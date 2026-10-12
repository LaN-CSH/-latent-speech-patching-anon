# eval_shellaswag_patched.py
#
# sHellaSwag evaluation with LST-style speech patching.
# This keeps the original eval flow:
#   audio -> HuBERT features -> kmeans speech tokens -> speech-conditioned option scoring
# but replaces raw speech-token prefix with patch-level z vectors:
#   <s> + z_1 ... z_P + <t> + option_text
# Labels are -100 on <s>, z patches, and <t>; only option text tokens are scored.

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import fairseq
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

IGNORE_INDEX = -100

try:
    RMSNorm = nn.RMSNorm
except AttributeError:  # pragma: no cover
    RMSNorm = nn.LayerNorm


# -----------------------------
# Original eval utilities
# -----------------------------


def load_jsonl_files(meta_dir: str):
    rows = []
    for path in sorted(Path(meta_dir).glob("meta_*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def load_meta(model_dir_or_file: str) -> Dict[str, Any]:
    p = Path(model_dir_or_file)
    if p.is_dir():
        p = p / "speech_token_meta.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tokenizer_robust(tokenizer_name_or_path: str):
    try:
        return AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    except AttributeError as e:
        if "extra_special_tokens" in str(e):
            raise RuntimeError(
                f"Failed to load tokenizer from {tokenizer_name_or_path!r}. "
                "Your checkpoint tokenizer_config.json likely has extra_special_tokens as a list. "
                "Use --tokenizer_name_or_path Qwen/Qwen2.5-0.5B."
            ) from e
        raise


def load_hubert_model(ckpt_path: str, device: str):
    models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    model = models[0].to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_hubert_tokens(hubert_model, kmeans, wav_path: str, device: str, output_layer: int = 9):
    wav, sr = librosa.load(wav_path, sr=16000, mono=True)
    wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).to(device)

    feats, _ = hubert_model.extract_features(
        source=wav_tensor,
        padding_mask=None,
        output_layer=output_layer,
    )

    feats = feats.squeeze(0).cpu().numpy()
    tokens = kmeans.predict(feats).tolist()
    return tokens


def speech_ids(hubert_tokens: List[int], hubert_offset: int, codebook_size: int):
    return [hubert_offset + int(x) for x in hubert_tokens if 0 <= int(x) < codebook_size]


def text_ids(tokenizer, text: str, add_leading_space: bool = False):
    text = " ".join(text.lower().strip().split())
    if add_leading_space and text:
        text = " " + text
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def resolve_wav_path(ex: Dict[str, Any], audio_dir: Path) -> Optional[Path]:
    """Use ex['audio_path'] when available; otherwise fall back to audio_dir / f'{id}.wav'."""
    if "audio_path" in ex and ex["audio_path"]:
        p = Path(ex["audio_path"])
        if p.is_file():
            return p
        p2 = audio_dir / ex["audio_path"]
        if p2.is_file():
            return p2
    ex_id = ex.get("id")
    if ex_id is not None:
        p3 = audio_dir / f"{ex_id}.wav"
        if p3.is_file():
            return p3
    return None


# -----------------------------
# Patch planning
# -----------------------------


def chunk_static_lengths(n_tokens: int, patch_size: int) -> List[int]:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    out: List[int] = []
    pos = 0
    while pos < n_tokens:
        length = min(patch_size, n_tokens - pos)
        out.append(length)
        pos += length
    return out


# -----------------------------
# LST-style patch encoder
# Same architecture as train_lslm_lst_patch_finetune_tokenizerfix_v4_savefix.py
# -----------------------------


class LocalSelfAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        self.attn_norm = RMSNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            bias=False,
            batch_first=True,
        )
        self.ffn_norm = RMSNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size, bias=False),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor], causal: bool) -> torch.Tensor:
        seq_len = x.size(1)
        attn_mask = None
        if causal:
            attn_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
                diagonal=1,
            )
        h = self.attn_norm(x)
        attn_out, _ = self.attn(
            h,
            h,
            h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class PatchCrossAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_norm = RMSNorm(hidden_size)
        self.kv_norm = RMSNorm(hidden_size)
        self.wq = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wk = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wv = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wo = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor, kv: torch.Tensor, allow_mask: torch.Tensor) -> torch.Tensor:
        B, P, D = x.shape
        T = kv.size(1)
        q = self.wq(self.q_norm(x)).view(B, P, self.num_heads, self.head_dim).transpose(1, 2)
        kv_norm = self.kv_norm(kv)
        k = self.wk(kv_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(kv_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        disallow = ~allow_mask
        attn_mask = torch.zeros((B, 1, P, T), dtype=q.dtype, device=q.device)
        attn_mask = attn_mask.masked_fill(disallow.unsqueeze(1), torch.finfo(q.dtype).min)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(B, P, D)
        return x + self.wo(out)


class LSTSpeechPatchEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        local_layers: int = 1,
        local_heads: int = 8,
        cross_heads: int = 8,
        dropout: float = 0.0,
        pooling: str = "mean",
        causal_local_encoder: bool = True,
    ):
        super().__init__()
        if pooling not in {"mean", "avg", "max"}:
            raise ValueError("pooling must be one of: mean, avg, max")
        self.hidden_size = hidden_size
        self.pooling = "mean" if pooling == "avg" else pooling
        self.causal_local_encoder = causal_local_encoder
        self.local_layers = nn.ModuleList(
            [LocalSelfAttentionBlock(hidden_size, local_heads, dropout=dropout) for _ in range(local_layers)]
        )
        self.cross_attn = PatchCrossAttention(hidden_size, cross_heads, dropout=dropout)

    @staticmethod
    def _pad_speech(speech_embeds_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        B = len(speech_embeds_list)
        max_len = max(x.size(0) for x in speech_embeds_list)
        D = speech_embeds_list[0].size(-1)
        device = speech_embeds_list[0].device
        dtype = speech_embeds_list[0].dtype
        padded = torch.zeros((B, max_len, D), dtype=dtype, device=device)
        key_padding_mask = torch.ones((B, max_len), dtype=torch.bool, device=device)
        for b, x in enumerate(speech_embeds_list):
            L = x.size(0)
            padded[b, :L] = x
            key_padding_mask[b, :L] = False
        return padded, key_padding_mask

    @staticmethod
    def _build_patch_tensors(
        patch_lengths_list: List[List[int]],
        max_speech_len: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = len(patch_lengths_list)
        max_patches = max(len(x) for x in patch_lengths_list)
        patch_ids = torch.zeros((B, max_speech_len), dtype=torch.long, device=device)
        valid_token = torch.zeros((B, max_speech_len), dtype=torch.bool, device=device)
        valid_patch = torch.zeros((B, max_patches), dtype=torch.bool, device=device)

        for b, lengths in enumerate(patch_lengths_list):
            pos = 0
            for p, length in enumerate(lengths):
                length = int(length)
                if length <= 0:
                    continue
                patch_ids[b, pos : pos + length] = p
                valid_token[b, pos : pos + length] = True
                valid_patch[b, p] = True
                pos += length
            if pos > max_speech_len:
                raise ValueError("patch lengths exceed speech length")
        return patch_ids, valid_token, valid_patch

    def _pool_patches(
        self,
        h: torch.Tensor,
        patch_ids: torch.Tensor,
        valid_token: torch.Tensor,
        valid_patch: torch.Tensor,
    ) -> torch.Tensor:
        B, T, D = h.shape
        P = valid_patch.size(1)
        idx = patch_ids.unsqueeze(-1).expand(B, T, D)

        if self.pooling == "mean":
            sums = h.new_zeros((B, P, D))
            sums.scatter_add_(dim=1, index=idx, src=h * valid_token.unsqueeze(-1))
            counts = h.new_zeros((B, P, 1))
            counts.scatter_add_(dim=1, index=patch_ids.unsqueeze(-1), src=valid_token.to(dtype=h.dtype).unsqueeze(-1))
            return sums / counts.clamp_min(1.0)

        neg = torch.finfo(h.dtype).min
        src = h.masked_fill(~valid_token.unsqueeze(-1), neg)
        out = h.new_full((B, P, D), neg)
        out.scatter_reduce_(dim=1, index=idx, src=src, reduce="amax", include_self=True)
        out = out.masked_fill(~valid_patch.unsqueeze(-1), 0.0)
        return out

    def forward(
        self,
        speech_embeds_list: List[torch.Tensor],
        speech_patch_lengths: List[List[int]],
    ) -> List[torch.Tensor]:
        if len(speech_embeds_list) != len(speech_patch_lengths):
            raise ValueError("speech_embeds_list and speech_patch_lengths batch size mismatch")

        speech, key_padding_mask = self._pad_speech(speech_embeds_list)
        B, T, _ = speech.shape

        for b, lengths in enumerate(speech_patch_lengths):
            if sum(int(x) for x in lengths) != speech_embeds_list[b].size(0):
                raise ValueError(
                    f"batch {b}: sum(patch_lengths)={sum(lengths)} != speech_len={speech_embeds_list[b].size(0)}"
                )

        h = speech
        for layer in self.local_layers:
            h = layer(h, key_padding_mask=key_padding_mask, causal=self.causal_local_encoder)

        patch_ids, valid_token, valid_patch = self._build_patch_tensors(
            speech_patch_lengths,
            max_speech_len=T,
            device=speech.device,
        )

        patch_init = self._pool_patches(h, patch_ids, valid_token, valid_patch)

        P = valid_patch.size(1)
        q_ids = torch.arange(P, device=speech.device).view(1, P, 1)
        allow = (patch_ids.unsqueeze(1) == q_ids) & valid_token.unsqueeze(1) & valid_patch.unsqueeze(-1)
        if T > 0:
            allow = torch.where(valid_patch.unsqueeze(-1), allow, torch.zeros_like(allow))
            allow[:, :, 0] |= ~valid_patch

        z = self.cross_attn(patch_init, h, allow_mask=allow)

        out: List[torch.Tensor] = []
        for b, lengths in enumerate(speech_patch_lengths):
            out.append(z[b, : len(lengths)])
        return out


# -----------------------------
# Patched option scoring
# -----------------------------


def load_patch_encoder(
    model,
    patch_ckpt_path: Path,
    meta: Dict[str, Any],
    device: str,
    dtype: Optional[torch.dtype],
    args,
) -> LSTSpeechPatchEncoder:
    hidden_size = model.config.hidden_size

    local_layers = args.local_layers if args.local_layers is not None else int(meta.get("local_layers", 1))
    local_heads = args.local_heads if args.local_heads is not None else int(meta.get("local_heads", 8))
    cross_heads = args.cross_heads if args.cross_heads is not None else int(meta.get("cross_heads", 8))
    pooling = args.pooling if args.pooling is not None else meta.get("pooling", "mean")
    bidirectional = bool(meta.get("bidirectional_local_encoder", False))
    if args.bidirectional_local_encoder:
        bidirectional = True

    patch_encoder = LSTSpeechPatchEncoder(
        hidden_size=hidden_size,
        local_layers=local_layers,
        local_heads=local_heads,
        cross_heads=cross_heads,
        dropout=0.0,
        pooling=pooling,
        causal_local_encoder=not bidirectional,
    )

    state = torch.load(patch_ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    patch_encoder.load_state_dict(state, strict=True)
    patch_encoder.to(device)
    if dtype is not None:
        patch_encoder.to(dtype=dtype)
    patch_encoder.eval()
    return patch_encoder


@torch.no_grad()
def build_patched_inputs_embeds(
    model,
    patch_encoder: LSTSpeechPatchEncoder,
    s_id: int,
    t_id: int,
    speech_token_ids: List[int],
    option_ids: List[int],
    patch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    device = model.device
    embed = model.get_input_embeddings()

    if len(speech_token_ids) == 0:
        # no-speech control: <s> <t> option
        ids = [s_id, t_id] + option_ids
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        inputs_embeds = embed(input_ids)
        attention_mask = torch.ones((1, len(ids)), dtype=torch.long, device=device)
        labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX] + option_ids], dtype=torch.long, device=device)
        return inputs_embeds, attention_mask, labels, 0

    speech_tensor = torch.tensor([speech_token_ids], dtype=torch.long, device=device)
    speech_embeds = embed(speech_tensor)[0]

    patch_lengths = chunk_static_lengths(len(speech_token_ids), patch_size)
    z = patch_encoder([speech_embeds], [patch_lengths])[0]

    s_emb = embed(torch.tensor([[s_id]], dtype=torch.long, device=device))[0]
    suffix_ids = torch.tensor([[t_id] + option_ids], dtype=torch.long, device=device)
    suffix_embeds = embed(suffix_ids)[0]

    patched = torch.cat([s_emb, z, suffix_embeds], dim=0).unsqueeze(0)
    attention_mask = torch.ones((1, patched.size(1)), dtype=torch.long, device=device)
    labels = torch.tensor(
        [[IGNORE_INDEX] * (1 + len(patch_lengths) + 1) + option_ids],
        dtype=torch.long,
        device=device,
    )
    return patched, attention_mask, labels, len(patch_lengths)


@torch.no_grad()
def score_option_patched(
    model,
    patch_encoder: LSTSpeechPatchEncoder,
    s_id: int,
    t_id: int,
    speech_token_ids: List[int],
    option_ids: List[int],
    patch_size: int,
):
    inputs_embeds, attention_mask, labels, n_patches = build_patched_inputs_embeds(
        model=model,
        patch_encoder=patch_encoder,
        s_id=s_id,
        t_id=t_id,
        speech_token_ids=speech_token_ids,
        option_ids=option_ids,
        patch_size=patch_size,
    )

    out = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
    )

    logits = out.logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="sum")
    nll = loss_fct(
        logits.reshape(-1, logits.size(-1)),
        shift_labels.reshape(-1),
    )

    n_tokens = (shift_labels != IGNORE_INDEX).sum().item()
    n_input_tokens = int(attention_mask.sum().item())
    avg_nll = nll.item() / max(n_tokens, 1)
    avg_logprob = -avg_nll
    return avg_logprob, avg_nll, n_tokens, n_patches, n_input_tokens


# -----------------------------
# Compute-cost utilities
# -----------------------------


def sync_if_cuda(device: str):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def bytes_to_gib(n_bytes: int) -> float:
    return float(n_bytes) / (1024.0 ** 3)


def _cfg_int(config, names: List[str], default: Optional[int] = None) -> Optional[int]:
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    return default


def estimate_causal_lm_forward_flops(config, seq_len: int, vocab_size: Optional[int] = None) -> int:
    """Approximate dense decoder-only LM inference FLOPs for one forward pass.

    This is a model-agnostic estimate based on common Transformer blocks. It includes
    attention/MLP matmuls and the LM head projection, but excludes embeddings, norms,
    activations, softmax, loss computation, HuBERT, k-means, and IO overhead.
    """
    if seq_len <= 0:
        return 0

    n_layers = _cfg_int(config, ["num_hidden_layers", "n_layer", "num_layers"], 0) or 0
    hidden = _cfg_int(config, ["hidden_size", "n_embd", "d_model"], 0) or 0
    intermediate = _cfg_int(config, ["intermediate_size", "n_inner", "ffn_dim"], None)
    vocab = int(vocab_size or getattr(config, "vocab_size", 0) or 0)
    if n_layers <= 0 or hidden <= 0:
        return 0
    if intermediate is None:
        intermediate = 4 * hidden

    model_type = str(getattr(config, "model_type", "")).lower()
    gated_mlp = any(x in model_type for x in ["llama", "mistral", "qwen", "gemma", "phi"])

    # 2 FLOPs per multiply-add. Dense self-attention estimate.
    qkvo_proj = 8.0 * seq_len * hidden * hidden
    attention_scores_and_values = 4.0 * seq_len * seq_len * hidden
    mlp = (6.0 if gated_mlp else 4.0) * seq_len * hidden * intermediate
    lm_head = 2.0 * seq_len * hidden * vocab if vocab > 0 else 0.0

    return int(round(n_layers * (qkvo_proj + attention_scores_and_values + mlp) + lm_head))


def estimate_patch_encoder_forward_flops(
    patch_encoder: LSTSpeechPatchEncoder,
    n_speech_tokens: int,
    n_patches: int,
) -> int:
    """Approximate FLOPs for one LST patch-encoder forward pass.

    The implementation uses dense attention masks, so the cross-attention term is
    estimated as P x T, not as sparse patch-local attention.
    """
    T = int(n_speech_tokens)
    P = int(n_patches)
    if T <= 0 or P <= 0:
        return 0

    D = int(patch_encoder.hidden_size)
    n_local_layers = len(patch_encoder.local_layers)

    local_attn_proj = 8.0 * T * D * D
    local_attn_scores_values = 4.0 * T * T * D
    local_ffn = 16.0 * T * D * D  # D -> 4D -> D
    local_total = n_local_layers * (local_attn_proj + local_attn_scores_values + local_ffn)

    cross_qkvo_proj = 4.0 * D * D * (P + T)
    cross_attn_scores_values = 4.0 * P * T * D

    return int(round(local_total + cross_qkvo_proj + cross_attn_scores_values))


def default_cost_output_path(output_jsonl: str) -> str:
    p = Path(output_jsonl)
    return str(p.with_name(p.name + ".cost.json"))


def cuda_memory_snapshot(device: str) -> Dict[str, Any]:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return {}
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "allocated_gib": bytes_to_gib(int(torch.cuda.memory_allocated(device))),
        "reserved_gib": bytes_to_gib(int(torch.cuda.memory_reserved(device))),
        "max_allocated_gib": bytes_to_gib(int(torch.cuda.max_memory_allocated(device))),
        "max_reserved_gib": bytes_to_gib(int(torch.cuda.max_memory_reserved(device))),
    }


# -----------------------------
# Main
# -----------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Patched output dir containing base model, speech_token_meta.json, and speech_patch_encoder.pt")
    parser.add_argument("--base_lslm_dir", default=None, help="Load frozen/base LSLM from here instead of --model_dir. Useful for patch-encoder-only checkpoints.")
    parser.add_argument("--tokenizer_name_or_path", default=None, help="Use Qwen/Qwen2.5-0.5B if checkpoint tokenizer_config.json is broken.")
    parser.add_argument("--patch_encoder_ckpt", default=None, help="Defaults to <model_dir>/speech_patch_encoder.pt")
    parser.add_argument("--meta_path", default=None, help="Defaults to <model_dir>/speech_token_meta.json")

    parser.add_argument("--meta_dir", required=True)
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--hubert_ckpt", required=True)
    parser.add_argument("--km_path", required=True)
    parser.add_argument("--output_jsonl", default="shellaswag_patched_eval_results.jsonl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hubert_layer", type=int, default=9)
    parser.add_argument("--max_speech_tokens", type=int, default=1600)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    parser.add_argument("--patch_size", type=int, default=None, help="Override patch size. Default: value in speech_token_meta.json")
    parser.add_argument("--pooling", choices=["mean", "avg", "max"], default=None)
    parser.add_argument("--local_layers", type=int, default=None)
    parser.add_argument("--local_heads", type=int, default=None)
    parser.add_argument("--cross_heads", type=int, default=None)
    parser.add_argument("--bidirectional_local_encoder", action="store_true")
    parser.add_argument("--allow_resize", action="store_true", help="Allow resizing token embeddings if vocab size does not match meta. Usually false for eval.")

    parser.add_argument("--add_leading_space", action="store_true", help="Prepend a space to each ending before tokenization.")
    parser.add_argument("--speech_control", choices=["real", "shuffled", "no_speech"], default="real")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--compute_cost",
        action="store_true",
        help=(
            "Collect compute-cost metrics: wall-clock time, option forward count, "
            "token/patch counts, CUDA peak memory, and approximate FLOPs."
        ),
    )
    parser.add_argument(
        "--cost_output_json",
        default=None,
        help="Where to save aggregate compute-cost summary. Default: <output_jsonl>.cost.json when --compute_cost is set.",
    )
    parser.add_argument(
        "--record_example_cost",
        action="store_true",
        help="Add per-example cost metrics to each output_jsonl row. Implies --compute_cost.",
    )
    parser.add_argument(
        "--skip_flop_estimate",
        action="store_true",
        help="Skip approximate FLOPs estimates and only record measured timing/memory/counts.",
    )
    args = parser.parse_args()

    compute_cost = bool(args.compute_cost or args.record_example_cost or args.cost_output_json is not None)

    random.seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model_dir = Path(args.model_dir)
    meta_path = Path(args.meta_path) if args.meta_path is not None else model_dir / "speech_token_meta.json"
    patch_ckpt_path = Path(args.patch_encoder_ckpt) if args.patch_encoder_ckpt is not None else model_dir / "speech_patch_encoder.pt"
    base_lslm_dir = args.base_lslm_dir or args.model_dir
    tokenizer_path = args.tokenizer_name_or_path or base_lslm_dir

    meta = load_meta(str(meta_path))
    s_id = int(meta["s_id"])
    t_id = int(meta["t_id"])
    hubert_offset = int(meta["hubert_offset"])
    total_vocab_size = int(meta["total_vocab_size"])
    codebook_size = int(meta["codebook_size"])
    patch_size = args.patch_size if args.patch_size is not None else int(meta.get("patch_size", 4))

    print("model_dir:", args.model_dir)
    print("base_lslm_dir:", base_lslm_dir)
    print("tokenizer_path:", tokenizer_path)
    print("patch_encoder_ckpt:", str(patch_ckpt_path))
    print("text_vocab_size:", meta.get("text_vocab_size"))
    print("<s> id:", s_id)
    print("<t> id:", t_id)
    print("hubert_offset:", hubert_offset)
    print("total_vocab_size:", total_vocab_size)
    print("codebook_size:", codebook_size)
    print("patch_size:", patch_size)
    print("speech_control:", args.speech_control)
    print("compute_cost:", compute_cost)

    tokenizer = load_tokenizer_robust(tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else None
    model = AutoModelForCausalLM.from_pretrained(
        base_lslm_dir,
        trust_remote_code=True,
        torch_dtype=dtype,
    )

    current_vocab = model.get_input_embeddings().weight.shape[0]
    if current_vocab != total_vocab_size:
        if not args.allow_resize:
            raise ValueError(
                f"Model vocab size {current_vocab} != total_vocab_size {total_vocab_size}. "
                "For eval this usually means --base_lslm_dir or --model_dir is wrong. "
                "Use --allow_resize only if you intentionally want random new rows."
            )
        print(f"Resizing token embeddings: {current_vocab} -> {total_vocab_size}")
        model.resize_token_embeddings(total_vocab_size)

    model.to(device)
    model.eval()

    if not patch_ckpt_path.is_file():
        raise FileNotFoundError(f"Patch encoder checkpoint not found: {patch_ckpt_path}")

    patch_encoder = load_patch_encoder(
        model=model,
        patch_ckpt_path=patch_ckpt_path,
        meta=meta,
        device=device,
        dtype=dtype,
        args=args,
    )

    hubert_model = load_hubert_model(args.hubert_ckpt, device)
    kmeans = joblib.load(args.km_path)

    rows = load_jsonl_files(args.meta_dir)
    print("num examples:", len(rows))

    cost_output_json = args.cost_output_json or default_cost_output_path(args.output_jsonl)
    cost: Optional[Dict[str, Any]] = None
    if compute_cost:
        sync_if_cuda(device)
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        cost = {
            "enabled": True,
            "notes": [
                "Wall-clock timings include HuBERT feature extraction, k-means, patch encoder, CausalLM forward, loss, and Python/IO overhead.",
                "FLOPs are approximate dense matmul estimates and exclude HuBERT, k-means, softmax, norms, activations, loss, and IO.",
                "Patch encoder is currently run once per answer option, matching the original scoring flow.",
            ],
            "config": {
                "model_dir": str(args.model_dir),
                "base_lslm_dir": str(base_lslm_dir),
                "tokenizer_path": str(tokenizer_path),
                "patch_encoder_ckpt": str(patch_ckpt_path),
                "speech_control": args.speech_control,
                "patch_size": patch_size,
                "max_speech_tokens": args.max_speech_tokens,
                "device": device,
                "dtype": "bf16" if args.bf16 else "fp16" if args.fp16 else "fp32/default",
                "skip_flop_estimate": args.skip_flop_estimate,
            },
            "model": {
                "base_model_params": count_parameters(model),
                "patch_encoder_params": count_parameters(patch_encoder),
                "total_loaded_scored_params": count_parameters(model) + count_parameters(patch_encoder),
                "hidden_size": int(getattr(model.config, "hidden_size", 0) or 0),
                "num_hidden_layers": int(getattr(model.config, "num_hidden_layers", 0) or 0),
                "intermediate_size": int(getattr(model.config, "intermediate_size", 0) or 0),
                "vocab_size": int(total_vocab_size),
            },
            "counts": {
                "requested_examples": len(rows),
                "evaluated_examples": 0,
                "missing_audio_examples": 0,
                "option_forwards": 0,
                "scored_text_tokens": 0,
                "input_tokens_equivalent": 0,
                "unique_speech_tokens": 0,
                "option_conditioning_speech_tokens": 0,
                "option_conditioning_speech_patches": 0,
            },
            "timing_seconds": {
                "total_eval": 0.0,
                "audio_tokenization_hubert_kmeans": 0.0,
                "option_scoring": 0.0,
            },
            "throughput": {},
            "memory_cuda": {
                "after_model_load": cuda_memory_snapshot(device),
                "during_eval_peak": {},
            },
            "estimated_flops": {
                "causal_lm_forward": 0,
                "patch_encoder_forward": 0,
                "total": 0,
            },
        }

    correct = 0
    total = 0
    audio_dir = Path(args.audio_dir)
    eval_start_time = time.perf_counter()

    with open(args.output_jsonl, "w", encoding="utf-8") as out:
        for ex in tqdm(rows):
            ex_id = ex["id"]
            label = int(ex["label"])
            example_audio_seconds = 0.0
            example_scoring_seconds = 0.0
            example_scored_tokens = 0
            example_input_tokens = 0
            example_patch_encoder_flops = 0
            example_lm_flops = 0

            wav_path = resolve_wav_path(ex, audio_dir)
            if wav_path is None:
                print(f"[WARN] missing audio for id={ex_id}")
                if cost is not None:
                    cost["counts"]["missing_audio_examples"] += 1
                continue

            if args.speech_control == "no_speech":
                sp_ids: List[int] = []
            else:
                if compute_cost:
                    sync_if_cuda(device)
                    audio_start = time.perf_counter()

                hubert_tokens = extract_hubert_tokens(
                    hubert_model,
                    kmeans,
                    str(wav_path),
                    device,
                    output_layer=args.hubert_layer,
                )

                if compute_cost:
                    sync_if_cuda(device)
                    example_audio_seconds = time.perf_counter() - audio_start
                    assert cost is not None
                    cost["timing_seconds"]["audio_tokenization_hubert_kmeans"] += example_audio_seconds

                if args.max_speech_tokens is not None:
                    hubert_tokens = hubert_tokens[: args.max_speech_tokens]

                sp_ids = speech_ids(hubert_tokens, hubert_offset, codebook_size)
                if args.speech_control == "shuffled":
                    rng = random.Random(f"{args.seed}-{ex_id}")
                    rng.shuffle(sp_ids)

            scores = []

            for ending in ex["endings"]:
                option_ids = text_ids(tokenizer, ending, add_leading_space=args.add_leading_space)

                if compute_cost:
                    sync_if_cuda(device)
                    scoring_start = time.perf_counter()

                avg_logprob, avg_nll, n_tokens, n_patches, n_input_tokens = score_option_patched(
                    model=model,
                    patch_encoder=patch_encoder,
                    s_id=s_id,
                    t_id=t_id,
                    speech_token_ids=sp_ids,
                    option_ids=option_ids,
                    patch_size=patch_size,
                )

                if compute_cost:
                    sync_if_cuda(device)
                    option_scoring_seconds = time.perf_counter() - scoring_start
                    example_scoring_seconds += option_scoring_seconds
                    assert cost is not None
                    cost["timing_seconds"]["option_scoring"] += option_scoring_seconds

                    lm_flops = 0
                    patch_flops = 0
                    if not args.skip_flop_estimate:
                        lm_flops = estimate_causal_lm_forward_flops(
                            model.config,
                            seq_len=n_input_tokens,
                            vocab_size=total_vocab_size,
                        )
                        patch_flops = estimate_patch_encoder_forward_flops(
                            patch_encoder,
                            n_speech_tokens=len(sp_ids),
                            n_patches=n_patches,
                        )
                        cost["estimated_flops"]["causal_lm_forward"] += lm_flops
                        cost["estimated_flops"]["patch_encoder_forward"] += patch_flops
                        example_lm_flops += lm_flops
                        example_patch_encoder_flops += patch_flops

                    cost["counts"]["option_forwards"] += 1
                    cost["counts"]["scored_text_tokens"] += int(n_tokens)
                    cost["counts"]["input_tokens_equivalent"] += int(n_input_tokens)
                    cost["counts"]["option_conditioning_speech_tokens"] += int(len(sp_ids))
                    cost["counts"]["option_conditioning_speech_patches"] += int(n_patches)
                    example_scored_tokens += int(n_tokens)
                    example_input_tokens += int(n_input_tokens)

                score_row = {
                    "ending": ending,
                    "avg_logprob": avg_logprob,
                    "avg_nll": avg_nll,
                    "n_tokens": n_tokens,
                    "n_input_tokens": n_input_tokens,
                    "n_speech_tokens": len(sp_ids),
                    "n_speech_patches": n_patches,
                }
                if compute_cost and args.record_example_cost:
                    score_row["estimated_lm_forward_flops"] = lm_flops
                    score_row["estimated_patch_encoder_flops"] = patch_flops
                scores.append(score_row)

            pred = max(range(len(scores)), key=lambda i: scores[i]["avg_logprob"])
            is_correct = int(pred == label)
            correct += is_correct
            total += 1

            row = {
                "id": ex_id,
                "label": label,
                "pred": pred,
                "correct": is_correct,
                "speech_control": args.speech_control,
                "patch_size": patch_size,
                "scores": scores,
            }

            if cost is not None:
                cost["counts"]["evaluated_examples"] += 1
                cost["counts"]["unique_speech_tokens"] += int(len(sp_ids))

                if args.record_example_cost:
                    row["compute_cost"] = {
                        "audio_tokenization_hubert_kmeans_seconds": example_audio_seconds,
                        "option_scoring_seconds": example_scoring_seconds,
                        "option_forwards": len(ex["endings"]),
                        "scored_text_tokens": example_scored_tokens,
                        "input_tokens_equivalent": example_input_tokens,
                        "unique_speech_tokens": len(sp_ids),
                        "option_conditioning_speech_tokens": len(sp_ids) * len(ex["endings"]),
                        "estimated_lm_forward_flops": example_lm_flops,
                        "estimated_patch_encoder_flops": example_patch_encoder_flops,
                        "estimated_total_flops": example_lm_flops + example_patch_encoder_flops,
                    }

            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    if compute_cost:
        sync_if_cuda(device)

    acc = correct / max(total, 1)
    print("==== patched sHellaSwag Results ====")
    print("total:", total)
    print("correct:", correct)
    print("accuracy:", acc)
    print("saved:", args.output_jsonl)

    if cost is not None:
        total_eval_seconds = time.perf_counter() - eval_start_time
        cost["timing_seconds"]["total_eval"] = total_eval_seconds
        cost["estimated_flops"]["total"] = (
            cost["estimated_flops"]["causal_lm_forward"]
            + cost["estimated_flops"]["patch_encoder_forward"]
        )
        counts = cost["counts"]
        cost["accuracy"] = {
            "total": total,
            "correct": correct,
            "accuracy": acc,
        }
        cost["throughput"] = {
            "examples_per_second": counts["evaluated_examples"] / total_eval_seconds if total_eval_seconds > 0 else None,
            "option_forwards_per_second": counts["option_forwards"] / cost["timing_seconds"]["option_scoring"]
            if cost["timing_seconds"]["option_scoring"] > 0
            else None,
            "scored_text_tokens_per_second": counts["scored_text_tokens"] / cost["timing_seconds"]["option_scoring"]
            if cost["timing_seconds"]["option_scoring"] > 0
            else None,
            "input_tokens_equivalent_per_second": counts["input_tokens_equivalent"] / cost["timing_seconds"]["option_scoring"]
            if cost["timing_seconds"]["option_scoring"] > 0
            else None,
        }
        cost["memory_cuda"]["during_eval_peak"] = cuda_memory_snapshot(device)

        print("==== Compute Cost ====")
        print("wall_seconds:", total_eval_seconds)
        print("audio_tokenization_hubert_kmeans_seconds:", cost["timing_seconds"]["audio_tokenization_hubert_kmeans"])
        print("option_scoring_seconds:", cost["timing_seconds"]["option_scoring"])
        print("option_forwards:", counts["option_forwards"])
        print("scored_text_tokens:", counts["scored_text_tokens"])
        print("input_tokens_equivalent:", counts["input_tokens_equivalent"])
        if not args.skip_flop_estimate:
            print("estimated_total_flops:", cost["estimated_flops"]["total"])
            print("estimated_total_tflops:", cost["estimated_flops"]["total"] / 1e12)
        if cost["memory_cuda"]["during_eval_peak"]:
            print("cuda_peak_allocated_gib:", cost["memory_cuda"]["during_eval_peak"]["max_allocated_gib"])
            print("cuda_peak_reserved_gib:", cost["memory_cuda"]["during_eval_peak"]["max_reserved_gib"])

        with open(cost_output_json, "w", encoding="utf-8") as f:
            json.dump(cost, f, ensure_ascii=False, indent=2)
        print("cost_saved:", cost_output_json)


if __name__ == "__main__":
    main()
