# train_lslm_lst_patch_finetune.py
#
# LST-style speech patching for finetuning a pretrained LSLM/LLM.
# - The model input embedding matrix is frozen.
# - Speech token embeddings are read from the frozen embedding matrix.
# - Speech tokens are encoded into patch-level z vectors using an LST-inspired local encoder:
#     speech embeddings -> local self-attention -> patch pooling init -> patch-to-token cross-attention
# - The pretrained causal LM receives: <s> embedding + speech patch z's + <t> embedding + text embeddings
# - Loss is only applied to text tokens. <s>, speech patches, and <t> are ignored with -100.
#
# Expected JSONL examples:
#   {"input_ids": [s_id, speech_id..., t_id, text_id...], "utt_id": ..., "split_word_idx": ...}
# Optional existing labels are ignored; labels are rebuilt after patching.

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

IGNORE_INDEX = -100


def maybe_load_speech_meta(model_name_or_path: str) -> Optional[Dict[str, Any]]:
    meta_path = os.path.join(model_name_or_path, "speech_token_meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_tokenizer_robust(tokenizer_name_or_path: str):
    try:
        return AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    except AttributeError as e:
        if "extra_special_tokens" in str(e):
            raise RuntimeError(
                f"Failed to load tokenizer from {tokenizer_name_or_path!r} because "
                "tokenizer_config.json has extra_special_tokens as a list. "
                "Use --tokenizer_name_or_path Qwen/Qwen2.5-0.5B, or edit that "
                "checkpoint tokenizer_config.json so extra_special_tokens is {}."
            ) from e
        raise


try:
    RMSNorm = nn.RMSNorm
except AttributeError:  # pragma: no cover
    RMSNorm = nn.LayerNorm


# -----------------------------
# JSONL / patch planning helpers
# -----------------------------


def load_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_alignment_map(path: Optional[str]) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    if path is None:
        return None
    out: Dict[str, List[Dict[str, Any]]] = {}
    for row in load_jsonl(path):
        out[row["utt_id"]] = row["words"]
    return out


def find_t_pos(input_ids: List[int], t_id: int) -> int:
    try:
        return input_ids.index(t_id)
    except ValueError as e:
        raise ValueError(f"<t> token id {t_id} not found in input_ids") from e


def validate_speech_to_text_format(input_ids: List[int], s_id: int, t_id: int) -> int:
    if len(input_ids) < 4:
        raise ValueError("input_ids too short; expected <s> speech... <t> text...")
    if input_ids[0] != s_id:
        raise ValueError(f"input_ids[0]={input_ids[0]} != s_id={s_id}")
    t_pos = find_t_pos(input_ids, t_id)
    if t_pos <= 1:
        raise ValueError("No speech tokens between <s> and <t>")
    if t_pos >= len(input_ids) - 1:
        raise ValueError("No text tokens after <t>")
    return t_pos


def chunk_static_lengths(n_tokens: int, patch_size: int) -> List[int]:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    out: List[int] = []
    i = 0
    while i < n_tokens:
        length = min(patch_size, n_tokens - i)
        out.append(length)
        i += length
    return out


def split_long_lengths(lengths: List[int], max_patch_length: Optional[int]) -> List[int]:
    out: List[int] = []
    for length in lengths:
        length = int(length)
        if length <= 0:
            continue
        if max_patch_length is None:
            out.append(length)
            continue
        while length > max_patch_length:
            out.append(max_patch_length)
            length -= max_patch_length
        if length > 0:
            out.append(length)
    return out


def build_align_lengths_from_words(
    ex: Dict[str, Any],
    words: List[Dict[str, Any]],
    speech_len: int,
    max_patch_length: Optional[int],
) -> List[int]:
    """Build one patch per word span and gap/silence span.

    This assumes the speech prefix corresponds to words[:split_word_idx], as in the
    LibriSpeech S->T data construction used earlier in this project.
    """
    utt_id = ex.get("utt_id", "<unknown>")
    if "split_word_idx" not in ex:
        raise ValueError("align patching requires split_word_idx in each JSONL example")

    split_idx = int(ex["split_word_idx"])
    if split_idx <= 0 or split_idx > len(words):
        raise ValueError(f"[{utt_id}] invalid split_word_idx={split_idx}, len(words)={len(words)}")

    speech_words = words[:split_idx]
    base_start = int(speech_words[0]["hubert_start"])
    base_end = int(speech_words[-1]["hubert_end"])
    expected_len = base_end - base_start
    if expected_len != speech_len:
        raise ValueError(
            f"[{utt_id}] alignment mismatch: word span length {expected_len} != speech_len {speech_len}. "
            "Make sure train_jsonl was built from the same alignment and HuBERT tokenization."
        )

    lengths: List[int] = []
    cur = base_start
    for w in speech_words:
        ws = int(w["hubert_start"])
        we = int(w["hubert_end"])
        if ws > cur:
            lengths.append(ws - cur)  # gap/silence before word
        if we > ws:
            lengths.append(we - ws)   # word span
        cur = we
    if cur < base_end:
        lengths.append(base_end - cur)

    lengths = split_long_lengths(lengths, max_patch_length)
    if sum(lengths) != speech_len:
        raise ValueError(f"[{utt_id}] align lengths sum {sum(lengths)} != speech_len {speech_len}")
    return lengths


def make_patched_text_only_labels(
    input_ids: List[int],
    t_pos: int,
    num_speech_patches: int,
) -> List[int]:
    """Labels for patched input: [<s>, z_1..z_P, <t>, text...].

    Loss only on text tokens after <t>.
    """
    text_ids = input_ids[t_pos + 1 :]
    return [IGNORE_INDEX] * (1 + num_speech_patches + 1) + text_ids


# -----------------------------
# Dataset / collator
# -----------------------------


class SpeechToTextPatchDataset(Dataset):
    def __init__(
        self,
        train_jsonl: str,
        s_id: int,
        t_id: int,
        patch_mode: str,
        patch_size: int,
        alignment_jsonl: Optional[str] = None,
        max_patch_length: Optional[int] = None,
        align_mismatch_policy: str = "error",
        max_raw_length: Optional[int] = None,
        mixed_static_prob: float = 0.5,
        seed: int = 42,
        curriculum_align_frac: float = 1.0 / 3.0,
        curriculum_mixed_frac: float = 1.0 / 3.0,
    ):
        if patch_mode not in {"static", "align", "mixed", "curriculum"}:
            raise ValueError(f"Unsupported patch_mode: {patch_mode}")
        if not (0.0 <= mixed_static_prob <= 1.0):
            raise ValueError("mixed_static_prob must be in [0, 1]")
        if curriculum_align_frac < 0 or curriculum_mixed_frac < 0:
            raise ValueError("curriculum fractions must be non-negative")
        if curriculum_align_frac + curriculum_mixed_frac > 1.0:
            raise ValueError("curriculum_align_frac + curriculum_mixed_frac must be <= 1")

        self.data: List[Dict[str, Any]] = []
        self.patch_mode = patch_mode
        self.patch_size = patch_size
        self.mixed_static_prob = float(mixed_static_prob)
        self.seed = int(seed)
        self.curriculum_align_frac = float(curriculum_align_frac)
        self.curriculum_mixed_frac = float(curriculum_mixed_frac)
        self.curriculum_progress = 0.0

        needs_alignment = patch_mode in {"align", "mixed", "curriculum"}
        alignment_map = load_alignment_map(alignment_jsonl)
        if needs_alignment and alignment_map is None:
            raise ValueError(
                "--alignment_jsonl is required when --patch_mode is align, mixed, or curriculum"
            )

        n_total, n_skip, n_fallback = 0, 0, 0
        for ex in load_jsonl(train_jsonl):
            n_total += 1
            input_ids = [int(x) for x in ex["input_ids"]]
            if max_raw_length is not None and len(input_ids) > max_raw_length:
                # Preserve the format only if <t> and at least one text token remain.
                input_ids = input_ids[:max_raw_length]

            try:
                t_pos = validate_speech_to_text_format(input_ids, s_id=s_id, t_id=t_id)
                speech_len = t_pos - 1
                static_lengths = chunk_static_lengths(speech_len, patch_size)

                align_lengths: Optional[List[int]] = None
                if needs_alignment:
                    utt_id = ex["utt_id"]
                    if utt_id not in alignment_map:  # type: ignore[operator]
                        raise ValueError(f"{utt_id} not found in alignment_jsonl")
                    align_lengths = build_align_lengths_from_words(
                        ex=ex,
                        words=alignment_map[utt_id],  # type: ignore[index]
                        speech_len=speech_len,
                        max_patch_length=max_patch_length,
                    )
            except Exception:
                if needs_alignment and align_mismatch_policy == "static":
                    t_pos = validate_speech_to_text_format(input_ids, s_id=s_id, t_id=t_id)
                    speech_len = t_pos - 1
                    static_lengths = chunk_static_lengths(speech_len, patch_size)
                    align_lengths = static_lengths
                    n_fallback += 1
                elif align_mismatch_policy == "skip":
                    n_skip += 1
                    continue
                else:
                    raise

            self.data.append(
                {
                    "input_ids": input_ids,
                    "raw_len": len(input_ids),
                    "t_pos": t_pos,
                    "speech_start": 1,
                    "speech_len": speech_len,
                    "static_speech_patch_lengths": [int(x) for x in static_lengths],
                    "align_speech_patch_lengths": None if align_lengths is None else [int(x) for x in align_lengths],
                    "utt_id": ex.get("utt_id"),
                }
            )

        print(f"Loaded {len(self.data)} examples from {n_total}")
        print(f"patch_mode: {self.patch_mode}")
        if self.patch_mode == "mixed":
            print(f"mixed_static_prob: {self.mixed_static_prob}")
        if self.patch_mode == "curriculum":
            print(
                "curriculum schedule: "
                f"align first {self.curriculum_align_frac:.3f}, "
                f"mixed next {self.curriculum_mixed_frac:.3f}, "
                "static final"
            )
        if n_fallback:
            print(f"align fallback to static: {n_fallback}")
        if n_skip:
            print(f"skipped examples: {n_skip}")

    def __len__(self) -> int:
        return len(self.data)

    def set_curriculum_progress(self, progress: float):
        self.curriculum_progress = max(0.0, min(1.0, float(progress)))

    def current_curriculum_stage(self) -> str:
        p = self.curriculum_progress
        if p < self.curriculum_align_frac:
            return "align"
        if p < self.curriculum_align_frac + self.curriculum_mixed_frac:
            return "mixed"
        return "static"

    def _mixed_choice(self, idx: int, salt: int = 0) -> str:
        # Deterministic per index/progress so runs are reproducible.
        rng = random.Random(self.seed + idx * 1_000_003 + salt * 9_176)
        return "static" if rng.random() < self.mixed_static_prob else "align"

    def _select_patch_lengths(self, idx: int, item: Dict[str, Any]) -> Tuple[List[int], str]:
        static_lengths = item["static_speech_patch_lengths"]
        align_lengths = item["align_speech_patch_lengths"]

        mode = self.patch_mode
        if mode == "curriculum":
            mode = self.current_curriculum_stage()

        if mode == "static":
            return static_lengths, "static"
        if mode == "align":
            if align_lengths is None:
                return static_lengths, "static_fallback"
            return align_lengths, "align"
        if mode == "mixed":
            # The salt changes across curriculum progress, so the mixed middle stage is not
            # exactly the same draw if curriculum is updated during training.
            salt = int(self.curriculum_progress * 1_000_000)
            chosen = self._mixed_choice(idx, salt=salt)
            if chosen == "align" and align_lengths is not None:
                return align_lengths, "align"
            return static_lengths, "static"
        raise ValueError(f"Unsupported patch_mode: {self.patch_mode}")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        speech_patch_lengths, used_patch_mode = self._select_patch_lengths(idx, item)
        labels = make_patched_text_only_labels(
            input_ids=item["input_ids"],
            t_pos=item["t_pos"],
            num_speech_patches=len(speech_patch_lengths),
        )
        patched_len = 1 + len(speech_patch_lengths) + (len(item["input_ids"]) - item["t_pos"])
        if patched_len != len(labels):
            raise RuntimeError(f"patched_len {patched_len} != labels len {len(labels)}")

        return {
            "input_ids": item["input_ids"],
            "raw_len": item["raw_len"],
            "t_pos": item["t_pos"],
            "speech_start": item["speech_start"],
            "speech_len": item["speech_len"],
            "speech_patch_lengths": [int(x) for x in speech_patch_lengths],
            "labels": labels,
            "utt_id": item.get("utt_id"),
            "used_patch_mode": used_patch_mode,
        }


@dataclass
class SpeechPatchCollator:
    pad_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_raw_len = max(len(x["input_ids"]) for x in features)
        max_patch_len = max(len(x["labels"]) for x in features)

        input_ids, raw_attention_mask, labels = [], [], []
        speech_start, speech_len, speech_patch_lengths = [], [], []

        for x in features:
            ids = x["input_ids"]
            labs = x["labels"]
            raw_pad = max_raw_len - len(ids)
            label_pad = max_patch_len - len(labs)

            input_ids.append(ids + [self.pad_token_id] * raw_pad)
            raw_attention_mask.append([1] * len(ids) + [0] * raw_pad)
            labels.append(labs + [IGNORE_INDEX] * label_pad)
            speech_start.append(int(x["speech_start"]))
            speech_len.append(int(x["speech_len"]))
            speech_patch_lengths.append([int(v) for v in x["speech_patch_lengths"]])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "raw_attention_mask": torch.tensor(raw_attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "speech_start": torch.tensor(speech_start, dtype=torch.long),
            "speech_len": torch.tensor(speech_len, dtype=torch.long),
            "speech_patch_lengths": speech_patch_lengths,
        }


# -----------------------------
# LST-style patch encoder
# -----------------------------


class LocalSelfAttentionBlock(nn.Module):
    """Small pre-norm transformer block over speech token embeddings.

    This mirrors the LST local encoder idea: token-level local/causal processing before
    reducing to patch-level states. It intentionally uses PyTorch SDPA/MHA to avoid repo
    dependencies such as xformers/flex_attention.
    """

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
            # PyTorch MultiheadAttention bool mask: True means disallowed.
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
    """Patch queries attend to token-level speech states.

    Based on the LST CrossAttention pattern: RMSNorm on q and kv, linear q/k/v/o,
    residual return x + attention(x, kv, kv). The mask restricts each patch query to
    its own speech patch unless a cross-window is requested upstream.
    """

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
        """
        x: [B, P, D] patch query/init states
        kv: [B, T, D] token-level speech states
        allow_mask: [B, P, T] bool, True where patch p may attend to token t
        """
        B, P, D = x.shape
        T = kv.size(1)
        q = self.wq(self.q_norm(x)).view(B, P, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(self.kv_norm(kv)).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(self.kv_norm(kv)).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Additive mask: 0 allowed, -inf disallowed. Shape broadcastable to [B,H,P,T].
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
    """Encode speech token embeddings into patch-level z vectors.

    LST-inspired path:
      frozen speech token embeddings x
      -> optional local self-attention over speech token sequence
      -> patch pooling to initialize patch states
      -> patch-to-token cross-attention within each patch
      -> z patch vectors
    """

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
            key_padding_mask[b, :L] = False  # False = valid for MultiheadAttention
        return padded, key_padding_mask

    @staticmethod
    def _build_patch_tensors(
        patch_lengths_list: List[List[int]],
        max_speech_len: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return patch_ids [B,T], valid_token [B,T], valid_patch [B,P]."""
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

        # max pooling. Invalid token positions are set to a large negative value.
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

        # LST-style patch cross-attention mask: each patch query attends to tokens in its patch.
        P = valid_patch.size(1)
        q_ids = torch.arange(P, device=speech.device).view(1, P, 1)
        allow = (patch_ids.unsqueeze(1) == q_ids) & valid_token.unsqueeze(1) & valid_patch.unsqueeze(-1)
        # Avoid NaNs for padded patch rows by allowing token 0; those rows are removed later.
        if T > 0:
            allow = torch.where(valid_patch.unsqueeze(-1), allow, torch.zeros_like(allow))
            allow[:, :, 0] |= ~valid_patch

        z = self.cross_attn(patch_init, h, allow_mask=allow)

        out: List[torch.Tensor] = []
        for b, lengths in enumerate(speech_patch_lengths):
            out.append(z[b, : len(lengths)])
        return out


# -----------------------------
# LLM wrapper
# -----------------------------


class LSLMWithFrozenEmbeddingLSTPatching(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        patch_encoder: LSTSpeechPatchEncoder,
        freeze_lm_head: bool = False,
        freeze_input_embeddings: bool = True,
    ):
        super().__init__()
        self.base_model = base_model
        self.patch_encoder = patch_encoder
        if freeze_input_embeddings:
            self.freeze_embeddings(freeze_lm_head=freeze_lm_head)

    def freeze_embeddings(self, freeze_lm_head: bool = False):
        emb = self.base_model.get_input_embeddings()
        for p in emb.parameters():
            p.requires_grad = False

        if freeze_lm_head and hasattr(self.base_model, "lm_head"):
            for p in self.base_model.lm_head.parameters():
                p.requires_grad = False

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            return self.base_model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.base_model, "gradient_checkpointing_disable"):
            return self.base_model.gradient_checkpointing_disable()

    def build_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        raw_attention_mask: torch.Tensor,
        speech_start: torch.Tensor,
        speech_len: torch.Tensor,
        speech_patch_lengths: List[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embed = self.base_model.get_input_embeddings()
        # Embedding is frozen, but gradients still flow from z into patch_encoder and base_model.
        raw_embeds = embed(input_ids)

        prefix_list: List[torch.Tensor] = []
        speech_list: List[torch.Tensor] = []
        suffix_list: List[torch.Tensor] = []

        for b in range(input_ids.size(0)):
            raw_len = int(raw_attention_mask[b].sum().item())
            ss = int(speech_start[b].item())
            sl = int(speech_len[b].item())
            se = ss + sl
            raw = raw_embeds[b, :raw_len]
            prefix_list.append(raw[:ss])       # normally just <s>
            speech_list.append(raw[ss:se])     # HuBERT token embeddings
            suffix_list.append(raw[se:])       # <t> + text tokens

        z_list = self.patch_encoder(speech_list, speech_patch_lengths)
        patched = [torch.cat([p, z, s], dim=0) for p, z, s in zip(prefix_list, z_list, suffix_list)]

        max_len = max(x.size(0) for x in patched)
        hidden = patched[0].size(-1)
        inputs_embeds = raw_embeds.new_zeros((len(patched), max_len, hidden))
        attention_mask = torch.zeros((len(patched), max_len), dtype=torch.long, device=raw_embeds.device)
        for b, emb in enumerate(patched):
            L = emb.size(0)
            inputs_embeds[b, :L] = emb
            attention_mask[b, :L] = 1
        return inputs_embeds, attention_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        raw_attention_mask: torch.Tensor,
        labels: torch.Tensor,
        speech_start: torch.Tensor,
        speech_len: torch.Tensor,
        speech_patch_lengths: List[List[int]],
        **kwargs,
    ):
        inputs_embeds, attention_mask = self.build_inputs_embeds(
            input_ids=input_ids,
            raw_attention_mask=raw_attention_mask,
            speech_start=speech_start,
            speech_len=speech_len,
            speech_patch_lengths=speech_patch_lengths,
        )
        return self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )


# -----------------------------
# Debug / parameter utilities
# -----------------------------


def print_trainable_parameters(model: nn.Module):
    total = 0
    trainable = 0
    for _, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    pct = 100.0 * trainable / max(total, 1)
    print(f"trainable params: {trainable:,} / {total:,} ({pct:.4f}%)")


@torch.no_grad()
def debug_sanity(model: LSLMWithFrozenEmbeddingLSTPatching, dataset: Dataset, collator: SpeechPatchCollator, device: torch.device):
    model.eval()
    batch = collator([dataset[i] for i in range(min(2, len(dataset)))])
    batch_t = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    out = model(**batch_t)
    raw_lens = batch_t["raw_attention_mask"].sum(dim=1).tolist()
    num_patches = [len(x) for x in batch_t["speech_patch_lengths"]]
    supervised = [(row != IGNORE_INDEX).sum().item() for row in batch_t["labels"]]

    print("==== DEBUG SANITY ====")
    print("loss_patched_text_only:", float(out.loss.detach().cpu()))
    print("raw_lens:", raw_lens)
    print("num_speech_patches:", num_patches)
    print("num_supervised_text_tokens:", supervised)
    print("input_embedding_trainable:", any(p.requires_grad for p in model.base_model.get_input_embeddings().parameters()))
    print("======================")




class PatchCurriculumCallback(TrainerCallback):
    """Update dataset patching stage during training.

    The schedule is based on training progress:
      [0, align_frac)                    -> align
      [align_frac, align_frac+mixed_frac) -> mixed
      remaining                          -> static

    Use dataloader_num_workers=0 for strict step-level updates, because PyTorch workers
    hold their own dataset copies.
    """

    def __init__(self, dataset: SpeechToTextPatchDataset, total_epochs: float):
        self.dataset = dataset
        self.total_epochs = max(float(total_epochs), 1e-8)
        self._last_stage: Optional[str] = None

    def _update(self, state):
        epoch = 0.0 if state.epoch is None else float(state.epoch)
        progress = max(0.0, min(1.0, epoch / self.total_epochs))
        self.dataset.set_curriculum_progress(progress)
        stage = self.dataset.current_curriculum_stage()
        if stage != self._last_stage:
            print(f"[curriculum] progress={progress:.4f}, stage={stage}")
            self._last_stage = stage

    def on_train_begin(self, args, state, control, **kwargs):
        self._update(state)

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._update(state)

    def on_step_end(self, args, state, control, **kwargs):
        self._update(state)


# -----------------------------
# Main
# -----------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--alignment_jsonl", default=None)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--tokenizer_name_or_path", default=None, help="Tokenizer path. Use the original Qwen tokenizer if the checkpoint tokenizer_config.json is broken.")
    parser.add_argument("--output_dir", default="./outputs/lslm_lst_patch_ft")
    parser.add_argument("--codebook_size", type=int, default=500)

    parser.add_argument("--patch_mode", choices=["static", "align", "mixed", "curriculum"], default="static")
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--max_patch_length", type=int, default=None)
    parser.add_argument("--align_mismatch_policy", choices=["error", "static", "skip"], default="error")
    parser.add_argument("--max_raw_length", type=int, default=None)
    parser.add_argument("--mixed_static_prob", type=float, default=0.5, help="For --patch_mode mixed/curriculum mixed stage, probability of using static patching per sequence.")
    parser.add_argument("--curriculum_align_frac", type=float, default=1.0/3.0, help="Fraction of training progress using align patching in curriculum mode.")
    parser.add_argument("--curriculum_mixed_frac", type=float, default=1.0/3.0, help="Fraction of training progress using mixed patching after the align stage.")

    parser.add_argument("--local_layers", type=int, default=1)
    parser.add_argument("--local_heads", type=int, default=8)
    parser.add_argument("--cross_heads", type=int, default=8)
    parser.add_argument("--pooling", choices=["mean", "avg", "max"], default="mean")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bidirectional_local_encoder", action="store_true")
    parser.add_argument("--cross_gate_init", type=float, default=0.0, help="0.0 starts from pooled z; use -999 to disable gate")
    parser.add_argument("--output_norm", action="store_true")
    parser.add_argument("--patch_encoder_ckpt", default=None)
    parser.add_argument("--allow_resize", action="store_true", help="Allow resizing vocab if checkpoint vocab size does not match speech meta")
    parser.add_argument("--save_safetensors", default=False)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Use 0 for curriculum so dataset stage updates are visible to the collator.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--freeze_lm_head", action="store_true")
    parser.add_argument("--train_input_embeddings", action="store_true", help="One-stage ablation: do not freeze the input embedding matrix (default keeps it frozen).")
    parser.add_argument("--debug_sanity", action="store_true")

    args = parser.parse_args()
    set_seed(args.seed)

    if args.patch_mode in {"align", "mixed", "curriculum"} and args.alignment_jsonl is None:
        raise ValueError("--alignment_jsonl is required for --patch_mode align, mixed, or curriculum")

    if args.patch_mode == "curriculum" and args.dataloader_num_workers != 0:
        print("[WARN] curriculum mode works best with --dataloader_num_workers 0; worker copies may not see stage updates.")

    tokenizer_path = args.tokenizer_name_or_path or args.model_name_or_path
    tokenizer = load_tokenizer_robust(tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    saved_meta = maybe_load_speech_meta(args.model_name_or_path)
    if saved_meta is not None:
        text_vocab_size = int(saved_meta["text_vocab_size"])
        s_id = int(saved_meta["s_id"])
        t_id = int(saved_meta["t_id"])
        hubert_offset = int(saved_meta["hubert_offset"])
        total_vocab_size = int(saved_meta["total_vocab_size"])
        codebook_size = int(saved_meta.get("codebook_size", args.codebook_size))
        print("Loaded speech_token_meta.json from checkpoint")
    else:
        text_vocab_size = len(tokenizer)
        s_id = text_vocab_size
        t_id = text_vocab_size + 1
        hubert_offset = text_vocab_size + 2
        codebook_size = args.codebook_size
        total_vocab_size = text_vocab_size + 2 + codebook_size
        print("No speech_token_meta.json found; inferred speech token ids from tokenizer length")

    print("tokenizer_path:", tokenizer_path)
    print("text_vocab_size:", text_vocab_size)
    print("<s> id:", s_id)
    print("<t> id:", t_id)
    print("hubert_offset:", hubert_offset)
    print("total_vocab_size:", total_vocab_size)
    print("codebook_size:", codebook_size)
    print("patch_mode:", args.patch_mode)
    print("patch_size:", args.patch_size)
    if args.patch_mode in {"mixed", "curriculum"}:
        print("mixed_static_prob:", args.mixed_static_prob)
    if args.patch_mode == "curriculum":
        print("curriculum_align_frac:", args.curriculum_align_frac)
        print("curriculum_mixed_frac:", args.curriculum_mixed_frac)

    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else None
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    )

    current_vocab = base_model.get_input_embeddings().weight.shape[0]
    if current_vocab != total_vocab_size:
        print(f"Resizing token embeddings: {current_vocab} -> {total_vocab_size}")
        base_model.resize_token_embeddings(total_vocab_size)

    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False

    hidden_size = base_model.config.hidden_size
    patch_encoder = LSTSpeechPatchEncoder(
        hidden_size=hidden_size,
        local_layers=args.local_layers,
        local_heads=args.local_heads,
        cross_heads=args.cross_heads,
        dropout=args.dropout,
        pooling=args.pooling,
        causal_local_encoder=not args.bidirectional_local_encoder,
    )
    if dtype is not None:
        patch_encoder = patch_encoder.to(dtype=dtype)

    model = LSLMWithFrozenEmbeddingLSTPatching(
        base_model=base_model,
        patch_encoder=patch_encoder,
        freeze_lm_head=args.freeze_lm_head,
        freeze_input_embeddings=not args.train_input_embeddings,
    )
    print("input_embeddings_frozen:", not args.train_input_embeddings)
    print_trainable_parameters(model)

    train_dataset = SpeechToTextPatchDataset(
        train_jsonl=args.train_jsonl,
        s_id=s_id,
        t_id=t_id,
        patch_mode=args.patch_mode,
        patch_size=args.patch_size,
        alignment_jsonl=args.alignment_jsonl,
        max_patch_length=args.max_patch_length,
        align_mismatch_policy=args.align_mismatch_policy,
        max_raw_length=args.max_raw_length,
        mixed_static_prob=args.mixed_static_prob,
        seed=args.seed,
        curriculum_align_frac=args.curriculum_align_frac,
        curriculum_mixed_frac=args.curriculum_mixed_frac,
    )
    collator = SpeechPatchCollator(pad_token_id=tokenizer.pad_token_id)

    if args.debug_sanity:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        debug_sanity(model, train_dataset, collator, device)
        return

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=args.logging_steps,
        logging_strategy="steps",
        save_strategy="no",
        report_to=["tensorboard"],
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        optim="adamw_torch",
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        max_grad_norm=args.max_grad_norm,
        save_safetensors=args.save_safetensors,
    )

    callbacks = []
    if args.patch_mode == "curriculum":
        callbacks.append(PatchCurriculumCallback(train_dataset, total_epochs=args.epochs))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )
    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    base_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    torch.save(patch_encoder.state_dict(), os.path.join(args.output_dir, "speech_patch_encoder.pt"))

    meta = {
        "model_name_or_path": args.model_name_or_path,
        "text_vocab_size": text_vocab_size,
        "s_id": s_id,
        "t_id": t_id,
        "hubert_offset": hubert_offset,
        "total_vocab_size": total_vocab_size,
        "codebook_size": codebook_size,
        "patch_mode": args.patch_mode,
        "patch_size": args.patch_size,
        "max_patch_length": args.max_patch_length,
        "mixed_static_prob": args.mixed_static_prob,
        "curriculum_align_frac": args.curriculum_align_frac,
        "curriculum_mixed_frac": args.curriculum_mixed_frac,
        "pooling": args.pooling,
        "local_layers": args.local_layers,
        "local_heads": args.local_heads,
        "cross_heads": args.cross_heads,
        "bidirectional_local_encoder": args.bidirectional_local_encoder,
        "input_embeddings_frozen": not args.train_input_embeddings,
        "freeze_lm_head": args.freeze_lm_head,
        "objective": "speech_patches_to_text_only",
        "patch_encoder_state": "speech_patch_encoder.pt",
    }
    with open(os.path.join(args.output_dir, "speech_token_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Training finished.")
    print("Saved to:", args.output_dir)


if __name__ == "__main__":
    main()
