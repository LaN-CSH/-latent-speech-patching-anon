# eval_shellaswag.py
#
# Baseline sHellaSwag evaluation with raw speech-token prefix.
# Adds optional compute-cost accounting compatible with the patched-cost eval script.

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import librosa
import torch
import torch.nn as nn
import fairseq
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

IGNORE_INDEX = -100


def load_jsonl_files(meta_dir):
    rows = []
    for path in sorted(Path(meta_dir).glob("meta_*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def load_meta(model_dir):
    with open(Path(model_dir) / "speech_token_meta.json", "r") as f:
        return json.load(f)


def load_hubert_model(ckpt_path, device):
    models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    model = models[0].to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_hubert_tokens(hubert_model, kmeans, wav_path, device, output_layer=9):
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


def speech_ids(hubert_tokens, hubert_offset, codebook_size):
    return [hubert_offset + int(x) for x in hubert_tokens if 0 <= int(x) < codebook_size]


def text_ids(tokenizer, text):
    text = " ".join(text.lower().strip().split())
    return tokenizer(text, add_special_tokens=False)["input_ids"]


@torch.no_grad()
def score_option(model, input_ids, labels, device):
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    labels = torch.tensor([labels], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    out = model(
        input_ids=input_ids,
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

    return avg_logprob, avg_nll, n_tokens, n_input_tokens


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

    Includes attention/MLP matmuls and the LM head projection. Excludes embeddings,
    norms, activations, softmax, loss computation, HuBERT, k-means, and IO overhead.
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

    # 2 FLOPs per multiply-add. Dense full-sequence self-attention estimate.
    qkvo_proj = 8.0 * seq_len * hidden * hidden
    attention_scores_and_values = 4.0 * seq_len * seq_len * hidden
    mlp = (6.0 if gated_mlp else 4.0) * seq_len * hidden * intermediate
    lm_head = 2.0 * seq_len * hidden * vocab if vocab > 0 else 0.0

    return int(round(n_layers * (qkvo_proj + attention_scores_and_values + mlp) + lm_head))


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--base_model_name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--meta_dir", required=True)
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--hubert_ckpt", required=True)
    parser.add_argument("--km_path", required=True)
    parser.add_argument("--output_jsonl", default="shellaswag_eval_results.jsonl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hubert_layer", type=int, default=9)
    parser.add_argument("--max_speech_tokens", type=int, default=1600)
    parser.add_argument("--bf16", action="store_true")

    parser.add_argument(
        "--compute_cost",
        action="store_true",
        help=(
            "Collect compute-cost metrics: wall-clock time, option forward count, "
            "token counts, CUDA peak memory, and approximate LM FLOPs."
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

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    meta = load_meta(args.model_dir)
    s_id = int(meta["s_id"])
    t_id = int(meta["t_id"])
    hubert_offset = int(meta["hubert_offset"])
    total_vocab_size = int(meta["total_vocab_size"])
    codebook_size = int(meta["codebook_size"])

    print("model_dir:", args.model_dir)
    print("base_model_name:", args.base_model_name)
    print("text_vocab_size:", meta.get("text_vocab_size"))
    print("<s> id:", s_id)
    print("<t> id:", t_id)
    print("hubert_offset:", hubert_offset)
    print("total_vocab_size:", total_vocab_size)
    print("codebook_size:", codebook_size)
    print("max_speech_tokens:", args.max_speech_tokens)
    print("compute_cost:", compute_cost)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
    )

    if model.get_input_embeddings().weight.shape[0] != total_vocab_size:
        model.resize_token_embeddings(total_vocab_size)

    model.to(device)
    model.eval()

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
            "variant": "baseline_raw_speech_prefix",
            "notes": [
                "Wall-clock timings include HuBERT feature extraction, k-means, CausalLM forward, loss, and Python/IO overhead.",
                "FLOPs are approximate dense matmul estimates and exclude HuBERT, k-means, softmax, norms, activations, loss, and IO.",
                "Baseline uses raw speech-token prefix; no patch encoder is run, so patch_encoder_forward FLOPs are zero.",
                "The model is run once per answer option, matching the original scoring flow.",
            ],
            "config": {
                "model_dir": str(args.model_dir),
                "base_model_name": str(args.base_model_name),
                "max_speech_tokens": args.max_speech_tokens,
                "device": device,
                "dtype": "bf16" if args.bf16 else "fp32/default",
                "skip_flop_estimate": args.skip_flop_estimate,
            },
            "model": {
                "base_model_params": count_parameters(model),
                "patch_encoder_params": 0,
                "total_loaded_scored_params": count_parameters(model),
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
            example_lm_flops = 0

            wav_path = audio_dir / f"{ex_id}.wav"
            if not wav_path.exists():
                print(f"[WARN] missing audio: {wav_path}")
                if cost is not None:
                    cost["counts"]["missing_audio_examples"] += 1
                continue

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

            sp_ids = speech_ids(
                hubert_tokens,
                hubert_offset,
                codebook_size,
            )
            prefix_ids = [s_id] + sp_ids + [t_id]

            scores = []

            for ending in ex["endings"]:
                option_ids = text_ids(tokenizer, ending)

                input_ids = prefix_ids + option_ids
                labels = [IGNORE_INDEX] * len(prefix_ids) + option_ids

                if compute_cost:
                    sync_if_cuda(device)
                    scoring_start = time.perf_counter()

                avg_logprob, avg_nll, n_tokens, n_input_tokens = score_option(
                    model,
                    input_ids,
                    labels,
                    device,
                )

                lm_flops = 0
                if compute_cost:
                    sync_if_cuda(device)
                    option_scoring_seconds = time.perf_counter() - scoring_start
                    example_scoring_seconds += option_scoring_seconds
                    assert cost is not None
                    cost["timing_seconds"]["option_scoring"] += option_scoring_seconds

                    if not args.skip_flop_estimate:
                        lm_flops = estimate_causal_lm_forward_flops(
                            model.config,
                            seq_len=n_input_tokens,
                            vocab_size=total_vocab_size,
                        )
                        cost["estimated_flops"]["causal_lm_forward"] += lm_flops
                        example_lm_flops += lm_flops

                    cost["counts"]["option_forwards"] += 1
                    cost["counts"]["scored_text_tokens"] += int(n_tokens)
                    cost["counts"]["input_tokens_equivalent"] += int(n_input_tokens)
                    cost["counts"]["option_conditioning_speech_tokens"] += int(len(sp_ids))
                    example_scored_tokens += int(n_tokens)
                    example_input_tokens += int(n_input_tokens)

                score_row = {
                    "ending": ending,
                    "avg_logprob": avg_logprob,
                    "avg_nll": avg_nll,
                    "n_tokens": n_tokens,
                    "n_input_tokens": n_input_tokens,
                    "n_speech_tokens": len(sp_ids),
                    "n_speech_patches": 0,
                }
                if compute_cost and args.record_example_cost:
                    score_row["estimated_lm_forward_flops"] = lm_flops
                    score_row["estimated_patch_encoder_flops"] = 0
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
                        "option_conditioning_speech_patches": 0,
                        "estimated_lm_forward_flops": example_lm_flops,
                        "estimated_patch_encoder_flops": 0,
                        "estimated_total_flops": example_lm_flops,
                    }

            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    if compute_cost:
        sync_if_cuda(device)

    acc = correct / max(total, 1)

    print("==== sHellaSwag Results ====")
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
