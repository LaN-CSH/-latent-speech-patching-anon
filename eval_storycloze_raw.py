#!/usr/bin/env python3
"""
Evaluate paired StoryCloze speech benchmark with raw speech-token prefix.

Expected files per problem:
  0_<problem_id>_correct.wav
  1_<problem_id>_incorrect.wav
and matching text files:
  0_<problem_id>_correct.txt
  1_<problem_id>_incorrect.txt

Each wav may contain all 5 story sentences. This script uses wav2vec2+CTC
forced alignment on the first 4 text sentences, cuts the selected wav to that
4-sentence prefix in memory, extracts HuBERT+kmeans speech tokens from the cut
prefix, then scores the two candidate fifth sentences by normalized log
likelihood.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import fairseq
import joblib
import librosa
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

IGNORE_INDEX = -100
PAIR_RE = re.compile(r"^(?P<num>\d+)_(?P<qid>.+)_(?P<label>correct|incorrect)(?:\(\d+\))?$")


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class PairExample:
    qid: str
    correct_wav: Path
    incorrect_wav: Path
    correct_txt: Path
    incorrect_txt: Path
    context: str
    endings: List[Dict[str, Any]]


def clean_stem(path: Path) -> str:
    return re.sub(r"\(\d+\)$", "", path.stem)


def parse_pair_stem(path: Path) -> Optional[Tuple[str, str, int]]:
    m = PAIR_RE.match(clean_stem(path))
    if not m:
        return None
    return m.group("qid"), m.group("label"), int(m.group("num"))


def normalize_text(s: str) -> str:
    return " ".join(s.lower().strip().split())


def split_sentences(text: str) -> List[str]:
    text = " ".join(text.strip().split())
    if not text:
        return []
    parts = re.findall(r"[^.!?]+[.!?]+[\"']?|[^.!?]+$", text)
    return [p.strip() for p in parts if p.strip()]


def split_context_ending(text: str, n_context_sentences: int = 4) -> Tuple[str, str, List[str]]:
    sents = split_sentences(text)
    if len(sents) < n_context_sentences + 1:
        raise ValueError(f"Expected at least {n_context_sentences + 1} sentences, got {len(sents)}: {text[:200]!r}")
    context = " ".join(sents[:n_context_sentences])
    ending = " ".join(sents[n_context_sentences:])
    return context, ending, sents


def find_text_for_wav(wav_path: Path, text_dir: Path, label: str, qid: str) -> Optional[Path]:
    candidates = [text_dir / f"{clean_stem(wav_path)}.txt", wav_path.with_suffix(".txt")]
    for c in candidates:
        if c.is_file():
            return c
    matches = sorted(text_dir.rglob(f"*_{qid}_{label}*.txt"))
    return matches[0] if matches else None


def collect_pair_examples(
    audio_dir: Path,
    text_dir: Path,
    n_context_sentences: int = 4,
    strict_context_match: bool = False,
    limit: Optional[int] = None,
) -> List[PairExample]:
    groups: Dict[str, Dict[str, Tuple[Path, int]]] = {}
    for wav in sorted(audio_dir.rglob("*.wav")):
        parsed = parse_pair_stem(wav)
        if parsed is None:
            continue
        qid, label, num = parsed
        groups.setdefault(qid, {})[label] = (wav, num)

    examples: List[PairExample] = []
    n_missing = 0
    n_bad = 0
    for qid, g in sorted(groups.items(), key=lambda kv: min(v[1] for v in kv[1].values())):
        if "correct" not in g or "incorrect" not in g:
            n_missing += 1
            continue
        correct_wav = g["correct"][0]
        incorrect_wav = g["incorrect"][0]
        correct_txt = find_text_for_wav(correct_wav, text_dir, "correct", qid)
        incorrect_txt = find_text_for_wav(incorrect_wav, text_dir, "incorrect", qid)
        if correct_txt is None or incorrect_txt is None:
            n_missing += 1
            continue
        try:
            correct_text = correct_txt.read_text(encoding="utf-8").strip()
            incorrect_text = incorrect_txt.read_text(encoding="utf-8").strip()
            c_ctx, c_end, _ = split_context_ending(correct_text, n_context_sentences=n_context_sentences)
            i_ctx, i_end, _ = split_context_ending(incorrect_text, n_context_sentences=n_context_sentences)
            if normalize_text(c_ctx) != normalize_text(i_ctx):
                msg = f"context mismatch for {qid}: {correct_txt} vs {incorrect_txt}"
                if strict_context_match:
                    raise ValueError(msg)
                print(f"[WARN] {msg}")
            examples.append(
                PairExample(
                    qid=qid,
                    correct_wav=correct_wav,
                    incorrect_wav=incorrect_wav,
                    correct_txt=correct_txt,
                    incorrect_txt=incorrect_txt,
                    context=c_ctx,
                    endings=[
                        {"label_name": "correct", "text": c_end, "source_wav": str(correct_wav), "source_txt": str(correct_txt)},
                        {"label_name": "incorrect", "text": i_end, "source_wav": str(incorrect_wav), "source_txt": str(incorrect_txt)},
                    ],
                )
            )
        except Exception as e:
            n_bad += 1
            print(f"[WARN] failed to parse pair {qid}: {type(e).__name__}: {e}")
        if limit is not None and len(examples) >= limit:
            break

    print(f"pairs loaded: {len(examples)}")
    if n_missing:
        print(f"pairs skipped due to missing counterpart/text: {n_missing}")
    if n_bad:
        print(f"pairs skipped due to parse errors: {n_bad}")
    return examples


def choose_speech_wav(ex: PairExample, source: str) -> Path:
    if source == "correct":
        return ex.correct_wav
    if source == "incorrect":
        return ex.incorrect_wav
    if source == "first":
        paths = sorted([ex.correct_wav, ex.incorrect_wav], key=lambda p: parse_pair_stem(p)[2])  # type: ignore[index]
        return paths[0]
    raise ValueError(f"unknown speech_source: {source}")


# -----------------------------
# wav2vec2 CTC prefix alignment
# -----------------------------


def normalize_for_ctc(text: str, labels: Sequence[str]) -> str:
    label_set = set(labels)
    text = text.upper().replace("’", "'").replace("`", "'")
    chars: List[str] = []
    for ch in text:
        if ch == " ":
            chars.append(" ")
        elif ch in label_set and ch != "|":
            chars.append(ch)
        elif ch == "'" and "'" in label_set:
            chars.append(ch)
        elif "A" <= ch <= "Z" and ch in label_set:
            chars.append(ch)
    norm = "".join(chars)
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm


def transcript_to_token_ids(normalized_text: str, dictionary: Dict[str, int]) -> Tuple[List[int], str]:
    ctc_text = normalized_text.replace(" ", "|")
    token_ids: List[int] = []
    kept_chars: List[str] = []
    for ch in ctc_text:
        if ch in dictionary:
            token_ids.append(dictionary[ch])
            kept_chars.append(ch)
    return token_ids, "".join(kept_chars)


def load_wav_16k(wav_path: Path, sample_rate: int = 16000) -> Tuple[np.ndarray, float]:
    wav, _ = librosa.load(str(wav_path), sr=sample_rate, mono=True)
    wav = wav.astype(np.float32, copy=False)
    duration = float(len(wav)) / float(sample_rate)
    return wav, duration


@torch.no_grad()
def get_emission_from_wav(wav2vec_model, wav: np.ndarray, device: str) -> torch.Tensor:
    waveform = torch.from_numpy(wav).float().unsqueeze(0).to(device)
    emissions, _ = wav2vec_model(waveform)
    emissions = torch.log_softmax(emissions, dim=-1)
    return emissions[0].cpu()


def get_trellis(emission: torch.Tensor, tokens: List[int], blank_id: int) -> torch.Tensor:
    num_frames = emission.size(0)
    num_tokens = len(tokens)
    trellis = torch.empty((num_frames + 1, num_tokens + 1), dtype=emission.dtype)
    trellis[0, 0] = 0.0
    trellis[0, 1:] = -float("inf")
    trellis[1:, 0] = torch.cumsum(emission[:, blank_id], dim=0)
    for t in range(num_frames):
        stay = trellis[t, 1:] + emission[t, blank_id]
        change = trellis[t, :-1] + emission[t, tokens]
        trellis[t + 1, 1:] = torch.maximum(stay, change)
    return trellis


def backtrack(trellis: torch.Tensor, emission: torch.Tensor, tokens: List[int], blank_id: int) -> List[Point]:
    j = trellis.size(1) - 1
    t = int(torch.argmax(trellis[:, j]).item())
    path: List[Point] = []
    while j > 0 and t > 0:
        p_stay = emission[t - 1, blank_id]
        p_change = emission[t - 1, tokens[j - 1]]
        stayed = trellis[t - 1, j] + p_stay
        changed = trellis[t - 1, j - 1] + p_change
        if changed > stayed:
            prob = float(torch.exp(p_change).item())
            path.append(Point(token_index=j - 1, time_index=t - 1, score=prob))
            j -= 1
        t -= 1
    if j != 0:
        raise RuntimeError("Failed to align prefix transcript with wav2vec2 CTC emissions")
    return path[::-1]


def merge_repeats(path: List[Point], transcript: str) -> List[Segment]:
    if not path:
        return []
    segments: List[Segment] = []
    i1 = 0
    while i1 < len(path):
        i2 = i1
        while i2 < len(path) and path[i2].token_index == path[i1].token_index:
            i2 += 1
        score = sum(p.score for p in path[i1:i2]) / max(i2 - i1, 1)
        token_idx = path[i1].token_index
        segments.append(Segment(transcript[token_idx], path[i1].time_index, path[i2 - 1].time_index + 1, score))
        i1 = i2
    return segments


def merge_words(segments: List[Segment], separator: str = "|") -> List[Segment]:
    words: List[Segment] = []
    i1 = 0
    while i1 < len(segments):
        if segments[i1].label == separator:
            i1 += 1
            continue
        i2 = i1
        while i2 < len(segments) and segments[i2].label != separator:
            i2 += 1
        if i2 > i1:
            word = "".join(seg.label for seg in segments[i1:i2])
            total_len = sum(seg.length for seg in segments[i1:i2])
            score = sum(seg.score * seg.length for seg in segments[i1:i2]) / max(total_len, 1)
            words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
        i1 = i2
    return words


def align_prefix_and_trim(
    wav2vec_model,
    labels: Sequence[str],
    dictionary: Dict[str, int],
    wav_path: Path,
    prefix_text: str,
    device: str,
    sample_rate: int = 16000,
    hubert_hz: float = 50.0,
    trim_pad_sec: float = 0.08,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    wav, duration = load_wav_16k(wav_path, sample_rate=sample_rate)
    normalized = normalize_for_ctc(prefix_text, labels)
    token_ids, ctc_text = transcript_to_token_ids(normalized, dictionary)
    if not token_ids:
        raise RuntimeError("CTC-normalized prefix transcript is empty")

    emission = get_emission_from_wav(wav2vec_model, wav, device)
    trellis = get_trellis(emission, token_ids, blank_id=0)
    path = backtrack(trellis, emission, token_ids, blank_id=0)
    char_segments = merge_repeats(path, ctc_text)
    word_segments = merge_words(char_segments, separator="|")
    if not word_segments:
        raise RuntimeError("No word segments produced by CTC alignment")

    ratio = duration / max(emission.size(0), 1)
    raw_context_end_sec = min(duration, max(0.0, word_segments[-1].end * ratio))
    context_end_sec = min(duration, raw_context_end_sec + trim_pad_sec)
    cut_samples = max(1, min(len(wav), int(round(context_end_sec * sample_rate))))
    trimmed = wav[:cut_samples]
    trimmed_duration = float(len(trimmed)) / float(sample_rate)
    num_hubert_frames_est = max(1, int(round(trimmed_duration * hubert_hz)))

    out_words: List[Dict[str, Any]] = []
    for seg in word_segments:
        start = max(0.0, min(trimmed_duration, seg.start * ratio))
        end = max(start, min(trimmed_duration, seg.end * ratio))
        hs = max(0, min(num_hubert_frames_est, int(round(start * hubert_hz))))
        he = max(0, min(num_hubert_frames_est, int(round(end * hubert_hz))))
        if hs < num_hubert_frames_est:
            he = max(he, hs + 1)
        he = min(num_hubert_frames_est, he)
        out_words.append({
            "word": seg.label.lower(),
            "start": round(start, 4),
            "end": round(end, 4),
            "hubert_start": int(hs),
            "hubert_end": int(he),
            "score": round(float(seg.score), 4),
        })

    info = {
        "full_audio_duration_sec": round(duration, 4),
        "raw_context_end_sec": round(raw_context_end_sec, 4),
        "context_end_sec": round(context_end_sec, 4),
        "trimmed_duration_sec": round(trimmed_duration, 4),
        "normalized_context": normalized,
        "num_ctc_frames": int(emission.size(0)),
        "num_hubert_frames_est": int(num_hubert_frames_est),
        "words": out_words,
    }
    return trimmed, info


def save_trimmed_wav_if_requested(trimmed: np.ndarray, sample_rate: int, out_dir: Optional[Path], qid: str) -> Optional[str]:
    if out_dir is None:
        return None
    import soundfile as sf
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{qid}_context4.wav"
    sf.write(str(path), trimmed, sample_rate)
    return str(path)


# -----------------------------
# Model/audio utilities
# -----------------------------


def load_meta(model_dir_or_file: str) -> Dict[str, Any]:
    p = Path(model_dir_or_file)
    if p.is_dir():
        p = p / "speech_token_meta.json"
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_tokenizer_robust(tokenizer_name_or_path: str):
    try:
        return AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    except AttributeError as e:
        if "extra_special_tokens" in str(e):
            raise RuntimeError(
                f"Failed to load tokenizer from {tokenizer_name_or_path!r}. "
                "Use --tokenizer_name_or_path Qwen/Qwen2.5-0.5B if the checkpoint tokenizer_config.json is broken."
            ) from e
        raise


def load_hubert_model(ckpt_path: str, device: str):
    models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    model = models[0].to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_hubert_tokens_from_wav(hubert_model, kmeans, wav: np.ndarray, device: str, output_layer: int = 9) -> List[int]:
    if wav.size == 0:
        return []
    wav_tensor = torch.from_numpy(wav.astype(np.float32, copy=False)).float().unsqueeze(0).to(device)
    feats, _ = hubert_model.extract_features(source=wav_tensor, padding_mask=None, output_layer=output_layer)
    feats = feats.squeeze(0).cpu().numpy()
    return kmeans.predict(feats).tolist()


def speech_ids(hubert_tokens: List[int], hubert_offset: int, codebook_size: int) -> List[int]:
    return [hubert_offset + int(x) for x in hubert_tokens if 0 <= int(x) < codebook_size]


def text_ids(tokenizer, text: str, add_leading_space: bool = False) -> List[int]:
    text = normalize_text(text)
    if add_leading_space and text:
        text = " " + text
    return tokenizer(text, add_special_tokens=False)["input_ids"]


@torch.no_grad()
def score_option(model, input_ids: List[int], labels: List[int], device: str) -> Tuple[float, float, int]:
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    label_tensor = torch.tensor([labels], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_tensor)
    out = model(input_ids=input_tensor, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]
    shift_labels = label_tensor[:, 1:]
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="sum")
    nll = loss_fct(logits.reshape(-1, logits.size(-1)), shift_labels.reshape(-1))
    n_tokens = int((shift_labels != IGNORE_INDEX).sum().item())
    avg_nll = float(nll.item()) / max(n_tokens, 1)
    return -avg_nll, avg_nll, n_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Speech-token LSLM checkpoint containing speech_token_meta.json")
    parser.add_argument("--tokenizer_name_or_path", default=None, help="Defaults to --model_dir. Use Qwen/Qwen2.5-0.5B if checkpoint tokenizer is broken.")
    parser.add_argument("--audio_dir", required=True, help="Directory containing *_correct.wav and *_incorrect.wav")
    parser.add_argument("--text_dir", default=None, help="Directory containing matching txt files. Defaults to --audio_dir")
    parser.add_argument("--hubert_ckpt", required=True)
    parser.add_argument("--km_path", required=True)
    parser.add_argument("--output_jsonl", default="storycloze_eval_results.jsonl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hubert_layer", type=int, default=9)
    parser.add_argument("--max_speech_tokens", type=int, default=1600)
    parser.add_argument("--speech_source", choices=["correct", "incorrect", "first"], default="correct", help="Which full 5-sentence wav to align and trim to 4-sentence speech context")
    parser.add_argument("--speech_control", choices=["real", "shuffled", "no_speech"], default="real")
    parser.add_argument("--n_context_sentences", type=int, default=4)
    parser.add_argument("--trim_pad_sec", type=float, default=0.08)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--hubert_hz", type=float, default=50.0)
    parser.add_argument("--trim_cache_dir", default=None, help="Optional directory to save the CTC-trimmed 4-sentence wavs")
    parser.add_argument("--on_alignment_error", choices=["skip", "error", "full_audio"], default="skip", help="full_audio may leak the answer; use only for debugging")
    parser.add_argument("--add_leading_space", action="store_true")
    parser.add_argument("--shuffle_options", action="store_true")
    parser.add_argument("--strict_context_match", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--allow_resize", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    meta = load_meta(args.model_dir)
    s_id = int(meta["s_id"])
    t_id = int(meta["t_id"])
    hubert_offset = int(meta["hubert_offset"])
    total_vocab_size = int(meta["total_vocab_size"])
    codebook_size = int(meta["codebook_size"])

    tokenizer_path = args.tokenizer_name_or_path or args.model_dir
    tokenizer = load_tokenizer_robust(tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else None
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, trust_remote_code=True, torch_dtype=dtype)
    current_vocab = model.get_input_embeddings().weight.shape[0]
    if current_vocab != total_vocab_size:
        if not args.allow_resize:
            raise ValueError(f"Model vocab size {current_vocab} != total_vocab_size {total_vocab_size}. Use --allow_resize only intentionally.")
        model.resize_token_embeddings(total_vocab_size)
    model.to(device)
    model.eval()

    hubert_model = load_hubert_model(args.hubert_ckpt, device)
    kmeans = joblib.load(args.km_path)

    print("Loading wav2vec2 CTC model: torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H")
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    wav2vec_model = bundle.get_model().to(device).eval()
    ctc_labels = bundle.get_labels()
    ctc_dictionary = {c: i for i, c in enumerate(ctc_labels)}

    examples = collect_pair_examples(
        audio_dir=Path(args.audio_dir),
        text_dir=Path(args.text_dir or args.audio_dir),
        n_context_sentences=args.n_context_sentences,
        strict_context_match=args.strict_context_match,
        limit=args.limit,
    )

    trim_cache_dir = Path(args.trim_cache_dir) if args.trim_cache_dir else None
    correct = 0
    total = 0
    skipped = 0
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out:
        for ex in tqdm(examples):
            speech_wav_path = choose_speech_wav(ex, args.speech_source)
            trim_info: Dict[str, Any] = {}
            trimmed_wav_path: Optional[str] = None

            if args.speech_control == "no_speech":
                sp_ids: List[int] = []
                hubert_tokens: List[int] = []
            else:
                try:
                    trimmed_wav, trim_info = align_prefix_and_trim(
                        wav2vec_model=wav2vec_model,
                        labels=ctc_labels,
                        dictionary=ctc_dictionary,
                        wav_path=speech_wav_path,
                        prefix_text=ex.context,
                        device=device,
                        sample_rate=args.sample_rate,
                        hubert_hz=args.hubert_hz,
                        trim_pad_sec=args.trim_pad_sec,
                    )
                    trimmed_wav_path = save_trimmed_wav_if_requested(trimmed_wav, args.sample_rate, trim_cache_dir, ex.qid)
                except Exception as e:
                    msg = f"[WARN] CTC trim failed for {ex.qid} ({speech_wav_path}): {type(e).__name__}: {e}"
                    print(msg)
                    if args.on_alignment_error == "error":
                        raise
                    if args.on_alignment_error == "skip":
                        skipped += 1
                        continue
                    trimmed_wav, dur = load_wav_16k(speech_wav_path, sample_rate=args.sample_rate)
                    trim_info = {"alignment_error": msg, "full_audio_duration_sec": dur, "context_end_sec": dur, "used_full_audio_with_possible_leakage": True, "words": []}

                hubert_tokens = extract_hubert_tokens_from_wav(hubert_model, kmeans, trimmed_wav, device, args.hubert_layer)
                if args.max_speech_tokens is not None:
                    hubert_tokens = hubert_tokens[: args.max_speech_tokens]
                if args.speech_control == "shuffled":
                    rng = random.Random(f"{args.seed}-{ex.qid}")
                    rng.shuffle(hubert_tokens)
                sp_ids = speech_ids(hubert_tokens, hubert_offset, codebook_size)

            prefix_ids = [s_id] + sp_ids + [t_id]
            endings = list(ex.endings)
            if args.shuffle_options:
                random.Random(f"{args.seed}-{ex.qid}-options").shuffle(endings)

            scores = []
            for opt_idx, opt in enumerate(endings):
                option_ids = text_ids(tokenizer, opt["text"], add_leading_space=args.add_leading_space)
                input_ids = prefix_ids + option_ids
                labels = [IGNORE_INDEX] * len(prefix_ids) + option_ids
                avg_logprob, avg_nll, n_tokens = score_option(model, input_ids, labels, device)
                scores.append({
                    "option_index": opt_idx,
                    "label_name": opt["label_name"],
                    "ending": opt["text"],
                    "avg_logprob": avg_logprob,
                    "avg_nll": avg_nll,
                    "n_tokens": n_tokens,
                    "source_txt": opt["source_txt"],
                })

            pred_idx = max(range(len(scores)), key=lambda i: scores[i]["avg_logprob"])
            pred_label_name = scores[pred_idx]["label_name"]
            is_correct = int(pred_label_name == "correct")
            correct += is_correct
            total += 1

            out.write(json.dumps({
                "id": ex.qid,
                "label_name": "correct",
                "pred_label_name": pred_label_name,
                "pred_index": pred_idx,
                "correct": is_correct,
                "context_text": ex.context,
                "speech_wav_full": str(speech_wav_path) if args.speech_control != "no_speech" else None,
                "speech_wav_trimmed": trimmed_wav_path,
                "speech_source": args.speech_source,
                "speech_control": args.speech_control,
                "trim_info": trim_info,
                "n_hubert_tokens": len(hubert_tokens),
                "n_speech_tokens": len(sp_ids),
                "scores": scores,
            }, ensure_ascii=False) + "\n")

    acc = correct / max(total, 1)
    print("==== StoryCloze speech Results ====")
    print("total:", total)
    print("correct:", correct)
    print("accuracy:", acc)
    print("skipped:", skipped)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
