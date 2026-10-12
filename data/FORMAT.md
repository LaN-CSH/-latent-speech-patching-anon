# Data formats

Data files are not included in this repository.

## Training jsonl (`--train_jsonl`, both stages)

One utterance per line. LibriSpeech train-clean-100; the first part of each utterance is given as speech units and the rest of the transcript is the text target.

| Field | Meaning |
|---|---|
| `utt_id` | LibriSpeech utterance id |
| `type` | `s_plus_t_to_t` |
| `split_word_idx` | index of the first word of the text target |
| `speech_text` | transcript of the speech part |
| `target_text` | text target |
| `input_ids` | `[S]` id, speech-unit ids (HuBERT unit + offset), `[T]` id, text ids |
| `labels` | `-100` on the speech part, text ids on the target |

Speech-unit ids are `text_vocab_size + 2 + unit`, where the two added ids are `[S]` and `[T]`.

Example (ids shortened):

```json
{"utt_id": "103-1240-0", "type": "s_plus_t_to_t", "split_word_idx": 23, "speech_text": "chapter one missus rachel lynde is surprised ...", "target_text": "fringed with alders and ladies eardrops and traversed by a brook", "input_ids": [128256, 128365, 128365, 128653, "..."], "labels": [-100, -100, -100, -100, "..."]}
```

## Word alignment jsonl (`--alignment_jsonl`, Stage 2)

One utterance per line, word boundaries in 50 Hz HuBERT frames. Used by the aligned phase of the curriculum.

```json
{"utt_id": "103-1240-0", "transcript": "CHAPTER ONE ...", "words": [{"word": "chapter", "start": 0.4802, "end": 0.8003, "hubert_start": 24, "hubert_end": 40}, {"word": "one", "start": 0.9003, "end": 1.0604, "hubert_start": 45, "hubert_end": 53}]}
```

## spoken HellaSwag (`--meta_dir`, `--audio_dir`)

`--meta_dir` holds `meta_*.jsonl`; `--audio_dir` holds one 16 kHz wav per item named `<id>.wav`. The same `--meta_dir` is used by `eval_text_hellaswag.py`.

```json
{"id": "validation_0000000", "context": "A man is sitting on a roof. he", "endings": ["is using wrap to wrap a pair of skis.", "..."], "label": 3, "split": "validation"}
```

## spoken StoryCloze (`--audio_dir`, `--text_dir`)

For each item, four files: `<name>_correct.wav`, `<name>_correct.txt`, `<name>_incorrect.wav`, `<name>_incorrect.txt`. Each wav/txt holds the full five-sentence story with the correct or the incorrect ending.

## Per-item output

Every evaluation script writes one json line per item with at least `id` and `correct` (0/1).
