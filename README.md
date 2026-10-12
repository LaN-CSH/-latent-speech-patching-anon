# Latent Speech Patching

Anonymous code release for ARR submission. This repository currently contains the training and evaluation code needed to reproduce Table 1 (raw discrete baseline and our two-stage latent patching model). **It will be updated progressively** — ablation variants, checkpoints, and processed evaluation sets will be added upon acceptance.

## Setup

```bash
pip install -r requirements.txt
```

- Backbones: `meta-llama/Llama-3.2-1B` and `meta-llama/Llama-3.2-3B` (gated on the Hugging Face Hub).
- Speech units: HuBERT-base layer 9, 50 Hz, k-means with 500 clusters. The evaluation scripts take the HuBERT checkpoint (`--hubert_ckpt`) and the k-means model (`--km_path`) and extract units on the fly.
- Training data: LibriSpeech train-clean-100, converted to the jsonl format described in `data/FORMAT.md`. Data files are not included.

## Files

| File | Purpose |
|---|---|
| `train_stage1.py` | Stage 1: raw discrete adaptation (text embedding rows frozen) |
| `train_stage2.py` | Stage 2: patch encoder + LLM (`--patch_mode curriculum --patch_size 8`) |
| `eval_shellaswag.py` / `eval_shellaswag_raw.py` | spoken HellaSwag likelihood scoring, patched model / raw baseline |
| `eval_storycloze.py` / `eval_storycloze_raw.py` | spoken StoryCloze, patched model / raw baseline |
| `eval_text_hellaswag.py` | text-only HellaSwag |
| `configs/*.yaml` | hyperparameters of the runs reported in Table 1 (reference; the scripts take command-line arguments) |
| `data/FORMAT.md` | jsonl formats of the training and evaluation data |

Each evaluation script writes one json line per item (`--output_jsonl`). The patch encoder is defined inside `train_stage2.py` and the patched evaluation scripts, so no separate module directory is needed.

## Reproduce (3B, seed 42)

```bash
# 1. Stage 1: raw discrete adaptation
python train_stage1.py --train_jsonl data/train.jsonl --model_name meta-llama/Llama-3.2-3B --output_dir ckpt/3b_stage1 --codebook_size 500 --epochs 2 --lr 1e-5 --batch_size 2 --grad_accum 8 --bf16 --seed 42

# 2. Stage 2: latent patching, initialised from the Stage-1 checkpoint
python train_stage2.py --train_jsonl data/train.jsonl --alignment_jsonl data/word_alignments_50hz.jsonl --model_name_or_path ckpt/3b_stage1 --tokenizer_name_or_path meta-llama/Llama-3.2-3B --output_dir ckpt/3b_stage2 --codebook_size 500 --patch_mode curriculum --patch_size 8 --mixed_static_prob 0.5 --curriculum_align_frac 0.333333 --curriculum_mixed_frac 0.333333 --pooling mean --local_layers 1 --local_heads 8 --cross_heads 8 --epochs 2 --lr 1e-5 --weight_decay 0.0 --max_grad_norm 0.3 --batch_size 2 --grad_accum 8 --bf16 --dataloader_num_workers 0 --seed 42

# 3. spoken HellaSwag
python eval_shellaswag.py --model_dir ckpt/3b_stage2 --tokenizer_name_or_path meta-llama/Llama-3.2-3B --meta_dir data/shellaswag --audio_dir data/shellaswag/audio_16k --hubert_ckpt data/hubert_base_ls960.pt --km_path data/hubert_base_ls960_L9_km500.bin --patch_size 8 --pooling mean --local_layers 1 --local_heads 8 --cross_heads 8 --output_jsonl out/3b_shellaswag.jsonl --device cuda --bf16

# 4. spoken StoryCloze
python eval_storycloze.py --model_dir ckpt/3b_stage2 --tokenizer_name_or_path meta-llama/Llama-3.2-3B --audio_dir data/sSC --text_dir data/sSC --hubert_ckpt data/hubert_base_ls960.pt --km_path data/hubert_base_ls960_L9_km500.bin --patch_mode static --patch_size 8 --output_jsonl out/3b_storycloze.jsonl --device cuda --bf16

# 5. text HellaSwag
python eval_text_hellaswag.py --model_name ckpt/3b_stage2 --meta_dir data/shellaswag --output_jsonl out/3b_text_hellaswag.jsonl --device cuda --bf16
```

Table 1 reports means over three seeds (42/43/44); the raw baseline is the Stage-1 checkpoint evaluated without patching:

```bash
python eval_shellaswag_raw.py --model_dir ckpt/3b_stage1 --base_model_name meta-llama/Llama-3.2-3B --meta_dir data/shellaswag --audio_dir data/shellaswag/audio_16k --hubert_ckpt data/hubert_base_ls960.pt --km_path data/hubert_base_ls960_L9_km500.bin --output_jsonl out/3b_raw_shellaswag.jsonl --device cuda --bf16
python eval_storycloze_raw.py --model_dir ckpt/3b_stage1 --tokenizer_name_or_path meta-llama/Llama-3.2-3B --audio_dir data/sSC --text_dir data/sSC --hubert_ckpt data/hubert_base_ls960.pt --km_path data/hubert_base_ls960_L9_km500.bin --output_jsonl out/3b_raw_storycloze.jsonl --device cuda --bf16
python eval_text_hellaswag.py --model_name ckpt/3b_stage1 --meta_dir data/shellaswag --output_jsonl out/3b_raw_text_hellaswag.jsonl --device cuda --bf16
```

For the 1B backbone use `meta-llama/Llama-3.2-1B` and the values in `configs/llama32_1b_stage{1,2}.yaml`. Other seeds: `--seed 43` / `--seed 44` in both stages (each Stage-2 run starts from the Stage-1 checkpoint of the same seed).

## License

MIT.
