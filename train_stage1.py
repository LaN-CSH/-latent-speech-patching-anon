# train_qwen_st_to_t.py

import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


class JsonlCausalDataset(Dataset):
    def __init__(self, path: str):
        self.data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ex = json.loads(line)
                    self.data.append({
                        "input_ids": ex["input_ids"],
                        "labels": ex["labels"],
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


@dataclass
class DataCollatorForCausalLMJsonl:
    pad_token_id: int

    def __call__(self, features: List[Dict[str, Any]]):
        max_len = max(len(x["input_ids"]) for x in features)

        input_ids = []
        labels = []
        attention_mask = []

        for x in features:
            ids = x["input_ids"]
            labs = x["labels"]

            pad_len = max_len - len(ids)

            input_ids.append(ids + [self.pad_token_id] * pad_len)
            labels.append(labs + [-100] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--output_dir", default="./outputs/qwen_st_to_t")
    parser.add_argument("--codebook_size", type=int, default=500)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    text_vocab_size = len(tokenizer)
    s_id = text_vocab_size
    t_id = text_vocab_size + 1
    hubert_offset = text_vocab_size + 2
    total_vocab_size = text_vocab_size + 2 + args.codebook_size

    print("text_vocab_size:", text_vocab_size)
    print("<s> id:", s_id)
    print("<t> id:", t_id)
    print("hubert_offset:", hubert_offset)
    print("total_vocab_size:", total_vocab_size)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
    )

    model.resize_token_embeddings(total_vocab_size)

    # 기존 Qwen text embedding은 freeze하고,
    # 새로 추가한 <s>, <t>, HuBERT token embedding만 학습
    emb = model.get_input_embeddings().weight

    def freeze_old_embedding_grad(grad):
        grad[:text_vocab_size] = 0
        return grad

    emb.register_hook(freeze_old_embedding_grad)

    # Qwen은 보통 input embedding과 lm_head가 tie되어 있을 수 있음.
    # lm_head도 같은 weight면 위 hook으로 같이 제어됨.
    # untied라면 lm_head에도 동일하게 적용
    if hasattr(model, "lm_head") and model.lm_head.weight is not emb:
        lm_head_weight = model.lm_head.weight

        def freeze_old_lm_head_grad(grad):
            grad[:text_vocab_size] = 0
            return grad

        lm_head_weight.register_hook(freeze_old_lm_head_grad)

    train_dataset = JsonlCausalDataset(args.train_jsonl)

    collator = DataCollatorForCausalLMJsonl(
        pad_token_id=tokenizer.pad_token_id,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        #overwrite_output_dir=True,

        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,

        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,

        logging_dir=f"{args.output_dir}/logs",
        logging_steps=args.logging_steps,
        logging_strategy="steps",

        save_strategy="no",

        report_to=["tensorboard"],
        bf16=args.bf16,
        fp16=args.fp16,

        gradient_checkpointing=True,
        optim="adamw_torch",

        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    meta = {
        "text_vocab_size": text_vocab_size,
        "s_id": s_id,
        "t_id": t_id,
        "hubert_offset": hubert_offset,
        "total_vocab_size": total_vocab_size,
        "codebook_size": args.codebook_size,
    }

    with open(f"{args.output_dir}/speech_token_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Training finished.")
    print("Saved to:", args.output_dir)


if __name__ == "__main__":
    main()