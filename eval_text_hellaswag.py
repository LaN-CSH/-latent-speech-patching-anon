# eval_hellaswag_text_baseline.py

import argparse, json
from pathlib import Path
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl_files(meta_dir):
    rows = []
    for path in sorted(Path(meta_dir).glob("meta_*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def text_ids(tokenizer, text):
    return tokenizer(
        " ".join(text.strip().split()),
        add_special_tokens=False,
    )["input_ids"]


@torch.no_grad()
def score_option(model, input_ids, labels, device):
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    labels = torch.tensor([labels], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    nll = loss_fct(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
    )

    n_tokens = (shift_labels != -100).sum().item()
    avg_nll = nll.item() / max(n_tokens, 1)
    return -avg_nll


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--meta_dir", required=True)
    parser.add_argument("--output_jsonl", default="hellaswag_text_baseline_results.jsonl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
    ).to(device)
    model.eval()

    rows = load_jsonl_files(args.meta_dir)

    correct = 0
    total = 0

    with open(args.output_jsonl, "w", encoding="utf-8") as out:
        for ex in tqdm(rows):
            prefix_ids = text_ids(tokenizer, ex["context"])

            scores = []
            for ending in ex["endings"]:
                option_ids = text_ids(tokenizer, ending)
                input_ids = prefix_ids + option_ids
                labels = [-100] * len(prefix_ids) + option_ids

                score = score_option(model, input_ids, labels, device)
                scores.append(score)

            pred = max(range(len(scores)), key=lambda i: scores[i])
            label = int(ex["label"])
            ok = int(pred == label)

            correct += ok
            total += 1

            out.write(json.dumps({
                "id": ex["id"],
                "label": label,
                "pred": pred,
                "correct": ok,
                "scores": scores,
            }) + "\n")

    print("==== Text-only HellaSwag baseline ====")
    print("total:", total)
    print("correct:", correct)
    print("accuracy:", correct / max(total, 1))


if __name__ == "__main__":
    main()