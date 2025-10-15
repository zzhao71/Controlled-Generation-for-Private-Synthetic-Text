import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)


def read_jsonl_texts(path: str, text_key: str = "generated_text", max_samples: Optional[int] = None) -> List[str]:
    texts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if text_key in obj and isinstance(obj[text_key], str):
                t = obj[text_key].strip()
                if t:
                    texts.append(t)
            if max_samples is not None and len(texts) >= max_samples:
                break
    return texts


def read_csv_column_texts(path: str, column: str = "output", max_samples: Optional[int] = None) -> List[str]:
    import csv

    texts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = str(row.get(column, "")).strip()
            if val:
                texts.append(val)
            if max_samples is not None and len(texts) >= max_samples:
                break
    return texts


def read_json_list_texts(path: str, field: str, max_samples: Optional[int] = None) -> List[str]:
    """Read a JSON file containing a list of objects and extract text from a given field."""
    texts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return texts
    if not isinstance(data, list):
        return texts
    for obj in data:
        if not isinstance(obj, dict):
            continue
        val = obj.get(field, "")
        if isinstance(val, str):
            t = val.strip()
            if t:
                texts.append(t)
        if max_samples is not None and len(texts) >= max_samples:
            break
    return texts


class CausalTextDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        tokenizer: AutoTokenizer,
        max_length: int = 1024,
        pad_to_max_length: bool = True,
    ):
        self.texts = texts
        self.tk = tokenizer
        self.max_length = max_length
        self.pad_to_max_length = pad_to_max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.texts[idx]
        enc = self.tk(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=("max_length" if self.pad_to_max_length else False),
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        # Labels for causal LM; pad tokens will be masked by the collator
        item["labels"] = item["input_ids"].clone()
        return item


def compute_perplexity_from_loss(eval_loss: float) -> float:
    try:
        return float(math.exp(eval_loss))
    except OverflowError:
        return float("inf")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Sheared-LLaMA-1.3B on JSONL outputs and evaluate perplexity on CSV outputs.")

    # Data (match train_sheared_llama_ppl.py)
    parser.add_argument("--train_jsonl", type=str, default="prefix_tuning_eval_results_private.json", help="Path to training JSONL with a 'generated_text' field.")
    parser.add_argument("--train_text_key", type=str, default="reference", help="Field name in JSONL to use as training text.")
    parser.add_argument("--eval_csv", type=str, default="fine_tuning_data_first_six_paragraphs_test.csv", help="Path to CSV file for perplexity evaluation.")
    parser.add_argument("--eval_text_column", type=str, default="output", help="CSV column to evaluate perplexity on (default: output).")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Limit number of training samples.")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Limit number of evaluation samples.")

    # Model
    parser.add_argument("--model_name", type=str, default="princeton-nlp/Sheared-LLaMA-1.3B", help="Pretrained model name or path.")
    parser.add_argument("--output_dir", type=str, default="sheared-llama-ft-synthetic", help="Where to save the fine-tuned model.")
    parser.add_argument("--max_length", type=int, default=1024, help="Max sequence length for tokenization.")

    # Training hyperparameters
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    # Precision / memory
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")

    # Eval control
    parser.add_argument("--eval_only", action="store_true", help="Skip training and only evaluate perplexity on the eval set.")

    args = parser.parse_args()

    set_seed(args.seed)

    # Load model and tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # Data loading (support JSONL or JSON list for training; CSV/JSON/JSONL for eval)
    print(f"Reading training texts from: {args.train_jsonl} (key/field={args.train_text_key})")
    train_texts: List[str] = []
    # Try by extension first
    lower_train = args.train_jsonl.lower()
    if lower_train.endswith(".jsonl"):
        train_texts = read_jsonl_texts(args.train_jsonl, text_key=args.train_text_key, max_samples=args.max_train_samples)
    elif lower_train.endswith(".json"):
        train_texts = read_json_list_texts(args.train_jsonl, field=args.train_text_key, max_samples=args.max_train_samples)
    # Fallback attempts if empty
    if not train_texts:
        train_texts = read_jsonl_texts(args.train_jsonl, text_key=args.train_text_key, max_samples=args.max_train_samples)
    if not train_texts:
        train_texts = read_json_list_texts(args.train_jsonl, field=args.train_text_key, max_samples=args.max_train_samples)
    if not train_texts:
        raise RuntimeError(f"No training texts loaded from {args.train_jsonl}. Ensure it is JSONL or a JSON array with field '{args.train_text_key}'.")
    print(f"Loaded {len(train_texts)} training texts")

    print(f"Reading eval texts from: {args.eval_csv} (column/field={args.eval_text_column})")
    eval_texts: List[str] = []
    lower_eval = args.eval_csv.lower()
    if lower_eval.endswith(".csv"):
        eval_texts = read_csv_column_texts(args.eval_csv, column=args.eval_text_column, max_samples=args.max_eval_samples)
    elif lower_eval.endswith(".jsonl"):
        eval_texts = read_jsonl_texts(args.eval_csv, text_key=args.eval_text_column, max_samples=args.max_eval_samples)
    elif lower_eval.endswith(".json"):
        eval_texts = read_json_list_texts(args.eval_csv, field=args.eval_text_column, max_samples=args.max_eval_samples)
    # Fallback attempts if empty
    if not eval_texts:
        eval_texts = read_csv_column_texts(args.eval_csv, column=args.eval_text_column, max_samples=args.max_eval_samples)
    if not eval_texts:
        eval_texts = read_jsonl_texts(args.eval_csv, text_key=args.eval_text_column, max_samples=args.max_eval_samples)
    if not eval_texts:
        eval_texts = read_json_list_texts(args.eval_csv, field=args.eval_text_column, max_samples=args.max_eval_samples)
    if not eval_texts:
        raise RuntimeError(f"No evaluation texts loaded from {args.eval_csv}. Ensure it is CSV, JSONL, or a JSON array with field '{args.eval_text_column}'.")
    print(f"Loaded {len(eval_texts)} eval texts")

    train_dataset = CausalTextDataset(train_texts, tokenizer, max_length=args.max_length)
    eval_dataset = CausalTextDataset(eval_texts, tokenizer, max_length=args.max_length)

    # Data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Optionally skip training
    if not args.eval_only:
        print("Starting training...")
        train_result = trainer.train()
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        metrics_train = train_result.metrics
        trainer.log_metrics("train", metrics_train)
        trainer.save_metrics("train", metrics_train)
        trainer.save_state()
    else:
        print("Eval-only mode: skipping training.")

    # Evaluate perplexity on eval texts
    print("Evaluating perplexity on eval set...")
    metrics_eval = trainer.evaluate(eval_dataset=eval_dataset)
    eval_loss = metrics_eval.get("eval_loss", None)
    ppl = compute_perplexity_from_loss(eval_loss) if eval_loss is not None else float("nan")
    metrics_eval["perplexity"] = ppl
    print(f"Eval loss: {eval_loss:.4f} | Perplexity: {ppl:.4f}")
    trainer.log_metrics("eval", metrics_eval)
    trainer.save_metrics("eval", metrics_eval)

    # Persist a concise summary to output_dir
    summary = {
        "model_name": args.model_name,
        "output_dir": args.output_dir,
        "train_jsonl": args.train_jsonl,
        "train_text_key": args.train_text_key,
        "num_train_epochs": args.num_train_epochs,
        "eval_path": args.eval_csv,
        "eval_text_column": args.eval_text_column,
        "num_train_texts": len(train_texts),
        "num_eval_texts": len(eval_texts),
        "eval_loss": float(eval_loss) if eval_loss is not None else None,
        "perplexity": float(ppl),
        "eval_only": bool(args.eval_only),
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summary to {os.path.join(args.output_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
