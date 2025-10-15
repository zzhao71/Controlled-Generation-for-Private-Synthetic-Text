import argparse
import csv
from pathlib import Path
from typing import Optional

import torch
from peft import PrefixTuningConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

DEFAULT_MODEL_NAME = "princeton-nlp/Sheared-LLaMA-1.3B"
DEFAULT_TRAIN_CSV = Path("fine_tuning_data_echr_test_first_6_paragraphs_direct.csv")
DEFAULT_EVAL_CSV = Path("fine_tuning_data_echr_test_first_6_paragraphs_direct.csv")
DEFAULT_OUTPUT_DIR = Path("sheared-llama-prefix-tuned")
DEFAULT_VIRTUAL_TOKENS = 20
DEFAULT_BATCH_SIZE = 4
DEFAULT_GRAD_ACCUM = 4
DEFAULT_EPOCHS = 3
DEFAULT_LR = 5e-5
DEFAULT_EVAL_STEPS = 100
DEFAULT_SAVE_STEPS = 100
DEFAULT_LOGGING_STEPS = 10
DEFAULT_MAX_LENGTH = 1024


class PrivacyDataset(Dataset):
    def __init__(
        self,
        file_path: Path,
        tokenizer,
        max_length: int = DEFAULT_MAX_LENGTH,
        max_samples: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        with file_path.open(newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                control_code = row["input"]
                text_output = row["output"]
                self.examples.append({
                    "input_text": control_code,
                    "output_text": text_output,
                })
                if max_samples is not None and len(self.examples) >= max_samples:
                    break

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        half_length = self.max_length // 2

        input_encoding = self.tokenizer(
            example["input_text"],
            max_length=half_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        output_encoding = self.tokenizer(
            example["output_text"],
            max_length=half_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = torch.cat([
            input_encoding.input_ids.squeeze(),
            output_encoding.input_ids.squeeze(),
        ])[: self.max_length]
        attention_mask = torch.cat([
            input_encoding.attention_mask.squeeze(),
            output_encoding.attention_mask.squeeze(),
        ])[: self.max_length]
        labels = torch.cat([
            torch.full_like(input_encoding.input_ids.squeeze(), -100),
            output_encoding.input_ids.squeeze(),
        ])[: self.max_length]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def data_collator(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prefix-tuning fine-tuning script with configurable hyperparameters.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Base model identifier to load.")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV, help="Path to the training CSV file.")
    parser.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV, help="Path to the evaluation CSV file.")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to store checkpoints.")
    parser.add_argument("--final-checkpoint-name", default="final_checkpoint", help="Subdirectory name for the final checkpoint.")
    parser.add_argument("--virtual-token-count", type=int, default=DEFAULT_VIRTUAL_TOKENS, help="Number of virtual tokens in the prefix.")
    parser.add_argument("--no-prefix-projection", action="store_true", help="Disable prefix projection layer.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Per-device batch size for training.")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Per-device batch size for evaluation; defaults to --batch-size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=DEFAULT_GRAD_ACCUM, help="Gradient accumulation steps.")
    parser.add_argument("--num-epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR, help="Learning rate.")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Maximum sequence length for tokenization.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional cap on the number of training samples.")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Optional cap on the number of evaluation samples.")
    parser.add_argument("--eval-steps", type=int, default=DEFAULT_EVAL_STEPS, help="Evaluation interval (set <=0 to disable).")
    parser.add_argument("--save-steps", type=int, default=DEFAULT_SAVE_STEPS, help="Checkpoint save interval (set <=0 for epoch-level saves).")
    parser.add_argument("--save-total-limit", type=int, default=3, help="Maximum number of checkpoints to keep.")
    parser.add_argument("--logging-steps", type=int, default=DEFAULT_LOGGING_STEPS, help="Logging interval in steps.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--fp16", dest="fp16", action="store_true", help="Enable FP16 training (default).")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false", help="Disable FP16 training.")
    parser.set_defaults(fp16=True)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    peft_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=args.virtual_token_count,
        prefix_projection=not args.no_prefix_projection,
        inference_mode=False,
    )

    model = get_peft_model(base_model, peft_config)

    train_dataset = PrivacyDataset(
        args.train_csv,
        tokenizer,
        max_length=args.max_length,
        max_samples=args.max_train_samples,
    )

    eval_dataset = None
    if args.eval_csv is not None and args.eval_csv.exists():
        eval_dataset = PrivacyDataset(
            args.eval_csv,
            tokenizer,
            max_length=args.max_length,
            max_samples=args.max_eval_samples,
        )

    evaluation_strategy = "no"
    eval_steps = None
    if eval_dataset is not None:
        evaluation_strategy = "steps" if args.eval_steps and args.eval_steps > 0 else "epoch"
        eval_steps = args.eval_steps if args.eval_steps and args.eval_steps > 0 else None

    save_strategy = "steps" if args.save_steps and args.save_steps > 0 else "epoch"

    eval_batch_size = args.eval_batch_size or args.batch_size

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(args.checkpoint_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=args.save_steps if save_strategy == "steps" else None,
        learning_rate=args.learning_rate,
        fp16=args.fp16 and torch.cuda.is_available(),
        load_best_model_at_end=False,
        logging_dir=str(args.checkpoint_dir / "logs"),
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    final_dir = args.checkpoint_dir / args.final_checkpoint_name
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Model saved to {final_dir}")


if __name__ == "__main__":
    main()
