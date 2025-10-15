import argparse
import csv
import random
import string
from pathlib import Path
from typing import Dict, List, Optional

import torch
from presidio_analyzer import AnalyzerEngine
from peft import PrefixTuningConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

DEFAULT_MODEL_NAME = "princeton-nlp/Sheared-LLaMA-1.3B"
DEFAULT_TRAIN_CSV = Path("fine_tuning_data_echr_test_first_6_paragraphs_direct.csv")
DEFAULT_EVAL_CSV = Path("fine_tuning_data_echr_test_first_6_paragraphs_direct.csv")
DEFAULT_OUTPUT_DIR = Path("sheared-llama-partial-prefix")
DEFAULT_VIRTUAL_TOKENS = 20
DEFAULT_MAX_LENGTH = 1024
DEFAULT_BATCH_SIZE = 4
DEFAULT_GRAD_ACCUM = 4
DEFAULT_NUM_EPOCHS = 3
DEFAULT_LR = 5e-5
DEFAULT_EVAL_STEPS = 200
DEFAULT_SAVE_STEPS = 200
DEFAULT_LOGGING_STEPS = 50
DEFAULT_DROP_PROB = 0.35
DEFAULT_NOISE_PROB = 0.2

ENTITY_MAPPING = {
    "CODE": "CODE",
    "PERSON": "PERSON",
    "LOCATION": "LOC",
    "DATE_TIME": "DATETIME",
    "ORGANIZATION": "ORG",
    "DEMOGRAPHICS": "DEM",
    "QUANTITY": "QUANTITY",
}


def random_code() -> str:
    body = "".join(random.choices(string.ascii_uppercase + string.digits, k=5))
    tail = "".join(random.choices(string.ascii_uppercase + string.digits, k=2))
    return f"{body}/{tail}"


FIRST_NAMES = [
    "Alex",
    "Blake",
    "Casey",
    "Dana",
    "Elliot",
    "Finley",
    "Harper",
    "Jordan",
    "Kai",
    "Logan",
    "Morgan",
    "Quinn",
    "Riley",
    "Skyler",
]
LAST_NAMES = [
    "Adams",
    "Baker",
    "Carson",
    "Dawson",
    "Ellis",
    "Foster",
    "Griffin",
    "Hayes",
    "Irwin",
    "Johnson",
    "Kennedy",
    "Lewis",
]


def random_person() -> str:
    return f"{random.choice(['Mr', 'Ms', 'Dr', 'Prof'])} {random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


CITIES = ["Baltimore", "Seattle", "Tokyo", "Munich", "Cairo"]
COUNTRIES = ["USA", "Germany", "Japan", "Kenya", "Brazil"]
ADDRESSES = ["221B Baker St", "1600 Amphitheatre Pkwy", "350 Fifth Ave"]
INFRASTRUCTURE = ["London Bridge", "Central Station", "Pier 39"]


def random_loc() -> str:
    return random.choice(CITIES + COUNTRIES + ADDRESSES + INFRASTRUCTURE)


ORGS = [
    "OpenAI",
    "World Health Organization",
    "Harvard University",
    "UNICEF",
    "St. Mary's Hospital",
    "SpaceX",
    "NASA",
    "MIT",
    "Stanford University",
    "Google",
]


def random_org() -> str:
    return random.choice(ORGS)


HERITAGE = ["Irish-American", "Nigerian", "Chinese", "Latinx", "Punjabi"]
JOBS = ["software engineer", "nurse", "professor", "mechanic", "pilot"]


def random_age() -> str:
    return f"{random.randint(18, 85)}-year-old"


def random_dem() -> str:
    return random.choice([
        f"{random_age()} {random.choice(JOBS)}",
        f"{random.choice(HERITAGE)} descent",
        f"{random.choice(JOBS).title()}",
        random_age(),
    ])


def random_datetime() -> str:
    from datetime import datetime, timedelta

    start = datetime(1990, 1, 1)
    end = datetime(2024, 12, 31)
    delta = end - start
    day = start + timedelta(days=random.randint(0, delta.days))
    return day.strftime("%d %B %Y")


def random_quantity() -> str:
    if random.random() < 0.5:
        return f"{random.randint(1, 100)}%"
    return f"${random.randint(1, 999_999):,}"


NOISE_GENERATORS = {
    "CODE": random_code,
    "PERSON": random_person,
    "DATETIME": random_datetime,
    "LOC": random_loc,
    "ORG": random_org,
    "DEM": random_dem,
    "QUANTITY": random_quantity,
}


def build_control_code(
    text: str,
    analyzer: AnalyzerEngine,
    drop_prob: float,
    noise_prob: float,
    rng: random.Random,
) -> str:
    entities = analyzer.analyze(text=text, language="en")
    grouped: Dict[str, List[str]] = {}
    for result in entities:
        mapped = ENTITY_MAPPING.get(result.entity_type, "MISE")
        value = text[result.start : result.end].strip()
        if value:
            grouped.setdefault(mapped, []).append(value)

    lines: List[str] = []
    for tag, values in grouped.items():
        kept = [v for v in values if rng.random() > drop_prob]
        if not kept and values:
            kept = [values[rng.randrange(len(values))]]
        if kept:
            unique = list(dict.fromkeys(kept))
            lines.append(f"{tag}: {', '.join(unique)}")

    for tag, generator in NOISE_GENERATORS.items():
        if rng.random() < noise_prob:
            lines.append(f"{tag}: {generator()}")

    return "\n".join(lines)


class PartialPrefixDataset(Dataset):
    def __init__(
        self,
        file_path: Path,
        tokenizer,
        analyzer: AnalyzerEngine,
        max_length: int,
        drop_prob: float,
        noise_prob: float,
        max_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples: List[Dict[str, str]] = []
        rng = random.Random(seed)

        with file_path.open(newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                output_text = row.get("output", "").strip()
                if not output_text:
                    continue
                control_code = build_control_code(output_text, analyzer, drop_prob, noise_prob, rng)
                if not control_code:
                    continue
                self.examples.append({
                    "input_text": control_code,
                    "output_text": output_text,
                })
                if max_samples is not None and len(self.examples) >= max_samples:
                    break

        if not self.examples:
            raise RuntimeError("No training examples constructed from the provided CSV. Check that Presidio detects entities.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        half = self.max_length // 2
        input_enc = self.tokenizer(
            example["input_text"],
            max_length=half,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        output_enc = self.tokenizer(
            example["output_text"],
            max_length=half,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = torch.cat([input_enc.input_ids.squeeze(), output_enc.input_ids.squeeze()])[: self.max_length]
        attention_mask = torch.cat([
            input_enc.attention_mask.squeeze(),
            output_enc.attention_mask.squeeze(),
        ])[: self.max_length]
        labels = torch.cat([
            torch.full_like(input_enc.input_ids.squeeze(), -100),
            output_enc.input_ids.squeeze(),
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
    parser = argparse.ArgumentParser(description="Prefix tuning using partial control codes extracted with Presidio.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Base model identifier to load.")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV, help="CSV containing outputs for training.")
    parser.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV, help="Optional CSV for evaluation (same pipeline).")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to store checkpoints.")
    parser.add_argument("--final-checkpoint-name", default="final_checkpoint", help="Subdirectory name for the saved adapter.")
    parser.add_argument("--virtual-token-count", type=int, default=DEFAULT_VIRTUAL_TOKENS, help="Number of virtual tokens in the prefix.")
    parser.add_argument("--no-prefix-projection", action="store_true", help="Disable the prefix projection layer.")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Maximum combined sequence length.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Per-device training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Per-device evaluation batch size (defaults to training batch size).")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=DEFAULT_GRAD_ACCUM, help="Gradient accumulation steps.")
    parser.add_argument("--num-epochs", type=int, default=DEFAULT_NUM_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR, help="Learning rate.")
    parser.add_argument("--eval-steps", type=int, default=DEFAULT_EVAL_STEPS, help="Evaluation interval in steps (<=0 evaluates each epoch).")
    parser.add_argument("--save-steps", type=int, default=DEFAULT_SAVE_STEPS, help="Checkpoint save interval in steps (<=0 saves each epoch).")
    parser.add_argument("--save-total-limit", type=int, default=3, help="Maximum number of checkpoints to keep.")
    parser.add_argument("--logging-steps", type=int, default=DEFAULT_LOGGING_STEPS, help="Logging interval in steps.")
    parser.add_argument("--drop-prob", type=float, default=DEFAULT_DROP_PROB, help="Probability of dropping an extracted entity line.")
    parser.add_argument("--noise-prob", type=float, default=DEFAULT_NOISE_PROB, help="Probability of adding a random synthetic line.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional cap on training samples.")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Optional cap on evaluation samples.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--fp16", dest="fp16", action="store_true", help="Enable FP16 training when CUDA is available.")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false", help="Disable FP16 training.")
    parser.set_defaults(fp16=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        set_seed(args.seed)
        random.seed(args.seed)

    analyzer = AnalyzerEngine()

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

    train_dataset = PartialPrefixDataset(
        args.train_csv,
        tokenizer,
        analyzer,
        max_length=args.max_length,
        drop_prob=args.drop_prob,
        noise_prob=args.noise_prob,
        max_samples=args.max_train_samples,
        seed=args.seed,
    )

    eval_dataset = None
    if args.eval_csv and args.eval_csv.exists():
        eval_dataset = PartialPrefixDataset(
            args.eval_csv,
            tokenizer,
            analyzer,
            max_length=args.max_length,
            drop_prob=args.drop_prob,
            noise_prob=args.noise_prob,
            max_samples=args.max_eval_samples,
            seed=args.seed,
        )

    evaluation_strategy = "no"
    eval_steps = None
    if eval_dataset is not None:
        if args.eval_steps and args.eval_steps > 0:
            evaluation_strategy = "steps"
            eval_steps = args.eval_steps
        else:
            evaluation_strategy = "epoch"

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
        logging_dir=str(args.checkpoint_dir / "logs"),
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        report_to="none",
        load_best_model_at_end=False,
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
