import argparse
import csv
import json
import math
import random
import re
import string
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import evaluate
import torch
from presidio_analyzer import AnalyzerEngine
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from privacy_metrics.Metrics import entities_in_paragraph, leaked_percentage

DEFAULT_MODEL_NAME = "princeton-nlp/Sheared-LLaMA-1.3B"
DEFAULT_CHECKPOINT_DIR = Path("sheared-llama-privacy-prefix-partial")
DEFAULT_CHECKPOINT_NAME = "final_checkpoint"
DEFAULT_INPUT_CSV = Path("fine_tuning_data_first_six_paragraphs_direct.csv")
DEFAULT_OUTPUT_JSONL = Path("synthetic_ft_data_partial_prefix.jsonl")
DEFAULT_CONTEXT_SIZE = 3
DEFAULT_MAX_ROWS: Optional[int] = 100
DEFAULT_MAX_NEW_TOKENS = 400
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9

STOP_STRINGS = [
    "CODE:",
    "LOC:",
    "ORG:",
    "DEM:",
    "QUANTITY:",
    "DATETIME:",
    "PERSON:",
    "MISE:",
]


@dataclass
class PartialPrefixConfig:
    model_name: str = DEFAULT_MODEL_NAME
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR
    checkpoint_name: str = DEFAULT_CHECKPOINT_NAME
    input_csv: Path = DEFAULT_INPUT_CSV
    output_jsonl: Path = DEFAULT_OUTPUT_JSONL
    context_size: int = DEFAULT_CONTEXT_SIZE
    max_rows: Optional[int] = DEFAULT_MAX_ROWS
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P


def parse_args() -> PartialPrefixConfig:
    parser = argparse.ArgumentParser(description="Partially known prefix tuning evaluation with Presidio control codes.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Base model identifier to load.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory containing the PEFT checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default=DEFAULT_CHECKPOINT_NAME,
        help="Checkpoint subdirectory name inside --checkpoint-dir.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="CSV file containing official control codes and reference outputs.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=DEFAULT_OUTPUT_JSONL,
        help="Destination for generated prompt/output pairs.",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=DEFAULT_CONTEXT_SIZE,
        help="Number of examples to concatenate before generation.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS if DEFAULT_MAX_ROWS is not None else -1,
        help="Limit the number of CSV rows to process (<=0 to use all).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum number of tokens to generate per context window.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature for generation.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help="Nucleus sampling top-p.",
    )

    args = parser.parse_args()
    max_rows = args.max_rows
    if max_rows is not None and max_rows <= 0:
        max_rows = None

    return PartialPrefixConfig(
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.checkpoint_name,
        input_csv=args.input_csv,
        output_jsonl=args.output_jsonl,
        context_size=max(1, args.context_size),
        max_rows=max_rows,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )


class StopOnCode(StoppingCriteria):
    def __init__(self, tokenizer):
        super().__init__()
        self.stop_ids_list = [tokenizer.encode(pattern, add_special_tokens=False) for pattern in STOP_STRINGS]

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> bool:
        for stop_ids in self.stop_ids_list:
            if input_ids.shape[1] < len(stop_ids):
                continue
            if input_ids[0, -len(stop_ids) :].tolist() == stop_ids:
                return True
        return False


def extract_private_entities(text: str, analyzer: AnalyzerEngine) -> str:
    entity_mapping = {
        "CODE": "CODE",
        "PERSON": "PERSON",
        "LOCATION": "LOC",
        "DATE_TIME": "DATETIME",
        "ORGANIZATION": "ORG",
        "DEMOGRAPHICS": "DEM",
        "QUANTITY": "QUANTITY",
    }

    results = analyzer.analyze(text=text, language="en")
    grouped: Dict[str, List[str]] = {}
    for result in results:
        mapped = entity_mapping.get(result.entity_type, "MISE")
        grouped.setdefault(mapped, []).append(text[result.start : result.end])

    formatted = []
    for entity_type, values in grouped.items():
        formatted.append(f"{entity_type}: {', '.join(values)}")
    return "\n".join(formatted)


def build_generators() -> Dict[str, callable]:
    def random_code():
        body = "".join(random.choices(string.ascii_uppercase + string.digits, k=5))
        tail = "".join(random.choices(string.ascii_uppercase + string.digits, k=2))
        return f"{body}/{tail}"

    first_names = [
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
    last_names = [
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

    def random_person():
        return f"{random.choice(['Mr', 'Ms', 'Dr', 'Prof'])} {random.choice(first_names)} {random.choice(last_names)}"

    def random_datetime(start_year: int = 1990, end_year: int = 2024):
        start_dt = datetime(start_year, 1, 1)
        span_days = (datetime(end_year, 12, 31) - start_dt).days
        day = start_dt + timedelta(days=random.randint(0, span_days))
        return day.strftime("%d %B %Y")

    cities = ["Baltimore", "Seattle", "Tokyo", "Munich", "Cairo"]
    countries = ["USA", "Germany", "Japan", "Kenya", "Brazil"]
    addresses = ["221B Baker St", "1600 Amphitheatre Pkwy", "350 Fifth Ave"]
    infrastructure = ["London Bridge", "Central Station", "Pier 39"]

    def random_loc():
        return random.choice(cities + countries + addresses + infrastructure)

    orgs = [
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

    def random_org():
        return random.choice(orgs)

    heritage = ["Irish-American", "Nigerian", "Chinese", "Latinx", "Punjabi"]
    jobs = ["software engineer", "nurse", "professor", "mechanic", "pilot"]

    def random_age():
        return f"{random.randint(18, 85)}-year-old"

    def random_dem():
        pattern = random.choice(
            [
                f"{random_age()} {random.choice(jobs)}",
                f"{random.choice(heritage)} descent",
                f"{random.choice(jobs).title()}",
                f"{random_age()}",
            ]
        )
        return pattern

    def random_quantity():
        if random.random() < 0.5:
            return f"{random.randint(1, 100)}%"
        return f"${random.randint(1, 999_999):,}"

    return {
        "CODE": random_code,
        "PERSON": random_person,
        "DATETIME": random_datetime,
        "LOC": random_loc,
        "ORG": random_org,
        "DEM": random_dem,
        "QUANTITY": random_quantity,
    }


def load_prefix_model(config: PartialPrefixConfig):
    base_model = AutoModelForCausalLM.from_pretrained(config.model_name, use_cache=False)
    from peft import PeftModel

    model_path = config.checkpoint_dir / config.checkpoint_name
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {model_path}")

    model = PeftModel.from_pretrained(base_model, model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, device


def run_partially_known_prefix(config: PartialPrefixConfig) -> None:
    analyzer = AnalyzerEngine()
    model, tokenizer, device = load_prefix_model(config)
    stopper = StoppingCriteriaList([StopOnCode(tokenizer)])
    rouge = evaluate.load("rouge")
    generators = build_generators()

    rows: List[dict] = []
    with config.input_csv.open(newline="", encoding="utf-8") as csvfile:
        for idx, row in enumerate(csv.DictReader(csvfile), start=1):
            rows.append(row)
            if config.max_rows is not None and idx >= config.max_rows:
                break

    if not rows:
        print("No rows found to process.")
        return

    total_expected_contexts = max(1, math.ceil(len(rows) / config.context_size))
    pbar = tqdm(total=total_expected_contexts)

    pattern = re.compile(r"^(CODE|PERSON|DATETIME|LOC|ORG|DEM|QUANTITY|MISE):\s*(.*)$", re.MULTILINE)

    privacy_hits = 0
    leaked_percentage_total = 0.0
    processed_contexts = 0
    rouge_2_total = 0.0
    rouge_l_total = 0.0
    generate_records: List[dict] = []

    prompt = ""
    current_count = 0
    latest_output = ""
    latest_official_code = ""

    def flush(force: bool = False) -> None:
        nonlocal prompt, current_count, privacy_hits, leaked_percentage_total
        nonlocal processed_contexts, rouge_2_total, rouge_l_total, latest_output, latest_official_code

        if current_count == 0:
            return
        if current_count < config.context_size and not force:
            return

        base_prompt = prompt
        matches = pattern.findall(base_prompt)

        official_matches = pattern.findall(latest_official_code)
        official_values = []
        for _, raw_value in official_matches:
            parts = [item.strip() for item in raw_value.split(",") if item.strip()]
            official_values.extend(parts)

        latest_by_tag: Dict[str, str] = {}
        for tag, raw in matches:
            latest_by_tag.setdefault(tag, raw.strip())

        random_lines = [f"{tag}: {generators[tag]()}" for tag in latest_by_tag if tag in generators]
        generation_prompt = base_prompt
        if random_lines:
            generation_prompt = f"{base_prompt}\n" + "\n".join(random_lines)

        inputs = tokenizer(generation_prompt, return_tensors="pt").to(device)
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            stopping_criteria=stopper,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        full_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        generated_text = full_text[len(generation_prompt) :]

        stop_positions = []
        for token in STOP_STRINGS:
            pos = generated_text.find(token)
            if pos >= 0:
                stop_positions.append(pos)
        if stop_positions:
            generated_text = generated_text[: min(stop_positions)].rstrip()

        result_dict = entities_in_paragraph(generated_text, official_values)
        if any(result_dict.values()):
            privacy_hits += 1
        processed_contexts += 1

        leaked_amount = leaked_percentage(generated_text, official_values)
        leaked_percentage_total += leaked_amount

        scores = rouge.compute(
            predictions=[generated_text],
            references=[latest_output],
            use_stemmer=True,
            rouge_types=["rouge2", "rougeL"],
        )
        rouge_2_total += scores["rouge2"]
        rouge_l_total += scores["rougeL"]

        generate_records.append({"input": generation_prompt, "generated_text": generated_text})

        privacy_pct = privacy_hits / processed_contexts * 100
        avg_leaked = leaked_percentage_total / processed_contexts
        pbar.update(1)
        pbar.set_postfix({"context": processed_contexts, "privacy": f"{privacy_pct:5.2f}%", "leaked": f"{avg_leaked:5.2f}%"})

        prompt = ""
        current_count = 0

    for row in rows:
        text_output = row["output"]
        control_code = extract_private_entities(text_output, analyzer)
        prompt += f"{control_code}\n{text_output}\n\n"
        latest_output = text_output
        latest_official_code = row["input"]
        current_count += 1

        if current_count == config.context_size:
            flush()

    flush(force=True)
    pbar.close()

    if processed_contexts > 0:
        print(f"Final Privacy percentage: {privacy_hits / processed_contexts * 100:.2f}%")
        print(f"Final Leaked percentage: {leaked_percentage_total / processed_contexts:.2f}%")
        print(f"Final Rouge-2 score: {rouge_2_total / processed_contexts:.4f}")
        print(f"Final Rouge-L score: {rouge_l_total / processed_contexts:.4f}")
        print(f"Total contexts processed: {processed_contexts}")

    config.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with config.output_jsonl.open("w", encoding="utf-8") as f:
        for record in generate_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Generated records have been saved to {config.output_jsonl}.")


def main() -> None:
    config = parse_args()
    run_partially_known_prefix(config)


if __name__ == "__main__":
    main()
