import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import evaluate
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from privacy_metrics.Metrics import entities_in_paragraph, leaked_percentage

DEFAULT_MODEL_NAME = "princeton-nlp/Sheared-LLaMA-1.3B"
DEFAULT_INPUT_CSV = "fine_tuning_data_echr_test_first_6_paragraphs_direct.csv"
DEFAULT_OUTPUT_JSONL = "icl_baseline.jsonl"
DEFAULT_CONTEXT_SIZE = 3
DEFAULT_MAX_NEW_TOKENS = 400
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9


@dataclass
class ICLConfig:
    model_name: str = DEFAULT_MODEL_NAME
    input_csv: Path = Path(DEFAULT_INPUT_CSV)
    output_jsonl: Path = Path(DEFAULT_OUTPUT_JSONL)
    context_size: int = DEFAULT_CONTEXT_SIZE
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P


def parse_args() -> ICLConfig:
    parser = argparse.ArgumentParser(description="Baseline ICL generation with privacy evaluation.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Base model identifier to load.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path(DEFAULT_INPUT_CSV),
        help="CSV file containing control codes and reference outputs.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path(DEFAULT_OUTPUT_JSONL),
        help="Destination for generated prompt/output pairs.",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=DEFAULT_CONTEXT_SIZE,
        help="Number of examples to concatenate before generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum number of tokens to generate per context.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help="Nucleus sampling top-p.",
    )

    args = parser.parse_args()
    return ICLConfig(
        model_name=args.model_name,
        input_csv=args.input_csv,
        output_jsonl=args.output_jsonl,
        context_size=max(1, args.context_size),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )


class StopOnCode(StoppingCriteria):
    def __init__(self, tokenizer, stop_string="CODE:"):
        super().__init__()
        self.stop_ids = tokenizer.encode(stop_string, add_special_tokens=False)

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> bool:
        if input_ids.shape[1] < len(self.stop_ids):
            return False
        tail = input_ids[0, -len(self.stop_ids) :].tolist()
        return tail == self.stop_ids


def run_baseline(config: ICLConfig) -> None:
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    stopper = StoppingCriteriaList([StopOnCode(tokenizer)])
    rouge = evaluate.load("rouge")

    with config.input_csv.open(newline="", encoding="utf-8") as csvfile:
        rows: List[dict] = list(csv.DictReader(csvfile))

    if not rows:
        print("No rows found to process.")
        return

    total_contexts = max(1, math.ceil(len(rows) / config.context_size))
    pbar = tqdm(total=total_contexts)

    pattern = re.compile(r"^(CODE|PERSON|DATETIME|LOC|ORG|DEM|QUANTITY|MISE):\s*(.*)$", re.MULTILINE)

    privacy_hits = 0
    leaked_percentage_total = 0.0
    processed_contexts = 0
    rouge_2_total = 0.0
    rouge_l_total = 0.0
    generate_records = []

    current_prompt = ""
    current_extract = ""
    current_count = 0
    latest_reference = ""

    def flush(latest_output: str, force: bool = False) -> None:
        nonlocal current_prompt, current_extract, current_count
        nonlocal privacy_hits, leaked_percentage_total, processed_contexts
        nonlocal rouge_2_total, rouge_l_total

        if current_count == 0:
            return
        if current_count < config.context_size and not force:
            return

        matches = pattern.findall(current_extract)
        extracted_values = []
        for _, raw_value in matches:
            parts = [item.strip() for item in raw_value.split(",")]
            extracted_values.extend(part for part in parts if part)

        #print(extracted_values)  # Uncommented to enable printing extracted values

        inputs = tokenizer(current_prompt, return_tensors="pt").to(model.device)
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            stopping_criteria=stopper,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=True,
        )
        generated_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)[len(current_prompt) :]
        generated_text = generated_text.split("CODE:")[0].rstrip()

        preds = [generated_text]
        refs = [latest_output]
        generate_records.append({"input": current_prompt, "generated_text": generated_text})

        result_dict = entities_in_paragraph(generated_text, extracted_values)
        if any(result_dict.values()):
            privacy_hits += 1
        processed_contexts += 1

        leaked_amount = leaked_percentage(generated_text, extracted_values)
        leaked_percentage_total += leaked_amount

        privacy_pct = privacy_hits / processed_contexts * 100
        avg_leaked = leaked_percentage_total / processed_contexts

        pbar.update(1)
        pbar.set_postfix(
            {
                "context": processed_contexts,
                "privacy": f"{privacy_pct:5.2f}%",
                "leaked": f"{avg_leaked:5.2f}%",
            }
        )

        scores = rouge.compute(
            predictions=preds,
            references=refs,
            use_stemmer=True,
            rouge_types=["rouge2", "rougeL"],
        )
        rouge_2_total += scores["rouge2"]
        rouge_l_total += scores["rougeL"]

        current_prompt = ""
        current_extract = ""
        current_count = 0

    for row in rows:
        text_output = row["output"]
        control_code = row["input"]
        current_prompt += f"{text_output}\n\n"
        current_extract += f"{control_code}\n{text_output}\n\n"
        current_count += 1
        latest_reference = text_output

        if current_count == config.context_size:
            flush(latest_reference)

    flush(latest_reference, force=True)
    pbar.close()

    if processed_contexts > 0:
        print(f"Final Privacy percentage: {privacy_hits / processed_contexts * 100:.2f}%")
        print(f"Final Leaked percentage: {leaked_percentage_total / processed_contexts:.2f}%")
        print(f"Final Rouge-2 score: {rouge_2_total / processed_contexts:.4f}")
        print(f"Final Rouge-L score: {rouge_l_total / processed_contexts:.4f}")
        print(f"Total records processed: {processed_contexts}")

    config.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with config.output_jsonl.open("w", encoding="utf-8") as f:
        for rec in generate_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Generated records have been saved to {config.output_jsonl}.")


def main() -> None:
    config = parse_args()
    run_baseline(config)


if __name__ == "__main__":
    main()
