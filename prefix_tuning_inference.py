import argparse
import csv
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
import gc
import logging
from pathlib import Path
from typing import Optional

import evaluate
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from privacy_metrics.Metrics import entities_in_paragraph, leaked_percentage

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_MODEL_NAME = "princeton-nlp/Sheared-LLaMA-1.3B"
DEFAULT_OUTPUT_CSV_FILE = "fine_tuning_data_first_six_paragraphs_direct.csv"
DEFAULT_CHECKPOINT_DIR = "sheared-llama-privacy-prefix"
DEFAULT_FINAL_CHECKPOINT = "final_checkpoint"
DEFAULT_OUTPUT_FILE = "prefix_tuning_eval_results.json"
DEFAULT_MAX_EVAL_SAMPLES: Optional[int] = None
DEFAULT_MAX_NEW_TOKENS = 400
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9


@dataclass
class InferenceConfig:
    model_name: str = DEFAULT_MODEL_NAME
    checkpoint_dir: Path = Path(DEFAULT_CHECKPOINT_DIR)
    checkpoint_name: str = DEFAULT_FINAL_CHECKPOINT
    output_csv_file: Path = Path(DEFAULT_OUTPUT_CSV_FILE)
    output_file: Path = Path(DEFAULT_OUTPUT_FILE)
    max_eval_samples: Optional[int] = DEFAULT_MAX_EVAL_SAMPLES
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P


def parse_args() -> InferenceConfig:
    parser = argparse.ArgumentParser(description="Evaluate a prefix-tuned (PEFT) model on privacy metrics.")
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=f"Base model identifier to load (default: {DEFAULT_MODEL_NAME}).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(DEFAULT_CHECKPOINT_DIR),
        help=f"Directory containing the PEFT checkpoint (default: {DEFAULT_CHECKPOINT_DIR}).",
    )
    parser.add_argument(
        "--checkpoint-name",
        default=DEFAULT_FINAL_CHECKPOINT,
        help=f"Name of the checkpoint subdirectory (default: {DEFAULT_FINAL_CHECKPOINT}).",
    )
    parser.add_argument(
        "--output-csv-file",
        type=Path,
        default=Path(DEFAULT_OUTPUT_CSV_FILE),
        help="CSV file used to extract control codes for evaluation.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path(DEFAULT_OUTPUT_FILE),
        help="Where to store the evaluation results JSON.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=DEFAULT_MAX_EVAL_SAMPLES,
        help="Limit the number of evaluation rows (default: evaluate all).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Maximum new tokens to generate (default: {DEFAULT_MAX_NEW_TOKENS}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE}).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help=f"Nucleus sampling top-p (default: {DEFAULT_TOP_P}).",
    )

    args = parser.parse_args()
    return InferenceConfig(
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.checkpoint_name,
        output_csv_file=args.output_csv_file,
        output_file=args.output_file,
        max_eval_samples=args.max_eval_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )


# Helper functions for generating random values
def random_code():
    """Generate a random code in format XXXXX/XX"""
    from string import ascii_uppercase, digits
    body = "".join(random.choices(ascii_uppercase + digits, k=5))
    tail = "".join(random.choices(ascii_uppercase + digits, k=2))
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


def random_person():
    """Generate a random person's name with title"""
    return f"{random.choice(['Mr', 'Ms', 'Dr', 'Prof'])} {random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def random_datetime(start_year=1990, end_year=2024):
    """Generate a random date between start_year and end_year"""
    start_dt = datetime(start_year, 1, 1)
    span_days = (datetime(end_year, 12, 31) - start_dt).days
    day = start_dt + timedelta(days=random.randint(0, span_days))
    return day.strftime("%d %B %Y")


CITIES = ["Baltimore", "Seattle", "Tokyo", "Munich", "Cairo"]
COUNTRIES = ["USA", "Germany", "Japan", "Kenya", "Brazil"]
ADDRESSES = ["221B Baker St", "1600 Amphitheatre Pkwy", "350 Fifth Ave"]
INFRA = ["London Bridge", "Central Station", "Pier 39"]


def random_loc():
    """Generate a random location"""
    return random.choice(CITIES + COUNTRIES + ADDRESSES + INFRA)


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


def random_org():
    """Generate a random organization name"""
    return random.choice(ORGS)


HERITAGE = ["Irish-American", "Nigerian", "Chinese", "Latinx", "Punjabi"]
JOB = ["software engineer", "nurse", "professor", "mechanic", "pilot"]


def random_age():
    """Generate a random age"""
    return f"{random.randint(18, 85)}-year-old"


def random_dem():
    """Generate a random demographic description"""
    pattern = random.choice(
        [
            f"{random_age()} {random.choice(JOB)}",
            f"{random.choice(HERITAGE)} descent",
            f"{random.choice(JOB).title()}",
            f"{random_age()}",
        ]
    )
    return pattern


def random_quantity():
    """Generate a random quantity (percentage or money value)"""
    if random.random() < 0.5:  # 50-50: percent vs money
        return f"{random.randint(1, 100)}%"
    else:
        return f"${random.randint(1, 999_999):,}"


# Generator mapping
GENERATOR = {
    "CODE": random_code,
    "PERSON": random_person,
    "DATETIME": random_datetime,
    "LOC": random_loc,
    "ORG": random_org,
    "DEM": random_dem,
    "QUANTITY": random_quantity,
}


def load_model_for_inference(config: InferenceConfig):
    """Load the prefix-tuned model and tokenizer for inference"""
    try:
        logger.info(f"Loading base model: {config.model_name}")

        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            use_cache=False,
        )

        model_path = Path(config.checkpoint_dir) / config.checkpoint_name
        logger.info(f"Loading adapter weights from: {model_path}")

        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        from peft import PeftModel

        model = PeftModel.from_pretrained(base_model, model_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        model = model.to(device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def extract_unique_values(csv_file: Path):
    """Extract all unique values from the control codes in the CSV file"""
    logger.info(f"Extracting unique values from {csv_file}")
    unique_values = []
    pattern = re.compile(r"^(CODE|PERSON|DATETIME|LOC|ORG|DEM|QUANTITY|MISE):\s*(.*)$", re.MULTILINE)

    try:
        with open(csv_file, newline="", encoding="utf-8") as csvfile:
            for row in csv.DictReader(csvfile):
                input_col = row["input"]
                for tag, raw in pattern.findall(input_col):
                    for val in map(str.strip, raw.split(",")):
                        if val and val not in unique_values:
                            unique_values.append(val)

        logger.info(f"Found {len(unique_values)} unique values for privacy evaluation")
        return unique_values

    except Exception as e:
        logger.error(f"Error extracting unique values: {e}")
        raise


def generate_with_peft_model(
    model,
    tokenizer,
    input_ids,
    attention_mask=None,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    """A safer generation function for PEFT models that works around common issues"""
    device = input_ids.device

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    try:
        logger.info("Attempting generation with base model direct access")
        with torch.no_grad():
            output_ids = model.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
            )
        return output_ids
    except Exception as e1:
        logger.warning(f"Base model generation failed: {e1}")

        try:
            logger.info("Attempting manual token generation as fallback")

            current_ids = input_ids.clone()
            current_mask = attention_mask.clone() if attention_mask is not None else None

            for _ in range(max_new_tokens):
                with torch.no_grad():
                    outputs = model(
                        input_ids=current_ids,
                        attention_mask=current_mask,
                        use_cache=False,
                    )

                next_token_logits = outputs.logits[:, -1, :]
                next_token_logits = next_token_logits / temperature

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float("Inf"))

                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                current_ids = torch.cat([current_ids, next_token], dim=-1)

                if current_mask is not None:
                    current_mask = torch.cat(
                        [
                            current_mask,
                            torch.ones((current_mask.shape[0], 1), device=device),
                        ],
                        dim=-1,
                    )

                if next_token[0, 0].item() == tokenizer.eos_token_id:
                    break

            return current_ids

        except Exception as e2:
            logger.error(f"Both generation methods failed. Final error: {e2}")
            return input_ids


def evaluate_model(model, tokenizer, unique_values, config: InferenceConfig):
    """Evaluate the model on privacy metrics and generation quality"""
    logger.info("Starting model evaluation")
    model.eval()

    privacy_violation_count = 0
    leaked_percentage_total = 0
    rouge_2_total = 0
    rouge_l_total = 0
    total_samples = 0
    leaked = []

    device = next(model.parameters()).device
    rouge = evaluate.load("rouge")
    pattern = re.compile(r"^(CODE|PERSON|DATETIME|LOC|ORG|DEM|QUANTITY|MISE):\s*(.*)$", re.MULTILINE)

    results = []
    try:
        with open(config.output_csv_file, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)

            for idx, row in enumerate(reader, start=1):
                if config.max_eval_samples is not None and idx > config.max_eval_samples:
                    break

                control_code = row["input"]
                reference_output = row["output"]

                matches = pattern.findall(control_code)
                latest = {}
                for tag, raw in matches:
                    latest.setdefault(tag, raw.strip())

                lines = []
                for tag in latest:
                    if tag in GENERATOR:
                        lines.append(f"{tag}: {GENERATOR[tag]()}")

                input_text = "\n".join(lines)
                inputs = tokenizer(input_text, return_tensors="pt").to(device)

                gen_ids = generate_with_peft_model(
                    model,
                    tokenizer,
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask if hasattr(inputs, "attention_mask") else None,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                )

                generated_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

                input_text_decoded = tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
                if generated_text.startswith(input_text_decoded):
                    generated_text = generated_text[len(input_text_decoded):].strip()

                result_dict = entities_in_paragraph(generated_text, unique_values)

                for entity_value, found in result_dict.items():
                    if found:
                        logger.info(f"Privacy violation detected for entity: {entity_value}")
                        if entity_value not in leaked:
                            leaked.append(entity_value)

                if any(result_dict.values()):
                    privacy_violation_count += 1

                leaked_percentage_amount = leaked_percentage(generated_text, unique_values)
                leaked_percentage_total += leaked_percentage_amount

                scores = rouge.compute(
                    predictions=[generated_text],
                    references=[reference_output],
                    use_stemmer=True,
                    rouge_types=["rouge2", "rougeL"],
                )
                rouge_2_total += scores["rouge2"]
                rouge_l_total += scores["rougeL"]

                results.append(
                    {
                        "input": control_code,
                        "reference": reference_output,
                        "generated": generated_text,
                        "privacy_violated": any(result_dict.values()),
                        "leaked_percentage": leaked_percentage_amount,
                        "rouge2": scores["rouge2"],
                        "rougeL": scores["rougeL"],
                    }
                )

                total_samples += 1

                if idx % 10 == 0:
                    logger.info(f"Processed {idx} samples")

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise

    if total_samples > 0:
        logger.info("=== Evaluation Results ===")
        logger.info(f"Privacy violation percentage: {privacy_violation_count / total_samples * 100:.2f}%")
        logger.info(f"Average leaked percentage: {leaked_percentage_total / total_samples:.2f}%")
        logger.info(f"Average Rouge-2 score: {rouge_2_total / total_samples:.4f}")
        logger.info(f"Average Rouge-L score: {rouge_l_total / total_samples:.4f}")
        logger.info(f"Total examples evaluated: {total_samples}")
        logger.info(f"Unique leaked entities percentage: {len(set(leaked)) / total_samples * 100:.2f}%")
    else:
        logger.warning("No samples were evaluated")

    try:
        with open(config.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {config.output_file}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")

    return results


def cleanup_resources(model=None):
    """Clean up GPU memory and other resources"""
    if model is not None and torch.cuda.is_available():
        logger.info("Moving model to CPU for cleanup")
        model.to("cpu")

    gc.collect()

    if torch.cuda.is_available():
        logger.info("Clearing CUDA cache")
        torch.cuda.empty_cache()


def main(config: InferenceConfig):
    """Main function to run evaluation"""
    model = None
    try:
        model, tokenizer = load_model_for_inference(config)
        logger.info("Successfully loaded trained model for evaluation")

        unique_values = extract_unique_values(config.output_csv_file)

        evaluation_results = evaluate_model(model, tokenizer, unique_values, config)
        logger.info(f"Evaluation completed with {len(evaluation_results)} samples")

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise

    finally:
        cleanup_resources(model)
        logger.info("Resources cleaned up")


if __name__ == "__main__":
    main(parse_args())
