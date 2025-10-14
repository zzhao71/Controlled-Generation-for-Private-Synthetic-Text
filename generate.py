import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Set

import pandas as pd

DEFAULT_INPUT = Path("data/tab/echr_test.json")
DEFAULT_PARAGRAPH_LIMIT = 6
DEFAULT_IDENTIFIER_TYPES = ("DIRECT",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fine-tuning pairs from ECHR annotations with controllable filtering."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to source annotations JSON (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the generated CSV; defaults are derived from the input/filters.",
    )
    parser.add_argument(
        "--paragraphs",
        type=int,
        default=None,
        help="Number of non-empty paragraphs to keep; set <= 0 to keep all (default: 6)",
    )
    parser.add_argument(
        "--identifier-types",
        nargs="+",
        choices=["DIRECT", "QUASI"],
        default=list(DEFAULT_IDENTIFIER_TYPES),
        metavar="IDENTIFIER",
        help="Identifier types to include (choose from DIRECT QUASI; default: DIRECT)",
    )
    return parser.parse_args()


def derive_output_path(input_path: Path, paragraph_limit: int, identifier_types: Set[str]) -> Path:
    identifier_tag = "_".join(sorted(identifier.lower() for identifier in identifier_types))
    paragraph_tag = "all_paragraphs" if paragraph_limit <= 0 else f"first_{paragraph_limit}_paragraphs"
    stem = input_path.stem or "annotations"
    return Path(f"fine_tuning_data_{stem}_{paragraph_tag}_{identifier_tag}.csv")


def extract_first_annotator_mentions(annotation: dict) -> Iterable[dict]:
    annotations = annotation.get("annotations", {})
    if not annotations:
        return []

    first_key = next(iter(annotations))
    first_annotator = annotations.get(first_key, {})
    return first_annotator.get("entity_mentions", [])


def combine_paragraphs(text: str, limit: int | None) -> str:
    paragraphs = [paragraph for paragraph in text.split("\n") if paragraph]
    if limit is None or limit <= 0:
        return "\n".join(paragraphs)
    return "\n".join(paragraphs[:limit])


def build_dataset_rows(annotations: list[dict], identifier_types: Set[str], paragraph_limit: int | None) -> list[dict]:
    rows = []
    for annotation in annotations:
        control_code_dict: defaultdict[str, set[str]] = defaultdict(set)
        text = annotation.get("text") or ""
        combined_paragraphs = combine_paragraphs(text, paragraph_limit)

        for entity_mention in extract_first_annotator_mentions(annotation):
            identifier = entity_mention.get("identifier_type")
            if identifier not in identifier_types:
                continue

            control_code = entity_mention.get("entity_type")
            span_text = entity_mention.get("span_text")
            if not control_code or not span_text:
                continue

            if span_text in combined_paragraphs:
                control_code_dict[control_code].add(span_text)

        control_code_list = [
            f"{code}: {', '.join(sorted(spans))}" for code, spans in control_code_dict.items()
        ]
        rows.append({
            "input": "\n".join(control_code_list),
            "output": combined_paragraphs,
        })
    return rows


def main() -> None:
    args = parse_args()
    identifier_types = set(args.identifier_types or DEFAULT_IDENTIFIER_TYPES)
    paragraph_limit = args.paragraphs if args.paragraphs is not None else DEFAULT_PARAGRAPH_LIMIT

    with args.input.open("r", encoding="utf-8") as file:
        annotations = json.load(file)

    dataset_rows = build_dataset_rows(annotations, identifier_types, paragraph_limit)
    df = pd.DataFrame(dataset_rows)

    output_path = args.output or derive_output_path(args.input, paragraph_limit, identifier_types)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, quotechar='"', quoting=csv.QUOTE_ALL)

    print(
        "Dataset has been created and saved to "
        f"'{output_path}' using identifier types {sorted(identifier_types)} "
        f"and paragraph limit {paragraph_limit}."
    )
    print(df.head())


if __name__ == "__main__":
    main()
