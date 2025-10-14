"""Unified entry point for running the different generation and evaluation experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

SCRIPT_MAP = {
    "baseline": PROJECT_ROOT / "baseline_icl.py",
    "partially_known_icl": PROJECT_ROOT / "partially_known_icl.py",
    "partially_known_prefix": PROJECT_ROOT / "partially_known_prefix_tuning.py",
    "icl": PROJECT_ROOT / "in_context_learning.py",
    "icl_privacy": PROJECT_ROOT / "in_context_learning_privacy_guarantee.py",
    "prefix_tuning": PROJECT_ROOT / "prefix_tuning.py",
    "masking": PROJECT_ROOT / "masking.py",
    "prefix_eval": PROJECT_ROOT / "prefix_tuning_inference.py",
    "masking_eval": PROJECT_ROOT / "prefix_tuning_inference.py",
}


DEFAULT_EXTRA_ARGS = {
    "prefix_eval": [
        "--checkpoint-dir",
        "sheared-llama-prefix-tuned",
        "--checkpoint-name",
        "final_checkpoint",
        "--output-file",
        "prefix_tuning_eval_results.json",
    ],
    "masking_eval": [
        "--checkpoint-dir",
        "sheared-llama-privacy-prefix",
        "--checkpoint-name",
        "final",
        "--output-file",
        "masking_eval_results.json",
    ],
}

TASK_CHOICES = tuple(SCRIPT_MAP.keys()) + ("evaluate_all",)


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run one of the generation or evaluation pipelines (ICL, prefix tuning, masking)."
    )
    parser.add_argument(
        "task",
        choices=TASK_CHOICES,
        help="Which pipeline to execute.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use when launching the selected script.",
    )
    return parser.parse_known_args(argv)


def run_task(task: str, python_bin: str, extra_args: list[str]) -> int:
    if task == "evaluate_all":
        for sub_task in ("prefix_eval", "masking_eval"):
            return_code = run_task(sub_task, python_bin, extra_args)
            if return_code != 0:
                return return_code
        return 0

    script = SCRIPT_MAP[task]
    if not script.exists():
        raise FileNotFoundError(f"Expected script not found: {script}")

    default_args = DEFAULT_EXTRA_ARGS.get(task, [])
    cmd = [python_bin, str(script), *default_args, *extra_args]
    print("$", " ".join(cmd))
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    args, extra = parse_args(argv)
    return run_task(args.task, args.python, extra)


if __name__ == "__main__":
    sys.exit(main())
