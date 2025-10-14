# Controlled Generation for Private Synthetic Text

Implementation of prefix tuning and prompt-based strategies for generating privacy-preserving synthetic text on the Text Anonymization Benchmark (TAB).


## Repository layout
- `generate.py` – prepares fine-tuning CSVs by selecting the first six paragraphs of each document and aggregating control codes.
- `masking.py`, `baseline_icl.py`, `partially_known_icl.py`, `partially_known_prefix_tuning.py`, `in_context_learning*.py` – prompt-based and partially observed baselines used for comparison with prefix tuning.
- `prefix_tuning.py` – trains a prefix-tuned `princeton-nlp/Sheared-LLaMA-1.3B` model on the generated CSVs.
- `prefix_tuning_inference.py` – evaluates a tuned model against privacy and utility metrics using controllable prompts.
- `run_experiments.py` – unified entry point that dispatches to any of the training, inference, and evaluation scripts.
- `privacy_metrics/` – scripts for evaluating privacy and utility metrics.

## Prerequisites


Install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate 
pip install --upgrade pip
pip install -r requirements.txt
```


## Dataset setup
This project consumes the datasets released with TAB and MIMIC-III. Clone the TAB repository next to this project (or anywhere you prefer) and expose its path through an environment variable that the scripts can read.

```bash
python data/download_tab_dataset.py \
  --dest data/tab \
  --branch master \
  --file echr_train.json \
  --file echr_test.json
```

Alternatively, run `python data/download_tab_dataset.py --sync-lfs` to clone the benchmark data into `external/text-anonymization-benchmark`. Use `--dest` to change the output directory, `--update` to pull new commits, and `--branch master` if the default branch name differs. To retrieve only selected files (for example, the ECHR train/test splits), pass `--file` multiple times, e.g. `python data/download_tab_dataset.py --file echr_train.json --file echr_test.json`.


## Unified runner
Launch any of the major pipelines from a single entry point. Available tasks include:

- `baseline` – baseline in-context learning pipeline
- `partially_known_icl` – partially observed control codes with the base model
- `partially_known_prefix` – partially observed control codes with the prefix-tuned model
- `icl`, `icl_privacy` – prompt-based baselines
- `prefix_tuning`, `masking` – finetuning scripts
- `prefix_eval`, `masking_eval` – evaluation shortcuts with sensible defaults
- `evaluate_all` – runs `prefix_eval` followed by `masking_eval`

```bash
python run_experiments.py prefix_tuning
python run_experiments.py prefix_eval --max-eval-samples 50
python run_experiments.py partially_known_prefix --context-size 2 --max-rows 60
python run_experiments.py masking -- --learning-rate 1e-4
```

Any extra arguments after the task name are forwarded to the underlying script. Use `--` to separate runner flags from script-specific options when needed.

## Preparing fine-tuning data
`generate.py` reads `extracted_data.json`, gathers annotated entity mentions within the first six paragraphs, and produces a CSV ready for prefix tuning.

```bash
python generate.py
```

The script writes `fine_tuning_data_first_six_paragraphs_direct.csv` to the working directory. Adjust the filtering logic inside `generate.py` if you want to keep additional identifier types (e.g., QUASI entities).

## Training with prefix tuning
`prefix_tuning.py` fine-tunes the Sheared LLaMA model using the generated CSVs. Ensure the training and evaluation CSV paths in the script point to the files you created.

```bash
python prefix_tuning.py
```

The script saves checkpoints under `sheared-llama-prefix-tuned/final_checkpoint/`. Modify the hyperparameters (batch size, number of epochs, learning rate, virtual token count) to match your hardware budget.

### Running inference
After training, run inference using `prefix_tuning_inference.py`. The script exposes CLI flags so you can point to any checkpoint and evaluation CSV without editing the file.

```bash
python prefix_tuning_inference.py \
  --checkpoint-dir sheared-llama-prefix-tuned \
  --checkpoint-name final_checkpoint \
  --input-csv fine_tuning_data_first_six_paragraphs_direct.csv \
  --output-file prefix_tuning_eval_results.json \
  --max-eval-samples 100 \
  --temperature 0.7 --top-p 0.9
```

Use `--model-name` to override the base model, `--max-new-tokens` to adjust generation length, or omit `--max-eval-samples` to evaluate the full dataset.

### Partially known control codes
Use the partially observed baselines to test scenarios where only limited private information is known:

```bash
python partially_known_icl.py --context-size 2 --max-rows 60
python partially_known_prefix_tuning.py --checkpoint-dir sheared-llama-privacy-prefix --checkpoint-name final_checkpoint --context-size 2
```

Both scripts derive control codes with Presidio, mix in synthetic distractors, and report privacy/utility metrics while writing JSONL outputs for inspection.

## Privacy and utility evaluation
The utilities in `privacy_metrics/` and the prompt-based baselines (`baseline_icl.py`, `partially_known_icl.py`, `partially_known_prefix_tuning.py`, `in_context_learning_privacy_guarantee.py`) reproduce the metrics reported in the project. Run these scripts after generation to compare against the anonymization baselines.

## Citation
If you use this code or the TAB data in academic work, please cite the original Text Anonymization Benchmark publication and any relevant privacy-preserving generation papers referenced in your research.

## License
This repository inherits the licensing terms of the TAB dataset for any redistributed data. See `LICENSE.txt` in the TAB repository for details, and consult the license headers within this project before distributing derivative work.
