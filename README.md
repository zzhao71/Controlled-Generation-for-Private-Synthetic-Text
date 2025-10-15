# Controlled Generation for Private Synthetic Text

Implementation of prefix tuning and prompt-based strategies for generating privacy-preserving synthetic text.


## Repository layout
- `generate.py` – prepares fine-tuning CSVs by selecting the first six paragraphs of each document and aggregating control codes.
- `masking.py`, `baseline_icl.py`, `partially_known_icl.py`, `partially_known_prefix_tuning.py`, `in_context_learning*.py` – prompt-based and partially observed baselines used for comparison with prefix tuning.
- `prefix_tuning.py` – trains a prefix-tuned `princeton-nlp/Sheared-LLaMA-1.3B` model on the generated CSVs.
- `prefix_tuning_inference.py` – evaluates a tuned model against privacy and utility metrics using controllable prompts.
- `run_experiments.py` – unified entry point that dispatches to any of the training, inference, and evaluation scripts.
- `privacy_metrics/` – scripts for evaluating privacy and utility metrics.

## Prerequisites
- Python 3.8 or newer 
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
- `icl`, `icl_privacy` – in-context learning with or without privacy enhancement
- `prefix_tuning`, `masking` – prefix tuning and masking-based prefix tuning
- `prefix_eval`, `masking_eval` – privacy evaluation scripts
- `perplexity` – evaluate perplexity of the base model on the dataset

To run a task, here are some example commands:
```bash
python run_experiments.py prefix_tuning
python run_experiments.py prefix_eval --max-eval-samples 50
python run_experiments.py partially_known_prefix --context-size 2 --max-rows 60
```

Any extra arguments after the task name are forwarded to the underlying script. Use `--` to separate runner flags from script-specific options when needed.

## Preparing fine-tuning data
`generate.py` reads `extracted_data.json`, gathers annotated entity mentions within the first six paragraphs, and produces a CSV ready for prefix tuning.

```bash
python generate.py \
  --input data/tab/echr_train.json \
  --output fine_tuning_data_first_six_paragraphs_direct.csv \
  --paragraphs 6 \
  --identifier-types DIRECT
```
The script writes `fine_tuning_data_first_six_paragraphs_direct.csv` to the working directory. Adjust the filtering logic inside `generate.py` if you want to keep additional identifier types (e.g., QUASI entities).
Adjust paragraphs, input, and output paths as needed.

## Finetuning with prefix tuning

### Prefix tuning
`prefix_tuning.py` fine-tunes the Sheared LLaMA model using the generated CSVs. Ensure the training and evaluation CSV paths in the script point to the files you created.

```bash
python run_experiments.py prefix_tuning \
  --model-name princeton-nlp/Sheared-LLaMA-1.3B \
  --train-csv fine_tuning_data_first_six_paragraphs_direct.csv \
  --eval-csv fine_tuning_data_first_six_paragraphs_direct.csv \
  --checkpoint-dir sheared-llama-prefix-tuned \
  --virtual-token-count 20 \
  --batch-size 4 \
  --num-epochs 3 \
  --learning-rate 5e-5
```
The script saves checkpoints under `sheared-llama-prefix-tuned/final_checkpoint/`. Modify the hyperparameters (batch size, number of epochs, learning rate, virtual token count) to match your hardware budget.

### Masking-based prefix tuning
`masking.py` implements prefix tuning with entity masking as described in the paper. Run it similarly to `prefix_tuning.py`, but add the `--masking` flag to enable masking.

```bash
python run_experiments.py masking \ 
  --model-name princeton-nlp/Sheared-LLaMA-1.3B \
  --train-csv fine_tuning_data_first_six_paragraphs_direct.csv \
  --eval-csv fine_tuning_data_first_six_paragraphs_direct.csv \
  --output-dir sheared-llama-privacy-prefix \
```
The specific hyperparameters can be adjusted as needed inside the `masking.py` file.

### Running inference
After training, run inference using `prefix_tuning_inference.py` for both standard and masking-based prefix-tuned models.

```bash
python prefix_tuning_inference.py \
  --checkpoint-dir sheared-llama-prefix-tuned \
  --checkpoint-name final_checkpoint \
  --input-csv fine_tuning_data_first_six_paragraphs_direct.csv \
  --output-file prefix_tuning_eval_results.json \
  --max-eval-samples 100 \
  --temperature 0.7 \
  --top-p 0.9
```

Use `--model-name` to override the base model, `--max-new-tokens` to adjust generation length, or omit `--max-eval-samples` to evaluate the full dataset. `checkpoint-name` can choose which model checkpoint to load.

### Partially known control codes
Use the partially observed baselines to test scenarios where only limited private information is known:

```bash
python partially_known_icl.py --context-size 3
python partially_known_prefix_tuning.py \ 
  --checkpoint-dir sheared-llama-partial-privacy-prefix \
```

Both scripts derive control codes with Presidio. The hyperparameters can be adjusted as needed.

## Privacy and utility evaluation
The privacy metrics in `privacy_metrics/` and each in-context learning or finetuning method file has already incorporated it to report the values. Utility metrics include perplexity is in `evaluation_perplexity.py`. It can be ran with
```bash
python evaluation_perplexity.py 
```     

## Citation
If you use this code in your research, please cite the following paper:

```@inproceedings{zhao2025controlled,
  title={Controlled Generation for Private Synthetic Text},
  author={Zhao, Zihao and Field, Anjalie},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2025}
}
``` 

