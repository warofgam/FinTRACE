# Target-aware pipeline: instruct dataset, SFT, inference

A single pipeline for **churn** and **gender** tasks: build rule-based instruct datasets, run SFT training, and run inference with vLLM. Everything is driven by config.

## Structure

```
target-aware/
├── accelerate_config.yaml   # Accelerate config (multi-GPU, bf16)
├── configs/
│   ├── config.yaml          # Main config (task, model, sft, inference)
│   └── datasets/
│       ├── churn.yaml       # Data and rules for churn
│       └── gender.yaml     # Data and rules for gender
├── config_loader.py         # Config loading and path resolution
├── create_instruct_dataset.py  # Build train/test jsonl
├── sft_train.py             # SFT on instruct dataset
├── instruct_inference.py   # Inference and metrics
└── README.md
```

## Switching tasks

In `configs/config.yaml` set:

- **task**: `churn` or `gender`
- **dataset**: path to the dataset config relative to `configs/` (e.g. `datasets/churn.yaml` or `datasets/gender.yaml`)

Other paths (train/eval/test, model_dir, etc.) default from `data_dir` and `task`.

## Commands

### 1. Create instruct dataset

Generates `{data_dir}/{task}_instruct_train.jsonl` and `{task}_instruct_test.jsonl` from the rules in the dataset config’s `rules_file`.

```bash
python create_instruct_dataset.py --config configs/config.yaml
# Override outputs:
python create_instruct_dataset.py --config configs/config.yaml --train_out /path/to/train.jsonl --test_out /path/to/test.jsonl
```

### 2. SFT training

Trains LoRA on the instruct dataset (messages format: system / user / assistant). Run with **Accelerate** (multi-GPU):

```bash
accelerate launch --config_file accelerate_config.yaml sft_train.py --config configs/config.yaml
```

Override script arguments:

```bash
accelerate launch --config_file accelerate_config.yaml sft_train.py \
  --config configs/config.yaml \
  --output_dir models/my-run --learning_rate 1e-4 --num_train_epochs 3
```

`accelerate_config.yaml` in the target-aware root: 8 GPUs, bf16, MULTI_GPU (DDP). For a different number of GPUs, change `num_processes` or generate a new config with `accelerate config`.

### 3. Inference and metrics

Loads a checkpoint, runs on the test set, and computes accuracy / F1 / MCC.

```bash
python instruct_inference.py --config configs/config.yaml
# Overrides:
python instruct_inference.py --config configs/config.yaml --checkpoint checkpoint-240 --output results.csv --metrics metrics.json
```

## Dataset configs

- **churn**: `stats_csv`, `test_ids_csv`, `rules_file` (churn.txt), `feature_mapping`, `rules_feature_to_col`. Verdict in dataset: YES/NO → metrics CHURN/LOYAL.
- **gender**: `stats_csv`, `test_ids_csv`, `rules_file` (gender.txt), `feature_order`. Verdict: Male/Female → metrics 0/1. Optional `few_shot_ids_file` to restrict training to specific IDs.

Paths in the dataset config are relative to `data_dir` (except `data_dir` itself).

## Adding a new task

1. Add `configs/datasets/<task>.yaml` with: `task`, `data_dir`, `stats_csv`, `test_ids_csv`, `id_col`, `target_col`, `rules_file`, `verdict_output`, `verdict_to_label`, `pos_label`, and either `feature_mapping` + `rules_feature_to_col` (like churn) or `feature_order` (like gender).
2. In `create_instruct_dataset.py`, add a branch `elif task == "<task>":` and a function `generate_<task>_instruct(...)`.
3. In `instruct_inference.py`, add `extract_prediction_<task>`, label mapping, and if needed target loading in `evaluate_model`.
