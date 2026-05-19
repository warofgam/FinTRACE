# FinTRACE

**Authors:** Artem Sakhno, Daniil Tomilov, Yuliana Shakhvalieva, Inessa Fedorova, Daria Ruzanova, Omar Zoloev, Andrey Savchenko, Maksim Makarenko  
**Paper:** [Financial Transaction Retrieval and Contextual Evidence for Knowledge-Grounded Reasoning](https://arxiv.org/abs/2603.15459)  
**Code:** [github.com/warofgam/FinTRACE](https://github.com/warofgam/FinTRACE)

<p align="center">
  <img src="assets/fintrace_overview.svg" alt="FinTRACE overview" width="920">
</p>

FinTRACE is a retrieval-first framework for knowledge-grounded reasoning over financial transaction histories. Instead of serializing raw transaction rows directly into an LLM prompt, FinTRACE converts transaction sequences into a structured behavioral knowledge base: reusable feature essences, interpretable behavioral patterns, white-box rules, and task-level evidence. The resulting context can be used for zero-shot and few-shot prediction, as well as for instruction tuning of LLMs on transaction analytics tasks.

The repository contains research code for two complementary pipelines:

- **Target-agnostic reasoning:** few-shot prompting over transaction-derived features and behavioral context.
- **Target-aware instruction tuning:** rule-grounded instruction dataset construction, LoRA SFT, and vLLM inference.

## Why FinTRACE?

General-purpose LLMs are not naturally good at long, irregular, tabular transaction histories. Transactional signal is distributed across time, amounts, merchant categories, frequency patterns, and institution-specific schemas. FinTRACE inserts an explicit intermediate layer between raw events and the LLM:

1. Extract numerical transaction features.
2. Convert them into behavioral evidence.
3. Connect evidence with white-box rules.
4. Retrieve task-relevant facts and rules.
5. Ask the LLM to reason over structured evidence instead of raw rows.

This makes the prediction process more interpretable and reusable across downstream tasks such as churn prediction, demographic inference, and other financial behavior modeling problems.

## Method overview

FinTRACE organizes transactional evidence into three semantic layers:

| Layer | Meaning | Example |
| --- | --- | --- |
| Feature essences | Task-agnostic numerical descriptors computed from transaction histories | transaction frequency, spending variance, income regularity |
| Behavioral patterns | Higher-level interpretable concepts built from feature essences | financial stability, behavioral loyalty, discretionary spending |
| Downstream targets | Task-specific objectives connected to the behavioral evidence | churn, default risk, user attributes |

White-box rules connect these layers and produce traceable evidence chains. A rule can be rendered in natural language, for example:

```text
IF activity_period_days <= 70.5 -> strong churn signal
```

At inference time, the model receives a compact context made of retrieved behavioral facts and rules, not the full transaction table.

## Repository structure

```text
FinTRACE/
├── target-agnostic/
│   ├── configs/
│   │   ├── config.py
│   │   ├── config.yaml
│   │   └── dataset/
│   ├── prompts/
│   │   ├── df_2024.py
│   │   ├── gender.py
│   │   └── rosbank.py
│   ├── whitebox/
│   │   ├── datafusion.ipynb
│   │   ├── gender.ipynb
│   │   └── rosbank.ipynb
│   ├── few_shot.py
│   └── utils.py
├── target-aware/
│   ├── configs/
│   │   ├── config.yaml
│   │   └── datasets/
│   │       ├── churn.yaml
│   │       └── gender.yaml
│   ├── accelerate_config.yaml
│   ├── config_loader.py
│   ├── create_instruct_dataset.py
│   ├── sft_train.py
│   ├── instruct_inference.py
│   └── README.md
└── README.md
```

## Pipelines

### 1. Target-agnostic few-shot reasoning

The target-agnostic pipeline evaluates an LLM directly at inference time. It builds prompts from transaction-derived features, feature statistics, and balanced few-shot examples.

Main entry point:

```bash
cd target-agnostic
python few_shot.py
```

Configuration is controlled by:

```text
target-agnostic/configs/config.yaml
```

The main config selects the dataset config and the vLLM model. The current setup is designed around `openai/gpt-oss-120b` with tensor parallel inference, but the model can be replaced in the config.

### 2. Target-aware instruction tuning

The target-aware pipeline converts white-box behavioral rules into instruction-following training examples, trains a LoRA adapter, and evaluates it with vLLM.

Main steps:

```bash
cd target-aware
python create_instruct_dataset.py --config configs/config.yaml
accelerate launch --config_file accelerate_config.yaml sft_train.py --config configs/config.yaml
python instruct_inference.py --config configs/config.yaml
```

The pipeline supports two task configs out of the box:

| Task | Config | Output format |
| --- | --- | --- |
| Churn prediction | `configs/datasets/churn.yaml` | `Verdict: YES` or `Verdict: NO` |
| Gender prediction | `configs/datasets/gender.yaml` | `Verdict: Male` or `Verdict: Female` |

## Installation

The repository is research code and does not currently ship a locked environment file. A minimal setup should include the libraries used by the scripts:

```bash
git clone https://github.com/warofgam/FinTRACE.git
cd FinTRACE
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn pyyaml datasets transformers trl peft accelerate vllm torch
```

For SFT with FlashAttention, install a PyTorch and `flash-attn` build compatible with your CUDA environment.

## Data preparation

Raw datasets are expected to be placed locally and referenced from the config files. The repository uses precomputed transaction statistics, test IDs, and rule files.

For the target-aware churn setup, the default config expects a layout similar to:

```text
data/
├── stats.csv
├── sber_ai_test_ids.csv
└── rules/
    └── churn.txt
```

For gender prediction, the default config expects:

```text
data/gender/
├── sakhno_stats_dataset_renamed.csv
├── test_ids.csv
└── rules/
    └── gender.txt
```

Update the corresponding dataset YAML files if your files are stored elsewhere.

## Configuration

### Target-agnostic config

Edit:

```text
target-agnostic/configs/config.yaml
```

Important fields:

| Field | Description |
| --- | --- |
| `dataset` | Dataset config path, for example `dataset/df_2024.yaml` |
| `model.name` | vLLM model name |
| `model.tensor_parallel_size` | Number of GPUs for tensor parallel inference |
| `model.max_model_len` | Maximum context length |
| `sampling.temperature` | Generation temperature |
| `inference.num_shots` | Number of few-shot examples |

### Target-aware config

Edit:

```text
target-aware/configs/config.yaml
```

Important fields:

| Field | Description |
| --- | --- |
| `task` | `churn` or `gender` |
| `dataset` | Dataset-specific YAML path |
| `base_model` | Base LLM for SFT |
| `sft.*` | LoRA and training hyperparameters |
| `inference.*` | vLLM inference and adapter settings |

## Results reported in the paper

### Few-shot performance

| Method | Rosbank F1 | Rosbank MCC | Gender F1 | Gender MCC | DataFusion F1 | DataFusion MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt-oss, 0-shot | 0.55 | 0.19 | 0.60 | 0.24 | 0.68 | 0.04 |
| gpt-oss, 16-shot | 0.58 | 0.25 | 0.59 | 0.22 | 0.68 | 0.04 |
| TabPFN v2, 16-shot | 0.49 | 0.01 | 0.61 | 0.27 | 0.71 | 0.09 |
| TabLLM, 16-shot | 0.59 | 0.25 | 0.53 | 0.09 | 0.42 | 0.00 |
| FinTRACE, zero-shot | 0.69 | 0.38 | 0.63 | 0.31 | 0.65 | 0.05 |
| FinTRACE, 16-shot | 0.70 | 0.40 | 0.60 | 0.24 | 0.77 | 0.10 |

### Knowledge-grounded instruction tuning

| Method | Rosbank MCC | Gender MCC | Text score |
| --- | ---: | ---: | ---: |
| Zero-shot | 0.13 | 0.18 | 0.53 |
| FinTRACE target-aware | 0.41 | 0.37 | 0.53 |
| NTP, LLM4ES | 0.40 | 0.04 | 0.00 |
| SFT | 0.43 | 0.52 | 0.53 |
| SFT + Instruct | 0.03 | 0.00 | 0.52-0.53 |
| FinTRACE Instruct | 0.48 | 0.53 | 0.53 |

## Adding a new task

To add a new target, create a dataset config and connect it to the dataset generation and inference code.

1. Add a new YAML file under `target-aware/configs/datasets/`.
2. Define paths for statistics, test IDs, rules, ID column, target column, and label mapping.
3. Add task-specific instruction dataset generation in `create_instruct_dataset.py`.
4. Add task-specific prediction extraction and metrics in `instruct_inference.py`.
5. Point `target-aware/configs/config.yaml` to the new task config.

## Practical notes

- Keep raw transaction data out of the repository.
- Store only reproducible configs, scripts, and non-sensitive derived artifacts.
- Check that label semantics are consistent across configs, prompts, metrics, and final verdict parsing.
- For vLLM inference, match `tensor_parallel_size` to the number of available GPUs.
- For LoRA SFT, adjust `per_device_train_batch_size`, `gradient_accumulation_steps`, and `max_length` to fit your GPU memory.

## Citation

If you use this repository or build on FinTRACE, please cite:

```bibtex
@misc{sakhno2026fintrace,
  title={Financial Transaction Retrieval and Contextual Evidence for Knowledge-Grounded Reasoning},
  author={Sakhno, Artem and Tomilov, Daniil and Shakhvalieva, Yuliana and Fedorova, Inessa and Ruzanova, Daria and Zoloev, Omar and Savchenko, Andrey and Makarenko, Maksim},
  year={2026},
  eprint={2603.15459},
  archivePrefix={arXiv},
  primaryClass={cs.IR}
}
```

## License

No license file is currently included in the repository. Add a license before releasing the code for external reuse.
