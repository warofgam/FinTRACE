"""Load configuration from YAML. Dataset config in configs/dataset/."""

from pathlib import Path

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent

# Load main config
with open(_CONFIG_DIR / "config.yaml", encoding="utf-8") as f:
    _main = yaml.safe_load(f)

# Load dataset config (path in main config is relative to config dir)
_dataset_path = _CONFIG_DIR / _main["dataset"]
with open(_dataset_path, encoding="utf-8") as f:
    _ds = yaml.safe_load(f)

# Dataset name for few-shot/parsing dispatch: gender, rosbank, df_2024
DATASET_NAME = _dataset_path.stem

# --- Expose dataset (for backward compatibility) ---
DATA_DIR = _ds["data_dir"]
STATS_CSV = _ds["stats_csv"]
TEST_IDS_CSV = _ds["test_ids_csv"]
ID_COL = _ds["id_col"]
TARGET_COL = _ds["target_col"]

# Optional: mapping from original column names to human-readable (e.g. for rosbank)
FEATURE_MAPPING = _ds.get("feature_mapping") or {}

TRANSACTION_FREQUENCY_FEATURES = _ds["transaction_frequency_features"]
TEMPORAL_PATTERN_FEATURES = _ds["temporal_pattern_features"]
FINANCIAL_METRIC_FEATURES = _ds["financial_metric_features"]
SPENDING_PATTERN_FEATURES = _ds["spending_pattern_features"]
MCC_PATTERN_FEATURES = _ds["mcc_pattern_features"]

# --- Expose main (model, sampling, inference) ---
MODEL_CONFIG = {
    "model": _main["model"]["name"],
    "tensor_parallel_size": _main["model"]["tensor_parallel_size"],
    "gpu_memory_utilization": _main["model"]["gpu_memory_utilization"],
    "max_model_len": _main["model"]["max_model_len"],
    "enable_prefix_caching": _main["model"]["enable_prefix_caching"],
    "dtype": _main["model"]["dtype"],
}
SAMPLING_CONFIG = _main["sampling"]
NUM_SHOTS = _main["inference"]["num_shots"]
