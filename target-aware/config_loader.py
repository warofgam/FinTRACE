"""
Загрузка конфигурации для target-aware pipeline.
Все пути в dataset конфиге считаются относительно data_dir, кроме data_dir.
"""
from pathlib import Path
import argparse
import yaml

_ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = _ROOT / "configs"


def load_config(config_path=None):
    if config_path is None:
        config_path = CONFIGS_DIR / "config.yaml"
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = _ROOT / config_path
    with open(config_path, encoding="utf-8") as f:
        main = yaml.safe_load(f)

    dataset_path = config_path.parent / main["dataset"]
    if not dataset_path.is_absolute():
        dataset_path = config_path.parent / main["dataset"]
    with open(dataset_path, encoding="utf-8") as f:
        dataset = yaml.safe_load(f)

    data_dir = main.get("data_dir") or dataset.get("data_dir", ".")
    if not Path(data_dir).is_absolute():
        data_dir = str(_ROOT / data_dir)
    dataset["data_dir"] = data_dir
    dataset["_dataset_path"] = str(dataset_path)

    return {
        "task": main.get("task") or dataset.get("task"),
        "base_model": main.get("base_model", "meta-llama/Meta-Llama-3-8B-Instruct"),
        "data_dir": data_dir,
        "dataset": dataset,
        "sft": main.get("sft", {}),
        "inference": main.get("inference", {}),
    }


def resolve_path(data_dir, rel_path):
    if not rel_path:
        return None
    p = Path(rel_path)
    if p.is_absolute():
        return str(p)
    return str(Path(data_dir) / rel_path)


def get_train_eval_paths(cfg):
    data_dir = Path(cfg["data_dir"])
    task = cfg["task"]
    sft = cfg["sft"]
    train = sft.get("train_data") or str(data_dir / f"{task}_instruct_train.jsonl")
    eval_ = sft.get("eval_data") or str(data_dir / f"{task}_instruct_test.jsonl")
    if not Path(train).is_absolute():
        train = str(data_dir / train)
    if not Path(eval_).is_absolute():
        eval_ = str(data_dir / eval_)
    return train, eval_


def get_output_dir(cfg):
    sft = cfg["sft"]
    out = sft.get("output_dir")
    if out:
        return out
    return str(Path("models") / f"llama-8b-{cfg['task']}-instruct")


def get_inference_paths(cfg):
    data_dir = cfg["data_dir"]
    task = cfg["task"]
    inf = cfg["inference"]
    test_data = inf.get("test_data") or str(Path(data_dir) / f"{task}_instruct_test.jsonl")
    if not Path(test_data).is_absolute():
        test_data = str(Path(data_dir) / Path(test_data).name)
    model_dir = inf.get("model_dir") or str(Path("models") / f"llama-8b-{task}-instruct")
    return test_data, model_dir


def parse_args_common(description=""):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    return p
