"""
Универсальный инференс модели, обученной на instruct-датасете (churn / gender), по конфигу.
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
from tqdm import tqdm

from config_loader import load_config, resolve_path, get_inference_paths


def load_model_and_tokenizer(model_path: str, base_model_id: str, max_lora_rank: int = 128, tensor_parallel_size: int = 8, max_model_len: int = 4096):
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    llm = LLM(
        model=base_model_id,
        enable_lora=True,
        max_lora_rank=max_lora_rank,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
    )
    lora_name = "sft_adapter"
    lora_request = LoRARequest(lora_name, 1, model_path)
    return llm, tokenizer, lora_request


# --------------- Churn: YES/NO -> CHURN/LOYAL ---------------

def extract_prediction_churn(generated_text: str) -> str | None:
    text = generated_text.strip()
    verdict_pattern = r"Verdict:\s*(YES|NO)\b"
    match = re.search(verdict_pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    text_upper = text.upper()
    yes_idx, no_idx = text_upper.rfind("YES"), text_upper.rfind("NO")
    if yes_idx == -1 and no_idx == -1:
        return None
    return "YES" if yes_idx > no_idx else "NO"


def predicted_to_label_churn(verdict: str, verdict_to_label: dict) -> str | None:
    if verdict is None:
        return None
    return verdict_to_label.get(verdict.upper())


# --------------- Gender: Male/Female -> 0/1 ---------------

def extract_prediction_gender(generated_text: str) -> int | None:
    text = generated_text.strip()
    verdict_pattern = r"Verdict:\s*(Male|Female)\b"
    match = re.search(verdict_pattern, text, re.IGNORECASE)
    if match:
        return 1 if match.group(1).lower() == "male" else 0
    text_upper = text.upper()
    male_idx, female_idx = text_upper.rfind("MALE"), text_upper.rfind("FEMALE")
    if male_idx == -1 and female_idx == -1:
        return None
    return 1 if male_idx > female_idx else 0


def predicted_to_label_gender(pred: int, verdict_to_label: dict) -> int:
    return pred  # already 0/1


# --------------- Evaluation ---------------

def load_true_targets_churn(stats_path: str, id_col: str, target_col: str) -> dict:
    stats_df = pd.read_csv(stats_path)
    target_dict = {}
    for _, row in stats_df.iterrows():
        cid = int(row[id_col])
        flag = int(row[target_col])
        target_dict[cid] = "CHURN" if flag == 0 else "LOYAL"
    return target_dict


def load_true_targets_gender(stats_path: str, test_ids_path: str, id_col: str, target_col: str, by_order: bool = False) -> dict | list:
    stats_df = pd.read_csv(stats_path)
    test_ids = pd.read_csv(test_ids_path)[id_col].tolist()
    test_stats = stats_df[stats_df[id_col].isin(test_ids)]

    def to_01(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        v = str(val).upper().strip()
        if v in ("M", "MALE", "1"):
            return 1
        if v in ("F", "FEMALE", "0"):
            return 0
        try:
            return int(val) if int(val) in (0, 1) else None
        except (ValueError, TypeError):
            return None

    if by_order:
        return [to_01(test_stats[test_stats[id_col] == cid].iloc[0][target_col]) if len(test_stats[test_stats[id_col] == cid]) else None for cid in test_ids]
    return {row[id_col]: to_01(row[target_col]) for _, row in test_stats.iterrows()}


def evaluate_model(
    cfg,
    llm,
    tokenizer,
    test_df,
    lora_request,
    output_file=None,
    metrics_file=None,
    max_new_tokens=1024,
    temperature=0.0,
    batch_size=8,
    log_file=None,
):
    task = cfg["task"]
    ds = cfg["dataset"]
    data_dir = cfg["data_dir"]
    id_col = ds["id_col"]
    verdict_to_label = ds.get("verdict_to_label") or {}
    pos_label = ds.get("pos_label", "CHURN")
    stats_path = resolve_path(data_dir, ds["stats_csv"])
    test_ids_path = resolve_path(data_dir, ds["test_ids_csv"])

    def _pred_to_label_churn(v):
        return predicted_to_label_churn(v, verdict_to_label)

    def _pred_to_label_gender(v):
        return v  # already 0/1

    if task == "churn":
        extract_pred = extract_prediction_churn
        predicted_to_label = _pred_to_label_churn
        true_targets = load_true_targets_churn(stats_path, id_col, ds["target_col"])
        labels_metric = ["CHURN", "LOYAL"]
        label_names = None
    else:
        extract_pred = extract_prediction_gender
        predicted_to_label = _pred_to_label_gender
        use_order = id_col not in test_df.columns
        true_targets = load_true_targets_gender(stats_path, test_ids_path, id_col, ds["target_col"], by_order=use_order)
        labels_metric = ds.get("labels_metric", [0, 1])
        label_names = ds.get("label_names", ["0 (F)", "1 (M)"])
        pos_label = ds.get("pos_label", 1)

    # Build prompts and meta
    rows_meta = []
    prompts = []
    for pos, (idx, row) in enumerate(test_df.iterrows()):
        messages = row.get("prompt")
        if messages is None and "instruction" in row:
            messages = [
                {"role": "system", "content": row["instruction"]},
                {"role": "user", "content": row["input"]},
            ]
        if messages is None:
            continue
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(text)
        sid = row.get(id_col, idx)
        if task == "churn":
            gt = true_targets.get(int(sid)) if isinstance(sid, (int, float)) or (isinstance(sid, str) and sid.isdigit()) else row.get("answer") or row.get("output")
        else:
            gt = true_targets.get(sid) if isinstance(true_targets, dict) else (true_targets[pos] if pos < len(true_targets) else None)
            if gt is None and "output" in row:
                m = re.search(r"Verdict:\s*(Male|Female)", str(row.get("output", "")), re.I)
                if m:
                    gt = 1 if m.group(1).lower() == "male" else 0
        rows_meta.append({"idx": idx, "pos": pos, "sample_id": sid, "true_label": gt})

    sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=temperature, top_p=0.95)
    predictions = []
    true_labels = []
    sample_ids = []
    full_outputs = []
    verdicts_raw = []
    log_f = open(log_file, "a", encoding="utf-8") if log_file else None

    for start in tqdm(range(0, len(prompts), batch_size), desc=f"Instruct vLLM ({task})"):
        batch_prompts = prompts[start : start + batch_size]
        batch_meta = rows_meta[start : start + batch_size]
        outputs = llm.generate(batch_prompts, sampling_params, lora_request=lora_request)
        for out, meta in zip(outputs, batch_meta):
            generated_text = out.outputs[0].text.strip()
            pred_verdict = extract_pred(generated_text)
            pred_label = predicted_to_label(pred_verdict)
            if log_f:
                log_f.write(f"=== {id_col}: {meta['sample_id']} ===\n")
                log_f.write(f"GT: {meta['true_label']} | PRED: {pred_verdict} -> {pred_label}\n")
                log_f.write(f"OUTPUT:\n{generated_text}\n")
                log_f.write("=" * 50 + "\n")
            if pred_label is None:
                continue
            if meta["true_label"] is not None:
                predictions.append(pred_label)
                true_labels.append(meta["true_label"])
                sample_ids.append(meta["sample_id"])
                full_outputs.append(generated_text)
                if task == "churn":
                    verdicts_raw.append(pred_verdict)
    if log_f:
        log_f.close()

    if not predictions:
        print("Нет предсказаний для метрик.")
        return

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, pos_label=pos_label)
    mcc = matthews_corrcoef(true_labels, predictions)
    cm = confusion_matrix(true_labels, predictions, labels=labels_metric)
    report_str = classification_report(true_labels, predictions, labels=labels_metric, target_names=label_names)

    metrics = {
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "mcc": float(mcc),
        "confusion_matrix": cm.tolist(),
        "labels": labels_metric,
        "classification_report": report_str,
        "n_predictions": len(predictions),
    }
    print(f"\n{'='*60}\nРезультаты ({task} instruct):\n{'='*60}")
    print(f"Accuracy: {accuracy:.4f}\nF1: {f1:.4f}\nMCC: {mcc:.4f}\n")
    print(f"Confusion Matrix:\n{cm}\n")
    print(report_str)
    print("=" * 60)

    if metrics_file:
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"Метрики сохранены в {metrics_file}")

    if output_file:
        out_df = pd.DataFrame({id_col: sample_ids, "true_label": true_labels, "predicted_label": predictions, "full_response": full_outputs})
        if task == "churn":
            out_df.insert(2, "predicted_verdict", verdicts_raw)
        out_df.to_csv(output_file, index=False)
        print(f"Результаты сохранены в {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Inference on instruct model from config")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint name inside model_dir")
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--test_data", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--metrics", type=str, default=None)
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--log_file", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    test_data, model_dir = get_inference_paths(cfg)
    inf = cfg["inference"]

    model_dir = args.model_dir or model_dir
    checkpoint = args.checkpoint or inf.get("checkpoint", "checkpoint-108")
    model_path = f"{model_dir}/{checkpoint}"
    test_data = args.test_data or test_data
    output_file = args.output or inf.get("output_file", "instruct_inference_results.csv")
    metrics_file = args.metrics or inf.get("metrics_file")
    base_model = args.base_model or cfg["base_model"]
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else inf.get("max_new_tokens", 1024)
    temperature = args.temperature if args.temperature is not None else inf.get("temperature", 0.0)
    batch_size = args.batch_size or inf.get("batch_size", 1000)
    log_file = args.log_file or f"{cfg['task']}_instruct_generated_log.txt"

    if not Path(test_data).exists():
        raise FileNotFoundError(f"Test data not found: {test_data}")
    test_df = pd.read_json(test_data, lines=True)
    if cfg["task"] == "churn" and "cl_id" not in test_df.columns and "customer_id" in test_df.columns:
        test_df = test_df.rename(columns={"customer_id": "cl_id"})

    llm, tokenizer, lora_request = load_model_and_tokenizer(
        model_path,
        base_model,
        max_lora_rank=inf.get("max_lora_rank", 128),
        tensor_parallel_size=inf.get("tensor_parallel_size", 8),
        max_model_len=inf.get("max_model_len", 8192),
    )
    evaluate_model(
        cfg, llm, tokenizer, test_df, lora_request,
        output_file=output_file,
        metrics_file=metrics_file,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        batch_size=batch_size,
        log_file=log_file,
    )


if __name__ == "__main__":
    main()
