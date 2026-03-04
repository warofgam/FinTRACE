"""
Универсальное создание instruct-датасета (churn / gender) по конфигу.
Использует правила из rules_file и формирует записи instruction / input / output.
"""
import re
import json
import argparse
from pathlib import Path

import pandas as pd

from config_loader import load_config, CONFIGS_DIR, resolve_path


# --------------- Churn: rule parsing and record building ---------------

def _parse_rules_churn(rules_path: str, rules_feature_to_col: dict) -> dict:
    path = Path(rules_path)
    if not path.exists():
        raise FileNotFoundError(f"Rules file not found: {rules_path}")
    text = path.read_text(encoding="utf-8").strip()
    missing_re = re.compile(
        r"If the value of the feature (.+?) is missing, this is (.+?) signal\.", re.IGNORECASE
    )
    cond_re = re.compile(
        r"If (.+?) is (less than or equal to|between|greater than) ([^,]+), this is (.+?) signal\."
    )
    rules_by_col = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = missing_re.match(line)
        if m:
            feat, signal = m.group(1).strip(), m.group(2).strip()
            col = rules_feature_to_col.get(feat)
            if col is None:
                continue
            rules_by_col.setdefault(col, []).append(("missing", None, None, signal))
            continue
        m = cond_re.match(line)
        if not m:
            continue
        feat, cond_type, rest, signal = m.group(1).strip(), m.group(2).strip().lower(), m.group(3).strip(), m.group(4).strip()
        col = rules_feature_to_col.get(feat)
        if col is None:
            continue
        if "less than or equal to" in cond_type:
            try:
                bound = float(rest)
                rules_by_col.setdefault(col, []).append(("le", bound, None, signal))
            except ValueError:
                pass
        elif "between" in cond_type:
            part = re.match(r"([\d.]+)\s+and\s+([\d.]+)", rest)
            if part:
                low, high = float(part.group(1)), float(part.group(2))
                rules_by_col.setdefault(col, []).append(("between", low, high, signal))
        elif "greater than" in cond_type:
            try:
                bound = float(rest)
                rules_by_col.setdefault(col, []).append(("gt", bound, None, signal))
            except ValueError:
                pass
    return rules_by_col


def _format_val_churn(v: float, col: str) -> str:
    if col in ("transaction_period", "unique_transaction_days", "transaction_count"):
        return str(int(round(v)))
    if col == "income_expense_ratio":
        return f"{v:.2f}"
    return f"{v:,.0f}".replace(",", " ")


def _apply_rules_churn(row: pd.Series, rules_by_col: dict, col_to_readable: dict) -> list:
    lines = []
    for col, rule_list in rules_by_col.items():
        val = row.get(col)
        readable = col_to_readable.get(col, col)
        if pd.isna(val) or (isinstance(val, float) and str(val).strip() == ""):
            for cond, low, high, signal in rule_list:
                if cond == "missing":
                    lines.append(f"The value for {readable.lower()} is missing; this is a {signal} signal.")
                    break
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        v_str = _format_val_churn(v, col)
        for cond, low, high, signal in rule_list:
            if cond == "missing":
                continue
            if cond == "le" and v <= low:
                lines.append(f"{readable} is {v_str}, at or below the threshold and indicates a {signal} signal.")
                break
            if cond == "between" and low < v <= high:
                lines.append(f"{readable} is {v_str}, in the range that gives a {signal} signal.")
                break
            if cond == "gt" and v > low:
                lines.append(f"{readable} is {v_str}, above the threshold, which corresponds to a {signal} signal.")
                break
    return lines


def generate_churn_instruct(df: pd.DataFrame, ds_cfg: dict, output_file: str) -> None:
    data_dir = ds_cfg["data_dir"]
    rules_path = resolve_path(data_dir, ds_cfg["rules_file"])
    id_col = ds_cfg["id_col"]
    target_col = ds_cfg["target_col"]
    feature_mapping = ds_cfg.get("feature_mapping", {})
    rules_feature_to_col = {k: v for k, v in ds_cfg.get("rules_feature_to_col", {}).items()}
    col_to_readable = {v: feature_mapping.get(v, v) for v in rules_feature_to_col.values() if v in feature_mapping}
    col_to_readable.update({c: feature_mapping.get(c, c) for c in feature_mapping})

    rules_by_col = _parse_rules_churn(rules_path, rules_feature_to_col)
    instruction = (
        "You are an expert banking analyst. Based on the customer description below, "
        "classify whether the customer will churn: answer YES (will churn) or NO (will stay loyal). "
        "First provide your analysis in a section starting with 'Reasoning:'. "
        "Then state the classification in a section starting with 'Verdict:'. "
        "The Verdict must be exactly 'YES' or 'NO'. "
        "The most important thing is to answer YES or NO correctly in the final Verdict."
    )
    records = []
    for _, row in df.iterrows():
        cl_id = row.get(id_col)
        cl_id = "unknown" if pd.isna(cl_id) else int(cl_id)
        profile_lines = []
        for key, description in feature_mapping.items():
            val = row.get(key, "N/A")
            if val == "N/A" or (isinstance(val, float) and pd.isna(val)):
                continue
            if isinstance(val, (int, float)):
                val = round(val, 4)
            profile_lines.append(f"{description}: {val}")
        user_input = "Customer Profile Data:\n" + "\n".join(profile_lines)
        signal_lines = _apply_rules_churn(row, rules_by_col, col_to_readable)
        verdict = "YES" if row[target_col] == 0 else "NO"
        output = "Reasoning: " + "\n".join(signal_lines) + "\nVerdict: " + verdict
        records.append({id_col: cl_id, "instruction": instruction, "input": user_input, "output": output})

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Churn instruct dataset saved to {output_file} with {len(records)} entries.")


# --------------- Gender: rule parsing and record building ---------------

def _normalize_feature_name(s: str) -> str:
    return s.lower().strip().replace('"', '"')


def _parse_rules_gender(rules_path: str, df_columns: list) -> dict:
    path = Path(rules_path)
    if not path.exists():
        raise FileNotFoundError(f"Rules file not found: {rules_path}")
    text = path.read_text(encoding="utf-8").strip()
    col_by_normalized = {_normalize_feature_name(c): c for c in df_columns}
    missing_re = re.compile(
        r"If the value of the feature (.+?) is missing, this is neutral signal \(gender unclear\)\.",
        re.IGNORECASE,
    )
    cond_re = re.compile(
        r"If (.+?) is (less than or equal to|between|greater than) ([^,]+), this is (strong|medium|mild) signal for (Male|Female)\.",
        re.IGNORECASE,
    )
    rules_by_col = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = missing_re.match(line)
        if m:
            feat = _normalize_feature_name(m.group(1).strip())
            col = col_by_normalized.get(feat)
            if col is None:
                continue
            rules_by_col.setdefault(col, []).append(("missing", None, None, "neutral", None))
            continue
        m = cond_re.match(line)
        if not m:
            continue
        rest, cond_type, strength, signal = m.group(3).strip(), m.group(2).strip().lower(), m.group(4).strip().lower(), m.group(5).strip()
        feat = _normalize_feature_name(m.group(1).strip())
        col = col_by_normalized.get(feat)
        if col is None:
            continue
        if "less than or equal to" in cond_type:
            try:
                bound = float(rest)
                rules_by_col.setdefault(col, []).append(("le", bound, None, signal, strength))
            except ValueError:
                pass
        elif "between" in cond_type:
            part = re.match(r"([\d.-]+)\s+and\s+([\d.-]+)", rest)
            if part:
                low, high = float(part.group(1)), float(part.group(2))
                rules_by_col.setdefault(col, []).append(("between", low, high, signal, strength))
        elif "greater than" in cond_type:
            try:
                bound = float(rest)
                rules_by_col.setdefault(col, []).append(("gt", bound, None, signal, strength))
            except ValueError:
                pass
    return rules_by_col


def _format_val_gender(v: float, col: str) -> str:
    if "per day" in col.lower():
        return f"{v:.2f}"
    if "count" in col.lower():
        return str(int(round(v)))
    if abs(v) >= 1e6 or (abs(v) < 0.01 and v != 0):
        return f"{v:.2f}"
    return f"{v:,.0f}".replace(",", " ")


def _format_strength(strength: str) -> str:
    if strength is None:
        return ""
    return "STRONG" if strength.strip().lower() == "strong" else strength.strip()


def _apply_rules_gender(row: pd.Series, rules_by_col: dict) -> list:
    signals = []
    for col, rule_list in rules_by_col.items():
        val = row.get(col)
        readable = col.capitalize().replace("expences", "expenses")
        if pd.isna(val) or (isinstance(val, str) and val.strip() == ""):
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        v_str = _format_val_gender(v, col)
        for cond, low, high, signal, strength in rule_list:
            if cond == "missing":
                continue
            strength_str = _format_strength(strength)
            text_desc = f"{readable} is {v_str}, which acts as a {strength_str.lower()} indicator."
            is_match = (cond == "le" and v <= low) or (cond == "between" and low < v <= high) or (cond == "gt" and v > low)
            if is_match:
                signals.append({
                    "feature": readable, "value": v_str, "signal_type": signal.lower(),
                    "strength": strength_str, "text": text_desc,
                })
                break
    return signals


def _generate_conclusion_gender(male_signals: list, female_signals: list, verdict: str) -> str:
    def score(s):
        return {"STRONG": 3, "medium": 2, "mild": 1}.get(s["strength"], 0)
    sorted_m = sorted(male_signals, key=score, reverse=True)
    sorted_f = sorted(female_signals, key=score, reverse=True)
    conclusion = "Conclusion:\n"
    if verdict == "Male":
        top_m = [s["feature"] for s in sorted_m[:2]] if sorted_m else ["overall spending patterns"]
        top_f = sorted_f[0]["feature"] if sorted_f else None
        conclusion += f"While there are some female-leaning indicators (such as {top_f}), " if top_f else "Looking at the data, "
        conclusion += f"the male signals heavily dominate this profile. Specifically, traits like {', '.join(top_m)} strongly align with male spending behavior, outweighing other factors."
    else:
        top_f = [s["feature"] for s in sorted_f[:2]] if sorted_f else ["overall spending patterns"]
        top_m = sorted_m[0]["feature"] if sorted_m else None
        conclusion += f"Although there are certain male-leaning metrics (e.g., {top_m}), " if top_m else "Looking at the data, "
        conclusion += f"the profile is primarily defined by female indicators. Features like {', '.join(top_f)} are strong predictors that ultimately tip the scale to Female."
    return conclusion


def generate_gender_instruct(df: pd.DataFrame, ds_cfg: dict, output_file: str) -> None:
    data_dir = ds_cfg["data_dir"]
    rules_path = resolve_path(data_dir, ds_cfg["rules_file"])
    id_col = ds_cfg["id_col"]
    target_col = ds_cfg["target_col"]
    feature_order = ds_cfg.get("feature_order", [])

    rules_by_col = _parse_rules_gender(rules_path, list(df.columns))
    instruction = (
        "You are an expert banking analyst. Based on the customer description below, "
        "predict the customer's gender: answer Male or Female. "
        "First provide your analysis in a section starting with 'Reasoning:', "
        "grouping and weighing both male and female indicators. "
        "Then state the classification in a section starting with 'Verdict:'. "
        "The Verdict must be exactly 'Male' or 'Female'."
    )
    records = []
    for _, row in df.iterrows():
        cid = row.get(id_col)
        cid = "unknown" if pd.isna(cid) else (int(cid) if isinstance(cid, (int, float)) else cid)
        profile_lines = []
        for col in feature_order:
            if col not in df.columns:
                continue
            val = row.get(col, "N/A")
            if val == "N/A" or pd.isna(val):
                continue
            val = round(val, 4) if isinstance(val, (int, float)) else val
            profile_lines.append(f"{col}: {val}")
        user_input = "Customer Profile Data:\n" + "\n".join(profile_lines)
        g = row[target_col]
        verdict = "Male" if g in (1, "1", "Male", "MALE", "male", "M") else "Female"
        if g not in (0, "0", 1, "1", "Male", "Female", "M", "F", "male", "female"):
            verdict = "Male" if str(g).upper() in ("1", "M", "MALE") else "Female"
        signals = _apply_rules_gender(row, rules_by_col)
        male_signals = [s for s in signals if s["signal_type"] == "male"]
        female_signals = [s for s in signals if s["signal_type"] == "female"]
        for lst in (male_signals, female_signals):
            lst.sort(key=lambda x: {"STRONG": 3, "medium": 2, "mild": 1}.get(x["strength"], 0), reverse=True)
        reasoning_parts = ["To determine the customer's gender, we evaluate the spending patterns across various categories.\n"]
        if male_signals:
            reasoning_parts.append("Male Indicators:")
            for s in male_signals:
                reasoning_parts.append(f"- {s['text']}")
            reasoning_parts.append("")
        if female_signals:
            reasoning_parts.append("Female Indicators:")
            for s in female_signals:
                reasoning_parts.append(f"- {s['text']}")
            reasoning_parts.append("")
        conclusion_text = _generate_conclusion_gender(male_signals, female_signals, verdict)
        reasoning_parts.append(conclusion_text)
        output = "Reasoning:\n" + "\n".join(reasoning_parts) + f"\n\nVerdict: {verdict}"
        records.append({id_col: cid, "instruction": instruction, "input": user_input, "output": output})

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Gender instruct dataset saved to {output_file} with {len(records)} entries.")


# --------------- Main ---------------

def main():
    parser = argparse.ArgumentParser(description="Create instruct dataset (churn or gender) from config")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--train_out", type=str, default=None, help="Override train output path")
    parser.add_argument("--test_out", type=str, default=None, help="Override test output path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    task = cfg["task"]
    ds = cfg["dataset"]
    data_dir = cfg["data_dir"]

    stats_csv = resolve_path(data_dir, ds["stats_csv"])
    test_ids_csv = resolve_path(data_dir, ds["test_ids_csv"])
    id_col = ds["id_col"]

    df = pd.read_csv(stats_csv)
    test_ids = pd.read_csv(test_ids_csv)[id_col].tolist()
    df_train = df[~df[id_col].isin(test_ids)]
    df_test = df[df[id_col].isin(test_ids)]

    if args.train_out:
        train_out = args.train_out
        test_out = args.test_out or str(Path(train_out).parent / f"{task}_instruct_test.jsonl")
    else:
        train_out = str(Path(data_dir) / f"{task}_instruct_train.jsonl")
        test_out = str(Path(data_dir) / f"{task}_instruct_test.jsonl")
    if args.test_out:
        test_out = args.test_out

    # Optional: few_shot filter for gender
    few_shot_file = ds.get("few_shot_ids_file")
    if few_shot_file:
        few_path = resolve_path(data_dir, few_shot_file)
        if Path(few_path).exists():
            with open(few_path) as f:
                allowed = {line.strip() for line in f if line.strip()}
            df_train = df_train[df_train[id_col].astype(str).isin({str(x) for x in allowed})]

    if task == "churn":
        generate_churn_instruct(df_train, ds, train_out)
        generate_churn_instruct(df_test, ds, test_out)
    elif task == "gender":
        generate_gender_instruct(df_train, ds, train_out)
        generate_gender_instruct(df_test, ds, test_out)
    else:
        raise ValueError(f"Unknown task: {task}. Use churn or gender.")


if __name__ == "__main__":
    main()
