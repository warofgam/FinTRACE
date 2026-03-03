"""Utility functions for gender prediction."""

import json
import re
import pandas as pd
import numpy as np
from typing import List, Optional

from configs.config import (
    TRANSACTION_FREQUENCY_FEATURES,
    TEMPORAL_PATTERN_FEATURES,
    FINANCIAL_METRIC_FEATURES,
    SPENDING_PATTERN_FEATURES,
    MCC_PATTERN_FEATURES,
)


def format_value(val):
    """Format a value for display in prompts."""
    if pd.isna(val):
        return "N/A"
    if isinstance(val, (int, float)):
        if isinstance(val, float) and val.is_integer():
            return str(int(val))
        return f"{val:.4f}" if abs(val) < 1e-4 or abs(val) >= 1e6 else f"{val:.2f}"
    return str(val)


def format_stat_value(val):
    """Format statistical value for display."""
    if pd.isna(val):
        return "N/A"
    if isinstance(val, (int, float)):
        if isinstance(val, float) and val.is_integer():
            return str(int(val))
        # Use scientific notation for very large or very small numbers
        if abs(val) < 1e-3 or abs(val) >= 1e6:
            return f"{val:.4e}"
        return f"{val:.2f}"
    return str(val)


def get_feature_statistics(df: pd.DataFrame) -> str:
    """
    Generate statistics (quantiles, mean, std) for each feature in the dataset.
    
    Args:
        df: DataFrame with features (using human-readable names)
        
    Returns:
        Formatted string with feature statistics
    """
    stats_parts = []
    stats_parts.append("## Feature Statistics (from training data)")
    stats_parts.append("")
    stats_parts.append("The following statistics provide context for understanding the distribution of each feature:")
    stats_parts.append("")
    
    def get_feature_stats(feature_name: str) -> str:
        """Get statistics for a single feature."""
        if feature_name not in df.columns:
            return None
        
        col = df[feature_name]
        if not pd.api.types.is_numeric_dtype(col):
            return None
        
        # Удаляем NaN значения для статистики
        col_clean = col.dropna()
        if len(col_clean) == 0:
            return None
        
        quantiles = col_clean.quantile([0.0, 0.25, 0.5, 0.75, 1.0])
        mean_val = col_clean.mean()
        std_val = col_clean.std()
        
        stats_text = (
            f"  - Min: {format_stat_value(quantiles[0.0])}, "
            f"Q25: {format_stat_value(quantiles[0.25])}, "
            f"Median: {format_stat_value(quantiles[0.5])}, "
            f"Q75: {format_stat_value(quantiles[0.75])}, "
            f"Max: {format_stat_value(quantiles[1.0])}, "
            f"Mean: {format_stat_value(mean_val)}, "
            f"Std: {format_stat_value(std_val)}"
        )
        return stats_text
    
    sections = [
        ("Transaction Frequency Features", TRANSACTION_FREQUENCY_FEATURES),
        ("Temporal Pattern Features", TEMPORAL_PATTERN_FEATURES),
        ("Financial Metric Features", FINANCIAL_METRIC_FEATURES),
        ("Spending Pattern Features", SPENDING_PATTERN_FEATURES),
        ("MCC (Merchant Category) Pattern Features", MCC_PATTERN_FEATURES)
    ]
    
    for section_name, feature_list in sections:
        section_lines = []
        for feature in feature_list:
            if feature in df.columns:
                feature_stats = get_feature_stats(feature)
                if feature_stats:
                    section_lines.append(f"- **{feature}**:")
                    section_lines.append(feature_stats)
        if section_lines:
            stats_parts.append(f"### {section_name}")
            stats_parts.extend(section_lines)
            stats_parts.append("")
    
    stats_parts.append("Use these statistics to understand whether a customer's values are typical, unusually high, or unusually low compared to the training distribution.")
    
    return "\n".join(stats_parts)


def extract_prediction(text: str) -> int:
    """
    Extract prediction from model response for a single customer.
    Returns 1 for Male, 0 for Female, -1 if not found.
    """
    text = text.strip().upper()
    if text.startswith("MALE"):
        return 1
    elif text.startswith("FEMALE"):
        return 0
    else:
        words = text.split()[:5]
        for word in words:
            if word == "MALE":
                return 1
            elif word == "FEMALE":
                return 0
        return -1


def _extract_json_from_text(text: str) -> Optional[str]:
    """
    Extract JSON object from text that may contain reasoning before/after.
    Model often outputs reasoning first, then JSON. Looks for:
    1) ```json ... ``` or ``` ... ``` block (last one if multiple)
    2) {"predictions": ...} — find by brace matching from first occurrence
    """
    raw = text.strip()
    if not raw:
        return None

    # 1) Code fence: take last ``` block (most likely the JSON after reasoning)
    code_fence = "```"
    start_idx = raw.rfind(code_fence)
    if start_idx != -1:
        after_start = raw[start_idx + len(code_fence):]
        # Skip optional "json" after opening ```
        if after_start.lstrip().lower().startswith("json"):
            after_start = after_start[4:].lstrip()
        end_idx = after_start.find(code_fence)
        if end_idx != -1:
            snippet = after_start[:end_idx].strip()
            if "predictions" in snippet and "{" in snippet:
                return snippet

    # 2) Find {"predictions" and extract object by matching braces
    idx = -1
    for key in ('{"predictions"', '{"predictions":', '{ "predictions"', '{ "predictions":'):
        idx = raw.find(key)
        if idx != -1:
            break
    if idx == -1:
        # Last resort: find "predictions" and take the { before it
        p = raw.find('"predictions"')
        if p != -1:
            idx = raw.rfind("{", 0, p)
        else:
            idx = -1
    if idx != -1:
        depth = 0
        in_string = False
        escape = False
        quote_char = None
        i = idx
        while i < len(raw):
            c = raw[i]
            if in_string:
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == quote_char:
                    in_string = False
            elif c in ('"', "'"):
                in_string = True
                quote_char = c
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return raw[idx : i + 1]
            i += 1
    return None


def _normalize_gender_json(raw: str) -> str:
    """
    Fix invalid JSON where gender value is unquoted: "gender": MALE -> "gender": "MALE".
    Handles both quoted and unquoted values from model output.
    """
    # Only add quotes when value is not already quoted (no " after colon)
    return re.sub(
        r'"gender"\s*:\s*(?!")(MALE|FEMALE)\b',
        lambda m: f'"gender": "{m.group(1)}"',
        raw,
        flags=re.IGNORECASE,
    )

def extract_predictions_from_json(text: str, num_customers: int) -> List[int]:
    """
    Extract predictions from model response when it is JSON.
    Handles reasoning before JSON: finds JSON block (code fence or {"predictions": ...}) and parses it.
    Accepts both "gender": "MALE" and "gender": MALE (normalized before parsing).
    Returns list of predictions (1 for Male, 0 for Female, -1 if not found).
    """
    raw = _extract_json_from_text(text)
    if raw is None:
        raw = text.strip()
        for prefix in ("```json", "```"):
            if raw.startswith(prefix):
                raw = raw[len(prefix):].strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()
    raw = _normalize_gender_json(raw)
    try:
        data = json.loads(raw)
        preds = data.get("predictions")
        if not isinstance(preds, list):
            return [-1] * num_customers
        by_idx = {}
        for p in preds:
            if isinstance(p, dict):
                idx = p.get("customer_index")
                gender = p.get("gender")
                if idx is not None and gender is not None:
                    g = str(gender).strip().lower()
                    if g == "male":
                        by_idx[int(idx)] = 1
                    elif g == "female":
                        by_idx[int(idx)] = 0
                    else:
                        by_idx[int(idx)] = -1
        return [by_idx.get(i, -1) for i in range(1, num_customers + 1)]
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


def extract_predictions_from_multi_response(text: str, num_customers: int) -> List[int]:
    """
    Extract predictions from model response for multiple customers.
    Tries JSON first, then falls back to Male/Female patterns.
    Returns list of predictions (1 for Male, 0 for Female, -1 if not found).
    """
    json_preds = extract_predictions_from_json(text, num_customers)
    if json_preds and all(p != -1 for p in json_preds):
        return json_preds
    if len(json_preds) == num_customers and any(p != -1 for p in json_preds):
        return json_preds

    # Fallback: Male/Female patterns
    predictions = []
    text_upper = text.upper()
    for i in range(1, num_customers + 1):
        patterns = [
            rf"CUSTOMER\s+{i}\s*:\s*(MALE|FEMALE)",
            rf"CUSTOMER\s+{i}\s+VERDICT\s*:\s*(MALE|FEMALE)",
            rf"CUSTOMER\s+{i}\s+(MALE|FEMALE)",
            rf"{i}\s*:\s*(MALE|FEMALE)",
            rf"CUSTOMER\s+{i}\s*-\s*(MALE|FEMALE)",
        ]
        found = False
        for pattern in patterns:
            match = re.search(pattern, text_upper)
            if match:
                pred = match.group(1)
                predictions.append(1 if pred == "MALE" else 0)
                found = True
                break
        if not found:
            customer_section = re.search(rf"CUSTOMER\s+{i}[^\d]*", text_upper, re.IGNORECASE)
            if customer_section:
                section_text = customer_section.group(0)
                if "MALE" in section_text[:50]:
                    predictions.append(1)
                elif "FEMALE" in section_text[:50]:
                    predictions.append(0)
                else:
                    predictions.append(-1)
            else:
                predictions.append(-1)
    return predictions


def extract_predictions_from_json_churn(text: str, num_customers: int) -> List[int]:
    """
    Extract churn predictions from model response when it is JSON.
    Expected format: {"predictions": [{"customer_index": 1, "churn": true/false}, ...]}
    Returns list of predictions (1 for churn, 0 for no churn, -1 if not found).
    Used for rosbank and df_2024 datasets.
    """
    raw = _extract_json_from_text(text)
    if raw is None:
        raw = text.strip()
        for prefix in ("```json", "```"):
            if raw.startswith(prefix):
                raw = raw[len(prefix):].strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()
    try:
        data = json.loads(raw)
        preds = data.get("predictions")
        if not isinstance(preds, list):
            return [-1] * num_customers
        by_idx = {}
        for p in preds:
            if isinstance(p, dict):
                idx = p.get("customer_index")
                churn = p.get("churn")
                if idx is not None and churn is not None:
                    by_idx[int(idx)] = 1 if churn else 0
        return [by_idx.get(i, -1) for i in range(1, num_customers + 1)]
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


def extract_predictions_from_multi_response_churn(text: str, num_customers: int) -> List[int]:
    """
    Extract churn predictions from model response (rosbank / df_2024).
    Tries JSON first (churn: true/false), then falls back to YES/NO / churn: true|false patterns.
    Returns list of predictions (1 for churn, 0 for no churn, -1 if not found).
    """
    json_preds = extract_predictions_from_json_churn(text, num_customers)
    if json_preds and all(p != -1 for p in json_preds):
        return json_preds
    if len(json_preds) == num_customers and any(p != -1 for p in json_preds):
        return json_preds

    text_upper = text.upper().strip()
    # Single customer: look for Verdict: YES/NO or churn: true/false in text
    if num_customers == 1:
        for pattern, val in [
            (r"VERDICT\s*:\s*YES", 1),
            (r"VERDICT\s*:\s*NO", 0),
            (r"churn\s*:\s*true", 1),
            (r"churn\s*:\s*false", 0),
        ]:
            if re.search(pattern, text_upper):
                return [val]
        if text_upper.startswith("YES"):
            return [1]
        if text_upper.startswith("NO"):
            return [0]
        for w in text_upper.split()[:8]:
            if w == "YES":
                return [1]
            if w == "NO":
                return [0]
        return [-1]

    predictions = []
    for i in range(1, num_customers + 1):
        patterns = [
            rf"CUSTOMER\s+{i}\s*:\s*(YES|NO)",
            rf"CUSTOMER\s+{i}\s+VERDICT\s*:\s*(YES|NO)",
            rf"CUSTOMER\s+{i}\s+(YES|NO)",
            rf"{i}\s*:\s*(YES|NO)",
        ]
        found = False
        for pattern in patterns:
            match = re.search(pattern, text_upper)
            if match:
                pred = match.group(1)
                predictions.append(1 if pred == "YES" else 0)
                found = True
                break
        if not found:
            customer_section = re.search(rf"CUSTOMER\s+{i}[^\d]*", text_upper, re.IGNORECASE)
            if customer_section:
                section_text = customer_section.group(0)
                if "YES" in section_text[:50]:
                    predictions.append(1)
                elif "NO" in section_text[:50]:
                    predictions.append(0)
                else:
                    predictions.append(-1)
            else:
                predictions.append(-1)
    return predictions
