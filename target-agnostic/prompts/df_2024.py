"""Prompt templates for DataFusion 2024 churn (from reasoning/finetuning/df_2024/prompts.py)."""

import pandas as pd
from typing import List

from configs.config import (
    TRANSACTION_FREQUENCY_FEATURES,
    TEMPORAL_PATTERN_FEATURES,
    FINANCIAL_METRIC_FEATURES,
    SPENDING_PATTERN_FEATURES,
    MCC_PATTERN_FEATURES,
    ID_COL,
)
from utils import format_value

JSON_OUTPUT_EXAMPLE = '{"predictions": [{"customer_index": 1, "churn": true}, {"customer_index": 2, "churn": false}]}'

SYSTEM_PROMPT_RULES_CHURN = """You are a Senior Behavioral Data Analyst specializing in banking retention.

## Task
Your task is to predict whether a customer will churn (stop using the bank's service) based on their transactional activity patterns. You are provided with a comprehensive set of rules derived from multiple white-box models. Your task is to apply these rules to predict Bank Churn.

## Input Format
You will receive a set of transaction statistics including:
- Transaction frequency
- Spending metrics 
- Behavioral patterns

## Output Format
**CRITICAL**: You MUST respond with a single valid JSON object and nothing else. No markdown, no code fence, no explanation outside JSON.

- For each customer: "churn": true if you predict they WILL churn, "churn": false if they will NOT churn.
- Use customer_index 1-based (1, 2, 3, ...) to match the order of customers in the request.

Example:
{"predictions": [{"customer_index": 1, "churn": true}, {"customer_index": 2, "churn": false}]}

### THE GOLDEN RULE:
"High Value equals Retention." 
Even if a customer has a short lifetime duration (< 175 days), they are NOT churned if they show high spending volume or deep integration.

### I. THE "IMMUNITY" FILTERS (Default to RETAINED)
Before checking for churn, check if the customer has "Immunity". If ANY of these are true, classify as **RETAINED**:
1. **High Spending Volume:** `Total_cumulative_spending_volume` is less than -65,000 (meaning they spent 65k+). 
   *Example: ID 354864 (Ex 11) has only 15 tx, but spent -90k. Result: RETAINED.*
2. **Deep Ecosystem Integration:** `Diversity_of_spending_categories` > 15.
   *Example: ID 220903 (Ex 14) has short duration but 18 categories. Result: RETAINED.*
3. **High Frequency Habit:** `Total_number_of_transactions` > 150.

---

### II. THE "CONFIRMED CHURN" PROFILES (Classify as CHURN)
If the customer has NO "Immunity", classify as **CHURN** only if they fit these strict profiles:

#### PROFILE 1: The "Small & Quiet" (Low Engagement)
- `Total_number_of_transactions` < 40 
- **AND** `Total_cumulative_spending_volume` > -30,000 (Spent very little)
- **AND** `Diversity_of_spending_categories` < 10.
*Example: IDs 499336, 531138 fit this perfectly.*

#### PROFILE 2: The "High-Frequency Ghost" (Low Value despite activity)
- `Total_number_of_transactions` is high (up to 130)
- **BUT** `Total_cumulative_spending_volume` is weak (>-20,000) OR `Average_value_per_transaction` is very low (closer to 0 than -300).
*Example: ID 444821 (90 tx but only -17k spent).*

#### PROFILE 3: The "Robotic/Temporary" User
- `Maximum_single_transaction_value` is suspiciously low (e.g., < 100) while `Average` is high.
- **OR** `Consistency_of_transaction_timing` > 8.0 (Extreme randomness).
*Example: ID 370257 (Ex 2) had a max spend of only 92 despite high volume. This is non-human behavior.*

---

### III. FINAL DECISION LOGIC
1. Check for **Immunity** first. 
2. If no Immunity, check for **Profiles 1, 2, or 3**.
3. If still unsure, or if the metrics are "average", default to churn: false.
Respond only with the JSON object {"predictions": [{"customer_index": 1, "churn": true/false}, ...]}."""


def create_system_prompt_rules(feature_statistics: str = "") -> str:
    if feature_statistics:
        return SYSTEM_PROMPT_RULES_CHURN.rstrip() + "\n\n" + feature_statistics
    return SYSTEM_PROMPT_RULES_CHURN


def create_system_prompt(feature_statistics: str) -> str:
    return f"""You are an expert data analyst specializing in customer behavior prediction and churn analysis.

## Task
Your task is to predict whether a customer will churn (stop using the service) based on their transactional activity patterns. You will be provided with comprehensive transaction statistics for a customer, and you must analyze these features to determine the likelihood of churn.

Use this class distribution as context when making your predictions, but base your decision primarily on the transaction patterns provided. Note that churn is a rare event: predict churn only when the evidence clearly supports it; most customers should be classified as non-churning.

## Input Format
You will receive a set of transaction statistics including:
- Transaction frequency and timing patterns
- Income and expense metrics
- Activity patterns (weekdays, weekends, monthly patterns)
- Transaction amount distributions
- Temporal activity indicators

## Output Format
**CRITICAL**: You MUST respond with a single valid JSON object and nothing else. No markdown, no code fence, no explanation outside JSON.

- For each customer: `"churn": true` if you predict they WILL churn, `"churn": false` if they will NOT churn.
- Use `customer_index` 1-based (1, 2, 3, ...) to match the order of customers in the request.

Example: {JSON_OUTPUT_EXAMPLE}

Return only this JSON object (no other text).

## Analysis Guidelines
1. Consider recent activity trends 
2. Evaluate transaction frequency patterns 
3. Assess spending behavior changes 
4. Examine temporal patterns 
5. Look for signs of engagement decline

Remember: Respond only with the JSON object {{\"predictions\": [{{\"customer_index\": 1, \"churn\": true/false}}, ...]}}."""


def create_user_prompt(row: pd.Series) -> str:
    stats = row.drop(ID_COL) if ID_COL in row.index else row
    prompt_parts = [
        "Please analyze the following customer transaction statistics and predict whether this customer will churn:",
        "",
        "## Transaction Statistics",
        ""
    ]
    sections = [
        ("Transaction Frequency", TRANSACTION_FREQUENCY_FEATURES),
        ("Temporal Patterns", TEMPORAL_PATTERN_FEATURES),
        ("Financial Metrics", FINANCIAL_METRIC_FEATURES),
        ("Spending Patterns", SPENDING_PATTERN_FEATURES),
        ("MCC (Merchant Category) Patterns", MCC_PATTERN_FEATURES)
    ]
    for section_name, feature_list in sections:
        prompt_parts.append(f"### {section_name}")
        for feature in feature_list:
            if feature in stats.index:
                value = format_value(stats[feature])
                prompt_parts.append(f"- **{feature}**: {value}")
        prompt_parts.append("")
    prompt_parts.append("Based on these statistics, will this customer churn?")
    prompt_parts.append('Respond only with JSON: {"predictions": [{"customer_index": 1, "churn": true/false}]}.')
    return "\n".join(prompt_parts)


def create_multi_user_prompt(rows: List[pd.Series], cl_ids: List) -> str:
    prompt_parts = [
        f"Please analyze the following transaction statistics for {len(rows)} customers and predict whether each customer will churn.",
        "",
        'Respond with a single JSON object: {"predictions": [{"customer_index": 1, "churn": true/false}, ...]} (customer_index 1-based, churn boolean). No other text.',
        "",
        "## Customer Transaction Statistics",
        ""
    ]
    for customer_idx, (row, cl_id) in enumerate(zip(rows, cl_ids), 1):
        prompt_parts.append(f"### Customer {customer_idx} (ID: {cl_id})")
        stats = row.drop(ID_COL) if ID_COL in row.index else row
        sections = [
            ("Transaction Frequency", TRANSACTION_FREQUENCY_FEATURES),
            ("Temporal Patterns", TEMPORAL_PATTERN_FEATURES),
            ("Financial Metrics", FINANCIAL_METRIC_FEATURES),
            ("Spending Patterns", SPENDING_PATTERN_FEATURES),
            ("MCC (Merchant Category) Patterns", MCC_PATTERN_FEATURES)
        ]
        for section_name, feature_list in sections:
            prompt_parts.append(f"#### {section_name}")
            for feature in feature_list:
                if feature in stats.index:
                    value = format_value(stats[feature])
                    prompt_parts.append(f"- **{feature}**: {value}")
            prompt_parts.append("")
        prompt_parts.append("")
    prompt_parts.append("Based on these statistics, will each customer churn?")
    prompt_parts.append('Respond only with JSON: {"predictions": [{"customer_index": 1, "churn": true/false}, ...]}.')
    return "\n".join(prompt_parts)
