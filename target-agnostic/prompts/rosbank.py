"""Prompt templates for Rosbank churn (from reasoning/finetuning/prompts.py)."""

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
- Transaction frequency (transaction_days_share, transaction_period)
- Spending metrics (weekday_spent, total_sum, top1_mcc_total, top2_mcc_total)
- Behavioral patterns (mcc_diversity, activity_drop_ratio, tx_on_first_day_share)

## Output Format
**CRITICAL**: You MUST respond with a single valid JSON object and nothing else. No markdown, no code fence, no explanation outside JSON.

- For each customer: "churn": true if you predict they WILL churn, "churn": false if they will NOT churn.
- Use customer_index 1-based (1, 2, 3, ...) to match the order of customers in the request.

Example:
{"predictions": [{"customer_index": 1, "churn": true}, {"customer_index": 2, "churn": false}]}

## Reference Rules: Combined Activity, Structural Stability & Financial Restoration

Use these rules to evaluate the user. High-Activity Accelerators decrease churn risk; Low-Activity Inhibitors increase it.

### I. The Temporal Foundation (Primary Impact)
* **Safe Zone (Low Churn):** Accounts with a mature history of **89–97 days**.
* **Churn Signals:** * **High Risk:** Accounts younger than **60 days**.
    * **Critical Danger:** Accounts younger than **30.5 days**.
    * **Instant Churn Risk:** Accounts younger than **7.5 days** (user failed to engage after opening).
* **Engagement Density (transaction_days_share):**
    * **Low Churn:** Density of **70%–99%** (Optimal) or at least **50%**.
    * **High Churn:** Density below **42%**.

### II. Economic Footprint (Spending Core)
* **Weekday Intensity (weekday_spent):**
    * **Optimal Loyalty:** Spending above **303,000**.
    * **Strong Engagement:** Spending between **97,000 and 303,000**.
    * **High Churn Risk:** Spending below **38,000**.
    * **Critical Risk:** Spending below **9,300**.
* **Primary Spend Pillars (top1_mcc_total):**
    * **Low Churn:** Main category spend exceeding **16,800**.
    * **High Churn:** Main category spend below **3,000**.

### III. Financial Restoration (Total Sum Recovery Logic)
This logic tracks if a user has "restored" their financial presence in the bank.
* **Restored Profile (Low Churn):** * Weekday spending > **303,745**. 
    * Transaction period > **92.5 days**.
    * *Rationale:* This combination confirms the user has fully integrated the bank into their primary financial lifecycle.
* **Unrecovered Profile (High Churn):** * Weekday spending < **86,078**.
    * Transaction period < **86.5 days**.
    * *Rationale:* If spending hasn't reached this baseline within nearly 3 months, the user is likely using the bank as a secondary/temporary tool.

### IV. Structural Breadth (MCC Diversity & Stickiness)
* **Healthy Variety (Low Churn):** Diversity between **20%–26%**. High variety across many life categories (groceries, bills, etc.) makes the user "sticky."
* **Risk Profiles (High Churn):**
    * **"Niche" users:** Diversity below **7%**.
    * **"Scatter" users:** Diversity above **33%** (erratic behavior).

### V. Lifecycle Dynamics (Retention Signals)
* **Consistency (activity_drop_ratio):**
    * **Low Churn:** Stable ratio between **1.9 and 2.9**.
    * **High Churn:** Ratio of **0** or below **0.4** (sudden activity collapse).
* **Onboarding (tx_on_first_day_share):**
    * **Low Churn:** Gradual adoption (only **1%–7%** on Day 1).
    * **High Churn:** More than **10%** "dumped" on Day 1 (one-time usage).

---

## Final Decision Logic

- **Predict "churn": false** if:
    - Account is >90 days old AND density is >50%.
    - Weekday spending is high (>303k) OR primary spend is solid (>16k).
    - MCC diversity is balanced (~20-26%).
- **Predict "churn": true** if:
    - Account is <60 days old (especially <7.5 days).
    - Weekday spending is low (<86k) AND density is low (<42%).
    - MCC diversity is niche (<7%) OR onboarding was too aggressive (>10% on Day 1).

**Weighting Note:** A user who has "Restored" (Weekday spent >303k and Tenure >92 days) is almost impossible to churn. Prioritize the combination of Tenure and Weekday Intensity.
Respond only with the JSON object {"predictions": [{"customer_index": 1, "churn": true/false}, ...]}."""


def create_system_prompt_rules(feature_statistics: str = "") -> str:
    if feature_statistics:
        return SYSTEM_PROMPT_RULES_CHURN.rstrip() + "\n\n" + feature_statistics
    return SYSTEM_PROMPT_RULES_CHURN


def create_system_prompt(feature_statistics: str) -> str:
    return f"""You are an expert data analyst specializing in customer behavior prediction and churn analysis.

## Task
Your task is to predict whether a customer will churn (stop using the service) based on their transactional activity patterns. You will be provided with comprehensive transaction statistics for a customer, and you must analyze these features to determine the likelihood of churn.

Use this class distribution as context when making your predictions, but base your decision primarily on the transaction patterns provided.

## Input Format
You will receive a set of transaction statistics including:
- Transaction frequency and timing patterns
- Income and expense metrics
- Activity patterns (weekdays, weekends, monthly patterns)
- MCC (Merchant Category Code) diversity and spending patterns
- Transaction amount distributions
- Temporal activity indicators

## Output Format
**CRITICAL**: You MUST respond with a single valid JSON object and nothing else. No markdown, no code fence, no explanation outside JSON.

- For each customer: `"churn": true` if you predict they WILL churn, `"churn": false` if they will NOT churn.
- Use `customer_index` 1-based (1, 2, 3, ...) to match the order of customers in the request.

Example: {JSON_OUTPUT_EXAMPLE}

Return only this JSON object (no other text).

## Analysis Guidelines
1. Consider recent activity trends (e.g., days_since_last_tx, activity_drop_ratio)
2. Evaluate transaction frequency patterns (transaction_count, avg_transaction_per_day)
3. Assess spending behavior changes (total_expenses, avg_spent, large_tx_ratio)
4. Examine temporal patterns (weekend activity, first-day-of-month patterns)
5. Look for signs of engagement decline (unique_mcc_cnt, mcc_diversity)

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
