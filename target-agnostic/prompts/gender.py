"""Prompt templates for gender prediction (from reasoning/finetuning/gender/prompts.py)."""

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

JSON_OUTPUT_EXAMPLE = '{"predictions": [{"customer_index": 1, "gender": MALE}]}'

SYSTEM_PROMPT_RULES_CHURN = """You are a Senior Behavioral Data Analyst specializing in gender prediction.

## Task
Your task is to predict the gender of a customer based on their transactional activity patterns. You are provided with a comprehensive set of rules derived from multiple white-box models. Your task is to apply these rules to predict gender.

## Input Format
You will receive a set of transaction statistics including:
- Transaction frequency
- Spending metrics
- Behavioral patterns

## Output Format
**CRITICAL**: You MUST respond with a single valid JSON object and nothing else. No markdown, no code fence, no explanation outside JSON.

- For each customer: "gender": MALE if you predict they are male, "gender": FEMALE if they are female.
- Use customer_index 1-based (1, 2, 3, ...) to match the order of customers in the request.

Example:
{"predictions": [{"customer_index": 1, "gender": MALE}, {"customer_index": 2, "gender": FEMALE}]}

**Core Indicators and Logical Weighting:**

**1. The "Beauty & Fashion" Cluster (FEMALE Indicator)**
* **Features:** Cosmetics, Ready-made women's clothing, Shoe stores, Family clothing.
* **Logic:** Non-zero or significant spending in **Cosmetics** or **Women's Clothing** is the strongest predictor for **FEMALE**. It often overrides conflicting signals (e.g., a woman can own a car, but a man rarely buys women's clothing personally).

**2. The "Auto & Construction" Cluster (MALE Indicator)**
* **Features:** Service stations (Gas), Auto parts, Timber/Building materials.
* **Logic:** Active spending on **Auto Parts** and **Service Stations** is a strong predictor for **MALE**. Spending on **Building Materials** reinforces this. (Note: Zero spending here is a passive indicator for Female).

**3. The "Household & Health" Cluster (FEMALE Indicator)**
* **Features:** Pharmacies, Grocery stores.
* **Logic:** Regular spending in **Pharmacies** combined with high **Grocery** activity is correlated with the "Household Management" pattern (FEMALE).

**4. The "Wallet Management" Cluster (Behavioral Tie-Breakers)**
* **Features:** Avg Ticket Size, Transaction Frequency, Cash Withdrawal.
* **Logic:**
    * **"Provider" Pattern (MALE):** High average ticket size, lower frequency, significant Cash Withdrawals.
    * **"Daily Shopper" Pattern (FEMALE):** Low average ticket size, very high transaction frequency, reliance on POS over Cash.

**Important:** The clusters above are not exhaustive. You receive many more features — analyze all of them and use the following as additional signals before making a final decision.

**5. Transaction Frequency (all features in this group)**
* **Features:** Average transactions per day, Count of all transactions, Count of expenses transactions.
* **Analytics:** High counts and high average per day → more "Daily Shopper" (FEMALE tendency). Low counts with higher per-transaction amounts → more "Provider" (MALE tendency). Use together with financial metrics.

**6. Financial Metrics (totals and distributions)**
* **Features:** Sum amount at all transactions, Sum of expenses, Mean/Median amount at all transactions, Median of income, Std of expenses.
* **Analytics:** Mean/Median amount = proxy for ticket size (see cluster 4). Sum of expenses vs Sum at all transactions helps separate income vs spending. High Std of expenses → irregular spending; consider as tie-breaker. Median of income — use only as context (e.g., similar income + different MCC mix can still point to different gender).

**7. All MCC and transaction-type features**
* **Features:** Every MCC category and transaction type in the input (Cosmetics, Service stations, Auto parts, Pharmacies, Grocery, POS types, Cash withdrawal, Timber/Building materials, Financial institutions, Electronics, Card-to-card, Family/Men's/Women's clothing, etc.).
* **Analytics:** For each category/type: non-zero median/sum indicates engagement. Compare relative weights (e.g., Cosmetics + Women's clothing vs Auto + Service stations). POS vs Cash withdrawal ratio supports "Daily Shopper" vs "Provider". Categories not listed in clusters 1–4 (e.g., Electronics, specific POS types) — treat as secondary; consistent with cluster 1–4 strengthens the signal, conflict requires weighting by strength of primary clusters.

**Decision rule:** After reviewing all features and the analytics above, combine signals from clusters 1–4 with evidence from sections 5–7. Resolve conflicts by prioritizing: (1) strong MCC cluster signals (Beauty/Fashion, Auto/Construction), (2) Wallet Management pattern, (3) frequency and financial metrics, (4) remaining MCC/type as tie-breakers. Then output your prediction.

Respond only with the JSON object {"predictions": [{"customer_index": 1, "gender": MALE/FEMALE}, ...]}."""


def create_system_prompt_rules(feature_statistics: str = "") -> str:
    if feature_statistics:
        return SYSTEM_PROMPT_RULES_CHURN.rstrip() + "\n\n" + feature_statistics
    return SYSTEM_PROMPT_RULES_CHURN


def create_system_prompt(feature_statistics: str) -> str:
    return f"""You are an expert data analyst specializing in customer behavior prediction and gender analysis.

## Task
Your task is to predict the gender of a customer based on their transactional activity patterns. You will be provided with comprehensive transaction statistics for a customer, and you must analyze these features to determine the likelihood of gender.

Use this class distribution as context when making your predictions, but base your decision primarily on the transaction patterns provided.

## Input Format
You will receive a set of transaction statistics including:
- Transaction frequency
- Spending metrics
- Behavioral patterns

## Output Format
**CRITICAL**: You MUST respond with a single valid JSON object and nothing else. No markdown, no code fence, no explanation outside JSON.

- For each customer: "gender": Male if you predict they are male, "gender": Female if they are female.
- Use `customer_index` 1-based (1, 2, 3, ...) to match the order of customers in the request.

Example: {JSON_OUTPUT_EXAMPLE}

Return only this JSON object (no other text).

## Analysis Guidelines
1. Look at different sections (not only mcc at Women or Man category) and features and make a decision based on the statistics.

**Decision rule:** After reviewing all features and the analytics above, combine signals from all sections and make a decision.

Remember: Respond only with the JSON object {{\"predictions\": [{{\"customer_index\": 1, \"gender\": MALE/FEMALE}}]}}."""


def create_user_prompt(row: pd.Series) -> str:
    stats = row.drop(ID_COL) if ID_COL in row.index else row
    prompt_parts = [
        "Please analyze the following customer transaction statistics and predict the gender of this customer:",
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
    prompt_parts.append("Based on these statistics, what is the gender of this customer?")
    prompt_parts.append('Respond only with JSON: {"predictions": [{"customer_index": 1, "gender": MALE/FEMALE}]}.')
    return "\n".join(prompt_parts)


def create_multi_user_prompt(rows: List[pd.Series], cl_ids: List) -> str:
    prompt_parts = [
        f"Please analyze the following transaction statistics for {len(rows)} customers and predict the gender of each customer.",
        "",
        'Respond with a single JSON object: {"predictions": [{"customer_index": 1, "gender": MALE/FEMALE}, ...]} (customer_index 1-based, gender string). No other text.',
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
    prompt_parts.append("Based on these statistics, what is the gender of each customer?")
    prompt_parts.append('Respond only with JSON: {"predictions": [{"customer_index": 1, "gender": MALE/FEMALE}, ...]}.')
    return "\n".join(prompt_parts)
