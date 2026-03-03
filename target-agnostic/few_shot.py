"""Main script for few-shot churn prediction using vLLM with 10 examples in context."""

import pandas as pd
import numpy as np
from vllm import LLM, SamplingParams
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef

from configs.config import (
    MODEL_CONFIG,
    SAMPLING_CONFIG,
    STATS_CSV,
    TEST_IDS_CSV,
    DATA_DIR,
    FEATURE_MAPPING,
    DATASET_NAME,
    TRANSACTION_FREQUENCY_FEATURES,
    TEMPORAL_PATTERN_FEATURES,
    FINANCIAL_METRIC_FEATURES,
    SPENDING_PATTERN_FEATURES,
    MCC_PATTERN_FEATURES,
    NUM_SHOTS,
    ID_COL,
    TARGET_COL,
)
from prompts import (
    create_system_prompt_rules,
    create_user_prompt as base_create_user_prompt,
)
from utils import (
    get_feature_statistics,
    extract_predictions_from_multi_response,
    extract_predictions_from_multi_response_churn,
    format_value,
)
import os


def load_data():
    """Load and prepare data."""
    print("Loading data...")
    df = pd.read_csv(STATS_CSV)
    targets = df[[ID_COL, TARGET_COL]]
    df = df.drop(columns=[TARGET_COL])
    if FEATURE_MAPPING:
        df = df.rename(columns=FEATURE_MAPPING)
    
    test_ids = pd.read_csv(TEST_IDS_CSV)[ID_COL].tolist()
    train_df = df[~df[ID_COL].isin(test_ids)]
    test_df = df[df[ID_COL].isin(test_ids)]
    
    # Выделяем target-метки для train выборки
    train_targets = targets[~targets[ID_COL].isin(test_ids)]
    num_zeros = (train_targets[TARGET_COL] == 0).sum()
    num_ones = (train_targets[TARGET_COL] == 1).sum()
    
    print(f"Train targets: 0s = {num_zeros}, 1s = {num_ones}")
    print(f"Test set size: {len(test_df)}")
    
    return train_df, test_df, targets, test_ids, num_zeros, num_ones, train_targets


def select_few_shot_examples(train_df, train_targets, n_examples=10):
    """
    Select balanced few-shot examples. Labels depend on DATASET_NAME:
    - gender: 1 → Male, 0 → Female
    - rosbank: 0 → churn: true, 1 → churn: false
    - df_2024: 1 → churn: true, 0 → churn: false
    """
    train_with_targets = train_df.merge(train_targets, on=ID_COL, how='inner')
    n_per_class = n_examples // 2

    if DATASET_NAME == "gender":
        class_a = train_with_targets[train_with_targets[TARGET_COL] == 1]
        class_b = train_with_targets[train_with_targets[TARGET_COL] == 0]
        label_a, label_b = "gender: Male", "gender: Female"
        def extra_label(row):
            return "gender: Male" if row[TARGET_COL] == 1 else "gender: Female"
    elif DATASET_NAME == "rosbank":
        class_a = train_with_targets[train_with_targets[TARGET_COL] == 0]
        class_b = train_with_targets[train_with_targets[TARGET_COL] == 1]
        label_a, label_b = "churn: true", "churn: false"
        def extra_label(row):
            return "churn: true" if row[TARGET_COL] == 0 else "churn: false"
    else:
        class_a = train_with_targets[train_with_targets[TARGET_COL] == 1]
        class_b = train_with_targets[train_with_targets[TARGET_COL] == 0]
        label_a, label_b = "churn: true", "churn: false"
        def extra_label(row):
            return "churn: true" if row[TARGET_COL] == 1 else "churn: false"

    samples_a = class_a.sample(n=min(n_per_class, len(class_a)), random_state=42)
    samples_b = class_b.sample(n=min(n_per_class, len(class_b)), random_state=42)
    examples = []
    for _, row in samples_a.iterrows():
        examples.append((row, row[ID_COL], label_a))
    for _, row in samples_b.iterrows():
        examples.append((row, row[ID_COL], label_b))

    shots_ids = [cl_id for _, cl_id, _ in examples]
    shots_path = os.path.join(DATA_DIR, "few_shot_ids.txt")
    with open(shots_path, "w") as f:
        for cl_id in shots_ids:
            f.write(f"{cl_id}\n")
    print(f"Few-shot example IDs saved to {shots_path}")
    
    # If we need more examples, add from the class with more samples
    remaining = n_examples - len(examples)
    if remaining > 0:
        if len(class_a) > len(class_b):
            additional = class_a.sample(n=min(remaining, len(class_a) - n_per_class), random_state=43)
        else:
            additional = class_b.sample(n=min(remaining, len(class_b) - n_per_class), random_state=43)
        for _, row in additional.iterrows():
            examples.append((row, row[ID_COL], extra_label(row)))
    print(f"Selected {len(examples)} few-shot examples (dataset={DATASET_NAME})")
    
    return examples


def format_example(row, cl_id, target_flag, example_idx):
    """
    Format a single example for the few-shot prompt.
    
    Args:
        row: pandas Series with customer features
        cl_id: Customer ID
        target_flag: Label in LLM output format ('gender: Male' or 'gender: Female')
        example_idx: Example number (1-indexed)
        
    Returns:
        Formatted example string
    """
    # Exclude cl_id and target_flag from stats
    stats = row.drop([ID_COL, TARGET_COL]) if TARGET_COL in row.index else row.drop(ID_COL)
    
    parts = [
        f"### Example {example_idx} (ID: {cl_id})",
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
        parts.append(f"#### {section_name}")
        for feature in feature_list:
            if feature in stats.index:
                value = format_value(stats[feature])
                parts.append(f"- **{feature}**: {value}")
        parts.append("")
    
    parts.append(f"**Prediction**: {target_flag}")
    parts.append("")
    
    return "\n".join(parts)


def create_few_shot_examples_section(examples):
    """Create the few-shot examples section. Intro text depends on DATASET_NAME."""
    if DATASET_NAME == "gender":
        intro = "Use these examples to understand the patterns that indicate gender:"
    elif DATASET_NAME == "rosbank":
        intro = "Use these examples to understand the patterns that indicate churn vs. no churn:"
    else:
        intro = "Use these examples to understand the patterns that indicate churn:"
    parts = [
        "## Few-Shot Examples",
        "",
        f"Below are examples from the training data with their correct labels. {intro}",
        ""
    ]
    
    for idx, (row, cl_id, target_flag) in enumerate(examples, 1):
        example_text = format_example(row, cl_id, target_flag, idx)
        parts.append(example_text)
    
    parts.append("---")
    parts.append("")
    
    return "\n".join(parts)


def create_few_shot_system_prompt(num_zeros: int, num_ones: int, feature_statistics: str, examples_section: str) -> str:
    """
    Create system prompt for few-shot churn prediction task.

    Args:
        num_zeros: Number of customers who did not churn in training data
        num_ones: Number of customers who churned in training data
        feature_statistics: Formatted string with feature statistics
        examples_section: Formatted string with few-shot examples

    Returns:
        System prompt string
    """
    base_prompt = create_system_prompt_rules(feature_statistics)
    print(base_prompt + "\n\n" + examples_section)
    return base_prompt

def initialize_model():
    """Initialize vLLM model and sampling parameters."""
    print("\nInitializing vLLM model...")
    llm = LLM(**MODEL_CONFIG)
    
    # Get eos_token_id for stop_token_ids
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        **SAMPLING_CONFIG,
        stop_token_ids=[tokenizer.eos_token_id]
    )
    
    print("Model initialized successfully.")
    return llm, sampling_params


def run_inference(llm, sampling_params, tokenizer, system_prompt, df, targets_dict, description="inference"):
    """
    One customer per prompt. Parser chosen by DATASET_NAME (gender vs churn).
    """
    extract_fn = extract_predictions_from_multi_response_churn if DATASET_NAME in ("rosbank", "df_2024") else extract_predictions_from_multi_response
    formatted_prompts = []
    prompt_meta_data = []
    for _, row in df.iterrows():
        cl_id = row[ID_COL]
        true_label = targets_dict[cl_id]
        prompt_meta_data.append([(cl_id, true_label)])
        user_prompt = base_create_user_prompt(row)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        formatted_prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    print(f"Running vLLM generation for {len(formatted_prompts)} prompts ({description})...")
    outputs = llm.generate(formatted_prompts, sampling_params)

    rows_for_csv = []
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        cl_id, true_label = prompt_meta_data[i][0]
        batch_preds = extract_fn(generated_text, 1)
        pred = batch_preds[0] if batch_preds else -1
        pred_target = pred if pred != -1 else 0
        rows_for_csv.append({
            ID_COL: cl_id,
            "model_reasoning": generated_text,
            "pred_target": pred_target,
            "true_target": int(true_label),
        })
    return rows_for_csv


def main():
    """Main inference pipeline (OPTIMIZED)."""
    # Load data
    train_df, test_df, targets, test_ids, num_zeros, num_ones, train_targets = load_data()

    # Select few-shot examples
    print("\nSelecting few-shot examples...")
    few_shot_examples = select_few_shot_examples(train_df, train_targets, n_examples=NUM_SHOTS)
    examples_section = create_few_shot_examples_section(few_shot_examples)

    # Compute feature statistics
    print("\nComputing feature statistics...")
    feature_statistics = get_feature_statistics(train_df)

    # Initialize model
    llm, sampling_params = initialize_model()
    tokenizer = llm.get_tokenizer()

    # Create system prompt
    system_prompt = create_few_shot_system_prompt(num_zeros, num_ones, feature_statistics, examples_section)

    test_targets_dict = dict(zip(targets[targets[ID_COL].isin(test_ids)][ID_COL], targets[targets[ID_COL].isin(test_ids)][TARGET_COL]))
    train_targets_dict = dict(zip(train_targets[ID_COL], train_targets[TARGET_COL]))

    print("\n" + "="*80)
    print(f"Dataset: {DATASET_NAME}")
    print("="*80 + "\n")

    # --- Test set: инференс и метрики ---
    print(f"Preparing prompts for {len(test_df)} test customers")
    test_rows = run_inference(llm, sampling_params, tokenizer, system_prompt, test_df, test_targets_dict, "test")
    predictions = np.array([r["pred_target"] for r in test_rows])
    true_labels = np.array([r["true_target"] for r in test_rows])

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    f1_weighted = f1_score(true_labels, predictions, average='weighted')
    f1_macro = f1_score(true_labels, predictions, average='macro')
    mcc = matthews_corrcoef(true_labels, predictions)

    print("\n" + "="*80)
    print(f"RESULTS TEST (Few-Shot with {NUM_SHOTS} examples, dataset={DATASET_NAME})")
    print("="*80)
    print(f"Total customers: {len(test_df)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"MCC: {mcc:.4f}")
    print("\n" + "-" * 80)
    print(f"Predicted positive (1): {np.sum(predictions == 1)}")
    print(f"Actual positive (1): {np.sum(true_labels == 1)}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
