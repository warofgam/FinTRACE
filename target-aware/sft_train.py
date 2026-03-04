"""
Универсальное SFT-обучение на instruct-датасете (churn / gender) по конфигу.
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

from config_loader import load_config, get_train_eval_paths, get_output_dir


def convert_to_llama_messages(row):
    return {
        "messages": [
            {"role": "system", "content": row["instruction"]},
            {"role": "user", "content": row["input"]},
            {"role": "assistant", "content": row["output"]},
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="SFT training on instruct dataset from config")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--train_data", type=str, default=None, help="Override train jsonl path")
    parser.add_argument("--eval_data", type=str, default=None, help="Override eval jsonl path")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output dir")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=None, help="Override epochs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_path, eval_path = get_train_eval_paths(cfg)
    if args.train_data:
        train_path = args.train_data
    if args.eval_data:
        eval_path = args.eval_data

    df_train = pd.read_json(train_path, lines=True)
    df_test = pd.read_json(eval_path, lines=True)
    dataset_train = Dataset.from_pandas(df_train)
    dataset_train = dataset_train.map(convert_to_llama_messages, remove_columns=dataset_train.column_names)
    dataset_test = Dataset.from_pandas(df_test)
    dataset_test = dataset_test.map(convert_to_llama_messages, remove_columns=dataset_test.column_names)

    model_id = cfg["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    sft_cfg = cfg["sft"]
    lora_target = sft_cfg.get("lora_target_modules") or [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ]
    peft_config = LoraConfig(
        r=sft_cfg.get("lora_r", 128),
        lora_alpha=sft_cfg.get("lora_alpha", 256),
        target_modules=lora_target,
        lora_dropout=sft_cfg.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )

    output_dir = args.output_dir or get_output_dir(cfg)
    training_args = SFTConfig(
        output_dir=output_dir,
        max_length=sft_cfg.get("max_length", 4096),
        dataset_text_field="messages",
        per_device_train_batch_size=sft_cfg.get("per_device_train_batch_size", 8),
        gradient_accumulation_steps=sft_cfg.get("gradient_accumulation_steps", 2),
        learning_rate=args.learning_rate or sft_cfg.get("learning_rate", 3e-4),
        logging_steps=sft_cfg.get("logging_steps", 5),
        num_train_epochs=args.num_train_epochs or sft_cfg.get("num_train_epochs", 5),
        bf16=sft_cfg.get("bf16", True),
        save_strategy=sft_cfg.get("save_strategy", "epoch"),
        eval_strategy=sft_cfg.get("eval_strategy", "epoch"),
        report_to=sft_cfg.get("report_to", "none"),
        gradient_checkpointing=sft_cfg.get("gradient_checkpointing", True),
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_train,
        eval_dataset=dataset_test,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
    )
    trainer.train()


if __name__ == "__main__":
    main()
