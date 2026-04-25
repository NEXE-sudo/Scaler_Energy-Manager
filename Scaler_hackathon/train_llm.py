"""
train_llm.py

Fine-tunes an LLM on energy grid dispatch data using TRL SFTTrainer + LoRA.
Designed to run on a single GPU (T4/A100) in Google Colab.

Supports:
  - Standard HuggingFace path (transformers + trl + peft)
  - Unsloth path (faster, lower VRAM) — auto-detected

Usage:
    # Standard
    python train_llm.py --dataset dataset_clean.jsonl

    # With Unsloth (if installed)
    python train_llm.py --dataset dataset_clean.jsonl --use-unsloth

    # Override model
    python train_llm.py --model meta-llama/Meta-Llama-3-8B-Instruct

Requirements (install in Colab):
    pip install transformers trl peft accelerate bitsandbytes datasets
    # Optional: pip install unsloth
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS = {
    "model":           "meta-llama/Meta-Llama-3-8B-Instruct",
    "output_dir":      "./lora_energy_grid",
    "dataset":         "dataset_clean.jsonl",
    "max_seq_length":  1024,
    # LoRA
    "lora_r":          16,
    "lora_alpha":      32,
    "lora_dropout":    0.05,
    "lora_targets":    ["q_proj", "v_proj", "k_proj", "o_proj"],
    # Training
    "epochs":          2,
    "batch_size":      4,
    "grad_accum":      4,       # effective batch = 16
    "lr":              2e-4,
    "warmup_steps":    20,
    "max_steps":       -1,      # -1 = run full epochs; set >0 to cap (hackathon mode)
    "save_steps":      100,
    "logging_steps":   10,
    # Efficiency
    "load_in_4bit":    True,
    "dtype":           "float16",  # bfloat16 on A100, float16 on T4
    "gradient_checkpointing": True,
}

# Template used by DataCollatorForCompletionOnlyLM to identify response start
RESPONSE_TEMPLATE = "<|assistant|>\n"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_clean_dataset(path: str) -> List[Dict[str, str]]:
    """Load TRL-formatted JSONL (prompt + completion fields)."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                # Validate expected fields
                if "prompt" in rec and "completion" in rec:
                    records.append(rec)
    print(f"Loaded {len(records)} training samples from {path}")
    return records


def records_to_hf_dataset(records: List[Dict[str, str]]):
    """Convert list of dicts to a HuggingFace Dataset."""
    from datasets import Dataset
    return Dataset.from_list(records)


def formatting_func(example: Dict[str, str]) -> str:
    """
    Combine prompt + completion into a single training string.
    TRL SFTTrainer uses this when format='text' mode.
    The RESPONSE_TEMPLATE separator tells the collator which tokens to train on.
    """
    return example["prompt"] + example["completion"]


# ─────────────────────────────────────────────────────────────────────────────
# Standard HuggingFace training path
# ─────────────────────────────────────────────────────────────────────────────

def train_standard(cfg: Dict[str, Any], dataset) -> None:
    """Train using transformers + trl + peft (no Unsloth dependency)."""
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, TaskType, get_peft_model
    try:
        from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
    except ImportError:
        from trl.trainer import SFTTrainer, DataCollatorForCompletionOnlyLM

    print(f"\nLoading model: {cfg['model']}")

    # 4-bit quantisation for low-VRAM training
    bnb_config = None
    if cfg["load_in_4bit"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False  # Required for gradient checkpointing

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=True)
    tokenizer.pad_token     = tokenizer.eos_token
    tokenizer.padding_side  = "right"

    # LoRA config
    lora_config = LoraConfig(
        task_type     = TaskType.CAUSAL_LM,
        r             = cfg["lora_r"],
        lora_alpha    = cfg["lora_alpha"],
        lora_dropout  = cfg["lora_dropout"],
        target_modules = cfg["lora_targets"],
        bias          = "none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Collator — trains ONLY on completion tokens (not the prompt)
    collator = DataCollatorForCompletionOnlyLM(
        response_template = RESPONSE_TEMPLATE,
        tokenizer         = tokenizer,
    )

    training_args = TrainingArguments(
        output_dir              = cfg["output_dir"],
        num_train_epochs        = cfg["epochs"],
        per_device_train_batch_size = cfg["batch_size"],
        gradient_accumulation_steps = cfg["grad_accum"],
        learning_rate           = cfg["lr"],
        warmup_steps            = cfg["warmup_steps"],
        max_steps               = cfg["max_steps"] if cfg["max_steps"] > 0 else -1,
        fp16                    = (cfg["dtype"] == "float16"),
        bf16                    = (cfg["dtype"] == "bfloat16"),
        gradient_checkpointing  = cfg["gradient_checkpointing"],
        logging_steps           = cfg["logging_steps"],
        save_steps              = cfg["save_steps"],
        save_total_limit        = 2,
        report_to               = "none",
        dataloader_num_workers  = 0,
    )

    trainer = SFTTrainer(
        model             = model,
        tokenizer         = tokenizer,
        train_dataset     = dataset,
        formatting_func   = formatting_func,
        data_collator     = collator,
        max_seq_length    = cfg["max_seq_length"],
        args              = training_args,
    )

    print("\nStarting training...")
    trainer.train()

    # Save LoRA adapter weights
    model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"\n[✓] Model saved to {cfg['output_dir']}")


# ─────────────────────────────────────────────────────────────────────────────
# Unsloth training path (faster, lower VRAM)
# ─────────────────────────────────────────────────────────────────────────────

def train_unsloth(cfg: Dict[str, Any], dataset) -> None:
    """Train using Unsloth for 2x speed and ~50% lower VRAM."""
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
        from transformers import TrainingArguments
    except ImportError:
        print("Unsloth not installed. Run: pip install unsloth")
        print("Falling back to standard training...")
        train_standard(cfg, dataset)
        return

    import torch

    dtype = torch.float16 if cfg["dtype"] == "float16" else torch.bfloat16

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name    = cfg["model"],
        max_seq_length = cfg["max_seq_length"],
        dtype         = dtype,
        load_in_4bit  = cfg["load_in_4bit"],
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r              = cfg["lora_r"],
        lora_alpha     = cfg["lora_alpha"],
        lora_dropout   = cfg["lora_dropout"],
        target_modules = cfg["lora_targets"],
        bias           = "none",
        use_gradient_checkpointing = cfg["gradient_checkpointing"],
    )

    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    collator = DataCollatorForCompletionOnlyLM(
        response_template = RESPONSE_TEMPLATE,
        tokenizer         = tokenizer,
    )

    training_args = TrainingArguments(
        output_dir              = cfg["output_dir"],
        num_train_epochs        = cfg["epochs"],
        per_device_train_batch_size = cfg["batch_size"],
        gradient_accumulation_steps = cfg["grad_accum"],
        learning_rate           = cfg["lr"],
        warmup_steps            = cfg["warmup_steps"],
        max_steps               = cfg["max_steps"] if cfg["max_steps"] > 0 else -1,
        fp16                    = (cfg["dtype"] == "float16"),
        bf16                    = (cfg["dtype"] == "bfloat16"),
        logging_steps           = cfg["logging_steps"],
        save_steps              = cfg["save_steps"],
        save_total_limit        = 2,
        report_to               = "none",
        dataloader_num_workers  = 0,
        optim                   = "adamw_8bit",  # Unsloth-optimised
    )

    trainer = SFTTrainer(
        model             = model,
        tokenizer         = tokenizer,
        train_dataset     = dataset,
        formatting_func   = formatting_func,
        data_collator     = collator,
        max_seq_length    = cfg["max_seq_length"],
        args              = training_args,
    )

    print("\nStarting training (Unsloth)...")
    trainer.train()

    model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"\n[✓] Model saved to {cfg['output_dir']}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM on energy grid dispatch data")
    parser.add_argument("--dataset",     type=str, default=DEFAULTS["dataset"])
    parser.add_argument("--model",       type=str, default=DEFAULTS["model"])
    parser.add_argument("--output-dir",  type=str, default=DEFAULTS["output_dir"])
    parser.add_argument("--epochs",      type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--lr",          type=float, default=DEFAULTS["lr"])
    parser.add_argument("--max-steps",   type=int, default=DEFAULTS["max_steps"],
                        help="Set to e.g. 200 for quick hackathon run")
    parser.add_argument("--lora-r",      type=int, default=DEFAULTS["lora_r"])
    parser.add_argument("--no-4bit",     action="store_true", help="Disable 4-bit quantisation")
    parser.add_argument("--use-unsloth", action="store_true", help="Use Unsloth backend")
    parser.add_argument("--dtype",       type=str, default=DEFAULTS["dtype"],
                        choices=["float16", "bfloat16"])
    parser.add_argument("--batch-size",  type=int, default=DEFAULTS["batch_size"])
    args = parser.parse_args()

    cfg = {**DEFAULTS}
    cfg["model"]       = args.model
    cfg["output_dir"]  = args.output_dir
    cfg["epochs"]      = args.epochs
    cfg["batch_size"]  = args.batch_size
    cfg["lr"]          = args.lr
    cfg["max_steps"]   = args.max_steps
    cfg["lora_r"]      = args.lora_r
    cfg["load_in_4bit"] = not args.no_4bit
    cfg["dtype"]       = args.dtype

    print(f"Model      : {cfg['model']}")
    print(f"Dataset    : {args.dataset}")
    print(f"Output     : {cfg['output_dir']}")
    print(f"Epochs     : {cfg['epochs']}")
    print(f"Max steps  : {cfg['max_steps']} (-1 = full epochs)")
    print(f"LoRA r     : {cfg['lora_r']}")
    print(f"4-bit      : {cfg['load_in_4bit']}")
    print(f"Unsloth    : {args.use_unsloth}")

    # Load dataset
    records = load_clean_dataset(args.dataset)
    if not records:
        print("ERROR: No valid records in dataset. Run data_generation.py first.")
        sys.exit(1)

    dataset = records_to_hf_dataset(records)

    # Train
    if args.use_unsloth:
        train_unsloth(cfg, dataset)
    else:
        train_standard(cfg, dataset)


if __name__ == "__main__":
    main()