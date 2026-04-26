"""
train_llm.py

Fine-tunes an LLM on energy grid dispatch data using TRL SFTTrainer + LoRA.
Designed to run on a single GPU (T4/A100) in Google Colab.

Supports:
  - Standard HuggingFace path (transformers + trl + peft)
  - Unsloth path (faster, lower VRAM)  auto-detected

Usage:
    # Standard
    python train_llm.py --dataset dataset_clean.jsonl

    # With Unsloth (if installed)
    python train_llm.py --dataset dataset_clean.jsonl --use-unsloth

    # Override model
    python train_llm.py --model mistralai/Mistral-7B-Instruct-v0.2

Requirements (install in Colab):
    pip install transformers trl peft accelerate bitsandbytes datasets
    # Optional: pip install unsloth
"""

import argparse
from tomlkit import item
import torch
import os
import sys
import json
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
from pathlib import Path
from typing import Any, Dict, List, Optional
import warnings
warnings.filterwarnings("ignore")

# 
# Config
# 

DEFAULTS = {
    "model":           "mistralai/Mistral-7B-Instruct-v0.2",
    "output_dir":      "./lora_energy_grid",
    "dataset":         "dataset_clean.jsonl",
    "max_seq_length":  2048,
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
    "load_in_4bit":    False,
    "dtype":           "float16",  # bfloat16 on A100, float16 on T4
    "gradient_checkpointing": True,
}

# Template used by DataCollatorForCompletionOnlyLM to identify response start
RESPONSE_TEMPLATE = "### Response:\n"

def detect_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def resolve_device(arg_device: str) -> str:
    import torch
    if arg_device == "cpu":
        return "cpu"
    if arg_device == "cuda":
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"

# 
# Dataset loading
# 

def load_clean_dataset(path: str) -> List[Dict[str, str]]:
    """Load TRL-formatted JSONL (prompt + completion fields)."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
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
    """Convert list of dicts to a HuggingFace Dataset (prompt+completion merged into 'text')."""
    from datasets import Dataset
    merged = [{"text": r["prompt"] + r["completion"]} for r in records]
    return Dataset.from_list(merged)


def formatting_func(example: Dict[str, Any]) -> List[str]:
    """
    Combine prompt + completion into a single training string.
    TRL SFTTrainer uses this when format='text' mode.
    The RESPONSE_TEMPLATE separator tells the collator which tokens to train on.
    """
    if isinstance(example.get("prompt"), list):
        return [p + c for p, c in zip(example["prompt"], example["completion"])]
    else:
        return [example["prompt"] + example["completion"]]

device = detect_device()
print(f"Detected device: {device}")

# 
# Standard HuggingFace training path
# 

def load_model_and_tokenizer(cfg: dict, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = None
    if cfg["load_in_4bit"] and device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"],
        quantization_config=bnb_config,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    )

    if device == "cpu":
        model.to("cpu")

    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

def train_standard(cfg, dataset, device):
    import torch
    from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    print(" Starting training pipeline...")

    #  Ensure output path exists
    cfg["output"] = cfg.get("output_dir", "./lora_energy_grid")

    #  DEBUG: inspect dataset structure
    if len(dataset) == 0:
        raise ValueError(" Dataset is completely empty before processing")

    print("\n RAW DATA SAMPLE:")
    print(dataset[0])

    #  Robust merge function (auto-detect format)
    def merge_prompt_completion(dataset):
        merged = []

        for item in dataset:
            text = None

            # Case 1: already merged
            if "text" in item:
                text = item["text"]

            # Case 2: prompt + completion
            elif "prompt" in item and "completion" in item:
                text = item["prompt"] + item["completion"]

            # Case 3: prompt + response
            elif "prompt" in item and "response" in item:
                text = item["prompt"] + item["response"]

            # Case 4: input + output
            elif "input" in item and "output" in item:
                text = item["input"] + item["output"]

            # Case 5: system + prompt + response
            elif "system" in item and "prompt" in item and "response" in item:
                text = f"<|system|>\n{item['system']}\n<|user|>\n{item['prompt']}\n<|assistant|>\n{item['response']}"

            if text:
                merged.append({"text": text})
                
            else:
                print(f"[WARN] Skipping record with unrecognised format: {list(item.keys())}")
        return merged
    

    dataset = merge_prompt_completion(dataset)

    print(f"\n Dataset size after merge: {len(dataset)}")

    #  HARD STOP if still empty
    if len(dataset) == 0:
        raise ValueError(" Dataset is EMPTY after merge. Format mismatch.")

    print("\n SAMPLE AFTER MERGE:")
    print(dataset[0]["text"][:500])

    #  Load model
    print(f"\nLoading model: {cfg['model']}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"],
        torch_dtype=torch.float32,
        device_map=None
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model.to("cpu")
    model.config.use_cache = False

    #  LoRA config
    lora_config = LoraConfig(
        r=cfg.get("lora_r", 16),
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    #  Training arguments
    training_args = TrainingArguments(
        output_dir=cfg["output"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=cfg.get("epochs", 2),
        max_steps=cfg.get("max_steps", -1) if cfg.get("max_steps", -1) > 0 else -1,
        logging_steps=1,
        save_strategy="no",
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        report_to="none"
    )

    from transformers import Trainer

    #  Tokenization function
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )

        #  CRITICAL FIX: add labels
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    # Convert dataset (list  HF dataset)
    from datasets import Dataset
    hf_dataset = Dataset.from_list(dataset)

    # Tokenize
    tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)

    # Set format for PyTorch
    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print("\n Starting training...")
    trainer.train()

    print("\n Saving LoRA model...")
    model.save_pretrained(cfg["output"])
    tokenizer.save_pretrained(cfg["output"])

    print("\n Training complete!")    
# 
# Unsloth training path (faster, lower VRAM)
# 

def train_unsloth(cfg: Dict[str, Any], dataset, device) -> None:
    """Train using Unsloth for 2x speed and ~50% lower VRAM."""
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
        from transformers import TrainingArguments
    except ImportError:
        print("Unsloth not installed. Run: pip install unsloth")
        print("Falling back to standard training...")
        train_standard(cfg, dataset, device)
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
    print(f"\n[] Model saved to {cfg['output_dir']}")


# 
# Entry point
# 

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
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    
    def apply_hardware_config(cfg: dict, device: str, user_model: str, default_model: str) -> dict:
        """
        Adjust config safely based on hardware.
        Only overrides model if user did NOT explicitly set one.
        """
        if device == "cpu" and user_model == default_model:
            print("Switching to CPU-safe config")

            cfg["model"] = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            cfg["load_in_4bit"] = False
            cfg["batch_size"] = 1
            cfg["grad_accum"] = 1
            cfg["gradient_checkpointing"] = False

        return cfg
    
    if args.device == "cpu":
        device = "cpu"
    elif args.device == "cuda":
        device = "cuda"
    else:
        device = detect_device()

    cfg = {**DEFAULTS}
    cfg["model"]       = args.model
    cfg["output_dir"]  = args.output_dir
    cfg["epochs"]      = args.epochs
    cfg["batch_size"]  = args.batch_size
    cfg["lr"]          = args.lr
    cfg["max_steps"]   = args.max_steps
    cfg["lora_r"]      = args.lora_r
    cfg["load_in_4bit"] = not args.no_4bit
    cfg["dtype"] = args.dtype

    cfg = apply_hardware_config(cfg, device, args.model, DEFAULTS["model"])

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

    if args.use_unsloth:
        dataset = records_to_hf_dataset(records)
        train_unsloth(cfg, dataset, device)
    else:
        train_standard(cfg, records, device)  # train_standard expects raw list


if __name__ == "__main__":
    main()