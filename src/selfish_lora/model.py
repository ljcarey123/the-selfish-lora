"""
Load the base LLM with a LoRA adapter attached.

The adapter's A matrices keep PEFT's random initialisation and are shared by
every genome; only the B matrices are evolved (see genome.py). Seeding before
get_peft_model makes that shared A identical across runs.
"""

import os

import torch
from dotenv import load_dotenv, find_dotenv
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def load_model(
    model_name: str = DEFAULT_MODEL,
    lora_rank: int = 4,
    target_modules: tuple = ("q_proj", "v_proj"),
    seed: int = 0,
):
    """
    Load a causal LM with a LoRA adapter, ready for inference.

    Args:
        model_name: Hugging Face model id
        lora_rank: LoRA rank r (small keeps genomes cheap to mutate and store)
        target_modules: Linear layers to adapt
        seed: Seed for the shared random A matrices

    Returns:
        (peft_model, tokenizer)
    """
    load_dotenv(find_dotenv())
    hf_token = os.getenv("HF_TOKEN")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    if device == "cpu":
        print("WARNING: CUDA not available - using CPU (will be very slow!)")

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, token=hf_token)

    torch.manual_seed(seed)
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=2 * lora_rank,
        target_modules=list(target_modules),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config).to(device)
    model.eval()

    print(f"Loaded {model_name} on {device} ({dtype}), LoRA r={lora_rank} on {list(target_modules)}")
    return model, tokenizer
