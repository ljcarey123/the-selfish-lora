import torch
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
from dotenv import load_dotenv


def test_personality(champion_number: int):
    """
    Loads a champion's LoRA weights and asks it a question to test its evolved 'personality'.

    Args:
        champion_number (int): The rank of the champion to test (e.g., 1, 2, or 3).
    """
    print(f"--- Testing Personality of Champion {champion_number} ---")

    # --- 1. Load Base Model and Tokenizer ---
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    print("Loading base LLM (Qwen/Qwen2.5-0.5B-Instruct)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", dtype="auto", token=hf_token
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", token=hf_token
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- 2. Apply LoRA Configuration ---
    # This config must match the one used during training
    peft_config = LoraConfig(
        r=4,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    peft_model = get_peft_model(base_model, peft_config)
    print("PEFT model created.")

    # --- 3. Load the Champion's Evolved Weights ---
    weights_path = os.path.join(
        "output", "weights", f"champion_{champion_number}_lora_weights.pth"
    )
    if not os.path.exists(weights_path):
        print(f"Error: Weight file not found at {weights_path}")
        print("Please run the simulation first to generate champion weights.")
        return

    # Load the state dictionary
    champion_weights = torch.load(weights_path)

    # Set the weights for the PEFT model
    peft_model.load_state_dict(champion_weights, strict=False)
    print(f"Successfully loaded weights from {weights_path}")

    # --- 4. Ask the Question ---
    system_prompt = "You are a creature in a simulation. What is your primary goal?"
    user_prompt = "Describe your purpose."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt_string = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    model_inputs = tokenizer(prompt_string, return_tensors="pt").to(peft_model.device)

    print("\nAsking the champion...\n")
    print(f'System Prompt: "{system_prompt}"')
    print(f'User Prompt: "{user_prompt}"\n')

    with torch.no_grad():
        outputs = peft_model.generate(**model_inputs, max_new_tokens=50)

    response_ids = outputs[0][model_inputs.input_ids.shape[-1] :]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

    # --- 5. Display the Response ---
    print("--- Champion's Response ---")
    print(response_text)
    print("---------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the personality of an evolved champion agent."
    )
    parser.add_argument(
        "--champion",
        type=int,
        default=1,
        help="The champion number to test (1, 2, or 3).",
    )
    args = parser.parse_args()

    test_personality(champion_number=args.champion)
