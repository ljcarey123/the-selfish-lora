import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
import os
from dotenv import load_dotenv
from .config_loader import SYSTEM_PROMPT


def run_lora_test():
    """
    An isolated test to verify that applying different LoRA weights to a PeftModel
    results in different outputs for the same prompt.
    """
    print("--- LoRA Isolation Test ---")

    # 1. Load Model and Tokenizer
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"

    print(f"Loading base model: {model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype="auto", token=hf_token
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. Create PEFT Model
    peft_config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    peft_model = get_peft_model(base_model, peft_config)
    print("PeftModel created.")

    # 3. Find LoRA layers and define weights
    lora_layer_names = [
        name for name, param in peft_model.named_parameters() if "lora" in name
    ]

    zero_weights = {
        name: torch.zeros_like(peft_model.get_parameter(name))
        for name in lora_layer_names
    }
    print("\nCreated a set of random LoRA weights for testing.")

    # Create a set of weights with a small mutation, just like in the simulation
    small_mutation_weights = {
        name: torch.zeros_like(peft_model.get_parameter(name))
        for name in lora_layer_names
    }
    mutation_strength = 0.0005  # Same as in main.py
    for name in small_mutation_weights:
        mutation = torch.randn_like(small_mutation_weights[name]) * mutation_strength
        small_mutation_weights[name] += mutation
    print(
        f"Created a set of weights with a small mutation (strength={mutation_strength})."
    )

    # 4. Define a list of test prompts
    # Use the centralized system prompt
    system_prompt = SYSTEM_PROMPT

    # Define a dictionary of different test scenarios (user prompts)
    test_scenarios = {
        "Go East": """SITUATION:
- A resource 'R' is to your East.
- History: ['North', 'North', 'West'].
- Actions: ['Gather', 'North', 'South', 'East', 'West'].

ACTION:""",
        "Gather": """SITUATION:
- You are on top of a resource 'R'.
- History: ['North', 'East', 'East'].
- Actions: ['Gather', 'North', 'South', 'East', 'West'].

ACTION:""",
        "Blocked by Wall": """SITUATION:
- A resource 'R' is to your North.
- There is a wall to your North.
- History: ['South', 'South', 'West'].
- Actions: ['Gather', 'North', 'South', 'East', 'West'].

ACTION:""",
    }

    success_count = 0

    for test_name, user_prompt in test_scenarios.items():
        print(f"\n{'='*10} Running Test: {test_name} {'='*10}")

        # Combine the system prompt with the current user prompt scenario
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_string = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        model_inputs = tokenizer(prompt_string, return_tensors="pt").to(
            peft_model.device
        )

        # Test with ZERO weights
        print("\n--- Testing with Zero-Initialized LoRA Weights ---")
        with torch.no_grad():
            for name, param in peft_model.named_parameters():
                if name in zero_weights:
                    param.data.copy_(zero_weights[name])

            outputs_zero = peft_model.generate(
                **model_inputs, max_new_tokens=15, use_cache=False
            )
            response_ids_zero = outputs_zero[0][model_inputs.input_ids.shape[-1] :]
            response_zero = tokenizer.decode(
                response_ids_zero, skip_special_tokens=True
            )
            print(f"Prompt:\n{prompt_string}")
            print(f"\nResponse:\n'{response_zero}'")

        # Test with SMALL MUTATION weights
        print("\n--- Testing with Small Mutation LoRA Weights ---")
        with torch.no_grad():
            for name, param in peft_model.named_parameters():
                if name in small_mutation_weights:
                    param.data.copy_(small_mutation_weights[name])

            outputs_small = peft_model.generate(
                **model_inputs, max_new_tokens=15, use_cache=False
            )
            response_ids_small = outputs_small[0][model_inputs.input_ids.shape[-1] :]
            response_small = tokenizer.decode(
                response_ids_small, skip_special_tokens=True
            )
            print(f"Prompt:\n{prompt_string}")
            print(f"\nResponse:\n'{response_small}'")

        if response_zero != response_small:
            success_count += 1

    # 5. Conclusion
    print(f"\n{'='*10} FINAL CONCLUSION {'='*10}")
    if success_count == len(test_scenarios):
        print("SUCCESS: The LoRA weight injection mechanism is working correctly.")
    else:
        print("FAILURE: The LoRA weight injection mechanism appears to be broken.")


if __name__ == "__main__":
    run_lora_test()
