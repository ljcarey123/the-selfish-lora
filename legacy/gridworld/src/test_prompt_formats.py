"""
Test different prompt formats with Qwen model to find optimal format.

This script tests various ways of prompting Qwen2.5-0.5B-Instruct to ensure
we're using the correct chat template and getting valid action outputs.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import os

# Load environment
load_dotenv()
model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")

print("="*70)
print("PROMPT FORMAT TESTING")
print("="*70)
print(f"\nLoading model: {model_name}")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Check for CUDA
if torch.cuda.is_available():
    print("CUDA available - using GPU")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda"
    )
else:
    print("CUDA not available - using CPU (slower)")
    model = AutoModelForCausalLM.from_pretrained(model_name)

model.eval()

print(f"Model loaded. Device: {next(model.parameters()).device}")
print(f"Tokenizer chat template exists: {tokenizer.chat_template is not None}")
print()

# Test scenario
scenario = {
    "energy": 19,
    "status": "OK",
    "vision": "WALL NORTH, WALL EAST, FOOD NORTH-WEST, WALL SOUTH-EAST",
    "last_action": "MOVE_NORTH"
}


def test_format(name: str, messages: list, max_new_tokens: int = 10):
    """Test a specific prompt format."""
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"{'='*70}")

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    print(f"\nFormatted prompt:\n{prompt}")
    print(f"\n{'-'*70}")

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_output[len(prompt):].strip()

    print(f"Response: {response}")
    print(f"Response length: {len(response)} chars")

    # Check if valid action
    valid_actions = ["MOVE_NORTH", "MOVE_SOUTH", "MOVE_EAST", "MOVE_WEST"]
    found_action = None
    for action in valid_actions:
        if action in response.upper():
            found_action = action
            break

    if found_action:
        print(f"[OK] Valid action found: {found_action}")
    else:
        print(f"[FAIL] No valid action found")

    return response


# Test 1: Current format (instructions in user message)
print("\n" + "="*70)
print("FORMAT 1: Instructions as user message (current)")
print("="*70)

messages1 = [
    {
        "role": "user",
        "content": f"""Move to FOOD for energy. Move into AGENTs to steal energy. Avoid WALLs.
Energy: {scenario['energy']} ({scenario['status']})
{scenario['vision']}
Last: {scenario['last_action']}
Output one: MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, MOVE_WEST"""
    }
]

test_format("Current format", messages1)


# Test 2: System prompt + user message
print("\n" + "="*70)
print("FORMAT 2: System prompt + user message")
print("="*70)

messages2 = [
    {
        "role": "system",
        "content": "You are a game agent. Output only one movement command: MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, or MOVE_WEST."
    },
    {
        "role": "user",
        "content": f"""Move to FOOD for energy. Move into AGENTs to steal energy. Avoid WALLs.
Energy: {scenario['energy']} ({scenario['status']})
{scenario['vision']}
Last: {scenario['last_action']}
Choose one: MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, MOVE_WEST"""
    }
]

test_format("System + user", messages2)


# Test 3: System prompt with clear task definition
print("\n" + "="*70)
print("FORMAT 3: Clear system prompt with structured user input")
print("="*70)

messages3 = [
    {
        "role": "system",
        "content": "You control an agent in a grid world. Output exactly one action: MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, or MOVE_WEST. Output ONLY the action, nothing else."
    },
    {
        "role": "user",
        "content": f"""Rules: Move to FOOD for energy. Move into AGENTs to steal energy. Avoid WALLs.

Current state:
- Energy: {scenario['energy']} ({scenario['status']})
- Vision: {scenario['vision']}
- Last action: {scenario['last_action']}

Action:"""
    }
]

test_format("Structured system + user", messages3)


# Test 4: Few-shot examples
print("\n" + "="*70)
print("FORMAT 4: Few-shot examples")
print("="*70)

messages4 = [
    {
        "role": "system",
        "content": "You are a game agent. Output only one command: MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, or MOVE_WEST."
    },
    {
        "role": "user",
        "content": "Energy: 25 (OK)\nFOOD NORTH, WALL SOUTH\nAction:"
    },
    {
        "role": "assistant",
        "content": "MOVE_NORTH"
    },
    {
        "role": "user",
        "content": "Energy: 15 (LOW)\nWALL NORTH, FOOD EAST, AGENT WEST\nAction:"
    },
    {
        "role": "assistant",
        "content": "MOVE_EAST"
    },
    {
        "role": "user",
        "content": f"Energy: {scenario['energy']} ({scenario['status']})\n{scenario['vision']}\nAction:"
    }
]

test_format("Few-shot examples", messages4)


# Test 5: Ultra-minimal format
print("\n" + "="*70)
print("FORMAT 5: Ultra-minimal")
print("="*70)

messages5 = [
    {
        "role": "system",
        "content": "Output one: MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, MOVE_WEST"
    },
    {
        "role": "user",
        "content": f"E:{scenario['energy']} {scenario['vision']}"
    }
]

test_format("Ultra-minimal", messages5)


# Test 6: Question format
print("\n" + "="*70)
print("FORMAT 6: Question format")
print("="*70)

messages6 = [
    {
        "role": "user",
        "content": f"""You are at Energy: {scenario['energy']}
You see: {scenario['vision']}
Which direction should you move? Choose MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, or MOVE_WEST."""
    }
]

test_format("Question format", messages6)


print("\n" + "="*70)
print("TESTING COMPLETE")
print("="*70)
print("\nReview the outputs above to determine which format produces the most reliable action outputs.")
