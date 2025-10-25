import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_PATH = os.path.join(SCRIPT_DIR, "data", "test_data.json")
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


class StopOnNewlineOrSpace(StoppingCriteria):
    """Stop generation when we encounter newline, space after a word, or punctuation."""
    def __init__(self, tokenizer, start_length):
        self.tokenizer = tokenizer
        self.start_length = start_length
        self.stop_tokens = set([
            tokenizer.encode("\n", add_special_tokens=False)[0] if tokenizer.encode("\n", add_special_tokens=False) else None,
            tokenizer.encode(",", add_special_tokens=False)[0] if tokenizer.encode(",", add_special_tokens=False) else None,
        ])
        self.stop_tokens.discard(None)
    
    def __call__(self, input_ids, scores, **kwargs):
        # Check if we've generated at least one token beyond the prompt
        if input_ids.shape[1] <= self.start_length:
            return False
        
        # Get the last generated token
        last_token = input_ids[0, -1].item()
        
        # Stop if we hit a stop token
        if last_token in self.stop_tokens:
            return True
        
        # Decode the generated text to check for word boundaries
        generated_ids = input_ids[0, self.start_length:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Stop if we have a complete word (contains space or has length > 2)
        if " " in generated_text or len(generated_text.split()) > 1:
            return True
            
        return False


def get_model_prediction(model, tokenizer, example):
    chat_template = tokenizer.apply_chat_template(
        example, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_template, return_tensors="pt").to(model.device)
    start_length = inputs.input_ids.shape[1]

    # Create stopping criteria
    stopping_criteria = StoppingCriteriaList([
        StopOnNewlineOrSpace(tokenizer, start_length)
    ])

    # Generate with constrained parameters for single-word action prediction
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,  # Allow some buffer
        temperature=0.1,  # Low temperature for more deterministic output
        do_sample=False,  # Greedy decoding for consistency
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        stopping_criteria=stopping_criteria,
    )

    response_ids = outputs[0][inputs.input_ids.shape[1] :]
    prediction = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    # Extract only the first word (the action)
    # This handles cases where model generates extra text
    first_word = prediction.split()[0] if prediction.split() else prediction
    # Remove any punctuation or special characters
    first_word = first_word.strip(".,!?;:").strip()

    return first_word


def run_evaluation(model, tokenizer, test_data, model_name):
    print(f"\n{'='*20} EVALUATING: {model_name.upper()} {'='*20}")
    correct_predictions = 0
    total_predictions = len(test_data)
    results = []
    for example in tqdm(test_data, desc=f"Evaluating {model_name}"):
        prompt_messages = example[:-1]
        true_action = example[-1]["content"]
        predicted_action = get_model_prediction(model, tokenizer, prompt_messages)
        results.append({"predicted": predicted_action, "true": true_action})
        if predicted_action.strip() == true_action:
            correct_predictions += 1

    print("\nDetailed Predictions (Predicted vs. True):")
    for res in results:
        status = "✅" if res["predicted"] == res["true"] else "❌"
        print(f"- {status} Predicted: {res['predicted']:<7} | True: {res['true']}")

    accuracy = (correct_predictions / total_predictions) * 100
    print(f"\n{model_name} Accuracy: {accuracy:.2f}%")
    return accuracy


if __name__ == "__main__":
    adapter_path = os.path.join(SCRIPT_DIR, "output", "final_champion_model")
    test_data = None
    with open(TEST_DATA_PATH, "r") as f:
        test_data = json.load(f)

    # --- 1. Evaluate Base Model ---
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, trust_remote_code=True
    )
    base_tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID, trust_remote_code=True
    )
    base_tokenizer.pad_token = base_tokenizer.eos_token
    base_accuracy = run_evaluation(base_model, base_tokenizer, test_data, "Base Model")
    del base_model
    del base_tokenizer
    torch.cuda.empty_cache()

    # --- 2. Evaluate Fine-Tuned Model ---
    tuned_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, trust_remote_code=True
    )
    tuned_model = PeftModel.from_pretrained(tuned_model, adapter_path)
    tuned_tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID, trust_remote_code=True
    )
    tuned_tokenizer.pad_token = tuned_tokenizer.eos_token
    tuned_accuracy = run_evaluation(
        tuned_model, tuned_tokenizer, test_data, "Fine-Tuned Model"
    )
    del tuned_model
    del tuned_tokenizer
    torch.cuda.empty_cache()

    # --- 3. Final Summary ---
    print(f"\n{'='*25} FINAL SUMMARY {'='*25}")
    print(f"Base Model Accuracy:     {base_accuracy:.2f}%")
    print(f"Fine-Tuned Accuracy:   {tuned_accuracy:.2f}%")
    print(f"Improvement:             {tuned_accuracy - base_accuracy:+.2f}%")
    print("=" * 67)
