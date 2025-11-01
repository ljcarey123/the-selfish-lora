import os
import json
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)
from peft import PeftModel
import math

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TEST_DATA_PATH = os.path.join(SCRIPT_DIR, "data", "test.jsonl")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
BASE_MODEL_PATH = MODEL_ID
LORA_ADAPTER_PATH = os.path.join(OUTPUT_DIR, "best_model")


class StopOnNewlineOrSpace(StoppingCriteria):
    """Stop generation when we encounter newline, space after a word, or punctuation."""

    def __init__(self, tokenizer, start_length):
        self.tokenizer = tokenizer
        self.start_length = start_length
        self.stop_tokens = set(
            [
                (
                    tokenizer.encode("\n", add_special_tokens=False)[0]
                    if tokenizer.encode("\n", add_special_tokens=False)
                    else None
                ),
                (
                    tokenizer.encode(",", add_special_tokens=False)[0]
                    if tokenizer.encode(",", add_special_tokens=False)
                    else None
                ),
            ]
        )
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
        generated_ids = input_ids[0, self.start_length :]
        generated_text = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()

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
    stopping_criteria = StoppingCriteriaList(
        [StopOnNewlineOrSpace(tokenizer, start_length)]
    )

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
    raw_output = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    # Extract only the first word (the action)
    # This handles cases where model generates extra text
    first_word = raw_output.split()[0] if raw_output.split() else raw_output
    # Remove any punctuation or special characters
    parsed_action = first_word.strip(".,!?;:").strip()

    return raw_output, parsed_action


def run_evaluation(model, tokenizer, test_data, model_name):
    print(f"\n{'='*20} EVALUATING: {model_name.upper()} {'='*20}")
    correct_predictions = 0
    total_predictions = len(test_data)
    valid_predictions = 0
    results = []

    # Track action distributions
    predicted_actions = {}
    true_actions = {}
    valid_actions = ["North", "South", "East", "West", "Gather"]

    for example in tqdm(test_data, desc=f"Evaluating {model_name}"):
        prompt_messages = example[:-1]
        true_action = example[-1]["content"]
        raw_output, predicted_action = get_model_prediction(
            model, tokenizer, prompt_messages
        )

        # Track distributions
        predicted_actions[predicted_action] = (
            predicted_actions.get(predicted_action, 0) + 1
        )
        true_actions[true_action] = true_actions.get(true_action, 0) + 1

        # Check if valid
        if predicted_action in valid_actions:
            valid_predictions += 1

        results.append(
            {
                "raw_output": raw_output,
                "predicted": predicted_action,
                "true": true_action,
            }
        )
        if predicted_action.strip() == true_action:
            correct_predictions += 1

    accuracy = (correct_predictions / total_predictions) * 100
    valid_rate = (valid_predictions / total_predictions) * 100

    # Calculate diversity metrics
    num_unique_actions = len(
        [a for a in predicted_actions.keys() if a in valid_actions]
    )
    diversity_score = num_unique_actions / len(valid_actions)  # 0.0 to 1.0

    # Calculate action balance (entropy-based)
    # Perfect balance = all actions used equally
    total_valid = sum(predicted_actions.get(a, 0) for a in valid_actions)
    if total_valid > 0:
        action_probs = [
            predicted_actions.get(a, 0) / total_valid for a in valid_actions
        ]
        entropy = -sum(p * math.log(p + 1e-10) for p in action_probs if p > 0)
        max_entropy = math.log(len(valid_actions))  # Maximum possible entropy
        balance_score = entropy / max_entropy  # 0.0 to 1.0
    else:
        balance_score = 0.0

    # Evolution readiness score (for warm-start)
    # Prioritizes: valid outputs > diversity > balance
    evolution_score = (
        (valid_rate / 100.0) * 0.5 + diversity_score * 0.3 + balance_score * 0.2
    )
    evolution_score *= 100  # Convert to percentage

    print(f"\n{model_name} Metrics:")
    print(f"  Accuracy:          {accuracy:.2f}%")
    print(f"  Valid Action Rate: {valid_rate:.2f}%")
    print(
        f"  Diversity:         {num_unique_actions}/5 actions ({diversity_score*100:.1f}%)"
    )
    print(f"  Balance Score:     {balance_score*100:.1f}%")
    print(f"  Evolution Score:   {evolution_score:.1f}%")

    return (
        accuracy,
        valid_rate,
        predicted_actions,
        results,
        diversity_score,
        balance_score,
        evolution_score,
    )


if __name__ == "__main__":
    test_data = None
    with open(TEST_DATA_PATH, "r") as f:
        test_data = [json.loads(line) for line in f.readlines()]

    # --- 1. Evaluate Base Model ---
    print("\n" + "=" * 70)
    print("EVALUATING BASE MODEL")
    print("=" * 70)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, trust_remote_code=True
    )
    base_tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH, trust_remote_code=True
    )
    base_tokenizer.pad_token = base_tokenizer.eos_token
    (
        base_accuracy,
        base_valid_rate,
        base_dist,
        base_results,
        base_diversity,
        base_balance,
        base_evolution,
    ) = run_evaluation(base_model, base_tokenizer, test_data, "Base Model")
    del base_model
    del base_tokenizer
    torch.cuda.empty_cache()

    # --- 2. Evaluate Best LoRA Adapter ---
    print("\n" + "=" * 70)
    print("EVALUATING BEST LoRA ADAPTER")
    print("=" * 70)

    tuned_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, trust_remote_code=True
    )
    tuned_model = PeftModel.from_pretrained(tuned_model, LORA_ADAPTER_PATH)
    tuned_tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH, trust_remote_code=True
    )
    tuned_tokenizer.pad_token = tuned_tokenizer.eos_token

    accuracy, valid_rate, dist, results, diversity, balance, evolution = (
        run_evaluation(tuned_model, tuned_tokenizer, test_data, "Best LoRA Adapter")
    )

    del tuned_model
    del tuned_tokenizer
    torch.cuda.empty_cache()

    # --- 3. Final Summary ---
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\nBase Model:")
    print(f"  Accuracy:         {base_accuracy:.2f}%")
    print(f"  Valid Rate:       {base_valid_rate:.2f}%")
    print(f"  Evolution Score:  {base_evolution:.1f}%")

    print(f"\nBest LoRA Adapter:")
    print(f"  Accuracy:         {accuracy:.2f}%")
    print(f"  Valid Rate:       {valid_rate:.2f}%")
    print(f"  Diversity:        {diversity*100:.1f}%")
    print(f"  Balance:          {balance*100:.1f}%")
    print(f"  Evolution Score:  {evolution:.1f}%")

    print("=" * 70)

    # --- 4. Save Evaluation Report ---
    eval_report_path = os.path.join(OUTPUT_DIR, "EVALUATION_REPORT_FINAL.txt")
    with open(eval_report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("FINAL EVALUATION REPORT - TEST SET PERFORMANCE\n")
        f.write("=" * 70 + "\n\n")

        f.write("Base Model Performance:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Accuracy:         {base_accuracy:.2f}%\n")
        f.write(f"  Valid Rate:       {base_valid_rate:.2f}%\n")
        f.write(f"  Diversity:        {base_diversity*100:.1f}%\n")
        f.write(f"  Balance:          {base_balance*100:.1f}%\n")
        f.write(f"  Evolution Score:  {base_evolution:.1f}%\n\n")

        f.write("Best LoRA Adapter Performance:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Accuracy:         {accuracy:.2f}%\n")
        f.write(f"  Valid Rate:       {valid_rate:.2f}%\n")
        f.write(f"  Diversity:        {diversity*100:.1f}%\n")
        f.write(f"  Balance:          {balance*100:.1f}%\n")
        f.write(f"  Evolution Score:  {evolution:.1f}%\n\n")

        f.write("\nAction Distribution Comparison:\n")
        f.write("=" * 70 + "\n")

        # Base model distribution
        f.write("\nBase Model Predicted Actions:\n")
        for action in sorted(base_dist.keys()):
            count = base_dist[action]
            pct = (count / len(test_data)) * 100
            f.write(f"  {action}: {count} ({pct:.1f}%)\n")

        # Best LoRA adapter distribution
        f.write(f"\nBest LoRA Adapter Predicted Actions:\n")
        for action in sorted(dist.keys()):
            count = dist[action]
            pct = (count / len(test_data)) * 100
            f.write(f"  {action}: {count} ({pct:.1f}%)\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("\nMetric Definitions:\n")
        f.write("- Evolution Score: Composite metric for warm-start quality\n")
        f.write("  Formula: 50% Valid Rate + 30% Diversity + 20% Balance\n")
        f.write("  Higher is better for evolutionary simulation starting point\n")
        f.write("- Valid Rate: Percentage of outputs that are valid actions\n")
        f.write("- Diversity: Percentage of unique valid actions used (0-100%, max=5 actions)\n")
        f.write("- Balance: How evenly distributed the actions are (entropy-based, 0-100%)\n")
        f.write("- Accuracy: Percentage matching expected actions (less relevant for evolution)\n")
        f.write("\nTest set size: {} examples\n".format(len(test_data)))
        f.write("\nFor training metrics, see TRAINING_REPORT_FINAL.txt\n")

        # Add detailed predictions for best run
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"\nDETAILED PREDICTIONS - Best LoRA Adapter\n")
        f.write("=" * 70 + "\n\n")

        for i, res in enumerate(results, 1):
            status = "✅" if res["predicted"] == res["true"] else "❌"
            f.write(f"{i:2d}. {status} Raw Output: '{res['raw_output']}'\n")
            f.write(
                f"       → Parsed as: '{res['predicted']}'  |  Expected: '{res['true']}'\n\n"
            )

    print(f"\nEvaluation report saved to: {eval_report_path}")
