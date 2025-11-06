import os
import json
import torch
import glob
import argparse
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union, Any, Generator, ContextManager
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
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


# Default paths
def find_best_checkpoint(run_dir: str) -> Optional[str]:
    """Find the best checkpoint in a run directory based on validation loss."""
    checkpoint_dirs = []

    # Look for checkpoint directories
    for root, dirs, _ in os.walk(run_dir):
        if "checkpoint-" in root:
            checkpoint_dirs.append(root)

    if not checkpoint_dirs:
        # If no checkpoints, check if there's an adapter in the root
        if "adapter_model.safetensors" in os.listdir(run_dir):
            return run_dir
        return None

    # Find checkpoint with best validation loss
    best_checkpoint = None
    best_loss = float("inf")

    for checkpoint_dir in checkpoint_dirs:
        trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.json")
        if os.path.exists(trainer_state_path):
            try:
                with open(trainer_state_path, "r") as f:
                    state = json.load(f)
                    if "best_model_checkpoint" in state:
                        return state["best_model_checkpoint"]

                    # Fallback: find checkpoint with lowest validation loss
                    for log in state.get("log_history", []):
                        if "eval_loss" in log and log["eval_loss"] < best_loss:
                            best_loss = log["eval_loss"]
                            best_checkpoint = checkpoint_dir
            except (json.JSONDecodeError, KeyError):
                continue

    return (
        best_checkpoint or checkpoint_dirs[-1]
    )  # Return most recent if no validation loss found


def get_run_dirs() -> List[str]:
    """Get all run directories, sorted by modification time (newest first)."""
    run_dirs = glob.glob(os.path.join(OUTPUT_DIR, "run-*"))
    return sorted(
        [d for d in run_dirs if os.path.isdir(d)], key=os.path.getmtime, reverse=True
    )


def get_run_dir(
    run_spec: Optional[Union[int, str]] = None,
) -> Optional[Tuple[str, str]]:
    """Get a specific run directory and its best checkpoint by number, path, or return the latest run.

    Returns:
        Tuple of (run_dir, checkpoint_dir) or None if not found
    """
    if run_spec is None:
        # Default to most recent run
        runs = get_run_dirs()
        if not runs:
            return None
        run_dir = runs[0]
    elif isinstance(run_spec, int):
        # Get run by number (1-based index)
        runs = get_run_dirs()
        if not (0 < run_spec <= len(runs)):
            return None
        run_dir = runs[run_spec - 1]
    else:
        # Assume it's a path
        if not os.path.isdir(run_spec):
            return None
        run_dir = run_spec

    # Find best checkpoint in the run directory
    checkpoint_dir = find_best_checkpoint(run_dir)
    if checkpoint_dir is None:
        print(f"Warning: No valid checkpoints found in {run_dir}")
        return None

    return run_dir, checkpoint_dir


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
    # Prepare the input with chat template
    chat_template = tokenizer.apply_chat_template(
        example, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_template, return_tensors="pt").to(model.device)
    start_length = inputs.input_ids.shape[1]

    # Create stopping criteria - more permissive to allow different action words
    stopping_criteria = StoppingCriteriaList([
        StopOnNewlineOrSpace(tokenizer, start_length)
    ])

    # Generate with deterministic settings for evolutionary tasks
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,  # Just enough for action words
            temperature=0.1,  # Low temperature for deterministic outputs
            do_sample=False,  # Greedy decoding for consistency
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
        )

    response_ids = outputs[0][inputs.input_ids.shape[1]:]
    raw_output = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    # Return the full generated text (up to 20 tokens)
    # Remove any leading/trailing whitespace and newlines
    parsed_action = raw_output.strip()
    
    # Truncate to first 20 tokens if needed
    tokens = parsed_action.split()
    if len(tokens) > 20:
        parsed_action = ' '.join(tokens[:20])
    
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
    # Standardized action names from config/prompts.json
    valid_actions = ["move_north", "move_south", "move_east", "move_west", "gather", "stand_still"]

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


@contextmanager
def model_manager(
    model_path: str, is_lora: bool = False, checkpoint_path: Optional[str] = None
) -> Generator[Tuple[Any, Any], None, None]:
    """Context manager for loading and cleaning up models.

    Args:
        model_path: Path to the base model
        is_lora: Whether to load a LoRA adapter
        checkpoint_path: Path to the checkpoint (for LoRA)

    Yields:
        Tuple of (model, tokenizer)
    """
    model = None
    tokenizer = None

    try:
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        # Load LoRA adapter if specified
        if is_lora and checkpoint_path:
            model = PeftModel.from_pretrained(model, checkpoint_path, device_map="auto")
            model = model.merge_and_unload()

        model.eval()
        yield model, tokenizer

    except Exception as e:
        print(f"Error in model manager: {str(e)}")
        raise

    finally:
        # Cleanup
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        torch.cuda.empty_cache()


def evaluate_run(
    run_dir: str,
    checkpoint_dir: str,
    test_data: List[dict],
    base_model_path: str = BASE_MODEL_PATH,
) -> dict:
    """Evaluate a single model run.

    Args:
        run_dir: Directory containing the run
        checkpoint_dir: Directory containing the checkpoint to evaluate
        test_data: List of test examples
        base_model_path: Path to base model

    Returns:
        Dictionary containing evaluation metrics
    """
    print(f"\n{'='*70}")
    print(
        f"EVALUATING: {os.path.basename(run_dir)} - {os.path.basename(checkpoint_dir)}"
    )
    print("=" * 70)

    is_lora = checkpoint_dir != base_model_path

    try:
        with model_manager(base_model_path, is_lora, checkpoint_dir) as (
            model,
            tokenizer,
        ):
            accuracy, valid_rate, dist, results, diversity, balance, evolution = (
                run_evaluation(model, tokenizer, test_data, os.path.basename(run_dir))
            )

            return {
                "accuracy": accuracy,
                "valid_rate": valid_rate,
                "diversity": diversity,
                "balance": balance,
                "evolution": evolution,
                "action_distribution": dist,
                "results": results,
            }

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise


def evaluate_base_model(
    test_data: List[dict], base_model_path: str = BASE_MODEL_PATH
) -> dict:
    """Evaluate the base model without any fine-tuning."""
    print("\n" + "=" * 70)
    print("EVALUATING BASE MODEL")
    print("=" * 70)

    try:
        with model_manager(base_model_path, is_lora=False) as (model, tokenizer):
            accuracy, valid_rate, dist, results, diversity, balance, evolution = (
                run_evaluation(model, tokenizer, test_data, "Base Model")
            )

            return {
                "accuracy": accuracy,
                "valid_rate": valid_rate,
                "diversity": diversity,
                "balance": balance,
                "evolution": evolution,
                "action_distribution": dist,
                "results": results,
            }

    except Exception as e:
        print(f"Error evaluating base model: {str(e)}")
        raise


def save_evaluation_report(results: dict, output_path: str):
    """Save evaluation results to a report file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("MODEL EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")

        for model_name, metrics in results.items():
            f.write(f"{model_name} Performance:\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Accuracy:         {metrics['accuracy']:.2f}%\n")
            f.write(f"  Valid Rate:       {metrics['valid_rate']:.2f}%\n")
            f.write(f"  Diversity:        {metrics['diversity']*100:.1f}%\n")
            f.write(f"  Balance:          {metrics['balance']*100:.1f}%\n")
            f.write(f"  Evolution Score:  {metrics['evolution']:.1f}%\n\n")

            # Action distribution
            f.write("  Action Distribution:\n")
            for action, count in metrics["action_distribution"].items():
                f.write(f"    {action}: {count}\n")
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned models")
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Run to evaluate (number, path, or 'all'). If not specified, evaluates the most recent run.",
    )
    parser.add_argument(
        "--base",
        action="store_true",
        help="Include base model in evaluation",
    )
    args = parser.parse_args()

    # Load test data
    with open(TEST_DATA_PATH, "r") as f:
        test_data = [json.loads(line) for line in f.readlines()]

    results = {}

    # Evaluate base model if requested
    if args.base:
        results["Base Model"] = evaluate_base_model(test_data)

    # Determine which runs to evaluate
    if args.run == "all":
        run_dirs = get_run_dirs()
        print(f"Found {len(run_dirs)} runs to evaluate")
    else:
        run_dirs = []
        if args.run:
            # Try to interpret as a run number or path
            run_info = get_run_dir(args.run)
            if run_info:
                run_dirs = [
                    args.run
                ]  # Pass the original spec to get proper checkpoint handling
            else:
                print(f"Could not find run: {args.run}")
        else:
            # Default to most recent run
            print(f"Looking for runs in: {os.path.abspath(OUTPUT_DIR)}")
            print(f"Directory exists: {os.path.exists(OUTPUT_DIR)}")
            if os.path.exists(OUTPUT_DIR):
                print(f"Contents of {OUTPUT_DIR}:")
                for f in os.listdir(OUTPUT_DIR):
                    print(f"- {f} (dir: {os.path.isdir(os.path.join(OUTPUT_DIR, f))})")

            run_specs = get_run_dirs()
            print(f"Found {len(run_specs)} runs: {run_specs}")
            if run_specs:
                run_dirs = [run_specs[0]]
            else:
                print("No runs found to evaluate")

    # Evaluate each run
    for run_spec in run_dirs:
        try:
            # Get run directory and best checkpoint
            run_info = get_run_dir(run_spec)
            if run_info is None:
                print(f"Skipping invalid run: {run_spec}")
                continue

            run_dir, checkpoint_dir = run_info
            run_name = os.path.basename(run_dir)

            # Add checkpoint info to run name if it's a specific checkpoint
            if checkpoint_dir != run_dir:
                checkpoint_name = os.path.basename(checkpoint_dir)
                run_name = f"{run_name} ({checkpoint_name})"

            results[run_name] = evaluate_run(run_dir, checkpoint_dir, test_data)
        except Exception as e:
            print(f"Error evaluating {run_spec}: {str(e)}")
            import traceback

            traceback.print_exc()

    # Print and save summary
    if results:
        # Print summary
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            print(f"  Accuracy:         {metrics['accuracy']:.2f}%")
            print(f"  Valid Rate:       {metrics['valid_rate']:.2f}%")
            print(f"  Evolution Score:  {metrics['evolution']:.1f}%")

        # Save detailed report
        report_path = os.path.join(OUTPUT_DIR, "evaluation_report.txt")
        save_evaluation_report(results, report_path)
        print(f"\nDetailed report saved to: {report_path}")
        
        # Print metric definitions
        print("\n" + "=" * 70)
        print("Metric Definitions:")
        print("- Evolution Score: Composite metric for warm-start quality")
        print("  Formula: 50% Valid Rate + 30% Diversity + 20% Balance")
        print("  Higher is better for evolutionary simulation starting point")
        print("- Valid Rate: Percentage of outputs that are valid actions")
        print("- Diversity: Percentage of unique valid actions used (0-100%, max=5 actions)")
        print("- Balance: How evenly distributed the actions are (entropy-based, 0-100%)")
        print("- Accuracy: Percentage matching expected actions (less relevant for evolution)")
        print(f"\nTest set size: {len(test_data)} examples")
        print("\nFor training metrics, see TRAINING_REPORT_FINAL.txt")

        # Add detailed predictions for best run if available
        best_run = next(iter(results.values()), None)
        if best_run and 'results' in best_run and best_run['results']:
            print("\n" + "=" * 70)
            print("DETAILED PREDICTIONS - Best Run")
            print("=" * 70 + "\n")
            
            for i, res in enumerate(best_run['results'][:5], 1):  # Show first 5 examples
                status = "✅" if res["predicted"] == res["true"] else "❌"
                print(f"{i:2d}. {status} Raw Output: '{res['raw_output']}'")
                print(f"       → Parsed as: '{res['predicted']}'  |  Expected: '{res['true']}'\n")
