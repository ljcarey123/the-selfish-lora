import os
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import optuna
from datasets import Dataset

# --- CONFIGURATION ---
load_dotenv()
plt.switch_backend("Agg")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_DATA_PATH = os.path.join(SCRIPT_DIR, "data", "train_data.json")
VALIDATION_DATA_PATH = os.path.join(SCRIPT_DIR, "data", "validation_data.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output", "final_champion_model")

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

TRAINING_ARGS = {
    "output_dir": OUTPUT_DIR,
    "num_train_epochs": 1,  # Just need a warm-start for evolution, not perfect performance
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "optim": "adamw_torch",
    "save_steps": 50,
    "logging_steps": 1,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.03,
    "group_by_length": True,
    "lr_scheduler_type": "constant",
    "report_to": "none",
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",  # Save checkpoint at each epoch
    "load_best_model_at_end": True,  # Automatically load best checkpoint
    "metric_for_best_model": "eval_loss",  # Use eval_loss to determine best epoch
    "save_total_limit": 3,  # Keep only the 3 best checkpoints to save disk space
}

# Early stopping configuration
EARLY_STOPPING_PATIENCE = (
    1  # Stop if no improvement for 1 epoch (aggressive for small dataset)
)
EARLY_STOPPING_THRESHOLD = 0.0  # Minimum change to qualify as an improvement


def format_dataset(example, tokenizer):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}


def create_compute_metrics(tokenizer):
    """
    Factory function that creates a compute_metrics function with access to the tokenizer.
    This avoids using global variables.
    """

    def compute_metrics(eval_pred):
        """
        Computes action-level metrics by decoding predictions and checking against expected actions.

        Returns:
        - valid_action_rate: % of predictions that are valid actions (North/South/East/West/Gather)
        - correct_action_rate: % of predictions that match the expected action exactly
        """
        predictions, labels = eval_pred
        # predictions shape: (batch, seq_len, vocab_size) - take argmax over vocab dimension
        predicted_ids = np.argmax(predictions, axis=-1)

        # Valid actions
        valid_actions = {"North", "South", "East", "West", "Gather"}

        # Process each example in the batch
        num_valid = 0
        num_correct = 0
        total_examples = 0
        unique_predictions = set()  # Track unique predicted actions for diversity

        for i in range(len(predicted_ids)):
            # Get the predicted tokens for this example (filter out -100 padding)
            pred_tokens = predicted_ids[i]
            label_tokens = labels[i]

            # Find non-padding tokens in labels
            mask = label_tokens != -100
            if not mask.any():
                continue

            total_examples += 1

            # Decode predicted and expected tokens
            pred_text = tokenizer.decode(
                pred_tokens[mask], skip_special_tokens=True
            ).strip()
            label_text = tokenizer.decode(
                label_tokens[mask], skip_special_tokens=True
            ).strip()

            # Extract first word from prediction (the action)
            pred_action = pred_text.split()[0] if pred_text.split() else ""
            # Clean up punctuation
            pred_action = pred_action.strip(".,!?;:").strip()

            # Extract expected action
            expected_action = label_text.split()[0] if label_text.split() else ""
            expected_action = expected_action.strip(".,!?;:").strip()

            # Check if predicted action is valid
            if pred_action in valid_actions:
                num_valid += 1
                unique_predictions.add(pred_action)  # Track for diversity

            # Check if predicted action matches expected
            if pred_action == expected_action:
                num_correct += 1

        if total_examples == 0:
            return {
                "valid_action_rate": 0.0,
                "correct_action_rate": 0.0,
                "action_diversity": 0.0,
            }

        # Diversity: number of unique actions / 5 possible actions
        diversity_score = len(unique_predictions) / len(valid_actions)

        return {
            "valid_action_rate": num_valid / total_examples,
            "correct_action_rate": num_correct / total_examples,
            "action_diversity": diversity_score,  # 0.0 to 1.0 (1.0 = all 5 actions used)
        }

    return compute_metrics


def load_and_prepare_dataset(tokenizer, path):
    print(f"Loading dataset from {path}...")
    with open(path, "r") as f:
        data = json.load(f)
    formatted_data = [{"messages": conversation} for conversation in data]
    dataset = Dataset.from_list(formatted_data)
    dataset = dataset.map(lambda x: format_dataset(x, tokenizer))
    print(f"Dataset loaded and formatted successfully with {len(dataset)} examples.")
    return dataset


def load_model_and_tokenizer():
    print(f"Loading base model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)
    print("Base model and tokenizer loaded.")
    return model, tokenizer


def model_init():
    """Initializes a fresh model for each hyperparameter search trial."""
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, peft_config)
    return model


def save_training_report(
    trainer_state, output_dir, hyperparameters=None, trial_number=None
):
    """
    Generates a detailed text report of the training run.
    Includes hyperparameters, per-epoch metrics, and best epoch information.
    """
    logs = trainer_state.log_history
    eval_logs = [log for log in logs if "eval_loss" in log]

    # Determine report filename
    if trial_number is not None:
        report_filename = f"training_report_trial_{trial_number}.txt"
    else:
        report_filename = "training_report_final.txt"

    report_path = os.path.join(output_dir, report_filename)

    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("TRAINING REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Hyperparameters section
        if hyperparameters:
            f.write("HYPERPARAMETERS:\n")
            f.write("-" * 40 + "\n")
            for key, value in sorted(hyperparameters.items()):
                f.write(f"  {key}: {value}\n")
            f.write("\n")

        # Per-epoch metrics
        if eval_logs:
            f.write("PER-EPOCH METRICS:\n")
            f.write("-" * 100 + "\n")
            f.write(
                f"{'Epoch':<8} {'Eval Loss':<12} {'Valid %':<10} {'Correct %':<12} {'Diversity':<12} {'Runtime (s)':<12}\n"
            )
            f.write("-" * 100 + "\n")

            best_loss = float("inf")
            best_epoch = 0
            best_correct_rate = 0.0

            for log in eval_logs:
                epoch = log.get("epoch", "N/A")
                eval_loss = log.get("eval_loss", "N/A")
                valid_rate = log.get("eval_valid_action_rate", 0.0)
                correct_rate = log.get("eval_correct_action_rate", 0.0)
                diversity = log.get("eval_action_diversity", 0.0)
                runtime = log.get("eval_runtime", "N/A")

                f.write(
                    f"{epoch:<8} {eval_loss:<12.6f} {valid_rate*100:<10.2f} {correct_rate*100:<12.2f} {diversity:<12.3f} {runtime:<12.2f}\n"
                )

                if isinstance(eval_loss, float) and eval_loss < best_loss:
                    best_loss = eval_loss
                    best_epoch = epoch
                    best_correct_rate = correct_rate

            f.write("\n")
            f.write("BEST EPOCH (by eval_loss):\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Epoch: {best_epoch}\n")
            f.write(f"  Eval Loss: {best_loss:.6f}\n")
            f.write(f"  Correct Action Rate: {best_correct_rate*100:.2f}%\n")
            f.write("\n")

        # Training summary
        if trainer_state.best_metric is not None:
            f.write("TRAINING SUMMARY:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Best Metric (eval_loss): {trainer_state.best_metric:.6f}\n")
            f.write(f"  Best Model Checkpoint: {trainer_state.best_model_checkpoint}\n")
            f.write("\n")

        f.write("=" * 80 + "\n")

    print(f"Training report saved to: {report_path}")
    return report_path


def plot_training_results(trainer_state, output_dir):
    print("Generating and saving metrics plots...")
    logs = trainer_state.log_history
    train_logs = [log for log in logs if "loss" in log]
    eval_logs = [log for log in logs if "eval_loss" in log]

    if train_logs:
        train_epochs = [log["epoch"] for log in train_logs]
        train_losses = [log["loss"] for log in train_logs]
        plt.figure(figsize=(10, 6))
        plt.plot(train_epochs, train_losses, marker=".", linestyle="-", color="blue")
        plt.title("Training Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        train_loss_path = os.path.join(output_dir, "training_loss_curve.png")
        plt.savefig(train_loss_path)
        plt.close()
        print(f"Training loss plot saved to {train_loss_path}")

    if eval_logs:
        eval_epochs = [log["epoch"] for log in eval_logs]
        eval_losses = [log["eval_loss"] for log in eval_logs]

        # Plot validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(eval_epochs, eval_losses, marker=".", linestyle="-", color="orange")
        plt.title("Validation Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        val_loss_path = os.path.join(output_dir, "validation_loss_curve.png")
        plt.savefig(val_loss_path)
        plt.close()
        print(f"Validation loss plot saved to {val_loss_path}")

        # Plot action metrics if available
        if "eval_valid_action_rate" in eval_logs[0]:
            valid_rates = [
                log.get("eval_valid_action_rate", 0) * 100 for log in eval_logs
            ]
            correct_rates = [
                log.get("eval_correct_action_rate", 0) * 100 for log in eval_logs
            ]
            diversity_scores = [
                log.get("eval_action_diversity", 0) * 100 for log in eval_logs
            ]

            plt.figure(figsize=(10, 6))
            plt.plot(
                eval_epochs,
                valid_rates,
                marker="o",
                linestyle="-",
                color="green",
                label="Valid Action %",
            )
            plt.plot(
                eval_epochs,
                correct_rates,
                marker="s",
                linestyle="-",
                color="blue",
                label="Correct Action %",
            )
            plt.plot(
                eval_epochs,
                diversity_scores,
                marker="^",
                linestyle="--",
                color="purple",
                label="Diversity % (unique actions)",
            )
            plt.title("Action Prediction Metrics")
            plt.xlabel("Epoch")
            plt.ylabel("Percentage (%)")
            plt.legend()
            plt.grid(True)
            action_metrics_path = os.path.join(output_dir, "action_metrics_curve.png")
            plt.savefig(action_metrics_path)
            plt.close()
            print(f"Action metrics plot saved to {action_metrics_path}")


def hyperparameter_space(trial):
    """Defines the hyperparameter search space for optuna."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-4, log=True),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.03, 0.1),
        # Note: per_device_train_batch_size must be a power of 2 for efficiency
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [2, 4, 8]
        ),
    }


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Fine-tune a language model with optional hyperparameter search"
    )
    parser.add_argument(
        "--hyper", action="store_true", help="Enable hyperparameter search using Optuna"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=15,
        help="Number of hyperparameter search trials (default: 15)",
    )
    args = parser.parse_args()

    # Load tokenizer and datasets
    _, tokenizer = load_model_and_tokenizer()
    train_dataset = load_and_prepare_dataset(tokenizer, TRAIN_DATA_PATH)
    validation_dataset = load_and_prepare_dataset(tokenizer, VALIDATION_DATA_PATH)

    if args.hyper:
        # --- HYPERPARAMETER SEARCH MODE ---
        print("--- Starting Hyperparameter Search ---")
        training_args_for_search = TrainingArguments(**TRAINING_ARGS)

        # Create compute_metrics function with tokenizer
        compute_metrics_fn = create_compute_metrics(tokenizer)

        # Initialize a special trainer for hyperparameter search
        # Note: peft_config is applied inside model_init, not passed here
        search_trainer = SFTTrainer(
            model_init=model_init,  # The trainer will call this to get a fresh model for each trial
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            dataset_text_field="text",
            max_seq_length=512,
            tokenizer=tokenizer,
            args=training_args_for_search,
            packing=False,
            compute_metrics=compute_metrics_fn,
            data_collator=DataCollatorForCompletionOnlyLM(
                response_template="<|im_start|>assistant", tokenizer=tokenizer
            ),
        )

        # Define custom objective function to ensure we're minimizing eval_loss
        def compute_objective(metrics):
            """
            Extract the evaluation loss from metrics.
            This is what Optuna will try to minimize.
            """
            eval_loss = metrics.get("eval_loss", float("inf"))
            print(f"[DEBUG] Objective value (eval_loss): {eval_loss}")
            return eval_loss

        # Run the search
        best_run = search_trainer.hyperparameter_search(
            direction="minimize",
            backend="optuna",
            hp_space=hyperparameter_space,
            n_trials=args.trials,
            compute_objective=compute_objective,
        )

        print("--- Hyperparameter Search Complete ---")
        print(f"Best run found: {best_run}")

        # Update training arguments with the best hyperparameters
        final_training_args = TrainingArguments(**TRAINING_ARGS)
        final_training_args.learning_rate = best_run.hyperparameters["learning_rate"]
        # num_train_epochs is not in search space, use default from TRAINING_ARGS
        final_training_args.warmup_ratio = best_run.hyperparameters["warmup_ratio"]
        final_training_args.per_device_train_batch_size = best_run.hyperparameters[
            "per_device_train_batch_size"
        ]

        print(f"\n--- Training Final Model with Best Hyperparameters ---")
        print(f"Learning Rate: {final_training_args.learning_rate}")
        print(f"Epochs: {final_training_args.num_train_epochs}")
        print(f"Warmup Ratio: {final_training_args.warmup_ratio}")
        print(f"Batch Size: {final_training_args.per_device_train_batch_size}")
    else:
        # --- NORMAL TRAINING MODE ---
        print("--- Starting Normal Training (no hyperparameter search) ---")
        print("Tip: Use --hyper flag to enable hyperparameter search")
        final_training_args = TrainingArguments(**TRAINING_ARGS)
        print(f"Learning Rate: {final_training_args.learning_rate}")
        print(f"Epochs: {final_training_args.num_train_epochs}")

    # Load a fresh model for final training
    final_model = model_init()

    # Create compute_metrics function with tokenizer in closure
    compute_metrics_fn = create_compute_metrics(tokenizer)

    # Train the model (no early stopping needed for 1 epoch)
    final_trainer = SFTTrainer(
        model=final_model,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=final_training_args,
        packing=False,
        compute_metrics=compute_metrics_fn,
        data_collator=DataCollatorForCompletionOnlyLM(
            response_template="<|im_start|>assistant", tokenizer=tokenizer
        ),
    )

    final_trainer.train()

    print("--- Training Finished ---")
    final_trainer.save_model(OUTPUT_DIR)
    print(f"Model saved to: {OUTPUT_DIR}")

    # Generate training plots
    plot_training_results(final_trainer.state, OUTPUT_DIR)

    # Generate detailed training report
    hyperparams_used = {
        "learning_rate": final_training_args.learning_rate,
        "num_train_epochs": final_training_args.num_train_epochs,
        "warmup_ratio": final_training_args.warmup_ratio,
        "per_device_train_batch_size": final_training_args.per_device_train_batch_size,
        "gradient_accumulation_steps": final_training_args.gradient_accumulation_steps,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
    }
    save_training_report(
        final_trainer.state, OUTPUT_DIR, hyperparameters=hyperparams_used
    )

    print("--- Fine-Tuning Process Complete ---")
