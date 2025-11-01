import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset
import optuna
import time
import torch
from transformers import BitsAndBytesConfig

# --- CONFIGURATION ---
load_dotenv()
plt.switch_backend("Agg")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_DATA_PATH = os.path.join(SCRIPT_DIR, "data", "train.jsonl")
VALIDATION_DATA_PATH = os.path.join(SCRIPT_DIR, "data", "validation.jsonl")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

STUDY_NAME = "sft-search"
STORAGE_URL = "sqlite:///sft_study.db"

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
    # Core training settings
    "output_dir": OUTPUT_DIR,
    "num_train_epochs": 2,  # More epochs to allow learning, with early stopping
    "per_device_train_batch_size": 4,  # Batch size per GPU/CPU
    "gradient_accumulation_steps": 4,  # Number of steps to accumulate gradients before updating weights
    # Memory optimization settings
    "fp16": True,  # Enable mixed precision training (faster, less memory)
    # Optimizer settings
    "optim": "adamw_torch",
    "learning_rate": 2e-5,  # Will be tuned by hyperparameter search
    "weight_decay": 0.01,  # L2 regularization prevents overfitting
    "warmup_ratio": 0.03,  # Will be tuned by hyperparameter search
    # Regularization to prevent output collapse
    "label_smoothing_factor": 0.05,  # Reduced - was preventing confident predictions
    # Evaluation and saving
    "evaluation_strategy": "epoch",  # Evaluate at end of epoch
    "save_strategy": "epoch",  # Save at end of epoch
    "save_total_limit": 1,  # Keep only the best checkpoint
    "load_best_model_at_end": True,  # Load best checkpoint (only one anyway)
    "metric_for_best_model": "eval_loss",
    # Logging and misc
    "logging_steps": 10,  # Log every 10 steps
    "report_to": "none",  # Don't send to wandb/tensorboard
}

# Early stopping configuration
EARLY_STOPPING_PATIENCE = 2  # Stop if no improvement for 2 epochs
EARLY_STOPPING_THRESHOLD = 0.01  # Minimum change to qualify as an improvement


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
        data = [json.loads(line) for line in f]
    formatted_data = [{"messages": conversation} for conversation in data]
    dataset = Dataset.from_list(formatted_data)
    dataset = dataset.map(lambda x: format_dataset(x, tokenizer))
    print(f"Dataset loaded and formatted successfully with {len(dataset)} examples.")
    return dataset


def load_tokenizer():
    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("Tokenizer loaded.")
    return tokenizer


def model_init(trial=None, use_cpu=False):
    """Initializes a fresh model for each hyperparameter search trial."""
    if trial is not None:
        # Create output directory for this trial
        output_dir = os.path.join(OUTPUT_DIR, f"run-{trial.number}")
        os.makedirs(output_dir, exist_ok=True)
        trial.set_user_attr("output_dir", output_dir)
    else:
        # Normal training mode
        output_dir = OUTPUT_DIR

    # Set up device map and quantization config
    device_map = "cpu" if use_cpu else "auto"
    
    if use_cpu:
        # For CPU, use full precision
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype=torch.float32
        )
    else:
        # For GPU, use 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load model with device_map='auto' and no additional device mapping
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            device_map=device_map,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
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
    if hyperparameters and "run_name" in hyperparameters:
        report_filename = f"training_report_{hyperparameters['run_name']}.txt"
    elif trial_number is not None:
        report_filename = f"training_report_trial_{trial_number}.txt"
    else:
        report_filename = "training_report_final.txt"

    report_path = os.path.join(output_dir, report_filename)

    with open(report_path, "w", encoding="utf-8") as f:
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
    # Create output directory for this trial
    output_dir = os.path.join(OUTPUT_DIR, f"run-{trial.number}")
    os.makedirs(output_dir, exist_ok=True)

    # Store output_dir in trial user attributes
    trial.set_user_attr("output_dir", output_dir)

    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.01, 0.1),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [1, 2, 4]
        ),
        "weight_decay": trial.suggest_float("weight_decay", 0.001, 0.1, log=True),
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
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU training even if GPU is available",
    )
    args = parser.parse_args()

    train_path = TRAIN_DATA_PATH
    val_path = VALIDATION_DATA_PATH

    # Load tokenizer and datasets
    tokenizer = load_tokenizer()
    train_dataset = load_and_prepare_dataset(tokenizer, train_path)
    validation_dataset = load_and_prepare_dataset(tokenizer, val_path)

    # Update training args based on CPU/GPU mode
    training_args = TRAINING_ARGS.copy()
    if args.cpu:
        training_args["use_cpu"] = True
        print("--- Using CPU for training ---")
    else:
        print("--- Using GPU for training (if available) ---")

    if args.hyper:
        # --- HYPERPARAMETER SEARCH MODE ---
        print("--- Starting Hyperparameter Search ---")
        training_args_for_search = TrainingArguments(**training_args)

        # Create compute_metrics function with tokenizer
        compute_metrics_fn = create_compute_metrics(tokenizer)

        # Initialize a special trainer for hyperparameter search
        # Note: peft_config is applied inside model_init, not passed here
        search_trainer = SFTTrainer(
            model_init=lambda: model_init(
                use_cpu=args.cpu
            ),  # Pass CPU flag to model_init
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            dataset_text_field="text",
            max_seq_length=512,
            tokenizer=tokenizer,
            args=training_args_for_search,
            packing=False,
            compute_metrics=compute_metrics_fn,
            data_collator=DataCollatorForCompletionOnlyLM(
                response_template="assistant", tokenizer=tokenizer
            ),
        )

        # Generate unique study name with timestamp
        unique_study_name = f"{STUDY_NAME}-{int(time.time())}"

        # Run the search
        best_run = search_trainer.hyperparameter_search(
            study_name=unique_study_name,
            storage=STORAGE_URL,
            backend="optuna",
            hp_space=hyperparameter_space,
            n_trials=args.trials,
            compute_objective=lambda metrics: metrics.get("eval_loss", float("inf")),
        )

        study = optuna.load_study(
            study_name=unique_study_name,
            storage=STORAGE_URL,
        )
        trials = study.trials

        print("--- Hyperparameter Search Complete ---")
        print(f"Best run found: {best_run}")
        print(f"\nBest Hyperparameters:")
        print(f"  Learning Rate: {best_run.hyperparameters['learning_rate']}")
        print(f"  Warmup Ratio: {best_run.hyperparameters['warmup_ratio']}")
        print(
            f"  Batch Size: {best_run.hyperparameters['per_device_train_batch_size']}"
        )
        print(f"  Weight Decay: {best_run.hyperparameters['weight_decay']}")
        print(f"  Objective (eval_loss): {best_run.objective}")

        # Generate plots and reports for each run in the current search
        print("\n--- Generating Reports for Current Runs ---")
        all_run_metrics = []

        for i, run in enumerate(trials):
            run_dir = run.user_attrs["output_dir"]
            run_name = os.path.basename(run_dir)

            # Find the final checkpoint in this run's directory
            checkpoints = []
            for item in os.listdir(run_dir):
                if item.startswith("checkpoint-"):
                    try:
                        checkpoint_num = int(item.split("-")[1])
                        checkpoints.append((checkpoint_num, item))
                    except (IndexError, ValueError):
                        pass

            if not checkpoints:
                continue

            checkpoints.sort(key=lambda x: x[0])
            final_checkpoint = checkpoints[-1][1]
            checkpoint_path = os.path.join(run_dir, final_checkpoint)

            # Load trainer state
            state_file = os.path.join(checkpoint_path, "trainer_state.json")
            if not os.path.exists(state_file):
                continue

            with open(state_file, "r") as f:
                state_dict = json.load(f)

            # Create a minimal TrainerState object
            from transformers.trainer_callback import TrainerState

            state = TrainerState(**state_dict)

            # Generate plots
            plot_training_results(state, checkpoint_path)

            # Generate report
            run_hyperparams = {
                "run_name": run_name,
                "checkpoint": final_checkpoint,
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "lora_dropout": LORA_DROPOUT,
                "learning_rate": run.params["learning_rate"],
                "warmup_ratio": run.params["warmup_ratio"],
                "per_device_train_batch_size": run.params[
                    "per_device_train_batch_size"
                ],
                "weight_decay": run.params["weight_decay"],
            }
            save_training_report(
                state, checkpoint_path, hyperparameters=run_hyperparams
            )
            print(f"  Generated report for {run_name}/{final_checkpoint}")

            # Collect metrics for final report
            log_history = state_dict.get("log_history", [])
            eval_logs = [log for log in log_history if "eval_loss" in log]
            if eval_logs:
                final_eval = eval_logs[-1]
                all_run_metrics.append(
                    {
                        "run_name": run_name,
                        "eval_loss": final_eval.get("eval_loss", float("inf")),
                        "valid_action_rate": final_eval.get(
                            "eval_valid_action_rate", 0.0
                        ),
                        "correct_action_rate": final_eval.get(
                            "eval_correct_action_rate", 0.0
                        ),
                        "action_diversity": final_eval.get(
                            "eval_action_diversity", 0.0
                        ),
                    }
                )

        # Generate comprehensive final report
        final_report_path = os.path.join(OUTPUT_DIR, "TRAINING_REPORT_FINAL.txt")
        with open(final_report_path, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("FINAL TRAINING REPORT - HYPERPARAMETER SEARCH\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Number of Trials: {args.trials}\n")
            f.write(f"Best Run: {best_run.run_id}\n")
            f.write(f"Best Objective (eval_loss): {best_run.objective:.4f}\n\n")

            f.write("Best Hyperparameters:\n")
            f.write("-" * 70 + "\n")
            f.write(
                f"  Learning Rate: {best_run.hyperparameters['learning_rate']:.2e}\n"
            )
            f.write(f"  Warmup Ratio: {best_run.hyperparameters['warmup_ratio']:.4f}\n")
            f.write(
                f"  Batch Size: {best_run.hyperparameters['per_device_train_batch_size']}\n"
            )
            f.write(
                f"  Weight Decay: {best_run.hyperparameters['weight_decay']:.4f}\n\n"
            )

            f.write("All Runs Performance:\n")
            f.write("=" * 70 + "\n")
            f.write(
                f"{'Run':<10} {'Eval Loss':<12} {'Valid %':<10} {'Correct %':<12} {'Diversity':<10}\n"
            )
            f.write("-" * 70 + "\n")

            # Sort by eval_loss (best first)
            all_run_metrics.sort(key=lambda x: x["eval_loss"])

            for metrics in all_run_metrics:
                f.write(f"{metrics['run_name']:<10} ")
                f.write(f"{metrics['eval_loss']:<12.4f} ")
                f.write(f"{metrics['valid_action_rate']*100:<10.1f} ")
                f.write(f"{metrics['correct_action_rate']*100:<12.1f} ")
                f.write(f"{metrics['action_diversity']:<10.2f}\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write("\nNotes:\n")
            f.write("- Eval Loss: Lower is better\n")
            f.write("- Valid %: Percentage of outputs that are valid actions\n")
            f.write("- Correct %: Percentage of outputs that match expected action\n")
            f.write("- Diversity: Number of unique actions produced (max 5)\n")
            f.write("\nEach run's detailed report: training_report_run-X.txt\n")
            f.write("Use evaluate_model.py for test set evaluation.\n")

        print(f"\nTraining report saved to: {final_report_path}")
        print("\n--- Fine-Tuning Process Complete ---")
        print(
            f"All {len(trials)} runs have been saved. Use evaluate_model.py to compare them."
        )

    else:
        # --- NORMAL TRAINING MODE ---
        print("--- Starting Normal Training (no hyperparameter search) ---")
        print("Tip: Use --hyper flag to enable hyperparameter search")
        final_training_args = TrainingArguments(**training_args)
        print(f"Learning Rate: {final_training_args.learning_rate}")
        print(f"Epochs: {final_training_args.num_train_epochs}")

        # Load a fresh model for final training
        final_model = model_init(use_cpu=args.cpu)

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
                response_template="assistant", tokenizer=tokenizer
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
