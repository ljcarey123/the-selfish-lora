import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
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
    "num_train_epochs": 2,
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
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
}


def format_dataset(example, tokenizer):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predicted_ids = np.argmax(predictions, axis=-1)
    mask = labels != -100
    acc = accuracy_score(labels[mask], predicted_ids[mask])
    return {"accuracy": acc}


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
        eval_accuracies = [log["eval_accuracy"] for log in eval_logs]
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

        plt.figure(figsize=(10, 6))
        plt.plot(eval_epochs, eval_accuracies, marker="o", linestyle="-", color="green")
        plt.title("Validation Accuracy Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True)
        val_acc_path = os.path.join(output_dir, "validation_accuracy_curve.png")
        plt.savefig(val_acc_path)
        plt.close()
        print(f"Validation accuracy plot saved to {val_acc_path}")


if __name__ == "__main__":
    print("--- Starting Fine-Tuning Process ---")
    model, tokenizer = load_model_and_tokenizer()
    train_dataset = load_and_prepare_dataset(tokenizer, TRAIN_DATA_PATH)
    validation_dataset = load_and_prepare_dataset(tokenizer, VALIDATION_DATA_PATH)

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_arguments = TrainingArguments(**TRAINING_ARGS)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorForCompletionOnlyLM(
            # Tell the collator that the part to learn starts with the assistant's turn.
            # It will automatically mask everything before this.
            response_template="<|im_start|>assistant",
            tokenizer=tokenizer,
        ),
    )

    print("--- Starting Training ---")
    try:
        trainer.train()
        print("--- Training Finished ---")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback

        traceback.print_exc()

    print(f"Saving fine-tuned model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    print("Model saved successfully.")

    plot_training_results(trainer.state, OUTPUT_DIR)

    print("--- Fine-Tuning Process Complete ---")
