import os
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, TrainingArguments, Trainer
from scipy.stats import spearmanr
from peft import LoraConfig, get_peft_model, TaskType

from dataset import SimplePlausibilityDataset
from model import create_model


# Model configurations
MODELS = {
    "deberta-base": "microsoft/deberta-v3-base",
    "deberta-large": "microsoft/deberta-v3-large",
    "roberta-base": "roberta-base",
    "roberta-large": "roberta-large",
    "modernbert-base": "answerdotai/ModernBERT-base",
    "modernbert-large": "answerdotai/ModernBERT-large",
}


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze() if predictions.ndim > 1 else predictions

    spearman, _ = spearmanr(predictions, labels)
    mae = np.mean(np.abs(predictions - labels))
    rmse = np.sqrt(np.mean((predictions - labels) ** 2))
    acc_05 = np.mean(np.abs(predictions - labels) <= 0.5)
    acc_10 = np.mean(np.abs(predictions - labels) <= 1.0)

    return {
        "spearman": spearman,
        "mae": mae,
        "rmse": rmse,
        "acc_0.5": acc_05,
        "acc_1.0": acc_10,
        "acc_std": 0.0,
    }


def apply_lora(model, model_name):
    if "deberta" in model_name:
        target_modules = ["query_proj", "key_proj", "value_proj"]
    else:
        target_modules = ["query", "key", "value"]

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=target_modules,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    return model


def train(
    model_name,
    train_file,
    dev_file,
    output_dir,
    epochs=5,
    batch_size=1,
    learning_rate=1e-4,
    freeze_transformer=False,
    use_lora=True,
):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Training {model_name}... Output dir: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(MODELS[model_name], use_fast=False)
    train_dataset = SimplePlausibilityDataset(train_file, tokenizer)
    eval_dataset = SimplePlausibilityDataset(dev_file, tokenizer)

    model = create_model(MODELS[model_name], freeze_transformer=freeze_transformer or use_lora)
    if use_lora:
        model = apply_lora(model, model_name)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="spearman",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()

    final_model_dir = os.path.join(output_dir, "final_model")
    os.makedirs(final_model_dir, exist_ok=True)

    print("Merging LoRA (if any) and saving FULL model object...")

    model_to_save = trainer.model
    if use_lora:
        model_to_save = model_to_save.merge_and_unload()

    # Save the full model object
    torch.save(model_to_save, os.path.join(final_model_dir, "full_model.pt"))

    # Also save tokenizer and config for reference
    model_to_save.config.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    print(f"Full model object saved to {final_model_dir}/full_model.pt")

    import json
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--dev_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze_transformer", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    args = parser.parse_args()

    train(
        args.model_name,
        args.train_file,
        args.dev_file,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        freeze_transformer=args.freeze_transformer,
        use_lora=args.use_lora,
    )
