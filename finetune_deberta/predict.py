import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
from dataset import SimplePlausibilityDataset
from model import PlausibilityRegressionModel, PlausibilityConfig

# =========================================================
# Load model + tokenizer using training config
# =========================================================
def load_model_and_tokenizer(experiment_dir, device):
    model_dir = os.path.join(experiment_dir, "final_model_merged")
    params_path = os.path.join(experiment_dir, "params.json")

    if not os.path.exists(model_dir):
        raise ValueError(f"No merged model directory found at {model_dir}")
    if not os.path.exists(params_path):
        raise ValueError(f"No params.json found at {params_path}")

    # Load training parameters (for reference)
    with open(params_path, "r") as f:
        params = json.load(f)

    print(f"Loading model from: {model_dir}")
    print(f"Pooling method: {params.get('pooling', 'cls')}")
    print(f"Loss type: {params.get('loss_type', 'huber')}")

    # Load tokenizer from merged model directory
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)

    # Load the model directly using from_pretrained
    # This will automatically load the config and weights
    model = PlausibilityRegressionModel.from_pretrained(model_dir)
    
    print("Successfully loaded model and tokenizer")

    model.to(device)
    model.eval()
    return model, tokenizer

# =========================================================
# Prediction
# =========================================================
def predict(model, dataloader, device):
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            all_predictions.extend(logits.squeeze(-1).cpu().tolist())

    return all_predictions

# =========================================================
# Save predictions
# =========================================================
def save_jsonl(ids, predictions, output_file):
    with open(output_file, "w") as f:
        for sample_id, score in zip(ids, predictions):
            f.write(json.dumps({"id": sample_id, "score": float(score)}) + "\n")
    print(f"\nSaved predictions to {output_file}")

# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_file", type=str, default="predictions.jsonl")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Load model & tokenizer
    model, tokenizer = load_model_and_tokenizer(args.experiment_dir, device)

    # Load test data
    with open(args.test_file, "r") as f:
        test_data = json.load(f)
    ids = [sample["id"] for sample in test_data]

    # Dataset & DataLoader
    test_dataset = SimplePlausibilityDataset(
        args.test_file, tokenizer, max_length=args.max_length, is_test=True
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Predictions
    predictions = predict(model, test_loader, device)

    # Save
    save_jsonl(ids, predictions, args.output_file)

if __name__ == "__main__":
    main()