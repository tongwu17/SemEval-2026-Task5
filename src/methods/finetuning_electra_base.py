import os
import sys
import json
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_utils import load_train_data, load_dev_data, load_test_data, save_predictions, get_sample_text
from src.config import DEV_PREDICTIONS_DIR, TEST_PREDICTIONS_DIR, evaluate_predictions

# Hugging Face imports
from transformers import (
    ElectraTokenizer,
    ElectraForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from torch.utils.data import Dataset

@dataclass
class ElectraConfig:
    """Configuration for ELECTRA fine-tuning"""
    model_name: str = "google/electra-base-discriminator"  # base + MPS for speed
    max_length: int = 256  # reduced for speed
    batch_size: int = 32  # larger batch for MPS efficiency
    learning_rate: float = 2e-5  # slightly higher for base
    num_epochs: int = 10  # Max epochs, early stopping will stop earlier
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1  # no accumulation needed with large batch
    fp16: bool = False  # MPS doesn't support fp16
    seed: int = 42
    early_stopping_patience: int = 3  # Stop if no improvement for 3 epochs
    output_dir: str = "./src/checkpoints/electra_base/train"
    
class PlausibilityDataset(Dataset):
    """Dataset for Word Sense Plausibility Rating"""
    
    def __init__(self, data: Dict, tokenizer: ElectraTokenizer, max_length: int = 512, 
                 is_test: bool = False):
        """
        Args:
            data: Dictionary of samples
            tokenizer: ELECTRA tokenizer
            max_length: Maximum sequence length
            is_test: Whether this is test data (no labels)
        """
        self.samples = list(data.items())
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample_id, sample = self.samples[idx]
        
        # Build input text with context
        # Format: [CLS] homonym: meaning [SEP] story context [SEP]
        story = get_sample_text(sample, include_ending=True)
        meaning_text = f"{sample['homonym']}: {sample['judged_meaning']}"
        
        # Include example sentence for better context
        example_sent = sample.get('example_sentence', '')
        if example_sent:
            meaning_text += f" (Example: {example_sent})"
        
        # Tokenize with pair format
        encoding = self.tokenizer(
            meaning_text,
            story,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'sample_id': sample_id
        }
        
        # Add labels for training/dev
        if not self.is_test:
            # Normalize label from 1-5 to 0-1 range for better training
            label = (sample['average'] - 1) / 4.0
            item['labels'] = torch.tensor(label, dtype=torch.float)
        
        return item


class TrainingLogger:
    """Logger for training process"""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.start_time = datetime.now()
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
    def log(self, message: str, print_to_console: bool = True):
        """Log message to file and optionally to console"""
        if print_to_console:
            print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')
    
    def log_header(self, config: dict, train_samples: int, val_samples: int, output_file: str, mode: str = "train"):
        """Log training configuration header"""
        self.log("=" * 60)
        self.log(f"Training Configuration - {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("=" * 60)
        self.log(f"Mode: {mode}")
        for key, value in config.items():
            if key not in ['output_dir', 'fp16']:
                self.log(f"{key}: {value}")
        self.log(f"Train samples: {train_samples}")
        self.log(f"Val samples: {val_samples}")
        self.log(f"Output: {output_file}")
        self.log(f"Log file: {self.log_file}")
        self.log("=" * 60 + "\n")


class DetailedLoggingCallback(TrainerCallback):
    """Custom callback for detailed epoch-by-epoch logging"""
    
    def __init__(self, logger: TrainingLogger, patience: int = 3):
        self.logger = logger
        self.patience = patience
        self.best_metric = None
        self.best_epoch = 0
        self.no_improvement_count = 0
        self.current_epoch = 0
        self.train_loss = 0
        self.last_logged_epoch = 0  # Track last logged epoch to avoid duplicates
        self.training_finished = False  # Track if training has ended
        
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Called when training logs are available"""
        if logs is None:
            return
        
        # Capture training loss
        if 'loss' in logs:
            self.train_loss = logs['loss']
    
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        """Called after evaluation at end of each epoch"""
        if metrics is None:
            return
        
        # Skip if training already finished (avoid duplicate logging from manual evaluate() call)
        if self.training_finished:
            return
            
        self.current_epoch = int(state.epoch)
        
        # Skip if already logged this epoch
        if self.current_epoch == self.last_logged_epoch and self.last_logged_epoch > 0:
            return
        self.last_logged_epoch = self.current_epoch
        
        # Extract metrics
        eval_loss = metrics.get('eval_loss', 0)
        eval_spearman = metrics.get('eval_spearman', 0)
        eval_rmse = metrics.get('eval_rmse', 0)
        eval_acc = metrics.get('eval_acc_within_1', 0)
        
        # Log epoch results
        self.logger.log(f"\nEpoch {self.current_epoch}:")
        self.logger.log(f"  Train Loss: {self.train_loss}")
        self.logger.log(f"  Val Loss: {eval_loss}")
        self.logger.log(f"  Val RMSE: {eval_rmse}")
        self.logger.log(f"  Val Spearman: {eval_spearman}")
        self.logger.log(f"  Val Acc within 1: {eval_acc}")
        
        # Check for improvement (using spearman as the metric)
        if self.best_metric is None or eval_spearman > self.best_metric:
            self.best_metric = eval_spearman
            self.best_epoch = self.current_epoch
            self.no_improvement_count = 0
            self.logger.log(f"  → Saved best model (Spearman: {eval_spearman})")
        else:
            self.no_improvement_count += 1
            self.logger.log(f"  No improvement ({self.no_improvement_count}/{self.patience})")
    
    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of training"""
        self.training_finished = True
        
        # Check if early stopping occurred
        if self.no_improvement_count >= self.patience:
            self.logger.log(f"\nEarly stopping at epoch {self.current_epoch}")
        
        self.logger.log(f"Best model at epoch {self.best_epoch} with Spearman: {self.best_metric}")
    

def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    
    # Convert back from 0-1 to 1-5 scale
    pred_scores = predictions * 4 + 1
    true_scores = labels * 4 + 1
    
    # Clip predictions to valid range
    pred_scores = np.clip(pred_scores, 1, 5)
    
    # Round to nearest integer for accuracy calculation
    pred_rounded = np.round(pred_scores)
    true_rounded = np.round(true_scores)
    
    # Compute metrics
    mse = mean_squared_error(true_scores, pred_scores)
    rmse = np.sqrt(mse)
    spearman, _ = spearmanr(pred_scores, true_scores)
    
    # Accuracy within 1 point (approximate)
    acc_within_1 = np.mean(np.abs(pred_rounded - true_rounded) <= 1)
    
    return {
        'rmse': rmse,
        'spearman': spearman,
        'acc_within_1': acc_within_1
    }


class ElectraPredictor:
    """ELECTRA-large based predictor for plausibility rating"""
    
    def __init__(self, config: Optional[ElectraConfig] = None):
        """
        Initialize ELECTRA predictor
        
        Args:
            config: Configuration object
        """
        self.config = config or ElectraConfig()
        
        # Device selection: CUDA > MPS (Apple Silicon) > CPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("[ELECTRA] Using Apple Silicon MPS acceleration!")
        else:
            self.device = torch.device('cpu')
        
        print(f"[ELECTRA] Initializing with model: {self.config.model_name}")
        print(f"[ELECTRA] Device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = ElectraTokenizer.from_pretrained(self.config.model_name)
        
        # Model will be loaded during training or prediction
        self.model = None
        self.trainer = None
        
    def _prepare_model(self):
        """Prepare model for training"""
        print(f"[ELECTRA] Loading model: {self.config.model_name}...")
        
        self.model = ElectraForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=1,  # Regression
            problem_type="regression"
        )
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[ELECTRA] Total parameters: {total_params:,}")
        print(f"[ELECTRA] Trainable parameters: {trainable_params:,}")
        
    def train(self, train_data: Dict, dev_data: Dict, logger: Optional[TrainingLogger] = None):
        """
        Fine-tune ELECTRA on training data
        
        Args:
            train_data: Training data dictionary
            dev_data: Development data dictionary
            logger: Optional logger
        """
        # Prepare model
        self._prepare_model()
        
        # Create datasets
        print("[ELECTRA] Creating datasets...")
        train_dataset = PlausibilityDataset(
            train_data, self.tokenizer, self.config.max_length, is_test=False
        )
        dev_dataset = PlausibilityDataset(
            dev_data, self.tokenizer, self.config.max_length, is_test=False
        )
        
        print(f"[ELECTRA] Training samples: {len(train_dataset)}")
        print(f"[ELECTRA] Dev samples: {len(dev_dataset)}")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size * 2,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            fp16=False,  # MPS doesn't support fp16
            bf16=False,  # MPS doesn't support bf16 either
            logging_dir='./logs',
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="spearman",
            greater_is_better=True,
            save_total_limit=2,
            seed=self.config.seed,
            report_to="none",  # Disable wandb/tensorboard
            remove_unused_columns=False,  # Keep sample_id
            save_safetensors=False,  # Use pytorch native format to avoid contiguity issues
            dataloader_num_workers=0,  # Avoid multiprocessing issues on Mac
            dataloader_pin_memory=False,  # Not supported on MPS
        )
        
        # Data collator to handle sample_id
        def data_collator(features):
            batch = {
                'input_ids': torch.stack([f['input_ids'] for f in features]),
                'attention_mask': torch.stack([f['attention_mask'] for f in features]),
            }
            if 'labels' in features[0]:
                batch['labels'] = torch.stack([f['labels'] for f in features])
            return batch
        
        # Create callbacks
        callbacks = [EarlyStoppingCallback(early_stopping_patience=self.config.early_stopping_patience)]
        if logger:
            callbacks.append(DetailedLoggingCallback(logger, patience=self.config.early_stopping_patience))
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            callbacks=callbacks
        )
        
        # Train
        print("\n[ELECTRA] Starting training...")
        if logger:
            logger.log("\n[ELECTRA] Starting training...")
            
        train_result = self.trainer.train()
        
        # Log results
        if logger:
            logger.log(f"Training loss: {train_result.training_loss}")
            
        # Get final eval results from trainer state (no need to re-evaluate)
        # The best model is already loaded due to load_best_model_at_end=True
        eval_results = self.trainer.evaluate()
        
        # Log final results without triggering callback again
        print(f"\n[ELECTRA] Final Val Spearman: {eval_results['eval_spearman']}")
        print(f"[ELECTRA] Final Val Acc within 1: {eval_results['eval_acc_within_1']}")
            
        return eval_results
    
    def predict(self, data: Dict, is_test: bool = False) -> List[Dict]:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a checkpoint.")
        
        # Create dataset
        dataset = PlausibilityDataset(
            data, self.tokenizer, self.config.max_length, is_test=is_test
        )
        
        # Get sample IDs
        sample_ids = [s[0] for s in dataset.samples]
        
        # Make predictions
        self.model.eval()
        all_preds = []
        
        with torch.no_grad():
            for i in tqdm(range(len(dataset)), desc="Predicting"):
                item = dataset[i]
                input_ids = item['input_ids'].unsqueeze(0).to(self.device)
                attention_mask = item['attention_mask'].unsqueeze(0).to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                pred = outputs.logits.squeeze().cpu().item()
                
                # Convert from 0-1 back to 1-5 scale
                pred_score = pred * 4 + 1
                pred_score = np.clip(pred_score, 1, 5)
                
                all_preds.append(pred_score)
        
        # Format predictions
        predictions = [
            {"id": sid, "prediction": round(pred)}
            for sid, pred in zip(sample_ids, all_preds)
        ]
        
        return predictions
    
    def save_model(self, path: str):
        """Save model to path"""
        if self.model is not None:
            os.makedirs(path, exist_ok=True)
            # Make all tensors contiguous before saving to avoid safetensors error
            state_dict = {k: v.contiguous() for k, v in self.model.state_dict().items()}
            torch.save(state_dict, os.path.join(path, 'pytorch_model.bin'))
            self.model.config.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            print(f"[ELECTRA] Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from path"""
        print(f"[ELECTRA] Loading model from {path}...")
        # Load config and create model
        self.model = ElectraForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=1,
            problem_type="regression"
        )
        # Load saved weights
        state_dict = torch.load(os.path.join(path, 'pytorch_model.bin'), map_location=self.device)
        self.model.load_state_dict(state_dict)
        # Load tokenizer from original model (checkpoint may not have tokenizer files)
        if os.path.exists(os.path.join(path, 'tokenizer_config.json')):
            self.tokenizer = ElectraTokenizer.from_pretrained(path)
        else:
            self.tokenizer = ElectraTokenizer.from_pretrained(self.config.model_name)
        self.model.to(self.device)
        print("[ELECTRA] Model loaded successfully!")


def main(val_split: float = 0.1):
    # Setup logger
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_log_finetuning_electra_base_dev_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    logger = TrainingLogger(log_file)
    
    # Use default configuration from ElectraConfig dataclass
    config = ElectraConfig()
    
    # Load data first to get counts for header
    print("[ELECTRA-Base] Loading data...")
    train_data = load_train_data()
    dev_data = load_dev_data()
    
    # Apply val_split if specified
    if val_split > 0:
        from sklearn.model_selection import train_test_split
        all_ids = list(train_data.keys())
        train_ids, val_ids = train_test_split(all_ids, test_size=val_split, random_state=42)
        actual_train_data = {k: train_data[k] for k in train_ids}
        val_data = {k: train_data[k] for k in val_ids}
        train_pct = int((1 - val_split) * 100)
        val_pct = int(val_split * 100)
        print(f"[ELECTRA-Base] Using {train_pct}/{val_pct} split from Train data")
        print(f"  - Training: {len(actual_train_data)} ({train_pct}%)")
        print(f"  - Validation (for early stopping): {len(val_data)} ({val_pct}%)")
    else:
        actual_train_data = train_data
        val_data = dev_data
        print(f"[ELECTRA-Base] Using Train 100% + Dev for validation")
    
    # Define output file
    pred_file = os.path.join(DEV_PREDICTIONS_DIR, "finetuning_electra_base_predictions.jsonl")
    
    # Log config
    logger.log_header(vars(config), len(actual_train_data), len(val_data), pred_file, mode="train")
    
    print(f"[ELECTRA-Base] Training samples: {len(actual_train_data)}")
    print(f"[ELECTRA-Base] Validation samples: {len(val_data)}")
    print(f"[ELECTRA-Base] Dev samples (for final eval): {len(dev_data)}")
    
    # Initialize predictor
    predictor = ElectraPredictor(config)
    
    # Train
    eval_results = predictor.train(actual_train_data, val_data, logger)
    
    # Make predictions on dev set
    print("\n[ELECTRA-Base] Making predictions on dev set...")
    
    dev_predictions = predictor.predict(dev_data, is_test=False)
    
    # Evaluate
    metrics = evaluate_predictions(dev_predictions, dev_data)
    print(f"\n[ELECTRA-Base] Final Dev Evaluation:")
    print(f"  Spearman: {metrics['spearman']}")
    print(f"  Acc: {metrics['acc_within_stdev']}")
    
    # Save predictions
    save_predictions(dev_predictions, pred_file)
    print(f"Predictions saved to: {pred_file}")
    
    # Write to evaluation_results.txt
    eval_results_file = os.path.join(os.path.dirname(__file__), '..', 'evaluation_results.txt')
    with open(eval_results_file, 'a') as f:
        result_line = (
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"finetuning_electra_base_predictions.jsonl | "
            f"Train Samples: {len(train_data)} | "
            f"Predict Samples (Dev Dataset): {len(dev_data)} | "
            f"Spearman: {metrics['spearman']} | "
            f"Acc: {metrics['acc_within_stdev']}\n"
        )
        f.write(result_line)
    print(f"[ELECTRA] Results appended to evaluation_results.txt")
    
    # Save model to train checkpoint directory
    model_save_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'electra_base', 'train', 'best_model')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    predictor.save_model(model_save_path)
    
    end_time = datetime.now()
    duration = end_time - logger.start_time
    logger.log("\n" + "=" * 60)
    logger.log("Training completed!")
    logger.log(f"Duration: {duration}")
    logger.log(f"Train Samples (Train Dataset): {len(train_data)}")
    logger.log(f"Predict Samples (Dev Dataset): {len(dev_data)}")
    logger.log(f"Mode: dev")
    logger.log(f"Output: {pred_file}")
    logger.log(f"Spearman: {metrics['spearman']}")
    logger.log(f"Acc: {metrics['acc_within_stdev']}")
    logger.log("=" * 60)
    
    print(f"\n[ELECTRA] Training complete! Log saved to: {log_file}")
    
    return metrics


def predict_test():
    """Make predictions on test set using saved model from train mode"""
    
    # Load model from train checkpoint
    model_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'electra_base', 'train', 'best_model')
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at {model_path}. Run --mode train first.")
        return
    
    # Load test data
    test_data = load_test_data()
    print(f"[ELECTRA] Test samples: {len(test_data)}")
    
    # Initialize and load model
    config = ElectraConfig()
    predictor = ElectraPredictor(config)
    predictor.load_model(model_path)
    
    # Make predictions
    print("[ELECTRA] Making predictions on test set...")
    test_predictions = predictor.predict(test_data, is_test=True)
    
    # Save predictions
    pred_file = os.path.join(TEST_PREDICTIONS_DIR, "finetuning_electra_base_predictions.jsonl")
    save_predictions(test_predictions, pred_file)
    
    print(f"[ELECTRA] Test predictions saved to: {pred_file}")


def train_dev_and_predict_test(val_split: float = 0.1): 
    # Setup logger
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_log_finetuning_electra_base_train_dev_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    logger = TrainingLogger(log_file)
    
    # Use default configuration
    config = ElectraConfig()
    # Update output dir for train_dev mode
    config.output_dir = "./src/checkpoints/electra_base/train_dev"
    
    # Load and combine train + dev data
    print("[ELECTRA-Base] Loading data...")
    train_data = load_train_data()
    dev_data = load_dev_data()
    test_data = load_test_data()
    
    # Rename dev IDs to avoid collision
    dev_data_renamed = {f"dev_{k}": v for k, v in dev_data.items()}
    full_data = {**train_data, **dev_data_renamed}
    
    # Split for training/validation (for early stopping)
    from sklearn.model_selection import train_test_split
    all_ids = list(full_data.keys())
    train_ids, val_ids = train_test_split(all_ids, test_size=val_split, random_state=42)
    
    full_train_data = {k: full_data[k] for k in train_ids}
    val_data = {k: full_data[k] for k in val_ids}
    
    train_pct = int((1 - val_split) * 100)
    val_pct = int(val_split * 100)
    
    print(f"[ELECTRA-Base] Total samples (train+dev): {len(full_data)}")
    print(f"  - Original Train: {len(train_data)}")
    print(f"  - Original Dev: {len(dev_data)}")
    print(f"[ELECTRA-Base] After {train_pct}/{val_pct} split:")
    print(f"  - Training: {len(full_train_data)} ({train_pct}%)")
    print(f"  - Validation (for early stopping): {len(val_data)} ({val_pct}%)")
    print(f"[ELECTRA-Base] Test samples: {len(test_data)}")
    
    # Define output file
    pred_file = os.path.join(TEST_PREDICTIONS_DIR, "finetuning_electra_base_predictions.jsonl")
    
    # Log config header
    logger.log("=" * 60)
    logger.log(f"Training Configuration - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log("=" * 60)
    logger.log(f"Mode: train_dev (train+dev combined, {train_pct}/{val_pct} split)")
    for key, value in vars(config).items():
        if key not in ['output_dir', 'fp16']:
            logger.log(f"{key}: {value}")
    logger.log(f"Total samples (train+dev): {len(full_data)}")
    logger.log(f"Training samples ({train_pct}%): {len(full_train_data)}")
    logger.log(f"Validation samples ({val_pct}%): {len(val_data)}")
    logger.log(f"Test samples: {len(test_data)}")
    logger.log(f"Output: {pred_file}")
    logger.log(f"Log file: {log_file}")
    logger.log("=" * 60 + "\n")
    
    # Initialize predictor
    predictor = ElectraPredictor(config)
    
    # Train with proper validation split for early stopping
    predictor.train(full_train_data, val_data, logger)
    
    # Make predictions on test set
    print("\n[ELECTRA-Base] Making predictions on test set...")
    test_predictions = predictor.predict(test_data, is_test=True)
    
    # Save predictions
    save_predictions(test_predictions, pred_file)
    print(f"Predictions saved to: {pred_file}")
    
    # Save model
    model_save_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'electra_base', 'train_dev', 'best_model')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    predictor.save_model(model_save_path)
    
    end_time = datetime.now()
    duration = end_time - logger.start_time
    logger.log("\n" + "=" * 60)
    logger.log("Training completed!")
    logger.log(f"Duration: {duration}")
    logger.log(f"Train Samples ({train_pct}% of train+dev): {len(full_train_data)}")
    logger.log(f"Validation Samples ({val_pct}%): {len(val_data)}")
    logger.log(f"Test Samples: {len(test_data)}")
    logger.log(f"Mode: train_dev")
    logger.log(f"Output: {pred_file}")
    logger.log("=" * 60)
    
    print(f"\n[ELECTRA] Training complete! Log saved to: {log_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ELECTRA-base Fine-tuning for Plausibility Rating")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "train_dev", "test"],
                       help="Mode: train (train only), train_dev (train+dev for test), test (predict on test)")
    parser.add_argument("--val_split", type=float, default=0.1,
                       help="Validation split ratio (default: 0.1). For train mode: 0=use Dev as val, >0=split from Train")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        main(val_split=args.val_split)
    elif args.mode == "train_dev":
        train_dev_and_predict_test(val_split=args.val_split)
    else:
        predict_test()
