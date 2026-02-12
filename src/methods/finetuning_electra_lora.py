import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from tqdm import tqdm
from scipy.stats import spearmanr

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_utils import load_train_data, load_dev_data, load_test_data, save_predictions, get_sample_text
from src.config import DEV_PREDICTIONS_DIR, TEST_PREDICTIONS_DIR, evaluate_predictions

# Hugging Face imports
from transformers import (
    ElectraModel,
    ElectraTokenizer,
    ElectraPreTrainedModel,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.utils.data import Dataset

# PEFT (Parameter-Efficient Fine-Tuning) for LoRA
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)

@dataclass
class ElectraLoraConfig:
    """Configuration for ELECTRA + LoRA fine-tuning"""
    # Model config
    model_name: str = "google/electra-large-discriminator"  # 335M params
    max_length: int = 256
    
    # Training config
    batch_size: int = 16 
    learning_rate: float = 1e-4  
    num_epochs: int = 10  
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 2  # Effective batch = 32
    seed: int = 42
    early_stopping_patience: int = 3
    
    lora_r: int = 8  
    lora_alpha: int = 32  
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "query", "key", "value", "dense"  # Attention layers + FFN
    ])
    
    # Loss config
    huber_delta: float = 1.0  # Huber loss delta
    
    # Pooling
    pooling_strategy: str = "mean"  # "mean" or "cls"
    
    # Output
    output_dir: str = "./src/checkpoints/electra_lora/train"


class ElectraForRegressionWithMeanPooling(ElectraPreTrainedModel):
    """
    ELECTRA model with Mean Pooling head for regression
    
    Mean pooling aggregates all token representations, not just [CLS]
    This gives better sentence-level representations
    """
    
    def __init__(self, config, pooling_strategy: str = "mean", huber_delta: float = 1.0):
        super().__init__(config)
        self.electra = ElectraModel(config)
        self.pooling_strategy = pooling_strategy
        self.huber_delta = huber_delta
        
        # Regression head with dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.regressor = nn.Linear(config.hidden_size, 1)
        
        # Huber loss
        self.huber_loss = nn.HuberLoss(delta=huber_delta, reduction='mean')
        
        # Initialize weights
        self.post_init()
    
    def mean_pooling(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Mean pooling over token representations
        
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: (batch, seq_len)
            
        Returns:
            pooled: (batch, hidden_size)
        """
        # Expand attention mask to hidden size
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        
        # Sum token embeddings (masked)
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        
        # Sum mask (for averaging)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        
        # Mean
        return sum_embeddings / sum_mask
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> SequenceClassifierOutput:
        
        outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)
        
        # Pooling
        if self.pooling_strategy == "mean":
            pooled = self.mean_pooling(hidden_states, attention_mask)
        else:  # CLS
            pooled = hidden_states[:, 0, :]
        
        # Regression head
        pooled = self.dropout(pooled)
        logits = self.regressor(pooled)  # (batch, 1)
        
        # Compute loss
        loss = None
        if labels is not None:
            # Huber loss (more robust than MSE)
            loss = self.huber_loss(logits.squeeze(-1), labels)
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )


class PlausibilityDataset(Dataset):
    """Dataset for Word Sense Plausibility Rating"""
    
    def __init__(self, data: Dict, tokenizer: ElectraTokenizer, max_length: int = 256, 
                 is_test: bool = False):
        self.samples = list(data.items())
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample_id, sample = self.samples[idx]
        
        # Build input text with context
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
        
        # Add labels for training/dev (normalize 1-5 to 0-1)
        if not self.is_test:
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
        if print_to_console:
            print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')
    
    def log_header(self, config: dict, train_samples: int, val_samples: int, output_file: str, mode: str = "train"):
        self.log("=" * 60)
        self.log(f"Training Configuration - {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("=" * 60)
        self.log(f"Mode: {mode}")
        for key, value in config.items():
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
        
        # Round epoch to nearest integer (state.epoch can be 0.99, 1.99, etc.)
        self.current_epoch = round(state.epoch)
        
        # Skip if already logged this epoch
        if self.current_epoch == self.last_logged_epoch:
            return
        self.last_logged_epoch = self.current_epoch
        
        # Extract metrics
        eval_loss = metrics.get('eval_loss', 0)
        eval_spearman = metrics.get('eval_spearman', 0)
        
        # Log epoch results
        self.logger.log(f"\nEpoch {self.current_epoch}:")
        self.logger.log(f"  Train Loss: {self.train_loss}")
        self.logger.log(f"  Val Loss: {eval_loss}")
        self.logger.log(f"  Val Spearman: {eval_spearman}")
        
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
    """Compute evaluation metrics (official metrics only)"""
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    
    # Convert back from 0-1 to 1-5 scale
    pred_scores = predictions * 4 + 1
    true_scores = labels * 4 + 1
    
    # Clip predictions to valid range
    pred_scores = np.clip(pred_scores, 1, 5)
    
    # Spearman correlation (official metric)
    spearman, _ = spearmanr(pred_scores, true_scores)
    
    return {
        'spearman': spearman,
    }


class ElectraLoraPredictor:
    """ELECTRA + LoRA predictor for plausibility rating"""
    
    def __init__(self, config: Optional[ElectraLoraConfig] = None):
        self.config = config or ElectraLoraConfig()
        
        # Device selection
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("[ELECTRA-LoRA] Using Apple Silicon MPS acceleration!")
        else:
            self.device = torch.device('cpu')
        
        print(f"[ELECTRA-LoRA] Model: {self.config.model_name}")
        print(f"[ELECTRA-LoRA] Device: {self.device}")
        print(f"[ELECTRA-LoRA] Pooling: {self.config.pooling_strategy}")
        print(f"[ELECTRA-LoRA] LoRA r={self.config.lora_r}, alpha={self.config.lora_alpha}")
        
        # Load tokenizer
        self.tokenizer = ElectraTokenizer.from_pretrained(self.config.model_name)
        
        self.model = None
        self.trainer = None
        
    def _prepare_model(self):
        """Prepare model with LoRA"""
        print(f"[ELECTRA-LoRA] Loading base model: {self.config.model_name}...")
        
        # Load base model with mean pooling
        base_model = ElectraForRegressionWithMeanPooling.from_pretrained(
            self.config.model_name,
            pooling_strategy=self.config.pooling_strategy,
            huber_delta=self.config.huber_delta
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            inference_mode=False
        )
        
        # Apply LoRA
        print("[ELECTRA-LoRA] Applying LoRA...")
        self.model = get_peft_model(base_model, lora_config)
        self.model.to(self.device)
        
        # Print parameter stats
        self.model.print_trainable_parameters()
        
    def train(self, train_data: Dict, dev_data: Dict, logger: Optional[TrainingLogger] = None, resume_checkpoint: Optional[str] = None):
        """Fine-tune ELECTRA+LoRA on training data"""
        
        # Prepare model with LoRA
        self._prepare_model()
        
        # Create datasets
        print("[ELECTRA-LoRA] Creating datasets...")
        train_dataset = PlausibilityDataset(
            train_data, self.tokenizer, self.config.max_length, is_test=False
        )
        dev_dataset = PlausibilityDataset(
            dev_data, self.tokenizer, self.config.max_length, is_test=False
        )
        
        print(f"[ELECTRA-LoRA] Training samples: {len(train_dataset)}")
        print(f"[ELECTRA-LoRA] Dev samples: {len(dev_dataset)}")
        
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
            bf16=False,
            logging_dir='./logs',
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="spearman",
            greater_is_better=True,
            save_total_limit=2,
            seed=self.config.seed,
            report_to="none",
            remove_unused_columns=False,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
        )
        
        # Data collator
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
        print("\n[ELECTRA-LoRA] Starting training...")
        if logger:
            logger.log("\n[ELECTRA-LoRA] Starting training...")
            
        # Resume training from checkpoint if provided
        if resume_checkpoint:
            train_result = self.trainer.train(resume_from_checkpoint=resume_checkpoint)
        else:
            train_result = self.trainer.train()
        
        if logger:
            logger.log(f"Training loss: {train_result.training_loss}")
            
        # Evaluate on dev
        print("\n[ELECTRA-LoRA] Evaluating on dev set...")
        eval_results = self.trainer.evaluate()
        
        print(f"[ELECTRA-LoRA] Dev Spearman: {eval_results['eval_spearman']}")
        
        if logger:
            logger.log(f"Dev Spearman: {eval_results['eval_spearman']}")
            
        return eval_results
    
    def predict(self, data: Dict, is_test: bool = False) -> List[Dict]:
        """Make predictions on data"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a checkpoint.")
        
        dataset = PlausibilityDataset(
            data, self.tokenizer, self.config.max_length, is_test=is_test
        )
        
        sample_ids = [s[0] for s in dataset.samples]
        
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
        
        predictions = [
            {"id": sid, "prediction": round(pred)}
            for sid, pred in zip(sample_ids, all_preds)
        ]
        
        return predictions
    
    def save_model(self, path: str):
        """Save LoRA model"""
        if self.model is not None:
            os.makedirs(path, exist_ok=True)
            # Save only LoRA weights (much smaller!)
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            print(f"[ELECTRA-LoRA] LoRA adapter saved to {path}")
    
    def load_model(self, path: str):
        """Load LoRA model"""
        print(f"[ELECTRA-LoRA] Loading model from {path}...")
        
        # Load base model
        base_model = ElectraForRegressionWithMeanPooling.from_pretrained(
            self.config.model_name,
            pooling_strategy=self.config.pooling_strategy,
            huber_delta=self.config.huber_delta
        )
        
        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(base_model, path)
        self.tokenizer = ElectraTokenizer.from_pretrained(path)
        self.model.to(self.device)
        print("[ELECTRA-LoRA] Model loaded successfully!")


def main(val_split: float = 0.1):
    # Setup logger
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_log_finetuning_electra_lora_dev_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    logger = TrainingLogger(log_file)
    
    # Use default configuration
    config = ElectraLoraConfig()
    
    # Load data first to get counts for header
    print("[ELECTRA-LoRA] Loading data...")
    train_data = load_train_data()
    dev_data = load_dev_data()
    
    # Apply val_split: split Train into train/val
    from sklearn.model_selection import train_test_split
    all_ids = list(train_data.keys())
    train_ids, val_ids = train_test_split(all_ids, test_size=val_split, random_state=42)
    actual_train_data = {k: train_data[k] for k in train_ids}
    val_data = {k: train_data[k] for k in val_ids}
    train_pct = int((1 - val_split) * 100)
    val_pct = int(val_split * 100)
    print(f"[ELECTRA-LoRA] Using {train_pct}/{val_pct} split from Train data")
    print(f"  - Training: {len(actual_train_data)} ({train_pct}%)")
    print(f"  - Validation (for early stopping): {len(val_data)} ({val_pct}%)")
    
    # Define output file
    pred_file = os.path.join(DEV_PREDICTIONS_DIR, "finetuning_electra_large_predictions.jsonl")
    
    # Log config
    logger.log_header({
        'Model': config.model_name,
        'pooling_strategy': config.pooling_strategy,
        'huber_delta': config.huber_delta,
        'lora_r': config.lora_r,
        'lora_alpha': config.lora_alpha,
        'lora_dropout': config.lora_dropout,
        'learning_rate': config.learning_rate,
        'batch_size': config.batch_size,
        'gradient_accumulation_steps': config.gradient_accumulation_steps,
        'effective_batch_size': config.batch_size * config.gradient_accumulation_steps,
        'num_epochs': config.num_epochs,
        'early_stopping_patience': config.early_stopping_patience,
        'val_split': val_split,
    }, len(actual_train_data), len(val_data), pred_file)
    
    print(f"[ELECTRA-LoRA] Training samples: {len(actual_train_data)}")
    print(f"[ELECTRA-LoRA] Validation samples: {len(val_data)}")
    print(f"[ELECTRA-LoRA] Dev samples (for final eval): {len(dev_data)}")
    
    # Initialize predictor
    predictor = ElectraLoraPredictor(config)
    
    # Train (support resume from checkpoint via env var ELECTRA_LORA_RESUME)
    resume_checkpoint = os.environ.get('ELECTRA_LORA_RESUME') if 'ELECTRA_LORA_RESUME' in os.environ else None
    eval_results = predictor.train(actual_train_data, val_data, logger, resume_checkpoint=resume_checkpoint)
    
    # Make predictions on dev set (using full dev_data for final evaluation)
    print("\n[ELECTRA-LoRA] Making predictions on dev set...")
    
    dev_predictions = predictor.predict(dev_data, is_test=False)
    
    # Evaluate
    metrics = evaluate_predictions(dev_predictions, dev_data)
    print(f"\n[ELECTRA-LoRA] Final Dev Evaluation:")
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
            f"finetuning_electra_large_predictions.jsonl | "
            f"Train Samples: {len(train_data)} | "
            f"Predict Samples (Dev Dataset): {len(dev_data)} | "
            f"Pooling: {config.pooling_strategy} | "
            f"LoRA r={config.lora_r} alpha={config.lora_alpha} | "
            f"Huber delta={config.huber_delta} | "
            f"Spearman: {metrics['spearman']} | "
            f"Acc: {metrics['acc_within_stdev']}\n"
        )
        f.write(result_line)
    print(f"[ELECTRA-LoRA] Results appended to evaluation_results.txt")
    
    # Save model
    model_save_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'electra_lora', 'best_model')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    predictor.save_model(model_save_path)
    
    # Log footer (similar to embedding_roberta_xgboost format)
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
    
    print(f"\n[ELECTRA-LoRA] Training complete! Log saved to: {log_file}")
    
    return metrics


def train_dev_and_predict_test(val_split: float = 0.1): 
    # Setup logger
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_log_finetuning_electra_lora_train_dev_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    logger = TrainingLogger(log_file)
    
    # Use default configuration
    config = ElectraLoraConfig()
    # Update output dir for train_dev mode
    config.output_dir = "./src/checkpoints/electra_lora/train_dev"
    
    # Load and combine train + dev data
    print("[ELECTRA-LoRA] Loading data...")
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
    
    print(f"[ELECTRA-LoRA] Total samples (train+dev): {len(full_data)}")
    print(f"  - Original Train: {len(train_data)}")
    print(f"  - Original Dev: {len(dev_data)}")
    print(f"[ELECTRA-LoRA] After {train_pct}/{val_pct} split:")
    print(f"  - Training: {len(full_train_data)} ({train_pct}%)")
    print(f"  - Validation (for early stopping): {len(val_data)} ({val_pct}%)")
    print(f"[ELECTRA-LoRA] Test samples: {len(test_data)}")
    
    # Define output file
    pred_file = os.path.join(TEST_PREDICTIONS_DIR, "finetuning_electra_large_predictions.jsonl")
    
    # Log config header
    logger.log("=" * 60)
    logger.log(f"Training Configuration - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log("=" * 60)
    logger.log(f"Mode: train_dev (train+dev combined, {train_pct}/{val_pct} split)")
    logger.log(f"Model: {config.model_name}")
    logger.log(f"pooling_strategy: {config.pooling_strategy}")
    logger.log(f"huber_delta: {config.huber_delta}")
    logger.log(f"lora_r: {config.lora_r}")
    logger.log(f"lora_alpha: {config.lora_alpha}")
    logger.log(f"lora_dropout: {config.lora_dropout}")
    logger.log(f"learning_rate: {config.learning_rate}")
    logger.log(f"batch_size: {config.batch_size}")
    logger.log(f"gradient_accumulation_steps: {config.gradient_accumulation_steps}")
    logger.log(f"effective_batch_size: {config.batch_size * config.gradient_accumulation_steps}")
    logger.log(f"num_epochs: {config.num_epochs}")
    logger.log(f"early_stopping_patience: {config.early_stopping_patience}")
    logger.log(f"val_split: {val_split}")
    logger.log(f"Total samples (train+dev): {len(full_data)}")
    logger.log(f"Training samples ({train_pct}%): {len(full_train_data)}")
    logger.log(f"Validation samples ({val_pct}%): {len(val_data)}")
    logger.log(f"Test samples: {len(test_data)}")
    logger.log(f"Output: {pred_file}")
    logger.log(f"Log file: {log_file}")
    logger.log("=" * 60 + "\n")
    
    # Initialize predictor
    predictor = ElectraLoraPredictor(config)
    
    # Train with proper validation split for early stopping
    predictor.train(full_train_data, val_data, logger)
    
    # Make predictions on test set
    print("\n[ELECTRA-LoRA] Making predictions on test set...")
    test_predictions = predictor.predict(test_data, is_test=True)
    
    # Save predictions
    save_predictions(test_predictions, pred_file)
    print(f"Predictions saved to: {pred_file}")
    
    # Save model
    model_save_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'electra_lora', 'train_dev', 'best_model')
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
    
    print(f"\n[ELECTRA-LoRA] Training complete! Log saved to: {log_file}")


def predict_test():
    """Make predictions on test set using saved model from train mode"""
    
    model_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'electra_lora', 'train', 'best_model')
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at {model_path}. Run training first.")
        return
    
    test_data = load_test_data()
    print(f"[ELECTRA-LoRA] Test samples: {len(test_data)}")
    
    config = ElectraLoraConfig()
    predictor = ElectraLoraPredictor(config)
    predictor.load_model(model_path)
    
    print("[ELECTRA-LoRA] Making predictions on test set...")
    test_predictions = predictor.predict(test_data, is_test=True)
    
    pred_file = os.path.join(TEST_PREDICTIONS_DIR, "finetuning_electra_large_predictions.jsonl")
    save_predictions(test_predictions, pred_file)
    
    print(f"[ELECTRA-LoRA] Test predictions saved to: {pred_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ELECTRA + LoRA Fine-tuning for Plausibility Rating")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "train_dev", "test"],
                       help="Mode: train (train only), train_dev (train+dev for test), test (predict on test)")
    parser.add_argument("--val_split", type=float, default=0.1,
                       help="Validation split ratio for train_dev mode (default: 0.1)")
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from (e.g. src/checkpoints/electra_lora/train/checkpoint-500)")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        # Pass resume checkpoint to main via environment or direct call
        if args.resume_checkpoint:
            os.environ['ELECTRA_LORA_RESUME'] = args.resume_checkpoint
        main(val_split=args.val_split)
    elif args.mode == "train_dev":
        train_dev_and_predict_test(val_split=args.val_split)
    else:
        predict_test()
