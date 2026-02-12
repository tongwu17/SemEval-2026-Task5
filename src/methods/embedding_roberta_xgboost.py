import os
import sys
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine, euclidean
import re
from collections import Counter
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_utils import load_train_data, load_dev_data, load_test_data, save_predictions, get_sample_text
from src.config import DEV_PREDICTIONS_DIR, TEST_PREDICTIONS_DIR

class TrainingLogger:
    """Logger for training process"""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.start_time = datetime.now()
        
    def log(self, message: str, print_to_console: bool = True):
        """Log message to file and optionally to console"""
        if print_to_console:
            print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')
    
    def log_header(self, config: dict):
        """Log training configuration header"""
        self.log("=" * 60)
        self.log(f"Training Configuration - {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("=" * 60)
        for key, value in config.items():
            self.log(f"{key}: {value}")
        self.log("=" * 60 + "\n")
    
    def log_footer(self, results: dict):
        """Log training completion footer"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        self.log("\n" + "=" * 60)
        self.log("Training completed!")
        self.log(f"Duration: {duration}")
        for key, value in results.items():
            self.log(f"{key}: {value}")
        self.log("=" * 60)


class EmbeddingPredictorV2:
    def __init__(self, model_name: str = "all-roberta-large-v1", 
                 n_estimators: int = 100, max_depth: int = 4, learning_rate: float = 0.1):
        """
        Initialize embedding model
        
        Args:
            model_name: sentence-transformers model name
            n_estimators: XGBoost n_estimators (reduced to prevent overfitting)
            max_depth: XGBoost max_depth (reduced to prevent overfitting)
            learning_rate: XGBoost learning_rate (increased for better generalization)
        """
        print(f"[Embedding-RoBERTa-XGBoost] Loading embedding model: {model_name}...")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.regressor = None
        print("[Embedding-RoBERTa-XGBoost] Model loaded successfully!")
    
    def extract_features(self, sample: Dict) -> np.ndarray:
        # Get texts
        story = get_sample_text(sample, include_ending=True)
        meaning = f"{sample['homonym']}: {sample['judged_meaning']}"
        
        # Get embeddings
        story_emb = self.model.encode(story, convert_to_numpy=True)
        meaning_emb = self.model.encode(meaning, convert_to_numpy=True)
        
        # Semantic similarity features
        cos_sim = 1 - cosine(story_emb, meaning_emb)
        eucl_dist = euclidean(story_emb, meaning_emb)
        dot_prod = np.dot(story_emb, meaning_emb)
        manhattan_dist = np.sum(np.abs(story_emb - meaning_emb))
        
        # Text length features
        story_len = len(story)
        meaning_len = len(meaning)
        story_words = len(story.split())
        meaning_words = len(meaning.split())
        len_ratio = meaning_len / (story_len + 1)
        word_ratio = meaning_words / (story_words + 1)
        
        # Lexical overlap features
        story_words_set = set(story.lower().split())
        meaning_words_set = set(meaning.lower().split())
        overlap = len(story_words_set & meaning_words_set)
        jaccard = overlap / (len(story_words_set | meaning_words_set) + 1) if story_words_set or meaning_words_set else 0
        
        # Character-level features
        story_chars = Counter(story.lower())
        meaning_chars = Counter(meaning.lower())
        char_overlap = sum((story_chars & meaning_chars).values())
        
        # Sentence structure
        story_sentences = len([s for s in re.split(r'[.!?]+', story) if s.strip()])
        meaning_sentences = len([s for s in re.split(r'[.!?]+', meaning) if s.strip()])
        
        # Punctuation features
        story_punct = sum(1 for c in story if c in '.,!?;:')
        meaning_punct = sum(1 for c in meaning if c in '.,!?;:')
        
        # Binary feature
        has_ending = 1 if sample.get('ending', '').strip() else 0
        
        # Combine all features (23 features)
        features = np.array([
            # Semantic similarity (4)
            cos_sim,
            eucl_dist,
            dot_prod,
            manhattan_dist,
            # Length features (6)
            story_len,
            meaning_len,
            story_words,
            meaning_words,
            len_ratio,
            word_ratio,
            # Lexical overlap (3)
            overlap,
            jaccard,
            char_overlap,
            # Structure (4)
            story_sentences,
            meaning_sentences,
            story_punct,
            meaning_punct,
            # Binary (1)
            has_ending,
            # Interaction features (5)
            cos_sim * len_ratio,
            cos_sim * jaccard,
            eucl_dist * len_ratio,
            dot_prod * word_ratio,
            jaccard * has_ending
        ])
        
        return features
    
    def train(self, train_data: Dict, validation_split: float = 0.2, logger: 'TrainingLogger' = None):
        msg = "\n[Embedding-RoBERTa-XGBoost] Extracting features from training data..."
        print(msg)
        if logger:
            logger.log(msg, print_to_console=False)
            
        X_train = []
        y_train = []
        
        for sample_id, sample in tqdm(train_data.items(), desc="Processing"):
            features = self.extract_features(sample)
            X_train.append(features)
            y_train.append(sample['average'])
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        from xgboost import XGBRegressor
        self.regressor = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            # Regularization parameters
            min_child_weight=3,  # Prevent overfitting
            gamma=0.1,  # Minimum loss reduction
            subsample=0.8,  # Row sampling
            colsample_bytree=0.8,  # Column sampling
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=1.0,  # L2 regularization
            # NO early_stopping_rounds - we'll use Spearman-based selection
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        
        print(f"\n[Embedding-RoBERTa-XGBoost] Training XGBoost (n_estimators={self.n_estimators}, max_depth={self.max_depth}, lr={self.learning_rate})...")
        if logger:
            logger.log(f"\nXGBoost Parameters:", print_to_console=False)
            logger.log(f"  n_estimators: {self.n_estimators}", print_to_console=False)
            logger.log(f"  max_depth: {self.max_depth}", print_to_console=False)
            logger.log(f"  learning_rate: {self.learning_rate}", print_to_console=False)
            logger.log(f"  min_child_weight: 3", print_to_console=False)
            logger.log(f"  gamma: 0.1", print_to_console=False)
            logger.log(f"  subsample: 0.8", print_to_console=False)
            logger.log(f"  colsample_bytree: 0.8", print_to_console=False)
            logger.log(f"  reg_alpha: 0.1, reg_lambda: 1.0", print_to_console=False)
        
        # Use validation split + early stopping (empirically better than 100% training)
        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=validation_split, random_state=42
        )
        
        train_pct = int((1 - validation_split) * 100)
        val_pct = int(validation_split * 100)
        
        print(f"[Embedding-RoBERTa-XGBoost] Training samples: {len(X_tr)}, Validation samples: {len(X_val)} ({train_pct}/{val_pct} split + early stopping)")
        if logger:
            logger.log(f"\nData Split (Early Stopping Mode):", print_to_console=False)
            logger.log(f"  Training samples: {len(X_tr)} ({train_pct}%)", print_to_console=False)
            logger.log(f"  Validation samples: {len(X_val)} ({val_pct}%)", print_to_console=False)
        
        print("\n[Embedding-RoBERTa-XGBoost] Starting training...")
        if logger:
            logger.log("\n[Embedding-RoBERTa-XGBoost] Starting training...", print_to_console=False)
        
        # Train all n_estimators rounds (no early stopping)
        self.regressor.fit(
            X_tr, y_tr,
            eval_set=[(X_tr, y_tr), (X_val, y_val)],
            verbose=False
        )
        
        # Get training history
        evals_result = self.regressor.evals_result()
        train_rmse_history = evals_result['validation_0']['rmse']
        val_rmse_history = evals_result['validation_1']['rmse']
        
        total_rounds = len(val_rmse_history)
        
        # Calculate Spearman for each round and find best by Spearman
        print(f"\nTraining Progress ({total_rounds} rounds, selecting by Val Spearman):")
        if logger:
            logger.log(f"\nTraining Progress ({total_rounds} rounds, selecting by Val Spearman):", print_to_console=False)
        
        best_spearman = -1
        best_spearman_round = 0
        patience = 20  # Early stopping patience
        no_improvement_count = 0
        
        for i in range(total_rounds):
            # Predict using only first i+1 trees
            val_pred = self.regressor.predict(X_val, iteration_range=(0, i+1))
            val_spearman_i, _ = spearmanr(y_val, val_pred)
            
            train_rmse = train_rmse_history[i]
            val_rmse = val_rmse_history[i]
            
            # Check if best
            if val_spearman_i > best_spearman:
                best_spearman = val_spearman_i
                best_spearman_round = i
                no_improvement_count = 0
                is_best = " ← best"
            else:
                no_improvement_count += 1
                is_best = ""
            
            msg = f"  Round {i+1:3d}: Train RMSE={train_rmse:.4f}, Val RMSE={val_rmse:.4f}, Val Spearman={val_spearman_i:.4f}{is_best}"
            print(msg)
            if logger:
                logger.log(msg, print_to_console=False)
            
            # Early stop if no improvement for patience rounds
            if no_improvement_count >= patience:
                msg = f"\nEarly stopping at round {i+1} (no improvement for {patience} rounds)"
                print(msg)
                if logger:
                    logger.log(msg, print_to_console=False)
                break
        
        # Store best iteration for prediction
        self.best_iteration = best_spearman_round
        
        msg = f"Best model at round {best_spearman_round + 1} (Val Spearman: {best_spearman:.4f})"
        print(msg)
        if logger:
            logger.log(msg, print_to_console=False)
        
        # Evaluate on both sets using best iteration
        train_pred = self.regressor.predict(X_tr, iteration_range=(0, best_spearman_round+1))
        val_pred = self.regressor.predict(X_val, iteration_range=(0, best_spearman_round+1))
        
        train_mse = mean_squared_error(y_tr, train_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        train_spearman, _ = spearmanr(y_tr, train_pred)
        val_spearman, _ = spearmanr(y_val, val_pred)
        
        print(f"\nFinal Results:")
        print(f"  Train: RMSE={train_mse**0.5}, Spearman={train_spearman}")
        print(f"  Val:   RMSE={val_mse**0.5}, Spearman={val_spearman}")
        if logger:
            logger.log(f"\nFinal Results:", print_to_console=False)
            logger.log(f"  Train: RMSE={train_mse**0.5}, Spearman={train_spearman}", print_to_console=False)
            logger.log(f"  Val:   RMSE={val_mse**0.5}, Spearman={val_spearman}", print_to_console=False)
        
        # Check for overfitting
        if train_spearman - val_spearman > 0.15:
            msg = "Warning: Potential overfitting detected!"
            print(msg)
            if logger:
                logger.log(msg, print_to_console=False)
    
    def predict(self, test_data: Dict) -> Dict[str, int]:
        if self.regressor is None:
            raise ValueError("Model not trained yet!")
        
        print("\n[Embedding-RoBERTa-XGBoost] Predicting...")
        predictions = {}
        
        # Use best iteration based on Spearman
        iteration_range = (0, self.best_iteration + 1) if hasattr(self, 'best_iteration') else None
        
        for sample_id, sample in tqdm(test_data.items(), desc="Predicting"):
            features = self.extract_features(sample)
            if iteration_range:
                score = self.regressor.predict(features.reshape(1, -1), iteration_range=iteration_range)[0]
            else:
                score = self.regressor.predict(features.reshape(1, -1))[0]
            # Round to integer and clip to valid range [1, 5]
            # Like human annotators who give integer ratings (1-5)
            score = int(round(np.clip(score, 1.0, 5.0)))
            predictions[sample_id] = score
        
        return predictions


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Embedding-based prediction v2")
    parser.add_argument("--model", default="all-roberta-large-v1",
                        help="Sentence-BERT model name")
    parser.add_argument("--n-estimators", type=int, default=100,
                        help="XGBoost n_estimators (reduced for regularization)")
    parser.add_argument("--max-depth", type=int, default=4,
                        help="XGBoost max_depth (reduced for regularization)")
    parser.add_argument("--learning-rate", type=float, default=0.1,
                        help="XGBoost learning_rate")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples for testing (train and dev)")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Validation split ratio for early stopping (default: 0.2)")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "train_dev"],
                        help="Mode: train (Train→Dev prediction), train_dev (Train+Dev→Test prediction)")
    parser.add_argument("--output", default=None,
                        help="Output file path")
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    train_data = load_train_data()
    
    # Setup logging
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(src_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_str = "train_dev" if args.mode == "train_dev" else "train"
    log_filename = f"training_log_embedding_roberta_xgboost_{mode_str}_{timestamp}.txt"
    log_path = os.path.join(log_dir, log_filename)
    logger = TrainingLogger(log_path)
    
    if args.mode == "train_dev":
        # train_dev mode: use train + dev for training, test for prediction
        dev_data = load_dev_data()
        test_data = load_test_data()
        # Rename dev IDs to avoid collision with train IDs
        dev_data_renamed = {f"dev_{k}": v for k, v in dev_data.items()}
        full_train_data = {**train_data, **dev_data_renamed}
        predict_data = test_data
        output_filename = "embedding_roberta_xgboost_predictions.jsonl"
        output_dir = TEST_PREDICTIONS_DIR
        print(f"Train samples (train+dev): {len(full_train_data)}")
        print(f"  - Train: {len(train_data)}")
        print(f"  - Dev: {len(dev_data)}")
        print(f"Test samples: {len(predict_data)}")
    else:
        # train mode: use train for training, dev for prediction
        dev_data = load_dev_data()
        full_train_data = train_data
        predict_data = dev_data
        output_filename = "embedding_roberta_xgboost_predictions.jsonl"
        output_dir = DEV_PREDICTIONS_DIR
        print(f"Train samples: {len(full_train_data)}")
        print(f"Dev samples: {len(predict_data)}")
    
    # Log training configuration
    logger.log_header({
        "Mode": mode_str,
        "Model": args.model,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "Train samples": len(train_data),
        "Dev samples": len(dev_data) if 'dev_data' in dir() else "N/A",
        "Total train samples": len(full_train_data),
        "Predict samples": len(predict_data),
        "Output": os.path.join(output_dir, output_filename),
        "Log file": log_path,
    })
    
    # Limit samples if specified
    if args.max_samples:
        full_train_data = dict(list(full_train_data.items())[:args.max_samples])
        predict_data = dict(list(predict_data.items())[:args.max_samples])
    
    # Initialize predictor
    predictor = EmbeddingPredictorV2(
        model_name=args.model,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate
    )
    
    predictor.train(full_train_data, validation_split=args.val_split, logger=logger)
    
    # Predict
    predictions = predictor.predict(predict_data)
    
    # Save predictions (convert to list format)
    output_path = args.output or os.path.join(output_dir, output_filename)
    pred_list = [{"id": sample_id, "prediction": score} for sample_id, score in predictions.items()]
    save_predictions(pred_list, output_path)
    print(f"Predictions saved to: {output_path}")
    
    # Evaluate (only for train mode, test set has no labels)
    if args.mode == "train":
        print("\n" + "=" * 50)
        print("\n[Embedding-RoBERTa-XGBoost] Evaluation Result: " + output_filename)
        print("-" * 40)
        
        # Calculate Spearman on dev set
        gold_scores = [sample['average'] for sample_id, sample in predict_data.items()]
        pred_scores = [predictions[sample_id] for sample_id in predict_data.keys()]
        
        spearman_corr, p_value = spearmanr(gold_scores, pred_scores)
        
        # Calculate Accuracy (within std or distance < 1)
        def is_within_std(prediction: float, sample: dict) -> bool:
            """Check if prediction is within acceptable range"""
            avg = sample['average']
            stdev = sample['stdev']
            # Within avg ± stdev
            if (avg - stdev) < prediction < (avg + stdev):
                return True
            # Distance < 1
            if abs(avg - prediction) < 1:
                return True
            return False
        
        correct_count = 0
        for sample_id, sample in predict_data.items():
            if is_within_std(predictions[sample_id], sample):
                correct_count += 1
        acc_within_std = correct_count / len(pred_scores)
        
        print(f"  Train Samples (Train Dataset): {len(full_train_data)}")
        print(f"  Predict Samples (Dev Dataset): {len(pred_scores)}")
        print(f"  Spearman: {spearman_corr}")
        print(f"  Acc within std: {acc_within_std} ({correct_count}/{len(pred_scores)})")
        print(f"  p-value: {p_value}")
        
        # Log evaluation results
        logger.log_footer({
            "Train Samples (Train Dataset)": len(full_train_data),
            "Predict Samples (Dev Dataset)": len(pred_scores),
            "Spearman": f"{spearman_corr}",
            "Accuracy": f"{acc_within_std} ({correct_count}/{len(pred_scores)})",
            "p-value": f"{p_value}",
            "Output": output_path,
        })
        
        # Save to results file 
        src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_file = os.path.join(src_dir, "evaluation_results.txt")
        with open(results_file, 'a') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} | {output_filename} | Train Samples (Train Dataset): {len(full_train_data)} | Predict Samples (Dev Dataset): {len(pred_scores)} | Spearman: {spearman_corr} | Acc: {acc_within_std}\n")
        print(f"  Result saved to: {results_file}")
    else:
        print("\n" + "=" * 50)
        print(f"\n[Embedding-RoBERTa-XGBoost] Test predictions saved!")
        print(f"  Train Samples (Train + Dev Dataset): {len(full_train_data)}")
        print(f"  Predict Samples (Test Dataset): {len(predictions)}")
        print(f"  Output: {output_path}")
        print("  (No evaluation - test set has no labels)")
        
        # Log completion
        logger.log_footer({
            "Train Samples (Train + Dev Dataset)": len(full_train_data),
            "Predict Samples (Test Dataset)": len(predictions),
            "Mode": "test",
            "Output": output_path,
            "Note": "No evaluation - test set has no labels",
        })
        
        # Save to results file (test mode - no evaluation metrics)
        src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_file = os.path.join(src_dir, "evaluation_results.txt")
        with open(results_file, 'a') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} | {output_filename} | Train Samples (Train + Dev Dataset): {len(full_train_data)} | Predict Samples (Test Dataset): {len(predictions)} | Mode: test | (no labels)\n")
        print(f"  Result saved to: {results_file}")
        
    print(f"\n  Training log saved to: {log_path}")


if __name__ == "__main__":
    main()
