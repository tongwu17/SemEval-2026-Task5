import os
import sys
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine, euclidean
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


class EmbeddingPredictor:
    """
    Embedding-based predictor: MPNet + Ridge Regression
    Uses all-mpnet-base-v2 model with Ridge regression
    """
    
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """
        Initialize embedding model
        
        Args:
            model_name: sentence-transformers model name
        """
        print(f"[Embedding-MPNet-Ridge] Loading embedding model: {model_name}...")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.regressor = None
        print("[Embedding-MPNet-Ridge] Model loaded successfully!")
    
    def extract_features(self, sample: Dict) -> np.ndarray:
        """
        Extract features from a sample
        
        Args:
            sample: Data sample
        
        Returns:
            Feature vector
        """
        # Get texts
        story = get_sample_text(sample, include_ending=True)
        meaning = f"{sample['homonym']}: {sample['judged_meaning']}"
        
        # Get embeddings
        story_emb = self.model.encode(story, convert_to_numpy=True)
        meaning_emb = self.model.encode(meaning, convert_to_numpy=True)
        
        # Calculate similarities
        cos_sim = 1 - cosine(story_emb, meaning_emb)  # Cosine similarity
        eucl_dist = euclidean(story_emb, meaning_emb)  # Euclidean distance
        dot_prod = np.dot(story_emb, meaning_emb)      # Dot product
        
        # Text-based features
        story_len = len(story.split())
        meaning_len = len(meaning.split())
        has_ending = 1 if sample.get('ending', '').strip() else 0
        
        # Combine features
        features = np.array([
            cos_sim,
            eucl_dist,
            dot_prod,
            story_len,
            meaning_len,
            has_ending,
            cos_sim * story_len,  # Interaction features
            cos_sim * meaning_len,
        ])
        
        return features
    
    def train(self, train_data: Dict, alpha: float = 1.0, validation_split: float = 0.1, logger: 'TrainingLogger' = None):
        msg = "\n[Embedding-MPNet-Ridge] Extracting features from training data..."
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
        
        print(f"\n[Embedding-MPNet-Ridge] Training Ridge regression (alpha={alpha})...")
        if logger:
            logger.log(f"\nRidge Parameters:", print_to_console=False)
            logger.log(f"  alpha: {alpha}", print_to_console=False)
            
        self.regressor = Ridge(alpha=alpha)
        
        if validation_split > 0:
            # Use validation split for monitoring
            from sklearn.model_selection import train_test_split
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=validation_split, random_state=42
            )
            
            train_pct = int((1 - validation_split) * 100)
            val_pct = int(validation_split * 100)
            
            print(f"[Embedding-MPNet-Ridge] Training samples: {len(X_tr)}, Validation samples: {len(X_val)} ({train_pct}/{val_pct} split)")
            if logger:
                logger.log(f"\nData Split:", print_to_console=False)
                logger.log(f"  Training samples: {len(X_tr)} ({train_pct}%)", print_to_console=False)
                logger.log(f"  Validation samples: {len(X_val)} ({val_pct}%)", print_to_console=False)
            
            self.regressor.fit(X_tr, y_tr)
            
            # Evaluate on both sets
            train_pred = self.regressor.predict(X_tr)
            val_pred = self.regressor.predict(X_val)
            
            train_mse = mean_squared_error(y_tr, train_pred)
            val_mse = mean_squared_error(y_val, val_pred)
            train_spearman, _ = spearmanr(y_tr, train_pred)
            val_spearman, _ = spearmanr(y_val, val_pred)
            
            print(f"Training MSE: {train_mse}, Spearman: {train_spearman}")
            print(f"Validation MSE: {val_mse}, Spearman: {val_spearman}")
            
            if logger:
                logger.log(f"\nTraining Results:", print_to_console=False)
                logger.log(f"  Train MSE: {train_mse}, Spearman: {train_spearman}", print_to_console=False)
                logger.log(f"  Val MSE: {val_mse}, Spearman: {val_spearman}", print_to_console=False)
            
            # Check for overfitting
            if train_spearman - val_spearman > 0.15:
                msg = "Warning: Potential overfitting detected!"
                print(msg)
                if logger:
                    logger.log(msg, print_to_console=False)
        else:
            # 100% training (no split)
            print(f"[Embedding-MPNet-Ridge] Training samples: {len(X_train)} (100% data)")
            if logger:
                logger.log(f"\nData:", print_to_console=False)
                logger.log(f"  Training samples: {len(X_train)} (100%)", print_to_console=False)
            
            self.regressor.fit(X_train, y_train)
            
            # Evaluate on training set
            train_pred = self.regressor.predict(X_train)
            train_mse = mean_squared_error(y_train, train_pred)
            train_spearman, _ = spearmanr(y_train, train_pred)
            
            print(f"Training MSE: {train_mse}, Spearman: {train_spearman}")
            
            if logger:
                logger.log(f"\nTraining Results:", print_to_console=False)
                logger.log(f"  Train MSE: {train_mse}, Spearman: {train_spearman}", print_to_console=False)
    
    def predict(self, data: Dict) -> List[Dict]:
        """
        Predict on new data
        
        Args:
            data: Data dictionary
        
        Returns:
            Predictions list
        """
        if self.regressor is None:
            raise ValueError("Model not trained! Call train() first.")
        
        print("\n[Embedding-MPNet-Ridge] Predicting...")
        predictions = []
        
        for sample_id, sample in tqdm(data.items(), desc="Predicting"):
            features = self.extract_features(sample)
            score = self.regressor.predict(features.reshape(1, -1))[0]
            
            # Clip to 1-5 range and round to integer
            score = max(1, min(5, round(score)))
            
            predictions.append({
                "id": sample_id,
                "prediction": int(score)
            })
        
        return predictions


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Embedding-based prediction")
    parser.add_argument("--model", default="all-mpnet-base-v2", 
                        help="Sentence-transformers model name")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Ridge regression alpha")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples for testing (train and dev)")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio (default: 0.1)")
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
    log_filename = f"training_log_embedding_mpnet_ridge_{mode_str}_{timestamp}.txt"
    log_path = os.path.join(log_dir, log_filename)
    logger = TrainingLogger(log_path)
    output_filename = "embedding_mpnet_ridge_predictions.jsonl"
    
    if args.mode == "train_dev":
        # train_dev mode: use train + dev for training, test for prediction
        dev_data = load_dev_data()
        test_data = load_test_data()
        # Rename dev IDs to avoid collision with train IDs
        dev_data_renamed = {f"dev_{k}": v for k, v in dev_data.items()}
        full_train_data = {**train_data, **dev_data_renamed}
        predict_data = test_data
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
        output_dir = DEV_PREDICTIONS_DIR
        print(f"Train samples: {len(full_train_data)}")
        print(f"Dev samples: {len(predict_data)}")
    
    # Log training configuration
    logger.log_header({
        "Mode": mode_str,
        "Model": args.model,
        "alpha": args.alpha,
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
    
    # Initialize and train
    predictor = EmbeddingPredictor(model_name=args.model)
    # Dev mode: use 80/20 validation to monitor overfitting
    # Test mode: use 100% data to maximize training data
    predictor.train(full_train_data, alpha=args.alpha, validation_split=args.val_split, logger=logger)
    
    # Predict
    predictions = predictor.predict(predict_data)
    
    # Save predictions
    output_path = args.output or os.path.join(output_dir, output_filename)
    save_predictions(predictions, output_path)
    print(f"Predictions saved to: {output_path}")
    
    # Evaluate (only for train mode, test set has no labels)
    if args.mode == "train":
        print("\n" + "=" * 50)
        print("\n[Embedding-MPNet-Ridge] Evaluation Result: " + output_filename)
        print("-" * 40)
        
        # Calculate Spearman on dev set
        gold_scores = [sample['average'] for sample_id, sample in predict_data.items()]
        pred_scores = [p['prediction'] for p in predictions]
        
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
        for i, (sample_id, sample) in enumerate(predict_data.items()):
            if is_within_std(predictions[i]['prediction'], sample):
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
        results_file = os.path.join(src_dir, "evaluation_results.txt")
        with open(results_file, 'a') as f:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{ts} | {output_filename} | Train Samples (Train Dataset): {len(full_train_data)} | Predict Samples (Dev Dataset): {len(pred_scores)} | Spearman: {spearman_corr} | Acc: {acc_within_std}\n")
        print(f"  Result saved to: {results_file}")
    else:
        print("\n" + "=" * 50)
        print(f"\n[Embedding-MPNet-Ridge] Test predictions saved!")
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
        results_file = os.path.join(src_dir, "evaluation_results.txt")
        with open(results_file, 'a') as f:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{ts} | {output_filename} | Train Samples (Train + Dev Dataset): {len(full_train_data)} | Predict Samples (Test Dataset): {len(predictions)} | Mode: test | (no labels)\n")
        print(f"  Result saved to: {results_file}")
    
    print(f"\n  Training log saved to: {log_path}")


if __name__ == "__main__":
    main()
