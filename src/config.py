import os
import json
import statistics
from typing import Dict, List
from datetime import datetime
from scipy.stats import spearmanr

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "semeval26-05-scripts")
DATA_DIR = os.path.join(SCRIPTS_DIR, "data")
PREDICTIONS_DIR = os.path.join(PROJECT_ROOT, "src", "predictions")
DEV_PREDICTIONS_DIR = os.path.join(PREDICTIONS_DIR, "dev")
TEST_PREDICTIONS_DIR = os.path.join(PREDICTIONS_DIR, "test")

# Data files
TRAIN_FILE = os.path.join(DATA_DIR, "train.json")
DEV_FILE = os.path.join(DATA_DIR, "dev.json")
TEST_FILE = os.path.join(DATA_DIR, "test.json")

# Results file
RESULTS_FILE = os.path.join(PROJECT_ROOT, "src", "evaluation_results.txt")

# Ensure predictions directories exist
os.makedirs(DEV_PREDICTIONS_DIR, exist_ok=True)
os.makedirs(TEST_PREDICTIONS_DIR, exist_ok=True)


# ============================================================
# Scoring functions
# ============================================================

def get_average(l):
    return sum(l) / len(l)


def get_standard_deviation(l):
    return statistics.stdev(l)


def is_within_standard_deviation(prediction, labels):
    """Official accuracy check: within avg +/- stdev OR distance < 1."""
    avg = get_average(labels)
    stdev = get_standard_deviation(labels)

    # Is prediction within the range of the average +/- the standard deviation?
    if (avg - stdev) < prediction < (avg + stdev):
        return True

    # Is the distance between average and prediction less than one?
    if abs(avg - prediction) < 1:
        return True

    return False


def evaluate_predictions(predictions: List[Dict], ground_truth: Dict) -> Dict[str, float]:
    """
    Evaluate predictions using official scoring logic.

    Args:
        predictions: [{"id": "0", "prediction": 3}, ...]
        ground_truth: dev.json dict {id: {choices, average, stdev, ...}}

    Returns:
        {"spearman": float, "acc_within_stdev": float}
    """
    pred_dict = {str(p['id']): p['prediction'] for p in predictions}

    y_true = []
    y_pred = []
    correct_count = 0

    for sample_id, sample in ground_truth.items():
        sid = str(sample_id)
        if sid in pred_dict:
            avg = get_average(sample['choices'])
            y_true.append(avg)
            y_pred.append(pred_dict[sid])

            if is_within_standard_deviation(pred_dict[sid], sample['choices']):
                correct_count += 1

    if len(y_pred) < 2:
        return {'spearman': 0.0, 'acc_within_stdev': 0.0}

    corr, _ = spearmanr(y_pred, y_true)
    acc = correct_count / len(y_pred)

    return {
        'spearman': corr,
        'acc_within_stdev': acc,
    }


def save_result(pred_path: str, samples: int, spearman: float, acc_within_std: float):
    """Save evaluation result to file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = os.path.basename(pred_path)
    line = f"{timestamp} | {filename} | Samples: {samples} | Spearman: {spearman} | Acc: {acc_within_std}\n"
    with open(RESULTS_FILE, 'a', encoding='utf-8') as f:
        f.write(line)
    print(f"  Result saved to: {RESULTS_FILE}")
