import json
import sys
import os
from typing import Dict, List

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import TRAIN_FILE, DEV_FILE, TEST_FILE


def load_data(filepath: str) -> Dict:
    """Load JSON data file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_train_data() -> Dict:
    """Load training data"""
    return load_data(TRAIN_FILE)


def load_dev_data() -> Dict:
    """Load dev data"""
    return load_data(DEV_FILE)


def load_test_data() -> Dict:
    """Load test data"""
    return load_data(TEST_FILE)


def get_sample_text(sample: Dict, include_ending: bool = True) -> str:
    """
    Convert sample to text format
    
    Args:
        sample: Data sample
        include_ending: Whether to include ending
    
    Returns:
        Combined story text
    """
    text = sample['precontext'] + " " + sample['sentence']
    if include_ending and sample.get('ending', '').strip():
        text += " " + sample['ending']
    return text


def save_predictions(predictions: List[Dict], filepath: str):
    """
    Save predictions in JSONL format
    
    Args:
        predictions: Prediction list [{"id": "xxx", "prediction": 3}, ...]
        filepath: Output path
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')
    print(f"Predictions saved to: {filepath}")


def load_predictions(filepath: str) -> List[Dict]:
    """Load predictions"""
    predictions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            predictions.append(json.loads(line.strip()))
    return predictions


if __name__ == "__main__":
    # Test
    train_data = load_train_data()
    dev_data = load_dev_data()
    
    print(f"Training samples: {len(train_data)}")
    print(f"Dev samples: {len(dev_data)}")
    
    # Print first sample
    sample = list(dev_data.values())[0]
    print(f"\nExample sample:")
    print(f"Story: {get_sample_text(sample)[:200]}...")
    print(f"Homonym: {sample['homonym']}")
    print(f"Meaning: {sample['judged_meaning']}")
    print(f"Human avg: {sample['average']}")
    print(f"Stdev: {sample['stdev']}")
