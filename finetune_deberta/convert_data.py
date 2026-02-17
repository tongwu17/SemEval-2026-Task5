"""
Convert AmbiStory data from original format to expected format
"""

import json
import argparse
from pathlib import Path


def convert_ambistory_data(input_file: str, output_file: str):
    # Load data
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Convert to list format
    converted_data = []
    
    for idx, sample in data.items():
        # Split precontext into sentences (basic approach)
        # The precontext is usually 3 sentences
        precontext = sample['precontext']
        
        # Simple sentence splitting (you may want to use a better sentence tokenizer)
        sentences = precontext.replace('! ', '!|').replace('? ', '?|').replace('. ', '.|').split('|')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Extract target word from sentence
        target_word = sample['homonym']
        
        converted = {
            "id": idx,
            "story_id": sample['sample_id'],
            "homonym": sample['homonym'],
            "precontext_sentences": sentences,
            "ambiguous_sentence": sample['sentence'],
            "ending": sample['ending'],  # Can be empty string
            "target_word": target_word,
            "word_sense": sample['judged_meaning'],
            "avg_score": sample['average'],
            "std_score": sample['stdev'],
            "example_sentence": sample.get('example_sentence', ''),
            "individual_scores": sample['choices'],
            "nonsensical_flags": sample.get('nonsensical', [])
        }
        
        converted_data.append(converted)
    
    # Save converted data
    print(f"Converted {len(converted_data)} samples")
    print(f"Saving to {output_file}...")
    
    with open(output_file, 'w') as f:
        json.dump(converted_data, f, indent=2)
    
    print(f" Conversion complete!")

    scores = [s['avg_score'] for s in converted_data]
    if all([isinstance(score, float) for score in scores]):
        # Print statistics
        print(f"\nData Statistics:")
        print(f"  Total samples: {len(converted_data)}")
        print(f"  Score range: [{min(scores):.2f}, {max(scores):.2f}]")
        print(f"  Score mean: {sum(scores)/len(scores):.2f}")
    
    # Count samples with/without endings
    with_ending = sum(1 for s in converted_data if s['ending'])
    without_ending = len(converted_data) - with_ending
    print(f"  Samples with ending: {with_ending}")
    print(f"  Samples without ending: {without_ending}")
    
    # Unique homonyms
    unique_homonyms = set(s['homonym'] for s in converted_data)
    print(f"  Unique homonyms: {len(unique_homonyms)}")
    
    return converted_data


def split_train_val(data_file: str, train_file: str, val_file: str, val_ratio: float = 0.2):
    """Split data into train and validation sets"""
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # Shuffle data
    import random
    random.seed(42)
    random.shuffle(data)
    
    # Split
    split_idx = int(len(data) * (1 - val_ratio))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    # Save
    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(val_file, 'w') as f:
        json.dump(val_data, f, indent=2)
    
    print(f"Split into:")
    print(f"  Train: {len(train_data)} samples -> {train_file}")
    print(f"  Val: {len(val_data)} samples -> {val_file}")


def main():
    parser = argparse.ArgumentParser(description='Convert AmbiStory data format')
    parser.add_argument('--input', type=str, required=True,
                       help='Input JSON file (original format)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON file (converted format)')
    parser.add_argument('--split', action='store_true',
                       help='Also split into train/val')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='Validation ratio (default: 0.2)')
    
    args = parser.parse_args()
    
    # Convert data
    converted_data = convert_ambistory_data(args.input, args.output)
    
    # Optionally split
    if args.split:
        base_name = Path(args.output).stem
        base_dir = Path(args.output).parent
        train_file = base_dir / f"{base_name}_train.json"
        val_file = base_dir / f"{base_name}_val.json"
        
        split_train_val(args.output, str(train_file), str(val_file), args.val_ratio)


if __name__ == "__main__":
    main()
