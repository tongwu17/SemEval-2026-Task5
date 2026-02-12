import os
import sys
import re
import time
from openai import OpenAI
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_utils import load_train_data, load_dev_data, load_test_data, save_predictions
from src.config import DEV_PREDICTIONS_DIR, TEST_PREDICTIONS_DIR

# Change model
MODEL_NAME = "gpt-5.2"

def parse_response(text: str) -> int:
    if not text:
        return 3
    
    text = text.strip()
    
    # Direct match (expected case)
    if text in ['1', '2', '3', '4', '5']:
        return int(text)
    
    # Fallback: find first 1-5 digit in text
    match = re.search(r'[1-5]', text)
    if match:
        return int(match.group())
    
    print(f"Warning: Could not extract rating from: {text[:50]}...")
    return 3  # Middle value as safe default

def call_openai_prompt(sample: dict, model: str = MODEL_NAME, api_key: str = None, max_retries: int = 3) -> int:
    """
    Call OpenAI API with structured prompt (system + few-shot + user).
    """
    # -------------------- System Prompt (Task description + Rating scale)
    system_prompt = """You are evaluating whether a proposed meaning of a homonym is supported by its narrative context.

Input format:
- Homonym: The ambiguous word
- Meaning: The proposed interpretation
- Precontext: Background narrative
- Sentence: The sentence containing the homonym
- Ending: The conclusion (may be none)

Rating scale:
1 = Completely implausible. The meaning clearly conflicts with the narrative.
2 = Mostly implausible. Weak or contradictory support.
3 = Moderately plausible. Possible but ambiguous.
4 = Very plausible. Strong and consistent support.
5 = Highly plausible. Clearly intended and strongly confirmed.

The ending is the most important factor for disambiguation.
Return only a single integer (1-5). No explanation."""

    # -------------------- Few-shot Examples (from train.json, stdev=0)
    few_shot_examples = """Examples:

1) Homonym: drive
   Meaning: hitting a golf ball off of a tee with a driver
   Precontext: Lisa had always been competitive. Every weekend, she dedicated herself to her passion. She believed that her relentless practice would pay off someday.
   Sentence: Her drive was what ultimately got her into the top university.
   Ending: She made that long trip to show the course coordinators her dedication to going to that university, and they said that was one of the reasons why they accepted her.
   Rating: 1

2) Homonym: storm
   Meaning: a violent weather condition with winds 64-72 knots and precipitation and thunder and lightning
   Precontext: The village lay quiet under the clouded sky. Tension hung in the air as whispers filled the crowded hall. Soldiers and townspeople alike exchanged knowing looks.
   Sentence: It's uneasy in battle; everyone is preparing for the storm.
   Ending: And everyone was expecting bloodshed; it would be merciless.
   Rating: 2

3) Homonym: interest
   Meaning: a fixed charge for borrowing money; usually a percentage of the amount borrowed
   Precontext: John opened the envelope with trembling hands. He had applied for a loan last week to help start his small business. The bank's decision letter finally lay before him on the kitchen table.
   Sentence: There was a large interest shown.
   Ending: (none)
   Rating: 3

4) Homonym: heated
   Meaning: marked by emotional heat; vehement
   Precontext: The group of friends gathered around the campfire on a chilly night. They started discussing their differing opinions about the best scary stories. One person threw another log onto the fire as the conversation continued.
   Sentence: Things began to get very heated.
   Ending: (none)
   Rating: 4

5) Homonym: drive
   Meaning: a journey in a vehicle (usually an automobile)
   Precontext: John glanced at his watch and sighed. It had been a busy afternoon filled with back-to-back meetings. The evening sky was beginning to darken as he packed up his bag.
   Sentence: He was ready to drive after a long day.
   Ending: He started the car, and the engine roared to life. He sped off down the country lanes into the sunset; he'd been waiting for this moment all day.
   Rating: 5"""

    # -------------------- User Prompt
    ending_text = sample.get('ending', '').strip()
    user_prompt = f"""Homonym: {sample['homonym']}
Meaning: {sample['judged_meaning']}
Precontext: {sample['precontext']}
Sentence: {sample['sentence']}
Ending: {ending_text if ending_text else '(none)'}

Rating:"""

    # -------------------- API call
    if api_key:
        client = OpenAI(api_key=api_key)
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please set OPENAI_API_KEY environment variable")
        client = OpenAI(api_key=api_key)

    use_responses_api = model.startswith("gpt-5")
    answer = ""
    last_error = None

    for attempt in range(max_retries):
        try:
            if use_responses_api:
                # Responses API (for gpt-5.x)
                response = client.responses.create(
                    model=model,
                    input=[
                        {
                            "role": "developer",
                            "content": [{"type": "input_text", "text": system_prompt}]
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": few_shot_examples},
                                {"type": "input_text", "text": user_prompt},
                            ],
                        },
                    ],
                    temperature=0.0,
                    max_output_tokens=16,  
                )

                if hasattr(response, "output_text") and response.output_text:
                    answer = response.output_text.strip()
                elif getattr(response, "output", None):
                    for item in response.output:
                        content = getattr(item, "content", None)
                        if not content:
                            continue
                        for part in content:
                            text_val = getattr(part, "text", None)
                            if text_val:
                                answer = text_val.strip()
                                break
                        if answer:
                            break
            else:
                # Chat Completions API (gpt-4 family)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": few_shot_examples + "\n\nNow rate this:\n" + user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=10,  # Only need a single digit
                )
                answer = response.choices[0].message.content.strip()

            if answer:
                break

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                print(f"\n[WARN] API call failed, retrying ({attempt + 1}/{max_retries})...")
                time.sleep(1)

    if answer:
        return parse_response(answer)
    
    if last_error:
        print(f"\nError: {last_error}")
    return 3  # Default to middle value on failure

def predict_with_llm(data: dict, model: str = None, max_samples: int = None, api_key: str = None) -> tuple:
    if model is None:
        model = MODEL_NAME
    
    all_samples = list(data.items())
    sample_count = min(max_samples, len(all_samples)) if max_samples else len(all_samples)
    
    print("=" * 60)
    print("LLM Prompting")
    print(f"Model: {model}")
    print(f"Samples: {sample_count} / {len(all_samples)}")
    print("=" * 60)
    
    predictions = []
    
    for key, sample in tqdm(all_samples[:sample_count], desc="Predicting with LLM"):
        rating = call_openai_prompt(sample, model, api_key)
        
        predictions.append({
            "id": key,
            "prediction": rating,
        })
        
        # Rate limiting
        time.sleep(0.3)
    
    return predictions, model


def main():
    import argparse
    parser = argparse.ArgumentParser(description="LLM Prompting for Narrative Understanding")
    parser.add_argument("--model", default=None,
                        help="Model name")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum samples to process")
    parser.add_argument("--output", default=None,
                        help="Output file path")
    parser.add_argument("--test", action="store_true",
                        help="Run on test set instead of dev set")
    parser.add_argument("--api-key", default=None,
                        help="API key (optional, reads from environment if not provided)")
    args = parser.parse_args()
    
    # Load data
    if args.test:
        print("Loading test data...")
        data = load_test_data()
        output_dir = TEST_PREDICTIONS_DIR
        print(f"Loaded {len(data)} test samples")
    else:
        print("Loading dev data...")
        data = load_dev_data()
        output_dir = DEV_PREDICTIONS_DIR
        print(f"Loaded {len(data)} dev samples")
    
    # Predict with LLM prompting
    predictions, model_used = predict_with_llm(
        data,
        model=args.model,
        max_samples=args.max_samples,
        api_key=args.api_key
    )
    
    # Generate output filename
    if args.output:
        output_path = args.output
    else:
        model_suffix = model_used.replace("-", "_").replace(".", "_")
        output_path = os.path.join(output_dir, f"{model_suffix}_predictions.jsonl")
    
    # Save predictions
    save_predictions(predictions, output_path)
    print(f"\n[SUCCESS] Predictions saved to: {output_path}")
    
    # Save to evaluation results file
    from datetime import datetime
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_file = os.path.join(src_dir, "evaluation_results.txt")
    output_filename = os.path.basename(output_path)
    
    if args.test:
        print("\n" + "="*50)
        print(f"[LLM Prompting] Test predictions saved!")
        print(f"  Samples: {len(predictions)}")
        print(f"  Output: {output_path}")
        print("  (No evaluation - test set has no labels)")
        
        # Write to results file (test mode)
        with open(results_file, 'a') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} | {output_filename} | Samples: {len(predictions)} | Mode: test | (no labels)\n")
        print(f"  Result saved to: {results_file}")


if __name__ == "__main__":
    main()
