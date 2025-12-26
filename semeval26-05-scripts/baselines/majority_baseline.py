import json
import random

SET = "dev"

MAJORITY_LABEL = 4  # majority label on the test set.
OUTPUT_FILEPATH = "predictions/majority_predictions_dev.jsonl"

if __name__ == "__main__":
    pred_list = []
    with open("data/" + SET + ".json", "r") as f:
        gold_data = json.load(f)

    for i in gold_data:
        pred_list.append({"id": i, "prediction": MAJORITY_LABEL})

    with open(OUTPUT_FILEPATH, "w") as f:
        for pred in pred_list:
            f.write(json.dumps(pred) + "\n")