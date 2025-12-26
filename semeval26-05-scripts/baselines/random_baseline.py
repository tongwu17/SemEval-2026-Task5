import json
import random

SET = "dev"
labels = [1, 2, 3, 4, 5]
OUTPUT_FILEPATH = "predictions/random_predictions_dev.jsonl"

if __name__ == "__main__":
    pred_list = []
    with open("data/" + SET + ".json", "r") as f:
        gold_data = json.load(f)

    for i in gold_data:
        random.shuffle(labels)
        pred_list.append({"id": i, "prediction": labels[0]})

    with open(OUTPUT_FILEPATH, "w") as f:
        for pred in pred_list:
            f.write(json.dumps(pred) + "\n")