print("Importing...", flush=True)

import sys
import os
import json

import statistics
from scipy.stats import spearmanr

"""
Usage: python3 scoring.py ref_filepath pred_filepath output_filepath
Spearman and Accuracy scores will be printed on command line, and scores will be written to the output filepath.
"""

from format_check import check_formatting

print("Starting Scoring script...", flush=True)

def get_standard_deviation(l):
    return statistics.stdev(l)

def get_average(l):
    return sum(l)/len(l)

def get_gold_by_id(id, gold_data):
    for line in gold_data:
        if str(line["id"]) == str(id):
            return line

def is_within_standard_deviation(prediction, labels):
    avg = get_average(labels)
    stdev = get_standard_deviation(labels)

    # Is prediction within the range of the average +/- the standard deviation?
    if (avg - stdev) < prediction < (avg + stdev):
        return True

    # Is the distance between average and prediction less than one?
    if abs(avg - prediction) < 1:
        return True

    # If neither one applies, then this prediction will be counted as "wrong".
    return False

def spearman_evaluation_score(predictions_filepath: str, gold_data: dict):
    """
    Get the spearman score for a prediction filepath on the gold data.
    It calculates the correlation between the list of predictions and the list of human averages.
    Score is printed on command line.
    """
    gold_list = ["-"] * len(gold_data)
    pred_list = ["-"] * len(gold_data)

    with open(predictions_filepath, "r") as f:
        pred_lines = f.readlines()

    for line in pred_lines:
        line = json.loads(line)
        gold_list[int(line["id"])] = get_average(get_gold_by_id(line["id"], gold_data)["label"])
        pred_list[int(line["id"])] = line["prediction"]

    corr, value = spearmanr(pred_list, gold_list)
    print(f"----------\nSpearman Correlation: {corr}\nSpearman p-Value: {value}")

    return corr



def accuracy_within_standard_deviation_score(predictions_filepath, gold_data):
    """
    Get the Acc. w/in SD score.
    It calculates the proportion of samples where the prediction is within either 1 or +/- standard deviation 
    of the average human judgment.
    Score is printed on command line.
    """
    with open(predictions_filepath, "r") as f:
        pred_lines = f.readlines()

    correct_guesses = 0
    wrong_guesses = 0

    for line in pred_lines:
        line = json.loads(line)
        labels = get_gold_by_id(line["id"], gold_data)["label"]
        if is_within_standard_deviation(line["prediction"], labels):
            correct_guesses += 1
        else:
            wrong_guesses += 1

    print(f"----------\nAccuracy: {correct_guesses / (correct_guesses + wrong_guesses)} ({correct_guesses}/{correct_guesses+wrong_guesses})")

    return correct_guesses / (correct_guesses + wrong_guesses)

if __name__ == "__main__":
    arguments = sys.argv
    if len(arguments) < 3:
        print("Please provide the path to the jsonl predictions file and the jsonl gold file. \n" 
        "Example Usage: python3 scoring.py ref_filepath pred_filepath output_dir", flush=True)
        sys.exit()

    testset = arguments[1]
    try:
        with open(testset, "r") as f:
            testset_lines = f.readlines()
            gold_data = []
            for line in testset_lines:
                gold_data.append(json.loads(line))
    except:
        print("No reference file " + testset + " found.", flush=True)
        sys.exit()

    predictions_filepath = arguments[2]
    if not os.path.exists(predictions_filepath):
        print("Not a valid predictions filepath, file not found: ", predictions_filepath, flush=True)
        sys.exit()

    if not check_formatting(predictions_filepath, gold_data):
        sys.exit()

    print(f"Everything looks OK. Evaluating file {predictions_filepath} on {testset}", flush=True)

    corr = spearman_evaluation_score(predictions_filepath, gold_data)
    acc = accuracy_within_standard_deviation_score(predictions_filepath, gold_data)

    results = {"accuracy": acc, "spearman": corr}

    with open(os.path.join(sys.argv[3]), "w") as f:
        json.dump(results, f)

    print("Results dumped into scores.json successfully.")

