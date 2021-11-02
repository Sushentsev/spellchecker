import argparse
import json
import os
from typing import List

import pandas as pd
from tqdm import tqdm

from dataset import DatasetEntry
from metrics import acc_k, mrr
from spell_checker import SpellChecker


def log_metrics(y_true: List[str], y_pred: List[List[str]]):
    for k in [1, 5, 10]:
        print(f"Acc@{k}: {acc_k(y_true, y_pred, k):.2f}")

    print(f"MRR: {mrr(y_true, y_pred):.2f}")


def train(spell_checker: SpellChecker, data_path: str):
    with open(os.path.join(data_path, "train/train_5k.json")) as file:
        data = json.load(file)

    dataset_size = len(data)
    train_size = int(dataset_size * 0.75)
    train_data, val_data = data[:train_size], data[train_size:]

    train_dataset = [DatasetEntry(misspelled=entry["wrong"], correct=entry["correct"], candidates=entry["candidates"])
                     for entry in train_data if entry["correct"] != "?"]
    val_dataset = [DatasetEntry(misspelled=entry["wrong"], correct=entry["correct"], candidates=entry["candidates"])
                   for entry in val_data if entry["correct"] != "?"]

    spell_checker.train(train_dataset, val_dataset)
    print(f"Fitted.")


def eval(spell_checker: SpellChecker, data_path: str):
    test_df = pd.read_csv("./data/test.tsv", sep="\t")
    y_pred = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        y_pred.append(spell_checker.ranked_candidates(row["wrong"]))

    log_metrics(test_df.correct.values, y_pred)


def main(data_path: str, ranker_type: str):
    spell_checker = SpellChecker.from_files("en_US", "./data/features_data", ranker_type=ranker_type)
    train(spell_checker, data_path)
    eval(spell_checker, data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpellChecker")
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--ranker_type", type=str, default="boosting")
    args = parser.parse_args()

    main(args.data_path, args.ranker_type)

# Basic ranker
# Acc@1: 0.52
# Acc@5: 0.70
# Acc@10: 0.72
# MRR: 0.60

# CatBoost ranker
# Acc@1: 0.59
# Acc@5: 0.72
# Acc@10: 0.73
# MRR: 0.65
