"""Data loading and validation for the Kaggle credit-card dataset."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
TARGET = "Class"
RANDOM_STATE = 42


def load_dataset(path):
    frame = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    frame = frame[FEATURES + [TARGET]].copy()
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="raise").astype(int)
    if not set(frame[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Class must contain only 0 (genuine) and 1 (fraud).")
    frame[FEATURES] = frame[FEATURES].apply(pd.to_numeric, errors="raise")
    return frame


def split_dataset(frame):
    X, y = frame[FEATURES], frame[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
