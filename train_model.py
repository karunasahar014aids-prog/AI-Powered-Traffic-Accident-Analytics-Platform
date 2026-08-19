from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    BASE_DIR / "data" / "dataset.csv",
    BASE_DIR / "data" / "victorian_road_crash_data(1).csv",
]
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "accident_severity_model.pkl"
META_PATH = MODEL_DIR / "model_metadata.json"

FEATURES = ["SPEED_ZONE", "ACCIDENT_TYPE", "LIGHT_CONDITION", "ROAD_GEOMETRY", "DAY_OF_WEEK", "HOUR", "NO_OF_VEHICLES"]
TARGET = "SEVERITY"
CATEGORICAL = ["ACCIDENT_TYPE", "LIGHT_CONDITION", "ROAD_GEOMETRY", "DAY_OF_WEEK"]
NUMERIC = ["SPEED_ZONE", "HOUR", "NO_OF_VEHICLES"]


def find_dataset():
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    return None


def load_data():
    path = find_dataset()
    if path is None:
        raise FileNotFoundError("Dataset not found. Put the CSV in data/dataset.csv")

    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in ["SPEED_ZONE", "ACCIDENT_TYPE", "LIGHT_CONDITION", "ROAD_GEOMETRY", "DAY_OF_WEEK", "NO_OF_VEHICLES", "SEVERITY"] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    if "HOUR" not in df.columns:
        if "ACCIDENT_TIME" not in df.columns:
            raise ValueError("Dataset needs HOUR or ACCIDENT_TIME")
        df["HOUR"] = pd.to_datetime(df["ACCIDENT_TIME"].astype(str), errors="coerce").dt.hour

    for col in ["SPEED_ZONE", "HOUR", "NO_OF_VEHICLES"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[FEATURES + [TARGET]].dropna(subset=[TARGET]).copy()
    return df


def main():
    df = load_data()
    X, y = df[FEATURES], df[TARGET].astype(str)

    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC),
        ("cat", categorical_pipe, CATEGORICAL),
    ])

    classifier = RandomForestClassifier(
        n_estimators=250,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision_macro": round(float(precision_score(y_test, pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_test, pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_test, pred, average="macro", zero_division=0)), 4),
        "classes": sorted(y.unique().tolist()),
        "rows": int(len(df)),
        "dataset": "Victorian Road Crash Data",
        "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
    }

    with MODEL_PATH.open("wb") as f:
        pickle.dump(model, f)
    META_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Model training complete")
    print(f"Rows: {len(df):,}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['f1_macro']:.4f}")
    print(f"Saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
