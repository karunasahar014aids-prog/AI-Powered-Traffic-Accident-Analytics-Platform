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
DATA_PATH = BASE_DIR / "data" / "dataset.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "accident_severity_model.pkl"
META_PATH = MODEL_DIR / "model_metadata.json"

FEATURES = [
    "SPEED_ZONE",
    "ACCIDENT_TYPE",
    "LIGHT_CONDITION",
    "ROAD_GEOMETRY",
    "DAY_OF_WEEK",
    "HOUR",
    "NO_OF_VEHICLES",
]
TARGET = "SEVERITY"
CATEGORICAL = ["ACCIDENT_TYPE", "LIGHT_CONDITION", "ROAD_GEOMETRY", "DAY_OF_WEEK"]
NUMERIC = ["SPEED_ZONE", "HOUR", "NO_OF_VEHICLES"]


def make_demo_data(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    speed = rng.choice([30, 40, 50, 60, 70, 80, 100, 110], n)
    accident_type = rng.choice(["Collision", "Single vehicle", "Pedestrian", "Other"], n)
    light = rng.choice(["Daylight", "Dark", "Dawn/Dusk"], n)
    geometry = rng.choice(["Straight", "Curve", "Intersection", "Roundabout"], n)
    day = rng.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], n)
    hour = rng.integers(0, 24, n)
    vehicles = rng.integers(1, 8, n)
    risk = (speed >= 80).astype(int) + (vehicles >= 4).astype(int) + (light == "Dark").astype(int) + (accident_type == "Pedestrian").astype(int)
    severity = np.where(risk >= 3, "Fatal accident", np.where(risk >= 1, "Serious injury accident", "Other injury accident"))
    return pd.DataFrame({
        "SPEED_ZONE": speed, "ACCIDENT_TYPE": accident_type, "LIGHT_CONDITION": light,
        "ROAD_GEOMETRY": geometry, "DAY_OF_WEEK": day, "HOUR": hour,
        "NO_OF_VEHICLES": vehicles, "SEVERITY": severity,
    })


def load_data():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset is missing columns: {missing}")
        return df[FEATURES + [TARGET]].copy(), False
    return make_demo_data(), True


def main():
    df, demo_mode = load_data()
    df = df.dropna(subset=[TARGET])
    X = df[FEATURES].copy()
    y = df[TARGET].astype(str)

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
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision_macro": round(float(precision_score(y_test, pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_test, pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_test, pred, average="macro", zero_division=0)), 4),
        "classes": sorted(y.unique().tolist()),
        "rows": int(len(df)),
        "demo_mode": demo_mode,
        "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
    }

    with MODEL_PATH.open("wb") as f:
        pickle.dump(model, f)
    META_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nModel training complete")
    print(f"Rows: {len(df):,}")
    print(f"Demo mode: {demo_mode}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['f1_macro']:.4f}")
    print(f"Saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
