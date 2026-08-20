from pathlib import Path
import json
import os
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [BASE_DIR / "data" / "dataset.csv", BASE_DIR / "data" / "victorian_road_crash_data(1).csv"]
MODEL_PATH = BASE_DIR / "models" / "accident_severity_model.pkl"
META_PATH = BASE_DIR / "models" / "model_metadata.json"
FEATURES = ["SPEED_ZONE", "ACCIDENT_TYPE", "LIGHT_CONDITION", "ROAD_GEOMETRY", "DAY_OF_WEEK", "HOUR", "NO_OF_VEHICLES"]
TARGET = "SEVERITY"
app = Flask(__name__)


def find_dataset():
    return next((p for p in DATA_CANDIDATES if p.exists()), None)


def load_model():
    if not MODEL_PATH.exists():
        try:
            subprocess.run([sys.executable, str(BASE_DIR / "train_model.py")], cwd=BASE_DIR, check=True, timeout=900)
        except Exception:
            return None
    if not MODEL_PATH.exists():
        return None
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


def load_metadata():
    if META_PATH.exists():
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    return {"demo_mode": True, "accuracy": None, "f1_macro": None, "classes": []}


def make_demo_data(n=2500, seed=42):
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
    return pd.DataFrame({"SPEED_ZONE": speed, "ACCIDENT_TYPE": accident_type, "LIGHT_CONDITION": light, "ROAD_GEOMETRY": geometry, "DAY_OF_WEEK": day, "HOUR": hour, "NO_OF_VEHICLES": vehicles, "SEVERITY": severity})


def load_data():
    path = find_dataset()
    if path is None:
        return make_demo_data(), True
    df = pd.read_csv(path, low_memory=False)
    if "HOUR" not in df.columns and "ACCIDENT_TIME" in df.columns:
        df["HOUR"] = pd.to_datetime(df["ACCIDENT_TIME"].astype(str), errors="coerce").dt.hour
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        return make_demo_data(), True
    for col in ["SPEED_ZONE", "HOUR", "NO_OF_VEHICLES"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[FEATURES + [TARGET]].dropna(subset=[TARGET]).copy(), False


def dashboard_data():
    df, demo = load_data()
    severity = df[TARGET].value_counts().to_dict()
    by_hour = df.groupby("HOUR").size().reindex(range(24), fill_value=0).tolist()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    by_day = df.groupby("DAY_OF_WEEK").size().reindex(day_order, fill_value=0).tolist()
    speed = df.groupby("SPEED_ZONE").size().sort_index()
    accident_type = df.groupby("ACCIDENT_TYPE").size().sort_values(ascending=False).head(8)
    light = df.groupby("LIGHT_CONDITION").size().sort_values(ascending=False).head(8)
    return {
        "demo_mode": demo,
        "total": int(len(df)),
        "fatal": int(severity.get("Fatal accident", 0)),
        "serious": int(severity.get("Serious injury accident", 0)),
        "other": int(severity.get("Other injury accident", 0)),
        "severity_labels": list(severity.keys()),
        "severity_values": [int(v) for v in severity.values()],
        "hour_labels": list(range(24)),
        "hour_values": by_hour,
        "day_labels": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        "day_values": by_day,
        "speed_labels": [str(x) for x in speed.index.tolist()],
        "speed_values": [int(v) for v in speed.values.tolist()],
        "type_labels": [str(x) for x in accident_type.index.tolist()],
        "type_values": [int(v) for v in accident_type.values.tolist()],
        "light_labels": [str(x) for x in light.index.tolist()],
        "light_values": [int(v) for v in light.values.tolist()],
    }


def build_recommendation(risk, row):
    if risk == "High":
        text = "High-severity pattern detected. Prioritize emergency response, speed control, and immediate safety intervention."
    elif risk == "Medium":
        text = "Medium-severity pattern detected. Increase monitoring and consider warning signs and speed-control measures."
    else:
        text = "Lower-severity pattern detected. Continue routine monitoring and standard road-safety practices."
    if row["LIGHT_CONDITION"].lower() in {"dark", "darkness", "night"}:
        text += " Reduced-light condition: use additional caution and visibility measures."
    if row["SPEED_ZONE"] >= 80:
        text += " Higher speed zone: reinforce speed compliance."
    return text


def explanation_factors(row):
    factors = []
    if row["SPEED_ZONE"] >= 80:
        factors.append("Higher speed zone")
    if row["NO_OF_VEHICLES"] >= 4:
        factors.append("Higher vehicle count")
    if row["LIGHT_CONDITION"].lower() in {"dark", "darkness", "night"}:
        factors.append("Reduced-light condition")
    if row["ACCIDENT_TYPE"].lower() == "pedestrian":
        factors.append("Pedestrian involvement")
    if not factors:
        factors.append("Selected road and traffic conditions")
    return factors


@app.route("/")
def index():
    return render_template("index.html", stats=dashboard_data(), metadata=load_metadata())


@app.post("/predict")
def predict():
    model = load_model()
    if model is None:
        return jsonify({"error": "Model training failed. Check deployment build logs."}), 503
    payload = request.get_json(silent=True) or request.form.to_dict()
    try:
        row = {"SPEED_ZONE": float(payload["SPEED_ZONE"]), "ACCIDENT_TYPE": payload["ACCIDENT_TYPE"], "LIGHT_CONDITION": payload["LIGHT_CONDITION"], "ROAD_GEOMETRY": payload["ROAD_GEOMETRY"], "DAY_OF_WEEK": payload["DAY_OF_WEEK"], "HOUR": int(payload["HOUR"]), "NO_OF_VEHICLES": int(payload["NO_OF_VEHICLES"])}
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid input: {exc}"}), 400
    frame = pd.DataFrame([row], columns=FEATURES)
    prediction = model.predict(frame)[0]
    probabilities = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(frame)[0]
        probabilities = {str(c): round(float(p) * 100, 2) for c, p in zip(model.classes_, probs)}
    risk = "High" if "Fatal" in str(prediction) else "Medium" if "Serious" in str(prediction) else "Low"
    return jsonify({
        "prediction": str(prediction),
        "risk": risk,
        "probabilities": probabilities,
        "recommendation": build_recommendation(risk, row),
        "factors": explanation_factors(row),
        "input_context": row,
    })


@app.get("/api/stats")
def api_stats():
    return jsonify(dashboard_data())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
