"""
Predictive Maintenance API
==========================
Flask REST API that:
  1. Trains a Gradient Boosting model on the AI4I 2020 dataset at startup
  2. Exposes POST /predict  — returns failure probability + risk level
  3. Exposes GET  /health   — liveness check
  4. Exposes GET  /stats    — dataset stats (class balance, feature ranges)
  5. Serves index.html on GET /

Run:
    pip install flask pandas scikit-learn
    python app.py

The CSV file (ai4i2020.csv) must be in the same directory.
"""

import utils.session
import os
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=".")

# ─────────────────────────────────────────────────────────────────────────────
# Model training at startup
# ─────────────────────────────────────────────────────────────────────────────
print("Training model on startup...")

CSV_PATH = os.path.join(os.path.dirname(__file__), "ai4i2020.csv")
df_raw = pd.read_csv(CSV_PATH)

# Clean
df = df_raw.drop(columns=["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"])
le = LabelEncoder()
df["Type"] = le.fit_transform(df["Type"])  # H→0, L→1, M→2

FEATURES = ["Type", "Air temperature [K]", "Process temperature [K]",
            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
TARGET   = "Machine failure"

X = df[FEATURES]
y = df[TARGET]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split & train
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
MODEL_STATS = {
    "accuracy":  round(accuracy_score(y_test, y_pred), 4),
    "recall":    round(recall_score(y_test, y_pred), 4),
    "f1":        round(f1_score(y_test, y_pred), 4),
    "train_samples": int(X_train.shape[0]),
    "test_samples":  int(X_test.shape[0]),
    "failure_rate":  round(float(y.mean()), 4),
}

# Feature ranges for the UI sliders
FEATURE_RANGES = {}
for feat in FEATURES:
    col = df[feat]
    FEATURE_RANGES[feat] = {
        "min":  round(float(col.min()), 2),
        "max":  round(float(col.max()), 2),
        "mean": round(float(col.mean()), 2),
        "std":  round(float(col.std()), 2),
    }

# Feature importance
FEATURE_IMPORTANCE = {
    feat: round(float(imp), 4)
    for feat, imp in zip(FEATURES, model.feature_importances_)
}

print(f"Model ready — Accuracy: {MODEL_STATS['accuracy']:.2%}, "
      f"Recall(Failure): {MODEL_STATS['recall']:.2%}")


# ─────────────────────────────────────────────────────────────────────────────
# Risk classification helper
# ─────────────────────────────────────────────────────────────────────────────
def classify_risk(prob: float) -> dict:
    """Convert a probability to a human-readable risk band."""
    if prob < 0.20:
        return {"level": "LOW",      "color": "#00d084", "action": "Continue normal operation. No action required."}
    elif prob < 0.45:
        return {"level": "MODERATE", "color": "#f7c948", "action": "Schedule inspection within 7 days. Monitor closely."}
    elif prob < 0.70:
        return {"level": "HIGH",     "color": "#ff8c42", "action": "Inspect within 48 hours. Prepare maintenance crew."}
    else:
        return {"level": "CRITICAL", "color": "#ff3b30", "action": "IMMEDIATE ACTION REQUIRED. Stop machine if safe to do so."}


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the frontend."""
    return send_from_directory(".", "index.html")


@app.route("/health", methods=["GET"])
def health():
    """Liveness check."""
    return jsonify({"status": "ok", "model": "GradientBoosting", **MODEL_STATS})


@app.route("/stats", methods=["GET"])
def stats():
    """Return dataset stats, feature ranges and model performance."""
    return jsonify({
        "model_stats":       MODEL_STATS,
        "feature_ranges":    FEATURE_RANGES,
        "feature_importance": FEATURE_IMPORTANCE,
        "type_encoding":     {"H": 0, "L": 1, "M": 2},
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict machine failure.

    Expected JSON body:
    {
        "type":                 "M",      // "H", "L", or "M"
        "air_temperature":      300.5,    // Kelvin
        "process_temperature":  310.2,    // Kelvin
        "rotational_speed":     1400,     // RPM
        "torque":               45.0,     // Nm
        "tool_wear":            120       // minutes
    }

    Returns:
    {
        "failure_probability": 0.23,
        "prediction":          0,         // 0 or 1
        "risk": {
            "level":  "MODERATE",
            "color":  "#f7c948",
            "action": "Schedule inspection..."
        },
        "input_summary": { ... },
        "top_risk_factors": [ ... ]
    }
    """
    data = request.get_json(force=True)

    # ── Validate ──────────────────────────────────────────────────────────────
    required = ["type", "air_temperature", "process_temperature",
                "rotational_speed", "torque", "tool_wear"]
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    type_map = {"H": 0, "L": 1, "M": 2}
    type_raw = str(data["type"]).upper()
    if type_raw not in type_map:
        return jsonify({"error": "type must be 'H', 'L', or 'M'"}), 400

    try:
        raw_input = {
            "Type":                     type_map[type_raw],
            "Air temperature [K]":      float(data["air_temperature"]),
            "Process temperature [K]":  float(data["process_temperature"]),
            "Rotational speed [rpm]":   float(data["rotational_speed"]),
            "Torque [Nm]":              float(data["torque"]),
            "Tool wear [min]":          float(data["tool_wear"]),
        }
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid numeric value: {e}"}), 400

    # ── Predict ───────────────────────────────────────────────────────────────
    X_input = np.array([[raw_input[f] for f in FEATURES]])
    X_input_scaled = scaler.transform(X_input)
    prob      = float(model.predict_proba(X_input_scaled)[0][1])
    prediction = int(model.predict(X_input_scaled)[0])
    risk      = classify_risk(prob)

    # ── Top risk factors ──────────────────────────────────────────────────────
    # How many std-devs above mean is each feature? Weighted by importance.
    risk_factors = []
    for i, feat in enumerate(FEATURES):
        mean = FEATURE_RANGES[feat]["mean"]
        std  = FEATURE_RANGES[feat]["std"] or 1
        z    = (raw_input[feat] - mean) / std
        importance = FEATURE_IMPORTANCE[feat]
        risk_factors.append({
            "feature":    feat,
            "value":      raw_input[feat],
            "z_score":    round(z, 2),
            "importance": importance,
            "risk_score": round(abs(z) * importance, 4),
        })
    risk_factors.sort(key=lambda x: x["risk_score"], reverse=True)

    return jsonify({
        "failure_probability": round(prob, 4),
        "prediction":          prediction,
        "risk":                risk,
        "input_summary":       {k: round(v, 2) for k, v in raw_input.items()},
        "top_risk_factors":    risk_factors[:3],
    })


@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """
    Predict for multiple machines at once.
    Body: { "machines": [ { ...same as /predict... }, ... ] }
    """
    data = request.get_json(force=True)
    machines = data.get("machines", [])
    if not machines:
        return jsonify({"error": "No machines provided"}), 400

    results = []
    for i, m in enumerate(machines):
        # Re-use the single predict logic by faking a request context
        type_map = {"H": 0, "L": 1, "M": 2}
        type_raw = str(m.get("type", "M")).upper()
        try:
            raw = {
                "Type":                     type_map.get(type_raw, 2),
                "Air temperature [K]":      float(m.get("air_temperature", 300)),
                "Process temperature [K]":  float(m.get("process_temperature", 310)),
                "Rotational speed [rpm]":   float(m.get("rotational_speed", 1500)),
                "Torque [Nm]":              float(m.get("torque", 40)),
                "Tool wear [min]":          float(m.get("tool_wear", 100)),
            }
            X_in = scaler.transform(np.array([[raw[f] for f in FEATURES]]))
            prob = float(model.predict_proba(X_in)[0][1])
            pred = int(model.predict(X_in)[0])
            results.append({
                "machine_id": m.get("id", f"M{i+1}"),
                "failure_probability": round(prob, 4),
                "prediction": pred,
                "risk": classify_risk(prob),
            })
        except Exception as e:
            results.append({"machine_id": m.get("id", f"M{i+1}"), "error": str(e)})

    return jsonify({"results": results, "total": len(results)})


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting Predictive Maintenance API on http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
