import sqlite3
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

DB_PATH = "predictions.db"

# Valid ranges - apne dataset ke hisab se badal sakti ho
RANGES = {
    "study_hours": (0, 24),
    "attendance": (0, 100),
    "previous_score": (0, 100),
    "assignments_completed": (0, 100),
}

# Trained model load karna
bundle = joblib.load("model/best_model.pkl")
model = bundle["model"]
FEATURES = bundle["features"]


# ---------- Database ----------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                study_hours REAL,
                attendance REAL,
                previous_score REAL,
                assignments_completed REAL,
                prediction TEXT,
                confidence REAL
            )"""
        )


init_db()


# ---------- Helpers ----------
def validate(form):
    """Form ki values check karta hai. (values, errors) return karta hai."""
    values, errors = {}, []
    for field in FEATURES:
        raw = form.get(field, "").strip()
        label = field.replace("_", " ").title()
        if raw == "":
            errors.append(f"{label} cannot be empty.")
            continue
        try:
            value = float(raw)
        except ValueError:
            errors.append(f"{label} must be a number.")
            continue
        low, high = RANGES[field]
        if not low <= value <= high:
            errors.append(f"{label} must be between {low} and {high}.")
            continue
        values[field] = value
    return values, errors


def get_advice(v):
    """Student ki values dekh kar personalised suggestions."""
    advice = []
    if v["study_hours"] < 3:
        advice.append("Study hours are low. Try to study at least 3-4 hours every day.")
    if v["attendance"] < 75:
        advice.append("Attendance is below 75%. Attend classes regularly.")
    if v["previous_score"] < 50:
        advice.append("Previous score is low. Revise weak topics and practice more.")
    if v["assignments_completed"] < 50:
        advice.append("Few assignments are completed. Submit every assignment on time.")
    if not advice:
        advice.append("Great job! Keep up the same routine.")
    return advice


MESSAGES = {
    "High": "The student is performing very well based on the provided information.",
    "Medium": "The student has medium performance. More study and practice may help improve the result.",
    "Low": "The student has low performance. More study time and academic support may help.",
}


# ---------- Routes ----------
@app.route("/")
def home():
    return render_template("index.html", form_values={}, errors=[])


@app.route("/predict", methods=["POST"])
def predict():
    values, errors = validate(request.form)

    if errors:
        return render_template("index.html", errors=errors, form_values=request.form)

    student = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
    prediction = str(model.predict(student)[0])

    # Har class ki probability (confidence)
    probabilities = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(student)[0]
        probabilities = {
            str(c): round(float(p) * 100, 1) for c, p in zip(model.classes_, probs)
        }
    confidence = probabilities.get(prediction)

    with get_db() as conn:
        conn.execute(
            """INSERT INTO predictions
               (created_at, study_hours, attendance, previous_score,
                assignments_completed, prediction, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                values["study_hours"],
                values["attendance"],
                values["previous_score"],
                values["assignments_completed"],
                prediction,
                confidence,
            ),
        )

    importance = {
        k: round(v * 100, 1) for k, v in bundle.get("importance", {}).items()
    }

    return render_template(
        "index.html",
        errors=[],
        form_values=values,
        prediction=prediction,
        message=MESSAGES.get(prediction, ""),
        probabilities=probabilities,
        advice=get_advice(values),
        importance=importance,
    )


@app.route("/history")
def history():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT 100"
        ).fetchall()
    return render_template("history.html", rows=rows)


if __name__ == "__main__":
    app.run(debug=True)
