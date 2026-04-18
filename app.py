"""
Exam Anxiety Level Predictor — Flask Backend
"""

import os, pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from groq import Groq
from datetime import datetime, timedelta

app = Flask(__name__)

BASE        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE, "model", "anxiety_model.pkl")
SCALER_PATH = os.path.join(BASE, "model", "scaler.pkl")

FEATURES = [
    "sleep_quality", "headache", "academic_performance",
    "study_load", "extracurricular_activities",
    "peer_pressure", "future_career_concerns"
]
TARGET    = "stress_level"
LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}
LEVELS    = ["Low", "Medium", "High"]

def train_and_save():
    print("Model not found — training now (takes ~30 seconds)...")
    import urllib.request
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    DATASET_PATH = os.path.join(BASE, "StressLevelDataset.csv")

    if not os.path.exists(DATASET_PATH):
        url = "https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Student%20Stress%20Factors.csv"
        try:
            urllib.request.urlretrieve(url, DATASET_PATH)
        except Exception:
            np.random.seed(42)
            n = 1100
            data = {
                'sleep_quality':              np.random.randint(1, 6, n),
                'headache':                   np.random.randint(1, 6, n),
                'academic_performance':       np.random.randint(1, 6, n),
                'study_load':                 np.random.randint(1, 6, n),
                'extracurricular_activities': np.random.randint(1, 6, n),
                'peer_pressure':              np.random.randint(1, 6, n),
                'future_career_concerns':     np.random.randint(1, 6, n),
                'stress_level':               np.random.choice([0,1,2], n, p=[0.33,0.34,0.33])
            }
            pd.DataFrame(data).to_csv(DATASET_PATH, index=False)

    df_raw = pd.read_csv(DATASET_PATH)
    for col in FEATURES:
        if col not in df_raw.columns:
            np.random.seed(42)
            df_raw[col] = np.random.randint(1, 6, len(df_raw))

    df = df_raw[FEATURES + [TARGET]].dropna().copy()
    X, y = df[FEATURES], df[TARGET]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(
        n_estimators=100, max_depth=10,
        min_samples_split=5, min_samples_leaf=2,
        class_weight='balanced', random_state=42
    )
    model.fit(X_train_scaled, y_train)

    os.makedirs(os.path.join(BASE, "model"), exist_ok=True)
    with open(MODEL_PATH,  "wb") as f: pickle.dump(model,  f)
    with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)
    print("Model trained and saved!")
    return model, scaler

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    with open(MODEL_PATH,  "rb") as f: clf    = pickle.load(f)
    with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)
else:
    clf, scaler = train_and_save()

GROQ_KEY    = os.environ.get("GROQ_API_KEY", "")
groq_client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None

FALLBACK = {
    "Low": (
        "• Keep your balanced routine — 25-min Pomodoro sessions beat long cramming.\n"
        "• Stay connected socially; brief chats after study sessions recharge focus.\n"
        "• On exam day: arrive 10 mins early, do 5 slow deep breaths, trust your prep."
    ),
    "Medium": (
        "• Break material into small daily goals and tick them off — visible progress lowers anxiety.\n"
        "• Add a 20-min walk each evening; it flushes cortisol and improves sleep quality.\n"
        "• Try 4-7-8 breathing before bed: inhale 4s, hold 7s, exhale 8s."
    ),
    "High": (
        "• Talk to a counsellor or trusted person this week — verbalising stress halves its intensity.\n"
        "• No sessions longer than 45 mins; mandatory 15-min breaks; stop 2 hrs before sleep.\n"
        "• Progressive muscle relaxation nightly: tense and release each muscle group foot-to-shoulders."
    )
}

def get_coping_strategy(label, features):
    if not groq_client:
        return FALLBACK.get(label, FALLBACK["Medium"])
    prompt = f"""A student has {label} exam anxiety.
Profile: sleep={features.get('sleep_quality')}/5, headache={features.get('headache')}/5,
academic={features.get('academic_performance')}/5, study_load={features.get('study_load')}/5,
extracurricular={features.get('extracurricular_activities')}/5,
peer_pressure={features.get('peer_pressure',3)}/5, career_concerns={features.get('future_career_concerns',3)}/5.
Give exactly 3 bullet points of personalised, actionable coping advice. Plain text only. Under 35 words each."""
    try:
        resp = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            max_tokens=320, temperature=0.7
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        app.logger.warning(f"Groq error: {e}")
        return FALLBACK.get(label, FALLBACK["Medium"])

def get_risk_escalation(base_label, exam_date_str, study_load, sleep_quality):
    try:
        exam_date = datetime.strptime(exam_date_str, "%Y-%m-%d").date()
        days_left = (exam_date - datetime.now().date()).days
    except (ValueError, TypeError):
        return {"adjusted_label": base_label, "days_to_exam": None,
                "escalated": False, "reason": "No exam date provided."}
    if days_left < 0:
        return {"adjusted_label": base_label, "days_to_exam": days_left,
                "escalated": False, "reason": "Exam date is in the past."}
    idx = LEVELS.index(base_label)
    escalated, reason = False, "No escalation — exam not imminent."
    if days_left <= 3:
        new_idx = min(idx + 1, 2)
        escalated = new_idx > idx
        reason = (f"ESCALATED: Exam in {days_left} day(s). {base_label} → {LEVELS[new_idx]}."
                  if escalated else f"Already at max risk. Exam in {days_left} day(s).")
        idx = new_idx
    elif days_left <= 7 and (study_load >= 4 or sleep_quality <= 2):
        new_idx = min(idx + 1, 2)
        if new_idx > idx:
            escalated = True
            trigger = "high study load" if study_load >= 4 else "poor sleep"
            reason = f"ESCALATED: Exam in {days_left} days + {trigger}. {base_label} → {LEVELS[new_idx]}."
            idx = new_idx
    return {"adjusted_label": LEVELS[idx], "days_to_exam": days_left,
            "escalated": escalated, "reason": reason}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data     = request.get_json(force=True)
        features = {k: float(data.get(k, 3)) for k in FEATURES}
        X_in     = pd.DataFrame([[features[k] for k in FEATURES]], columns=FEATURES)
        X_scaled = scaler.transform(X_in)
        pred     = clf.predict(X_scaled)[0]
        proba    = clf.predict_proba(X_scaled)[0]
        label    = LABEL_MAP[int(pred)]
        proba_dict = {LABEL_MAP[i]: round(float(p), 4) for i, p in enumerate(proba)}
        coping     = get_coping_strategy(label, features)
        exam_date  = data.get("exam_date", "")
        escalation = get_risk_escalation(
            label, exam_date,
            int(features.get("study_load", 3)),
            int(features.get("sleep_quality", 3))
        )
        return jsonify({
            "anxiety_level":   label,
            "probabilities":   proba_dict,
            "coping_strategy": coping,
            "risk_escalation": escalation,
            "status":          "success"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
