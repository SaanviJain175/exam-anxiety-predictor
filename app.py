import os
import pickle
import json
import numpy as np
from flask import Flask, request, jsonify, render_template
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
LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}
LEVELS    = ["Low", "Medium", "High"]

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
        "• Progressive muscle relaxation nightly: tense each muscle group foot-to-shoulders."
    )
}


def train_and_save():
    print("Training model for first time...")
    import urllib.request
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import pandas as pd

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

    df = pd.read_csv(DATASET_PATH)
    for col in FEATURES:
        if col not in df.columns:
            df[col] = np.random.randint(1, 6, len(df))
    if 'stress_level' not in df.columns:
        df['stress_level'] = np.random.choice([0,1,2], len(df))

    df = df[FEATURES + ['stress_level']].dropna()
    X = df[FEATURES].values
    y = df['stress_level'].values

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


try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        with open(MODEL_PATH,  "rb") as f: clf    = pickle.load(f)
        with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)
        print("Model loaded!")
    else:
        clf, scaler = train_and_save()
except Exception as e:
    print(f"Model error: {e}, retraining...")
    clf, scaler = train_and_save()

try:
    from groq import Groq
    GROQ_KEY    = os.environ.get("GROQ_API_KEY", "")
    groq_client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None
except Exception:
    groq_client = None


def get_coping_strategy(label, features):
    if not groq_client:
        return FALLBACK.get(label, FALLBACK["Medium"])
    try:
        prompt = f"""A student has {label} exam anxiety.
Profile: sleep={features.get('sleep_quality')}/5, headache={features.get('headache')}/5,
academic={features.get('academic_performance')}/5, study_load={features.get('study_load')}/5,
extracurricular={features.get('extracurricular_activities')}/5,
peer_pressure={features.get('peer_pressure',3)}/5, career_concerns={features.get('future_career_concerns',3)}/5.
Give exactly 3 bullet points of personalised, actionable coping advice. Plain text only. Under 35 words each."""
        resp = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            max_tokens=320, temperature=0.7
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return FALLBACK.get(label, FALLBACK["Medium"])


def get_timetable_risk(base_label, exam_schedule, study_load, sleep_quality):
    today   = datetime.now().date()
    results = []

    for exam in exam_schedule:
        try:
            exam_date = datetime.strptime(exam["date"], "%Y-%m-%d").date()
            days_left = (exam_date - today).days
            subject   = exam.get("subject", "Exam")

            if days_left < 0:
                results.append({"subject": subject, "date": exam["date"],
                    "days_left": days_left, "status": "past",
                    "message": f"{subject} exam has passed.",
                    "escalated": False, "adjusted_label": base_label})
                continue

            idx = LEVELS.index(base_label)
            escalated = False
            message   = ""

            if days_left <= 3:
                new_idx   = min(idx + 1, 2)
                escalated = new_idx > idx
                message   = (f"🚨 CRITICAL: {subject} in {days_left} day(s)! Risk escalated {base_label} → {LEVELS[new_idx]}."
                             if escalated else f"⚠️ {subject} in {days_left} day(s). Already at maximum risk.")
                idx = new_idx
            elif days_left <= 5:
                if study_load >= 4 or sleep_quality <= 2:
                    new_idx   = min(idx + 1, 2)
                    escalated = new_idx > idx
                    trigger   = "high study load" if study_load >= 4 else "poor sleep"
                    message   = (f"⚠️ ESCALATED: {subject} in {days_left} days + {trigger}. {base_label} → {LEVELS[new_idx]}."
                                 if escalated else f"⚠️ {subject} in {days_left} days. Already at max risk.")
                    idx = new_idx
                else:
                    message = f"📅 {subject} in {days_left} days. Monitor your stress levels."
            elif days_left <= 7:
                message = f"📌 {subject} in {days_left} days. Start preparation now."
            else:
                message = f"✅ {subject} in {days_left} days. Stay consistent."

            results.append({"subject": subject, "date": exam["date"],
                "days_left": days_left, "status": "upcoming",
                "message": message, "escalated": escalated,
                "adjusted_label": LEVELS[idx]})
        except Exception:
            continue

    all_labels  = [r["adjusted_label"] for r in results if r["status"] == "upcoming"]
    final_label = base_label
    if "High" in all_labels:
        final_label = "High"
    elif "Medium" in all_labels and base_label == "Low":
        final_label = "Medium"

    return {"base_label": base_label, "final_label": final_label,
            "exam_results": results, "overall_escalated": final_label != base_label}


def generate_study_plan(exam_schedule, study_load, sleep_quality, subjects_input):
    today    = datetime.now().date()
    upcoming = []

    for exam in exam_schedule:
        try:
            exam_date = datetime.strptime(exam["date"], "%Y-%m-%d").date()
            days_left = (exam_date - today).days
            if 0 < days_left <= 30:
                upcoming.append({"subject": exam.get("subject", "Subject"),
                    "date": exam["date"], "days_left": days_left, "exam_date": exam_date})
        except Exception:
            continue

    if not upcoming:
        return {"error": "No upcoming exams within 30 days to plan for."}

    upcoming.sort(key=lambda x: x["days_left"])

    if study_load >= 4 and sleep_quality <= 2:
        session_mins = 45; break_mins = 15; sessions_day = 4; daily_hours = 4
    elif study_load >= 3:
        session_mins = 50; break_mins = 10; sessions_day = 3; daily_hours = 3
    else:
        session_mins = 25; break_mins = 5;  sessions_day = 4; daily_hours = 2

    plan         = []
    all_subjects = [e["subject"] for e in upcoming]

    for exam in upcoming:
        days_available = max(1, exam["days_left"] - 1)
        subject        = exam["subject"]
        day_plan       = []

        for i in range(min(days_available, 14)):
            date          = today + timedelta(days=i)
            sessions      = []
            other_subjects = [s for s in all_subjects if s != subject]

            for j in range(sessions_day):
                if j == 0 or (exam["days_left"] <= 3 and j <= 1):
                    sub = subject
                elif other_subjects:
                    sub = other_subjects[(i + j) % len(other_subjects)]
                else:
                    sub = subject

                start_hour = 9 + (j * (session_mins + break_mins)) // 60
                start_min  = (j * (session_mins + break_mins)) % 60
                sessions.append({"subject": sub, "duration": f"{session_mins} mins",
                    "time": f"{start_hour:02d}:{start_min:02d}", "type": "Focus Session"})

            day_plan.append({"day": i + 1, "date": date.strftime("%d %b %Y"),
                "weekday": date.strftime("%A"), "sessions": sessions,
                "note": "⚡ Revision only" if i == days_available - 1 else
                        "📖 Deep focus day" if exam["days_left"] <= 3 else "✅ Regular study day"})

        plan.append({"exam_subject": subject, "exam_date": exam["date"],
            "days_left": exam["days_left"], "daily_hours": daily_hours,
            "session_mins": session_mins, "break_mins": break_mins, "schedule": day_plan})

    return {"plan": plan, "daily_hours": daily_hours, "session_mins": session_mins,
        "break_mins": break_mins,
        "tip": ("⚠️ High load + poor sleep. Keep sessions short with frequent breaks."
                if study_load >= 4 and sleep_quality <= 2 else
                "💡 Study in focused blocks. Take breaks seriously — they improve retention.")}


TRACKING_FILE = os.path.join(BASE, "tracking_data.json")

def load_tracking():
    if os.path.exists(TRACKING_FILE):
        try:
            with open(TRACKING_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_tracking(data):
    try:
        with open(TRACKING_FILE, "w") as f:
            json.dump(data, f)
    except Exception:
        pass

def add_tracking_entry(features, label, escalated_label):
    history = load_tracking()
    entry   = {"date": datetime.now().strftime("%Y-%m-%d"),
               "time": datetime.now().strftime("%H:%M"),
               "base_label": label, "escalated_label": escalated_label,
               "sleep_quality": features.get("sleep_quality", 3),
               "study_load": features.get("study_load", 3),
               "headache": features.get("headache", 2)}
    history.append(entry)
    history = history[-30:]
    save_tracking(history)
    return history

def get_pattern_analysis(history):
    if len(history) < 3:
        return {"message": "Track at least 3 sessions to see patterns.", "entries": history}

    recent       = history[-7:]
    high_count   = sum(1 for e in recent if e["escalated_label"] == "High")
    medium_count = sum(1 for e in recent if e["escalated_label"] == "Medium")
    avg_sleep    = sum(e.get("sleep_quality", 3) for e in recent) / len(recent)
    avg_load     = sum(e.get("study_load", 3) for e in recent) / len(recent)

    alerts = []
    if high_count >= 3:
        alerts.append("🚨 High anxiety in 3+ of last 7 sessions. Please speak to a counsellor.")
    if avg_sleep < 2.5:
        alerts.append("😴 Sleep quality consistently poor. Prioritise sleep this week.")
    if avg_load > 4:
        alerts.append("📚 Study load very high. Consider reducing optional commitments.")
    if not alerts:
        alerts.append("✅ Anxiety levels look manageable. Keep maintaining your routine.")

    return {"total_entries": len(history), "recent_entries": recent,
            "high_count": high_count, "medium_count": medium_count,
            "avg_sleep": round(avg_sleep, 1), "avg_load": round(avg_load, 1),
            "alerts": alerts, "entries": history}


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data     = request.get_json(force=True)
        features = {k: float(data.get(k, 3)) for k in FEATURES}
        X_in     = np.array([[features[k] for k in FEATURES]])
        X_scaled = scaler.transform(X_in)
        pred     = clf.predict(X_scaled)[0]
        proba    = clf.predict_proba(X_scaled)[0]
        label    = LABEL_MAP[int(pred)]

        proba_dict     = {LABEL_MAP[i]: round(float(p), 4) for i, p in enumerate(proba)}
        coping         = get_coping_strategy(label, features)
        exam_schedule  = data.get("exam_schedule", [])
        timetable_risk = get_timetable_risk(label, exam_schedule,
                            int(features.get("study_load", 3)),
                            int(features.get("sleep_quality", 3)))
        final_label = timetable_risk["final_label"]

        study_plan = None
        if exam_schedule:
            study_plan = generate_study_plan(exam_schedule,
                            int(features.get("study_load", 3)),
                            int(features.get("sleep_quality", 3)),
                            [e.get("subject", "") for e in exam_schedule])

        history  = add_tracking_entry(features, label, final_label)
        patterns = get_pattern_analysis(history)

        return jsonify({"anxiety_level": label, "final_label": final_label,
            "probabilities": proba_dict, "coping_strategy": coping,
            "timetable_risk": timetable_risk, "study_plan": study_plan,
            "patterns": patterns, "status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/history", methods=["GET"])
def get_history():
    history  = load_tracking()
    patterns = get_pattern_analysis(history)
    return jsonify({"history": history, "patterns": patterns})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
