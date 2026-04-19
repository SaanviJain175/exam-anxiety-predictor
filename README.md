# 🎓 ExamMind — Exam Anxiety Predictor
### Live AI Web Application | Python · Flask · Random Forest · LLaMA 3.1 | Saanvi Jain


## 🌐 LIVE DEMO — Click to Open the App

# 👉 [https://exam-anxiety-predictor.onrender.com](https://exam-anxiety-predictor.onrender.com)

> ⚠️ Free hosting — may take 30-50 seconds to wake up on first click. Wait for it, it works.


## 📸 App Preview

<img width="2512" height="1400" alt="image" src="https://github.com/user-attachments/assets/ee37be5a-d3ef-44fb-8d8c-d945f22213dd" />


## 🔬 What This Project Does

A fully deployed AI-powered web application that predicts a student's exam anxiety level
using Machine Learning and generates personalised interventions.


## ⚙️ Technical Stack

| Layer | Technology |
|-------|-----------|
| ML Model | Random Forest Classifier (scikit-learn) |
| AI Layer | Groq API · LLaMA 3.1 8B Instant |
| Backend | Python · Flask · Gunicorn |
| Frontend | HTML · CSS · JavaScript |
| Hosting | Render.com (free tier) |
| Dataset | Zomato Bangalore — 1,12,500 records |


## 🚀 3 Advanced Features Built

### 1. Timetable-Aware Risk Escalation Module
- Enter your college exam schedule (subject + date)
- Model automatically detects proximity to exam dates
- Student profiled as **Medium** stress → flagged **High** when exam is within 3-5 days
- Considers high study load and poor sleep as escalation triggers

### 2. Personalised Day-by-Day Study Plan Generator
- When high study load + poor sleep detected, generates full study timetable
- Distributes subjects across available days with exact time slots
- Mandatory break intervals built in (10-15 mins between sessions)
- Last day automatically set to revision only

### 3. Longitudinal Anxiety Tracking Module
- Every prediction is saved automatically
- After 3+ sessions: pattern recognition kicks in
- Alerts if High anxiety detected 3+ times in a week
- Tracks sleep quality and study load trends over time


## 🧠 ML Model Details

- **Algorithm:** Random Forest Classifier
- **Features:** Sleep quality, headache frequency, academic performance,
  study load, extracurricular activities, peer pressure, future career concerns
- **Classes:** Low / Medium / High anxiety
- **Training:** Auto-trains on first deployment using Zomato Student Stress dataset
- **Validation:** 5-fold cross validation


## 📁 Project Structure

exam-anxiety-predictor/
├── app.py                  ← Flask backend + all 3 feature modules
├── templates/
│   └── index.html          ← Full frontend UI
├── requirements.txt        ← Dependencies
├── runtime.txt             ← Python 3.11.9
├── render.yaml             ← Render deployment config
└── Procfile                ← Gunicorn start command

## 👩‍💻 About

**Saanvi Jain** | BSc Student
- 📧 sjain72006@gmail.com
