# HealthPredict AI – Multi-Disease Prediction System

![HealthPredict AI Logo](assets/screenshots/home.png)

## 📋 Overview

**HealthPredict AI** is a machine learning-based health prediction platform that estimates the risk of multiple diseases using trained models and an interactive web interface.

The system integrates a FastAPI backend with multiple pre-trained machine learning models and a responsive frontend to provide real-time predictions based on user input.

---

## ✨ Features

* Multi-disease prediction (Heart, Diabetes, Liver, Kidney, Cancer, Respiratory, Parkinson, Stroke, Thyroid)
* FastAPI-based backend for high-performance predictions
* Interactive web interface using HTML, CSS, and JavaScript
* Real-time risk prediction
* Multiple trained ML models
* API-based architecture
* Health check endpoint for backend monitoring

---

## 🧬 Models & Accuracy

| Disease             | Algorithm          | Accuracy |
| ------------------- | ------------------ | -------- |
| Heart Disease       | Stacking Ensemble  | ~88%     |
| Diabetes            | XGBoost Classifier | ~84%     |
| Liver Disease       | Voting Ensemble    | ~90.59%  |
| Kidney Disease      | Voting Ensemble    | ~78.75%  |
| Breast Cancer       | Voting Ensemble    | ~96.49%  |
| Respiratory Disease | Voting Ensemble    | ~94.78%  |
| Parkinson's Disease | Voting Ensemble    | ~92.31%  |
| Stroke              | Voting Ensemble    | ~95.21%  |
| Thyroid Disorder    | Voting Ensemble    | ~94%     |

---

## 🎨 Screenshots

![Home UI](assets/screenshots/home.png)
![Prediction UI](assets/screenshots/prediction.png)
![Result UI](assets/screenshots/result.png)

---

## 🛠️ Tech Stack

### Backend

* Python
* FastAPI
* Scikit-learn
* XGBoost
* Joblib
* Pandas
* NumPy

### Frontend

* HTML
* CSS
* JavaScript

---

## 📚 Dataset Sources

* Heart Disease Dataset — UCI / Kaggle
* Diabetes Dataset — Pima Indians Diabetes Dataset
* Liver Disease Dataset — Kaggle
* Chronic Kidney Disease Dataset — UCI
* Breast Cancer Dataset — UCI
* Asthma/Respiratory Dataset — Kaggle
* Parkinson’s Dataset — UCI
* Stroke Dataset — Kaggle
* Thyroid Dataset — UCI

---

## 📦 Installation

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Backend runs on:

```
http://127.0.0.1:8000
```

API Docs:

```
http://127.0.0.1:8000/docs
```

---

### Frontend Setup

```bash
cd frontend
```

Open:

```
healthpredict-ai.html
```

---

## 🔌 API Endpoints

### Health Check

```
GET /health
```

Response:

```json
{
  "message": "HealthPredict AI Backend Running",
  "loaded_models": ["heart", "diabetes", "liver", "kidney", "cancer", "respiratory", "parkinson", "stroke", "thyroid"]
}
```

---

### Prediction Endpoint Example

```
POST /predict/heart
```

Response:

```json
{
  "disease": "heart",
  "prediction": 1,
  "label": "High Risk",
  "confidence": 92.45
}
```

---

## 🚀 Usage

1. Start backend server
2. Open frontend
3. Select disease
4. Enter details
5. Click predict

---

## 📊 Project Structure

```
HealthPredict-AI/
│
├── frontend/
│   └── healthpredict-ai.html
│
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   └── models/
│       ├── *.pkl
│
├── assets/
│   └── screenshots/
│
├── docs/
│
├── README.md
├── .gitignore
└── LICENSE
```

---

## 🔮 Future Scope

* AI-based health assistant
* Mobile application
* Real-time health monitoring
* Wearable device integration
* Improved model accuracy

---

## 👨‍💻 Author

Omkar Dubey

---

## ⚠️ Disclaimer

This project is for educational purposes only.
It should not be used as a substitute for professional medical advice.

---

## 🙏 Acknowledgment

* Scikit-learn
* XGBoost
* FastAPI
* Open-source community

---

**Version:** 1.0
