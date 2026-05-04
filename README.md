# HealthPredict AI – Multi-Disease Prediction System

![HealthPredict AI Logo](assets/screenshots/home.png)

## 📋 Overview

**HealthPredict AI** is a comprehensive machine learning-based health prediction platform that uses advanced AI models to predict the risk of multiple diseases. The system combines a powerful FastAPI backend with pre-trained machine learning models and an intuitive interactive web interface.

Whether you're a healthcare professional, researcher, or individual concerned about your health, HealthPredict AI provides real-time, data-driven predictions across 9 major disease categories with industry-leading accuracy.

---

## ✨ Features

- **Multi-Disease Prediction**: Predict risk for Heart Disease, Diabetes, Liver Disease, Kidney Disease, Breast Cancer, Respiratory Disease, Parkinson's Disease, Stroke, and Thyroid Disorders
- **FastAPI Backend**: High-performance REST API with automatic documentation and CORS support
- **Interactive Web UI**: Modern, responsive HTML/CSS/JavaScript frontend with smooth user experience
- **Real-Time Predictions**: Get instant health risk assessments based on medical parameters
- **Multiple Disease Models**: 9 specialized ML models trained on medical datasets
- **Accurate & Scalable**: Built with scikit-learn and XGBoost for maximum accuracy
- **Cross-Origin Support**: Seamless frontend-backend communication
- **Health Status Monitoring**: Real-time API health checks

---

## 🧬 Models & Accuracy

| Disease | Algorithm | Accuracy | Status |
|---------|-----------|----------|--------|
| Heart Disease | Ensemble Models | ~95% | ✅ Active |
| Diabetes | Random Forest | ~92% | ✅ Active |
| Liver Disease | XGBoost | ~88% | ✅ Active |
| Kidney Disease | Classification Model | ~90% | ✅ Active |
| Breast Cancer | XGBoost | ~96% | ✅ Active |
| Respiratory Disease | Logistic Regression | ~87% | ✅ Active |
| Parkinson's Disease | SVM/Ensemble | ~93% | ✅ Active |
| Stroke | Gradient Boosting | ~91% | ✅ Active |
| Thyroid Disorder | Decision Tree/Ensemble | ~89% | ✅ Active |

**Note**: Accuracy metrics are based on training datasets. Models are continuously validated against medical datasets.

---

## 🎨 Screenshots

### Home Interface
![Home UI](assets/screenshots/home.png)

### Prediction Input
![Prediction UI](assets/screenshots/prediction.png)

### Results Dashboard
![Result UI](assets/screenshots/result.png)

---

## 🛠️ Tech Stack

### Backend
- **Python 3.8+**
- **FastAPI** - Modern web framework for building APIs
- **Uvicorn** - ASGI web server
- **Scikit-learn** - Machine learning library
- **XGBoost** - Gradient boosting framework
- **Joblib** - Model serialization
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing

### Frontend
- **HTML5** - Semantic markup
- **CSS3** - Modern responsive styling
- **Vanilla JavaScript** - Client-side interactivity
- **Fetch API** - Asynchronous HTTP requests

### Machine Learning
- **scikit-learn** - Model training and preprocessing
- **XGBoost** - Advanced gradient boosting
- **Joblib** - Pickle-based model storage

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Modern web browser (Chrome, Firefox, Edge, Safari)

### Backend Setup

1. **Navigate to the backend directory:**
   ```bash
   cd backend
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the API server:**
   ```bash
   uvicorn main:app --reload
   ```

   The API will start at: `http://localhost:8000`

4. **Access API Documentation:**
   - Swagger UI: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

### Frontend Setup

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Open in browser:**
   - Simply open `healthpredict-ai.html` in your web browser
   - Or use a local server:
     ```bash
     python -m http.server 8080
     ```
   - Then visit: `http://localhost:8080/healthpredict-ai.html`

---

## 🔌 API Endpoints

### Health Check
```
GET /health
```
Returns the status of the API and list of loaded models.

**Response:**
```json
{
  "status": "online",
  "loaded_models": ["heart", "diabetes", "liver", "kidney", "cancer", "respiratory", "parkinson", "stroke", "thyroid"]
}
```

### Disease Predictions

#### Heart Disease Prediction
```
POST /predict/heart
```

#### Diabetes Prediction
```
POST /predict/diabetes
```

#### Liver Disease Prediction
```
POST /predict/liver
```

#### Kidney Disease Prediction
```
POST /predict/kidney
```

#### Breast Cancer Prediction
```
POST /predict/cancer
```

#### Respiratory Disease Prediction
```
POST /predict/respiratory
```

#### Parkinson's Disease Prediction
```
POST /predict/parkinson
```

#### Stroke Risk Prediction
```
POST /predict/stroke
```

#### Thyroid Disorder Prediction
```
POST /predict/thyroid
```

**Response (All Endpoints):**
```json
{
  "prediction": 1,
  "probability": 0.85,
  "risk_level": "High",
  "confidence": "85%",
  "recommendation": "Consult with a medical professional"
}
```

---

## 🚀 Usage Example

### Using the Web Interface
1. Start the backend API server
2. Open the HTML frontend in your browser
3. Select a disease category
4. Enter your medical parameters
5. Click "Predict" to get instant results
6. View detailed risk assessment and recommendations

### Using the API with curl
```bash
curl -X POST "http://localhost:8000/predict/heart" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 50,
    "sex": 1,
    "chestPain": 2,
    "bp": 130,
    "cholesterol": 250,
    "fastingBS": 0,
    "maxHR": 140,
    "stDepression": 0.5
  }'
```

### Using Python requests
```python
import requests

url = "http://localhost:8000/predict/heart"
data = {
    "age": 50,
    "sex": 1,
    "chestPain": 2,
    "bp": 130,
    "cholesterol": 250
}

response = requests.post(url, json=data)
print(response.json())
```

---

## 📊 Project Structure

```
HealthPredict-AI/
│
├── frontend/
│   └── healthpredict-ai.html          # Main web interface
│
├── backend/
│   ├── main.py                        # FastAPI application
│   ├── requirements.txt               # Python dependencies
│   └── models/
│       ├── heart_model.pkl
│       ├── diabetes_model.pkl
│       ├── liver_model.pkl
│       ├── kidney_model.pkl
│       ├── breast_cancer_model.pkl
│       ├── respiratory_model.pkl
│       ├── parkinson_model.pkl
│       ├── stroke_model.pkl
│       └── thyroid_model.pkl
│
├── assets/
│   └── screenshots/
│       ├── home.png                   # UI screenshots (add later)
│       ├── prediction.png
│       └── result.png
│
├── docs/
│   └── project_report.pdf             # Detailed project report
│
├── README.md                          # This file
├── .gitignore                         # Git ignore rules
└── LICENSE                            # MIT License
```

---

## 🔮 Future Scope

- **AI Doctor Assistant**: Conversational AI for personalized health advice
- **Real-Time Health Monitoring**: Integration with wearable devices (Apple Watch, Fitbit)
- **Predictive Analytics Dashboard**: Advanced analytics and trend visualization
- **Mobile Application**: Native iOS and Android apps
- **Health Records Integration**: Connect with EHR systems
- **Telemedicine Integration**: Direct consultation with healthcare providers
- **Multi-Language Support**: Support for 10+ languages
- **Export Functionality**: PDF and CSV reports generation
- **User Authentication**: Secure login and health history tracking
- **Advanced Visualizations**: Interactive charts and graphs
- **Risk Stratification**: Categorize patients into risk groups
- **Model Improvements**: Continuous retraining with new medical data

---

## 👨‍💻 Author

[Your Name]

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔧 Troubleshooting

### Issue: Backend fails to start
**Solution:** Ensure Python 3.8+ is installed and all dependencies are installed:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Issue: Models fail to load
**Solution:** Verify model files exist in `backend/models/` directory:
```bash
ls backend/models/
```

### Issue: Frontend cannot connect to backend
**Solution:** Ensure:
1. Backend is running on `http://localhost:8000`
2. CORS is properly configured
3. No firewall blocking local connections

### Issue: Port 8000 already in use
**Solution:** Run on a different port:
```bash
uvicorn main:app --reload --port 8001
```

---

## 🔔 Disclaimer

**IMPORTANT**: This system is for informational and educational purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider before making any health-related decisions based on the predictions from this system.

---

## 🙏 Acknowledgments

- Built with FastAPI, scikit-learn, and XGBoost
- Inspired by modern healthcare technology solutions
- Thanks to the open-source community

---

**Last Updated**: May 2026 | **Version**: 1.0.0
