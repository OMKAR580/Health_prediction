import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sklearn.compose
import sklearn.impute
import numpy as np

# Patch for older models loading in newer scikit-learn
try:
    sklearn.compose._column_transformer._RemainderColsList = type('_RemainderColsList', (list,), {})
except AttributeError:
    pass
    
try:
    sklearn.impute.SimpleImputer._fill_dtype = property(lambda self: self.statistics_.dtype if hasattr(self, 'statistics_') else object)
except AttributeError:
    pass

app = FastAPI(title="HealthPredict API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

model_files = {
    "heart": "heart_model.pkl",
    "diabetes": "diabetes_model.pkl",
    "liver": "liver_model.pkl",
    "kidney": "kidney_model.pkl",
    "cancer": "breast_cancer_model.pkl",
    "respiratory": "respiratory_model.pkl",
    "parkinson": "parkinson_model.pkl",
    "stroke": "stroke_model.pkl",
    "thyroid": "thyroid_model.pkl"
}

model_accuracy = {
    "heart": 88,
    "diabetes": 84,
    "liver": 90.59,
    "kidney": 78.75,
    "cancer": 96.49,
    "respiratory": 94.78,
    "parkinson": 92.31,
    "stroke": 95.21,
    "thyroid": 94
}

models = {}
for name, filename in model_files.items():
    filepath = os.path.join(MODEL_DIR, filename)
    if os.path.exists(filepath):
        try:
            models[name] = joblib.load(filepath)
            print(f"Loaded {filename}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")

@app.get("/health")
def health_check():
    return {"status": "online", "loaded_models": list(models.keys())}

@app.get("/model-info")
def model_info():
    return model_accuracy

def map_features(disease: str, f: dict):
    mapped = {}
    if disease == 'heart':
        mapped = {
            'Age': f.get('age', 50),
            'Sex': f.get('sex', 1),
            'ChestPainType': f.get('chestPain', 0),
            'RestingBP': f.get('bp', 120),
            'Cholesterol': f.get('cholesterol', 200),
            'FastingBS': f.get('fastingBS', 0),
            'RestingECG': 1,
            'MaxHR': f.get('maxHR', 150),
            'ExerciseAngina': 0,
            'Oldpeak': f.get('stDepression', 0.0),
            'ST_Slope': 1,
            'Cholesterol_Is_Zero': 1 if f.get('cholesterol', 200) == 0 else 0,
            'RestingBP_Is_Zero': 1 if f.get('bp', 120) == 0 else 0
        }
    elif disease == 'diabetes':
        mapped = {
            'Pregnancies': f.get('pregnancies', 0),
            'Glucose': f.get('glucose', 100),
            'BloodPressure': 70,
            'SkinThickness': f.get('skinThickness', 20),
            'Insulin': f.get('insulin', 80),
            'BMI': f.get('bmi', 25.0),
            'DiabetesPedigreeFunction': f.get('hba1c', 5) * 0.1, 
            'Age': f.get('age', 45)
        }
    elif disease == 'liver':
        mapped = {
            'Age': f.get('age', 40),
            'Gender': f.get('sex', 1),
            'BMI': 25.0,
            'AlcoholConsumption': 0,
            'Smoking': 0,
            'GeneticRisk': 0,
            'PhysicalActivity': 5,
            'Diabetes': 0,
            'Hypertension': 0,
            'LiverFunctionTest': f.get('alt', 40)
        }
    elif disease == 'kidney':
        mapped = {
            'age': f.get('age', 55),
            'bp': 80,
            'rbc': 1,
            'pc': 1,
            'pcc': 0,
            'ba': 0,
            'htn': f.get('hypertension', 0),
            'dm': f.get('dm', 0),
            'cad': 0,
            'appet': 1,
            'pe': 0,
            'ane': 0
        }
    elif disease == 'cancer':
        mapped = {
            'radius_mean': f.get('radiusMean', 14),
            'texture_mean': f.get('textureMean', 19),
            'perimeter_mean': f.get('perimeterMean', 90),
            'area_mean': f.get('areaMean', 600),
            'smoothness_mean': f.get('smoothnessMean', 10),
            'compactness_mean': 0.1,
            'concavity_mean': f.get('concavityMean', 10),
            'concave points_mean': 0.05,
            'symmetry_mean': 0.2,
            'fractal_dimension_mean': 0.06,
            'radius_se': 0.5,
            'texture_se': 1.0,
            'perimeter_se': 3.0,
            'area_se': 40.0,
            'smoothness_se': 0.005,
            'compactness_se': 0.02,
            'concavity_se': 0.03,
            'concave points_se': 0.01,
            'symmetry_se': 0.02,
            'fractal_dimension_se': 0.003,
            'radius_worst': f.get('radiusMean', 14) * 1.2,
            'texture_worst': f.get('textureMean', 19) * 1.2,
            'perimeter_worst': f.get('perimeterMean', 90) * 1.2,
            'area_worst': f.get('areaMean', 600) * 1.2,
            'smoothness_worst': f.get('smoothnessMean', 10) * 1.2,
            'compactness_worst': 0.2,
            'concavity_worst': f.get('concavityMean', 10) * 1.2,
            'concave points_worst': 0.1,
            'symmetry_worst': 0.3,
            'fractal_dimension_worst': 0.08
        }
    elif disease == 'respiratory':
        mapped = {
            'Age': f.get('age', 50),
            'Gender': f.get('sex', 1),
            'Ethnicity': 0,
            'EducationLevel': 1,
            'BMI': 25.0,
            'Smoking': f.get('smokingStatus', 0),
            'PhysicalActivity': 5,
            'DietQuality': 5,
            'SleepQuality': 5,
            'PollutionExposure': 5,
            'PollenExposure': 5,
            'DustExposure': 5,
            'PetAllergy': 0,
            'FamilyHistoryAsthma': 0,
            'HistoryOfAllergies': 0,
            'Eczema': 0,
            'HayFever': 0,
            'GastroesophagealReflux': 0,
            'LungFunctionFEV1': f.get('fev1', 2.5),
            'LungFunctionFVC': f.get('fvc', 3.5),
            'Wheezing': 0,
            'ShortnessOfBreath': f.get('dyspneaGrade', 0),
            'ChestTightness': 0,
            'Coughing': 0,
            'NighttimeSymptoms': 0,
            'ExerciseInduced': 0,
            'DoctorInCharge': 0
        }
    elif disease == 'parkinson':
        mapped = {
            'mdvp:fo(hz)': 150.0,
            'mdvp:fhi(hz)': 200.0,
            'mdvp:flo(hz)': 100.0,
            'mdvp:jitter(%)': f.get('jitter', 0.03),
            'mdvp:jitter(abs)': 0.00005,
            'mdvp:rap': 0.01,
            'mdvp:ppq': 0.01,
            'jitter:ddp': 0.03,
            'mdvp:shimmer': f.get('shimmer', 0.02),
            'mdvp:shimmer(db)': 0.2,
            'shimmer:apq3': 0.01,
            'shimmer:apq5': 0.01,
            'mdvp:apq': 0.01,
            'shimmer:dda': 0.03,
            'nhr': f.get('nhr', 0.02),
            'hnr': f.get('hnr', 20.0),
            'rpde': f.get('rpde', 0.5),
            'dfa': f.get('dfa', 0.7),
            'spread1': -5.0,
            'spread2': 0.2,
            'd2': 2.0,
            'ppe': 0.2
        }
    elif disease == 'stroke':
        mapped = {
            'gender': f.get('sex', 1),
            'age': f.get('age', 60),
            'hypertension': f.get('hypertension', 0),
            'heart_disease': f.get('heartDisease', 0),
            'ever_married': 1,
            'work_type': 2,
            'residence_type': 1,
            'avg_glucose_level': f.get('avgGlucose', 100),
            'bmi': f.get('bmi', 25.0),
            'smoking_status': f.get('smokingStatus', 0)
        }
    elif disease == 'thyroid':
        mapped = {
            'age': f.get('age', 40),
            'sex': f.get('sex', 1),
            'on_thyroxine': f.get('onThyroxine', 0),
            'query_on_thyroxine': 0,
            'on_antithyroid_meds': 0,
            'sick': 0,
            'pregnant': 0,
            'thyroid_surgery': 0,
            'i131_treatment': 0,
            'query_hypothyroid': 0,
            'query_hyperthyroid': 0,
            'lithium': 0,
            'goitre': f.get('goitre', 0),
            'tumor': 0,
            'hypopituitary': 0,
            'psych': 0,
            'tsh_measured': 1,
            'tsh': f.get('tsh', 1.5),
            't3_measured': 1,
            't3': f.get('t3', 2.0),
            'tt4_measured': 1,
            'tt4': f.get('tt4', 100),
            't4u_measured': 1,
            't4u': f.get('t4u', 1.0),
            'fti_measured': 1,
            'fti': 100.0,
            'tbg': 0.0,
            'referral_source': 0
        }
    else:
        mapped = f
    return mapped

@app.post("/predict/{disease}")
def predict_endpoint(disease: str, data: dict):
    if disease not in models:
        raise HTTPException(status_code=500, detail=f"Model for {disease} is not loaded.")
    
    model = models[disease]
    mapped_features = map_features(disease, data)
    
    df = pd.DataFrame([mapped_features])
    
    try:
        if hasattr(model, "predict"):
            pred = model.predict(df)[0]
        else:
            raise HTTPException(status_code=500, detail="Model does not have a predict method")
            
        prediction = 1 if pred in [1, "1", "High Risk", True, "Yes"] else 0
        
        confidence = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(df)[0]
                if len(proba) > int(prediction):
                    confidence = float(proba[int(prediction)] * 100)
                else:
                    confidence = float(proba[-1] * 100)
                confidence = round(confidence, 2)
            except Exception:
                confidence = None
            
        label = "High Risk" if prediction == 1 else "Low Risk"
        
        return {
            "disease": disease,
            "prediction": prediction,
            "label": label,
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
