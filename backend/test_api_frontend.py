import requests

base_url = "http://localhost:8000"

payloads = {
    "heart": {"age":52, "sex":1, "bp":120, "cholesterol":220, "maxHR":150, "chestPain":0, "fastingBS":0, "stDepression":1},
    "diabetes": {"age":45, "bmi":26, "glucose":110, "insulin":80, "hba1c":5, "familyHistory":0, "pregnancies":0, "skinThickness":20},
    "liver": {"age":40, "totalBilirubin":10, "directBilirubin":4, "alt":40, "ast":38, "albumin":4, "totalProteins":6},
    "kidney": {"age":55, "bloodUrea":40, "creatinine":2, "hemoglobin":12, "sodium":135, "potassium":4, "hypertension":0, "dm":0},
    "cancer": {"radiusMean":14, "textureMean":19, "perimeterMean":90, "areaMean":600, "smoothnessMean":10, "concavityMean":10},
    "respiratory": {"fev1":2.5, "fvc":3.5, "smokingStatus":0, "packYears":0, "dyspneaGrade":0},
    "parkinson": {"jitter":0.03, "shimmer":0.02, "nhr":0.025, "hnr":20.0, "rpde":0.45, "dfa":0.7},
    "stroke": {"age":67, "hypertension":0, "heartDisease":0, "bmi":28, "avgGlucose":100, "smokingStatus":0},
    "thyroid": {"age":42, "tsh":1.5, "t3":2.0, "tt4":100, "t4u":1.0, "onThyroxine":0, "goitre":0}
}

for disease, payload in payloads.items():
    print(f"\nTesting {disease} with frontend payload...")
    try:
        res = requests.post(f"{base_url}/predict/{disease}", json=payload)
        if res.status_code == 200:
            print("SUCCESS:", res.json())
        else:
            print("ERROR:", res.text)
    except Exception as e:
        print("REQUEST FAILED:", e)
