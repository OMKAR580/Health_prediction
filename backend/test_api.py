import requests

endpoints = [
    "heart", "diabetes", "liver", "kidney", "cancer", "respiratory", "parkinson", "stroke", "thyroid"
]

base_url = "http://localhost:8000"

def test_endpoints():
    try:
        health = requests.get(f"{base_url}/health")
        print("Health check:", health.json())
    except Exception as e:
        print("Health check failed:", e)
        return

    for disease in endpoints:
        print(f"\nTesting {disease}...")
        try:
            # We'll just send an empty dict to see if the map_features uses defaults and predicts without error
            res = requests.post(f"{base_url}/predict/{disease}", json={})
            print(res.status_code)
            if res.status_code == 200:
                print("SUCCESS:", res.json())
            else:
                print("ERROR:", res.text)
        except Exception as e:
            print("REQUEST FAILED:", e)

if __name__ == "__main__":
    test_endpoints()
