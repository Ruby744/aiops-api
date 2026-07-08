from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="AIOps Hard Drive Failure Predictor")

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

THRESHOLD = 0.2

# Input Schema
class DriveMetrics(BaseModel):
    smart_5_raw: float    #Reallocated Sectors
    smart_9_raw: float    # Power-On Hours
    smart_187_raw: float  # Uncorrectable Errors
    smart_194_raw: float  # Temperature
    smart_197_raw: float  # Pending Sectors

@app.post("/predict/")
def predict(metrics: DriveMetrics):
    # Prepare input array in the same feature order
    features = np.array([[
        metrics.smart_5_raw,
        metrics.smart_9_raw,
        metrics.smart_187_raw,
        metrics.smart_194_raw,
        metrics.smart_197_raw
    ]])
    
    scaled = scaler.transform(features)
    prob = model.predict_proba(scaled)[0][1]
    failed = int(prob >= THRESHOLD)
    
    return {
        "failure_probability": round(float(prob), 4),
        "prediction": failed,
        "status": "CRITICAL - Immediate Inspection Required" if failed else "HEALTHY",
        "threshold_used": THRESHOLD,
        "features_received": {
            "reallocated_sectors": metrics.smart_5_raw,
            "power_on_hours": metrics.smart_9_raw,
            "uncorrectable_errors": metrics.smart_187_raw,
            "temperature": metrics.smart_194_raw,
            "pending_sectors": metrics.smart_197_raw
        }
    }
    