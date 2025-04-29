# predict.py
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import datetime
import os

# Import shared configuration
from shared_config import TOP_FEATURES, BASE_SENSORS, MODEL_FILENAME, SCALER_FILENAME

# --- Load Model and Scaler ---
model = None
scaler = None
app_startup_error = None

try:
    if not os.path.exists(MODEL_FILENAME):
        raise FileNotFoundError(f"Model file not found: {MODEL_FILENAME}")
    if not os.path.exists(SCALER_FILENAME):
        raise FileNotFoundError(f"Scaler file not found: {SCALER_FILENAME}")

    model = joblib.load(MODEL_FILENAME)
    scaler = joblib.load(SCALER_FILENAME)
    print("Prediction API: Model and Scaler loaded successfully.")

    # Basic check: Does the scaler expect the right number of features?
    if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ != len(TOP_FEATURES):
         raise ValueError(f"Scaler expects {scaler.n_features_in_} features, but config has {len(TOP_FEATURES)} top features.")

except Exception as e:
    app_startup_error = f"FATAL ERROR loading model/scaler: {e}"
    print(app_startup_error)
    model = None # Ensure they are None on error
    scaler = None

# --- FastAPI App ---
app = FastAPI(title="Predictive Maintenance Prediction API")

# --- Pydantic Models ---
# Re-define or import if placing TimeSeriesDataPoint in shared_models.py
class TimeSeriesDataPoint(BaseModel):
    timestamp: datetime.datetime
    TP2: float
    TP3: float
    Oil_temperature: float
    Motor_current: float

class PredictionInput(BaseModel):
    data: List[TimeSeriesDataPoint]

class PredictionResult(BaseModel):
    prediction_message: str


# --- Core Prediction Logic (New Function) ---
def _predict_logic(data_list: List[dict], loaded_model, loaded_scaler) -> str:
    """Contains the core logic for making a prediction on a list of data points."""
    if not data_list:
        raise ValueError("Received empty data list for prediction.")

    # No try-except here, let the caller handle it
    # Convert list of dicts to DataFrame
    # We need to handle potential datetime objects if they weren't serialized
    try:
        received_ts_data = pd.DataFrame(data_list)
        # Ensure timestamp is parsed if it's coming as string (might happen if generated data is serialized)
        if 'timestamp' in received_ts_data.columns and not pd.api.types.is_datetime64_any_dtype(received_ts_data['timestamp']):
             received_ts_data['timestamp'] = pd.to_datetime(received_ts_data['timestamp'])
    except Exception as e:
        raise ValueError(f"Failed to create DataFrame from input data: {e}")


    calculated_features = {}
    for sensor in BASE_SENSORS:
        if sensor not in received_ts_data.columns:
            raise ValueError(f"Sensor column '{sensor}' missing in input data.")
        calculated_features[f'{sensor}_mean'] = received_ts_data[sensor].mean(skipna=True)
        if pd.isna(calculated_features[f'{sensor}_mean']):
            raise ValueError(f"Could not calculate mean for sensor '{sensor}'. Input data might be invalid.")

    feature_vector = pd.DataFrame([calculated_features], columns=TOP_FEATURES)
    scaled_features = loaded_scaler.transform(feature_vector)
    prediction = loaded_model.predict(scaled_features)

    if prediction[0] == 1:
        message = "Pre failure detected!!"
    else:
        message = "Sensor readings are normal!"

    return message


# --- API Endpoint (Uses the new logic function) ---
@app.post("/predict_segment", response_model=PredictionResult)
async def predict_data_endpoint(input_data: PredictionInput):
    if model is None or scaler is None: # Check if loading failed
         raise HTTPException(status_code=500, detail=f"API startup error or model/scaler not loaded: {app_startup_error}")

    try:
        # Convert Pydantic models back to dicts for the logic function
        data_list_dict = [item.dict() for item in input_data.data]
        message = _predict_logic(data_list_dict, model, scaler)
        return PredictionResult(prediction_message=message)

    except ValueError as e: # Catch issues like NaN means or missing columns
        print(f"Value Error during prediction: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid data or configuration: {str(e)}")
    except Exception as e:
        # Log the actual error in a real application
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error during prediction processing: {str(e)}")

@app.get("/")
async def root():
    if app_startup_error:
        return {"error": app_startup_error}
    return {"message": "Predictive Maintenance Prediction API is running. POST to /predict_segment"}