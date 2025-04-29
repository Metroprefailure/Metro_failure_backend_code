import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import datetime

# Import logic and objects from other modules
from generate import _generate_data_logic, TimeSeriesDataPoint as GenerateTimeSeriesDataPoint # Avoid name clash
from predict import _predict_logic, model as loaded_model, scaler as loaded_scaler, app_startup_error as predict_app_startup_error
from shared_config import config_load_error as shared_config_load_error # Import config load error status

# --- FastAPI App ---
app = FastAPI(title="Generation and Prediction Orchestrator API")

# --- Pydantic Models ---
# Use the TimeSeriesDataPoint definition from generate.py
class CombinedResponse(BaseModel):
    generated_segment_type: str
    prediction_message: str
    data: List[GenerateTimeSeriesDataPoint] # Include the generated data in the response

# --- API Endpoint ---
@app.get("/generate_and_predict", response_model=CombinedResponse)
async def generate_and_predict_endpoint():
    # --- Pre-checks ---
    if shared_config_load_error:
        raise HTTPException(status_code=500, detail=f"Shared configuration error: {shared_config_load_error}")
    if predict_app_startup_error:
        raise HTTPException(status_code=500, detail=f"Prediction module startup error: {predict_app_startup_error}")
    if loaded_model is None or loaded_scaler is None:
        raise HTTPException(status_code=500, detail="Prediction model or scaler not loaded.")

    try:
        # --- Step 1: Generate Data ---
        segment_type, data_list = _generate_data_logic()

        # --- Step 2: Predict on Generated Data ---
        # The data_list is already a list of dicts, suitable for _predict_logic
        prediction_msg = _predict_logic(data_list, loaded_model, loaded_scaler)

        # --- Step 3: Format Response ---
        # Convert data_list back to Pydantic models for the response
        response_data = [GenerateTimeSeriesDataPoint(**item) for item in data_list]

        return CombinedResponse(
            generated_segment_type=segment_type,
            prediction_message=prediction_msg,
            data=response_data
        )

    except ValueError as e: # Catch errors from generation or prediction logic
        print(f"Orchestration Value Error: {e}")
        raise HTTPException(status_code=400, detail=f"Error during processing: {str(e)}")
    except Exception as e:
        # Log error properly in production
        print(f"Orchestration error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during orchestration: {str(e)}")


@app.get("/")
async def root():
    if shared_config_load_error or predict_app_startup_error:
         return {"status": "API is running, but encountered errors on startup.",
                 "config_error": shared_config_load_error,
                 "prediction_module_error": predict_app_startup_error}
    return {"message": "Orchestrator API is running. Call /generate_and_predict"}