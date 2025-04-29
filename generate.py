# generate.py
import pandas as pd
import numpy as np
from scipy import stats
import random
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import datetime
import requests # Add requests library for potential future use if needed

# Import shared configuration and data
from shared_config import (
    normal_params, pre_failure_params, avg_intra_segment_std_normal,
    avg_intra_segment_std_pre_failure, BASE_SENSORS,
    SAMPLING_INTERVAL_SECONDS, POINTS_PER_MINUTE, TOP_FEATURES,
    config_load_error
)

# --- FastAPI App ---
app = FastAPI(title="Synthetic Data Generation API")

# --- Pydantic Models ---
class TimeSeriesDataPoint(BaseModel):
    timestamp: datetime.datetime
    TP2: float
    TP3: float
    Oil_temperature: float
    Motor_current: float

class GeneratedDataResponse(BaseModel):
    generated_segment_type: str # 'Normal' or 'Pre-Failure'
    data: List[TimeSeriesDataPoint] # The generated time series

# --- Helper Function ---
def generate_segment_ts(duration_min: float, target_means: dict, intra_segment_stds: dict) -> pd.DataFrame:
    num_points = int(np.ceil(duration_min * POINTS_PER_MINUTE))
    num_points = max(num_points, 2)
    time_index = pd.date_range(start=datetime.datetime.now(), periods=num_points, freq=f'{SAMPLING_INTERVAL_SECONDS}S')
    generated_ts = pd.DataFrame(index=time_index)
    generated_ts.index.name = 'timestamp'

    for sensor in BASE_SENSORS:
        target_mean = target_means.get(f'{sensor}_mean', 0)
        noise_std_dev = intra_segment_stds.get(f'{sensor}_std', 0.01)
        noise_std_dev = max(noise_std_dev, 1e-6)
        generated_ts[sensor] = stats.norm.rvs(loc=target_mean, scale=noise_std_dev, size=num_points, random_state=np.random.randint(10000))
    return generated_ts

# --- Core Generation Logic (New Function) ---
def _generate_data_logic():
    """Contains the core logic for generating one segment of data."""
    if config_load_error: # Check if shared config failed
        raise ValueError(f"Configuration error: {config_load_error}") # Raise standard error
    if normal_params['duration_minutes'] is None or pre_failure_params['duration_minutes'] is None \
       or normal_params['duration_minutes'].empty or pre_failure_params['duration_minutes'].empty:
         raise ValueError("Duration data not loaded in shared config.") # Raise standard error

    # No try-except here, let the caller handle it
    segment_type = random.choice(['Normal', 'Pre-Failure'])
    params = normal_params if segment_type == 'Normal' else pre_failure_params
    intra_segment_stds = avg_intra_segment_std_normal if segment_type == 'Normal' else avg_intra_segment_std_pre_failure

    target_characteristics = {}
    sampled_duration = params['duration_minutes'].sample(n=1, random_state=np.random.randint(10000)).iloc[0]
    target_characteristics['duration_minutes'] = sampled_duration

    for feature in TOP_FEATURES:
        mean = params[feature]['mean']
        std = params[feature]['std']
        std = max(std, 0)
        target_characteristics[feature] = stats.norm.rvs(loc=mean, scale=std, random_state=np.random.randint(10000))

    generated_ts_data = generate_segment_ts(
        target_characteristics['duration_minutes'],
        target_characteristics,
        intra_segment_stds
    )

    data_list = generated_ts_data.reset_index().to_dict(orient='records')

    # Convert datetime objects to ISO format strings for broader compatibility if needed later
    # for item in data_list:
    #     item['timestamp'] = item['timestamp'].isoformat()

    return segment_type, data_list


# --- API Endpoint (Uses the new logic function) ---
@app.get("/generate_synthetic_data", response_model=GeneratedDataResponse)
async def generate_data_endpoint():
    try:
        segment_type, data_list = _generate_data_logic()
        return GeneratedDataResponse(
            generated_segment_type=segment_type,
            data=data_list
        )
    except ValueError as e: # Catch config/duration errors
         raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # Log error properly in production
        print(f"Data generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Error during data generation: {str(e)}")

@app.get("/")
async def root():
    if config_load_error:
        return {"error": config_load_error}
    return {"message": "Synthetic Data Generation API is running. Call /generate_synthetic_data"}