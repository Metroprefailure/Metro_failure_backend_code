# shared_config.py
import pandas as pd
import numpy as np
import os

# --- Configuration ---
MODEL_FILENAME = 'rf_model_4features.joblib'
SCALER_FILENAME = 'scaler_4features.joblib'
MODEL_DATA_FILENAME = 'model_data.csv' # Or your path

SAMPLING_INTERVAL_SECONDS = 10
POINTS_PER_MINUTE = 60 / SAMPLING_INTERVAL_SECONDS
TOP_FEATURES = ['TP2_mean', 'Oil_temperature_mean', 'TP3_mean', 'Motor_current_mean']
BASE_SENSORS = ['TP2', 'TP3', 'Oil_temperature', 'Motor_current']

# --- Load Duration Data ---
normal_durations = None
pre_failure_durations = None
config_load_error = None

try:
    if not os.path.exists(MODEL_DATA_FILENAME):
         raise FileNotFoundError(f"'{MODEL_DATA_FILENAME}' not found. Cannot load durations.")

    model_data_df = pd.read_csv(MODEL_DATA_FILENAME) # Add index_col if needed
    normal_durations = model_data_df[model_data_df['target'] == 0]['duration_minutes'].dropna()
    pre_failure_durations = model_data_df[model_data_df['target'] == 1]['duration_minutes'].dropna()

    if normal_durations.empty or pre_failure_durations.empty:
        raise ValueError("Could not extract normal or pre-failure durations from model_data.")
    print("Shared Config: Duration lists loaded successfully.")

except Exception as e:
    config_load_error = f"ERROR loading duration data in shared_config: {e}"
    print(config_load_error)


# --- Define Statistical Profiles (Using Top 4 Features Only) ---
# Ensure duration lists are pandas Series for .sample()
normal_params = {
    'TP2_mean': {'mean': 0.0161, 'std': 0.2316},
    'TP3_mean': {'mean': 8.2478, 'std': 0.1538},
    'Oil_temperature_mean': {'mean': 62.9788, 'std': 3.8592},
    'Motor_current_mean': {'mean': 0.0844, 'std': 0.3193},
    'duration_minutes': normal_durations if normal_durations is not None else pd.Series(dtype=float)
}
pre_failure_params = {
    'TP2_mean': {'mean': -0.0043, 'std': 0.1988},
    'TP3_mean': {'mean': 8.2518, 'std': 0.0528},
    'Oil_temperature_mean': {'mean': 49.5398, 'std': 1.2784},
    'Motor_current_mean': {'mean': 0.0550, 'std': 0.2127},
    'duration_minutes': pre_failure_durations if pre_failure_durations is not None else pd.Series(dtype=float)
}
avg_intra_segment_std_normal = {
    'TP2_std': 0.0624, 'TP3_std': 0.0131,
    'Oil_temperature_std': 0.0847, 'Motor_current_std': 0.0437
}
avg_intra_segment_std_pre_failure = {
    'TP2_std': 0.0210, 'TP3_std': 0.0035,
    'Oil_temperature_std': 0.0341, 'Motor_current_std': 0.0122
}