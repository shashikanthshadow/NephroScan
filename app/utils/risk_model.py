# app/utils/risk_model.py

import joblib
import numpy as np

# Load model and scaler once
model = joblib.load("models/kidney_stone_rf_model.joblib")
scaler = joblib.load("models/kidney_stone_scaler.joblib")

def predict_kidney_risk(form_data):
    """
    Predict kidney stone risk based on urine analysis inputs.
    Expects form_data with keys: gravity, ph, osmo, cond, urea, calc.
    Returns 'Kidney Stone' or 'No Kidney Stone' with explanation.
    """
    # Define expected features and their aliases for user-friendly error messages
    features = {
        'gravity': 'Urine Density',
        'ph': 'Urine Acidity',
        'osmo': 'Urine Concentration',
        'cond': 'Urine Conductivity',
        'urea': 'Urea Level',
        'calc': 'Calcium Level'
    }
    
    # Validate and collect inputs
    input_values = []
    for feature in ['gravity', 'ph', 'osmo', 'cond', 'urea', 'calc']:
        value = form_data.get(feature)
        if value is None or value == "":
            raise ValueError(f"Please enter a value for {features[feature]}")
        try:
            value = float(value)
        except ValueError:
            raise ValueError(f"{features[feature]} must be a number")
        
        # Basic range validation (based on dataset ranges)
        if feature == 'gravity' and not (1.005 <= value <= 1.030):
            raise ValueError(f"{features[feature]} should be between 1.005 and 1.030")
        elif feature == 'ph' and not (4.5 <= value <= 8.0):
            raise ValueError(f"{features[feature]} should be between 4.5 and 8.0")
        elif feature == 'osmo' and not (200 <= value <= 1200):
            raise ValueError(f"{features[feature]} should be between 200 and 1200")
        elif feature == 'cond' and not (5 <= value <= 40):
            raise ValueError(f"{features[feature]} should be between 5 and 40")
        elif feature == 'urea' and not (50 <= value <= 500):
            raise ValueError(f"{features[feature]} should be between 50 and 500")
        elif feature == 'calc' and not (1 <= value <= 10):
            raise ValueError(f"{features[feature]} should be between 1 and 10")
        
        input_values.append(value)
    
    # Prepare input for model
    input_array = np.array([input_values])
    input_scaled = scaler.transform(input_array)
    
    # Predict
    prediction = model.predict(input_scaled)
    result = 'Kidney Stone' if prediction[0] == 1 else 'No Kidney Stone'
    
    # Generate user-friendly explanation
    explanation = []
    if result == 'Kidney Stone':
        explanation.append("The urine test suggests you may have a kidney stone.")
        explanation.append("Possible reasons (common in India):")
        if input_values[0] > 1.020:  # High gravity
            explanation.append("- Low water intake, especially in hot weather")
        if input_values[1] < 6.0:  # Low pH
            explanation.append("- Diet high in acidic foods (e.g., tea, spinach)")
        if input_values[5] > 5.0:  # High calcium
            explanation.append("- High intake of oxalate-rich foods (e.g., nuts, tea)")
        explanation.append("Please see a doctor for tests and advice (e.g., drink more water, reduce oxalate foods).")
    else:
        explanation.append("The urine test suggests you are unlikely to have a kidney stone.")
        explanation.append("To stay safe, drink plenty of water and eat a balanced diet.")
    
    return result, explanation
