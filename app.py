# app.py — YOUR FINAL, BEAUTIFUL, WORKING STREAMLIT APP
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# MUST BE THE VERY FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="AI Soil Moisture Advisor",
    page_icon="leaf",
    layout="centered"
)


# ------------------- LOAD MODEL & PREPROCESSOR -------------------
@st.cache_resource
def load_resources():
    model = load_model('optimized_soil_moisture_ann.h5', compile=False)
    model.compile(optimizer='adam', loss='mse')
    preprocessor = joblib.load('preprocessor.pkl')
    return model, preprocessor


model, preprocessor = load_resources()

# ------------------- IDEAL RANGES -------------------
ideal_ranges = {
    'Tomato': (20, 30),
    'Rice': (25, 35),
}


# ------------------- BULLETPROOF PREDICTION FUNCTION -------------------
def monitor_soil(sample_dict, crop):
    expected_cols = ['Crop Type', 'Soil Type', 'Temperature (°C)',
                     'Humidity (%)', 'Rainfall (mm)', 'Irrigation Applied',
                     'Irrigation Amount (litres)']

    df = pd.DataFrame(columns=expected_cols, index=[0])

    for col in expected_cols:
        df.loc[0, col] = sample_dict.get(col, 0)

    df['Crop Type'] = df['Crop Type'].astype(str)
    df['Soil Type'] = df['Soil Type'].astype(str)
    df['Irrigation Applied'] = df['Irrigation Applied'].astype(str)

    X_processed = preprocessor.transform(df)
    pred = float(model.predict(X_processed, verbose=False)[0][0])

    low, high = ideal_ranges.get(crop, (15, 35))
    if pred < low:
        msg = f"Dry soil! Irrigate to reach {low}% for optimum {crop} production."
    elif pred > high:
        msg = f"Overly wet! Reduce irrigation or improve drainage."
    else:
        msg = "Optimal moisture level. No immediate action needed."

    return round(pred, 2), msg


# ------------------- STREAMLIT UI -------------------
st.title("AI-Powered Soil Moisture Monitoring")
st.markdown("### Precision Irrigation Advisor for Farmers")

st.markdown("""
Get real-time soil moisture predictions and smart irrigation recommendations  
using your trained Neural Network model.
""")

with st.form("soil_form"):
    col1, col2 = st.columns(2)

    with col1:
        crop_type = st.selectbox('Crop Type', ['Tomato', 'Rice'])
        soil_type = st.selectbox('Soil Type',
                                 ['Clay loam', 'Sand', 'Loamy sand', 'Sandy loam', 'Loam', 'Clay'])
        temperature = st.slider('Temperature (°C)', 15.0, 45.0, 28.0)

    with col2:
        humidity = st.slider('Humidity (%)', 30.0, 100.0, 65.0)
        rainfall = st.slider('Rainfall Today (mm)', 0.0, 100.0, 0.0)
        irrigation_applied = st.radio('Irrigation Today?', ['No', 'Yes'])
        irrigation_amount = st.slider('Irrigation Amount (litres)', 0.0, 50.0,
                                      0.0) if irrigation_applied == 'Yes' else 0.0

    predict_btn = st.form_submit_button("Predict & Get Advice", use_container_width=True, type="primary")

if predict_btn:
    sample = {
        'Crop Type': crop_type,
        'Soil Type': soil_type,
        'Temperature (°C)': temperature,
        'Humidity (%)': humidity,
        'Rainfall (mm)': rainfall,
        'Irrigation Applied': irrigation_applied,
        'Irrigation Amount (litres)': irrigation_amount
    }

    pred, advice = monitor_soil(sample, crop_type)

    st.markdown("---")
    st.metric("Predicted Soil Moisture", f"{pred:.2f}%")

    if "Optimal" in advice:
        st.success(f"**{advice}**")
    elif "Dry" in advice:
        st.warning(f"**{advice}**")
    else:
        st.error(f"**{advice}**")

    st.info("Model trained on real field data • Ready for farm use")

st.markdown("---")
st.caption("Made with love for agriculture | Your AI Soil Doctor")
