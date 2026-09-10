import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# Robust path handling — works locally AND on Streamlit Cloud
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

st.set_page_config(page_title="Churn Predictor", page_icon="📊")
st.title("📊 Telco Customer Churn Predictor")

# Load model artifacts using absolute paths
model = joblib.load(MODELS_DIR / "churn_model.pkl")
scaler = joblib.load(MODELS_DIR / "scaler.pkl")
feature_cols = joblib.load(MODELS_DIR / "feature_columns.pkl")

# USD to INR conversion rate (update as needed)
USD_TO_INR = 83.0

st.write("Enter customer details to predict churn probability:")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_inr = st.number_input("Monthly Charges (₹)", 0.0, 20000.0, 5800.0)
    total_inr = st.number_input("Total Charges (₹)", 0.0, 800000.0, 83000.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

with col2:
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])

if st.button("🔮 Predict Churn"):
    # Convert ₹ back to $ for the model
    monthly = monthly_inr / USD_TO_INR
    total = total_inr / USD_TO_INR

    # Build a row with defaults for unseen features
    row = {col: 0 for col in feature_cols}
    row['tenure'] = tenure
    row['MonthlyCharges'] = monthly
    row['TotalCharges'] = total

    # Set categorical encodings
    if contract == "One year":
        row['Contract_One year'] = 1
    if contract == "Two year":
        row['Contract_Two year'] = 1
    if internet == "Fiber optic":
        row['InternetService_Fiber optic'] = 1
    if internet == "No":
        row['InternetService_No'] = 1
    if payment == "Electronic check":
        row['PaymentMethod_Electronic check'] = 1
    if payment == "Mailed check":
        row['PaymentMethod_Mailed check'] = 1
    if payment == "Credit card (automatic)":
        row['PaymentMethod_Credit card (automatic)'] = 1
    if paperless == "Yes":
        row['PaperlessBilling_Yes'] = 1
    if senior == "Yes":
        row['SeniorCitizen'] = 1

    input_df = pd.DataFrame([row])[feature_cols]
    input_scaled = scaler.transform(input_df)

    prob = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")
    if prob > 0.5:
        st.error(f"⚠️ **High Churn Risk** — Probability: {prob:.1%}")
    else:
        st.success(f"✅ **Low Churn Risk** — Probability: {prob:.1%}")

    st.progress(float(prob))