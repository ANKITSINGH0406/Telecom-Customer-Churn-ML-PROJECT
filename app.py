import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.title("📡 Telecom Customer Churn Prediction")

# User inputs
tenure = st.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

if st.button("Predict Churn"):

    # Create input dictionary
    input_dict = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Scale numerical features
    input_df[["tenure", "MonthlyCharges", "TotalCharges"]] = scaler.transform(
        input_df[["tenure", "MonthlyCharges", "TotalCharges"]]
    )

    # Add missing columns
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[columns]

    # Prediction
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠ Customer is likely to CHURN\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Customer is likely to STAY\nProbability: {probability:.2f}")
