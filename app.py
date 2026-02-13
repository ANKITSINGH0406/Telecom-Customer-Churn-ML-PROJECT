import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model and scaler
model = pickle.load(open("churn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("📞 Telecom Customer Churn Prediction")

st.write("Enter customer details to predict churn.")

# Assuming `X` from your notebook (df_processed.drop('Churn', axis=1)) holds the column names in correct order
# In a real app, you would save X.columns or feature names during training and load them here.
# For now, let's reconstruct the inputs based on the original X.columns structure.

# Define the input fields for all 30 features in the correct order as they appear in X
# This requires careful mapping of Streamlit inputs to the model's expected features.

# Helper function to get default values for number inputs
def get_default_numerical_value(feature_name):
    if feature_name == 'tenure': return 32 # Average tenure from df.describe()
    if feature_name == 'MonthlyCharges': return 64.76 # Average monthly charges
    if feature_name == 'TotalCharges': return 2283.3 # Average total charges
    return 0.0 # Default for other numerical features if any

# Create input dictionary that mirrors the structure of X
input_dict = {}

# Binary features (0 or 1)
input_dict['gender'] = st.selectbox('Gender', ['Female', 'Male'], format_func=lambda x: 'Male' if x == 'Male' else 'Female')
input_dict['SeniorCitizen'] = st.selectbox('Senior Citizen', ['No', 'Yes'])
input_dict['Partner'] = st.selectbox('Partner', ['No', 'Yes'])
input_dict['Dependents'] = st.selectbox('Dependents', ['No', 'Yes'])
input_dict['PhoneService'] = st.selectbox('Phone Service', ['No', 'Yes'])
input_dict['PaperlessBilling'] = st.selectbox('Paperless Billing', ['No', 'Yes'])

# Numerical features
input_dict['tenure'] = st.number_input('Tenure (Months)', min_value=0, max_value=72, value=int(get_default_numerical_value('tenure')))
input_dict['MonthlyCharges'] = st.number_input('Monthly Charges', min_value=18.25, max_value=118.75, value=float(get_default_numerical_value('MonthlyCharges')))
input_dict['TotalCharges'] = st.number_input('Total Charges', min_value=0.0, max_value=8684.8, value=float(get_default_numerical_value('TotalCharges')))

# Multi-class categorical features (one-hot encoded)
# The `drop_first=True` means 'No' or the first category will be represented by all zeros

# MultipleLines
multiple_lines_choice = st.selectbox('Multiple Lines', ['No phone service', 'No', 'Yes'])
input_dict['MultipleLines_No phone service'] = 1 if multiple_lines_choice == 'No phone service' else 0
input_dict['MultipleLines_Yes'] = 1 if multiple_lines_choice == 'Yes' else 0

# InternetService
internet_service_choice = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
input_dict['InternetService_Fiber optic'] = 1 if internet_service_choice == 'Fiber optic' else 0
input_dict['InternetService_No'] = 1 if internet_service_choice == 'No' else 0

# OnlineSecurity
online_security_choice = st.selectbox('Online Security', ['No internet service', 'No', 'Yes'])
input_dict['OnlineSecurity_No internet service'] = 1 if online_security_choice == 'No internet service' else 0
input_dict['OnlineSecurity_Yes'] = 1 if online_security_choice == 'Yes' else 0

# OnlineBackup
online_backup_choice = st.selectbox('Online Backup', ['No internet service', 'No', 'Yes'])
input_dict['OnlineBackup_No internet service'] = 1 if online_backup_choice == 'No internet service' else 0
input_dict['OnlineBackup_Yes'] = 1 if online_backup_choice == 'Yes' else 0

# DeviceProtection
device_protection_choice = st.selectbox('Device Protection', ['No internet service', 'No', 'Yes'])
input_dict['DeviceProtection_No internet service'] = 1 if device_protection_choice == 'No internet service' else 0
input_dict['DeviceProtection_Yes'] = 1 if device_protection_choice == 'Yes' else 0

# TechSupport
tech_support_choice = st.selectbox('Tech Support', ['No internet service', 'No', 'Yes'])
input_dict['TechSupport_No internet service'] = 1 if tech_support_choice == 'No internet service' else 0
input_dict['TechSupport_Yes'] = 1 if tech_support_choice == 'Yes' else 0

# StreamingTV
streaming_tv_choice = st.selectbox('Streaming TV', ['No internet service', 'No', 'Yes'])
input_dict['StreamingTV_No internet service'] = 1 if streaming_tv_choice == 'No internet service' else 0
input_dict['StreamingTV_Yes'] = 1 if streaming_tv_choice == 'Yes' else 0

# StreamingMovies
streaming_movies_choice = st.selectbox('Streaming Movies', ['No internet service', 'No', 'Yes'])
input_dict['StreamingMovies_No internet service'] = 1 if streaming_movies_choice == 'No internet service' else 0
input_dict['StreamingMovies_Yes'] = 1 if streaming_movies_choice == 'Yes' else 0

# Contract
contract_choice = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
input_dict['Contract_One year'] = 1 if contract_choice == 'One year' else 0
input_dict['Contract_Two year'] = 1 if contract_choice == 'Two year' else 0

# PaymentMethod
payment_method_choice = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
input_dict['PaymentMethod_Credit card (automatic)'] = 1 if payment_method_choice == 'Credit card (automatic)' else 0
input_dict['PaymentMethod_Electronic check'] = 1 if payment_method_choice == 'Electronic check' else 0
input_dict['PaymentMethod_Mailed check'] = 1 if payment_method_choice == 'Mailed check' else 0


# When the predict button is pressed
if st.button("Predict"):
    # Convert categorical inputs to numerical values (0 or 1)
    input_dict['gender'] = 1 if input_dict['gender'] == 'Male' else 0
    input_dict['SeniorCitizen'] = 1 if input_dict['SeniorCitizen'] == 'Yes' else 0
    input_dict['Partner'] = 1 if input_dict['Partner'] == 'Yes' else 0
    input_dict['Dependents'] = 1 if input_dict['Dependents'] == 'Yes' else 0
    input_dict['PhoneService'] = 1 if input_dict['PhoneService'] == 'Yes' else 0
    input_dict['PaperlessBilling'] = 1 if input_dict['PaperlessBilling'] == 'Yes' else 0

    # Create DataFrame from input_dict
    # It's crucial to ensure the column order is exactly the same as X used for training
    # For this, we'll assume `X` (from the notebook kernel) is available and extract its columns
    # In a deployed app, you'd save X.columns during training and load them here.

    # For the purpose of making this runnable, let's manually define the feature order
    # based on the `X.columns` from the notebook's kernel state.
    feature_order = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                     'PhoneService', 'MultipleLines_No phone service', 'MultipleLines_Yes',
                     'InternetService_Fiber optic', 'InternetService_No',
                     'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
                     'OnlineBackup_No internet service', 'OnlineBackup_Yes',
                     'DeviceProtection_No internet service', 'DeviceProtection_Yes',
                     'TechSupport_No internet service', 'TechSupport_Yes',
                     'StreamingTV_No internet service', 'StreamingTV_Yes',
                     'StreamingMovies_No internet service', 'StreamingMovies_Yes',
                     'Contract_One year', 'Contract_Two year',
                     'PaperlessBilling', 'PaymentMethod_Credit card (automatic)',
                     'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
                     'MonthlyCharges', 'TotalCharges']

    # Initialize with zeros for all one-hot encoded columns that might not be directly set
    processed_input = {col: 0 for col in feature_order}
    for key, value in input_dict.items():
        if key in processed_input: # Only update if the key is in our expected features
            processed_input[key] = value

    # Create a DataFrame with a single row, ensuring column order
    input_df_final = pd.DataFrame([processed_input], columns=feature_order)

    # Scale numerical features using the loaded scaler
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges'] # Should match numerical_features from training
    input_df_final[numerical_features] = scaler.transform(input_df_final[numerical_features])

    # Make prediction
    prediction = model.predict(input_df_final)
    prediction_proba = model.predict_proba(input_df_final)[:, 1]

    if prediction[0] == 1:
        st.error(f"⚠️ Customer is likely to CHURN (Probability: {prediction_proba[0]:.2f})")
    else:
        st.success(f"✅ Customer is NOT likely to churn (Probability: {prediction_proba[0]:.2f})")
