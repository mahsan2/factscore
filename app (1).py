import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# --- 1. SETUP AUTH ---
credentials = {
    "usernames": {
        "zahed1": {
            "name": "Authorized User",
            "password": '$2b$12$3J.QpuvRADGvYrXTi6tkfOtmRAgLfmzxlL19o1ebpHgN5NGwUiiJy'
        }
    }
}

authenticator = stauth.Authenticate(
    credentials,
    cookie_name="app_cookie",
    key="abcdef",
    cookie_expiry_days=1
)


# --- 2. CONDITIONAL ACCESS ---
if authentication_status:
    authenticator.logout('Logout', 'sidebar')
    st.title('🔒 Factor Weight Predictor')

    # --- Upload Excel File ---
    uploaded_file = st.file_uploader("Upload your Excel file", type=['xlsx'])

    if uploaded_file:
        df = pd.read_excel(uploaded_file, skiprows=1)
        df.columns = ['Sample', 'Factor_Weight', 'A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12']

        for col in ['A5', 'A6']:
            df[col] = df[col].replace('[\$,]', '', regex=True).astype(float)
        for col in ['A3', 'A4', 'A10', 'A11']:
            df[col] = df[col].replace('%','',regex=True).astype(float)

        df['A10'] = df['A10'].max() - df['A10']

        X = df[['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12']]
        y = df['Factor_Weight']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = RandomForestRegressor(n_estimators=300, random_state=42)
        model.fit(X_scaled, y)

        st.success("Model trained! Enter values below to make prediction.")

        # --- Prediction Inputs ---
        with st.form("prediction_form"):
            A_inputs = [st.number_input(f"A{i+1}") for i in range(12)]
            submit = st.form_submit_button("Predict")

        if submit:
            A_inputs[9] = df['A10'].max() - A_inputs[9]  # A10 correction
            scaled_inputs = scaler.transform([A_inputs])
            prediction = model.predict(scaled_inputs)[0]
            st.metric("Predicted Factor Weight", round(prediction, 2))

elif authentication_status is False:
    st.error('❌ Username or password is incorrect')

elif authentication_status is None:
    st.warning('🔑 Please enter your username and password')
