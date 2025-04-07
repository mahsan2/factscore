import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from io import BytesIO

# --- Authenticator Setup ---
credentials = {
    "usernames": {
        "zahed1": {
            "name": "Authorized User",
            "password": "$2b$12$3J.QpuvRADGvYrXTi6tkfOtmRAgLfmzxlL19o1ebpHgN5NGwUiiJy"
        }
    }
}
authenticator = stauth.Authenticate(
    credentials,
    cookie_name="app_cookie",
    key="abcdef",
    cookie_expiry_days=1
)
name, authentication_status, username = authenticator.login("Login", "main")

# --- App After Login ---
if authentication_status:
    authenticator.logout("Logout", "sidebar")
    st.title("🔐 Factor Weight Predictor with Explainability")

    uploaded_file = st.file_uploader("📄 Upload your Excel file (.xlsx)", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file, skiprows=1)
        df.columns = ['Sample', 'Factor_Weight', 'A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12']

        # Clean and preprocess
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
        explainer = shap.Explainer(model, X_scaled)

        st.success("✅ Model trained successfully!")

        with st.form("predict_form"):
            st.subheader("📊 Enter input values")
            A_inputs = [st.number_input(f"A{i+1}") for i in range(12)]
            submit = st.form_submit_button("🔍 Predict")

        if submit:
            A_inputs[9] = df['A10'].max() - A_inputs[9]  # Fix A10
            input_scaled = scaler.transform([A_inputs])
            prediction = model.predict(input_scaled)[0]
            rounded_pred = int(round(prediction))

            st.metric("🎯 Predicted Factor Score", rounded_pred)

            # Explainability (SHAP)
            shap_values = explainer(input_scaled)
            st.subheader("📈 SHAP Explanation")
            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_values[0], max_display=12, show=False)
            st.pyplot(fig)

            # Downloadable Report
            st.subheader("📥 Download Prediction Report")
            report_df = pd.DataFrame({
                'Input Feature': [f'A{i+1}' for i in range(12)],
                'Value': A_inputs,
                'SHAP Value': shap_values[0].values
            })
            report_df.loc[len(report_df.index)] = ['Prediction', rounded_pred, '']
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                report_df.to_excel(writer, index=False, sheet_name='Prediction Report')
            st.download_button("📄 Download Excel Report", output.getvalue(), file_name='prediction_report.xlsx')

elif authentication_status is False:
    st.error("❌ Username or password is incorrect.")
elif authentication_status is None:
    st.warning("🔑 Please enter your username and password.")
