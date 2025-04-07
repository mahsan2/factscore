import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('factor_weight_model.pkl')

# Application title
st.title("Factor Weight Prediction")

# User inputs
a1 = st.number_input("A1 (numeric)", value=24.1)
a2 = st.number_input("A2 (numeric)", value=2.27)
a3 = st.number_input("A3 (%)", value=36.10) / 100
a4 = st.number_input("A4 (%)", value=75.40) / 100
a5 = st.number_input("A5 ($)", value=448.77)
a6 = st.number_input("A6 ($)", value=1136)
a7 = st.number_input("A7 (numeric)", value=4.9)
a8 = st.number_input("A8 (numeric)", value=4.7)
a9 = st.number_input("A9 (numeric)", value=386)
a10 = st.number_input("A10 (%) - lower is better", value=5.90) / 100
a11 = st.number_input("A11 (%)", value=12.80) / 100
a12 = st.number_input("A12 (numeric)", value=5.6)

# Arrange input data into DataFrame
input_df = pd.DataFrame({
    'A1': [a1], 'A2': [a2], 'A3': [a3], 'A4': [a4],
    'A5': [a5], 'A6': [a6], 'A7': [a7], 'A8': [a8],
    'A9': [a9], 'A10': [a10], 'A11': [a11], 'A12': [a12]
})

# Predict
if st.button('Predict Factor Weight'):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Factor Weight Overall: {prediction:.2f}")
