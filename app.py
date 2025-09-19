import streamlit as st
import joblib
import numpy as np


model = joblib.load("house_price_model.pkl")
features = joblib.load("model_features.pkl")


st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏡 House Price Prediction App")
st.write("Enter the house details below to estimate the price:")


user_input = []
for feature in features:
    value = st.number_input(f"{feature}", value=0.0)
    user_input.append(value)


if st.button("Predict Price"):
    input_array = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    st.success(f"💰 Estimated House Price: ${prediction:,.2f}")