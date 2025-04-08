import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
import numpy as np

# === Set up UI ===
st.set_page_config(page_title="Crude Oil Price Forecasting", layout="centered")
st.title("\U0001F4C8 Crude Oil Price Forecasting")
st.markdown("""
<style>
    .main {
        background-color: #f4f1ee;
    }
    h1 {
        color: #0b3d91;
    }
    .stButton>button {
        background-color: #0b3d91;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# === Load Data ===
data = pd.read_csv("crude_oil_macro_data.csv")
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# === Display data ===
if st.checkbox("Show Raw Data"):
    st.write(data.tail())

# === Load models ===
model_option = st.selectbox("Select Forecasting Model", ["ARIMA", "LSTM"])

if model_option == "ARIMA":
    model = joblib.load("arima_model.pkl")
    steps = st.slider("Forecast Steps (Days)", 30, 180, 90)
    forecast = model.forecast(steps=steps)
    st.line_chart(forecast)
    st.success("Forecast plotted using ARIMA")

elif model_option == "LSTM":
    model = load_model("lstm_model.h5")
    scaler = joblib.load("lstm_scaler.save")

    st.write("Forecasting next 30 days using LSTM")
    prices = data['Close_CL=F'].fillna(method='ffill').values.reshape(-1, 1)
    scaled_prices = scaler.transform(prices)

    window = 60
    last_window = scaled_prices[-window:]
    predictions = []

    for _ in range(30):
        pred_input = last_window.reshape(1, window, 1)
        pred = model.predict(pred_input)
        predictions.append(pred[0][0])
        last_window = np.append(last_window[1:], pred, axis=0)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    st.line_chart(predictions)
    st.success("Forecast plotted using LSTM")
