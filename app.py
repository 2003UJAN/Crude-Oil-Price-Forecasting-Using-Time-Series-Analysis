import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from datetime import timedelta

# Load ARIMA model
with open('arima_model.pkl', 'rb') as f:
    arima_model = pickle.load(f)

# Load LSTM model and scaler
lstm_model = load_model('lstm_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load dataset
data = pd.read_csv('crude_oil_dataset.csv', parse_dates=['Date'], index_col='Date')

# Streamlit UI
st.set_page_config(page_title="Crude Oil Price Forecasting", page_icon=":oil_drum:", layout="wide")
st.title("Crude Oil Price Forecasting")
st.sidebar.header("User Input")

# User input for forecasting
forecast_days = st.sidebar.slider('Days of forecast:', 1, 30, 7)

# ARIMA Forecasting
arima_forecast = arima_model.forecast(steps=forecast_days)
arima_forecast_dates = [data.index[-1] + timedelta(days=i) for i in range(1, forecast_days + 1)]

# LSTM Forecasting
seq_length = 60
lstm_input = data['Crude_Oil_Price'].values[-seq_length:]
lstm_input = scaler.transform(lstm_input.reshape(-1, 1))
lstm_input = np.reshape(lstm_input, (1, seq_length, 1))

lstm_forecast = []
for _ in range(forecast_days):
    pred = lstm_model.predict(lstm_input)
    lstm_forecast.append(pred[0][0])
    lstm_input = np.append(lstm_input[:, 1:, :], [[pred]], axis=1)

lstm_forecast = scaler.inverse_transform(np.array(lstm_forecast).reshape(-1, 1))
lstm_forecast_dates = [data.index[-1] + timedelta(days=i) for i in range(1, forecast_days + 1)]

# Plot results
st.subheader('Forecast Results')
fig, ax = plt.subplots()
ax.plot(data.index, data['Crude_Oil_Price'], label='Historical Prices')
ax.plot(arima_forecast_dates, arima_forecast, label='ARIMA Forecast')
ax.plot(lstm_forecast_dates, lstm_forecast, label='LSTM Forecast')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig)
