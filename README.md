# 🛢️ Crude Oil Price Forecasting Using Time Series Analysis

This project forecasts crude oil prices using both **ARIMA** and **LSTM** models based on historical price data and macroeconomic indicators. The web app is built with **Streamlit** and features a petroleum-themed UI to provide real-time predictions.

## 🔍 Overview

- 📈 Historical crude oil price analysis
- 🔮 Forecasting with ARIMA and LSTM models
- 🌐 Streamlit-based interactive web app
- 🧮 Macroeconomic indicators: GDP & CPI (via FRED)
- 🧰 Auto-refreshing future predictions
- 💡 Intuitive UI for petroleum analysts & traders

## 📁 File Structure

├── generate_data.py # Data collection and preprocessing ├── train_arima.py # ARIMA model training ├── train_lstm.py # LSTM model training ├── app.py # Streamlit web app with petroleum UI ├── crude_oil_dataset.csv # Generated dataset ├── arima_model.pkl # Trained ARIMA model ├── lstm_model.h5 # Trained LSTM model ├── scaler.pkl # LSTM scaler for inverse transform ├── requirements.txt # Python dependencies └── README.md # Project documentation


## ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/crude-oil-price-forecast.git
   cd crude-oil-price-forecast

    Install dependencies

pip install -r requirements.txt

Set your FRED API Key

    Sign up at https://fred.stlouisfed.org/

    Replace 'your_api_key' in generate_data.py with your FRED API key

Generate Dataset

python generate_data.py

Train the Models

python train_arima.py
python train_lstm.py

Run the Streamlit App

    streamlit run app.py

📊 Models Used
✅ ARIMA (AutoRegressive Integrated Moving Average)

    Suitable for linear trends and stationary data

    Fast & interpretable

🧠 LSTM (Long Short-Term Memory)

    Captures non-linear and long-range temporal dependencies

    Ideal for financial time series

💡 Use Cases

    Price prediction for crude oil traders

    Government policy planning

    Petrochemical market analysis

    Risk & hedging strategy planning
