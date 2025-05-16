Stock Market Prediction
A Predictive Framework for Stock Market Movements Using Hybrid XGBoost-LSTM Architecture
Date: May 16, 2025

Overview
This project implements a hybrid XGBoost-LSTM model for stock price prediction on NASDAQ data, combining advanced feature engineering with sequential deep learning. The model is deployed as an interactive Streamlit web app for real-time forecasting and visualization.

Streamlit App:
https://stock-prediction-i9vsz2k6zqu6wwtvcpnbyr.streamlit.app

Key Features
Hybrid XGBoost for feature selection and LSTM for time-series prediction
Technical indicators: MA, EMA, RSI, MACD, volatility
Robust preprocessing and normalization
Evaluation metrics: RMSE, MAE, R²
Streamlit app for real-time predictions

Dataset
NASDAQ historical stock data (619,040 rows, 5 years)
Features: Open, High, Low, Close, Volume, Date, Symbol


Project Structure
data/: Datasets
notebooks/: EDA and prototyping
src/: Scripts for preprocessing, modeling, evaluation
app/: Steamlit app code
README.md: Documentation


How It Works
Data preprocessing and technical indicator computation
Feature selection via XGBoost
Sequence generation for LSTM input
Model training and evaluation
Visualization and deployment via Streamlit

Results
RMSE: 3.25
MAE: 2.75
R²: 0.854

Usage
bash
pip install -r requirements.txt
streamlit run app/stock_prediction_app.py
Or use the hosted app:
https://stock-prediction-i9vsz2k6zqu6wwtvcpnbyr.streamlit.app

Limitations & Future Work
Sensitive to high volatility and anomalies
Computationally intensive
Future: Integrate sentiment analysis, attention mechanisms, real-time adaptation

Authors
Yuvan Karthik G
Nihal Muhammad Ali
