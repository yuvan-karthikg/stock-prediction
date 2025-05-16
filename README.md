Stock Market Prediction

A Predictive Framework for Stock Market Movements Using Hybrid XGBoost-LSTM Architecture

Date: May 16, 2025


OVERVIEW:

This project implements a hybrid XGBoost-LSTM model for stock price prediction on NASDAQ data, combining advanced feature engineering with sequential deep learning. The model is deployed as an interactive Streamlit web app for real-time forecasting and visualization.


STREAMLIT APP:

https://stock-prediction-i9vsz2k6zqu6wwtvcpnbyr.streamlit.app


KEY FEATURES:

Hybrid XGBoost for feature selection and LSTM for time-series prediction
Technical indicators: MA, EMA, RSI, MACD, volatility
Robust preprocessing and normalization
Evaluation metrics: RMSE, MAE, R²
Streamlit app for real-time predictions


DATASET:

NASDAQ historical stock data (619,040 rows, 5 years) - 
Features: Open, High, Low, Close, Volume, Date, Symbol


PROJECT STRUCTURES:

data/: Datasets

notebooks/: EDA and prototyping

src/: Scripts for preprocessing, modeling, evaluation

app/: Steamlit app code

README.md: Documentation


HOW IT WORKS:

Data preprocessing and technical indicator computation

Feature selection via XGBoost

Sequence generation for LSTM input

Model training and evaluation

Visualization and deployment via Streamlit


RESULTS:

RMSE: 3.25

MAE: 2.75

R²: 0.854


USAGE:

bash

pip install -r requirements.txt

streamlit run app/stock_prediction_app.py

Or use the hosted app:

https://stock-prediction-i9vsz2k6zqu6wwtvcpnbyr.streamlit.app


LIMITATIONS & FUTURE WORK:

Sensitive to high volatility and anomalies

Computationally intensive

Future: Integrate sentiment analysis, attention mechanisms, real-time adaptation


AUTHORS:

Yuvan Karthik G

Nihal Muhammad Ali
