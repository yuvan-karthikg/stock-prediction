import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

@st.cache_data
def load_symbols(symbols_file):
    symbols = pd.read_csv(symbols_file)
    return symbols[symbols['Listing Exchange'] == 'Q']['Symbol'].unique()

def add_technical_indicators(df):
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df.dropna()

def preprocess(df):
    features = ['Open','High','Low','Close','Volume','MA_10','RSI']
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    scaled_features = X_scaler.fit_transform(df[features])
    scaled_target = y_scaler.fit_transform(df[['Close']])
    return scaled_features, scaled_target, X_scaler, y_scaler

def select_features(scaled, target, split_index):
    xgb = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    xgb.fit(scaled[:split_index], target[:split_index].ravel())
    return xgb.feature_importances_.argsort()[-3:][::-1]

def create_sequences(scaled, target, important_idx, lookback=60):
    X, y = [], []
    for i in range(lookback, len(scaled) - 1):
        X.append(scaled[i-lookback:i, important_idx])
        y.append(target[i + 1])  # Predict next day's close
    return np.array(X), np.array(y)

def build_lstm(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Streamlit app starts
st.title('📈 Hybrid XGBoost + LSTM Stock Predictor')
st.write("Upload your NASDAQ symbols file and one stock's price data CSV to begin.")

symbols_file = st.sidebar.file_uploader("Upload NASDAQ Symbols CSV", type=['csv'])
prices_file = st.sidebar.file_uploader("Upload Stock Prices CSV (e.g., AAPL.csv)", type=['csv'])

if symbols_file and prices_file:
    nasdaq_symbols = load_symbols(symbols_file)
    selected_symbol = st.sidebar.selectbox('Select Stock Ticker', nasdaq_symbols)
    epochs = st.sidebar.slider('Training Epochs', 10, 50, 30)
    lookback = st.sidebar.slider('Lookback Days', 30, 100, 60)

    df = pd.read_csv(prices_file, parse_dates=['Date'])
    df['Symbol'] = selected_symbol
    df = df.sort_values('Date')

    if len(df) < 100:
        st.error("Insufficient data points.")
    else:
        df = add_technical_indicators(df)
        if len(df) < lookback + 100:
            st.error("Not enough data after feature engineering.")
        else:
            scaled_X, scaled_y, X_scaler, y_scaler = preprocess(df)
            split = int(0.8 * len(scaled_X))
            important_idx = select_features(scaled_X, scaled_y, split)

            X, y = create_sequences(scaled_X, scaled_y, important_idx, lookback)
            split = int(0.8 * len(X))
            X_train, y_train = X[:split], y[:split]
            X_test, y_test = X[split:], y[split:]

            st.write('Training LSTM model...')
            model = build_lstm((lookback, len(important_idx)))
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=32,
                                validation_data=(X_test, y_test), verbose=0)

            preds_scaled = model.predict(X_test)
            preds_rescaled = y_scaler.inverse_transform(preds_scaled)
            actual_rescaled = y_scaler.inverse_transform(y_test)

            st.subheader(f'📊 Predictions vs Actual: {selected_symbol}')
            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(df['Date'].values[-len(preds_rescaled):], actual_rescaled, label='Actual')
            ax.plot(df['Date'].values[-len(preds_rescaled):], preds_rescaled, label='Predicted')
            ax.legend()
            st.pyplot(fig)

            rmse = np.sqrt(mean_squared_error(actual_rescaled, preds_rescaled))
            mae = mean
