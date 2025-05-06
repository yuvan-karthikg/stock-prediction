import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# 1. Load NASDAQ symbols from uploaded file
@st.cache_data
def load_symbols(symbols_file):
    symbols = pd.read_csv(symbols_file)
    nasdaq_symbols = symbols[symbols['Listing Exchange'] == 'Q']['Symbol'].unique()
    return nasdaq_symbols

# 2. Preprocessing & Feature Engineering
def add_technical_indicators(df):
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(14).mean()))
    return df.dropna()

def preprocess(df):
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_10', 'RSI']
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])
    return scaled, scaler

def select_features(scaled, target):
    xgb = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    xgb.fit(scaled[:-100], target[:-100])
    return xgb.feature_importances_.argsort()[-3:][::-1]

def create_sequences(scaled, important_idx, lookback=60):
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i, important_idx])
        y.append(scaled[i, 3])  # Close price index
    return np.array(X), np.array(y)

def build_lstm(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Streamlit UI
st.title('Stock Market Prediction: Hybrid XGBoost-LSTM')
st.write('Upload your Kaggle stock price data and NASDAQ symbols file to begin.')

symbols_file = st.sidebar.file_uploader("Upload NASDAQ Symbols CSV (symbols_valid_meta.csv)", type=['csv'])
prices_file = st.sidebar.file_uploader("Upload Stock Prices CSV (Kaggle dataset)", type=['csv'])

if symbols_file and prices_file:
    nasdaq_symbols = load_symbols(symbols_file)
    selected_symbol = st.sidebar.selectbox('Select NASDAQ Stock Ticker', nasdaq_symbols)
    epochs = st.sidebar.slider('LSTM Training Epochs', 10, 50, 30)
    lookback = st.sidebar.slider('Lookback Window (days)', 30, 100, 60)

    # Load and filter data
    df = pd.read_csv(prices_file, parse_dates=['Date'])
    stock_df = df[df['Name'] == selected_symbol].sort_values('Date')
    stock_df = add_technical_indicators(stock_df)
    if len(stock_df) < lookback + 100:
        st.error("Not enough data for this ticker after feature engineering. Try another ticker.")
    else:
        scaled, scaler = preprocess(stock_df)
        important_idx = select_features(scaled, stock_df['Close'].values)
        X, y = create_sequences(scaled, important_idx, lookback)

        split = int(0.8 * len(X))
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        st.write('Training LSTM...')
        model = build_lstm((lookback, len(important_idx)))
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test), verbose=0)

        preds = model.predict(X_test)
        # Reconstruct for inverse scaling
        preds_full = np.zeros((len(preds), 7))
        preds_full[:, 3] = preds.flatten()
        preds_rescaled = scaler.inverse_transform(preds_full)[:, 3]
        actual_rescaled = stock_df['Close'].values[-len(preds_rescaled):]

        # Plot
        st.subheader(f'Predicted vs Actual Close Price for {selected_symbol}')
        fig, ax = plt.subplots()
        ax.plot(stock_df['Date'].values[-len(preds_rescaled):], actual_rescaled, label='Actual')
        ax.plot(stock_df['Date'].values[-len(preds_rescaled):], preds_rescaled, label='Predicted')
        ax.legend()
        st.pyplot(fig)

        # Metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        rmse = np.sqrt(mean_squared_error(actual_rescaled, preds_rescaled))
        mae = mean_absolute_error(actual_rescaled, preds_rescaled)
        r2 = r2_score(actual_rescaled, preds_rescaled)
        st.write(f'**RMSE:** {rmse:.2f}')
        st.write(f'**MAE:** {mae:.2f}')
        st.write(f'**R² Score:** {r2:.3f}')

        st.write('**Model Training History**')
        st.line_chart({'loss': history.history['loss'], 'val_loss': history.history['val_loss']})

else:
    st.info("Please upload both the NASDAQ symbols and stock prices CSV files.")

st.write('---')
st.write('**Instructions:**')
st.write('1. Upload the NASDAQ symbols file (symbols_valid_meta.csv).')
st.write('2. Upload your Kaggle stock price data (should have columns: Date, Open, High, Low, Close, Volume, Name).')
st.write('3. Select a ticker, adjust parameters, and view predictions.')

