import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

@st.cache_data
def load_symbols(symbols_file):
    symbols = pd.read_csv(symbols_file)
    nasdaq_symbols = symbols[symbols['Listing Exchange'] == 'Q']['Symbol'].unique()
    return nasdaq_symbols

def add_technical_indicators(df):
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(14).mean()))
    df['EMA_5'] = df['Close'].ewm(span=5).mean()
    df['EMA_20'] = df['Close'].ewm(span=20).mean()
    df['Volatility'] = df['Close'].rolling(window=20).std()
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    return df.dropna()

def preprocess(df):
    features = ['Open','High','Low','Close','Volume','MA_10','RSI','EMA_5','EMA_20','Volatility','MACD']
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])
    return scaled, scaler, features

def select_features(scaled, target):
    xgb = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    xgb.fit(scaled[:-100], target[:-100])
    return xgb.feature_importances_.argsort()[-5:][::-1]  

def create_sequences(scaled, important_idx, lookback=60):
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i, important_idx])
        y.append(scaled[i, 3]) 
    return np.array(X), np.array(y)

def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape, dropout=0.2, recurrent_dropout=0.2),
        LSTM(32, dropout=0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

st.title('Hybrid XGBoost-LSTM Stock Price Predictor')
st.write('Upload NASDAQ symbols file & stock price CSV (e.g., ACER.csv).')

symbols_file = st.sidebar.file_uploader("Upload NASDAQ Symbols CSV (symbols_valid_meta.csv)", type=['csv'])
prices_file = st.sidebar.file_uploader("Upload Single Stock Prices CSV (e.g., ACER.csv)", type=['csv'])

if symbols_file and prices_file:
    nasdaq_symbols = load_symbols(symbols_file)
    selected_symbol = st.sidebar.selectbox('Select Stock Ticker', nasdaq_symbols)
    epochs = st.sidebar.slider('LSTM Training Epochs', 10, 50, 30)
    lookback = st.sidebar.slider('Lookback Window (days)', 30, 100, 60)

    
    df = pd.read_csv(prices_file, parse_dates=['Date'])
    df['Symbol'] = selected_symbol

    stock_df = df.sort_values('Date')
    if len(stock_df) < 100:
        st.error("Insufficient data for selected symbol")
    else:
        stock_df = add_technical_indicators(stock_df)
        if len(stock_df) < lookback + 100:
            st.error("Not enough data points after feature engineering")
        else:
            scaled, scaler, features = preprocess(stock_df)
            important_idx = select_features(scaled, stock_df['Close'].values)
            X, y = create_sequences(scaled, important_idx, lookback)

            
            split = int(0.8 * len(X))
            X_train, y_train = X[:split], y[:split]
            X_test, y_test = X[split:], y[split:]

            st.write('Training LSTM...')
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model = build_lstm((lookback, len(important_idx)))
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, 
                                validation_data=(X_test, y_test), callbacks=[early_stop], verbose=0)

            preds = model.predict(X_test)
            
            X_sample = scaled[split+lookback-1:split+lookback]
            pred_template = np.repeat(X_sample, len(preds), axis=0)
            pred_template[:, 3] = preds.flatten()
            preds_rescaled = scaler.inverse_transform(pred_template)[:,3]
            actual_rescaled = stock_df['Close'].values[-len(preds_rescaled):]

            st.subheader(f'Predictions vs Actual: {selected_symbol}')
            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(stock_df['Date'].values[-len(preds_rescaled):], actual_rescaled, label='Actual')
            ax.plot(stock_df['Date'].values[-len(preds_rescaled):], preds_rescaled, label='Predicted')
            ax.legend()
            st.pyplot(fig)

            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            rmse = np.sqrt(mean_squared_error(actual_rescaled, preds_rescaled))
            mae = mean_absolute_error(actual_rescaled, preds_rescaled)
            r2 = r2_score(actual_rescaled, preds_rescaled)
            
            st.metric("RMSE", f"{rmse:.2f}")
            st.metric("MAE", f"{mae:.2f}")
            st.metric("R² Score", f"{r2:.3f}")

            st.line_chart({
                'Training Loss': history.history['loss'],
                'Validation Loss': history.history['val_loss']
            })

else:
    st.info("Please upload both the NASDAQ & stock prices CSV file.")

st.markdown("---")
st.write("**Note:** Your stock prices CSV must contain these columns: Date, Open, High, Low, Close, Volume, and optionally Adj Close.")
