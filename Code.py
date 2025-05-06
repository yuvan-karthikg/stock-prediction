import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt


@st.cache_data
def load_symbols():
    symbols = pd.read_csv('symbols_valid_meta.csv')
    return symbols[symbols['Listing Exchange'] == 'Q']['Symbol'].unique()


@st.cache_data
def load_stock_data(symbol):
    df = pd.read_csv('stock_prices.csv', parse_dates=['Date'])
    df = df[df['Name'] == symbol].sort_values('Date')
    df['MA_10'] = df['Close'].rolling(10).mean()
    df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(14).mean()))
    df = df.dropna()
    return df

def preprocess(df):
    scaler = MinMaxScaler()
    features = ['Open','High','Low','Close','Volume','MA_10','RSI']
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

st.title('Hybrid XGBoost-LSTM Stock Price Predictor')


symbols = load_symbols()
selected_symbol = st.sidebar.selectbox('Select NASDAQ Stock Ticker', symbols)
epochs = st.sidebar.slider('LSTM Training Epochs', 10, 50, 30)
lookback = st.sidebar.slider('Lookback Window (days)', 30, 100, 60)


df = load_stock_data(selected_symbol)
scaled, scaler = preprocess(df)
important_idx = select_features(scaled, df['Close'].values)
X, y = create_sequences(scaled, important_idx, lookback)


split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]


st.write('Training LSTM...')
model = build_lstm((lookback, len(important_idx)))
history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test), verbose=0)


preds = model.predict(X_test)
preds_rescaled = scaler.inverse_transform(np.concatenate([
    np.zeros((preds.shape[0], 3)),  # Fill for Open, High, Low
    preds,                          # Predicted Close
    np.zeros((preds.shape[0], 3))   # Fill for Volume, MA_10, RSI
], axis=1))[:,3]

actual_rescaled = df['Close'].values[-len(preds_rescaled):]


st.subheader(f'Predicted vs Actual Close Price for {selected_symbol}')
fig, ax = plt.subplots()
ax.plot(df['Date'].values[-len(preds_rescaled):], actual_rescaled, label='Actual')
ax.plot(df['Date'].values[-len(preds_rescaled):], preds_rescaled, label='Predicted')
ax.legend()
st.pyplot(fig)


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
rmse = np.sqrt(mean_squared_error(actual_rescaled, preds_rescaled))
mae = mean_absolute_error(actual_rescaled, preds_rescaled)
r2 = r2_score(actual_rescaled, preds_rescaled)
st.write(f'**RMSE:** {rmse:.2f}')
st.write(f'**MAE:** {mae:.2f}')
st.write(f'**R² Score:** {r2:.3f}')

st.write('**Model Training History**')
st.line_chart({'loss': history.history['loss'], 'val_loss': history.history['val_loss']})

st.write('---')
st.write('**How to use:**')
st.write('Select a NASDAQ stock, adjust parameters, and view the hybrid model’s prediction vs. actual closing prices.')

