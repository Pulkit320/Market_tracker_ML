import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class MarketPreprocessor:
    def __init__(self, window_size=60):
        self.window_size = window_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def get_technical_indicators(self, df):
        df = df.copy()

        # Ensure numeric Close
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df.dropna(subset=['Close'], inplace=True)

        # Moving Average
        df['MA20'] = df['Close'].rolling(window=20, min_periods=20).mean()

        # RSI
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).rolling(window=14).mean()
        loss = -delta.clip(upper=0).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        return df.dropna()

    def create_sequences(self, data):
        scaled_data = self.scaler.fit_transform(data)
        X, y = [], []

        for i in range(self.window_size, len(scaled_data)):
            X.append(scaled_data[i-self.window_size:i])
            y.append(scaled_data[i, 0])

        return np.array(X), np.array(y)
