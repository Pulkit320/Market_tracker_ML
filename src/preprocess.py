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
        X, y = [], []
    
        for i in range(len(data) - self.window_size - 1):
            X.append(data[i:i+self.window_size])

            close_today = data[i + self.window_size - 1, 0]
            close_tomorrow = data[i + self.window_size, 0]

        # Direction: 1 = UP, 0 = DOWN or SAME
            direction = 1 if close_tomorrow > close_today else 0
            y.append(direction)

        return np.array(X), np.array(y)


