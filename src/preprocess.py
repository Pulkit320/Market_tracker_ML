import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class MarketPreprocessor:
    def __init__(self, window_size=60):
        self.window_size = window_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def get_technical_indicators(self, df):
    df = df.copy()

    # --------------------------------------------------
    # 1️⃣ Flatten MultiIndex columns (yfinance quirk)
    # --------------------------------------------------
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # --------------------------------------------------
    # 2️⃣ Standardize price column
    # --------------------------------------------------
    if 'Close' in df.columns:
        price = df['Close']
    elif 'Adj Close' in df.columns:
        price = df['Adj Close']
    else:
        raise ValueError(f"No Close or Adj Close column found. Columns: {df.columns}")

    df['Close'] = pd.to_numeric(price, errors='coerce')

    # --------------------------------------------------
    # 3️⃣ Drop invalid rows SAFELY
    # --------------------------------------------------
    df = df.loc[df['Close'].notna()].copy()

    # --------------------------------------------------
    # 4️⃣ Technical indicators
    # --------------------------------------------------
    df['MA20'] = df['Close'].rolling(window=20, min_periods=20).mean()

    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()

    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    return df



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


