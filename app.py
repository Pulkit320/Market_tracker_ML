import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from src.preprocess import MarketPreprocessor

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Market Tracer (LSTM)",
    layout="wide"
)

st.title("📈 Market Tracer – LSTM Direction Prediction")
st.write("Predicts short-term market direction (UP / DOWN) using a pre-trained LSTM model.")

# --------------------------------------------------
# Load model & preprocessor
# --------------------------------------------------
MODEL_PATH = "models/lstm_direction_model.h5"

@st.cache_resource
def load_lstm_model():
    return load_model(MODEL_PATH)

model = load_lstm_model()
preprocessor = MarketPreprocessor(window_size=60)

# --------------------------------------------------
# Sidebar inputs
# --------------------------------------------------
st.sidebar.header("Input Parameters")

ticker = st.sidebar.text_input(
    "Stock Ticker",
    value="AAPL",
    help="Example: AAPL, MSFT, GOOGL, RELIANCE.NS"
)

period = st.sidebar.selectbox(
    "Lookback Period",
    ["6mo", "1y", "2y"],
    index=1
)

interval = st.sidebar.selectbox(
    "Interval",
    ["1d", "1h"],
    index=0
)

predict_btn = st.sidebar.button("Fetch Data & Predict")

# --------------------------------------------------
# Prediction logic
# --------------------------------------------------
if predict_btn:

    with st.spinner("Fetching market data..."):
        df = yf.download(ticker, period=period, interval=interval)

    if df.empty or len(df) < 70:
        st.error("❌ Not enough data to generate prediction.")
        st.stop()

    # --------------------------------------------------
    # Feature engineering
    # --------------------------------------------------
    df_feat = preprocessor.get_technical_indicators(df)

    feature_cols = ["Close", "MA20", "RSI"]
    df_feat = df_feat[feature_cols].dropna()

    if len(df_feat) < 61:
        st.error("❌ Not enough data after preprocessing.")
        st.stop()

    # --------------------------------------------------
    # Create sequences
    # --------------------------------------------------
    X, _ = preprocessor.create_sequences(df_feat.values)

    # --------------------------------------------------
    # Model prediction
    # --------------------------------------------------
    probs = model.predict(X, verbose=0).reshape(-1)
    last_prob = probs[-1]

    direction = "📈 UP" if last_prob >= 0.5 else "📉 DOWN"

    # --------------------------------------------------
    # Display results
    # --------------------------------------------------
    st.subheader("Latest Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Predicted Direction",
            value=direction
        )

    with col2:
        st.metric(
            label="UP Probability",
            value=f"{last_prob:.2f}"
        )

    # --------------------------------------------------
    # Probability plot
    # --------------------------------------------------
    st.subheader("UP Probability Over Time")

    prob_df = pd.DataFrame(
        {"UP Probability": probs},
        index=df_feat.index[-len(probs):]
    )

    st.line_chart(prob_df)

    # --------------------------------------------------
    # Price chart (context)
    # --------------------------------------------------
    st.subheader("Closing Price (Context Only)")
    st.line_chart(df_feat["Close"])

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption(
    "⚠️ Educational project. Predictions are probabilistic and not financial advice."
)

