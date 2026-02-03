📈 Market Tracer using LSTM (Directional Prediction)

This project implements an LSTM-based market tracer that predicts the short-term direction (UP / DOWN) of stock prices using historical market data and technical indicators.
The model is trained and evaluated using walk-forward validation and deployed as a Streamlit web application that fetches near-real-time data via Yahoo Finance.

🚀 Features

LSTM-based time series model

Directional prediction (UP / DOWN)

Technical indicators:

Moving Average (MA20)

Relative Strength Index (RSI)

Walk-forward (TimeSeriesSplit) validation

Comparison with a naive baseline

Live inference using yfinance

Interactive Streamlit deployment

🧠 Problem Formulation

Instead of predicting exact stock prices (which are highly noisy and non-stationary), the task is formulated as a binary classification problem:

1 → Price goes UP

0 → Price goes DOWN or stays the same

This formulation better aligns with market trend analysis and avoids dominance by persistence-based baselines.

📊 Dataset

Source: Yahoo Finance (via yfinance)

Data: OHLCV historical stock data

Interval: Daily / Hourly (configurable)

Lookback window: 60 timesteps

⚠️ Note: Yahoo Finance data may vary in structure (e.g., Close vs Adj Close).
The preprocessing pipeline is designed to handle these variations robustly.

⚙️ Preprocessing Pipeline

Handles MultiIndex columns returned by Yahoo Finance

Standardizes price column (Close / Adj Close)

Removes invalid rows safely

Computes technical indicators:

MA20

RSI

Converts data into sliding window sequences for LSTM input

🏗 Model Architecture

Model: Long Short-Term Memory (LSTM)

Layers:

LSTM (50 units)

Dropout (0.2)

LSTM (50 units)

Dropout (0.2)

Dense (sigmoid output)

Optimizer: Adam

Loss: Binary Crossentropy

The output represents the probability of an upward movement.

📈 Evaluation Strategy

Validation method: Walk-forward validation (TimeSeriesSplit)

Primary metric: Accuracy

Baseline: Naive directional predictor

🔍 Results (Typical)
Model	Directional Accuracy
Naive Baseline	~44%
LSTM Model	~53%

While the absolute accuracy is modest, the model consistently outperforms the naive baseline, which is considered meaningful in financial time-series prediction.

🧪 Key Observations

Model output probabilities are clustered near 0.5, indicating weak but real signal

The model occasionally predicts DOWN, verified via probability analysis

Results align with the Efficient Market Hypothesis, highlighting the difficulty of short-term market prediction using historical technical indicators alone

🌐 Deployment

The trained model is deployed using Streamlit Cloud.

Application Capabilities:

Fetches live market data via Yahoo Finance

Applies the same preprocessing used during training

Displays:

Predicted direction (UP / DOWN)

Probability of upward movement

Historical probability trend

Price chart for context

⚠️ This application is for educational purposes only and does not provide trading advice.

🛠 Project Structure
market_tracker_ml/
│
├── app.py                  # Streamlit app
├── src/
│   ├── preprocess.py       # Feature engineering & sequence creation
│   └── __init__.py
├── models/
│   └── lstm_direction_model.h5
├── requirements.txt
└── README.md

▶️ Running Locally
git clone https://github.com/<your-username>/market-tracker-ml.git
cd market-tracker-ml
pip install -r requirements.txt
streamlit run app.py

📦 Dependencies

Python 3.9+

TensorFlow

NumPy

Pandas

Scikit-learn

Streamlit

yfinance

Matplotlib

All dependencies are listed in requirements.txt.

🎓 Academic Notes

Results are reported honestly without overstating predictive power

The project demonstrates:

Proper time-series validation

Baseline comparison

Deployment-aware preprocessing

Real-world data handling issues

📌 Disclaimer

This project is intended for learning and demonstration purposes only.
It does not constitute financial or investment advice.

✨ Author

Developed as an end-to-end machine learning project covering:

Data preprocessing

Time-series modeling

Evaluation

Deployment

