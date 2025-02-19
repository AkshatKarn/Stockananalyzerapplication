import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="📊 AI-Powered Stock Analyzer", layout="wide")
st.title("📊 AI-Powered Stock Analyzer")
st.write("Analyze stocks, visualize trends, and get AI-driven insights!")

stocks = ["AAPL", "GOOGL", "TSLA", "AMZN", "MSFT", "NFLX", "NVDA", "META", "IBM", "INTC", "AMD", "BABA", "ORCL", "PYPL", "DIS", "PEP", "KO", "CSCO", "UBER", "LYFT"]
selected_stock = st.sidebar.selectbox("📌 Select a Stock", stocks)

st.sidebar.header("📊 Stock Selection & Customization")
if st.sidebar.button("🔄 Refresh Data"):
    st.session_state.clear()
    st.experimental_rerun()
if st.sidebar.button("🔍 Compare Multiple Stocks"):
    st.warning("Feature coming soon!")
if st.sidebar.button("📊 View Market Trends"):
    st.info("Market trend analysis coming soon!")
if st.sidebar.button("💡 AI Stock Picks"):
    st.success("Get AI-powered stock recommendations soon!")

@st.cache_data
def load_data(stock):
    date_rng = pd.date_range(start="2025-01-01", end="2026-01-01", freq="D")  # Fixed to start from 2025
    data = np.random.randn(len(date_rng)) * 10 + 100
    df = pd.DataFrame({"Date": date_rng, "Open": data-2, "High": data+2, "Low": data-4, "Close": data})
    df["Date"] = df["Date"].dt.tz_localize(None)
    return df

df = load_data(selected_stock)

st.sidebar.header("📅 Select Date Range")
start_date = st.sidebar.date_input("Start Date", df["Date"].min())
end_date = st.sidebar.date_input("End Date", df["Date"].max())

# Ensure proper filtering by converting date inputs to datetime
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

df_filtered = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

st.write(f"### 📜 Historical Data for {selected_stock}")
st.dataframe(df_filtered.head())

fig = px.line(df_filtered, x="Date", y="Close", title="Stock Price Over Time", color_discrete_sequence=["blue"])
st.plotly_chart(fig)

fig_candle = go.Figure(data=[go.Candlestick(x=df_filtered["Date"], open=df_filtered["Open"],
    high=df_filtered["High"], low=df_filtered["Low"], close=df_filtered["Close"], name="Candlestick")])
st.plotly_chart(fig_candle)

st.write("### 📊 Moving Averages & Bollinger Bands")
df_filtered['SMA_20'] = df_filtered['Close'].rolling(window=20).mean()
df_filtered['Upper_BB'] = df_filtered['SMA_20'] + 2 * df_filtered['Close'].rolling(window=20).std()
df_filtered['Lower_BB'] = df_filtered['SMA_20'] - 2 * df_filtered['Close'].rolling(window=20).std()
fig_ma = px.line(df_filtered, x="Date", y=["Close", "SMA_20", "Upper_BB", "Lower_BB"],
                  labels={"value": "Stock Price"}, title="Moving Averages & Bollinger Bands")
st.plotly_chart(fig_ma)

def train_arima(df):
    model = ARIMA(df["Close"], order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=180)
    future_dates = pd.date_range(start=df["Date"].iloc[-1], periods=181, freq="D")[1:]
    return pd.DataFrame({"Date": future_dates, "Predicted Price": forecast})

forecast_df = train_arima(df_filtered)
st.write(f"### 🔮 ARIMA Prediction for {selected_stock}")
fig_pred = px.line(forecast_df, x="Date", y="Predicted Price", title="Predicted Stock Prices", color_discrete_sequence=["red"])
st.plotly_chart(fig_pred)

st.write("### 📊 Volatility Analysis")
df_filtered['Volatility'] = df_filtered['Close'].pct_change()
fig_volatility = px.line(df_filtered, x="Date", y="Volatility", title="Stock Volatility Over Time")
st.plotly_chart(fig_volatility)

st.write("### 📊 RSI (Relative Strength Index) Analysis")
df_filtered['RSI'] = 100 - (100 / (1 + df_filtered['Close'].pct_change().rolling(14).mean()))
fig_rsi = px.line(df_filtered, x="Date", y="RSI", title="Relative Strength Index (RSI)")
st.plotly_chart(fig_rsi)

st.write("### 📈 MACD (Moving Average Convergence Divergence)")
df_filtered['EMA_12'] = df_filtered['Close'].ewm(span=12, adjust=False).mean()
df_filtered['EMA_26'] = df_filtered['Close'].ewm(span=26, adjust=False).mean()
df_filtered['MACD'] = df_filtered['EMA_12'] - df_filtered['EMA_26']
fig_macd = px.line(df_filtered, x="Date", y="MACD", title="MACD Indicator")
st.plotly_chart(fig_macd)

st.write("### 📉 Support & Resistance Levels")
resistance = df_filtered['High'].max()
support = df_filtered['Low'].min()
st.write(f"Resistance Level: {resistance}")
st.write(f"Support Level: {support}")

st.write("### 📊 Intraday Price Movement")
df_filtered['Intraday Change'] = df_filtered['Close'] - df_filtered['Open']
fig_intraday = px.bar(df_filtered, x="Date", y="Intraday Change", title="Intraday Price Changes")
st.plotly_chart(fig_intraday)

st.write("### 🤖 AI-Powered Stock Recommendations")
if forecast_df['Predicted Price'].iloc[-1] > df_filtered['Close'].iloc[-1] * 1.05:
    st.success("📈 **BUY:** Expected upward trend.")
elif forecast_df['Predicted Price'].iloc[-1] < df_filtered['Close'].iloc[-1] * 0.95:
    st.error("📉 **SELL:** Expected downward trend.")
else:
    st.warning("⚖ **HOLD:** Market stable.")
