import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="📊 AI-Powered Stock Analyzer", layout="wide")
st.title("📊 AI-Powered Stock Analyzer")
st.write("Analyze stocks, visualize trends, and get AI-driven insights!")

stocks = ["AAPL", "GOOGL", "TSLA", "AMZN", "MSFT", "NFLX", "NVDA", "META", "IBM", "INTC", "AMD", "BABA",
          "ORCL", "PYPL", "DIS", "PEP", "KO", "CSCO", "UBER", "LYFT"]
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
    date_rng = pd.date_range(start="2020-01-01", end="2026-12-31", freq="D")
    data = np.random.randn(len(date_rng)) * 10 + 100  # Simulated price data
    df = pd.DataFrame({
        "Date": date_rng,
        "Open": data - 2,
        "High": data + 2,
        "Low": data - 4,
        "Close": data
    })
    df["Date"] = df["Date"].dt.tz_localize(None)
    return df

df = load_data(selected_stock)

st.sidebar.header("📅 Select Date Range")
min_date = pd.to_datetime("2020-01-01")
max_date = pd.to_datetime("2026-12-31")

start_date = st.sidebar.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)
df_filtered = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

if df_filtered.empty:
    st.error("🚫 No data available for the selected date range. Please choose a different range.")
    st.stop()

# Layout: 2 columns left (7 units), right (5 units) for roughly 58%-42%
left_col, right_col = st.columns([7,5])

with left_col:
    # Limit width with container div using markdown & CSS trick
    st.markdown("<div style='max-width: 600px;'>", unsafe_allow_html=True)

    # Price line chart
    fig = px.line(df_filtered, x="Date", y="Close", title="Stock Price Over Time", color_discrete_sequence=["blue"])
    st.plotly_chart(fig, use_container_width=False, width=600)

    # Candlestick chart
    fig_candle = go.Figure(data=[go.Candlestick(x=df_filtered["Date"], open=df_filtered["Open"],
                                               high=df_filtered["High"], low=df_filtered["Low"],
                                               close=df_filtered["Close"], name="Candlestick")])
    st.plotly_chart(fig_candle, use_container_width=False, width=600)

    # Moving Averages & Bollinger Bands
    st.write("### 📊 Moving Averages & Bollinger Bands")
    df_filtered['SMA_20'] = df_filtered['Close'].rolling(window=20).mean()
    df_filtered['Upper_BB'] = df_filtered['SMA_20'] + 2 * df_filtered['Close'].rolling(window=20).std()
    df_filtered['Lower_BB'] = df_filtered['SMA_20'] - 2 * df_filtered['Close'].rolling(window=20).std()
    fig_ma = px.line(df_filtered, x="Date", y=["Close", "SMA_20", "Upper_BB", "Lower_BB"],
                     labels={"value": "Stock Price"}, title="Moving Averages & Bollinger Bands")
    st.plotly_chart(fig_ma, use_container_width=False, width=600)

    def train_arima(df):
        model = ARIMA(df["Close"], order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=180)
        future_dates = pd.date_range(start=df["Date"].iloc[-1], periods=181, fr
