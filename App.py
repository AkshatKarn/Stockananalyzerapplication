# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go

# Function to fetch stock data
@st.cache_data()
def load_data(stock):
    try:
        data = yf.download(stock, period="1y")
        if data.empty:
            return None
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Failed to load data for {stock}: {e}")
        return None

# Streamlit UI setup
st.set_page_config(page_title="📊 AI-Powered Stock Analyzer", layout="wide")
st.title("📊 AI-Powered Stock Analyzer")
st.write("Analyze stocks, visualize trends, and get AI-driven insights!")

# Stock selection
stocks = ["AAPL", "GOOGL", "TSLA", "AMZN", "MSFT", "NFLX", "NVDA", "META", "IBM", "INTC",
          "AMD", "BABA", "ORCL", "PYPL", "DIS", "PEP", "KO", "CSCO", "UBER", "LYFT"]
st.sidebar.header("📊 Stock Selection & Date Range")
selected_stocks = st.sidebar.multiselect("📌 Select Stocks", stocks, default=["AAPL"])

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.experimental_rerun()

if not selected_stocks:
    st.error("No stocks selected. Please choose at least one stock.")
    st.stop()

# Fetch stock data
stock_data = {stock: load_data(stock) for stock in selected_stocks}
stock_data = {k: v for k, v in stock_data.items() if v is not None}

if not stock_data:
    st.error("Failed to fetch stock data. Please check your internet connection or stock symbols.")
    st.stop()

# Date Range Filter based on first selected stock's data
first_stock = next(iter(stock_data))
df = stock_data[first_stock]
start_date = st.sidebar.date_input("Start Date", df["Date"].min())
end_date = st.sidebar.date_input("End Date", df["Date"].max())
start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)

if start_date > end_date:
    st.error("Start date must be before end date.")
    st.stop()

# Filter data by date range for all stocks
for stock in stock_data:
    stock_data[stock] = stock_data[stock][(stock_data[stock]["Date"] >= start_date) & (stock_data[stock]["Date"] <= end_date)].copy()

# Warn about stocks with no data in selected range
no_data_stocks = [stock for stock in selected_stocks if stock_data[stock].empty]
if no_data_stocks:
    st.warning(f"No data for these stocks in selected date range: {', '.join(no_data_stocks)}")

# Build merged dataframe for price comparison (fixed to avoid misalignment)
merged_df = stock_data[first_stock][["Date"]].copy()
for stock in selected_stocks:
    if "Close" in stock_data[stock]:
        temp_df = stock_data[stock][["Date", "Close"]].rename(columns={"Close": stock})
        merged_df = pd.merge(merged_df, temp_df, on="Date", how="outer")
merged_df = merged_df.sort_values("Date").reset_index(drop=True)

# Stock price comparison line chart
fig_compare = px.line(merged_df, x="Date", y=selected_stocks, title="📈 Stock Price Comparison")
st.plotly_chart(fig_compare, use_container_width=True)

# Candlestick Chart function
def show_candlestick_chart(df, stock_name):
    if not {'Open', 'High', 'Low', 'Close', 'Date'}.issubset(df.columns) or df.empty:
        st.warning(f"Not enough data to display candlestick for {stock_name}.")
        return
    st.write(f"### 📉 Candlestick Chart for {stock_name}")
    fig = go.Figure(data=[go.Candlestick(
        x=df["Date"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"]
    )])
    fig.update_layout(xaxis_title="Date", yaxis_title="Price", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# Moving Averages function
def show_moving_averages(df, stock_name):
    df_ma = df.copy()
    df_ma["MA20"] = df_ma["Close"].rolling(window=20).mean()
    df_ma["MA50"] = df_ma["Close"].rolling(window=50).mean()
    fig = px.line(df_ma, x="Date", y=["Close", "MA20", "MA50"], title=f"📈 {stock_name} Moving Averages")
    st.plotly_chart(fig, use_container_width=True)

# Summary Statistics
stats_df = pd.DataFrame({
    "Stock": selected_stocks,
    "Mean Price": [stock_data[stock]["Close"].mean() for stock in selected_stocks],
    "Max Price": [stock_data[stock]["Close"].max() for stock in selected_stocks],
    "Min Price": [stock_data[stock]["Close"].min() for stock in selected_stocks],
})
st.write("### 📊 Stock Summary Statistics")
st.dataframe(stats_df)

# Performance Comparison
def show_performance():
    rows = []
    for stock in selected_stocks:
        close_series = stock_data[stock]["Close"]
        if close_series.empty:
            continue
        start_price, end_price = close_series.iloc[0], close_series.iloc[-1]
        one_year_return = ((end_price - start_price) / start_price) * 100
        volatility = close_series.pct_change().std() * np.sqrt(252)
        rows.append({
            "Stock": stock,
            "Return (%)": round(one_year_return, 2),
            "Volatility": round(volatility, 4)
        })
    st.write("### 📊 Performance Comparison")
    st.dataframe(pd.DataFrame(rows))

show_performance()

# ARIMA Model and AI Insights for the first selected stock
def train_arima(df):
    if len(df) < 10:
        raise ValueError("Not enough data for ARIMA.")
    model = ARIMA(df["Close"].dropna(), order=(1, 1, 1))
    return model.fit()

def show_ai_insights(df, stock_name):
    df_copy = df.copy()
    df_copy.set_index("Date", inplace=True)
    try:
        if df_copy["Close"].dropna().shape[0] < 10:
            st.warning(f"Not enough data to train ARIMA model for {stock_name}.")
            return     
        forecast_model = train_arima(df_copy)
        forecast_value = forecast_model.forecast(steps=1).iloc[0]
        st.write(f"### 🔮 ARIMA Forecast for {stock_name}: {forecast_value:.2f}")
        current_price = df_copy["Close"].iloc[-1]
        if forecast_value > current_price * 1.05:
            st.success("📈 BUY signal")
        elif forecast_value < current_price * 0.95:
            st.error("📉 SELL signal")
        else:
            st.warning("⚖️ HOLD signal")
    except Exception as e:
        st.error(f"ARIMA failed: {e}")

# Show charts for the first selected stock only (to keep UI clean)
show_candlestick_chart(stock_data[first_stock], first_stock)
show_moving_averages(stock_data[first_stock], first_stock)
show_ai_insights(stock_data[first_stock], first_stock)
