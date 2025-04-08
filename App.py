# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import seaborn as sns
import matplotlib.pyplot as plt
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
st.sidebar.header("📊 Stock Selection & Customization")
selected_stocks = st.sidebar.multiselect("📌 Select Stocks", stocks, default=["AAPL"])

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

if not selected_stocks:
    st.error("No stocks selected. Please choose at least one stock.")
    st.stop()

# Fetch stock data
stock_data = {stock: load_data(stock) for stock in selected_stocks}
stock_data = {k: v for k, v in stock_data.items() if v is not None}

if not stock_data:
    st.error("Failed to fetch stock data. Please check your internet connection or stock symbols.")
    st.stop()

# Build merged data for comparison
first_stock = next(iter(stock_data))
merged_df = pd.DataFrame({"Date": stock_data[first_stock]["Date"]})
for stock in selected_stocks:
    if "Close" in stock_data[stock]:
        merged_df[stock] = stock_data[stock]["Close"]

# Display merged stock comparison
st.write("### 📋 Stock Comparison Data")
st.dataframe(merged_df.head())

# Charts and Analysis Functions
def add_moving_averages(df):
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()
    return df

def show_candlestick_chart(df, stock_name):
    if not {'Open', 'High', 'Low', 'Close', 'Date'}.issubset(df.columns):
        st.warning("Not enough data to display candlestick.")
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
    st.plotly_chart(fig)

def show_volume_chart(df, stock_name):
    if "Volume" not in df.columns:
        st.warning("No volume data available.")
        return
    fig = px.bar(df, x="Date", y="Volume", title=f"📊 Volume for {stock_name}")
    st.plotly_chart(fig)

def show_moving_averages(df, stock_name):
    df_ma = add_moving_averages(df.copy())
    fig = px.line(df_ma, x="Date", y=["Close", "MA20", "MA50"], title=f"📈 {stock_name} Moving Averages")
    st.plotly_chart(fig)

def show_volatility_chart(df, stock_name):
    df["Returns"] = df["Close"].pct_change()
    df["Volatility"] = df["Returns"].rolling(window=20).std()
    fig = px.line(df, x="Date", y="Volatility", title=f"📉 Volatility of {stock_name}")
    st.plotly_chart(fig)

def show_correlation_heatmap(stock_data):
    st.write("### 🔍 Correlation Heatmap")
    df_corr = pd.DataFrame({stock: data["Close"] for stock, data in stock_data.items() if "Close" in data})
    df_corr.dropna(inplace=True)
    corr = df_corr.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Line Chart
fig_compare = px.line(merged_df, x="Date", y=selected_stocks, title="📈 Stock Price Comparison")
st.plotly_chart(fig_compare)

# Summary statistics
stats_df = pd.DataFrame({
    "Stock": selected_stocks,
    "Mean Price": [stock_data[stock]["Close"].mean() for stock in selected_stocks],
    "Max Price": [stock_data[stock]["Close"].max() for stock in selected_stocks],
    "Min Price": [stock_data[stock]["Close"].min() for stock in selected_stocks],
})
st.write("### 📊 Stock Comparison Summary")
st.dataframe(stats_df)

# Performance Comparison
def show_comparison():
    rows = []
    for stock in selected_stocks:
        close_series = stock_data[stock]["Close"]
        start_price, end_price = close_series.iloc[0], close_series.iloc[-1]
        one_year_return = ((end_price - start_price) / start_price) * 100
        volatility = close_series.pct_change().std() * np.sqrt(252)
        rows.append({
            "Stock": stock,
            "1-Year Return (%)": round(one_year_return, 2),
            "Volatility": round(volatility, 4)
        })
    st.write("### 📊 Performance Comparison")
    st.dataframe(pd.DataFrame(rows))

# Date Range Filter
df = stock_data[first_stock]
start_date = st.sidebar.date_input("Start Date", df["Date"].min())
end_date = st.sidebar.date_input("End Date", df["Date"].max())
start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)
df_filtered = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].copy()

if df_filtered.empty or "Close" not in df_filtered.columns:
    st.error("No valid data in selected date range.")
    st.stop()

st.write(f"### 📋 Historical Data for {first_stock}")
st.dataframe(df_filtered.head())

# Trend visualization
def show_trends(df_filtered):
    fig = px.line(df_filtered, x="Date", y="Close", title="Stock Price Over Time")
    st.plotly_chart(fig)
    plt.clf()


# ARIMA
def train_arima(df):
    if len(df) < 10:
        raise ValueError("Not enough data for ARIMA.")
    model = ARIMA(df["Close"].dropna(), order=(1, 1, 1))
    return model.fit()

def show_insights(df_filtered):
    df_filtered.set_index("Date", inplace=True)
    try:
        if df_filtered["Close"].dropna().shape[0] < 10:
            st.error("Not enough data to train ARIMA model.")
            return     
        forecast_model = train_arima(df_filtered)
        forecast_value = forecast_model.forecast(steps=1).iloc[0]
        st.write(f"### 🔮 ARIMA Forecast: {forecast_value:.2f}")
        current_price = df_filtered["Close"].iloc[-1]
        if forecast_value > current_price * 1.05:
            st.success("📈 BUY signal")
        elif forecast_value < current_price * 0.95:
            st.error("📉 SELL signal")
        else:
            st.warning("⚖️ HOLD signal")
    except Exception as e:
        st.error(f"ARIMA failed: {e}")

# Sidebar Buttons
if st.sidebar.button("📊 Compare Stocks"):
    show_comparison()
if st.sidebar.button("📈 View Trends"):
    show_trends(df_filtered)
if st.sidebar.button("🔮 AI Insights"):
    show_insights(df_filtered)
if st.sidebar.button("📉 Candlestick Chart"):
    show_candlestick_chart(df_filtered, first_stock)
if st.sidebar.button("📊 Volume Chart"):
    show_volume_chart(df_filtered, first_stock)
if st.sidebar.button("📈 Moving Averages"):
    show_moving_averages(df_filtered, first_stock)
if st.sidebar.button("📉 Volatility Chart"):
    show_volatility_chart(df_filtered, first_stock)
if st.sidebar.button("🔍 Correlation Heatmap"):
    show_correlation_heatmap(stock_data)
if st.sidebar.button("📝 Generate Report"):
    st.write("📄 Report generation coming soon!")
