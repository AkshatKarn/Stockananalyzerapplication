# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go  # For candlestick


# Function to fetch stock data
@st.cache_data
def load_data(stock):
    data = yf.download(stock, period="1y")
    if data.empty:
        return None
    data.reset_index(inplace=True)
    return data

# Streamlit UI setup
st.set_page_config(page_title="📊 AI-Powered Stock Analyzer", layout="wide")
st.title("📊 AI-Powered Stock Analyzer")
st.write("Analyze stocks, visualize trends, and get AI-driven insights!")

# Stock selection
stocks = ["AAPL", "GOOGL", "TSLA", "AMZN", "MSFT", "NFLX", "NVDA", "META", "IBM", "INTC",
          "AMD", "BABA", "ORCL", "PYPL", "DIS", "PEP", "KO", "CSCO", "UBER", "LYFT"]
selected_stocks = st.sidebar.multiselect("📌 Select Stocks", stocks, default=["AAPL"])

# Sidebar refresh button
st.sidebar.header("📊 Stock Selection & Customization")
if st.sidebar.button("🔄 Refresh Data"):
    st.session_state.clear()
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

st.write("Selected Stocks:", selected_stocks)
st.write("Available Stock Data:", list(stock_data.keys()))

first_stock = next(iter(stock_data), None)
if first_stock and "Date" in stock_data[first_stock]:
    merged_df = pd.DataFrame({"Date": stock_data[first_stock]["Date"]})
    for stock in selected_stocks:
        if stock in stock_data:
            merged_df[stock] = stock_data[stock]["Close"]
else:
    st.error("Stock data is unavailable. Please check the data source.")
    st.stop()

# Display stock comparison table
st.write("### 📋 Stock Comparison Data")
st.dataframe(merged_df.head())
def add_moving_averages(df):
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()
    return df
#Graphs and Charts
def show_candlestick_chart(df, stock_name):
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
    st.write(f"### 📊 Volume Traded for {stock_name}")
    fig = px.bar(df, x="Date", y="Volume", labels={'Volume': 'Volume Traded'})
    st.plotly_chart(fig)

def show_moving_averages(df, stock_name):
    df_ma = add_moving_averages(df.copy())
    fig = px.line(df_ma, x="Date", y=["Close", "MA20", "MA50"], title=f"📈 {stock_name} Price with Moving Averages")
    st.plotly_chart(fig)

def show_volatility_chart(df, stock_name):
    df["Returns"] = df["Close"].pct_change()
    df["Volatility"] = df["Returns"].rolling(window=20).std()
    fig = px.line(df, x="Date", y="Volatility", title=f"📉 {stock_name} 20-day Rolling Volatility")
    st.plotly_chart(fig)

def show_correlation_heatmap(stock_data):
    st.write("### 🔍 Correlation Heatmap of Selected Stocks")
    df_corr = pd.DataFrame({stock: data["Close"] for stock, data in stock_data.items()})
    df_corr.dropna(inplace=True)

    corr = df_corr.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Line chart for multiple stocks
fig_compare = px.line(merged_df, x="Date", y=selected_stocks, title="📈 Stock Price Comparison")
st.plotly_chart(fig_compare)

# Summary statistics
stats_df = pd.DataFrame({
    "Stock": selected_stocks,
    "Mean Price": [float(stock_data[stock]["Close"].mean()) for stock in selected_stocks if stock in stock_data],
    "Max Price": [float(stock_data[stock]["Close"].max()) for stock in selected_stocks if stock in stock_data],
    "Min Price": [float(stock_data[stock]["Close"].min()) for stock in selected_stocks if stock in stock_data],
})
st.write("### 📊 Stock Comparison Summary")
st.dataframe(stats_df)

# Stock Performance Comparison
def show_comparison():
    st.write("### 📊 Stock Performance Comparison")
    rows = []
    for stock in selected_stocks:
        if stock in stock_data and not stock_data[stock].empty:
            close_series = stock_data[stock]["Close"]
            start_price = close_series.iloc[0]
            end_price = close_series.iloc[-1]
            one_year_return = ((end_price - start_price) / start_price) * 100
            volatility = close_series.pct_change().std() * np.sqrt(252)
            rows.append({
                "Stock": stock,
                "1-Year Return (%)": round(float(one_year_return), 2),
                "Volatility": round(float(volatility), 4)
            })
        else:
            st.warning(f"Data for {stock} not available or empty.")

    performance_df = pd.DataFrame(rows)
    st.dataframe(performance_df)

# Date Range Selection
st.sidebar.header("🗕 Select Date Range")
df = stock_data[first_stock]
start_date = st.sidebar.date_input("Start Date", df["Date"].min())
end_date = st.sidebar.date_input("End Date", df["Date"].max())
start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)
df_filtered = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].copy()

if df_filtered.empty or "Close" not in df_filtered.columns:
    st.error("No data available for the selected date range.")
    st.stop()

st.write(f"### 📋 Historical Data for {first_stock}")
st.dataframe(df_filtered.head())

# Stock Price Visualization
def show_trends(df_filtered):
    if df_filtered.empty:
        st.warning("No data available to display trends. Please check your date range or stock selection.")
        return
    if "Date" not in df_filtered.columns or "Close" not in df_filtered.columns:
        st.error("Missing required columns in data. Make sure 'Date' and 'Close' columns are present.")
        return
    try:
        df_filtered["Date"] = pd.to_datetime(df_filtered["Date"])
        fig = px.line(df_filtered, x="Date", y="Close", title="Stock Price Over Time", color_discrete_sequence=["blue"])
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"An error occurred while plotting: {e}")

# ARIMA Model
def train_arima(df):
    if len(df) < 10:
        raise ValueError("Not enough data points to fit ARIMA model.")
    model = ARIMA(df["Close"], order=(1, 1, 1))
    model_fit = model.fit()
    return model_fit

def show_insights(df_filtered):
    if df_filtered.empty:
        st.error("No data available for the selected date range.")
        return
    st.write("### Filtered Data Preview for ARIMA Prediction")
    st.dataframe(df_filtered.tail())
    df_filtered = df_filtered.dropna(subset=["Close"])
    df_filtered.set_index("Date", inplace=True)
    try:
        forecast_model = train_arima(df_filtered)
        forecast_value = forecast_model.forecast(steps=1).iloc[0]
        st.write(f"### 🔮 ARIMA Prediction for {first_stock}")
        if forecast_value > df_filtered['Close'].iloc[-1] * 1.05:
            st.success("📈 **BUY:** Expected upward trend.")
        elif forecast_value < df_filtered['Close'].iloc[-1] * 0.95:
            st.error("📉 **SELL:** Expected downward trend.")
        else:
            st.warning("⚖️ **HOLD:** Market stable.")
    except Exception as e:
        st.error(f"Error occurred during ARIMA prediction: {e}")

# Buttons with Functionality
if st.sidebar.button("📊 Compare Stocks"):
    show_comparison()
if st.sidebar.button("📈 View Trends"):
    show_trends(df_filtered)
if st.sidebar.button("🔮 AI Insights"):
    show_insights(df_filtered)
if st.sidebar.button("📋 Generate Report"):
    st.write("Report generation feature coming soon!")
if st.sidebar.button("📈 Compare Stocks"):
    show_comparison()

if st.sidebar.button("📉 View Trends"):
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
    st.write("Report generation feature coming soon!")
    
