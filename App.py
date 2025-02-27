import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf 
from statsmodels.tsa.arima.model import ARIMA

@st.cache_data
def load_data(stock):
    """Fetches stock data from Yahoo Finance for the past year."""
    data = yf.download(stock, period="1y")
    data.reset_index(inplace=True)  # Ensure Date is a column
    return data
    
st.set_page_config(page_title="📊 AI-Powered Stock Analyzer", layout="wide")
st.title("📊 AI-Powered Stock Analyzer")
st.write("Analyze stocks, visualize trends, and get AI-driven insights!")

stocks = ["AAPL", "GOOGL", "TSLA", "AMZN", "MSFT", "NFLX", "NVDA", "META", "IBM", "INTC", "AMD", "BABA", "ORCL", "PYPL", "DIS", "PEP", "KO", "CSCO", "UBER", "LYFT"]
selected_stock = st.sidebar.selectbox("📌 Select a Stock", stocks)

st.sidebar.header("📊 Stock Selection & Customization")
if st.sidebar.button("🔄 Refresh Data"):
    st.session_state.clear()
    st.rerun()

# Ensuring 'task' is always initialized in session state
if "task" not in st.session_state:
    st.session_state["task"] = None

if st.sidebar.button("🔍 Compare Multiple Stocks"):
    st.session_state["task"] = "compare"

if st.session_state["task"] == "compare":
    selected_stocks = st.sidebar.multiselect("📌 Select Stocks to Compare", stocks, default=["AAPL", "GOOGL"])
    
    if len(selected_stocks) < 2:
        st.warning("Please select at least two stocks to compare.")
    else:
        # Load data for selected stocks
        stock_data = {stock: load_data(stock) for stock in selected_stocks}

        # Merge data into one DataFrame
        merged_df = pd.DataFrame({"Date": stock_data[selected_stocks[0]]["Date"]})
        for stock in selected_stocks:
            merged_df[stock] = stock_data[stock]["Close"]

        # Display stock comparison table
        st.write("### 📜 Stock Comparison Data")
        st.dataframe(merged_df.head())

        # Line chart for multiple stocks
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

st.sidebar.header("📅 Select Date Range")
start_date = st.sidebar.date_input("Start Date")
end_date = st.sidebar.date_input("End Date")

# Load and filter stock data
df = load_data(selected_stock)
df_filtered = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]

st.write(f"### 📜 Historical Data for {selected_stock}")
st.dataframe(df_filtered.head())

# Stock price line chart
fig = px.line(df_filtered, x="Date", y="Close", title="Stock Price Over Time")
st.plotly_chart(fig)

# Candlestick chart
fig_candle = go.Figure(data=[go.Candlestick(x=df_filtered["Date"], open=df_filtered["Open"],
    high=df_filtered["High"], low=df_filtered["Low"], close=df_filtered["Close"])])
st.plotly_chart(fig_candle)

# ARIMA Model Training
def train_arima(df):
    if len(df) < 10:
        raise ValueError("Not enough data points to fit ARIMA model.")
    
    model = ARIMA(df["Close"], order=(1, 1, 1))  
    model_fit = model.fit()
    return model_fit

st.write(f"Number of data points: {len(df_filtered)}")
if len(df_filtered) < 10:
    st.error("Not enough data to run ARIMA prediction.")
else:
    forecast_df = train_arima(df_filtered)
    st.write(f"### 🔮 ARIMA Prediction for {selected_stock}")
    forecast_df_df = pd.DataFrame({
        "Date": df_filtered["Date"],
        "Predicted Price": forecast_df.fittedvalues
    })
    fig_pred = px.line(forecast_df_df, x="Date", y="Predicted Price", title="Predicted Stock Prices", color_discrete_sequence=["red"])
    st.plotly_chart(fig_pred)
