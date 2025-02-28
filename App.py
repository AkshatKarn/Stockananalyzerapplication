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
    data.reset_index(inplace=True)
    return data

st.set_page_config(page_title="📊 AI-Powered Stock Analyzer", layout="wide")
st.title("📊 AI-Powered Stock Analyzer")
st.write("Analyze stocks, visualize trends, and get AI-driven insights!")

stocks = ["AAPL", "GOOGL", "TSLA", "AMZN", "MSFT", "NFLX", "NVDA", "META", "IBM", "INTC", "AMD", "BABA", "ORCL", "PYPL", "DIS", "PEP", "KO", "CSCO", "UBER", "LYFT"]
selected_stocks = st.sidebar.multiselect("📌 Select Stocks", stocks, default=["AAPL"])

st.sidebar.header("📊 Stock Selection & Customization")
if st.sidebar.button("🔄 Refresh Data"):
    st.session_state.clear()
    st.rerun()

# Load and merge data for selected stocks
stock_data = {stock: load_data(stock) for stock in selected_stocks}
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

# Stock Performance Comparison
st.write("### 📊 Stock Performance Comparison")
performance_df = pd.DataFrame({
    "Stock": selected_stocks,
    "1-Year Return (%)": [(stock_data[stock]["Close"].iloc[-1] - stock_data[stock]["Close"].iloc[0]) / stock_data[stock]["Close"].iloc[0] * 100 for stock in selected_stocks],
    "Volatility": [stock_data[stock]['Close'].pct_change().std() * np.sqrt(252) for stock in selected_stocks]
})
st.dataframe(performance_df)

# Date Range Selection
st.sidebar.header("📅 Select Date Range")
df = stock_data[selected_stocks[0]]  # Use the first selected stock for date range reference
start_date = st.sidebar.date_input("Start Date", df["Date"].min())
end_date = st.sidebar.date_input("End Date", df["Date"].max())

start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)
df_filtered = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

st.write(f"### 📜 Historical Data for {selected_stocks[0]}")
st.dataframe(df_filtered.head())

# Stock Price Visualization
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

# ARIMA Prediction Function
def train_arima(df):
    if len(df) < 10:
        raise ValueError("Not enough data points to fit ARIMA model.")
    model = ARIMA(df["Close"], order=(1, 1, 1))
    try:
        model_fit = model.fit()
    except IndexError:
        raise ValueError("ARIMA model failed due to insufficient data.")
    return model_fit

forecast_df = train_arima(df_filtered)
st.write(f"### 🔮 ARIMA Prediction for {selected_stocks[0]}")

# AI-Powered Stock Recommendations
st.write("### 🤖 AI-Powered Stock Recommendations")
if forecast_df.forecast(steps=1)[0] > df_filtered['Close'].iloc[-1] * 1.05:
    st.success("📈 **BUY:** Expected upward trend.")
elif forecast_df.forecast(steps=1)[0] < df_filtered['Close'].iloc[-1] * 0.95:
    st.error("📉 **SELL:** Expected downward trend.")
else:
    st.warning("⚖ **HOLD:** Market stable.")

# Add Buttons
st.sidebar.button("📊 Compare Stocks")
st.sidebar.button("📈 View Trends")
st.sidebar.button("🔮 AI Insights")
st.sidebar.button("📉 Risk Analysis")
st.sidebar.button("📜 Generate Report")
