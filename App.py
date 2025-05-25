# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Cache stock data loading for performance
@st.cache_data()
def load_data(stock: str, period: str = "1y") -> pd.DataFrame | None:
    """Download stock data for given period."""
    try:
        data = yf.download(stock, period=period)
        if data.empty:
            return None
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Failed to load data for {stock}: {e}")
        return None

# Helper function: Calculate moving averages
def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()
    return df

# Visualization Functions
def show_candlestick_chart(df: pd.DataFrame, stock_name: str) -> None:
    if not {'Open', 'High', 'Low', 'Close', 'Date'}.issubset(df.columns):
        st.warning("Insufficient data for candlestick chart.")
        return
    fig = go.Figure(data=[go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"]
    )])
    fig.update_layout(title=f"Candlestick Chart for {stock_name}", xaxis_title="Date", yaxis_title="Price", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

def show_volume_chart(df: pd.DataFrame, stock_name: str) -> None:
    if "Volume" not in df.columns:
        st.warning("No volume data available.")
        return
    fig = px.bar(df, x="Date", y="Volume", title=f"Volume for {stock_name}")
    st.plotly_chart(fig)

def show_moving_averages(df: pd.DataFrame, stock_name: str) -> None:
    df_ma = add_moving_averages(df.copy())
    fig = px.line(df_ma, x="Date", y=["Close", "MA20", "MA50"], title=f"{stock_name} Moving Averages")
    st.plotly_chart(fig)

def show_volatility_chart(df: pd.DataFrame, stock_name: str) -> None:
    df = df.copy()
    df["Returns"] = df["Close"].pct_change()
    df["Volatility"] = df["Returns"].rolling(window=20).std()
    fig = px.line(df, x="Date", y="Volatility", title=f"Volatility of {stock_name}")
    st.plotly_chart(fig)

def show_correlation_heatmap(stock_data: dict[str, pd.DataFrame]) -> None:
    st.write("### Correlation Heatmap")
    df_corr = pd.DataFrame({stock: data["Close"] for stock, data in stock_data.items() if "Close" in data})
    df_corr.dropna(inplace=True)
    corr = df_corr.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

def show_trends(df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("No data available for the selected stock or date range.")
        return
    fig = px.line(df, x="Date", y="Close", title="Stock Price Over Time")
    st.plotly_chart(fig)

def train_arima(df: pd.DataFrame) -> ARIMA:
    if len(df) < 10:
        raise ValueError("Not enough data for ARIMA.")
    model = ARIMA(df["Close"].dropna(), order=(1, 1, 1))
    return model.fit()

def show_insights(df: pd.DataFrame) -> None:
    df_copy = df.copy()
    df_copy.set_index("Date", inplace=True)
    try:
        if df_copy["Close"].dropna().shape[0] < 10:
            st.error("Not enough data to train ARIMA model.")
            return
        forecast_model = train_arima(df_copy)
        forecast_value = forecast_model.forecast(steps=1).iloc[0]
        st.write(f"### ARIMA Forecast: {forecast_value:.2f}")
        current_price = df_copy["Close"].iloc[-1]
        if forecast_value > current_price * 1.05:
            st.success("📈 BUY signal")
        elif forecast_value < current_price * 0.95:
            st.error("📉 SELL signal")
        else:
            st.warning("⚖️ HOLD signal")
    except Exception as e:
        st.error(f"ARIMA failed: {e}")

def show_performance_comparison(stock_data: dict[str, pd.DataFrame], selected_stocks: list[str]) -> None:
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
    st.write("### Performance Comparison")
    st.dataframe(pd.DataFrame(rows))

# Streamlit Page Setup
st.set_page_config(page_title="📊 AI-Powered Stock Analyzer", layout="wide")
st.title("📊 AI-Powered Stock Analyzer")
st.write("Analyze stocks, visualize trends, and get AI-driven insights!")

# Sidebar Stock Selection and Options
stocks = ["AAPL", "GOOGL", "TSLA", "AMZN", "MSFT", "NFLX", "NVDA", "META", "IBM", "INTC",
          "AMD", "BABA", "ORCL", "PYPL", "DIS", "PEP", "KO", "CSCO", "UBER", "LYFT"]
st.sidebar.header("Stock Selection & Customization")
selected_stocks = st.sidebar.multiselect("Select Stocks", stocks, default=["AAPL"])

if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.experimental_rerun()

if not selected_stocks:
    st.error("No stocks selected. Please choose at least one stock.")
    st.stop()

# Load Data for selected stocks
stock_data = {stock: load_data(stock) for stock in selected_stocks}
stock_data = {k: v for k, v in stock_data.items() if v is not None}

if not stock_data:
    st.error("Failed to fetch stock data. Check your internet connection or stock symbols.")
    st.stop()

# Merge Close Prices for comparison chart
first_stock = next(iter(stock_data))
merged_df = pd.DataFrame({"Date": stock_data[first_stock]["Date"]})
for stock in selected_stocks:
    merged_df[stock] = stock_data[stock]["Close"]

# Date range filter (based on first stock)
min_date = stock_data[first_stock]["Date"].min()
max_date = stock_data[first_stock]["Date"].max()
start_date = st.sidebar.date_input("Start Date", min_date)
end_date = st.sidebar.date_input("End Date", max_date)
start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)

# Filter data by selected date range
df_filtered = stock_data[first_stock][
    (stock_data[first_stock]["Date"] >= start_date) & (stock_data[first_stock]["Date"] <= end_date)
].copy()

if df_filtered.empty:
    st.error("No valid data in selected date range.")
    st.stop()

# Display merged stock price comparison
st.write("### Stock Price Comparison")
fig_compare = px.line(merged_df, x="Date", y=selected_stocks, title="Stock Price Comparison")
st.plotly_chart(fig_compare)

# Summary statistics
stats_df = pd.DataFrame({
    "Stock": selected_stocks,
    "Mean Price": [stock_data[stock]["Close"].mean() for stock in selected_stocks],
    "Max Price": [stock_data[stock]["Close"].max() for stock in selected_stocks],
    "Min Price": [stock_data[stock]["Close"].min() for stock in selected_stocks],
})
st.write("### Stock Summary Statistics")
st.dataframe(stats_df)

# Sidebar action buttons
st.sidebar.header("Analysis Options")
if st.sidebar.button("Compare Stocks"):
    show_performance_comparison(stock_data, selected_stocks)
if st.sidebar.button("View Trends"):
    show_trends(df_filtered)
if st.sidebar.button("AI Insights"):
    show_insights(df_filtered)
if st.sidebar.button("Candlestick Chart"):
    show_candlestick_chart(df_filtered, first_stock)
if st.sidebar.button("Volume Chart"):
    show_volume_chart(df_filtered, first_stock)
if st.sidebar.button("Moving Averages"):
    show_moving_averages(df_filtered, first_stock)
if st.sidebar.button("Volatility Chart"):
    show_volatility_chart(df_filtered, first_stock)
if st.sidebar.button("Correlation Heatmap"):
    show_correlation_heatmap(stock_data)
if st.sidebar.button("Generate Report"):
    st.info("📄 Report generation coming soon!")


