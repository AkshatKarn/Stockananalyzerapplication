import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA

# Streamlit UI setup
st.set_page_config(page_title="📊 AI-Powered Stock Analyzer", layout="wide")
st.title("📊 AI-Powered Stock Analyzer")
st.write("Analyze stocks, visualize trends, and get AI-driven insights!")

# Stock list and selection
stocks = ["AAPL", "GOOGL", "TSLA", "AMZN", "MSFT", "NFLX", "NVDA", "META", "IBM", "INTC",
          "AMD", "BABA", "ORCL", "PYPL", "DIS", "PEP", "KO", "CSCO", "UBER", "LYFT"]
selected_stocks = st.sidebar.multiselect("📌 Select Stocks", stocks, default=["AAPL"])

# Sidebar - Refresh
st.sidebar.header("📊 Stock Selection & Customization")
if st.sidebar.button("🔄 Refresh Data"):
    st.session_state.clear()
    st.rerun()

# Sidebar - Date Range
st.sidebar.header("📅 Select Date Range")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2024-05-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-05-28"))

start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)

# Data loader function with caching
@st.cache_data
def load_data(stock, start_date, end_date):
    data = yf.download(stock, start=start_date, end=end_date)
    if data.empty:
        return None
    data.reset_index(inplace=True)
    return data

# Check stock selection
if not selected_stocks:
    st.error("No stocks selected. Please choose at least one.")
    st.stop()

# Load data for each selected stock
stock_data = {stock: load_data(stock, start_date, end_date) for stock in selected_stocks}
stock_data = {k: v for k, v in stock_data.items() if v is not None}

# Stop if no valid data
if not stock_data:
    st.error("No stock data retrieved. Check stock symbols or internet.")
    st.stop()

# Show available stocks
st.write("Selected Stocks:", selected_stocks)
st.write("Available Stock Data:", list(stock_data.keys()))

# Create merged dataframe for comparison
first_stock = next(iter(stock_data))

if len(selected_stocks) == 1:
    # Show data for a single selected stock
    stock = selected_stocks[0]
    df = stock_data[stock]
    df_filtered = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].copy()

    if df_filtered.empty:
        st.warning(f"No data available for {stock} in the selected date range.")
    else:
        st.write(f"### 📜 Historical Data for {stock}")
        st.dataframe(df_filtered.head())
        st.write("Date Range in Data:", df_filtered['Date'].min(), "to", df_filtered['Date'].max())

        fig = px.line(df_filtered, x="Date", y="Close", title=f"📈 Stock Price of {first_stock}")
        st.plotly_chart(fig)
else:
    # Multiple stocks — show comparison
    merged_df = pd.DataFrame({"Date": stock_data[first_stock]["Date"]})
    for stock in selected_stocks:
        merged_df[stock] = stock_data[stock]["Close"]

    st.write("### 📜 Stock Comparison Data")
    st.dataframe(merged_df.head())

    fig_compare = px.line(merged_df, x="Date", y=selected_stocks, title="📈 Stock Price Comparison")
    st.plotly_chart(fig_compare)

    stats_df = pd.DataFrame({
        "Stock": selected_stocks,
        "Mean Price": [stock_data[stock]["Close"].mean() for stock in selected_stocks],
        "Max Price": [stock_data[stock]["Close"].max() for stock in selected_stocks],
        "Min Price": [stock_data[stock]["Close"].min() for stock in selected_stocks],
    })
    st.write("### 📊 Stock Comparison Summary")
    st.dataframe(stats_df)


# Stock performance comparison
def show_comparison():
    st.write("### 📊 Stock Performance Comparison")
    performance_df = pd.DataFrame({
        "Stock": selected_stocks,
        "1-Year Return (%)": [(stock_data[stock]["Close"].iloc[-1] - stock_data[stock]["Close"].iloc[0]) /
                              stock_data[stock]["Close"].iloc[0] * 100 for stock in selected_stocks],
        "Volatility": [stock_data[stock]['Close'].pct_change().std() * np.sqrt(252) for stock in selected_stocks]
    })
    st.dataframe(performance_df)

# Filtered data for trends/ARIMA
df = stock_data[first_stock]
df_filtered = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].copy()

if df_filtered.empty or "Close" not in df_filtered.columns:
    st.error("No data in selected date range.")
    st.stop()

st.write(f"### 📜 Historical Data for {first_stock}")
st.dataframe(df_filtered.head())
st.write("Date Range in Data:", df_filtered["Date"].min(), "to", df_filtered["Date"].max())

# Plot trends
def show_trends(df_filtered):
    if df_filtered.empty:
        st.warning("No data to show.")
        return
    fig = px.line(df_filtered, x="Date", y="Close", title="Stock Price Over Time")
    st.plotly_chart(fig)

# ARIMA prediction
def train_arima(df):
    model = ARIMA(df["Close"], order=(1, 1, 1))
    return model.fit()

def show_insights(df_filtered):
    df_filtered = df_filtered.dropna(subset=["Close"])
    df_filtered.set_index("Date", inplace=True)

    try:
        model = train_arima(df_filtered)
        forecast = model.forecast(steps=1)[0]
        current = df_filtered["Close"].iloc[-1]
        st.write(f"### 🔮 Forecast for {first_stock}: {forecast:.2f}")

        if forecast > current * 1.05:
            st.success("📈 BUY: Expected uptrend")
        elif forecast < current * 0.95:
            st.error("📉 SELL: Expected downtrend")
        else:
            st.warning("⚖ HOLD: Market stable")
    except Exception as e:
        st.error(f"ARIMA error: {e}")

# Sidebar buttons
if st.sidebar.button("📊 Compare Stocks"):
    show_comparison()
if st.sidebar.button("📈 View Trends"):
    show_trends(df_filtered)
if st.sidebar.button("🔮 AI Insights"):
    show_insights(df_filtered)
if st.sidebar.button("📜 Generate Report"):
    st.info("Report generation feature coming soon!")

