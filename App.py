import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA

# ---------- CONFIGURATION ----------
st.set_page_config(page_title="📊 AI-Powered Stock Analyzer", layout="wide")
st.title("📊 AI-Powered Stock Analyzer")
st.write("Analyze stocks, visualize trends, and get AI-driven insights!")

# ---------- UTILITY FUNCTIONS ----------
def plot_line_chart(df, x_col="Date", y_col="Close", title="📈 Stock Chart"):
    if df is None or df.empty:
        st.warning("No data to plot.")
        return

    if isinstance(y_col, list):
        missing_cols = [col for col in y_col if col not in df.columns]
        if missing_cols:
            st.warning(f"Missing columns for plotting: {missing_cols}")
            return
    else:
        if x_col not in df.columns or y_col not in df.columns:
            st.warning(f"Missing '{x_col}' or '{y_col}' in the data.")
            return

    try:
        fig = px.line(df, x=x_col, y=y_col, title=title)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Plotting error: {e}")

def load_data(stock, start_date, end_date):
    data = yf.download(stock, start=start_date, end=end_date)
    if data.empty:
        return None
    data.reset_index(inplace=True)
    data["Date"] = pd.to_datetime(data["Date"], errors='coerce')
    data.dropna(subset=["Date", "Close"], inplace=True)
    return data

@st.cache_data(show_spinner=False)
def get_data(stock, start, end):
    return load_data(stock, start, end)

def show_comparison(stock_data, selected_stocks):
    st.write("### 📊 Stock Performance Comparison")
    performance_df = pd.DataFrame({
        "Stock": selected_stocks,
        "1-Year Return (%)": [
            (stock_data[stock]["Close"].iloc[-1] - stock_data[stock]["Close"].iloc[0]) / stock_data[stock]["Close"].iloc[0] * 100
            for stock in selected_stocks
        ],
        "Volatility": [
            stock_data[stock]['Close'].pct_change().std() * np.sqrt(252)
            for stock in selected_stocks
        ]
    })
    st.dataframe(performance_df)

def show_trends(df_filtered):
    plot_line_chart(df_filtered, title="📈 Stock Price Over Time")

def train_arima(df):
    model = ARIMA(df["Close"], order=(1, 1, 1))
    return model.fit()

def show_insights(df_filtered, stock):
    df_copy = df_filtered.dropna(subset=["Close"]).copy()

    if len(df_copy) < 20:
        st.warning("Not enough data points for reliable ARIMA forecast.")
        return

    df_copy.set_index("Date", inplace=True)

    try:
        model = train_arima(df_copy)
        forecast = model.forecast(steps=1)[0]
        current = df_copy["Close"].iloc[-1]
        st.write(f"### 🔮 Forecast for {stock}: {forecast:.2f}")

        if forecast > current * 1.05:
            st.success("📈 BUY: Expected uptrend")
        elif forecast < current * 0.95:
            st.error("📉 SELL: Expected downtrend")
        else:
            st.warning("⚖️ HOLD: Market stable")
    except Exception as e:
        st.error(f"ARIMA error: {e}")

# ---------- SIDEBAR INPUT ----------
stocks = ["AAPL", "GOOGL", "TSLA", "AMZN", "MSFT", "NFLX", "NVDA", "META", "IBM", "INTC",
          "AMD", "BABA", "ORCL", "PYPL", "DIS", "PEP", "KO", "CSCO", "UBER", "LYFT"]

selected_stocks = st.sidebar.multiselect("📌 Select Stocks", stocks, default=["AAPL"])

st.sidebar.header("📊 Stock Selection & Customization")
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.experimental_rerun()

st.sidebar.header("📅 Select Date Range")
min_date = pd.to_datetime("2010-01-01")
max_date = pd.to_datetime("today")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2024-05-01"), min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-05-28"), min_value=min_date, max_value=max_date)

start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)

if start_date > end_date:
    st.sidebar.error("Start Date must be before End Date")
    st.stop()

# ---------- DATA LOADING ----------
if not selected_stocks:
    st.error("No stocks selected. Please choose at least one.")
    st.stop()

stock_data = {stock: get_data(stock, start_date, end_date) for stock in selected_stocks}
stock_data = {k: v for k, v in stock_data.items() if v is not None}

if not stock_data:
    st.error("No stock data retrieved. Check stock symbols or internet connection.")
    st.stop()

# ---------- DISPLAY DATA ----------
st.write("Selected Stocks:", selected_stocks)
st.write("Available Stock Data:", list(stock_data.keys()))

first_stock = next(iter(stock_data))

if len(selected_stocks) == 1:
    stock = selected_stocks[0]
    df = stock_data[stock]
    df_filtered = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].copy()

    if df_filtered.empty:
        st.warning(f"No data available for {stock} in the selected date range.")
    else:
        st.write(f"### 📜 Historical Data for {stock}")
        st.dataframe(df_filtered.head())

        st.write("Date Range in Data:", df_filtered['Date'].min().date(), "to", df_filtered['Date'].max().date())
        plot_line_chart(df_filtered, title=f"📈 Stock Price of {stock}")

else:
    # Merge dates and close prices from all stocks for comparison
    merged_df = pd.DataFrame({"Date": stock_data[first_stock]["Date"]})
    for stock in selected_stocks:
        merged_df[stock] = stock_data[stock]["Close"].values

    st.write("### 📜 Stock Comparison Data")
    st.dataframe(merged_df.head())
    plot_line_chart(merged_df, x_col="Date", y_col=selected_stocks, title="📈 Stock Price Comparison")

    stats_df = pd.DataFrame({
        "Stock": selected_stocks,
        "Mean Price": [stock_data[stock]["Close"].mean() for stock in selected_stocks],
        "Max Price": [stock_data[stock]["Close"].max() for stock in selected_stocks],
        "Min Price": [stock_data[stock]["Close"].min() for stock in selected_stocks],
    })
    st.write("### 📊 Stock Comparison Summary")
    st.dataframe(stats_df)

# ---------- SIDEBAR ACTIONS ----------
df = stock_data[first_stock]
df_filtered = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].copy()

if st.sidebar.button("📊 Compare Stocks"):
    show_comparison(stock_data, selected_stocks)

if st.sidebar.button("📈 View Trends"):
    show_trends(df_filtered)

if st.sidebar.button("🔮 AI Insights"):
    show_insights(df_filtered, first_stock)

if st.sidebar.button("📜 Generate Report"):
    st.info("Report generation feature coming soon!")
