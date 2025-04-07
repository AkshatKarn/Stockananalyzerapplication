import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA

# Function to fetch stock data
@st.cache_data
def load_data(stock):
    data = yf.download(stock, period="1y")
    if data.empty:
        return None
    data.reset_index(inplace=True)
    return data

# Streamlit UI setup
st.set_page_config(page_title="\ud83d\udcca AI-Powered Stock Analyzer", layout="wide")
st.title("\ud83d\udcca AI-Powered Stock Analyzer")
st.write("Analyze stocks, visualize trends, and get AI-driven insights!")

# Stock selection
stocks = ["AAPL", "GOOGL", "TSLA", "AMZN", "MSFT", "NFLX", "NVDA", "META", "IBM", "INTC",
          "AMD", "BABA", "ORCL", "PYPL", "DIS", "PEP", "KO", "CSCO", "UBER", "LYFT"]
selected_stocks = st.sidebar.multiselect("\ud83d\udccc Select Stocks", stocks, default=["AAPL"])

# Sidebar refresh button
st.sidebar.header("\ud83d\udcca Stock Selection & Customization")
if st.sidebar.button("\ud83d\udd04 Refresh Data"):
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
st.write("### \ud83d\udcdc Stock Comparison Data")
st.dataframe(merged_df.head())

# Line chart for multiple stocks
fig_compare = px.line(merged_df, x="Date", y=selected_stocks, title="\ud83d\udcc8 Stock Price Comparison")
st.plotly_chart(fig_compare)

# Summary statistics
stats_df = pd.DataFrame({
    "Stock": selected_stocks,
    "Mean Price": [float(stock_data[stock]["Close"].mean()) for stock in selected_stocks if stock in stock_data],
    "Max Price": [float(stock_data[stock]["Close"].max()) for stock in selected_stocks if stock in stock_data],
    "Min Price": [float(stock_data[stock]["Close"].min()) for stock in selected_stocks if stock in stock_data],
})
st.write("### \ud83d\udcca Stock Comparison Summary")
st.dataframe(stats_df)

# Stock Performance Comparison
def show_comparison():
    st.write("### \ud83d\udcca Stock Performance Comparison")
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
st.sidebar.header("\ud83d\uddd5 Select Date Range")
df = stock_data[first_stock]
start_date = st.sidebar.date_input("Start Date", df["Date"].min())
end_date = st.sidebar.date_input("End Date", df["Date"].max())
start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)
df_filtered = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].copy()

if df_filtered.empty or "Close" not in df_filtered.columns:
    st.error("No data available for the selected date range.")
    st.stop()

st.write(f"### \ud83d\udcdc Historical Data for {first_stock}")
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
        st.write(f"### \ud83d\udd2e ARIMA Prediction for {first_stock}")
        if forecast_value > df_filtered['Close'].iloc[-1] * 1.05:
            st.success("\ud83d\udcc8 **BUY:** Expected upward trend.")
        elif forecast_value < df_filtered['Close'].iloc[-1] * 0.95:
            st.error("\ud83d\udcc9 **SELL:** Expected downward trend.")
        else:
            st.warning("\u2696 **HOLD:** Market stable.")
    except Exception as e:
        st.error(f"Error occurred during ARIMA prediction: {e}")

# Buttons with Functionality
if st.sidebar.button("\ud83d\udcca Compare Stocks"):
    show_comparison()
if st.sidebar.button("\ud83d\udcc8 View Trends"):
    show_trends(df_filtered)
if st.sidebar.button("\ud83d\udd2e AI Insights"):
    show_insights(df_filtered)
if st.sidebar.button("\ud83d\udcdc Generate Report"):
    st.write("Report generation feature coming soon!")
