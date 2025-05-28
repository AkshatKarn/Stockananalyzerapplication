import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf

# ---------- CONFIGURATION ----------
st.set_page_config(page_title="\ud83d\udcca AI-Powered Stock Analyzer", layout="wide")
st.title("\ud83d\udcca AI-Powered Stock Analyzer")
st.write("Analyze stocks, visualize trends, and get smart insights!")

# ---------- UTILITY FUNCTIONS ----------
def plot_line_chart(df, x_col="Date", y_col="Close", title="\ud83d\udcc8 Stock Chart"):
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

def plot_candlestick_chart(df, title="\ud83d\udcc9 Candlestick Chart"):
    if {'Date', 'Open', 'High', 'Low', 'Close'}.issubset(df.columns):
        fig = go.Figure(data=[
            go.Candlestick(x=df['Date'],
                           open=df['Open'], high=df['High'],
                           low=df['Low'], close=df['Close'])
        ])
        fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Missing required columns for candlestick chart.")

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
    st.write("### \ud83d\udcca Stock Performance Comparison")
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
    st.write("### \ud83d\udcc8 Line Chart Trend")
    plot_line_chart(df_filtered, title="\ud83d\udcc8 Stock Price Over Time")
    st.write("### \ud83d\udcc9 Candlestick Trend")
    plot_candlestick_chart(df_filtered)

def show_insights(df_filtered, stock):
    st.write(f"### \ud83e\udd14 Insights for {stock}")
    mean_price = df_filtered["Close"].mean()
    max_price = df_filtered["Close"].max()
    min_price = df_filtered["Close"].min()
    st.markdown(f"- Average Price: **{mean_price:.2f}**")
    st.markdown(f"- Max Price: **{max_price:.2f}**")
    st.markdown(f"- Min Price: **{min_price:.2f}**")

def generate_report(df_filtered, stock):
    st.write("### \ud83d\udcc4 Generated Report")
    st.markdown(f"#### Summary for **{stock}**")
    st.dataframe(df_filtered.describe())
    show_insights(df_filtered, stock)
    plot_line_chart(df_filtered, title=f"\ud83d\udcc8 {stock} Line Chart")
    plot_candlestick_chart(df_filtered, title=f"\ud83d\udcc9 {stock} Candlestick Chart")

# ---------- SIDEBAR INPUT ----------
stocks = ["AAPL", "GOOGL", "TSLA", "AMZN", "MSFT", "NFLX", "NVDA", "META", "IBM", "INTC",
          "AMD", "BABA", "ORCL", "PYPL", "DIS", "PEP", "KO", "CSCO", "UBER", "LYFT"]

selected_stocks = st.sidebar.multiselect("\ud83d\udccc Select Stocks", stocks, default=["AAPL"])

st.sidebar.header("\ud83d\udcca Stock Selection & Customization")
if st.sidebar.button("\ud83d\udd04 Refresh Data"):
    st.cache_data.clear()
    st.experimental_rerun()

st.sidebar.header("\ud83d\udcc5 Select Date Range")
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

df = stock_data[first_stock]
df_filtered = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].copy()

if len(selected_stocks) == 1:
    if df_filtered.empty:
        st.warning(f"No data available for {first_stock} in the selected date range.")
    else:
        st.write(f"### \ud83d\udcdc Historical Data for {first_stock}")
        st.dataframe(df_filtered.head())

        st.write("Date Range in Data:", df_filtered['Date'].min().date(), "to", df_filtered['Date'].max().date())
        plot_line_chart(df_filtered, title=f"\ud83d\udcc8 Stock Price of {first_stock}")
else:
    merged_df = pd.DataFrame({"Date": stock_data[first_stock]["Date"]})
    for stock in selected_stocks:
        merged_df[stock] = stock_data[stock]["Close"].values

    st.write("### \ud83d\udcdc Stock Comparison Data")
    st.dataframe(merged_df.head())
    plot_line_chart(merged_df, x_col="Date", y_col=selected_stocks, title="\ud83d\udcc8 Stock Price Comparison")

    stats_df = pd.DataFrame({
        "Stock": selected_stocks,
        "Mean Price": [stock_data[stock]["Close"].mean() for stock in selected_stocks],
        "Max Price": [stock_data[stock]["Close"].max() for stock in selected_stocks],
        "Min Price": [stock_data[stock]["Close"].min() for stock in selected_stocks],
    })
    st.write("### \ud83d\udcca Stock Comparison Summary")
    st.dataframe(stats_df)

# ---------- SIDEBAR ACTIONS ----------
if st.sidebar.button("\ud83d\udcca Compare Stocks"):
    show_comparison(stock_data, selected_stocks)

if st.sidebar.button("\ud83d\udcc8 View Trends"):
    show_trends(df_filtered)

if st.sidebar.button("\ud83e\udd2e AI Insights"):
    show_insights(df_filtered, first_stock)

if st.sidebar.button("\ud83d\udcc4 Generate Report"):
    generate_report(df_filtered, first_stock)
