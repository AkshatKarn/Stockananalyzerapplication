import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="\ud83d\udcca AI-Powered Stock Analyzer", layout="wide")
st.title("\ud83d\udcca AI-Powered Stock Analyzer")
st.write("Analyze stocks, visualize trends, and get AI-driven insights!")

stocks = ["AAPL", "GOOGL", "TSLA", "AMZN", "MSFT", "NFLX", "NVDA", "META", "IBM", "INTC", "AMD", "BABA",
          "ORCL", "PYPL", "DIS", "PEP", "KO", "CSCO", "UBER", "LYFT"]
selected_stock = st.sidebar.selectbox("\ud83d\udccc Select a Stock", stocks)

st.sidebar.header("\ud83d\udcca Stock Selection & Customization")
if st.sidebar.button("\ud83d\udd04 Refresh Data"):
    st.session_state.clear()
    st.experimental_rerun()
if st.sidebar.button("\ud83d\udd0d Compare Multiple Stocks"):
    st.warning("Feature coming soon!")
if st.sidebar.button("\ud83d\udcca View Market Trends"):
    st.info("Market trend analysis coming soon!")
if st.sidebar.button("\ud83d\udca1 AI Stock Picks"):
    st.success("Get AI-powered stock recommendations soon!")

@st.cache_data
def load_data(stock):
    date_rng = pd.date_range(start="2020-01-01", end="2026-12-31", freq="D")
    data = np.random.randn(len(date_rng)) * 10 + 100
    df = pd.DataFrame({
        "Date": date_rng,
        "Open": data - 2,
        "High": data + 2,
        "Low": data - 4,
        "Close": data
    })
    df["Date"] = df["Date"].dt.tz_localize(None)
    return df

df = load_data(selected_stock)

st.sidebar.header("\ud83d\udcc5 Select Date Range")
min_date = pd.to_datetime("2020-01-01")
max_date = pd.to_datetime("2026-12-31")

start_date = st.sidebar.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)
df_filtered = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

if df_filtered.empty:
    st.error("\ud83d\udeab No data available for the selected date range. Please choose a different range.")
    st.stop()

def generate_insights(df):
    insights = []
    if df["Close"].iloc[-1] > df["Close"].iloc[0]:
        insights.append("\ud83d\udcc8 The stock showed an overall **uptrend** during the selected period.")
    else:
        insights.append("\ud83d\udcc9 The stock showed an overall **downtrend** during the selected period.")

    change = df["Close"].iloc[-1] - df["Close"].iloc[0]
    pct_change = (change / df["Close"].iloc[0]) * 100
    insights.append(f"\ud83d\udd0d The stock changed by **{change:.2f} USD** (**{pct_change:.2f}%**) from start to end.")

    max_row = df.loc[df["Close"].idxmax()]
    min_row = df.loc[df["Close"].idxmin()]
    insights.append(f"\ud83d\ude80 The highest price was **{max_row['Close']:.2f} USD** on **{max_row['Date'].date()}**.")
    insights.append(f"\ud83d\udcc9 The lowest price was **{min_row['Close']:.2f} USD** on **{min_row['Date'].date()}**.")

    volatility = df["Close"].pct_change().std() * 100
    insights.append(f"\ud83d\udcca The stock had a volatility of approximately **{volatility:.2f}%**.")

    return insights

left_col, right_col = st.columns([7, 5])

with left_col:
    st.markdown("<div style='max-width: 600px;'>", unsafe_allow_html=True)

    fig = px.line(df_filtered, x="Date", y="Close", title="Stock Price Over Time", color_discrete_sequence=["blue"])
    st.plotly_chart(fig, use_container_width=False, width=600)

    fig_candle = go.Figure(data=[go.Candlestick(x=df_filtered["Date"], open=df_filtered["Open"],
                                                high=df_filtered["High"], low=df_filtered["Low"],
                                                close=df_filtered["Close"], name="Candlestick")])
    st.plotly_chart(fig_candle, use_container_width=False, width=600)

    df_filtered['SMA_20'] = df_filtered['Close'].rolling(window=20).mean()
    df_filtered['Upper_BB'] = df_filtered['SMA_20'] + 2 * df_filtered['Close'].rolling(window=20).std()
    df_filtered['Lower_BB'] = df_filtered['SMA_20'] - 2 * df_filtered['Close'].rolling(window=20).std()
    fig_ma = px.line(df_filtered, x="Date", y=["Close", "SMA_20", "Upper_BB", "Lower_BB"],
                     labels={"value": "Stock Price"}, title="Moving Averages & Bollinger Bands")
    st.plotly_chart(fig_ma, use_container_width=False, width=600)

with right_col:
    st.subheader("\ud83d\udcd8 Stock Insights")
    for insight in generate_insights(df_filtered):
        st.markdown(insight)
