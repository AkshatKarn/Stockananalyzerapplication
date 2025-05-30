import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="AI-Powered Stock Analyzer", layout="wide")
st.title(" AI-Powered Stock Analyzer")
st.write("Analyze stocks, visualize trends, and get AI-driven insights!")

stocks = ["AAPL", "GOOGL", "TSLA", "AMZN", "MSFT", "NFLX", "NVDA", "META", "IBM", "INTC", "AMD", "BABA",
          "ORCL", "PYPL", "DIS", "PEP", "KO", "CSCO", "UBER", "LYFT"]
selected_stock = st.sidebar.selectbox("Select a Stock", stocks)

st.sidebar.header("Stock Selection & Customization")
if st.sidebar.button("Refresh Data"):
    st.session_state.clear()
    st.experimental_rerun()

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

st.sidebar.header("Select Date Range")
min_date = pd.to_datetime("2020-01-01")
max_date = pd.to_datetime("2026-12-31")

start_date = st.sidebar.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)
df_filtered = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

if df_filtered.empty:
    st.error("No data available for the selected date range. Please choose a different range.")
    st.stop()

def generate_insights(df):
    insights = []
    if df["Close"].iloc[-1] > df["Close"].iloc[0]:
        insights.append("The stock showed an overall **uptrend** during the selected period.")
    else:
        insights.append("The stock showed an overall **downtrend** during the selected period.")

    change = df["Close"].iloc[-1] - df["Close"].iloc[0]
    pct_change = (change / df["Close"].iloc[0]) * 100
    insights.append(f"The stock changed by **{change:.2f} USD** (**{pct_change:.2f}%**) from start to end.")

    max_row = df.loc[df["Close"].idxmax()]
    min_row = df.loc[df["Close"].idxmin()]
    insights.append(f"Highest price: **{max_row['Close']:.2f} USD** on **{max_row['Date'].date()}**.")
    insights.append(f"Lowest price: **{min_row['Close']:.2f} USD** on **{min_row['Date'].date()}**.")

    volatility = df["Close"].pct_change().std() * 100
    insights.append(f"Approx. volatility: **{volatility:.2f}%**.")

    return insights

# Layout with tabs for better organization
with st.container():
    tab1, tab2 = st.tabs(["Visualizations", "\ud83d\udcd8 Insights"])

    with tab1:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Line Chart")
            fig = px.line(df_filtered, x="Date", y="Close", title="Stock Price Over Time", color_discrete_sequence=["blue"])
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Candlestick Chart")
            fig_candle = go.Figure(data=[go.Candlestick(x=df_filtered["Date"], open=df_filtered["Open"],
                                                        high=df_filtered["High"], low=df_filtered["Low"],
                                                        close=df_filtered["Close"])])
            st.plotly_chart(fig_candle, use_container_width=True)

            st.subheader("Moving Averages & Bollinger Bands")
            df_filtered['SMA_20'] = df_filtered['Close'].rolling(window=20).mean()
            df_filtered['Upper_BB'] = df_filtered['SMA_20'] + 2 * df_filtered['Close'].rolling(window=20).std()
            df_filtered['Lower_BB'] = df_filtered['SMA_20'] - 2 * df_filtered['Close'].rolling(window=20).std()
            fig_ma = px.line(df_filtered, x="Date", y=["Close", "SMA_20", "Upper_BB", "Lower_BB"],
                             title="Moving Averages & Bollinger Bands")
            st.plotly_chart(fig_ma, use_container_width=True)

        with col2:
            st.subheader("Data Preview")
            st.dataframe(df_filtered[['Date', 'Open', 'High', 'Low', 'Close']].tail(10), use_container_width=True)

    with tab2:
        st.subheader("Stock Insights")
        for insight in generate_insights(df_filtered):
            st.markdown(insight)

        st.markdown("---")
        st.markdown("**Note:** All data is simulated for demonstration purposes only.")
