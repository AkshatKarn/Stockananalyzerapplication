import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="📊 AI-Powered Stock Analyzer", layout="wide")
st.title("📊 AI-Powered Stock Analyzer")
st.write("Analyze stocks, visualize trends, and get AI-driven insights!")

stocks = ["AAPL", "GOOGL", "TSLA", "AMZN", "MSFT", "NFLX", "NVDA", "META", "IBM", "INTC", "AMD", "BABA",
          "ORCL", "PYPL", "DIS", "PEP", "KO", "CSCO", "UBER", "LYFT"]
selected_stock = st.sidebar.selectbox("📌 Select a Stock", stocks)

st.sidebar.header("📊 Stock Selection & Customization")
if st.sidebar.button("🔄 Refresh Data"):
    st.session_state.clear()
    st.experimental_rerun()
if st.sidebar.button("🔍 Compare Multiple Stocks"):
    st.warning("Feature coming soon!")
if st.sidebar.button("📊 View Market Trends"):
    st.info("Market trend analysis coming soon!")
if st.sidebar.button("💡 AI Stock Picks"):
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

st.sidebar.header("📅 Select Date Range")
min_date = pd.to_datetime("2020-01-01")
max_date = pd.to_datetime("2026-12-31")

start_date = st.sidebar.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)
df_filtered = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

if df_filtered.empty:
    st.error("🚫 No data available for the selected date range. Please choose a different range.")
    st.stop()

left_col, right_col = st.columns([7, 5])

with left_col:
    st.markdown("<div style='max-width: 600px;'>", unsafe_allow_html=True)

    fig = px.line(df_filtered, x="Date", y="Close", title="Stock Price Over Time", color_discrete_sequence=["blue"])
    st.plotly_chart(fig, use_container_width=False, width=600)

    fig_candle = go.Figure(data=[go.Candlestick(x=df_filtered["Date"], open=df_filtered["Open"],
                                               high=df_filtered["High"], low=df_filtered["Low"],
                                               close=df_filtered["Close"])])
    st.plotly_chart(fig_candle, use_container_width=False, width=600)

    st.write("### 📊 Moving Averages & Bollinger Bands")
    df_filtered['SMA_20'] = df_filtered['Close'].rolling(window=20).mean()
    df_filtered['Upper_BB'] = df_filtered['SMA_20'] + 2 * df_filtered['Close'].rolling(window=20).std()
    df_filtered['Lower_BB'] = df_filtered['SMA_20'] - 2 * df_filtered['Close'].rolling(window=20).std()
    fig_ma = px.line(df_filtered, x="Date", y=["Close", "SMA_20", "Upper_BB", "Lower_BB"],
                     labels={"value": "Stock Price"}, title="Moving Averages & Bollinger Bands")
    st.plotly_chart(fig_ma, use_container_width=False, width=600)

    def train_arima(df):
        model = ARIMA(df["Close"], order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=180)
        future_dates = pd.date_range(start=df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=180)
        return pd.DataFrame({"Date": future_dates, "Predicted Price": forecast})

    forecast_df = train_arima(df_filtered)

    st.write(f"### 🔮 ARIMA Prediction for {selected_stock}")
    fig_pred = px.line(forecast_df, x="Date", y="Predicted Price", title="Predicted Stock Prices",
                       color_discrete_sequence=["red"])
    st.plotly_chart(fig_pred, use_container_width=False, width=600)

    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.write(f"### 📜 Historical Data for {selected_stock}")
    st.dataframe(df_filtered.head())

    st.write("### 📉 Support & Resistance Levels")
    resistance = df_filtered['High'].max()
    support = df_filtered['Low'].min()
    st.write(f"Resistance Level: {resistance:.2f}")
    st.write(f"Support Level: {support:.2f}")

    st.write("### 🤖 AI-Powered Stock Recommendations")
    if forecast_df['Predicted Price'].iloc[-1] > df_filtered['Close'].iloc[-1] * 1.05:
        st.success("📈 BUY: Expected upward trend.")
    elif forecast_df['Predicted Price'].iloc[-1] < df_filtered['Close'].iloc[-1] * 0.95:
        st.error("📉 SELL: Expected downward trend.")
    else:
        st.warning("⚖ HOLD: Market stable.")

    st.markdown("---")

    st.write("## 📚 Understanding the Indicators and Charts")
    st.markdown("""
    **Stock Price Over Time:** Visual trend of closing prices.

    **Candlestick Chart:** Visualizes OHLC data and reveals sentiment.

    **Moving Averages & Bollinger Bands:** Identify trends and volatility.

    **ARIMA Forecasting:** Predicts future stock prices based on historical data.

    **Support & Resistance:** Shows key psychological price levels.
    """)
