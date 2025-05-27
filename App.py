import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Page config and title
st.set_page_config(page_title="AI-Powered Stock Analyzer", layout="wide")
st.title("📊 AI-Powered Stock Analyzer")

# Sidebar - Stock Selection and Date Range
st.sidebar.header("🗂️ Stock Selection & Date Range")
stock_list = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NFLX", "NVDA"]
selected_stocks = st.sidebar.multiselect("📈 Select Stocks", stock_list, default=["AAPL"])

start_date = st.sidebar.date_input("Start Date", value=datetime.today() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", value=datetime.today())

if start_date >= end_date:
    st.sidebar.error("❌ End date must be after start date.")

# Fetch Data Function
@st.cache_data
def get_stock_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end)

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[1] if isinstance(col, tuple) else col for col in df.columns]

        df.reset_index(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()


# Moving Averages Function
def show_moving_averages(df, stock_name):
    if df.empty or "Close" not in df.columns:
        st.warning(f"⚠️ No valid data to display moving averages for {stock_name}.")
        return

    df_ma = df.copy()
    df_ma["MA20"] = df_ma["Close"].rolling(window=20).mean()
    df_ma["MA50"] = df_ma["Close"].rolling(window=50).mean()

    plot_cols = ["Close"]
    if df_ma["MA20"].notna().sum() > 0:
        plot_cols.append("MA20")
    if df_ma["MA50"].notna().sum() > 0:
        plot_cols.append("MA50")

    if len(plot_cols) <= 1:
        st.warning(f"⚠️ Not enough data to calculate moving averages for {stock_name}.")
        return

    try:
        fig = px.line(df_ma, x="Date", y=plot_cols, title=f"📊 {stock_name} Moving Averages")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Plotting error for {stock_name}: {e}")

# Candlestick Chart Function
def show_candlestick_chart(df, stock_name):
    if df.empty or len(df) < 2:
        st.warning(f"⚠️ Not enough data to display candlestick for {stock_name}.")
        return

    fig = go.Figure(data=[go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    )])
    fig.update_layout(title=f"🕯️ Candlestick Chart - {stock_name}", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# Main Dashboard Logic
if st.sidebar.button("🔄 Refresh Data") or st.session_state.get("data_loaded"):
    if not selected_stocks:
        st.warning("⚠️ Please select at least one stock.")
    else:
        st.session_state["data_loaded"] = True
        for stock in selected_stocks:
            st.subheader(f"📊 {stock} Analysis")
            data = get_stock_data(stock, start_date, end_date)

            if data.empty:
                st.error(f"❌ No data found for {stock} in the selected range.")
                continue

            show_candlestick_chart(data, stock)
            show_moving_averages(data, stock)

            with st.expander(f"📄 Raw Data for {stock}"):
                st.dataframe(data)
