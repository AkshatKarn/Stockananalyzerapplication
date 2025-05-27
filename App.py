import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="AI-Powered Stock Analyzer", layout="centered")

st.title("📊 AI-Powered Stock Analyzer")
ticker = st.text_input("🔍 Enter Stock Ticker (e.g., AAPL)", value="AAPL")

start_date = st.date_input("📅 Start Date")
end_date = st.date_input("📅 End Date")

if st.button("Analyze"):
    try:
        st.subheader(f"{ticker.upper()} Analysis")

        # Download stock data
        df = yf.download(ticker, start=start_date, end=end_date)

        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(0)

        # Check required columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_cols):
            st.error(f"❌ Missing required columns in data. Found: {df.columns.tolist()}")
        else:
            # Reset index to use 'Date' for x-axis
            df.reset_index(inplace=True)

            # Plot candlestick chart
            fig = go.Figure(data=[go.Candlestick(
                x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close']
            )])

            fig.update_layout(title=f"{ticker.upper()} Candlestick Chart",
                              xaxis_title="Date", yaxis_title="Price (USD)",
                              xaxis_rangeslider_visible=False)

            st.plotly_chart(fig)

    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")
