import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from io import BytesIO
import yfinance as yf

st.set_page_config(page_title="AI-Powered Stock Analyzer", layout="wide")
st.title("AI-Powered Stock Analyzer")
st.write("Analyze stocks, visualize trends, get AI-driven insights, and generate reports!")

stocks = ["AAPL", "GOOGL", "TSLA", "AMZN", "MSFT", "NFLX", "NVDA", "META", "IBM", "INTC", "AMD", "BABA",
          "ORCL", "PYPL", "DIS", "PEP", "KO", "CSCO", "UBER", "LYFT"]
stock_input_method = st.sidebar.radio("Select stock input method:", ["Choose from list", "Enter manually"])

if stock_input_method == "Choose from list":
    selected_stock = st.sidebar.selectbox("Select a Stock", stocks)
else:
    selected_stock = st.sidebar.text_input("Enter Stock Symbol (e.g. AAPL)", value="AAPL")

st.sidebar.header("Stock Selection & Customization")
if st.sidebar.button("Refresh Data"):
    st.session_state.clear()
    st.experimental_rerun()

@st.cache_data

def load_data_yfinance(stock, start, end):
    df = yf.download(stock, start=start, end=end)
    df.reset_index(inplace=True)
    return df

st.sidebar.header("Select Date Range")
min_date = pd.to_datetime("2015-01-01")
max_date = pd.to_datetime("2025-12-31")

start_date = st.sidebar.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

df = load_data_yfinance(selected_stock, start_date, end_date)

if df.empty:
    st.error("No data available for the selected date range. Please choose a different range or check the stock symbol.")
    st.stop()

# Rest of the code remains unchanged

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

def generate_report(df):
    output = BytesIO()
    report_text = "Stock Report\n\n"
    report_text += f"Date Range: {start_date.date()} to {end_date.date()}\n"
    report_text += f"Total Days: {len(df)}\n"
    report_text += "\n".join(generate_insights(df))
    output.write(report_text.encode())
    output.seek(0)
    return output

with st.container():
    tab1, tab2, tab3 = st.tabs(["Visualizations", "Insights", "Report Generator"])

    with tab1:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Line Chart")
            fig = px.line(df_filtered, x="Date", y="Close", title="Stock Price Over Time", color_discrete_sequence=["blue"])
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Candlestick Chart")
            fig_candle = go.Figure(data=[go.Candlestick(x=df["Date"], open=df["Open"],
                                                        high=df["High"], low=df["Low"],
                                                        close=df["Close"])])
            st.plotly_chart(fig_candle, use_container_width=True)

            st.subheader("Moving Averages & Bollinger Bands")
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['Upper_BB'] = df['SMA_20'] + 2 * df['Close'].rolling(window=20).std()
            df['Lower_BB'] = df['SMA_20'] - 2 * df['Close'].rolling(window=20).std()
            fig_ma = px.line(df, x="Date", y=["Close", "SMA_20", "Upper_BB", "Lower_BB"],
                             title="Moving Averages & Bollinger Bands")
            st.plotly_chart(fig_ma, use_container_width=True)

        with col2:
            st.subheader("Data Preview")
            st.dataframe(df[['Date', 'Open', 'High', 'Low', 'Close']].tail(10), use_container_width=True)

    with tab2:
        st.subheader("Stock Insights")
        for insight in generate_insights(df):
            st.markdown(insight)

    with tab3:
        st.subheader("Generate Report")
        if st.button("Generate Text Report"):
            report = generate_report(df)
            st.download_button(label="Download Report", data=report, file_name=f"{selected_stock}_report.txt", mime="text/plain")
        st.markdown("Note: This report is based on real-time data fetched using yfinance.")
