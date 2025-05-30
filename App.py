import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import matplotlib.pyplot as plt

# Streamlit page config
st.set_page_config(page_title="Simple Stock Analyzer", layout="wide")
st.title("📈 Simple Stock Analyzer")
st.markdown("Enter a stock symbol and date range to view price trends and basic stats.")

# Sidebar inputs
st.sidebar.header("Stock Selection")
symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, MSFT)", value="AAPL")
start_date = st.sidebar.date_input("Start Date", datetime.date(2022, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

# Validate date input
if start_date >= end_date:
    st.error("❗ Start date must be before end date.")
    st.stop()

# Fetch stock data
@st.cache_data
def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    df.reset_index(inplace=True)
    return df

df = load_data(symbol, start_date, end_date)

# Validate data
if df.empty or "Close" not in df.columns:
    st.error("⚠️ No data found. Please check the stock symbol and date range.")
    st.stop()

# Clean Close column
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df.dropna(subset=["Close"], inplace=True)

# Show basic statistics
st.subheader(f"📊 Basic Statistics for {symbol}")
st.write(df["Close"].describe()[["mean", "min", "max"]].rename({
    "mean": "Average Close",
    "min": "Minimum Close",
    "max": "Maximum Close"
}))

# Plot
st.subheader(f"📉 Closing Price Trend for {symbol}")
fig, ax = plt.subplots()
ax.plot(df["Date"], df["Close"], color='skyblue', linewidth=2)
ax.set_xlabel("Date")
ax.set_ylabel("Close Price")
ax.set_title(f"{symbol} Closing Price")
st.pyplot(fig)

# Toggle to show raw data
if st.checkbox("Show Raw Data"):
    st.subheader("📄 Raw Data")
    st.dataframe(df)
