import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# App title and description
st.set_page_config(page_title="AI-Powered Stock Analyzer", layout="wide")
st.title("💹 AI-Powered Stock Analyzer")
st.markdown("Analyze stocks, visualize trends, get AI-driven insights, and generate reports!")

# Sidebar inputs
st.sidebar.header("User Input")
selected_stock = st.sidebar.text_input("Enter Stock Symbol (e.g. AAPL, MSFT)", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.date.today())

# Load Data Function
@st.cache_data
def load_data_yfinance(stock, start, end):
    try:
        df = yf.download(stock, start=start, end=end)
        if not df.empty:
            df.reset_index(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load and validate data
df = load_data_yfinance(selected_stock, start_date, end_date)

# Debug output
st.write("📌 Selected Stock Symbol:", selected_stock)
st.write("📅 Date Range:", start_date, "to", end_date)
st.write("🧾 Raw Data Shape:", df.shape)

# Validate data
required_columns = {"Date", "Open", "High", "Low", "Close", "Volume"}
if df.empty or not required_columns.issubset(df.columns.union(["Date"])):
    st.error("❌ The dataset is empty or missing required columns.")
    st.stop()

# Ensure columns are numeric
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

# Drop rows with NaNs in Close
df.dropna(subset=["Close"], inplace=True)

# Display raw data
st.subheader("📊 Raw Stock Data")
st.dataframe(df.head())

# Plotting
st.subheader("📈 Stock Closing Price Over Time")
fig, ax = plt.subplots()
ax.plot(df["Date"], df["Close"], label="Close Price")
ax.set_xlabel("Date")
ax.set_ylabel("Close Price")
ax.set_title(f"{selected_stock} Closing Price")
ax.legend()
st.pyplot(fig)

# Moving Averages
st.subheader("📉 Moving Averages")
df["MA20"] = df["Close"].rolling(window=20).mean()
df["MA50"] = df["Close"].rolling(window=50).mean()
fig, ax = plt.subplots()
ax.plot(df["Date"], df["Close"], label="Close")
ax.plot(df["Date"], df["MA20"], label="20-Day MA")
ax.plot(df["Date"], df["MA50"], label="50-Day MA")
ax.set_title(f"{selected_stock} Moving Averages")
ax.legend()
st.pyplot(fig)

# Correlation Heatmap (if numerical columns exist)
st.subheader("📌 Correlation Matrix")
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
if len(numeric_cols) > 1:
    fig, ax = plt.subplots()
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
else:
    st.info("Not enough numeric columns for correlation matrix.")
