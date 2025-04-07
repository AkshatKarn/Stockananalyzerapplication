import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA

# Function to fetch stock data
@st.cache_data
def load_data(stock):
    """Fetches stock data from Yahoo Finance for the past year."""
    data = yf.download(stock, period="1y")
    if data.empty:  # Check if no data was retrieved
        return None
    data.reset_index(inplace=True)
    return data

# Streamlit UI setup
st.set_page_config(page_title="📊 AI-Powered Stock Analyzer", layout="wide")
st.title("📊 AI-Powered Stock Analyzer")
st.write("Analyze stocks, visualize trends, and get AI-driven insights!")

# Stock selection
stocks = ["AAPL", "GOOGL", "TSLA", "AMZN", "MSFT", "NFLX", "NVDA", "META", "IBM", "INTC",
          "AMD", "BABA", "ORCL", "PYPL", "DIS", "PEP", "KO", "CSCO", "UBER", "LYFT"]
selected_stocks = st.sidebar.multiselect("📌 Select Stocks", stocks, default=["AAPL"])

# Sidebar refresh button
st.sidebar.header("📊 Stock Selection & Customization")
if st.sidebar.button("🔄 Refresh Data"):
    st.session_state.clear()
    st.rerun()

# Ensure stocks are selected
if not selected_stocks:
    st.error("No stocks selected. Please choose at least one stock.")
    st.stop()

# Fetch stock data
stock_data = {stock: load_data(stock) for stock in selected_stocks}
stock_data = {k: v for k, v in stock_data.items() if v is not None}  # Remove None values

# Ensure at least one valid stock is available
if not stock_data:
    st.error("Failed to fetch stock data. Please check your internet connection or stock symbols.")
    st.stop()

# Debugging output
st.write("Selected Stocks:", selected_stocks)
st.write("Available Stock Data:", list(stock_data.keys()))

# Ensure at least one stock has data before proceeding
# Ensure at least one stock has data before proceeding
first_stock = next(iter(stock_data), None)
if first_stock and "Date" in stock_data[first_stock]:
    merged_df = pd.DataFrame({"Date": stock_data[first_stock]["Date"]})
    for stock in selected_stocks:
        if stock in stock_data:
            merged_df[stock] = stock_data[stock]["Close"]

    # ✅ Flatten columns (in case they're MultiIndex)
    if isinstance(df_filtered.columns, pd.MultiIndex):
        df_filtered.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df_filtered.columns]
    else:
        st.error("Stock data is unavailable. Please check the data source.")
        st.stop()
    
def show_trends(df_filtered):
    import plotly.express as px
    import streamlit as st

    if df_filtered.empty:
        st.warning("No data available to display trends. Please check your date range or stock selection.")
        return

    # Flatten MultiIndex if present
    if isinstance(df_filtered.columns, pd.MultiIndex):
        df_filtered.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df_filtered.columns]

    # Debug: check column names
    st.write("📌 DataFrame Columns:", df_filtered.columns.tolist())

    # Try to detect the correct column names
    date_col = [col for col in df_filtered.columns if "Date" in col][0]
    close_col = [col for col in df_filtered.columns if "Close" in col and first_stock in col]
    
    if not close_col:
        close_col = [col for col in df_filtered.columns if col.lower() == "close"]

    if not close_col:
        st.error("Couldn't find the correct 'Close' column to plot.")
        return

    close_col = close_col[0]  # Get the first match

    try:
        df_filtered[date_col] = pd.to_datetime(df_filtered[date_col])
        fig = px.line(df_filtered, x=date_col, y=close_col, title="📉 Stock Price Trend")
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"An error occurred while plotting: {e}")


# Display stock comparison table
st.write("### 📜 Stock Comparison Data")
st.dataframe(merged_df.head())

# Line chart for multiple stocks
fig_compare = px.line(merged_df, x="Date", y=selected_stocks, title="📈 Stock Price Comparison")
st.plotly_chart(fig_compare)

# Summary statistics
# Summary statistics
stats_df = pd.DataFrame({
    "Stock": selected_stocks,
    "Mean Price": [float(stock_data[stock]["Close"].mean()) for stock in selected_stocks if stock in stock_data],
    "Max Price": [float(stock_data[stock]["Close"].max()) for stock in selected_stocks if stock in stock_data],
    "Min Price": [float(stock_data[stock]["Close"].min()) for stock in selected_stocks if stock in stock_data],
})
st.write("### 📊 Stock Comparison Summary")
st.dataframe(stats_df)


# Stock Performance Comparison
def show_comparison():
    st.write("### 📊 Stock Performance Comparison")

    rows = []
    for stock in selected_stocks:
        if stock in stock_data and not stock_data[stock].empty:
            close_series = stock_data[stock]["Close"]

            # Extract first and last close prices
            start_price = close_series.iloc[0]
            end_price = close_series.iloc[-1]

            # Calculate return and volatility
            one_year_return = ((end_price - start_price) / start_price) * 100
            volatility = close_series.pct_change().std() * np.sqrt(252)

            # Make sure they are scalar float values (not Series)
            one_year_return = float(one_year_return)
            volatility = float(volatility)

            rows.append({
                "Stock": stock,
                "1-Year Return (%)": round(one_year_return, 2),
                "Volatility": round(volatility, 4)
            })
        else:
            st.warning(f"Data for {stock} not available or empty.")

    # Build DataFrame with clean scalar values
    performance_df = pd.DataFrame(rows)

    # Show in Streamlit
    st.dataframe(performance_df)

# Date Range Selection
st.sidebar.header("📅 Select Date Range")
df = stock_data[first_stock]  # Use the first available stock for reference
start_date = st.sidebar.date_input("Start Date", df["Date"].min())
end_date = st.sidebar.date_input("End Date", df["Date"].max())

start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)
df_filtered = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].copy()

if df_filtered.empty or "Close" not in df_filtered.columns:
    st.error("No data available for the selected date range.")
    st.stop()

st.write(f"### 📜 Historical Data for {first_stock}")
st.dataframe(df_filtered.head())

# Stock Price Visualization
def show_trends(df_filtered):
    import plotly.express as px
    import streamlit as st

    if df_filtered.empty:
        st.warning("No data available to display trends. Please check your date range or stock selection.")
        return

    if "Date" not in df_filtered.columns or "Close" not in df_filtered.columns:
        st.error("Missing required columns in data. Make sure 'Date' and 'Close' columns are present.")
        st.write("Columns in DataFrame:", df_filtered.columns.tolist())
        return

    try:
        df_filtered["Date"] = pd.to_datetime(df_filtered["Date"])  # Ensure correct datetime format
        fig = px.line(df_filtered, x="Date", y="Close", title="Stock Price Over Time", color_discrete_sequence=["blue"])
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"An error occurred while plotting: {e}")

# ARIMA Prediction Function
def train_arima(df):
    if len(df) < 10:
        raise ValueError("Not enough data points to fit ARIMA model.")
    model = ARIMA(df["Close"], order=(1, 1, 1))
    try:
        model_fit = model.fit()
    except IndexError:
        raise ValueError("ARIMA model failed due to insufficient data.")
    return model_fit

def show_insights(df_filtered):
    # Display filtered data for debugging
    if df_filtered.empty:
        st.error("No data available for the selected date range.")
        return
    
    st.write("### Filtered Data Preview for ARIMA Prediction")
    st.dataframe(df_filtered.tail())  # Display the last few rows for debugging
    
    # Handle missing values
    df_filtered = df_filtered.dropna(subset=["Close"])

    # Set Date column as the index for ARIMA model
    df_filtered.set_index("Date", inplace=True)

    # ARIMA prediction logic
    try:
        forecast_df = train_arima(df_filtered)
        st.write(f"### 🔮 ARIMA Prediction for {first_stock}")
        forecast_value = forecast_df.forecast(steps=1)[0]
        
        if forecast_value > df_filtered['Close'].iloc[-1] * 1.05:
            st.success("📈 **BUY:** Expected upward trend.")
        elif forecast_value < df_filtered['Close'].iloc[-1] * 0.95:
            st.error("📉 **SELL:** Expected downward trend.")
        else:
            st.warning("⚖ **HOLD:** Market stable.")
    except Exception as e:
        st.error(f"Error occurred during ARIMA prediction: {e}")

# Buttons with Functionality
if st.sidebar.button("📊 Compare Stocks"):
    show_comparison()
if st.sidebar.button("📈 View Trends"):
    show_trends(df_filtered)
if st.sidebar.button("🔮 AI Insights"):
    show_insights(df_filtered)
if st.sidebar.button("📜 Generate Report"):
    st.write("Report generation feature coming soon!")

