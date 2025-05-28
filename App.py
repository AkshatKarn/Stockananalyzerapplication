import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import date
from fpdf import FPDF
import os

# Helper to fetch stock data
def get_stock_data(symbol, start, end):
    return yf.download(symbol, start=start, end=end)

# Helper to predict future price
def predict_price(data):
    data = data.reset_index()
    data['Date_ordinal'] = pd.to_datetime(data['Date']).map(pd.Timestamp.toordinal)
    X = data['Date_ordinal'].values.reshape(-1, 1)
    y = data['Close'].values
    model = LinearRegression()
    model.fit(X, y)
    next_day = pd.Timestamp.today().toordinal() + 1
    predicted_price = model.predict([[next_day]])[0]
    return predicted_price, model

# Helper to create and download a PDF report
def create_report(symbol, df, predicted_price):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, f"Stock Analysis Report: {symbol}", ln=True, align='C')

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Predicted Closing Price for Next Day: ₹{predicted_price:.2f}", ln=True)

    plot_path = "stock_plot.png"
    plt.figure(figsize=(10, 4))
    plt.plot(df['Close'], label='Close Price')
    plt.title(f"{symbol} Close Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    pdf.image(plot_path, x=10, y=50, w=190)
    pdf.output("stock_report.pdf")
    os.remove(plot_path)

    with open("stock_report.pdf", "rb") as f:
        st.download_button("Download Report", f, file_name="stock_report.pdf")

# Streamlit UI
st.set_page_config(page_title="Stock Analyzer", layout="wide")

st.title("📈 Stock Analysis & Prediction App")

# Sidebar for user input
st.sidebar.header("Select Stock and Date Range")
stock_list = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'INFY.NS', 'RELIANCE.NS']
selected_stock = st.sidebar.selectbox("Choose a Stock Symbol", stock_list)
custom_stock = st.sidebar.text_input("Or Enter Custom Stock Symbol (e.g., TCS.NS)")

symbol = custom_stock if custom_stock else selected_stock

start_date = st.sidebar.date_input("Start Date", value=date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", value=date.today())

if start_date >= end_date:
    st.error("End date must be after start date.")
else:
    if st.sidebar.button("Fetch & Analyze"):
        with st.spinner("Fetching data..."):
            data = get_stock_data(symbol, start_date, end_date)

        if data.empty:
            st.error("No data found. Try a different stock symbol or date range.")
        else:
            st.success(f"Showing data for {symbol}")

            st.subheader(f"📊 Historical Closing Prices: {symbol}")
            st.line_chart(data['Close'])

            predicted_price, model = predict_price(data)
            st.metric("Predicted Closing Price (Next Day)", f"₹{predicted_price:.2f}")

            st.subheader("📉 Trendline (Linear Regression)")
            data_reset = data.reset_index()
            data_reset['Date_ordinal'] = pd.to_datetime(data_reset['Date']).map(pd.Timestamp.toordinal)
            plt.figure(figsize=(10, 4))
            plt.plot(data_reset['Date'], data_reset['Close'], label='Actual')
            plt.plot(data_reset['Date'], model.predict(data_reset[['Date_ordinal']]), label='Trendline', linestyle='--')
            plt.xlabel("Date")
            plt.ylabel("Close Price")
            plt.title(f"{symbol} Price Trend")
            plt.legend()
            st.pyplot(plt)

            st.subheader("📄 Generate Report")
            if st.button("Create PDF Report"):
                create_report(symbol, data, predicted_price)

