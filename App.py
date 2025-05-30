import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
import plotly.io as pio
from fpdf import FPDF
import tempfile
import os

st.set_page_config(page_title="AI-Powered Stock Analyzer", layout="wide")
st.title("AI-Powered Stock Analyzer")
st.write("Analyze stocks, visualize trends, and get AI-driven insights!")

stocks = ["AAPL", "GOOGL", "TSLA", "AMZN", "MSFT", "NFLX", "NVDA", "META", "IBM", "INTC", "AMD", "BABA",
          "ORCL", "PYPL", "DIS", "PEP", "KO", "CSCO", "UBER", "LYFT"]

st.sidebar.header("Stock Selection & Customization")
selected_stocks = st.sidebar.multiselect("Select Stocks to Compare", stocks, default=["AAPL"])

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

st.sidebar.header("Select Date Range")
min_date = pd.to_datetime("2020-01-01")
max_date = pd.to_datetime("2026-12-31")

start_date = st.sidebar.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

def generate_insights(df):
    insights = []
    change = df["Close"].iloc[-1] - df["Close"].iloc[0]
    pct_change = (change / df["Close"].iloc[0]) * 100
    volatility = df["Close"].pct_change().std() * 100

    if pct_change > 10 and volatility < 2:
        recommendation = "BUY"
        reason = "Strong positive momentum with low volatility."
    elif pct_change < -10 and volatility > 2:
        recommendation = "SELL"
        reason = "Significant negative trend with high volatility."
    else:
        recommendation = "HOLD"
        reason = "Moderate changes with acceptable volatility."

    insights.append(f"**Recommendation: {recommendation}** — {reason}")
    trend = "uptrend" if change > 0 else "downtrend"
    insights.append(f"The stock showed an overall **{trend}** during the selected period.")
    insights.append(f"The stock changed by **{change:.2f} USD** (**{pct_change:.2f}%**) from start to end.")
    max_row = df.loc[df["Close"].idxmax()]
    min_row = df.loc[df["Close"].idxmin()]
    insights.append(f"Highest price: **{max_row['Close']:.2f} USD** on **{max_row['Date'].date()}**.")
    insights.append(f"Lowest price: **{min_row['Close']:.2f} USD** on **{min_row['Date'].date()}**.")
    insights.append(f"Approx. volatility: **{volatility:.2f}%**.")

    return insights

def create_pdf_report(insights, figures):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, "Stock Analysis Report", align='C')
    pdf.ln()
    for insight in insights:
        pdf.multi_cell(0, 10, insight.replace("**", ""))

    for fig in figures:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            pio.write_image(fig, tmpfile.name, format='png')
            pdf.add_page()
            pdf.image(tmpfile.name, x=10, y=10, w=pdf.w - 20)
            os.unlink(tmpfile.name)

    return pdf.output(dest='S').encode('utf-8')

# Load and visualize
data_dict = {}
for stock in selected_stocks:
    df = load_data(stock)
    df_filtered = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
    if df_filtered.empty:
        st.error(f"No data for {stock} in the selected range.")
        continue
    data_dict[stock] = df_filtered

# Visualization Section
if data_dict:
    for stock, df_filtered in data_dict.items():
        st.subheader(f"Stock: {stock}")
        tab1, tab2 = st.tabs(["Visualizations", "Insights"])

        with tab1:
            col1, col2 = st.columns([2, 1])

            with col1:
                fig = px.line(df_filtered, x="Date", y="Close", title=f"{stock} Price Over Time", color_discrete_sequence=["blue"])
                st.plotly_chart(fig, use_container_width=True)

                fig_candle = go.Figure(data=[go.Candlestick(x=df_filtered["Date"], open=df_filtered["Open"],
                                                            high=df_filtered["High"], low=df_filtered["Low"],
                                                            close=df_filtered["Close"])])
                fig_candle.update_layout(title="Candlestick Chart")
                st.plotly_chart(fig_candle, use_container_width=True)

                df_filtered['SMA_20'] = df_filtered['Close'].rolling(window=20).mean()
                df_filtered['Upper_BB'] = df_filtered['SMA_20'] + 2 * df_filtered['Close'].rolling(window=20).std()
                df_filtered['Lower_BB'] = df_filtered['SMA_20'] - 2 * df_filtered['Close'].rolling(window=20).std()
                fig_ma = px.line(df_filtered, x="Date", y=["Close", "SMA_20", "Upper_BB", "Lower_BB"],
                                 title="Moving Averages & Bollinger Bands")
                st.plotly_chart(fig_ma, use_container_width=True)

            with col2:
                st.dataframe(df_filtered[['Date', 'Open', 'High', 'Low', 'Close']].tail(10), use_container_width=True)

        with tab2:
            insights = generate_insights(df_filtered)
            for insight in insights:
                st.markdown(insight)
            st.markdown("---")
            st.markdown("**Note:** All data is simulated for demonstration purposes only.")

        # PDF download button for each stock
        if st.sidebar.button(f"Download {stock} Report"):
            pdf_data = create_pdf_report(insights, [fig, fig_candle, fig_ma])
            st.sidebar.download_button("Download Report", data=pdf_data, file_name=f"{stock}_report.pdf", mime="application/pdf")

    # Combined comparison
    st.subheader("Stock Comparison")
    fig_compare = go.Figure()
    for stock, df_filtered in data_dict.items():
        fig_compare.add_trace(go.Scatter(x=df_filtered["Date"], y=df_filtered["Close"], mode='lines', name=stock))
    fig_compare.update_layout(title="Closing Prices Comparison", xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig_compare, use_container_width=True)
