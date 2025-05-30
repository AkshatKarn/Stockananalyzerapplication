import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from fpdf import FPDF
import tempfile
import os

st.set_page_config(page_title="AI-Powered Stock Analyzer", layout="wide")
st.title("AI-Powered Stock Analyzer")
st.write("Analyze stocks, visualize trends, and get AI-driven insights!")

stocks = ["AAPL", "GOOGL", "TSLA", "AMZN", "MSFT", "NFLX", "NVDA", "META", "IBM", "INTC", "AMD", "BABA",
          "ORCL", "PYPL", "DIS", "PEP", "KO", "CSCO", "UBER", "LYFT"]

st.sidebar.header("Stock Selection & Customization")
menu_option = st.sidebar.radio("Menu", ["Stock Analysis", "Stock Comparison"])

selected_stocks = st.sidebar.multiselect("Select Stocks", stocks, default=["AAPL"])

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

    return insights, recommendation

def create_pdf_report(insights, df, stock):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Report for {stock}", ln=True, align='C')
    pdf.ln(10)
    for insight in insights:
        pdf.multi_cell(0, 10, insight.replace("**", ""))
    pdf.ln(10)
    pdf.multi_cell(0, 10, "Summary Table:")
    for index, row in df.iterrows():
        pdf.cell(0, 10, f"{row['Date'].date()} - {row['Close']:.2f} USD", ln=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        return tmp.name

# Main App Logic
if menu_option == "Stock Analysis":
    for stock in selected_stocks:
        df = load_data(stock)
        df_filtered = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
        if df_filtered.empty:
            st.error(f"No data for {stock} in the selected range.")
            continue

        st.subheader(f"Stock: {stock}")
        tab1, tab2, tab3 = st.tabs(["Visualizations", "Insights", "Table"])

        with tab1:
            fig = px.line(df_filtered, x="Date", y="Close", title=f"{stock} Price Over Time")
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

        with tab2:
            insights, _ = generate_insights(df_filtered)
            for insight in insights:
                st.markdown(insight)

        with tab3:
            st.dataframe(df_filtered[['Date', 'Open', 'High', 'Low', 'Close']], use_container_width=True)

        if st.button(f"Download Report for {stock}"):
            report_path = create_pdf_report(insights, df_filtered, stock)
            with open(report_path, "rb") as f:
                st.download_button("Download Report", f, file_name=f"{stock}_report.pdf", mime="application/pdf")

elif menu_option == "Stock Comparison":
    st.subheader("Stock Comparison")
    comparison_data = {}
    comparison_results = []

    for stock in selected_stocks:
        df = load_data(stock)
        df_filtered = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
        if df_filtered.empty:
            st.warning(f"No data for {stock} in selected range.")
            continue
        insights, recommendation = generate_insights(df_filtered)
        comparison_data[stock] = df_filtered
        comparison_results.append((stock, insights, recommendation))

    fig_compare = go.Figure()
    for stock, df_filtered in comparison_data.items():
        fig_compare.add_trace(go.Scatter(x=df_filtered["Date"], y=df_filtered["Close"], mode='lines', name=stock))
    fig_compare.update_layout(title="Closing Prices Comparison", xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig_compare, use_container_width=True)

    st.markdown("## Combined Insights")
    for stock, insights, recommendation in comparison_results:
        st.markdown(f"### {stock} — **{recommendation}**")
        for insight in insights:
            st.markdown(f"- {insight}")
        st.markdown("---")

    if st.button("Download Combined Report"):
        combined_pdf = FPDF()
        combined_pdf.add_page()
        combined_pdf.set_font("Arial", size=12)
        combined_pdf.cell(0, 10, "Combined Stock Comparison Report", ln=True, align='C')
        combined_pdf.ln(10)
        for stock, insights, _ in comparison_results:
            combined_pdf.multi_cell(0, 10, f"{stock} Insights:")
            for insight in insights:
                combined_pdf.multi_cell(0, 10, insight.replace("**", ""))
            combined_pdf.ln(5)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            combined_pdf.output(tmp.name)
            with open(tmp.name, "rb") as f:
                st.download_button("Download Combined Report", f, file_name="combined_report.pdf", mime="application/pdf")
