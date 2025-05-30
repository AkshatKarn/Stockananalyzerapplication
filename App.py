import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="AI-Powered Stock Analyzer", layout="wide")
st.title("AI-Powered Stock Analyzer")
st.write("Analyze stocks, visualize trends, get AI-driven insights!")

stocks = ["AAPL", "GOOGL", "TSLA", "AMZN", "MSFT", "NFLX", "NVDA", "META", "IBM", "INTC", "AMD", "BABA",
          "ORCL", "PYPL", "DIS", "PEP", "KO", "CSCO", "UBER", "LYFT"]

menu = st.sidebar.radio("Menu", ["Stock Analysis", "Stock Comparison"])

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

def arima_forecast(df):
    df = df.set_index("Date")
    model = ARIMA(df["Close"], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)
    forecast_index = pd.date_range(start=df.index[-1], periods=30, freq='D')
    return forecast_index, forecast

if menu == "Stock Analysis":
    st.sidebar.header("Stock Selection")
    selected_stocks = st.sidebar.multiselect("Select Stocks to Analyze", stocks, default=["AAPL"])

    data_dict = {}
    for stock in selected_stocks:
        df = load_data(stock)
        df_filtered = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
        if df_filtered.empty:
            st.error(f"No data for {stock} in the selected range.")
            continue
        data_dict[stock] = df_filtered

    if data_dict:
        for stock, df_filtered in data_dict.items():
            st.subheader(f"Stock: {stock}")
            tab1, tab2, tab3, tab4 = st.tabs(["Visualizations", "Table", "Insights", "Investment Analysis"])

            with tab1:
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

                forecast_index, forecast = arima_forecast(df_filtered)
                fig_forecast = go.Figure()
                fig_forecast.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['Close'], name='Historical'))
                fig_forecast.add_trace(go.Scatter(x=forecast_index, y=forecast, name='ARIMA Forecast'))
                fig_forecast.update_layout(title="ARIMA Forecast")
                st.plotly_chart(fig_forecast, use_container_width=True)

            with tab2:
                st.dataframe(df_filtered[['Date', 'Open', 'High', 'Low', 'Close']], use_container_width=True)

            with tab3:
                insights, _ = generate_insights(df_filtered)
                for insight in insights:
                    st.markdown(insight)

            with tab4:
                st.markdown("### 📈 Investment Details")
                num_stocks = st.number_input(f"How many {stock} stocks did you buy?", min_value=0, value=0, key=f"{stock}_num")
                buy_price = st.number_input(f"At what price per stock did you buy {stock}?", min_value=0.0, value=0.0, key=f"{stock}_price")
                
                if num_stocks > 0 and buy_price > 0:
                    current_price = df_filtered["Close"].iloc[-1]
                    total_invested = num_stocks * buy_price
                    current_value = num_stocks * current_price
                    profit_loss = current_value - total_invested

                    st.write(f"**Current Price:** ${current_price:.2f}")
                    st.write(f"**Total Invested:** ${total_invested:.2f}")
                    st.write(f"**Current Value:** ${current_value:.2f}")
                    st.write(f"**Profit/Loss:** ${profit_loss:.2f}")
                    if profit_loss > 0:
                        st.success("You're in profit! 🎉")
                    elif profit_loss < 0:
                        st.error("You're at a loss. 📉")
                    else:
                        st.info("No gain, no loss. 📊")
                else:
                    st.info("Enter valid stock count and purchase price to see investment summary.")
