# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prophet import Prophet
import plotly.express as px
from sentiment_model import load_model, analyze_sentiment

st.set_page_config(page_title="Sentilyst: Intelligent Sentiment Forecasting Agent", layout="wide")

st.title("🧠 Sentilyst: Intelligent Sentiment Forecasting Agent")
st.write("Analyze and forecast sentiment trends from text data using transformers + Prophet.")

# ---- Section 1: Input ----
st.header("Step 1: Input Text Data")

option = st.radio("Select data source:", ["Manual Input", "Upload CSV"])

if option == "Manual Input":
    user_text = st.text_area("Enter text samples (one per line):")
    if user_text:
        texts = [t.strip() for t in user_text.split("\n") if t.strip()]
        df = pd.DataFrame({"date": [datetime.now()] * len(texts), "text": texts})
    else:
        df = pd.DataFrame(columns=["date", "text"])
else:
    uploaded = st.file_uploader("Upload CSV with 'text' and optional 'date' columns", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        if "date" not in df.columns:
            df["date"] = datetime.now()
    else:
        df = pd.DataFrame(columns=["date", "text"])

if df.empty:
    st.stop()

# ---- Section 2: Sentiment Analysis ----
st.header("Step 2: Sentiment Analysis")

tokenizer, model = load_model()
with st.spinner("Analyzing sentiments..."):
    sentiment_df = analyze_sentiment(df["text"].tolist(), tokenizer, model)
    df = df.reset_index(drop=True).join(sentiment_df.drop(columns=["text"]))
st.success("✅ Sentiment analysis completed!")

st.dataframe(df.head())

# ---- Section 3: Aggregate by Date ----
st.header("Step 3: Aggregate & Visualize")

df["date"] = pd.to_datetime(df["date"])
agg = (
    df.groupby(df["date"].dt.date)["sentiment_score"]
    .mean()
    .reset_index()
    .rename(columns={"date": "ds", "sentiment_score": "y"})
)
st.write("Aggregated daily sentiment:")
st.dataframe(agg.tail())

fig = px.line(agg, x="ds", y="y", title="Sentiment over time", markers=True)
st.plotly_chart(fig, use_container_width=True)

# ---- Section 4: Forecast ----
st.header("Step 4: Forecast Future Sentiment")

days = st.slider("Forecast days into the future", 7, 90, 30)
if len(agg) >= 5:
    m = Prophet(daily_seasonality=True)
    m.fit(agg)
    future = m.make_future_dataframe(periods=days)
    forecast = m.predict(future)
    fig2 = px.line(forecast, x="ds", y="yhat", title="Forecasted Sentiment Trend")
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning("Need at least 5 data points to forecast sentiment.")

# ---- Section 5: Save or Download ----
st.header("Step 5: Download Results")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Sentiment Results", csv, "sentilyst_results.csv", "text/csv")

st.success("Done! You’ve built and deployed an intelligent sentiment forecasting app.")
