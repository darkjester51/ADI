import streamlit as st
import pandas as pd
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from adi_core import run_adi_daily, HISTORICAL_BASELINES

LOG_FILE = "data/adi_log.csv"

# Page configuration
st.set_page_config(page_title="Authoritarian Drift Index (ADI) v4.6.1", layout="centered")
st.title("Authoritarian Drift Index (ADI) Dashboard v4.6.1")
st.subheader("Time to Leave Gauge with Historical Context")
st.markdown(
    """
    <div style='text-align: center; margin-bottom: 20px;'>
        <a href='https://buymeacoffee.com/yourname' target='_blank'>
            <button style='background-color:#ff9800; color:white; padding:10px 20px; border:none; border-radius:5px; font-size:16px; cursor:pointer;'>
                ☕ Support This Project
            </button>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

# Shoe Level Gauge
def shoe_meter(level):
    colors = {1: "green", 2: "limegreen", 3: "gold", 4: "orange", 5: "red"}
    icons = "👟" * level
    st.markdown(f"<h3 style='color:{colors[level]};'>{icons} - Level {level}</h3>", unsafe_allow_html=True)

# Static Historical Chart
def plot_static_historical_chart(current_adi):
    plt.figure(figsize=(9, 5))

    # Plot historical baselines
    for name, values in HISTORICAL_BASELINES.items():
        plt.plot(range(len(values)), values, linestyle='--', linewidth=2, label=name)

    # Add current ADI as horizontal line
    plt.axhline(current_adi, color='red', linestyle='-', linewidth=2, label=f"Current U.S. ADI ({current_adi})")

    plt.title("Historical Authoritarian Drift (Reference)")
    plt.xlabel("Relative Timeline (Index)")
    plt.ylabel("ADI Score (0-100)")
    plt.legend()
    st.pyplot(plt)

# Refresh Button
if st.button("🔄 Refresh Now"):
    with st.spinner("Pulling most current data..."):
        summary, adi_score, shoe_level, shoe_status, forecast, historical_context = run_adi_daily()
        st.success(f"Current ADI Score: {adi_score} (Shoe Level {shoe_level} – {shoe_status})")
        shoe_meter(shoe_level)
        st.markdown(summary)
        st.info(forecast)
        st.subheader("Historical Context:")
        for line in historical_context:
            st.write("- " + line)

        # Display static historical chart
        st.subheader("📊 Historical Reference Chart")
        plot_static_historical_chart(adi_score)

# U.S. ADI Trend Chart
if os.path.exists(LOG_FILE):
    st.subheader("📈 U.S. ADI Trend (Last 30 Days)")
    df_log = pd.read_csv(LOG_FILE)
    df_log["Date"] = pd.to_datetime(df_log["Date"])
    df_log = df_log.sort_values("Date")

    # Filter for last 30 days
    thirty_days_ago = datetime.datetime.now() - datetime.timedelta(days=30)
    df_30 = df_log[df_log["Date"] >= thirty_days_ago]

    if len(df_30) > 1:
        st.line_chart(df_30.set_index("Date")["ADI Score"])
    else:
        st.warning("Not enough data for a 30-day trend. Wait for more daily updates.")
