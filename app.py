import streamlit as st
import pandas as pd
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from adi_core import run_adi_daily, HISTORICAL_BASELINES

LOG_FILE = "data/adi_log.csv"

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# Page configuration
st.set_page_config(page_title="Authoritarian Drift Index (ADI) v4.6.1", layout="centered")
st.title("Authoritarian Drift Index (ADI) Dashboard v4.6.1")
st.subheader("Real Time Democracy Gauge with Historical Context")

st.markdown(
    """
    <div style='text-align: center; margin-bottom: 20px;'>
        <b>Support ADI and Further Projects</b><br>
        Your support helps keep ADI live and improving.<br><br>
        <a href='https://cash.app/$Stoller139' target='_blank' style='text-decoration:none;'>
            <div style='display:inline-block; background-color:#28a745; color:white; padding:10px 20px; border-radius:5px; font-size:16px; margin-right:10px;'>
                💵 Donate via CashApp
            </div>
        </a>
        <a href='https://venmo.com/ZetaCronSolutions' target='_blank' style='text-decoration:none;'>
            <div style='display:inline-block; background-color:#3D95CE; color:white; padding:10px 20px; border-radius:5px; font-size:16px;'>
                💳 Donate via Venmo
            </div>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

# =======================
# About ADI Section
# =======================
st.markdown(
    """
    ### About the Authoritarian Drift Index (ADI)

    The **Authoritarian Drift Index (ADI)** is a data-driven, educational tool designed to track the state of democratic health and authoritarian trends in the United States. 
    It is based on a **mathematical scoring algorithm** inspired by **Freedom House's Global Freedom Index**, focusing on civil liberties, rule of law, free press, and political rights.

    **ADI is not propaganda or a political endorsement.** It does not aim to influence voting or policy decisions but provides an **objective, historical context** for current events.

    **Shoe Level System:**  
    - **👟 Level 1 – Stable (0–29):** Healthy democracy, minimal authoritarian drift.  
    - **👟👟 Level 2 – Caution (30–49):** Early warning signs of instability.  
    - **👟👟👟 Level 3 – Warning (50–69):** Authoritarian trends are increasing.  
    - **👟👟👟👟 Level 4 – Critical (70–84):** Significant risk to democratic institutions.  
    - **👟👟👟👟👟 Level 5 – Emergency (85–100):** Conditions resemble authoritarian regimes.

    **Disclaimer:** ADI is for educational use only. It is an independent project and is not affiliated with any government, organization, or advocacy group.
    """
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

# Logging function
def log_adi_score(score):
    os.makedirs("data", exist_ok=True)
    log_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow(["Date", "ADI Score"])
        writer.writerow([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), score])

# Refresh Button
if st.button("🔄 Refresh Now"):
    with st.spinner("Pulling most current data..."):
        summary, scaled_adi, shoe_level, shoe_status, forecast, historical_context, raw_adi = run_adi_daily()
        
        st.write(f"DEBUG: Raw ADI Score = {raw_adi}")
        st.success(f"Current ADI Score: {scaled_adi} (Shoe Level {shoe_level} – {shoe_status})")
        
        shoe_meter(shoe_level)
        st.markdown(summary)
        st.info(forecast)

        st.subheader("Historical Context:")
        for line in historical_context:
            st.write("- " + line)

        log_adi_score(scaled_adi)  # Still logging the scaled score

        st.subheader("📊 Historical Reference Chart")
        plot_static_historical_chart(scaled_adi)

# U.S. ADI Trend Chart
if os.path.exists(LOG_FILE):
    try:
        df_log = pd.read_csv(LOG_FILE)
        if "Date" in df_log.columns and "ADI Score" in df_log.columns:
            # Clean and parse Date column
            df_log["Date"] = pd.to_datetime(df_log["Date"], errors='coerce')
            df_log = df_log.dropna(subset=["Date"])
            df_log = df_log.sort_values("Date")

            if len(df_log) > 1:
                # Determine the time span in years
                date_span = (df_log["Date"].max() - df_log["Date"].min()).days / 365.25
                if date_span >= 10:
                    chart_title = "📈 U.S. ADI Trend (All Time View)"
                elif date_span >= 5:
                    chart_title = "📈 U.S. ADI Trend (5-Year View)"
                else:
                    chart_title = "📈 U.S. ADI Trend (Recent View)"

                st.subheader(chart_title)
                st.line_chart(df_log.set_index("Date")["ADI Score"])
            else:
                st.warning("Not enough data for trend visualization.")
        else:
            st.warning("CSV file missing required columns: Date and ADI Score.")
    except Exception as e:
        st.error(f"Error loading ADI log: {e}")
