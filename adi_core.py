import requests
import feedparser
import datetime
import pandas as pd
import os
import numpy as np
from bs4 import BeautifulSoup

# Categories and their weights
CATEGORIES = {
    "judicial": 0.15,
    "civil_service": 0.15,
    "civil_rights": 0.15,
    "media": 0.10,
    "rule_of_law": 0.10,
    "polarization": 0.10,
    "economy": 0.10,
    "foreign_policy": 0.05,
    "elections": 0.10
}

# Expanded Severity mapping
SEVERITY_MAP = {
    "ending crime and disorder": ("civil_rights", 10),
    "export of the american ai technology stack": ("foreign_policy", 5),
    "captive nations week": ("polarization", 4),
    "ban": ("civil_rights", 4),
    "tariff": ("economy", 3),
    "voter suppression": ("elections", 7),
    "court ruling overturns": ("rule_of_law", -5),
    "rights restored": ("civil_rights", -6),
    "supreme court blocks": ("rule_of_law", -6),
    "election safeguards": ("elections", -4),
    "freedom of press": ("media", -3),
    "judicial independence": ("judicial", -5),
    "civil liberties upheld": ("civil_rights", -5),
}

DECAY_FACTOR = 0.95

# Historical baselines
HISTORICAL_BASELINES = {
    "Weimar Germany (1929-1933)": [20, 25, 30, 40, 55, 70, 85],
    "Chile (1970-1973)": [15, 20, 28, 40, 60, 80],
    "Turkey (2013-2017)": [18, 22, 30, 45, 65, 78, 85]
}

# -------------------------------
# Data Sources
# -------------------------------
def scrape_whitehouse_actions():
    url = "https://www.whitehouse.gov/presidential-actions/"
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        actions = []
        for link in soup.select("h2 a, article a"):
            title = link.get_text(strip=True)
            href = link.get('href')
            if href and not href.startswith('http'):
                href = "https://www.whitehouse.gov" + href
            actions.append((title.lower(), href))
        return actions[:5]
    except Exception:
        return [("No White House actions found", "")]

def fetch_us_politics_news():
    try:
        # Reuters first
        feed = feedparser.parse("https://feeds.reuters.com/Reuters/PoliticsNews")
        if feed.entries:
            return [(entry.title.lower(), entry.link) for entry in feed.entries[:10]]
        # Fallback to BBC
        feed = feedparser.parse("http://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml")
        if feed.entries:
            return [(entry.title.lower(), entry.link) for entry in feed.entries[:10]]
        # AP News fallback
        feed = feedparser.parse("https://apnews.com/hub/ap-top-news?format=atom")
        return [(entry.title.lower(), entry.link) for entry in feed.entries[:10]]
    except Exception:
        return [("No political headlines found", "")]

# -------------------------------
# Scoring & ADI Calculations
# -------------------------------
def score_events(events):
    scores = {cat: 0 for cat in CATEGORIES}
    for event in [e[0].lower() for e in events]:
        for key, (cat, points) in SEVERITY_MAP.items():
            if key in event:
                scores[cat] = min(max(scores.get(cat, 0) + points, 0), 10)
    return scores

def calculate_adi(scores):
    return round(sum(scores[cat] * 10 * CATEGORIES[cat] for cat in CATEGORIES), 2)

def scale_to_historical(raw_score):
    if raw_score <= 0:
        return 0.0
    avg_end = np.mean([values[-1] for values in HISTORICAL_BASELINES.values()])
    return round((raw_score / 30) * avg_end, 2)

def get_shoe_level(adi_score):
    if adi_score < 30:
        return 1, "Stable"
    elif adi_score < 50:
        return 2, "Caution"
    elif adi_score < 70:
        return 3, "Warning"
    elif adi_score < 85:
        return 4, "Critical"
    else:
        return 5, "Emergency"

# -------------------------------
# Main Daily Run with Decay
# -------------------------------
def run_adi_daily():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data")
    os.makedirs(data_path, exist_ok=True)
    log_file = os.path.join(data_path, "adi_log.csv")

    # Load existing log
    df_old = pd.read_csv(log_file) if os.path.exists(log_file) else pd.DataFrame(columns=["Date", "ADI Score", "Shoe Level", "Status"])
    prev_score = df_old["ADI Score"].iloc[-1] if not df_old.empty else 0

    # Get events
    whitehouse_actions = scrape_whitehouse_actions()
    headlines = fetch_us_politics_news()
    events = whitehouse_actions + headlines
    scores = score_events(events)

    # Apply decay and calculate
    raw_adi_today = calculate_adi(scores)
    raw_adi = prev_score * DECAY_FACTOR + raw_adi_today
    scaled_adi = scale_to_historical(raw_adi)
    shoe_level, shoe_status = get_shoe_level(scaled_adi)
    today = datetime.date.today().strftime("%Y-%m-%d")

    # Update log
    df_new = pd.DataFrame([[today, scaled_adi, shoe_level, shoe_status]], columns=["Date", "ADI Score", "Shoe Level", "Status"])
    df = pd.concat([df_old, df_new]).drop_duplicates(subset=["Date"], keep="last")
    df.to_csv(log_file, index=False)

    # Forecast and historical context
    forecast = forecast_trend(df)
    historical_context = historical_comparison(scaled_adi)

    # Summary for display
    summary = format_summary(today, raw_adi, scaled_adi, shoe_level, shoe_status, whitehouse_actions, headlines)

    return summary, scaled_adi, shoe_level, shoe_status, forecast, historical_context
