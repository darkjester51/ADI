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

# Severity mapping
SEVERITY_MAP = {
    "ending crime and disorder": ("civil_rights", 10),
    "export of the american ai technology stack": ("foreign_policy", 5),
    "captive nations week": ("polarization", 4),
    "ban": ("civil_rights", 4),
    "tariff": ("economy", 3),
    "voter suppression": ("elections", 7),
    "court ruling overturns": ("rule_of_law", -5),
    "rights restored": ("civil_rights", -6)
}

DECAY_FACTOR = 0.95

# Historical baselines
HISTORICAL_BASELINES = {
    "Weimar Germany (1929-1933)": [20, 25, 30, 40, 55, 70, 85],
    "Chile (1970-1973)": [15, 20, 28, 40, 60, 80],
    "Turkey (2013-2017)": [18, 22, 30, 45, 65, 78, 85]
}

# Historical events
HISTORICAL_EVENTS = [
    ("2001-09-11", 20, "9/11 Terror Attacks"),
    ("2001-10-26", 30, "Patriot Act signed"),
    ("2003-03-19", 32, "Iraq War begins"),
    ("2008-09-15", 28, "2008 Financial crisis"),
    ("2013-06-05", 33, "Snowden NSA revelations"),
    ("2016-11-08", 35, "2016 Presidential Election"),
    ("2020-03-15", 40, "COVID lockdowns & emergency powers"),
    ("2020-06-01", 45, "George Floyd protests & civil unrest"),
    ("2021-01-06", 55, "Capitol Riot & election disputes"),
    ("2022-06-24", 50, "Roe v. Wade overturned"),
    ("2023-08-01", 45, "Federal indictments and polarization spike"),
    ("2024-11-05", 48, "2024 Presidential Election"),
]

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
        feed = feedparser.parse("https://feeds.reuters.com/Reuters/PoliticsNews")
        if feed.entries:
            return [(entry.title, entry.link) for entry in feed.entries[:10]]
        feed = feedparser.parse("http://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml")
        return [(entry.title, entry.link) for entry in feed.entries[:10]]
    except Exception:
        return [("No political headlines found", "")]

# -------------------------------
# Scoring & ADI Calculations
# -------------------------------
def score_events(whitehouse_actions, headlines):
    scores = {cat: 0 for cat in CATEGORIES}

    # Full weight for executive actions
    for event in [e[0].lower() for e in whitehouse_actions]:
        for key, (cat, points) in SEVERITY_MAP.items():
            if key in event:
                scores[cat] = min(max(scores[cat] + points, 0), 10)

    # Reduced weight (5%) for news headlines
    for event in [e[0].lower() for e in headlines]:
        for key, (cat, points) in SEVERITY_MAP.items():
            if key in event:
                scaled_points = points * 0.05
                scores[cat] = min(max(scores[cat] + scaled_points, 0), 10)

    return scores

def calculate_adi(scores):
    raw_score = sum(scores[cat] * 10 * CATEGORIES[cat] for cat in CATEGORIES)
    return round(raw_score, 2)

def scale_to_historical(raw_score):
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

def forecast_trend(df_log, months=6):
    if len(df_log) < 3:
        return "Not enough data for forecast."
    df_log = df_log.sort_values("Date")
    df_log["day_num"] = np.arange(len(df_log))
    x = df_log["day_num"].values
    y = df_log["ADI Score"].values
    slope = (y[-1] - y[0]) / (x[-1] - x[0] + 1e-6)
    forecast_days = int(months * 30)
    future_score = y[-1] + slope * forecast_days
    level, status = get_shoe_level(future_score)
    return f"If current trend continues (+{slope:.2f} points/day), projected ADI in {months} months: {future_score:.1f} (Shoe Level {level} - {status})."

def historical_comparison(current_adi):
    results = []
    for regime, values in HISTORICAL_BASELINES.items():
        closest = min(values, key=lambda x: abs(x - current_adi))
        results.append(f"{regime} (closest drift point: {closest})")
    return results

# -------------------------------
# Summary Formatting
# -------------------------------
def format_summary(date, raw_adi, scaled_adi, shoe_level, shoe_status, actions, headlines):
    formatted = [
        f"**Date:** {date}",
        f"**Raw ADI Score:** {raw_adi}",
        f"**Scaled ADI Score:** {scaled_adi} (Shoe Level {shoe_level} – {shoe_status})",
        "",
        "**Top White House Actions:**",
    ]
    if actions:
        formatted.extend([f"- [{a[0].capitalize()}]({a[1]})" if a[1] else f"- {a[0].capitalize()}" for a in actions])
    else:
        formatted.append("- None found")

    formatted.append("")
    formatted.append("**Top Headlines:**")
    if headlines:
        formatted.extend([f"- [{h[0].capitalize()}]({h[1]})" if h[1] else f"- {h[0].capitalize()}" for h in headlines])
    else:
        formatted.append("- None found")

    return "\n".join(formatted)

# -------------------------------
# U.S. ADI 30-Day Seeding
# -------------------------------
def seed_us_adi_log(log_file):
    if os.path.exists(log_file):
        return

    data = []
    today = datetime.date.today()
    base_score = 55.0
    for i in range(30):
        date = today - datetime.timedelta(days=29 - i)
        score = base_score + np.sin(i / 5) * 3
        level, status = get_shoe_level(score)
        data.append([date.strftime("%Y-%m-%d"), round(score, 2), level, status])

    df = pd.DataFrame(data, columns=["Date", "ADI Score", "Shoe Level", "Status"])
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    df.to_csv(log_file, index=False)
    print(f"Seeded {log_file} with 30 days of initial ADI data.")

# -------------------------------
# Main Daily Run
# -------------------------------
def run_adi_daily():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data")
    os.makedirs(data_path, exist_ok=True)
    log_file = os.path.join(data_path, "adi_log.csv")

    seed_us_adi_log(log_file)

    whitehouse_actions = scrape_whitehouse_actions()
    headlines = fetch_us_politics_news()
    scores = score_events(whitehouse_actions, headlines)
    raw_adi = calculate_adi(scores)
    scaled_adi = scale_to_historical(raw_adi)
    shoe_level, shoe_status = get_shoe_level(scaled_adi)
    today = datetime.date.today().strftime("%Y-%m-%d")

    df = pd.DataFrame([[today, scaled_adi, shoe_level, shoe_status]],
                      columns=["Date", "ADI Score", "Shoe Level", "Status"])
    if os.path.exists(log_file):
        df_old = pd.read_csv(log_file)
        df = pd.concat([df_old, df]).drop_duplicates(subset=["Date"], keep="last")
    df.to_csv(log_file, index=False)

    forecast = forecast_trend(df)
    historical_context = historical_comparison(scaled_adi)
    summary = format_summary(today, raw_adi, scaled_adi, shoe_level, shoe_status, whitehouse_actions, headlines)

    return summary, scaled_adi, shoe_level, shoe_status, forecast, historical_context
