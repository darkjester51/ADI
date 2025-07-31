import requests
import feedparser
import datetime
import pandas as pd
import os
import numpy as np
from bs4 import BeautifulSoup

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

SEVERITY_MAP = {
    "ban": ("civil_rights", 4),
    "curfew": ("civil_rights", 5),
    "martial law": ("civil_rights", 8),
    "emergency declaration": ("civil_rights", 6),
    "suspension of rights": ("civil_rights", 8),
    "voter suppression": ("civil_rights", 7),
    "mass surveillance": ("civil_rights", 5),
    "protest ban": ("civil_rights", 6),
    "ending crime and disorder": ("civil_rights", 10),
    "rights restored": ("civil_rights", -6),

    "court ruling overturns": ("rule_of_law", -5),
    "indictment": ("rule_of_law", 5),
    "conviction": ("rule_of_law", 4),
    "arrest warrant": ("rule_of_law", 5),
    "legal immunity": ("rule_of_law", 4),
    "rule of law": ("rule_of_law", 5),
    "refusal to comply": ("rule_of_law", 6),
    "military tribunal": ("rule_of_law", 7),

    "executive order": ("civil_service", 3),
    "nominations sent": ("civil_service", 2),
    "appointed": ("civil_service", 2),
    "confirmed by senate": ("civil_service", -2),
    "presidential memorandum": ("civil_service", 3),
    "agency directive": ("civil_service", 2),
    "bureaucratic overhaul": ("civil_service", 4),

    "media censorship": ("media", 6),
    "social media ban": ("media", 5),
    "disinformation law": ("media", 4),
    "press credentials revoked": ("media", 6),
    "broadcast license revoked": ("media", 7),

    "supreme court": ("judicial", 4),
    "federal court": ("judicial", 3),
    "judicial review": ("judicial", 4),
    "court packing": ("judicial", 7),

    "election security": ("elections", 3),
    "voter id law": ("elections", 3),
    "gerrymandering": ("elections", 4),
    "vote recount": ("elections", 3),
    "electoral reform": ("elections", 2),
    "polling access limited": ("elections", 5),

    "tariff": ("economy", 3),
    "sanctions": ("foreign_policy", 5),
    "export controls": ("foreign_policy", 4),
    "foreign interference": ("foreign_policy", 6),
    "ai export": ("foreign_policy", 4),
    "expulsion of diplomats": ("foreign_policy", 5),
    "military aid": ("foreign_policy", 4),
    "nuclear weapons": ("foreign_policy", 7),
    "global conflict": ("foreign_policy", 6),
    "export of the american ai technology stack": ("foreign_policy", 5),

    "economic crisis": ("economy", 5),
    "inflation": ("economy", 4),
    "job losses": ("economy", 4),
    "recession": ("economy", 5),
    "stimulus": ("economy", -3),
    "imports": ("economy", 3),
    "duty-free": ("economy", 4),
    "trade restriction": ("economy", 5),
    "adjusting imports": ("economy", 3),

    "captive nations week": ("polarization", 4),
    "division": ("polarization", 3),
    "extremism": ("polarization", 4),
    "hate speech": ("polarization", 3),
    "civil unrest": ("polarization", 5),
    "armed protest": ("polarization", 6),
    "militia": ("polarization", 5),

    "rights expanded": ("civil_rights", -5),
    "judicial independence": ("rule_of_law", -5),
    "increased transparency": ("civil_service", -3),
    "anti-corruption effort": ("rule_of_law", -4),
}

HISTORICAL_BASELINES = {
    "Weimar Germany (1929-1933)": [20, 25, 30, 40, 55, 70, 85],
    "Chile (1970-1973)": [15, 20, 28, 40, 60, 80],
    "Turkey (2013-2017)": [18, 22, 30, 45, 65, 78, 85]
}

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

def score_events(whitehouse_actions, headlines):
    scores = {cat: 0 for cat in CATEGORIES}
    for event in [e[0].lower() for e in whitehouse_actions]:
        for key, (cat, points) in SEVERITY_MAP.items():
            if key in event:
                scores[cat] += points
    for event in [e[0].lower() for e in headlines]:
        for key, (cat, points) in SEVERITY_MAP.items():
            if key in event:
                scores[cat] += points * 0.05
    return scores

def calculate_adi_delta(scores):
    return round(sum(scores[cat] * CATEGORIES[cat] for cat in scores), 2)

def get_shoe_level(score):
    if score < 30:
        return 1, "Stable"
    elif score < 50:
        return 2, "Caution"
    elif score < 70:
        return 3, "Warning"
    elif score < 85:
        return 4, "Critical"
    else:
        return 5, "Emergency"

def historical_comparison(current_adi):
    results = []
    for regime, values in HISTORICAL_BASELINES.items():
        closest = min(values, key=lambda x: abs(x - current_adi))
        results.append(f"{regime} (closest drift point: {closest})")
    return results

def forecast_trend(df_log, months=6):
    if len(df_log) < 3:
        return "Not enough data for forecast."
    df_log = df_log.sort_values("Date")
    df_log["day_num"] = np.arange(len(df_log))
    x = df_log["day_num"].values
    y = df_log["ADI Score"].values
    slope = (y[-1] - y[0]) / (x[-1] - x[0] + 1e-6)
    future_score = y[-1] + slope * months * 30
    level, status = get_shoe_level(future_score)
    return f"If trend holds (+{slope:.2f}/day), ADI in {months}mo: {future_score:.1f} (Level {level} – {status})"

def format_summary(date, raw_delta, new_score, shoe_level, shoe_status, actions, headlines):
    lines = [
        f"**Date:** {date}",
        f"**Raw Daily Change:** {raw_delta}",
        f"**New ADI Score:** {new_score} (Level {shoe_level} – {shoe_status})",
        "",
        "**Top White House Actions:**"
    ]
    lines += [f"- [{a[0]}]({a[1]})" if a[1] else f"- {a[0]}" for a in actions]
    lines.append("")
    lines.append("**Top Headlines:**")
    lines += [f"- [{h[0]}]({h[1]})" if h[1] else f"- {h[0]}" for h in headlines]
    return "\n".join(lines)

def run_adi_daily():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data")
    os.makedirs(data_path, exist_ok=True)
    log_file = os.path.join(data_path, "adi_log.csv")

    today = datetime.date.today().strftime("%Y-%m-%d")
    last_score = 55.0
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        df = df.sort_values("Date")
        last_score = df.iloc[-1]["ADI Score"]

    whitehouse_actions = scrape_whitehouse_actions()
    headlines = fetch_us_politics_news()
    scores = score_events(whitehouse_actions, headlines)
    delta = calculate_adi_delta(scores)
    new_score = round(last_score + delta, 2)
    level, status = get_shoe_level(new_score)

    # Log the score
    df_today = pd.DataFrame([[today, new_score, level, status]],
                            columns=["Date", "ADI Score", "Shoe Level", "Status"])
    if os.path.exists(log_file):
        df_old = pd.read_csv(log_file)
        df = pd.concat([df_old, df_today]).drop_duplicates(subset=["Date"], keep="last")
    else:
        df = df_today
    df.to_csv(log_file, index=False)

    forecast = forecast_trend(df)
    context = historical_comparison(new_score)
    summary = format_summary(today, delta, new_score, level, status, whitehouse_actions, headlines)

    return summary, new_score, level, status, forecast, context, delta
