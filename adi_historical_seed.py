import os
import pandas as pd
from adi_core import get_shoe_level

# Historical events with approximate ADI scores (scaled to 0–100 trajectory)
HISTORICAL_EVENTS = [
    ("2001-09-11", 20, "9/11 Terror Attacks"),
    ("2001-10-26", 30, "Patriot Act signed"),
    ("2003-03-19", 32, "Iraq War begins"),
    ("2008-09-15", 28, "Financial crisis"),
    ("2013-06-05", 33, "Snowden NSA revelations"),
    ("2016-11-08", 35, "2016 Presidential Election"),
    ("2020-03-15", 40, "COVID lockdowns & emergency powers"),
    ("2020-06-01", 45, "George Floyd protests & civil unrest"),
    ("2021-01-06", 55, "Capitol Riot & election disputes"),
    ("2022-06-24", 50, "Roe v. Wade overturned"),
    ("2023-08-01", 45, "Federal indictments and polarization spike"),
    ("2024-11-05", 48, "2024 Presidential Election"),
]

def seed_historical_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data")
    os.makedirs(data_path, exist_ok=True)
    log_file = os.path.join(data_path, "adi_log.csv")

    data = []
    for date, score, event in HISTORICAL_EVENTS:
        shoe_level, status = get_shoe_level(score)
        data.append([date, score, shoe_level, status, event])

    df = pd.DataFrame(data, columns=["Date", "ADI Score", "Shoe Level", "Status", "Event"])
    df = df.sort_values("Date")
    df.to_csv(log_file, index=False)

    print(f"Seeded {len(HISTORICAL_EVENTS)} historical events into {log_file}")

if __name__ == "__main__":
    seed_historical_data()
