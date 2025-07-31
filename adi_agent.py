import hashlib
import json
import os
from datetime import datetime

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# AI Agent for ADI Reflex Checkpoint
# -------------------------------
class ADIAgent:
    def __init__(self, cache_path="adi_event_cache.json"):
        self.cache_path = cache_path
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

    def _save_cache(self):
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f, indent=2)

    def _hash_events(self, events):
        flat = " ".join([e[0].lower() for e in events])
        return hashlib.sha256(flat.encode("utf-8")).hexdigest()

    def _get_event_text(self, events):
        return " ".join([e[0] for e in events])

    def detect_meaningful_change(self, current_wh, current_news):
        today = datetime.today().strftime("%Y-%m-%d")
        cur_hash = self._hash_events(current_wh + current_news)
        cur_text = self._get_event_text(current_wh + current_news)

        # Compare with last stored
        if "last_hash" in self.cache and self.cache["last_hash"] == cur_hash:
            print("🔁 No meaningful change – identical hash.")
            return False

        if "last_text" in self.cache:
            vec = TfidfVectorizer().fit_transform([self.cache["last_text"], cur_text])
            similarity = cosine_similarity(vec[0:1], vec[1:2])[0][0]
            print(f"🧠 Cosine similarity to last: {similarity:.3f}")
            if similarity > 0.9:
                print("⚖️ No meaningful change – high semantic similarity.")
                return False

        # Update cache
        self.cache = {
            "last_date": today,
            "last_hash": cur_hash,
            "last_text": cur_text
        }
        self._save_cache()
        print("✅ Meaningful change detected.")
        return True

# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    agent = ADIAgent()

    # Dummy test events (replace with live scraped data)
    yesterday_wh = [("President signs executive order on AI", "")]
    yesterday_news = [("Congress debates border security", "")]

    today_wh = [("President signs executive order on AI", "")]
    today_news = [("Congress debates border security and tech", "")]

    change = agent.detect_meaningful_change(today_wh, today_news)
    print("Should recalculate ADI:", change)
