import hashlib
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Reuse your original scoring logic
from adi_core import CATEGORIES, SEVERITY_MAP

CACHE_FILE = "adi_event_cache.json"
SIMILARITY_THRESHOLD = 0.9  # Above this, assume no real change

def hash_events(events):
    """Generate a hash from the content of events."""
    concat = " ".join([title.lower() for title, _ in events])
    return hashlib.sha256(concat.encode('utf-8')).hexdigest()

def compute_similarity(events_a, events_b):
    """Use TF-IDF cosine similarity to detect similarity between event sets."""
    texts = [" ".join([t.lower() for t, _ in events_a]), " ".join([t.lower() for t, _ in events_b])]
    tfidf = TfidfVectorizer().fit_transform(texts)
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])
    return similarity[0][0]

def score_events(whitehouse_actions, headlines):
    """Score events as before."""
    scores = {cat: 0 for cat in CATEGORIES}
    matched = []

    for event in [e[0].lower() for e in whitehouse_actions]:
        for key, (cat, points) in SEVERITY_MAP.items():
            if key in event:
                scores[cat] = min(max(scores[cat] + points, 0), 10)
                matched.append(f"[WH] '{key}' matched '{event}' → +{points} to {cat}")

    for event in [e[0].lower() for e in headlines]:
        for key, (cat, points) in SEVERITY_MAP.items():
            if key in event:
                scaled_points = points * 0.05
                scores[cat] = min(max(scores[cat] + scaled_points, 0), 10)
                matched.append(f"[NEWS] '{key}' matched '{event}' → +{scaled_points:.2f} to {cat}")

    print("Matched Events:")
    for m in matched:
        print(m)

    return scores

def run_adi_agent(whitehouse_actions, headlines):
    """Run the smart agent logic before scoring."""
    if not os.path.exists(CACHE_FILE):
        cache = {}
    else:
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)

    # Build new hashes
    wh_hash = hash_events(whitehouse_actions)
    news_hash = hash_events(headlines)

    wh_sim = 0.0
    news_sim = 0.0

    if "wh_last" in cache and "news_last" in cache:
        wh_sim = compute_similarity(whitehouse_actions, cache["wh_last"])
        news_sim = compute_similarity(headlines, cache["news_last"])

    if wh_sim > SIMILARITY_THRESHOLD and news_sim > SIMILARITY_THRESHOLD:
        print(f"No significant change in content (WH sim: {wh_sim:.2f}, News sim: {news_sim:.2f}) — reusing last score.")
        return cache.get("last_score", {cat: 0 for cat in CATEGORIES})

    # Compute new score
    scores = score_events(whitehouse_actions, headlines)

    # Save state
    cache["wh_last"] = whitehouse_actions
    cache["news_last"] = headlines
    cache["last_score"] = scores

    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

    return scores
