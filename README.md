# Authoritarian Drift Index (ADI) Dashboard v4.2

## How to Run
1. Install Python 3.9+.
2. Navigate to the ADIS_v4_2 folder.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Launch Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Features
- Weighted severity mapping for events.
- Bidirectional scoring with corrective events reducing the ADI.
- 3-6 month projected outlook based on trend velocity.
- Historical baseline comparison with Weimar Germany, Chile, and Turkey.
- Shoe Level gauge (1-5 shoes) and 30-day trend chart.
