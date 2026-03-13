import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Ensure we load exactly what app.py loads
env_path = str(Path("/Users/knvsaditya/Documents/Playground 2/.env").absolute())
load_dotenv(env_path)

# Insert the path to import local modules
sys.path.insert(0, "/Users/knvsaditya/Documents/Playground 2")

import technical_guardrails as tg
from signal_combiner import HybridSignalCombiner

def main():
    print("Testing Metals Sector...")
    print(f"Index Ticker: {tg.SECTOR_INDEX_TICKERS['Metals']}")
    print(f"Stock Tickers: {tg.SECTOR_TICKERS['Metals']}")
    
    tg_instance = tg.TechnicalGuardrails()
    tech_data = tg_instance.analyze_sector("Metals")
    
    print("\n--- TECH DATA RETURNED ---")
    print(json.dumps(tech_data, indent=2, default=str))

    print("\n--- COMBINER REASONS ---")
    combiner = HybridSignalCombiner()
    # Mock sentiment for testing
    sent_data = {"sentiment_score": 0.5, "sentiment_label": "NEUTRAL", "headline_count": 0, "ranked_evidence": [], "crash_risk": False, "short_squeeze_risk": False}
    hs = combiner.combine_signals(tech_data, sent_data, "Metals", daily_change_pct=-4.82)
    
    print(f"Signal: {hs.signal}")
    print(f"Tech reasons passed to LLM (should be 5):")
    for r in getattr(hs, "tech_reasons", []):
         print(f"   -> {r}")
         
    print(f"\nFinal reasoning list:")
    for r in hs.reasoning:
         print(f"   -> {r}")

if __name__ == "__main__":
    main()
