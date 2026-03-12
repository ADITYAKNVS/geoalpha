import os
from technical_guardrails import fetch_historical_data, fyers
from datetime import datetime, timedelta

def test_sym(sym):
    range_to = datetime.now().strftime("%Y-%m-%d")
    range_from = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
    data = {
        "symbol": sym,
        "resolution": "1D",
        "date_format": "1",
        "range_from": range_from,
        "range_to": range_to,
        "cont_flag": "1"
    }
    response = fyers.history(data=data)
    if response and response.get('s') == 'ok':
        print(f"SUCCESS {sym}: Candles: {len(response.get('candles', []))}")
    else:
        print(f"FAILED {sym}: {response}")

months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
for m in months:
    test_sym(f"MCX:GOLD26{m}FUT")
    test_sym(f"MCX:GOLDPETAL26{m}FUT")
    test_sym(f"MCX:GOLDM26{m}FUT")
