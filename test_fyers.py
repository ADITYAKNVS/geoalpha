import os
from technical_guardrails import fetch_historical_data, fyers
from datetime import datetime, timedelta

def test_sym(sym):
    print(f"Testing {sym}...")
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
        print("SUCCESS! Candles:", len(response.get('candles', [])))
    else:
        print("FAILED:", response)

test_sym("MCX:MCXCOMDEX-INDEX")
test_sym("MCX:CRUDEOIL-INDEX")
test_sym("MCX:CRUDEOIL25MARFUT")
test_sym("MCX:GOLD25APRZUT")
test_sym("MCX:GOLD25APRFUT")
test_sym("NSE:USDINR24AUGFUT")
test_sym("NSE:USDINR25MARFUT")
