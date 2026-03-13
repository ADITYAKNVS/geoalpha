"""
GeoAlpha Technical Guardrails — Hard-coded market indicators
============================================================
Provides RSI, Moving Averages, Volume anomaly detection as
strict boundaries that the ML model cannot override.
"""

import os
import logging
try:
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).with_name(".env"))
except ImportError:
    pass

import pandas as pd
from datetime import datetime, timedelta, timezone
from fyers_apiv3 import fyersModel
import numpy as np
try:
    import streamlit as st
except ImportError:
    st = None

logger = logging.getLogger(__name__)


def _get_runtime_secret(key: str) -> str:
    value = os.environ.get(key, "")
    if value:
        return value
    if st is not None:
        try:
            secret_value = st.secrets.get(key, "")
            if secret_value:
                return str(secret_value)
        except Exception:
            pass
    return ""


def _mask_secret(value: str, keep: int = 4) -> str:
    if not value:
        return "missing"
    if len(value) <= keep * 2:
        return "*" * len(value)
    return f"{value[:keep]}...{value[-keep:]}"


def get_fyers_credentials() -> tuple[str, str]:
    return _get_runtime_secret("FYERS_CLIENT_ID"), _get_runtime_secret("FYERS_ACCESS_TOKEN")


FYERS_CLIENT_ID, FYERS_ACCESS_TOKEN = get_fyers_credentials()
fyers = fyersModel.FyersModel(client_id=FYERS_CLIENT_ID, is_async=False, token=FYERS_ACCESS_TOKEN, log_path="")


def get_fyers_client():
    client_id, access_token = get_fyers_credentials()
    if not client_id or not access_token:
        raise RuntimeError("FYERS credentials are missing")
    return fyersModel.FyersModel(client_id=client_id, is_async=False, token=access_token, log_path="")


def get_fyers_debug_status() -> dict:
    client_id, access_token = get_fyers_credentials()
    source = "environment"
    if not os.environ.get("FYERS_CLIENT_ID") and st is not None:
        try:
            if st.secrets.get("FYERS_CLIENT_ID", ""):
                source = "streamlit_secrets"
        except Exception:
            pass
    return {
        "client_id_present": bool(client_id),
        "access_token_present": bool(access_token),
        "client_id_masked": _mask_secret(client_id),
        "access_token_masked": _mask_secret(access_token),
        "source": source if client_id or access_token else "missing",
    }


def check_token_expiry() -> dict:
    """Decode the Fyers JWT access token and return expiry status."""
    _, access_token = get_fyers_credentials()
    if not access_token:
        return {"status": "missing", "message": "No access token configured", "hours_remaining": 0}
    try:
        import base64, json as _json
        payload = access_token.split(".")[1]
        payload += "=" * (4 - len(payload) % 4)
        data = _json.loads(base64.urlsafe_b64decode(payload))
        exp_ts = data.get("exp", 0)
        now_ts = datetime.now().timestamp()
        hours_remaining = (exp_ts - now_ts) / 3600
        if hours_remaining <= 0:
            return {"status": "expired", "message": "Token has expired", "hours_remaining": 0}
        elif hours_remaining <= 1:
            return {"status": "expiring_soon", "message": f"Expires in {int(hours_remaining * 60)}min", "hours_remaining": round(hours_remaining, 1)}
        else:
            return {"status": "valid", "message": f"Valid for {hours_remaining:.1f}h", "hours_remaining": round(hours_remaining, 1)}
    except Exception as exc:
        logger.warning("Token expiry check failed: %s", exc)
        return {"status": "unknown", "message": "Could not decode token", "hours_remaining": 0}


def generate_fyers_auth_url() -> str:
    """Generate the Fyers OAuth login URL."""
    client_id, _ = get_fyers_credentials()
    secret_key = _get_runtime_secret("FYERS_SECRET_KEY")
    redirect_uri = _get_runtime_secret("FYERS_REDIRECT_URI") or "http://127.0.0.1"
    session = fyersModel.SessionModel(
        client_id=client_id,
        secret_key=secret_key,
        redirect_uri=redirect_uri,
        response_type="code",
        grant_type="authorization_code",
    )
    return session.generate_authcode()


def exchange_fyers_auth_code(auth_code: str) -> dict:
    """Exchange an auth code for an access token and save to .env."""
    client_id, _ = get_fyers_credentials()
    secret_key = _get_runtime_secret("FYERS_SECRET_KEY")
    redirect_uri = _get_runtime_secret("FYERS_REDIRECT_URI") or "http://127.0.0.1"
    session = fyersModel.SessionModel(
        client_id=client_id,
        secret_key=secret_key,
        redirect_uri=redirect_uri,
        response_type="code",
        grant_type="authorization_code",
    )
    session.set_token(auth_code)
    response = session.generate_token()
    if "access_token" in response:
        new_token = response["access_token"]
        # Save to .env file
        env_path = Path(__file__).with_name(".env")
        _set_env_var(str(env_path), "FYERS_ACCESS_TOKEN", new_token)
        # Also update the running environment
        os.environ["FYERS_ACCESS_TOKEN"] = new_token
        return {"success": True, "message": "Token refreshed!"}
    else:
        return {"success": False, "message": f"Fyers error: {response.get('message', 'Unknown error')}"}


def _set_env_var(filepath: str, key: str, value: str):
    """Write or update a key=value in a .env file."""
    lines = []
    found = False
    try:
        with open(filepath, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        pass
    with open(filepath, "w") as f:
        for line in lines:
            if line.startswith(f"{key}="):
                f.write(f"{key}={value}\n")
                found = True
            else:
                f.write(line)
        if not found:
            f.write(f"{key}={value}\n")


def map_yf_to_fyers(ticker: str) -> str:
    index_map = {
        "^CNXIT": "NSE:NIFTYIT-INDEX",
        "^NSEBANK": "NSE:NIFTYBANK-INDEX",
        "^CNXMETAL": "NSE:NIFTYMETAL-INDEX",
        "^CNXENERGY": "NSE:NIFTYENERGY-INDEX",
        "^CNXPHARMA": "NSE:NIFTYPHARMA-INDEX",
        "^CNXFMCG": "NSE:NIFTYFMCG-INDEX",
        "^CNXINFRA": "NSE:NIFTYINFRA-INDEX",
        "^NSEI": "NSE:NIFTY50-INDEX",
        "^BSESN": "BSE:SENSEX-INDEX",
    }
    if ticker in index_map:
        return index_map[ticker]

    if ticker in ["GC=F", "CL=F", "INR=X", "SI=F", "HG=F", "ALI=F"]:
        now = datetime.now()
        yy = str(now.year)[-2:]
        m_idx = now.month - 1
        months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

        if now.day > 18:
            m_idx = (m_idx + 1) % 12
            if m_idx == 0:
                yy = str(int(yy) + 1)

        cur_month = months[m_idx]

        if ticker == "GC=F":
            return f"MCX:GOLDPETAL{yy}{cur_month}FUT"
        if ticker == "CL=F":
            return f"MCX:CRUDEOIL{yy}{cur_month}FUT"
        if ticker == "INR=X":
            return f"NSE:USDINR{yy}{cur_month}FUT"
        if ticker == "SI=F":
            if m_idx % 2 == 0:
                m_idx = (m_idx + 1) % 12
                cur_month = months[m_idx]
            return f"MCX:SILVERMIC{yy}{cur_month}FUT"
        if ticker == "HG=F":
            return f"MCX:COPPER{yy}{cur_month}FUT"
        if ticker == "ALI=F":
            return f"MCX:ALUMINIUM{yy}{cur_month}FUT"

    if ticker.endswith(".NS"):
        return f"NSE:{ticker.replace('.NS', '')}-EQ"
    if ticker.endswith(".BO"):
        return f"BSE:{ticker.replace('.BO', '')}-EQ"
    return ticker


def get_live_quotes(tickers: list) -> dict:
    """Fetch live quotes (LTP, CH, CHP) for multiple tickers in one lightweight Fyers API call."""
    req_map = {map_yf_to_fyers(t): t for t in tickers}
    data = {"symbols": ",".join(req_map.keys())}
    results = {t: {"lp": 0, "ch": 0, "chp": 0} for t in tickers}

    try:
        response = get_fyers_client().quotes(data=data)
    except Exception as exc:
        logger.warning("Fyers live quote fetch failed for %s: %s", data["symbols"], exc)
        return results

    if response and response.get("s") == "ok":
        for item in response.get("d", []):
            f_sym = item.get("n")
            orig_ticker = req_map.get(f_sym)
            if orig_ticker:
                v = item.get("v", {})
                results[orig_ticker] = {
                    "lp": v.get("lp", 0),
                    "ch": v.get("ch", 0),
                    "chp": v.get("chp", 0),
                }
    else:
        logger.warning("Fyers live quote response not ok: %s", response)
    return results


def is_market_open() -> bool:
    """Check if current IST time is within Indian equity market hours (9:15 AM – 3:30 PM, weekdays)."""
    try:
        import pytz
        ist = pytz.timezone('Asia/Kolkata')
        now_ist = datetime.now(timezone.utc).astimezone(ist)
    except ImportError:
        # Fallback: assume the local timezone is IST
        now_ist = datetime.now()
    if now_ist.weekday() >= 5:  # Saturday / Sunday
        return False
    market_open = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now_ist <= market_close


def fetch_historical_data(ticker: str, period: str = "60d") -> pd.DataFrame:
    days = 60
    if period.endswith("d"):
        try:
            days = int(period[:-1]) * 2
        except ValueError:
            pass

    # AFTER-HOURS FIX: When the market is closed, bumping range_to to
    # tomorrow ensures the Fyers API includes today's completed daily
    # candle instead of omitting it or returning a placeholder row.
    now = datetime.now()
    if is_market_open():
        range_to = now.strftime("%Y-%m-%d")
    else:
        range_to = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    range_from = (now - timedelta(days=days)).strftime("%Y-%m-%d")
    fyers_symbol = map_yf_to_fyers(ticker)

    data = {
        "symbol": fyers_symbol,
        "resolution": "1D",
        "date_format": "1",
        "range_from": range_from,
        "range_to": range_to,
        "cont_flag": "1",
    }
    df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    try:
        response = get_fyers_client().history(data=data)
    except Exception as exc:
        logger.warning("Fyers historical fetch failed for %s (%s): %s", ticker, fyers_symbol, exc)
        return df
    if response and response.get("s") == "ok" and "candles" in response and len(response["candles"]) > 0:
        cols = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
        df = pd.DataFrame(response["candles"], columns=cols)
        df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")
        df.set_index("Datetime", inplace=True)
    else:
        logger.warning("Fyers historical response not ok for %s (%s): %s", ticker, fyers_symbol, response)

    # CLEANING & VALIDATION:
    # - Drop rows with NaN OHLC values
    # - Drop rows where prices are non-positive or all zeros
    # - After hours: drop any future-date placeholder candles
    # This ensures we only keep valid completed daily candles and never treat
    # placeholder / bad data as a real bar.
    if not df.empty:
        df = df[~df[["Open", "High", "Low", "Close"]].isna().any(axis=1)]
        df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]

        # After hours: trim any candle whose date is strictly after today
        # (could appear because we set range_to = tomorrow).
        if not is_market_open():
            today_end = pd.Timestamp(now.strftime("%Y-%m-%d")) + pd.Timedelta(days=1)
            df = df[df.index < today_end]

        if df.empty:
            logger.warning("All candles filtered out as invalid for %s (%s)", ticker, fyers_symbol)
            return df
        df.sort_index(inplace=True)

    return df



# ── Sector → representative tickers mapping ──────────────────
SECTOR_TICKERS = {
    "IT":             ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS"],
    "Banking":        ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS"],
    "Metals":         ["TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS"],
    "Oil & Gas":      ["RELIANCE.NS", "ONGC.NS", "BPCL.NS", "HINDPETRO.NS", "IOC.NS"],
    "Pharma":         ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS"],
    "FMCG":           ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS"],
    "Gold":           ["GOLDBEES.NS"],
    "Infrastructure": ["LT.NS", "IRB.NS"],
}

# Nifty sector index tickers (for sector-level signals)
SECTOR_INDEX_TICKERS = {
    "IT":             "^CNXIT",
    "Banking":        "^NSEBANK",
    "Metals":         "^CNXMETAL",
    "Oil & Gas":      "^CNXENERGY",
    "Pharma":         "^CNXPHARMA",
    "FMCG":           "^CNXFMCG",
    "Gold":           "GOLDBEES.NS",
    "Infrastructure": "^CNXINFRA",
}

# ── Subsector classification (upstream vs downstream) ─────────
SUBSECTOR_MAP = {
    "Oil & Gas": {
        "upstream": {
            "tickers": ["ONGC.NS", "OIL.NS"],
            "label": "Upstream (Exploration & Production)",
            "crude_impact": "POSITIVE",  # high crude = more revenue
            "logic": "Higher crude → higher realizations → better revenue",
        },
        "downstream": {
            "tickers": ["BPCL.NS", "HINDPETRO.NS", "IOC.NS"],
            "label": "Downstream (Refining & Marketing)",
            "crude_impact": "NEGATIVE",  # high crude = squeezed GRM
            "logic": "Higher crude → input cost rises → GRM compression → margin pressure",
        },
        "integrated": {
            "tickers": ["RELIANCE.NS"],
            "label": "Integrated (Refining + Petchem + Retail)",
            "crude_impact": "MIXED",  # hedged, diversified
            "logic": "Diversified across refining, petrochemicals, and retail — crude impact is mixed",
        },
    },
    "Metals": {
        "ferrous": {
            "tickers": ["TATASTEEL.NS", "JSWSTEEL.NS", "SAIL.NS"],
            "label": "Ferrous (Steel)",
            "driver": "China demand + steel HRC prices",
        },
        "non_ferrous": {
            "tickers": ["HINDALCO.NS", "NATIONALUM.NS"],
            "label": "Non-Ferrous (Aluminium/Copper)",
            "driver": "LME copper/aluminium prices + energy costs",
        },
    },
}

# ── Stock pick reasoning — WHY each stock benefits/suffers ────
STOCK_PICK_REASONS = {
    # Oil & Gas
    "ONGC.NS":       {"name": "ONGC",        "why_bull": "Largest upstream producer — directly benefits from higher crude realizations",
                                               "why_bear": "Revenue falls linearly with crude price decline"},
    "OIL.NS":        {"name": "Oil India",    "why_bull": "Pure-play upstream — crude price increase flows directly to top line",
                                               "why_bear": "High operating costs + aging fields reduce margin buffer"},
    "RELIANCE.NS":   {"name": "Reliance",     "why_bull": "Diversified across O2C, retail, Jio — crude impact hedged naturally",
                                               "why_bear": "Petchem margins under pressure if crude stays elevated"},
    "BPCL.NS":       {"name": "BPCL",         "why_bull": "Benefits from crude price drop — GRM expands, auto-fuel margins improve",
                                               "why_bear": "Rising crude compresses gross refining margin (GRM)"},
    "HINDPETRO.NS":  {"name": "HPCL",         "why_bull": "Marketing margin improvement when crude drops",
                                               "why_bear": "Most exposed to crude spikes — weakest balance sheet among OMCs"},
    "IOC.NS":        {"name": "IOC",           "why_bull": "Largest refiner — benefits from stable/falling crude",
                                               "why_bear": "Government-price-controlled fuel = margin squeeze on crude spike"},
    # IT
    "TCS.NS":        {"name": "TCS",           "why_bull": "Market leader — biggest beneficiary of weak INR + strong deal wins",
                                               "why_bear": "US recession fears → enterprise deal deferrals"},
    "INFY.NS":       {"name": "Infosys",       "why_bull": "Strong AI/cloud practice — benefits from digital transformation spend",
                                               "why_bear": "Client concentration risk + Nasdaq weakness"},
    "WIPRO.NS":      {"name": "Wipro",         "why_bull": "Turnaround story — cost optimization improving margins",
                                               "why_bear": "Weakest growth among top-4 IT — vulnerable to spending cuts"},
    "HCLTECH.NS":    {"name": "HCL Tech",      "why_bull": "Strong infrastructure management business — recurring revenue",
                                               "why_bear": "Product business volatility"},
    # Banking
    "HDFCBANK.NS":   {"name": "HDFC Bank",     "why_bull": "Best asset quality — CASA franchise strongest in India",
                                               "why_bear": "Post-merger deposit mobilization pressure"},
    "ICICIBANK.NS":  {"name": "ICICI Bank",    "why_bull": "Strong retail loan growth + improving asset quality",
                                               "why_bear": "Elevated wholesale book exposure"},
    "SBIN.NS":       {"name": "SBI",           "why_bull": "Largest lender — leads recovery after corrections",
                                               "why_bear": "PSU governance overhang — but never avoid in downturns"},
    "KOTAKBANK.NS":  {"name": "Kotak Bank",    "why_bull": "Premium franchise — low NPA, high CASA ratio",
                                               "why_bear": "Expensive valuations limit upside"},
    # Metals
    "TATASTEEL.NS":  {"name": "Tata Steel",    "why_bull": "Vertically integrated — benefits when steel HRC prices rise",
                                               "why_bear": "Europe operations drag + coking coal cost spikes"},
    "HINDALCO.NS":   {"name": "Hindalco",      "why_bull": "LME aluminium/copper rally directly boosts realizations",
                                               "why_bear": "Energy cost inflation + Novelis capex pressure"},
    "JSWSTEEL.NS":   {"name": "JSW Steel",     "why_bull": "Lowest cost domestic producer — benefits from India infra demand",
                                               "why_bear": "High debt + import competition from China"},
    # Pharma
    "SUNPHARMA.NS":  {"name": "Sun Pharma",    "why_bull": "Specialty portfolio de-risks from generic price erosion",
                                               "why_bear": "USFDA regulatory overhang"},
    "DRREDDY.NS":    {"name": "Dr Reddy's",    "why_bull": "Strong US generics pipeline + weak INR tailwind",
                                               "why_bear": "Product concentration risk"},
    "CIPLA.NS":      {"name": "Cipla",          "why_bull": "India + emerging markets strength — respiratory portfolio",
                                               "why_bear": "US business margin pressure"},
    # FMCG
    "HINDUNILVR.NS": {"name": "HUL",           "why_bull": "Pricing power + rural recovery benefits volume growth",
                                               "why_bear": "Input cost inflation + weak rural demand"},
    "ITC.NS":        {"name": "ITC",            "why_bull": "FMCG business scaling + cigarette pricing power + hotel recovery",
                                               "why_bear": "ESG concerns on tobacco + conglomerate discount"},
    "NESTLEIND.NS":  {"name": "Nestle India",  "why_bull": "Premium positioning — resilient margins",
                                               "why_bear": "Expensive valuations + limited rural penetration"},
    # Infrastructure
    "LT.NS":         {"name": "L&T",           "why_bull": "Largest order book — direct beneficiary of govt capex cycle",
                                               "why_bear": "Execution delays + working capital pressure"},
    "IRB.NS":        {"name": "IRB Infra",     "why_bull": "Toll road recovery + strong project execution pipeline",
                                               "why_bear": "Interest rate sensitivity + high leverage"},
}

# ── Sector-specific macro relevance weights ───────────────────
# What macro factors actually matter for each sector (0.0 = irrelevant, 1.0 = critical)
SECTOR_MACRO_WEIGHTS = {
    "IT":             {"nasdaq": 0.6, "usd_inr": 0.8, "crude": 0.1, "opec": 0.0, "china": 0.1, "rbi": 0.1,
                       "us_10y_yield": 0.15, "tech_spending": 0.9, "ai_capex": 0.9, "outsourcing_demand": 0.8},
    "Banking":        {"nasdaq": 0.1, "usd_inr": 0.1, "crude": 0.2, "opec": 0.0, "china": 0.1, "rbi": 0.9,
                       "domestic_bond_yield": 0.9, "credit_growth": 0.8, "us_10y_yield": 0.1},
    "Metals":         {"nasdaq": 0.0, "usd_inr": 0.3, "crude": 0.1, "opec": 0.0, "china": 0.9, "rbi": 0.1},
    "Oil & Gas":      {"nasdaq": 0.0, "usd_inr": 0.2, "crude": 1.0, "opec": 0.9, "china": 0.7, "rbi": 0.1},
    "Pharma":         {"nasdaq": 0.1, "usd_inr": 0.7, "crude": 0.1, "opec": 0.0, "china": 0.1, "rbi": 0.1},
    "FMCG":           {"nasdaq": 0.0, "usd_inr": 0.3, "crude": 0.3, "opec": 0.1, "china": 0.0, "rbi": 0.2},
    "Gold":           {"nasdaq": 0.1, "usd_inr": 0.5, "crude": 0.2, "opec": 0.1, "china": 0.1, "rbi": 0.2},
    "Infrastructure": {"nasdaq": 0.0, "usd_inr": 0.2, "crude": 0.2, "opec": 0.0, "china": 0.1, "rbi": 0.6},
}


class TechnicalGuardrails:
    """Hard-coded technical analysis engine."""

    def __init__(self, lookback_period: int = 60):
        self.lookback = f"{lookback_period}d"

    # ── RSI Calculation ───────────────────────────────────────
    @staticmethod
    def compute_rsi(closes: np.ndarray, period: int = 14) -> float:
        """
        Compute RSI from a close price array.
        RSI > 70 = overbought (bearish signal)
        RSI < 30 = oversold (bullish signal)
        """
        if len(closes) < period + 1:
            # Not enough history to compute a stable RSI signal
            return None

        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        if avg_loss == 0:
            return 100.0

        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return round(100.0 - (100.0 / (1.0 + rs)), 2)

    # ── Moving Average Crossover ──────────────────────────────
    @staticmethod
    def compute_ma_signal(closes: np.ndarray) -> dict:
        """
        20-day vs 50-day MA crossover detection.
        MA20 > MA50  = bullish (uptrend)
        MA20 < MA50  = bearish (downtrend)
        Recent crossover = strong signal
        """
        if len(closes) < 50:
            # Explicitly mark as insufficient data rather than fake neutral.
            return {
                "ma20": None,
                "ma50": None,
                "crossover": "INSUFFICIENT_DATA",
                "signal": "INSUFFICIENT_DATA",
                "strength": 0.0,
            }

        ma20 = np.mean(closes[-20:])
        ma50 = np.mean(closes[-50:])

        # Check for recent crossover (within last 5 days)
        recent_cross = False
        if len(closes) >= 55:
            for i in range(-5, 0):
                prev_ma20 = np.mean(closes[i - 20:i])
                prev_ma50 = np.mean(closes[i - 50:i])
                if (prev_ma20 < prev_ma50 and ma20 > ma50) or \
                   (prev_ma20 > prev_ma50 and ma20 < ma50):
                    recent_cross = True
                    break

        spread_pct = ((ma20 - ma50) / ma50) * 100

        # ── MA Spread Sanity Check ──
        # Typical spreads: 0-2% = weak, 2-5% = strong, >5% = extreme.
        # Spreads > 8% are extremely unusual and likely indicate:
        #   (1) a legitimate extreme trend, or
        #   (2) data quality issues (stale/adjusted prices)
        spread_warning = ""
        if abs(spread_pct) > 8.0:
            spread_warning = (f"⚠️ EXTREME MA SPREAD ({spread_pct:+.2f}%): "
                            f"This is unusually large. Verify data quality. "
                            f"Typical range: ±2-5%. Signal weight reduced.")
            # Cap the effective spread for scoring purposes
            spread_pct_effective = 8.0 if spread_pct > 0 else -8.0
        else:
            spread_pct_effective = spread_pct

        if ma20 > ma50:
            signal = "BULLISH"
            strength = min(1.0, abs(spread_pct_effective) / 5.0)
            if recent_cross:
                strength = min(1.0, strength + 0.3)
        elif ma20 < ma50:
            signal = "BEARISH"
            strength = min(1.0, abs(spread_pct_effective) / 5.0)
            if recent_cross:
                strength = min(1.0, strength + 0.3)
        else:
            signal = "NEUTRAL"
            strength = 0.0

        return {
            "ma20": round(ma20, 2),
            "ma50": round(ma50, 2),
            "spread_pct": round(spread_pct, 2),
            "spread_pct_effective": round(spread_pct_effective, 2),
            "spread_warning": spread_warning,
            "crossover": "recent_cross" if recent_cross else ("golden" if signal == "BULLISH" else "death" if signal == "BEARISH" else "flat"),
            "signal": signal,
            "strength": round(strength, 2),
        }

    # ── Volume Anomaly Detection ──────────────────────────────
    @staticmethod
    def detect_volume_anomaly(volumes: np.ndarray) -> dict:
        """
        Flag unusual volume (> 1.5x 20-day average).
        High volume on down move = elevated selling pressure
        High volume on up move = elevated buying pressure

        FIX v2.1: Handles zero-volume data (sector indices don't have volume).
        """
        if len(volumes) < 21:
            # Too few candles to build a robust 20‑day average.
            return {
                "is_anomaly": False,
                "ratio": None,
                "signal": "NO_DATA",
                "data_valid": False,
            }

        avg_20 = np.mean(volumes[-21:-1])  # 20-day avg excluding today
        today_vol = volumes[-1]

        # FIX: Sector indices (^CNXMETAL etc.) return 0 volume
        # Flag as invalid so combiner can fall back to stock-level volume
        if avg_20 == 0 or today_vol == 0:
            return {
                "is_anomaly": False,
                "ratio": 0.0,
                "signal": "NO_DATA",
                "data_valid": False,
            }

        ratio = today_vol / avg_20

        if ratio > 2.5:
            signal = "EXTREME"
        elif ratio > 1.5:
            signal = "HIGH"
        elif ratio >= 1.2:
            signal = "ABOVE_AVERAGE"
        elif ratio < 0.8:
            signal = "VERY_WEAK"
        elif ratio < 1.0:
            signal = "WEAK"
        else:
            signal = "NORMAL"

        return {
            "is_anomaly": ratio > 1.2 or ratio < 1.0,  # 1.2+ and <1.0 are anomalies
            "ratio": round(ratio, 2),
            "signal": signal,
            "data_valid": True,
        }

    # ── Daily Change Calculator ───────────────────────────────
    @staticmethod
    def compute_daily_change(closes: np.ndarray) -> dict:
        """Compute today's percentage change and classify volatility."""
        if len(closes) < 2:
            return {"change_pct": 0.0, "volatility_class": "normal"}

        change = ((closes[-1] - closes[-2]) / closes[-2]) * 100

        if abs(change) < 1.0:
            vol_class = "normal"
        elif abs(change) < 2.0:
            vol_class = "moderate"
        else:
            vol_class = "extreme"

        return {
            "change_pct": round(change, 2),
            "volatility_class": vol_class,
            "direction": "up" if change > 0 else "down" if change < 0 else "flat",
        }

    # ── Weekly Change Calculator ────────────────────────────────
    @staticmethod
    def compute_weekly_change(closes: np.ndarray) -> float:
        """
        v2.1: Compute 5-day (weekly) % change for multi-timeframe momentum.
        """
        if len(closes) < 6:
            return None
        return round(((closes[-1] - closes[-6]) / closes[-6]) * 100, 2)

    # ── Monthly Change Calculator ───────────────────────────────
    @staticmethod
    def compute_monthly_change(closes: np.ndarray) -> float:
        """
        v2.8: Compute 20-day (monthly) % change for multi-timeframe momentum.
        """
        if len(closes) < 21:
            return None
        return round(((closes[-1] - closes[-21]) / closes[-21]) * 100, 2)

    @staticmethod
    def compute_gap_context(opens: np.ndarray, closes: np.ndarray) -> dict:
        if len(opens) < 1 or len(closes) < 2:
            return {"gap_pct": 0.0, "abs_gap_pct": 0.0, "direction": "flat"}

        gap_pct = ((opens[-1] - closes[-2]) / closes[-2]) * 100
        return {
            "gap_pct": round(gap_pct, 2),
            "abs_gap_pct": round(abs(gap_pct), 2),
            "direction": "gap_up" if gap_pct > 0.15 else "gap_down" if gap_pct < -0.15 else "flat",
        }

    @staticmethod
    def compute_breakout_context(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> dict:
        if len(closes) < 21:
            return {"state": "INSIDE_RANGE", "distance_pct": 0.0}

        recent_high = np.max(highs[-21:-1])
        recent_low = np.min(lows[-21:-1])
        latest_close = closes[-1]

        if latest_close > recent_high:
            distance_pct = ((latest_close - recent_high) / recent_high) * 100
            state = "BREAKOUT"
        elif latest_close < recent_low:
            distance_pct = ((latest_close - recent_low) / recent_low) * 100
            state = "BREAKDOWN"
        else:
            upper_distance = ((recent_high - latest_close) / recent_high) * 100 if recent_high else 0.0
            lower_distance = ((latest_close - recent_low) / recent_low) * 100 if recent_low else 0.0
            distance_pct = min(upper_distance, lower_distance)
            state = "INSIDE_RANGE"

        return {
            "state": state,
            "distance_pct": round(distance_pct, 2),
            "recent_high": round(float(recent_high), 2),
            "recent_low": round(float(recent_low), 2),
        }

    @staticmethod
    def compute_intraday_context(highs: np.ndarray, lows: np.ndarray) -> dict:
        if len(highs) < 21 or len(lows) < 21:
            return {"range_pct": 0.0, "range_expansion": 1.0}

        latest_range = highs[-1] - lows[-1]
        avg_range = np.mean(highs[-21:-1] - lows[-21:-1])
        range_expansion = (latest_range / avg_range) if avg_range else 1.0

        return {
            "range_pct": round(float(latest_range), 2),
            "range_expansion": round(float(range_expansion), 2),
        }

    # ── Full Technical Analysis for a Ticker ──────────────────
    def analyze_ticker(self, ticker: str) -> dict:
        """Run full technical analysis on a single ticker."""
        try:
            data = fetch_historical_data(ticker, period=self.lookback)
            if len(data) < 5:
                return self._neutral_result(ticker, "insufficient data")

            closes = data["Close"].values.astype(float)
            opens = data["Open"].values.astype(float)
            highs = data["High"].values.astype(float)
            lows = data["Low"].values.astype(float)
            volumes = data["Volume"].values.astype(float)

            rsi = self.compute_rsi(closes)
            ma_signal = self.compute_ma_signal(closes)
            volume_info = self.detect_volume_anomaly(volumes)
            daily = self.compute_daily_change(closes)
            weekly_change = self.compute_weekly_change(closes)
            monthly_change = self.compute_monthly_change(closes)
            gap_context = self.compute_gap_context(opens, closes)
            breakout_context = self.compute_breakout_context(highs, lows, closes)
            intraday_context = self.compute_intraday_context(highs, lows)

            # If we don't have enough data for the core indicators (RSI, MA, volume),
            # return an explicit insufficient‑data result instead of pretending to be neutral.
            core_insufficient = (
                rsi is None
                and ma_signal.get("signal") == "INSUFFICIENT_DATA"
                and not volume_info.get("data_valid", False)
            )
            if core_insufficient:
                return self._neutral_result(ticker, "insufficient candles for RSI/MA/volume")

            # ── Composite signal ──
            score = 0.0  # -1.0 (full bearish) to +1.0 (full bullish)
            reasons = []

            # RSI contribution (weight: 0.30)
            # v2.9: 60-70 is moderately bullish (strong momentum), >70 is overbought (bearish)
            if rsi is not None and rsi >= 70:
                score -= 0.30
                reasons.append(f"RSI {rsi} — overbought")
            elif rsi is not None and rsi >= 60:
                score += 0.15
                reasons.append(f"RSI {rsi} — moderately bullish (strong momentum)")
            elif rsi is not None and rsi <= 30:
                score += 0.30
                reasons.append(f"RSI {rsi} — oversold (reversal likely)")
            elif rsi is not None and rsi <= 40:
                score -= 0.10
                reasons.append(f"RSI {rsi} — weak (near oversold)")
            elif rsi is not None:
                reasons.append(f"RSI {rsi} — neutral zone")

            # MA crossover contribution (weight: 0.35)
            if ma_signal["signal"] == "BULLISH":
                score += 0.35 * ma_signal["strength"]
                reasons.append(f"MA20 > MA50 (spread {ma_signal['spread_pct']}%) — uptrend")
            elif ma_signal["signal"] == "BEARISH":
                score -= 0.35 * ma_signal["strength"]
                reasons.append(f"MA20 < MA50 (spread {ma_signal['spread_pct']}%) — downtrend")
            if ma_signal.get("spread_warning"):
                reasons.append(ma_signal["spread_warning"])

            # ── RSI / MA Contradiction Detection ──
            if rsi < 30 and ma_signal["signal"] == "BULLISH":
                score -= 0.15
                reasons.append(
                    f"⚠️ RSI/MA CONTRADICTION: RSI {rsi} oversold "
                    f"BUT MA20>MA50 — sharp pullback in uptrend."
                )
            elif rsi > 70 and ma_signal["signal"] == "BEARISH":
                score += 0.15
                reasons.append(
                    f"⚠️ RSI/MA CONTRADICTION: RSI {rsi} overbought "
                    f"BUT MA20<MA50 — bounce in downtrend."
                )

            # Volume contribution (weight: 0.20)
            if volume_info.get("data_valid", True) and volume_info["is_anomaly"]:
                vol_direction = 1 if daily["change_pct"] > 0 else -1
                score += 0.20 * vol_direction
                reasons.append(
                    f"Volume {volume_info['ratio']}x avg — "
                    f"{'elevated buying pressure' if vol_direction > 0 else 'elevated selling pressure'}"
                )
            elif not volume_info.get("data_valid", True):
                reasons.append("Volume data unavailable for this index")

            # Daily change contribution (weight: 0.15)
            if daily["volatility_class"] == "extreme":
                change_dir = 1 if daily["change_pct"] > 0 else -1
                score += 0.15 * change_dir
                reasons.append(f"Extreme move {daily['change_pct']}%")

            if breakout_context["state"] == "BREAKOUT":
                score += 0.12
                reasons.append(f"Breakout above 20-day high ({breakout_context['distance_pct']}%)")
            elif breakout_context["state"] == "BREAKDOWN":
                score -= 0.12
                reasons.append(f"Breakdown below 20-day low ({breakout_context['distance_pct']}%)")

            if gap_context["abs_gap_pct"] >= 1.0:
                reasons.append(f"Opening gap {gap_context['gap_pct']:+.2f}%")

            # Classify final direction
            if score > 0.15:
                direction = "BULLISH"
            elif score < -0.15:
                direction = "BEARISH"
            else:
                direction = "NEUTRAL"

            confidence = min(1.0, abs(score) / 0.5)

            return {
                "ticker": ticker,
                "direction": direction,
                "confidence": round(confidence, 2),
                "score": round(score, 3),
                "rsi": rsi,
                "ma": ma_signal,
                "volume": volume_info,
                "daily_change": daily,
                "weekly_change": weekly_change,
                "monthly_change": monthly_change,
                "gap": gap_context,
                "breakout": breakout_context,
                "intraday_context": intraday_context,
                "reasons": reasons,
                "status": "ok",
            }

        except Exception as e:
            return self._neutral_result(ticker, str(e))

    # ── Sector-level Technical Analysis ───────────────────────
    def analyze_sector(self, sector: str) -> dict:
        """
        Analyze a sector using its index ticker + component stocks.
        Returns aggregated technical signal with subsector divergence.
        """
        index_ticker = SECTOR_INDEX_TICKERS.get(sector)
        stock_tickers = SECTOR_TICKERS.get(sector, [])

        # Index-level analysis (primary)
        index_result = None
        if index_ticker:
            index_result = self.analyze_ticker(index_ticker)

        # Stock-level analysis (supporting) — analyze ALL stocks for subsector detection
        stock_results = []
        for t in stock_tickers:
            r = self.analyze_ticker(t)
            if r["status"] == "ok":
                # Attach stock reasoning
                pick_info = STOCK_PICK_REASONS.get(t, {})
                r["stock_name"] = pick_info.get("name", t.replace(".NS", ""))
                r["why_bull"] = pick_info.get("why_bull", "Sector leader")
                r["why_bear"] = pick_info.get("why_bear", "Sector weakness")
                stock_results.append(r)

        # ── v2.1: Aggregate volume from stocks if index has no volume ──
        # Sector indices (^CNXMETAL etc.) return 0 volume.
        # Fall back to average of stock-level volume ratios.
        aggregated_volume = None
        if index_result and index_result.get("status") == "ok":
            idx_vol = index_result.get("volume", {})
            if not idx_vol.get("data_valid", True):
                # Index has no volume data — aggregate from stocks
                valid_vols = [
                    r["volume"] for r in stock_results
                    if r.get("volume", {}).get("data_valid", True)
                ]
                if valid_vols:
                    avg_ratio = float(np.mean([v["ratio"] for v in valid_vols]))
                    any_anomaly = any(v["is_anomaly"] for v in valid_vols)
                    if avg_ratio > 2.5:
                        agg_signal = "EXTREME"
                    elif avg_ratio > 1.5:
                        agg_signal = "HIGH"
                    elif avg_ratio >= 1.2:
                        agg_signal = "ABOVE_AVERAGE"
                    elif avg_ratio < 0.8:
                        agg_signal = "VERY_WEAK"
                    elif avg_ratio < 1.0:
                        agg_signal = "WEAK"
                    else:
                        agg_signal = "NORMAL"
                    aggregated_volume = {
                        "ratio": round(avg_ratio, 2),
                        "is_anomaly": any_anomaly,
                        "signal": agg_signal,
                        "data_valid": True,
                        "source": "stock_aggregate",
                    }
                    # Patch the index result so combiner sees correct volume
                    index_result["volume"] = aggregated_volume

        # Combine signals
        if index_result and index_result["status"] == "ok":
            primary_direction = index_result["direction"]
            primary_confidence = index_result["confidence"]
            primary_score = index_result["score"]
        elif stock_results:
            avg_score = np.mean([r["score"] for r in stock_results])
            primary_score = avg_score
            if avg_score > 0.15:
                primary_direction = "BULLISH"
            elif avg_score < -0.15:
                primary_direction = "BEARISH"
            else:
                primary_direction = "NEUTRAL"
            primary_confidence = min(1.0, abs(avg_score) / 0.5)
        else:
            primary_direction = "NEUTRAL"
            primary_confidence = 0.3
            primary_score = 0.0

        # Stock consensus
        if stock_results:
            bullish_count = sum(1 for r in stock_results if r["direction"] == "BULLISH")
            bearish_count = sum(1 for r in stock_results if r["direction"] == "BEARISH")
            consensus = "SPLIT"
            if bullish_count == len(stock_results):
                consensus = "ALL_BULLISH"
            elif bearish_count == len(stock_results):
                consensus = "ALL_BEARISH"
            elif bullish_count > bearish_count:
                consensus = "MOSTLY_BULLISH"
            elif bearish_count > bullish_count:
                consensus = "MOSTLY_BEARISH"
        else:
            consensus = "NO_DATA"

        # ── Subsector divergence detection ──
        subsector_analysis = {}
        if sector in SUBSECTOR_MAP:
            for sub_name, sub_info in SUBSECTOR_MAP[sector].items():
                sub_tickers = sub_info["tickers"]
                sub_results = [r for r in stock_results if r["ticker"] in sub_tickers]
                if sub_results:
                    avg_score = float(np.mean([r["score"] for r in sub_results]))
                    if avg_score > 0.15:
                        sub_dir = "BULLISH"
                    elif avg_score < -0.15:
                        sub_dir = "BEARISH"
                    else:
                        sub_dir = "NEUTRAL"
                    subsector_analysis[sub_name] = {
                        "label": sub_info["label"],
                        "direction": sub_dir,
                        "score": round(avg_score, 3),
                        "stocks": [r["stock_name"] for r in sub_results],
                        "extra": {k: v for k, v in sub_info.items() if k not in ("tickers", "label")},
                    }

        # Detect divergence
        has_divergence = False
        divergence_note = ""
        if len(subsector_analysis) >= 2:
            dirs = [v["direction"] for v in subsector_analysis.values()]
            if "BULLISH" in dirs and "BEARISH" in dirs:
                has_divergence = True
                bull_subs = [v["label"] for v in subsector_analysis.values() if v["direction"] == "BULLISH"]
                bear_subs = [v["label"] for v in subsector_analysis.values() if v["direction"] == "BEARISH"]
                divergence_note = f"⚠️ SUBSECTOR DIVERGENCE: {', '.join(bull_subs)} bullish vs {', '.join(bear_subs)} bearish"

        # ── Momentum-aware stock picks ──
        # v2.0: Stock picks MUST pass a momentum gate.
        # A stock with bullish technical score should NEVER be in avoid list.
        # A stock with bearish technical score should NEVER be in top picks.
        
        # v2.7: Top picks must also OUTPERFORM the aggregate sector change and be GREEN.
        if stock_results:
            sector_avg_change = np.mean([r.get("daily_change", {}).get("change_pct", 0.0) for r in stock_results])
        else:
            sector_avg_change = 0.0
            
        top_picks = []
        avoid_picks = []
        sorted_bullish = sorted(stock_results, key=lambda x: x["score"], reverse=True)
        sorted_bearish = sorted(stock_results, key=lambda x: x["score"])

        for r in sorted_bullish:
            if r["score"] > 0.05 and len(top_picks) < 2:
                daily = r.get("daily_change", {})
                vol = r.get("volume", {})
                chg = daily.get("change_pct", 0.0)
                v_ratio = vol.get("ratio", 1.0)
                
                # Rule: Protect against underperforming / red stocks being labeled 'Top Picks'
                if chg <= 0.0 or chg < sector_avg_change:
                    continue
                
                # Dynamic technical reason instead of hardcoded fundamentals
                if v_ratio > 1.2:
                    reason = f"Strong price/volume confirmation ({v_ratio:.1f}x avg volume)"
                elif chg > 1.5:
                    reason = "Strong intraday momentum"
                else:
                    reason = "Positive technical structure"
                    
                top_picks.append({
                    "name": r["stock_name"],
                    "reason": reason,
                    "score": r["score"],
                    "daily_change": daily.get("change_pct", 0.0),
                    "volume_ratio": vol.get("ratio", 1.0),
                    "rsi": r.get("rsi", 50.0),
                })

        for r in sorted_bearish:
            # MOMENTUM GATE: Only flag as avoid if technical score is ACTUALLY bearish
            daily = r.get("daily_change", {})
            d_chg = daily.get("change_pct", 0.0)
            
            # Rule: Protect ALL positive intraday movers from being 'Avoid'
            # If stock is green (> 0%), never flag as avoid for intraday.
            if d_chg > 0.0:
                continue
                
            if r["score"] < -0.05 and len(avoid_picks) < 2:
                vol = r.get("volume", {})
                v_ratio = vol.get("ratio", 1.0)
                
                # Dynamic technical reason instead of hardcoded fundamentals
                if d_chg < -1.5 and v_ratio > 1.2:
                    reason = f"Heavy selling pressure ({v_ratio:.1f}x avg volume)"
                elif d_chg < -1.0:
                    reason = "Sharp intraday weakness"
                elif r.get("rsi", 50.0) < 40:
                    reason = "Technically weak (RSI approaching oversold)"
                else:
                    reason = "Negative technical structure"

                avoid_picks.append({
                    "name": r["stock_name"],
                    "reason": reason,
                    "score": r["score"],
                    "daily_change": d_chg,
                    "volume_ratio": v_ratio,
                    "rsi": r.get("rsi", 50.0),
                })

        reasons = index_result["reasons"] if index_result and index_result["status"] == "ok" else []
        if divergence_note:
            reasons.append(divergence_note)

        # Sector-level daily change
        sector_daily_change = 0.0
        if index_result and index_result.get("status") == "ok":
            sector_daily_change = index_result.get("daily_change", {}).get("change_pct", 0.0)

        return {
            "sector": sector,
            "direction": primary_direction,
            "confidence": round(float(primary_confidence), 2),
            "score": round(float(primary_score), 3),
            "index_analysis": index_result,
            "stock_analyses": stock_results,
            "stock_consensus": consensus,
            "subsectors": subsector_analysis,
            "has_divergence": has_divergence,
            "top_picks": top_picks,
            "avoid_picks": avoid_picks,
            "macro_weights": SECTOR_MACRO_WEIGHTS.get(sector, {}),
            "reasons": reasons,
            "sector_daily_change": sector_daily_change,
        }

    @staticmethod
    def _neutral_result(ticker: str, reason: str) -> dict:
        return {
            "ticker": ticker,
            "direction": "NEUTRAL",
            "confidence": 0.3,
            "score": 0.0,
            "rsi": None,
            "ma": {
                "signal": "NEUTRAL",
                "strength": 0.0,
                "ma20": None,
                "ma50": None,
                "spread_pct": 0.0,
                "crossover": "N/A",
            },
            "volume": {
                "is_anomaly": False,
                "ratio": None,
                "signal": "NO_DATA",
                "data_valid": False,
            },
            "daily_change": {"change_pct": 0.0, "volatility_class": "normal", "direction": "flat"},
            "gap": {"gap_pct": 0.0, "abs_gap_pct": 0.0, "direction": "flat"},
            "breakout": {"state": "INSIDE_RANGE", "distance_pct": 0.0},
            "intraday_context": {"range_pct": 0.0, "range_expansion": 1.0},
            "reasons": [f"Data unavailable: {reason}"],
            "status": "error",
        }
