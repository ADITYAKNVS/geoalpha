import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st
from fyers_apiv3 import fyersModel
import requests
import plotly.graph_objects as go
import logging
import json
import re
from html import escape
from groq import Groq
from google import genai as google_genai
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytz
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh

# ── Market Hours Auto-refresh ──
ist = pytz.timezone("Asia/Kolkata")
now = datetime.now(ist)

market_open = (
    now.weekday() < 5 and
    (now.hour > 9 or (now.hour == 9 and now.minute >= 15)) and
    (now.hour < 15 or (now.hour == 15 and now.minute <= 30))
)

if market_open:
    st_autorefresh(interval=10000, key="data_refresh")
else:
    st.info("🕒 Market is closed. Live tracking paused.")

# ── Local ML modules ──
from sentiment_engine import SentimentEngine
from sector_report_utils import (
    build_sector_stock_contribution_lines,
    build_sector_technical_snapshot,
    inject_sector_technical_sections,
    classify_rsi_zone,
)
from technical_guardrails import (
    TechnicalGuardrails,
    SECTOR_TICKERS,
    STOCK_PICK_REASONS,
    fetch_historical_data,
    map_yf_to_fyers,
    get_live_quotes,
    get_fyers_client,
    get_fyers_debug_status,
    check_token_expiry,
    generate_fyers_auth_url,
    exchange_fyers_auth_code,
)
from signal_combiner import HybridSignalCombiner

# ── API Keys ──
logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
NEWSDATA_API_KEY = os.environ.get("NEWSDATA_API_KEY", "")
MARKETAUX_API_KEY = os.environ.get("MARKETAUX_API_KEY", "")
GNEWS_API_KEY = os.environ.get("GNEWS_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
FYERS_CLIENT_ID = os.environ.get("FYERS_CLIENT_ID", "")
FYERS_ACCESS_TOKEN = os.environ.get("FYERS_ACCESS_TOKEN", "")

client = Groq(api_key=GROQ_API_KEY)
gemini_client = google_genai.Client(api_key=GEMINI_API_KEY)
fyers = fyersModel.FyersModel(client_id=FYERS_CLIENT_ID, is_async=False, token=FYERS_ACCESS_TOKEN, log_path="")

# ── ML Engine Singletons (cached) ──
@st.cache_resource
def load_sentiment_engine():
    return SentimentEngine()

def load_technical_guardrails():
    return TechnicalGuardrails()

def load_signal_combiner():
    return HybridSignalCombiner()

# ── Page Config ──
st.set_page_config(page_title="GeoAlpha — Hybrid Market Intelligence", page_icon="🌍", layout="wide")

# ── DARK THEME CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg-primary: #0a0a0f;
    --bg-secondary: #12121a;
    --bg-card: rgba(20, 20, 35, 0.8);
    --border-glass: rgba(255, 255, 255, 0.06);
    --text-primary: #e8e8f0;
    --text-secondary: #8888a0;
    --accent-green: #00e676;
    --accent-red: #ff1744;
    --accent-yellow: #ffab00;
    --accent-blue: #2979ff;
    --accent-purple: #7c4dff;
    --glow-green: rgba(0, 230, 118, 0.3);
    --glow-red: rgba(255, 23, 68, 0.3);
    --glow-yellow: rgba(255, 171, 0, 0.3);
}

.stApp {
    background: linear-gradient(135deg, var(--bg-primary) 0%, #0d0d1a 50%, var(--bg-primary) 100%) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
}

header[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding-top: 2rem; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #141428 100%) !important;
    border-right: 1px solid var(--border-glass);
}
section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* Glassmorphic Cards */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--border-glass);
    border-radius: 16px;
    padding: 20px;
    margin: 8px 0;
    transition: transform 0.2s ease, box-shadow 0.3s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
}

/* Signal Cards */
.signal-card-invest {
    background: linear-gradient(135deg, rgba(0,230,118,0.08), rgba(0,230,118,0.02));
    border: 1px solid rgba(0,230,118,0.2);
    border-radius: 16px;
    padding: 20px;
    margin: 8px 0;
    box-shadow: 0 0 20px var(--glow-green);
}
.signal-card-hold {
    background: linear-gradient(135deg, rgba(255,171,0,0.08), rgba(255,171,0,0.02));
    border: 1px solid rgba(255,171,0,0.2);
    border-radius: 16px;
    padding: 20px;
    margin: 8px 0;
    box-shadow: 0 0 20px var(--glow-yellow);
}
.signal-card-avoid {
    background: linear-gradient(135deg, rgba(255,23,68,0.08), rgba(255,23,68,0.02));
    border: 1px solid rgba(255,23,68,0.2);
    border-radius: 16px;
    padding: 20px;
    margin: 8px 0;
    box-shadow: 0 0 20px var(--glow-red);
}

/* Metric Styles */
.metric-box {
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-glass);
    border-radius: 14px;
    padding: 16px;
    text-align: center;
}
.metric-label { font-size: 12px; color: var(--text-secondary); font-weight: 500; letter-spacing: 0.5px; text-transform: uppercase; }
.metric-value { font-size: 24px; font-weight: 700; color: var(--text-primary); margin: 6px 0; }
.metric-delta-up { color: var(--accent-green); font-size: 13px; font-weight: 600; }
.metric-delta-down { color: var(--accent-red); font-size: 13px; font-weight: 600; }

/* Confidence Bar */
.conf-bar-bg { background: rgba(255,255,255,0.05); border-radius: 8px; height: 8px; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 8px; transition: width 0.8s ease; }

/* Fear & Greed Gauge */
.fg-gauge { text-align: center; padding: 20px; }
.fg-score { font-size: 64px; font-weight: 800; margin: 10px 0; }
.fg-label { font-size: 18px; font-weight: 600; letter-spacing: 1px; }

/* Pulse animation */
@keyframes pulse-green { 0%, 100% { box-shadow: 0 0 10px var(--glow-green); } 50% { box-shadow: 0 0 30px var(--glow-green); } }
@keyframes pulse-red { 0%, 100% { box-shadow: 0 0 10px var(--glow-red); } 50% { box-shadow: 0 0 30px var(--glow-red); } }
.crash-warning { animation: pulse-red 2s infinite; }
.rally-alert { animation: pulse-green 2s infinite; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }

/* Dividers */
hr { border-color: var(--border-glass) !important; }

/* Streamlit overrides */
.stMetric label, .stMetric [data-testid="stMetricValue"] { color: var(--text-primary) !important; }
.stMarkdown, .stMarkdown p, .stMarkdown li { color: var(--text-primary) !important; }
div[data-testid="stExpander"] { background: var(--bg-card); border: 1px solid var(--border-glass); border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ── Logo loading ──
import base64, os as _os
_logo_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "geoalpha_logo.png")
_logo_b64 = ""
_header_logo = '<span style="font-size:50px;margin-right:10px;">🌍</span>'

if _os.path.exists(_logo_path):
    with open(_logo_path, "rb") as _f:
        _logo_b64 = base64.b64encode(_f.read()).decode()
    _header_logo = f'<img src="data:image/png;base64,{_logo_b64}" style="width:60px;height:60px;border-radius:14px;margin-right:14px;vertical-align:middle;">'

st.markdown(f"""
<div style="text-align:center; padding: 10px 0 5px;">
    <div style="display:inline-flex;align-items:center;justify-content:center;">
        {_header_logo}
        <div style="text-align:left;">
            <h1 style="font-size:42px; font-weight:800; background: linear-gradient(135deg, #7c4dff, #2979ff, #00e676); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin:0;">GeoAlpha</h1>
        </div>
    </div>
    <p style="color: var(--text-secondary); font-size: 15px; margin-top:4px;">Hybrid ML + Technical Analysis • FinBERT NLP Engine • Real-time Intelligence</p>
</div>
""", unsafe_allow_html=True)
st.divider()

# ── Sidebar ──
_sidebar_logo_html = f'<img src="data:image/png;base64,{_logo_b64}" style="width:36px;height:36px;border-radius:8px;vertical-align:middle;margin-right:8px;">' if _logo_b64 else '🌍'
st.sidebar.markdown(f'{_sidebar_logo_html} <span style="font-size:18px;font-weight:700;vertical-align:middle;">GeoAlpha</span>', unsafe_allow_html=True)
st.sidebar.markdown("### Select Sectors")
all_sectors = {
    "💻 IT": "IT", "⛽ Oil & Gas": "Oil & Gas", "🏦 Banking": "Banking",
    "🛒 FMCG": "FMCG", "⚙️ Metals": "Metals", "💊 Pharma": "Pharma",
    "🏗️ Infrastructure": "Infrastructure", "🥇 Gold": "Gold"
}
selected = [v for l, v in all_sectors.items() if st.sidebar.checkbox(l, value=False)]

available_stock_spotlights = []
for sector in selected:
    for ticker in SECTOR_TICKERS.get(sector, []):
        stock_name = STOCK_PICK_REASONS.get(ticker, {}).get("name", ticker.replace(".NS", ""))
        available_stock_spotlights.append((f"{stock_name} ({ticker})", sector, ticker))

st.sidebar.divider()
st.sidebar.markdown("### News Display")
show_global = st.sidebar.checkbox("🌐 Show Global News", value=True)
show_indian = st.sidebar.checkbox("📰 Show Indian News", value=True)

st.sidebar.divider()
st.sidebar.markdown("### Analysis Depth")
deep_dive = st.sidebar.checkbox("🔬 Deep Dive Mode", value=False)
st.sidebar.caption("Institutional-grade report with Gemini 2.5 Flash")

st.sidebar.divider()
st.sidebar.markdown("### Stock Spotlight")
stock_spotlight_options = ["None"] + [label for label, _, _ in available_stock_spotlights]
selected_stock_label = st.sidebar.selectbox("Explain one stock in detail", stock_spotlight_options, index=0)
selected_stock = next(
    ((sector, ticker) for label, sector, ticker in available_stock_spotlights if label == selected_stock_label),
    None,
)

st.sidebar.divider()
generate = st.sidebar.button("🔍 Generate Report", width="stretch")

# ── Fyers Token Management ──
st.sidebar.divider()
token_info = check_token_expiry()
if token_info["status"] == "valid":
    _token_icon = "🟢"
    _token_color = "#00e676"
elif token_info["status"] == "expiring_soon":
    _token_icon = "🟡"
    _token_color = "#ffab00"
else:
    _token_icon = "🔴"
    _token_color = "#ff1744"

st.sidebar.markdown(f"""
<div style="background:rgba(255,255,255,0.03);border-radius:10px;padding:12px;border-left:3px solid {_token_color};">
    <div style="font-size:13px;font-weight:700;color:#e8e8f0;">🔑 Fyers Token</div>
    <div style="font-size:12px;color:{_token_color};margin-top:4px;">{_token_icon} {token_info['message']}</div>
</div>
""", unsafe_allow_html=True)

if token_info["status"] in ("expired", "expiring_soon", "missing"):
    st.sidebar.warning("⚠️ Refresh your Fyers token to get live Indian market data.")

with st.sidebar.expander("🔄 Refresh Token", expanded=token_info["status"] in ("expired", "missing")):
    # Security gate — prevent exposing client ID in auth URL to random users
    if "fyers_admin_unlocked" not in st.session_state:
        st.session_state.fyers_admin_unlocked = False

    if not st.session_state.fyers_admin_unlocked:
        st.caption("🔒 Enter admin PIN to access token refresh")
        admin_pin = st.text_input("Admin PIN", type="password", key="fyers_admin_pin", label_visibility="collapsed", placeholder="Enter PIN...")
        # PIN = last 4 chars of FYERS_CLIENT_ID (e.g. "-100" from "DRPRLG8GTH-100")
        clean_client_id = FYERS_CLIENT_ID.strip()
        expected_pin = clean_client_id[-4:] if len(clean_client_id) >= 4 else clean_client_id
        if st.button("🔓 Unlock", use_container_width=True):
            if admin_pin.strip() == expected_pin:
                st.session_state.fyers_admin_unlocked = True
                st.rerun()
            else:
                st.error("Incorrect PIN")
                print(f"Failed login attempt: Entered '{admin_pin}', Expected '{expected_pin}'")
    else:
        st.caption("**Step 1:** Click the link below to log in to Fyers")
        try:
            auth_url = generate_fyers_auth_url()
            st.markdown(f"[🔗 Open Fyers Login]({auth_url})")
        except Exception as exc:
            st.error(f"Could not generate auth URL: {exc}")
            auth_url = None

        st.caption("**Step 2:** After login, copy the `auth_code` from the redirect URL")
        auth_code_input = st.text_input("Paste auth_code here", key="fyers_auth_code", label_visibility="collapsed", placeholder="Paste auth_code here...")

        if st.button("✅ Exchange Token", disabled=not auth_code_input, use_container_width=True):
            with st.spinner("Exchanging token..."):
                result = exchange_fyers_auth_code(auth_code_input.strip())
            if result["success"]:
                st.success(result["message"])
                st.session_state.fyers_admin_unlocked = False
                st.cache_data.clear()
                st.rerun()
            else:
                st.error(result["message"])


def log_fetch_warning(context, exc):
    logger.warning("%s failed: %s", context, exc)


def canonicalize_headline(text):
    normalized = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    normalized = re.sub(r"\b(reuters|bloomberg|cnbc|marketaux|gnews|finnhub|newsdata)\b", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    tokens = normalized.split()
    return " ".join(tokens[:12])


def normalize_article(headline, summary="", url="", image="", source="", published_at="", bucket="sectoral"):
    return {
        "headline": (headline or "").strip(),
        "summary": (summary or "").strip(),
        "url": url or "",
        "image": image or "",
        "source": source or "",
        "published_at": published_at or "",
        "bucket": bucket or "sectoral",
    }


def _is_article_fresh(article, max_age_days=5):
    """Return True if the article was published within the last max_age_days days."""
    pub = article.get("published_at", "")
    if not pub:
        return True  # No date → let it through (better to show than miss)

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=max_age_days)

    # Try epoch timestamp (Finnhub sends integer epoch)
    try:
        ts = float(pub)
        if ts > 1e9:  # Looks like a valid epoch
            pub_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            return pub_dt >= cutoff
    except (ValueError, TypeError, OSError):
        pass

    # Try ISO / common string formats
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d %H:%M:%S %z", "%Y-%m-%d"):
        try:
            pub_dt = datetime.strptime(str(pub).strip(), fmt)
            if pub_dt.tzinfo is None:
                pub_dt = pub_dt.replace(tzinfo=timezone.utc)
            return pub_dt >= cutoff
        except ValueError:
            continue

    # Fallback: try dateutil if available
    try:
        from dateutil import parser as du_parser
        pub_dt = du_parser.parse(str(pub))
        if pub_dt.tzinfo is None:
            pub_dt = pub_dt.replace(tzinfo=timezone.utc)
        return pub_dt >= cutoff
    except Exception:
        pass

    return True  # Unparseable date → let it through


def append_article(target, seen, article):
    headline = article.get("headline", "").strip()
    canonical = canonicalize_headline(headline)
    if not headline or canonical in seen:
        return
    # STRICT: Reject articles older than 5 days
    if not _is_article_fresh(article, max_age_days=5):
        return
    seen.add(canonical)
    target.append(article)


def classify_news_bucket(headline, summary=""):
    text = f"{headline} {summary}".lower()
    global_macro_keywords = [
        "war", "sanction", "opec", "crude", "fed", "bond yield", "dollar",
        "china", "global", "nasdaq", "geopolitical", "tariff", "export"
    ]
    regulatory_keywords = [
        "rbi", "policy", "regulation", "regulatory", "usfda", "approval",
        "government", "cabinet", "ministry", "tax", "duty", "tender", "order"
    ]

    if any(keyword in text for keyword in global_macro_keywords):
        return "global_macro"
    if any(keyword in text for keyword in regulatory_keywords):
        return "regulatory"
    return "sectoral"


def fetch_marketaux_articles(search_query, limit=5, countries=None):
    articles = []
    five_days_ago = (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y-%m-%dT%H:%M")
    params = {
        "api_token": MARKETAUX_API_KEY,
        "search": search_query,
        "language": "en",
        "limit": limit,
        "filter_entities": "true",
        "published_after": five_days_ago,
    }
    if countries:
        params["countries"] = countries

    try:
        response = requests.get(
            "https://api.marketaux.com/v1/news/all",
            params=params,
            timeout=8,
        )
        response.raise_for_status()
        for article in response.json().get("data", []):
            headline = article.get("title", "")
            summary = article.get("description", "")
            articles.append(
                normalize_article(
                    headline=headline,
                    summary=summary,
                    url=article.get("url", ""),
                    image=article.get("image_url", ""),
                    source="MarketAux",
                    published_at=article.get("published_at", ""),
                    bucket=classify_news_bucket(headline, summary),
                )
            )
    except Exception as exc:
        log_fetch_warning(f"MarketAux query [{search_query}]", exc)

    return articles


def fetch_gnews_articles(search_query, limit=5):
    articles = []
    five_days_ago = (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        response = requests.get(
            "https://gnews.io/api/v4/search",
            params={
                "q": search_query,
                "lang": "en",
                "max": limit,
                "apikey": GNEWS_API_KEY,
                "from": five_days_ago,
            },
            timeout=8,
        )
        response.raise_for_status()
        for article in response.json().get("articles", []):
            headline = article.get("title", "")
            summary = article.get("description", "")
            articles.append(
                normalize_article(
                    headline=headline,
                    summary=summary,
                    url=article.get("url", ""),
                    image=article.get("image", ""),
                    source="GNews",
                    published_at=article.get("publishedAt", ""),
                    bucket=classify_news_bucket(headline, summary),
                )
            )
    except Exception as exc:
        log_fetch_warning(f"GNews query [{search_query}]", exc)

    return articles


SOURCE_QUALITY_SCORES = {
    "MarketAux": 0.90,
    "Finnhub": 0.85,
    "NewsData": 0.72,
    "GNews": 0.65,
}


def parse_published_at(value):
    if not value:
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        if text.isdigit():
            return datetime.fromtimestamp(int(text), tz=timezone.utc)
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def compute_recency_score(published_at):
    dt = parse_published_at(published_at)
    if dt is None:
        return 0.5

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    age_hours = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 3600)
    if age_hours <= 6:
        return 1.0
    if age_hours <= 24:
        return 0.9
    if age_hours <= 48:
        return 0.75
    if age_hours <= 96:
        return 0.6
    return 0.4


def infer_time_horizon(article):
    text = f"{article.get('headline', '')} {article.get('summary', '')}".lower()
    recency = compute_recency_score(article.get("published_at", ""))

    immediate_keywords = [
        "today", "intraday", "sanction", "war", "tariff", "yield", "fed",
        "opec", "policy decision", "hearing", "court", "expiry", "block deal"
    ]
    medium_keywords = [
        "credit growth", "order book", "capex", "guidance", "expansion",
        "roadmap", "investment", "demand outlook", "pipeline", "festival season"
    ]

    if any(keyword in text for keyword in immediate_keywords) or recency >= 0.9:
        return "immediate"
    if any(keyword in text for keyword in medium_keywords):
        return "medium_term"
    return "near_term"


def describe_time_horizon(time_horizon):
    return {
        "immediate": "immediate",
        "near_term": "near-term",
        "medium_term": "medium-term",
    }.get(time_horizon, "near-term")


def annotate_articles_with_horizon(articles):
    annotated = []
    for article in articles:
        item = dict(article)
        item["time_horizon"] = infer_time_horizon(item)
        annotated.append(item)
    return annotated


def classify_reason_category(article):
    text = f"{article.get('headline', '')} {article.get('summary', '')}".lower()
    if any(token in text for token in ("court orders", "court order", "embassy", "drama platform", "entertainment", "streaming platform")):
        return "macro"
    if any(token in text for token in ("earnings", "results", "profit", "revenue", "guidance", "quarter", "ebitda")):
        return "earnings"
    if any(token in text for token in ("approval", "policy", "regulation", "regulatory", "rbi", "usfda", "tax", "duty", "ministry")):
        return "regulation"
    if any(token in text for token in ("order win", "contract", "deal", "project award", "tender", "order inflow")):
        return "order win"
    if any(token in text for token in ("crude", "oil", "gold", "bullion", "copper", "aluminium", "iron ore", "yield", "dollar")):
        return "commodity"
    if article.get("bucket") == "global_macro":
        return "macro"
    if article.get("bucket") == "regulatory":
        return "regulation"
    return "sector sympathy"


def derive_event_taxonomy(ranked_evidence, move_type, technical=None, contradiction=False):
    taxonomy = []
    for item in ranked_evidence:
        tag = item.get("reason_category")
        if tag and tag not in taxonomy:
            taxonomy.append(tag)

    if move_type == "TECHNICAL_ONLY" and "technical-only" not in taxonomy:
        taxonomy.append("technical-only")

    if technical:
        if technical.get("index_analysis"):
            breakout_state = technical.get("index_analysis", {}).get("breakout", {}).get("state")
        else:
            breakout_state = technical.get("breakout", {}).get("state")
        if breakout_state in {"BREAKOUT", "BREAKDOWN"} and "technical breakout" not in taxonomy:
            taxonomy.append("technical breakout")

    if contradiction and "contradiction/divergence" not in taxonomy:
        taxonomy.append("contradiction/divergence")

    return taxonomy


def compute_peer_confirmation(stock_consensus, article_label, daily_change_pct):
    bullish_consensus = stock_consensus in {"ALL_BULLISH", "MOSTLY_BULLISH"}
    bearish_consensus = stock_consensus in {"ALL_BEARISH", "MOSTLY_BEARISH"}

    if article_label == "positive" and daily_change_pct > 0 and bullish_consensus:
        return 1.0
    if article_label == "negative" and daily_change_pct < 0 and bearish_consensus:
        return 1.0
    if stock_consensus == "SPLIT":
        return 0.55
    return 0.4


def score_ranked_reason(article, daily_change_pct, stock_consensus):
    relevance = article.get("relevance", {})
    direct_match_count = len(relevance.get("matched_keywords", []))
    contextual_match_count = len(relevance.get("macro_matches", [])) + len(relevance.get("regulatory_matches", []))
    entity_strength = min(1.0, (direct_match_count * 2 + contextual_match_count) / 6)
    recency = compute_recency_score(article.get("published_at", ""))
    source_quality = SOURCE_QUALITY_SCORES.get(article.get("source", ""), 0.55)

    label = article.get("label", "neutral")
    if label == "positive" and daily_change_pct > 0.2:
        price_confirmation = 1.0
    elif label == "negative" and daily_change_pct < -0.2:
        price_confirmation = 1.0
    elif label == "neutral":
        price_confirmation = 0.6
    else:
        price_confirmation = 0.25

    peer_confirmation = compute_peer_confirmation(stock_consensus, label, daily_change_pct)
    relevance_conf = relevance.get("confidence", 0.2)

    score = (
        relevance_conf * 0.28 +
        entity_strength * 0.18 +
        recency * 0.16 +
        source_quality * 0.14 +
        price_confirmation * 0.14 +
        peer_confirmation * 0.10
    )

    return round(score, 3), {
        "relevance": round(relevance_conf, 2),
        "entity_strength": round(entity_strength, 2),
        "recency": round(recency, 2),
        "source_quality": round(source_quality, 2),
        "price_confirmation": round(price_confirmation, 2),
        "peer_confirmation": round(peer_confirmation, 2),
    }


def build_causality_note(top_evidence, move_type, evidence_state):
    if evidence_state == "INSUFFICIENT_EVIDENCE":
        return "Evidence is too weak to claim a causal driver; treat the move as correlation plus flow/technical context."
    if move_type == "SIDEWAYS_CONSOLIDATION":
        return "The sector traded flat with low volume; no catalyst is needed — treat as sideways consolidation."
    if move_type == "TECHNICAL_ONLY":
        return "No high-confidence news catalyst was confirmed; the move is better treated as technical or positioning-driven."
    if not top_evidence:
        return "No ranked evidence survived the relevance filter, so causation is not established."

    top_score = top_evidence[0].get("reason_score", 0.0)
    horizon = describe_time_horizon(top_evidence[0].get("time_horizon"))
    if top_score >= 0.78:
        return f"The top article is a plausible {horizon} driver, but still not proof of causation."
    return f"The evidence suggests a likely {horizon} driver, but it should be framed as correlation rather than proof."


def has_distribution_evidence(technical, daily_change_pct):
    index_analysis = technical.get("index_analysis") or {}
    volume_signal = index_analysis.get("volume", {}).get("signal", "NORMAL")
    weekly_change = index_analysis.get("weekly_change", 0.0)
    monthly_change = index_analysis.get("monthly_change", 0.0)
    bearish_breadth = technical.get("stock_consensus") in {"ALL_BEARISH", "MOSTLY_BEARISH"}

    return (
        daily_change_pct < 0 and
        weekly_change < 0 and
        monthly_change < 0 and
        volume_signal in {"HIGH", "EXTREME", "ABOVE_AVERAGE"} and
        bearish_breadth
    )


def build_pressure_language(technical):
    index_analysis = technical.get("index_analysis") or {}
    volume_ratio = index_analysis.get("volume", {}).get("ratio", 1.0)
    daily_change = index_analysis.get("daily_change", {}).get("change_pct", 0.0)
    if daily_change < 0 and volume_ratio >= 1.2:
        return f"elevated selling pressure ({volume_ratio:.1f}x volume)"
    if daily_change > 0 and volume_ratio >= 1.2:
        return f"elevated buying pressure ({volume_ratio:.1f}x volume)"
    return f"normal volume context ({volume_ratio:.1f}x volume)"


def classify_move_type(technical, sentiment, daily_change_pct):
    relevant_count = sentiment.get("relevant_count", 0)
    technical_direction = technical.get("direction", "NEUTRAL")
    stock_consensus = technical.get("stock_consensus", "NO_DATA")
    index_analysis = technical.get("index_analysis") or {}
    volume_info = index_analysis.get("volume", {}) or {}
    volume_signal = volume_info.get("signal", "NORMAL")
    volume_ratio = volume_info.get("ratio", 1.0 if volume_signal != "NO_DATA" else 0.0)

    # If technical status is error / insufficient data, do NOT treat this as a
    # valid neutral technical move. Fall back to price + sentiment only.
    if technical.get("status") == "error":
        if relevant_count >= 3 and abs(daily_change_pct) >= 0.75:
            return "NEWS_DRIVEN"
        return "SECTOR_DRIVEN"

    # SIDEWAYS GATE: flat move + low volume = no real move to explain
    if abs(daily_change_pct) < 0.25 and volume_ratio < 0.8:
        return "SIDEWAYS_CONSOLIDATION"

    if relevant_count == 0 and (
        technical_direction != "NEUTRAL" or abs(daily_change_pct) >= 0.75 or volume_signal in {"HIGH", "EXTREME", "ABOVE_AVERAGE"}
    ):
        return "TECHNICAL_ONLY"

    if relevant_count >= 3 and abs(daily_change_pct) >= 0.75:
        return "NEWS_DRIVEN"

    if stock_consensus in {"ALL_BULLISH", "MOSTLY_BULLISH", "ALL_BEARISH", "MOSTLY_BEARISH"} and relevant_count >= 1:
        return "MIXED"

    return "SECTOR_DRIVEN"


def build_technical_confirmation(technical):
    return build_sector_technical_snapshot(technical)["summary"]


def build_key_risk(technical, sentiment, daily_change_pct):
    sector_sentiment = sentiment.get("sector_sentiment", "neutral")
    stock_consensus = technical.get("stock_consensus", "NO_DATA")
    dominant_horizon = sentiment.get("dominant_time_horizon", "near_term")

    if sector_sentiment == "positive" and daily_change_pct < 0:
        if has_distribution_evidence(technical, daily_change_pct):
            return "Positive news is being ignored while multi-day price-volume weakness persists; distribution risk is building, not confirmed."
        return f"Positive {describe_time_horizon(dominant_horizon)} news is being outweighed by shorter-term risk or positioning."
    if sector_sentiment == "negative" and daily_change_pct > 0:
        return f"Negative {describe_time_horizon(dominant_horizon)} news is being absorbed by price; this suggests resilience, not proven institutional accumulation."
    if technical.get("has_divergence"):
        return "Subsector divergence is high; the sector move is not uniform."
    if stock_consensus == "SPLIT":
        return "Breadth is split across major stocks; follow-through risk is higher."
    if sentiment.get("relevant_count", 0) == 0:
        return "No strong sector-specific news was retrieved; treat this as a flow/technical move."
    return "No major contradiction detected in the current evidence stack."


def build_peer_snapshot(technical):
    peers = []
    for stock in technical.get("stock_analyses", []):
        peers.append(
            f"{stock.get('stock_name', stock.get('ticker', ''))}: {stock.get('daily_change', {}).get('change_pct', 0.0):+.2f}%"
        )
    return ", ".join(peers[:4]) if peers else "No peer snapshot available."


def build_sector_driver_dossier(sector, technical, sentiment, daily_change_pct):
    ranked_evidence = []
    for article in sentiment.get("relevant_headlines", []):
        time_horizon = infer_time_horizon(article)
        reason_score, score_breakdown = score_ranked_reason(
            article, daily_change_pct, technical.get("stock_consensus", "NO_DATA")
        )
        ranked_evidence.append({
            "headline": article.get("headline", ""),
            "summary": article.get("summary", ""),
            "source": article.get("source", ""),
            "published_at": article.get("published_at", ""),
            "bucket": article.get("bucket", "sectoral"),
            "label": article.get("label", "neutral"),
            "reason_category": classify_reason_category(article),
            "time_horizon": time_horizon,
            "reason_score": reason_score,
            "score_breakdown": score_breakdown,
            "relevance_confidence": article.get("relevance", {}).get("confidence", 0.0),
        })

    ranked_evidence.sort(key=lambda article: article["reason_score"], reverse=True)
    move_type = classify_move_type(technical, sentiment, daily_change_pct)
    contradiction = (
        (sentiment.get("sector_sentiment") == "positive" and daily_change_pct < 0) or
        (sentiment.get("sector_sentiment") == "negative" and daily_change_pct > 0) or
        technical.get("has_divergence", False)
    )
    evidence_state = "SUFFICIENT_EVIDENCE"
    if not ranked_evidence or ranked_evidence[0]["reason_score"] < 0.58:
        evidence_state = "INSUFFICIENT_EVIDENCE"

    if ranked_evidence:
        primary_driver = (
            f"Likely {describe_time_horizon(ranked_evidence[0]['time_horizon'])} "
            f"{ranked_evidence[0]['reason_category']}: {ranked_evidence[0]['headline']}"
        )
        secondary_driver = (
            f"Likely {describe_time_horizon(ranked_evidence[1]['time_horizon'])} "
            f"{ranked_evidence[1]['reason_category']}: {ranked_evidence[1]['headline']}"
            if len(ranked_evidence) > 1 else "Technical and flow confirmation dominate after the top driver."
        )
    else:
        if move_type == "TECHNICAL_ONLY":
            primary_driver = "Technical-only move: price, breadth, and volume explain the action better than news."
            secondary_driver = "Likely sector rotation / short-covering / mean reversion rather than a discrete headline."
        else:
            primary_driver = "Sector flow appears broader than any single article."
            secondary_driver = "Use technical confirmation and breadth to judge persistence."

    technical_snapshot = build_sector_technical_snapshot(technical)
    stock_contribution_lines = build_sector_stock_contribution_lines(technical)

    return {
        "sector": sector,
        "move_type": move_type,
        "evidence_state": evidence_state,
        "primary_driver": primary_driver,
        "secondary_driver": secondary_driver,
        "technical_confirmation": technical_snapshot["summary"],
        "technical_indicator_lines": technical_snapshot["lines"],
        "stock_contribution_lines": stock_contribution_lines,
        "key_risk": build_key_risk(technical, sentiment, daily_change_pct),
        "pressure_context": build_pressure_language(technical),
        "causality_note": build_causality_note(ranked_evidence, move_type, evidence_state),
        "ranked_evidence": ranked_evidence[:5],
        "peer_snapshot": build_peer_snapshot(technical),
        "taxonomy": derive_event_taxonomy(ranked_evidence, move_type, technical=technical, contradiction=contradiction),
    }

def get_nifty_50_change():
    try:
        from datetime import datetime, timedelta
        range_to = datetime.now().strftime("%Y-%m-%d")
        range_from = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        data = {
            "symbol": "NSE:NIFTY 50-INDEX",
            "resolution": "1D",
            "date_format": "1",
            "range_from": range_from,
            "range_to": range_to,
            "cont_flag": "1"
        }
        response = get_fyers_client().history(data=data)
        if response and response.get("s") == "ok" and "candles" in response and len(response["candles"]) >= 2:
            candles = response["candles"]
            last_close = candles[-1][4]
            prev_close = candles[-2][4]
            return round(((last_close - prev_close) / prev_close) * 100, 2)
    except Exception as exc:
        log_fetch_warning("Nifty 50 change", exc)
    return 0.0


def render_fyers_debug_message():
    status = get_fyers_debug_status()
    return (
        f"Fyers source: `{status['source']}` | "
        f"client id: `{status['client_id_masked']}` | "
        f"access token: `{status['access_token_masked']}`"
    )


def build_stock_search_queries(ticker, stock_name, sector):
    sector_query = SECTOR_SEARCH_QUERIES.get(sector, [""])[0]
    base_name = stock_name.replace(" ", " OR ")
    queries = [
        f"\"{stock_name}\" OR {ticker.replace('.NS', '')} OR {base_name}",
    ]
    if sector_query:
        queries.append(f"\"{stock_name}\" OR {ticker.replace('.NS', '')} OR ({sector_query})")
    return queries


@st.cache_data(ttl=900)
def get_stock_news(ticker, stock_name, sector, limit=10):
    articles, seen = [], set()
    queries = build_stock_search_queries(ticker, stock_name, sector)
    per_source_limit = max(4, min(6, limit))

    for query in queries:
        for article in fetch_marketaux_articles(query, limit=per_source_limit, countries="in"):
            append_article(articles, seen, article)
        for article in fetch_gnews_articles(query, limit=per_source_limit):
            append_article(articles, seen, article)

    return articles[:limit]


def classify_stock_move_type(stock_analysis, ranked_evidence):
    # SIDEWAYS GATE: if stock barely moved and volume is thin, it's consolidation
    daily_change = abs(stock_analysis.get("daily_change_pct", 0.0))
    vol_ratio = stock_analysis.get("volume", {}).get("ratio", 1.0)
    if daily_change < 0.25 and vol_ratio < 0.8:
        return "SIDEWAYS_CONSOLIDATION"

    if not ranked_evidence and (
        stock_analysis.get("breakout", {}).get("state") in {"BREAKOUT", "BREAKDOWN"} or
        stock_analysis.get("gap", {}).get("abs_gap_pct", 0.0) >= 1.0 or
        stock_analysis.get("volume", {}).get("signal") in {"HIGH", "EXTREME", "ABOVE_AVERAGE"}
    ):
        return "TECHNICAL_ONLY"
    if ranked_evidence and ranked_evidence[0]["reason_score"] >= 0.68:
        return "NEWS_DRIVEN"
    if abs(stock_analysis.get("relative_strength_vs_sector", 0.0)) >= 1.0:
        return "SECTOR_DRIVEN"
    return "MIXED"


def build_stock_technical_summary(stock_analysis):
    gap = stock_analysis.get("gap", {})
    breakout = stock_analysis.get("breakout", {})
    intraday = stock_analysis.get("intraday_context", {})
    return (
        f"Gap {gap.get('gap_pct', 0.0):+.2f}% ({gap.get('direction', 'flat')}), "
        f"{breakout.get('state', 'INSIDE_RANGE')} vs 20d range, "
        f"relative volume {stock_analysis.get('volume', {}).get('ratio', 1.0):.1f}x, "
        f"range expansion {intraday.get('range_expansion', 1.0):.2f}x"
    )


def build_stock_driver_dossier(
    sector,
    ticker,
    stock_name,
    stock_analysis,
    sector_dossier,
    sector_change,
    nifty_change,
    stock_articles,
    engine,
):
    ranked_evidence = []
    stock_sentiment = engine.analyze_batch(
        [f"{article['headline']}. {article.get('summary', '')}".strip() for article in stock_articles]
    )

    for article in stock_articles:
        enriched = dict(article)
        time_horizon = infer_time_horizon(article)
        enriched["label"] = engine.analyze_headline(
            f"{article.get('headline', '')}. {article.get('summary', '')}".strip()
        )["label"]
        enriched["relevance"] = {
            "confidence": 0.92 if stock_name.lower() in article.get("headline", "").lower() else 0.72,
            "matched_keywords": [stock_name.lower()],
            "macro_matches": [],
            "regulatory_matches": [],
        }
        reason_score, score_breakdown = score_ranked_reason(
            enriched, stock_analysis.get("daily_change", {}).get("change_pct", 0.0), "MOSTLY_BULLISH"
        )
        ranked_evidence.append({
            "headline": article.get("headline", ""),
            "summary": article.get("summary", ""),
            "source": article.get("source", ""),
            "published_at": article.get("published_at", ""),
            "bucket": article.get("bucket", "sectoral"),
            "label": enriched["label"],
            "reason_category": classify_reason_category(article),
            "time_horizon": time_horizon,
            "reason_score": reason_score,
            "score_breakdown": score_breakdown,
        })

    ranked_evidence.sort(key=lambda article: article["reason_score"], reverse=True)
    daily_change = stock_analysis.get("daily_change", {}).get("change_pct", 0.0)
    relative_strength_vs_sector = round(daily_change - sector_change, 2)
    relative_strength_vs_nifty = round(daily_change - nifty_change, 2)
    stock_analysis["relative_strength_vs_sector"] = relative_strength_vs_sector
    stock_analysis["relative_strength_vs_nifty"] = relative_strength_vs_nifty

    move_type = classify_stock_move_type(stock_analysis, ranked_evidence)
    contradiction = (
        (stock_sentiment.get("aggregate_label") == "positive" and daily_change < 0) or
        (stock_sentiment.get("aggregate_label") == "negative" and daily_change > 0) or
        (relative_strength_vs_sector < 0 and sector_change > 0.5) or
        (relative_strength_vs_sector > 0 and sector_change < -0.5)
    )
    evidence_state = "SUFFICIENT_EVIDENCE"
    if not ranked_evidence or ranked_evidence[0]["reason_score"] < 0.58:
        evidence_state = "INSUFFICIENT_EVIDENCE"

    if ranked_evidence:
        primary_driver = (
            f"Likely {describe_time_horizon(ranked_evidence[0]['time_horizon'])} "
            f"{ranked_evidence[0]['reason_category']}: {ranked_evidence[0]['headline']}"
        )
        secondary_driver = (
            f"Likely {describe_time_horizon(ranked_evidence[1]['time_horizon'])} "
            f"{ranked_evidence[1]['reason_category']}: {ranked_evidence[1]['headline']}"
            if len(ranked_evidence) > 1 else "Sector and technical context are the next most important drivers."
        )
    elif move_type == "TECHNICAL_ONLY":
        primary_driver = "Technical-only move: gap, breakout state, and stock-level volume explain the move better than news."
        secondary_driver = "Sector context is secondary; no strong stock-specific article dominated."
    else:
        primary_driver = "Stock move is being carried more by sector/macro flow than a discrete stock-specific headline."
        secondary_driver = sector_dossier.get("primary_driver", "Sector driver dominates.")

    peer_snapshot = []
    for peer in SECTOR_TICKERS.get(sector, []):
        if peer == ticker:
            continue
        peer_name = STOCK_PICK_REASONS.get(peer, {}).get("name", peer.replace(".NS", ""))
        peer_snapshot.append(peer_name)

    key_risk = build_key_risk({"has_divergence": contradiction, "stock_consensus": "SPLIT"}, {
        "sector_sentiment": stock_sentiment.get("aggregate_label", "neutral"),
        "relevant_count": len(ranked_evidence),
    }, daily_change)

    return {
        "ticker": ticker,
        "stock_name": stock_name,
        "sector": sector,
        "move_type": move_type,
        "evidence_state": evidence_state,
        "primary_driver": primary_driver,
        "secondary_driver": secondary_driver,
        "technical_confirmation": build_stock_technical_summary(stock_analysis),
        "key_risk": key_risk,
        "pressure_context": (
            f"price-volume context suggests {build_pressure_language({'index_analysis': stock_analysis})}"
        ),
        "causality_note": build_causality_note(ranked_evidence, move_type, evidence_state),
        "relative_strength_vs_sector": relative_strength_vs_sector,
        "relative_strength_vs_nifty": relative_strength_vs_nifty,
        "daily_change": daily_change,
        "sector_change": sector_change,
        "nifty_change": nifty_change,
        "peer_snapshot": ", ".join(peer_snapshot[:4]) if peer_snapshot else "No peers available.",
        "ranked_evidence": ranked_evidence[:5],
        "stock_sentiment": stock_sentiment,
        "taxonomy": derive_event_taxonomy(ranked_evidence, move_type, contradiction=contradiction),
    }


def build_evaluation_cases(run_timestamp, sector_dossiers, stock_dossier=None):
    cases = []
    for sector, dossier in sector_dossiers.items():
        cases.append({
            "case_id": f"{run_timestamp}_{sector.lower().replace(' ', '_')}",
            "run_timestamp": run_timestamp,
            "instrument_type": "sector",
            "symbol": sector,
            "move_type": dossier.get("move_type"),
            "evidence_state": dossier.get("evidence_state"),
            "primary_driver_generated": dossier.get("primary_driver"),
            "secondary_driver_generated": dossier.get("secondary_driver"),
            "technical_confirmation_generated": dossier.get("technical_confirmation"),
            "taxonomy": dossier.get("taxonomy", []),
            "top_evidence": [item.get("headline") for item in dossier.get("ranked_evidence", [])[:3]],
            "labels": {
                "primary_driver_correct": None,
                "article_relevance_correct": None,
                "technical_explanation_correct": None,
                "hallucination": None,
                "forced_narrative": None,
                "missed_main_driver": None,
            },
            "notes": "",
        })

    if stock_dossier:
        cases.append({
            "case_id": f"{run_timestamp}_{stock_dossier['ticker'].replace('.NS', '').lower()}",
            "run_timestamp": run_timestamp,
            "instrument_type": "stock",
            "symbol": stock_dossier["ticker"],
            "sector": stock_dossier["sector"],
            "move_type": stock_dossier.get("move_type"),
            "evidence_state": stock_dossier.get("evidence_state"),
            "primary_driver_generated": stock_dossier.get("primary_driver"),
            "secondary_driver_generated": stock_dossier.get("secondary_driver"),
            "technical_confirmation_generated": stock_dossier.get("technical_confirmation"),
            "taxonomy": stock_dossier.get("taxonomy", []),
            "top_evidence": [item.get("headline") for item in stock_dossier.get("ranked_evidence", [])[:3]],
            "labels": {
                "primary_driver_correct": None,
                "article_relevance_correct": None,
                "technical_explanation_correct": None,
                "hallucination": None,
                "forced_narrative": None,
                "missed_main_driver": None,
            },
            "notes": "",
        })

    return cases


def append_evaluation_cases(cases, output_path=None):
    if output_path is None:
        output_path = Path(__file__).with_name("evaluation_cases.jsonl")

    with output_path.open("a", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=True) + "\n")

    return output_path


@st.cache_data(ttl=900)
def analyze_stock_with_llm(stock_dossier, deep_dive=False):
    system_prompt = """You explain why a single Indian stock moved today.

Rules:
1. Use the supplied stock-specific evidence first, sector context second, macro third.
2. Respect the move type exactly. If it is TECHNICAL_ONLY, do not invent a news catalyst.
3. Use the taxonomy tags as hints for what type of driver matters most.
4. Mention relative strength vs sector and vs Nifty explicitly.
5. Mention stock-level technical context: gap, breakout/breakdown, relative volume, range expansion.
6. If evidence state is INSUFFICIENT_EVIDENCE, say the driver confidence is weak and the move may be flow-driven.
7. If price contradicts the news, call out contradiction/divergence directly.
8. Do not say a headline caused the move unless the data proves it; use "likely driver" or "plausible catalyst".
9. Treat elevated volume as buying/selling pressure, not confirmed institutional flow.
10. Use the article time horizons to explain why immediate risk can outweigh medium-term positives.

Output:
- 1 short heading line
- 4-6 sentences
- End with one key risk sentence
"""

    user_prompt = f"""
STOCK: {stock_dossier['stock_name']} ({stock_dossier['ticker']})
SECTOR: {stock_dossier['sector']}
MOVE TYPE: {stock_dossier['move_type']}
EVIDENCE STATE: {stock_dossier['evidence_state']}
TAXONOMY: {", ".join(stock_dossier.get('taxonomy', []))}
DAILY CHANGE: {stock_dossier['daily_change']:+.2f}%
RELATIVE STRENGTH VS SECTOR: {stock_dossier['relative_strength_vs_sector']:+.2f}%
RELATIVE STRENGTH VS NIFTY: {stock_dossier['relative_strength_vs_nifty']:+.2f}%
PRIMARY DRIVER: {stock_dossier['primary_driver']}
SECONDARY DRIVER: {stock_dossier['secondary_driver']}
TECHNICAL CONFIRMATION: {stock_dossier['technical_confirmation']}
PEER SNAPSHOT: {stock_dossier['peer_snapshot']}
KEY RISK: {stock_dossier['key_risk']}
PRESSURE CONTEXT: {stock_dossier['pressure_context']}
CAUSALITY NOTE: {stock_dossier['causality_note']}
RANKED EVIDENCE:
{chr(10).join(f"- {item['headline']} | {item['reason_category']} | {describe_time_horizon(item.get('time_horizon'))} | score {item['reason_score']:.2f}" for item in stock_dossier.get('ranked_evidence', [])[:4])}
"""

    if deep_dive:
        try:
            response = gemini_client.models.generate_content(
                model="models/gemini-2.5-flash", contents=f"{system_prompt}\n\n{user_prompt}"
            )
            return response.text
        except Exception as exc:
            log_fetch_warning("Gemini stock analysis", exc)
            return f"⚠️ Stock analysis failed: {str(exc)}"

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=350,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content
    except Exception as exc:
        log_fetch_warning("Groq stock analysis", exc)
        return f"⚠️ Stock analysis failed: {str(exc)}"
# ══════════════════════════════════════════════════════════════
# DATA FETCHING FUNCTIONS
# ══════════════════════════════════════════════════════════════

@st.cache_data(ttl=900)
def get_commodity_prices():
    def safe_price(ticker):
        try:
            hist = fetch_historical_data(ticker, period="5d")
            return round(hist['Close'].iloc[-1], 2) if len(hist) >= 1 else "N/A"
        except Exception as exc:
            log_fetch_warning(f"Price fetch [{ticker}]", exc)
            return "N/A"

    def safe_yf_price(ticker):
        """Fallback to yfinance for tickers not available on Fyers."""
        try:
            import yfinance as yf
            hist = yf.Ticker(ticker).history(period="5d")
            return round(hist['Close'].iloc[-1], 2) if len(hist) >= 1 else "N/A"
        except Exception as exc:
            log_fetch_warning(f"YF price fetch [{ticker}]", exc)
            return "N/A"

    return {
        # Fyers-available tickers
        "oil": safe_price("CL=F"), "gold": safe_price("GC=F"),
        "usd_inr": safe_price("INR=X"), "copper": safe_price("HG=F"),
        "aluminium": safe_price("ALI=F"),
        # yfinance fallback — no Fyers equivalent
        "steel": safe_yf_price("HRC=F"),
        "bond_10y": safe_yf_price("^TNX"), "india_gsec": safe_yf_price("^INBMK10Y"),
        "iron_ore": safe_yf_price("TIO=F"), "china_etf": safe_yf_price("FXI"),
        "pmi_proxy": safe_yf_price("XLI"),
    }

def is_indian_market_hours():
    """Check if current IST time is within Indian market hours (8 AM - 3:30 PM, weekdays)."""
    import pytz
    ist = pytz.timezone('Asia/Kolkata')
    now_ist = datetime.now(timezone.utc).astimezone(ist)
    if now_ist.weekday() >= 5:  # Sat/Sun
        return False
    market_open = now_ist.replace(hour=8, minute=0, second=0, microsecond=0)
    market_close = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now_ist <= market_close


SECTOR_YF_TICKERS = {
    "IT": "^CNXIT", "Banking": "^NSEBANK", "FMCG": "^CNXFMCG",
    "Oil & Gas": "^CNXENERGY", "Pharma": "^CNXPHARMA",
    "Metals": "^CNXMETAL", "Infrastructure": "^CNXINFRA", "Gold": "GOLDBEES.NS"
}


def _get_sector_changes_from_fyers():
    """Fetch intraday % change for all sector indices from Fyers live quotes."""
    tickers = list(SECTOR_YF_TICKERS.values())
    live_q = get_live_quotes(tickers)  # returns {ticker: {lp, ch, chp}}
    changes = {}
    for name, ticker in SECTOR_YF_TICKERS.items():
        chp = live_q.get(ticker, {}).get("chp", 0)
        if chp:  # non-zero = Fyers returned real data
            changes[name] = round(chp, 2)
    return changes  # empty dict if Fyers failed


def get_nifty_sector_data_live():
    """Sector performance strings using Fyers live quotes (market hours only)."""
    changes = _get_sector_changes_from_fyers()
    performance = {}
    for name in SECTOR_YF_TICKERS:
        chp = changes.get(name)
        if chp is not None:
            arrow = "📈" if chp > 0 else "📉"
            performance[name] = f"{arrow} {'+' if chp > 0 else ''}{chp}%"
        else:
            performance[name] = "N/A"
    return performance


@st.cache_data(ttl=900)
def get_nifty_sector_data_afterhours():
    """Sector performance strings using yfinance last-close (after hours)."""
    import yfinance as yf
    performance = {}
    for name, ticker in SECTOR_YF_TICKERS.items():
        try:
            hist = yf.Ticker(ticker).history(period="2d")
            if len(hist) >= 2:
                change = round(((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100, 2)
                arrow = "📈" if change > 0 else "📉"
                performance[name] = f"{arrow} {'+' if change > 0 else ''}{change}%"
            else:
                performance[name] = "N/A"
        except Exception as exc:
            try:
                hist = fetch_historical_data(ticker, period="5d")
                if len(hist) >= 2:
                    change = round(((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100, 2)
                    arrow = "📈" if change > 0 else "📉"
                    performance[name] = f"{arrow} {'+' if change > 0 else ''}{change}%"
                else:
                    performance[name] = "N/A"
            except Exception:
                log_fetch_warning(f"Nifty sector data [{name}]", exc)
                performance[name] = "N/A"
    return performance


def get_nifty_sector_data():
    """Always try Fyers first (works even after market close for that day's data).
    Only fall back to yfinance if Fyers returns insufficient data."""
    live = get_nifty_sector_data_live()
    # If Fyers returned real data for at least half the sectors, use it
    valid = sum(1 for v in live.values() if v != "N/A")
    if valid >= 4:
        return live
    return get_nifty_sector_data_afterhours()


def get_nifty_sector_changes_live():
    """Sector % changes as floats using Fyers live quotes (market hours only)."""
    return _get_sector_changes_from_fyers()


@st.cache_data(ttl=900)
def get_nifty_sector_changes_afterhours():
    """Sector % changes as floats using yfinance last-close (after hours)."""
    import yfinance as yf
    changes = {}
    for name, ticker in SECTOR_YF_TICKERS.items():
        try:
            hist = yf.Ticker(ticker).history(period="2d")
            if len(hist) >= 2:
                changes[name] = round(((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100, 2)
            else:
                changes[name] = 0.0
        except Exception as exc:
            try:
                hist = fetch_historical_data(ticker, period="5d")
                if len(hist) >= 2:
                    changes[name] = round(((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100, 2)
                else:
                    changes[name] = 0.0
            except Exception:
                log_fetch_warning(f"Nifty sector change [{name}]", exc)
                changes[name] = 0.0
    return changes


def get_nifty_sector_changes():
    """Always try Fyers first (works even after market close for that day's data).
    Only fall back to yfinance if Fyers returns insufficient data."""
    live = get_nifty_sector_changes_live()
    if len(live) >= 4:  # Fyers returned data for enough sectors
        return live
    return get_nifty_sector_changes_afterhours()

@st.cache_data(ttl=900)
def get_all_news():
    global_articles, indian_articles, seen = [], [], set()
    for cat in ["general", "forex", "merger"]:
        try:
            response = requests.get(
                "https://finnhub.io/api/v1/news",
                params={"category": cat, "token": FINNHUB_API_KEY},
                timeout=5,
            )
            response.raise_for_status()
            for article in response.json():
                append_article(
                    global_articles,
                    seen,
                    normalize_article(
                        headline=article.get("headline", ""),
                        summary=article.get("summary", ""),
                        url=article.get("url", ""),
                        image=article.get("image", ""),
                        source="Finnhub",
                        published_at=str(article.get("datetime", "")),
                        bucket=classify_news_bucket(
                            article.get("headline", ""), article.get("summary", "")
                        ),
                    ),
                )
        except Exception as exc:
            log_fetch_warning(f"Finnhub category [{cat}]", exc)
    try:
        response = requests.get(
            "https://newsdata.io/api/1/news",
            params={
                "apikey": NEWSDATA_API_KEY,
                "country": "in",
                "category": "business",
                "language": "en",
            },
            timeout=5,
        )
        response.raise_for_status()
        for article in response.json().get("results", []):
            append_article(
                indian_articles,
                seen,
                normalize_article(
                    headline=article.get("title", ""),
                    summary=article.get("description", ""),
                    url=article.get("link", ""),
                    image=article.get("image_url", ""),
                    source="NewsData",
                    published_at=article.get("pubDate", ""),
                    bucket=classify_news_bucket(
                        article.get("title", ""), article.get("description", "")
                    ),
                ),
            )
    except Exception as exc:
        log_fetch_warning("NewsData India business feed", exc)

    for article in fetch_marketaux_articles(
        "india business markets sector rotation rbi policy", limit=10, countries="in"
    ):
        append_article(global_articles, seen, article)

    for article in fetch_gnews_articles(
        "oil OPEC war sanctions crude geopolitical India markets", limit=10
    ):
        append_article(global_articles, seen, article)

    return global_articles[:15], indian_articles[:15]

SECTOR_SEARCH_QUERIES = {
    "Banking": [
        "RBI OR bank credit growth OR deposit growth OR NIM OR slippages OR HDFC Bank OR ICICI Bank OR SBI OR Kotak OR NBFC OR bond yields"
    ],
    "IT": [
        "TCS OR Infosys OR Wipro OR HCLTech OR AI spending OR outsourcing demand OR cloud deals OR Nasdaq OR US tech spending"
    ],
    "Oil & Gas": [
        "ONGC OR Reliance OR BPCL OR HPCL OR IOC OR crude oil OR OPEC OR refining margins OR sanctions"
    ],
    "Metals": [
        "Tata Steel OR JSW Steel OR Hindalco OR aluminium OR copper OR iron ore OR China demand OR LME"
    ],
    "Pharma": [
        "Sun Pharma OR Dr Reddy OR Cipla OR USFDA OR generic drugs OR drug approval OR API"
    ],
    "FMCG": [
        "HUL OR ITC OR Nestle India OR Britannia OR rural demand OR consumer spending OR input cost inflation"
    ],
    "Infrastructure": [
        "L&T OR IRB OR capex OR highway projects OR railway capex OR infrastructure spending OR order inflow"
    ],
    "Gold": [
        "gold prices OR bullion OR Titan OR Kalyan Jewellers OR jewellery demand OR ETF flows"
    ],
}

@st.cache_data(ttl=900)
def get_sector_news(sector_name, limit=10):
    queries = SECTOR_SEARCH_QUERIES.get(sector_name, [])
    articles, seen = [], set()
    if not queries:
        return []

    per_source_limit = max(4, min(6, limit))
    for query in queries:
        for article in fetch_marketaux_articles(query, limit=per_source_limit, countries="in"):
            append_article(articles, seen, article)
        for article in fetch_gnews_articles(query, limit=per_source_limit):
            append_article(articles, seen, article)

    articles.sort(key=lambda article: article.get("published_at", ""), reverse=True)
    return articles[:limit]

# ══════════════════════════════════════════════════════════════
# SLIMMED LLM ANALYSIS (Prompt reduced ~70%)
# ══════════════════════════════════════════════════════════════

@st.cache_data(ttl=900)
def analyze_with_llm(prices, nifty_data, selected_sectors, hybrid_results,
                     global_sentiment, sector_changes, deep_dive=False, sector_news=None,
                     sector_dossiers=None):
    """
    The LLM now EXPLAINS pre-computed hybrid signals. It no longer decides them.
    Prompt reduced from ~750 lines to ~100 lines.
    """
    sectors_text = ", ".join(selected_sectors)

    # Build hybrid signal context for the LLM
    signal_context = ""
    for sector, hs in hybrid_results.items():
        signal_context += f"\n--- {sector} ---\n"
        signal_context += f"Signal: {hs.signal} (confidence: {hs.confidence:.0%})\n"
        m_score = getattr(hs, "momentum_score", 0.5)
        v_score = getattr(hs, "volume_score", 0.5)
        breadth = getattr(hs, "breadth", "UNKNOWN")

        signal_context += f"Momentum: score {m_score:.2f} (daily change {sector_changes.get(sector, 0.0):+.2f}%) — WEIGHT 40% (25% Daily, 35% Weekly, 25% Monthly, 15% MA Bias)\n"
        signal_context += f"Technical: {hs.technical_direction} ({hs.technical_confidence:.0%}) — WEIGHT 25%\n"
        signal_context += f"Volume: score {v_score:.2f} — WEIGHT 20%\n"
        signal_context += f"Sentiment: {hs.sentiment_label} ({hs.sentiment_score:.0%}) — WEIGHT 15%\n"
        signal_context += f"Agreement: {hs.agreement}\n"
        signal_context += f"Breadth: {breadth}\n"
        for r in hs.reasoning:
            signal_context += f"  {r}\n"
        if hs.crash_warning:
            signal_context += "  \U0001f6a8 CRASH WARNING ACTIVE\n"
        if hs.rally_alert:
            signal_context += "  \U0001f680 RALLY ALERT ACTIVE\n"

        # ── INDEX-LEVEL TECHNICAL DETAILS (RSI, MA, Volume, Breakout) ──
        tech_details = hs.technical_details
        idx = tech_details.get("index_analysis") or {}
        if idx and idx.get("status") == "ok":
            idx_rsi = idx.get("rsi", "N/A")
            idx_ma = idx.get("ma", {})
            idx_vol = idx.get("volume", {})
            idx_daily = idx.get("daily_change", {})
            idx_breakout = idx.get("breakout", {})
            idx_weekly = idx.get("weekly_change", 0.0)
            idx_monthly = idx.get("monthly_change", 0.0)
            signal_context += f"  INDEX TECHNICALS:\n"
            signal_context += f"    RSI: {idx_rsi:.1f}\n" if isinstance(idx_rsi, (int, float)) else ""
            signal_context += f"    MA20: {idx_ma.get('ma20', 'N/A')}, MA50: {idx_ma.get('ma50', 'N/A')}, Crossover: {idx_ma.get('crossover', 'N/A')}, MA Signal: {idx_ma.get('signal', 'N/A')}\n"
            signal_context += f"    Volume Ratio: {idx_vol.get('ratio', 'N/A')}x ({idx_vol.get('signal', 'N/A')})\n"
            signal_context += f"    Daily Change: {idx_daily.get('change_pct', 0):+.2f}% ({idx_daily.get('volatility_class', 'normal')})\n"
            signal_context += f"    Weekly Change: {idx_weekly:+.2f}%, Monthly Change: {idx_monthly:+.2f}%\n"
            signal_context += f"    Breakout State: {idx_breakout.get('state', 'N/A')} (distance: {idx_breakout.get('distance_pct', 0):+.2f}%)\n"

        # ── ALL STOCK ANALYSES (so LLM can discuss individual stocks) ──
        stock_analyses = tech_details.get("stock_analyses", [])
        if stock_analyses:
            signal_context += "  STOCK-LEVEL ANALYSIS:\n"
            for sa in stock_analyses:
                sa_name = sa.get("stock_name", sa.get("ticker", "Unknown"))
                sa_daily = sa.get("daily_change", {}).get("change_pct", 0)
                sa_rsi = sa.get("rsi", "N/A")
                sa_vol = sa.get("volume", {})
                sa_score = sa.get("score", 0)
                sa_dir = sa.get("direction", "NEUTRAL")
                sa_breakout = sa.get("breakout", {})
                signal_context += (
                    f"    {sa_name}: {sa_dir} (score: {sa_score:.3f}, daily: {sa_daily:+.2f}%, "
                    f"RSI: {sa_rsi:.0f}, volume: {sa_vol.get('ratio', 'N/A')}x {sa_vol.get('signal', '')}, "
                    f"breakout: {sa_breakout.get('state', 'N/A')})\n"
                )

        # Subsector divergence data
        if tech_details.get("has_divergence"):
            signal_context += "  \u26a0\ufe0f SUBSECTOR DIVERGENCE DETECTED:\n"
            for sub_name, sub_data in tech_details.get("subsectors", {}).items():
                signal_context += f"    {sub_data['label']}: {sub_data['direction']} (score {sub_data['score']})\n"
                extra = sub_data.get('extra', {})
                if 'crude_impact' in extra:
                    signal_context += f"      Crude impact: {extra['crude_impact']} — {extra.get('logic', '')}\n"
                if 'driver' in extra:
                    signal_context += f"      Key driver: {extra['driver']}\n"

        # Macro relevance weights
        macro_w = tech_details.get("macro_weights", {})
        if macro_w:
            irrelevant = [k for k, v in macro_w.items() if v < 0.2]
            critical = [k for k, v in macro_w.items() if v >= 0.7]
            if irrelevant:
                signal_context += f"  IRRELEVANT macro factors (DO NOT use): {', '.join(irrelevant)}\n"
            if critical:
                signal_context += f"  CRITICAL macro factors (MUST use): {', '.join(critical)}\n"

        # Stock picks with reasons + momentum data
        top_picks = tech_details.get("top_picks", [])
        avoid_picks = tech_details.get("avoid_picks", [])
        if top_picks:
            signal_context += "  TOP PICKS (momentum-verified):\n"
            for p in top_picks:
                signal_context += f"    \U0001f7e2 {p['name']}: {p['reason']} (tech score: {p.get('score', 0):.3f}, daily: {p.get('daily_change', 0):+.2f}%, RSI: {p.get('rsi', 50):.0f})\n"
        if avoid_picks:
            signal_context += "  STOCKS TO AVOID (momentum-verified bearish):\n"
            for p in avoid_picks:
                signal_context += f"    \U0001f534 {p['name']}: {p['reason']} (tech score: {p.get('score', 0):.3f}, daily: {p.get('daily_change', 0):+.2f}%, RSI: {p.get('rsi', 50):.0f})\n"
        if not avoid_picks:
            signal_context += "  NO STOCKS TO AVOID: The system currently does not explicitly flag specific stocks for avoidance (could be due to broad-based movement or lack of individual volume anomalies).\n"

        dossier = (sector_dossiers or {}).get(sector, {})
        if dossier:
            signal_context += f"  Movement Classification: {dossier.get('move_type', 'MIXED')}\n"
            signal_context += f"  Evidence State: {dossier.get('evidence_state', 'SUFFICIENT_EVIDENCE')}\n"
            signal_context += f"  Event Taxonomy: {', '.join(dossier.get('taxonomy', []))}\n"
            signal_context += f"  Primary Driver: {dossier.get('primary_driver', '')}\n"
            signal_context += f"  Secondary Driver: {dossier.get('secondary_driver', '')}\n"
            signal_context += f"  Technical Confirmation: {dossier.get('technical_confirmation', '')}\n"
            technical_lines = dossier.get("technical_indicator_lines", [])
            if technical_lines:
                signal_context += "  MANDATORY TECHNICAL INDICATORS:\n"
                for line in technical_lines:
                    signal_context += f"    - {line}\n"
            stock_contribution_lines = dossier.get("stock_contribution_lines", [])
            if stock_contribution_lines:
                signal_context += "  STOCK CONTRIBUTION:\n"
                for line in stock_contribution_lines[:2]:
                    signal_context += f"    - {line}\n"
            signal_context += f"  Volume Interpretation: {dossier.get('pressure_context', '')}\n"
            signal_context += f"  Causality Note: {dossier.get('causality_note', '')}\n"
            signal_context += f"  Key Risk / Contradiction: {dossier.get('key_risk', '')}\n"
            signal_context += f"  Peer Snapshot: {dossier.get('peer_snapshot', '')}\n"
            evidence_lines = dossier.get("ranked_evidence", [])
            if evidence_lines:
                signal_context += "  Ranked Evidence:\n"
                for evidence in evidence_lines[:3]:
                    signal_context += (
                        f"    - [{evidence.get('reason_category', 'sector sentiment')}] "
                        f"{describe_time_horizon(evidence.get('time_horizon'))}, "
                        f"score {evidence.get('reason_score', 0):.2f}: {evidence.get('headline', '')}\n"
                    )

    # Global sentiment summary
    sent_summary = (
        f"Overall News Sentiment: {global_sentiment.get('aggregate_label', 'neutral')} "
        f"(score: {global_sentiment.get('aggregate_score', 0.5):.0%})\n"
        f"Positive: {global_sentiment.get('positive_pct', 0):.0f}% | "
        f"Negative: {global_sentiment.get('negative_pct', 0):.0f}% | "
        f"Neutral: {global_sentiment.get('neutral_pct', 0):.0f}%"
    )

    nifty_text = "\n".join(f"- Nifty {k}: {v}" for k, v in nifty_data.items())

    try:
        nq_hist = fetch_historical_data("^IXIC", period="2d")
        nq_change = round(((nq_hist['Close'].iloc[-1] - nq_hist['Close'].iloc[-2]) / nq_hist['Close'].iloc[-2]) * 100, 2)
        nasdaq_text = f"{'+' if nq_change > 0 else ''}{nq_change}%"
    except Exception as exc:
        log_fetch_warning("Nasdaq context fetch", exc)
        nasdaq_text = "N/A"

    sector_news_text = ""
    if sector_news:
        bucket_labels = {
            "global_macro": "GLOBAL MACRO / GEOPOLITICAL",
            "regulatory": "REGULATORY / POLICY",
            "sectoral": "SECTORAL NEWS",
        }
        for sector, articles in sector_news.items():
            if not articles:
                continue

            sector_news_text += f"\n{sector.upper()} NEWS DRIVERS:\n"
            bucketed_articles = {"global_macro": [], "regulatory": [], "sectoral": []}
            for article in articles:
                bucketed_articles.setdefault(article.get("bucket", "sectoral"), []).append(article)

            for bucket in ("global_macro", "regulatory", "sectoral"):
                bucket_articles = bucketed_articles.get(bucket, [])
                if not bucket_articles:
                    continue
                sector_news_text += f"{bucket_labels[bucket]}:\n"
                for article in bucket_articles[:3]:
                    sector_news_text += (
                        f"- {article['headline']} | {article.get('summary', '')} "
                        f"| Source: {article.get('source', '')} "
                        f"| Time: {article.get('published_at', '')}\n"
                    )
            sector_news_text += "\n"

    if deep_dive:
        output_instructions = """
For each sector provide:
### [Sector] | Nifty: [exact %] | Signal: [use the pre-computed signal above]

📊 MACRO PICTURE: ONLY use macro factors marked CRITICAL above. DO NOT reference factors marked IRRELEVANT.

🧭 MOVE TYPE:
  - Use the supplied movement classification exactly.
  - If the move is TECHNICAL_ONLY or widely diverges from the news (MIXED), explicitly state that near-term positioning and institutional flows dominated the session.
  - Do NOT force a fake headline-driven explanation if the price moved against the news.
  - If it is NEWS_DRIVEN, lead with the ranked evidence.

📈 TECHNICAL INDICATORS:
  - This section is MANDATORY for every sector.
  - Explicitly show RSI with zone, MA20 vs MA50 with the trend signal, support/resistance, breakout/breakdown state, and volume ratio.
  - Use the exact technical lines supplied above. If a value is unavailable, say unavailable instead of skipping the section.

🌍 GLOBAL / REGULATORY / SECTOR NEWS DRIVERS:
  - Use the provided news buckets.
  - Prefer geopolitical/global macro themes if present.
  - If those are thin, use regulatory or sectoral themes instead.
  - Mention at least 2 concrete themes from the supplied articles.
  - Do NOT write "No relevant geopolitical headlines" unless every bucket is empty.
  - NEWS RECENCY RULE: Treat any news older than 5 days as historical context only. NEVER cite old news (like old COVID pill approvals from years ago) as the driver for TODAY'S price movement.

🏭 FUNDAMENTAL ANALYSIS: Company-level impact, margins, supply chains
  - If SUBSECTOR DIVERGENCE is flagged, you MUST explain which subsector benefits and which suffers, and WHY.
  - For Oil & Gas: Always separate upstream (ONGC, Oil India) vs downstream (BPCL, HPCL, IOC) impact.
  - If there is no divergence, explicitly state that the move suggests sector-wide positioning rather than company-specific catalysts.
  - DO NOT list Top Picks or Stocks to Avoid here. Reserve those for the SIGNAL EXPLANATION section ONLY.

🔄 RELATIVE STRENGTH: Compare this sector's daily % change against the other analyzed sectors to identify if capital is rotating IN or OUT compared to peers.

📰 NEWS SENTIMENT: Reference the FinBERT polarity — cite specific bullish/bearish news themes. Do NOT mention specific small headline counts.
  - If there is a bearish sentiment-price divergence (good news but falling prices), explicitly state this often signals institutional distribution or capital rotation out of the sector.
  - If there is a bullish divergence (bad news but rising prices), cite price absorption or institutional accumulation.

🎯 SIGNAL EXPLANATION: Explain WHY the hybrid system produced this signal
   - Explicitly state that Momentum (40% overall weight) is driven by a stable multi-timeframe blend (25% Daily, 35% Weekly, 25% Monthly, 15% MA Bias), and Technicals are 25% of the composite score.
   - Explicitly state that Sentiment (15%) acts only as a weak filter and DOES NOT override the price trend.
   - Top Picks: Use the TOP PICKS provided above — explain each stock's specific mechanism
   - Avoid: Use the STOCKS TO AVOID provided above — explain why each suffers
   - Key Risk: one thing to watch tomorrow

IMPORTANT FORMATTING RULE: You MUST leave a blank line between every major section (between Macro, News Drivers, Fundamental, Relative Strength, News, and Signal) so they do not blend into a single paragraph.
"""
    else:
        output_instructions = """
For each sector provide:
### [Sector] | Nifty: [exact %] | Signal: [use the pre-computed signal above]

- Reasoning: [3 points — technical + sentiment + ONLY relevant macro factors]
- Mandatory Technical Indicators: include RSI, MA20 vs MA50, support/resistance, breakout/breakdown state, and volume ratio for every sector.
- IMPORTANT: Only reference macro factors marked as CRITICAL. Do NOT use IRRELEVANT factors.
- Move Type: Use the supplied classification. If it says TECHNICAL_ONLY or widely diverges from the news, explicitly state that the move was driven by near-term institutional positioning and capital flows. Do NOT invent a news catalyst if price moved against the news.
- News Drivers: Use the supplied sector/global/regulatory news buckets and mention 1-2 concrete themes. Treat any news older than 5 days as historical context only, never as today's catalyst.
- If SUBSECTOR DIVERGENCE flagged: Explain which subsector benefits vs suffers. If no divergence, explicitly state the move suggests sector-wide positioning rather than company-specific catalysts.
- \U0001f504 RELATIVE STRENGTH: Compare this sector's daily % change against the other analyzed sectors to identify capital rotation.
- Top Picks: Use the provided picks with their specific reasons. Explain WHY each stock benefits/suffers today.
"""

    system_prompt = f"""You are a data-driven Indian stock market analyst. Your role is to EXPLAIN
pre-computed hybrid signals, NOT to decide them.

ARCHITECTURE (v2.8 — Momentum-First):
The signal is computed as: 40% Price Momentum + 25% Technical Indicators + 20% Volume Confirmation + 15% Sentiment.
Price momentum is the PRIMARY signal and uses a stable multi-timeframe calculation (25% Daily, 35% Weekly, 25% Monthly, 15% MA Bias). Sentiment is a SECONDARY filter.

RULES:
1. The signal (\U0001f7e2/\U0001f7e1/\U0001f534) has been computed by a hybrid ML system. Use it EXACTLY as provided.
2. Explain the reasoning using the technical, momentum, volume, and sentiment data provided.
2A. EVERY sector output MUST include a dedicated "TECHNICAL INDICATORS" section with RSI, MA20/MA50, support/resistance, breakout or breakdown state, and volume ratio.
3. Use ONLY the data provided. Never invent news, fundamentals, or data.
4. Use exact prices and percentages from the data.
5. Only reference currently listed NSE companies.
6. MACRO RELEVANCE: Each sector has CRITICAL and IRRELEVANT macro factors listed.
   - NEVER mention IRRELEVANT factors (e.g., do NOT mention Nasdaq for Oil & Gas).
   - ALWAYS reference CRITICAL factors with exact values.
7. SUBSECTOR DIVERGENCE (CRITICAL RULE): When explaining a sector, you MUST account for its subsectors if they diverge.
   - For Oil & Gas: This behaves as TWO SEPARATE SECTORS when crude spikes. NEVER say the "whole sector is bullish" just because the index is up.
     * Upstream (ONGC, Oil India) benefits from HIGH crude.
     * Downstream/Refining (BPCL, HPCL, IOC) suffers from HIGH crude (margin compression).
     * If the index is up but refiners are down, explicitly state that Upstream dragged the index higher while Downstream suffered.
   - For Metals: Ferrous (Tata Steel) driven by China demand, Non-Ferrous (Hindalco) by LME prices. 
     * CRITICAL: Gold is a SAFE HAVEN asset. DO NOT use gold price movements or gold-related news (like Iran conflicts) to explain industrial metals (steel/copper/aluminium) performance.
   - For Infrastructure: Heavyweights like L&T dominate the index. If the sector is down but others are up, explicitly check if L&T dragged the sector down.
8. STOCK PICKS: Use the pre-computed momentum-verified picks EXACTLY as provided.
   - NEVER override stock picks. If the system says a stock is a top pick, explain WHY using the provided reason.
   - NEVER add a stock to the avoid list unless the system explicitly flagged it as avoid.
   - If no stocks are flagged to avoid, state that the system does not explicitly flag any specific stocks for avoidance (do NOT claim they have positive momentum if the sector is down).
   - Explain WHY each stock benefits/suffers based on the SPECIFIC mechanism and data provided.
9. ANTI-HALLUCINATION RULES (CRITICAL):
   - You are evaluating INTRADAY MOMENTUM. Long-term fundamental analysis is strictly FORBIDDEN.
   - NEVER generate your own fundamental analysis. Use ONLY reasoning data from the hybrid signal.
   - DO NOT USE words like "capex", "margins", "subsidiary", "inflation", or "valuation" unless explicitly present in the provided reasoning data.
   - NEVER claim that sentiment or news "overrides" the technical trend. Mathematically, Momentum is 40% and Sentiment is only 15%. Sentiment is a mathematically weak filter.
   - If the data says a stock is bullish, DO NOT contradict it with your own analysis.
10. SENTIMENT-PRICE DIVERGENCE (CRITICAL): If flagged as DIVERGENCE_BULLISH or DIVERGENCE_BEARISH, STOP forcing the news to explain the price move.
    - Real markets are driven by institutional positioning, not just headlines.
    - If news is positive but price falls (DIVERGENCE_BEARISH), explicitly state that institutional flows and near-term positioning dominated the session, leading to distribution despite positive news.
    - If news is negative but price rises (DIVERGENCE_BULLISH), attribute the move to price absorption, capital rotation, or technical buying overriding the bad news.
11. NEWS USAGE: The supplied sector news is already bucketed into global macro, regulatory, and sectoral drivers.
    - If any bucket is populated, use it to establish the fundamental context.
    - Prefer concrete article themes over generic fallback language.
    - NEVER invent a news catalyst for a technical breakdown/breakout. Provide the news for context, but attribute the daily move to flows/positioning if the news contradicts the move.
12. DRIVER RANKING: Each sector also includes a movement classification plus ranked evidence.
    - Lead with the Primary Driver when it is credible.
    - Mention Secondary Driver only if it adds information.
    - Use Key Risk / Contradiction to explain when price is ignoring news.
13. EVIDENCE STATE: If evidence state is INSUFFICIENT_EVIDENCE, say the move may be flow-driven and avoid overclaiming.
14. CAUSALITY: Do not say a headline "caused" the move unless the data explicitly proves it. Use phrasing like "likely driver", "coincided with", or "plausible catalyst".
15. VOLUME CLAIMS & MOVE CLASSIFICATION: 
    - Volume 1.2x-2.0x indicates elevated activity/positioning, NOT necessarily an established trend.
    - If sector move is small (e.g., -0.5% to +0.5%) but volume is HIGH (>1.2x), this is NOT a quiet/flat "TECHNICAL_ONLY" day. It is FLOW_DRIVEN (institutional distribution/accumulation or sector rebalancing).
    - If sector move is small and volume is LOW (<1.0x), classify as "POSITIONING / LOW CONVICTION". Do not call it "SECTOR_DRIVEN".
16. DISTRIBUTION / ACCUMULATION / REBALANCING: Use these terms when volume contradicts the price magnitude. High volume on a flat day suggests a battle between buyers and sellers or index rebalancing.
17. TIME HORIZON: Respect the supplied article horizons. Immediate macro risk can outweigh medium-term positive news without being a true contradiction.
18. NO CLEAR CATALYST RULE: If the daily price move is strictly within -1.0% to +1.0% AND volume is BELOW_AVERAGE (<1.0x), explicitly classify the Move Type as "POSITIONING / LOW-CONVICTION MOVE" and explicitly state "No clear catalyst detected."
    - Explain that the sector is experiencing internal subsector rotation, minor profit-taking, index flows, or exhaustion.
    - NEVER force a geopolitical or breaking news narrative on a low volume, low conviction day.
19. DEFENSIVE ROTATION (PHARMA/FMCG/IT): If historically defensive sectors (like Pharma, FMCG, IT) show positive performance on a day when high-beta sectors (like Banking or Metals) are falling, explicitly classify the move as "Defensive Rotation" or "Selective Accumulation."
     - DO NOT force background news (like minor clinical trials, generic approvals, or international drug developments) to be the primary catalyst.
     - The PRIMARY explanation MUST be: "The sector's resilience likely reflects defensive capital rotation rather than a single news catalyst."
     - Any news should be introduced ONLY as: "News provides a supportive sector backdrop" — never as "News drove the move."
     - For export-oriented pharma names (e.g., Dr Reddy's, Sun Pharma), add currency sensitivity as a secondary factor when USD/INR moves > 0.2%.
20. SENTIMENT FORMATTING (STRICT BANDS):
     - Score 0%-44%: Label as "negative" or "bearish skew".
     - Score 45%-55%: Label as "neutral".
     - Score 56%-65%: Label as "mildly positive" or "slight positive skew".
     - Score 66%-80%: Label as "positive" or "limited bearish coverage".
     - Score 81%-100%: Label as "strongly positive".
     - CRITICAL: NEVER pair the word "neutral" with a score above 55%. This is contradictory and confusing for traders.
21. WEAK NEWS DOWNGRADE: If cited news articles describe background developments (international drug trials, long-term infrastructure plans, routine regulatory filings, foreign company earnings) rather than direct same-day catalysts for the Indian market, you MUST:
     - Label them as "context" or "supportive backdrop", NOT as drivers.
     - Use phrasing like: "Background sector developments include..." or "News provides context but is not a direct catalyst."
     - Lead with the actual driver: institutional flows, rotation, technical momentum, or positioning.
     - NEVER attribute a sector move primarily to background news when the data shows it was flow-driven or technically driven.
22. SIDEWAYS CONSOLIDATION RULE (ALL SECTORS): If the move type is "SIDEWAYS_CONSOLIDATION", the sector traded essentially flat.
     - DO NOT force any catalyst, news driver, or structural explanation.
     - Describe it as: "The sector traded largely sideways with limited participation and no directional conviction."
     - Attribute to: index rebalancing flows, routine profit-taking, or absence of fresh catalysts.
     - Single operational events (e.g., "one company loading cargo at a port") are NEVER sector-wide catalysts. Ignore them.
     - Keep the analysis SHORT — flat moves need SHORT explanations.
23. SENTIMENT-EVIDENCE CONSISTENCY: If the evidence state is "INSUFFICIENT_EVIDENCE" or "INSUFFICIENT_DATA", do NOT cite the raw sentiment percentage.
     - Instead say: "Sentiment data is limited; the move is likely flow-driven."
     - NEVER pair INSUFFICIENT_DATA/INSUFFICIENT_EVIDENCE with a high confidence sentiment score. This is contradictory.

{output_instructions}

End with: \u26a0\ufe0f This is not financial advice. For educational purposes only."""

    # Determine currency based on magnitude (MCX is ~8000+, YF is ~90)
    try:
        _cur = "₹" if float(prices.get('oil', 0)) > 1000 else "$"
    except:
        _cur = "$"
        
    user_prompt = f"""
LIVE DATA (READ CURRENCY CAREFULLY):
- Crude Oil: {_cur}{prices['oil']}/barrel | Gold: {_cur}{prices['gold']} | USD/INR: ₹{prices['usd_inr']}
- US 10Y Yield: {prices.get('bond_10y', 'N/A')}% | India G-Sec: {prices.get('india_gsec', 'N/A')}%
- Copper: {_cur}{prices.get('copper', 'N/A')} | Steel: {_cur}{prices.get('steel', 'N/A')}
- Nasdaq: {nasdaq_text}

SECTOR PERFORMANCE TODAY:
{chr(10).join(f"- {s}: {sector_changes.get(s, 0):.2f}%" for s in selected)}

NIFTY SECTORS:
{nifty_text}

FINBERT ML SENTIMENT:
{sent_summary}

HYBRID SIGNAL RESULTS (USE THESE — DO NOT OVERRIDE):
{signal_context}
{sector_news_text}
Analyze: {sectors_text}"""

    if deep_dive:
        try:
            response = gemini_client.models.generate_content(
                model="models/gemini-2.5-flash", contents=f"{system_prompt}\n\n{user_prompt}")
            return inject_sector_technical_sections(response.text, sector_dossiers or {}, deep_dive=True)
        except Exception as exc:
            log_fetch_warning("Gemini analysis", exc)
            return f"⚠️ Gemini error: {str(exc)}"
    else:
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant", temperature=0.0, max_tokens=1000,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
            return inject_sector_technical_sections(
                response.choices[0].message.content,
                sector_dossiers or {},
                deep_dive=False,
            )
        except Exception as exc:
            log_fetch_warning("Groq analysis", exc)
            if "rate_limit" in str(exc).lower():
                return "⚠️ Groq rate limit hit. Please wait 60 seconds."
            return f"⚠️ Analysis failed: {str(exc)}"

# ══════════════════════════════════════════════════════════════
# UI HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════

def render_metric_card(label, value, pts, pct):
    sign = "+" if pts > 0 else ""
    cls = "metric-delta-up" if pts > 0 else "metric-delta-down"
    arrow = "▲" if pts > 0 else "▼"
    return f"""
    <div class="metric-box">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="{cls}">{arrow} {sign}{pts} ({sign}{pct}%)</div>
    </div>"""

def render_signal_card(hybrid_signal):
    hs = hybrid_signal
    if "Invest" in hs.signal:
        card_class = "signal-card-invest"
        accent = "#00e676"
        pulse = "rally-alert" if hs.rally_alert else ""
    elif "Avoid" in hs.signal:
        card_class = "signal-card-avoid"
        accent = "#ff1744"
        pulse = "crash-warning" if hs.crash_warning else ""
    else:
        card_class = "signal-card-hold"
        accent = "#ffab00"
        pulse = ""

    conf_pct = int(hs.confidence * 100)
    conf_color = accent

    # Color-coded momentum/volume/sentiment mini-bars
    def score_color(score):
        if score >= 0.65: return "#00e676"
        elif score >= 0.45: return "#ffab00"
        else: return "#ff1744"

    m_score = getattr(hs, "momentum_score", 0.5)
    v_score = getattr(hs, "volume_score", 0.5)

    m_color = score_color(m_score)
    v_color = score_color(v_score)

    reasoning_html = ""
    for r in hs.reasoning[:6]:
        reasoning_html += f'<div style="color:#aaa;font-size:12px;padding:2px 0;">  {r}</div>'

    alert_html = ""
    if hs.crash_warning:
        alert_html = '<div style="background:rgba(255,23,68,0.15);border:1px solid rgba(255,23,68,0.3);border-radius:8px;padding:8px;margin-top:8px;color:#ff6b6b;font-weight:600;text-align:center;">🚨 CRASH WARNING — Momentum + Volume + Technicals all bearish</div>'
    elif hs.rally_alert:
        alert_html = '<div style="background:rgba(0,230,118,0.15);border:1px solid rgba(0,230,118,0.3);border-radius:8px;padding:8px;margin-top:8px;color:#69f0ae;font-weight:600;text-align:center;">🚀 RALLY ALERT — Momentum + Volume + Technicals all bullish</div>'

    # Agreement badge
    agreement_display = hs.agreement
    agreement_color = "#aaa"
    if hs.agreement == "DIVERGENCE_BULLISH":
        agreement_display = "🔥 BULLISH DIVERGENCE"
        agreement_color = "#00e676"
    elif hs.agreement == "DIVERGENCE_BEARISH":
        agreement_display = "⚠️ BEARISH DIVERGENCE"
        agreement_color = "#ff1744"
    elif hs.agreement == "ALIGNED":
        agreement_color = "#00e676"
    elif hs.agreement == "CONFLICT":
        agreement_color = "#ff1744"

    return f"""
    <div class="{card_class} {pulse}">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div>
                <div style="font-size:20px;font-weight:700;color:#e8e8f0;">{hs.sector}</div>
                <div style="font-size:28px;font-weight:800;color:{accent};margin-top:4px;">{hs.signal}</div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:36px;font-weight:800;color:{accent};">{conf_pct}%</div>
                <div style="font-size:11px;color:#888;text-transform:uppercase;">Confidence</div>
            </div>
        </div>
        <div style="margin-top:12px;">
            <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{conf_pct}%;background:{conf_color};"></div></div>
        </div>
        <div style="display:flex;gap:8px;margin-top:12px;flex-wrap:wrap;">
            <div style="flex:1;min-width:70px;background:rgba(255,255,255,0.03);border-radius:8px;padding:8px;border-left:3px solid {m_color};">
                <div style="font-size:10px;color:#888;text-transform:uppercase;">📈 Momentum</div>
                <div style="font-size:16px;font-weight:700;color:{m_color};">{m_score:.0%}</div>
                <div style="font-size:10px;color:#666;">40% weight</div>
            </div>
            <div style="flex:1;min-width:70px;background:rgba(255,255,255,0.03);border-radius:8px;padding:8px;">
                <div style="font-size:10px;color:#888;text-transform:uppercase;">⚙️ Technical</div>
                <div style="font-size:14px;font-weight:600;color:#e8e8f0;">{hs.technical_direction}</div>
                <div style="font-size:10px;color:#666;">25% · {hs.technical_confidence:.0%}</div>
            </div>
            <div style="flex:1;min-width:70px;background:rgba(255,255,255,0.03);border-radius:8px;padding:8px;border-left:3px solid {v_color};">
                <div style="font-size:10px;color:#888;text-transform:uppercase;">📊 Volume</div>
                <div style="font-size:16px;font-weight:700;color:{v_color};">{v_score:.0%}</div>
                <div style="font-size:10px;color:#666;">20% weight</div>
            </div>
            <div style="flex:1;min-width:70px;background:rgba(255,255,255,0.03);border-radius:8px;padding:8px;">
                <div style="font-size:10px;color:#888;text-transform:uppercase;">🧠 Sentiment</div>
                <div style="font-size:14px;font-weight:600;color:#e8e8f0;">{hs.sentiment_label.title()}</div>
                <div style="font-size:10px;color:#666;">15% · {hs.sentiment_score:.0%}</div>
            </div>
        </div>
        <div style="margin-top:8px;text-align:center;">
            <span style="font-size:12px;font-weight:600;color:{agreement_color};padding:3px 10px;background:rgba(255,255,255,0.05);border-radius:12px;">{agreement_display}</span>
        </div>
        {reasoning_html}
        {alert_html}
    </div>"""


def render_evidence_panel(dossier):
    taxonomy = ", ".join(dossier.get("taxonomy", [])) if dossier.get("taxonomy") else "none"
    evidence_html = ""
    technical_lines = dossier.get("technical_indicator_lines", [])
    technical_html = ""
    if technical_lines:
        technical_html = "".join(
            f'<div style="color:#b8c6db;font-size:12px;margin-top:4px;">• {line}</div>'
            for line in technical_lines
        )

    stock_contribution_lines = dossier.get("stock_contribution_lines", [])
    stock_contribution_html = ""
    if stock_contribution_lines:
        stock_contribution_html = "".join(
            f'<div style="color:#e8e8f0;font-size:12px;margin-top:4px;">• {line}</div>'
            for line in stock_contribution_lines[:2]
        )

    for item in dossier.get("ranked_evidence", []):
        evidence_html += f"""
        <div style="padding:10px 0;border-top:1px solid rgba(255,255,255,0.06);">
            <div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start;">
                <div style="flex:1;">
                    <div style="color:#e8e8f0;font-weight:600;font-size:13px;">{item.get('headline', '')}</div>
                    <div style="color:#888;font-size:11px;margin-top:4px;">{item.get('reason_category', '').title()} · {item.get('source', '')} · {item.get('published_at', '')}</div>
                    <div style="color:#aaa;font-size:12px;margin-top:6px;">{item.get('summary', '')}</div>
                </div>
                <div style="min-width:86px;text-align:right;">
                    <div style="color:#69f0ae;font-weight:700;font-size:15px;">{int(item.get('reason_score', 0) * 100)}%</div>
                    <div style="color:#777;font-size:10px;">reason score</div>
                </div>
            </div>
        </div>"""

    if not evidence_html:
        evidence_html = '<div style="color:#888;font-size:12px;">No ranked sector-specific evidence. Current explanation should rely on technical/flow context.</div>'

    return f"""
    <div class="glass-card" style="padding:16px;">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;">
            <div>
                <div style="font-size:18px;font-weight:700;color:#e8e8f0;">{dossier.get('sector', '')}</div>
                <div style="font-size:12px;color:#69f0ae;margin-top:4px;">{dossier.get('move_type', 'MIXED')}</div>
            </div>
            <div style="font-size:12px;color:#888;text-align:right;">{dossier.get('evidence_state', 'SUFFICIENT_EVIDENCE')}</div>
        </div>
        <div style="margin-top:8px;color:#888;font-size:12px;">taxonomy: {taxonomy}</div>
        <div style="margin-top:12px;color:#e8e8f0;font-size:13px;"><strong>Primary:</strong> {dossier.get('primary_driver', '')}</div>
        <div style="margin-top:6px;color:#e8e8f0;font-size:13px;"><strong>Secondary:</strong> {dossier.get('secondary_driver', '')}</div>
        <div style="margin-top:6px;color:#e8e8f0;font-size:13px;"><strong>Technical:</strong> {dossier.get('technical_confirmation', '')}</div>
        <div style="margin-top:8px;padding:10px;border-radius:10px;background:rgba(41,121,255,0.08);border:1px solid rgba(41,121,255,0.18);">
            <div style="color:#7fb3ff;font-size:11px;font-weight:700;letter-spacing:0.04em;">MANDATORY TECHNICAL INDICATORS</div>
            {technical_html or '<div style="color:#b8c6db;font-size:12px;margin-top:4px;">No technical detail lines available.</div>'}
        </div>
        <div style="margin-top:8px;color:#e8e8f0;font-size:13px;"><strong>Stock Contribution:</strong></div>
        {stock_contribution_html or '<div style="color:#888;font-size:12px;margin-top:4px;">No stock contribution lines available.</div>'}
        <div style="margin-top:6px;color:#e8e8f0;font-size:13px;"><strong>Pressure:</strong> {dossier.get('pressure_context', '')}</div>
        <div style="margin-top:6px;color:#888;font-size:13px;"><strong>Causality:</strong> {dossier.get('causality_note', '')}</div>
        <div style="margin-top:6px;color:#e8e8f0;font-size:13px;"><strong>Peers:</strong> {dossier.get('peer_snapshot', '')}</div>
        <div style="margin-top:6px;color:#ffab00;font-size:13px;"><strong>Risk:</strong> {dossier.get('key_risk', '')}</div>
        <div style="margin-top:12px;">{evidence_html}</div>
    </div>"""


def render_stock_spotlight(stock_dossier, stock_report):
    escaped_stock_report = escape(stock_report)
    taxonomy = ", ".join(stock_dossier.get("taxonomy", [])) if stock_dossier.get("taxonomy") else "none"
    evidence_rows = ""
    for item in stock_dossier.get("ranked_evidence", []):
        evidence_rows += f"""
        <div style="padding:10px 0;border-top:1px solid rgba(255,255,255,0.06);">
            <div style="display:flex;justify-content:space-between;gap:12px;">
                <div style="flex:1;">
                    <div style="color:#e8e8f0;font-weight:600;font-size:13px;">{item.get('headline', '')}</div>
                    <div style="color:#888;font-size:11px;margin-top:4px;">{item.get('reason_category', '')} · {item.get('source', '')} · {item.get('published_at', '')}</div>
                </div>
                <div style="color:#69f0ae;font-weight:700;font-size:14px;">{int(item.get('reason_score', 0) * 100)}%</div>
            </div>
        </div>"""

    if not evidence_rows:
        evidence_rows = '<div style="color:#888;font-size:12px;">No strong stock-specific article cluster. This move is being treated as technical/flow-led.</div>'

    return f"""
    <div class="glass-card" style="padding:18px;">
        <div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start;">
            <div>
                <div style="font-size:20px;font-weight:700;color:#e8e8f0;">{stock_dossier['stock_name']} ({stock_dossier['ticker']})</div>
                <div style="font-size:12px;color:#69f0ae;margin-top:4px;">{stock_dossier['move_type']} · {stock_dossier['evidence_state']}</div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:24px;font-weight:800;color:#e8e8f0;">{stock_dossier['daily_change']:+.2f}%</div>
                <div style="font-size:11px;color:#888;">today</div>
            </div>
        </div>
        <div style="margin-top:10px;color:#888;font-size:12px;">taxonomy: {taxonomy}</div>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-top:12px;">
            <div style="background:rgba(255,255,255,0.03);padding:10px;border-radius:10px;"><div style="font-size:10px;color:#888;">VS SECTOR</div><div style="font-size:18px;font-weight:700;color:#e8e8f0;">{stock_dossier['relative_strength_vs_sector']:+.2f}%</div></div>
            <div style="background:rgba(255,255,255,0.03);padding:10px;border-radius:10px;"><div style="font-size:10px;color:#888;">VS NIFTY</div><div style="font-size:18px;font-weight:700;color:#e8e8f0;">{stock_dossier['relative_strength_vs_nifty']:+.2f}%</div></div>
            <div style="background:rgba(255,255,255,0.03);padding:10px;border-radius:10px;"><div style="font-size:10px;color:#888;">TECHNICAL</div><div style="font-size:12px;color:#e8e8f0;">{stock_dossier['technical_confirmation']}</div></div>
        </div>
        <div style="margin-top:12px;color:#e8e8f0;font-size:13px;"><strong>Primary:</strong> {stock_dossier['primary_driver']}</div>
        <div style="margin-top:6px;color:#e8e8f0;font-size:13px;"><strong>Secondary:</strong> {stock_dossier['secondary_driver']}</div>
        <div style="margin-top:6px;color:#e8e8f0;font-size:13px;"><strong>Pressure:</strong> {stock_dossier['pressure_context']}</div>
        <div style="margin-top:6px;color:#888;font-size:13px;"><strong>Causality:</strong> {stock_dossier['causality_note']}</div>
        <div style="margin-top:6px;color:#e8e8f0;font-size:13px;"><strong>Peers:</strong> {stock_dossier['peer_snapshot']}</div>
        <div style="margin-top:6px;color:#ffab00;font-size:13px;"><strong>Risk:</strong> {stock_dossier['key_risk']}</div>
        <div style="margin-top:14px;color:#e8e8f0;font-size:13px;white-space:pre-wrap;">{escaped_stock_report}</div>
        <div style="margin-top:12px;">{evidence_rows}</div>
    </div>"""


def render_fear_greed_gauge(fg_data):
    score = fg_data["score"]
    label = fg_data["label"]
    if score >= 60: color = "#00e676"
    elif score >= 40: color = "#ffab00"
    else: color = "#ff1744"

    components_html = ""
    for k, v in fg_data.get("components", {}).items():
        components_html += f'<div style="display:flex;justify-content:space-between;padding:3px 0;"><span style="color:#888;font-size:12px;">{k.replace("_", " ").title()}</span><span style="color:#e8e8f0;font-size:12px;font-weight:600;">{v}</span></div>'

    return f"""
    <div class="glass-card">
        <div class="fg-gauge">
            <div style="font-size:12px;color:#888;text-transform:uppercase;letter-spacing:2px;">Market Mood</div>
            <div class="fg-score" style="color:{color};">{score}</div>
            <div class="fg-label" style="color:{color};">{label}</div>
            <div style="margin-top:16px;max-width:200px;margin-left:auto;margin-right:auto;">
                <div class="conf-bar-bg" style="height:10px;">
                    <div class="conf-bar-fill" style="width:{score}%;background:linear-gradient(90deg, #ff1744, #ffab00, #00e676);"></div>
                </div>
            </div>
            <div style="margin-top:16px;">{components_html}</div>
        </div>
    </div>"""

def create_sentiment_gauge(sentiment_data):
    pos = sentiment_data.get("positive_pct", 0)
    neg = sentiment_data.get("negative_pct", 0)
    neu = sentiment_data.get("neutral_pct", 0)
    fig = go.Figure(go.Bar(
        x=[pos, neu, neg],
        y=["Bullish", "Neutral", "Bearish"],
        orientation='h',
        marker_color=["#00e676", "#ffab00", "#ff1744"],
        text=[f"{pos:.0f}%", f"{neu:.0f}%", f"{neg:.0f}%"],
        textposition='auto',
        textfont=dict(color="white", size=14),
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8e8f0"), height=180, margin=dict(l=60,r=20,t=10,b=10),
        xaxis=dict(showgrid=False, showticklabels=False, range=[0, 100]),
        yaxis=dict(showgrid=False),
    )
    return fig

def create_sector_ticker_tape(sector_changes):
    """Build a scrolling ticker-tape HTML strip with live sector % changes."""
    import time
    
    sector_icons = {
        "IT": "💻", "Banking": "🏦", "FMCG": "🛒", "Oil & Gas": "⛽",
        "Pharma": "💊", "Metals": "⚙️", "Infrastructure": "🏗️", "Gold": "🥇"
    }
    items = []
    for name, chp in sector_changes.items():
        icon = sector_icons.get(name, "📊")
        color = "#00e676" if chp > 0 else "#ff1744" if chp < 0 else "#8888a0"
        arrow = "▲" if chp > 0 else "▼" if chp < 0 else "▬"
        items.append(
            f'<div style="display:inline-flex; align-items:center; padding: 6px 14px; background: rgba(255,255,255,0.03); border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); margin: 0 10px;">'
            f'<span style="margin-right:6px;">{icon}</span>'
            f'<strong style="color:#e8e8f0;font-size:14px;white-space:nowrap;margin-right:6px;">{name}</strong>'
            f'<span style="color:{color};font-weight:700;font-size:14px;white-space:nowrap;">{arrow} {chp:+.2f}%</span>'
            f'</div>'
        )
    tape_content = "".join(items)
    
    # Generate a unique ID for this specific render tick
    # This completely overrides Streamlit's virtual DOM diffing logic and forces it 
    # to render a brand new HTML node, effectively restarting the scroll animation 
    # but keeping it in sync.
    tick_id = int(time.time() * 1000)
    
    return f"""
    <style>
    @keyframes scrollAnimation_{tick_id} {{
        0%   {{ transform: translateX(0); }}
        100% {{ transform: translateX(calc(-50% - 10px)); }}
    }}
    .live-ticker-container_{tick_id} {{
        overflow: hidden;
        white-space: nowrap;
        background: linear-gradient(90deg, rgba(20,20,35,0.95), rgba(15,15,26,0.95));
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 12px 0;
        margin: 8px 0 16px;
        position: relative;
        width: 100%;
        display: flex;
    }}
    .live-ticker-track_{tick_id} {{
        display: inline-flex;
        animation: scrollAnimation_{tick_id} 30s linear infinite;
        will-change: transform;
    }}
    .live-ticker-track_{tick_id}:hover {{
        animation-play-state: paused;
    }}
    </style>
    <div class="live-ticker-container_{tick_id}">
        <div class="live-ticker-track_{tick_id}">
            {tape_content}{tape_content}
        </div>
    </div>
    """


def create_relative_strength_chart(sector_changes, nifty50_chp):
    """Build a divergence bar chart: each sector's relative strength vs Nifty50."""
    sectors = list(sector_changes.keys())
    rel_strength = [round(sector_changes.get(s, 0) - nifty50_chp, 2) for s in sectors]

    # Sort by relative strength (strongest first)
    paired = sorted(zip(sectors, rel_strength), key=lambda x: x[1], reverse=True)
    sectors_sorted = [p[0] for p in paired]
    rs_sorted = [p[1] for p in paired]

    colors = [
        "#00e676" if v > 0.5 else "#69f0ae" if v > 0
        else "#ff1744" if v < -0.5 else "#ff5252"
        for v in rs_sorted
    ]

    fig = go.Figure(go.Bar(
        x=rs_sorted, y=sectors_sorted, orientation='h', marker_color=colors,
        text=[f"{v:+.2f}%" for v in rs_sorted], textposition='auto',
        textfont=dict(color="white", size=13, family="Inter"),
    ))
    # Add zero-line for Nifty50 benchmark
    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    fig.add_annotation(
        x=0, y=1.05, yref="paper", text=f"Nifty50 ({nifty50_chp:+.2f}%)",
        showarrow=False, font=dict(color="#8888a0", size=11),
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8e8f0", family="Inter"), height=300,
        margin=dict(l=100, r=30, t=30, b=10),
        xaxis=dict(
            showgrid=True, gridcolor="rgba(255,255,255,0.05)",
            title="Relative Strength vs Nifty50 (%)",
            zeroline=True, zerolinecolor="rgba(255,255,255,0.2)",
        ),
        yaxis=dict(showgrid=False),
    )
    return fig

# ══════════════════════════════════════════════════════════════
# MAIN APP FLOW
# ══════════════════════════════════════════════════════════════

# Auto-load market data
with st.spinner("🌍 Loading live market data..."):
    prices = get_commodity_prices()
    nifty_data = get_nifty_sector_data()
    sector_changes = get_nifty_sector_changes()

try:
    import yfinance as yf
    nq_hist = yf.Ticker("^IXIC").history(period="2d")
    nq_change = round(((nq_hist['Close'].iloc[-1] - nq_hist['Close'].iloc[-2]) / nq_hist['Close'].iloc[-2]) * 100, 2)
    nasdaq_text = f"{'+' if nq_change > 0 else ''}{nq_change}%"
except Exception as exc:
    log_fetch_warning("Nasdaq dashboard fetch", exc)
    nasdaq_text = "N/A"

ist = pytz.timezone('Asia/Kolkata')
now = datetime.now(ist).strftime("%d %b %Y, %I:%M %p IST")

def get_delta(hist):
    if len(hist) >= 2:
        prev, today = hist['Close'].iloc[-2], hist['Close'].iloc[-1]
        return round(today, 2), round(today - prev, 2), round(((today - prev) / prev) * 100, 2)
    return 0, 0, 0

# ── DASHBOARD FRAGMENT: Auto-refreshing Snapshot ──
# is_indian_market_hours() is defined above near get_nifty_sector_data()


def get_yf_quotes(tickers):
    """Fallback: fetch last-close data from yfinance for the market snapshot."""
    import yfinance as yf
    results = {}
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period="2d")
            if len(hist) >= 2:
                lp = round(hist['Close'].iloc[-1], 2)
                ch = round(hist['Close'].iloc[-1] - hist['Close'].iloc[-2], 2)
                chp = round(((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100, 2)
                results[t] = {"lp": lp, "ch": ch, "chp": chp}
            elif len(hist) >= 1:
                results[t] = {"lp": round(hist['Close'].iloc[-1], 2), "ch": 0, "chp": 0}
            else:
                results[t] = {"lp": 0, "ch": 0, "chp": 0}
        except Exception:
            results[t] = {"lp": 0, "ch": 0, "chp": 0}
    return results


def get_smart_quotes(tickers):
    """Auto-switch between Fyers (market hours) and yfinance (after hours)."""
    market_open = is_indian_market_hours()
    source = "fyers"

    if market_open:
        try:
            live_q = get_live_quotes(tickers)
            live_count = sum(1 for q in live_q.values() if q.get("lp", 0))
            if live_count > 0:
                return live_q, "fyers"
            # Fyers returned all zeros — fall back
            source = "yfinance"
        except Exception:
            source = "yfinance"
    else:
        source = "yfinance"

    return get_yf_quotes(tickers), source


# ── Auto-refresh config based on market hours ──
refresh_time = "10s" if market_open else None

@st.fragment(run_every=refresh_time)
def dashboard_snapshot():
    now_refresh = datetime.now(timezone.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d %H:%M:%S")
    col_r1, col_r2 = st.columns([6, 1])
    with col_r1:
        st.caption(f"🕐 Live Ticking... last fetched: {now_refresh}")
    with col_r2:
        if st.button("🔄 Full Refresh"):
            st.cache_data.clear()
            st.rerun()

    fyers_tickers = ["^NSEI", "^BSESN", "^NSEBANK", "^CNXIT", "GC=F", "CL=F", "INR=X", "SI=F"]
    live_q, data_source = get_smart_quotes(fyers_tickers)

    # ── Fetch sector data fresh on every fragment tick ──
    # get_nifty_sector_changes() uses ttl=10s cache during market hours (Fyers live)
    # and ttl=900s cache after hours (yfinance). The fragment fires every 10s,
    # so the cache expires and fresh data is picked up automatically.
    live_sector_changes = get_nifty_sector_changes()
    is_live = is_indian_market_hours() and len(live_sector_changes) >= 4
    sector_data_source = "Fyers Live" if is_live else "Last Close"

    # Get Nifty50 change for relative strength calculation
    nifty50_chp = live_q.get("^NSEI", {}).get("chp", 0)

    live_quote_count = sum(1 for quote in live_q.values() if quote.get("lp", 0))

    if live_quote_count == 0:
        st.warning(
            "Market data unavailable. If during market hours, check your Fyers token. "
            "After hours, yfinance may not have data for MCX futures."
        )
        st.caption(render_fyers_debug_message())

    if data_source == "yfinance" and live_quote_count > 0:
        st.info("📊 After hours — showing **last close** data via yfinance. Live Fyers polling resumes at 8 AM IST.")

    def q_val(t):
        q = live_q.get(t, {"lp": 0, "ch": 0, "chp": 0})
        return q["lp"], q["ch"], q["chp"]

    # Data source label
    source_label = "Live Polling" if data_source == "fyers" else "Last Close"

    # Market Snapshot
    st.markdown("""<div class="glass-card" style="padding:24px;">
        <h3 style="color:#e8e8f0;margin-top:0;">📡 Live Market Snapshot</h3>
    """, unsafe_allow_html=True)

    st.markdown(f"##### 🇮🇳 Indian Indices ({source_label})")
    c1, c2, c3, c4 = st.columns(4)
    n_v, n_p, n_pc = q_val("^NSEI")
    s_v, s_p, s_pc = q_val("^BSESN")
    b_v, b_p, b_pc = q_val("^NSEBANK")
    i_v, i_p, i_pc = q_val("^CNXIT")
    with c1: st.markdown(render_metric_card("NIFTY 50", f"{n_v:,.2f}" if n_v else "N/A", n_p, n_pc), unsafe_allow_html=True)
    with c2: st.markdown(render_metric_card("SENSEX", f"{s_v:,.2f}" if s_v else "N/A", s_p, s_pc), unsafe_allow_html=True)
    with c3: st.markdown(render_metric_card("BANK NIFTY", f"{b_v:,.2f}" if b_v else "N/A", b_p, b_pc), unsafe_allow_html=True)
    with c4: st.markdown(render_metric_card("NIFTY IT", f"{i_v:,.2f}" if i_v else "N/A", i_p, i_pc), unsafe_allow_html=True)

    st.markdown("##### 🌍 Global Indices (Context)")
    @st.cache_data(ttl=60)
    def fetch_yf_delta(ticker):
        import yfinance as yf
        try:
            df = yf.Ticker(ticker).history(period="2d")
            return get_delta(df)
        except Exception:
            return None, None, None

    c1, c2, c3, c4 = st.columns(4)
    sp_v, sp_p, sp_pc = fetch_yf_delta("^GSPC")
    nq_v, nq_p, nq_pc = fetch_yf_delta("^IXIC")
    dw_v, dw_p, dw_pc = fetch_yf_delta("^DJI")
    ft_v, ft_p, ft_pc = fetch_yf_delta("^FTSE")
    with c1: st.markdown(render_metric_card("S&P 500", f"{sp_v:,.2f}" if sp_v else "N/A", sp_p, sp_pc), unsafe_allow_html=True)
    with c2: st.markdown(render_metric_card("NASDAQ", f"{nq_v:,.2f}" if nq_v else "N/A", nq_p, nq_pc), unsafe_allow_html=True)
    with c3: st.markdown(render_metric_card("DOW JONES", f"{dw_v:,.2f}" if dw_v else "N/A", dw_p, dw_pc), unsafe_allow_html=True)
    with c4: st.markdown(render_metric_card("FTSE 100", f"{ft_v:,.2f}" if ft_v else "N/A", ft_p, ft_pc), unsafe_allow_html=True)

    st.markdown(f"##### 🛢️ Commodities & Forex ({source_label})")
    c1, c2, c3, c4 = st.columns(4)
    g_v, g_p, g_pc = q_val("GC=F")
    o_v, o_p, o_pc = q_val("CL=F")
    r_v, r_p, r_pc = q_val("INR=X")
    sv_v, sv_p, sv_pc = q_val("SI=F")
    # Currency symbol: Fyers returns MCX prices in ₹, yfinance returns international prices in $
    _cur = "₹" if data_source == "fyers" else "$"
    _cur_label = "(₹)" if data_source == "fyers" else "($)"
    with c1: st.markdown(render_metric_card(f"🥇 GOLD {_cur_label}", f"{_cur}{g_v:,.0f}" if g_v else "N/A", g_p, g_pc), unsafe_allow_html=True)
    with c2: st.markdown(render_metric_card(f"🛢️ CRUDE {_cur_label}", f"{_cur}{o_v:,.0f}" if o_v else "N/A", o_p, o_pc), unsafe_allow_html=True)
    with c3: st.markdown(render_metric_card("💱 USD/INR", f"₹{r_v:,.4f}" if r_v else "N/A", r_p, r_pc), unsafe_allow_html=True)
    with c4: st.markdown(render_metric_card(f"🥈 SILVER {_cur_label}", f"{_cur}{sv_v:,.0f}" if sv_v else "N/A", sv_p, sv_pc), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.divider()

    # ── Sector Live Performance Strip ──
    tape_html = create_sector_ticker_tape(live_sector_changes)
    st.markdown(tape_html, unsafe_allow_html=True)

    # ── Sector vs Nifty50 Relative Strength ──
    st.markdown(f"### ⚡ Sector Relative Strength vs Nifty50 ({sector_data_source})")
    st.plotly_chart(create_relative_strength_chart(live_sector_changes, nifty50_chp), width="stretch", config={"displayModeBar": False})

# ── REPORT MODE ──
if generate:
    if not selected:
        st.warning("Please select at least one sector!")
    else:
        if st.button("⬅ Back to Dashboard"):
            st.rerun()

        # Step 1: Fetch news
        with st.spinner("📰 Fetching latest news..."):
            global_articles, indian_articles = get_all_news()
            sector_news = {}
            for sector in selected:
                sector_news[sector] = get_sector_news(sector)

        # Step 2: Run FinBERT sentiment analysis
        with st.spinner("🧠 Running FinBERT sentiment analysis..."):
            engine = load_sentiment_engine()
            all_headlines = [
                f"{article['headline']}. {article.get('summary', '')}".strip()
                for article in global_articles + indian_articles
            ]
            global_sentiment = engine.analyze_batch(all_headlines)
            nifty_50_change = get_nifty_50_change()

        # Step 3: Run technical guardrails
        with st.spinner("⚙️ Computing technical guardrails (RSI, MA, Volume)..."):
            tg = load_technical_guardrails()
            technical_results = {}
            for sector in selected:
                technical_results[sector] = tg.analyze_sector(sector)

        # Step 4: Run hybrid signal combiner
        with st.spinner("⚡ Combining signals — Hybrid ML engine..."):
            combiner = load_signal_combiner()
            hybrid_results = {}
            sector_dossiers = {}
            for sector in selected:
                # Use direct sector retrieval first; fall back to broad market context only if needed.
                sector_articles = annotate_articles_with_horizon(list(sector_news.get(sector, [])))
                if len(sector_articles) < 4:
                    sector_articles += annotate_articles_with_horizon(global_articles + indian_articles)
                sector_sentiment = engine.analyze_sector_headlines(sector_articles, sector)

                daily_change = sector_changes.get(sector, 0.0)
                hybrid_results[sector] = combiner.combine_signals(
                    technical=technical_results[sector],
                    sentiment=sector_sentiment,
                    sector=sector,
                    daily_change_pct=daily_change,
                )
                sector_dossiers[sector] = build_sector_driver_dossier(
                    sector,
                    technical_results[sector],
                    sector_sentiment,
                    daily_change,
                )

            # Fear & Greed
            fg_data = combiner.compute_fear_greed(
                list(hybrid_results.values()), global_sentiment)

        stock_dossier = None
        stock_report = ""
        if selected_stock:
            spotlight_sector, spotlight_ticker = selected_stock
            stock_name = STOCK_PICK_REASONS.get(spotlight_ticker, {}).get("name", spotlight_ticker.replace(".NS", ""))
            stock_analysis = next(
                (
                    stock for stock in technical_results.get(spotlight_sector, {}).get("stock_analyses", [])
                    if stock.get("ticker") == spotlight_ticker
                ),
                None,
            )
            if stock_analysis:
                stock_articles = get_stock_news(spotlight_ticker, stock_name, spotlight_sector)
                stock_dossier = build_stock_driver_dossier(
                    spotlight_sector,
                    spotlight_ticker,
                    stock_name,
                    stock_analysis,
                    sector_dossiers.get(spotlight_sector, {}),
                    sector_changes.get(spotlight_sector, 0.0),
                    nifty_50_change,
                    stock_articles,
                    engine,
                )
                stock_report = analyze_stock_with_llm(stock_dossier, deep_dive=deep_dive)

        # ── RENDER: Signal Cards ──
        st.markdown("""<div style="text-align:center;margin:20px 0;">
            <h2 style="font-size:28px;font-weight:800;color:#e8e8f0;">⚡ Momentum-First Signal Dashboard</h2>
            <p style="color:#888;font-size:13px;">40% Momentum · 25% Technical · 20% Volume · 15% Sentiment</p>
        </div>""", unsafe_allow_html=True)

        # Fear & Greed + Sentiment Overview
        col_fg, col_sent = st.columns([1, 2])
        with col_fg:
            st.markdown(render_fear_greed_gauge(fg_data), unsafe_allow_html=True)
        with col_sent:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 🧠 FinBERT News Sentiment")
            st.plotly_chart(create_sentiment_gauge(global_sentiment), width="stretch", config={"displayModeBar": False})
            st.caption(f"Analyzed {global_sentiment.get('headline_count', 0)} headlines with ProsusAI/FinBERT")
            st.markdown('</div>', unsafe_allow_html=True)

        st.divider()

        # Signal Cards Grid
        st.markdown("### 🎯 Sector Signals")
        cols = st.columns(min(len(selected), 3))
        for i, sector in enumerate(selected):
            with cols[i % min(len(selected), 3)]:
                st.markdown(render_signal_card(hybrid_results[sector]), unsafe_allow_html=True)

        st.markdown("### 🧾 Driver Evidence")
        for sector in selected:
            with st.expander(f"{sector} evidence stack", expanded=False):
                st.markdown(render_evidence_panel(sector_dossiers[sector]), unsafe_allow_html=True)

        if stock_dossier:
            st.markdown("### 📌 Stock Spotlight")
            st.markdown(render_stock_spotlight(stock_dossier, stock_report), unsafe_allow_html=True)

        st.divider()

        # News sections
        if show_global and global_articles:
            with st.expander("🌐 Global News Today", expanded=False):
                for a in global_articles[:10]:
                    sent = engine.analyze_headline(a['headline'])
                    sent_color = "#00e676" if sent['label'] == 'positive' else "#ff1744" if sent['label'] == 'negative' else "#ffab00"
                    st.markdown(f"""
                    <div class="glass-card" style="padding:12px;margin:4px 0;">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <a href="{a['url']}" target="_blank" style="color:#e8e8f0;text-decoration:none;font-weight:500;flex:1;">{a['headline']}</a>
                            <span style="color:{sent_color};font-size:12px;font-weight:700;padding:3px 8px;background:rgba(255,255,255,0.05);border-radius:12px;margin-left:8px;white-space:nowrap;">
                                {sent['label'].upper()} {sent['score']:.0%}
                            </span>
                        </div>
                    </div>""", unsafe_allow_html=True)

        if show_indian and indian_articles:
            with st.expander("📰 Indian Market News", expanded=False):
                for a in indian_articles[:10]:
                    sent = engine.analyze_headline(a['headline'])
                    sent_color = "#00e676" if sent['label'] == 'positive' else "#ff1744" if sent['label'] == 'negative' else "#ffab00"
                    st.markdown(f"""
                    <div class="glass-card" style="padding:12px;margin:4px 0;">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <a href="{a['url']}" target="_blank" style="color:#e8e8f0;text-decoration:none;font-weight:500;flex:1;">{a['headline']}</a>
                            <span style="color:{sent_color};font-size:12px;font-weight:700;padding:3px 8px;background:rgba(255,255,255,0.05);border-radius:12px;margin-left:8px;white-space:nowrap;">
                                {sent['label'].upper()} {sent['score']:.0%}
                            </span>
                        </div>
                    </div>""", unsafe_allow_html=True)

        st.divider()

        # AI Deep Analysis
        with st.expander("🤖 AI Analysis Report" if not deep_dive else "🔬 Deep Institutional Report", expanded=True):
            with st.spinner("🤖 Generating analysis..." if not deep_dive else "🔬 Compiling institutional report..."):
                report = analyze_with_llm(
                    prices, nifty_data, selected, hybrid_results,
                    global_sentiment, sector_changes, deep_dive,
                    sector_news=sector_news,
                    sector_dossiers=sector_dossiers)
            st.markdown(report.replace("$", "\\$"))

        eval_run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        eval_cases = build_evaluation_cases(eval_run_timestamp, sector_dossiers, stock_dossier=stock_dossier)
        if st.button("🧪 Save Evaluation Snapshot"):
            output_path = append_evaluation_cases(eval_cases)
            st.success(f"Saved {len(eval_cases)} cases to {output_path.name}")



# ── DASHBOARD MODE ──
else:
    dashboard_snapshot()

    st.divider()
    st.markdown("""<div class="glass-card" style="text-align:center;padding:30px;">
        <div style="font-size:18px;color:#888;">👈 Select sectors from the sidebar</div>
        <div style="font-size:14px;color:#666;margin-top:8px;">Click <b>Generate Report</b> to run the Hybrid ML Analysis</div>
        <div style="margin-top:16px;display:flex;justify-content:center;gap:20px;">
            <span style="color:#00e676;">🧠 FinBERT NLP</span>
            <span style="color:#2979ff;">⚙️ RSI + MA + Volume</span>
            <span style="color:#7c4dff;">⚡ Hybrid Combiner</span>
        </div>
    </div>""", unsafe_allow_html=True)

# Force Streamlit reload to clear cached imported modules
