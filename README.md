# 🌍 GeoAlpha — Momentum-First Market Intelligence

Real-time Indian stock market intelligence powered by a **momentum-first hybrid ML/NLP system** combining multi-timeframe price momentum, technical indicators, volume confirmation, and FinBERT sentiment analysis.

## Architecture (v2.9)

```
yfinance  → 📈 Multi-Timeframe Momentum (40%)  ─┐
yfinance  → ⚙️ RSI / MA / Technicals (25%)      ├──→ ⚡ Signal Combiner → AI Report
yfinance  → 📊 Volume Confirmation (20%)         │
News APIs → 🧠 FinBERT Sentiment (15%)         ──┘
```

## Key Features

- **Momentum-First Scoring** — 40% momentum, 25% technicals, 20% volume, 15% sentiment
- **Multi-Timeframe Momentum** — Blends daily (25%), weekly (35%), monthly (25%), and MA bias (15%)
- **Sentiment-Price Divergence** — Negative news + rising price = institutional buying (bullish), not conflict
- **Volume Confirmation** — Aggregates stock-level volume when sector index data unavailable
- **Expanded RSI Zones** — 30-40 leans bearish, 60-70 leans bullish (not all neutral)
- **LLM Anti-Hallucination** — Strict rules prevent AI from inventing fundamentals
- **Subsector Divergence** — Upstream vs downstream Oil & Gas, Ferrous vs Non-Ferrous Metals
- **Fear & Greed Index** — Momentum-weighted composite market mood gauge
- **Sector Driver Ranking** — Ranked evidence, event taxonomy, contradiction handling, and insufficient-evidence states
- **Stock Spotlight** — Stock-specific news first, sector second, macro third, with gap/breakout/relative-strength context
- **Evaluation Snapshots** — Save live runs to `evaluation_cases.jsonl` and score them with `explanation_evaluator.py`
- **Premium Dark UI** — Glassmorphic 4-panel signal cards with animated alerts

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # Fill in your API keys
streamlit run app.py
```

## Environment Variables

Create a `.env` file with your API keys (see `.env.example`).

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit + Custom CSS |
| ML/NLP | FinBERT (HuggingFace Transformers) |
| Technical Analysis | NumPy + yfinance |
| Charts | Plotly |
| LLM | Groq (Llama 3.1) + Google Gemini 2.5 Flash |
| Testing | Pytest (25 unit tests) |

## Evaluation Workflow

1. Run the app and generate a report.
2. Click `Save Evaluation Snapshot` to append real sector/stock cases to `evaluation_cases.jsonl`.
3. Manually label the `labels` block for each case.
4. Run:

```bash
python3 explanation_evaluator.py evaluation_cases.jsonl
```
