"""
GeoAlpha Sentiment Engine — FinBERT-powered financial NLP
=========================================================
Uses ProsusAI/finbert (fine-tuned BERT for financial text) to:
  1. Classify headline sentiment (positive / negative / neutral)
  2. Score confidence 0.0–1.0
  3. Aggregate batch sentiment across multiple headlines
  4. Detect sector/global/regulatory relevance with richer context
"""

import re
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Sector-headline relevance keywords ────────────────────────────
SECTOR_RELEVANCE = {
    "Metals": [
        "steel", "copper", "aluminium", "aluminum", "iron ore", "metal",
        "china manufacturing", "china pmi", "lme", "hindalco", "tata steel",
        "jsw", "sail", "mining", "coking coal", "zinc", "nickel",
        "china stimulus", "china property", "china demand"
    ],
    "Banking": [
        "rbi", "rate cut", "rate hike", "interest rate", "credit growth",
        "npa", "fii", "bank", "hdfc", "icici", "sbi", "kotak", "axis",
        "liquidity", "monetary policy", "g-sec", "bond yield", "nbfc",
        "deposit", "lending", "nim", "slippage", "slippages", "credit cost",
        "loan growth", "casa", "deposit growth", "net interest margin"
    ],
    "IT": [
        "it services", "tech spending", "nasdaq", "infosys", "tcs",
        "wipro", "hcl tech", "outsourcing", "ai contract", "cloud",
        "digital transformation", "enterprise tech", "silicon valley",
        "us recession", "tech layoff", "software", "deal win", "cloud deal",
        "genai", "ai spending", "digital deal", "us tech spending"
    ],
    "Pharma": [
        "usfda", "fda", "drug approval", "pharma", "generic", "sun pharma",
        "dr reddy", "lupin", "cipla", "biotech", "clinical trial",
        "pharma export", "api", "drug", "healthcare", "warning letter",
        "inspection", "form 483", "abbreviated new drug application"
    ],
    "Oil & Gas": [
        "crude oil", "opec", "oil price", "refining", "grm", "ongc",
        "reliance", "hpcl", "bpcl", "iocl", "petroleum", "natural gas",
        "oil sanction", "oil import", "fuel", "petrol", "diesel",
        "oil production", "refining margin", "refining margins", "lng",
        "brent", "upstream", "downstream"
    ],
    "Gold": [
        "gold price", "jewellery", "jewelry", "titan", "kalyan",
        "senco gold", "gold import", "gold etf", "wedding season",
        "festival", "precious metal", "bullion", "gold demand", "safe haven",
        "etf flow", "bullion import"
    ],
    "Infrastructure": [
        "infrastructure", "capex", "road project", "railway", "l&t",
        "larsen", "irb", "gmr", "smart city", "highway", "port",
        "construction", "cement", "real estate", "housing", "order book",
        "railway capex", "project award", "epc"
    ],
    "FMCG": [
        "fmcg", "rural demand", "consumer spending", "hul", "itc",
        "nestle", "britannia", "dabur", "marico", "colgate",
        "volume growth", "inflation", "consumer goods", "retail",
        "input cost", "palm oil", "urban demand", "pricing power"
    ],
}

SECTOR_MACRO_CONTEXT = {
    "Metals": [
        "china", "china demand", "china stimulus", "lme", "commodity demand",
        "manufacturing pmi", "iron ore", "copper price", "aluminium price"
    ],
    "Banking": [
        "rbi policy", "bond yields", "deposit growth", "loan demand",
        "credit cycle", "liquidity", "inflation print", "g-sec", "yield curve"
    ],
    "IT": [
        "nasdaq", "us tech spending", "cloud spending", "ai spending",
        "enterprise demand", "outsourcing demand", "recession", "deal pipeline"
    ],
    "Pharma": [
        "us healthcare", "drug pricing", "regulatory inspection", "approval",
        "generic demand", "api pricing"
    ],
    "Oil & Gas": [
        "brent", "crude", "opec", "sanctions", "refining margins",
        "lng", "oil demand", "energy market"
    ],
    "Gold": [
        "safe haven", "bullion", "dollar", "real yields", "etf flows",
        "central bank buying"
    ],
    "Infrastructure": [
        "government spending", "capex", "project pipeline", "tender",
        "railway spending", "highway spending", "order inflow"
    ],
    "FMCG": [
        "consumer demand", "input costs", "rural recovery", "urban demand",
        "inflation", "monsoon", "pricing"
    ],
}

REGULATORY_KEYWORDS = [
    "rbi", "sebi", "policy", "regulation", "regulatory", "approval",
    "government", "cabinet", "ministry", "tax", "duty", "tariff",
    "ban", "fda", "usfda", "tender", "order award"
]

# Geopolitical/global macro keywords
GEO_KEYWORDS = [
    "war", "conflict", "sanction", "tariff", "trade war", "missile",
    "military", "invasion", "ceasefire", "treaty", "nato", "un security",
    "embargo", "nuclear", "border tension", "coup", "election",
    "geopolitical", "diplomatic", "alliance", "fed", "bond yields",
    "dollar", "opec", "china stimulus", "global recession"
]

GEO_SENSITIVE_SECTORS = {"Oil & Gas", "Gold", "Banking", "Metals", "IT"}

STRICT_RELEVANCE_THRESHOLD = 0.65

SECTOR_COMPANY_ALIASES = {
    "Metals": ["tata steel", "hindalco", "jsw steel", "sail", "jindal steel"],
    "Banking": ["hdfc bank", "icici bank", "sbi", "kotak bank", "axis bank"],
    "IT": ["tcs", "infosys", "wipro", "hcl tech", "hcltech", "tech mahindra"],
    "Pharma": ["sun pharma", "dr reddy", "cipla", "lupin", "aurobindo pharma"],
    "Oil & Gas": ["reliance", "ongc", "bpcl", "hpcl", "ioc", "oil india"],
    "Gold": ["titan", "kalyan jewellers", "senco gold"],
    "Infrastructure": ["l&t", "larsen", "irb", "gmr", "nbcc"],
    "FMCG": ["hul", "hindustan unilever", "itc", "nestle india", "britannia", "dabur", "marico"],
}


def keyword_matches(text: str, phrases: list[str]) -> list[str]:
    matches = []
    for phrase in phrases:
        pattern = r"(?<!\w)" + re.escape(phrase).replace(r"\ ", r"\s+") + r"(?!\w)"
        if re.search(pattern, text):
            matches.append(phrase)
    return matches


class SentimentEngine:
    """FinBERT-based financial sentiment analyzer."""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._labels = ["positive", "negative", "neutral"]

    def _load_model(self):
        """Lazy-load FinBERT model (called once, then cached)."""
        if self._model is None:
            model_name = "ProsusAI/finbert"
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self._model.eval()
        return self._model, self._tokenizer

    # ── Core sentiment analysis ───────────────────────────────────
    def analyze_headline(self, text: str) -> dict:
        """
        Analyze a single headline.
        Returns: {"headline": str, "label": str, "score": float, "scores": dict}
        """
        model, tokenizer = self._load_model()

        inputs = tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=512, padding=True
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        scores = {
            self._labels[i]: round(probs[0][i].item(), 4)
            for i in range(3)
        }
        best_idx = probs[0].argmax().item()

        return {
            "headline": text,
            "label": self._labels[best_idx],
            "score": round(probs[0][best_idx].item(), 4),
            "scores": scores,
        }

    def analyze_batch(self, headlines: list[str]) -> dict:
        """
        Analyze multiple headlines and return aggregated sentiment.
        Returns:
          {
            "individual": [list of per-headline results],
            "aggregate_label": str,
            "aggregate_score": float,
            "positive_pct": float,
            "negative_pct": float,
            "neutral_pct": float,
            "headline_count": int,
            "bullish_headlines": [top positive],
            "bearish_headlines": [top negative],
          }
        """
        if not headlines:
            return {
                "individual": [],
                "aggregate_label": "neutral",
                "aggregate_score": 0.5,
                "positive_pct": 0.0,
                "negative_pct": 0.0,
                "neutral_pct": 100.0,
                "headline_count": 0,
                "bullish_headlines": [],
                "bearish_headlines": [],
            }

        results = [self.analyze_headline(h) for h in headlines]

        # Aggregate
        pos_scores = [r["scores"]["positive"] for r in results]
        neg_scores = [r["scores"]["negative"] for r in results]
        neu_scores = [r["scores"]["neutral"] for r in results]

        avg_pos = np.mean(pos_scores)
        avg_neg = np.mean(neg_scores)
        avg_neu = np.mean(neu_scores)

        # Determine aggregate direction
        if avg_pos > avg_neg and avg_pos > avg_neu:
            agg_label = "positive"
            agg_score = avg_pos
        elif avg_neg > avg_pos and avg_neg > avg_neu:
            agg_label = "negative"
            agg_score = avg_neg
        else:
            agg_label = "neutral"
            agg_score = avg_neu

        # Label counts
        total = len(results)
        pos_count = sum(1 for r in results if r["label"] == "positive")
        neg_count = sum(1 for r in results if r["label"] == "negative")
        neu_count = sum(1 for r in results if r["label"] == "neutral")

        # Top bullish / bearish headlines
        sorted_pos = sorted(results, key=lambda r: r["scores"]["positive"], reverse=True)
        sorted_neg = sorted(results, key=lambda r: r["scores"]["negative"], reverse=True)

        return {
            "individual": results,
            "aggregate_label": agg_label,
            "aggregate_score": round(float(agg_score), 4),
            "positive_pct": round(pos_count / total * 100, 1),
            "negative_pct": round(neg_count / total * 100, 1),
            "neutral_pct": round(neu_count / total * 100, 1),
            "headline_count": total,
            "bullish_headlines": [
                r for r in sorted_pos[:3] if r["scores"]["positive"] > 0.5
            ],
            "bearish_headlines": [
                r for r in sorted_neg[:3] if r["scores"]["negative"] > 0.5
            ],
        }

    # ── Sector relevance detection ────────────────────────────────
    @staticmethod
    def _normalize_article(article) -> dict:
        if isinstance(article, str):
            return {
                "headline": article.strip(),
                "summary": "",
                "source": "",
                "published_at": "",
            }

        return {
            "headline": str(article.get("headline") or article.get("title") or "").strip(),
            "summary": str(article.get("summary") or article.get("description") or "").strip(),
            "source": str(article.get("source") or "").strip(),
            "published_at": str(article.get("published_at") or article.get("publishedAt") or "").strip(),
        }

    def classify_sector_relevance(self, headline: str, sector: str, summary: str = "") -> dict:
        """
        Determine if a headline is actually relevant to a specific sector.
        Uses richer keyword + macro context matching over headline and summary.
        
        Returns:
          {
            "is_relevant": bool,
            "relevance_type": "direct" | "indirect" | "none",
            "is_geopolitical": bool,
            "confidence": float,
          }
        """
        text_lower = f"{headline} {summary}".lower().strip()
        sector_kws = SECTOR_RELEVANCE.get(sector, [])
        macro_kws = SECTOR_MACRO_CONTEXT.get(sector, [])
        company_aliases = SECTOR_COMPANY_ALIASES.get(sector, [])

        # Direct keyword match
        direct_matches = keyword_matches(text_lower, sector_kws)
        macro_matches = keyword_matches(text_lower, macro_kws)
        regulatory_matches = keyword_matches(text_lower, REGULATORY_KEYWORDS)
        geo_matches = keyword_matches(text_lower, GEO_KEYWORDS)
        company_mentions = keyword_matches(text_lower, company_aliases)
        is_geopolitical = bool(geo_matches)

        if direct_matches or company_mentions:
            confidence = min(
                0.95,
                0.65 + len(direct_matches) * 0.08 + len(macro_matches) * 0.03 + len(company_mentions) * 0.12
            )
            return {
                "is_relevant": True,
                "relevance_type": "direct",
                "is_geopolitical": is_geopolitical,
                "confidence": round(confidence, 2),
                "matched_keywords": direct_matches,
                "company_mentions": company_mentions,
                "macro_matches": macro_matches,
                "regulatory_matches": regulatory_matches,
                "geo_matches": geo_matches,
            }

        contextual_hits = len(macro_matches) + len(regulatory_matches)
        if contextual_hits >= 2:
            confidence = min(
                0.9,
                0.52 + len(macro_matches) * 0.08 + len(regulatory_matches) * 0.06
            )
            return {
                "is_relevant": True,
                "relevance_type": "indirect",
                "is_geopolitical": is_geopolitical,
                "confidence": round(confidence, 2),
                "matched_keywords": [],
                "company_mentions": company_mentions,
                "macro_matches": macro_matches,
                "regulatory_matches": regulatory_matches,
                "geo_matches": geo_matches,
            }

        if is_geopolitical and sector in GEO_SENSITIVE_SECTORS and (macro_matches or regulatory_matches):
            return {
                "is_relevant": True,
                "relevance_type": "indirect",
                "is_geopolitical": True,
                "confidence": 0.55,
                "matched_keywords": [],
                "company_mentions": company_mentions,
                "macro_matches": macro_matches,
                "regulatory_matches": regulatory_matches,
                "geo_matches": geo_matches,
            }

        return {
            "is_relevant": False,
            "relevance_type": "none",
            "is_geopolitical": False,
            "confidence": 0.1,
            "matched_keywords": [],
            "company_mentions": company_mentions,
            "macro_matches": macro_matches,
            "regulatory_matches": regulatory_matches,
            "geo_matches": geo_matches,
        }

    def analyze_sector_headlines(
        self, headlines: list, sector: str
    ) -> dict:
        """
        Filter article/headline inputs relevant to a sector, then run sentiment on them.
        Returns combined relevance + sentiment analysis.
        """
        relevant = []
        irrelevant = []

        for item in headlines:
            article = self._normalize_article(item)
            if not article["headline"]:
                continue

            rel = self.classify_sector_relevance(
                article["headline"], sector, article["summary"]
            )
            passes_strict_filter = (
                rel["confidence"] >= STRICT_RELEVANCE_THRESHOLD or
                bool(rel.get("company_mentions"))
            )

            if rel["is_relevant"] and passes_strict_filter:
                text = article["headline"]
                if article["summary"]:
                    text = f"{article['headline']}. {article['summary']}"
                sent = self.analyze_headline(text)
                time_horizon = article.get("time_horizon", "near_term")
                relevant.append({**sent, **article, "relevance": rel, "time_horizon": time_horizon})
            else:
                irrelevant.append({**article, "relevance": rel})

        # Aggregate only relevant headlines
        if relevant:
            pos_scores = [r["scores"]["positive"] for r in relevant]
            neg_scores = [r["scores"]["negative"] for r in relevant]
            neu_scores = [r["scores"]["neutral"] for r in relevant]

            avg_pos = np.mean(pos_scores)
            avg_neg = np.mean(neg_scores)
            avg_neu = np.mean(neu_scores)

            if avg_pos > avg_neg + 0.1:
                sector_sentiment = "positive"
                sector_score = float(avg_pos)
            elif avg_neg > avg_pos + 0.1:
                sector_sentiment = "negative"
                sector_score = float(avg_neg)
            else:
                sector_sentiment = "neutral"
                sector_score = float(avg_neu)
        else:
            sector_sentiment = "neutral"
            sector_score = 0.5

        total_relevant = len(relevant)
        if total_relevant:
            pos_count = sum(1 for r in relevant if r["label"] == "positive")
            neg_count = sum(1 for r in relevant if r["label"] == "negative")
            neu_count = sum(1 for r in relevant if r["label"] == "neutral")
        else:
            pos_count = neg_count = 0
            neu_count = 1

        geo_count = sum(
            1 for r in relevant if r.get("relevance", {}).get("is_geopolitical")
        )
        time_horizon_counts = {"immediate": 0, "near_term": 0, "medium_term": 0}
        for article in relevant:
            horizon = article.get("time_horizon", "near_term")
            if horizon not in time_horizon_counts:
                horizon = "near_term"
            time_horizon_counts[horizon] += 1

        dominant_time_horizon = max(
            time_horizon_counts,
            key=lambda key: time_horizon_counts[key],
        ) if relevant else "near_term"

        return {
            "sector": sector,
            "sector_sentiment": sector_sentiment,
            "sector_score": round(sector_score, 4),
            "relevant_count": total_relevant,
            "total_count": len(headlines),
            "geo_headlines": geo_count,
            "positive_pct": round((pos_count / total_relevant) * 100, 1) if total_relevant else 0.0,
            "negative_pct": round((neg_count / total_relevant) * 100, 1) if total_relevant else 0.0,
            "neutral_pct": round((neu_count / total_relevant) * 100, 1) if total_relevant else 100.0,
            "dominant_time_horizon": dominant_time_horizon,
            "relevant_headlines": relevant[:8],
            "irrelevant_count": len(irrelevant),
        }
