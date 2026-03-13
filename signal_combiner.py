"""
GeoAlpha Hybrid Signal Combiner v2.0 — Momentum-First Architecture
====================================================================
Price momentum is the PRIMARY signal. Technical indicators confirm.
Volume validates. Sentiment is a secondary filter.

Signal Hierarchy (by weight):
  1. Price Momentum (40%) — daily change drives direction
  2. Technical Indicators (25%) — RSI, MA crossovers confirm
  3. Volume Confirmation (20%) — institutional flow validation
  4. Sentiment (15%) — FinBERT NLP as a secondary filter

Key Design Principles:
  - LLM never decides signals, only explains them
  - Negative sentiment + rising price = bullish divergence / absorption
  - Stock picks must pass a momentum gate
  - Market breadth modulates confidence
"""

from dataclasses import dataclass, field


@dataclass
class HybridSignal:
    """The final output of the hybrid system."""
    sector: str
    signal: str            # "🟢 Strong Invest" / "🟡 Hold" / "🔴 Strong Avoid"
    confidence: float      # 0.0 – 1.0
    technical_direction: str
    technical_confidence: float
    sentiment_label: str
    sentiment_score: float
    agreement: str         # "ALIGNED" / "CONFLICT" / "PARTIAL" / "DIVERGENCE_BULLISH" / "DIVERGENCE_BEARISH"
    reasoning: list = field(default_factory=list)
    crash_warning: bool = False
    rally_alert: bool = False
    technical_details: dict = field(default_factory=dict)
    sentiment_details: dict = field(default_factory=dict)
    # v2.0 additions
    momentum_score: float = 0.5
    volume_score: float = 0.5
    breadth: str = "UNKNOWN"


class HybridSignalCombiner:
    """
    Momentum-first signal combiner.

    Weighting:
      Final Score = 0.40 × Momentum + 0.25 × Technical + 0.20 × Volume + 0.15 × Sentiment

    Score-to-Signal mapping:
      > 0.65  → 🟢 Strong Invest
      > 0.55  → 🟢 Invest
      > 0.45  → 🟡 Hold
      > 0.35  → 🔴 Avoid
      ≤ 0.35  → 🔴 Strong Avoid
    """

    # ── Momentum Scoring ──────────────────────────────────
    @staticmethod
    def _daily_to_score(change_pct: float) -> float:
        """Convert a single % change to 0.0–1.0 score."""
        if change_pct >= 3.0:   return 0.95
        elif change_pct >= 2.0: return 0.85
        elif change_pct >= 1.0: return 0.70
        elif change_pct >= 0.5: return 0.58
        elif change_pct >= -0.5: return 0.50
        elif change_pct >= -1.0: return 0.42
        elif change_pct >= -2.0: return 0.30
        elif change_pct >= -3.0: return 0.15
        else: return 0.05

    @staticmethod
    def compute_momentum_score(
        daily_change_pct: float,
        weekly_change_pct: float = 0.0,
        monthly_change_pct: float = 0.0,
        ma_direction: str = "NEUTRAL",
    ) -> tuple[float, str]:
        """
        v2.8: Multi-timeframe momentum scoring.
        Blends daily (25%) + weekly (35%) + monthly (25%) + MA trend bias (15%).
        Reduces false signals from single-day spikes while allowing MA direction
        to tilt the score instead of being ignored.

        Returns: (score, label)
        """
        # Daily component (25%)
        daily_score = HybridSignalCombiner._daily_to_score(daily_change_pct)

        # Weekly component (5-day change) (35%)
        weekly_score = HybridSignalCombiner._daily_to_score(weekly_change_pct / 2.0)
        # Divide by 2 because weekly moves are typically 2x daily

        # Monthly component (20-day change) (25%)
        # Divide by 4 because monthly moves are typically 4x daily
        monthly_score = HybridSignalCombiner._daily_to_score(monthly_change_pct / 4.0)

        if ma_direction == "BULLISH":
            ma_score = 0.75
        elif ma_direction == "BEARISH":
            ma_score = 0.20
        else:
            ma_score = 0.50

        # Blend: 25% daily + 35% weekly + 25% monthly + 15% MA bias
        score = (
            daily_score * 0.25 +
            weekly_score * 0.35 +
            monthly_score * 0.25 +
            ma_score * 0.15
        )
        score = max(0.0, min(1.0, score))

        # Label
        if score >= 0.80: label = "STRONG_BULLISH"
        elif score >= 0.65: label = "BULLISH"
        elif score >= 0.55: label = "MODERATE_BULLISH"
        elif score >= 0.52: label = "LEAN_BULLISH"
        elif score >= 0.48: label = "NEUTRAL"
        elif score >= 0.42: label = "LEAN_BEARISH"
        elif score >= 0.35: label = "MODERATE_BEARISH"
        elif score >= 0.20: label = "BEARISH"
        else: label = "STRONG_BEARISH"

        return (round(score, 3), label)
    # ── Volume Scoring ──────────────────────────────────────
    @staticmethod
    def compute_volume_score(
        volume_data: dict,
        daily_change_pct: float,
    ) -> tuple[float, str]:
        """
        Convert volume anomaly + price direction into a 0.0–1.0 score.
        v2.1: Handles NO_DATA from sector indices gracefully.

        Returns: (score, label)
        """
        ratio = volume_data.get("ratio", 1.0)
        is_anomaly = volume_data.get("is_anomaly", False)
        vol_signal = volume_data.get("signal", "NORMAL")
        data_valid = volume_data.get("data_valid", True)

        # FIX v2.1: If volume data is missing/invalid, return neutral
        if not data_valid or vol_signal == "NO_DATA":
            return (0.50, "NO_DATA")

        if not is_anomaly:
            # Fix: Volume < 1.0 is weak, > 1.0 is normal
            if ratio < 0.8:
                return (0.45, "VERY_WEAK")
            elif ratio < 1.0:
                return (0.48, "WEAK")
            else:
                return (0.50, "NORMAL")

        # Anomalous volume — direction matters (HIGH/EXTREME/ABOVE_AVERAGE)
        if vol_signal in ("HIGH", "EXTREME", "ABOVE_AVERAGE"):
            if daily_change_pct > 0.5:
                # Limit the max score. A ratio of 1.2x will give a slight bump, 1.5x gives more.
                score = min(0.95, 0.65 + (ratio - 1.2) * 0.15)
                return (score, "INSTITUTIONAL_BUYING")
            elif daily_change_pct < -0.5:
                score = max(0.05, 0.35 - (ratio - 1.2) * 0.15)
                return (score, "INSTITUTIONAL_SELLING")
            else:
                return (0.50, "HIGH_VOL_FLAT")
        elif vol_signal == "VERY_WEAK":
            return (0.45, "VERY_WEAK_PARTICIPATION")
        elif vol_signal == "WEAK":
            return (0.48, "WEAK_PARTICIPATION")

        return (0.50, "NORMAL")

    # ── Sentiment Scoring ─────────────────────────────────────
    @staticmethod
    def compute_sentiment_score(
        sentiment: dict,
        daily_change_pct: float,
    ) -> tuple[float, str]:
        """
        Convert FinBERT sentiment into a 0.0–1.0 score.
        CRITICAL: Sentiment-price DIVERGENCE is a signal, not noise.

        - Negative sentiment + rising price → BULLISH divergence / absorption
        - Positive sentiment + falling price → BEARISH divergence / distribution risk
        - Aligned sentiment → confirms direction

        Returns: (score, label)
        """
        raw_label = sentiment.get("sector_sentiment", "neutral")
        raw_score = sentiment.get("sector_score", 0.5)
        relevant_count = sentiment.get("relevant_count", 0)

        # Reclassify weak sentiment
        if raw_label == "positive" and raw_score < 0.40:
            label = "neutral"
        elif raw_label == "negative" and raw_score < 0.40:
            label = "neutral"
        else:
            label = raw_label

        # Low relevance count → drastically reduce sentiment influence
        # v2.1: <5 headlines is not statistically meaningful
        if relevant_count < 2:
            return (0.50, "INSUFFICIENT_DATA")
        elif relevant_count < 5:
            # Partial data: pull heavily toward neutral
            # Still detect divergence but with reduced magnitude
            pass  # continue but scores below will be dampened

        # ── Sentiment-Price Divergence Detection ──
        # This is the KEY fix: divergence is a SIGNAL, not noise
        # v2.7: True divergence requires heavily skewed sentiment (>= 60%), not just a 40/40/20 mix.
        neg_pct = sentiment.get("negative_pct", 0.0)
        pos_pct = sentiment.get("positive_pct", 0.0)

        if label == "negative" and daily_change_pct > 0.5 and neg_pct >= 60.0:
            # Market rising despite heavily negative news → bullish divergence / absorption
            divergence_boost = min(0.20, raw_score * 0.20)  # up to +0.20
            score = 0.60 + divergence_boost
            return (score, "DIVERGENCE_BULLISH")

        if label == "positive" and daily_change_pct < -0.5 and pos_pct >= 60.0:
            # Market falling despite heavily positive news → bearish divergence / distribution risk
            divergence_penalty = min(0.20, raw_score * 0.20)
            score = 0.40 - divergence_penalty
            return (score, "DIVERGENCE_BEARISH")

        # ── Aligned sentiment ──
        if label == "positive":
            return (0.55 + raw_score * 0.15, "POSITIVE")
        elif label == "negative":
            return (0.45 - raw_score * 0.15, "NEGATIVE")
        else:
            return (0.50, "NEUTRAL")

    # ── Technical Scoring ─────────────────────────────────────
    @staticmethod
    def compute_technical_score(technical: dict) -> tuple[float, str]:
        """
        Convert technical analysis into a 0.0–1.0 score.
        Uses the pre-computed direction + confidence from guardrails.

        Returns: (score, label)
        """
        # If the guardrails explicitly reported an error / insufficient data,
        # treat technicals as neutral in the composite and surface that fact
        # via the label so the report can say "INSUFFICIENT_DATA" instead of
        # pretending to have a real bullish/bearish read.
        if technical.get("status") == "error":
            return (0.50, "INSUFFICIENT_DATA")

        direction = technical.get("direction", "NEUTRAL")
        raw_score = technical.get("score", 0.0)

        # Map technical score (-1.0 to +1.0) → (0.0 to 1.0)
        # score of +0.5 → 0.75, score of -0.5 → 0.25, score of 0 → 0.50
        normalized = (raw_score + 1.0) / 2.0
        normalized = max(0.0, min(1.0, normalized))

        # If all key fields are missing, also downgrade to INS UFFICIENT_DATA.
        ma = technical.get("index_analysis", {}).get("ma", {}) if "index_analysis" in technical else technical.get("ma", {})
        rsi = technical.get("index_analysis", {}).get("rsi") if "index_analysis" in technical else technical.get("rsi")
        if rsi is None and ma.get("ma20") is None and ma.get("ma50") is None:
            return (0.50, "INSUFFICIENT_DATA")

        return (normalized, direction)

    # ── Market Breadth ────────────────────────────────────────
    @staticmethod
    def compute_breadth_modifier(
        stock_consensus: str,
        has_divergence: bool,
    ) -> tuple[float, str]:
        """
        Market breadth modulates confidence.
        Returns: (multiplier, label)
          multiplier > 1.0 = confidence boost
          multiplier < 1.0 = confidence reduction
        """
        if stock_consensus == "ALL_BULLISH":
            return (1.20, "STRONG_BREADTH")
        elif stock_consensus == "MOSTLY_BULLISH":
            return (1.10, "GOOD_BREADTH")
        elif stock_consensus == "ALL_BEARISH":
            return (1.20, "STRONG_BEARISH_BREADTH")
        elif stock_consensus == "MOSTLY_BEARISH":
            return (1.10, "BEARISH_BREADTH")
        elif stock_consensus == "SPLIT" or has_divergence:
            return (0.85, "SPLIT_BREADTH")
        else:
            return (1.0, "NEUTRAL_BREADTH")

    # ══════════════════════════════════════════════════════════
    # CORE SIGNAL COMBINATION
    # ══════════════════════════════════════════════════════════
    def combine_signals(
        self,
        technical: dict,
        sentiment: dict,
        sector: str,
            daily_change_pct: float = 0.0,
        ) -> HybridSignal:
        """
        Momentum-first signal combination.

        Formula: Final = 0.40×Momentum + 0.25×Technical + 0.20×Volume + 0.15×Sentiment

        Args:
            technical: from TechnicalGuardrails.analyze_sector()
            sentiment: from SentimentEngine.analyze_sector_headlines()
            sector: sector name
            daily_change_pct: Nifty sector % change today
        """
        # Defensive check for None
        if daily_change_pct is None:
            daily_change_pct = 0.0
            
        reasoning = []

        # ── Step 1: Compute component scores ──

        # Get weekly and monthly change from technical data for multi-timeframe momentum
        index_analysis = technical.get("index_analysis") or {}
        weekly_change = index_analysis.get("weekly_change") or 0.0
        monthly_change = index_analysis.get("monthly_change") or 0.0
        
        # If index doesn't have weekly/monthly change, try stock average
        if weekly_change == 0.0:
            stock_analyses = technical.get("stock_analyses", [])
            if stock_analyses:
                weekly_changes = [s.get("weekly_change", 0.0) for s in stock_analyses if s.get("weekly_change", 0.0) != 0.0]
                if weekly_changes:
                    weekly_change = sum(weekly_changes) / len(weekly_changes)
                
                monthly_changes = [s.get("monthly_change", 0.0) for s in stock_analyses if s.get("monthly_change", 0.0) != 0.0]
                if monthly_changes:
                    monthly_change = sum(monthly_changes) / len(monthly_changes)
                    
        ma_direction = index_analysis.get("ma", {}).get("signal", "NEUTRAL")

        momentum_score, momentum_label = self.compute_momentum_score(
            daily_change_pct, weekly_change, monthly_change, ma_direction
        )

        tech_score, tech_label = self.compute_technical_score(technical)
        tech_conf = technical.get("confidence", 0.3)
        tech_reasons = technical.get("reasons", [])

        # Get volume data from technical analysis (sector-level)
        volume_data = index_analysis.get("volume", {"ratio": 1.0, "is_anomaly": False, "signal": "NORMAL", "data_valid": True})
        # Sanitize ratio if it exists but is None
        if volume_data.get("ratio") is None:
            volume_data["ratio"] = 1.0
            
        volume_score, volume_label = self.compute_volume_score(volume_data, daily_change_pct)

        sent_score, sent_label = self.compute_sentiment_score(sentiment, daily_change_pct)
        raw_sent_label = sentiment.get("sector_sentiment", "neutral")
        raw_sent_score = sentiment.get("sector_score", 0.5)
        relevant_count = sentiment.get("relevant_count", 0)

        # ── Step 2: Dynamic Weighting ──
        # If <5 relevant headlines, sentiment is noisy. Reduce weight to 5% and give 10% to momentum.
        if relevant_count < 5:
            w_mom = 0.50
            w_tech = 0.25
            w_vol = 0.20
            w_sent = 0.05
        else:
            w_mom = 0.40
            w_tech = 0.25
            w_vol = 0.20
            w_sent = 0.15

        final_score = (
            momentum_score * w_mom +
            tech_score * w_tech +
            volume_score * w_vol +
            sent_score * w_sent
        )

        # ── Step 3: Market breadth modifier ──
        stock_consensus = technical.get("stock_consensus", "NO_DATA")
        has_divergence = technical.get("has_divergence", False)
        breadth_mult, breadth_label = self.compute_breadth_modifier(
            stock_consensus, has_divergence
        )

        # Breadth modulates confidence, not direction
        # (shifts final_score slightly toward/away from 0.5)
        if breadth_mult > 1.0:
            # Breadth confirms direction → push away from 0.5
            if final_score > 0.5:
                final_score = 0.5 + (final_score - 0.5) * breadth_mult
            elif final_score < 0.5:
                final_score = 0.5 - (0.5 - final_score) * breadth_mult
        elif breadth_mult < 1.0:
            # Breadth conflicts → pull toward 0.5
            final_score = 0.5 + (final_score - 0.5) * breadth_mult

        final_score = max(0.0, min(1.0, final_score))

        # ── Step 4: Score → Signal mapping ──
        # v2.6: Less conservative. >0.55 triggers Invest.
        if final_score >= 0.70:
            signal = "🟢 Strong Invest"
        elif final_score >= 0.55:  # Was 0.58
            signal = "🟢 Invest"
        elif final_score >= 0.45:  # Was 0.42
            signal = "🟡 Hold"
        elif final_score >= 0.30:
            signal = "🔴 Avoid"
        else:
            signal = "🔴 Strong Avoid"

        # ── Step 5: Determine agreement type ──
        if sent_label == "DIVERGENCE_BULLISH":
            agreement = "DIVERGENCE_BULLISH"
        elif sent_label == "DIVERGENCE_BEARISH":
            agreement = "DIVERGENCE_BEARISH"
        elif (momentum_label in ("STRONG_BULLISH", "BULLISH", "MODERATE_BULLISH") and
              tech_label == "BULLISH"):
            agreement = "ALIGNED"
        elif (momentum_label in ("STRONG_BEARISH", "BEARISH", "MODERATE_BEARISH") and
              tech_label == "BEARISH"):
            agreement = "ALIGNED"
        elif (momentum_label in ("STRONG_BULLISH", "BULLISH", "MODERATE_BULLISH") and
              tech_label == "BEARISH"):
            agreement = "CONFLICT"
        elif (momentum_label in ("STRONG_BEARISH", "BEARISH", "MODERATE_BEARISH") and
              tech_label == "BULLISH"):
            agreement = "CONFLICT"
        else:
            agreement = "PARTIAL"

        # ── Step 6: Crash / Rally detection ──
        crash_warning = (
            final_score < 0.25 and
            daily_change_pct < -2.0 and
            volume_label == "INSTITUTIONAL_SELLING"
        )
        rally_alert = (
            final_score > 0.75 and
            daily_change_pct > 2.0 and
            volume_label == "INSTITUTIONAL_BUYING"
        )

        if crash_warning:
            signal = "🔴 Strong Avoid"
            final_score = max(final_score, 0.10)  # keep it low
        if rally_alert:
            signal = "🟢 Strong Invest"
            final_score = min(final_score, 0.95)

        # ── Step 7: Build reasoning chain ──
        # Momentum (primary)
        reasoning.append(
            f"📈 Momentum: {momentum_label} (score {momentum_score:.2f}, "
            f"daily change {daily_change_pct:+.2f}%) — WEIGHT: 40%"
        )

        # Technical
        reasoning.append(
            f"⚙️ Technical: {tech_label} (score {tech_score:.2f}, "
            f"confidence {tech_conf:.0%}) — WEIGHT: 25%"
        )
        for r in tech_reasons[:5]:
            reasoning.append(f"   → {r}")

        # Volume
        reasoning.append(
            f"📊 Volume: {volume_label} (score {volume_score:.2f}, "
            f"ratio {volume_data.get('ratio', 1.0):.1f}x) — WEIGHT: 20%"
        )

        # Sentiment
        reasoning.append(
            f"🧠 Sentiment: {sent_label} (score {sent_score:.2f}, "
            f"raw: {raw_sent_label} {raw_sent_score:.0%}) — WEIGHT: 15%"
        )

        # Divergence explanation
        if sent_label == "DIVERGENCE_BULLISH":
            reasoning.append(
                f"🔥 BULLISH DIVERGENCE: Price rising {daily_change_pct:+.2f}% "
                f"despite negative sentiment ({raw_sent_score:.0%}). "
                f"This suggests absorption/resilience, not proven institutional buying."
            )
        elif sent_label == "DIVERGENCE_BEARISH":
            reasoning.append(
                f"⚠️ BEARISH DIVERGENCE: Price falling {daily_change_pct:+.2f}% "
                f"despite positive sentiment ({raw_sent_score:.0%}). "
                f"This suggests selling pressure or distribution risk, not confirmed distribution."
            )

        # Breadth
        if breadth_label != "NEUTRAL_BREADTH":
            reasoning.append(
                f"🏭 Breadth: {breadth_label} (consensus: {stock_consensus})"
            )

        # Agreement
        if agreement == "ALIGNED":
            reasoning.append("✅ Momentum + Technicals AGREE — high conviction")
        elif agreement == "CONFLICT":
            reasoning.append("⚠️ Momentum vs Technicals CONFLICT — reduced conviction")

        # Crash/Rally
        if crash_warning:
            reasoning.append(
                f"🚨 CRASH WARNING: Large drop + elevated selling pressure + bearish technicals"
            )
        if rally_alert:
            reasoning.append(
                f"🚀 RALLY ALERT: Large rally + elevated buying pressure + bullish technicals"
            )

        # Composite score
        reasoning.append(
            f"📐 Composite Score: {final_score:.2f} "
            f"(M:{momentum_score:.2f}×{w_mom:.2f} + T:{tech_score:.2f}×{w_tech:.2f} + "
            f"V:{volume_score:.2f}×{w_vol:.2f} + S:{sent_score:.2f}×{w_sent:.2f})"
        )

        # Realign confidence strictly with the mathematical composite score
        # so Confidence % exactly matches the calculated Final Score %.
        if final_score >= 0.5:
            confidence = final_score
        else:
            confidence = 1.0 - final_score
        
        confidence = round(confidence, 2)

        # Reclassify sent_label for display purposes
        display_sent_label = raw_sent_label
        if raw_sent_label == "positive" and raw_sent_score < 0.40:
            display_sent_label = "neutral"
        elif raw_sent_label == "negative" and raw_sent_score < 0.40:
            display_sent_label = "neutral"

        return HybridSignal(
            sector=sector,
            signal=signal,
            confidence=confidence,
            technical_direction=tech_label,
            technical_confidence=tech_conf,
            sentiment_label=display_sent_label,
            sentiment_score=raw_sent_score,
            agreement=agreement,
            reasoning=reasoning,
            crash_warning=crash_warning,
            rally_alert=rally_alert,
            technical_details=technical,
            sentiment_details=sentiment,
            momentum_score=momentum_score,
            volume_score=volume_score,
            breadth=breadth_label,
        )

    # ── Fear & Greed Index ────────────────────────────────────
    @staticmethod
    def compute_fear_greed(
        sector_signals: list,
        global_sentiment: dict,
    ) -> dict:
        """
        Compute a market-wide Fear & Greed score (0-100).
        0 = Extreme Fear, 100 = Extreme Greed

        v2.0: Adds momentum component, reduces sentiment weight.
        Components:
          - Sector signal consensus (30%)
          - Momentum average (30%)
          - Average technical score (25%)
          - Global news sentiment (15%)
        """
        if not sector_signals:
            return {"score": 50, "label": "Neutral", "components": {}}

        total = len(sector_signals) or 1

        # Sector consensus
        bullish = sum(1 for s in sector_signals if "Invest" in s.signal)
        bearish = sum(1 for s in sector_signals if "Avoid" in s.signal)
        sector_score = ((bullish - bearish) / total + 1) / 2 * 100

        # Momentum average
        momentum_avg = sum(s.momentum_score for s in sector_signals) / total
        momentum_score = momentum_avg * 100  # 0-100

        # Technical average
        tech_scores = [s.technical_confidence for s in sector_signals]
        tech_dirs = [
            1 if s.technical_direction == "BULLISH"
            else -1 if s.technical_direction == "BEARISH"
            else 0
            for s in sector_signals
        ]
        avg_tech = sum(d * c for d, c in zip(tech_dirs, tech_scores)) / total
        tech_score = (avg_tech + 1) / 2 * 100

        # Sentiment score (reduced weight)
        sent_score_raw = global_sentiment.get("aggregate_score", 0.5)
        sent_label = global_sentiment.get("aggregate_label", "neutral")
        if sent_label == "negative":
            sent_score = (1 - sent_score_raw) * 100
        elif sent_label == "positive":
            sent_score = sent_score_raw * 100
        else:
            sent_score = 50

        # Weighted composite (momentum-first)
        composite = (
            sector_score * 0.30 +
            momentum_score * 0.30 +
            tech_score * 0.25 +
            sent_score * 0.15
        )
        composite = max(0, min(100, composite))

        # Label
        if composite >= 75:
            label = "Extreme Greed"
        elif composite >= 60:
            label = "Greed"
        elif composite >= 40:
            label = "Neutral"
        elif composite >= 25:
            label = "Fear"
        else:
            label = "Extreme Fear"

        return {
            "score": round(composite),
            "label": label,
            "components": {
                "sector_consensus": round(sector_score),
                "momentum": round(momentum_score),
                "technical_avg": round(tech_score),
                "news_sentiment": round(sent_score),
            },
        }
