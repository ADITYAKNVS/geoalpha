"""
Unit tests for GeoAlpha Signal Combiner v2.1
=============================================
Covers:
  1. Metals scenario (negative sentiment + rising price)
  2. Multi-timeframe momentum scoring
  3. Volume confirmation + NO_DATA handling
  4. Sentiment-price divergence
  5. Low headline count filter
  6. Market breadth modulation
  7. RSI zone detection (via integration)
"""

import pytest
from signal_combiner import HybridSignalCombiner, HybridSignal


@pytest.fixture
def combiner():
    return HybridSignalCombiner()


# ── Helper factories ──────────────────────────────────────────

def make_technical(direction="NEUTRAL", confidence=0.5, score=0.0,
                   consensus="NO_DATA", has_divergence=False,
                   volume_ratio=1.0, volume_anomaly=False, volume_signal="NORMAL",
                   volume_data_valid=True,
                   daily_change_pct=0.0, weekly_change=0.0,
                   ma_signal="NEUTRAL"):
    """Create a mock technical analysis result."""
    return {
        "direction": direction,
        "confidence": confidence,
        "score": score,
        "reasons": [f"RSI 50 — neutral zone", f"MA20 > MA50 — uptrend"],
        "stock_consensus": consensus,
        "has_divergence": has_divergence,
        "index_analysis": {
            "status": "ok",
            "volume": {
                "ratio": volume_ratio,
                "is_anomaly": volume_anomaly,
                "signal": volume_signal,
                "data_valid": volume_data_valid,
            },
            "daily_change": {"change_pct": daily_change_pct},
            "weekly_change": weekly_change,
            "ma": {"signal": ma_signal, "strength": 0.5, "spread_pct": 1.0},
        },
        "stock_analyses": [],
        "subsectors": {},
        "top_picks": [],
        "avoid_picks": [],
        "macro_weights": {},
    }


def make_sentiment(label="neutral", score=0.5, relevant_count=5, negative_pct=0.0, positive_pct=0.0):
    """Create a mock sector sentiment result."""
    if label == "negative" and negative_pct == 0.0:
        negative_pct = score * 100
    elif label == "positive" and positive_pct == 0.0:
        positive_pct = score * 100

    return {
        "sector_sentiment": label,
        "sector_score": score,
        "relevant_count": relevant_count,
        "negative_pct": negative_pct,
        "positive_pct": positive_pct,
    }


# ══════════════════════════════════════════════════
# TEST: The EXACT Metals failure scenario
# ══════════════════════════════════════════════════

class TestMetalsScenario:

    def test_metals_negative_sentiment_rising_price(self, combiner):
        """Negative sentiment + rising price = institutional buying = bullish."""
        technical = make_technical(
            direction="NEUTRAL", confidence=0.12, score=0.06,
            consensus="MOSTLY_BULLISH",
            weekly_change=2.5,
            ma_signal="BULLISH",
        )
        sentiment = make_sentiment(label="negative", score=0.83, relevant_count=4)

        result = combiner.combine_signals(
            technical=technical,
            sentiment=sentiment,
            sector="Metals",
            daily_change_pct=1.26,
        )

        assert "Hold" not in result.signal, (
            f"Metals +1.26% with negative sentiment should NOT be Hold, got {result.signal}"
        )
        assert result.agreement == "DIVERGENCE_BULLISH"

    def test_metals_strong_rise_beats_negative_sentiment(self, combiner):
        """Strong +3% rise should dominate over negative sentiment."""
        technical = make_technical(
            direction="BULLISH", confidence=0.6, score=0.3,
            consensus="ALL_BULLISH",
            weekly_change=5.0,
            ma_signal="BULLISH",
        )
        sentiment = make_sentiment(label="negative", score=0.90, relevant_count=6)

        result = combiner.combine_signals(
            technical=technical,
            sentiment=sentiment,
            sector="Metals",
            daily_change_pct=3.0,
        )

        assert "Invest" in result.signal, (
            f"Strong momentum +3% should produce Invest, got {result.signal}"
        )


# ══════════════════════════════════════════════════
# TEST: Multi-timeframe momentum scoring
# ══════════════════════════════════════════════════

class TestMultiTimeframeMomentum:

    def test_daily_spike_dampened_by_weak_weekly(self, combiner):
        """A +2% daily spike with flat weekly trend should NOT be STRONG_BULLISH."""
        score, label = combiner.compute_momentum_score(2.0, weekly_change_pct=0.0, ma_direction="NEUTRAL")
        # Daily component: 0.85 * 0.4 = 0.34
        # Weekly component: 0.50 * 0.3 = 0.15
        # MA component: 0.50 * 0.3 = 0.15
        # Total ≈ 0.64
        assert label != "STRONG_BULLISH", "Single-day spike should be dampened"
        assert score < 0.80

    def test_strong_across_all_timeframes(self, combiner):
        """When daily, weekly, and MA all bullish → STRONG_BULLISH."""
        score, label = combiner.compute_momentum_score(3.0, weekly_change_pct=5.0, ma_direction="BULLISH")
        assert score > 0.75
        assert label in ("STRONG_BULLISH", "BULLISH")

    def test_bearish_across_all_timeframes(self, combiner):
        """When daily, weekly, and MA all bearish → STRONG_BEARISH or BEARISH."""
        score, label = combiner.compute_momentum_score(-3.0, weekly_change_pct=-5.0, ma_direction="BEARISH")
        assert score < 0.25
        assert "BEARISH" in label

    def test_neutral_daily(self, combiner):
        """Flat day + flat week + neutral MA → NEUTRAL."""
        score, label = combiner.compute_momentum_score(0.0, weekly_change_pct=0.0, ma_direction="NEUTRAL")
        assert 0.45 <= score <= 0.55
        assert label == "NEUTRAL"

    def test_backwards_compatible_single_arg(self, combiner):
        """Old-style single-arg call should still work with defaults."""
        score, label = combiner.compute_momentum_score(1.5)
        assert 0.45 <= score <= 0.75  # Reasonable range

    def test_ma_direction_tilts_momentum(self, combiner):
        """Bullish MA should score above bearish MA when price changes are identical."""
        bullish_score, _ = combiner.compute_momentum_score(
            0.5, weekly_change_pct=1.0, monthly_change_pct=2.0, ma_direction="BULLISH"
        )
        bearish_score, _ = combiner.compute_momentum_score(
            0.5, weekly_change_pct=1.0, monthly_change_pct=2.0, ma_direction="BEARISH"
        )
        assert bullish_score > bearish_score


# ══════════════════════════════════════════════════
# TEST: Volume scoring — especially NO_DATA fix
# ══════════════════════════════════════════════════

class TestVolumeScoring:

    def test_no_data_volume_returns_neutral(self, combiner):
        """CRITICAL FIX: Volume 0.0x (index has no volume) → neutral, NOT institutional buying."""
        score, label = combiner.compute_volume_score(
            {"ratio": 0.0, "is_anomaly": False, "signal": "NO_DATA", "data_valid": False},
            daily_change_pct=1.5,
        )
        assert label == "NO_DATA"
        assert score == 0.50

    def test_high_volume_with_price_up(self, combiner):
        """High volume + price up = institutional buying."""
        score, label = combiner.compute_volume_score(
            {"ratio": 2.0, "is_anomaly": True, "signal": "HIGH", "data_valid": True},
            daily_change_pct=1.5,
        )
        assert label == "INSTITUTIONAL_BUYING"
        assert score > 0.65

    def test_high_volume_with_price_down(self, combiner):
        """High volume + price down = institutional selling."""
        score, label = combiner.compute_volume_score(
            {"ratio": 2.0, "is_anomaly": True, "signal": "HIGH", "data_valid": True},
            daily_change_pct=-1.5,
        )
        assert label == "INSTITUTIONAL_SELLING"
        assert score < 0.35

    def test_normal_volume(self, combiner):
        """Normal volume = neutral."""
        score, label = combiner.compute_volume_score(
            {"ratio": 1.0, "is_anomaly": False, "signal": "NORMAL", "data_valid": True},
            daily_change_pct=0.5,
        )
        assert 0.45 <= score <= 0.60


# ══════════════════════════════════════════════════
# TEST: Low headline count filter
# ══════════════════════════════════════════════════

class TestLowHeadlineFilter:

    def test_zero_headlines_gives_neutral(self, combiner):
        """No headlines → INSUFFICIENT_DATA."""
        score, label = combiner.compute_sentiment_score(
            {"sector_sentiment": "negative", "sector_score": 0.95, "relevant_count": 0},
            daily_change_pct=1.0,
        )
        assert label == "INSUFFICIENT_DATA"
        assert score == 0.50

    def test_one_headline_gives_neutral(self, combiner):
        """1 headline → INSUFFICIENT_DATA."""
        score, label = combiner.compute_sentiment_score(
            {"sector_sentiment": "negative", "sector_score": 0.90, "relevant_count": 1},
            daily_change_pct=1.0,
        )
        assert label == "INSUFFICIENT_DATA"

    def test_three_headlines_still_processes(self, combiner):
        """3 headlines → processes but with dampened influence."""
        score, label = combiner.compute_sentiment_score(
            {"sector_sentiment": "negative", "sector_score": 0.80, "relevant_count": 3},
            daily_change_pct=1.5,
        )
        # Should still detect divergence
        assert "DIVERGENCE" in label or "NEGATIVE" in label or score != 0.50


# ══════════════════════════════════════════════════
# TEST: Sentiment-price divergence
# ══════════════════════════════════════════════════

class TestSentimentDivergence:

    def test_negative_sentiment_rising_price_is_bullish(self, combiner):
        """Negative sentiment + rising price = BULLISH."""
        score, label = combiner.compute_sentiment_score(
            {"sector_sentiment": "negative", "sector_score": 0.83, "relevant_count": 5, "negative_pct": 83.0, "positive_pct": 10.0},
            daily_change_pct=1.26,
        )
        assert label == "DIVERGENCE_BULLISH"
        assert score > 0.60

    def test_positive_sentiment_falling_price_is_bearish(self, combiner):
        """Positive sentiment + falling price = retail trap."""
        score, label = combiner.compute_sentiment_score(
            {"sector_sentiment": "positive", "sector_score": 0.75, "relevant_count": 5, "negative_pct": 10.0, "positive_pct": 75.0},
            daily_change_pct=-1.0,
        )
        assert label == "DIVERGENCE_BEARISH"
        assert score < 0.40


# ══════════════════════════════════════════════════
# TEST: Market breadth
# ══════════════════════════════════════════════════

class TestMarketBreadth:

    def test_all_bullish_boosts(self, combiner):
        mult, label = combiner.compute_breadth_modifier("ALL_BULLISH", False)
        assert mult > 1.0

    def test_split_reduces(self, combiner):
        mult, label = combiner.compute_breadth_modifier("SPLIT", True)
        assert mult < 1.0

    def test_neutral(self, combiner):
        mult, label = combiner.compute_breadth_modifier("NO_DATA", False)
        assert mult == 1.0


# ══════════════════════════════════════════════════
# TEST: Signal output ranges
# ══════════════════════════════════════════════════

class TestSignalOutput:

    def test_strong_bullish_scenario(self, combiner):
        """All signals aligned bullish → Strong Invest."""
        technical = make_technical(
            direction="BULLISH", confidence=0.8, score=0.5,
            consensus="ALL_BULLISH",
            volume_ratio=2.5, volume_anomaly=True, volume_signal="EXTREME",
            daily_change_pct=3.0,
            weekly_change=6.0,
            ma_signal="BULLISH",
        )
        sentiment = make_sentiment(label="positive", score=0.80, relevant_count=8)

        result = combiner.combine_signals(
            technical=technical,
            sentiment=sentiment,
            sector="IT",
            daily_change_pct=3.0,
        )

        assert "Invest" in result.signal
        assert result.confidence > 0.6

    def test_strong_bearish_scenario(self, combiner):
        """All signals aligned bearish → Avoid."""
        technical = make_technical(
            direction="BEARISH", confidence=0.7, score=-0.4,
            consensus="ALL_BEARISH",
            weekly_change=-4.0,
            ma_signal="BEARISH",
        )
        sentiment = make_sentiment(label="negative", score=0.85, relevant_count=6)

        result = combiner.combine_signals(
            technical=technical,
            sentiment=sentiment,
            sector="Banking",
            daily_change_pct=-2.5,
        )

        assert "Avoid" in result.signal

    def test_no_forced_hold_on_small_move(self, combiner):
        """Old system forced Hold for <1% moves. New system allows signals."""
        technical = make_technical(
            direction="BULLISH", confidence=0.8, score=0.5,
            consensus="ALL_BULLISH",
            volume_ratio=2.0, volume_anomaly=True, volume_signal="HIGH",
            daily_change_pct=0.5,
            weekly_change=3.0,
            ma_signal="BULLISH",
        )
        sentiment = make_sentiment(label="positive", score=0.70, relevant_count=5)

        result = combiner.combine_signals(
            technical=technical,
            sentiment=sentiment,
            sector="IT",
            daily_change_pct=0.5,
        )

        # With strong technicals + bullish weekly trend, should not force Hold
        assert result.signal != "🟡 Hold" or result.confidence > 0.3


# ══════════════════════════════════════════════════
# TEST: Fear & Greed includes momentum
# ══════════════════════════════════════════════════

class TestFearGreed:

    def test_includes_momentum_component(self, combiner):
        signals = [
            HybridSignal(
                sector="IT", signal="🟢 Invest", confidence=0.7,
                technical_direction="BULLISH", technical_confidence=0.6,
                sentiment_label="positive", sentiment_score=0.7,
                agreement="ALIGNED", momentum_score=0.7, volume_score=0.6,
            ),
        ]
        global_sentiment = {
            "aggregate_label": "positive",
            "aggregate_score": 0.65,
        }
        result = combiner.compute_fear_greed(signals, global_sentiment)

        assert "momentum" in result["components"]
        assert result["components"]["momentum"] > 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
