from sector_report_utils import (
    build_sector_stock_contribution_lines,
    build_sector_technical_snapshot,
    inject_sector_technical_sections,
)


def make_technical_payload():
    return {
        "sector_daily_change": -4.82,
        "index_analysis": {
            "status": "ok",
            "rsi": 33.4,
            "ma": {
                "ma20": 8421.4,
                "ma50": 8703.9,
                "signal": "BEARISH",
                "spread_pct": -3.24,
                "crossover": "death",
            },
            "volume": {
                "ratio": 1.3,
                "signal": "HIGH",
                "data_valid": True,
                "source": "stock_aggregate",
            },
            "daily_change": {"change_pct": -4.82},
            "weekly_change": -6.11,
            "monthly_change": -9.45,
            "breakout": {
                "state": "BREAKDOWN",
                "distance_pct": 1.2,
                "recent_high": 8710.0,
                "recent_low": 8350.0,
            },
            "gap": {
                "gap_pct": -0.85,
                "direction": "gap_down",
            },
        },
        "stock_analyses": [
            {"stock_name": "Tata Steel", "daily_change": {"change_pct": -5.60}},
            {"stock_name": "Hindalco", "daily_change": {"change_pct": -4.10}},
            {"stock_name": "JSW Steel", "daily_change": {"change_pct": -3.85}},
        ],
    }


def test_build_sector_technical_snapshot_surfaces_mandatory_indicators():
    snapshot = build_sector_technical_snapshot(make_technical_payload())

    assert "RSI 33.4" in snapshot["summary"]
    assert "MA20 8421.4 vs MA50 8703.9" in snapshot["summary"]
    assert "support 8350.0 / resistance 8710.0" in snapshot["summary"]
    assert "volume 1.3x HIGH" in snapshot["summary"]
    assert any("Support / resistance" in line for line in snapshot["lines"])
    assert any("Price structure" in line for line in snapshot["lines"])


def test_build_sector_stock_contribution_lines_ranks_constituents():
    contribution_lines = build_sector_stock_contribution_lines(make_technical_payload())

    assert contribution_lines
    assert contribution_lines[0].startswith("Main drags:")
    assert "Tata Steel -5.60%" in contribution_lines[0]
    assert "Best relative performer: JSW Steel -3.85%" in contribution_lines[1]


def test_inject_sector_technical_sections_adds_one_block_per_sector():
    report = """### Metals | Nifty: -4.82% | Signal: 🔴 Avoid

🧭 MOVE TYPE: TECHNICAL_ONLY

🎯 SIGNAL EXPLANATION: Flow-led weakness.

### Banking | Nifty: -2.44% | Signal: 🟡 Hold

🧭 MOVE TYPE: MIXED

🎯 SIGNAL EXPLANATION: Mixed session."""
    dossiers = {
        "Metals": {
            "technical_indicator_lines": [
                "RSI: 33.4 (weak / near oversold)",
                "Moving averages: MA20 8421.4 vs MA50 8703.9 -> BEARISH trend (spread -3.24%, crossover death)",
            ]
        },
        "Banking": {
            "technical_indicator_lines": [
                "RSI: 48.1 (neutral)",
                "Moving averages: MA20 51234.0 vs MA50 50990.0 -> BULLISH trend (spread +0.48%, crossover golden)",
            ]
        },
    }

    enforced = inject_sector_technical_sections(report, dossiers, deep_dive=True)

    assert enforced.count("📈 TECHNICAL INDICATORS:") == 2
    assert "RSI: 33.4 (weak / near oversold)" in enforced
    assert "RSI: 48.1 (neutral)" in enforced

    enforced_again = inject_sector_technical_sections(enforced, dossiers, deep_dive=True)
    assert enforced_again.count("📈 TECHNICAL INDICATORS:") == 2
