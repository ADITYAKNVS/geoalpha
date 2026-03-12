from analysis_writer import (
    render_sector_report,
    render_stock_report,
    validate_sector_report,
    validate_stock_report,
)


def make_sector_payload():
    return {
        "sector": "Metals",
        "nifty_label": "📉 -0.42%",
        "signal": "🟡 Hold",
        "confidence": "61%",
        "move_type": "MIXED",
        "evidence_state": "SUFFICIENT_EVIDENCE",
        "primary_driver": "Likely immediate sector sympathy: China auto sales slowdown",
        "secondary_driver": "Likely near-term commodity: LME metals softened",
        "technical_confirmation": "MA trend BULLISH (spread +1.20%), volume NORMAL (1.0x)",
        "pressure_context": "normal volume context (1.0x volume)",
        "causality_note": "The evidence suggests a likely immediate driver, but it should be framed as correlation rather than proof.",
        "key_risk": "Breadth is split across major stocks; follow-through risk is higher.",
        "peer_snapshot": "Tata Steel: -0.13%, Hindalco: +0.26%",
        "sentiment_summary": "Neutral (52%), agreement PARTIAL, breadth SPLIT_BREADTH",
        "relative_strength_note": "-0.42% today vs peer average -0.18% (relative strength -0.24%).",
        "evidence_lines": ["sector sympathy (immediate, score 0.74): China auto sales slowdown"],
        "pick_lines": ["Top pick Hindalco: Positive technical structure"],
        "macro_critical": ["china"],
        "macro_irrelevant": ["nasdaq"],
        "taxonomy": ["sector sympathy", "commodity"],
        "subsector_notes": ["Ferrous: BEARISH (-0.22)", "Non-ferrous: BULLISH (+0.15)"],
        "news_bucket_lines": ["global macro: China auto sales slowdown"],
    }


def test_render_and_validate_sector_report():
    payload = make_sector_payload()
    report = render_sector_report([payload], deep_dive=True)
    valid, issues = validate_sector_report(report, [payload], deep_dive=True)
    assert valid, issues
    assert "### Metals |" in report
    assert "Drivers:" in report
    assert "Risk:" in report


def test_validate_sector_report_rejects_forbidden_wording():
    payload = make_sector_payload()
    report = "### Metals | Nifty: test | Signal: Hold\nDrivers: caused by institutional selling\nTechnicals: test\nRisk: test"
    valid, issues = validate_sector_report(report, [payload], deep_dive=False)
    assert not valid
    assert issues


def test_render_and_validate_stock_report():
    payload = {
        "stock_name": "Tata Steel",
        "ticker": "TATASTEEL.NS",
        "sector": "Metals",
        "move_type": "TECHNICAL_ONLY",
        "evidence_state": "INSUFFICIENT_EVIDENCE",
        "daily_change": -0.13,
        "primary_driver": "Technical-only move: gap and breakout context dominate.",
        "secondary_driver": "Sector context is secondary.",
        "causality_note": "No high-confidence catalyst survived the filter.",
        "technical_confirmation": "Gap -0.40% (gap_down), INSIDE_RANGE vs 20d range",
        "pressure_context": "price-volume context suggests normal volume context (1.0x volume)",
        "relative_strength_vs_sector": 0.12,
        "relative_strength_vs_nifty": -0.08,
        "peer_snapshot": "Hindalco, JSW Steel",
        "evidence_lines": [],
        "key_risk": "No strong stock-specific news was retrieved; treat this as a flow/technical move.",
    }
    report = render_stock_report(payload)
    valid, issues = validate_stock_report(report, payload)
    assert valid, issues
