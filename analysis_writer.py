"""
Compact writer prompts plus deterministic report rendering/validation.
The LLM polishes structured payloads; Python owns report structure and facts.
"""

from __future__ import annotations

import re


FORBIDDEN_PATTERNS = [
    r"\bcaused by\b",
    r"\bbecause of\b",
    r"\binstitutional buying\b",
    r"\binstitutional selling\b",
]


def render_market_context(context: dict) -> str:
    return (
        f"Market context: crude {context['oil']}, gold {context['gold']}, USD/INR {context['usd_inr']}, "
        f"US10Y {context['bond_10y']}, India G-Sec {context['india_gsec']}, Nasdaq {context['nasdaq']}. "
        f"Global news sentiment: {context['global_sentiment_label']} ({context['global_sentiment_score']})."
    )


def _join_or_default(values: list[str], default: str) -> str:
    cleaned = [value for value in values if value]
    return "; ".join(cleaned) if cleaned else default


def render_sector_report(payloads: list[dict], deep_dive: bool) -> str:
    sections = []
    for payload in payloads:
        heading = f"### {payload['sector']} | Nifty: {payload['nifty_label']} | Signal: {payload['signal']}"
        base_lines = [
            heading,
            f"Move Type: {payload['move_type']} | Evidence State: {payload['evidence_state']} | Confidence: {payload['confidence']}",
            f"Drivers: {payload['primary_driver']}. {payload['secondary_driver']}",
            f"Causality: {payload['causality_note']}",
            f"Technicals: {payload['technical_confirmation']}. Pressure: {payload['pressure_context']}",
            f"Sentiment: {payload['sentiment_summary']}",
            f"Peers: {payload['peer_snapshot']}",
            f"Relative Strength: {payload['relative_strength_note']}",
            f"Evidence: {_join_or_default(payload['evidence_lines'], 'No high-confidence evidence lines survived the filter.')}",
            f"Picks: {_join_or_default(payload['pick_lines'], 'No additional stock pick note.')}",
            f"Risk: {payload['key_risk']}",
        ]

        if deep_dive:
            deep_lines = [
                f"Macro Focus: {_join_or_default(payload['macro_critical'], 'No critical macro factor supplied.')}",
                f"Ignore: {_join_or_default(payload['macro_irrelevant'], 'No explicit irrelevant factor list.')}",
                f"Taxonomy: {_join_or_default(payload['taxonomy'], 'none')}",
                f"Subsector Notes: {_join_or_default(payload['subsector_notes'], 'No subsector split supplied.')}",
                f"News Buckets: {_join_or_default(payload['news_bucket_lines'], 'No bucketed news lines available.')}",
            ]
            base_lines[2:2] = deep_lines

        sections.append("\n".join(base_lines))

    return "\n\n".join(sections)


def render_stock_report(payload: dict) -> str:
    return "\n".join([
        f"### {payload['stock_name']} ({payload['ticker']}) | Signal Context: {payload['move_type']}",
        f"Evidence State: {payload['evidence_state']} | Daily Move: {payload['daily_change']:+.2f}%",
        f"Drivers: {payload['primary_driver']}. {payload['secondary_driver']}",
        f"Causality: {payload['causality_note']}",
        f"Technicals: {payload['technical_confirmation']}. Pressure: {payload['pressure_context']}",
        f"Relative Strength: vs sector {payload['relative_strength_vs_sector']:+.2f}%, vs Nifty {payload['relative_strength_vs_nifty']:+.2f}%",
        f"Peers: {payload['peer_snapshot']}",
        f"Evidence: {_join_or_default(payload['evidence_lines'], 'No high-confidence stock-specific evidence lines survived the filter.')}",
        f"Risk: {payload['key_risk']}",
    ])


def sector_writer_prompts(market_context: dict, deterministic_report: str, deep_dive: bool) -> tuple[str, str]:
    system_prompt = (
        "Rewrite the supplied deterministic Indian market report into clean analyst prose. "
        "Preserve every markdown heading exactly. Do not add sectors, claims, or data. "
        "Keep causal language probabilistic, keep volume language as buying/selling pressure, "
        "and keep insufficient-evidence caveats intact."
    )
    style = (
        "Write 5-8 sentences per sector with clear section flow."
        if deep_dive else
        "Write 3-5 concise sentences per sector."
    )
    user_prompt = (
        f"{render_market_context(market_context)}\n"
        f"{style}\n"
        "Polish the following deterministic report without changing headings or factual meaning:\n\n"
        f"{deterministic_report}"
    )
    return system_prompt, user_prompt


def stock_writer_prompts(stock_payload: dict, deterministic_report: str) -> tuple[str, str]:
    system_prompt = (
        "Rewrite the supplied deterministic stock note into clean analyst prose. "
        "Preserve the heading exactly. Do not add facts. "
        "Use likely-driver wording, not hard causation, and do not convert pressure into institutional-flow claims."
    )
    user_prompt = (
        f"Stock context: {stock_payload['stock_name']} in {stock_payload['sector']}. "
        f"Move type {stock_payload['move_type']}, evidence state {stock_payload['evidence_state']}.\n"
        "Polish the following deterministic note without changing meaning:\n\n"
        f"{deterministic_report}"
    )
    return system_prompt, user_prompt


def validate_sector_report(report: str, payloads: list[dict], deep_dive: bool) -> tuple[bool, list[str]]:
    issues = []
    for payload in payloads:
        heading = f"### {payload['sector']} |"
        if heading not in report:
            issues.append(f"missing heading: {payload['sector']}")

    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, report, flags=re.IGNORECASE):
            issues.append(f"forbidden wording: {pattern}")

    if deep_dive:
        for marker in ("Drivers:", "Technicals:", "Risk:"):
            if report.count(marker) < len(payloads):
                issues.append(f"missing marker count: {marker}")

    return (len(issues) == 0, issues)


def validate_stock_report(report: str, payload: dict) -> tuple[bool, list[str]]:
    issues = []
    heading = f"### {payload['stock_name']} ({payload['ticker']})"
    if heading not in report:
        issues.append("missing stock heading")

    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, report, flags=re.IGNORECASE):
            issues.append(f"forbidden wording: {pattern}")

    for marker in ("Drivers:", "Technicals:", "Risk:"):
        if marker not in report:
            issues.append(f"missing marker: {marker}")

    return (len(issues) == 0, issues)
