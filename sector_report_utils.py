from __future__ import annotations

import re


def _is_number(value) -> bool:
    return isinstance(value, (int, float))


def _fmt_number(value, decimals: int = 1) -> str:
    if not _is_number(value):
        return "N/A"
    return f"{value:.{decimals}f}"


def classify_rsi_zone(rsi) -> str:
    if not _is_number(rsi):
        return "unavailable"
    if rsi >= 70:
        return "overbought"
    if rsi >= 60:
        return "bullish momentum"
    if rsi <= 30:
        return "oversold"
    if rsi <= 40:
        return "weak / near oversold"
    return "neutral"


def build_sector_technical_snapshot(technical: dict) -> dict:
    # analyze_sector() nests the index-level data inside "index_analysis".
    # If it's None (index ticker failed) or status == "error", treat this as
    # an explicit INS UFFICIENT_DATA case rather than silently fabricating neutral values.
    idx = technical.get("index_analysis") or {}
    if idx.get("status") == "error" or technical.get("status") == "error":
        return {
            "summary": "Technical data: INSUFFICIENT_DATA (daily RSI/MA/volume not reliable for this sector/index).",
            "lines": [
                "RSI: N/A (insufficient data)",
                "Moving averages: MA20 N/A vs MA50 N/A (INSUFFICIENT_DATA)",
                "Support / resistance: N/A (recent range cannot be computed reliably)",
                "Volume: N/A (NO_DATA or unstable history); daily move shown from price only",
                "Price structure: gaps / weekly / monthly structure omitted due to limited history",
            ],
        }

    ma = idx.get("ma", {})
    volume = idx.get("volume", {})
    breakout = idx.get("breakout", {})
    gap = idx.get("gap", {})
    daily = idx.get("daily_change", {})

    rsi = idx.get("rsi")
    ma20 = ma.get("ma20")
    ma50 = ma.get("ma50")
    ma_signal = ma.get("signal", "NEUTRAL")
    ma_spread = ma.get("spread_pct", 0.0)
    volume_ratio = volume.get("ratio", 1.0)
    volume_signal = volume.get("signal", "NORMAL")
    support = breakout.get("recent_low")
    resistance = breakout.get("recent_high")
    breakout_state = breakout.get("state", "INSIDE_RANGE")
    breakout_distance = breakout.get("distance_pct", 0.0)
    gap_pct = gap.get("gap_pct", 0.0)
    gap_direction = gap.get("direction", "flat")
    daily_change = daily.get("change_pct", technical.get("sector_daily_change", 0.0))
    weekly_change = idx.get("weekly_change", 0.0)
    monthly_change = idx.get("monthly_change", 0.0)

    volume_source_suffix = ""
    if volume.get("source") == "stock_aggregate":
        volume_source_suffix = ", stock basket fallback"

    breakout_phrase = f"{breakout_state} ({breakout_distance:+.2f}%)"
    if breakout_state == "BREAKOUT":
        breakout_phrase = f"BREAKOUT ({breakout_distance:+.2f}% above resistance)"
    elif breakout_state == "BREAKDOWN":
        breakout_phrase = f"BREAKDOWN ({breakout_distance:+.2f}% below support)"

    summary = (
        f"RSI {_fmt_number(rsi, 1)} ({classify_rsi_zone(rsi)}), "
        f"MA20 {_fmt_number(ma20, 1)} vs MA50 {_fmt_number(ma50, 1)} ({ma_signal}, spread {ma_spread:+.2f}%), "
        f"support {_fmt_number(support, 1)} / resistance {_fmt_number(resistance, 1)}, "
        f"volume {_fmt_number(volume_ratio, 1)}x {volume_signal}, "
        f"weekly {weekly_change:+.2f}%, monthly {monthly_change:+.2f}%"
    )

    lines = [
        f"RSI: {_fmt_number(rsi, 1)} ({classify_rsi_zone(rsi)})",
        (
            f"Moving averages: MA20 {_fmt_number(ma20, 1)} vs MA50 {_fmt_number(ma50, 1)} "
            f"-> {ma_signal} trend (spread {ma_spread:+.2f}%, crossover {ma.get('crossover', 'N/A')})"
        ),
        (
            f"Support / resistance: 20-day support {_fmt_number(support, 1)}, "
            f"20-day resistance {_fmt_number(resistance, 1)}; state {breakout_phrase}"
        ),
        (
            f"Volume: {_fmt_number(volume_ratio, 1)}x average ({volume_signal}{volume_source_suffix}); "
            f"daily move {daily_change:+.2f}%"
        ),
        (
            f"Price structure: gap {gap_pct:+.2f}% ({gap_direction}), "
            f"weekly {weekly_change:+.2f}%, monthly {monthly_change:+.2f}%"
        ),
    ]

    return {
        "summary": summary,
        "lines": lines,
    }


def build_sector_stock_contribution_lines(technical: dict, limit: int = 3) -> list[str]:
    stock_analyses = technical.get("stock_analyses", [])
    if not stock_analyses:
        return []

    ordered = sorted(
        stock_analyses,
        key=lambda stock: stock.get("daily_change", {}).get("change_pct", 0.0),
    )
    weakest = ordered[0]
    strongest = ordered[-1]

    biggest_drags = ", ".join(
        f"{stock.get('stock_name', stock.get('ticker', ''))} {stock.get('daily_change', {}).get('change_pct', 0.0):+.2f}%"
        for stock in ordered[:limit]
    )
    biggest_lifts = ", ".join(
        f"{stock.get('stock_name', stock.get('ticker', ''))} {stock.get('daily_change', {}).get('change_pct', 0.0):+.2f}%"
        for stock in reversed(ordered[-limit:])
    )

    return [
        f"Main drags: {biggest_drags}",
        (
            f"Best relative performer: {strongest.get('stock_name', strongest.get('ticker', ''))} "
            f"{strongest.get('daily_change', {}).get('change_pct', 0.0):+.2f}% | "
            f"Weakest constituent: {weakest.get('stock_name', weakest.get('ticker', ''))} "
            f"{weakest.get('daily_change', {}).get('change_pct', 0.0):+.2f}%"
        ),
        f"Strongest lifts: {biggest_lifts}",
    ]


def build_sector_technical_markdown(dossier: dict) -> str:
    lines = dossier.get("technical_indicator_lines", [])
    if not lines:
        return ""

    markdown_lines = ["📈 TECHNICAL INDICATORS:"]
    markdown_lines.extend(f"- {line}" for line in lines)
    return "\n".join(markdown_lines)


def inject_sector_technical_sections(report: str, sector_dossiers: dict[str, dict], deep_dive: bool) -> str:
    if not report or not sector_dossiers:
        return report

    sections = [section for section in re.split(r"(?m)(?=^### )", report.strip()) if section.strip()]
    enhanced_sections = []

    for section in sections:
        heading_match = re.match(r"###\s+(.+?)\s+\|", section.strip())
        if not heading_match:
            enhanced_sections.append(section)
            continue

        sector = heading_match.group(1).strip()
        dossier = sector_dossiers.get(sector)
        tech_block = build_sector_technical_markdown(dossier or {})
        if not dossier or not tech_block:
            enhanced_sections.append(section)
            continue

        if "TECHNICAL INDICATORS:" in section.upper():
            enhanced_sections.append(section)
            continue

        if deep_dive and "🎯 SIGNAL EXPLANATION:" in section:
            section = section.replace("🎯 SIGNAL EXPLANATION:", f"{tech_block}\n\n🎯 SIGNAL EXPLANATION:", 1)
        else:
            section = f"{section.rstrip()}\n\n{tech_block}"

        enhanced_sections.append(section)

    return "\n\n".join(enhanced_sections)
