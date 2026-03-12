"""
Utilities for evaluating explanation quality on captured benchmark cases.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


LABEL_FIELDS = [
    "primary_driver_correct",
    "article_relevance_correct",
    "technical_explanation_correct",
    "hallucination",
    "forced_narrative",
    "missed_main_driver",
]


def load_cases(path: str | Path) -> list[dict]:
    file_path = Path(path)
    if not file_path.exists():
        return []

    cases = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def labeled_cases(cases: list[dict]) -> list[dict]:
    labeled = []
    for case in cases:
        labels = case.get("labels", {})
        if any(labels.get(field) is not None for field in LABEL_FIELDS):
            labeled.append(case)
    return labeled


def _ratio(true_count: int, total: int) -> float:
    return round((true_count / total), 4) if total else 0.0


def evaluate_cases(cases: list[dict]) -> dict:
    labeled = labeled_cases(cases)
    totals = {field: 0 for field in LABEL_FIELDS}
    positives = {field: 0 for field in LABEL_FIELDS}

    for case in labeled:
        labels = case.get("labels", {})
        for field in LABEL_FIELDS:
            value = labels.get(field)
            if value is None:
                continue
            totals[field] += 1
            if bool(value):
                positives[field] += 1

    metrics = {
        "case_count": len(cases),
        "labeled_case_count": len(labeled),
        "primary_driver_accuracy": _ratio(positives["primary_driver_correct"], totals["primary_driver_correct"]),
        "article_relevance_accuracy": _ratio(positives["article_relevance_correct"], totals["article_relevance_correct"]),
        "technical_explanation_accuracy": _ratio(positives["technical_explanation_correct"], totals["technical_explanation_correct"]),
        "hallucination_rate": _ratio(positives["hallucination"], totals["hallucination"]),
        "forced_narrative_rate": _ratio(positives["forced_narrative"], totals["forced_narrative"]),
        "missed_main_driver_rate": _ratio(positives["missed_main_driver"], totals["missed_main_driver"]),
    }

    metrics["signal_quality_score"] = round(
        (
            metrics["primary_driver_accuracy"] * 0.35 +
            metrics["article_relevance_accuracy"] * 0.20 +
            metrics["technical_explanation_accuracy"] * 0.25 +
            (1.0 - metrics["hallucination_rate"]) * 0.10 +
            (1.0 - metrics["forced_narrative_rate"]) * 0.05 +
            (1.0 - metrics["missed_main_driver_rate"]) * 0.05
        ),
        4,
    )

    return metrics


def format_report(metrics: dict) -> str:
    return "\n".join([
        f"Cases loaded: {metrics['case_count']}",
        f"Labeled cases: {metrics['labeled_case_count']}",
        f"Primary driver accuracy: {metrics['primary_driver_accuracy']:.1%}",
        f"Article relevance accuracy: {metrics['article_relevance_accuracy']:.1%}",
        f"Technical explanation accuracy: {metrics['technical_explanation_accuracy']:.1%}",
        f"Hallucination rate: {metrics['hallucination_rate']:.1%}",
        f"Forced narrative rate: {metrics['forced_narrative_rate']:.1%}",
        f"Missed main driver rate: {metrics['missed_main_driver_rate']:.1%}",
        f"Signal quality score: {metrics['signal_quality_score']:.1%}",
    ])


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else Path("evaluation_cases.jsonl")
    metrics = evaluate_cases(load_cases(path))
    print(format_report(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
