from explanation_evaluator import evaluate_cases


def test_evaluate_cases_metrics():
    cases = [
        {
            "case_id": "1",
            "labels": {
                "primary_driver_correct": True,
                "article_relevance_correct": True,
                "technical_explanation_correct": False,
                "hallucination": False,
                "forced_narrative": True,
                "missed_main_driver": False,
            },
        },
        {
            "case_id": "2",
            "labels": {
                "primary_driver_correct": False,
                "article_relevance_correct": True,
                "technical_explanation_correct": True,
                "hallucination": True,
                "forced_narrative": False,
                "missed_main_driver": True,
            },
        },
        {
            "case_id": "3",
            "labels": {
                "primary_driver_correct": None,
                "article_relevance_correct": None,
                "technical_explanation_correct": None,
                "hallucination": None,
                "forced_narrative": None,
                "missed_main_driver": None,
            },
        },
    ]

    metrics = evaluate_cases(cases)

    assert metrics["case_count"] == 3
    assert metrics["labeled_case_count"] == 2
    assert metrics["primary_driver_accuracy"] == 0.5
    assert metrics["article_relevance_accuracy"] == 1.0
    assert metrics["technical_explanation_accuracy"] == 0.5
    assert metrics["hallucination_rate"] == 0.5
    assert metrics["forced_narrative_rate"] == 0.5
    assert metrics["missed_main_driver_rate"] == 0.5
