from sentiment_engine import SentimentEngine


def fake_sentiment(_text):
    return {
        "headline": _text,
        "label": "neutral",
        "score": 0.5,
        "scores": {"positive": 0.2, "negative": 0.2, "neutral": 0.6},
    }


def test_strict_sector_filter_rejects_generic_noise(monkeypatch):
    engine = SentimentEngine()
    monkeypatch.setattr(engine, "analyze_headline", fake_sentiment)

    result = engine.analyze_sector_headlines(
        [
            {"headline": "Sri Lanka court orders 84 Iranian sailors' bodies be handed to Iran embassy"},
            {"headline": "UPERC tariff hearing scheduled next week"},
            {"headline": "JioStar launches a new micro drama platform"},
        ],
        "Metals",
    )

    assert result["relevant_count"] == 0
    assert result["irrelevant_count"] == 3


def test_company_mention_overrides_threshold(monkeypatch):
    engine = SentimentEngine()
    monkeypatch.setattr(engine, "analyze_headline", fake_sentiment)

    result = engine.analyze_sector_headlines(
        [
            {"headline": "Tata Steel board to review expansion roadmap"},
        ],
        "Metals",
    )

    assert result["relevant_count"] == 1
    assert result["relevant_headlines"][0]["relevance"]["company_mentions"] == ["tata steel"]
