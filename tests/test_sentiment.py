"""Offline tests with made-up posts (no internet). Run: pytest -q"""

from datetime import datetime, timezone

from sentiment import Post, Scorer, aggregate, is_relevant, report

NOW = datetime.now(timezone.utc)
S = Scorer(force_builtin=True)   # tests use the built-in word list so they don't need VADER installed


def P(text, source="Reddit r/stocks", label=None, likes=0):
    return Post(source, text, NOW, engagement=likes, self_label=label)


def test_scorer_direction():
    assert S.score("AAPL to the moon, so bullish, great quarter") > 0.3
    assert S.score("total crash, bagholder city, terrible guidance") < -0.3
    assert abs(S.score("Apple reports earnings Thursday")) < 0.05


def test_negation_flips():
    assert S.score("not bullish") < 0


def test_relevance_filter():
    assert is_relevant("loading up on $AAPL", "AAPL", "Apple Inc.")
    assert is_relevant("Apple just beat estimates", "AAPL", "Apple Inc.")
    assert not is_relevant("bought a banana", "AAPL", "Apple Inc.")
    # short tickers need a cashtag so ordinary words don't count
    assert not is_relevant("turn it ON and off", "ON", "ON Semiconductor Corporation")
    assert is_relevant("$ON looks cheap", "ON", "ON Semiconductor Corporation")


def test_aggregate_positive_crowd():
    posts = [P("so bullish, great growth") for _ in range(30)] + [P("terrible, sell") for _ in range(5)]
    r = aggregate("AAPL", "Apple", posts, 7, S, {})
    assert r.overall > 10 and "positive" in r.label


def test_stocktwits_self_labels_count():
    posts = [P("earnings tomorrow", "StockTwits", "Bearish") for _ in range(20)]
    r = aggregate("X", "X Co", posts, 7, S, {})
    assert r.overall < -10
    st = next(s for s in r.sources if s.source == "StockTwits")
    assert st.bull_bear == (0, 20)


def test_disagreement_and_errors_are_reported():
    posts = [P("great, bullish")] * 15 + [P("crash, bearish dump", "News headlines")] * 15
    r = aggregate("X", "X Co", posts, 7, S, {"StockTwits": "HTTPError: 403"})
    text = report(r)
    assert "disagree" in text and "unavailable" in text


def test_one_viral_post_cannot_dominate():
    posts = [P("terrible, sell") for _ in range(20)] + [P("amazing moon", likes=100000)]
    assert aggregate("X", "X", posts, 7, S, {}).overall < 0
