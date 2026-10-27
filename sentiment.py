"""
Social sentiment scrubber: what are people saying about a company right now?

Sources (all free):
  - Reddit      r/stocks, r/investing, r/wallstreetbets, r/StockMarket, r/options (+ any you add)
                Uses Reddit's official API if REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET are set
                (free "script" app at https://www.reddit.com/prefs/apps), otherwise Reddit's public JSON.
  - StockTwits  public symbol stream; users often tag their own posts Bullish / Bearish.
  - News        recent headlines from Yahoo Finance (via yfinance).
X/Twitter isn't included: its API is paid.

Each post is scored -1 (very negative) to +1 (very positive) with VADER, a sentiment model built for
social media, extended with stock slang ("to the moon", "bagholder", "puts"...). If VADER isn't installed,
a smaller built-in word list is used instead.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from statistics import mean
from typing import Optional

import requests

USER_AGENT = "windows:social-sentiment-bot:0.1 (personal research)"
DEFAULT_SUBREDDITS = ["stocks", "investing", "wallstreetbets", "StockMarket", "options"]

# Stock-market slang VADER doesn't know. Values on VADER's -4..+4 scale.
STOCK_LEXICON = {
    "moon": 3.0, "mooning": 3.0, "rocket": 2.5, "bullish": 2.5, "bull": 1.5, "calls": 1.2,
    "undervalued": 2.0, "breakout": 2.0, "beat": 1.5, "beats": 1.5, "upgrade": 2.0, "upgraded": 2.0,
    "rally": 2.0, "rallying": 2.0, "squeeze": 1.0, "buy": 1.2, "buying": 1.2, "long": 1.0, "hodl": 1.5,
    "tendies": 2.0, "green": 1.0, "ath": 1.5, "outperform": 2.0, "strong": 1.5,
    "bearish": -2.5, "bear": -1.5, "puts": -1.2, "overvalued": -2.0, "dump": -2.5, "dumping": -2.5,
    "crash": -3.0, "crashing": -3.0, "bagholder": -2.5, "bagholding": -2.5, "bags": -1.5,
    "downgrade": -2.0, "downgraded": -2.0, "miss": -1.5, "missed": -1.5, "sell": -1.2, "selling": -1.2,
    "short": -1.0, "shorting": -1.5, "red": -1.0, "rugpull": -3.0, "bankrupt": -3.5, "bankruptcy": -3.5,
    "fraud": -3.5, "lawsuit": -2.0, "recall": -2.0, "layoffs": -1.5, "underperform": -2.0, "weak": -1.5,
    "drilling": -2.0, "tanking": -2.5, "tank": -2.0, "plunge": -2.5, "plunging": -2.5, "loss": -1.5,
    "good": 1.5, "great": 2.5, "love": 2.5, "amazing": 3.0, "excellent": 3.0, "profit": 2.0, "growth": 1.5,
    "bad": -2.0, "terrible": -3.0, "awful": -3.0, "hate": -2.5, "worst": -3.0, "scam": -3.0, "risky": -1.0,
}
NEUTRAL_BAND = 0.05   # |score| below this counts as neutral (VADER's standard cutoff)


# ---------------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------------

class Scorer:
    def __init__(self, force_builtin: bool = False):
        self.engine = "built-in word list"
        self._vader = None
        if not force_builtin:
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self._vader = SentimentIntensityAnalyzer()
                self._vader.lexicon.update(STOCK_LEXICON)
                self.engine = "VADER + stock slang"
            except ImportError:
                pass

    def score(self, text: str) -> float:
        if self._vader:
            return float(self._vader.polarity_scores(text)["compound"])
        words = re.findall(r"[a-z']+", text.lower())
        total, negate = 0.0, False
        for w in words:
            if w in ("not", "no", "never", "isn't", "don't", "won't", "can't", "aren't"):
                negate = True
                continue
            v = STOCK_LEXICON.get(w)
            if v is not None:
                total += -v if negate else v
            negate = False
        # same squashing VADER uses, so scores land in -1..+1
        return total / ((total * total + 15) ** 0.5) if total else 0.0


# ---------------------------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------------------------

@dataclass
class Post:
    source: str
    text: str
    created: datetime
    url: str = ""
    engagement: int = 0                 # upvotes / likes, used as a light weight
    self_label: Optional[str] = None    # StockTwits users can tag Bullish / Bearish
    score: float = 0.0


@dataclass
class SourceSummary:
    source: str
    posts: int
    avg_score: float
    pct_positive: float
    pct_negative: float
    pct_neutral: float
    bull_bear: Optional[tuple[int, int]] = None
    error: Optional[str] = None


@dataclass
class SentimentResult:
    ticker: str
    company: str
    days: int
    engine: str
    overall: float                       # -100 .. +100
    label: str
    confidence: str
    sources: list[SourceSummary]
    top_positive: list[Post] = field(default_factory=list)
    top_negative: list[Post] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------------------------
# Fetchers (each returns a list of Post; failures are reported, not fatal)
# ---------------------------------------------------------------------------------------------

def _reddit_session() -> tuple[requests.Session, str]:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    cid, secret = os.getenv("REDDIT_CLIENT_ID"), os.getenv("REDDIT_CLIENT_SECRET")
    if cid and secret:
        r = s.post("https://www.reddit.com/api/v1/access_token", auth=(cid, secret),
                   data={"grant_type": "client_credentials"}, timeout=20)
        r.raise_for_status()
        s.headers["Authorization"] = f"bearer {r.json()['access_token']}"
        return s, "https://oauth.reddit.com"
    return s, "https://www.reddit.com"


def fetch_reddit(query: str, since: datetime, subreddits=DEFAULT_SUBREDDITS, limit: int = 100) -> list[Post]:
    s, base = _reddit_session()
    window = "week" if (datetime.now(timezone.utc) - since).days <= 7 else "month"
    posts = []
    for sub in subreddits:
        r = s.get(f"{base}/r/{sub}/search.json", timeout=20,
                  params={"q": query, "restrict_sr": 1, "sort": "new", "t": window, "limit": limit})
        r.raise_for_status()
        for child in r.json().get("data", {}).get("children", []):
            d = child["data"]
            created = datetime.fromtimestamp(d["created_utc"], tz=timezone.utc)
            if created < since:
                continue
            posts.append(Post(f"Reddit r/{sub}", f"{d.get('title', '')}. {d.get('selftext', '')[:1500]}",
                              created, "https://reddit.com" + d.get("permalink", ""), int(d.get("score", 0))))
        time.sleep(1)   # be polite: Reddit rate-limits unauthenticated clients hard
    return posts


def fetch_stocktwits(ticker: str, since: datetime) -> list[Post]:
    r = requests.get(f"https://api.stocktwits.com/api/2/streams/symbol/{ticker.upper()}.json",
                     headers={"User-Agent": USER_AGENT}, timeout=20)
    r.raise_for_status()
    posts = []
    for m in r.json().get("messages", []):
        created = datetime.strptime(m["created_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        if created < since:
            continue
        label = ((m.get("entities") or {}).get("sentiment") or {}).get("basic")
        posts.append(Post("StockTwits", m.get("body", ""), created,
                          f"https://stocktwits.com/message/{m['id']}",
                          int((m.get("likes") or {}).get("total", 0)), label))
    return posts


def _parse_rss(xml_text: str, source: str, since: datetime) -> list[Post]:
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime
    posts = []
    root = ET.fromstring(xml_text)
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        desc = re.sub(r"<[^>]+>", " ", item.findtext("description") or "")
        try:
            created = parsedate_to_datetime(item.findtext("pubDate") or "")
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
        except Exception:
            created = datetime.now(timezone.utc)
        if title and created >= since:
            posts.append(Post(source, f"{title}. {desc.strip()}"[:1500], created, item.findtext("link") or ""))
    return posts


def _news_yfinance(ticker: str, since: datetime) -> list[Post]:
    import yfinance as yf
    posts = []
    for item in yf.Ticker(ticker).news or []:
        c = item.get("content", item)          # yfinance changed its news format; handle both
        title = c.get("title", "")
        summary = c.get("summary", "") or ""
        ts = c.get("pubDate") or item.get("providerPublishTime")
        if isinstance(ts, (int, float)):
            created = datetime.fromtimestamp(ts, tz=timezone.utc)
        elif ts:
            created = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        else:
            created = datetime.now(timezone.utc)
        if created < since or not title:
            continue
        link = (c.get("canonicalUrl") or {}).get("url") if isinstance(c.get("canonicalUrl"), dict) else item.get("link", "")
        posts.append(Post("News headlines", f"{title}. {summary}", created, link or ""))
    return posts


def fetch_news(ticker: str, since: datetime, company: str = "") -> list[Post]:
    """Tries yfinance first, then Yahoo's RSS feed, then Google News RSS. Returns the first non-empty result."""
    attempts = [
        lambda: _news_yfinance(ticker, since),
        lambda: _parse_rss(requests.get("https://feeds.finance.yahoo.com/rss/2.0/headline",
                                        params={"s": ticker.upper(), "region": "US", "lang": "en-US"},
                                        headers={"User-Agent": USER_AGENT}, timeout=20).text,
                           "News headlines", since),
        lambda: _parse_rss(requests.get("https://news.google.com/rss/search",
                                        params={"q": f"{ticker.upper()} {company} stock".strip(),
                                                "hl": "en-US", "gl": "US", "ceid": "US:en"},
                                        headers={"User-Agent": USER_AGENT}, timeout=20).text,
                           "News headlines", since),
    ]
    last_error = None
    for attempt in attempts:
        try:
            got = attempt()
            if got:
                return got
        except Exception as e:                   # try the next feed
            last_error = e
    if last_error:
        raise last_error
    return []


def company_name(ticker: str) -> str:
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        return info.get("shortName") or info.get("longName") or ticker.upper()
    except Exception:
        return ticker.upper()


# ---------------------------------------------------------------------------------------------
# Filtering + aggregation
# ---------------------------------------------------------------------------------------------

def _name_core(name: str) -> str:
    core = re.sub(r"[,.()]|\b(inc|corp|corporation|co|ltd|plc|holdings|group|company|the|class [ab]|"
                  r"sponsored|adr|ads|oyj|ag|sa|nv|se|asa|ab|spa|s\.?a)\b", "", name, flags=re.I)
    return " ".join(core.split())


def is_relevant(text: str, ticker: str, name: str) -> bool:
    """Keep posts that really mention the company. Short tickers like ON, IT, ALL need a $cashtag."""
    t = ticker.upper()
    if re.search(rf"\${re.escape(t)}\b", text, re.I):
        return True
    core = _name_core(name)
    if core and len(core) >= 3 and re.search(rf"\b{re.escape(core)}\b", text, re.I):
        return True
    return len(t) >= 3 and re.search(rf"\b{re.escape(t)}\b", text) is not None


def summarize_source(source: str, posts: list[Post]) -> SourceSummary:
    if not posts:
        return SourceSummary(source, 0, 0.0, 0.0, 0.0, 0.0)
    n = len(posts)
    pos = sum(p.score > NEUTRAL_BAND for p in posts)
    neg = sum(p.score < -NEUTRAL_BAND for p in posts)
    bb = None
    labels = [p.self_label for p in posts if p.self_label]
    if labels:
        bb = (labels.count("Bullish"), labels.count("Bearish"))
    return SourceSummary(source, n, mean(p.score for p in posts), pos / n, neg / n, (n - pos - neg) / n, bb)


def aggregate(ticker: str, company: str, posts: list[Post], days: int, scorer: Scorer,
              errors: dict[str, str], attempted: Optional[list[str]] = None) -> SentimentResult:
    for p in posts:
        p.score = scorer.score(p.text)
        # A StockTwits author's own Bullish/Bearish tag is stronger evidence than word-scoring.
        if p.self_label == "Bullish":
            p.score = max(p.score, 0.0) * 0.5 + 0.5
        elif p.self_label == "Bearish":
            p.score = min(p.score, 0.0) * 0.5 - 0.5

    groups: dict[str, list[Post]] = {}
    for p in posts:
        key = "Reddit" if p.source.startswith("Reddit") else p.source
        groups.setdefault(key, []).append(p)
    sources = [summarize_source(k, v) for k, v in groups.items()]
    for name in attempted or []:            # sources that worked but found nothing still get a line
        if name not in groups:
            sources.append(SourceSummary(name, 0, 0, 0, 0, 0))
    for name, err in errors.items():
        sources.append(SourceSummary(name, 0, 0, 0, 0, 0, error=err))

    # Each post counts once, with a gentle boost for engagement (log scale so one viral post can't dominate).
    import math
    weights = [1 + math.log1p(max(p.engagement, 0)) * 0.25 for p in posts]
    overall = (sum(p.score * w for p, w in zip(posts, weights)) / sum(weights) * 100) if posts else 0.0

    label = ("very positive" if overall >= 40 else "positive" if overall >= 10 else
             "very negative" if overall <= -40 else "negative" if overall <= -10 else "mixed / neutral")
    active = [s for s in sources if s.posts]
    n = len(posts)
    confidence = "high" if n >= 150 and len(active) >= 2 else "medium" if n >= 40 else "low"

    notes = []
    if n < 20:
        notes.append(f"Only {n} relevant posts found; this is a small sample.")
    if len(active) >= 2:
        signs = {s.source: (1 if s.avg_score > NEUTRAL_BAND else -1 if s.avg_score < -NEUTRAL_BAND else 0)
                 for s in active}
        if 1 in signs.values() and -1 in signs.values():
            notes.append("Sources disagree: " + ", ".join(f"{k} {'+' if v > 0 else '-' if v < 0 else '0'}"
                                                           for k, v in signs.items()) + ".")

    seen, unique = set(), []
    for p in posts:                      # cross-posts and spam repeat the same text; show each once
        key = " ".join(p.text.lower().split())[:200]
        if key not in seen:
            seen.add(key)
            unique.append(p)
    ranked = sorted(unique, key=lambda p: p.score)
    return SentimentResult(ticker.upper(), company, days, scorer.engine, overall, label, confidence, sources,
                           top_positive=[p for p in reversed(ranked[-3:]) if p.score > NEUTRAL_BAND],
                           top_negative=[p for p in ranked[:3] if p.score < -NEUTRAL_BAND], notes=notes)


def run(ticker: str, days: int = 7, subreddits=DEFAULT_SUBREDDITS) -> SentimentResult:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    name = company_name(ticker)
    query = f'"${ticker.upper()}" OR "{_name_core(name)}"'
    fetchers = {
        "Reddit": lambda: fetch_reddit(query, since, subreddits),
        "StockTwits": lambda: fetch_stocktwits(ticker, since),
        "News headlines": lambda: fetch_news(ticker, since, _name_core(name)),
    }
    posts, errors, attempted = [], {}, []
    for src, fn in fetchers.items():
        try:
            got = fn()
            # StockTwits streams are per-symbol; Reddit and news searches get filtered for real mentions.
            kept = got if src == "StockTwits" else [p for p in got if is_relevant(p.text, ticker, name)]
            posts += kept
            attempted.append(src)
        except Exception as e:
            errors[src] = f"{type(e).__name__}: {str(e)[:120]}"
    return aggregate(ticker, name, posts, days, Scorer(), errors, attempted)


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------

def _clip(text: str, n: int = 140) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3] + "..."


def report(r: SentimentResult) -> str:
    total = sum(s.posts for s in r.sources)
    lines = [
        f"**Social sentiment: {r.company} ({r.ticker}), last {r.days} days**",
        f"Overall: **{r.overall:+.0f} / 100 ({r.label})** from {total} posts - confidence {r.confidence}",
        "",
        "**By source**",
    ]
    for s in r.sources:
        if s.error:
            lines.append(f"- {s.source}: unavailable ({s.error})")
        elif s.posts == 0:
            lines.append(f"- {s.source}: no relevant posts")
        else:
            extra = ""
            if s.bull_bear and sum(s.bull_bear):
                b, br = s.bull_bear
                extra = f"; authors tagged {b} Bullish vs {br} Bearish ({b / (b + br):.0%} bullish)"
            lines.append(f"- {s.source}: {s.posts} posts, average {s.avg_score * 100:+.0f}; "
                         f"{s.pct_positive:.0%} positive / {s.pct_negative:.0%} negative / "
                         f"{s.pct_neutral:.0%} neutral{extra}")
    if r.top_positive:
        lines += ["", "**Most positive**"] + [f"- ({p.source}) {_clip(p.text)}" for p in r.top_positive]
    if r.top_negative:
        lines += ["", "**Most negative**"] + [f"- ({p.source}) {_clip(p.text)}" for p in r.top_negative]
    lines += [
        "",
        "**How this was scored**",
        f"Each post is scored -100 to +100 with {r.engine}. Posts only count if they mention ${r.ticker} "
        f"or the company name. StockTwits posts the author tagged Bullish/Bearish are pushed that way. "
        f"Popular posts count a little more (log scale, so one viral post can't dominate). "
        f"Between -10 and +10 is treated as mixed.",
    ]
    lines += [f"Note: {n}" for n in r.notes]
    lines += ["", "_Social media is noisy: sarcasm, bots and hype skew it, and sentiment often lags price. "
                  "Use it as one input, not a signal on its own._"]
    return "\n".join(lines)
