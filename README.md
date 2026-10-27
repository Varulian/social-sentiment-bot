# Social Sentiment Bot

What are people saying about a company right now, and is it positive or negative?

```
pip install -r requirements.txt
python sentiment_cli.py AAPL        # last 7 days
python sentiment_cli.py TSLA 3      # last 3 days
python discord_bot.py               # then in Discord: !sentiment AAPL   or   !sentiment TSLA 3
```

## Setup

Put these in a `.env` file in this folder (or set them as environment variables):

```
DISCORD_TOKEN=your-discord-bot-token
REDDIT_CLIENT_ID=optional-but-recommended
REDDIT_CLIENT_SECRET=optional-but-recommended
```

Reddit credentials are free: go to https://www.reddit.com/prefs/apps, choose **create app**, then **script**.
Without them the bot falls back to Reddit's public pages, which get rate-limited or blocked more often.
Discord: enable **Message Content Intent** for the bot.

## Sources (all free)

| Source | What's pulled |
|---|---|
| Reddit | Recent posts mentioning the $ticker or company name in r/stocks, r/investing, r/wallstreetbets, r/StockMarket, r/options |
| StockTwits | The ticker's public message stream, including authors' own Bullish/Bearish tags |
| News headlines | Recent Yahoo Finance headlines and summaries |

X/Twitter isn't included because its API is paid. If one source fails, the report says so and carries on.

## How it scores

- Each post gets a score from -100 to +100 using **VADER**, a sentiment model built for social media, extended
  with stock slang (moon, tendies, bagholder, puts, rug pull, downgrade...). If VADER isn't installed, a smaller
  built-in word list is used.
- **Relevance filter:** a post must mention `$TICKER` or the company name. Short tickers (ON, IT, ALL) need the `$`
  cashtag so ordinary words don't count.
- **StockTwits self-tags** (Bullish/Bearish) push that post's score in that direction.
- **Engagement weighting:** popular posts count a little more, on a log scale so one viral post can't dominate.
- **Overall label:** +40 or more very positive, +10 to +40 positive, -10 to +10 mixed, then negative and very negative.
- **Confidence:** high with 150+ posts from 2+ sources, medium with 40+ posts, otherwise low. The report also flags
  when sources disagree.

## Files

```
sentiment.py       fetching, scoring, aggregation, report (all the logic)
sentiment_cli.py   terminal front end
discord_bot.py     Discord front end (!sentiment)
tests/             offline tests with made-up posts (pytest -q)
```

## Caveats

Sarcasm, bots, pump-and-dump hype and small samples all distort social sentiment, and it often reacts to price
rather than predicting it. Treat it as one input alongside the odds and weather bots, not a trading signal on its own.
