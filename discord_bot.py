"""
Discord front end.

    !sentiment AAPL        last 7 days
    !sentiment TSLA 3      last 3 days

Setup: set DISCORD_TOKEN (and optionally REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET) as environment
variables or in a .env file, enable "Message Content Intent" in the Discord Developer Portal, then:
    python discord_bot.py
"""

import asyncio
import os

import discord
from discord.ext import commands

import sentiment

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)


def _chunks(text: str, size: int = 1900):
    buf = ""
    for line in text.split("\n"):
        if len(buf) + len(line) + 1 > size:
            yield buf
            buf = ""
        buf += line + "\n"
    if buf.strip():
        yield buf


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} - ready for !sentiment commands")


@bot.command(name="sentiment")
async def sentiment_cmd(ctx: commands.Context, ticker: str, days: int = 7):
    async with ctx.typing():
        try:
            res = await asyncio.to_thread(sentiment.run, ticker, max(1, min(days, 30)))
        except Exception as e:
            await ctx.reply(f"Couldn't check sentiment for {ticker.upper()}: {type(e).__name__}: {e}")
            return
    for part in _chunks(sentiment.report(res)):
        await ctx.send(part)


@bot.command(name="help_sentiment")
async def help_sentiment(ctx: commands.Context):
    await ctx.reply("`!sentiment TICKER [days]` - what Reddit, StockTwits and news are saying (default 7 days)")


if __name__ == "__main__":
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        raise SystemExit("Set the DISCORD_TOKEN environment variable first.")
    bot.run(token)
