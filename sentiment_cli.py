"""
Terminal front end.

    python sentiment_cli.py AAPL          # last 7 days
    python sentiment_cli.py TSLA 3        # last 3 days
"""

import sys

from sentiment import report, run


def main() -> None:
    args = sys.argv[1:] or [input("Ticker: ").strip()]
    days = int(args[1]) if len(args) > 1 else 7
    print(report(run(args[0], days)).replace("**", "").replace("_", ""))


if __name__ == "__main__":
    main()
