"""A curated list of ~100 of the largest US-listed companies (mega/large cap,
liquid, across every sector) — the "safe" universe for live trading while the
arena is still being validated. Hand-curated from the S&P 500's largest
constituents, NOT a live market-cap ranking (the S&P constituent list we use
has no market caps), so treat it as "the big, liquid names", not "exactly
the top 100 today". Every symbol is checked against real data when the brief
is built; any that stop resolving are dropped automatically."""

TOP100 = """
AAPL MSFT NVDA AMZN GOOGL META AVGO TSLA BRK-B LLY JPM WMT V ORCL MA XOM NFLX
COST UNH JNJ PG HD ABBV BAC KO CRM CVX WFC CSCO TMUS AMD PM IBM MRK ABT GE MCD
LIN NOW PEP ACN TXN ISRG DIS INTU QCOM GS CAT AXP VZ T BKNG RTX MS SPGI AMGN
PGR NEE TMO PFE LOW UBER BA HON ETN SCHW BSX UNP DHR SYK TJX C PANW COP ADP
BLK LMT VRTX MDT ANET MU AMAT ADI GILD DE CB KKR PLD SBUX BMY SO ICE LRCX
KLAC APH CMCSA NKE MO MCK PLTR
""".split()
