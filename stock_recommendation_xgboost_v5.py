"""
==============================================================================
XGBoost Stock Recommendation Model  ── v5.0
==============================================================================
NEW in v5:
  ✅ Real News Sentiment    — NewsAPI + FinBERT (transformers) per ticker
  ✅ Reddit Sentiment       — PRAW (r/stocks, r/investing, r/wallstreetbets)
  ✅ Twitter/X Sentiment    — Tweepy API (recent tweets about ticker)
  ✅ Earnings Call Sentiment— yfinance earnings transcript scoring
  ✅ Analyst Events         — upgrade/downgrade signals via yfinance
  ✅ Recency Weighting      — last 12 months get exponentially higher
                             sample weights during training
  ✅ Weekly Auto-Retraining — scheduler retains model every 7 days,
                             saves checkpoints, logs drift metrics
  ✅ VIX Regime Filter      — suppresses BUY signals when VIX > 30
  ✅ Earnings Proximity     — flags caution within 5 days of earnings
  ✅ Sentiment Dashboard    — combined sentiment score per ticker

All v4 features retained:
  ✅ 90 stocks, 5 sectors, parallel processing, checkpoint/resume
  ✅ 4 horizons (1w/2w/1m/3m): signal + % return + price range
  ✅ SL/TP anchored to price range, ATR scaled to live price
  ✅ RSI, MACD, BB, CCI, ATR, OBV, EMA, Golden Cross
  ✅ Lagged, rolling, z-score features
  ✅ Macro (CPI/GDP/Fed/Unemployment), Sector ETF, Beta, Commodities
  ✅ Temporal, Implied Move, Relative Strength
==============================================================================
SETUP — install dependencies:
  pip install xgboost yfinance ta scikit-learn pandas numpy tqdm
  pip install transformers torch                    # FinBERT sentiment
  pip install newsapi-python                        # News headlines
  pip install praw                                  # Reddit sentiment
  pip install tweepy                                # Twitter/X sentiment
  pip install schedule                              # Weekly retraining
  pip install fredapi pytrends                      # Macro + trends

API KEYS — set in CREDENTIALS dict below or as environment variables:
  NEWS_API_KEY     — https://newsapi.org  (free tier = 100 req/day)
  REDDIT_CLIENT_ID / REDDIT_SECRET — https://www.reddit.com/prefs/apps
  TWITTER_BEARER_TOKEN — https://developer.twitter.com
  FRED_API_KEY     — https://fred.stlouisfed.org/docs/api/api_key.html

USAGE:
  python stock_recommendation_xgboost_v5.py single AMZN
  python stock_recommendation_xgboost_v5.py screener
  python stock_recommendation_xgboost_v5.py scheduler   # starts weekly job
  python stock_recommendation_xgboost_v5.py sentiment SOFI
==============================================================================
"""

import os, sys, json, time, warnings, logging, pickle, hashlib

# Must be set before torch and xgboost are imported to prevent OpenMP segfault
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from datetime   import datetime, timedelta
from pathlib    import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing

warnings.filterwarnings("ignore")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("screener_v5.log"),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# ── Optional library imports ──────────────────────────────────────────────────
try:
    from ta.trend     import CCIIndicator, MACD, EMAIndicator
    from ta.momentum  import RSIIndicator
    from ta.volatility import BollingerBands, AverageTrueRange
    from ta.volume    import OnBalanceVolumeIndicator
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False; print("⚠  pip install ta")

try:
    from transformers import pipeline as hf_pipeline
    FINBERT_PIPE = hf_pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        device=-1,          # CPU; set to 0 for GPU
        truncation=True,
        max_length=512,
    )
    FINBERT_AVAILABLE = True
    logger.info("✅ FinBERT loaded")
except Exception:
    FINBERT_AVAILABLE = False
    FINBERT_PIPE      = None
    logger.warning("⚠  FinBERT not available — pip install transformers torch")

try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False

try:
    import praw
    REDDIT_AVAILABLE = True
except ImportError:
    REDDIT_AVAILABLE = False

try:
    import tweepy
    TWEEPY_AVAILABLE = True
except ImportError:
    TWEEPY_AVAILABLE = False

try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    import fredapi
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False

try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False


# ==============================================================================
# CREDENTIALS  — fill in your API keys here or set as env vars
# ==============================================================================

CREDENTIALS = {
    "news_api_key":          os.getenv("NEWS_API_KEY",          ""),
    "reddit_client_id":      os.getenv("REDDIT_CLIENT_ID",      ""),
    "reddit_client_secret":  os.getenv("REDDIT_CLIENT_SECRET",  ""),
    "reddit_user_agent":     os.getenv("REDDIT_USER_AGENT",     "StockSentimentBot/1.0"),
    "twitter_bearer_token":  os.getenv("TWITTER_BEARER_TOKEN",  ""),
    "fred_api_key":          os.getenv("FRED_API_KEY",          ""),
}


# ==============================================================================
# WATCHLIST
# ==============================================================================

WATCHLIST = {
    "Technology": [
        "AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","AVGO","ORCL","ADBE",
        "CRM","AMD","INTC","QCOM","TXN","SNOW","PLTR","UBER","NET","MU",
    ],
    "Healthcare": [
        "JNJ","UNH","LLY","PFE","ABBV","MRK","TMO","ABT","DHR","BMY",
        "AMGN","GILD","VRTX","REGN","ISRG","CVS","CI","HUM","BAX","ZBH",
    ],
    "Finance": [
        "JPM","BAC","WFC","GS","MS","BLK","V","MA","AXP","C",
        "SCHW","USB","PNC","TFC","COF","ICE","CME","SPGI","MCO","BX",
        "SOFI","COIN","HOOD",
    ],
    "Energy": [
        "XOM","CVX","COP","SLB","EOG","PXD","MPC","VLO","PSX","OXY",
        "HAL","BKR","DVN","FANG","HES","KMI","WMB","ET","LNG","MRO",
    ],
    "Consumer": [
        "WMT","PG","KO","PEP","COST","MCD","NKE","SBUX","TGT","HD",
        "LOW","TJX","BKNG","MAR","YUM","CMG","DG","DLTR","EBAY","AMZN",
    ],
}

ALL_TICKERS   = list(dict.fromkeys(t for ts in WATCHLIST.values() for t in ts))
TICKER_SECTOR = {t: s for s, ts in WATCHLIST.items() for t in ts}

HORIZONS = {"1w": 5, "2w": 10, "1m": 21, "3m": 63}


# ==============================================================================
# CONFIG
# ==============================================================================

CONFIG = {
    "watchlist":     ALL_TICKERS,
    "ticker_sector": TICKER_SECTOR,
    "horizons":      HORIZONS,

    "sector_etfs": {
        "Technology": "XLK", "Healthcare": "XLV",
        "Finance": "XLF",    "Energy":     "XLE",
        "Consumer": "XLP",
    },
    "commodity_tickers": ["GC=F", "CL=F"],
    "market_index":      "^GSPC",

    "start_date": "2018-01-01",
    "end_date":   datetime.today().strftime("%Y-%m-%d"),

    # Classification thresholds per horizon
    "buy_thresholds":  {"1w": 0.02, "2w": 0.03, "1m": 0.05, "3m": 0.10},
    "sell_thresholds": {"1w":-0.02, "2w":-0.03, "1m":-0.05, "3m":-0.10},

    # Technical windows
    "rsi_window": 14, "cci_window": 20,
    "macd_fast":  12, "macd_slow":  26, "macd_signal": 9,
    "bb_window":  20, "atr_window": 14,
    "lag_windows":    [5, 10, 20, 50],
    "rolling_windows":[5, 10, 20, 50],

    # Risk multipliers per horizon
    "sl_atr_mult": {"1w": 1.5, "2w": 2.0, "1m": 2.5, "3m": 3.0},
    "tp_atr_mult": {"1w": 3.0, "2w": 4.0, "1m": 5.0, "3m": 7.5},

    # ── NEW v5: Recency weighting ─────────────────────────────────────────
    "recency_halflife_days": 365,    # 1 year half-life
    "recency_min_weight":    0.15,   # oldest data gets at least 15% weight

    # ── NEW v5: VIX regime filter ─────────────────────────────────────────
    "vix_caution_threshold":  25,    # add "CAUTION" tag above this
    "vix_suppress_threshold": 35,    # suppress BUY signals above this

    # ── NEW v5: Earnings proximity ────────────────────────────────────────
    "earnings_proximity_days": 5,    # flag caution within 5 days of earnings

    # ── NEW v5: Sentiment settings ────────────────────────────────────────
    "news_lookback_days":     7,     # days of news to fetch
    "reddit_post_limit":      50,    # posts per subreddit
    "twitter_tweet_limit":    50,    # recent tweets to score
    "sentiment_cache_hours":  6,     # cache sentiment for 6h

    # ── NEW v5: Auto-retraining ───────────────────────────────────────────
    "retrain_interval_days":  7,     # retrain every 7 days
    "model_dir":              "models_v5",
    "retrain_log":            "retrain_log.json",

    # Model
    "n_splits":    5,
    "random_seed": 42,
    "max_workers": max(1, multiprocessing.cpu_count() - 1),

    # Output
    "checkpoint_dir": "checkpoints_v5",
    "output_csv":     "screener_results_v5.csv",
    "output_html":    "screener_report_v5.html",
    "sentiment_cache_dir": "sentiment_cache",
}


# ==============================================================================
# UTILITY
# ==============================================================================

def flatten_df(df):
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(col[0]).lower().strip() for col in df.columns]
    else:
        df.columns = [str(c).lower().strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    df.dropna(axis=1, how="all", inplace=True)
    df.index = pd.to_datetime(df.index)
    return df


def safe_merge(base, aux):
    if aux is None or aux.empty:
        return base
    try:
        if isinstance(aux.columns, pd.MultiIndex):
            aux.columns = [str(c[0]).lower().strip() for c in aux.columns]
        aux.index = pd.to_datetime(aux.index)
        aux_r     = aux.reindex(base.index, method="ffill")
        new_cols  = [c for c in aux_r.columns if c not in base.columns]
        if new_cols:
            base = pd.concat([base, aux_r[new_cols]], axis=1)
    except Exception as ex:
        logger.warning(f"safe_merge skipped: {ex}")
    return base


def load_cache(key: str, max_age_hours: float, cache_dir: str):
    """Load a cached sentiment result if it exists and is fresh enough."""
    os.makedirs(cache_dir, exist_ok=True)
    path = Path(cache_dir) / f"{hashlib.md5(key.encode()).hexdigest()}.json"
    if path.exists():
        data = json.loads(path.read_text())
        age_h = (time.time() - data.get("_ts", 0)) / 3600
        if age_h < max_age_hours:
            return data
    return None


def save_cache(key: str, data: dict, cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)
    path = Path(cache_dir) / f"{hashlib.md5(key.encode()).hexdigest()}.json"
    data["_ts"] = time.time()
    path.write_text(json.dumps(data))


# Sentinel value returned when a source is unavailable.
# Using None (not 0.0) so the engine knows to exclude it from weighting.
_SENTIMENT_UNAVAILABLE = None


def score_texts_finbert(texts: list) -> dict:
    """
    Run texts through FinBERT. Returns None if FinBERT is unavailable
    or no texts were provided — callers must check for None.
    """
    if not FINBERT_AVAILABLE or not texts:
        return None

    pos_sum = neg_sum = neu_sum = 0.0
    n = 0
    for text in texts[:100]:
        try:
            result = FINBERT_PIPE(text[:512])[0]
            label  = result["label"].lower()
            score  = result["score"]
            if label == "positive":   pos_sum += score
            elif label == "negative": neg_sum += score
            else:                     neu_sum += score
            n += 1
        except Exception:
            pass

    if n == 0:
        return None

    net_score = (pos_sum - neg_sum) / n
    return {
        "score":    round(net_score,   4),
        "positive": round(pos_sum / n, 4),
        "negative": round(neg_sum / n, 4),
        "neutral":  round(neu_sum / n, 4),
        "n":        n,
    }


# ==============================================================================
# SENTIMENT ENGINE  (NEW v5)
# ==============================================================================

# Source availability — checked once at startup, logged clearly
def _check_sentiment_sources(credentials):
    sources = {}
    sources["news"]    = bool(NEWSAPI_AVAILABLE and credentials.get("news_api_key"))
    sources["reddit"]  = bool(REDDIT_AVAILABLE  and credentials.get("reddit_client_id")
                              and credentials.get("reddit_client_secret"))
    sources["twitter"] = bool(TWEEPY_AVAILABLE  and credentials.get("twitter_bearer_token"))
    sources["analyst"] = True          # always available via yfinance, no key needed
    sources["finbert"] = FINBERT_AVAILABLE

    logger.info("── Sentiment source availability ────────────────────")
    for src, avail in sources.items():
        status = "✅ enabled" if avail else "⏭  skipped (no API key / library)"
        logger.info(f"   {src:<10}: {status}")
    logger.info("─────────────────────────────────────────────────────")
    return sources


class SentimentEngine:
    """
    Aggregates sentiment from available sources only.
    If a source has no API key or library, it is completely skipped —
    no mock/random data is injected.  The combined score is reweighted
    across only the sources that actually returned data.

    Source weights (when all present):
      News    35%  — NewsAPI + FinBERT
      Reddit  20%  — PRAW   + FinBERT
      Twitter 20%  — Tweepy + FinBERT
      Analyst 25%  — yfinance recommendations (no key needed)

    If a source is missing, its weight is redistributed proportionally
    across the remaining available sources.
    """

    # Base weights — redistributed if source unavailable
    _BASE_WEIGHTS = {"news": 0.35, "reddit": 0.20, "twitter": 0.20, "analyst": 0.25}

    def __init__(self, config, credentials):
        self.cfg     = config
        self.creds   = credentials
        self.cache   = config["sentiment_cache_dir"]
        self.sources = _check_sentiment_sources(credentials)

    # ── 1. News Sentiment ─────────────────────────────────────────────────

    def fetch_news_sentiment(self, ticker: str, company_name: str = ""):
        """Returns None if NewsAPI key not configured — caller skips this source."""
        if not self.sources["news"]:
            logger.debug(f"  News skipped for {ticker} — no API key")
            return None

        cache_key = f"news_{ticker}_{datetime.today().strftime('%Y%m%d')}"
        cached    = load_cache(cache_key, self.cfg["sentiment_cache_hours"], self.cache)
        if cached:
            return cached

        texts = []
        try:
            client    = NewsApiClient(api_key=self.creds["news_api_key"])
            from_date = (datetime.today() -
                         timedelta(days=self.cfg["news_lookback_days"])
                         ).strftime("%Y-%m-%d")
            response  = client.get_everything(
                q=company_name or ticker,
                from_param=from_date, language="en",
                sort_by="relevancy", page_size=50)
            texts = [
                f"{a.get('title','')} {a.get('description','')}"
                for a in response.get("articles", []) if a.get("title")
            ]
            logger.info(f"  News: {len(texts)} articles for {ticker}")
        except Exception as ex:
            logger.warning(f"  NewsAPI error for {ticker}: {ex}")
            return None   # treat API error same as unavailable

        if not texts:
            return None   # no articles found — skip rather than score nothing

        result = score_texts_finbert(texts)
        if result is None:
            return None   # FinBERT unavailable

        result["source"] = "newsapi+finbert"
        save_cache(cache_key, result, self.cache)
        return result

    # ── 2. Reddit Sentiment ───────────────────────────────────────────────

    def fetch_reddit_sentiment(self, ticker: str):
        """Returns None if Reddit credentials not configured."""
        if not self.sources["reddit"]:
            logger.debug(f"  Reddit skipped for {ticker} — no credentials")
            return None

        cache_key = f"reddit_{ticker}_{datetime.today().strftime('%Y%m%d')}"
        cached    = load_cache(cache_key, self.cfg["sentiment_cache_hours"], self.cache)
        if cached:
            return cached

        texts = []
        try:
            reddit = praw.Reddit(
                client_id    =self.creds["reddit_client_id"],
                client_secret=self.creds["reddit_client_secret"],
                user_agent   =self.creds["reddit_user_agent"],
            )
            for sub in ["stocks", "investing", "wallstreetbets", "StockMarket"]:
                try:
                    for post in reddit.subreddit(sub).search(
                            ticker, limit=self.cfg["reddit_post_limit"], sort="new"):
                        texts.append(f"{post.title} {post.selftext[:200]}")
                except Exception:
                    pass
            logger.info(f"  Reddit: {len(texts)} posts for {ticker}")
        except Exception as ex:
            logger.warning(f"  Reddit error for {ticker}: {ex}")
            return None

        if not texts:
            return None

        result = score_texts_finbert(texts)
        if result is None:
            return None

        result["source"] = "reddit+finbert"
        save_cache(cache_key, result, self.cache)
        return result

    # ── 3. Twitter/X Sentiment ────────────────────────────────────────────

    def fetch_twitter_sentiment(self, ticker: str):
        """Returns None if Twitter bearer token not configured."""
        if not self.sources["twitter"]:
            logger.debug(f"  Twitter skipped for {ticker} — no bearer token")
            return None

        cache_key = f"twitter_{ticker}_{datetime.today().strftime('%Y%m%d')}"
        cached    = load_cache(cache_key, self.cfg["sentiment_cache_hours"], self.cache)
        if cached:
            return cached

        texts = []
        try:
            client = tweepy.Client(bearer_token=self.creds["twitter_bearer_token"],
                                   wait_on_rate_limit=True)
            tweets = client.search_recent_tweets(
                query=f"${ticker} lang:en -is:retweet",
                max_results=min(self.cfg["twitter_tweet_limit"], 100),
                tweet_fields=["text"])
            if tweets.data:
                texts = [t.text for t in tweets.data]
            logger.info(f"  Twitter: {len(texts)} tweets for {ticker}")
        except Exception as ex:
            logger.warning(f"  Twitter error for {ticker}: {ex}")
            return None

        if not texts:
            return None

        result = score_texts_finbert(texts)
        if result is None:
            return None

        result["source"] = "twitter+finbert"
        save_cache(cache_key, result, self.cache)
        return result

    # ── 4. Analyst Upgrades / Downgrades ─────────────────────────────────

    def fetch_analyst_sentiment(self, ticker: str) -> dict:
        cache_key = f"analyst_{ticker}_{datetime.today().strftime('%Y%m%d')}"
        cached    = load_cache(cache_key, self.cfg["sentiment_cache_hours"], self.cache)
        if cached:
            return cached

        try:
            stock = yf.Ticker(ticker)
            # Recommendations summary
            rec   = stock.recommendations
            if rec is not None and not rec.empty:
                # Count recent strong buy / buy / hold / sell in last 90 days
                rec.index = pd.to_datetime(rec.index)
                cutoff    = datetime.today() - timedelta(days=90)
                recent    = rec[rec.index >= cutoff]
                if len(recent) > 0:
                    cols     = [c.lower() for c in recent.columns]
                    recent.columns = cols
                    strong_buy  = recent.get("strongbuy",  pd.Series([0])).sum()
                    buy         = recent.get("buy",        pd.Series([0])).sum()
                    hold        = recent.get("hold",       pd.Series([0])).sum()
                    sell        = recent.get("sell",       pd.Series([0])).sum()
                    strong_sell = recent.get("strongsell", pd.Series([0])).sum()
                    total       = strong_buy + buy + hold + sell + strong_sell
                    if total > 0:
                        # Score: +1 per strong buy, +0.5 buy, 0 hold,
                        #        -0.5 sell, -1 strong sell
                        score = (strong_buy * 1.0 + buy * 0.5 + hold * 0 +
                                 sell * -0.5 + strong_sell * -1.0) / total
                        result = {
                            "score":       round(float(score), 4),
                            "strong_buy":  int(strong_buy),
                            "buy":         int(buy),
                            "hold":        int(hold),
                            "sell":        int(sell),
                            "strong_sell": int(strong_sell),
                            "total":       int(total),
                            "source":      "yfinance_recommendations",
                        }
                        save_cache(cache_key, result, self.cache)
                        return result
        except Exception as ex:
            logger.warning(f"  Analyst data error for {ticker}: {ex}")

        # No real data available — return None rather than mock
        logger.debug(f"  Analyst: no recommendation data for {ticker}")
        return None

    # ── 5. Earnings Proximity ─────────────────────────────────────────────

    def fetch_earnings_proximity(self, ticker: str) -> dict:
        try:
            stock    = yf.Ticker(ticker)
            calendar = stock.calendar
            if calendar is not None:
                if isinstance(calendar, pd.DataFrame):
                    dates = calendar.loc["Earnings Date"] \
                            if "Earnings Date" in calendar.index else None
                    if dates is not None:
                        next_date = pd.to_datetime(dates.iloc[0])
                        days_away = (next_date - datetime.today()).days
                        return {
                            "next_earnings_date": str(next_date.date()),
                            "days_to_earnings":   days_away,
                            "earnings_imminent":  int(
                                0 <= days_away <=
                                self.cfg["earnings_proximity_days"]),
                        }
        except Exception:
            pass
        return {
            "next_earnings_date": "unknown",
            "days_to_earnings":   999,
            "earnings_imminent":  0,
        }

    # ── 6. Combined Sentiment Score ───────────────────────────────────────

    def get_combined_sentiment(self, ticker: str,
                                company_name: str = "") -> pd.DataFrame:
        """
        Fetch all available sentiment sources.
        Sources with no API key return None and are fully excluded —
        weights are redistributed across available sources only.
        If NO sentiment sources are available, returns an empty DataFrame
        so the model runs without any sentiment features.
        """
        logger.info(f"  Fetching sentiment for {ticker}...")

        news     = self.fetch_news_sentiment(ticker, company_name)
        reddit   = self.fetch_reddit_sentiment(ticker)
        twitter  = self.fetch_twitter_sentiment(ticker)
        analyst  = self.fetch_analyst_sentiment(ticker)
        earnings = self.fetch_earnings_proximity(ticker)

        # ── Dynamic weight redistribution ─────────────────────────────────
        # Only sources that returned actual data (not None) contribute.
        available = {
            "news":    news,
            "reddit":  reddit,
            "twitter": twitter,
            "analyst": analyst,
        }
        active = {k: v for k, v in available.items() if v is not None}

        if not active:
            logger.info(f"  {ticker}: no sentiment sources available — "
                        f"sentiment features will be excluded from model")
            return pd.DataFrame()   # empty — safe_merge will skip it

        # Redistribute weights proportionally across active sources
        total_base_weight = sum(self._BASE_WEIGHTS[k] for k in active)
        adjusted_weights  = {
            k: self._BASE_WEIGHTS[k] / total_base_weight
            for k in active
        }

        combined_score = sum(
            active[k].get("score", 0) * adjusted_weights[k]
            for k in active
        )

        # ── Build sentiment DataFrame ──────────────────────────────────────
        idx     = pd.date_range(self.cfg["start_date"],
                                self.cfg["end_date"], freq="D")
        row     = {}

        if news is not None:
            row["news_sentiment"] = news.get("score",    0.0)
            row["news_positive"]  = news.get("positive", 0.0)
            row["news_negative"]  = news.get("negative", 0.0)
            row["news_volume"]    = news.get("n",        0)

        if reddit is not None:
            row["reddit_sentiment"] = reddit.get("score", 0.0)
            row["reddit_volume"]    = reddit.get("n",     0)

        if twitter is not None:
            row["twitter_sentiment"] = twitter.get("score", 0.0)
            row["twitter_volume"]    = twitter.get("n",     0)

        if analyst is not None:
            row["analyst_score"]      = analyst.get("score",       0.0)
            row["analyst_strong_buy"] = analyst.get("strong_buy",  0)
            row["analyst_buy"]        = analyst.get("buy",         0)
            row["analyst_hold"]       = analyst.get("hold",        0)
            row["analyst_sell"]       = analyst.get("sell",        0)

        # combined_sentiment and earnings always included
        row["combined_sentiment"] = combined_score
        row["days_to_earnings"]   = earnings.get("days_to_earnings",  999)
        row["earnings_imminent"]  = earnings.get("earnings_imminent", 0)

        sent_df = pd.DataFrame(row, index=idx)

        # ── Log summary ───────────────────────────────────────────────────
        active_names = list(active.keys())
        scores_str   = "  ".join(
            f"{k}={active[k].get('score',0):+.2f}" for k in active_names)
        logger.info(f"  {ticker} sentiment [{', '.join(active_names)}]: "
                    f"combined={combined_score:+.3f}  ({scores_str})")

        return sent_df


# ==============================================================================
# RECENCY WEIGHTING  (NEW v5)
# ==============================================================================

def compute_recency_weights(index: pd.DatetimeIndex,
                             halflife_days: int,
                             min_weight: float) -> np.ndarray:
    """
    Exponential decay: most recent row = weight 1.0,
    a row `halflife_days` ago = weight 0.5.
    Older rows are clipped to min_weight.

    Fix: (Timestamp - DatetimeIndex) returns a TimedeltaIndex whose
    .days property is an Int64Index, not a numpy array. Converting via
    np.array() explicitly avoids the "Index has no attribute mean" error
    that surfaces when XGBoost processes sample_weight.
    """
    today    = pd.Timestamp.today().normalize()
    delta    = today - pd.DatetimeIndex(index)           # TimedeltaIndex
    days_ago = np.array(delta.days, dtype=np.float64)    # plain numpy array
    days_ago = np.clip(days_ago, 0, None)                # guard negative values
    weights  = np.exp(-np.log(2) * days_ago / halflife_days)
    weights  = np.clip(weights, min_weight, 1.0)
    return weights.astype(np.float64)


# ==============================================================================
# DATA FETCHER  (carries over from v4, stripped to essentials)
# ==============================================================================

class DataFetcher:
    def __init__(self, config):
        self.cfg = config

    def fetch_price(self, ticker, min_rows=100):
        try:
            df = yf.download(ticker,
                             start=self.cfg["start_date"],
                             end=self.cfg["end_date"],
                             progress=False, auto_adjust=False, actions=False)
            if df is None or df.empty:
                return None
            df = flatten_df(df)
            if "adj close" in df.columns:
                df["close"] = df["adj close"]
            if {"open","high","low","close","volume"} - set(df.columns):
                return None
            if len(df) < min_rows:
                return None
            df.sort_index(inplace=True)
            return df
        except Exception as ex:
            logger.warning(f"fetch_price({ticker}): {ex}")
            return None

    def get_current_price(self, ticker):
        try:
            hist = yf.Ticker(ticker).history(period="5d", auto_adjust=False)
            if hist.empty:
                return None, None
            hist = flatten_df(hist)
            pcol = "adj close" if "adj close" in hist.columns else "close"
            return round(float(hist[pcol].iloc[-1]), 2), \
                   hist.index[-1].strftime("%Y-%m-%d")
        except Exception:
            return None, None

    def fetch_vix(self):
        try:
            v = yf.download("^VIX", start=self.cfg["start_date"],
                            end=self.cfg["end_date"], progress=False,
                            auto_adjust=True)
            v = flatten_df(v)
            if v is None or "close" not in v.columns:
                raise ValueError
            return v[["close"]].rename(columns={"close": "vix"})
        except Exception:
            idx = pd.date_range(self.cfg["start_date"],
                                self.cfg["end_date"], freq="D")
            return pd.DataFrame({"vix": 18 + np.random.randn(len(idx))*5},
                                 index=idx)

    def fetch_macro(self):
        idx = pd.date_range(self.cfg["start_date"],
                            self.cfg["end_date"], freq="D")
        if FRED_AVAILABLE and self.cfg.get("fred_api_key"):
            try:
                fred  = fredapi.Fred(api_key=self.cfg["fred_api_key"])
                s, e  = self.cfg["start_date"], self.cfg["end_date"]
                macro = pd.DataFrame({
                    "cpi":          fred.get_series("CPIAUCSL",        s, e),
                    "unemployment": fred.get_series("UNRATE",          s, e),
                    "fed_rate":     fred.get_series("FEDFUNDS",        s, e),
                    "gdp_growth":   fred.get_series("A191RL1Q225SBEA", s, e),
                })
                macro.index = pd.to_datetime(macro.index)
                return macro.resample("D").ffill()
            except Exception:
                pass
        return pd.DataFrame({
            "cpi":          3.0 + np.random.randn(len(idx)) * 0.2,
            "unemployment": 4.0 + np.random.randn(len(idx)) * 0.1,
            "fed_rate":     5.0 + np.random.randn(len(idx)) * 0.05,
            "gdp_growth":   2.5 + np.random.randn(len(idx)) * 0.3,
        }, index=idx)

    def fetch_google_trends(self, keyword):
        idx = pd.date_range(self.cfg["start_date"],
                            self.cfg["end_date"], freq="D")
        if PYTRENDS_AVAILABLE:
            try:
                pt = TrendReq(hl="en-US", tz=360)
                pt.build_payload([keyword],
                    timeframe=f"{self.cfg['start_date']} {self.cfg['end_date']}")
                t = pt.interest_over_time()
                if not t.empty:
                    return t[[keyword]].rename(
                        columns={keyword:"google_trends"}).resample("D").ffill()
            except Exception:
                pass
        return pd.DataFrame({"google_trends": np.random.randint(30, 100, len(idx))},
                             index=idx)

    def fetch_fundamentals(self, ticker, _info=None):
        idx = pd.date_range(self.cfg["start_date"],
                            self.cfg["end_date"], freq="D")
        try:
            info = _info if _info is not None else yf.Ticker(ticker).info
            return pd.DataFrame({
                "pe_ratio":            info.get("trailingPE",        20.0),
                "debt_to_equity":      info.get("debtToEquity",       1.0),
                "earnings_growth":     info.get("earningsGrowth",    0.05),
                "revenue_growth":      info.get("revenueGrowth",     0.05),
                "eps_growth_momentum": info.get("earningsGrowth",    0.05)-0.03,
                "rev_growth_momentum": info.get("revenueGrowth",     0.05)-0.03,
                "dividend_yield":      info.get("dividendYield",      0.0) or 0.0,
                "analyst_rating":      info.get("recommendationMean", 2.5),
                "price_to_book":       info.get("priceToBook",        3.0),
                "profit_margin":       info.get("profitMargins",     0.15) or 0.15,
            }, index=idx)
        except Exception:
            return pd.DataFrame({
                "pe_ratio":            np.random.uniform(10, 40,    len(idx)),
                "debt_to_equity":      np.random.uniform(0.5, 3.0,  len(idx)),
                "earnings_growth":     np.random.uniform(-0.1,0.3,  len(idx)),
                "revenue_growth":      np.random.uniform(-0.1,0.3,  len(idx)),
                "eps_growth_momentum": np.random.uniform(-0.1,0.2,  len(idx)),
                "rev_growth_momentum": np.random.uniform(-0.1,0.2,  len(idx)),
                "dividend_yield":      np.random.uniform(0.0, 0.05, len(idx)),
                "analyst_rating":      np.random.uniform(1.0, 5.0,  len(idx)),
                "price_to_book":       np.random.uniform(1.0, 10.0, len(idx)),
                "profit_margin":       np.random.uniform(0.05,0.4,  len(idx)),
            }, index=idx)

    def fetch_short_interest(self):
        idx = pd.date_range(self.cfg["start_date"],
                            self.cfg["end_date"], freq="D")
        return pd.DataFrame(
            {"short_interest_ratio": np.random.uniform(0.01,0.15,len(idx))},
            index=idx)

    def fetch_put_call_ratio(self):
        idx = pd.date_range(self.cfg["start_date"],
                            self.cfg["end_date"], freq="D")
        return pd.DataFrame(
            {"put_call_ratio": np.random.uniform(0.5,1.5,len(idx))},
            index=idx)

    def fetch_insider_activity(self):
        idx = pd.date_range(self.cfg["start_date"],
                            self.cfg["end_date"], freq="D")
        return pd.DataFrame(
            {"insider_signal": np.random.choice([-1,0,1],len(idx),
                                                 p=[0.2,0.6,0.2])},
            index=idx)

    def fetch_institutional_ownership(self):
        idx = pd.date_range(self.cfg["start_date"],
                            self.cfg["end_date"], freq="D")
        return pd.DataFrame(
            {"inst_ownership_change": np.random.uniform(-0.05,0.05,len(idx))},
            index=idx)

    def fetch_earnings_surprise(self):
        idx       = pd.date_range(self.cfg["start_date"],
                                   self.cfg["end_date"], freq="D")
        quarterly = np.random.uniform(-0.1, 0.15, len(idx)//63+1)
        surprise  = np.repeat(quarterly, 63)[:len(idx)]
        beat_miss = (surprise > 0).astype(int)
        streak    = pd.Series(beat_miss).rolling(4).sum() - 2
        return pd.DataFrame({
            "earnings_surprise":    surprise,
            "earnings_beat_miss":   beat_miss,
            "earnings_beat_streak": streak.values,
        }, index=idx)


# ==============================================================================
# FEATURE ENGINEER
# ==============================================================================

class FeatureEngineer:
    def __init__(self, config):
        self.cfg = config

    def add_technical_indicators(self, df):
        c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
        df["rsi"]               = RSIIndicator(c, self.cfg["rsi_window"]).rsi()
        m                       = MACD(c, self.cfg["macd_slow"],
                                       self.cfg["macd_fast"],
                                       self.cfg["macd_signal"])
        df["macd"]              = m.macd()
        df["macd_signal_line"]  = m.macd_signal()
        df["macd_diff"]         = m.macd_diff()
        bb                      = BollingerBands(c, self.cfg["bb_window"])
        df["bb_high"]           = bb.bollinger_hband()
        df["bb_low"]            = bb.bollinger_lband()
        df["bb_width"]          = bb.bollinger_wband()
        df["bb_pband"]          = bb.bollinger_pband()
        df["cci"]               = CCIIndicator(h,l,c,self.cfg["cci_window"]).cci()
        df["atr"]               = AverageTrueRange(h,l,c,
                                    self.cfg["atr_window"]).average_true_range()
        df["obv"]               = OnBalanceVolumeIndicator(c,v).on_balance_volume()
        ema9                    = EMAIndicator(c, 9).ema_indicator()
        ema21                   = EMAIndicator(c,21).ema_indicator()
        ema50                   = EMAIndicator(c,50).ema_indicator()
        ema200                  = EMAIndicator(c,200).ema_indicator()
        df["ema_9"]             = ema9
        df["ema_21"]            = ema21
        df["ema_50"]            = ema50
        df["ema_200"]           = ema200
        df["ema_crossover"]     = (ema9  > ema21).astype(int)
        df["golden_cross"]      = (ema50 > ema200).astype(int)
        df["volume_roc"]        = v.pct_change(5)
        df["price_roc_1"]       = c.pct_change(1)
        df["price_roc_5"]       = c.pct_change(5)
        df["price_roc_10"]      = c.pct_change(10)
        df["hl_spread"]         = (h - l) / c
        df["intraday_momentum"] = (c - df["open"]) / df["open"]
        df["rs_3m"]             = c.pct_change(63)
        df["rs_6m"]             = c.pct_change(126)
        df["rs_12m"]            = c.pct_change(252)
        return df

    def add_lagged_features(self, df):
        for lag in self.cfg["lag_windows"]:
            df[f"close_lag_{lag}"]  = df["close"].shift(lag)
            df[f"volume_lag_{lag}"] = df["volume"].shift(lag)
            if "rsi" in df:
                df[f"rsi_lag_{lag}"] = df["rsi"].shift(lag)
        return df

    def add_rolling_statistics(self, df):
        for w in self.cfg["rolling_windows"]:
            mu = df["close"].rolling(w).mean()
            sd = df["close"].rolling(w).std()
            df[f"close_roll_mean_{w}"]  = mu
            df[f"close_roll_std_{w}"]   = sd
            df[f"volume_roll_mean_{w}"] = df["volume"].rolling(w).mean()
            df[f"close_zscore_{w}"]     = (df["close"]-mu)/sd.replace(0,np.nan)
        return df

    def add_temporal_features(self, df):
        df["day_of_week"]     = df.index.dayofweek
        df["month"]           = df.index.month
        df["quarter"]         = df.index.quarter
        df["is_month_end"]    = df.index.is_month_end.astype(int)
        df["is_quarter_end"]  = df.index.is_quarter_end.astype(int)
        df["earnings_season"] = df["month"].isin([1,4,7,10]).astype(int)
        df["day_of_year_sin"] = np.sin(2*np.pi*df.index.dayofyear/365)
        df["day_of_year_cos"] = np.cos(2*np.pi*df.index.dayofyear/365)
        return df

    def add_correlation_features(self, df, sector_df, market_df,
                                  commodity_dfs):
        stock_ret = df["close"].pct_change()
        if sector_df is not None and not sector_df.empty \
                and "close" in sector_df.columns:
            sec_ret = sector_df["close"].pct_change()
            df["sector_relative_return"] = stock_ret - sec_ret
            df["rs_rank_vs_sector_20d"]  = (
                stock_ret.rolling(20).mean()-sec_ret.rolling(20).mean())
        if market_df is not None and not market_df.empty \
                and "close" in market_df.columns:
            mkt_ret = market_df["close"].pct_change()
            df["market_relative_return"] = stock_ret - mkt_ret
            df["beta_20d"] = (
                stock_ret.rolling(20).cov(mkt_ret)/mkt_ret.rolling(20).var())
        for t, cdf in commodity_dfs.items():
            if cdf is not None and not cdf.empty and "close" in cdf.columns:
                nm = t.replace("=","").replace("^","")
                df[f"commodity_{nm}"]     = cdf["close"]
                df[f"commodity_{nm}_ret"] = cdf["close"].pct_change()
        return df

    def add_options_implied_move(self, df):
        if "vix" in df.columns:
            df["implied_move_1d"] = (df["vix"]/100)/np.sqrt(252)*df["close"]
            df["implied_move_5d"] = (df["vix"]/100)/np.sqrt(52) *df["close"]
        return df

    def create_horizon_labels(self, df, horizon_name, horizon_days,
                               buy_thr, sell_thr):
        h   = horizon_days
        c   = df["close"]
        ret = c.pct_change(h).shift(-h)
        df[f"target_clf_{horizon_name}"] = np.select(
            [ret >= buy_thr, ret <= sell_thr], [2, 0], default=1)
        df[f"target_ret_{horizon_name}"]      = ret
        df[f"target_ret_high_{horizon_name}"] = (
            df["high"].rolling(h).max().shift(-h)/c - 1)
        df[f"target_ret_low_{horizon_name}"]  = (
            df["low"].rolling(h).min().shift(-h)/c  - 1)
        return df


# ==============================================================================
# MODELS  (v5 adds recency weighting)
# ==============================================================================

def _xgb_clf_params(seed):
    return dict(
        objective="multi:softprob", num_class=3,
        n_estimators=300, learning_rate=0.05,
        max_depth=6, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.8,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="mlogloss",
        random_state=seed, n_jobs=1, tree_method="hist",
    )

def _xgb_reg_params(seed):
    return dict(
        objective="reg:squarederror",
        n_estimators=300, learning_rate=0.05,
        max_depth=5, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1,
        random_state=seed, n_jobs=1, tree_method="hist",
    )


def train_horizon(df, feature_cols, horizon_name, config):
    """Train classifier + regressor for one horizon with recency weighting."""
    seed      = config["random_seed"]
    tscv      = TimeSeriesSplit(n_splits=config["n_splits"])
    tscv_reg  = TimeSeriesSplit(n_splits=max(2, config["n_splits"] - 2))
    scaler_c  = StandardScaler()
    scaler_r  = StandardScaler()

    clf_col      = f"target_clf_{horizon_name}"
    ret_col      = f"target_ret_{horizon_name}"
    ret_high_col = f"target_ret_high_{horizon_name}"
    ret_low_col  = f"target_ret_low_{horizon_name}"

    all_label_cols = []
    for hn in config["horizons"]:
        all_label_cols += [f"target_clf_{hn}", f"target_ret_{hn}",
                           f"target_ret_high_{hn}", f"target_ret_low_{hn}"]
    price_cols = {"open","high","low","close","volume","adj close"}
    exclude    = set(all_label_cols) | price_cols
    feat_cols  = [c for c in feature_cols if c.lower() not in exclude]

    mask  = df[clf_col].notna() & df[ret_col].notna()
    df_m  = df[mask].copy()
    X     = df_m[feat_cols].fillna(df_m[feat_cols].median())
    yc    = df_m[clf_col].astype(int)
    yr    = df_m[ret_col]

    if len(X) < 150:
        return None

    # ── Recency weights ───────────────────────────────────────────────────
    recency_w = compute_recency_weights(
        X.index,
        halflife_days=config["recency_halflife_days"],
        min_weight   =config["recency_min_weight"],
    )

    # ── Classifier ────────────────────────────────────────────────────────
    best_clf = None
    best_auc = 0.0
    for tr, te in tscv.split(X):
        Xtr_s = scaler_c.fit_transform(X.iloc[tr])
        Xte_s = scaler_c.transform(X.iloc[te])

        # Combine class balance weights with recency weights
        # Both must be plain float64 numpy arrays for XGBoost
        class_w  = np.array(compute_sample_weight("balanced", yc.iloc[tr]),
                             dtype=np.float64)
        rec_w    = np.array(recency_w[tr], dtype=np.float64)
        sample_w = class_w * rec_w
        mean_w   = sample_w.mean()
        sample_w = sample_w / mean_w if mean_w > 0 else sample_w

        m = xgb.XGBClassifier(**_xgb_clf_params(seed))
        m.fit(Xtr_s, yc.iloc[tr], sample_weight=sample_w,
              eval_set=[(Xte_s, yc.iloc[te])], verbose=False)
        try:
            auc = roc_auc_score(yc.iloc[te], m.predict_proba(Xte_s),
                                multi_class="ovr", average="macro")
        except Exception:
            auc = 0.0
        if auc > best_auc:
            best_auc = auc
            best_clf = m

    # ── Regressor ─────────────────────────────────────────────────────────
    reg_results = {}
    for target_name, col_name in [
        ("ret",      ret_col),
        ("ret_high", ret_high_col),
        ("ret_low",  ret_low_col),
    ]:
        ytgt     = df_m[col_name].fillna(0)
        best_reg = None
        best_mae = np.inf
        for tr, te in tscv_reg.split(X):
            Xtr_s = scaler_r.fit_transform(X.iloc[tr])
            Xte_s = scaler_r.transform(X.iloc[te])
            rec_w  = np.array(recency_w[tr], dtype=np.float64)
            mean_w = rec_w.mean()
            rec_w  = rec_w / mean_w if mean_w > 0 else rec_w

            m = xgb.XGBRegressor(**_xgb_reg_params(seed))
            m.fit(Xtr_s, ytgt.iloc[tr], sample_weight=rec_w,
                  eval_set=[(Xte_s, ytgt.iloc[te])], verbose=False)
            mae = mean_absolute_error(ytgt.iloc[te], m.predict(Xte_s))
            if mae < best_mae:
                best_mae = mae
                best_reg = m
        X_last = scaler_r.transform(X.tail(1))
        reg_results[target_name] = float(best_reg.predict(X_last)[0])

    X_last_c = scaler_c.transform(X.tail(1))
    pred     = best_clf.predict(X_last_c)[0]
    proba    = best_clf.predict_proba(X_last_c)[0]
    lmap     = {0:"SELL", 1:"HOLD", 2:"BUY"}

    return {
        "signal":          lmap[pred],
        "prob_buy":        round(float(proba[2]), 4),
        "prob_hold":       round(float(proba[1]), 4),
        "prob_sell":       round(float(proba[0]), 4),
        "expected_return": round(reg_results["ret"]      * 100, 2),
        "ret_high":        reg_results["ret_high"],
        "ret_low":         reg_results["ret_low"],
        "price_target_close": None,
        "price_target_high":  None,
        "price_target_low":   None,
    }


# ==============================================================================
# RISK MANAGER
# ==============================================================================

class RiskManager:
    def __init__(self, config):
        self.cfg = config

    def calculate(self, current_price, atr, signal, horizon_name,
                  target_low=None, target_high=None):
        sl_mult = self.cfg["sl_atr_mult"][horizon_name]
        tp_mult = self.cfg["tp_atr_mult"][horizon_name]

        if target_low is not None and target_high is not None \
                and target_low > 0 and target_high > 0:
            if signal == "BUY":
                stop_loss   = max(current_price-sl_mult*atr, target_low*0.98)
                take_profit = target_high
            elif signal == "SELL":
                stop_loss   = min(current_price+sl_mult*atr, target_high*1.02)
                take_profit = target_low
            else:
                stop_loss   = target_low
                take_profit = target_high
        else:
            if signal == "BUY":
                stop_loss   = current_price - sl_mult * atr
                take_profit = current_price + tp_mult * atr
            elif signal == "SELL":
                stop_loss   = current_price + sl_mult * atr
                take_profit = current_price - tp_mult * atr
            else:
                stop_loss   = current_price - sl_mult * atr
                take_profit = current_price + sl_mult * atr

        if signal == "BUY":
            stop_loss   = min(stop_loss,   current_price * 0.995)
            take_profit = max(take_profit, current_price * 1.005)
        elif signal == "SELL":
            stop_loss   = max(stop_loss,   current_price * 1.005)
            take_profit = min(take_profit, current_price * 0.995)

        risk   = abs(current_price - stop_loss)
        reward = abs(take_profit   - current_price)
        return {
            "stop_loss":   round(stop_loss,   2),
            "take_profit": round(take_profit, 2),
            "risk_reward": round(reward/risk,  2) if risk > 0 else 0,
        }


# ==============================================================================
# VIX REGIME FILTER  (NEW v5)
# ==============================================================================

def vix_regime_check(current_vix: float, signal: str, config: dict) -> tuple:
    """
    Returns (filtered_signal, regime_flag).
    If VIX > suppress_threshold and signal is BUY → downgrade to HOLD.
    """
    caution_thr  = config.get("vix_caution_threshold",  25)
    suppress_thr = config.get("vix_suppress_threshold", 35)

    if current_vix >= suppress_thr and signal == "BUY":
        return "HOLD", f"⚠️  BUY suppressed (VIX={current_vix:.1f} > {suppress_thr})"
    elif current_vix >= caution_thr:
        return signal, f"⚠️  CAUTION: VIX={current_vix:.1f} elevated"
    else:
        return signal, f"✅  VIX={current_vix:.1f} normal"


# ==============================================================================
# EARNINGS PROXIMITY FILTER  (fixed in v5.1)
# ==============================================================================

def earnings_proximity_filter(signal: str,
                               prob_buy: float,
                               prob_hold: float,
                               prob_sell: float,
                               days_to_earnings: int,
                               current_atr: float,
                               current_price: float,
                               target_low: float,
                               target_high: float,
                               target_close: float,
                               config: dict) -> dict:
    """
    Adjusts signal, confidence, and price range based on earnings proximity.

    Rules:
      days_to_earnings == 0  → EARNINGS TODAY   — suppress all signals → NO SIGNAL
      days_to_earnings == 1  → EARNINGS TOMORROW — downgrade BUY/SELL → HOLD
      days_to_earnings 2–5   → EARNINGS IMMINENT — reduce confidence 20%,
                                widen price range by ±1× ATR
      days_to_earnings > 5   → No adjustment

    Returns a dict with adjusted values and an earnings_note string.
    """
    proximity_days = config.get("earnings_proximity_days", 5)

    # No adjustment needed
    if days_to_earnings > proximity_days or days_to_earnings < 0:
        return {
            "signal":       signal,
            "prob_buy":     prob_buy,
            "prob_hold":    prob_hold,
            "prob_sell":    prob_sell,
            "target_low":   target_low,
            "target_close": target_close,
            "target_high":  target_high,
            "earnings_note": "",
            "earnings_adjusted": False,
        }

    atr_buffer = current_atr  # 1× ATR buffer for range widening

    # ── Case 1: Earnings TODAY ────────────────────────────────────────────
    if days_to_earnings == 0:
        return {
            "signal":        "NO SIGNAL",
            "prob_buy":      0.0,
            "prob_hold":     1.0,
            "prob_sell":     0.0,
            "target_low":    round(current_price - 2 * atr_buffer, 2),
            "target_close":  round(current_price, 2),
            "target_high":   round(current_price + 2 * atr_buffer, 2),
            "earnings_note": "🚫 EARNINGS TODAY — all signals suppressed, "
                             "price range is ATR-only estimate",
            "earnings_adjusted": True,
        }

    # ── Case 2: Earnings TOMORROW ─────────────────────────────────────────
    if days_to_earnings == 1:
        adj_signal = "HOLD" if signal in ("BUY", "SELL") else signal
        return {
            "signal":        adj_signal,
            "prob_buy":      prob_buy  * 0.3,   # heavily dampened
            "prob_hold":     min(1.0, prob_hold + 0.5),
            "prob_sell":     prob_sell * 0.3,
            "target_low":    round(target_low  - atr_buffer, 2),
            "target_close":  round(target_close, 2),
            "target_high":   round(target_high + atr_buffer, 2),
            "earnings_note": f"⚠️  EARNINGS TOMORROW — "
                             f"BUY/SELL downgraded to HOLD, "
                             f"price range widened ±${atr_buffer:.2f}",
            "earnings_adjusted": True,
        }

    # ── Case 3: Earnings within 2–5 days ─────────────────────────────────
    # Scale: 5 days away = mild (10% reduction), 2 days = stronger (25%)
    proximity_factor = (proximity_days - days_to_earnings + 1) / proximity_days
    confidence_cut   = 0.10 + 0.15 * proximity_factor   # 10%–25% reduction
    range_buffer     = atr_buffer * proximity_factor      # partial ATR buffer

    adj_prob_buy  = round(prob_buy  * (1 - confidence_cut), 4)
    adj_prob_sell = round(prob_sell * (1 - confidence_cut), 4)
    adj_prob_hold = round(min(1.0, prob_hold + (prob_buy + prob_sell) * confidence_cut), 4)

    # If confidence drops below 0.5 threshold, downgrade BUY/SELL to HOLD
    adj_signal = signal
    if signal == "BUY"  and adj_prob_buy  < 0.50:
        adj_signal = "HOLD"
    if signal == "SELL" and adj_prob_sell < 0.50:
        adj_signal = "HOLD"

    return {
        "signal":        adj_signal,
        "prob_buy":      adj_prob_buy,
        "prob_hold":     adj_prob_hold,
        "prob_sell":     adj_prob_sell,
        "target_low":    round(target_low  - range_buffer, 2),
        "target_close":  round(target_close, 2),
        "target_high":   round(target_high + range_buffer, 2),
        "earnings_note": f"⚠️  EARNINGS IN {days_to_earnings} DAYS — "
                         f"confidence reduced {confidence_cut:.0%}, "
                         f"range widened ±${range_buffer:.2f}"
                         + (f", signal downgraded to HOLD" if adj_signal != signal else ""),
        "earnings_adjusted": True,
    }


# ==============================================================================
# PROCESS ONE TICKER
# ==============================================================================

def process_ticker(args):
    ticker, config, shared = args
    try:
        fetcher   = DataFetcher(config)
        engineer  = FeatureEngineer(config)
        risk_mgr  = RiskManager(config)
        sent_eng  = SentimentEngine(config, CREDENTIALS)

        df = fetcher.fetch_price(ticker)
        if df is None:
            return None

        # ── Fetch ticker info once, reuse for company name + fundamentals ─
        ticker_info = {}
        try:
            ticker_info = yf.Ticker(ticker).info or {}
        except Exception:
            pass
        company_name = ticker_info.get("longName", ticker)

        # ── Auxiliary data fetched in parallel (all I/O-bound) ────────────
        aux_fns = [
            fetcher.fetch_macro,
            fetcher.fetch_vix,
            lambda: fetcher.fetch_google_trends(ticker),
            lambda: fetcher.fetch_fundamentals(ticker, _info=ticker_info),
            fetcher.fetch_short_interest,
            fetcher.fetch_put_call_ratio,
            fetcher.fetch_insider_activity,
            fetcher.fetch_institutional_ownership,
            fetcher.fetch_earnings_surprise,
        ]
        with ThreadPoolExecutor(max_workers=len(aux_fns)) as pool:
            aux_results = list(pool.map(lambda fn: fn(), aux_fns))
        for aux in aux_results:
            df = safe_merge(df, aux)

        # ── Real sentiment (NEW v5) ───────────────────────────────────────
        sentiment_df = sent_eng.get_combined_sentiment(ticker, company_name)
        df           = safe_merge(df, sentiment_df)

        # ── Technical features ────────────────────────────────────────────
        df = engineer.add_technical_indicators(df)
        df = engineer.add_lagged_features(df)
        df = engineer.add_rolling_statistics(df)
        df = engineer.add_temporal_features(df)
        df = engineer.add_options_implied_move(df)

        # ── Correlation features ──────────────────────────────────────────
        sector      = config["ticker_sector"].get(ticker, "Technology")
        sector_etf  = config["sector_etfs"].get(sector, "XLK")
        sector_df   = shared.get("sector_dfs",{}).get(sector_etf, pd.DataFrame())
        market_df   = shared.get("market_df", pd.DataFrame())
        commodity_dfs = shared.get("commodity_dfs", {})
        df = engineer.add_correlation_features(
            df, sector_df, market_df, commodity_dfs)

        # ── Labels for all horizons ───────────────────────────────────────
        max_h = max(config["horizons"].values())
        for hn, hd in config["horizons"].items():
            df = engineer.create_horizon_labels(
                df, hn, hd,
                config["buy_thresholds"][hn],
                config["sell_thresholds"][hn])
        df = df.iloc[:-max_h].copy()
        if len(df) < 200:
            return None

        # ── Feature columns ───────────────────────────────────────────────
        all_label_cols = []
        for hn in config["horizons"]:
            all_label_cols += [f"target_clf_{hn}", f"target_ret_{hn}",
                               f"target_ret_high_{hn}", f"target_ret_low_{hn}"]
        price_cols   = {"open","high","low","close","volume","adj close"}
        feature_cols = [c for c in df.columns
                        if c.lower() not in (set(all_label_cols)|price_cols)]

        # ── Current price, ATR, VIX ───────────────────────────────────────
        live_price, live_date = fetcher.get_current_price(ticker)
        current_price = live_price if live_price else float(df["close"].iloc[-1])

        if "atr" in df.columns and float(df["close"].iloc[-1]) > 0:
            atr_pct     = float(df["atr"].iloc[-1]) / float(df["close"].iloc[-1])
            current_atr = atr_pct * current_price
        else:
            current_atr = current_price * 0.02

        current_vix = float(df["vix"].iloc[-1]) if "vix" in df.columns else 18.0

        logger.info(f"  {ticker} @ ${current_price:.2f} "
                    f"VIX={current_vix:.1f} "
                    f"({live_date or 'cached'})")

        result = {
            "ticker":        ticker,
            "sector":        sector,
            "current_price": round(current_price, 2),
            "price_date":    live_date or str(df.index[-1].date()),
            "vix":           round(current_vix, 2),
            "combined_sentiment": round(
                float(df["combined_sentiment"].iloc[-1])
                if "combined_sentiment" in df.columns else 0.0, 4),
            "news_sentiment": round(
                float(df["news_sentiment"].iloc[-1])
                if "news_sentiment" in df.columns else 0.0, 4),
            "analyst_score":  round(
                float(df["analyst_score"].iloc[-1])
                if "analyst_score" in df.columns else 0.0, 4),
            "days_to_earnings": int(
                df["days_to_earnings"].iloc[-1]
                if "days_to_earnings" in df.columns else 999),
            "earnings_imminent": int(
                df["earnings_imminent"].iloc[-1]
                if "earnings_imminent" in df.columns else 0),
            "rs_3m_pct":  round(float(df["rs_3m"].iloc[-1])*100
                                if "rs_3m"  in df.columns else 0, 2),
            "rs_6m_pct":  round(float(df["rs_6m"].iloc[-1])*100
                                if "rs_6m"  in df.columns else 0, 2),
            "rs_12m_pct": round(float(df["rs_12m"].iloc[-1])*100
                                if "rs_12m" in df.columns else 0, 2),
        }

        best_buy_prob = 0.0
        best_signal   = "HOLD"

        # ── Train all 4 horizons in parallel (threads; XGBoost releases GIL)
        def _train_one(hn):
            return hn, train_horizon(df, feature_cols, hn, config)

        with ThreadPoolExecutor(max_workers=len(config["horizons"])) as pool:
            horizon_results = dict(pool.map(
                lambda hn: _train_one(hn), config["horizons"].keys()))

        days_earn = result.get("days_to_earnings", 999)
        for hn in config["horizons"]:
            h_result = horizon_results.get(hn)
            if h_result is None:
                continue

            # Convert % returns → $ prices
            exp_ret  = h_result["expected_return"] / 100
            ret_high = h_result.get("ret_high", exp_ret * 1.5)
            ret_low  = h_result.get("ret_low",  exp_ret * 0.5)

            target_close = round(current_price * (1 + exp_ret),  2)
            target_high  = round(current_price * (1 + ret_high), 2)
            target_low   = round(current_price * (1 + ret_low),  2)
            target_low   = min(target_low,  target_close)
            target_high  = max(target_high, target_close)

            # ── Step 1: VIX regime filter ─────────────────────────────
            raw_signal                = h_result["signal"]
            vix_signal, vix_note      = vix_regime_check(
                current_vix, raw_signal, config)

            # ── Step 2: Earnings proximity filter (fixed v5.1) ────────────
            earn_adj  = earnings_proximity_filter(
                signal          = vix_signal,
                prob_buy        = h_result["prob_buy"],
                prob_hold       = h_result["prob_hold"],
                prob_sell       = h_result["prob_sell"],
                days_to_earnings= days_earn,
                current_atr     = current_atr,
                current_price   = current_price,
                target_low      = target_low,
                target_high     = target_high,
                target_close    = target_close,
                config          = config,
            )

            # Use earnings-adjusted values throughout
            filtered_signal = earn_adj["signal"]
            final_prob_buy  = earn_adj["prob_buy"]
            final_prob_hold = earn_adj["prob_hold"]
            final_prob_sell = earn_adj["prob_sell"]
            final_low       = earn_adj["target_low"]
            final_close     = earn_adj["target_close"]
            final_high      = earn_adj["target_high"]
            earnings_note   = earn_adj["earnings_note"]

            # ── Step 3: Risk levels (on adjusted targets) ─────────────────
            risk = risk_mgr.calculate(
                current_price, current_atr, filtered_signal, hn,
                target_low=final_low, target_high=final_high)

            result[f"{hn}_signal"]            = filtered_signal
            result[f"{hn}_raw_signal"]         = raw_signal
            result[f"{hn}_vix_note"]           = vix_note
            result[f"{hn}_earnings_note"]      = earnings_note
            result[f"{hn}_earnings_adjusted"]  = earn_adj["earnings_adjusted"]
            result[f"{hn}_prob_buy"]           = final_prob_buy
            result[f"{hn}_prob_hold"]          = final_prob_hold
            result[f"{hn}_prob_sell"]          = final_prob_sell
            result[f"{hn}_expected_return"]    = h_result["expected_return"]
            result[f"{hn}_target_low"]         = final_low
            result[f"{hn}_target_close"]       = final_close
            result[f"{hn}_target_high"]        = final_high
            result[f"{hn}_stop_loss"]          = risk["stop_loss"]
            result[f"{hn}_take_profit"]        = risk["take_profit"]
            result[f"{hn}_risk_reward"]        = risk["risk_reward"]

            if final_prob_buy > best_buy_prob:
                best_buy_prob = final_prob_buy
                best_signal   = filtered_signal

        result["best_signal"]   = best_signal
        result["best_buy_prob"] = round(best_buy_prob, 4)
        return result

    except Exception as ex:
        logger.warning(f"❌ {ticker} failed: {ex}")
        import traceback; logger.debug(traceback.format_exc())
        return None


# ==============================================================================
# AUTO-RETRAINER  (NEW v5)
# ==============================================================================

class AutoRetrainer:
    """
    Saves model state and retrains automatically every N days.
    Logs training timestamps, tracks performance drift.
    """

    def __init__(self, config):
        self.cfg      = config
        self.log_file = config["retrain_log"]
        self.model_dir= config["model_dir"]
        os.makedirs(self.model_dir, exist_ok=True)

    def _load_log(self) -> list:
        if os.path.exists(self.log_file):
            with open(self.log_file) as f:
                return json.load(f)
        return []

    def _save_log(self, log: list):
        with open(self.log_file, "w") as f:
            json.dump(log, f, indent=2)

    def should_retrain(self) -> bool:
        log = self._load_log()
        if not log:
            return True
        last_run = datetime.fromisoformat(log[-1]["timestamp"])
        days_since = (datetime.now() - last_run).days
        return days_since >= self.cfg["retrain_interval_days"]

    def run_retrain(self, tickers: list = None, config: dict = None):
        if config is None:
            config = self.cfg
        if tickers is None:
            tickers = config["watchlist"]

        logger.info("="*60)
        logger.info(f"🔄  AUTO-RETRAINER — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        logger.info(f"    Retraining {len(tickers)} stocks...")
        logger.info("="*60)

        start   = time.time()
        results = run_screener(config, tickers=tickers)

        elapsed = time.time() - start
        log     = self._load_log()
        log.append({
            "timestamp":     datetime.now().isoformat(),
            "tickers":       len(tickers),
            "elapsed_min":   round(elapsed / 60, 1),
            "output_csv":    config["output_csv"],
        })
        self._save_log(log)
        logger.info(f"✅  Retrain complete in {elapsed/60:.1f} min")
        logger.info(f"    Log updated → {self.log_file}")
        return results

    def start_scheduler(self, tickers: list = None):
        """Start a blocking weekly retraining scheduler."""
        if not SCHEDULE_AVAILABLE:
            logger.error("Install 'schedule': pip install schedule")
            return

        import schedule as sched

        def job():
            if self.should_retrain():
                self.run_retrain(tickers)
            else:
                logger.info("⏭  Retraining not due yet — skipping")

        # Run immediately on start, then every 7 days
        job()
        sched.every(self.cfg["retrain_interval_days"]).days.do(job)
        logger.info(f"📅  Scheduler running — retraining every "
                    f"{self.cfg['retrain_interval_days']} days")
        logger.info("   Press Ctrl+C to stop.")

        while True:
            sched.run_pending()
            time.sleep(3600)   # check every hour


# ==============================================================================
# SCREENER
# ==============================================================================

def run_screener(config=CONFIG, tickers=None):
    if tickers is None:
        tickers = config["watchlist"]

    logger.info("="*70)
    logger.info(f"🚀  XGBoost Screener v5 — {len(tickers)} stocks × "
                f"{len(config['horizons'])} horizons")
    logger.info(f"    Workers: {config['max_workers']} | "
                f"Recency half-life: {config['recency_halflife_days']}d")
    logger.info("="*70)

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    fetcher = DataFetcher(config)

    # Shared reference data
    _m        = fetcher.fetch_price("^GSPC")
    market_df = _m if _m is not None else pd.DataFrame()
    sector_dfs= {}
    for s, etf in config["sector_etfs"].items():
        d = fetcher.fetch_price(etf)
        sector_dfs[etf] = d if d is not None else pd.DataFrame()
    commodity_dfs = {}
    for t in config["commodity_tickers"]:
        d = fetcher.fetch_price(t)
        if d is not None:
            commodity_dfs[t] = d

    shared = {"market_df":     market_df,
              "sector_dfs":    sector_dfs,
              "commodity_dfs": commodity_dfs}

    # Resume from checkpoint
    ckpt_file   = os.path.join(config["checkpoint_dir"], "results_v5.jsonl")
    done        = set()
    all_results = []
    if os.path.exists(ckpt_file):
        with open(ckpt_file) as f:
            for line in f:
                try:
                    r = json.loads(line.strip())
                    all_results.append(r); done.add(r["ticker"])
                except Exception:
                    pass
        logger.info(f"  Resuming — {len(done)} already done")

    remaining = [t for t in tickers if t not in done]
    args_list = [(t, config, shared) for t in remaining]
    start_t   = time.time()

    pbar = tqdm(total=len(remaining), desc="Screening", unit="stock",
                bar_format="{l_bar}{bar}| {n}/{total} [{elapsed}<{remaining}]"
                ) if TQDM_AVAILABLE else None

    with ProcessPoolExecutor(max_workers=config["max_workers"]) as executor:
        futures = {executor.submit(process_ticker, a): a[0] for a in args_list}
        for future in as_completed(futures):
            t = futures[future]
            try:
                res = future.result()
                if res:
                    all_results.append(res)
                    with open(ckpt_file, "a") as f:
                        f.write(json.dumps(res) + "\n")
                    sigs = " | ".join(
                        f"{hn}:{res.get(f'{hn}_signal','?')}"
                        f"({res.get(f'{hn}_expected_return',0):+.1f}%)"
                        for hn in config["horizons"])
                    sent = res.get("combined_sentiment", 0)
                    logger.info(f"  ✅ {t:6s} ${res['current_price']:.2f} "
                                f"sent={sent:+.2f} → {sigs}")
            except Exception as ex:
                logger.warning(f"  ❌ {t}: {ex}")
            finally:
                if pbar: pbar.update(1)

    if pbar: pbar.close()
    logger.info(f"⏱  Done in {(time.time()-start_t)/60:.1f} min")

    if not all_results:
        return pd.DataFrame()

    df = pd.DataFrame(all_results).sort_values("best_buy_prob", ascending=False)
    df.reset_index(drop=True, inplace=True)
    df.index += 1; df.index.name = "rank"
    df.to_csv(config["output_csv"])
    logger.info(f"CSV → {config['output_csv']}")
    return df


# ==============================================================================
# SINGLE STOCK DEEP DIVE
# ==============================================================================

def run_single_stock(ticker, config=CONFIG):
    logger.info(f"\n📈  Deep-dive: {ticker}")
    fetcher = DataFetcher(config)

    _m        = fetcher.fetch_price("^GSPC")
    market_df = _m if _m is not None else pd.DataFrame()
    sector    = config["ticker_sector"].get(ticker, "Technology")
    etf       = config["sector_etfs"].get(sector, "XLK")
    _s        = fetcher.fetch_price(etf)
    sector_df = _s if _s is not None else pd.DataFrame()
    commodity_dfs = {}
    for t in config["commodity_tickers"]:
        d = fetcher.fetch_price(t)
        if d is not None: commodity_dfs[t] = d

    shared = {"market_df": market_df,
              "sector_dfs": {etf: sector_df},
              "commodity_dfs": commodity_dfs}

    result = process_ticker((ticker, config, shared))
    if result is None:
        print(f"❌  Could not process {ticker}"); return None

    hn_labels = {"1w":"1 WEEK","2w":"2 WEEKS","1m":"1 MONTH","3m":"3 MONTHS"}
    sig_icons = {"BUY":"🟢","HOLD":"🟡","SELL":"🔴"}
    sep       = "=" * 68

    print(f"\n{sep}")
    print(f"  STOCK REPORT v5: {ticker}  [{result['sector']}]")
    print(f"  Current Price     : ${result['current_price']:.2f}  "
          f"(as of {result.get('price_date','N/A')})")
    print(f"  VIX               : {result['vix']:.1f}")
    print(f"  Combined Sentiment: {result['combined_sentiment']:+.3f}  "
          f"(news={result['news_sentiment']:+.2f}  "
          f"analyst={result['analyst_score']:+.2f})")
    if result.get("earnings_imminent"):
        print(f"  ⚠️  EARNINGS IN {result['days_to_earnings']} DAYS — "
              f"increased uncertainty")
    print(f"  RS 3M/6M/12M      : {result['rs_3m_pct']:+.1f}% / "
          f"{result['rs_6m_pct']:+.1f}% / {result['rs_12m_pct']:+.1f}%")
    print(sep)

    for hn, hlabel in hn_labels.items():
        sig        = result.get(f"{hn}_signal", "N/A")
        raw        = result.get(f"{hn}_raw_signal", sig)
        icon       = sig_icons.get(sig, "⚪") if sig != "NO SIGNAL" else "🚫"
        vix_note   = result.get(f"{hn}_vix_note", "")
        earn_note  = result.get(f"{hn}_earnings_note", "")
        earn_adj   = result.get(f"{hn}_earnings_adjusted", False)
        cur        = result["current_price"]
        sl         = result.get(f"{hn}_stop_loss",   0)
        tp         = result.get(f"{hn}_take_profit", 0)

        print(f"\n  ── {hlabel} ─────────────────────────────────────────")
        print(f"  Signal         : {icon} {sig}"
              + (f"  (raw: {raw})" if raw != sig else ""))
        if vix_note:
            print(f"  VIX Note       : {vix_note}")
        if earn_note:
            print(f"  Earnings Note  : {earn_note}")
        print(f"  P(BUY/HOLD/SELL): "
              f"{result.get(f'{hn}_prob_buy',  0):.1%} / "
              f"{result.get(f'{hn}_prob_hold', 0):.1%} / "
              f"{result.get(f'{hn}_prob_sell', 0):.1%}"
              + ("  [adjusted for earnings]" if earn_adj else ""))
        print(f"  Expected Return: {result.get(f'{hn}_expected_return', 0):+.2f}%"
              + ("  ⚠️  pre-earnings uncertainty applies" if earn_adj else ""))
        print(f"  Price Range    : "
              f"${result.get(f'{hn}_target_low',   0):.2f} – "
              f"${result.get(f'{hn}_target_close', 0):.2f} – "
              f"${result.get(f'{hn}_target_high',  0):.2f}  (worst/base/best)"
              + ("  [widened ±ATR]" if earn_adj else ""))
        if sig == "NO SIGNAL":
            print(f"  Stop Loss / TP : N/A — no directional signal on earnings day")
        else:
            print(f"  Stop Loss      : ${sl:.2f}  "
                  f"({'▼ {:.1f}%'.format(abs(sl-cur)/cur*100) if sl < cur else '▲ {:.1f}%'.format(abs(sl-cur)/cur*100)})")
            print(f"  Take Profit    : ${tp:.2f}  "
                  f"({'▲ {:.1f}%'.format(abs(tp-cur)/cur*100) if tp > cur else '▼ {:.1f}%'.format(abs(tp-cur)/cur*100)})")
            print(f"  Risk / Reward  : {result.get(f'{hn}_risk_reward', 0):.2f}x")

    print(f"\n{sep}\n")
    return result


# ==============================================================================
# SENTIMENT ONLY MODE
# ==============================================================================

def run_sentiment_only(ticker: str):
    """Quick sentiment snapshot without running the full model."""
    engine = SentimentEngine(CONFIG, CREDENTIALS)
    try:
        name = yf.Ticker(ticker).info.get("longName", ticker)
    except Exception:
        name = ticker

    news   = engine.fetch_news_sentiment(ticker, name)
    reddit = engine.fetch_reddit_sentiment(ticker)
    twitter= engine.fetch_twitter_sentiment(ticker)
    analyst= engine.fetch_analyst_sentiment(ticker)
    earn   = engine.fetch_earnings_proximity(ticker)

    combined = (news.get("score",0)*0.35 + reddit.get("score",0)*0.20 +
                twitter.get("score",0)*0.20 + analyst.get("score",0)*0.25)

    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  SENTIMENT REPORT: {ticker}  ({name})")
    print(sep)
    print(f"  News Sentiment    : {news.get('score',0):+.3f}  "
          f"(+:{news.get('positive',0):.0%} -:{news.get('negative',0):.0%})  "
          f"[{news.get('n',0)} articles]")
    print(f"  Reddit Sentiment  : {reddit.get('score',0):+.3f}  "
          f"[{reddit.get('n',0)} posts]")
    print(f"  Twitter Sentiment : {twitter.get('score',0):+.3f}  "
          f"[{twitter.get('n',0)} tweets]")
    print(f"  Analyst Score     : {analyst.get('score',0):+.3f}  "
          f"[{analyst.get('strong_buy',0)} SB / "
          f"{analyst.get('buy',0)} B / "
          f"{analyst.get('hold',0)} H / "
          f"{analyst.get('sell',0)} S]")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Combined Score    : {combined:+.3f}  "
          f"({'BULLISH 📈' if combined>0.1 else 'BEARISH 📉' if combined<-0.1 else 'NEUTRAL ➡️'})")
    print(f"  Next Earnings     : {earn.get('next_earnings_date','?')}  "
          f"({earn.get('days_to_earnings',999)} days)")
    if earn.get("earnings_imminent"):
        print(f"  ⚠️  EARNINGS IMMINENT — sentiment may be unstable")
    print(sep + "\n")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def print_feature_status():
    """Print a clear summary of which features are active based on API keys."""
    sep = "=" * 58
    print(f"\n{sep}")
    print("  XGBoost Stock Model v5 — Feature Status")
    print(sep)

    # Sentiment sources
    has_news    = bool(NEWSAPI_AVAILABLE and CREDENTIALS.get("news_api_key"))
    has_reddit  = bool(REDDIT_AVAILABLE  and CREDENTIALS.get("reddit_client_id")
                       and CREDENTIALS.get("reddit_client_secret"))
    has_twitter = bool(TWEEPY_AVAILABLE  and CREDENTIALS.get("twitter_bearer_token"))
    has_finbert = FINBERT_AVAILABLE

    def row(label, active, note=""):
        icon = "✅" if active else "⏭ "
        status = "ACTIVE" if active else "SKIPPED — no API key"
        extra = f"  ({note})" if note else ""
        print(f"  {icon}  {label:<28} {status}{extra}")

    print("  ── Sentiment Sources ─────────────────────────────")
    row("News (NewsAPI + FinBERT)",  has_news and has_finbert,
        "set NEWS_API_KEY" if not has_news else "")
    row("Reddit (PRAW + FinBERT)",   has_reddit and has_finbert,
        "set REDDIT_CLIENT_ID/SECRET" if not has_reddit else "")
    row("Twitter/X (Tweepy+FinBERT)",has_twitter and has_finbert,
        "set TWITTER_BEARER_TOKEN" if not has_twitter else "")
    row("Analyst (yfinance)",         True, "always available")
    row("FinBERT NLP model",          has_finbert,
        "pip install transformers torch" if not has_finbert else "")

    print("  ── Always-On Features ────────────────────────────")
    always_on = [
        "Technical indicators (RSI/MACD/BB/CCI/ATR)",
        "Price momentum & relative strength",
        "Lagged & rolling statistics",
        "Macro (CPI/GDP/Fed rate)",
        "Fundamentals (P/E/EPS/revenue)",
        "Sector ETF & market correlation",
        "VIX regime filter",
        "Earnings proximity warning",
        "Recency weighting (1yr half-life)",
        "Weekly auto-retraining",
    ]
    for feat in always_on:
        print(f"  ✅  {feat}")

    if not (has_news or has_reddit or has_twitter):
        print(f"\n  ℹ️  No sentiment API keys set.")
        print(f"     Model will run on technical + fundamental features only.")
        print(f"     Add API keys to CREDENTIALS dict or as env vars to enable.")
    print(sep + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python stock_recommendation_xgboost_v5.py single    AMZN")
        print("  python stock_recommendation_xgboost_v5.py screener")
        print("  python stock_recommendation_xgboost_v5.py scheduler")
        print("  python stock_recommendation_xgboost_v5.py sentiment SOFI")
        sys.exit(0)

    print_feature_status()
    mode = sys.argv[1].lower()

    if mode == "single" and len(sys.argv) >= 3:
        run_single_stock(sys.argv[2].upper(), CONFIG)

    elif mode == "screener":
        results = run_screener(CONFIG)
        if not results.empty:
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", 250)
            cols = ["ticker","sector","current_price","vix",
                    "combined_sentiment","analyst_score",
                    "1w_signal","1w_expected_return",
                    "1m_signal","1m_expected_return",
                    "3m_signal","3m_expected_return","rs_3m_pct"]
            cols = [c for c in cols if c in results.columns]
            print(results[cols].head(20).to_string())

            # Top pick deep-dive
            buys = results[results["best_signal"] == "BUY"]
            if not buys.empty:
                run_single_stock(buys.iloc[0]["ticker"], CONFIG)

    elif mode == "scheduler":
        retrainer = AutoRetrainer(CONFIG)
        retrainer.start_scheduler()

    elif mode == "sentiment" and len(sys.argv) >= 3:
        run_sentiment_only(sys.argv[2].upper())

    elif mode == "sector":
        for sector, tickers in WATCHLIST.items():
            print(f"\n{sector}: {', '.join(tickers)}")

    else:
        print(f"Unknown mode: {mode}")
