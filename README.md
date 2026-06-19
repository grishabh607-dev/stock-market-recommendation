# XGBoost Stock Recommendation System

A machine learning system that trains XGBoost models on historical stock data and generates **BUY / HOLD / SELL signals**, **price range forecasts**, and **risk management levels** across four forecast horizons — 1 week, 2 weeks, 1 month, and 3 months.

---

## Files

| File | Purpose |
|---|---|
| `stock_recommendation_xgboost_v5.py` | Main model — training, prediction, screener, scheduler |
| `simulation_framework.py` | Testing suite — backtest, walk-forward, paper trading |

---

## Quick Start

```bash
# 1. Install dependencies
pip install xgboost yfinance ta scikit-learn pandas numpy tqdm
pip install transformers torch        # FinBERT sentiment (optional)
pip install newsapi-python            # News API (optional)
pip install praw                      # Reddit (optional)
pip install tweepy                    # Twitter/X (optional)
pip install schedule                  # Weekly retraining (optional)
pip install fredapi pytrends          # Macro data + Google Trends (optional)

# 2. Run prediction for a single stock
python stock_recommendation_xgboost_v5.py single AAPL

# 3. Run full screener across all stocks
python stock_recommendation_xgboost_v5.py screener
```

---

## Commands

### Main Model

```bash
# Single stock — full report across all 4 horizons
python stock_recommendation_xgboost_v5.py single AMZN
python stock_recommendation_xgboost_v5.py single SOFI

# Full screener — trains + predicts all ~100 stocks, saves CSV + HTML report
python stock_recommendation_xgboost_v5.py screener

# Sentiment snapshot only — fast, no model training
python stock_recommendation_xgboost_v5.py sentiment NVDA

# Weekly auto-retrainer — runs forever, retrains every 7 days
python stock_recommendation_xgboost_v5.py scheduler

# List all sectors and their tickers
python stock_recommendation_xgboost_v5.py sector
```

### Simulation & Testing

```bash
# Backtest — replay historical data, measure signal accuracy
python simulation_framework.py backtest SOFI

# Walk-forward — rolling train/test windows, mimics real retraining
python simulation_framework.py walkforward AAPL

# Paper trading — generate live signals, track outcomes automatically
python simulation_framework.py paper AAPL MSFT SOFI NVDA

# Full suite — all three modes for one stock
python simulation_framework.py all SOFI
```

---

## API Keys (Optional)

The model runs without any API keys. When keys are missing, those specific sentiment sources are **skipped entirely** — no mock data is injected.

| Key | Source | What it enables | Free tier |
|---|---|---|---|
| `NEWS_API_KEY` | [newsapi.org](https://newsapi.org) | Real news headlines per ticker | 100 req/day |
| `REDDIT_CLIENT_ID` + `REDDIT_CLIENT_SECRET` | [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) | Reddit post sentiment | Free |
| `TWITTER_BEARER_TOKEN` | [developer.twitter.com](https://developer.twitter.com) | Tweet sentiment | Free tier |
| `FRED_API_KEY` | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) | Real CPI/GDP/Fed rate data | Free |

Set as environment variables:

```bash
export NEWS_API_KEY="your_key"
export REDDIT_CLIENT_ID="your_id"
export REDDIT_CLIENT_SECRET="your_secret"
export TWITTER_BEARER_TOKEN="your_token"
export FRED_API_KEY="your_key"
```

Or set directly in the `CREDENTIALS` dict at the top of `stock_recommendation_xgboost_v5.py`.

---

## Watchlist — 102 Stocks across 5 Sectors

| Sector | ETF | Count | Sample tickers |
|---|---|---|---|
| Technology | XLK | 20 | AAPL, MSFT, NVDA, GOOGL, META, TSLA, AMD, PLTR |
| Healthcare | XLV | 20 | JNJ, UNH, LLY, PFE, ABBV, ISRG, VRTX, REGN |
| Finance | XLF | 23 | JPM, GS, V, MA, BLK, SOFI, COIN, HOOD, BX |
| Energy | XLE | 20 | XOM, CVX, COP, SLB, OXY, HAL, LNG, MRO |
| Consumer | XLP | 20 | WMT, COST, MCD, NKE, HD, SBUX, TGT, CMG |

### Adding new stocks

Open `stock_recommendation_xgboost_v5.py` and edit **lines 166–189** (`WATCHLIST` dict):

```python
WATCHLIST = {
    "Finance": [
        "JPM", "BAC", ...,
        "YOUR_TICKER",    # ← add here under the right sector
    ],
    ...
}
```

`ALL_TICKERS` and `TICKER_SECTOR` are auto-built from `WATCHLIST` — no other changes needed.

To add a completely new sector, add a new key to `WATCHLIST` and a matching entry to `CONFIG["sector_etfs"]` at line 205.

---

## Features — 120+ inputs across 13 categories

### Technical Indicators (18 features)
RSI (14d), MACD + signal + diff, Bollinger Bands (high/low/width/%B), CCI (20d), ATR (14d), OBV, EMA 9/21/50/200, EMA crossover, Golden Cross, Volume ROC

### Price Momentum (8 features)
Price ROC 1d/5d/10d, High-Low spread, Intraday momentum, Relative Strength 3m/6m/12m

### Lagged Features (12 features)
Close/Volume/RSI lagged at 5, 10, 20, 50 days

### Rolling Statistics (16 features)
Rolling mean, std, volume mean, and z-score over 5/10/20/50 day windows

### Macroeconomic (4 features)
CPI (inflation), Unemployment rate, Fed funds rate, GDP growth — via FRED API when key is provided, otherwise mock fallback

### Sentiment — NEW in v5 (15 features)
News sentiment score/positive/negative/volume (NewsAPI + FinBERT), Reddit sentiment + volume, Twitter/X sentiment + volume, Analyst score + strong-buy/buy/hold/sell counts, Combined weighted sentiment score, Google Trends

> FinBERT is a BERT model specifically trained on financial text — it understands phrases like "beats estimates" and "guidance cut" correctly, unlike general-purpose sentiment models.

### Fundamentals (10 features)
P/E ratio, Debt/equity, Earnings growth, Revenue growth, EPS momentum, Revenue momentum, Dividend yield, Analyst rating, Price-to-book, Profit margin

### Options & Volatility (4 features)
VIX index, Put/call ratio, Implied move 1d, Implied move 5d

### Earnings History (5 features)
Earnings surprise, Beat/miss flag, Beat streak, Days to next earnings, Earnings imminent flag

### Correlation & Market (8 features)
Sector ETF relative return, Relative strength vs sector (20d), S&P 500 relative return, Beta (20d), Gold price + return, Oil price + return

### Alternative Data (3 features)
Short interest ratio, Insider trading signal, Institutional ownership change (13F delta)

### Temporal (8 features)
Day of week, Month, Quarter, Month-end flag, Quarter-end flag, Earnings season flag (Jan/Apr/Jul/Oct), Day-of-year sin/cos encoding

### Training Enhancements — NEW in v5
Exponential recency weighting (1-year half-life — last 12 months count far more than 2018 data), TimeSeriesSplit 5-fold cross-validation (no future leakage), Class balance weighting, Weekly auto-retraining via scheduler

---

## Model Architecture

### Two models trained per horizon per stock

| Model | Type | Predicts |
|---|---|---|
| Classifier | XGBoost multi-class | BUY / HOLD / SELL + probabilities |
| Regressor | XGBoost regression | Expected % return, price range (low/close/high) |

### Four forecast horizons

| Horizon | Trading days | BUY threshold | SELL threshold |
|---|---|---|---|
| 1 Week | 5 | +2% | −2% |
| 2 Weeks | 10 | +3% | −3% |
| 1 Month | 21 | +5% | −5% |
| 3 Months | 63 | +10% | −10% |

### XGBoost hyperparameters

```python
# Classifier
n_estimators=300, learning_rate=0.05, max_depth=6,
min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
gamma=0.1, reg_alpha=0.1, reg_lambda=1.0

# Regressor
n_estimators=300, learning_rate=0.05, max_depth=5,
subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1
```

### Price range methodology

The regressor predicts **% returns**, not absolute prices. Price targets are then calculated as:

```
Target Low   = current_price × (1 + predicted_low_return)
Target Close = current_price × (1 + predicted_base_return)
Target High  = current_price × (1 + predicted_high_return)
```

This makes predictions **scale-invariant** — they work correctly whether the stock trades at $5 or $5,000.

---

## Signal Safety Filters

Applied after the model generates its raw signal — these are guardrails, not training features.

### VIX Regime Filter

| VIX level | Action |
|---|---|
| VIX < 25 | No adjustment — normal market |
| VIX 25–35 | CAUTION flag added to signal |
| VIX > 35 | BUY signal automatically downgraded to HOLD |

### Earnings Proximity Filter

| Days to earnings | Action |
|---|---|
| > 5 days | No adjustment |
| 2–5 days | Confidence reduced 10–25%, price range widened ±ATR |
| 1 day | BUY/SELL downgraded to HOLD, range widened ±1×ATR |
| 0 days (today) | All signals suppressed → NO SIGNAL |

---

## Model Output per Stock per Horizon

```
── 1 WEEK FORECAST ─────────────────────────────────────
Signal         : 🟢 BUY
VIX Note       : ✅ VIX=18.4 normal
P(BUY/HOLD/SELL): 67.3% / 22.1% / 10.6%
Expected Return: +2.84%
Price Range    : $232.10 – $244.76 – $258.20  (worst/base/best)
Current Price  : $238.00
Stop Loss      : $232.10  (▼ 2.5% below entry)
Take Profit    : $258.20  (▲ 8.5% above entry)
Risk / Reward  : 3.40x
```

---

## Risk Management

Stop-loss and take-profit levels are:
- **Anchored** to the predicted price range (low → stop-loss, high → take-profit)
- **ATR-scaled** using the live current price, not stale historical ATR
- **Sanity-checked** so stop-loss is always below entry for BUY (and above for SELL)
- **Horizon-scaled** — longer horizons use wider ATR multipliers

| Horizon | Stop-loss multiplier | Take-profit multiplier |
|---|---|---|
| 1 Week | 1.5× ATR | 3.0× ATR |
| 2 Weeks | 2.0× ATR | 4.0× ATR |
| 1 Month | 2.5× ATR | 5.0× ATR |
| 3 Months | 3.0× ATR | 7.5× ATR |

---

## Screener Output Files

After running `screener`, two files are saved:

| File | Format | Contents |
|---|---|---|
| `screener_results_v5.csv` | CSV | All stocks, all horizons, all metrics — ranked by BUY probability |
| `screener_report_v5.html` | HTML | Dark-themed browser report — open in Chrome/Safari |

---

## Runtime

| Hardware | Screener runtime (102 stocks) |
|---|---|
| Mac M1/M2/M3 (8 cores) | 15–25 minutes |
| Mac M2 Pro / M3 Pro (10–12 cores) | 10–18 minutes |
| Mac M2 Max / M3 Max (12+ cores) | 8–14 minutes |
| Intel Mac i7 (4 cores) | 45–70 minutes |
| Intel Mac i9 (8 cores) | 25–40 minutes |

### Speed up tips

```python
# In CONFIG — reduce folds (fastest change, ~40% speed gain)
"n_splits": 3          # default is 5

# In _xgb_clf_params / _xgb_reg_params — reduce trees (~50% speed gain)
n_estimators=150       # default is 300
```

### Resume after crash

The screener saves each completed stock to `checkpoints_v5/results_v5.jsonl` immediately. Re-running the same command resumes from where it left off automatically.

To force a full fresh run:
```bash
rm -rf checkpoints_v5/
python stock_recommendation_xgboost_v5.py screener
```

---

## Simulation & Testing Framework

### Mode 1 — Backtest

Trains on 70% of historical data (2022–today), tests on remaining 30%. Simulates entering/exiting positions based on model signals.

```bash
python simulation_framework.py backtest SOFI
```

Outputs: win rate, total P&L, Sharpe ratio, max drawdown, equity sparkline, trade log CSV.

### Mode 2 — Walk-Forward

Rolls a training window forward month by month — trains on 24 months, tests on 3, advances, repeats. Most realistic simulation of real-world usage.

```bash
python simulation_framework.py walkforward AAPL
```

### Mode 3 — Paper Trading

Generates live signals for today, saves to `paper_trades.json`. On each subsequent run, automatically fetches the actual exit price for any signals whose target date has passed and marks them correct/incorrect.

```bash
python simulation_framework.py paper AAPL MSFT SOFI NVDA
```

**How exit price works:**
- Entry price = today's closing price (live from Yahoo Finance)
- Exit price = actual closing price on the target date (fetched automatically)
- Target date = today + (horizon_days × 1.4)
- BUY is correct if exit > entry
- SELL tracks directional accuracy only (no short selling simulated)

### Simulation config

```python
SIM_CONFIG = {
    "starting_capital":    10_000,    # $10,000 starting portfolio
    "position_size_pct":   0.20,      # 20% of capital per trade
    "commission_pct":      0.001,     # 0.1% commission per trade
    "slippage_pct":        0.0005,    # 0.05% slippage
    "min_confidence":      0.55,      # minimum P(BUY) to enter
    "horizon":             "1w",      # which horizon to trade
    "train_months":        24,        # walk-forward train window
    "test_months":         3,         # walk-forward test window
}
```

---

## Key Configuration

All main settings are in the `CONFIG` dict near the top of `stock_recommendation_xgboost_v5.py`:

| Setting | Default | Description |
|---|---|---|
| `start_date` | `"2018-01-01"` | Training data start — change to `"2010-01-01"` for more history |
| `forecast_horizon` | 5/10/21/63 days | Prediction windows — one model per horizon |
| `recency_halflife_days` | 365 | Recent data weighting — 1 year half-life |
| `recency_min_weight` | 0.15 | Oldest data still gets 15% weight |
| `vix_caution_threshold` | 25 | VIX level that adds caution flag |
| `vix_suppress_threshold` | 35 | VIX level that suppresses BUY signals |
| `earnings_proximity_days` | 5 | Days before earnings to start adjusting signals |
| `sentiment_cache_hours` | 6 | How long to cache sentiment before re-fetching |
| `retrain_interval_days` | 7 | Auto-retraining frequency |
| `max_workers` | CPU cores − 1 | Parallel workers for screener |
| `n_splits` | 5 | TimeSeriesSplit cross-validation folds |

---

## Known Limitations

| Scenario | Risk level | Reason |
|---|---|---|
| Black swan events (COVID crash, banking crisis) | High | Model trained on historical patterns — can't anticipate unseen events |
| Earnings day surprises | High | Earnings proximity filter suppresses signals but actual surprise is unpredictable |
| Short squeeze events (GME-style) | High | Pure sentiment momentum — no historical analogue |
| Regulatory / legal shocks | High | No feature captures SEC investigations or FDA rejections |
| Very new stocks (< 300 trading days) | High | Insufficient training data — model skips these |
| Macro regime shifts (rate hike cycles) | Medium | Rare in training data — recency weighting helps partially |
| Penny stocks / low volume stocks | Medium | Thin trading makes patterns unreliable |

> This model is intended as one analytical input among several — not a standalone trading decision system. Always verify signals against current news, upcoming earnings dates, and broader market conditions before acting.

---

## Project Structure

```
├── stock_recommendation_xgboost_v5.py   # Main model
├── simulation_framework.py              # Testing suite
├── README.md                            # This file
├── screener_results_v5.csv              # Generated after screener run
├── screener_report_v5.html              # Generated after screener run
├── screener_v5.log                      # Training + prediction log
├── paper_trades.json                    # Paper trading signal log
├── paper_trading_log.csv                # Paper trading CSV export
├── retrain_log.json                     # Auto-retrainer history
├── checkpoints_v5/
│   └── results_v5.jsonl                 # Per-stock checkpoint (resume support)
├── models_v5/                           # Saved model files
└── sentiment_cache/                     # Cached sentiment scores (6h TTL)
```

---

## Version History

| Version | Key additions |
|---|---|
| v1 | XGBoost classifier, basic technical indicators, single stock |
| v2 | Multi-stock screener, price range prediction, stop-loss/take-profit, insider/institutional features |
| v3 | 90 stocks, 5 sectors, parallel processing, sector ETFs, HTML report, checkpoint/resume |
| v4 | 4 forecast horizons (1w/2w/1m/3m), % return regression, price range anchored to live price |
| v5 | Real sentiment (NewsAPI/Reddit/Twitter + FinBERT), recency weighting, VIX regime filter, earnings proximity filter, weekly auto-retraining |
| v5.1 | Earnings proximity filter upgraded — confidence scaling, range widening, NO SIGNAL on earnings day |

---

## Disclaimer

This software is for educational and research purposes only. It does not constitute financial advice. Past model performance on historical data does not guarantee future results. Always consult a qualified financial advisor before making investment decisions.
