"""
==============================================================================
XGBoost Stock Model — Simulation & Testing Framework
==============================================================================
Three testing modes:

  1. BACKTEST          — replay 2018–today, measure how signals performed
  2. WALK-FORWARD      — train on N months, test on next M months, roll forward
  3. PAPER TRADING     — generate today's live signals, track daily (no real $)

Metrics tracked for all modes:
  ✅ Win Rate       — % of BUY/SELL calls that were correct
  ✅ P&L            — simulated dollar returns starting from $10,000
  ✅ Sharpe Ratio   — annualised return / annualised volatility
  ✅ Max Drawdown   — worst peak-to-trough equity drop

Usage:
  # Backtest one stock
  python simulation_framework.py backtest SOFI

  # Walk-forward one stock
  python simulation_framework.py walkforward AAPL

  # Paper trading (today's signals, saved to paper_trades.json)
  python simulation_framework.py paper AAPL MSFT SOFI NVDA

  # Full report across all three modes for one stock
  python simulation_framework.py all SOFI
==============================================================================
"""

import sys
import os

# Must be set before xgboost is imported to prevent OpenMP conflicts
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import warnings
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))


# ==============================================================================
# SIMULATION CONFIG
# ==============================================================================

SIM_CONFIG = {
    "starting_capital":    10_000,     # $ starting portfolio value
    "position_size_pct":   0.20,       # 20% of capital per trade
    "commission_pct":      0.001,      # 0.1% per trade (realistic broker fee)
    "slippage_pct":        0.0005,     # 0.05% slippage
    "min_confidence":      0.55,       # minimum P(BUY) to enter a trade
    "horizon":             "1w",       # which horizon to trade on
    "horizon_days":        5,          # trading days per trade

    # Walk-forward settings
    "train_months":        24,         # months of data to train on
    "test_months":         3,          # months to test before retraining
    "min_train_rows":      200,        # minimum rows needed to train

    # Backtest date range
    "backtest_start":      "2022-01-01",
    "backtest_end":        datetime.today().strftime("%Y-%m-%d"),

    # Paper trading output
    "paper_trades_file":   "paper_trades.json",
    "paper_log_file":      "paper_trading_log.csv",
}


# ==============================================================================
# METRICS ENGINE
# ==============================================================================

class MetricsEngine:
    """Calculates all performance metrics from a trades list."""

    @staticmethod
    def calculate(trades: list, starting_capital: float) -> dict:
        """
        trades: list of dicts with keys:
          date, ticker, signal, entry_price, exit_price,
          shares, pnl, cumulative_capital, correct
        """
        if not trades:
            return {"error": "No trades to analyse"}

        df = pd.DataFrame(trades)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # ── Win Rate ──────────────────────────────────────────────────────
        actionable = df[df["signal"].isin(["BUY", "SELL"])]
        win_rate   = (actionable["correct"].sum() / len(actionable) * 100
                      if len(actionable) > 0 else 0)

        # ── P&L ───────────────────────────────────────────────────────────
        total_pnl       = df["pnl"].sum()
        final_capital   = starting_capital + total_pnl
        total_return_pct= (final_capital - starting_capital) / starting_capital * 100

        # ── Daily equity curve ────────────────────────────────────────────
        equity_curve = df["cumulative_capital"].values

        # ── Sharpe Ratio (annualised) ─────────────────────────────────────
        if len(equity_curve) > 1:
            daily_returns = np.diff(equity_curve) / equity_curve[:-1]
            mean_r  = np.mean(daily_returns)
            std_r   = np.std(daily_returns)
            sharpe  = (mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0
        else:
            sharpe  = 0

        # ── Max Drawdown ──────────────────────────────────────────────────
        peak    = np.maximum.accumulate(equity_curve)
        dd      = (equity_curve - peak) / peak * 100
        max_dd  = float(np.min(dd))

        # ── Trade breakdown ───────────────────────────────────────────────
        buy_trades  = df[df["signal"] == "BUY"]
        sell_trades = df[df["signal"] == "SELL"]
        hold_trades = df[df["signal"] == "HOLD"]

        winning = actionable[actionable["correct"] == True]
        losing  = actionable[actionable["correct"] == False]

        avg_win  = winning["pnl"].mean() if len(winning) > 0 else 0
        avg_loss = losing["pnl"].mean()  if len(losing)  > 0 else 0
        profit_factor = (abs(winning["pnl"].sum()) / abs(losing["pnl"].sum())
                         if abs(losing["pnl"].sum()) > 0 else float("inf"))

        return {
            "total_trades":     len(actionable),
            "buy_trades":       len(buy_trades),
            "sell_trades":      len(sell_trades),
            "hold_signals":     len(hold_trades),
            "winning_trades":   int(actionable["correct"].sum()),
            "losing_trades":    int((~actionable["correct"]).sum()),
            "win_rate_pct":     round(win_rate, 2),
            "starting_capital": round(starting_capital, 2),
            "final_capital":    round(final_capital, 2),
            "total_pnl":        round(total_pnl, 2),
            "total_return_pct": round(total_return_pct, 2),
            "sharpe_ratio":     round(sharpe, 3),
            "max_drawdown_pct": round(max_dd, 2),
            "avg_win":          round(avg_win, 2),
            "avg_loss":         round(avg_loss, 2),
            "profit_factor":    round(profit_factor, 3),
            "equity_curve":     [round(v, 2) for v in equity_curve],
        }

    @staticmethod
    def print_report(metrics: dict, mode: str, ticker: str):
        sep = "=" * 60
        icons = {
            "win_rate":    "🎯" if metrics.get("win_rate_pct", 0) > 55 else "⚠️ ",
            "return":      "📈" if metrics.get("total_return_pct", 0) > 0  else "📉",
            "sharpe":      "✅" if metrics.get("sharpe_ratio", 0)     > 1.0 else "⚠️ ",
            "drawdown":    "✅" if metrics.get("max_drawdown_pct", 0) > -15 else "⚠️ ",
        }
        print(f"\n{sep}")
        print(f"  {mode.upper()} RESULTS — {ticker}")
        print(sep)
        print(f"  Starting Capital  : ${metrics['starting_capital']:>10,.2f}")
        print(f"  Final Capital     : ${metrics['final_capital']:>10,.2f}")
        print(f"  Total P&L         : ${metrics['total_pnl']:>+10,.2f}   {icons['return']}")
        print(f"  Total Return      : {metrics['total_return_pct']:>+9.2f}%")
        print(f"\n  ── Accuracy ──────────────────────────────────────")
        print(f"  Total Trades      : {metrics['total_trades']}")
        print(f"  Winning / Losing  : {metrics['winning_trades']} / {metrics['losing_trades']}")
        print(f"  Win Rate          : {metrics['win_rate_pct']:>8.2f}%   {icons['win_rate']}")
        print(f"  Avg Win / Loss    : ${metrics['avg_win']:+.2f} / ${metrics['avg_loss']:+.2f}")
        print(f"  Profit Factor     : {metrics['profit_factor']:.3f}")
        print(f"\n  ── Risk ──────────────────────────────────────────")
        print(f"  Sharpe Ratio      : {metrics['sharpe_ratio']:>8.3f}   {icons['sharpe']}")
        print(f"  Max Drawdown      : {metrics['max_drawdown_pct']:>8.2f}%   {icons['drawdown']}")
        print(sep)

        # Equity curve (ASCII sparkline)
        curve = metrics.get("equity_curve", [])
        if len(curve) > 1:
            mn, mx = min(curve), max(curve)
            rng    = mx - mn if mx != mn else 1
            blocks = "▁▂▃▄▅▆▇█"
            spark  = "".join(blocks[int((v - mn) / rng * 7)] for v in curve[-40:])
            print(f"  Equity Curve (last 40): {spark}")
            print(sep)


# ==============================================================================
# FEATURE BUILDER (standalone, no v4 dependency needed)
# ==============================================================================

def build_features_simple(ticker, start_date, end_date):
    """
    Build a feature DataFrame for a ticker using a simplified
    but self-contained feature set (no external auxiliary data needed).
    Used as fallback if v4 model not importable.
    """
    try:
        from ta.trend   import CCIIndicator, MACD, EMAIndicator
        from ta.momentum import RSIIndicator
        from ta.volatility import BollingerBands, AverageTrueRange
        from ta.volume  import OnBalanceVolumeIndicator
    except ImportError:
        raise ImportError("Install 'ta': pip install ta")

    df = yf.download(ticker, start=start_date, end=end_date,
                     progress=False, auto_adjust=False, actions=False)
    if df is None or df.empty:
        return None
    df = _flatten(df)

    if "adj close" in df.columns:
        df["close"] = df["adj close"]

    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    # Technical indicators
    df["rsi"]          = RSIIndicator(c, 14).rsi()
    macd               = MACD(c, 26, 12, 9)
    df["macd"]         = macd.macd()
    df["macd_diff"]    = macd.macd_diff()
    bb                 = BollingerBands(c, 20)
    df["bb_width"]     = bb.bollinger_wband()
    df["bb_pband"]     = bb.bollinger_pband()
    df["cci"]          = CCIIndicator(h, l, c, 20).cci()
    df["atr"]          = AverageTrueRange(h, l, c, 14).average_true_range()
    df["obv"]          = OnBalanceVolumeIndicator(c, v).on_balance_volume()
    ema9               = EMAIndicator(c,  9).ema_indicator()
    ema21              = EMAIndicator(c, 21).ema_indicator()
    ema50              = EMAIndicator(c, 50).ema_indicator()
    ema200             = EMAIndicator(c,200).ema_indicator()
    df["ema_cross"]    = (ema9 > ema21).astype(int)
    df["golden_cross"] = (ema50 > ema200).astype(int)
    df["price_roc_5"]  = c.pct_change(5)
    df["price_roc_10"] = c.pct_change(10)
    df["hl_spread"]    = (h - l) / c
    df["intraday_mom"] = (c - df["open"]) / df["open"]
    df["volume_roc"]   = v.pct_change(5)
    df["rs_3m"]        = c.pct_change(63)

    # Lagged features
    for lag in [5, 10, 20]:
        df[f"close_lag_{lag}"]  = c.shift(lag)
        df[f"volume_lag_{lag}"] = v.shift(lag)

    # Rolling stats
    for w in [5, 10, 20]:
        mu = c.rolling(w).mean()
        sd = c.rolling(w).std()
        df[f"close_mean_{w}"]   = mu
        df[f"close_std_{w}"]    = sd
        df[f"close_zscore_{w}"] = (c - mu) / sd.replace(0, np.nan)

    # Temporal
    df["day_of_week"]    = df.index.dayofweek
    df["month"]          = df.index.month
    df["earnings_season"]= df["month"].isin([1, 4, 7, 10]).astype(int)

    df.sort_index(inplace=True)
    return df


def _flatten(df):
    """Standalone flatten — used if v4 not importable."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(col[0]).lower().strip() for col in df.columns]
    else:
        df.columns = [str(c).lower().strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    df.dropna(axis=1, how="all", inplace=True)
    df.index = pd.to_datetime(df.index)
    return df


# ==============================================================================
# SIMPLE CLASSIFIER (used in backtest/walkforward without full v4 pipeline)
# ==============================================================================

def train_simple_model(X_train, y_train, X_test):
    """Train a lightweight XGBoost classifier and return test predictions."""
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_sample_weight

    scaler = StandardScaler()
    Xtr    = scaler.fit_transform(X_train.fillna(X_train.median()))
    Xte    = scaler.transform(X_test.fillna(X_train.median()))
    sw     = compute_sample_weight("balanced", y_train)

    model  = xgb.XGBClassifier(
        objective="multi:softprob", num_class=3,
        n_estimators=200, learning_rate=0.05,
        max_depth=5, subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss", random_state=42,
        n_jobs=1, tree_method="hist", verbosity=0,
    )
    model.fit(Xtr, y_train, sample_weight=sw, verbose=False)
    proba  = model.predict_proba(Xte)
    pred   = model.predict(Xte)
    return pred, proba


def make_labels(df, horizon_days=5, buy_thr=0.02, sell_thr=-0.02):
    """Create BUY=2, HOLD=1, SELL=0 labels for given horizon."""
    ret = df["close"].pct_change(horizon_days).shift(-horizon_days)
    labels = np.select(
        [ret >= buy_thr, ret <= sell_thr], [2, 0], default=1)
    return pd.Series(labels, index=df.index), ret


def get_feature_cols(df):
    """Return usable feature column names."""
    exclude = {"open", "high", "low", "close", "volume", "adj close",
               "labels", "future_ret"}
    return [c for c in df.columns if c.lower() not in exclude
            and not c.startswith("target_")]


# ==============================================================================
# MODE 1: BACKTEST
# ==============================================================================

class Backtester:
    """
    Trains once on 70% of history, tests on remaining 30%.
    Simulates entering/exiting trades based on model signals.
    """

    def __init__(self, sim_config=SIM_CONFIG):
        self.cfg = sim_config

    def run(self, ticker: str) -> dict:
        logger.info(f"\n{'='*55}")
        logger.info(f"  BACKTEST — {ticker}")
        logger.info(f"{'='*55}")

        # ── Build features ────────────────────────────────────────────────
        df = build_features_simple(
            ticker,
            self.cfg["backtest_start"],
            self.cfg["backtest_end"]
        )
        if df is None or len(df) < 300:
            logger.error(f"Insufficient data for {ticker}")
            return {}

        horizon  = self.cfg["horizon_days"]
        labels, future_ret = make_labels(df, horizon)
        df       = df.iloc[:-horizon].copy()
        labels   = labels.iloc[:-horizon]
        future_ret = future_ret.iloc[:-horizon]

        feat_cols = get_feature_cols(df)
        X         = df[feat_cols].fillna(df[feat_cols].median())

        # ── Train / test split (70/30, time-ordered) ──────────────────────
        split     = int(len(X) * 0.70)
        X_train   = X.iloc[:split]
        X_test    = X.iloc[split:]
        y_train   = labels.iloc[:split]
        y_test    = labels.iloc[split:]
        ret_test  = future_ret.iloc[split:]
        prices    = df["close"].iloc[split:]

        logger.info(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows")

        # ── Train & predict ───────────────────────────────────────────────
        pred, proba = train_simple_model(X_train, y_train, X_test)
        lmap = {2: "BUY", 1: "HOLD", 0: "SELL"}

        # ── Simulate trades ───────────────────────────────────────────────
        capital    = self.cfg["starting_capital"]
        pos_size   = self.cfg["position_size_pct"]
        commission = self.cfg["commission_pct"]
        slippage   = self.cfg["slippage_pct"]
        min_conf   = self.cfg["min_confidence"]
        trades     = []
        in_position= False
        entry_price= 0
        entry_date = None
        entry_shares = 0

        for i, (date, signal_int) in enumerate(zip(X_test.index, pred)):
            signal     = lmap[signal_int]
            confidence = float(proba[i][signal_int])
            price      = float(prices.iloc[i]) if i < len(prices) else None
            actual_ret = float(ret_test.iloc[i]) if i < len(ret_test) else 0

            if price is None or np.isnan(price):
                continue

            pnl     = 0
            correct = False

            # ── Enter position ────────────────────────────────────────────
            if not in_position and signal == "BUY" and confidence >= min_conf:
                trade_capital = capital * pos_size
                adj_price     = price * (1 + slippage)
                shares        = trade_capital / adj_price
                cost          = shares * adj_price * (1 + commission)
                if cost <= capital:
                    capital       -= cost
                    entry_price    = adj_price
                    entry_shares   = shares
                    entry_date     = date
                    in_position    = True

            # ── Exit position (hold for horizon_days or on SELL signal) ──
            if in_position:
                days_held = (date - entry_date).days if entry_date else 0
                should_exit = (days_held >= horizon * 1.4) or \
                              (signal == "SELL" and confidence >= min_conf)
                if should_exit:
                    exit_price = price * (1 - slippage)
                    proceeds   = entry_shares * exit_price * (1 - commission)
                    pnl        = proceeds - (entry_shares * entry_price * (1 + commission))
                    capital   += (entry_shares * entry_price) + pnl
                    correct    = pnl > 0
                    in_position= False

                    trades.append({
                        "date":               str(date.date()),
                        "ticker":             ticker,
                        "signal":             "BUY",
                        "entry_price":        round(entry_price, 2),
                        "exit_price":         round(exit_price, 2),
                        "shares":             round(entry_shares, 4),
                        "pnl":                round(pnl, 2),
                        "cumulative_capital": round(capital, 2),
                        "correct":            correct,
                        "confidence":         round(confidence, 4),
                    })
                    continue

            # ── SELL short simulation ─────────────────────────────────────
            if not in_position and signal == "SELL" and confidence >= min_conf:
                correct = actual_ret < 0
                trades.append({
                    "date":               str(date.date()),
                    "ticker":             ticker,
                    "signal":             "SELL",
                    "entry_price":        round(price, 2),
                    "exit_price":         round(price * (1 + actual_ret), 2),
                    "shares":             0,
                    "pnl":                0,
                    "cumulative_capital": round(capital, 2),
                    "correct":            correct,
                    "confidence":         round(confidence, 4),
                })

            # ── HOLD signal ───────────────────────────────────────────────
            if signal == "HOLD":
                trades.append({
                    "date":               str(date.date()),
                    "ticker":             ticker,
                    "signal":             "HOLD",
                    "entry_price":        round(price, 2),
                    "exit_price":         round(price, 2),
                    "shares":             0,
                    "pnl":                0,
                    "cumulative_capital": round(capital, 2),
                    "correct":            True,
                    "confidence":         round(confidence, 4),
                })

        metrics = MetricsEngine.calculate(trades, self.cfg["starting_capital"])
        MetricsEngine.print_report(metrics, "BACKTEST", ticker)

        # Save trade log
        log_path = f"backtest_{ticker}.csv"
        pd.DataFrame(trades).to_csv(log_path, index=False)
        logger.info(f"  Trade log saved → {log_path}")

        return {"ticker": ticker, "metrics": metrics, "trades": trades}


# ==============================================================================
# MODE 2: WALK-FORWARD SIMULATION
# ==============================================================================

class WalkForwardSimulator:
    """
    Rolls a training window forward through time.
    Train on N months → test on M months → advance → repeat.
    Mimics real-world model retraining.
    """

    def __init__(self, sim_config=SIM_CONFIG):
        self.cfg = sim_config

    def run(self, ticker: str) -> dict:
        logger.info(f"\n{'='*55}")
        logger.info(f"  WALK-FORWARD — {ticker}")
        logger.info(f"{'='*55}")

        df = build_features_simple(
            ticker,
            self.cfg["backtest_start"],
            self.cfg["backtest_end"]
        )
        if df is None or len(df) < 400:
            logger.error(f"Insufficient data for {ticker}")
            return {}

        horizon    = self.cfg["horizon_days"]
        labels, future_ret = make_labels(df, horizon)
        df         = df.iloc[:-horizon].copy()
        labels     = labels.iloc[:-horizon]
        future_ret = future_ret.iloc[:-horizon]
        feat_cols  = get_feature_cols(df)

        train_days = self.cfg["train_months"] * 21   # ~21 trading days/month
        test_days  = self.cfg["test_months"]  * 21

        capital    = self.cfg["starting_capital"]
        pos_size   = self.cfg["position_size_pct"]
        commission = self.cfg["commission_pct"]
        slippage   = self.cfg["slippage_pct"]
        min_conf   = self.cfg["min_confidence"]

        all_trades = []
        fold       = 0
        start_idx  = 0

        logger.info(f"  Train window: {self.cfg['train_months']}m "
                    f"({train_days}d) | "
                    f"Test window: {self.cfg['test_months']}m ({test_days}d)")

        while start_idx + train_days + test_days <= len(df):
            fold      += 1
            tr_end     = start_idx + train_days
            te_end     = min(tr_end + test_days, len(df))

            X_tr = df[feat_cols].iloc[start_idx:tr_end].fillna(
                df[feat_cols].iloc[start_idx:tr_end].median())
            y_tr = labels.iloc[start_idx:tr_end]
            X_te = df[feat_cols].iloc[tr_end:te_end].fillna(
                df[feat_cols].iloc[start_idx:tr_end].median())
            prices_te  = df["close"].iloc[tr_end:te_end]
            ret_te     = future_ret.iloc[tr_end:te_end]
            dates_te   = df.index[tr_end:te_end]

            if len(X_tr) < self.cfg["min_train_rows"] or len(X_te) < 5:
                start_idx += test_days
                continue

            logger.info(f"  Fold {fold:02d}: "
                        f"train [{df.index[start_idx].date()} → "
                        f"{df.index[tr_end-1].date()}] "
                        f"test [{df.index[tr_end].date()} → "
                        f"{df.index[te_end-1].date()}]")

            try:
                pred, proba = train_simple_model(X_tr, y_tr, X_te)
            except Exception as ex:
                logger.warning(f"  Fold {fold} failed: {ex}")
                start_idx += test_days
                continue

            lmap = {2: "BUY", 1: "HOLD", 0: "SELL"}
            in_pos = False
            entry_p = 0; entry_s = 0; entry_d = None

            for i in range(len(pred)):
                signal     = lmap[pred[i]]
                confidence = float(proba[i][pred[i]])
                price      = float(prices_te.iloc[i])
                actual_ret = float(ret_te.iloc[i])
                date       = dates_te[i]

                if np.isnan(price):
                    continue

                pnl = 0; correct = False

                if not in_pos and signal == "BUY" and confidence >= min_conf:
                    trade_cap = capital * pos_size
                    adj_p     = price * (1 + slippage)
                    shares    = trade_cap / adj_p
                    cost      = shares * adj_p * (1 + commission)
                    if cost <= capital:
                        capital  -= cost
                        entry_p   = adj_p
                        entry_s   = shares
                        entry_d   = date
                        in_pos    = True

                if in_pos:
                    days_held  = (date - entry_d).days if entry_d else 0
                    should_exit= days_held >= horizon * 1.4 or \
                                 (signal == "SELL" and confidence >= min_conf)
                    if should_exit:
                        exit_p  = price * (1 - slippage)
                        proceeds= entry_s * exit_p * (1 - commission)
                        pnl     = proceeds - entry_s * entry_p * (1 + commission)
                        capital += entry_s * entry_p + pnl
                        correct  = pnl > 0
                        in_pos   = False
                        all_trades.append({
                            "date": str(date.date()), "ticker": ticker,
                            "fold": fold, "signal": "BUY",
                            "entry_price": round(entry_p, 2),
                            "exit_price": round(exit_p, 2),
                            "shares": round(entry_s, 4),
                            "pnl": round(pnl, 2),
                            "cumulative_capital": round(capital, 2),
                            "correct": correct,
                            "confidence": round(confidence, 4),
                        })
                        continue

                if not in_pos and signal == "SELL" and confidence >= min_conf:
                    correct = actual_ret < 0
                    all_trades.append({
                        "date": str(date.date()), "ticker": ticker,
                        "fold": fold, "signal": "SELL",
                        "entry_price": round(price, 2),
                        "exit_price": round(price * (1 + actual_ret), 2),
                        "shares": 0, "pnl": 0,
                        "cumulative_capital": round(capital, 2),
                        "correct": correct, "confidence": round(confidence, 4),
                    })

                if signal == "HOLD":
                    all_trades.append({
                        "date": str(date.date()), "ticker": ticker,
                        "fold": fold, "signal": "HOLD",
                        "entry_price": round(price, 2),
                        "exit_price": round(price, 2),
                        "shares": 0, "pnl": 0,
                        "cumulative_capital": round(capital, 2),
                        "correct": True, "confidence": round(confidence, 4),
                    })

            start_idx += test_days

        if not all_trades:
            logger.error("No trades generated in walk-forward.")
            return {}

        metrics = MetricsEngine.calculate(all_trades, self.cfg["starting_capital"])
        MetricsEngine.print_report(metrics, "WALK-FORWARD", ticker)

        log_path = f"walkforward_{ticker}.csv"
        pd.DataFrame(all_trades).to_csv(log_path, index=False)
        logger.info(f"  Trade log saved → {log_path}  ({fold} folds)")

        return {"ticker": ticker, "metrics": metrics,
                "trades": all_trades, "folds": fold}


# ==============================================================================
# MODE 3: PAPER TRADING
# ==============================================================================

class PaperTrader:
    """
    Generates today's live signals for a list of tickers.
    Saves signals to a JSON log so you can track accuracy over time.
    Each time you run it, it appends today's predictions and checks
    if yesterday's predictions were correct.
    """

    def __init__(self, sim_config=SIM_CONFIG):
        self.cfg      = sim_config
        self.log_file = sim_config["paper_trades_file"]
        self.csv_file = sim_config["paper_log_file"]

    def _load_log(self) -> list:
        if os.path.exists(self.log_file):
            with open(self.log_file) as f:
                return json.load(f)
        return []

    def _save_log(self, log: list):
        # Deduplicate: keep last entry per (ticker, date) pair
        seen = {}
        for entry in log:
            key = (entry.get("ticker"), entry.get("date"))
            seen[key] = entry
        with open(self.log_file, "w") as f:
            json.dump(list(seen.values()), f, indent=2)

    def _get_signal(self, ticker: str) -> dict:
        """Generate today's signal using the same v5 model as stock_recommendation_xgboost_v5.py."""
        try:
            from stock_recommendation_xgboost_v5 import (
                process_ticker, DataFetcher, CONFIG as V5_CONFIG,
            )
        except ImportError:
            logger.error("stock_recommendation_xgboost_v5.py not found")
            return None

        try:
            fetcher = DataFetcher(V5_CONFIG)
            sector  = V5_CONFIG["ticker_sector"].get(ticker, "Technology")
            etf     = V5_CONFIG["sector_etfs"].get(sector, "XLK")

            market_df = fetcher.fetch_price("^GSPC")
            sector_df = fetcher.fetch_price(etf)
            commodity_dfs = {}
            for t in V5_CONFIG["commodity_tickers"]:
                d = fetcher.fetch_price(t)
                if d is not None:
                    commodity_dfs[t] = d

            shared = {
                "market_df":    market_df   if market_df  is not None else pd.DataFrame(),
                "sector_dfs":   {etf: sector_df if sector_df is not None else pd.DataFrame()},
                "commodity_dfs": commodity_dfs,
            }

            result = process_ticker((ticker, V5_CONFIG, shared))
        except Exception as ex:
            logger.warning(f"  {ticker} v5 signal failed: {ex}")
            return None

        if result is None:
            return None

        hn            = self.cfg["horizon"]          # e.g. "1w"
        horizon_days  = self.cfg["horizon_days"]     # e.g. 5
        signal        = result.get(f"{hn}_signal", "HOLD")
        prob_buy      = result.get(f"{hn}_prob_buy",  0.0)
        prob_hold     = result.get(f"{hn}_prob_hold", 0.0)
        prob_sell     = result.get(f"{hn}_prob_sell", 0.0)
        confidence    = max(prob_buy, prob_hold, prob_sell)

        return {
            "ticker":        ticker,
            "date":          datetime.today().strftime("%Y-%m-%d"),
            "price_date":    result.get("price_date"),
            "signal":        signal,
            "confidence":    round(confidence, 4),
            "prob_buy":      round(prob_buy,  4),
            "prob_hold":     round(prob_hold, 4),
            "prob_sell":     round(prob_sell, 4),
            "entry_price":   result.get("current_price"),
            "horizon_days":  horizon_days,
            "target_date":   (datetime.today() +
                              timedelta(days=horizon_days * 1.4)).strftime("%Y-%m-%d"),
            "outcome":       None,
            "exit_price":    None,
            "pnl_pct":       None,
            "correct":       None,
        }

    def _resolve_outcomes(self, log: list) -> list:
        """
        For any past signal whose target_date has passed and outcome is None,
        fetch the actual exit price and mark correct/incorrect.
        """
        today = datetime.today()
        for entry in log:
            if entry.get("outcome") is not None:
                continue
            target_dt = datetime.strptime(entry["target_date"], "%Y-%m-%d")
            if today < target_dt:
                continue
            try:
                hist  = yf.Ticker(entry["ticker"]).history(
                    start=entry["target_date"],
                    end=(target_dt + timedelta(days=5)).strftime("%Y-%m-%d"),
                    auto_adjust=False)
                hist  = _flatten(hist)
                if hist.empty:
                    continue
                pcol       = "adj close" if "adj close" in hist.columns else "close"
                exit_price = round(float(hist[pcol].iloc[0]), 2)
                entry_p    = entry.get("entry_price")
                if entry_p and entry_p > 0:
                    pnl_pct = (exit_price - entry_p) / entry_p * 100
                    if entry["signal"] == "BUY":
                        correct = pnl_pct > 0
                    elif entry["signal"] == "SELL":
                        correct = pnl_pct < 0
                    else:
                        correct = True
                    entry["exit_price"] = exit_price
                    entry["pnl_pct"]    = round(pnl_pct, 2)
                    entry["correct"]    = correct
                    entry["outcome"]    = "resolved"
                    logger.info(f"  Resolved {entry['ticker']} "
                                f"{entry['date']} {entry['signal']}: "
                                f"{'✅ CORRECT' if correct else '❌ WRONG'} "
                                f"({pnl_pct:+.2f}%)")
            except Exception as ex:
                logger.warning(f"  Could not resolve {entry['ticker']}: {ex}")
        return log

    def _simulate_pnl(self, log: list) -> dict:
        """
        Simulate a simple paper portfolio across all resolved trades.
        Each BUY uses position_size_pct of capital.
        SELL signals are tracked for directional accuracy only (no short selling).
        Returns portfolio metrics.
        """
        capital    = self.cfg["starting_capital"]
        pos_pct    = self.cfg["position_size_pct"]
        commission = self.cfg["commission_pct"]
        equity     = [capital]
        total_pnl  = 0.0

        resolved_buys = [e for e in log
                         if e.get("outcome") == "resolved"
                         and e["signal"] == "BUY"
                         and e.get("exit_price") is not None
                         and e.get("entry_price") is not None]

        for trade in resolved_buys:
            ep      = trade["entry_price"]
            xp      = trade["exit_price"]
            shares  = (capital * pos_pct) / ep
            cost    = shares * ep * (1 + commission)
            proceeds= shares * xp * (1 - commission)
            pnl     = proceeds - cost
            pnl_dollar = round(pnl, 2)
            capital += pnl
            total_pnl += pnl
            equity.append(round(capital, 2))
            trade["pnl_dollar"] = pnl_dollar   # enrich trade with $ P&L

        return {
            "final_capital": round(capital, 2),
            "total_pnl":     round(total_pnl, 2),
            "total_return":  round((capital - self.cfg["starting_capital"])
                                   / self.cfg["starting_capital"] * 100, 2),
            "equity_curve":  equity,
        }

    def run(self, tickers: list) -> dict:
        logger.info(f"\n{'='*60}")
        logger.info(f"  PAPER TRADING — {', '.join(tickers)}")
        logger.info(f"{'='*60}")

        log = self._load_log()

        # ── Resolve any past predictions whose target_date has passed ──────
        log = self._resolve_outcomes(log)

        # ── Generate today's signals ───────────────────────────────────────
        today_signals = []
        for ticker in tickers:
            logger.info(f"  Generating signal: {ticker}...")
            sig = self._get_signal(ticker)
            if sig:
                log.append(sig)
                today_signals.append(sig)

        self._save_log(log)
        pd.DataFrame(log).to_csv(self.csv_file, index=False)

        # ── Simulate portfolio P&L across resolved BUY trades ─────────────
        portfolio = self._simulate_pnl(log)

        # ── Partition log for display ──────────────────────────────────────
        resolved  = [e for e in log if e.get("outcome") == "resolved"]
        pending   = [e for e in log if e.get("outcome") is None]
        sep       = "=" * 65

        # ── OVERVIEW ──────────────────────────────────────────────────────
        print(f"\n{sep}")
        print(f"  PAPER TRADING SUMMARY")
        print(sep)
        print(f"  Starting capital     : ${self.cfg['starting_capital']:>10,.2f}")
        print(f"  Current capital      : ${portfolio['final_capital']:>10,.2f}")
        print(f"  Total P&L            : ${portfolio['total_pnl']:>+10,.2f}")
        print(f"  Total return         : {portfolio['total_return']:>+9.2f}%")
        print(f"  Total signals logged : {len(log)}")
        print(f"  Resolved             : {len(resolved)}")
        print(f"  Pending (awaiting)   : {len(pending)}")

        if resolved:
            correct   = sum(1 for e in resolved if e.get("correct"))
            wr        = correct / len(resolved) * 100
            pnl_list  = [e["pnl_pct"] for e in resolved
                         if e.get("pnl_pct") is not None]
            avg_ret   = np.mean(pnl_list) if pnl_list else 0
            best      = max(pnl_list)  if pnl_list else 0
            worst     = min(pnl_list)  if pnl_list else 0
            print(f"  Win rate             : {wr:>8.1f}%")
            print(f"  Avg return / trade   : {avg_ret:>+8.2f}%")
            print(f"  Best trade           : {best:>+8.2f}%")
            print(f"  Worst trade          : {worst:>+8.2f}%")

        # ── RESOLVED TRADES TABLE ─────────────────────────────────────────
        if resolved:
            print(f"\n  {'─'*63}")
            print(f"  Resolved trades:")
            print(f"  {'─'*63}")
            print(f"  {'Date':<12} {'Ticker':<7} {'Sig':<5} "
                  f"{'Entry':>8} {'Exit':>8} {'Move':>8} "
                  f"{'P&L $':>8} {'Result':<10}")
            print(f"  {'─'*63}")
            for e in sorted(resolved, key=lambda x: x.get("date",""), reverse=True)[:20]:
                ep      = e.get("entry_price", 0) or 0
                xp      = e.get("exit_price",  0) or 0
                pnl_pct = e.get("pnl_pct",     0) or 0
                pnl_usd = e.get("pnl_dollar",  "—")
                correct = e.get("correct", False)
                result  = "CORRECT" if correct else "WRONG"
                icon    = "" if correct else ""
                sig     = e.get("signal", "?")

                # For SELL signals, a price drop = correct
                if sig == "SELL":
                    move_str = f"{pnl_pct:+.2f}%"
                    usd_str  = "N/A (short)"
                else:
                    move_str = f"{pnl_pct:+.2f}%"
                    usd_str  = f"${pnl_usd:+.2f}" if isinstance(pnl_usd, float) else str(pnl_usd)

                print(f"  {e.get('date','?'):<12} "
                      f"{e.get('ticker','?'):<7} "
                      f"{sig:<5} "
                      f"${ep:>7.2f} "
                      f"${xp:>7.2f} "
                      f"{move_str:>8} "
                      f"{usd_str:>8} "
                      f"{icon} {result}")

        # ── PENDING TRADES TABLE ──────────────────────────────────────────
        if pending:
            print(f"\n  {'─'*63}")
            print(f"  Pending — waiting for target date:")
            print(f"  {'─'*63}")
            print(f"  {'Date':<12} {'Ticker':<7} {'Sig':<5} "
                  f"{'Entry':>8} {'Conf':>6} {'Target Date':<14} {'Days left'}")
            print(f"  {'─'*63}")
            today = datetime.today()
            for e in sorted(pending, key=lambda x: x.get("target_date",""))[:20]:
                ep        = e.get("entry_price", 0) or 0
                conf      = e.get("confidence", 0) or 0
                tgt_date  = e.get("target_date", "?")
                try:
                    days_left = (datetime.strptime(tgt_date, "%Y-%m-%d") - today).days
                    days_str  = f"{days_left}d"
                except Exception:
                    days_str  = "?"
                print(f"  {e.get('date','?'):<12} "
                      f"{e.get('ticker','?'):<7} "
                      f"{e.get('signal','?'):<5} "
                      f"${ep:>7.2f} "
                      f"{conf:>5.1%} "
                      f"{tgt_date:<14} "
                      f"{days_str}")

        # ── TODAY'S NEW SIGNALS ───────────────────────────────────────────
        if today_signals:
            print(f"\n  {'─'*63}")
            print(f"  Today's new signals (just added to pending):")
            print(f"  {'─'*63}")
            for s in today_signals:
                icon = " BUY " if s["signal"] == "BUY" else \
                       " SELL" if s["signal"] == "SELL" else " HOLD"
                ep   = s.get("entry_price") or 0
                print(f"  {icon}  {s['ticker']:<7} "
                      f"entry=${ep:.2f}  "
                      f"conf={s['confidence']:.1%}  "
                      f"horizon={s['horizon_days']}d  "
                      f"check on {s['target_date']}")
            print(f"\n  How exit price is determined:")
            print(f"  On the target date the model fetches the actual closing")
            print(f"  price from Yahoo Finance and compares it to the entry")
            print(f"  price above. BUY is correct if exit > entry.")
            print(f"  SELL is correct if exit < entry. Dollar P&L assumes")
            print(f"  {self.cfg['position_size_pct']*100:.0f}% of capital per trade.")

        print(f"\n  Logs saved → {self.log_file}  |  {self.csv_file}")
        print(sep)

        return {"signals": today_signals, "full_log": log, "portfolio": portfolio}


# ==============================================================================
# COMBINED REPORT
# ==============================================================================

def run_all(ticker: str):
    """Run all three modes for one ticker and print a combined summary."""
    print(f"\n{'#'*60}")
    print(f"  FULL SIMULATION SUITE — {ticker}")
    print(f"{'#'*60}")

    bt = Backtester().run(ticker)
    wf = WalkForwardSimulator().run(ticker)
    pt = PaperTrader().run([ticker])

    # ── Comparison table ──────────────────────────────────────────────────
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  COMPARISON SUMMARY — {ticker}")
    print(sep)
    print(f"  {'Mode':<18} {'Win Rate':>9} {'Return':>9} "
          f"{'Sharpe':>8} {'MaxDD':>8} {'Trades':>7}")
    print(f"  {'-'*56}")

    for label, result in [("Backtest", bt), ("Walk-Forward", wf)]:
        if result and "metrics" in result:
            m = result["metrics"]
            print(f"  {label:<18} "
                  f"{m.get('win_rate_pct', 0):>8.1f}% "
                  f"{m.get('total_return_pct', 0):>+8.2f}% "
                  f"{m.get('sharpe_ratio', 0):>8.3f} "
                  f"{m.get('max_drawdown_pct', 0):>7.2f}% "
                  f"{m.get('total_trades', 0):>7}")

    print(sep)
    print("\n  Interpretation guide:")
    print("  Win Rate > 55%   = model is directionally accurate")
    print("  Sharpe  > 1.0    = good risk-adjusted return")
    print("  Sharpe  > 2.0    = excellent")
    print("  Max DD  > -20%   = acceptable drawdown risk")
    print(sep)


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("\nUsage:")
        print("  python simulation_framework.py backtest   SOFI")
        print("  python simulation_framework.py walkforward AAPL")
        print("  python simulation_framework.py paper      AAPL MSFT SOFI")
        print("  python simulation_framework.py all        SOFI")
        sys.exit(0)

    mode    = sys.argv[1].lower()
    tickers = [t.upper() for t in sys.argv[2:]]

    if mode == "backtest":
        for t in tickers:
            Backtester().run(t)

    elif mode == "walkforward":
        for t in tickers:
            WalkForwardSimulator().run(t)

    elif mode == "paper":
        PaperTrader().run(tickers)

    elif mode == "all":
        run_all(tickers[0])

    else:
        print(f"Unknown mode: {mode}")
        print("Choose from: backtest | walkforward | paper | all")
