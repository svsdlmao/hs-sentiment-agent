"""n+1 price model with walk-forward (expanding-window) out-of-sample validation.

Two-stage design, because social data only covers ~10 days while prices cover a year:
  stage 1  price-only ridge on technical features, trained on the full pooled price history;
  stage 2  heavily-shrunk ridge on social + news features that explains stage-1 residuals,
           trained only on the sessions where sentiment exists.
Prediction  log(close[t+1]/close[t]) = stage1(x_price) + stage2(x_sentiment);  price = close[t]*exp(pred).

Validation is strictly walk-forward: for every test session d, both stages are re-fitted on
rows whose target was already known at close(d) (row date < d), then predict d -> d+1.
Every prediction is compared against two baselines on the same rows:
  naive       random walk, close[t+1] = close[t]
  price_only  stage 1 alone
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .features import PRICE_FEATURES, SOCIAL_FEATURES

STAGE1_ALPHA = 50.0
STAGE2_ALPHA = 150.0     # few rows (10 tickers x a handful of sessions): shrink hard
STAGE2_CAP = 0.5         # sentiment layer may move the forecast by at most 0.5 x daily volatility
MIN_TRAIN_SESSIONS = 2   # sentiment sessions needed before the first out-of-sample test

# Winsorise standardised features at +-3 sd so one spike (e.g. a 10x mention day) cannot dominate.
_CLIP = FunctionTransformer(lambda X: np.clip(X, -3, 3))


def _X(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].astype(float).fillna(0.0).to_numpy()


def _fit_stage1(train: pd.DataFrame):
    tr = train.dropna(subset=["target"] + ["ret_5", "vol_10", "dist_sma20"])
    m = make_pipeline(StandardScaler(), _CLIP, Ridge(alpha=STAGE1_ALPHA))
    m.fit(_X(tr, PRICE_FEATURES), tr["target"].to_numpy())
    return m


def _fit_stage2(train: pd.DataFrame, stage1):
    tr = train[train["in_social_window"]].dropna(subset=["target"])
    if len(tr) < 8:
        return None
    resid = tr["target"].to_numpy() - stage1.predict(_X(tr, PRICE_FEATURES))
    m = make_pipeline(StandardScaler(), _CLIP, Ridge(alpha=STAGE2_ALPHA))
    m.fit(_X(tr, SOCIAL_FEATURES), resid)
    return m


def _predict(rows: pd.DataFrame, s1, s2) -> tuple[np.ndarray, np.ndarray]:
    p1 = s1.predict(_X(rows, PRICE_FEATURES))
    p2 = s2.predict(_X(rows, SOCIAL_FEATURES)) if s2 is not None else np.zeros(len(rows))
    cap = STAGE2_CAP * rows["vol_10"].fillna(0.02).to_numpy()
    return p1, p1 + np.clip(p2, -cap, cap)


def walk_forward(panel: pd.DataFrame) -> pd.DataFrame:
    """Out-of-sample predictions for every sentiment session after the warm-up."""
    social_dates = sorted(panel.loc[panel["in_social_window"], "date"].unique())
    test_dates = [d for d in social_dates[MIN_TRAIN_SESSIONS:]]
    out = []
    for d in test_dates:
        test = panel[(panel["date"] == d) & panel["target"].notna() & panel["in_social_window"]]
        if test.empty:
            continue
        train = panel[panel["date"] < d]
        s1 = _fit_stage1(train)
        s2 = _fit_stage2(train, s1)
        p1, p2 = _predict(test, s1, s2)
        o = test[["ticker", "exchange", "date", "next_date", "close", "next_close", "target"]].copy()
        o["pred_naive"] = 0.0
        o["pred_price_only"] = p1
        o["pred_sentiment"] = p2
        o["stage2_used"] = s2 is not None
        out.append(o)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _binom_p_two_sided(k: int, n: int) -> float:
    if n == 0:
        return float("nan")
    probs = [math.comb(n, i) * 0.5 ** n for i in range(n + 1)]
    pk = probs[k]
    return float(min(1.0, sum(p for p in probs if p <= pk + 1e-12)))


def score_backtest(bt: pd.DataFrame) -> pd.DataFrame:
    if bt.empty:
        return pd.DataFrame()
    rows = []
    for scope, g in [("ALL", bt)] + list(bt.groupby("exchange")):
        actual = g["next_close"].to_numpy()
        for model in ["naive", "price_only", "sentiment"]:
            pred_ret = g[f"pred_{model}"].to_numpy()
            pred_px = g["close"].to_numpy() * np.exp(pred_ret)
            err = pred_px - actual
            ape = np.abs(err) / actual
            r = {"scope": scope, "model": model, "n": len(g), "sessions": int(g["date"].nunique()),
                 "MAE_pct": float(np.mean(ape) * 100), "RMSE_pct": float(np.sqrt(np.mean(ape ** 2)) * 100)}
            if model != "naive":
                moved = g["target"].to_numpy() != 0
                hits = int(np.sum(np.sign(pred_ret[moved]) == np.sign(g["target"].to_numpy()[moved])))
                n = int(moved.sum())
                r.update(dir_hits=hits, dir_n=n, dir_acc_pct=100 * hits / n if n else np.nan,
                         dir_p_value=_binom_p_two_sided(hits, n))
            rows.append(r)
    res = pd.DataFrame(rows)
    base = res[res["model"] == "naive"].set_index("scope")["MAE_pct"]
    res["MAE_vs_naive_pct"] = res.apply(lambda r: 100 * (base[r["scope"]] - r["MAE_pct"]) / base[r["scope"]], axis=1)
    return res


def predict_next(panel: pd.DataFrame) -> pd.DataFrame:
    """Live n+1 prediction from the latest completed session of every ticker."""
    train = panel[panel["target"].notna()]
    s1 = _fit_stage1(train)
    s2 = _fit_stage2(train, s1)
    last = panel.sort_values("date").groupby("ticker").tail(1).copy()
    p1, p2 = _predict(last, s1, s2)
    last["pred_ret_price_only"] = p1
    last["pred_ret"] = p2
    last["pred_close"] = last["close"] * np.exp(p2)
    sigma = last["vol_10"].fillna(last["vol_10"].median())
    last["pred_low_68"] = last["close"] * np.exp(p2 - sigma)
    last["pred_high_68"] = last["close"] * np.exp(p2 + sigma)
    coefs = {}
    if s2 is not None:
        coefs = dict(zip(SOCIAL_FEATURES, s2[-1].coef_.round(5)))
    last.attrs["stage2_coefs"] = coefs
    last.attrs["stage1_coefs"] = dict(zip(PRICE_FEATURES, s1[-1].coef_.round(5)))
    return last
