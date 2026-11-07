"""Aligns posts/news with price bars and builds the per-ticker-per-session feature panel.

Alignment rule (no look-ahead): every item is timestamped in UTC and assigned to the
first trading session whose *close* (exchange local close converted to UTC) is at or
after the item's timestamp. Features for session t therefore only use information
published before close(t), and they are used to predict close(t+1).
Items published after the latest close are kept separately as "after-close" buzz.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def engagement_weight(df: pd.DataFrame) -> pd.Series:
    likes = df["likes"].fillna(0)
    dislikes = df["dislikes"].fillna(0)
    replies = df["replies"].fillna(0)
    views = pd.to_numeric(df["views"], errors="coerce").fillna(0)
    foll = pd.to_numeric(df["author_followers"], errors="coerce").fillna(0)
    w = 1 + np.log1p(likes + dislikes + 2 * replies + views / 50) + 0.5 * np.log1p(foll / 1000)
    return w + 0.5 * df["holder_verified"].fillna(False).astype(float)  # Naver "verified holder" badge


def crowd_adjusted_score(df: pd.DataFrame) -> pd.Series:
    """A bullish post that the crowd down-votes counts less (Naver/LIHKG up/down votes)."""
    L, D = df["likes"].fillna(0), df["dislikes"].fillna(0)
    agree = (L - D) / (L + D + 2)
    return df["score"] * (1 + 0.5 * agree)


def assign_sessions(items: pd.DataFrame, prices: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """Adds `session` (trading date) to each item; NaT for items after the last close."""
    items = items.copy()
    items["session"] = pd.NaT
    for ex, grp in items.groupby("exchange"):
        cal = prices[prices["exchange"] == ex][["date", "close_utc"]].drop_duplicates("date").sort_values("close_utc")
        if cal.empty:
            continue
        closes = np.array([t.value for t in pd.to_datetime(cal["close_utc"], utc=True)], dtype=np.int64)
        ts = np.array([t.value for t in pd.to_datetime(grp[ts_col], utc=True)], dtype=np.int64)
        idx = np.searchsorted(closes, ts, side="left")  # first close >= timestamp
        dates = pd.to_datetime(cal["date"]).to_numpy()
        sess = [dates[i] if i < len(dates) else pd.NaT for i in idx]
        items.loc[grp.index, "session"] = pd.to_datetime(sess)
    items["session"] = pd.to_datetime(items["session"])
    return items


def _wmean(g: pd.DataFrame, col: str, wcol: str = "w") -> float:
    w = g[wcol]
    return float((g[col] * w).sum() / w.sum()) if len(g) and w.sum() > 0 else np.nan


def daily_social(posts: pd.DataFrame) -> pd.DataFrame:
    """One row per (ticker, session): mentions, engagement and weighted sentiment, per source and overall."""
    if posts.empty:
        return pd.DataFrame()
    p = posts.dropna(subset=["session"])
    rows = []
    for (t, s), g in p.groupby(["ticker", "session"]):
        r = {"ticker": t, "session": s, "mentions": len(g), "engagement": float(g["w"].sum()),
             "sent_social": _wmean(g, "score_adj"),
             "bull_share": float((g["score"] > 0.1).mean()), "bear_share": float((g["score"] < -0.1).mean())}
        for src, gs in g.groupby("source"):
            r[f"mentions_{src}"] = len(gs)
            r[f"sent_{src}"] = _wmean(gs, "score_adj")
        rows.append(r)
    return pd.DataFrame(rows)


def daily_news(news: pd.DataFrame) -> pd.DataFrame:
    if news.empty:
        return pd.DataFrame(columns=["ticker", "session", "news_n", "sent_news"])
    n = news.dropna(subset=["session"])
    return (n.groupby(["ticker", "session"]).agg(news_n=("score", "size"), sent_news=("score", "mean")).reset_index())


def price_features(prices: pd.DataFrame, index_prices: pd.DataFrame | None = None) -> pd.DataFrame:
    """Technical features per (ticker, date) and the n+1 target (next-session log return)."""
    out = []
    for t, g in prices.groupby("ticker"):
        g = g.sort_values("date").copy()
        lr = np.log(g["close"]).diff()
        g["ret_1"] = lr
        g["ret_5"] = np.log(g["close"]).diff(5)
        g["vol_10"] = lr.rolling(10).std()
        delta = g["close"].diff()
        up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        g["rsi_14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
        g["dist_sma20"] = g["close"] / g["close"].rolling(20).mean() - 1
        g["range_pos"] = (g["close"] - g["low"].rolling(20).min()) / (g["high"].rolling(20).max() - g["low"].rolling(20).min())
        g["vol_z"] = (g["volume"] - g["volume"].rolling(20).mean()) / g["volume"].rolling(20).std()
        g["target"] = lr.shift(-1)          # log(close[t+1] / close[t])
        g["next_close"] = g["close"].shift(-1)
        g["next_date"] = g["date"].shift(-1)
        out.append(g)
    df = pd.concat(out, ignore_index=True)
    if index_prices is not None and not index_prices.empty:
        idx = index_prices.sort_values("date")[["exchange", "date", "close"]].copy()
        idx["mkt_ret_1"] = idx.groupby("exchange")["close"].transform(lambda s: np.log(s).diff())
        df = df.merge(idx[["exchange", "date", "mkt_ret_1"]], on=["exchange", "date"], how="left")
    else:
        df["mkt_ret_1"] = np.nan
    return df


SOCIAL_FEATURES = ["sent_social", "sent_news", "d_sent", "mention_z", "bull_minus_bear"]
PRICE_FEATURES = ["ret_1", "ret_5", "vol_10", "rsi_c", "dist_sma20", "range_pos", "vol_z", "mkt_ret_1"]


def build_panel(price_feat: pd.DataFrame, social: pd.DataFrame, news_daily: pd.DataFrame) -> pd.DataFrame:
    panel = price_feat.copy()
    if not social.empty:
        panel = panel.merge(social.rename(columns={"session": "date"}), on=["ticker", "date"], how="left")
    if not news_daily.empty:
        panel = panel.merge(news_daily.rename(columns={"session": "date"}), on=["ticker", "date"], how="left")
    for c in ["mentions", "sent_social", "bull_share", "bear_share", "news_n", "sent_news"]:
        if c not in panel:
            panel[c] = np.nan
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel["in_social_window"] = panel["mentions"].notna()
    g = panel.groupby("ticker")
    panel["d_sent"] = panel["sent_social"] - g["sent_social"].shift(1)
    lm = np.log1p(panel["mentions"])
    panel["mention_z"] = (lm - g["mentions"].transform(lambda s: np.log1p(s).mean())) / g["mentions"].transform(
        lambda s: np.log1p(s).std()).replace(0, np.nan)
    panel["bull_minus_bear"] = panel["bull_share"] - panel["bear_share"]
    panel["rsi_c"] = (panel["rsi_14"] - 50) / 50
    return panel
