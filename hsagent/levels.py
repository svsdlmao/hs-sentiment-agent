"""Support / resistance from four independent methods, plus the trade plan per ticker.

1. Floor-trader pivots from the last session (P, R1/R2, S1/S2).
2. Swing levels: fractal highs/lows over ~9 months, clustered within 0.6 ATR and ranked by
   touches and recency.
3. Volume profile: the point of control (price bucket with the most traded volume, 90 sessions).
4. Crowd-cited levels: price numbers people actually write in the posts ("8만 깨지면", "500蚊")
   within ±25% of the current price. These are the levels the crowd is watching.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd


def atr(g: pd.DataFrame, n: int = 14) -> float:
    prev = g["close"].shift(1)
    tr = pd.concat([g["high"] - g["low"], (g["high"] - prev).abs(), (g["low"] - prev).abs()], axis=1).max(axis=1)
    return float(tr.rolling(n).mean().iloc[-1])


def pivots(last: pd.Series) -> dict:
    p = (last["high"] + last["low"] + last["close"]) / 3
    rng = last["high"] - last["low"]
    return {"P": p, "R1": 2 * p - last["low"], "S1": 2 * p - last["high"], "R2": p + rng, "S2": p - rng}


def swing_levels(g: pd.DataFrame, a: float, window: int = 3, lookback: int = 180) -> list[dict]:
    g = g.tail(lookback).reset_index(drop=True)
    pts = []
    for i in range(window, len(g) - window):
        hi, lo = g.loc[i, "high"], g.loc[i, "low"]
        if hi == g.loc[i - window:i + window, "high"].max():
            pts.append((hi, i))
        if lo == g.loc[i - window:i + window, "low"].min():
            pts.append((lo, i))
    pts.sort()
    clusters: list[list[tuple[float, int]]] = []
    for px, i in pts:
        if clusters and px - clusters[-1][-1][0] <= 0.6 * a:
            clusters[-1].append((px, i))
        else:
            clusters.append([(px, i)])
    n = len(g)
    out = []
    for c in clusters:
        prices = [p for p, _ in c]
        recency = max(i for _, i in c) / max(n - 1, 1)
        out.append({"price": float(np.mean(prices)), "touches": len(c), "strength": len(c) + 2 * recency})
    return out


def volume_poc(g: pd.DataFrame, lookback: int = 90, bins: int = 40) -> float:
    g = g.tail(lookback)
    tp = (g["high"] + g["low"] + g["close"]) / 3
    hist, edges = np.histogram(tp, bins=bins, weights=g["volume"].astype(float))
    k = int(np.argmax(hist))
    return float((edges[k] + edges[k + 1]) / 2)


# A number (optionally with Korean/Chinese 10k unit) that is NOT a percentage, count, date, multiple or index level.
_UNITS_NOT_PRICE = r"%|％|億|亿|倍|年|月|日|號|号|個|个|名|人|位|宗|間|间|天|分|時|时|點|点|년|월|일|개|명|주|배|조|억|위|시|분|포인트|bp|x|k"
_NUM = re.compile(r"(?<![\d.第])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(만|萬|万)?(?:\s*(\d)천)?(?!\s*(?:" + _UNITS_NOT_PRICE + r"))(?![\d.,])")


def crowd_levels(texts: list[str], price: float, code: str, top: int = 3) -> list[dict]:
    vals = []
    code_n = code.lstrip("0")
    for t in texts:
        for m in _NUM.finditer(t or ""):
            raw = m.group(1).replace(",", "")
            if raw.startswith("0") and "." not in raw:   # 00700, 005930 = tickers, not prices
                continue
            if raw == code_n:
                continue
            v = float(raw)
            if m.group(2):
                v *= 10_000
                if m.group(3):
                    v += int(m.group(3)) * 1000
            if 0.75 * price <= v <= 1.25 * price and not (1990 <= v <= 2035 and not m.group(2)):
                vals.append(v)
    if not vals:
        return []
    step = 10 ** max(0, int(np.floor(np.log10(price))) - 2)  # round to 3 significant figures
    c = Counter(round(v / step) * step for v in vals)
    return [{"price": float(p), "mentions": n} for p, n in c.most_common(top) if n >= 2]


def compute_levels(g: pd.DataFrame, texts: list[str], code: str) -> dict:
    g = g.sort_values("date")
    last = g.iloc[-1]
    price = float(last["close"])
    a = atr(g)
    swings = swing_levels(g, a)
    below = [s for s in swings if s["price"] < price - 0.1 * a]
    above = [s for s in swings if s["price"] > price + 0.1 * a]
    # strongest level within 3 ATR, otherwise the nearest one
    def pick(levels, side):
        near = [s for s in levels if abs(s["price"] - price) <= 3 * a]
        if near:
            return max(near, key=lambda s: (s["strength"], -abs(s["price"] - price)))
        if levels:
            return min(levels, key=lambda s: abs(s["price"] - price))
        return None
    sup, res = pick(below, "S"), pick(above, "R")
    piv = pivots(last)
    return {
        "price": price, "atr": a, "pivots": piv, "poc": volume_poc(g),
        "support": sup["price"] if sup else piv["S1"], "support_touches": sup["touches"] if sup else 0,
        "resistance": res["price"] if res else piv["R1"], "resistance_touches": res["touches"] if res else 0,
        "swing_levels": sorted(swings, key=lambda s: -s["strength"])[:8],
        "crowd_levels": crowd_levels(texts, price, code),
    }


def trade_plan(price: float, pred_ret: float, vol: float, sent: float, lv: dict) -> dict:
    """Turns the model output into a position. Requires model and crowd to agree."""
    thr = 0.25 * (vol if vol and not np.isnan(vol) else 0.02)
    sent_label = "Bullish" if sent > 0.08 else "Bearish" if sent < -0.08 else "Neutral"
    a = lv["atr"]
    if pred_ret > thr and sent_label != "Bearish":
        side = "LONG"
        stop = max(lv["support"] - 0.25 * a, price - 2 * a)
        target = max(lv["resistance"], price + a)
    elif pred_ret < -thr and sent_label != "Bullish":
        side = "SHORT"
        stop = min(lv["resistance"] + 0.25 * a, price + 2 * a)
        target = min(lv["support"], price - a)
    else:
        why = "model and crowd disagree" if (pred_ret > thr) or (pred_ret < -thr) else "predicted move below noise"
        return {"position": "FLAT", "reason": why, "entry": None, "stop": None, "target": None, "rr": None,
                "sentiment_label": sent_label}
    rr = abs(target - price) / max(abs(price - stop), 1e-9)
    reason = "model + crowd agree" + ("" if rr >= 1 else "; low reward/risk, size down")
    return {"position": side, "reason": reason, "entry": price, "stop": stop, "target": target, "rr": rr,
            "sentiment_label": sent_label}
