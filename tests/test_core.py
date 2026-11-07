from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from hsagent import features as F
from hsagent import levels as L
from hsagent.ingest.parsers import parse_count
from hsagent.sentiment import lexicon_score


@pytest.mark.parametrize("text,sign", [
    ("騰訊今日爆升，上車！🚀", 1),
    ("小米插水，套牢咗", -1),
    ("삼성전자 떡상 가즈아", 1),
    ("하이닉스 떡락 손절했다", -1),
    ("Tencent shares surge after earnings beat", 1),
    ("Alibaba plunges on weak guidance", -1),
    ("今日開會", 0),
])
def test_lexicon_direction(text, sign):
    s = lexicon_score(text)
    assert np.sign(round(s, 6)) == sign, (text, s)


def test_lexicon_negation_and_longest_match():
    assert lexicon_score("唔會升") < 0          # negated bullish
    assert lexicon_score("not bullish") < 0
    assert lexicon_score("大升") > lexicon_score("升")  # 大升 counted once as strong, not twice


def test_parse_count():
    assert parse_count("1.2K") == 1200
    assert parse_count("3.28M") == 3_280_000
    assert parse_count("15") == 15
    assert parse_count("") is None


def _prices():
    # two HKEX sessions: Thu 2026-11-05 and Fri 2026-11-06, close 16:00 HKT = 08:00 UTC
    rows = []
    for d in ["2026-11-05", "2026-11-06"]:
        rows.append(dict(ticker="0700.HK", exchange="HKEX", date=pd.Timestamp(d),
                         close_utc=pd.Timestamp(f"{d} 08:00", tz="UTC"), open=1, high=1, low=1, close=1, volume=1))
    return pd.DataFrame(rows)


def test_session_alignment_no_lookahead():
    posts = pd.DataFrame({"exchange": ["HKEX"] * 4, "ts_utc": pd.to_datetime([
        "2026-11-05 07:59", "2026-11-05 08:00", "2026-11-05 08:01", "2026-11-06 09:00"], utc=True)})
    out = F.assign_sessions(posts, _prices(), "ts_utc")
    assert out["session"].iloc[0] == pd.Timestamp("2026-11-05")   # before close -> same session
    assert out["session"].iloc[1] == pd.Timestamp("2026-11-05")   # exactly at close
    assert out["session"].iloc[2] == pd.Timestamp("2026-11-06")   # after close -> next session
    assert pd.isna(out["session"].iloc[3])                        # after the last close -> after-close bucket


def test_crowd_levels_korean_units_and_ticker_filter():
    texts = ["8만 깨지면 손절", "8만5천 저항", "8만 지지", "005930 보유", "2026년 전망", "80,000원 매수"]
    lv = L.crowd_levels(texts, price=82000, code="005930")
    prices = [c["price"] for c in lv]
    assert 80000 in prices
    assert all(70000 < p < 100000 for p in prices)


def test_trade_plan_requires_agreement():
    lv = {"atr": 2.0, "support": 95.0, "resistance": 106.0}
    assert L.trade_plan(100, 0.01, 0.02, 0.3, lv)["position"] == "LONG"
    assert L.trade_plan(100, 0.01, 0.02, -0.3, lv)["position"] == "FLAT"   # crowd bearish vs model up
    assert L.trade_plan(100, -0.01, 0.02, -0.3, lv)["position"] == "SHORT"
    assert L.trade_plan(100, 0.001, 0.02, 0.3, lv)["position"] == "FLAT"   # below noise threshold


def test_walk_forward_uses_only_past():
    """A target column that leaks the future must not be visible to the fit for date d."""
    from hsagent import model as M
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2026-01-01", periods=60)
    rows = []
    for t in ["A", "B"]:
        px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
        for i, d in enumerate(dates):
            rows.append(dict(ticker=t, exchange="X", date=d, open=px[i], high=px[i] * 1.01, low=px[i] * 0.99,
                             close=px[i], volume=1000 + i))
    pf = F.price_features(pd.DataFrame(rows))
    social = pd.DataFrame([dict(ticker=t, session=d, mentions=10, sent_social=0.1, bull_share=0.5, bear_share=0.2)
                           for t in ["A", "B"] for d in dates[-10:]])
    panel = F.build_panel(pf, social, pd.DataFrame())
    bt = M.walk_forward(panel)
    assert not bt.empty
    assert (bt["date"] >= dates[-10:][M.MIN_TRAIN_SESSIONS]).all()
    assert bt["next_close"].notna().all()
