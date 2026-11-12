"""Stale-price guard: a crawl that misses the last completed session must trigger a direct refetch."""
from datetime import datetime, timezone

import pandas as pd

from hsagent import pipeline
from hsagent.config import load_config


def _ex(cfg, code):
    return next(e for e in cfg.exchanges if e.code == code)


def test_last_completed_session():
    cfg = load_config()
    hk, kr = _ex(cfg, "HKEX"), _ex(cfg, "KRX")
    mon_evening = datetime(2026, 11, 9, 20, 10, tzinfo=timezone.utc)      # the run that went stale
    assert pipeline.last_completed_session(hk, mon_evening) == pd.Timestamp("2026-11-09")
    assert pipeline.last_completed_session(kr, mon_evening) == pd.Timestamp("2026-11-09")
    mon_morning = datetime(2026, 11, 9, 2, 0, tzinfo=timezone.utc)        # markets open, Monday not closed yet
    assert pipeline.last_completed_session(hk, mon_morning) == pd.Timestamp("2026-11-06")
    saturday = datetime(2026, 11, 7, 18, 0, tzinfo=timezone.utc)
    assert pipeline.last_completed_session(kr, saturday) == pd.Timestamp("2026-11-06")


def _bars(tickers, dates):
    return pd.DataFrame([{"ticker": t, "date": pd.Timestamp(d), "close": 1.0} for t in tickers for d in dates])


def test_stale_tickers_and_refetch(monkeypatch):
    cfg = load_config(now_utc=datetime(2026, 11, 9, 20, 10, tzinfo=timezone.utc))
    syms = [a.ticker for a in cfg.all_assets()] + [e.index_symbol for e in cfg.exchanges]
    stale = _bars(syms, ["2026-11-05", "2026-11-06"])
    assert pipeline.stale_tickers(stale, cfg) == set(syms)
    assert pipeline.stale_tickers(_bars(syms, ["2026-11-06", "2026-11-09"]), cfg) == set()

    fetched = {}

    def fake_parse(records, exchanges):
        if records == "fresh":
            return _bars(fetched["tickers"], ["2026-11-06", "2026-11-09"]).assign(
                close_utc=pd.Timestamp("2026-11-09 08:00", tz="UTC")), {}
        return stale.assign(close_utc=pd.Timestamp("2026-11-06 08:00", tz="UTC")), {}

    def fake_crawl(reqs):
        fetched["tickers"] = sorted(r["userData"]["ticker"] for r in reqs)
        return "fresh"

    monkeypatch.setattr(pipeline.parsers, "records_to_prices", fake_parse)
    monkeypatch.setattr(pipeline.local_crawler, "crawl", fake_crawl)
    prices, _ = pipeline.ensure_prices(cfg, [])
    assert fetched["tickers"] == sorted(syms)
    assert prices.groupby("ticker")["date"].max().eq(pd.Timestamp("2026-11-09")).all()

    fetched.clear()
    pipeline.ensure_prices(cfg, [], refresh_stale=False)                    # cached mode: never refetch
    assert not fetched


def _yahoo(bar_days, meta_utc, price=398.8):
    """Fake HK chart payload: daily bars stamped 01:30 UTC (09:30 HKT) plus a meta block."""
    ts = [int(datetime(2026, 11, d, 1, 30, tzinfo=timezone.utc).timestamp()) for d in bar_days]
    n = len(ts)
    meta = {"currency": "HKD", "regularMarketTime": int(meta_utc.timestamp()), "regularMarketPrice": price,
            "regularMarketDayHigh": 405.8, "regularMarketDayLow": 395.8, "regularMarketVolume": 25847007,
            "chartPreviousClose": 430.4}
    res = {"meta": meta, "timestamp": ts, "indicators": {"quote": [{
        "open": [410.0] * n, "high": [412.0] * n, "low": [405.0] * n, "close": [409.6] * n, "volume": [1] * n}]}}
    return [{"userData": {"kind": "yahoo_daily", "exchange": "HKEX", "ticker": "0700.HK"},
             "payload": {"chart": {"result": [res]}}}]


def test_meta_fills_session_missing_from_bars():
    from hsagent.ingest.parsers import records_to_prices
    cfg = load_config()
    # evening gap: bars stop at Nov 10, meta already printed Nov 11's close (16:08 HKT)
    p, _ = records_to_prices(_yahoo([9, 10], datetime(2026, 11, 11, 8, 8, tzinfo=timezone.utc)), cfg.exchanges)
    last = p.iloc[-1]
    assert last["date"] == pd.Timestamp("2026-11-11") and last["close"] == 398.8
    assert last["high"] == 405.8 and last["low"] == 395.8 and last["open"] == 409.6
    assert last["close_utc"] == pd.Timestamp("2026-11-11 08:00", tz="UTC")
    # normal case: the bar is already there -> nothing added
    p, _ = records_to_prices(_yahoo([10, 11], datetime(2026, 11, 11, 8, 8, tzinfo=timezone.utc)), cfg.exchanges)
    assert len(p) == 2
    # intraday print (11:00 HKT): session not finished -> no synthetic bar
    p, _ = records_to_prices(_yahoo([10, 11], datetime(2026, 11, 12, 3, 0, tzinfo=timezone.utc)), cfg.exchanges)
    assert p["date"].max() == pd.Timestamp("2026-11-11")
