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
