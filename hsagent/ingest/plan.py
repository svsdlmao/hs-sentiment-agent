"""Builds the crawl plan: the start requests for every source.

The same request list is fed to the Apify actor (as `startUrls`) and to the
local fallback crawler, so both paths fetch exactly the same pages. Pagination
rules live in `apify/page_function.js` (Apify) and `paginate.py` (local) and
are kept in sync by tests/test_page_function_parity.py.
"""
from __future__ import annotations

from datetime import timedelta
from urllib.parse import quote

from ..config import Asset, Config, Exchange

LIHKG_HEADERS = {"Referer": "https://lihkg.com/", "X-LI-DEVICE-TYPE": "browser"}
NAVER_HEADERS = {"Referer": "https://m.stock.naver.com/"}

NAVER_LIST = ("https://m.stock.naver.com/front-api/discussion/list?discussionType=domesticStock"
              "&itemCode={code}&pageSize={size}")
LIHKG_SEARCH = "https://lihkg.com/api_v2/thread/search?q={q}&page={{cursor}}&count=30&sort=desc_create_time&type=thread"
YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1={p1}&period2={p2}&interval=1d&includePrePost=false"


def _req(url: str, user_data: dict, headers: dict | None = None) -> dict:
    headers = headers or {}
    ud = dict(user_data, headers=headers)
    # uniqueKey includes the ticker so a page relevant to two tickers is kept for both.
    return {"url": url, "method": "GET", "headers": headers, "userData": ud,
            "uniqueKey": f"{url}#{ud.get('ticker', '')}"}


def naver_requests(cfg: Config, ex: Exchange, a: Asset) -> list[dict]:
    lim = cfg.limits
    base = NAVER_LIST.format(code=a.code, size=lim["naver_page_size"])
    return [_req(base, {
        "source": "naver", "kind": "naver_list", "exchange": ex.code, "ticker": a.ticker,
        "page": 1, "maxPages": lim["naver_max_pages"], "cutoffTs": int(cfg.cutoff_utc.timestamp()),
        "nextTemplate": base + "&offset={cursor}",
    }, NAVER_HEADERS)]


def lihkg_requests(cfg: Config, ex: Exchange, a: Asset) -> list[dict]:
    lim = cfg.limits
    out = []
    for kw in [k for k in a.keywords if not k.isdigit()]:  # names only; LIHKG search ignores stock codes
        tmpl = LIHKG_SEARCH.format(q=quote(kw))
        out.append(_req(tmpl.replace("{cursor}", "1"), {
            "source": "lihkg", "kind": "lihkg_search", "exchange": ex.code, "ticker": a.ticker,
            "keyword": kw, "page": 1, "maxPages": lim["lihkg_max_search_pages"],
            "threadsPerPage": lim["lihkg_threads_per_page"],
            "cutoffTs": int(cfg.cutoff_utc.timestamp()), "nextTemplate": tmpl,
        }, LIHKG_HEADERS))
    return out


def search_terms(a: Asset) -> list[str]:
    """Company names/nicknames from config plus the ticker code (00700, 005930).

    Cashtags ($0700, $005930) were tested on t.me search and return nothing (it drops the '$'),
    so the bare code is what catches cashtag-style mentions.
    """
    terms = list(a.keywords)
    if not any(k.isdigit() for k in terms):
        terms.append(a.code.zfill(5) if a.ticker.endswith(".HK") else a.code)
    return terms


def telegram_requests(cfg: Config, ex: Exchange, a: Asset) -> list[dict]:
    out = []
    for ch in ex.telegram_channels:
        for kw in search_terms(a):
            base = f"https://t.me/s/{ch}?q={quote(kw)}"
            out.append(_req(base, {
                "source": "telegram", "kind": "telegram_search", "exchange": ex.code,
                "ticker": a.ticker, "channel": ch, "keyword": kw, "page": 1,
                "maxPages": cfg.limits["telegram_max_pages"],
                "cutoffTs": int(cfg.cutoff_utc.timestamp()), "nextTemplate": base + "&before={cursor}",
            }))
    return out


def telegram_channel_requests(ex: Exchange) -> list[dict]:
    """One plain channel page each, only to read the subscriber count (author follower count)."""
    return [_req(f"https://t.me/s/{ch}", {"source": "telegram", "kind": "telegram_channel",
                                         "exchange": ex.code, "channel": ch}) for ch in ex.telegram_channels]


def yahoo_requests(cfg: Config) -> list[dict]:
    p2 = int(cfg.now_utc.timestamp()) + 86400
    p1 = int((cfg.now_utc - timedelta(days=int(cfg.run["price_history_days"]))).timestamp())
    out = []
    for ex in cfg.exchanges:
        syms = [(a.ticker, a.ticker) for a in ex.assets] + ([(ex.index_symbol, ex.index_symbol)] if ex.index_symbol else [])
        for sym, tick in syms:
            out.append(_req(YAHOO_CHART.format(sym=quote(sym), p1=p1, p2=p2),
                            {"source": "yahoo", "kind": "yahoo_daily", "exchange": ex.code, "ticker": tick}))
    return out


SOURCE_BUILDERS = {"naver": naver_requests, "lihkg": lihkg_requests, "telegram": telegram_requests}


def build_plan(cfg: Config, include_prices: bool = True) -> list[dict]:
    reqs: list[dict] = []
    for ex in cfg.exchanges:
        for src in ex.sources:
            for a in ex.assets:
                reqs += SOURCE_BUILDERS[src](cfg, ex, a)
        if "telegram" in ex.sources:
            reqs += telegram_channel_requests(ex)
    if include_prices:
        reqs += yahoo_requests(cfg)
    return reqs
