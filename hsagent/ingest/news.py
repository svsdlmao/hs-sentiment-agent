"""Recent news per asset from Exa (https://exa.ai), the task's news source.

Two searches per asset: an English query and a local-language query (Chinese / Korean),
restricted to the look-back window and Exa's `news` category.
Without EXA_API_KEY the module falls back to Google News RSS so the pipeline still
runs end-to-end; every row records which provider it came from.
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from urllib.parse import quote

import pandas as pd
import requests

from ..config import Asset, Config

log = logging.getLogger(__name__)
EXA_URL = "https://api.exa.ai/search"
NEWS_COLUMNS = ["ticker", "exchange", "provider", "query", "published_utc", "title", "url", "text", "source_domain"]


def _queries(a: Asset) -> list[str]:
    local = a.keywords[0]
    return [f"{a.name} ({a.ticker}) stock news", f"{local} 股價" if a.exchange == "HKEX" else f"{local} 주가"]


def exa_search(api_key: str, query: str, cfg: Config) -> tuple[list[dict], str | None]:
    body = {
        "query": query,
        "type": "auto",
        "category": "news",
        "numResults": int(cfg.exa["num_results"]),
        "startPublishedDate": cfg.cutoff_utc.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "endPublishedDate": cfg.now_utc.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "contents": {"text": {"maxCharacters": int(cfg.exa["max_characters"])},
                     "highlights": {"numSentences": 2, "highlightsPerUrl": 2}},
    }
    r = requests.post(EXA_URL, json=body, headers={"x-api-key": api_key, "Content-Type": "application/json"}, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"Exa HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    return data.get("results") or [], data.get("requestId")


def google_news_rss(query: str, lang: str) -> list[dict]:
    hl, gl = {"en": ("en-US", "US"), "zh": ("zh-HK", "HK"), "ko": ("ko", "KR")}[lang]
    url = f"https://news.google.com/rss/search?q={quote(query + ' when:10d')}&hl={hl}&gl={gl}&ceid={gl}:{hl.split('-')[0]}"
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    out = []
    for item in ET.fromstring(r.content).iter("item"):
        src = item.find("source")
        out.append({"title": item.findtext("title"), "url": item.findtext("link"),
                    "publishedDate": parsedate_to_datetime(item.findtext("pubDate")).isoformat(),
                    "text": "", "source": src.text if src is not None else None})
    return out


def fetch_news(cfg: Config, api_key: str | None) -> tuple[pd.DataFrame, list[dict]]:
    rows, audit = [], []
    for a in cfg.all_assets():
        for i, q in enumerate(_queries(a)):
            try:
                if api_key:
                    results, req_id = exa_search(api_key, q, cfg)
                    audit.append({"provider": "exa", "ticker": a.ticker, "query": q, "request_id": req_id, "results": len(results)})
                    for x in results:
                        text = " ".join(x.get("highlights") or []) or (x.get("text") or "")
                        rows.append(dict(ticker=a.ticker, exchange=a.exchange, provider="exa", query=q,
                                         published_utc=x.get("publishedDate"), title=x.get("title") or "", url=x.get("url"),
                                         text=text[:2000], source_domain=(x.get("url") or "").split("/")[2] if x.get("url") else None))
                else:
                    lang = "en" if i == 0 else ("zh" if a.exchange == "HKEX" else "ko")
                    results = google_news_rss(q.replace(" stock news", " stock"), lang)[: int(cfg.exa["num_results"])]
                    audit.append({"provider": "google_news_rss_fallback", "ticker": a.ticker, "query": q, "results": len(results)})
                    for x in results:
                        rows.append(dict(ticker=a.ticker, exchange=a.exchange, provider="google_news_rss_fallback", query=q,
                                         published_utc=x["publishedDate"], title=x["title"] or "", url=x["url"], text="",
                                         source_domain=x.get("source")))
            except Exception as e:  # noqa: BLE001 - one failed query should not stop the run
                log.warning("news query failed for %s (%s): %s", a.ticker, q, e)
                audit.append({"provider": "exa" if api_key else "google_news_rss_fallback", "ticker": a.ticker, "query": q, "error": str(e)})
    df = pd.DataFrame(rows, columns=NEWS_COLUMNS)
    if not df.empty:
        df["published_utc"] = pd.to_datetime(df["published_utc"], utc=True, errors="coerce")
        df = df.dropna(subset=["published_utc"])
        df = df[df["published_utc"] >= pd.Timestamp(cfg.cutoff_utc)]
        df = df.drop_duplicates(["ticker", "url"]).sort_values("published_utc").reset_index(drop=True)
    return df, audit
