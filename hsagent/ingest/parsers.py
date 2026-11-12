"""Turns raw crawl records (from Apify or the local crawler) into tidy tables.

posts  : one row per social post   (UTC timestamps, text, engagement, author followers)
prices : one row per daily bar      (local session date, OHLCV, UTC close timestamp)
"""
from __future__ import annotations

import html as htmlmod
import re
from datetime import datetime, timedelta, timezone

import pandas as pd
from bs4 import BeautifulSoup

POST_COLUMNS = ["post_id", "source", "exchange", "ticker", "ts_utc", "text", "likes", "dislikes",
                "replies", "views", "author", "author_followers", "holder_verified", "url", "channel", "keyword"]


def parse_count(s) -> float | None:
    """'1.2K' -> 1200, '3.28M' -> 3280000, '15' -> 15."""
    if s is None:
        return None
    s = str(s).strip().replace(",", "").upper()
    if not s:
        return None
    mult = 1.0
    if s[-1] in "KMB":
        mult = {"K": 1e3, "M": 1e6, "B": 1e9}[s[-1]]
        s = s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return None


def html_to_text(fragment: str) -> str:
    if not fragment:
        return ""
    soup = BeautifulSoup(fragment, "lxml")
    for q in soup.select("blockquote"):  # LIHKG quoted replies: drop the quoted part
        q.decompose()
    for br in soup.find_all("br"):
        br.replace_with("\n")
    return re.sub(r"[ \t\r\f\v]+", " ", htmlmod.unescape(soup.get_text())).strip()


# --- per source --------------------------------------------------------------

def _naver(rec: dict) -> list[dict]:
    ud = rec["userData"]
    code = ud["ticker"].split(".")[0]
    out = []
    for p in ((rec.get("payload") or {}).get("result") or {}).get("posts") or []:
        ts = datetime.fromisoformat(p["writtenAt"] + "+09:00").astimezone(timezone.utc)
        w = p.get("writer") or {}
        text = (p.get("title") or "") + "\n" + (p.get("contentSwReplacedButImg") or html_to_text(p.get("contentSwReplaced") or ""))
        out.append(dict(post_id=str(p["id"]), source="naver", exchange=ud["exchange"], ticker=ud["ticker"], ts_utc=ts,
                        text=text.strip(), likes=p.get("recommendCount") or 0, dislikes=p.get("notRecommendCount") or 0,
                        replies=p.get("commentCount") or p.get("replyCount") or 0, views=p.get("viewCount"),
                        author=w.get("nickname"), author_followers=None,
                        holder_verified=bool(w.get("isHolderVerified") or p.get("isHolderVerified")),
                        url=f"https://stock.naver.com/domestic/stock/{code}/discussion", channel=None, keyword=None))
    return out


def _lihkg_search(rec: dict) -> list[dict]:
    ud = rec["userData"]
    out = []
    for it in ((rec.get("payload") or {}).get("response") or {}).get("items") or []:
        user = it.get("user") or {}
        out.append(dict(post_id=f"t{it['thread_id']}", source="lihkg", exchange=ud["exchange"], ticker=ud["ticker"],
                        ts_utc=datetime.fromtimestamp(it["create_time"], timezone.utc), text=it.get("title") or "",
                        likes=it.get("like_count") or 0, dislikes=it.get("dislike_count") or 0,
                        replies=it.get("no_of_reply") or 0, views=None,
                        author=it.get("user_nickname") or user.get("nickname"), author_followers=None,
                        holder_verified=False, url=f"https://lihkg.com/thread/{it['thread_id']}/page/1",
                        channel=(it.get("category") or {}).get("name"), keyword=ud.get("keyword")))
    return out


def _lihkg_thread(rec: dict) -> list[dict]:
    ud = rec["userData"]
    resp = (rec.get("payload") or {}).get("response") or {}
    tid = ud.get("threadId") or resp.get("thread_id")
    title = resp.get("title") or ""
    out = []
    for i, it in enumerate(resp.get("item_data") or []):
        text = html_to_text(it.get("msg") or "")
        if i == 0 and title:
            text = f"{title}\n{text}"
        user = it.get("user") or {}
        out.append(dict(post_id=f"r{it.get('post_id')}", source="lihkg", exchange=ud["exchange"], ticker=ud["ticker"],
                        ts_utc=datetime.fromtimestamp(it["reply_time"], timezone.utc), text=text,
                        likes=it.get("like_count") or 0, dislikes=it.get("dislike_count") or 0, replies=0, views=None,
                        author=it.get("user_nickname") or user.get("nickname"), author_followers=None,
                        holder_verified=False, url=f"https://lihkg.com/thread/{tid}/page/1",
                        channel=(resp.get("category") or {}).get("name"), keyword=ud.get("keyword")))
    return out


def _telegram(rec: dict, subscribers: dict[str, float]) -> list[dict]:
    ud = rec["userData"]
    soup = BeautifulSoup(rec.get("payload") or "", "lxml")
    out = []
    for m in soup.select(".tgme_widget_message"):
        post = m.get("data-post") or ""
        tm = m.select_one(".tgme_widget_message_date time")
        if not post or not tm or not tm.get("datetime"):
            continue
        txt_el = m.select_one(".tgme_widget_message_text")
        text = html_to_text(str(txt_el)) if txt_el else ""
        if not text:
            continue
        views = parse_count((m.select_one(".tgme_widget_message_views") or {}).get_text() if m.select_one(".tgme_widget_message_views") else None)
        reactions = sum(parse_count(r.get_text()) or 0 for r in m.select(".tgme_reaction"))
        fwd = 1 if m.select_one(".tgme_widget_message_forwarded_from") else 0
        out.append(dict(post_id=post, source="telegram", exchange=ud["exchange"], ticker=ud["ticker"],
                        ts_utc=datetime.fromisoformat(tm["datetime"]).astimezone(timezone.utc), text=text,
                        likes=reactions, dislikes=0, replies=fwd, views=views, author=ud["channel"],
                        author_followers=subscribers.get(ud["channel"]), holder_verified=False,
                        url=f"https://t.me/{post}", channel=ud["channel"], keyword=ud.get("keyword")))
    return out


def records_to_posts(records: list[dict], cutoff_utc: datetime | None = None) -> pd.DataFrame:
    subs = {}
    for r in records:
        ud = r.get("userData") or {}
        if ud.get("source") == "telegram" and (r.get("meta") or {}).get("subscribers"):
            subs[ud["channel"]] = parse_count(r["meta"]["subscribers"])
    rows: list[dict] = []
    for r in records:
        kind = (r.get("userData") or {}).get("kind")
        if kind == "naver_list":
            rows += _naver(r)
        elif kind == "lihkg_search":
            rows += _lihkg_search(r)
        elif kind == "lihkg_thread":
            rows += _lihkg_thread(r)
        elif kind == "telegram_search":
            rows += _telegram(r, subs)
    df = pd.DataFrame(rows, columns=POST_COLUMNS)
    if df.empty:
        return df
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    if cutoff_utc is not None:
        df = df[df["ts_utc"] >= pd.Timestamp(cutoff_utc)]
    df = df[df["text"].str.len() > 0]
    # The same message can be found through several keywords/pages: keep one per ticker.
    df = df.drop_duplicates(subset=["source", "post_id", "ticker"]).sort_values("ts_utc").reset_index(drop=True)
    for c in ["likes", "dislikes", "replies"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


def _meta_bar(m: dict, ud: dict, zone, close_t, last: dict | None) -> dict | None:
    """Rebuild the last finished session's bar from the chart meta when the bars omit it.

    Yahoo drops the previous session's bar from the daily series for some hours after local midnight
    (seen at 17:40-20:10 UTC) while `meta` still carries that session's close, high, low and volume.
    Only used when the meta print is at/after the session close and newer than the last bar.
    """
    t, px = m.get("regularMarketTime"), m.get("regularMarketPrice")
    if not t or px is None:
        return None
    local = datetime.fromtimestamp(t, timezone.utc).astimezone(zone)
    session_close = datetime.combine(local.date(), close_t, tzinfo=zone)
    if local < session_close - timedelta(minutes=10):            # intraday print: session not finished
        return None
    d = pd.Timestamp(local.date())
    if last is not None and last["date"] >= d:
        return None
    prev = last["close"] if last is not None else m.get("chartPreviousClose") or px
    return dict(ticker=ud["ticker"], exchange=ud["exchange"], date=d, close_utc=session_close.astimezone(timezone.utc),
                open=prev, high=m.get("regularMarketDayHigh") or max(prev, px),
                low=m.get("regularMarketDayLow") or min(prev, px), close=px,
                volume=m.get("regularMarketVolume") or 0)


def records_to_prices(records: list[dict], exchanges) -> tuple[pd.DataFrame, dict]:
    """Daily bars + per-symbol meta (name, currency, last price) from Yahoo chart payloads."""
    close_times = {e.code: (e.zone, e.close_time) for e in exchanges}
    rows, meta = [], {}
    for r in records:
        ud = r.get("userData") or {}
        if ud.get("kind") != "yahoo_daily":
            continue
        res = (((r.get("payload") or {}).get("chart") or {}).get("result") or [None])[0]
        if not res or not res.get("timestamp"):
            continue
        q = res["indicators"]["quote"][0]
        m = res.get("meta") or {}
        zone, close_t = close_times[ud["exchange"]]
        meta[ud["ticker"]] = {"currency": m.get("currency"), "name": m.get("longName") or m.get("shortName"),
                              "regular_market_price": m.get("regularMarketPrice"),
                              "regular_market_time": m.get("regularMarketTime")}
        sym_rows = []
        for i, ts in enumerate(res["timestamp"]):
            if q["close"][i] is None:
                continue
            d = datetime.fromtimestamp(ts, timezone.utc).astimezone(zone).date()
            close_utc = datetime.combine(d, close_t, tzinfo=zone).astimezone(timezone.utc)
            sym_rows.append(dict(ticker=ud["ticker"], exchange=ud["exchange"], date=pd.Timestamp(d), close_utc=close_utc,
                                 open=q["open"][i], high=q["high"][i], low=q["low"][i], close=q["close"][i],
                                 volume=q["volume"][i] or 0))
        bar = _meta_bar(m, ud, zone, close_t, sym_rows[-1] if sym_rows else None)
        if bar:
            sym_rows.append(bar)
        rows += sym_rows
    df = pd.DataFrame(rows)
    if not df.empty:
        df["close_utc"] = pd.to_datetime(df["close_utc"], utc=True)
        df = df.drop_duplicates(["ticker", "date"], keep="last").sort_values(["ticker", "date"]).reset_index(drop=True)
    return df, meta
