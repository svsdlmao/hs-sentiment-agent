"""Python mirror of the pagination logic in apify/page_function.js.

Used by the local fallback crawler. Must stay behaviourally identical to the JS
version; tests/test_page_function_parity.py runs both on the same fixtures.
"""
from __future__ import annotations

from datetime import datetime
from urllib.parse import quote

from bs4 import BeautifulSoup

KST_SUFFIX = "+09:00"


NAVER_KEEP = ("id", "writtenAt", "title", "contentSwReplacedButImg", "recommendCount", "notRecommendCount",
              "viewCount", "commentCount", "replyCount", "isHolderVerified", "isCleanbotPassed")


def slim_naver(payload: dict) -> dict:
    """Mirror of slimNaver() in page_function.js."""
    res = (payload or {}).get("result") or {}
    posts = []
    for x in res.get("posts") or []:
        o = {k: x[k] for k in NAVER_KEEP if k in x}
        w = x.get("writer") or {}
        o["writer"] = {"nickname": w.get("nickname"), "isHolderVerified": w.get("isHolderVerified")}
        posts.append(o)
    return {"result": {"lastOffset": res.get("lastOffset"), "posts": posts}}


def _next_url(ud: dict, cursor) -> str:
    return ud["nextTemplate"].replace("{cursor}", quote(str(cursor), safe=""))


def telegram_page_info(html: str) -> tuple[str, dict, list[int], list[float]]:
    """Return (messages_html, meta, ids, unix_times) for a t.me/s/ page."""
    soup = BeautifulSoup(html, "lxml")
    payload = "\n".join(str(el) for el in soup.select(".tgme_widget_message_wrap"))
    meta: dict = {}
    for el in soup.select(".tgme_channel_info_counter"):
        t = el.select_one(".counter_type")
        v = el.select_one(".counter_value")
        if t and v and "subscriber" in t.get_text().lower():
            meta["subscribers"] = v.get_text().strip()
    if "subscribers" not in meta:
        h = soup.select_one(".tgme_header_counter")
        if h and "subscriber" in h.get_text().lower():
            meta["subscribers"] = h.get_text().lower().replace("subscribers", "").replace("subscriber", "").strip()
    ids, times = [], []
    for el in soup.select(".tgme_widget_message"):
        post = el.get("data-post") or ""
        tm = el.select_one(".tgme_widget_message_date time")
        try:
            mid = int(post.split("/")[-1])
        except ValueError:
            continue
        if tm and tm.get("datetime"):
            ids.append(mid)
            times.append(datetime.fromisoformat(tm["datetime"]).timestamp())
    return payload, meta, ids, times


def next_requests(ud: dict, payload, tg_ids: list[int] | None = None, tg_times: list[float] | None = None) -> list[dict]:
    kind = ud.get("kind")
    page = ud.get("page") or 1
    more = page < (ud.get("maxPages") or 1)
    out: list[dict] = []
    if kind == "naver_list":
        res = (payload or {}).get("result") or {}
        posts = res.get("posts") or []
        if posts and more and res.get("lastOffset"):
            oldest = min(datetime.fromisoformat(p["writtenAt"] + KST_SUFFIX).timestamp() for p in posts)
            if oldest > ud["cutoffTs"]:
                out.append({"url": _next_url(ud, res["lastOffset"]), "userData": {**ud, "page": page + 1}})
    elif kind == "lihkg_search":
        items = ((payload or {}).get("response") or {}).get("items") or []
        if len(items) >= 30 and more:
            if min(i["create_time"] for i in items) > ud["cutoffTs"]:
                out.append({"url": _next_url(ud, page + 1), "userData": {**ud, "page": page + 1}})
        active = [i for i in items if (i.get("last_reply_time") or i["create_time"]) >= ud["cutoffTs"]]
        active.sort(key=lambda i: (-(i.get("no_of_reply") or 0), str(i["thread_id"])))
        for t in active[: ud.get("threadsPerPage") or 10]:
            out.append({
                "url": f"https://lihkg.com/api_v2/thread/{t['thread_id']}/page/1?order=reply_time",
                "userData": {"source": "lihkg", "kind": "lihkg_thread", "exchange": ud["exchange"],
                             "ticker": ud["ticker"], "keyword": ud.get("keyword"),
                             "threadId": str(t["thread_id"]), "cutoffTs": ud["cutoffTs"],
                             "headers": ud.get("headers")},
            })
    elif kind == "telegram_search":
        if tg_ids and more and min(tg_times) > ud["cutoffTs"]:
            out.append({"url": _next_url(ud, min(tg_ids)), "userData": {**ud, "page": page + 1}})
    return out
