"""Local fallback crawler (no Apify token needed).

Executes the same crawl plan as the Apify actor and produces records with the same
shape, so everything downstream is identical. Intended for development and for
re-running when Apify credits run out; the submission run uses Apify.
"""
from __future__ import annotations

import json
import logging
import socket
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests

from .paginate import next_requests, slim_naver, telegram_page_info

log = logging.getLogger(__name__)

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")

# --- DNS-over-HTTPS fallback -------------------------------------------------
# Some ISPs fail to resolve m.stock.naver.com; Apify's cloud does not have this
# problem, but the local crawler patches getaddrinfo to fall back to Cloudflare DoH.
_orig_getaddrinfo = socket.getaddrinfo
_doh_cache: dict[str, str] = {}


def _doh_resolve(host: str) -> str | None:
    if host in _doh_cache:
        return _doh_cache[host]
    try:
        r = requests.get("https://1.1.1.1/dns-query", params={"name": host, "type": "A"},
                         headers={"accept": "application/dns-json"}, timeout=10)
        for ans in r.json().get("Answer", []):
            if ans.get("type") == 1:
                _doh_cache[host] = ans["data"]
                return ans["data"]
    except Exception:  # noqa: BLE001
        return None
    return None


def _getaddrinfo(host, *args, **kwargs):
    try:
        return _orig_getaddrinfo(host, *args, **kwargs)
    except socket.gaierror:
        ip = _doh_resolve(host) if isinstance(host, str) else None
        if not ip:
            raise
        log.debug("DoH resolved %s -> %s", host, ip)
        return _orig_getaddrinfo(ip, *args, **kwargs)


socket.getaddrinfo = _getaddrinfo
# -----------------------------------------------------------------------------


# Concurrent requests allowed per host. LIHKG rate-limits a single IP hard (HTTP 429);
# on Apify the proxy rotates IPs so the actor can go faster.
HOST_LIMITS = {"lihkg.com": 1, "t.me": 3}
_host_sems: dict[str, threading.Semaphore] = {}
_sem_lock = threading.Lock()


def _host_sem(url: str) -> threading.Semaphore:
    host = urlparse(url).netloc
    with _sem_lock:
        if host not in _host_sems:
            _host_sems[host] = threading.Semaphore(HOST_LIMITS.get(host, 4))
        return _host_sems[host]


# Circuit breaker: a host that answers 429 with a long Retry-After (LIHKG bans an IP for ~24 h)
# is skipped for the rest of the run instead of stalling the crawl. Reported in the run log.
BLOCKED_HOSTS: set[str] = set()
MAX_BACKOFF_S = 120


def _fetch(session: requests.Session, req: dict, retries: int = 5) -> dict | None:
    if urlparse(req["url"]).netloc in BLOCKED_HOSTS:
        return None
    with _host_sem(req["url"]):
        return _fetch_inner(session, req, retries)


def _fetch_inner(session: requests.Session, req: dict, retries: int) -> dict | None:
    ud = req["userData"]
    host = urlparse(req["url"]).netloc
    headers = {"User-Agent": UA, "Accept-Language": "en-US,en;q=0.8,zh-HK;q=0.6,ko;q=0.5", **(req.get("headers") or {})}
    for attempt in range(retries):
        if host in BLOCKED_HOSTS:
            return None
        try:
            r = session.get(req["url"], headers=headers, timeout=30)
            if r.status_code == 429 or (r.status_code == 403 and "1015" in r.text[:200]):
                try:
                    retry_after = float(r.headers.get("Retry-After") or 0)
                except ValueError:
                    retry_after = 0
                if retry_after > MAX_BACKOFF_S:
                    BLOCKED_HOSTS.add(host)
                    log.error("%s rate-limited this IP for %.1f h; skipping it for this run "
                              "(use --mode apify: the Apify proxy rotates IPs)", host, retry_after / 3600)
                    return None
                wait = retry_after or 5 * 2 ** attempt
                log.info("429 from %s, backing off %.0fs", host, wait)
                time.sleep(wait)
                continue
            if r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            rec = {"url": req["url"], "userData": ud, "meta": {},
                   "fetchedAt": datetime.now(timezone.utc).isoformat()}
            if ud["kind"].startswith("telegram"):
                payload, meta, ids, times = telegram_page_info(r.text)
                rec.update(payloadType="html", payload="" if ud["kind"] == "telegram_channel" else payload, meta=meta)
                rec["_tg"] = (ids, times)
            else:
                payload = r.json()
                rec.update(payloadType="json", payload=slim_naver(payload) if ud["kind"] == "naver_list" else payload)
            return rec
        except (requests.RequestException, json.JSONDecodeError) as e:
            wait = 2 ** attempt
            log.warning("fetch failed (%s) %s - retry in %ss", e, req["url"][:120], wait)
            time.sleep(wait)
    log.error("giving up on %s", req["url"][:160])
    return None


def crawl(plan: list[dict], max_workers: int = 6, polite_delay: float = 0.15) -> list[dict]:
    """Breadth-first crawl of the plan, following the same pagination as the Apify actor."""
    seen: set[str] = set()
    queue = deque()
    for r in plan:
        key = r.get("uniqueKey") or r["url"]
        if key not in seen:
            seen.add(key)
            queue.append(r)
    records: list[dict] = []
    per_host_sessions: dict[str, requests.Session] = {}

    def session_for(url: str) -> requests.Session:
        host = urlparse(url).netloc
        return per_host_sessions.setdefault(host, requests.Session())

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        while queue:
            batch = [queue.popleft() for _ in range(min(len(queue), max_workers * 2))]
            for rec in pool.map(lambda q: _fetch(session_for(q["url"]), q), batch):
                if rec is None:
                    continue
                ids, times = rec.pop("_tg", ([], []))
                for n in next_requests(rec["userData"], rec["payload"], ids, times):
                    key = f"{n['url']}#{n['userData'].get('ticker', '')}"
                    if key not in seen:
                        seen.add(key)
                        queue.append({"url": n["url"], "headers": n["userData"].get("headers") or {},
                                      "userData": n["userData"], "uniqueKey": key})
                records.append(rec)
            time.sleep(polite_delay)
            log.info("local crawl: %d records, %d queued", len(records), len(queue))
    return records
