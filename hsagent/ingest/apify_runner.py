"""Runs the crawl plan on Apify (actor `apify/cheerio-scraper`) through the Apify REST API."""
from __future__ import annotations

import logging
import time

import requests

from ..config import ROOT, Config

log = logging.getLogger(__name__)
API = "https://api.apify.com/v2"


class ApifyError(RuntimeError):
    pass


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def build_actor_input(cfg: Config, plan: list[dict]) -> dict:
    proxy = {"useApifyProxy": True}
    if cfg.apify.get("proxy_groups"):
        proxy["apifyProxyGroups"] = list(cfg.apify["proxy_groups"])
    return {
        "startUrls": plan,
        "pageFunction": (ROOT / "apify" / "page_function.js").read_text(encoding="utf-8"),
        "proxyConfiguration": proxy,
        "additionalMimeTypes": ["application/json", "text/json"],
        "maxConcurrency": int(cfg.apify.get("max_concurrency", 8)),
        "maxRequestRetries": 4,
        "requestTimeoutSecs": 45,
        "ignoreSslErrors": False,
        "keepUrlFragments": False,
        "debugLog": False,
    }


def fetch_dataset(token: str, dataset_id: str, page: int = 500) -> list[dict]:
    items: list[dict] = []
    offset = 0
    while True:
        r = requests.get(f"{API}/datasets/{dataset_id}/items", headers=_headers(token),
                         params={"clean": 1, "format": "json", "offset": offset, "limit": page}, timeout=120)
        r.raise_for_status()
        chunk = r.json()
        items += chunk
        if len(chunk) < page:
            return items
        offset += page


def run_plan(cfg: Config, token: str, plan: list[dict]) -> tuple[list[dict], dict]:
    actor = cfg.apify["actor_id"].replace("/", "~")
    r = requests.post(f"{API}/acts/{actor}/runs", headers=_headers(token), json=build_actor_input(cfg, plan),
                      params={"memory": cfg.apify.get("memory_mbytes", 1024),
                              "timeout": cfg.apify.get("timeout_secs", 1800)}, timeout=120)
    if r.status_code >= 400:
        raise ApifyError(f"could not start actor {cfg.apify['actor_id']}: HTTP {r.status_code} {r.text[:300]}")
    run = r.json()["data"]
    log.info("Apify run started: %s (https://console.apify.com/view/runs/%s)", run["id"], run["id"])
    while run["status"] in ("READY", "RUNNING"):
        rr = requests.get(f"{API}/actor-runs/{run['id']}", headers=_headers(token),
                          params={"waitForFinish": 60}, timeout=120)
        rr.raise_for_status()
        run = rr.json()["data"]
        stats = run.get("stats") or {}
        log.info("  status=%s requests_finished=%s", run["status"], stats.get("requestsFinished"))
        time.sleep(1)
    if run["status"] != "SUCCEEDED":
        raise ApifyError(f"Apify run {run['id']} ended with status {run['status']}")
    items = fetch_dataset(token, run["defaultDatasetId"])
    info = {
        "actor_id": cfg.apify["actor_id"], "actor_actId": run.get("actId"), "run_id": run["id"],
        "dataset_id": run["defaultDatasetId"], "status": run["status"],
        "started_at": run.get("startedAt"), "finished_at": run.get("finishedAt"),
        "requests_finished": (run.get("stats") or {}).get("requestsFinished"),
        "requests_failed": (run.get("stats") or {}).get("requestsFailed"),
        "console_url": f"https://console.apify.com/view/runs/{run['id']}",
        "items": len(items),
    }
    log.info("Apify run %s finished: %d dataset items", run["id"], len(items))
    return items, info
