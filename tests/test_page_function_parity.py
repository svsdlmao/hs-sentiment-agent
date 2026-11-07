"""The Apify pageFunction (JS) and the local crawler (Python) must paginate identically.

Runs apify/page_function.js through tests/js/harness.mjs (needs `npm install` in tests/js)
on saved real pages and compares the enqueued requests and extracted payloads with
hsagent.ingest.paginate. Skipped when Node or cheerio is not installed.
"""
import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from hsagent.ingest.paginate import next_requests, slim_naver, telegram_page_info
from hsagent.ingest.parsers import records_to_posts

HERE = Path(__file__).parent
FX = HERE / "fixtures"
HARNESS = HERE / "js" / "harness.mjs"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not (HERE / "js" / "node_modules" / "cheerio").exists(),
    reason="node + cheerio not installed (cd tests/js && npm install)")


def _cases():
    pages = json.loads((FX / "json_pages.json").read_text(encoding="utf-8"))
    tg_html = (FX / "telegram_search.html").read_text(encoding="utf-8")
    cases = []
    for name in ["naver", "lihkg"]:
        p = pages[name]
        for cutoff in [0, 4_000_000_000]:            # far past -> paginate; far future -> stop
            ud = dict(p["userData"], cutoffTs=cutoff, page=1)
            cases.append({"name": f"{name}-{cutoff}", "userData": ud, "url": p["url"],
                          "contentType": "json", "body": json.dumps(p["payload"], ensure_ascii=False)})
    tg_ud = {"source": "telegram", "kind": "telegram_search", "exchange": "KRX", "ticker": "005930.KS",
             "channel": "hedgecat0301", "keyword": "삼성전자", "page": 1, "maxPages": 4,
             "nextTemplate": "https://t.me/s/hedgecat0301?q=%EC%82%BC%EC%84%B1%EC%A0%84%EC%9E%90&before={cursor}"}
    for cutoff in [0, 4_000_000_000]:
        cases.append({"name": f"telegram-{cutoff}", "userData": dict(tg_ud, cutoffTs=cutoff),
                      "url": "https://t.me/s/hedgecat0301?q=x", "contentType": "html", "body": tg_html})
    return cases


def _run_js(cases, tmp_path):
    f = tmp_path / "cases.json"
    f.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
    out = subprocess.run(["node", str(HARNESS), str(f)], capture_output=True, check=True)
    return json.loads(out.stdout.decode("utf-8"))


def _norm(reqs):
    return sorted((r["url"], r["userData"].get("page"), r["userData"].get("kind"), r["userData"].get("threadId")) for r in reqs)


def test_pagination_parity(tmp_path):
    cases = _cases()
    js = _run_js(cases, tmp_path)
    for c, j in zip(cases, js):
        if c["contentType"] == "html":
            _, meta, ids, times = telegram_page_info(c["body"])
            py = next_requests(c["userData"], None, ids, times)
            assert j["meta"].get("subscribers") == meta.get("subscribers"), c["name"]
        else:
            py = next_requests(c["userData"], json.loads(c["body"]))
        assert _norm(j["enqueued"]) == _norm(py), c["name"]
    # sanity: the "far past" cutoffs must actually paginate, the "future" ones must not
    by = {c["name"]: j for c, j in zip(cases, js)}
    assert any(r["userData"].get("page") == 2 for r in by["naver-0"]["enqueued"])
    assert any(r["userData"].get("kind") == "lihkg_thread" for r in by["lihkg-0"]["enqueued"])
    assert by["telegram-0"]["enqueued"] and not by["telegram-4000000000"]["enqueued"]
    assert not [r for r in by["naver-4000000000"]["enqueued"] if r["userData"].get("page") == 2]


def test_telegram_payload_parses_identically(tmp_path):
    cases = [c for c in _cases() if c["name"] == "telegram-0"]
    js = _run_js(cases, tmp_path)[0]
    py_payload, meta, _, _ = telegram_page_info(cases[0]["body"])
    def posts(payload):
        rec = {"userData": cases[0]["userData"], "payload": payload, "meta": meta}
        return records_to_posts([rec])[["post_id", "ts_utc", "text", "views"]].reset_index(drop=True)
    a, b = posts(js["payload"]), posts(py_payload)
    assert len(a) > 0
    assert a.equals(b)


def test_slim_naver_keeps_parser_fields():
    p = json.loads((FX / "json_pages.json").read_text(encoding="utf-8"))["naver"]
    slim = slim_naver(copy.deepcopy(p["payload"]))
    df = records_to_posts([{"userData": p["userData"], "payload": slim}])
    assert len(df) == len(p["payload"]["result"]["posts"])
    assert df["ts_utc"].dt.tz is not None and df["text"].str.len().min() > 0
