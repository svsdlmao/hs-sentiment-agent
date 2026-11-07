"""End-to-end run: ingest -> score -> align -> features -> backtest -> n+1 prediction -> report."""
from __future__ import annotations

import gzip
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as F
from . import levels as L
from . import model as M
from .config import Config
from .ingest import apify_runner, local_crawler, news as news_mod, parsers, plan as plan_mod
from .sentiment import score_frame

log = logging.getLogger(__name__)


# --- ingestion ----------------------------------------------------------------

def ingest(cfg: Config, mode: str, dataset_id: str | None = None, records_path: list[str] | str | None = None) -> tuple[list[dict], dict]:
    out = cfg.output_dir
    raw_dir = out / "raw"
    raw_dir.mkdir(exist_ok=True)
    info: dict = {"mode": mode}
    if mode == "cached":
        if isinstance(records_path, str):
            records_path = [records_path]
        paths = [Path(p) for p in records_path] if records_path else             [max(raw_dir.glob("records_*.json.gz"), key=lambda p: p.stat().st_mtime)]
        records = []
        for path in paths:  # several crawls can be merged; duplicate posts are dropped in parsing
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                chunk = json.load(fh)
            records += chunk
            log.info("loaded %d cached records from %s", len(chunk), path)
        info["records_path"] = [str(p) for p in paths]
        return records, info
    plan = plan_mod.build_plan(cfg)
    log.info("crawl plan: %d start requests", len(plan))
    if mode == "apify":
        token = os.environ.get("APIFY_TOKEN")
        if not token:
            raise SystemExit("APIFY_TOKEN is not set (put it in .env). Use --mode local to crawl without Apify.")
        if dataset_id:
            records = apify_runner.fetch_dataset(token, dataset_id)
            info.update(dataset_id=dataset_id, actor_id=cfg.apify["actor_id"])
        else:
            records, run_info = apify_runner.run_plan(cfg, token, plan)
            info.update(run_info)
        runs_file = out / "apify_runs.json"
        hist = json.loads(runs_file.read_text(encoding="utf-8")) if runs_file.exists() else []
        hist.append({**info, "logged_at": datetime.now(timezone.utc).isoformat()})
        runs_file.write_text(json.dumps(hist, indent=2), encoding="utf-8")
    elif mode == "local":
        records = local_crawler.crawl(plan)
    else:
        raise ValueError(mode)
    stamp = cfg.now_utc.strftime("%Y%m%dT%H%M%SZ")
    path = raw_dir / f"records_{stamp}.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False)
    info["records_path"] = str(path)
    info["records"] = len(records)
    return records, info


def ensure_prices(cfg: Config, records: list[dict]) -> tuple[pd.DataFrame, dict]:
    prices, meta = parsers.records_to_prices(records, cfg.exchanges)
    wanted = {a.ticker for a in cfg.all_assets()} | {e.index_symbol for e in cfg.exchanges if e.index_symbol}
    have = set(prices["ticker"]) if not prices.empty else set()
    missing = wanted - have
    if missing:
        log.warning("price bars missing from the crawl for %s; fetching them directly", sorted(missing))
        extra = [r for r in plan_mod.yahoo_requests(cfg) if r["userData"]["ticker"] in missing]
        p2, m2 = parsers.records_to_prices(local_crawler.crawl(extra), cfg.exchanges)
        prices = pd.concat([prices, p2], ignore_index=True)
        meta.update(m2)
    prices = prices[prices["close_utc"] <= pd.Timestamp(cfg.now_utc)]  # drop the in-progress session
    return prices.reset_index(drop=True), meta


# --- helpers ------------------------------------------------------------------

def next_weekday(d: pd.Timestamp) -> pd.Timestamp:
    d = d + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _coverage(posts: pd.DataFrame, cfg: Config) -> dict:
    cov = {}
    cutoff = pd.Timestamp(cfg.cutoff_utc)
    for (t, src), g in posts.groupby(["ticker", "source"]):
        oldest = g["ts_utc"].min()
        cov[f"{t}|{src}"] = {"oldest": oldest.isoformat(), "posts": int(len(g)),
                             "complete": bool(oldest <= cutoff + pd.Timedelta(hours=12)) or src != "naver"}
    return cov


def _top_posts(g: pd.DataFrame, k: int = 3) -> dict:
    def fmt(df):
        return [{"source": r.source, "ts": r.ts_utc.strftime("%Y-%m-%d %H:%M"), "score": round(float(r.score), 2),
                 "text": " ".join(str(r.text).split())[:160], "url": r.url, "w": round(float(r.w), 1)}
                for r in df.itertuples()]
    return {"bull": fmt(g[g["score"] > 0.1].nlargest(k, "w")), "bear": fmt(g[g["score"] < -0.1].nlargest(k, "w"))}


# --- main ---------------------------------------------------------------------

def run(cfg: Config, mode: str = "apify", scorer: str = "lexicon", with_news: bool = True,
        dataset_id: str | None = None, records_path: list[str] | None = None, analyst: str = "auto") -> dict:
    out = cfg.output_dir
    records, ingest_info = ingest(cfg, mode, dataset_id, records_path)
    prices, meta = ensure_prices(cfg, records)
    names = {a.ticker: a.name for a in cfg.all_assets()}

    # social posts
    posts = parsers.records_to_posts(records, cfg.cutoff_utc)
    log.info("parsed %d posts", len(posts))
    posts["score"] = score_frame(posts, "text", scorer, names)
    posts["w"] = F.engagement_weight(posts)
    posts["score_adj"] = F.crowd_adjusted_score(posts)
    posts = F.assign_sessions(posts, prices, "ts_utc")

    # news
    news_audit: list[dict] = []
    news = pd.DataFrame(columns=news_mod.NEWS_COLUMNS)
    if with_news:
        news, news_audit = news_mod.fetch_news(cfg, os.environ.get("EXA_API_KEY"))
    if not news.empty:
        news["score"] = score_frame(news.assign(text_all=news["title"] + " " + news["text"].fillna("")), "text_all", scorer, names)
        news = F.assign_sessions(news, prices, "published_utc")
    else:
        news["score"] = pd.Series(dtype=float)
        news["session"] = pd.Series(dtype="datetime64[ns]")

    # features
    idx_syms = {e.index_symbol for e in cfg.exchanges}
    asset_px = prices[~prices["ticker"].isin(idx_syms)]
    index_px = prices[prices["ticker"].isin(idx_syms)]
    pf = F.price_features(asset_px, index_px)
    panel = F.build_panel(pf, F.daily_social(posts), F.daily_news(news))

    # out-of-sample check + n+1 prediction
    bt = M.walk_forward(panel)
    metrics = M.score_backtest(bt)
    nxt = M.predict_next(panel)

    # per-ticker summary
    coverage = _coverage(posts, cfg)
    summary = []
    for a in cfg.all_assets():
        g_px = asset_px[asset_px["ticker"] == a.ticker]
        if g_px.empty:
            log.warning("no prices for %s, skipped", a.ticker)
            continue
        gp = posts[posts["ticker"] == a.ticker]
        gn = news[news["ticker"] == a.ticker]
        row = nxt[nxt["ticker"] == a.ticker].iloc[0]
        pan = panel[(panel["ticker"] == a.ticker) & panel["in_social_window"]].sort_values("date")
        lv = L.compute_levels(g_px, gp["text"].tolist(), a.code)
        w = gp["w"].sum()
        sent_social = float((gp["score_adj"] * gp["w"]).sum() / w) if w > 0 else 0.0
        sent_news = float(gn["score"].mean()) if len(gn) else np.nan
        composite = sent_social if np.isnan(sent_news) else 0.75 * sent_social + 0.25 * sent_news
        after = gp[gp["session"].isna()]
        recent = pan.tail(3)["sent_social"].mean()
        prior = pan.head(max(len(pan) - 3, 0))["sent_social"].mean()
        mention_z = float(pan["mention_z"].iloc[-1]) if len(pan) and pd.notna(pan["mention_z"].iloc[-1]) else 0.0
        plan = L.trade_plan(float(row["close"]), float(row["pred_ret"]), float(row["vol_10"]), composite, lv)
        flags = []
        if mention_z > 1.5:
            flags.append("buzz spike")
        if composite > 0.3 and mention_z > 1.0:
            flags.append("crowded long, euphoria risk")
        if composite > 0.1 and row["ret_5"] < -0.03:
            flags.append("bullish divergence: crowd buying the dip")
        if composite < -0.1 and row["ret_5"] > 0.03:
            flags.append("bearish divergence: price up, crowd sceptical")
        incomplete = [k.split("|")[1] for k, v in coverage.items() if k.startswith(a.ticker + "|") and not v["complete"]]
        if incomplete:
            flags.append(f"{'/'.join(incomplete)} capped by page limit (busy board)")
        vol = float(row["vol_10"]) if pd.notna(row["vol_10"]) else 0.02
        direction = "Up" if row["pred_ret"] > 0.1 * vol else "Down" if row["pred_ret"] < -0.1 * vol else "Flat"
        by_src = gp.groupby("source").agg(n=("score", "size")).n.to_dict()
        sent_src = {s: float((g["score_adj"] * g["w"]).sum() / g["w"].sum()) for s, g in gp.groupby("source")}
        summary.append({
            "ticker": a.ticker, "name": a.name, "exchange": a.exchange, "currency": (meta.get(a.ticker) or {}).get("currency"),
            "price": float(row["close"]), "price_date": row["date"].strftime("%Y-%m-%d"),
            "ret_1d": float(row["ret_1"]), "ret_5d": float(row["ret_5"]),
            "mentions": int(len(gp)), "mentions_by_source": by_src, "sent_by_source": sent_src,
            "engagement": float(w), "sentiment_social": sent_social, "sentiment_news": None if np.isnan(sent_news) else sent_news,
            "sentiment_score": composite, "sentiment_label": plan["sentiment_label"],
            "sentiment_momentum": None if (np.isnan(recent) or np.isnan(prior)) else float(recent - prior),
            "after_close_posts": int(len(after)),
            "after_close_sentiment": float(after["score_adj"].mean()) if len(after) else None,
            "news_count": int(len(gn)), "mention_z": mention_z,
            "pred_session": next_weekday(row["date"]).strftime("%Y-%m-%d"),
            "pred_close": float(row["pred_close"]), "pred_ret": float(row["pred_ret"]),
            "pred_ret_price_only": float(row["pred_ret_price_only"]),
            "pred_low_68": float(row["pred_low_68"]), "pred_high_68": float(row["pred_high_68"]),
            "direction": direction, **plan, "levels": lv, "flags": flags,
            "top_posts": _top_posts(gp), "news": [{"title": r.title, "url": r.url, "ts": r.published_utc.strftime("%Y-%m-%d"),
                                                   "score": round(float(r.score), 2), "provider": r.provider}
                                                  for r in gn.sort_values("published_utc", ascending=False).head(6).itertuples()],
        })

    # persist
    posts.to_csv(out / "posts.csv", index=False, encoding="utf-8-sig")
    news.to_csv(out / "news.csv", index=False, encoding="utf-8-sig")
    panel.to_csv(out / "panel.csv", index=False)
    bt.to_csv(out / "backtest_predictions.csv", index=False)
    metrics.to_csv(out / "backtest_metrics.csv", index=False)
    (out / "news_audit.json").write_text(json.dumps(news_audit, indent=2, ensure_ascii=False), encoding="utf-8")
    # live track record: score every earlier forecast whose target session has closed (prices already in hand)
    log_path = out / "predictions_log.csv"
    track = score_log(pd.read_csv(log_path), asset_px) if log_path.exists() else pd.DataFrame()
    if not track.empty:
        track.to_csv(out / "prediction_check.csv", index=False)
    log_rows = pd.DataFrame([{"run_at": cfg.now_utc.isoformat(), "ticker": s["ticker"], "exchange": s["exchange"],
                              "base_date": s["price_date"], "base_close": s["price"], "target_session": s["pred_session"],
                              "pred_close": s["pred_close"], "pred_ret": s["pred_ret"], "position": s["position"]} for s in summary])
    log_rows.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)

    result = {
        "generated_at": cfg.now_utc.isoformat(), "cutoff_utc": cfg.cutoff_utc.isoformat(),
        "lookback_days": cfg.run["lookback_days"], "scorer": scorer, "ingest": ingest_info,
        "news_provider": "exa" if os.environ.get("EXA_API_KEY") else ("google_news_rss_fallback" if with_news else "disabled"),
        "exchanges": [{"code": e.code, "name": e.name, "sources": e.sources, "telegram_channels": e.telegram_channels}
                      for e in cfg.exchanges],
        "counts": {"posts": int(len(posts)), "news": int(len(news)),
                   "posts_by_source": posts.groupby(["exchange", "source"]).size().rename("n").reset_index().to_dict("records")},
        "coverage": coverage, "summary": summary,
        "metrics": metrics.to_dict("records"),
        "stage2_coefs": nxt.attrs.get("stage2_coefs", {}), "stage1_coefs": nxt.attrs.get("stage1_coefs", {}),
        "track_record": track_summary(track),
    }

    # analyst agent: narrative layer (Claude tool-use agent, or deterministic template)
    from .agent import RunData, run_analyst
    data = RunData(result, posts, news, panel)
    result["analyst"] = run_analyst(data, analyst)
    result["briefs"] = data.briefs
    for s in summary:
        s["thesis"] = data.theses.get(s["ticker"])

    (out / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    from .report import build_report
    build_report(result, prices, posts, panel, bt, out / "report.html")
    log.info("report written to %s", out / "report.html")
    return result


def score_log(preds: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Score logged n+1 forecasts against the first close after their base date (latest run per ticker/base date)."""
    preds = preds.sort_values("run_at").drop_duplicates(["ticker", "base_date"], keep="last")
    rows = []
    for p in preds.itertuples(index=False):
        g = prices[(prices["ticker"] == p.ticker) & (prices["date"] > pd.Timestamp(p.base_date))].sort_values("date")
        row = p._asdict()
        if g.empty:
            rows.append({**row, "status": "pending"})
            continue
        act = g.iloc[0]
        actual_ret = float(np.log(act["close"] / p.base_close))
        rows.append({**row, "status": "scored", "actual_date": act["date"].strftime("%Y-%m-%d"),
                     "actual_close": float(act["close"]), "actual_ret": actual_ret,
                     "abs_err_pct": 100 * abs(p.pred_close - act["close"]) / act["close"],
                     "naive_abs_err_pct": 100 * abs(p.base_close - act["close"]) / act["close"],
                     "direction_hit": bool(np.sign(p.pred_ret) == np.sign(actual_ret)),
                     "position_pnl_pct": 100 * actual_ret * {"LONG": 1, "SHORT": -1}.get(p.position, 0)})
    return pd.DataFrame(rows)


def track_summary(track: pd.DataFrame) -> dict:
    if track.empty or "status" not in track:
        return {"scored": 0, "pending": 0, "rows": []}
    sc = track[track["status"] == "scored"]
    out = {"scored": int(len(sc)), "pending": int((track["status"] == "pending").sum()), "rows": []}
    if len(sc):
        traded = sc[sc["position"] != "FLAT"]
        out.update(sessions=int(sc["actual_date"].nunique()), direction_hit_pct=float(100 * sc["direction_hit"].mean()),
                   mae_pct=float(sc["abs_err_pct"].mean()), naive_mae_pct=float(sc["naive_abs_err_pct"].mean()),
                   trades=int(len(traded)),
                   avg_trade_pnl_pct=float(traded["position_pnl_pct"].mean()) if len(traded) else None,
                   rows=sc.sort_values(["actual_date", "ticker"], ascending=[False, True]).head(60)
                   .astype(object).where(sc.notna(), None).to_dict("records"))
    return out


def check_predictions(cfg: Config) -> pd.DataFrame:
    """'Check your results': fetch fresh closes and score every logged n+1 forecast."""
    log_path = cfg.output_dir / "predictions_log.csv"
    if not log_path.exists():
        raise SystemExit("no predictions_log.csv yet - run the pipeline first")
    preds = pd.read_csv(log_path)
    plan = [r for r in plan_mod.yahoo_requests(cfg) if r["userData"]["ticker"] in set(preds["ticker"])]
    prices, _ = parsers.records_to_prices(local_crawler.crawl(plan), cfg.exchanges)
    prices = prices[prices["close_utc"] <= pd.Timestamp(datetime.now(timezone.utc))]
    res = score_log(preds, prices)
    res.to_csv(cfg.output_dir / "prediction_check.csv", index=False)
    return res


def load_last_run(cfg: Config):
    """Rebuild the agent's view of the last run from outputs/ (for `hsagent ask`)."""
    from .agent import RunData
    out = cfg.output_dir
    result = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    posts = pd.read_csv(out / "posts.csv", parse_dates=["ts_utc"])
    news = pd.read_csv(out / "news.csv", parse_dates=["published_utc"])
    panel = pd.read_csv(out / "panel.csv", parse_dates=["date"])
    return RunData(result, posts, news, panel)
