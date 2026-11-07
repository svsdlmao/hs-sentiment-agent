"""CLI.

  python -m hsagent run                      # full run through Apify + Exa (needs .env keys)
  python -m hsagent run --mode local         # same crawl plan, fetched from this machine (no Apify)
  python -m hsagent run --mode cached        # re-run analysis on the last saved raw crawl
  python -m hsagent run --dataset <id>       # re-use a finished Apify run's dataset
  python -m hsagent check                    # score past n+1 predictions against realised closes
  python -m hsagent plan                     # print the crawl plan / Apify actor input
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import load_config


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="hsagent", description="HK/KR social-sentiment n+1 price agent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--mode", choices=["apify", "local", "cached"], default="apify")
    r.add_argument("--dataset", help="existing Apify dataset id to analyse (implies --mode apify)")
    r.add_argument("--records", action="append", help="saved records_*.json.gz to analyse (with --mode cached); repeat to merge crawls")
    r.add_argument("--scorer", choices=["lexicon", "claude"], default="lexicon")
    r.add_argument("--no-news", action="store_true")
    r.add_argument("--analyst", choices=["auto", "claude", "template", "off"], default="auto",
                   help="narrative layer: Claude tool-use agent (auto = when ANTHROPIC_API_KEY is set) or template")
    sub.add_parser("check")
    q = sub.add_parser("ask", help="ask the analyst agent a question about the last run")
    q.add_argument("question")
    p = sub.add_parser("plan")
    p.add_argument("--actor-input", action="store_true", help="print the full Apify actor input JSON")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    cfg = load_config()

    if args.cmd == "run":
        from .pipeline import run
        res = run(cfg, mode=args.mode, scorer=args.scorer, with_news=not args.no_news,
                  dataset_id=args.dataset, records_path=args.records, analyst=args.analyst)
        print(f"\nposts={res['counts']['posts']} news={res['counts']['news']}  ->  {cfg.output_dir / 'report.html'}")
        for ex in cfg.exchanges:
            rows = sorted([s for s in res["summary"] if s["exchange"] == ex.code],
                          key=lambda s: (-s["mentions"], -s["sentiment_score"]))
            print(f"\n{ex.name}")
            print(f"  {'ticker':<10}{'price':>12}{'mentions':>10}{'sent':>7}  {'label':<8}{'n+1':>12}{'dir':>6}  position")
            for s in rows:
                print(f"  {s['ticker']:<10}{s['price']:>12,.2f}{s['mentions']:>10}{s['sentiment_score']:>7.2f}  "
                      f"{s['sentiment_label']:<8}{s['pred_close']:>12,.2f}{s['direction']:>6}  {s['position']}")
        m = [x for x in res["metrics"] if x["scope"] == "ALL"]
        if m:
            print("\nwalk-forward out-of-sample (all tickers):")
            for x in m:
                extra = f"  dir_acc={x['dir_acc_pct']:.1f}% (n={int(x['dir_n'])}, p={x['dir_p_value']:.2f})" if x["model"] != "naive" else ""
                print(f"  {x['model']:<11} MAE={x['MAE_pct']:.2f}%{extra}")
    elif args.cmd == "check":
        from .pipeline import check_predictions
        res = check_predictions(cfg)
        print(res.to_string(index=False))
    elif args.cmd == "ask":
        from .agent import ask
        from .pipeline import load_last_run
        print(ask(load_last_run(cfg), args.question))
    elif args.cmd == "plan":
        from .ingest.apify_runner import build_actor_input
        from .ingest.plan import build_plan
        plan = build_plan(cfg)
        if args.actor_input:
            print(json.dumps(build_actor_input(cfg, plan), ensure_ascii=False, indent=2))
        else:
            from collections import Counter
            print(Counter(r["userData"]["kind"] for r in plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
