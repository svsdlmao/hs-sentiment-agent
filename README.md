# HK & KR social-sentiment n+1 price agent

[![CI](https://github.com/svsdlmao/hs-sentiment-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/svsdlmao/hs-sentiment-agent/actions/workflows/ci.yml)
[![daily agent run](https://github.com/svsdlmao/hs-sentiment-agent/actions/workflows/daily.yml/badge.svg)](https://github.com/svsdlmao/hs-sentiment-agent/actions/workflows/daily.yml)

An agent that, every trading day, scrapes retail-investor chatter and news for **5 Hang Seng (HKEX) and 5 KOSPI (KRX)
stocks**, scores sentiment in Cantonese, Chinese, Korean and English, aligns it with price bars in UTC, finds support and
resistance, proposes a position, predicts the **next-session (n+1) close**, scores its own past forecasts, has a
**Claude analyst agent** write a thesis per ticker, and publishes a **visual HTML report**.

| | HKEX (Hang Seng) | KRX (KOSPI) |
|---|---|---|
| Assets | Tencent 0700, Alibaba 9988, Xiaomi 1810, Meituan 3690, BYD 1211 | Samsung Elec. 005930, SK hynix 000660, NAVER 035420, Kakao 035720, Hyundai Motor 005380 |
| Social source 1 | **LIHKG** (thread search per keyword, plus replies of the most active threads) | **Naver Finance discussion board** (종목토론실, every post in the 10-day window) |
| Social source 2 | **Telegram** public channels: hkej_news, mingpao_news, reuterszh, jin10data | **Telegram** public channels: hedgecat0301, YeouidoStory2, bornlupin, corevalue, sejongdata2013, hanaresearch, kiwoom_semibat, companyreport, ehdwl |
| Search terms | ticker code (`00700`), company names in 繁/简 | ticker code (`005930`), company names in Korean/English, retail nicknames (삼전) |
| News | **Exa** search, one English and one Chinese query per asset | **Exa** search, one English and one Korean query per asset |
| Prices | Yahoo chart API, daily bars (1 year), fetched through the same Apify run | same |

Everything (tickers, keywords, channels, limits) lives in [`config.toml`](config.toml). Adding an asset is a config change.
Cashtags: I tested `$0700.HK` and `$005930` on Telegram search and got no results, because t.me search drops the `$`.
The bare ticker code is therefore what catches cashtag-style mentions, and Naver's board is already per-ticker.

---

## Results from the Apify run (2026-11-07)

Live report: **https://svsdlmao.github.io/hs-sentiment-agent/**. It is refreshed every weekday by the
[daily workflow](.github/workflows/daily.yml).

| | |
|---|---|
| Apify run | `apify/cheerio-scraper` · run `vhuNiSIHsKRnbTT6I` · dataset `X9w6UTjTKOpgAodpX` · 221 start requests → 1,243 pages in 22 min |
| Posts collected (10 days) | **98,877**: Naver 98,292 · Telegram 495 · LIHKG 90 |
| News articles (Exa) | 236 (about 24 per asset, English plus local-language queries) |
| Out-of-sample forecasts | 40 (10 tickers × 4 sessions, walk-forward) |

| Model (walk-forward, out of sample) | MAE | Direction hit-rate |
|---|---|---|
| Random walk (tomorrow = today) | **3.69%** | – |
| Price-only ridge | 3.74% | 35% (p = 0.08) |
| Price + sentiment (this agent) | 3.96% | 52.5% (p = 0.87) |
| · HKEX only | 2.93% vs naive 2.83% | **70%** (price-only 30%) |
| · KRX only | 4.98% vs naive 4.54% | 35% (price-only 40%) |

**How to read this.** On price error nothing beats the random walk over 4 sessions, which is the normal result for
daily stock returns. The sentiment layer does help on Hong Kong: it called 14 of 20 directions there, against 6 of 20 for
price-only. It does not help on Korea, where the huge Naver boards are noisy and did not lead the moves in these
sessions. Four sessions in which stocks on one exchange move together is far too small a sample, so none of this is
statistically significant. The report shows these numbers anyway, because they are **what the agent can and cannot
claim yet**. The live track record (below) is how that sample grows.

Snapshot of the per-ticker view (the full version is in the live report):

| Ticker | Price | Mentions | Sentiment | n+1 | Position | Support / Resistance |
|---|---|---|---|---|---|---|
| 0700.HK Tencent | 419.8 | 65 | +0.30 Bullish | +1.2% | LONG | 411.1 / 438.8 |
| 3690.HK Meituan | 80.4 | 109 | +0.22 Bullish | +1.0% | LONG (low R:R, size down) | 74.0 / 84.8 |
| 005930.KS Samsung | 188,700 | 45,849 | −0.15 Bearish | +1.4% | FLAT (model and crowd disagree) | 167,150 / 208,467 |
| 000660.KS SK hynix | 1,132,000 | 42,171 | −0.01 Neutral | +1.8% | LONG | 1,007,571 / 1,351,000 |

All ten forecasts are "Up". The price-only stage expects a rebound after the 6 Nov selloff, which is mean reversion
learned from a year of history. Sentiment can only shift a forecast by up to 0.5 × daily volatility, so it mostly decides
the *position*: a call needs the forecast to clear the noise threshold **and** the crowd must not be against it
(Samsung's bearish board vetoes its LONG).

---

## Quick start

```bash
python -m pip install -r requirements.txt
cp .env.example .env        # fill in APIFY_TOKEN and EXA_API_KEY (ANTHROPIC_API_KEY optional)
python -m hsagent run       # Apify + Exa + analyst agent, ~10-15 min, writes outputs/report.html
```

| Command | What it does |
|---|---|
| `python -m hsagent run --mode local` | same crawl plan and pagination, fetched from your machine (no Apify credits) |
| `python -m hsagent run --mode cached` | re-run the analysis on the last saved raw crawl (offline, ~1 min) |
| `python -m hsagent run --dataset <id>` | analyse an existing Apify dataset, so any run can be reproduced exactly |
| `python -m hsagent run --scorer claude` | Claude sentiment scorer instead of the lexicon |
| `python -m hsagent run --analyst claude\|template\|off` | choose the narrative layer (default `auto` = Claude when a key is set) |
| `python -m hsagent ask "why is Samsung FLAT?"` | ask the analyst agent about the last run; it answers from tool calls |
| `python -m hsagent check` | score every logged n+1 forecast against the closes that printed since |
| `python -m hsagent plan --actor-input` | print the exact Apify actor input JSON |
| `python -m pytest -q` | 18 tests (the JS parity tests run after `npm ci --prefix tests/js`) |

## Apify and Exa usage (for reproduction)

* **Actor:** [`apify/cheerio-scraper`](https://apify.com/apify/cheerio-scraper), Apify's official actor. Its input is
  built by `hsagent/ingest/apify_runner.py`:
  * `startUrls` is the crawl plan of **221 start requests**: 5 Naver boards, 9 LIHKG searches, 182 Telegram
    channel × search-term searches, 13 Telegram channel pages and 12 Yahoo price series;
  * `pageFunction` is [`apify/page_function.js`](apify/page_function.js);
  * `additionalMimeTypes: ["application/json"]`, plus Apify Proxy (rotating IPs, which LIHKG needs).

  The page function returns each page's payload and keeps paginating until posts are older than the 10-day cutoff
  (Naver `offset` cursor, LIHKG `page`, Telegram `before=<message id>`). It also queues the reply pages of the
  most-discussed LIHKG threads. Each run's `run_id`, `dataset_id` and console URL are appended to
  `outputs/apify_runs.json`.
* **One crawl plan, two runtimes.** `local_crawler.py` runs the same plan with a Python mirror of the page function's
  pagination logic. A test runs the real `page_function.js` under Node + cheerio against saved pages and asserts that both
  produce the same follow-up requests. Apify and local mode therefore cannot drift apart silently.
* **Exa:** `POST https://api.exa.ai/search` with `category: "news"`, `startPublishedDate` set to the window start, and
  text plus highlights. Request IDs go to `outputs/news_audit.json`. Without `EXA_API_KEY` the pipeline falls back to
  Google News RSS, and the report says so.

## How it works

```
config.toml ─► plan.py ─► Apify cheerio-scraper + page_function.js ─┐   (or local_crawler.py: same plan, same pagination)
                                                                   ▼
                 parsers.py  ── posts (UTC ts, text, likes, dislikes, replies, views, author, followers)
                             └─ daily bars (OHLCV, session close in UTC)
news.py (Exa) ──────────────────────────────┐
sentiment.py  (lexicon | Claude) ───────────┤
features.py   UTC session alignment, engagement weighting, daily aggregates, technicals
model.py      2-stage ridge, walk-forward OOS backtest vs naive & price-only, live n+1 forecast
levels.py     swing S/R, pivots, volume POC, crowd-cited levels, position + stop/target
pipeline.py   live track record: score yesterday's forecasts, log today's
agent.py      Claude analyst agent (tool use) ─► thesis per ticker, brief per exchange
report.py     outputs/report.html  (plotly, self-contained, works offline)
```

**UTC alignment, no look-ahead.** Every post and article is stamped in UTC and assigned to the first trading session
whose close (16:00 HKT / 15:30 KST, converted to UTC) is at or after it. The features for session *t* contain only
information published before close(*t*), and they predict close(*t+1*). Posts after the latest close appear in the report
as after-close chatter, which is the freshest signal for tomorrow.

**Sentiment.** The default scorer is a multilingual finance lexicon covering Cantonese (爆升, 插水, 撈底), Traditional and
Simplified Chinese, Korean retail slang (떡상, 떡락, 설거지, 가즈아, 줍줍), English and emoji, with negation handling
(唔/冇/不/안/못/not). It is compiled to one regex and scores 20k posts in about 1 s. Each post's score is weighted by
engagement: log of likes, replies and views, the author's follower count (Telegram subscribers), and Naver's
verified-holder badge. It is then adjusted for crowd agreement, so a bullish post that gets down-voted counts for less.
`--scorer claude` uses Claude with structured JSON output in batches; a batch that fails falls back to the lexicon.

**Model.** Social data covers only about 7 sessions, so a single model would overfit. The model has two stages:
1. a price-only ridge (returns, volatility, RSI, distance from SMA20, range position, volume z-score, index return),
   trained on a year of pooled history;
2. a heavily shrunk ridge on sentiment features (social and news sentiment, change in sentiment, mention z-score,
   bull-minus-bear share), fitted to the stage-1 residuals. Its features are winsorised at ±3 sd, and it can move the
   forecast by at most 0.5 × daily volatility.

**Support and resistance.** Four methods, shown together on each chart:
* fractal swing highs and lows clustered by ATR;
* floor-trader pivots;
* the volume point-of-control;
* **crowd-cited levels**: prices people actually write in posts ("8만 깨지면", "500蚊"). They are parsed with unit
  handling (만/萬) and kept only when at least 2 posts cite them within ±25% of the price. These are the levels retail
  traders are watching.

**Position.** LONG or SHORT only when the forecast clears 0.25 × volatility **and** crowd sentiment is not against it. Otherwise the
position is FLAT, with the reason shown. Each position comes with a stop at support or resistance, a target and the R:R.

**Flags.** Buzz spike (mention z-score), crowded long, bullish divergence (crowd buying the dip) and bearish divergence.

## Analyst agent (Claude tool use)

`hsagent/agent.py` uses the Anthropic SDK tool runner (`claude-opus-5-5`). The agent investigates the run instead of
receiving one giant prompt:

| Read tools | Write tools |
|---|---|
| `list_tickers`, `get_ticker`, `get_posts(stance=bull/bear)`, `get_daily_sentiment`, `get_news`, `get_backtest` | `write_ticker_thesis(ticker, thesis, key_risk, confidence)`, `write_exchange_brief(exchange, brief)` |

* **Grounded:** the system prompt requires every number in a thesis to come from a tool result. The backtest tool is
  part of the toolset, so the brief has to say whether the model beats a random walk.
* **Prompt-injection safe:** scraped posts reach the model only inside tool results and are labelled untrusted data
  (one test fixture post literally says "IGNORE PREVIOUS INSTRUCTIONS"). The model can only write through the two typed
  write tools, and their schemas are strict.
* **Always produces output:** without an API key, a deterministic template analyst fills the same fields from the same
  data. The report labels which one wrote each thesis. `python -m hsagent ask "..."` opens the same toolset for
  free-form questions.

## Live track record and automation

Each run first scores every earlier forecast whose target session has since closed. It then appends its own forecasts to
`outputs/predictions_log.csv`. The report's **Live track record** section shows the hit-rate, MAE against naive, and the
P&L of the LONG/SHORT calls.

[`.github/workflows/daily.yml`](.github/workflows/daily.yml) runs the whole agent every weekday at 09:45 UTC, after both
markets close. It commits the track record and publishes the report to GitHub Pages, so the out-of-sample record grows
with no one touching it. One-time setup:
1. In repo Settings → Secrets → Actions, add `APIFY_TOKEN` and `EXA_API_KEY` (and optionally `ANTHROPIC_API_KEY`).
2. In Settings → Pages, set Source to **GitHub Actions**.

[`ci.yml`](.github/workflows/ci.yml) runs the Python tests and the Node page-function parity test on every push.

## Outputs (`outputs/`)

| file | content |
|---|---|
| `report.html` | the visual report: rankings per exchange sorted by mentions/sentiment, market briefs, bubble chart, sentiment heatmap, per-ticker charts with S/R and n+1 band, track record, backtest |
| `summary.json` | everything in the report as data |
| `posts.csv`, `news.csv` | scored, session-aligned posts and articles |
| `panel.csv` | ticker × session feature panel |
| `backtest_predictions.csv`, `backtest_metrics.csv` | out-of-sample predictions and scores |
| `predictions_log.csv`, `prediction_check.csv` | live n+1 forecasts and their realised scoring |
| `apify_runs.json`, `news_audit.json` | run IDs, dataset IDs and Exa request IDs for reproduction |
| `raw/records_*.json.gz` | raw crawl (git-ignored) |

## Design decisions

* **The official `cheerio-scraper` with my own page function, not marketplace actors per site.** No ready-made actor
  covers LIHKG, Naver's discussion board and Telegram search together with the engagement fields. One generic actor
  with a page function also gives a single run, a single dataset and a single bill.
* **JSON endpoints over HTML.** Naver's board is a JS app. Its bundles revealed the `front-api/discussion/list`
  endpoint, which returns likes, dislikes, views and the verified-holder badge, none of which is visible in the HTML.
* **A lexicon by default and Claude as an option.** A lexicon is free, deterministic and reproducible on 100k posts, and
  it knows slang that general models misread (e.g. "가즈아"). Claude is used where language understanding pays off: the
  optional scorer, and the analyst that explains the numbers.
* **The benchmark is the random walk.** Any daily price model has to beat "tomorrow = today". The report states this
  benchmark up front rather than reporting a hit-rate on its own.

## Limitations

* **Ten days is a very short sample.** Stocks on one exchange move together, so the effective sample is closer to the
  number of sessions (4) than to the number of predictions (40). Treat the backtest as a sanity check, not as an edge.
* **Coverage is uneven.** Naver is huge (98k posts), while HK retail chatter on LIHKG is small (90 posts in the
  Apify run). LIHKG blocks single IPs for about 24h after bursts, so the local crawler throttles to one
  request at a time and stops (circuit breaker); the Apify proxy avoids the ban by rotating IPs.
* **Follower counts** exist only on Telegram (channel subscribers). LIHKG and Naver don't expose them, so engagement is
  used instead.
* **Short selling** is restricted for many HK names and for retail investors in Korea, so read SHORT as
  "underweight/avoid".
* Research output only, not investment advice.
