"""Self-contained visual HTML report (plotly.js inlined, opens offline)."""
from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

UP, DOWN, NEU = "#1a9e5f", "#d0453b", "#8a8f98"
SRC_COLORS = {"naver": "#03c75a", "telegram": "#2aa1da", "lihkg": "#f2a33a"}
FONT = "Inter, Segoe UI, Noto Sans CJK, Malgun Gothic, Microsoft JhengHei, sans-serif"


def _fig_html(fig: go.Figure, first: bool = False) -> str:
    fig.update_layout(font=dict(family=FONT, size=12, color="#2b2f36"), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=50, r=20, t=40, b=40))
    fig.update_xaxes(gridcolor="#e6e8eb", zerolinecolor="#cfd3d8")
    fig.update_yaxes(gridcolor="#e6e8eb", zerolinecolor="#cfd3d8")
    return fig.to_html(full_html=False, include_plotlyjs="inline" if first else False,
                       config={"displaylogo": False, "responsive": True})


def _fmt_px(v, cur=None) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{v:,.0f}" if (cur == "KRW" or abs(v) >= 10000) else f"{v:,.2f}"


def _pct(v) -> str:
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:+.2f}%"


def _sent_color(s: float) -> str:
    return UP if s > 0.08 else DOWN if s < -0.08 else NEU


def _e(s) -> str:
    return html.escape(str(s))


# --- charts --------------------------------------------------------------------

def bubble(rows: list[dict]) -> go.Figure:
    fig = go.Figure()
    for r in rows:
        fig.add_trace(go.Scatter(
            x=[r["sentiment_score"]], y=[max(r["mentions"], 1)], mode="markers+text", text=[r["ticker"]],
            textposition="top center", name=r["ticker"], showlegend=False,
            marker=dict(size=14 + 40 * np.sqrt(r["engagement"] / max(max(x["engagement"] for x in rows), 1)),
                        color=UP if r["direction"] == "Up" else DOWN if r["direction"] == "Down" else NEU,
                        opacity=0.75, line=dict(width=1, color="white")),
            hovertemplate=f"<b>{_e(r['name'])}</b><br>sentiment %{{x:.2f}}<br>mentions %{{y}}<br>"
                          f"n+1 {_pct(r['pred_ret'])} ({r['direction']})<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="#9aa0a6", dash="dot"))
    fig.update_layout(title="Mentions vs sentiment (bubble = engagement, colour = predicted n+1 direction)",
                      height=380, xaxis_title="sentiment score (−1 … +1)", yaxis_title="mentions (10 days)", yaxis_type="log")
    return fig


def heatmap(panel: pd.DataFrame, tickers: list[str]) -> go.Figure:
    p = panel[panel["ticker"].isin(tickers) & panel["in_social_window"]]
    piv = p.pivot_table(index="ticker", columns="date", values="sent_social").reindex(tickers)
    cnt = p.pivot_table(index="ticker", columns="date", values="mentions").reindex(tickers)
    text = cnt.fillna(0).astype(int).astype(str).values
    fig = go.Figure(go.Heatmap(z=piv.values, x=[d.strftime("%m-%d") for d in piv.columns], y=piv.index, zmid=0,
                               zmin=-0.6, zmax=0.6, colorscale=[[0, DOWN], [0.5, "#f4f5f7"], [1, UP]],
                               text=text, texttemplate="%{text}", colorbar=dict(title="sent")))
    fig.update_layout(title="Daily sentiment by session (cell text = mentions)", height=90 + 45 * len(tickers))
    fig.update_xaxes(type="category")
    return fig


def mentions_by_source(posts: pd.DataFrame, tickers: list[str]) -> go.Figure:
    p = posts[posts["ticker"].isin(tickers)].copy()
    p["day"] = p["ts_utc"].dt.floor("D")
    fig = go.Figure()
    for src, g in p.groupby("source"):
        s = g.groupby("day").size()
        fig.add_trace(go.Bar(x=s.index, y=s.values, name=src, marker_color=SRC_COLORS.get(src, NEU)))
    fig.update_layout(barmode="stack", title="Posts per UTC day by source", height=300, legend=dict(orientation="h", y=1.12))
    return fig


def ticker_chart(s: dict, px: pd.DataFrame, panel: pd.DataFrame) -> go.Figure:
    g = px[px["ticker"] == s["ticker"]].sort_values("date").tail(45)
    pn = panel[(panel["ticker"] == s["ticker"]) & panel["in_social_window"]]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04,
                        specs=[[{}], [{"secondary_y": True}]])
    fig.add_trace(go.Candlestick(x=g["date"], open=g["open"], high=g["high"], low=g["low"], close=g["close"],
                                 increasing_line_color=UP, decreasing_line_color=DOWN, name="price", showlegend=False), 1, 1)
    lv = s["levels"]
    x0, x1 = g["date"].iloc[0], pd.Timestamp(s["pred_session"])
    def hline(y, color, label, dash="solid", width=1.4):
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y, y], mode="lines", line=dict(color=color, dash=dash, width=width),
                                 name=label, hovertemplate=f"{label}: %{{y:,.2f}}<extra></extra>"), 1, 1)
    hline(lv["support"], UP, f"support ({lv['support_touches']} touches)", width=2)
    hline(lv["resistance"], DOWN, f"resistance ({lv['resistance_touches']} touches)", width=2)
    hline(lv["pivots"]["P"], "#7b61ff", "pivot", "dot")
    hline(lv["poc"], "#b08900", "volume POC", "dash")
    for c in lv["crowd_levels"]:
        hline(c["price"], "#e05aa8", f"crowd-cited {c['price']:,.0f} ({c['mentions']}×)", "dashdot", 1)
    # n+1 prediction with 68% band
    fig.add_trace(go.Scatter(x=[x1, x1], y=[s["pred_low_68"], s["pred_high_68"]], mode="lines",
                             line=dict(color="#1f6feb", width=6), opacity=0.25, name="68% band"), 1, 1)
    fig.add_trace(go.Scatter(x=[g["date"].iloc[-1], x1], y=[s["price"], s["pred_close"]], mode="lines+markers",
                             line=dict(color="#1f6feb", dash="dot"), marker=dict(size=[4, 11], symbol="diamond"),
                             name=f"n+1 forecast {_fmt_px(s['pred_close'], s['currency'])}"), 1, 1)
    if len(pn):
        fig.add_trace(go.Bar(x=pn["date"], y=pn["sent_social"], marker_color=[_sent_color(v) for v in pn["sent_social"].fillna(0)],
                             name="daily sentiment", showlegend=False), 2, 1)
        fig.add_trace(go.Scatter(x=pn["date"], y=pn["mentions"], mode="lines+markers", line=dict(color="#555", width=1),
                                 name="mentions", showlegend=False), 2, 1, secondary_y=True)
    fig.update_layout(height=470, xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=-0.12, font=dict(size=10)),
                      margin=dict(l=50, r=40, t=10, b=10))
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(title_text="sent", row=2, col=1)
    fig.update_yaxes(title_text="posts", row=2, col=1, secondary_y=True, showgrid=False)
    return fig


def backtest_chart(bt: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    act = np.log(bt["next_close"] / bt["close"]) * 100
    for m, c in [("price_only", "#9aa0a6"), ("sentiment", "#1f6feb")]:
        fig.add_trace(go.Scatter(x=bt[f"pred_{m}"] * 100, y=act, mode="markers", name=m.replace("_", "-"),
                                 marker=dict(color=c, size=8, opacity=0.8), text=bt["ticker"] + " " + bt["date"].dt.strftime("%m-%d"),
                                 hovertemplate="%{text}<br>pred %{x:.2f}%<br>actual %{y:.2f}%<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="#bbb"))
    fig.add_vline(x=0, line=dict(color="#bbb"))
    fig.update_layout(title="Out-of-sample: predicted vs actual next-session return", height=380,
                      xaxis_title="predicted %", yaxis_title="actual %", legend=dict(orientation="h", y=1.1))
    return fig


# --- page ----------------------------------------------------------------------

CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#1d2127;--muted:#69707a;--line:#e3e6ea;--up:#1a9e5f;--down:#d0453b;--neu:#8a8f98;--acc:#1f6feb}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 Inter,"Segoe UI","Noto Sans CJK TC","Malgun Gothic","Microsoft JhengHei",sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding:24px 16px 80px}h1{font-size:26px;margin:0 0 4px}h2{font-size:20px;margin:36px 0 12px;padding-top:8px;border-top:2px solid var(--ink)}
h3{font-size:16px;margin:0}.muted{color:var(--muted)}.small{font-size:12px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:18px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}.kpi b{display:block;font-size:22px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:14px 0}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}@media(max-width:900px){.grid2{grid-template-columns:1fr}}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:7px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{cursor:pointer;user-select:none;color:var(--muted);font-weight:600;position:sticky;top:0;background:var(--card)}th:hover{color:var(--ink)}
td.l,th.l{text-align:left}.tbl{overflow-x:auto}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;font-weight:600;color:#fff}
.LONG,.Up,.Bullish{background:var(--up)}.SHORT,.Down,.Bearish{background:var(--down)}.FLAT,.Flat,.Neutral{background:var(--neu)}
.bar{display:inline-block;height:8px;border-radius:4px;vertical-align:middle}
.tcard-h{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:baseline;margin-bottom:6px}.tcard-h .big{font-size:22px;font-weight:700}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin:8px 0}.facts div{background:var(--bg);border-radius:8px;padding:6px 10px}
.facts span{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.03em}
.posts{display:grid;grid-template-columns:1fr 1fr;gap:12px;font-size:12.5px}@media(max-width:900px){.posts{grid-template-columns:1fr}}
.posts li{margin:0 0 6px}.flag{display:inline-block;background:#fff4d6;color:#7a5600;border-radius:6px;padding:1px 7px;font-size:12px;margin:2px 4px 0 0}
a{color:var(--acc);text-decoration:none}a:hover{text-decoration:underline}
.method li{margin-bottom:4px}
.brief{border-left:4px solid var(--acc);background:#f3f7ff}.brief p{margin:6px 0 0}
.thesis{background:#f3f7ff;border-radius:8px;padding:10px 12px;margin:8px 0;font-size:13.5px}
.by{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.conf-high{background:var(--up)}.conf-medium{background:#c78a00}.conf-low{background:var(--neu)}
.hit{color:var(--up);font-weight:600}.miss{color:var(--down);font-weight:600}
"""

SORT_JS = """
document.querySelectorAll('table.sortable').forEach(t=>{t.querySelectorAll('th').forEach((th,i)=>{th.addEventListener('click',()=>{
const b=t.tBodies[0],rows=[...b.rows],asc=th.dataset.asc!=='1';th.dataset.asc=asc?'1':'0';
rows.sort((a,c)=>{const x=a.cells[i].dataset.v??a.cells[i].innerText,y=c.cells[i].dataset.v??c.cells[i].innerText;
const nx=parseFloat(x),ny=parseFloat(y);const r=(!isNaN(nx)&&!isNaN(ny))?nx-ny:String(x).localeCompare(String(y));return asc?r:-r});
rows.forEach(r=>b.appendChild(r))})})});
"""


def _rank_table(rows: list[dict]) -> str:
    head = ("<tr><th>#</th><th class=l>Ticker</th><th class=l>Name</th><th>Price</th><th>1d</th><th>Mentions</th>"
            "<th class=l>By source</th><th>Sentiment</th><th class=l>Label</th><th>Momentum</th><th>n+1 forecast</th>"
            "<th>Δ</th><th class=l>Dir</th><th class=l>Position</th><th>Support</th><th>Resistance</th></tr>")
    body = []
    for i, s in enumerate(rows, 1):
        bysrc = " · ".join(f"<span style='color:{SRC_COLORS.get(k, NEU)}'>{k}</span> {v}" for k, v in s["mentions_by_source"].items())
        w = int(min(abs(s["sentiment_score"]), 1) * 60)
        sbar = f"<span class=bar style='width:{w}px;background:{_sent_color(s['sentiment_score'])}'></span> {s['sentiment_score']:+.2f}"
        mom = s["sentiment_momentum"]
        body.append(
            f"<tr><td>{i}</td><td class=l><a href='#t-{_e(s['ticker'])}'><b>{_e(s['ticker'])}</b></a></td><td class=l>{_e(s['name'])}</td>"
            f"<td data-v='{s['price']}'>{_fmt_px(s['price'], s['currency'])}</td><td data-v='{s['ret_1d']}'>{_pct(s['ret_1d'])}</td>"
            f"<td data-v='{s['mentions']}'>{s['mentions']:,}</td><td class=l>{bysrc}</td>"
            f"<td data-v='{s['sentiment_score']}'>{sbar}</td><td class=l><span class='pill {s['sentiment_label']}'>{s['sentiment_label']}</span></td>"
            f"<td data-v='{mom if mom is not None else 0}'>{'–' if mom is None else f'{mom:+.2f}'}</td>"
            f"<td data-v='{s['pred_close']}'>{_fmt_px(s['pred_close'], s['currency'])}</td><td data-v='{s['pred_ret']}'>{_pct(s['pred_ret'])}</td>"
            f"<td class=l><span class='pill {s['direction']}'>{'▲' if s['direction']=='Up' else '▼' if s['direction']=='Down' else '■'} {s['direction']}</span></td>"
            f"<td class=l><span class='pill {s['position']}'>{s['position']}</span></td>"
            f"<td>{_fmt_px(s['levels']['support'], s['currency'])}</td><td>{_fmt_px(s['levels']['resistance'], s['currency'])}</td></tr>")
    return f"<div class=tbl><table class=sortable><thead>{head}</thead><tbody>{''.join(body)}</tbody></table></div>"


def _author(a: str) -> str:
    return "Claude analyst agent" if a == "claude" else "rule-based analyst (no API key)"


def _thesis(s: dict) -> str:
    t = s.get("thesis")
    if not t:
        return ""
    return (f"<div class=thesis><span class=by>Thesis · {_author(t.get('author'))}</span> "
            f"<span class='pill conf-{_e(t['confidence'])}'>{_e(t['confidence'])} confidence</span>"
            f"<p style='margin:6px 0 4px'>{_e(t['thesis'])}</p><span class=muted><b>Key risk:</b> {_e(t['key_risk'])}</span></div>")


def _track_record(tr: dict) -> str:
    if not tr or not tr.get("scored"):
        pending = tr.get("pending", 0) if tr else 0
        return (f"<p class=muted>No forecast has reached its target session yet ({pending} pending). Each run logs its n+1 "
                f"forecasts; the next run (or <code>python -m hsagent check</code>) scores them against the real close. "
                f"The scheduled GitHub Action does this every trading day, so this table grows on its own.</p>")
    pnl = tr.get("avg_trade_pnl_pct")
    pnl_txt = "–" if pnl is None else f"{pnl:+.2f}%"
    head = (f"<div class=kpis><div class=kpi><span class=muted>Scored forecasts</span><b>{tr['scored']}</b>"
            f"<span class=small>{tr.get('sessions', '?')} sessions · {tr['pending']} pending</span></div>"
            f"<div class=kpi><span class=muted>Direction hit-rate</span><b>{tr['direction_hit_pct']:.0f}%</b></div>"
            f"<div class=kpi><span class=muted>Price MAE vs naive</span><b>{tr['mae_pct']:.2f}%</b>"
            f"<span class=small>naive {tr['naive_mae_pct']:.2f}%</span></div>"
            f"<div class=kpi><span class=muted>Avg position P&amp;L</span><b>"
            f"{pnl_txt}</b><span class=small>{tr.get('trades', 0)} LONG/SHORT calls, close-to-close</span></div></div>")
    rows = "".join(
        f"<tr><td class=l>{_e(r['actual_date'])}</td><td class=l>{_e(r['ticker'])}</td><td>{_fmt_px(r['base_close'])}</td>"
        f"<td>{_fmt_px(r['pred_close'])}</td><td>{_fmt_px(r['actual_close'])}</td><td>{r['abs_err_pct']:.2f}%</td>"
        f"<td class=l><span class={'hit' if r['direction_hit'] else 'miss'}>{'✓' if r['direction_hit'] else '✗'}</span></td>"
        f"<td class=l>{_e(r['position'])}</td><td>{r['position_pnl_pct']:+.2f}%</td></tr>" for r in tr["rows"])
    return head + ("<div class=tbl><table class=sortable><thead><tr><th class=l>Session</th><th class=l>Ticker</th><th>Base</th>"
                   "<th>Forecast</th><th>Actual</th><th>Error</th><th class=l>Dir</th><th class=l>Position</th><th>P&amp;L</th>"
                   f"</tr></thead><tbody>{rows}</tbody></table></div>")


def _ticker_card(s: dict, chart: str) -> str:
    cur = s["currency"]
    lv = s["levels"]
    plan = (f"<b>{s['position']}</b> @ {_fmt_px(s['entry'], cur)} · stop {_fmt_px(s['stop'], cur)} · target {_fmt_px(s['target'], cur)}"
            f" · R:R {s['rr']:.2f} — {_e(s['reason'])}") if s["position"] != "FLAT" else f"<b>FLAT</b> — {_e(s['reason'])}"
    srcs = " · ".join(f"{k} {v:+.2f}" for k, v in s["sent_by_source"].items())
    crowd = ", ".join(f"{c['price']:,.0f} ({c['mentions']}×)" for c in lv["crowd_levels"]) or "none cited"
    flags = "".join(f"<span class=flag>{_e(f)}</span>" for f in s["flags"])

    def plist(items):
        if not items:
            return "<li class=muted>none</li>"
        return "".join(f"<li><span class=muted>{_e(p['ts'])} · {_e(p['source'])} · {p['score']:+.2f}</span><br>"
                       f"<a href='{_e(p['url'])}' target=_blank rel=noopener>{_e(p['text'])}</a></li>" for p in items)
    news = "".join(f"<li><span class=muted>{_e(n['ts'])} · {n['score']:+.2f}</span> <a href='{_e(n['url'])}' target=_blank rel=noopener>{_e(n['title'])}</a></li>"
                   for n in s["news"]) or "<li class=muted>no news in window</li>"
    return f"""
<div class=card id="t-{_e(s['ticker'])}">
 <div class=tcard-h><h3>{_e(s['ticker'])} · {_e(s['name'])}</h3><span class=big>{_fmt_px(s['price'], cur)} {cur or ''}</span>
  <span class=muted>close {s['price_date']}</span><span class='pill {s['sentiment_label']}'>{s['sentiment_label']} {s['sentiment_score']:+.2f}</span>
  <span class='pill {s['direction']}'>n+1 {s['pred_session']}: {_fmt_px(s['pred_close'], cur)} ({_pct(s['pred_ret'])})</span></div>
 <div>{flags}</div>
 {_thesis(s)}
 <div class=facts>
  <div><span>Position</span>{plan}</div>
  <div><span>Support / Resistance</span>{_fmt_px(lv['support'], cur)} / {_fmt_px(lv['resistance'], cur)}</div>
  <div><span>Pivots S1 · P · R1</span>{_fmt_px(lv['pivots']['S1'], cur)} · {_fmt_px(lv['pivots']['P'], cur)} · {_fmt_px(lv['pivots']['R1'], cur)}</div>
  <div><span>Volume POC · ATR(14)</span>{_fmt_px(lv['poc'], cur)} · {_fmt_px(lv['atr'], cur)}</div>
  <div><span>Crowd-cited levels</span>{crowd}</div>
  <div><span>Sentiment by source</span>{srcs or '–'}{f" · news {s['sentiment_news']:+.2f}" if s['sentiment_news'] is not None else ''}</div>
  <div><span>68% n+1 range</span>{_fmt_px(s['pred_low_68'], cur)} – {_fmt_px(s['pred_high_68'], cur)}</div>
  <div><span>After-close chatter</span>{s['after_close_posts']} posts{f" · {s['after_close_sentiment']:+.2f}" if s['after_close_sentiment'] is not None else ''}</div>
 </div>
 {chart}
 <div class=posts><div><b style="color:var(--up)">Most-engaged bullish posts</b><ul>{plist(s['top_posts']['bull'])}</ul></div>
 <div><b style="color:var(--down)">Most-engaged bearish posts</b><ul>{plist(s['top_posts']['bear'])}</ul></div></div>
 <div class=small><b>News</b><ul>{news}</ul></div>
</div>"""


def _metrics_table(metrics: list[dict]) -> str:
    if not metrics:
        return "<p class=muted>Not enough sentiment sessions yet for an out-of-sample test.</p>"
    rows = []
    for m in metrics:
        d = (f"{m['dir_acc_pct']:.1f}% ({int(m['dir_hits'])}/{int(m['dir_n'])}, p={m['dir_p_value']:.2f})"
             if m["model"] != "naive" and m.get("dir_n") else "–")
        rows.append(f"<tr><td class=l>{m['scope']}</td><td class=l>{m['model'].replace('_', '-')}</td><td>{m['n']}</td><td>{m.get('sessions', '–')}</td>"
                    f"<td>{m['MAE_pct']:.2f}%</td><td>{m['RMSE_pct']:.2f}%</td><td>{m['MAE_vs_naive_pct']:+.1f}%</td><td>{d}</td></tr>")
    return ("<div class=tbl><table><thead><tr><th class=l>Scope</th><th class=l>Model</th><th>n</th><th>Sessions</th><th>Price MAE</th><th>RMSE</th>"
            "<th>MAE vs naive</th><th>Direction hit-rate</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


def build_report(res: dict, prices: pd.DataFrame, posts: pd.DataFrame, panel: pd.DataFrame, bt: pd.DataFrame, path: Path) -> None:
    first = True
    sections = []
    for ex in res["exchanges"]:
        rows = sorted([s for s in res["summary"] if s["exchange"] == ex["code"]], key=lambda s: (-s["mentions"], -s["sentiment_score"]))
        if not rows:
            continue
        tickers = [s["ticker"] for s in rows]
        b = _fig_html(bubble(rows), first); first = False
        h = _fig_html(heatmap(panel, tickers))
        mb = _fig_html(mentions_by_source(posts, tickers))
        cards = "".join(_ticker_card(s, _fig_html(ticker_chart(s, prices, panel))) for s in rows)
        btxt = (res.get("briefs") or {}).get(ex["code"])
        brief = (f"<div class='card brief'><span class=by>Market brief · {_author(res.get('analyst'))}</span>"
                 f"<p>{_e(btxt)}</p></div>") if btxt else ""
        chans = ", ".join(f"<a href='https://t.me/s/{c}' target=_blank rel=noopener>@{c}</a>" for c in ex["telegram_channels"])
        sections.append(f"""
<h2>{_e(ex['name'])}</h2>
<p class=muted>Social sources: <b>{' + '.join(ex['sources'])}</b>{(' · Telegram channels: ' + chans) if chans else ''}.
Sorted by mentions, then sentiment. Click a column header to re-sort.</p>
{brief}
<div class=card>{_rank_table(rows)}</div>
<div class=grid2><div class=card>{b}</div><div class=card>{h}</div></div>
<div class=card>{mb}</div>
{cards}""")

    all_m = [m for m in res["metrics"] if m["scope"] == "ALL"]
    sm = next((m for m in all_m if m["model"] == "sentiment"), None)
    kpi_dir = f"{sm['dir_acc_pct']:.0f}%" if sm and sm.get("dir_n") else "–"
    bt_html = _fig_html(backtest_chart(bt), first) if not bt.empty else ""
    src_counts = " · ".join(f"{r['exchange']} {r['source']} {r['n']:,}" for r in res["counts"]["posts_by_source"])
    ing = res["ingest"]
    ing_txt = (f"Apify actor <code>{_e(ing.get('actor_id'))}</code>, run <code>{_e(ing.get('run_id', '–'))}</code>, dataset <code>{_e(ing.get('dataset_id', '–'))}</code>"
               if ing.get("mode") == "apify" else f"mode <code>{_e(ing.get('mode'))}</code> (same crawl plan as the Apify actor, fetched locally)")
    coefs = ", ".join(f"{k} {v:+.4f}" for k, v in res.get("stage2_coefs", {}).items()) or "not fitted (too few sessions)"

    page = f"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>HK & KR Sentiment Report</title><style>{CSS}</style></head><body><div class=wrap>
<h1>Social-sentiment n+1 price report — HKEX & KRX</h1>
<div class=muted>Generated {res['generated_at'][:16].replace('T', ' ')} UTC · window {res['lookback_days']} days (from {res['cutoff_utc'][:16].replace('T', ' ')} UTC)
 · scorer <b>{_e(res['scorer'])}</b> · news <b>{_e(res['news_provider'])}</b></div>
<div class=small muted>Ingestion: {ing_txt}</div>
<div class=kpis>
 <div class=kpi><span class=muted>Social posts</span><b>{res['counts']['posts']:,}</b><span class=small>{src_counts}</span></div>
 <div class=kpi><span class=muted>News articles</span><b>{res['counts']['news']:,}</b></div>
 <div class=kpi><span class=muted>Assets</span><b>{len(res['summary'])}</b><span class=small>5 per exchange</span></div>
 <div class=kpi><span class=muted>OOS direction hit-rate</span><b>{kpi_dir}</b><span class=small>sentiment model, walk-forward</span></div>
</div>
{''.join(sections)}
<h2>Live track record</h2>
<p class=muted>Real n+1 forecasts made by earlier runs, scored against the close that printed afterwards.</p>
<div class=card>{_track_record(res.get('track_record') or {})}</div>
<h2>Did it work? Out-of-sample check</h2>
<p class=muted>Walk-forward: for each session in the sentiment window (after a 2-session warm-up) the model is re-fitted only on data
available at that session's close and predicts the next close. Baselines: <b>naive</b> = random walk (tomorrow = today), <b>price-only</b> =
the same model without sentiment. With ~10 days of social data there are only a handful of test <i>sessions</i>, and stocks on the same exchange move together on
each of them, so the effective sample is closer to the session count than to n. Treat hit-rates as a sanity check, not as proof of edge;
the p-value assumes independent predictions and is therefore optimistic.</p>
<div class=card>{_metrics_table(res['metrics'])}</div>
{f'<div class=card>{bt_html}</div>' if bt_html else ''}
<div class=card small>Sentiment-layer coefficients (standardised features → next-day log-return residual): {coefs}</div>
<p class=muted>Live predictions are appended to <code>outputs/predictions_log.csv</code>; run <code>python -m hsagent check</code> after the next session closes to score them.</p>
<h2>Method</h2>
<div class="card method"><ul>
<li><b>Ingestion.</b> All social pages and price bars are fetched by the Apify actor <code>apify/cheerio-scraper</code> with a custom page function that follows pagination back to the window cutoff. LIHKG and Naver are read from their JSON APIs, Telegram from public channel previews (<code>t.me/s/…?q=keyword</code>). News comes from Exa search (English + local-language query per asset, <code>category=news</code>).</li>
<li><b>Fields.</b> Every post keeps UTC timestamp, text, likes/dislikes, replies/forwards, views, author and author follower count (Telegram channel subscribers; Naver "verified holder" badge).</li>
<li><b>UTC alignment.</b> Each post is attached to the first session whose close (16:00 HKT / 15:30 KST, converted to UTC) is at or after it, so features never contain information from after the bar they predict from. Posts after the last close are shown separately as after-close chatter.</li>
<li><b>Sentiment.</b> Multilingual finance lexicon (Cantonese, Traditional/Simplified Chinese, Korean retail slang, English, emoji) with negation; optional Claude scorer. Scores are engagement-weighted (log of likes, replies, views and followers) and adjusted for crowd agreement (a bullish post the crowd down-votes counts less).</li>
<li><b>Model.</b> Stage 1: ridge on price features (returns, volatility, RSI, distance from SMA20, range position, volume z-score, index return) pooled over a year of history. Stage 2: shrunk ridge on sentiment features (social and news sentiment, change in sentiment, mention z-score, bull-minus-bear share) fitted to stage-1 residuals inside the sentiment window.</li>
<li><b>Levels.</b> Fractal swing levels clustered by ATR and ranked by touches and recency; floor pivots; 90-session volume point-of-control; and crowd-cited prices pulled from the posts themselves.</li>
<li><b>Analyst agent.</b> A Claude tool-use agent (Anthropic SDK tool runner) reads the run through six read-only tools
(tickers, ticker detail, posts, daily sentiment, news, backtest) and writes each thesis and market brief through two write tools.
Scraped text reaches it only as tool results and is treated as untrusted data; numbers must come from tools. Without an API key a
rule-based writer fills the same fields from the same data.</li>
<li><b>Position.</b> LONG/SHORT only when the forecast exceeds a quarter of daily volatility <i>and</i> crowd sentiment does not contradict it; stop beyond the nearest level (max 2 ATR), target at the opposite level. Otherwise FLAT. Research output, not investment advice.</li>
</ul></div>
</div><script>{SORT_JS}</script></body></html>"""
    Path(path).write_text(page, encoding="utf-8")
