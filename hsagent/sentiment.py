"""Post-level sentiment scoring.

Default: a multilingual finance lexicon (Cantonese / Traditional + Simplified Chinese /
Korean retail slang / English / emoji) with negation handling. Fast, deterministic,
no API cost, and it understands forum slang (e.g. 떡상, 插水, 上車) that generic
models miss.

Optional: `--scorer claude` sends posts in batches to the Claude API (official
`anthropic` SDK, structured JSON output). Any batch that fails falls back to the lexicon.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re

import pandas as pd

log = logging.getLogger(__name__)

# weight 1 = mild, 2 = strong
BULL = {
    # Chinese (HK Cantonese + Traditional + Simplified)
    "大升": 2, "爆升": 2, "急升": 2, "飆升": 2, "飙升": 2, "抽升": 2, "狂升": 2, "破頂": 2, "破顶": 2, "創新高": 2,
    "创新高": 2, "新高": 1, "突破": 1, "利好": 2, "看好": 1, "看漲": 2, "看涨": 2, "買入": 1, "买入": 1, "增持": 1,
    "跑贏": 1, "跑赢": 1, "強勁": 1, "强劲": 1, "反彈": 1, "反弹": 1, "上調": 1, "上调": 1, "超預期": 2, "超预期": 2,
    "回購": 1, "回购": 1, "派息": 1, "牛市": 2, "好消息": 1, "上車": 1, "加倉": 1, "加仓": 1, "入貨": 1, "撈底": 1,
    "起飛": 2, "起飞": 2, "上升": 1, "上漲": 1, "上涨": 1, "升": 1, "漲": 1, "涨": 1, "賺": 1, "赚": 1, "盈利增長": 2,
    "盈利增长": 2, "勁升": 2, "目標價上調": 2, "目标价上调": 2, "扭虧": 2, "扭亏": 2,
    # Korean
    "떡상": 2, "급등": 2, "상한가": 2, "폭등": 2, "신고가": 2, "불기둥": 2, "가즈아": 2, "풀매수": 2, "상승": 1,
    "매수": 1, "호재": 2, "돌파": 1, "반등": 1, "강세": 1, "올라": 1, "오른다": 1, "오름": 1, "흑자": 1, "수익": 1,
    "실적개선": 2, "서프라이즈": 2, "목표가 상향": 2, "상향": 1, "매집": 1, "축하": 1, "익절": 1, "날아": 1, "줍줍": 1,
    # English
    "surge": 2, "soar": 2, "rally": 2, "record high": 2, "beat": 1, "beats": 1, "upgrade": 2, "bullish": 2, "buy": 1,
    "outperform": 1, "rebound": 1, "jump": 1, "jumps": 1, "gain": 1, "gains": 1, "rise": 1, "rises": 1, "strong": 1,
    "growth": 1, "raised": 1, "raises": 1, "higher": 1, "up": 0.5, "moon": 2, "long": 0.5,
    # emoji
    "🚀": 2, "📈": 2, "🔥": 1, "💰": 1, "🐂": 2, "🤑": 1,
}
BEAR = {
    "大跌": 2, "暴跌": 2, "急跌": 2, "插水": 2, "狂跌": 2, "跌穿": 2, "瀉": 1, "泻": 1, "劈": 1, "新低": 2, "利淡": 2,
    "利空": 2, "看淡": 1, "看跌": 2, "沽出": 1, "賣出": 1, "卖出": 1, "減持": 1, "减持": 1, "跑輸": 1, "跑输": 1,
    "下調": 1, "下调": 1, "遜預期": 2, "逊预期": 2, "不及預期": 2, "不及预期": 2, "虧損": 1, "亏损": 1, "熊市": 2,
    "爆煲": 2, "蝕": 1, "蚀": 1, "套牢": 2, "割肉": 2, "斬倉": 2, "斩仓": 2, "止蝕": 1, "止损": 1, "罰款": 1, "罚款": 1,
    "調查": 1, "调查": 1, "制裁": 1, "走貨": 1, "落車": 1, "見頂": 2, "见顶": 2, "下跌": 1, "跌": 1, "輸": 1, "输": 1,
    "沽空": 1, "做空": 1, "目標價下調": 2, "目标价下调": 2, "崩": 2, "危": 1,
    "떡락": 2, "급락": 2, "하한가": 2, "폭락": 2, "신저가": 2, "개미지옥": 2, "반토막": 2, "설거지": 2, "하락": 1,
    "매도": 1, "악재": 2, "붕괴": 2, "약세": 1, "손절": 2, "물렸": 2, "망했": 2, "망함": 2, "폭망": 2, "공매도": 1,
    "인버스": 1, "손실": 1, "쇼크": 2, "하향": 1, "적자": 1, "탈출": 1, "개잡주": 2, "빠진다": 1, "떨어": 1, "내려": 1,
    "plunge": 2, "plunges": 2, "slump": 2, "tumble": 2, "crash": 2, "miss": 1, "misses": 1, "downgrade": 2, "bearish": 2,
    "sell": 1, "underperform": 1, "drop": 1, "drops": 1, "fall": 1, "falls": 1, "weak": 1, "loss": 1, "losses": 1,
    "probe": 1, "ban": 1, "cut": 1, "cuts": 1, "lawsuit": 1, "decline": 1, "declines": 1, "lower": 1,
    "down": 0.5, "short": 0.5,
    "📉": 2, "💀": 1, "😭": 1, "🩸": 2, "🐻": 2, "😱": 1,
}
NEGATORS_CJK = ("不", "未", "冇", "無", "无", "没", "沒", "唔", "非", "안 ", "못 ")
NEGATORS_EN = ("not", "no", "never", "n't", "without")

_ascii_word = re.compile(r"^[a-z ]+$")

# One signed weight per term; a term in both lists would cancel, so BEAR wins only if absent from BULL.
_WEIGHTS: dict[str, float] = {**{k: -w for k, w in BEAR.items()}, **BULL}
_EN = sorted((k for k in _WEIGHTS if _ascii_word.match(k)), key=len, reverse=True)
_CJK = sorted((k for k in _WEIGHTS if not _ascii_word.match(k)), key=len, reverse=True)
# Single pass, longest alternative first (大升 before 升); finditer matches never overlap,
# so a short term inside a longer one is not double counted.
_TERMS = re.compile(r"(?P<en>\b(?:" + "|".join(map(re.escape, _EN)) + r")\b)|(?P<cjk>" + "|".join(map(re.escape, _CJK)) + ")")
_NEG_EN = re.compile(r"\b(?:" + "|".join(NEGATORS_EN) + r")\b[\w ]{0,8}$")


def lexicon_score(text: str) -> float:
    """Score in (-1, 1). Positive = bullish."""
    if not text:
        return 0.0
    t = text.lower()
    raw = 0.0
    for m in _TERMS.finditer(t):
        w = _WEIGHTS[m.group(0)]
        s = m.start()
        if m.group("en"):
            negated = bool(_NEG_EN.search(t[max(0, s - 14):s]))
        else:
            pre = t[max(0, s - 3):s]
            negated = any(n in pre for n in NEGATORS_CJK)
        raw += -w if negated else w
    return math.tanh(raw / 2.5) if raw else 0.0


# --- optional Claude scorer ---------------------------------------------------

CLAUDE_MODEL = os.environ.get("HSAGENT_CLAUDE_MODEL", "claude-opus-5-5")
_SYSTEM = (
    "You score retail-investor and news posts about one listed company for short-term price sentiment. "
    "Posts may be in Cantonese, Traditional/Simplified Chinese, Korean or English and use forum slang. "
    "Return a score from -1 (strongly bearish for the stock) to 1 (strongly bullish); 0 when the post is "
    "neutral, off-topic or pure noise. Judge the stance of the writer toward the stock, not the general mood."
)
_SCHEMA = {
    "type": "object",
    "properties": {"scores": {"type": "array", "items": {
        "type": "object",
        "properties": {"i": {"type": "integer"}, "score": {"type": "number"}},
        "required": ["i", "score"], "additionalProperties": False}}},
    "required": ["scores"], "additionalProperties": False,
}


def claude_scores(texts: list[str], company: str, batch: int = 60) -> list[float | None]:
    import anthropic  # optional dependency

    client = anthropic.Anthropic()
    out: list[float | None] = [None] * len(texts)
    for start in range(0, len(texts), batch):
        chunk = texts[start:start + batch]
        listing = "\n".join(f"[{i}] {t[:400].replace(chr(10), ' ')}" for i, t in enumerate(chunk))
        try:
            resp = client.beta.messages.create(
                model=CLAUDE_MODEL, max_tokens=16000,
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": _SCHEMA}},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                system=_SYSTEM,
                messages=[{"role": "user", "content": f"Company: {company}\nScore every post.\n\n{listing}"}],
            )
            if resp.stop_reason == "refusal":
                log.warning("Claude declined a batch for %s; lexicon used for it", company)
                continue
            text = next(b.text for b in resp.content if b.type == "text")
            for row in json.loads(text)["scores"]:
                if 0 <= row["i"] < len(chunk):
                    out[start + row["i"]] = max(-1.0, min(1.0, float(row["score"])))
        except anthropic.RateLimitError as e:
            log.warning("Claude rate limited (%s); lexicon used for this batch", e)
        except anthropic.APIStatusError as e:
            log.warning("Claude API error %s; lexicon used for this batch", e.status_code)
        except anthropic.APIConnectionError as e:
            log.warning("Claude connection error (%s); lexicon used for this batch", e)
        except (StopIteration, json.JSONDecodeError, KeyError) as e:
            log.warning("unparseable Claude output (%s); lexicon used for this batch", e)
    return out


def score_frame(df: pd.DataFrame, text_col: str, scorer: str = "lexicon", names: dict[str, str] | None = None) -> pd.Series:
    lex = df[text_col].fillna("").map(lexicon_score)
    if scorer != "claude" or df.empty:
        return lex
    result = lex.copy()
    for ticker, idx in df.groupby("ticker").groups.items():
        llm = claude_scores(df.loc[idx, text_col].fillna("").tolist(), (names or {}).get(ticker, ticker))
        for i, v in zip(idx, llm):
            if v is not None:
                result.loc[i] = v
    return result
