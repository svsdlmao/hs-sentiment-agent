"""Configuration loading (config.toml + .env)."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Asset:
    ticker: str
    name: str
    keywords: list[str]
    exchange: str

    @property
    def code(self) -> str:
        """Local exchange code: '0700.HK' -> '0700', '005930.KS' -> '005930'."""
        return self.ticker.split(".")[0]


@dataclass
class Exchange:
    code: str
    name: str
    tz: str
    close_local: str
    index_symbol: str
    sources: list[str]
    telegram_channels: list[str]
    assets: list[Asset] = field(default_factory=list)

    @property
    def zone(self) -> ZoneInfo:
        return ZoneInfo(self.tz)

    @property
    def close_time(self) -> time:
        h, m = self.close_local.split(":")
        return time(int(h), int(m))


@dataclass
class Config:
    raw: dict
    exchanges: list[Exchange]
    now_utc: datetime

    @property
    def run(self) -> dict:
        return self.raw["run"]

    @property
    def limits(self) -> dict:
        return self.raw["limits"]

    @property
    def apify(self) -> dict:
        return self.raw["apify"]

    @property
    def exa(self) -> dict:
        return self.raw["exa"]

    @property
    def cutoff_utc(self) -> datetime:
        return self.now_utc - timedelta(days=int(self.run["lookback_days"]))

    @property
    def output_dir(self) -> Path:
        p = ROOT / self.run["output_dir"]
        p.mkdir(parents=True, exist_ok=True)
        return p

    def all_assets(self) -> list[Asset]:
        return [a for ex in self.exchanges for a in ex.assets]

    def exchange(self, code: str) -> Exchange:
        return next(e for e in self.exchanges if e.code == code)


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env reader (KEY=VALUE lines); never overrides real env vars."""
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_config(path: Path | None = None, now_utc: datetime | None = None) -> Config:
    load_dotenv()
    path = path or ROOT / "config.toml"
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    exchanges = []
    for ex in raw["exchanges"]:
        e = Exchange(
            code=ex["code"], name=ex["name"], tz=ex["tz"], close_local=ex["close_local"],
            index_symbol=ex.get("index_symbol", ""), sources=ex["sources"],
            telegram_channels=ex.get("telegram_channels", []),
        )
        e.assets = [Asset(a["ticker"], a["name"], a["keywords"], e.code) for a in ex["assets"]]
        exchanges.append(e)
    return Config(raw=raw, exchanges=exchanges, now_utc=now_utc or datetime.now(timezone.utc))
