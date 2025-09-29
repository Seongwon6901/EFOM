# -*- coding: utf-8 -*-
"""
pi_uploader.py — Compact, production‑ready publisher for sending EFOM outputs to
OSIsoft PI via PIconnect.

Key features
- Context‑managed connection with point cache
- Type‑aware writes (numeric / digital / string)
- Skips None/NaN safely; NO_REPLACE + BUFFER_IF_POSSIBLE by default
- Simple key→tag mapping (EFOM_TAG_MAP) + type hints for special tags
- Tiny CLI for ad‑hoc publishing from CSV/Parquet

Typical usage inside efom_runner:

    from datetime import datetime
    from pi_uploader import PIServerConfig, PIPublisher, publish_efom_record, EFOM_TAG_MAP

    ts = datetime.now()
    record = {
        "rcot1": 837.4, "rcot2": 835.7, "eth_prod": 66.24,
        "margin_hourly": 1186.55, "cycle_success": True,
        "timestamp_str": ts.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with PIPublisher(PIServerConfig(server="pirtdb", tz="Asia/Seoul")) as pub:
        pub.require_points(EFOM_TAG_MAP.values())
        publish_efom_record(pub, record, ts)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Mapping, Optional
import logging

import numpy as np
import pandas as pd
import PIconnect as PI
from PIconnect.PIConsts import AuthenticationMode, UpdateMode, BufferMode

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
_log = logging.getLogger("pi_uploader")
if not _log.handlers:
    _log.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _log.addHandler(h)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PIServerConfig:
    server: str = "pirtdb"  # or IP like "172.17.21.117"
    auth_mode: AuthenticationMode = AuthenticationMode.WINDOWS_AUTHENTICATION
    username: str = ""   # blank → current Windows user
    password: str = ""
    domain: Optional[str] = None
    tz: str = "Asia/Seoul"

    def server_kwargs(self) -> Dict[str, Any]:
        if self.auth_mode == AuthenticationMode.WINDOWS_AUTHENTICATION:
            return dict(
                server=self.server,
                authentication_mode=self.auth_mode,
                username=self.username or None,
                password=self.password or None,
                domain=self.domain,
            )
        return dict(
            server=self.server,
            authentication_mode=AuthenticationMode.PI_USER_AUTHENTICATION,
            username=self.username,
            password=self.password,
        )

# ─────────────────────────────────────────────────────────────────────────────
# Publisher
# ─────────────────────────────────────────────────────────────────────────────
class PIPublisher:
    """Context‑managed PI publisher with point caching and type‑aware writes."""

    def __init__(self, cfg: PIServerConfig):
        self.cfg = cfg
        PI.PIConfig.DEFAULT_TIMEZONE = cfg.tz
        self._srv: Optional[PI.PIServer] = None
        self._cache: Dict[str, Any] = {}

    def __enter__(self) -> "PIPublisher":
        self._srv = PI.PIServer(**self.cfg.server_kwargs())
        self._srv.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._srv is not None:
            self._srv.__exit__(exc_type, exc, tb)
            self._srv = None

    # ── Points ──────────────────────────────────────────────────────────────
    def _get_point(self, tag: str):
        assert self._srv is not None, "Use 'with PIPublisher(...) as pub:'"
        if tag in self._cache:
            return self._cache[tag]
        pts = self._srv.search(tag)
        if not pts:
            raise KeyError(f"PI tag not found: {tag}")
        exact = [p for p in pts if getattr(p, "name", None) == tag]
        pt = exact[0] if exact else (pts[0] if len(pts) == 1 else None)
        if pt is None:
            cand = ", ".join(getattr(p, "name", "<no-name>") for p in pts[:6])
            raise KeyError(f"Ambiguous search for '{tag}'. Candidates: {cand} ...")
        self._cache[tag] = pt
        return pt

    def require_points(self, tags: Iterable[str]) -> None:
        for t in tags:
            _ = self._get_point(t)
        _log.info("Validated %d PI points exist.", len(self._cache))

    # ── Helpers ─────────────────────────────────────────────────────────────
    def _ensure_ts(self, ts: datetime) -> datetime:
        tz = self.cfg.tz
        if getattr(ts, "tzinfo", None) is None:
            return pd.Timestamp(ts).tz_localize(tz).to_pydatetime()
        return pd.Timestamp(ts).tz_convert(tz).to_pydatetime()

    @staticmethod
    def _is_nan(x: Any) -> bool:
        try:
            return x is None or (isinstance(x, float) and np.isnan(x))
        except Exception:
            return False

    @staticmethod
    def _to_digital(x: Any) -> int:
        if isinstance(x, bool):
            return 1 if x else 0
        if isinstance(x, (int, np.integer)):
            return 1 if int(x) != 0 else 0
        if isinstance(x, str):
            s = x.strip().lower()
            if s in {"ok", "on", "true", "1", "success"}: return 1
            if s in {"error", "off", "false", "0", "fail", "failed"}: return 0
        return 1 if x else 0

    # ── Core write ──────────────────────────────────────────────────────────
    def publish_value(
        self,
        tag: str,
        value: Any,
        ts: datetime,
        *,
        update_mode: UpdateMode = UpdateMode.NO_REPLACE,
        buffer_mode: BufferMode = BufferMode.BUFFER_IF_POSSIBLE,
        skip_nan: bool = True,
        treat_as: Optional[str] = None,  # 'digital' | 'string' | None
    ) -> None:
        if skip_nan and self._is_nan(value):
            _log.debug("Skip NaN/None for %s", tag)
            return
        pt = self._get_point(tag)
        ts2 = self._ensure_ts(ts)

        try:
            pointtype = (getattr(pt, "pointtype", "") or "").lower()
        except Exception:
            pointtype = ""

        val = value
        if treat_as == "digital" or pointtype == "digital":
            val = self._to_digital(value)
        elif treat_as == "string" or pointtype == "string":
            val = str(value)
        else:
            try:
                val = float(value)
            except Exception:
                val = str(value)

        pt.update_value(val, ts2, update_mode, buffer_mode)
        _log.debug("Wrote %s=%r @ %s", tag, val, ts2)

    def publish_dict(self, values: Mapping[str, Any], ts: datetime, *, hints: Optional[Mapping[str, str]] = None) -> None:
        hints = hints or {}
        for tag, val in values.items():
            self.publish_value(tag, val, ts, treat_as=hints.get(tag))

    def publish_dict_mapped(
        self,
        values: Mapping[str, Any],  # your keys
        key_to_tag: Mapping[str, str],  # key → PI tag
        ts: datetime,
        *,
        hints: Optional[Mapping[str, str]] = None,  # key → 'digital'/'string'
    ) -> None:
        hints = hints or {}
        for key, tag in key_to_tag.items():
            if key in values:
                self.publish_value(tag, values[key], ts, treat_as=hints.get(key))

# ─────────────────────────────────────────────────────────────────────────────
# EFOM mapping (adjust keys to your efom_runner outputs)
# ─────────────────────────────────────────────────────────────────────────────
EFOM_TAG_MAP: Dict[str, str] = {
    # Digital status / meta
    "cycle_success": "M10_EFOM_CYCLE_SUCCESS",
    "timestamp_str": "M10_EFOM_TIMESTAMP",   # string
    "performance": "M10_EFOM_PERFORMANCE",

    # ── RCOTs (CURRENT snapshot vs RECOMMENDED) ──
    # If your plant already trends actual RCOTs from DCS, DO NOT overwrite them.
    # Instead, ask PI admin to create namespaced EFOM "_CURR" tags fed by your runner,
    # or just omit CURRENT here and only write *_REC.

    # Chamber 1–3 (single feed)
    "rcot1_curr": "M10_EFOM_RCOT1_CURR",
    "rcot1_rec":  "M10_EFOM_RCOT1",         # using existing tag for RECOMMENDED
    "rcot2_curr": "M10_EFOM_RCOT2_CURR",
    "rcot2_rec":  "M10_EFOM_RCOT2",
    "rcot3_curr": "M10_EFOM_RCOT3_CURR",
    "rcot3_rec":  "M10_EFOM_RCOT3",

    # Chamber 4–6 (split: NAP/GAS)
    "rcot4_nap_curr": "M10_EFOM_RCOT4_NAP_CURR",
    "rcot4_nap_rec":  "M10_EFOM_RCOT4_NAP",
    "rcot4_gas_curr": "M10_EFOM_RCOT4_GAS_CURR",
    "rcot4_gas_rec":  "M10_EFOM_RCOT4_GAS",

    "rcot5_nap_curr": "M10_EFOM_RCOT5_NAP_CURR",
    "rcot5_nap_rec":  "M10_EFOM_RCOT5_NAP",
    "rcot5_gas_curr": "M10_EFOM_RCOT5_GAS_CURR",
    "rcot5_gas_rec":  "M10_EFOM_RCOT5_GAS",

    "rcot6_nap_curr": "M10_EFOM_RCOT6_NAP_CURR",
    "rcot6_nap_rec":  "M10_EFOM_RCOT6_NAP",
    "rcot6_gas_curr": "M10_EFOM_RCOT6_GAS_CURR",
    "rcot6_gas_rec":  "M10_EFOM_RCOT6_GAS",

    # ── Quantities BEFORE/AFTER (t/h) ──
    "eth_before":  "M10_EFOM_ETH_BEFORE",
    "eth_after":   "M10_EFOM_ETH_AFTER",
    "prop_before": "M10_EFOM_PROP_BEFORE",
    "prop_after":  "M10_EFOM_PROP_AFTER",
    "mc4_before":  "M10_EFOM_MC4_BEFORE",
    "mc4_after":   "M10_EFOM_MC4_AFTER",
    "rpg_before":  "M10_EFOM_RPG_BEFORE",
    "rpg_after":   "M10_EFOM_RPG_AFTER",

    # Keep the legacy aggregate tags if operators are already used to them
    "eth_prod":    "M10_EFOM_ETH_PROD",      # interpret as AFTER if you still write it
    "prop_prod":   "M10_EFOM_PROP_PROD",
    "mc4_prod":    "M10_EFOM_MC4_PROD",
    "rpg_prod":    "M10_EFOM_RPG_PROD",

    # ── Economics ──
    "margin_hourly":  "M10_EFOM_MARGIN_HOURLY",   # AFTER snapshot $/h
    "margin_uplift":  "M10_EFOM_MARGIN_UPLIFT",   # Δ$/h (after - before)

    # Error metrics (optional)
    "mape_ethy": "M10_EFOM_MAPE_ETHY",
    "mape_prop": "M10_EFOM_MAPE_PROP",
    "mape_mc4":  "M10_EFOM_MAPE_MC4",
    "mape_rpg":  "M10_EFOM_MAPE_RPG",
}

EFOM_TYPE_HINTS: Dict[str, str] = {
    "cycle_success": "digital",
    "timestamp_str": "string",
}

# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrappers
# ─────────────────────────────────────────────────────────────────────────────
def publish_efom_record(pub: PIPublisher, record: Mapping[str, Any], ts: datetime) -> None:
    """Publish any pre-built record using the global EFOM_TAG_MAP/hints."""
    pub.publish_dict_mapped(record, EFOM_TAG_MAP, ts, hints=EFOM_TYPE_HINTS)


def publish_efom_outcome(
    pub: PIPublisher,
    *,
    ts: datetime,
    rcot_current: Mapping[str, Any],     # keys: rcot1_curr, rcot4_nap_curr, ...
    rcot_recommended: Mapping[str, Any], # keys: rcot1_rec, rcot4_nap_rec, ...
    qty_before: Mapping[str, Any],       # keys: eth_before, prop_before, mc4_before, rpg_before
    qty_after: Mapping[str, Any],        # keys: eth_after,  prop_after,  mc4_after,  rpg_after
    margin_after: Optional[float] = None,
    margin_uplift: Optional[float] = None,
    status_ok: Optional[bool] = None,
    performance: Optional[float] = None,
    mape: Optional[Mapping[str, Any]] = None,  # {mape_ethy:..., mape_prop:..., ...}
) -> None:
    """
    Convenience publisher that merges the provided blocks with standard meta fields
    and writes them in one go. Missing keys are simply skipped.
    """
    rec: Dict[str, Any] = {
        **{"timestamp_str": pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")},
        **(rcot_current or {}),
        **(rcot_recommended or {}),
        **(qty_before or {}),
        **(qty_after or {}),
    }
    if margin_after is not None:
        rec["margin_hourly"] = margin_after
    if margin_uplift is not None:
        rec["margin_uplift"] = margin_uplift
    if status_ok is not None:
        rec["cycle_success"] = status_ok
    if performance is not None:
        rec["performance"] = performance
    if mape:
        rec.update(mape)

    publish_efom_record(pub, rec, ts)

# ─────────────────────────────────────────────────────────────────────────────
# CLI (optional)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Publish one EFOM row to PI from CSV/Parquet")
    ap.add_argument("--server", default="pirtdb")
    ap.add_argument("--tz", default="Asia/Seoul")
    ap.add_argument("--csv", default="")
    ap.add_argument("--parquet", default="")
    ap.add_argument("--ts-col", default="timestamp")
    ap.add_argument("--row", type=int, default=-1, help="Row index to publish (-1=last)")
    args = ap.parse_args()

    cfg = PIServerConfig(server=args.server, tz=args.tz)

    if not args.csv and not args.parquet:
        ap.error("Provide --csv or --parquet")

    if args.csv:
        df = pd.read_csv(args.csv, parse_dates=[args.ts_col])
    else:
        df = pd.read_parquet(args.parquet)
        if args.ts_col in df.columns:
            df[args.ts_col] = pd.to_datetime(df[args.ts_col])

    i = args.row if args.row >= 0 else (len(df) - 1)
    row = df.iloc[i]
    ts = pd.Timestamp(row[args.ts_col]).to_pydatetime()

    record: Dict[str, Any] = {}
    for k in EFOM_TAG_MAP.keys():
        if k in row.index:
            record[k] = row[k]
    record.setdefault("timestamp_str", pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S"))

    with PIPublisher(cfg) as pub:
        pub.require_points(EFOM_TAG_MAP.values())
        publish_efom_record(pub, record, ts)
        _log.info("Published EFOM row %d to PI", i)
