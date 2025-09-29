# -*- coding: utf-8 -*-
"""
EFOM Runner v3 (notebook-friendly script)

What it does:
  1) (Optional) Incrementally download PI first-row CSV with end_date overridden to NOW (local).
  2) Load frames via DataPipeline (prioritizes downloaded_csv).
  3) Build merged_lims (PONA):
       - sample PI minute data at 07:00/19:00 targets based on X_12h coverage
       - map *_gas series from X_12h at nearest 09:00/21:00 (±3h), ffill small gaps
       - alias C11 → C11+ per-carbon names when needed
       - as-of backfill from merged_lims2 (Excel) to bridge short outages
  4) Sanitize composition for SRTO (numeric; limited ffill; keep canonical gas names)
  5) Build SRTO pipeline + memoized Spyro function (as-of composition)
  6) Run main.run_production with online/latest-only or historical/closed_loop.

Edit the knobs in the “User knobs” section.
"""

from __future__ import annotations
from src.pi_uploader import PIServerConfig, PIPublisher
from PIconnect.PIConsts import UpdateMode, BufferMode

from pathlib import Path
from typing import Dict, Any, List, Sequence, Tuple, Optional, Literal
import re
import pandas as pd
import numpy as np

# Core orchestrator + helpers
from src import main
from src.main import ensure_pi_download

# Data loading primitives
from src.data_loading import DataPaths, ResampleConfig, DataPipeline, load_feed_data

# SRTO / SPYRO plumbing
from src.srto_pipeline import SRTOConfig, RCOTSweepConfig, FeedConfig, SRTOPipeline
from src.srto_components import component_index, MW

# PI downloader types
from src.pipeline import DownloadConfig, AuthenticationMode


# =========================
# User knobs
# =========================
MODE           = "online"            # 'historical' | 'closed_loop' | 'online'
DOWNLOAD_PI    = True                # pull from PI first (incremental)
OUT_DIR        = Path("prod_out/jupyter_v3")
INPUT_DIR      = Path("input")
INTER_DIR      = Path("intermediate")
DOWNLOADED_CSV = "pi_firstrow.csv"   # relative to INTER_DIR

# Historical window if MODE != 'online'
START_STR      = "2024-09-10"
END_STR        = None                 # None → open end

# SRTO paths
SRTO_DLL = Path(r"C:\Program Files\Pyrotec\SRTO")
SPY7S = [
    r"01. GF_HYBRID MODE_SRTO7_NAPH.SPY7",
    r"04. LF_NAPH MODE_SRTO7.SPY7",
    r"07. GF_GAS MODE_SRTO7.SPY7",
]

# Prices cleanup
REPLACE_PRICE_ZEROS = True

# PI downloader overrides (make end_date = now)
OVERRIDE_END_WITH_NOW     = True
SAFETY_LAG_MINUTES        = 2
ALIGN_END_TO_INTERVAL     = True

# Composition fill limits (07:00/19:00 cadence → ~2 weeks ≈ 28 stamps; use 60 if you want longer)
SHORT_FILL_LIMIT = 60

# =========================
# Helpers
# =========================


def last_9_or_21(now_local: pd.Timestamp) -> pd.Timestamp:
    # returns the last 09:00 or 21:00 <= now (tz-naive)
    now_local = pd.Timestamp(now_local).tz_convert('Asia/Seoul') if now_local.tzinfo else now_local.tz_localize('Asia/Seoul')
    last = ((now_local - pd.Timedelta(hours=9)).floor('12H') + pd.Timedelta(hours=9))
    return last.tz_localize(None)

def last_complete_12h_stamp(now: pd.Timestamp) -> pd.Timestamp:
    """
    Return the last completed 09:00 or 21:00 (Asia/Seoul), tz-naive.
    At 09:05 -> 09:00, at 21:05 -> 21:00.
    """
    now = pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize('Asia/Seoul')
    else:
        now = now.tz_convert('Asia/Seoul')
    last = (now - pd.Timedelta(hours=9)).floor('12H') + pd.Timedelta(hours=9)
    return last.tz_localize(None)

# ===== EFOM → PI push helpers =====

# ===== EFOM → PI push helpers =====
TAG_MAP = {
    "rcot1": "M10_EFOM_RCOT1",
    "rcot2": "M10_EFOM_RCOT2",
    "rcot3": "M10_EFOM_RCOT3",
    "rcot4_nap": "M10_EFOM_RCOT4_NAP",
    "rcot4_gas": "M10_EFOM_RCOT4_GAS",
    "rcot5_nap": "M10_EFOM_RCOT5_NAP",
    "rcot5_gas": "M10_EFOM_RCOT5_GAS",
    "rcot6_nap": "M10_EFOM_RCOT6_NAP",
    "rcot6_gas": "M10_EFOM_RCOT6_GAS",
    "eth_prod": "M10_EFOM_ETH_PROD",
    "prop_prod": "M10_EFOM_PROP_PROD",
    "mc4_prod": "M10_EFOM_MC4_PROD",
    "rpg_prod": "M10_EFOM_RPG_PROD",
    "margin_hourly": "M10_EFOM_MARGIN_HOURLY",
    "performance": "M10_EFOM_PERFORMANCE",
    # "timestamp_str": "M10_EFOM_TIMESTAMP",
    "mape_ethy": "M10_EFOM_MAPE_ETHY",
    "mape_prop": "M10_EFOM_MAPE_PROP",
    "mape_mc4": "M10_EFOM_MAPE_MC4",
    "mape_rpg": "M10_EFOM_MAPE_RPG",
}
STRING_TAGS = {"M10_EFOM_TIMESTAMP"}  # tags we *must* write as strings

def _extract_mape_for_push(metrics: pd.DataFrame) -> dict:
    """Return {'mape_ethy': float|None, 'mape_prop': ..., 'mape_mc4': ..., 'mape_rpg': ...} from metrics df."""
    if metrics is None or metrics.empty:
        return {}
    def get(tgt):
        try:
            v =metrics[metrics['target'] == tgt]['mape_pct'].mean()
            return float(v) if pd.notna(v) else None
        except Exception:
            return None
    return {
        'mape_ethy': get('Ethylene_prod_t+1'),
        'mape_prop': get('Propylene_prod_t+1'),
        'mape_mc4':  get('MixedC4_prod_t+1'),
        'mape_rpg':  get('RPG_prod_t+1'),
    }

def _latest_recs_from_outdir(out_dir: Path):
    # try usual names, then wildcard
    cand = [
        out_dir /"online" / "rcot_recommendations_sim.csv",
        out_dir /"online" / "rcot_recommendations.csv",

    ]
    if not any(p.exists() for p in cand):
        # fallback to first matching file if any
        wild = sorted(out_dir.glob("rcot_recommendations*.csv"))
        if wild:
            cand.append(wild[-1])
    rec_path = next((p for p in cand if p.exists()), None)
    if not rec_path:
        return None, {}, {}, {}
    recs = pd.read_csv(rec_path, parse_dates=["timestamp"]).sort_values("timestamp")
    last = recs.iloc[-1]
    ts = pd.Timestamp(last["timestamp"])
    # rcots
    rcots = {}
    for c in recs.columns:
        if c.startswith("rcot_opt_") and pd.notna(last[c]):
            rcots[c.replace("rcot_opt_", "")] = float(last[c])
    # products (current baseline; switch to *_opt_tph if you want optimal)
    prods = {}
    for p in ["Ethylene","Propylene","MixedC4","RPG"]:
        col = f"{p}_current_tph"
        if col in recs.columns and pd.notna(last[col]):
            prods[p] = float(last[col])
    # margin/performance if you stored them (else zeros)
    extras = dict(
        margin_hourly=float(last.get("improvement_per_h", 0.0)) if pd.notna(last.get("improvement_per_h", np.nan)) else 0.0,
        performance=0.0,  # plug your own score if you have it
    )
    return ts, rcots, prods, extras

def _fallback_from_frames(X_12h: pd.DataFrame, Y_12h: pd.DataFrame):
    last = X_12h.dropna(how="all").iloc[-1]
    ts = last.name
    rcots = {}
    for k in ["RCOT_chamber1","RCOT_chamber2","RCOT_chamber3",
              "RCOT_naphtha_chamber4","RCOT_gas_chamber4",
              "RCOT_naphtha_chamber5","RCOT_gas_chamber5",
              "RCOT_naphtha_chamber6","RCOT_gas_chamber6"]:
        if k in last and pd.notna(last[k]): rcots[k] = float(last[k])
    prods = {}
    if not Y_12h.empty and ts in Y_12h.index:
        for p in ["Ethylene","Propylene","MixedC4","RPG"]:
            col = f"{p}_prod_t+1"
            if col in Y_12h.columns and pd.notna(Y_12h.at[ts, col]):
                prods[p] = float(Y_12h.at[ts, col])
    extras = dict(margin_hourly=0.0, performance=0.0)
    return ts, rcots, prods, extras

# def _build_push_record(out_dir: Path, X_12h: pd.DataFrame, Y_12h: pd.DataFrame, mape: dict | None = None):
#     ts, rcots, prods, extras = _latest_recs_from_outdir(out_dir)
#     if ts is None:
#         ts, rcots, prods, extras = _fallback_from_frames(X_12h, Y_12h)
def _build_push_record(out_dir: Path, X_12h: pd.DataFrame, Y_12h: pd.DataFrame,
                       mape: dict | None = None,
                       override_ts: pd.Timestamp | None = None):
    ts, rcots, prods, extras = _latest_recs_from_outdir(out_dir)
    if ts is None:
        ts, rcots, prods, extras = _fallback_from_frames(X_12h, Y_12h)

    # use the stamp we actually ran on
    if override_ts is not None:
        ts = pd.Timestamp(override_ts)

    def pick(d, *keys):
        for k in keys:
            if k in d: return d[k]
        return None

    rec = {
        "rcot1":     pick(rcots, "RCOT_chamber1"),
        "rcot2":     pick(rcots, "RCOT_chamber2"),
        "rcot3":     pick(rcots, "RCOT_chamber3"),
        "rcot4_nap": pick(rcots, "RCOT_naphtha_chamber4"),
        "rcot4_gas": pick(rcots, "RCOT_gas_chamber4"),
        "rcot5_nap": pick(rcots, "RCOT_naphtha_chamber5"),
        "rcot5_gas": pick(rcots, "RCOT_gas_chamber5"),
        "rcot6_nap": pick(rcots, "RCOT_naphtha_chamber6"),
        "rcot6_gas": pick(rcots, "RCOT_gas_chamber6"),
        "eth_prod":  prods.get("Ethylene"),
        "prop_prod": prods.get("Propylene"),
        "mc4_prod":  prods.get("MixedC4"),
        "rpg_prod":  prods.get("RPG"),
        "margin_hourly": extras.get("margin_hourly", 0.0),
        "performance":   extras.get("performance", 0.0),
        "timestamp_str": pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S"),
    }
    if mape:
        rec.update({k: (None if v is None else float(v)) for k, v in mape.items()})

    rec = {k: v for k, v in rec.items() if v is not None}
    return ts, rec

# def _safe_publish_record(pub: PIPublisher, record: dict, ts):
#     tz = pub.cfg.tz
#     ts_local = pd.Timestamp(ts)
#     ts_local = ts_local.tz_localize(tz) if ts_local.tz is None else ts_local.tz_convert(tz)

#     # clamp future just in case
#     now_pi = pd.Timestamp.now(tz=tz).floor("T")
#     if ts_local > now_pi:
#         ts_local = now_pi - pd.Timedelta(seconds=1)

#     for key, tag in TAG_MAP.items():
#         if key not in record:
#             continue
#         val = record[key]
#         if val is None or (isinstance(val, float) and np.isnan(val)):
#             continue
#         try:
#             pt = pub._get_point(tag)
#             ptype = (getattr(pt, "pointtype", "") or "").lower()

#             if tag in STRING_TAGS:
#                 # if PI point isn't actually string, skip to avoid float coercion errors
#                 if ptype != "string":
#                     print(f"[skip] {tag}: PI point type is {ptype}, expected 'string'")
#                     continue
#                 v = str(val)
#             else:
#                 # numeric path
#                 v = float(val)

#             pt.update_value(v, ts_local.to_pydatetime(),
#                             UpdateMode.NO_REPLACE, BufferMode.BUFFER_IF_POSSIBLE)
#         except Exception as e:
#             print(f"[fail] {tag}: {e}")

def _safe_publish_record(pub: PIPublisher, record: dict, ts):
    tz = pub.cfg.tz
    ts_local = pd.Timestamp(ts)
    ts_local = ts_local.tz_localize(tz) if ts_local.tz is None else ts_local.tz_convert(tz)

    # clamp future just in case
    now_pi = pd.Timestamp.now(tz=tz).floor("T")
    if ts_local > now_pi:
        ts_local = now_pi - pd.Timedelta(seconds=1)

    for key, tag in TAG_MAP.items():
        if key not in record:
            continue
        val = record[key]
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        try:
            pt = pub._get_point(tag)
            ptype = (getattr(pt, "pointtype", "") or "").lower()

            if tag in STRING_TAGS:
                # if PI point isn't actually string, skip to avoid float coercion errors
                if ptype != "string":
                    print(f"[skip] {tag}: PI point type is {ptype}, expected 'string'")
                    continue
                v = str(val)
            else:
                # numeric path
                v = float(val)

            pt.update_value(v, ts_local.to_pydatetime(),
                            UpdateMode.NO_REPLACE, BufferMode.BUFFER_IF_POSSIBLE)
        except Exception as e:
            print(f"[fail] {tag}: {e}")

# === gate helpers ===
GATE_TAG = "M10_EFOM_CYCLE_SUCCESS"  # handshake gate on the PI side

def _write_scalar(pub, tag: str, value: float, when_ts=None):
    """Write a numeric scalar to PI at 'when_ts' (or now) with clamped time."""
    tz = pub.cfg.tz
    ts_local = pd.Timestamp.now(tz=tz).floor("T") if when_ts is None else pd.Timestamp(when_ts)
    ts_local = ts_local.tz_localize(tz) if ts_local.tz is None else ts_local.tz_convert(tz)
    # clamp to <= now (PI rejects future)
    now_pi = pd.Timestamp.now(tz=tz).floor("T")
    if ts_local > now_pi:
        ts_local = now_pi - pd.Timedelta(seconds=1)
    pt = pub._get_point(tag)
    v = float(value)
    pt.update_value(v, ts_local.to_pydatetime(), UpdateMode.NO_REPLACE, BufferMode.BUFFER_IF_POSSIBLE)

def publish_with_gate(out_dir: Path, X_12h, Y_12h, mape=None, done_value: float = 1.0):
    """Gate=0 → payload → Gate=done_value (usually 1.0)."""
# right after main.run_production(...)
    try:
        ts_push, rec = _build_push_record(OUT_DIR, X_12h, Y_12h, mape=None, override_ts=latest)
        with PIPublisher(PIServerConfig(server="172.17.21.117", tz="Asia/Seoul")) as pub:
            _safe_publish_record(pub, rec, ts_push)
        print(f"[OK] Published EFOM setpoints at {ts_push}")
    except Exception as e:
        print(f"[WARN] PI publish skipped: {e}")

        # 2) payload at ts_push
        _safe_publish_record(pub, rec, ts_push)

        # 3) post-gate → done (default 1.0), at NOW
        try:
            _write_scalar(pub, GATE_TAG, done_value, when_ts=None)
            print(f"[GATE] {GATE_TAG}={done_value}")
        except Exception as e:
            print(f"[WARN] gate post-write failed: {e}")


def ml_prediction_check(
    *,
    X_12h: pd.DataFrame,
    Y_12h: pd.DataFrame,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    lookback: pd.Timedelta,
    target_cols: list[str],
    min_tr_rows: int,
    cache_tag: str = "_eval"
) -> tuple[pd.DataFrame, pd.DataFrame]:

    idx = pd.DatetimeIndex(X_12h.index)
    # normalize to tz-naive
    if idx.tz is not None:
        idx = idx.tz_convert('Asia/Seoul').tz_localize(None)
        X_12h = X_12h.copy()
        X_12h.index = idx
        if Y_12h.index.tz is not None:
            Y_12h = Y_12h.copy()
            Y_12h.index = Y_12h.index.tz_convert('Asia/Seoul').tz_localize(None)

    # default end/start if missing
    end = pd.Timestamp(end) if end is not None else X_12h.index.max()
    start = pd.Timestamp(start) if start is not None else (end - pd.Timedelta(days=90))

    # guard: start <= end
    if start > end:
        start, end = end - pd.Timedelta(days=90), end

    stamps = X_12h.index[(X_12h.index >= start) & (X_12h.index <= end)].sort_values()
    if len(stamps) == 0:
        return pd.DataFrame(), pd.DataFrame()

    tcols = [c for c in target_cols if c in Y_12h.columns]
    if not tcols:
        return pd.DataFrame(), pd.DataFrame()

    preds_all = main.ensure_ml_preds_for(
        stamps=stamps,
        Xsrc=X_12h,
        Ysrc=Y_12h,
        lookback=lookback,
        target_cols=tcols,
        mode="historical",
        train_mode="historical",
        cache_tag=cache_tag,
        Y_sim_state=None,
    )

    y_pred = preds_all.reindex(stamps)[tcols]
    y_true = Y_12h.reindex(stamps)[tcols]

    rows = []
    eps = 1e-9
    for c in tcols:
        yt = pd.to_numeric(y_true[c], errors="coerce")
        yp = pd.to_numeric(y_pred[c], errors="coerce")
        mask = yt.notna() & yp.notna()
        n = int(mask.sum())
        if n == 0:
            rows.append(dict(target=c, n=0, rmse=np.nan, mae=np.nan, mape_pct=np.nan, r2=np.nan, bias=np.nan, corr=np.nan))
            continue
        e = (yp[mask] - yt[mask]).to_numpy(float)
        ae = np.abs(e)
        rmse = float(np.sqrt(np.mean(e**2)))
        mae  = float(np.mean(ae))
        m_mask = np.abs(yt[mask].to_numpy(float)) > eps
        mape = float(np.mean(ae[m_mask] / np.abs(yt[mask].to_numpy(float)[m_mask]))*100.0) if m_mask.any() else np.nan
        bias = float(np.mean(e))
        yt_arr = yt[mask].to_numpy(float)
        ss_res = float(np.sum(e**2))
        ss_tot = float(np.sum((yt_arr - yt_arr.mean())**2))
        r2 = float(1.0 - ss_res/ss_tot) if ss_tot > eps else np.nan
        corr = float(np.corrcoef(yt_arr, yp[mask].to_numpy(float))[0,1]) if n > 1 else np.nan
        rows.append(dict(target=c, n=n, rmse=rmse, mae=mae, mape_pct=mape, r2=r2, bias=bias, corr=corr))

    metrics = pd.DataFrame(rows).set_index("target").sort_index()
    return y_pred, metrics

def _parse_interval(s: Optional[str]) -> pd.Timedelta:
    if not s: return pd.Timedelta(0)
    try:
        return pd.to_timedelta(s)
    except Exception:
        if s.endswith('m') and s[:-1].isdigit():
            return pd.Timedelta(minutes=int(s[:-1]))
        raise

def _override_end_date_now(cfg: DownloadConfig):
    if not OVERRIDE_END_WITH_NOW:
        return
    now_ts = pd.Timestamp.now(tz=cfg.tz) - pd.Timedelta(minutes=SAFETY_LAG_MINUTES)
    if (not cfg.recorded) and ALIGN_END_TO_INTERVAL:
        step = _parse_interval(cfg.interval)
        if step > pd.Timedelta(0):
            now_ts = pd.Timestamp(((now_ts.value // step.value) * step.value), tz=cfg.tz)
        else:
            now_ts = now_ts.floor('T')
    else:
        now_ts = now_ts.floor('T')
    cfg.end_date = now_ts.strftime('%Y-%m-%d %H:%M')
    print(f"[CONFIG] end_date overridden → {cfg.end_date} ({cfg.tz})")

def _ensure_date_col(df: pd.DataFrame) -> pd.DataFrame:
    if 'date' in df.columns:
        d = pd.to_datetime(df['date'], errors='coerce')
    elif isinstance(df.index, pd.DatetimeIndex):
        d = pd.to_datetime(df.index)
        df = df.reset_index(drop=True)
    else:
        raise ValueError("Need a datetime index or 'date' column")
    df = df.copy()
    # keep local-naive wall time (if tz-aware)
    if getattr(d, 'dt', None) is not None and d.dt.tz is not None:
        d = d.dt.tz_convert('Asia/Seoul').dt.tz_localize(None)
    df['date'] = d
    return df

def _ampm_targets_from_X_range(X_12h: pd.DataFrame,
                               times: Sequence[str] = ("07:00","19:00")) -> pd.DataFrame:
    idx = X_12h.index
    d0 = pd.to_datetime(idx.min()).normalize()
    d1 = pd.to_datetime(idx.max()).normalize()
    days = pd.date_range(d0 - pd.Timedelta(days=1), d1 + pd.Timedelta(days=1), freq="D")
    targets = []
    for d in days:
        for t in times:
            hh, mm = map(int, t.split(":"))
            targets.append(d + pd.Timedelta(hours=hh, minutes=mm))
    return pd.DataFrame({"date": pd.to_datetime(targets)}).sort_values("date").reset_index(drop=True)

def _build_gas_from_X(X_12h: pd.DataFrame,
                      targets: pd.DataFrame,
                      tol: pd.Timedelta = pd.Timedelta(hours=3)) -> pd.DataFrame:
    # prefer *_gas; if absent we’ll gracefully handle below
    gas_cols_x = ['Ethylene_gas','Ethane_gas','Propylene_gas','Propane_gas','n-Butane_gas','i-Butane_gas']
    have = [c for c in gas_cols_x if c in X_12h.columns]
    if not have:
        # fall back to canonical if *_gas missing
        have = [c for c in ['Ethylene','Ethane','Propylene','Propane','n-Butane','i-Butane'] if c in X_12h.columns]
        rename = {}  # already canonical
    else:
        rename = {c: c.replace('_gas','') for c in have}

    g = (X_12h[have].sort_index()
                    .reset_index()
                    .rename(columns={X_12h.index.name or X_12h.columns[0]: 'ts'}))
    g['ts'] = pd.to_datetime(g['ts'], errors='coerce')

    out = pd.merge_asof(
        left=targets.sort_values('date'),
        right=g.sort_values('ts'),
        left_on='date', right_on='ts',
        direction='nearest', tolerance=tol
    ).drop(columns=['ts'])

    if rename:
        out = out.rename(columns=rename)
    # ffill small gaps on the canonical names we produced
    canon = ['Ethylene','Ethane','Propylene','Propane','n-Butane','i-Butane']
    keep = [c for c in canon if c in out.columns]
    if keep:
        out[keep] = out[keep].ffill()
    return out  # contains 'date' + canonical gas columns present


def _alias_C11_plus(df: pd.DataFrame) -> pd.DataFrame:
    # If C11+ Foo missing but C11 Foo exists, create the + alias
    out = df.copy()
    for fam in ['n-Paraffin','i-Paraffin','Naphthene','Olefin','Aromatic']:
        src = f"C11 {fam}"
        tgt = f"C11+ {fam}"
        if src in out.columns and tgt not in out.columns:
            out[tgt] = pd.to_numeric(out[src], errors='coerce')
    return out

def _unify_gas_columns_for_srto(df: pd.DataFrame, keep_canon_only: bool = True) -> pd.DataFrame:
    # prefer *_gas when present but ensure canonical exists for SRTO
    out = df.copy()
    canon = ['Ethylene','Ethane','Propylene','Propane','n-Butane','i-Butane']
    for c in canon:
        cg = f"{c}_gas"
        if (c not in out.columns) and (cg in out.columns):
            out[c] = pd.to_numeric(out[cg], errors='coerce')
        elif (c in out.columns) and (cg in out.columns):
            # fill canonical from *_gas when *_gas has values
            a = pd.to_numeric(out[c], errors='coerce')
            b = pd.to_numeric(out[cg], errors='coerce')
            out[c] = b.where(b.notna(), a)
    if keep_canon_only:
        # drop *_gas after mirroring
        drop = [f"{c}_gas" for c in canon if f"{c}_gas" in out.columns]
        if drop:
            out = out.drop(columns=drop)
    return out

def _sanitize_comp_for_srto(merged_lims: pd.DataFrame) -> pd.DataFrame:
    ml = merged_lims.copy()
    if 'date' in ml.columns:
        ml['date'] = pd.to_datetime(ml['date'], errors='coerce')
        ml = ml.sort_values('date').set_index('date')

    # composition blocks for SRTO
    pona_fam  = ['Paraffins','Olefins','Naphthenes','Aromatics','n-Paraffin','i-Paraffin']
    canon_gas = ['Ethylene','Ethane','Propylene','Propane','n-Butane','i-Butane']
    pat = re.compile(r'^C(4|5|6|7|8|9|10|11\+?)\s+(n-?Paraffin|i-?Paraffin|Olefin|Naphthene|Aromatic)$', re.I)
    per_carbon = [c for c in ml.columns if pat.match(str(c))]

    cols = [c for c in pona_fam + canon_gas + per_carbon if c in ml.columns]
    if cols:
        ml[cols] = ml[cols].apply(pd.to_numeric, errors='coerce')
        # limited ffill/bfill to bridge short outages (no look-ahead beyond limit)
        ml[cols] = ml[cols].ffill(limit=SHORT_FILL_LIMIT).bfill(limit=SHORT_FILL_LIMIT)
        # remaining NA → 0.0 so SPYIN float(...) never crashes
        ml[cols] = ml[cols].fillna(0.0)

    # keep index AND have a 'date' column for any code expecting it
    ml['date'] = ml.index
    return ml.reset_index(drop=True)


# =========================
# Load & build
# =========================
if __name__ == "__main__":
    # Set output base
    main.set_out_dir_base(OUT_DIR)

    # Ensure intermediate dir
    INTER_DIR.mkdir(parents=True, exist_ok=True)
    pi_csv = INTER_DIR / DOWNLOADED_CSV

    # 0) (Optional) Ensure PI CSV with end_date = now
    if DOWNLOAD_PI:
        cfg = DownloadConfig(
            pi_server="172.17.21.117",
            auth_mode=AuthenticationMode.WINDOWS_AUTHENTICATION,
            pi_username="", pi_password="", pi_domain=None,
            tz="Asia/Seoul",
            start_date=(START_STR or "2024-01-01 00:00"),
            end_date=(END_STR),
            interval="1m", chunk_days=7, recorded=False,
            sheet_name="python_import", column_name="tags",
            input_dir=str(INPUT_DIR),
            tags_excel=str(INPUT_DIR / "EFOM_input_data_tag_list.xlsx"),
            out_csv=str(pi_csv), out_parquet="",
            incremental=True,
            # new flags (just stored; our wrapper will override end_date below)
            override_end_with_now=True, safety_lag_minutes=2, align_end_to_interval=True,
        )
        _override_end_date_now(cfg)
        ensure_pi_download(cfg)  # writes/updates pi_firstrow.csv (de-duped)

    # 1) Load frames via DataPipeline (prioritizes downloaded_csv)
    paths = DataPaths(
        input_dir=INPUT_DIR,
        inter_dir=INTER_DIR,
        downloaded_csv=DOWNLOADED_CSV,
        input_excel="EFOM_input_data_tag_list.xlsx",
        prod_excel="1. 생산량 Data_'23.07~'25.05_R1_송부용.xlsx",
        furn_excel="2. Furnace Data_'23.07~'25.05_R0.xlsx",
        nap_excel="Nap Feed 조성분석값.xlsx",
        gas_excel="Gas Feed 조성분석값.xlsx",
        recycle_excel="6. 에탄 및 프로판 데이터.xlsx",
        price_csv="price.csv",
        util_excel="#1ECU 유틸리티사용량일별데이터.xlsx",
        fresh_excel="7. Gas Furnace Feed Data_'23.07~'25.05_r2.xlsx",
        prod_pkl="df_production_v4.pkl",
        furn_pkl="furnace.pkl",
        nap_pkl="df_feed_naptha.pkl",
        gas_pkl="df_feed_gas.pkl",
        fresh_pkl="df_feed_fresh_v3.pkl",
        rec_pkl="df_recycle.pkl",
        prod_header=2, furn_header=2, nap_header=1, gas_header=1, rec_header=4, fresh_header=3,
    )

    cfg = ResampleConfig(hour_freq='h', win12_freq='12h', win12_offset='9h')

    feature_rename = {
        'Naph': 'Naphtha_chamber1', 'T-DAO': 'T-DAO_chamber1', 'DS': 'DS_chamber1',
        'RCOT Ave.': 'RCOT_chamber1', 'Excess O2': "Excess O2_chamber1",
        'Naph.1': 'Naphtha_chamber2', 'T-DAO.1': 'T-DAO_chamber2','DS.1': 'DS_chamber2',
        'RCOT Ave..1': 'RCOT_chamber2', 'Excess O2.1': "Excess O2_chamber2",
        'Naph.2': 'Naphtha_chamber3', 'T-DAO.2': 'T-DAO_chamber3','DS.2': 'DS_chamber3',
        'RCOT Ave..2': 'RCOT_chamber3', 'Excess O2.2': "Excess O2_chamber3",
        'Naph.3': 'Naphtha_chamber4', 'GAS': 'Gas Feed_chamber4','DS.3': 'DS_chamber4',
        'RCOT Ave..3': 'RCOT_chamber4', 'Excess O2.3': "Excess O2_chamber4",
        'Naph.4': 'Naphtha_chamber5', 'GAS.1': 'Gas Feed_chamber5','DS.4': 'DS_chamber5',
        'RCOT Ave..4': 'RCOT_chamber5', 'Excess O2.4': "Excess O2_chamber5",
        'Naph.5': 'Naphtha_chamber6', 'GAS.2': 'Gas Feed_chamber6','DS.5': 'DS_chamber6',
        'RCOT Ave..5': 'RCOT_chamber6', 'Excess O2.5': "Excess O2_chamber6",
    }
    target_rename  = { 'Unnamed: 36':'steam','ECU F/G':'fuel_gas','ECU Elec..1':'electricity' }

    dp = DataPipeline(paths, cfg).run(feature_rename, target_rename)
    art = dp.artifacts()

    X_12h = art['X_12h']
    Y_12h = art['Y_12h']
    prices_df = art['price_df']
    if REPLACE_PRICE_ZEROS:
        prices_df = prices_df.replace(0, pd.NA).ffill()
    prices_df = prices_df.loc[~prices_df.index.duplicated(keep='first')]

    print("Loaded:", X_12h.shape, Y_12h.shape, prices_df.shape)

    # 2) Build merged_lims from PI minute CSV
    pi_df = pd.read_csv(pi_csv)
    pi_df['timestamp'] = pd.to_datetime(pi_df['timestamp'], errors='coerce')
    pi_df = pi_df.sort_values('timestamp').dropna(subset=['timestamp'])

    # Select 07:00/19:00 targets and map *_gas from X_12h
    targets = _ampm_targets_from_X_range(X_12h, times=("07:00","19:00"))
    gas_map = _build_gas_from_X(X_12h, targets, tol=pd.Timedelta(hours=3))

    # Density conversions & tidy PONA (call existing normalizer in main)
    density_df = pd.read_excel(INPUT_DIR / "density_table.xlsx")
    # Use existing builder then normalize/clean names
    lims_full = main.build_merged_lims_full(
        pi_df, X_12h,
        density_df=density_df,
        bulk_density_col="M10L41004_Density",
        tolerance_days=7,
        enforce_100=False
    )
    # Clean column names: remove prefix + (wt%) → "C{N} Family"
    renames, drops = {}, []
    for c in lims_full.columns:
        m = re.match(r'^M10L41004_C(?P<C>\d+\+?)\s+(?P<FAM>[^()]+)\(wt%\)$', c)
        if m:
            renames[c] = f"C{m.group('C')} {m.group('FAM')}"
            continue
        m2 = re.match(r'^M10L41004_(?P<FAM>Paraffins|Olefins|Naphthenes|Aromatics|n-Paraffin|i-Paraffin)\(wt%\)$', c)
        if m2:
            fam = m2.group('FAM')
            if fam in ('Paraffins','Olefins','Naphthenes','Aromatics'):
                drops.append(c)
            else:
                renames[c] = fam

    lims = lims_full.drop(columns=drops).rename(columns=renames).copy()
    # Ensure we have a 'date' column & sort
    lims = _ensure_date_col(lims).sort_values('date').reset_index(drop=True)

    # # Merge in gas_map (canonical names) by 'date', then ffill
    # lims = lims.merge(gas_map, on='date', how='left').sort_values('date').reset_index(drop=True)
    # for c in ['Ethylene','Ethane','Propylene','Propane','n-Butane','i-Butane']:
    #     if c in lims.columns:
    #         lims[c] = pd.to_numeric(lims[c], errors='coerce')
    # lims[['Ethylene','Ethane','Propylene','Propane','n-Butane','i-Butane']] = \
    #     lims[['Ethylene','Ethane','Propylene','Propane','n-Butane','i-Butane']].ffill()

    # Merge in gas_map (canonical names) by 'date'
    canon = ['Ethylene','Ethane','Propylene','Propane','n-Butane','i-Butane']
    print("gas_map cols present:", [c for c in gas_map.columns if c in canon])


    lims = lims.merge(gas_map, on='date', how='left').sort_values('date').reset_index(drop=True)

    # Coerce numerics for whichever canonical gas columns we actually have
    for c in canon:
        if c in lims.columns:
            lims[c] = pd.to_numeric(lims[c], errors='coerce')

    # Forward-fill only the columns that are present
    present = [c for c in canon if c in lims.columns]
    if present:
        lims[present] = lims[present].ffill()

    # Ensure missing canonical gas columns exist (NaN) so later backfills can fill them
    for c in canon:
        if c not in lims.columns:
            lims[c] = pd.NA



    # 2b) Merge “merged_lims2” (Excel) for short outages and alias C11 →
    merged_lims2 = load_feed_data(
        nap_path=paths.input_dir / "복사본 (2024-25) ECU 투입 납사 세부성상-wt%.xlsx",
        gas_path=paths.input_dir / "Gas Feed 조성분석값.xlsx", header=1
    )
    merged_lims2['date'] = pd.to_datetime(merged_lims2['date'], errors='coerce')
    merged_lims2 = merged_lims2.dropna(subset=['date']).sort_values('date')
    # zero-rows → NaN → ffill/bfill
    gcan = [c for c in ['Ethylene','Ethane','Propylene','Propane','n-Butane','i-Butane'] if c in merged_lims2.columns]
    if gcan:
        zr = (merged_lims2[gcan].sum(axis=1) == 0)
        merged_lims2.loc[zr, gcan] = np.nan
        merged_lims2[gcan] = merged_lims2[gcan].ffill().bfill()
    # alias C11 → C11+
    merged_lims2 = _alias_C11_plus(merged_lims2)

    # as-of backward (≤ 1 day) mapping from merged_lims2 to lims by 'date'
    lims = _ensure_date_col(lims)
    merged_lims2 = _ensure_date_col(merged_lims2)
    lims = lims.sort_values('date').reset_index(drop=True)
    merged_lims2 = merged_lims2.sort_values('date').reset_index(drop=True)

    pona_fam = ['Paraffins','n-Paraffin','i-Paraffin','Olefins','Naphthenes','Aromatics']
    pat = re.compile(r'^C(4|5|6|7|8|9|10|11\+|12\+)\s+(n-?Paraffin|i-?Paraffin|Olefin|Naphthene|Aromatic)$', re.I)
    per_carbon_cols = [c for c in merged_lims2.columns if pat.match(str(c))]
    src_cols = [c for c in (pona_fam + gcan + per_carbon_cols) if c in merged_lims2.columns]

    if src_cols:
        ml2_map = pd.merge_asof(
            left=lims[['date']].sort_values('date'),
            right=merged_lims2[['date'] + src_cols].sort_values('date'),
            left_on='date', right_on='date',
            direction='backward', tolerance=pd.Timedelta(days=1)
        )
        for c in src_cols:
            if c not in lims.columns:
                lims[c] = pd.NA
            lims[c] = pd.to_numeric(lims[c], errors='coerce')
            lims[c] = lims[c].where(lims[c].notna(), pd.to_numeric(ml2_map[c], errors='coerce'))

        fill_cols = [c for c in (pona_fam + ['Ethylene','Ethane','Propylene','Propane','n-Butane','i-Butane'] + per_carbon_cols) if c in lims.columns]
        lims[fill_cols] = lims[fill_cols].ffill(limit=SHORT_FILL_LIMIT).bfill(limit=SHORT_FILL_LIMIT)

    # mirror canonical → *_gas for any downstream that still references *_gas
    for c in ['Ethylene','Ethane','Propylene','Propane','n-Butane','i-Butane']:
        cg = f'{c}_gas'
        if c in lims.columns and cg not in lims.columns:
            lims[cg] = lims[c]

    # keep canonical gas for SRTO, drop *_gas (we already mirrored)
    lims = _unify_gas_columns_for_srto(lims, keep_canon_only=True)

    # sanitize comps for SRTO (numeric, limited fills, no NAType)
    merged_lims = _sanitize_comp_for_srto(lims)

    # 3) SRTO + Spyro memo
    sel_spy7 = [(p if Path(p).is_absolute() else (SRTO_DLL / p)) for p in (SPY7S or [])]
    srto_config  = SRTOConfig(SRTO_DLL, sel_spy7, component_index, MW)
    sweep_config = RCOTSweepConfig(rcot_min=790.0, rcot_max=900.0, rcot_step=2.0,
                                   chunk_size=10, n_jobs=6, save_checkpoints=True)

    canonical_gas = ['Ethylene','Ethane','Propylene','Propane','n-Butane','i-Butane']
    gas_cols = [c for c in canonical_gas if c in X_12h.columns] or canonical_gas
    feed_config  = FeedConfig(gas_components=gas_cols)
    pipeline = SRTOPipeline(srto_config, sweep_config, feed_config)

    _SHORT_TO_SRTO = {
        'Ethylene':'Ethylene','Propylene':'Propylene','MixedC4':'MixedC4','RPG':'RPG',
        'Ethane':'Ethane','Propane':'Propane',
        'Fuel_Gas':'Fuel_Gas','Fuel Gas':'Fuel_Gas','FG':'Fuel_Gas','FuelGas':'Fuel_Gas',
        'Tail Gas':'Tail_Gas', 'Tail_Gas':'Tail_Gas'
    }

    class _SpyroMemo:
        def __init__(self, fn, key_cols=None, decimals=4, maxsize=200000):
            self.fn = fn; self.key_cols = tuple(key_cols) if key_cols is not None else None
            self.dec = decimals; self.cache = {}; self.maxsize = maxsize
        def _select_cols(self, row):
            if self.key_cols is None:
                return tuple(c for c in row.index if c.startswith('RCOT') or c.startswith('Naphtha_chamber') or c.startswith('Gas Feed_chamber'))
            return self.key_cols
        def _to_num(self, x):
            try: v = float(x)
            except Exception: v = 0.0
            if v != v: v = 0.0
            return round(v, self.dec)
        def _sig(self, row, short_key):
            cols = self._select_cols(row)
            vals = tuple(self._to_num(row.get(c, 0.0)) for c in cols)
            return (short_key, cols, vals)
        def __call__(self, row, short_key, ctx=None):
            k = self._sig(row, short_key); v = self.cache.get(k)
            if v is not None: return v
            out = self.fn(row, short_key, ctx)
            if len(self.cache) < self.maxsize: self.cache[k] = out
            return out

    def _make_spyro_fn(pipeline, merged_lims):
        def _spyro_row(row_like: pd.Series, short_key: str, ctx=None) -> float:
            ts = getattr(row_like, 'name', None)
            if ts is None:
                return 0.0
            # 'merged_lims' has tz-naive index in 'date' column and sanitized values
            try:
                comp_row = merged_lims.loc[merged_lims['date'] <= ts].iloc[-1]
            except Exception:
                comp_row = merged_lims.iloc[0]
            spot = pipeline.predict_spot_plant(row_like, comp_row, feed_thr=0.1)
            if spot.get('status') != 'ok':
                return 0.0
            key = _SHORT_TO_SRTO.get(short_key, short_key)
            return float(spot['totals_tph'].get(key, 0.0))
        return _SpyroMemo(_spyro_row)

    spyro_fn = _make_spyro_fn(pipeline, merged_lims)
    print("SRTO + Spyro ready. Gas cols for SRTO:", gas_cols)

    # 4) PICK WINDOW AND RUN
    if MODE == 'online':
        now_local = pd.Timestamp.now(tz='Asia/Seoul')

        # Pick the last *completed* 09:00/21:00 stamp
        target = last_complete_12h_stamp(now_local)

        # Ensure this stamp actually exists in X_12h; if not, fall back to the latest <= target
        if target not in X_12h.index:
            idx = X_12h.index[X_12h.index <= target]
            if len(idx) == 0:
                raise RuntimeError(f"No X_12h stamps <= {target} to run online.")
            target = idx.max()

        # IMPORTANT: run exactly on that stamp (do NOT normalize to midnight)
        latest = target
        assert latest.hour in (9, 21), f"Got {latest}, expected 09:00 or 21:00"

        start = latest
        end   = latest
        online_opts = dict(online_latest_only=True)
    else:
        start = pd.to_datetime(START_STR)
        end   = (pd.to_datetime(END_STR) if END_STR else None)
        online_opts = {}

    # Correct actuation log path (no double /online)
    act_hook = (main.default_actuation_logger_factory(OUT_DIR) if MODE == 'online' else None)

    main.run_production(
        X_12h=X_12h, Y_12h=Y_12h, merged_lims=merged_lims, pipeline=pipeline,
        prices_df=prices_df, total_spyro_yield_for_now=spyro_fn,
        start=start, end=end, mode=MODE,
        closed_loop_opts=dict(
            apply_timing='next_day',
            hold_policy='hold_until_next',
            ml_train_mode='historical',
            gp_train_mode='historical',
            cache_tag=('' if MODE=='historical' else '_sim'),
            **online_opts,
        ),
        act_hook=act_hook,
    )

    # =========================
    # ML prediction backtest (90D window ending at a chosen date)
    # =========================
    TARGET_COLS = [c for c in main.TARGET_COLS if c in Y_12h.columns]
    MIN_TR_ROWS = main.MIN_TR_ROWS

    end_eval = latest   # from the online section
    start_eval = end_eval - pd.Timedelta(days=90)

    preds, metrics = ml_prediction_check(
        X_12h=X_12h,
        Y_12h=Y_12h,
        start=start_eval,
        end=end_eval,
        lookback=pd.Timedelta("180D"),
        target_cols=TARGET_COLS,
        min_tr_rows=MIN_TR_ROWS,
        cache_tag="_eval"  # keep a separate cache
    )

    print("\n=== ML metrics (90D ending", end_eval, ") ===")
    print(metrics)
    

    # save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_excel(OUT_DIR / "metrics.xlsx")
    preds.to_csv(OUT_DIR / "preds_eval.csv")
    # Build MAPE dict from the just-computed metrics
    mape_for_push = _extract_mape_for_push(metrics)

    # # Push (only in online if you prefer, or in all modes)
    # try:
    #     ts_push, rec = _build_push_record(OUT_DIR, X_12h, Y_12h, mape=mape_for_push)
    #     with PIPublisher(PIServerConfig(server="172.17.21.117", tz="Asia/Seoul")) as pub:
    #         _safe_publish_record(pub, rec, ts_push)
    #     print(f"[OK] Published EFOM setpoints & MAPEs at {ts_push}")
    # except Exception as e:
    #     print(f"[WARN] PI publish skipped: {e}")

    # pick the correct OUT folder for this mode (usually OUT_DIR / "online")
    MODE_DIR = OUT_DIR / ("online" if MODE == "online" else MODE)

    # mape_for_push from your metrics (or {})
    mape_for_push = _extract_mape_for_push(metrics) if 'metrics' in locals() else {}

    try:
        publish_with_gate(MODE_DIR, X_12h, Y_12h, mape=mape_for_push, done_value=0.0)
        print('success')
    except Exception as e:
        print(f"[WARN] gated publish skipped: {e}")

