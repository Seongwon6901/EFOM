# -*- coding: utf-8 -*-
"""
EFOM Runner (thin CLI)

- Loads project data via your DataPipeline (DataPaths/ResampleConfig)
- Optionally merges LIMS by AM/PM rule
- Builds SRTO pipeline + memoized Spyro yield function
- Calls src.main.run_production(...) for historical / closed_loop / online

Keep business logic in src/main.py. This file should stay small and orchestration-only.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

# Core orchestrator
from src import main

# Your data loading primitives
from src.data_loading import DataPaths, ResampleConfig, DataPipeline, load_feed_data
# SRTO / SPYRO plumbing
from src.srto_pipeline import SRTOConfig, RCOTSweepConfig, FeedConfig, SRTOPipeline
from src.srto_components import component_index, MW


# -----------------------------
# CLI
# -----------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="EFOM Runner")

    # Run mode
    ap.add_argument('--mode', choices=['historical','closed_loop','online'], default='historical')
    ap.add_argument('--start', type=str, required=True)
    ap.add_argument('--end',   type=str, default=None)
    ap.add_argument('--out-dir', type=str, default='prod_out')

    # Policy & caches
    ap.add_argument('--apply-timing', choices=['next_day','next_stamp'], default='next_day')
    ap.add_argument('--hold-policy',  choices=['hold_until_next'], default='hold_until_next')
    ap.add_argument('--cache-tag', type=str, default='')
    ap.add_argument('--ml-train-mode', choices=['historical','simulated'], default='historical')
    ap.add_argument('--gp-train-mode', choices=['historical','simulated'], default='historical')

    # DataPaths-style inputs (defaults match your repo)
    ap.add_argument('--input-dir', type=str, default='input')
    ap.add_argument('--inter-dir', type=str, default='intermediate')
    ap.add_argument('--prod-excel', type=str, default="1. 생산량 Data_'23.07~'25.05_R1_송부용.xlsx")
    ap.add_argument('--furn-excel', type=str, default="2. Furnace Data_'23.07~'25.05_R0.xlsx")
    ap.add_argument('--nap-excel',  type=str, default='Nap Feed 조성분석값.xlsx')
    ap.add_argument('--gas-excel',  type=str, default='Gas Feed 조성분석값.xlsx')
    ap.add_argument('--recycle-excel', type=str, default='6. 에탄 및 프로판 데이터.xlsx')
    ap.add_argument('--price-csv',  type=str, default='price.csv')
    ap.add_argument('--util-excel', type=str, default='#1ECU 유틸리티사용량일별데이터.xlsx')
    ap.add_argument('--fresh-excel', type=str, default="7. Gas Furnace Feed Data_'23.07~'25.05_r2.xlsx")

    # Optional PKL caches + headers
    ap.add_argument('--prod-pkl', type=str, default='df_production_v4.pkl')
    ap.add_argument('--furn-pkl', type=str, default='furnace.pkl')
    ap.add_argument('--nap-pkl',  type=str, default='df_feed_naptha.pkl')
    ap.add_argument('--gas-pkl',  type=str, default='df_feed_gas.pkl')
    ap.add_argument('--fresh-pkl', type=str, default='df_feed_fresh_v3.pkl')
    ap.add_argument('--rec-pkl',  type=str, default='df_recycle.pkl')
    ap.add_argument('--prod-header', type=int, default=2)
    ap.add_argument('--furn-header', type=int, default=2)
    ap.add_argument('--nap-header',  type=int, default=1)
    ap.add_argument('--gas-header',  type=int, default=1)
    ap.add_argument('--rec-header',  type=int, default=4)
    ap.add_argument('--fresh-header',type=int, default=3)

    # LIMS merge inputs
    ap.add_argument('--nap-lims', type=str, default='복사본 (2024-25) ECU 투입 납사 세부성상-wt%.xlsx')
    ap.add_argument('--gas-lims', type=str, default='Gas Feed 조성분석값.xlsx')
    ap.add_argument('--lims-header', type=int, default=1)
    ap.add_argument('--lims-trim-head', type=int, default=4, help='drop first N rows after cleaning')
    ap.add_argument('--am-pm-lims', action='store_true', help='apply AM/PM rule for daily composition join')

    # SRTO / SPYRO
    ap.add_argument('--srto-dll', type=str, default=r"C:\\Program Files\\Pyrotec\\SRTO")
    ap.add_argument('--spy7', type=str, nargs='*', default=[
        r"01. GF_HYBRID MODE_SRTO7_NAPH.SPY7",
        r"04. LF_NAPH MODE_SRTO7.SPY7",
        r"07. GF_GAS MODE_SRTO7.SPY7",
    ], help='List of spy7 filenames under --srto-dll (or absolute paths)')

    return ap.parse_args()


# -----------------------------
# Loading helpers
# -----------------------------

def _build_paths(args: argparse.Namespace) -> DataPaths:
    return DataPaths(
        input_dir=Path(args.input_dir),
        inter_dir=Path(args.inter_dir),
        prod_excel=args.prod_excel,
        furn_excel=args.furn_excel,
        nap_excel=args.nap_excel,
        gas_excel=args.gas_excel,
        recycle_excel=args.recycle_excel,
        price_csv=args.price_csv,
        util_excel=args.util_excel,
        fresh_excel=args.fresh_excel,
        prod_pkl=args.prod_pkl,
        furn_pkl=args.furn_pkl,
        nap_pkl=args.nap_pkl,
        gas_pkl=args.gas_pkl,
        fresh_pkl=args.fresh_pkl,
        rec_pkl=args.rec_pkl,
        prod_header=args.prod_header,
        furn_header=args.furn_header,
        nap_header=args.nap_header,
        gas_header=args.gas_header,
        rec_header=args.rec_header,
        fresh_header=args.fresh_header,
    )


def _feature_target_renames() -> tuple[Dict[str, str], Dict[str, str]]:
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
    return feature_rename, target_rename


def _load_frames(args: argparse.Namespace):
    paths = _build_paths(args)
    cfg = ResampleConfig(hour_freq='h', win12_freq='12h', win12_offset='9h')
    feature_rename, target_rename = _feature_target_renames()

    dp = DataPipeline(paths, cfg).run(feature_rename, target_rename)
    art = dp.artifacts()

    X_12h = art['X_12h']
    Y_12h = art['Y_12h']
    prices_df = art['price_df']
    return paths, X_12h, Y_12h, prices_df


def _load_lims(paths: DataPaths, args: argparse.Namespace) -> pd.DataFrame:
    merged_lims = load_feed_data(
        nap_path=paths.input_dir / args.nap_lims,
        gas_path=paths.input_dir / args.gas_lims,
        header=args.lims_header,
    )
    merged_lims['date'] = pd.to_datetime(merged_lims['date'], errors='coerce')
    merged_lims = merged_lims.dropna(subset=['date']).sort_values('date')

    gas_cols = [c for c in ['Ethylene','Ethane','Propylene','Propane','n-Butane','i-Butane'] if c in merged_lims.columns]
    zr = (merged_lims[gas_cols].sum(axis=1) == 0)
    merged_lims.loc[zr, gas_cols] = np.nan
    merged_lims[gas_cols] = merged_lims[gas_cols].ffill().bfill()
    if args.lims_trim_head > 0 and len(merged_lims) > args.lims_trim_head:
        merged_lims = merged_lims.iloc[args.lims_trim_head:]
    return merged_lims


def _merge_lims_into_X(X_12h: pd.DataFrame, merged_lims: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    # composition columns possibly present in LIMS
    pona_cols = ['Paraffins','Olefins','Naphthenes','Aromatics']
    gas_cols = [c for c in ['Ethylene','Ethane','Propylene','Propane','n-Butane','i-Butane'] if c in merged_lims.columns]
    keep = [c for c in (pona_cols + gas_cols) if c in merged_lims.columns]

    if not args.am_pm_lims or not keep:
        return X_12h.loc[:, ~X_12h.columns.duplicated()].copy()

    x = X_12h.copy()
    ts = x.index.to_series()
    lims_date = ts.dt.normalize()
    lims_date = lims_date.where(ts.dt.hour >= 12, lims_date - pd.Timedelta(days=1))
    x = x.assign(lims_date=lims_date.values)

    m = merged_lims.copy()
    m['lims_date'] = pd.to_datetime(m['date'], errors='coerce').dt.normalize()
    m_daily = m.sort_values('date').groupby('lims_date', as_index=False).last()

    m_sel = m_daily[['lims_date'] + keep]
    xr = x.reset_index()
    idx_name = xr.columns[0]
    merged = xr.merge(m_sel, on='lims_date', how='left').set_index(idx_name)
    merged.index.name = X_12h.index.name

    # asof fallback for NaNs
    if merged[keep].isna().any().any():
        mr = m.sort_values('date')[['date'] + keep].rename(columns={'date': 'lims_ts'})
        xr_ts = xr.copy(); xr_ts['ts'] = pd.to_datetime(xr_ts[idx_name])
        asof = pd.merge_asof(xr_ts.sort_values('ts'), mr.sort_values('lims_ts'), left_on='ts', right_on='lims_ts', direction='backward').set_index(idx_name)
        for c in keep:
            merged[c] = merged[c].fillna(asof[c]).infer_objects(copy=False)

    # Drop any pre-existing *_gas columns before we rename new ones
    for g in gas_cols:
        gcol = f"{g}_gas"
        if gcol in X_12h.columns:
            X_12h = X_12h.drop(columns=[gcol])

    X_out = X_12h.join(merged[keep])
    gas_rename = {c: f"{c}_gas" for c in gas_cols if c in X_out.columns}
    X_out = X_out.rename(columns=gas_rename)
    return X_out.loc[:, ~X_out.columns.duplicated()].copy()


# -----------------------------
# SRTO / Spyro yield
# -----------------------------

def _build_srto(args: argparse.Namespace, gas_cols: List[str]) -> SRTOPipeline:
    dll_folder = Path(args.srto_dll)
    # Allow absolute spy7 paths or join to dll_folder
    sel_spy7 = []
    for s in (args.spy7 or []):
        p = Path(s)
        sel_spy7.append(p if p.is_absolute() else (dll_folder / s))
    srto_config  = SRTOConfig(dll_folder, sel_spy7, component_index, MW)
    sweep_config = RCOTSweepConfig(rcot_min=790.0, rcot_max=900.0, rcot_step=2.0,
                                   chunk_size=10, n_jobs=6, save_checkpoints=True)
    feed_config  = FeedConfig(gas_components=gas_cols)
    return SRTOPipeline(srto_config, sweep_config, feed_config)


class _SpyroMemo:
    def __init__(self, fn, key_cols=None, decimals=4, maxsize=200000):
        self.fn = fn
        self.key_cols = tuple(key_cols) if key_cols is not None else None
        self.dec = decimals
        self.cache: Dict[Any, float] = {}
        self.maxsize = maxsize

    def _select_cols(self, row: pd.Series):
        if self.key_cols is None:
            return tuple(c for c in row.index if c.startswith('RCOT') or c.startswith('Naphtha_chamber') or c.startswith('Gas Feed_chamber'))
        return self.key_cols

    def _to_num(self, x):
        try:
            v = float(x)
        except Exception:
            v = 0.0
        if v != v:  # NaN
            v = 0.0
        return round(v, self.dec)

    def _sig(self, row: pd.Series, short_key: str):
        cols = self._select_cols(row)
        vals = tuple(self._to_num(row.get(c, 0.0)) for c in cols)
        return (short_key, cols, vals)

    def __call__(self, row: pd.Series, short_key: str, ctx=None):
        k = self._sig(row, short_key)
        v = self.cache.get(k)
        if v is not None: return v
        out = self.fn(row, short_key, ctx)
        if len(self.cache) < self.maxsize:
            self.cache[k] = out
        return out


_SHORT_TO_SRTO = {
    'Ethylene':'Ethylene','Propylene':'Propylene','MixedC4':'MixedC4','RPG':'RPG',
    'Ethane':'Ethane','Propane':'Propane',
    'Fuel_Gas':'Fuel_Gas','Fuel Gas':'Fuel_Gas','FG':'Fuel_Gas','FuelGas':'Fuel_Gas',
    'Tail Gas':'Tail_Gas', 'Tail_Gas' :'Tail_Gas'
}


def _make_spyro_fn(pipeline: SRTOPipeline, merged_lims: pd.DataFrame) -> _SpyroMemo:
    def _spyro_row(row_like: pd.Series, short_key: str, ctx=None) -> float:
        ts = getattr(row_like, 'name', None)
        if ts is None:
            return 0.0
        sel = merged_lims.loc[merged_lims['date'] <= ts]
        comp_row = (sel.iloc[-1] if not sel.empty else merged_lims.iloc[0])
        spot = pipeline.predict_spot_plant(row_like, comp_row, feed_thr=0.1)
        if spot.get('status') != 'ok':
            return 0.0
        key = _SHORT_TO_SRTO.get(short_key, short_key)
        return float(spot['totals_tph'].get(key, 0.0))
    return _SpyroMemo(_spyro_row)


# -----------------------------
# Main
# -----------------------------

def main_cli():
    args = _parse_args()

    # Configure output base
    main.set_out_dir_base(args.out_dir)

    # Load frames
    paths, X_12h, Y_12h, prices_df = _load_frames(args)

    # LIMS
    merged_lims = _load_lims(paths, args)

    # Merge LIMS → X_12h if requested
    X_12h = _merge_lims_into_X(X_12h, merged_lims, args)

    # SRTO pipeline (needs gas component list from LIMS)
    gas_cols = [c for c in ['Ethylene','Ethane','Propylene','Propane','n-Butane','i-Butane'] if c in merged_lims.columns]
    pipeline = _build_srto(args, gas_cols)

    # Spyro memo
    spyro_fn = _make_spyro_fn(pipeline, merged_lims)

    # Parse dates
    start = pd.to_datetime(args.start)
    end   = pd.to_datetime(args.end) if args.end else None

    # Online actuation hook (CSV logger by default)
    act_hook = main.default_actuation_logger_factory(Path(args.out_dir) / 'online') if args.mode == 'online' else None

    # Run
    main.run_production(
        X_12h=X_12h, Y_12h=Y_12h, merged_lims=merged_lims, pipeline=pipeline,
        prices_df=prices_df, total_spyro_yield_for_now=spyro_fn,
        start=start, end=end,
        mode=args.mode,
        closed_loop_opts=dict(
            apply_timing=args.apply_timing,
            hold_policy=args.hold_policy,
            ml_train_mode=args.ml_train_mode,
            gp_train_mode=args.gp_train_mode,
            cache_tag=(args.cache_tag or ('' if args.mode=='historical' else '_sim')),
        ),
        act_hook=act_hook,
    )


if __name__ == '__main__':
    main_cli()
# EOF