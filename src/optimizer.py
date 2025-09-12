# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple, Optional, Any, Callable

import numpy as np
import pandas as pd

from scipy.optimize import minimize

try:
    from scipy.optimize import differential_evolution
    _HAS_DE = True
except Exception:
    _HAS_DE = False

# Prefer your src package; fall back to local import if used outside the repo
try:
    import src.gp_residuals as gpmod
except Exception:
    import gp_residuals as gpmod


# ─────────────────────────────────────────────────────────────────────────────
# 0) Fuel-gas constants (Excel-delta style)
# ─────────────────────────────────────────────────────────────────────────────

# @dataclass(frozen=True)
# class FuelGasConstants:
#     # cp values (weighted average), kcal/ton/K
#     cp_wavg_kcal_per_ton_K: float = 411.488209
#     # Heats of cracking (kcal/ton of product)
#     dH_eth_kcal_per_ton: float = 1_080_970.0     # Ethylene
#     dH_prop_kcal_per_ton: float =   673_409.0    # Propylene
#     dH_fg_kcal_per_ton: float =   926_147.0      # Proxy for FG delta (methane basis)
#     # Fuel gas heat content (kcal/ton of fuel gas)
#     fuel_gas_heat_content_kcal_per_ton: float = 15_294_088.0
#     # Reference RCOT for baseline
#     rcot_ref_C: float = 840.0

@dataclass
class FuelGasConstants:
    # RCOT references
    rc_ref_naph: float = 840.0
    rc_ref_gas:  float = 880.0
    kcal_per_nm3: float = 9000.0

    # Excel “slope” model (kcal/h per °C) – fill from your calibration
    slope_kcal_perC_13:   float = 0.0
    slope_kcal_perC_N456: float = 0.0
    slope_kcal_perC_G456: float = 0.0
    slope_kcal_per_tDS:   float = 0.0

    # Full energy-budget model used by delta_fg_excel
    cp_wavg_kcal_per_ton_K: float = 411.488209
    dH_eth_kcal_per_ton:    float = 1_080_970.0
    dH_prop_kcal_per_ton:   float =   673_409.0
    dH_fg_kcal_per_ton:     float =   926_147.0
    fuel_gas_heat_content_kcal_per_ton: float = 15_294_088.0


# ─────────────────────────────────────────────────────────────────────────────
# 1) Prices (monthly; forward-fill)
# ─────────────────────────────────────────────────────────────────────────────

def _active_rc_means(row: pd.Series,
                     flow_thr: float = 1.0,
                     rc_min: float = 800.0,
                     weight_by_feed: bool = True,
                     sum_ds_active_only: bool = True) -> tuple[float, float, float, float]:
    """
    Returns per-side RCOT means limited to *active* coils and optional DS sum over actives:
      rc13   = mean RCOT for chambers 1–3 (naphtha side) when Naphtha_flow>thr & RCOT>=rc_min
      rcn456 = mean RCOT for naphtha 4–6 when Naphtha_flow>thr & RCOT>=rc_min
      rcg456 = mean RCOT for gas 4–6 when Gas_flow>thr & RCOT>=rc_min
      ds_sum = sum of DS_chamber* over active coils (or over all if sum_ds_active_only=False)
    """
    def _collect(indices, rc_key_fmt, flow_key_fmt):
        pairs = []
        for i in indices:
            rc = float(row.get(rc_key_fmt.format(i=i), np.nan))
            f  = float(row.get(flow_key_fmt.format(i=i), 0.0))
            if np.isfinite(rc) and rc >= rc_min and f > flow_thr:
                pairs.append((rc, f))
        if not pairs:
            return np.nan
        if weight_by_feed:
            w = sum(f for _, f in pairs)
            return (sum(rc*f for rc, f in pairs) / w) if w > 0 else np.nan
        else:
            return float(np.mean([rc for rc, _ in pairs]))

    # 1–3: naphtha side
    rc13 = _collect(range(1,4), rc_key_fmt='RCOT_chamber{i}', flow_key_fmt='Naphtha_chamber{i}')
    # 4–6 naphtha side
    rcn456 = _collect(range(4,7), rc_key_fmt='RCOT_naphtha_chamber{i}', flow_key_fmt='Naphtha_chamber{i}')
    # 4–6 gas side
    rcg456 = _collect(range(4,7), rc_key_fmt='RCOT_gas_chamber{i}', flow_key_fmt='Gas Feed_chamber{i}')

    # DS sum (active only or all)
    if sum_ds_active_only:
        ds = 0.0
        # 1–3 naphtha actives
        for i in range(1,4):
            rc = float(row.get(f'RCOT_chamber{i}', np.nan))
            f  = float(row.get(f'Naphtha_chamber{i}', 0.0))
            if np.isfinite(rc) and rc >= rc_min and f > flow_thr:
                ds += float(row.get(f'DS_chamber{i}', 0.0))
        # 4–6 naphtha actives
        for i in range(4,7):
            rc = float(row.get(f'RCOT_naphtha_chamber{i}', np.nan))
            f  = float(row.get(f'Naphtha_chamber{i}', 0.0))
            if np.isfinite(rc) and rc >= rc_min and f > flow_thr:
                ds += float(row.get(f'DS_chamber{i}', 0.0))
        # 4–6 gas actives
        for i in range(4,7):
            rc = float(row.get(f'RCOT_gas_chamber{i}', np.nan))
            f  = float(row.get(f'Gas Feed_chamber{i}', 0.0))
            if np.isfinite(rc) and rc >= rc_min and f > flow_thr:
                ds += float(row.get(f'DS_chamber{i}', 0.0))
    else:
        ds = sum(float(row.get(f'DS_chamber{i}', 0.0)) for i in range(1,7))

    return float(rc13) if np.isfinite(rc13) else np.nan, \
           float(rcn456) if np.isfinite(rcn456) else np.nan, \
           float(rcg456) if np.isfinite(rcg456) else np.nan, \
           float(ds)

# (Optional) keep a thin shim with the old name so your existing calls still work
def _rc_means(row: pd.Series):
    return _active_rc_means(row, flow_thr=1.0, rc_min=800.0, weight_by_feed=True, sum_ds_active_only=True)
# class PriceProvider:
#     """
#     Get prices by month and item. `prices_df` must have columns: ['item','unit','date','value']
#     or be already wide with monthly index and canonical column names.
#     Canonical columns expected (wide mode): 'Ethylene','Propylene','Mixed C4','RPG',
#     'PN','Gas Feed','Fuel Gas','Steam','Electricity'
#     """
#     CANON = ['Ethylene','Propylene','Mixed C4','RPG','PN','Gas Feed','Fuel Gas','Steam','Electricity']

#     def __init__(self, df: pd.DataFrame):
#         if {'item','unit','date','value'}.issubset(df.columns):
#             dd = df.copy()
#             dd['item'] = dd['item'].astype(str).str.strip()
#             dd['unit'] = dd['unit'].astype(str).str.upper()
#             dd['date'] = pd.to_datetime(dd['date']).dt.to_period('M').dt.to_timestamp()

#             mapper = {'TPG (BTX 가치) (T/D)': 'RPG', 'TPG': 'RPG', 'Pygas': 'RPG'}
#             dd['item_std']  = dd['item'].map(mapper).fillna(dd['item'])

#             def conv(row):
#                 item, val, unit = row['item_std'], float(row['value']), row['unit']
#                 if item == 'Fuel Gas' and 'MMKCAL' in unit:
#                     return val * (15_294_088.0 / 1e6)  # $/MMkcal → $/t
#                 return val
#             dd['price_std'] = dd.apply(conv, axis=1)

#             wide = dd.pivot_table(index='date', columns='item_std', values='price_std', aggfunc='last')
#             for c in self.CANON:
#                 if c not in wide.columns: wide[c] = np.nan
#             wide = wide[self.CANON].sort_index().ffill()
#             self.table = wide
#         else:
#             wide = df.copy()
#             if 'date' in wide.columns:
#                 wide = wide.set_index(pd.to_datetime(wide['date']))
#             wide.index = pd.to_datetime(wide.index).to_period('M').to_timestamp()
#             wide = wide.sort_index().groupby(level=0).last().ffill()
#             for c in self.CANON:
#                 if c not in wide.columns: wide[c] = np.nan
#             self.table = wide[self.CANON]

#     def get(self, ts: pd.Timestamp | str, item: str, default: float = 0.0) -> float:
#         d = pd.Timestamp(ts).to_period('M').to_timestamp()
#         if item not in self.table.columns:
#             return float(default)
#         if d in self.table.index and pd.notnull(self.table.at[d, item]):
#             return float(self.table.at[d, item])
#         s = self.table.loc[:d, item].dropna()
#         return float(s.iloc[-1]) if len(s) else float(default)

def _to_month_start_index(idx):
    import pandas as pd
    if isinstance(idx, pd.PeriodIndex):
        # already monthly? force to M then to timestamp (month-start)
        return idx.asfreq('M').to_timestamp()
    dt = pd.to_datetime(idx, errors='coerce')
    return dt.to_period('M').to_timestamp()


class PriceProvider:
    def __init__(self, df):
        import numpy as np
        wide = df.copy()

        # allow either a wide DF with a datetime/period index OR a 'date' column
        if 'date' in wide.columns:
            wide = wide.set_index(pd.to_datetime(wide['date']))
            wide = wide.drop(columns=['date'])

        # ✅ robust monthly index
        wide.index = _to_month_start_index(wide.index)
        wide = wide.sort_index().groupby(level=0).last()

        # ✅ accept common aliases coming from your pipeline
        aliases = {
            'MixedC4':   'Mixed C4',
            'LPG':       'LPG',
            'Fuel_Gas':  'Fuel Gas',
            'Tail_Gas':  'Tail Gas',
        }
        wide = wide.rename(columns={k: v for k, v in aliases.items() if k in wide.columns})

        # ensure canonical columns exist (extras are fine; they’ll be kept)
        canon = ['Ethylene','Propylene','Mixed C4','RPG','PN',
                 'Gas Feed','Fuel Gas','Steam','Electricity',
                 'Hydrogen','Tail Gas','MX Offgas']
        for c in canon:
            if c not in wide.columns:
                wide[c] = np.nan

        # monthly ffill, keep canonical first for predictable order, then extras
        ordered = [c for c in canon if c in wide.columns]
        extras  = [c for c in wide.columns if c not in ordered]
        self.df = wide[ordered + extras].ffill()

    # --- add this method ---
    def get(self, ts, item, default=0.0) -> float:
        """
        Return price for `item` at timestamp ts.
        - ts is mapped to month-start and as-of backfilled
        - item aliases are normalized (MixedC4->Mixed C4, Fuel_Gas->Fuel Gas, Tail_Gas->Tail Gas, LPG->Gas Feed)
        """
        import numpy as np
        import pandas as pd

        # normalize item aliases to your canonical column names
        aliases = {
            'MixedC4': 'Mixed C4',
            'Fuel_Gas': 'Fuel Gas',
            'Tail_Gas': 'Tail Gas',
            'LPG Fresh': 'LPG',
            'Offgas Fresh': 'MX Offgas',
        }
        col = aliases.get(item, item)
        if col not in self.df.columns:
            return float(default)

        # month-start index + asof backfill
        ts = pd.Timestamp(ts)
        # m = _to_month_start_index(ts)[0]  # month start for ts
        # month start for ts (Timestamp, not index)
        m = pd.Timestamp(ts).to_period("M").start_time     # ← clean & explicit

        df = self.df

        # exact hit
        if m in df.index:
            val = df.at[m, col]
        else:
            # as-of backfill to previous available month
            i = df.index.searchsorted(m, side="right") - 1
            val = np.nan if i < 0 else df.iloc[i][col]

        return float(val) if pd.notna(val) else float(default)

    # (optional) convenience alias
    __call__ = get



# ─────────────────────────────────────────────────────────────────────────────
# 2) Geometry helper (no dependency on gp module)
# ─────────────────────────────────────────────────────────────────────────────

def geometry_from_row(row: pd.Series) -> str:
    """
    Infer geometry from active feeds in the row:
      - GF_HYB_NAPH: both naphtha (1–6) and gas (4–6) present
      - LF_NAPH: naphtha present, gas absent
      - GF_GAS: gas present, naphtha absent
      - NONE: neither present
    """
    n = sum(float(row.get(f'Naphtha_chamber{i}', 0.0)) for i in range(1,7))
    g = sum(float(row.get(f'Gas Feed_chamber{i}', 0.0)) for i in (4,5,6))
    if n > 0 and g > 0: return 'GF_HYB_NAPH'
    if n > 0:           return 'LF_NAPH'
    if g > 0:           return 'GF_GAS'
    return 'NONE'


# ─────────────────────────────────────────────────────────────────────────────
# 3) Excel-delta FG helpers & function (in this module)
# ─────────────────────────────────────────────────────────────────────────────

def _rc_pairs_for_effective_rcot(row: pd.Series) -> Tuple[float, float, float]:
    """
    Returns (rc_eff, naphtha_feed_tph, gas_feed_tph) for the current row.
    rc_eff is a feed-weighted mean of naphtha/gas RCOTs that are actually flowing.
    """
    # feeds
    n_feed = sum(float(row.get(f'Naphtha_chamber{ch}', 0.0)) for ch in range(1,7))
    g_feed = sum(float(row.get(f'Gas Feed_chamber{ch}', 0.0)) for ch in (4,5,6))

    # naphtha RCOTs: 1–3 and naphtha 4–6
    rc_n_list = []
    for ch in (1,2,3):
        rc = row.get(f'RCOT_chamber{ch}', np.nan)
        f  = float(row.get(f'Naphtha_chamber{ch}', 0.0))
        if f > 0 and pd.notnull(rc): rc_n_list.append(float(rc))
    for ch in (4,5,6):
        rc = row.get(f'RCOT_naphtha_chamber{ch}', np.nan)
        f  = float(row.get(f'Naphtha_chamber{ch}', 0.0))
        if f > 0 and pd.notnull(rc): rc_n_list.append(float(rc))
    rc_n = float(np.nanmean(rc_n_list)) if rc_n_list else np.nan

    # gas RCOTs: gas 4–6
    rc_g_list = []
    for ch in (4,5,6):
        rc = row.get(f'RCOT_gas_chamber{ch}', np.nan)
        f  = float(row.get(f'Gas Feed_chamber{ch}', 0.0))
        if f > 0 and pd.notnull(rc): rc_g_list.append(float(rc))
    rc_g = float(np.nanmean(rc_g_list)) if rc_g_list else np.nan

    # effective RCOT (feed-weighted) over sides present
    pairs = []
    if np.isfinite(rc_n) and n_feed > 0: pairs.append((rc_n, n_feed))
    if np.isfinite(rc_g) and g_feed > 0: pairs.append((rc_g, g_feed))
    if pairs:
        rc_eff = sum(rc*w for rc, w in pairs) / sum(w for _, w in pairs)
    else:
        rc_eff = np.nan
    return rc_eff, n_feed, g_feed


def _row_with_rcot_override(row: pd.Series, rcot_n=None, rcot_g=None) -> pd.Series:
    """
    Force all naphtha-side RCOTs (1–6) to rcot_n and all gas-side RCOTs (4–6) to rcot_g.
    """
    r = row.copy()
    if rcot_n is not None:
        for ch in (1,2,3):
            k = f'RCOT_chamber{ch}'
            if k in r: r[k] = float(rcot_n)
        for ch in (4,5,6):
            k = f'RCOT_naphtha_chamber{ch}'
            if k in r: r[k] = float(rcot_n)
    if rcot_g is not None:
        for ch in (4,5,6):
            k = f'RCOT_gas_chamber{ch}'
            if k in r: r[k] = float(rcot_g)
    return r


def _fuel_gas_abs(row: pd.Series, total_spyro_yield_for_now, spyro_ctx) -> float:
    """Fuel gas absolute tph via SPYRO callable."""
    for sk in ('Fuel_Gas', 'Fuel Gas', 'FG', 'FuelGas'):
        try:
            return float(total_spyro_yield_for_now(row, sk, ctx=spyro_ctx))
        except Exception:
            continue
    return 0.0


def delta_fg_excel(row_feats: pd.Series,
                   corrected: Dict[str, float],
                   total_spyro_yield_for_now,
                   spyro_ctx,
                   const: FuelGasConstants = FuelGasConstants(),
                   base_at_current_rcot: bool = False,   # NEW
                   debug: bool = False                   # NEW
                   ) -> float:
    """
    Excel-style ΔFG (tph).
    - Baseline yields can be at rc_ref (classic) or at current RCOT (toggle via base_at_current_rcot).
    - The RCOT firing term ALWAYS references rc_ref to retain RCOT sensitivity.
    - Includes robust guards to prevent numeric explosions.
    """
    # 1) Feed (tons per hour)
    if 'feed_qty' in row_feats:
        D = float(row_feats.get('feed_qty', 0.0))
    else:
        D  = sum(float(row_feats.get(f'Naphtha_chamber{ch}', 0.0)) for ch in range(1,7))
        D += sum(float(row_feats.get(f'Gas Feed_chamber{ch}', 0.0)) for ch in (4,5,6))
    if not np.isfinite(D) or D <= 0:
        return 0.0

    # 2) Current absolute product t/h (must be t/h, not kg/h!)
    E_abs = float(corrected.get('Ethylene_prod_t+1', 0.0))
    P_abs = float(corrected.get('Propylene_prod_t+1', 0.0))
    FG_abs= _fuel_gas_abs(row_feats, total_spyro_yield_for_now, spyro_ctx)

    # Sanity guard: absurd magnitudes → bail
    if any(abs(v) > 1e5 for v in (E_abs, P_abs, FG_abs, D)):
        if debug:
            print({'BAD_SCALE': True, 'E_abs':E_abs, 'P_abs':P_abs, 'FG_abs':FG_abs, 'D':D})
        return 0.0

    ratio_E  = E_abs / D
    ratio_P  = P_abs / D
    ratio_FG = FG_abs / D

    # 3) Effective RCOT (feed-weighted)
    rc_eff, _, _ = _rc_pairs_for_effective_rcot(row_feats)
    rc_eff = float(rc_eff) if np.isfinite(rc_eff) else const.rcot_ref_C

    # 4) Physics-only baselines at either reference or current RCOT
    rc_base_for_yields = rc_eff if base_at_current_rcot else const.rcot_ref_C
    r_base = _row_with_rcot_override(row_feats, rcot_n=rc_base_for_yields, rcot_g=rc_base_for_yields)

    base_E_abs  = float(total_spyro_yield_for_now(r_base, 'Ethylene',  ctx=spyro_ctx))
    base_P_abs  = float(total_spyro_yield_for_now(r_base, 'Propylene', ctx=spyro_ctx))
    base_FG_abs = _fuel_gas_abs(r_base, total_spyro_yield_for_now, spyro_ctx)

    base_E  = base_E_abs  / D
    base_P  = base_P_abs  / D
    base_FG = base_FG_abs / D

    # 5) Energy budget (kcal per ton of *feed*) → FG tph
    #    IMPORTANT: rc_term is ALWAYS vs reference, to keep RCOT sensitivity
    rc_term = (rc_eff - const.rcot_ref_C) * const.cp_wavg_kcal_per_ton_K
    term_E  = (ratio_E  - base_E ) * const.dH_eth_kcal_per_ton
    term_P  = (ratio_P  - base_P ) * const.dH_prop_kcal_per_ton
    term_FG = (ratio_FG - base_FG) * const.dH_fg_kcal_per_ton

    if debug:
        print({'D':D, 'rc_eff':rc_eff, 'rc_base_for_yields':rc_base_for_yields,
               'ratio_E':ratio_E, 'base_E':base_E, 'ratio_P':ratio_P, 'base_P':base_P,
               'ratio_FG':ratio_FG, 'base_FG':base_FG,
               'rc_term':rc_term, 'term_E':term_E, 'term_P':term_P, 'term_FG':term_FG})

    HHV = const.fuel_gas_heat_content_kcal_per_ton
    if not np.isfinite(HHV) or HHV <= 0:
        return 0.0

    # Guard rails for numeric sanity
    S = rc_term + term_E + term_P + term_FG
    if not np.isfinite(S) or abs(S) > 1e10:  # kcal/ton of feed
        if debug:
            print({'BAD_S': True, 'S':S})
        return 0.0

    delta_fg_tph = (S / HHV) * D
    # Clip ridiculous magnitudes (ΔFG should be << 1 t/h most of the time)
    if not np.isfinite(delta_fg_tph) or abs(delta_fg_tph) > 1e3:
        if debug:
            print({'BAD_DFG': True, 'delta_fg_tph':delta_fg_tph})
        return 0.0

    return float(delta_fg_tph)


# ─────────────────────────────────────────────────────────────────────────────
# 4) Margin builders
# ─────────────────────────────────────────────────────────────────────────────

def sum_naphtha_feed_tph(x_row: pd.Series) -> float:
    return float(sum(x_row.get(f'Naphtha_chamber{ch}', 0.0) for ch in range(1,7)))

def sum_gas_feed_tph(x_row: pd.Series) -> float:
    return float(sum(x_row.get(f'Gas Feed_chamber{ch}', 0.0) for ch in (4,5,6)))

def make_margin_fn(price_provider: PriceProvider, fg_cost_mode: str = 'none',
                   fg_constants: FuelGasConstants | None = None,
                   util_models: Optional[Dict[str, Any]] = None,
                   util_feature_cols: Optional[Sequence[str]] = None):
    """
    Returns margin(ts, x_row, yields_abs) in $/h for optimizer:
      Revenue(products incl. H2 & Tail Gas) − Fresh feed cost − (optional learned utils).
    The Excel-ΔFG fuel-gas consumption term is applied in make_margin_fn_excel_delta(...).
    """
    def _predict_utils(row: pd.Series):
        if not util_models or not util_feature_cols:
            return 0.0, 0.0, 0.0
        X = pd.DataFrame([{c: row.get(c, 0.0) for c in util_feature_cols}])[util_feature_cols]
        s = float(util_models.get('steam', None).predict(X)[0])       if 'steam' in util_models else 0.0
        f = float(util_models.get('fuel_gas', None).predict(X)[0])    if 'fuel_gas' in util_models else 0.0
        e = float(util_models.get('electricity', None).predict(X)[0]) if 'electricity' in util_models else 0.0
        return s, f, e

    # yield-key -> price item
    price_item_map = {
        'Ethylene': 'Ethylene',
        'Propylene': 'Propylene',
        'MixedC4': 'Mixed C4',
        'RPG': 'RPG',
        'Hydrogen': 'Hydrogen',
        'Tail_Gas': 'Tail Gas',
    }

    def margin(ts: pd.Timestamp, x_row: pd.Series, yields_abs: Dict[str, float]) -> float:
        # --- revenue (corrected absolute tph) ---
        rev = 0.0
        for key, item in price_item_map.items():
            yk = f'{key}_prod_t+1'         # e.g., MixedC4_prod_t+1, Tail_Gas_prod_t+1
            if yk in yields_abs:
                rev += float(yields_abs.get(yk, 0.0)) * price_provider.get(ts, item, 0.0)

        # --- fresh feed costs (prefer explicit fresh feeds; fallback to Gas Feed) ---
        p_PN  = price_provider.get(ts, 'PN', 0.0)
        p_LPG = price_provider.get(ts, 'LPG Fresh', np.nan)
        p_OFF = price_provider.get(ts, 'Offgas Fresh', np.nan)
        p_GF  = price_provider.get(ts, 'Gas Feed', 0.0)

        naph = sum(float(x_row.get(f'Naphtha_chamber{i}', 0.0)) for i in range(1, 7))
        gas_chambers = sum(float(x_row.get(f'Gas Feed_chamber{i}', 0.0)) for i in (4, 5, 6))
        lpg  = float(x_row.get('FreshFeed_C3 LPG',    0.0))
        off  = float(x_row.get('FreshFeed_MX Offgas', 0.0))

        if (lpg > 0 or off > 0) and (not np.isnan(p_LPG) or not np.isnan(p_OFF)):
            feed_cost  = naph * p_PN
            feed_cost += lpg * (p_LPG if not np.isnan(p_LPG) else p_GF)
            feed_cost += off * (p_OFF if not np.isnan(p_OFF) else p_GF)
        else:
            feed_cost = naph * p_PN + gas_chambers * p_GF

        # --- optional learned utilities (NOT Excel-ΔFG; that’s added in make_margin_fn_excel_delta) ---
        steam_tph, fg_cons_tph, elec_MWh = _predict_utils(x_row) if fg_cost_mode == 'ml' else (0.0, 0.0, 0.0)
        util_cost = (steam_tph * price_provider.get(ts, 'Steam', 0.0)
                   + fg_cons_tph * price_provider.get(ts, 'Fuel Gas', 0.0)
                   + elec_MWh   * price_provider.get(ts, 'Electricity', 0.0))

        return float(rev - feed_cost - util_cost)

    return margin


# def make_margin_fn_excel_delta(
#     price_provider: PriceProvider,
#     *,
#     total_spyro_yield_for_now: Any,
#     spyro_ctx: Any,
#     fg_constants: FuelGasConstants,
#     delta_fg_fn: Optional[Callable[[pd.Series, Dict[str, float], Any, Any, FuelGasConstants], float]] = None,
#     util_models: Optional[Dict[str, Any]] = None,
#     util_feature_cols: Optional[Sequence[str]] = None,
# ):
#     """
#     Wraps make_margin_fn(...) and subtracts Excel-delta FG consumption each time:
#       margin(ts, x_row, yields_abs) = base(ts,x_row,yields_abs) - ΔFG(x_row,yields_abs)*price(FG)
#     """
#     base = make_margin_fn(price_provider, fg_cost_mode='none',
#                           util_models=util_models, util_feature_cols=util_feature_cols)
#     if delta_fg_fn is None:
#         delta_fg_fn = delta_fg_excel  # use this module's implementation by default

#     def margin(ts: pd.Timestamp, x_row: pd.Series, yields_abs: Dict[str, float]) -> float:
#         p_FG = price_provider.get(ts, 'Fuel Gas', 0.0)
#         fg_cons_tph = float(delta_fg_fn(x_row, yields_abs, total_spyro_yield_for_now, spyro_ctx, fg_constants))
#         return base(ts, x_row, yields_abs) - fg_cons_tph * p_FG

#     return margin
# from typing import Optional, Dict, Any, Sequence, Callable, Tuple

# def make_margin_fn_excel_delta(price_provider, total_spyro_yield_for_now, spyro_ctx=None,
#                                fg_constants: FuelGasConstants = FuelGasConstants(),
#                                use_reference: bool = False):
#     """
#     Returns margin(ts, row_like, yields_abs_by_target) -> $/h
#     - Computes ΔFG cost vs the first call at (ts) (baseline) OR vs fixed refs (840/880) if use_reference=True.
#     """
#     baseline = {}  # keyed by ts

#     def _fg_delta_cost(ts, row_like):
#         rc13, rcn, rcg, ds = _rc_means(row_like)
#         # sensible fallbacks
#         rc13_0, rcn_0, rcg_0, ds_0 = (fg_constants.rc_ref_naph,
#                                     fg_constants.rc_ref_naph,
#                                     fg_constants.rc_ref_gas,
#                                     0.0)
#         if not use_reference:
#             if ts not in baseline:
#                 baseline[ts] = _rc_means(row_like)
#             rc13_0, rcn_0, rcg_0, ds_0 = baseline[ts]

#         rc13  = rc13  if np.isfinite(rc13)  else fg_constants.rc_ref_naph
#         rcn   = rcn   if np.isfinite(rcn)   else fg_constants.rc_ref_naph
#         rcg   = rcg   if np.isfinite(rcg)   else fg_constants.rc_ref_gas
#         rc13_0= rc13_0 if np.isfinite(rc13_0) else fg_constants.rc_ref_naph
#         rcn_0 = rcn_0 if np.isfinite(rcn_0)  else fg_constants.rc_ref_naph
#         rcg_0 = rcg_0 if np.isfinite(rcg_0)  else fg_constants.rc_ref_gas
#         # Excel-delta in kcal/h (or convert from Nm3/h if your slopes are Nm3/h/°C)
#         d_kcal_h  = fg_constants.slope_kcal_perC_13   * (rc13 - rc13_0)
#         d_kcal_h += fg_constants.slope_kcal_perC_N456 * (rcn  - rcn_0)
#         d_kcal_h += fg_constants.slope_kcal_perC_G456 * (rcg  - rcg_0)
#         d_kcal_h += fg_constants.slope_kcal_per_tDS   * (ds   - ds_0)

#         d_nm3_h = d_kcal_h / max(fg_constants.kcal_per_nm3, 1.0)
#         fg_price = price_provider.get(ts, 'Tail Gas', 0.0)
#         return d_nm3_h * fg_price

#     def margin(ts, row_like, yields_abs_by_target):
#         # revenue from corrected absolute t/h
#         rev = 0.0
#         for k,v in yields_abs_by_target.items():
#             # map 'Ethylene_prod_t+1' -> 'Ethylene' price key etc.
#             p = k.replace('_prod_t+1','').replace('_',' ')
#             price_key = {'MixedC4':'Mixed C4','Tail Gas':'Tail Gas'}.get(p, p)
#             rev += float(v) * price_provider.get(ts, price_key, 0.0)

#         # feed costs (simple)
#         pn = price_provider.get(ts, 'PN', 0.0)
#         pg = price_provider.get(ts, 'Gas Feed', 0.0)
#         naph = sum(float(row_like.get(f'Naphtha_chamber{i}', 0.0)) for i in range(1,7))
#         gas  = sum(float(row_like.get(f'Gas Feed_chamber{i}', 0.0)) for i in (4,5,6))
#         feed_cost = naph*pn + gas*pg

#         # FG delta cost (Excel delta)
#         fg_cost_delta = _fg_delta_cost(ts, row_like)
#         return float(rev - feed_cost - fg_cost_delta)

#     return margin

# def make_margin_fn_excel_delta(
#     price_provider: PriceProvider,
#     *,
#     total_spyro_yield_for_now: Any,
#     spyro_ctx: Any,
#     fg_constants: FuelGasConstants,
#     delta_fg_fn: Optional[Callable[[pd.Series, Dict[str, float], Any, Any, FuelGasConstants], float]] = None,
#     delta_fg_kwargs: Optional[Dict[str, Any]] = None,   # NEW
#     util_models: Optional[Dict[str, Any]] = None,
#     util_feature_cols: Optional[Sequence[str]] = None,
# ):
#     """
#     Wraps make_margin_fn(...) and subtracts Excel-delta FG consumption:
#       margin(ts, x_row, yields_abs) = base(ts,x_row,yields_abs) - ΔFG(x_row,yields_abs)*price(FG)
#     """
#     base = make_margin_fn(price_provider, fg_cost_mode='none',
#                           util_models=util_models, util_feature_cols=util_feature_cols)
#     if delta_fg_fn is None:
#         delta_fg_fn = delta_fg_excel  # use this module's implementation by default

#     def margin(ts: pd.Timestamp, x_row: pd.Series, yields_abs: Dict[str, float]) -> float:
#         p_FG = price_provider.get(ts, 'Fuel Gas', 0.0)
#         fg_cons_tph = float(delta_fg_fn(
#             x_row, yields_abs, total_spyro_yield_for_now, spyro_ctx, fg_constants,
#             **(delta_fg_kwargs or {})  # NEW
#         ))
#         return base(ts, x_row, yields_abs) - fg_cons_tph * p_FG

#     return margin

# def make_margin_fn_excel_delta(price_provider, total_spyro_yield_for_now, spyro_ctx=None,
#                                fg_constants: FuelGasConstants = FuelGasConstants(),
#                                use_reference: bool = False):
#     """
#     margin(ts, row_like, yields_abs_by_target) -> $/h
#     Uses energy-budget ΔFG (tph) anchored to a *fixed* baseline (per ts):
#       - If use_reference=True → baseline per-side RCOTs = (840 naph, 880 gas)
#       - Else → baseline per-side RCOTs captured from the *first* row at that ts.
#     Also adds recycle credit (Ethane+Propane × LPG), matching your audit.
#     """
#     import numpy as np

#     # --- helpers ---
#     def _per_side_rc_eff(r):
#         # active-only, flow-weighted per-side RCOTs
#         def _mean(keys, fkeys):
#             pairs = []
#             for k, fk in zip(keys, fkeys):
#                 rc = float(r.get(k, np.nan)); f = float(r.get(fk, 0.0))
#                 if np.isfinite(rc) and rc >= 800 and f > 1.0:
#                     pairs.append((rc, f))
#             if not pairs: return np.nan
#             w = sum(f for _, f in pairs)
#             return sum(rc*f for rc, f in pairs)/w if w > 0 else np.nan

#         rc13   = _mean([f'RCOT_chamber{i}'           for i in (1,2,3)],
#                        [f'Naphtha_chamber{i}'        for i in (1,2,3)])
#         rcn456 = _mean([f'RCOT_naphtha_chamber{i}'   for i in (4,5,6)],
#                        [f'Naphtha_chamber{i}'        for i in (4,5,6)])
#         rcg456 = _mean([f'RCOT_gas_chamber{i}'       for i in (4,5,6)],
#                        [f'Gas Feed_chamber{i}'       for i in (4,5,6)])

#         rc_n_eff = np.nanmean([rc13, rcn456]) if (np.isfinite(rc13) or np.isfinite(rcn456)) else np.nan
#         rc_g_eff = rcg456

#         naph = sum(float(r.get(f'Naphtha_chamber{i}', 0.0))   for i in range(1,7))
#         gas  = sum(float(r.get(f'Gas Feed_chamber{i}', 0.0)) for i in (4,5,6))
#         D    = max(float(naph + gas), 1e-9)
#         wN, wG = naph / D, gas / D
#         return rc_n_eff, rc_g_eff, wN, wG, D

#     def _fg_abs(r):
#         for sk in ('Fuel_Gas','Fuel Gas','FG','FuelGas','Tail Gas','Tail_Gas'):
#             try: return float(total_spyro_yield_for_now(r, sk, ctx=spyro_ctx))
#             except Exception: pass
#         return 0.0

#     # --- baseline store (per ts) ---
#     base = {}  # ts -> dict(rc_n, rc_g, wN, wG, D)

#     def _ensure_baseline(ts, row_like):
#         if ts in base and not use_reference:
#             return
#         if use_reference:
#             rc_n0, rc_g0 = float(fg_constants.rc_ref_naph), float(fg_constants.rc_ref_gas)
#             # weights come from current row (could also store first-call weights)
#             _, _, wN, wG, D = _per_side_rc_eff(row_like)
#         else:
#             rc_n0, rc_g0, wN, wG, D = _per_side_rc_eff(row_like)
#             if not np.isfinite(rc_n0): rc_n0 = float(fg_constants.rc_ref_naph)
#             if not np.isfinite(rc_g0): rc_g0 = float(fg_constants.rc_ref_gas)
#         base[ts] = dict(rc_n=rc_n0, rc_g=rc_g0, wN=float(wN), wG=float(wG), D=float(D))

#     def _dfg_tph(ts, row_like, yields_abs):
#         _ensure_baseline(ts, row_like)
#         b = base[ts]

#         # current per-side RCOTs and weights
#         rc_n, rc_g, wN, wG, D = _per_side_rc_eff(row_like)
#         if not np.isfinite(rc_n): rc_n = float(fg_constants.rc_ref_naph)
#         if not np.isfinite(rc_g): rc_g = float(fg_constants.rc_ref_gas)

#         # current absolute ratios
#         E_abs = float(yields_abs.get('Ethylene_prod_t+1',  0.0))
#         P_abs = float(yields_abs.get('Propylene_prod_t+1', 0.0))
#         FG_abs= _fg_abs(row_like)
#         rE, rP, rFG = E_abs/D, P_abs/D, FG_abs/D

#         # baseline ratios at *fixed* baseline per-side RCOTs (override RCOTs to base)
#         def _override(r, rc_n=None, rc_g=None):
#             rr = r.copy()
#             if rc_n is not None:
#                 for ch in (1,2,3): rr[f'RCOT_chamber{ch}'] = float(rc_n)
#                 for ch in (4,5,6): rr[f'RCOT_naphtha_chamber{ch}'] = float(rc_n)
#             if rc_g is not None:
#                 for ch in (4,5,6): rr[f'RCOT_gas_chamber{ch}'] = float(rc_g)
#             return rr

#         r_base = _override(row_like, b['rc_n'], b['rc_g'])
#         base_E_abs  = float(total_spyro_yield_for_now(r_base, 'Ethylene',  ctx=spyro_ctx))
#         base_P_abs  = float(total_spyro_yield_for_now(r_base, 'Propylene', ctx=spyro_ctx))
#         base_FG_abs = _fg_abs(r_base)
#         bE, bP, bFG = base_E_abs/D, base_P_abs/D, base_FG_abs/D

#         # energy budget
#         Cpw = float(fg_constants.cp_wavg_kcal_per_ton_K)
#         HHV = float(fg_constants.fuel_gas_heat_content_kcal_per_ton)
#         rc_term = Cpw * (wN * (rc_n - b['rc_n']) + wG * (rc_g - b['rc_g']))
#         term_E  = (rE  - bE ) * float(fg_constants.dH_eth_kcal_per_ton)
#         term_P  = (rP  - bP ) * float(fg_constants.dH_prop_kcal_per_ton)
#         term_FG = (rFG - bFG) * float(fg_constants.dH_fg_kcal_per_ton)
#         S = rc_term + term_E + term_P + term_FG
#         return float((S / HHV) * D)

#     def margin(ts, row_like, yields_abs_by_target):
#         # revenue
#         rev = 0.0
#         price_map = {'Ethylene':'Ethylene','Propylene':'Propylene','MixedC4':'Mixed C4',
#                      'RPG':'RPG','Hydrogen':'Hydrogen','Tail_Gas':'Tail Gas'}
#         for p,item in price_map.items():
#             col = f'{p}_prod_t+1'
#             if col in yields_abs_by_target:
#                 rev += float(yields_abs_by_target[col]) * price_provider.get(ts, item, 0.0)

#         # feed cost (prefer explicit fresh feeds; else Gas Feed)
#         p_PN  = price_provider.get(ts, 'PN', 0.0)
#         p_LPG = price_provider.get(ts, 'LPG', float(price_provider.get(ts, 'Gas Feed', 0.0)))
#         p_OFF = price_provider.get(ts, 'MX Offgas', 0.0)
#         naph  = sum(float(row_like.get(f'Naphtha_chamber{i}', 0.0)) for i in range(1,7))
#         gasC  = sum(float(row_like.get(f'Gas Feed_chamber{i}', 0.0)) for i in (4,5,6))
#         fresh_lpg = float(row_like.get('FreshFeed_C3 LPG',    0.0))
#         fresh_off = float(row_like.get('FreshFeed_MX Offgas', 0.0))
#         if (fresh_lpg > 0 or fresh_off > 0):
#             feed_cost = naph*p_PN + fresh_lpg*p_LPG + fresh_off*p_OFF
#         else:
#             feed_cost = naph*p_PN + gasC*price_provider.get(ts, 'Gas Feed', 0.0)

#         # recycle credit (match audit)
#         rec_credit = (float(yields_abs_by_target.get('Ethane_prod_t+1',  0.0)) +
#                       float(yields_abs_by_target.get('Propane_prod_t+1', 0.0))) * p_LPG

#         # ΔFG × Tail Gas price using *fixed* baseline
#         dfg_tph = _dfg_tph(ts, row_like, yields_abs_by_target)
#         fg_cost = dfg_tph * price_provider.get(ts, 'Tail Gas', 0.0)

#         return float(rev - feed_cost + rec_credit - fg_cost)

#     return margin

# --- Drop-in replacement in optimizer.py (or run as a cell if you import optimizer live) ---

def make_margin_fn_excel_delta(price_provider, total_spyro_yield_for_now, spyro_ctx=None,
                               fg_constants: FuelGasConstants = FuelGasConstants(),
                               use_reference: bool = False):
    """
    margin(ts, row_like, yields_abs_by_target) -> $/h

    • Energy-budget ΔFG (tph) vs a fixed baseline per ts (captured on first call).
    • Baseline = per-side RCOTs (naph/gas). If use_reference=True → (840/880).
    • Includes recycle credit (Ethane+Propane × LPG) to match audit.
    """
    import numpy as np

    def _per_side_rc_eff(r):
        def _mean(keys, fkeys, rc_min=800.0, flow_thr=1.0):
            pairs = []
            for k, fk in zip(keys, fkeys):
                rc = float(r.get(k, np.nan)); f = float(r.get(fk, 0.0))
                if np.isfinite(rc) and rc >= rc_min and f > flow_thr:
                    pairs.append((rc, f))
            if not pairs: return np.nan
            w = sum(f for _, f in pairs)
            return sum(rc*f for rc,f in pairs)/w if w>0 else np.nan

        rc13   = _mean([f'RCOT_chamber{i}'         for i in (1,2,3)],
                       [f'Naphtha_chamber{i}'      for i in (1,2,3)])
        rcn456 = _mean([f'RCOT_naphtha_chamber{i}' for i in (4,5,6)],
                       [f'Naphtha_chamber{i}'      for i in (4,5,6)])
        rcg456 = _mean([f'RCOT_gas_chamber{i}'     for i in (4,5,6)],
                       [f'Gas Feed_chamber{i}'     for i in (4,5,6)])

        rc_n_eff = np.nanmean([rc13, rcn456]) if (np.isfinite(rc13) or np.isfinite(rcn456)) else np.nan
        rc_g_eff = rcg456

        naph = sum(float(r.get(f'Naphtha_chamber{i}', 0.0))  for i in range(1,7))
        gas  = sum(float(r.get(f'Gas Feed_chamber{i}', 0.0)) for i in (4,5,6))
        D    = max(float(naph + gas), 1e-9)
        wN, wG = naph / D, gas / D
        return rc_n_eff, rc_g_eff, wN, wG, D

    def _fg_abs(r):
        for sk in ('Fuel_Gas','Fuel Gas','FG','FuelGas','Tail Gas','Tail_Gas'):
            try: return float(total_spyro_yield_for_now(r, sk, ctx=spyro_ctx))
            except Exception: pass
        return 0.0

    def _override(r, rc_n=None, rc_g=None):
        rr = r.copy()
        if rc_n is not None:
            for ch in (1,2,3): rr[f'RCOT_chamber{ch}'] = float(rc_n)
            for ch in (4,5,6): rr[f'RCOT_naphtha_chamber{ch}'] = float(rc_n)
        if rc_g is not None:
            for ch in (4,5,6): rr[f'RCOT_gas_chamber{ch}'] = float(rc_g)
        return rr

    # per-ts baseline cache
    _base = {}  # ts -> dict(rc_n, rc_g)

    def _dfg_tph(ts, row_like, yields_abs):
        # capture baseline RCOTs once per ts
        if (ts not in _base) or use_reference:
            if use_reference:
                _base[ts] = dict(rc_n=float(fg_constants.rc_ref_naph), rc_g=float(fg_constants.rc_ref_gas))
            else:
                rc_n0, rc_g0, *_ = _per_side_rc_eff(row_like)
                if not np.isfinite(rc_n0): rc_n0 = float(fg_constants.rc_ref_naph)
                if not np.isfinite(rc_g0): rc_g0 = float(fg_constants.rc_ref_gas)
                _base[ts] = dict(rc_n=float(rc_n0), rc_g=float(rc_g0))
        b = _base[ts]

        rc_n, rc_g, wN, wG, D = _per_side_rc_eff(row_like)
        if not np.isfinite(rc_n): rc_n = float(fg_constants.rc_ref_naph)
        if not np.isfinite(rc_g): rc_g = float(fg_constants.rc_ref_gas)

        E_abs = float(yields_abs.get('Ethylene_prod_t+1',  0.0))
        P_abs = float(yields_abs.get('Propylene_prod_t+1', 0.0))
        FG_ab = _fg_abs(row_like)
        rE, rP, rFG = E_abs/D, P_abs/D, FG_ab/D

        r_base = _override(row_like, rc_n=b['rc_n'], rc_g=b['rc_g'])
        base_E_abs  = float(total_spyro_yield_for_now(r_base, 'Ethylene',  ctx=spyro_ctx))
        base_P_abs  = float(total_spyro_yield_for_now(r_base, 'Propylene', ctx=spyro_ctx))
        base_FG_abs = _fg_abs(r_base)
        bE, bP, bFG = base_E_abs/D, base_P_abs/D, base_FG_abs/D

        Cpw = float(fg_constants.cp_wavg_kcal_per_ton_K)
        HHV = float(fg_constants.fuel_gas_heat_content_kcal_per_ton)
        rc_term = Cpw * (wN * (rc_n - b['rc_n']) + wG * (rc_g - b['rc_g']))
        term_E  = (rE - bE)  * float(fg_constants.dH_eth_kcal_per_ton)
        term_P  = (rP - bP)  * float(fg_constants.dH_prop_kcal_per_ton)
        term_FG = (rFG - bFG)* float(fg_constants.dH_fg_kcal_per_ton)
        S = rc_term + term_E + term_P + term_FG
        return float((S/HHV) * D)

    def margin(ts, row_like, yields_abs_by_target):
        # revenue
        price_map = {'Ethylene':'Ethylene','Propylene':'Propylene','MixedC4':'Mixed C4',
                     'RPG':'RPG','Hydrogen':'Hydrogen','Tail_Gas':'Tail Gas'}
        rev = 0.0
        for p,item in price_map.items():
            col = f'{p}_prod_t+1'
            if col in yields_abs_by_target:
                rev += float(yields_abs_by_target[col]) * price_provider.get(ts, item, 0.0)

        # feed cost (prefer explicit fresh; else Gas Feed)
        p_PN  = price_provider.get(ts, 'PN', 0.0)
        p_LPG = price_provider.get(ts, 'LPG', float(price_provider.get(ts, 'Gas Feed', 0.0)))
        p_OFF = price_provider.get(ts, 'MX Offgas', 0.0)
        naph  = sum(float(row_like.get(f'Naphtha_chamber{i}', 0.0)) for i in range(1,7))
        gasC  = sum(float(row_like.get(f'Gas Feed_chamber{i}', 0.0)) for i in (4,5,6))
        fresh_lpg = float(row_like.get('FreshFeed_C3 LPG',    0.0))
        fresh_off = float(row_like.get('FreshFeed_MX Offgas', 0.0))
        if (fresh_lpg > 0 or fresh_off > 0):
            feed_cost = naph*p_PN + fresh_lpg*p_LPG + fresh_off*p_OFF
        else:
            feed_cost = naph*p_PN + gasC*price_provider.get(ts, 'Gas Feed', 0.0)

        # recycle credit
        rec_credit = (float(yields_abs_by_target.get('Ethane_prod_t+1',  0.0)) +
                      float(yields_abs_by_target.get('Propane_prod_t+1', 0.0))) * p_LPG

        # ΔFG (tph) × Fuel Gas price
        dfg_tph = _dfg_tph(ts, row_like, yields_abs_by_target)
        fg_cost = dfg_tph * price_provider.get(ts, 'Fuel Gas', 0.0)

        return float(rev - feed_cost + rec_credit - fg_cost)

    return margin


# ─────────────────────────────────────────────────────────────────────────────
# 5) Scalar-RCOT optimizer (ML-anchored curve)
# ─────────────────────────────────────────────────────────────────────────────

def optimize_rcot_scalar_anchored(
    *,
    gp: gpmod.GPResiduals,
    X_12h: pd.DataFrame,
    merged_lims: pd.DataFrame,
    pipeline,
    ml_cached,
    ts: pd.Timestamp,
    rcot_setter,                         # gpmod.rcot_setter_lf_naph / rcot_setter_gf_gas / rcot_setter_hybrid
    rc_bounds: Tuple[float, float],
    step: float = 0.5,
    price_provider: PriceProvider | None = None,
    margin_fn = None,                    # if given, overrides price_provider
    use_gp_delta: bool = True,
    alpha: float = 0.2,
) -> dict:
    """
    Sweep RCOT via ML-anchored curve, compute margin per grid point, and pick argmax.
    Returns: {'rc_opt','margin','curve'}
    """
    lo, hi = rc_bounds
    rc_grid = np.arange(lo, hi + 1e-9, step, dtype=float)

    curve, x_row_base = gpmod.anchored_curve_at_ts(
        gp=gp, X_12h=X_12h, merged_lims=merged_lims, pipeline=pipeline, ml=ml_cached,
        ts=ts, rcot_setter=rcot_setter, rc_grid=rc_grid, use_gp_delta=use_gp_delta, alpha=alpha
    )
    if curve.empty:
        raise RuntimeError("Anchored sweep returned empty curve.")

    if margin_fn is None:
        if price_provider is None:
            raise ValueError("Provide either price_provider or margin_fn.")
        margin_fn = make_margin_fn(price_provider=price_provider)

    m = np.zeros(len(curve), float)
    for i, r in curve.iterrows():
        rc = float(r['RCOT'])
        x_row = rcot_setter(x_row_base.copy(), rc)  # RCOT-aware candidate row
        yields_abs = {f'{p}_prod_t+1': float(r.get(f'{p}_CORR_tph', 0.0)) for p in gpmod.PRODUCTS}
        m[i] = margin_fn(ts, x_row, yields_abs)

    curve = curve.copy()
    curve['MARGIN'] = m
    i_best = int(curve['MARGIN'].idxmax())
    best = curve.loc[i_best]

    return {
        'rc_opt': float(best['RCOT']),
        'margin': float(best['MARGIN']),
        'curve': curve,
        'row_base': x_row_base,   # <-- add this line
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6) Multi-knob optimizer (per-chamber RCOTs) — optional deep dive
# ─────────────────────────────────────────────────────────────────────────────

def _active_rcot_vars(row: pd.Series,
                      naph_bounds: Tuple[float,float] = (810.0, 853.0),
                      gas_bounds: Tuple[float,float]  = (850.0, 890.0),
                      feed_thr: float = 0.1):
    names, bnds = [], []
    # 1–3 naphtha
    for ch in (1,2,3):
        if float(row.get(f'Naphtha_chamber{ch}', 0.0)) > feed_thr:
            names.append(f'RCOT_chamber{ch}'); bnds.append(naph_bounds)
    # 4–6 naphtha + gas sides
    for ch in (4,5,6):
        if float(row.get(f'Naphtha_chamber{ch}', 0.0)) > feed_thr:
            names.append(f'RCOT_naphtha_chamber{ch}'); bnds.append(naph_bounds)
        if float(row.get(f'Gas Feed_chamber{ch}', 0.0)) > feed_thr:
            names.append(f'RCOT_gas_chamber{ch}');     bnds.append(gas_bounds)
    return names, bnds

def _apply_rcots(row: pd.Series, names: Sequence[str], x: Sequence[float]) -> pd.Series:
    r = row.copy()
    for k, v in zip(names, x):
        r[k] = float(v)
    return r

def corrected_yields_for_row(
    row: pd.Series,
    gps: Dict[str, Any],                 # dict: target -> pipe.predict(Xgp)
    feature_cols_gp: Sequence[str],
    total_spyro_yield_for_now,           # callable(row, short_key, ctx=...)
    spyro_ctx=None,
    alpha_overrides: Optional[Dict[Tuple[str,str], float]] = None,
) -> Dict[str, float]:
    """
    Compute corrected absolute t/h for each target using raw SPYRO + GP μ.
    (This is the non-anchored path — good for multi-knob study/DE.)
    """
    n = sum(float(row.get(f'Naphtha_chamber{ch}', 0.0)) for ch in range(1,7))
    g = sum(float(row.get(f'Gas Feed_chamber{ch}', 0.0)) for ch in (4,5,6))
    den = n + g
    flags = dict(is_hybrid = 1 if (n>0 and g>0) else 0,
                 is_naphtha= 1 if (n>0 and g==0) else 0,
                 ratio_naphtha = (n/den) if den>0 else 0.0)

    feat = {c: row.get(c, 0.0) for c in feature_cols_gp}
    for k,v in flags.items():
        if k in feat: feat[k] = v
    Xgp = pd.DataFrame([feat])[feature_cols_gp].to_numpy(dtype=float)

    out = {}
    geom = 'GF_HYB_NAPH' if flags['is_hybrid'] else ('LF_NAPH' if flags['is_naphtha'] else 'GF_GAS')
    for t, pipe in gps.items():
        short = gpmod.SHORT_MAP.get(t, t) if hasattr(gpmod, 'SHORT_MAP') else t
        raw_abs = float(total_spyro_yield_for_now(row, short, ctx=spyro_ctx))
        mu = float(pipe.predict(Xgp)[0])
        if alpha_overrides and (geom, t) in alpha_overrides:
            mu *= float(alpha_overrides[(geom, t)])
        out[t] = max(0.0, raw_abs + mu)
    return out

def optimize_rcot_multi(
    *,
    row: pd.Series,
    gps: Dict[str, Any],
    feature_cols_gp: Sequence[str],
    total_spyro_yield_for_now,
    spyro_ctx=None,
    price_provider: PriceProvider | None = None,
    margin_fn = None,                     # if provided, overrides price_provider
    naph_bounds: Tuple[float,float]=(810.0, 853.0),
    gas_bounds: Tuple[float,float]=(850.0, 890.0),
    method: str = 'slsqp',                # 'slsqp' | 'de' | 'hybrid'
    de_maxiter: int = 40,
    slsqp_maxiter: int = 120,
    alpha_overrides: Optional[Dict[Tuple[str,str], float]] = None,
):
    """
    Optimize per-chamber RCOTs directly (multiple knobs). Uses non-anchored corrected yields.
    Good for deep offline studies; stick to scalar-anchored for daily RCOT control.
    """
    rc_names, bounds = _active_rcot_vars(row, naph_bounds, gas_bounds)
    if not rc_names:
        return {'status':'no_active_knobs', 'rcot':{}, 'margin': np.nan}

    if margin_fn is None:
        if price_provider is None:
            raise ValueError("Provide either price_provider or margin_fn.")
        margin_fn = make_margin_fn(price_provider)

    x0 = []
    for k,(lo,hi) in zip(rc_names, bounds):
        v = float(row.get(k, (lo+hi)/2))
        x0.append(min(max(v, lo), hi))
    x0 = np.array(x0, float)

    def obj(x):
        r = _apply_rcots(row, rc_names, x)
        y = corrected_yields_for_row(
            r, gps, feature_cols_gp, total_spyro_yield_for_now, spyro_ctx, alpha_overrides
        )
        m = margin_fn(pd.Timestamp(row.name) if row.name is not None else pd.Timestamp('1970-01-01'), r, y)
        return -m

    if method in ('de','hybrid') and _HAS_DE:
        de = differential_evolution(obj, bounds, seed=0, maxiter=de_maxiter, polish=False)
        x_init = de.x
    else:
        x_init = x0

    sls = minimize(obj, x_init, method='SLSQP', bounds=bounds,
                   options={'maxiter': slsqp_maxiter, 'ftol': 1e-4, 'disp': False})
    x_best = sls.x if sls.success else x_init
    m_best = -obj(x_best)

    return {'status': 'ok' if sls.success else ('de_only' if method in ('de','hybrid') else 'local_only'),
            'rcot': {k: float(v) for k,v in zip(rc_names, x_best)},
            'margin': float(m_best)}


# ─────────────────────────────────────────────────────────────────────────────
# 7) Convenience wrappers (nice APIs)
# ─────────────────────────────────────────────────────────────────────────────

# def optimize_rcot_for_ts_scalar(
#     *,
#     ts: pd.Timestamp,
#     gp: gpmod.GPResiduals,
#     X_12h: pd.DataFrame,
#     merged_lims: pd.DataFrame,
#     pipeline,
#     ml_cached,
#     geometry: str,                       # 'LF_NAPH' | 'GF_GAS' | 'GF_HYB_NAPH'
#     rc_bounds: Tuple[float, float],
#     price_provider: PriceProvider,
#     margin_mode: str = 'excel_delta',    # 'excel_delta' | 'none' | 'ml'
#     total_spyro_yield_for_now=None,      # required for excel_delta
#     spyro_ctx=None,
#     fg_constants: FuelGasConstants = FuelGasConstants(),
#     step: float = 0.5,
#     use_gp_delta: bool = True,
#     alpha: float = 0.2,
#     util_models: Optional[Dict[str, Any]] = None,
#     util_feature_cols: Optional[Sequence[str]] = None,
#     delta_fg_fn: Optional[Callable[[pd.Series, Dict[str, float], Any, Any, FuelGasConstants], float]] = None,
# ):
#     """Pick the best scalar RCOT on the anchored curve for the given geometry."""
#     setter = {'LF_NAPH': gpmod.rcot_setter_lf_naph,
#               'GF_GAS': gpmod.rcot_setter_gf_gas,
#               'GF_HYB_NAPH': gpmod.rcot_setter_hybrid}[geometry]

#     if margin_mode == 'excel_delta':
#         if total_spyro_yield_for_now is None:
#             raise ValueError("total_spyro_yield_for_now is required for margin_mode='excel_delta'")
#         if delta_fg_fn is None:
#             delta_fg_fn = delta_fg_excel
#         margin_fn = make_margin_fn_excel_delta(
#             price_provider,
#             total_spyro_yield_for_now=total_spyro_yield_for_now, spyro_ctx=spyro_ctx,
#             fg_constants=fg_constants, delta_fg_fn=delta_fg_fn,
#             util_models=util_models, util_feature_cols=util_feature_cols
#         )
#     else:
#         margin_fn = make_margin_fn(
#             price_provider, fg_cost_mode=('ml' if margin_mode == 'ml' else 'none'),
#             util_models=util_models, util_feature_cols=util_feature_cols
#         )

#     return optimize_rcot_scalar_anchored(
#         gp=gp, X_12h=X_12h, merged_lims=merged_lims, pipeline=pipeline, ml_cached=ml_cached,
#         ts=ts, rcot_setter=setter, rc_bounds=rc_bounds, step=step,
#         price_provider=price_provider, margin_fn=margin_fn,
#         use_gp_delta=use_gp_delta, alpha=alpha
#     )

def optimize_rcot_for_ts_scalar(
    *,
    ts: pd.Timestamp,
    gp: gpmod.GPResiduals,
    X_12h: pd.DataFrame,
    merged_lims: pd.DataFrame,
    pipeline,
    ml_cached,
    geometry: str,                       # 'LF_NAPH' | 'GF_GAS' | 'GF_HYB_NAPH'
    rc_bounds: Tuple[float, float],
    price_provider: PriceProvider,
    margin_mode: str = 'excel_delta',    # 'excel_delta' | 'none' | 'ml'
    total_spyro_yield_for_now=None,      # required for excel_delta
    spyro_ctx=None,
    fg_constants: FuelGasConstants = FuelGasConstants(),
    step: float = 0.5,
    use_gp_delta: bool = True,
    alpha: float = 0.2,
    util_models: Optional[Dict[str, Any]] = None,
    util_feature_cols: Optional[Sequence[str]] = None,
    delta_fg_fn: Optional[Callable[[pd.Series, Dict[str, float], Any, Any, FuelGasConstants], float]] = None,
    delta_fg_kwargs: Optional[Dict[str, Any]] = None,   # NEW
):
    """Pick the best scalar RCOT on the anchored curve for the given geometry."""
    setter = {'LF_NAPH': gpmod.rcot_setter_lf_naph,
              'GF_GAS': gpmod.rcot_setter_gf_gas,
              'GF_HYB_NAPH': gpmod.rcot_setter_hybrid}[geometry]

    if margin_mode == 'excel_delta':
        if total_spyro_yield_for_now is None:
            raise ValueError("total_spyro_yield_for_now is required for margin_mode='excel_delta'")
        if delta_fg_fn is None:
            delta_fg_fn = delta_fg_excel
        margin_fn = make_margin_fn_excel_delta(
            price_provider, total_spyro_yield_for_now, spyro_ctx, fg_constants
        )

    else:
        margin_fn = make_margin_fn_excel_delta(
            price_provider, total_spyro_yield_for_now, spyro_ctx, fg_constants
        )

    return optimize_rcot_scalar_anchored(
        gp=gp, X_12h=X_12h, merged_lims=merged_lims, pipeline=pipeline, ml_cached=ml_cached,
        ts=ts, rcot_setter=setter, rc_bounds=rc_bounds, step=step,
        price_provider=price_provider, margin_fn=margin_fn,
        use_gp_delta=use_gp_delta, alpha=alpha
    )


# def optimize_rcot_for_ts_multi(
#     *,
#     ts: pd.Timestamp,
#     row0: pd.Series,
#     gps: Dict[str, Any],
#     feature_cols_gp: Sequence[str],
#     total_spyro_yield_for_now,
#     spyro_ctx=None,
#     price_provider: PriceProvider,
#     objective: str = 'per_hour',              # 'per_hour' | 'per_ton_fresh'
#     naph_bounds: Tuple[float,float]=(830.0, 853.0),
#     gas_bounds: Tuple[float,float]=(870.0, 890.0),
#     trust_delta_C: float = 5.0,
#     use_recycle_fixed_point: bool = True,
#     recycle_fn: Optional[Callable[..., Any]] = None,
#     recycle_iters: int = 10, recycle_damping: float = 0.5, recycle_tol: float = 1e-4,
#     alpha_overrides: Optional[Dict[Tuple[str,str], float]] = None,
#     margin_mode: str = 'excel_delta',        # 'excel_delta' | 'none' | 'ml'
#     fg_constants: FuelGasConstants = FuelGasConstants(),
#     util_models: Optional[Dict[str, Any]] = None,
#     util_feature_cols: Optional[Sequence[str]] = None,
#     enable_de: bool = False, de_maxiter: int = 40, de_popsize: int = 12,
#     enable_slsqp: bool = True, slsqp_maxiter: int = 120,
#     delta_fg_fn: Optional[Callable[[pd.Series, Dict[str, float], Any, Any, FuelGasConstants], float]] = None,
#     delta_fg_kwargs: Optional[Dict[str, Any]] = None,     # NEW
# ):
#     """
#     Optimize per-chamber RCOTs using corrected-yields + Excel-delta margin.
#     """
#     rc_names, bounds = _active_rcot_vars(row0, naph_bounds, gas_bounds)
#     if not rc_names:
#         return {'status': 'no_active_knobs', 'rcot': {}, 'margin': np.nan}

#     if trust_delta_C and trust_delta_C > 0:
#         new_bounds = []
#         for k, (lo, hi) in zip(rc_names, bounds):
#             xc = float(row0.get(k, 0.5*(lo+hi)))
#             a = max(lo, min(xc - trust_delta_C, hi))
#             b = max(lo, min(xc + trust_delta_C, hi))
#             if b < a: a = b = xc
#             new_bounds.append((a, b))
#         bounds = new_bounds

#     if margin_mode == 'excel_delta':
#         if total_spyro_yield_for_now is None:
#             raise ValueError("total_spyro_yield_for_now is required for margin_mode='excel_delta'")
#         margin_fn = make_margin_fn_excel_delta(
#             price_provider=price_provider,
#             total_spyro_yield_for_now=total_spyro_yield_for_now,
#             spyro_ctx=spyro_ctx,
#             fg_constants=fg_constants,   # carries your energy model constants
#         )
#     else:
#         margin_fn = make_margin_fn(
#             price_provider,
#             fg_cost_mode=('ml' if margin_mode == 'ml' else 'none'),
#             util_models=util_models,
#             util_feature_cols=util_feature_cols,
#         )
#     def _fresh_basis_tph(r_final: pd.Series, corrected: Dict[str, float]) -> float:
#         D   = float(r_final.get('feed_qty', 0.0))
#         L   = float(r_final.get('recycle_C2H6', 0.0))
#         M   = float(r_final.get('recycle_C3H8', 0.0))
#         MC4 = float(corrected.get('MixedC4_prod_t+1', 0.0))
#         return max(D - L - M - MC4, 1e-6)

#     def _apply(x):
#         r = row0.copy()
#         for k, v in zip(rc_names, x): r[k] = float(v)
#         if use_recycle_fixed_point and recycle_fn is not None:
#             r_ss, _, _ = recycle_fn(
#                 r, rc_names, x, gps, feature_cols_gp,
#                 total_spyro_yield_for_now=total_spyro_yield_for_now, spyro_ctx=spyro_ctx,
#                 damping=recycle_damping, iters=recycle_iters, tol=recycle_tol,
#                 alpha_overrides=alpha_overrides, geometry_from_row=None
#             )
#             return r_ss
#         return r

#     def _corrected(r_like):
#         return corrected_yields_for_row(
#             r_like, gps, feature_cols_gp, total_spyro_yield_for_now, spyro_ctx, alpha_overrides
#         )

#     # baseline
#     x0 = [min(max(float(row0.get(k, np.mean(bounds[i]))), bounds[i][0]), bounds[i][1]) for i, k in enumerate(rc_names)]
#     r_base = _apply(x0)
#     y_base = _corrected(r_base)
#     m_base = margin_fn(ts, r_base, y_base)
#     fresh_base = _fresh_basis_tph(r_base, y_base)
#     per_t_base = m_base / max(fresh_base, 1e-6)

#     # objective
#     def eval_candidate(x):
#         r_c = _apply(x)
#         y_c = _corrected(r_c)
#         m_h = margin_fn(ts, r_c, y_c)
#         fresh = _fresh_basis_tph(r_c, y_c)
#         m_t = m_h / max(fresh, 1e-6)
#         return m_h, m_t, y_c, r_c, fresh

#     def neg_obj(x):
#         m_h, m_t, *_ = eval_candidate(x)
#         return - (m_t if objective == 'per_ton_fresh' else m_h)

#     # warm start
#     x_ws = list(x0)
#     for j in range(len(x_ws)):
#         lo, hi = bounds[j]
#         grid = np.linspace(lo, hi, 5)
#         vals = [neg_obj([*(x_ws[:j]), g, *(x_ws[j+1:])]) for g in grid]
#         x_ws[j] = float(grid[int(np.argmin(vals))])

#     # solve
#     x_best, out_best = x_ws, eval_candidate(x_ws)
#     if enable_slsqp:
#         res = minimize(neg_obj, x0=x_ws, method='SLSQP',
#                        bounds=tuple(bounds), options={'maxiter': slsqp_maxiter, 'ftol': 1e-4, 'disp': False})
#         if res.success:
#             x_best, out_best = res.x, eval_candidate(res.x)

#     if enable_de and _HAS_DE:
#         res_d = differential_evolution(lambda z: neg_obj(list(z)), bounds=tuple(bounds),
#                                        strategy='best1bin', maxiter=de_maxiter,
#                                        popsize=de_popsize, tol=0.01, seed=0, polish=True)
#         x_de, out_de = res_d.x, eval_candidate(res_d.x)
#         if -neg_obj(x_de) > -neg_obj(x_best):
#             x_best, out_best = x_de, out_de

#     m_h_best, m_t_best, y_best, r_best, fresh_best = out_best

#     return {
#         'status': 'ok',
#         'ts': ts,
#         'rcot_names': rc_names,
#         'rcot_bounds': bounds,
#         'rcot_current': {k: float(row0.get(k, np.nan)) for k in rc_names},
#         'rcot_opt':     {k: float(v) for k, v in zip(rc_names, x_best)},
#         'margin_current_per_h': float(m_base),
#         'margin_opt_per_h':     float(m_h_best),
#         'improvement_per_h':    float(m_h_best - m_base),
#         'fresh_current_tph': float(fresh_base),
#         'fresh_opt_tph':     float(fresh_best),
#         'margin_current_per_t_fresh': float(per_t_base),
#         'margin_opt_per_t_fresh':     float(m_t_best),
#         'yields_current': {k: float(v) for k, v in y_base.items()},
#         'yields_opt':     {k: float(v) for k, v in y_best.items()},
#         'row_opt': r_best,
#     }
def anchored_expected_for_row(
    row_base: pd.Series,      # baseline row at rc0 (current)
    row_cand: pd.Series,      # candidate row with knob changes
    gp, pipeline, merged_lims,
    ml_cached,
    alpha_default: float = 0.0,
    alpha_overrides: Optional[Dict[Tuple[str,str], float]] = None
) -> Dict[str, float]:
    """
    Returns absolute t/h per *_prod_t+1 using:
      ML(rc0) + [ SRTO(row_cand) - SRTO(row_base) ] + α · [ μ(row_cand) - μ(row_base) ]
    Default α=0 → pure SRTO slopes; sets level to ML at rc0.
    """
    ts = getattr(row_base, 'name', None)
    comp_row = gp._comp_row_for_ts(merged_lims, ts) if hasattr(gp, '_comp_row_for_ts') else None

    spot0 = pipeline.predict_spot_plant(row_base, comp_row, feed_thr=0.1)['totals_tph']
    spot1 = pipeline.predict_spot_plant(row_cand, comp_row, feed_thr=0.1)['totals_tph']

    # ML anchor at rc0
    ml_point = ml_cached.predict_row(row_base) if hasattr(ml_cached, 'predict_row') else {}
    # Optional GP delta (usually off)
    if alpha_default != 0.0 or (alpha_overrides and len(alpha_overrides)):
        mu0, _ = gp._gp_mu_for_row(row_base,  return_std=False)
        mu1, _ = gp._gp_mu_for_row(row_cand, return_std=False)

    out = {}
    for p in gpmod.PRODUCTS:
        tcol = gpmod.TARGET_MAP[p]
        anchor = float(ml_point.get(tcol, 0.0))
        d_phys = float(spot1.get(p, 0.0)) - float(spot0.get(p, 0.0))
        if alpha_overrides:
            # per-geometry alpha is not meaningful across multi-knob;
            # use default alpha=0 (pure SRTO) for strict physics slopes
            a = alpha_overrides.get(('IGNORED', tcol), alpha_default)
        else:
            a = alpha_default
        d_mu = (mu1[p] - mu0[p]) * a if (alpha_default != 0.0) else 0.0
        out[tcol] = max(0.0, anchor + d_phys + d_mu)
    return out

# signature: add bounds_by_knob
def optimize_rcot_for_ts_multi(
    *,
    ts: pd.Timestamp,
    row0: pd.Series,
    gps: Dict[str, Any],
    feature_cols_gp: Sequence[str],
    total_spyro_yield_for_now,
    spyro_ctx=None,
    price_provider: PriceProvider,
    objective: str = 'per_hour',
    naph_bounds: Tuple[float,float]=(810.0, 853.0),
    gas_bounds: Tuple[float,float]=(850.0, 890.0),
    trust_delta_C: float = 10.0,
    use_recycle_fixed_point: bool = True,
    recycle_fn: Optional[Callable[..., Any]] = None,
    recycle_iters: int = 10, recycle_damping: float = 0.5, recycle_tol: float = 1e-4,
    alpha_overrides: Optional[Dict[Tuple[str,str], float]] = None,
    margin_mode: str = 'excel_delta',
    fg_constants: FuelGasConstants = FuelGasConstants(),
    util_models: Optional[Dict[str, Any]] = None,
    util_feature_cols: Optional[Sequence[str]] = None,
    enable_de: bool = False, de_maxiter: int = 40, de_popsize: int = 12,
    enable_slsqp: bool = True, slsqp_maxiter: int = 120,
    delta_fg_fn: Optional[Callable[[pd.Series, Dict[str, float], Any, Any, FuelGasConstants], float]] = None,
    delta_fg_kwargs: Optional[Dict[str, Any]] = None,
    bounds_by_knob: Optional[Dict[str, Tuple[float,float]]] = None,  # ← NEW
    anchored_from_ml: bool = True,
    gp: Any = None,
    pipeline: Any = None,
    merged_lims: Any = None,
    ml_cached: Any = None,
    alpha_default_for_anchor: float = 0.0,

):
    """
    Optimize per-chamber RCOTs using corrected-yields + Excel-delta margin.
    """
    rc_names, bounds = _active_rcot_vars(row0, naph_bounds, gas_bounds)
    if not rc_names:
        return {'status': 'no_active_knobs', 'rcot': {}, 'margin': np.nan}

    # 1) trust radius around current (if requested)
    if trust_delta_C and trust_delta_C > 0:
        shrink = []
        for k, (lo, hi) in zip(rc_names, bounds):
            xc = float(row0.get(k, 0.5*(lo+hi)))
            a = max(lo, min(xc - trust_delta_C, hi))
            b = max(lo, min(xc + trust_delta_C, hi))
            if b < a: a = b = xc
            shrink.append((a, b))
        bounds = shrink

    # 2) per-knob override (intersect with #1)
    if bounds_by_knob:
        merged = []
        for (k, (lo, hi)) in zip(rc_names, bounds):
            if k in bounds_by_knob:
                lo2, hi2 = bounds_by_knob[k]
                lo, hi = max(lo, lo2), min(hi, hi2)
                if hi < lo:  # fully frozen by gate
                    v = float(row0.get(k, (lo+hi)/2))
                    lo = hi = v
            merged.append((lo, hi))
        bounds = merged

    if margin_mode == 'excel_delta':
        if total_spyro_yield_for_now is None:
            raise ValueError("total_spyro_yield_for_now is required for margin_mode='excel_delta'")
        margin_fn = make_margin_fn_excel_delta(
            price_provider=price_provider,
            total_spyro_yield_for_now=total_spyro_yield_for_now,
            spyro_ctx=spyro_ctx,
            fg_constants=fg_constants,   # carries your energy model constants
        )
    else:
        margin_fn = make_margin_fn(
            price_provider,
            fg_cost_mode=('ml' if margin_mode == 'ml' else 'none'),
            util_models=util_models,
            util_feature_cols=util_feature_cols,
        )
    def _fresh_basis_tph(r_final: pd.Series, corrected: Dict[str, float]) -> float:
        D   = float(r_final.get('feed_qty', 0.0))
        L   = float(r_final.get('recycle_C2H6', 0.0))
        M   = float(r_final.get('recycle_C3H8', 0.0))
        MC4 = float(corrected.get('MixedC4_prod_t+1', 0.0))
        return max(D - L - M - MC4, 1e-6)

    def _apply(x):
        r = row0.copy()
        for k, v in zip(rc_names, x): r[k] = float(v)
        if use_recycle_fixed_point and recycle_fn is not None:
            r_ss, _, _ = recycle_fn(
                r, rc_names, x, gps, feature_cols_gp,
                total_spyro_yield_for_now=total_spyro_yield_for_now, spyro_ctx=spyro_ctx,
                damping=recycle_damping, iters=recycle_iters, tol=recycle_tol,
                alpha_overrides=alpha_overrides, geometry_from_row=None
            )
            return r_ss
        return r

    # def _corrected(r_like):
    #     return corrected_yields_for_row(
    #         r_like, gps, feature_cols_gp, total_spyro_yield_for_now, spyro_ctx, alpha_overrides
    #     )

    def _corrected(r_like):
        if anchored_from_ml:
            return anchored_expected_for_row(
                row_base=row0, row_cand=r_like,
                gp=gp, pipeline=pipeline, merged_lims=merged_lims,
                ml_cached=ml_cached,
                alpha_default=alpha_default_for_anchor,   # 0.0 → pure SRTO slopes
                alpha_overrides=None
            )
        else:
            return corrected_yields_for_row(
                r_like, gps, feature_cols_gp, total_spyro_yield_for_now, spyro_ctx, alpha_overrides
            )


    # baseline
    x0 = [min(max(float(row0.get(k, np.mean(bounds[i]))), bounds[i][0]), bounds[i][1]) for i, k in enumerate(rc_names)]
    r_base = _apply(x0)
    y_base = _corrected(r_base)
    m_base = margin_fn(ts, r_base, y_base)
    fresh_base = _fresh_basis_tph(r_base, y_base)
    per_t_base = m_base / max(fresh_base, 1e-6)

    # objective
    def eval_candidate(x):
        r_c = _apply(x)
        y_c = _corrected(r_c)
        m_h = margin_fn(ts, r_c, y_c)
        fresh = _fresh_basis_tph(r_c, y_c)
        m_t = m_h / max(fresh, 1e-6)
        return m_h, m_t, y_c, r_c, fresh

    def neg_obj(x):
        m_h, m_t, *_ = eval_candidate(x)
        return - (m_t if objective == 'per_ton_fresh' else m_h)

    # warm start
    x_ws = list(x0)
    for j in range(len(x_ws)):
        lo, hi = bounds[j]
        grid = np.linspace(lo, hi, 5)
        vals = [neg_obj([*(x_ws[:j]), g, *(x_ws[j+1:])]) for g in grid]
        x_ws[j] = float(grid[int(np.argmin(vals))])

    # solve
    x_best, out_best = x_ws, eval_candidate(x_ws)
    if enable_slsqp:
        res = minimize(neg_obj, x0=x_ws, method='SLSQP',
                       bounds=tuple(bounds), options={'maxiter': slsqp_maxiter, 'ftol': 1e-4, 'disp': False})
        if res.success:
            x_best, out_best = res.x, eval_candidate(res.x)

    if enable_de and _HAS_DE:
        res_d = differential_evolution(lambda z: neg_obj(list(z)), bounds=tuple(bounds),
                                       strategy='best1bin', maxiter=de_maxiter,
                                       popsize=de_popsize, tol=0.01, seed=0, polish=True)
        x_de, out_de = res_d.x, eval_candidate(res_d.x)
        if -neg_obj(x_de) > -neg_obj(x_best):
            x_best, out_best = x_de, out_de

    m_h_best, m_t_best, y_best, r_best, fresh_best = out_best

    return {
        'status': 'ok',
        'ts': ts,
        'rcot_names': rc_names,
        'rcot_bounds': bounds,
        'rcot_current': {k: float(row0.get(k, np.nan)) for k in rc_names},
        'rcot_opt':     {k: float(v) for k, v in zip(rc_names, x_best)},
        'margin_current_per_h': float(m_base),
        'margin_opt_per_h':     float(m_h_best),
        'improvement_per_h':    float(m_h_best - m_base),
        'fresh_current_tph': float(fresh_base),
        'fresh_opt_tph':     float(fresh_best),
        'margin_current_per_t_fresh': float(per_t_base),
        'margin_opt_per_t_fresh':     float(m_t_best),
        'yields_current': {k: float(v) for k, v in y_base.items()},
        'yields_opt':     {k: float(v) for k, v in y_best.items()},
        'row_opt': r_best,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 9) Expected vs Simulated audit (for scalar RCOT pick)
# ─────────────────────────────────────────────────────────────────────────────

# Canonical product list & token map
_PRODS_CANON = ['Ethylene','Propylene','Mixed C4','RPG','PFO','Hydrogen','Tail Gas']
_CANON2TOKEN = {'Mixed C4': 'MixedC4', 'Tail Gas': 'Tail_Gas'}  # others map to themselves

def _anchored_expected_from_curve_row(row: pd.Series) -> dict:
    """
    Pull expected absolute t/h from anchored curve row.
    The anchored curve uses columns like 'Ethylene_CORR_tph', 'MixedC4_CORR_tph', etc.
    """
    out = {}
    for p in _PRODS_CANON:
        tok = _CANON2TOKEN.get(p, p)
        col = f'{tok}_CORR_tph'
        out[p] = float(row.get(col, np.nan))
    return out

def _srto_simulated_abs(row_with_rcot: pd.Series,
                        total_spyro_yield_for_now,
                        spyro_ctx) -> dict:
    """
    Call your SRTO aggregation callable for each product to get absolute t/h.
    (This is the same callable you pass to delta_fg_excel.)
    """
    out = {}
    for p in _PRODS_CANON:
        tok = _CANON2TOKEN.get(p, p)  # SRTO short keys accept 'MixedC4', 'Tail_Gas', etc.
        try:
            out[p] = float(total_spyro_yield_for_now(row_with_rcot, tok, ctx=spyro_ctx))
        except Exception:
            out[p] = np.nan
    return out

def audit_expected_vs_sim_scalar(*,
    ts: pd.Timestamp,
    geometry: str,                               # 'LF_NAPH' | 'GF_GAS' | 'GF_HYB_NAPH'
    opt: dict,                                    # return of optimize_rcot_scalar_anchored
    total_spyro_yield_for_now,
    spyro_ctx=None,
    save_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Build a one-row audit with expected vs simulated for the chosen RCOT.
    """
    rc_opt   = float(opt['rc_opt'])
    curve    = opt['curve']
    row_base = opt['row_base']  # we added this return

    # pick the curve row nearest to rc_opt (float-safe)
    i = int((curve['RCOT'] - rc_opt).abs().idxmin())
    row_curve = curve.loc[i]
    expected_abs = _anchored_expected_from_curve_row(row_curve)

    # set RCOT on the base row for this geometry
    setter = {'LF_NAPH': gpmod.rcot_setter_lf_naph,
              'GF_GAS':  gpmod.rcot_setter_gf_gas,
              'GF_HYB_NAPH': gpmod.rcot_setter_hybrid}[geometry]
    row_for_sim = setter(row_base.copy(), rc_opt)

    # simulate absolute t/h via SRTO aggregate callable
    simulated_abs = _srto_simulated_abs(row_for_sim, total_spyro_yield_for_now, spyro_ctx)

    # assemble one-row DataFrame with expected/sim/diff
    rec = {'date': pd.Timestamp(ts), 'geometry': geometry, 'RCOT': rc_opt}
    for p in _PRODS_CANON:
        rec[f'expected_{p}'] = expected_abs.get(p, np.nan)
        rec[f'sim_{p}']      = simulated_abs.get(p, np.nan)
        rec[f'diff_{p}']     = rec[f'sim_{p}'] - rec[f'expected_{p}']
    df = pd.DataFrame([rec]).set_index('date').sort_index()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, mode='a', header=not Path(save_path).exists())
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 9) Expected vs Simulated audit (for multi-knob result)
# ─────────────────────────────────────────────────────────────────────────────

_PRODS_CANON = ['Ethylene','Propylene','Mixed C4','RPG','PFO','Hydrogen','Tail Gas']
_CANON2TOKEN = {'Mixed C4': 'MixedC4', 'Tail Gas': 'Tail_Gas'}  # SRTO short keys

# map corrected-yields keys → canonical
_EXP_KEYS = {
    'Ethylene':  'Ethylene_prod_t+1',
    'Propylene': 'Propylene_prod_t+1',
    'Mixed C4':  'MixedC4_prod_t+1',
    'RPG':       'RPG_prod_t+1',
    'PFO':       'PFO_prod_t+1',
    'Hydrogen':  'Hydrogen_prod_t+1',
    'Tail Gas':  'Tail_Gas_prod_t+1',
}

def _sim_abs_from_row(row: pd.Series, total_spyro_yield_for_now, spyro_ctx=None) -> dict:
    out = {}
    for p in _PRODS_CANON:
        tok = _CANON2TOKEN.get(p, p)
        try:
            out[p] = float(total_spyro_yield_for_now(row, tok, ctx=spyro_ctx))
        except Exception:
            out[p] = np.nan
    return out

def _exp_abs_from_corrected(yields_abs: dict) -> dict:
    out = {}
    for p, k in _EXP_KEYS.items():
        out[p] = float(yields_abs.get(k, np.nan))
    return out

def audit_expected_vs_sim_multi(
    *,
    ts: pd.Timestamp,
    geometry: str,
    row_baseline: pd.Series,
    exp_baseline: dict,     # corrected absolute t/h (y0)
    row_opt: pd.Series,
    exp_opt: dict,          # corrected absolute t/h (res['yields_opt'])
    total_spyro_yield_for_now,
    spyro_ctx=None
) -> pd.DataFrame:
    """
    Returns a two-row DF: baseline & optimized with expected_*, sim_*, diff_* for each product.
    """
    sim_base = _sim_abs_from_row(row_baseline, total_spyro_yield_for_now, spyro_ctx)
    sim_opt  = _sim_abs_from_row(row_opt,       total_spyro_yield_for_now, spyro_ctx)
    exp_base = _exp_abs_from_corrected(exp_baseline)
    exp_o    = _exp_abs_from_corrected(exp_opt)

    rows = []
    for tag, sim, exp in [('baseline', sim_base, exp_base), ('optimized', sim_opt, exp_o)]:
        rec = {'date': pd.Timestamp(ts), 'geometry': geometry, 'case': tag}
        for p in _PRODS_CANON:
            rec[f'expected_{p}'] = exp.get(p, np.nan)
            rec[f'sim_{p}']      = sim.get(p, np.nan)
            rec[f'diff_{p}']     = rec[f'sim_{p}'] - rec[f'expected_{p}']
        rows.append(rec)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 8) Public API
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    # constants & prices
    'FuelGasConstants', 'PriceProvider',
    # geometry & delta-FG
    'geometry_from_row', 'delta_fg_excel',
    # margin builders
    'make_margin_fn', 'make_margin_fn_excel_delta',
    # scalar-anchored optimizer + convenience
    'optimize_rcot_scalar_anchored', 'optimize_rcot_for_ts_scalar',
    # multi-knob optimizer + convenience
    'optimize_rcot_multi', 'optimize_rcot_for_ts_multi',
    # helpers
    'sum_naphtha_feed_tph', 'sum_gas_feed_tph','audit_expected_vs_sim_scalar'
]
__all__.extend(['audit_expected_vs_sim_multi'])
