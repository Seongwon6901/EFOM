# =========================
# Data Loading Module - Integrated Positions & Canonical Mapping
# =========================
from dataclasses import dataclass
from typing import Dict, Optional, Any, Union, Tuple, Iterable
from pathlib import Path
import pandas as pd
import numpy as np
import re

# ─────────────────────────────────────────────────────────────────────────────
# IO helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_pickle_or_excel(
    pkl_path: Path,
    excel_path: Optional[Path] = None,
    refresh_if_excel_newer: bool = True,
    **read_excel_kwargs
):
    """
    Load a DataFrame from pickle if available; otherwise from Excel and cache to pickle.
    If 'refresh_if_excel_newer' and Excel mtime > pickle mtime, refresh the pickle.
    Extra kwargs (e.g., header=..., usecols=...) are passed to read_excel.
    """
    pkl_path = Path(pkl_path)
    pkl_path.parent.mkdir(parents=True, exist_ok=True)

    if pkl_path.exists() and not refresh_if_excel_newer:
        return pd.read_pickle(pkl_path)

    if pkl_path.exists() and refresh_if_excel_newer and excel_path and Path(excel_path).exists():
        if Path(excel_path).stat().st_mtime > pkl_path.stat().st_mtime:
            df = pd.read_excel(excel_path, **read_excel_kwargs)
            df.to_pickle(pkl_path)
            return df
        return pd.read_pickle(pkl_path)

    if pkl_path.exists():
        return pd.read_pickle(pkl_path)

    if excel_path and Path(excel_path).exists():
        df = pd.read_excel(excel_path, **read_excel_kwargs)
        df.to_pickle(pkl_path)
        return df

    raise FileNotFoundError(f"Neither pickle nor Excel found.\n  pkl={pkl_path}\n  xlsx={excel_path}")

# ─────────────────────────────────────────────────────────────────────────────
# LIMS (daily feeds) loaders/cleaners
# ─────────────────────────────────────────────────────────────────────────────
# put this near your other top-level cleaners
def clean_pona_from_pi(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().reset_index(drop=True)
    ts_col = d.columns[0]
    out = d[[ts_col]].rename(columns={ts_col: 'Timestamp'})

    exact = {
        "M10L41004_Paraffins(vol%)":  "Paraffins",
        "M10L41004_Olefins(vol%)":    "Olefins",
        "M10L41004_Naphthenes(vol%)": "Naphthenes",
        "M10L41004_Aromatics(vol%)":  "Aromatics",
        "M10L41004_n-Paraffin(vol%)": "n-Paraffin",
        "M10L41004_i-Paraffin(vol%)": "i-Paraffin",
    }
    for src, dst in exact.items():
        if src in d.columns:
            out[dst] = pd.to_numeric(d[src], errors='coerce')

    if "Paraffins" not in out.columns:
        npar = out.get("n-Paraffin")
        ipar = out.get("i-Paraffin")
        if npar is not None or ipar is not None:
            out["Paraffins"] = (npar.fillna(0) if npar is not None else 0) + \
                               (ipar.fillna(0) if ipar is not None else 0)

    out["Timestamp"] = pd.to_datetime(out["Timestamp"], errors="coerce").dt.tz_localize(None)
    out = out.set_index("Timestamp").sort_index()

    keep = [c for c in ["Paraffins","Olefins","Naphthenes","Aromatics"] if c in out.columns]
    return out[keep]

def load_feed_data(nap_path=None, gas_path=None, paths=None, header=1):
    """
    Load and prepare feed data (daily LIMS).
    Returns a merged DataFrame keyed by 'date' with naphtha and gas comps.
    """
    if paths is not None:
        try:
            nap_path = nap_path or paths.nap_excel_path
            gas_path = gas_path or paths.gas_excel_path
        except Exception:
            pass

    if nap_path is None or gas_path is None:
        raise ValueError("nap_path and gas_path (or paths) must be provided")

    lims = pd.read_excel(nap_path, header=header).fillna(0.0)
    feed_naph = lims.rename(columns={'Unnamed: 2': 'date'})

    gas_feed = pd.read_excel(gas_path, header=header).fillna(0.0)
    gas_feed = gas_feed.rename(columns={'Unnamed: 2': 'date'})

    gas_cols = ['date', 'Ethylene', 'Ethane', 'Propylene', 'Propane', 'n-Butane', 'i-Butane']
    available_gas_cols = [col for col in gas_cols if col in gas_feed.columns]
    feed_gas = gas_feed[available_gas_cols]

    feed_naph['date'] = pd.to_datetime(feed_naph['date'], errors='coerce')
    feed_gas['date'] = pd.to_datetime(feed_gas['date'], errors='coerce')

    merged_lims = pd.merge(feed_naph, feed_gas, on='date', how='inner').sort_values('date')

    cols = [c for c in ['Ethylene', 'Ethane', 'Propylene', 'Propane', 'n-Butane', 'i-Butane']
            if c in merged_lims.columns]
    if cols:
        zero_rows = (merged_lims[cols].sum(axis=1) == 0)
        merged_lims.loc[zero_rows, cols] = np.nan
        merged_lims[cols] = merged_lims[cols].ffill().bfill()

    return merged_lims

# def clean_feed_df(df, is_gas=False):
#     """Clean feed dataframes exported from Excel into a timeseries-indexed numeric DF."""
#     df = df.copy()

#     if isinstance(df.index, pd.RangeIndex) and len(df) > 2:
#         df = df.drop([0, 1], axis=0)
#         df = df.iloc[:, 2:]
#         if 'Unnamed: 2' in df.columns:
#             df = df.rename(columns={'Unnamed: 2': 'Timestamp'})

#     if 'Timestamp' in df.columns:
#         df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
#         df = df.set_index('Timestamp')

#     if not isinstance(df.index, pd.DatetimeIndex):
#         try:
#             df.index = pd.to_datetime(df.index)
#         except Exception:
#             pass

#     df = df.drop(columns=['Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5'], errors='ignore')

#     if is_gas:
#         for col in ['Ethylene', 'Ethane', 'Propylene', 'Propane', 'n-Butane', 'i-Butane']:
#             if col in df.columns:
#                 df[col] = pd.to_numeric(df[col], errors='coerce')
#     else:
#         if 'Paraffins' in df.columns:
#             df['Paraffins'] = pd.to_numeric(df['Paraffins'], errors='coerce')

#     return df
def clean_feed_df(df, is_gas=False):
    df = df.copy()

    # normalize common lab tag names → simple headers
    rename_map = {
        "M10L41004_Paraffins(vol%)":  "Paraffins",
        "M10L41004_Olefins(vol%)":    "Olefins",
        "M10L41004_Naphthenes(vol%)": "Naphthenes",
        "M10L41004_Aromatics(vol%)":  "Aromatics",
    }
    df = df.rename(columns=rename_map)

    # existing timestamp handling…
    if isinstance(df.index, pd.RangeIndex) and len(df) > 2:
        df = df.drop([0, 1], axis=0)
        df = df.iloc[:, 2:]
        if 'Unnamed: 2' in df.columns:
            df = df.rename(columns={'Unnamed: 2': 'Timestamp'})
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df = df.set_index('Timestamp')
    if not isinstance(df.index, pd.DatetimeIndex):
        try: df.index = pd.to_datetime(df.index)
        except: pass
    df = df.drop(columns=['Unnamed: 3','Unnamed: 4','Unnamed: 5'], errors='ignore')

    if is_gas:
        for col in ['Ethylene','Ethane','Propylene','Propane','n-Butane','i-Butane']:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        for col in ['Paraffins','Olefins','Naphthenes','Aromatics']:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def _clean_pona_from_pi(self, df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().reset_index(drop=True)
    ts_col = d.columns[0]
    out = d[[ts_col]].rename(columns={ts_col: 'Timestamp'})

    # Exact → canonical
    exact = {
        "M10L41004_Paraffins(vol%)":  "Paraffins",
        "M10L41004_Olefins(vol%)":    "Olefins",
        "M10L41004_Naphthenes(vol%)": "Naphthenes",
        "M10L41004_Aromatics(vol%)":  "Aromatics",
        "M10L41004_n-Paraffin(vol%)": "n-Paraffin",
        "M10L41004_i-Paraffin(vol%)": "i-Paraffin",
    }
    for src, dst in exact.items():
        if src in d.columns:
            out[dst] = pd.to_numeric(d[src], errors='coerce')

    # If aggregate Paraffins missing, build from n-/i-
    if "Paraffins" not in out.columns:
        npar = out.get("n-Paraffin")
        ipar = out.get("i-Paraffin")
        if npar is not None or ipar is not None:
            out["Paraffins"] = (npar.fillna(0) if npar is not None else 0) + \
                               (ipar.fillna(0) if ipar is not None else 0)

    out["Timestamp"] = pd.to_datetime(out["Timestamp"], errors="coerce").dt.tz_localize(None)
    out = out.set_index("Timestamp").sort_index()

    # Keep only the 4 aggregates the rest of the pipeline expects
    keep = [c for c in ["Paraffins","Olefins","Naphthenes","Aromatics"] if c in out.columns]
    return out[keep]


def clean_furnace_or_production_df(df, n_drop=3, timestamp_colname="Timestamp"):
    """Legacy cleaner for furnace sheet."""
    df = df.copy()
    df.columns = df.iloc[0]
    df = df.drop(df.index[0]).reset_index(drop=True)
    df = df.drop(range(n_drop), axis=0).reset_index(drop=True)
    df.columns.values[0] = timestamp_colname
    df[timestamp_colname] = pd.to_datetime(df[timestamp_colname])
    df = df.set_index(timestamp_colname)
    df = df.iloc[:, 1:]
    return df

def clean_fresh_feed(df: pd.DataFrame,
                    ts_col_candidates=('Timestamp','Unnamed: 0')) -> pd.DataFrame:
    """
    Expect header ≈ 3, then 2 junk lines (like your example).
    Make a DateTimeIndex and numeric columns; create plant-level 'feed_qty' = row-wise sum.
    """
    d = df.copy()

    # Try to find timestamp column
    ts_col = next((c for c in ts_col_candidates if c in d.columns), d.columns[0])
    d = d.rename(columns={ts_col: 'Timestamp'})

    # Drop the two extra lines after header (like your fresh_feed.iloc[2:])
    if len(d) >= 2:
        d = d.iloc[2:].reset_index(drop=True)

    # Parse timestamp, set index
    d['Timestamp'] = pd.to_datetime(d['Timestamp'], errors='coerce')
    d = d.dropna(subset=['Timestamp']).set_index('Timestamp').sort_index()

    # Convert all non-time columns to numeric (coerce)
    num_cols = [c for c in d.columns if c != 'Timestamp']
    for c in num_cols:
        d[c] = pd.to_numeric(d[c], errors='coerce')

    # Optional: give columns a stable prefix (so we know where they came from)
    # Comment out if you want to keep original names.
    d = d.rename(columns={c: f'FreshFeed_{c}' for c in d.columns})

    # Plant-level feed_qty if not provided: sum across numeric cols
    if 'feed_qty' not in d.columns:
        d['feed_qty'] = d.select_dtypes(include=[np.number]).sum(axis=1)

    return d

# ─────────────────────────────────────────────────────────────────────────────
# Furnace columns cleaner
# ─────────────────────────────────────────────────────────────────────────────

def clean_furnace_columns(df):
    """Clean and rename furnace columns with chamber numbers."""
    new_cols = []
    col_counts = {}
    rcot_chamber_counts = {}
    rcot_start_chamber = 4

    for col in df.columns:
        col_clean = str(col).strip()
        if col_clean.startswith("RCOT #"):
            n = int(col_clean.split("#")[1])
            fuel = "naphtha" if n % 2 == 1 else "gas"
            rcot_chamber_counts[col_clean] = rcot_chamber_counts.get(col_clean, 0) + 1
            chamber_no = rcot_start_chamber + rcot_chamber_counts[col_clean] - 1
            new_col = f"{col_clean}_{fuel}_chamber{chamber_no}"
        else:
            start_chamber = 4 if col_clean == "Gas Feed" else 1
            col_counts[col_clean] = col_counts.get(col_clean, 0) + 1
            chamber_no = start_chamber + col_counts[col_clean] - 1
            new_col = f"{col_clean}_chamber{chamber_no}"
        new_cols.append(new_col)

    df.columns = new_cols
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

def resample_and_slice(df_numeric, freq, col_slice=None, **resample_kwargs):
    """Resample dataframe and optionally slice columns (kept for backwards compat)."""
    df = df_numeric.resample(freq, **resample_kwargs).mean()
    if col_slice is not None:
        df = df.iloc[:, col_slice].copy()
    return df

def reindex_ffill(df_daily, target_index):
    """Reindex and forward fill."""
    return df_daily.reindex(target_index, method='ffill')

def _mean_over_existing(df: pd.DataFrame, cols):
    valid = [c for c in cols if c in df.columns]
    if not valid:
        return pd.Series(0.0, index=df.index)
    return df[valid].mean(axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# RCOT virtuals
# ─────────────────────────────────────────────────────────────────────────────

def build_virtual_rcots_inplace(X: pd.DataFrame) -> None:
    """Create virtual RCOT_naphtha_chamber{4..6} and RCOT_gas_chamber{4..6}."""
    for ch in (4, 5, 6):
        naph_odds = [f"RCOT #{i}_naphtha_chamber{ch}" for i in (1, 3, 5, 7)]
        gas_evens = [f"RCOT #{i}_gas_chamber{ch}" for i in (2, 4, 6, 8)]

        has_naph = X.get(f"Naphtha_chamber{ch}", pd.Series(0, index=X.index)) > 0
        has_gas = X.get(f"Gas Feed_chamber{ch}", pd.Series(0, index=X.index)) > 0

        all8 = _mean_over_existing(X, naph_odds + gas_evens)
        navg = _mean_over_existing(X, naph_odds)
        gavg = _mean_over_existing(X, gas_evens)

        X[f"RCOT_naphtha_chamber{ch}"] = np.where(
            has_naph & has_gas, navg, np.where(has_naph, all8, 0.0)
        )
        X[f"RCOT_gas_chamber{ch}"] = np.where(
            has_naph & has_gas, gavg, np.where(has_gas, all8, 0.0)
        )

# ─────────────────────────────────────────────────────────────────────────────
# Production sheet canonicalization & exact-position cleaner
# ─────────────────────────────────────────────────────────────────────────────

def _canonize(s: str) -> str:
    """lowercase, remove spaces/underscores/()-%/[], etc."""
    s = str(s).lower()
    s = re.sub(r'[%/()\[\]\-]+', '', s)
    s = s.replace(' ', '').replace('_', '')
    return s

def _smart_rename_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Map messy headers to canonical labels."""
    df = df.copy()
    ren = {}
    for c in df.columns:
        k = _canonize(c)
        if k in ('time','timestamp','date'):
            ren[c] = 'Timestamp'
        elif 'ethylene' in k or 'ehtylene' in k or 'c2h4' in k:
            ren[c] = 'Ethylene'
        elif 'propylene' in k or 'c3h6' in k:
            ren[c] = 'Propylene'
        elif ('mixedc4' in k) or (('mixed' in k or 'mix' in k) and 'c4' in k) or k == 'c4':
            ren[c] = 'Mixed C4'
        elif 'rpg' in k:
            ren[c] = 'RPG'
        elif 'pfo' in k or 'fueloil' in k or 'pyrolysisfueloil' in k:
            ren[c] = 'PFO'
        elif k == 'hydrogen' or k == 'h2' or 'Hydrogen' in k or 'H2' in k:
            ren[c] = 'Hydrogen'
        elif 'tailgas' in k or ('tail' in k or 'Tail Gas' in k and 'gas' in k):
            ren[c] = 'Tail Gas'
    return df.rename(columns=ren)

def clean_production_df_from_positions(
    raw_df: pd.DataFrame,
    header_row: int = 3,                    # ← row 4 (0-based): YOUR FINAL
    data_start_row: int = 5,                # ← row 6 (0-based)
    # 1-based INCLUDING the Timestamp column BEFORE drop:
    # your zero-based AFTER-drop picks [2,7,10,11,12,15,16] ⇒ +2 ⇒ (4,9,12,13,14,17,18)
    select_positions_1b: Iterable[int] = (4, 9, 12, 13, 14, 17, 18),
    timestamp_colname: str = "Timestamp"
) -> pd.DataFrame:
    """
    - header = header_row
    - data starts at data_start_row
    - first column → datetime index (named 'Timestamp')
    - keep columns by 1-based positions (incl. Timestamp) converted AFTER drop
    - canonicalize names; if duplicates, KEEP LAST one
    """
    df = raw_df.copy()

    # header + data window
    df.columns = df.iloc[header_row]
    df = df.iloc[data_start_row:].reset_index(drop=True)

    # Timestamp from FIRST column (by position), then drop it
    ts = pd.to_datetime(df.iloc[:, 0], errors='coerce')
    df = df.iloc[:, 1:].copy()
    df.index = ts
    df.index.name = timestamp_colname
    df = df[df.index.notna()].sort_index()

    # select positions (1-based incl. Timestamp) → 0-based AFTER drop = p-2
    zero_idx = [p - 2 for p in select_positions_1b]
    zero_idx = [i for i in zero_idx if 0 <= i < df.shape[1]]
    df = df.iloc[:, zero_idx].copy()

    # canonicalize and KEEP LAST duplicate
    df = _smart_rename_to_canonical(df)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep='last')].copy()

    # numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Feature DF builder (no positional renaming)
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_df(df_prod, df_furn, df_naptha, df_gas):
    """
    df_prod already has canonical product names (later renamed to *_prod).
    df_naptha is daily Paraffins (Series or 1-col DF).
    df_gas gets _gas suffixes on component names.
    """
    gas_cols = ['Ethylene', 'Ethane', 'Propylene', 'Propane', 'n-Butane', 'i-Butane']
    gas_rename = {col: f"{col}_gas" for col in gas_cols if col in df_gas.columns}
    df_gas = df_gas.rename(columns=gas_rename)

    if isinstance(df_naptha, pd.Series):
        df_naptha = df_naptha.rename('Paraffins')
    else:
        if df_naptha.shape[1] == 1:
            df_naptha.columns = ['Paraffins']

    return pd.concat([df_prod, df_furn, df_naptha, df_gas], axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# Util cost / recycle cleaners
# ─────────────────────────────────────────────────────────────────────────────

def clean_recycle(df):
    """Clean recycle dataframe → columns Ethane/Propane, indexed by Timestamp."""
    df = df.iloc[2:, 1:]
    df.rename(columns={'Unnamed: 1': 'Timestamp'}, inplace=True)
    df.set_index('Timestamp', inplace=True)
    df.drop(columns=['Unnamed: 2', 'C2 Recycle', 'C3 Recycle', 'C3 to FG Drum'],
            inplace=True, errors='ignore')
    df.rename(columns={'C2 Recycle.1': 'Ethane', 'C3 Recycle.1': 'Propane'}, inplace=True)
    return df

def process_util_cost(path):
    """Process utility cost monthly table → tidy (item, unit, date, value)."""
    cost = pd.read_excel(path, header=1)

    def is_year_month(col):
        try:
            return isinstance(col, float) and 20 <= col < 40
        except Exception:
            return False

    date_columns = [col for col in cost.columns if is_year_month(col)]

    rows = {
        'PN': 1,
        'Gas Feed': 2,
        'Ethylene': 5,
        'Propylene': 6,
        'Mixed C4': 7,
        'TPG (BTX 가치) (T/D)': 8,
        'Fuel Gas': 10,
        'Steam ': 11,
        'Electricity': 12
    }

    all_results = []
    for label, row in rows.items():
        item = cost.iloc[row, 1]
        unit = cost.iloc[row, 2]
        data = cost.loc[row, date_columns]
        df = pd.DataFrame({
            'item': [item] * len(date_columns),
            'unit': [unit] * len(date_columns),
            'date': date_columns,
            'value': data.values
        })
        all_results.append(df)

    final = pd.concat(all_results, ignore_index=True)

    def parse_year_month(val):
        if isinstance(val, float):
            year = int(val)
            month = int(round((val - year) * 100))
            year += 2000 if year < 100 else 0
            try:
                return pd.Timestamp(year=year, month=month, day=1)
            except Exception:
                return pd.NaT
        return pd.NaT

    final['date'] = final['date'].apply(parse_year_month)
    return final

def process_price_csv(path_or_df) -> pd.DataFrame:
    """
    Expect columns like: ['timestamp','IMP_NAPH_PRICE','FEED_STOCK_PRICE1', ...]
    Returns a monthly wide table indexed by month with *canonical* columns:
      ['Ethylene','Propylene','Mixed C4','RPG','PN','Gas Feed','Fuel Gas','Steam','Electricity']
    """
    if isinstance(path_or_df, (str, Path)):
        df = pd.read_csv(path_or_df)
    else:
        df = path_or_df.copy()
    df.columns = pd.Index([str(c).strip() for c in df.columns])

    # timestamp → month index
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp']).copy()
    df['month'] = df['timestamp'].dt.to_period('M').dt.to_timestamp()

    # ---- DEFAULT MAPPING (adjust if needed) ----
    # You can override these by editing this dict.
    MAP = {
        'IMP_NAPH_PRICE':     'PN',           # naphtha import price
        'FEED_STOCK_PRICE1':  'LPG',     # LPG/other feed (part 1)
        'FEED_STOCK_PRICE2':  'C3',     # LPG/other feed (part 2) → will be averaged below
        'PL_EXP_PRICE':       'Propylene',
        'MC4_PRICE':          'Mixed C4',
        'RPG_PRICE':          'RPG',
        'EL_EXP_PRICE':       'Ethylene',
        'POWER_PRICE':        'Electricity',
        'SPS_PRCIE':          'Steam',
        'LNG_PRICE':          'Fuel Gas',     # assume LNG is your fuel-gas proxy
        # Optional extras (not used by PriceProvider, retained if present)
        'H2_PRICE':           'Hydrogen',
        'MX_OG_PRCIE':        'MX Offgas',           #
        'FG_PRICE':           'Tail Gas',
    }

    # melt → map → pivot
    value_cols = [c for c in df.columns if c not in ('timestamp','month')]
    long = df.melt(id_vars=['month'], value_vars=value_cols,
                   var_name='src', value_name='price').dropna(subset=['price'])

    long['item'] = long['src'].map(MAP)
    long = long[long['item'].notna()].copy()

    # If multiple sources map to same canonical (e.g., both FEED_STOCK_PRICE1/2 → Gas Feed),
    # aggregate by mean for that month.
    wide = long.groupby(['month','item'], as_index=False)['price'].mean()
    wide = wide.pivot(index='month', columns='item', values='price').sort_index()

    # Ensure canonical columns exist
    canon = ['Ethylene','Propylene','Mixed C4','RPG','PN','LPG','Fuel Gas','Steam','Electricity', 'MX Offgas', 'Hydrogen', 'Tail Gas']
    for c in canon:
        if c not in wide.columns:
            wide[c] = np.nan

    # order & forward-fill monthly
    wide = wide[canon].ffill()
    return wide

# ─────────────────────────────────────────────────────────────────────────────
# OOP config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataPaths:
    """Paths configuration for data files."""
    input_dir: Path
    inter_dir: Path

    # CSV file
    downloaded_csv: str

    # Excel files
    prod_excel: str
    furn_excel: str
    nap_excel: str
    gas_excel: str
    recycle_excel: str
    # cost_excel: str
    util_excel: str
    input_excel: str
    fresh_excel: str | None = None
    # NEW
    price_csv: str | None = None

    # Pickle files
    prod_pkl: str = "df_production.pkl"
    furn_pkl: str = "furnace.pkl"
    nap_pkl: str = "df_feed_naptha.pkl"
    gas_pkl: str = "df_feed_gas.pkl"
    rec_pkl: str = "df_recycle.pkl"
    fresh_pkl:   str = "fresh_feed.pkl"

    # Headers (prod_header kept for compatibility; prod read uses header=None)
    prod_header: int = 2
    furn_header: int = 2
    nap_header: int = 1
    gas_header: int = 1
    rec_header: int = 4
    fresh_header:int = 3

    # Exact production column selection (1-based, counting Timestamp as col 1)
    # Your final after-drop indices [2,7,10,11,12,15,16] → +2 here:
    prod_select_positions_1b: Tuple[int, ...] = (4, 9, 12, 13, 14, 17, 18)

    @property
    def downloaded_csv_path(self): return self.inter_dir / self.downloaded_csv if self.downloaded_csv else None
    @property
    def prod_excel_path(self): return self.input_dir / self.prod_excel
    @property
    def furn_excel_path(self): return self.input_dir / self.furn_excel
    @property
    def nap_excel_path(self): return self.input_dir / self.nap_excel
    @property
    def gas_excel_path(self): return self.input_dir / self.gas_excel
    @property
    def recycle_excel_path(self): return self.input_dir / self.recycle_excel
    @property
    def cost_excel_path(self): return self.input_dir / self.cost_excel
    @property
    def price_csv_path(self):  return self.input_dir / self.price_csv if self.price_csv else None
    @property
    def util_excel_path(self): return self.input_dir / self.util_excel
    @property
    def prod_pkl_path(self): return self.inter_dir / self.prod_pkl
    @property
    def furn_pkl_path(self): return self.inter_dir / self.furn_pkl
    @property
    def nap_pkl_path(self): return self.inter_dir / self.nap_pkl
    @property
    def gas_pkl_path(self): return self.inter_dir / self.gas_pkl
    @property
    def rec_pkl_path(self): return self.inter_dir / self.rec_pkl
    @property
    def input_excel_path(self): return self.input_dir / self.input_excel
    @property
    def fresh_excel_path(self): return self.input_dir / self.fresh_excel if self.fresh_excel else None
    @property
    def fresh_pkl_path(self):   return self.inter_dir / self.fresh_pkl

@dataclass
class ResampleConfig:
    """Resampling configuration."""
    hour_freq: str = 'h'
    win12_freq: str = '12h'
    win12_offset: str = '9h'

# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

class DataPipeline:
    """Main data loading and processing pipeline."""

    def __init__(self, paths: DataPaths, cfg: ResampleConfig):
        self.paths = paths
        self.cfg = cfg
        self.X_12h = None
        self.Y_12h = None
        self.util_df = None
        self.price_df = None
        self._prod = None
        self._furn = None
        self._nap = None
        self._gas = None
        self._rec = None
        self._fresh = None

    def load_raw(self):
        """Load raw data from Excel or pickle files for data unavailable from PI Server."""
        p = self.paths
        # Production read RAW; custom cleaner sets header & slices columns

        # Below data are not available from PI Server, so load from files.
        self._nap  = load_pickle_or_excel(p.nap_pkl_path,  p.nap_excel_path,  header=p.nap_header)
        self._gas  = load_pickle_or_excel(p.gas_pkl_path,  p.gas_excel_path,  header=p.gas_header)
        
        # Below data are available from PI Server, so skip loading from files.
        # self._prod = load_pickle_or_excel(p.prod_pkl_path, p.prod_excel_path, header=None)
        # self._furn = load_pickle_or_excel(p.furn_pkl_path, p.furn_excel_path, header=p.furn_header)
        # self._rec  = load_pickle_or_excel(p.rec_pkl_path,  p.recycle_excel_path, header=p.rec_header)
        # self._fresh = load_pickle_or_excel(p.fresh_pkl_path, p.fresh_excel_path, header=p.fresh_header)

        return self

    def clean(self):
        """Clean raw dataframes."""
        # Below data are not available from PI Server, so load from files.
        self._nap  = clean_feed_df(self._nap, is_gas=False)
        self._gas  = clean_feed_df(self._gas, is_gas=True)

        # Below data are available from PI Server, so skip cleaning from files.
        # self._rec  = clean_recycle(self._rec)
        # # Exact prod behavior (header_row=3, data_start_row=5)
        # self._prod = clean_production_df_from_positions(
        #     self._prod,
        #     header_row=3,
        #     data_start_row=5,
        #     select_positions_1b=self.paths.prod_select_positions_1b
        # )

        # # Furnace sheet via legacy cleaner
        # self._furn = clean_furnace_or_production_df(self._furn)
        # if self._fresh is not None:
        #     self._fresh = clean_fresh_feed(self._fresh)

        return self

    def load_data(self):
        """Load and clean data for the field available from the PI Server"""
        p = self.paths

        df = pd.read_csv(self.paths.downloaded_csv_path) if p.downloaded_csv_path and p.downloaded_csv_path.exists() else None
        df.columns = pd.Index([str(c).strip() for c in df.columns])
        if df is None:
            raise FileNotFoundError(f"Downloaded CSV not found at {p.downloaded_csv_path}")

        # Below data are available from PI Server, so load from the downloaded CSV.
        self._prod = self._clean_production_df(df)
        self._furn = self._clean_furnace_df(df)
        self._rec = self._clean_recycle_df(df)
        self._fresh = self._clean_fresh_df(df)
        
        # Below data are not available from PI Server, so load from files.
        self._nap = clean_pona_from_pi(df)
        self._gas = self._clean_gas_df(df)

        return self

    def _clean_production_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Custom cleaner for the downloaded production CSV - Refer 'source' sheet from EFOM_input_data_tag_list.xlsx"""
        df = df.copy()
        df.reset_index(drop=True)

        ts_col = df.columns[0]
        out = df[[ts_col]].copy().rename(columns={ts_col: 'Timestamp'})

        out["Ethylene"]   = df["M10FI5265"] - (df["M10FI5070"] / 1000.0)
        out["Propylene"]  = df["M10FIC6301"]
        out["MixedC4"]    = df["M10FIC6362_CORR"]
        out["RPG"]        = df["M15FIC002_CORR"] + df["M10FI6384_CORR"]
        out["PFO"]        = df["PFO_FLOW_TANK"]
        out["C2Recycle"]  = df["M10FIC5201"]
        out["C3Recycle"]  = np.where(df["M10FIC1008"] == -1, 0.0,
                                 df["M10FIC6101"])
        out["Hydrogen"]   = df["M10FI4246_CORR"] / 1000.0

        to_hmu = np.where(df["M10FIC3652"] > 0, df["M10FIC3652"], 0.0)
        out["Tail Gas"]   = df["M10FIC4247_CORR"] / 1000.0 + df["M10FIC3587_CORR"] + df["M10FI3588_CORR"] + to_hmu

        if 'Timestamp' in out.columns:
            out['Timestamp'] = pd.to_datetime(out['Timestamp'], errors='coerce')
            out['Timestamp'] = out['Timestamp'].dt.tz_localize(None)
            out = out.set_index('Timestamp')
        
        return out

    def _clean_furnace_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Custom cleaner for the downloaded furnace CSV"""
        df = df.copy()
        df.reset_index(drop=True)

        ts_col = df.columns[0]
        out = df[[ts_col]].copy().rename(columns={ts_col: 'Timestamp'})

        p = self.paths
        mapping = pd.read_excel(p.input_excel_path, sheet_name='furnace_tag_mapping', usecols=["tags", "output_column_name"]).dropna()
        mapping["tags"] = mapping["tags"].str.strip()
        mapping["output_column_name"] = mapping["output_column_name"].str.strip()

        available_tags = [c for c in mapping["tags"].tolist() if c in df.columns]
        tag_df = df[available_tags].copy()

        rename_dict = dict(zip(mapping["tags"], mapping["output_column_name"]))
        tag_df = tag_df.rename(columns=rename_dict)
        
        out = pd.concat([out, tag_df], axis=1)

        if 'Timestamp' in out.columns:
            out['Timestamp'] = pd.to_datetime(out['Timestamp'], errors='coerce')
            out['Timestamp'] = out['Timestamp'].dt.tz_localize(None)
            out = out.set_index('Timestamp')

        return out

    def _clean_recycle_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Custom cleaner for the downloaded recycle CSV - Refer 'source' sheet from EFOM_input_data_tag_list.xlsx"""
        df = df.copy()
        df.reset_index(drop=True)

        ts_col = df.columns[0]
        out = df[[ts_col]].copy().rename(columns={ts_col: 'Timestamp'})

        out["Ethane"]  = df["M10FIC5201"]
        out["Propane"] = df["M10FIC6101"] - np.where(df["M10FIC1008"] == -1, 0.0, df["M10FIC1008"] / 1000.0)

        if 'Timestamp' in out.columns:
            out['Timestamp'] = pd.to_datetime(out['Timestamp'], errors='coerce')
            out['Timestamp'] = out['Timestamp'].dt.tz_localize(None)
            out = out.set_index('Timestamp')

        return out
    
    def _clean_fresh_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Custom cleaner for the downloaded fresh feed CSV - Refer 'source' sheet from EFOM_input_data_tag_list.xlsx"""
        df = df.copy()
        df.reset_index(drop=True)

        ts_col = df.columns[0]
        out = df[[ts_col]].copy().rename(columns={ts_col: 'Timestamp'})

        out["FreshFeed_C3 LPG"]     = df["M10FIC1003"]
        out["FreshFeed_MX Offgas"]  = df["M10FI1016"]
        out["feed_qty"]             = out[["FreshFeed_C3 LPG", "FreshFeed_MX Offgas"]].sum(axis=1)

        if 'Timestamp' in out.columns:
            out['Timestamp'] = pd.to_datetime(out['Timestamp'], errors='coerce')
            out['Timestamp'] = out['Timestamp'].dt.tz_localize(None)
            out = out.set_index('Timestamp')

        return out
    
    def _clean_nap_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Custom cleaner for the downloaded naptha CSV - Refer 'source' sheet from EFOM_input_data_tag_list.xlsx"""
        df = df.copy()
        df.reset_index(drop=True)

        ts_col = df.columns[0]
        out = df[[ts_col]].copy().rename(columns={ts_col: 'Timestamp'})

        out["n-Paraffin"] = df["M10L41004_C4 n-Paraffin(vol%)"] + df["M10L41004_C5 n-Paraffin(vol%)"] + df["M10L41004_C6 n-Paraffin(vol%)"] + df["M10L41004_C7 n-Paraffin(vol%)"] + df["M10L41004_C8 n-Paraffin(vol%)"] + df["M10L41004_C9 n-Paraffin(vol%)"] + df["M10L41004_C10 n-Paraffin(vol%)"] + df["M10L41004_C11+ n-Paraffin(vol%)"]
        out["i-Paraffin"] = df["M10L41004_C4 i-Paraffin(vol%)"] + df["M10L41004_C5 i-Paraffin(vol%)"] + df["M10L41004_C6 i-Paraffin(vol%)"] + df["M10L41004_C7 i-Paraffin(vol%)"] + df["M10L41004_C8 i-Paraffin(vol%)"] + df["M10L41004_C9 i-Paraffin(vol%)"] + df["M10L41004_C10 i-Paraffin(vol%)"] + df["M10L41004_C11+ i-Paraffin(vol%)"]
        out["Paraffins"]  = out["n-Paraffin"] + out["i-Paraffin"]

        # TODO: Missing columns
        # Density, API도, D#IBP, D#10%, D#30%, D#50%, D#70%, D#90%, D#FBP, Sulfur, VP mini, Total Wafer, Olefin, Naphthene, Aromatic, As, Pb, V, N, Ni, Zn, Mercury, R-Cl, Total Nitrogen, MeOH, Total Oxygenates

        if 'Timestamp' in out.columns:
            out['Timestamp'] = pd.to_datetime(out['Timestamp'], errors='coerce')
            out['Timestamp'] = out['Timestamp'].dt.tz_localize(None)
            out = out.set_index('Timestamp')

        return out
    
    def _pick(df, *tokens):
        tokens = [t.lower() for t in tokens]
        for c in df.columns:
            s = str(c).lower()
            if all(t in s for t in tokens):
                return c
        return None

    
    # def _clean_gas_df(self, df: pd.DataFrame) -> pd.DataFrame:
    #     """Custom cleaner for the downloaded gas CSV - Refer 'source' sheet from EFOM_input_data_tag_list.xlsx"""
    #     df = df.copy()
    #     df.reset_index(drop=True)

    #     ts_col = df.columns[0]
    #     out = df[[ts_col]].copy().rename(columns={ts_col: 'Timestamp'})

    #     out["Ethylene"]  = df["M10G31003_Ethylene(mol%)"]
    #     out["Ethane"]    = df["M10G31003_Ethane(mol%)"]
    #     out["Propylene"] = df["M10G31003_Propylene(mol%)"]
    #     out["Propane"]   = df["M10G31003_Propane(mol%)"]
    #     out["n-Butane"]  = df["M10G31003_n-Butane(mol%)"]
    #     out["i-Butane"]  = df["M10G31003_i-Butane(mol%)"]

    #     # TODO: Missing columns
    #     # out["Total Sulfur"]     = df["xxxxxx"]
    #     # out["MeOH"]             = df["xxxxxx"]
    #     # out["Total Oxygenates"] = df["xxxxxx"]

    #     if 'Timestamp' in out.columns:
    #         out['Timestamp'] = pd.to_datetime(out['Timestamp'], errors='coerce')
    #         out['Timestamp'] = out['Timestamp'].dt.tz_localize(None)
    #         out = out.set_index('Timestamp')

    #     return out

    def _clean_gas_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().reset_index(drop=True)
        ts_col = df.columns[0]
        out = df[[ts_col]].copy().rename(columns={ts_col: 'Timestamp'})

        # prefer exact names; fallback to token search
        def col(name, *tokens):
            return name if name in df.columns else _pick(df, *tokens)

        c = {
            'Ethylene':  col("M10G31003_Ethylene(mol%)",  "G31003","ethylene","mol"),
            'Ethane':    col("M10G31003_Ethane(mol%)",    "G31003","ethane","mol"),
            'Propylene': col("M10G31003_Propylene(mol%)", "G31003","propylene","mol"),
            'Propane':   col("M10G31003_Propane(mol%)",   "G31003","propane","mol"),
            'n-Butane':  col("M10G31003_n-Butane(mol%)",  "G31003","n-butane","mol"),
            'i-Butane':  col("M10G31003_i-Butane(mol%)",  "G31003","i-butane","mol"),
        }

        for k, src in c.items():
            if src is not None:
                out[k] = pd.to_numeric(df[src], errors='coerce')

        out['Timestamp'] = pd.to_datetime(out['Timestamp'], errors='coerce').dt.tz_localize(None)
        out = out.set_index('Timestamp')
        return out


    def resample(self):
        # 1) Numeric copies
        prod_num = self._prod.apply(pd.to_numeric, errors='coerce')
        furn_num = self._furn.apply(pd.to_numeric, errors='coerce')
        rec_num  = self._rec.apply(pd.to_numeric, errors='coerce')

        # 2) 12h resampling (label/right, trailing bins)
        prod_12h = prod_num.resample(
            self.cfg.win12_freq, offset=self.cfg.win12_offset, label='right', closed='right'
        ).mean()
        furn_12h = clean_furnace_columns(
            furn_num.resample(
                self.cfg.win12_freq, offset=self.cfg.win12_offset, label='right', closed='right'
            ).mean()
        )
        rec_12h  = rec_num.resample(
            self.cfg.win12_freq, offset=self.cfg.win12_offset, label='right', closed='right'
        ).mean()

        # 3) Canonicalize production and rename → *_prod
        prod_12h = _smart_rename_to_canonical(prod_12h)
        to_prod = {
            'Ethylene':'Ethylene_prod','Propylene':'Propylene_prod','Mixed C4':'MixedC4_prod',
            'RPG':'RPG_prod','PFO':'PFO_prod','Hydrogen':'Hydrogen_prod','Tail Gas':'Tail_Gas_prod',
        }
        prod_12h = prod_12h.rename(columns={k:v for k,v in to_prod.items() if k in prod_12h.columns})

        # Recycle → *_prod
        prod_12h = pd.concat([prod_12h, rec_12h], axis=1).rename(columns={
            'Ethane':'Ethane_prod', 'Propane':'Propane_prod'
        })

        # 4) Now do PONA/gas daily → 12h, aligned to prod_12h.index
        pona_cols = [c for c in ['Paraffins','Olefins','Naphthenes','Aromatics'] if c in self._nap.columns]
        nap_daily = self._nap[pona_cols].resample('D').mean()
        gas_daily = self._gas.resample('D').mean()

        nap_12h = reindex_ffill(nap_daily, prod_12h.index)
        gas_12h = reindex_ffill(gas_daily, prod_12h.index)

        # Prefix naphtha columns
        if isinstance(nap_12h, pd.Series):
            nap_12h = nap_12h.rename('Paraffins')
        else:
            nap_12h = nap_12h.rename(columns={c: f'{c}' for c in nap_12h.columns})

        # 5) Build features
        self.X_12h = build_feature_df(prod_12h, furn_12h, nap_12h, gas_12h)
        try:
            build_virtual_rcots_inplace(self.X_12h)
        except Exception:
            pass

        # 6) Targets (t+1)
        target_pool = [
            'Ethylene_prod','Propylene_prod','MixedC4_prod','RPG_prod','PFO_prod',
            'Hydrogen_prod','Tail_Gas_prod','Ethane_prod','Propane_prod'
        ]
        available = [c for c in target_pool if c in prod_12h.columns]
        if not available:
            raise RuntimeError("No target columns found in production data after canonical mapping.")
        self.Y_12h = prod_12h[available].shift(-1)
        self.Y_12h.columns = [f"{c}_t+1" for c in self.Y_12h.columns]

        # 7) Fresh feed (optional)
        if self._fresh is not None:
            fresh_12h = (self._fresh
                        .resample(self.cfg.win12_freq, offset=self.cfg.win12_offset, label='right', closed='right')
                        .mean())
            fresh_12h = fresh_12h.reindex(self.X_12h.index).ffill()
            self.X_12h = pd.concat([self.X_12h, fresh_12h], axis=1)

        return self

    def build_util(self, util_feature_rename: Dict[str, str], util_target_rename: Dict[str, str]):
        util = pd.read_excel(self.paths.util_excel_path, header=4)
        util = util.drop([0], axis=0)
        util.columns.values[1] = "Timestamp"
        util['Timestamp'] = pd.to_datetime(util['Timestamp'])
        util = util.set_index('Timestamp')
        util = util.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
        self.util_df = util.rename(columns={**util_feature_rename, **util_target_rename})
        return self

    def build_prices(self):
        if getattr(self.paths, 'price_csv_path', None):
            self.price_df = process_price_csv(self.paths.price_csv_path)
        else:
            self.price_df = process_util_cost(self.paths.cost_excel_path)
        return self
    
    def build_prices_from_df(self):
        p = self.paths

        df = pd.read_csv(self.paths.downloaded_csv_path) if p.downloaded_csv_path and p.downloaded_csv_path.exists() else None
        if df is None:
            raise FileNotFoundError(f"Downloaded CSV not found at {p.downloaded_csv_path}")
        
        df.columns = pd.Index([str(c).strip() for c in df.columns])

        
        df = df.copy()
        df.reset_index(drop=True)

        ts_col = df.columns[0]
        out = df[[ts_col]].copy().rename(columns={ts_col: 'Timestamp'})

        mapping = pd.read_excel(p.input_excel_path, sheet_name='price_tag_mapping', usecols=["tags", "output_column_name"]).dropna()
        mapping["tags"] = mapping["tags"].str.strip()
        mapping["output_column_name"] = mapping["output_column_name"].str.strip()

        available_tags = [c for c in mapping["tags"].tolist() if c in df.columns]
        tag_df = df[available_tags].copy()

        rename_dict = dict(zip(mapping["tags"], mapping["output_column_name"]))
        tag_df = tag_df.rename(columns=rename_dict)
        
        out = pd.concat([out, tag_df], axis=1)

        if 'Timestamp' in out.columns:
            # Convert datetime
            out['Timestamp'] = pd.to_datetime(out['Timestamp'], errors='coerce')
            out['Timestamp'] = out['Timestamp'].dt.tz_localize(None)

            # Convert into YYYY-MM-DD format
            out['month'] = out['Timestamp'].dt.strftime('%Y-%m-%d')

            # Setting the column as an index
            out = out.drop(columns=['Timestamp']).set_index('month')

        self.price_df = out

        return self

    def run(self, util_feature_rename: Dict[str, str], util_target_rename: Dict[str, str]):
        return (self.load_raw()
                .clean()
                .load_data()
                .resample()
                .build_util(util_feature_rename, util_target_rename)
                # .build_prices())
                .build_prices_from_df())

    def artifacts(self) -> Dict[str, Any]:
        return {
            'X_12h': self.X_12h,
            'Y_12h': self.Y_12h,
            'util_df': self.util_df,
            'price_df': self.price_df
        }

# ─────────────────────────────────────────────────────────────────────────────
# Convenience
# ─────────────────────────────────────────────────────────────────────────────

def load_lims_data(nap_path: str, gas_path: str, header: int = 1) -> pd.DataFrame:
    """Alias for load_feed_data (backward compat)."""
    return load_feed_data(nap_path=nap_path, gas_path=gas_path, header=header)


__all__ = [
    'load_pickle_or_excel',
    'load_feed_data',
    'load_lims_data',
    'clean_feed_df',
    'clean_furnace_columns',
    'resample_and_slice',
    'reindex_ffill',
    'build_feature_df',
    'clean_production_df_from_positions',
    'clean_furnace_or_production_df',
    'clean_recycle',
    'process_util_cost',
    'process_price_csv',
    'DataPaths',
    'ResampleConfig',
    'DataPipeline',
    'build_virtual_rcots_inplace'
]
