# =========================
# Data Loading Module - Fixed Version
# =========================
from dataclasses import dataclass
from typing import Dict, Optional, Any
from pathlib import Path
import pandas as pd
import numpy as np


def load_pickle_or_excel(
    pkl_path: Path,
    excel_path: Optional[Path] = None,
    refresh_if_excel_newer: bool = True,
    **read_excel_kwargs
):
    """
    Load a DataFrame from pickle if available; otherwise from Excel and cache to pickle.
    If 'refresh_if_excel_newer' and Excel mtime > pickle mtime, refresh the pickle.
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


def load_feed_data(nap_path=None, gas_path=None, paths=None, header=1):
    """
    Load and prepare feed data for SRTO pipeline.
    
    Parameters:
        nap_path: Path to naphtha Excel file
        gas_path: Path to gas Excel file  
        paths: Optional DataPaths instance
        header: Header row for Excel files (default 1)
    
    Returns:
        Merged DataFrame with date column and feed compositions
    """
    # Resolve paths
    if paths is not None:
        try:
            nap_path = nap_path or paths.nap_excel_path
            gas_path = gas_path or paths.gas_excel_path
        except Exception:
            pass

    if nap_path is None or gas_path is None:
        raise ValueError("nap_path and gas_path (or paths) must be provided")

    # Load naphtha feed
    lims = pd.read_excel(nap_path, header=header).fillna(0.0)
    feed_naph = lims.rename(columns={'Unnamed: 2': 'date'})

    # Load gas feed
    gas_feed = pd.read_excel(gas_path, header=header).fillna(0.0)
    gas_feed = gas_feed.rename(columns={'Unnamed: 2': 'date'})
    
    # Select gas components
    gas_cols = ['date', 'Ethylene', 'Ethane', 'Propylene', 'Propane', 'n-Butane', 'i-Butane']
    available_gas_cols = [col for col in gas_cols if col in gas_feed.columns]
    feed_gas = gas_feed[available_gas_cols]

    # Ensure datetime
    feed_naph['date'] = pd.to_datetime(feed_naph['date'], errors='coerce')
    feed_gas['date'] = pd.to_datetime(feed_gas['date'], errors='coerce')

    # Merge on date
    merged_lims = pd.merge(feed_naph, feed_gas, on='date', how='inner')

    cols = [c for c in ['Ethylene', 'Ethane', 'Propylene', 'Propane', 'n-Butane', 'i-Butane'] if c in merged_lims.columns]

    # ensure chronological order
    merged_lims = merged_lims.sort_values('date')

    # mark rows where all gas comps sum to 0 as missing
    zero_rows = (merged_lims[cols].sum(axis=1) == 0)
    merged_lims.loc[zero_rows, cols] = np.nan

    # forward-fill from prior valid row; optionally backfill the very first block
    merged_lims[cols] = merged_lims[cols].ffill().bfill()

    
    return merged_lims


def clean_feed_df(df, is_gas=False):
    """
    Clean feed dataframes exported from Excel into a timeseries-indexed numeric DF.
    """
    df = df.copy()
    
    # Handle header rows
    if isinstance(df.index, pd.RangeIndex) and len(df) > 2:
        df = df.drop([0, 1], axis=0)
        df = df.iloc[:, 2:]
        if 'Unnamed: 2' in df.columns:
            df = df.rename(columns={'Unnamed: 2': 'Timestamp'})
    
    # Handle timestamp column
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df = df.set_index('Timestamp')
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    
    # Drop unused columns
    df = df.drop(columns=['Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5'], errors='ignore')
    
    # Convert to numeric
    if is_gas:
        gas_cols = ['Ethylene', 'Ethane', 'Propylene', 'Propane', 'n-Butane', 'i-Butane']
        for col in gas_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        if 'Paraffins' in df.columns:
            df['Paraffins'] = pd.to_numeric(df['Paraffins'], errors='coerce')
    
    return df


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


def resample_and_slice(df_numeric, freq, col_slice=None, **resample_kwargs):
    """Resample dataframe and optionally slice columns."""
    df = df_numeric.resample(freq, **resample_kwargs).mean()
    if col_slice is not None:
        df = df.iloc[:, col_slice].copy()
    return df


def reindex_ffill(df_daily, target_index):
    """Reindex and forward fill."""
    return df_daily.reindex(target_index, method='ffill')


def _mean_over_existing(df: pd.DataFrame, cols):
        """Return row-wise mean over existing columns (ignore missing columns)."""
        valid = [c for c in cols if c in df.columns]
        if not valid:
                return pd.Series(0.0, index=df.index)
        return df[valid].mean(axis=1)


def build_virtual_rcots_inplace(X: pd.DataFrame) -> None:
        """Create virtual RCOT_naphtha_chamber{4..6} and RCOT_gas_chamber{4..6}.

        Logic:
            - For chambers 4..6 there are 8 coil RCOTs named like
                'RCOT #1_naphtha_chamber4' ... 'RCOT #8_gas_chamber4'.
            - If both naphtha and gas feed are present for the chamber (Naphtha_chamber{ch} > 0
                and Gas Feed_chamber{ch} > 0) then compute separate means for odd (naphtha) and even (gas)
                coil groups.
            - If only one feed is present, use mean over all 8 coils for that feed's virtual RCOT.
        """
        for ch in (4, 5, 6):
                naph_odds = [f"RCOT #{i}_naphtha_chamber{ch}" for i in (1, 3, 5, 7)]
                gas_evens = [f"RCOT #{i}_gas_chamber{ch}" for i in (2, 4, 6, 8)]

                has_naph = X.get(f"Naphtha_chamber{ch}", pd.Series(0, index=X.index)) > 0
                has_gas = X.get(f"Gas Feed_chamber{ch}", pd.Series(0, index=X.index)) > 0

                all8 = _mean_over_existing(X, naph_odds + gas_evens)
                navg = _mean_over_existing(X, naph_odds)
                gavg = _mean_over_existing(X, gas_evens)

                X[f"RCOT_naphtha_chamber{ch}"] = np.where(has_naph & has_gas, navg,
                                                                                                    np.where(has_naph, all8, 0.0))
                X[f"RCOT_gas_chamber{ch}"] = np.where(has_naph & has_gas, gavg,
                                                                                            np.where(has_gas, all8, 0.0))


def build_feature_df(df_prod, df_furn, df_naptha, df_gas):
    """Build feature dataframe from components."""
    df_prod = df_prod.rename(columns={
        df_prod.columns[0]: 'Ethylene_prod',
        df_prod.columns[1]: 'Propylene_prod',
        df_prod.columns[2]: 'MixedC4_prod',
        df_prod.columns[3]: 'RPG_prod',
        df_prod.columns[4]: 'PFO_prod',
    })
    
    gas_cols = ['Ethylene', 'Ethane', 'Propylene', 'Propane', 'n-Butane', 'i-Butane']
    gas_rename = {col: col + '_gas' for col in gas_cols if col in df_gas.columns}
    df_gas = df_gas.rename(columns=gas_rename)
    
    return pd.concat([df_prod, df_furn, df_naptha.rename('Naptha_Paraffins'), df_gas], axis=1)


def clean_furnace_or_production_df(df, n_drop=3, timestamp_colname="Timestamp"):
    """Clean furnace or production dataframe."""
    df = df.copy()
    df.columns = df.iloc[0]
    df = df.drop(df.index[0]).reset_index(drop=True)
    df = df.drop(range(n_drop), axis=0).reset_index(drop=True)
    df.columns.values[0] = timestamp_colname
    df[timestamp_colname] = pd.to_datetime(df[timestamp_colname])
    df = df.set_index(timestamp_colname)
    df = df.iloc[:, 1:]
    return df


def clean_recycle(df):
    """Clean recycle dataframe."""
    df = df.iloc[2:, 1:]
    df.rename(columns={'Unnamed: 1': 'Timestamp'}, inplace=True)
    df.set_index('Timestamp', inplace=True)
    df.drop(columns=['Unnamed: 2', 'C2 Recycle', 'C3 Recycle', 'C3 to FG Drum'], 
            inplace=True, errors='ignore')
    df.rename(columns={'C2 Recycle.1': 'Ethane', 'C3 Recycle.1': 'Propane'}, inplace=True)
    return df


def process_util_cost(path):
    """Process utility cost data."""
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


# ==================== OOP Classes ====================

@dataclass
class DataPaths:
    """Paths configuration for data files."""
    input_dir: Path
    inter_dir: Path
    
    # Excel files
    prod_excel: str
    furn_excel: str
    nap_excel: str
    gas_excel: str
    recycle_excel: str
    cost_excel: str
    util_excel: str
    
    # Pickle files
    prod_pkl: str = "df_production.pkl"
    furn_pkl: str = "furnace.pkl"
    nap_pkl: str = "df_feed_naptha.pkl"
    gas_pkl: str = "df_feed_gas.pkl"
    rec_pkl: str = "df_recycle.pkl"
    
    # Headers
    prod_header: int = 2
    furn_header: int = 2
    nap_header: int = 1
    gas_header: int = 1
    rec_header: int = 4
    
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


@dataclass
class ResampleConfig:
    """Resampling configuration."""
    hour_freq: str = 'H'
    win12_freq: str = '12H'
    win12_offset: str = '9H'


class DataPipeline:
    """Main data loading and processing pipeline."""
    
    def __init__(self, paths: DataPaths, cfg: ResampleConfig):
        self.paths = paths
        self.cfg = cfg
        # Outputs
        self.X_12h = None
        self.Y_12h = None
        self.util_df = None
        self.price_df = None
        # Internal frames
        self._prod = None
        self._furn = None
        self._nap = None
        self._gas = None
        self._rec = None
    
    def load_raw(self):
        """Load raw data from files."""
        p = self.paths
        self._prod = load_pickle_or_excel(p.prod_pkl_path, p.prod_excel_path, header=p.prod_header)
        self._furn = load_pickle_or_excel(p.furn_pkl_path, p.furn_excel_path, header=p.furn_header)
        self._nap = load_pickle_or_excel(p.nap_pkl_path, p.nap_excel_path, header=p.nap_header)
        self._gas = load_pickle_or_excel(p.gas_pkl_path, p.gas_excel_path, header=p.gas_header)
        self._rec = load_pickle_or_excel(p.rec_pkl_path, p.recycle_excel_path, header=p.rec_header)
        return self
    
    def clean(self):
        """Clean loaded data."""
        self._nap = clean_feed_df(self._nap, is_gas=False)
        self._gas = clean_feed_df(self._gas, is_gas=True)
        self._rec = clean_recycle(self._rec)
        self._prod = clean_furnace_or_production_df(self._prod)
        self._furn = clean_furnace_or_production_df(self._furn)
        return self
    
    def resample(self):
        """Resample data to target frequencies."""
        # Daily feeds
        nap_daily = self._nap['Paraffins'].resample('D').mean()
        gas_daily = self._gas.resample('D').mean()
        
        # Numeric conversion
        prod_num = self._prod.apply(pd.to_numeric, errors='coerce')
        furn_num = self._furn.apply(pd.to_numeric, errors='coerce')
        rec_num = self._rec.apply(pd.to_numeric, errors='coerce')
        
        # Hourly resampling
        prod_hour = resample_and_slice(prod_num, self.cfg.hour_freq, np.r_[2:5, 7, 8])
        furn_hour = clean_furnace_columns(furn_num.resample(self.cfg.hour_freq).mean())
        
        # 12H resampling
        prod_12h = resample_and_slice(prod_num, self.cfg.win12_freq, 
                                      np.r_[2:5, 7, 8], offset=self.cfg.win12_offset)
        furn_12h = clean_furnace_columns(
            furn_num.resample(self.cfg.win12_freq, offset=self.cfg.win12_offset).mean()
        )
        rec_12h = rec_num.resample(self.cfg.win12_freq, offset=self.cfg.win12_offset).mean()
        
        # Append recycle
        prod_12h = pd.concat([prod_12h, rec_12h], axis=1).rename(columns={
            'Ethane': 'Ethane_prod', 'Propane': 'Propane_prod'
        })
        
        # Forward fill feeds
        nap_12h = reindex_ffill(nap_daily, prod_12h.index)
        gas_12h = reindex_ffill(gas_daily, prod_12h.index)
        
        # Rename production columns
        prod_rename = {
            'Ehtylene': 'Ethylene_prod', 'Ethylene': 'Ethylene_prod',
            'Propylene': 'Propylene_prod', 'Mixed C4': 'MixedC4_prod',
            'RPG': 'RPG_prod', 'PFO': 'PFO_prod', 'Hydrogen': 'Hydrogen_prod', 'Tail Gas': 'Tail_Gas_prod'
        }
        prod_12h = prod_12h.rename(columns=prod_rename)
        
        # Build features and targets
        self.X_12h = build_feature_df(prod_12h, furn_12h, nap_12h, gas_12h)
        # ensure virtual RCOTs are present for chambers 4..6
        try:
            build_virtual_rcots_inplace(self.X_12h)
        except Exception:
            # be tolerant for missing columns during incremental development
            pass
        self.Y_12h = prod_12h[['Ethylene_prod', 'Propylene_prod', 'MixedC4_prod', 
                               'RPG_prod', 'Ethane_prod', 'Propane_prod']].shift(-1)
        self.Y_12h.columns = [f"{c}_t+1" for c in self.Y_12h.columns]
        
        return self
    
    def build_util(self, util_feature_rename: Dict[str, str], 
                   util_target_rename: Dict[str, str]):
        """Build utility dataframe."""
        util = pd.read_excel(self.paths.util_excel_path, header=4)
        util = util.drop([0], axis=0)
        util.columns.values[1] = "Timestamp"
        util['Timestamp'] = pd.to_datetime(util['Timestamp'])
        util = util.set_index('Timestamp')
        util = util.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
        self.util_df = util.rename(columns={**util_feature_rename, **util_target_rename})
        return self
    
    def build_prices(self):
        """Build price dataframe."""
        self.price_df = process_util_cost(self.paths.cost_excel_path)
        return self
    
    def run(self, util_feature_rename: Dict[str, str], 
            util_target_rename: Dict[str, str]):
        """Run complete pipeline."""
        return (self.load_raw()
                .clean()
                .resample()
                .build_util(util_feature_rename, util_target_rename)
                .build_prices())
    
    def artifacts(self) -> Dict[str, Any]:
        """Return pipeline artifacts."""
        return {
            'X_12h': self.X_12h,
            'Y_12h': self.Y_12h,
            'util_df': self.util_df,
            'price_df': self.price_df
        }


# ==================== Convenience Functions ====================

def load_lims_data(nap_path: str, gas_path: str, header: int = 1) -> pd.DataFrame:
    """
    Convenience function to load LIMS data.
    Alias for load_feed_data for backward compatibility.
    """
    return load_feed_data(nap_path=nap_path, gas_path=gas_path, header=header)


# Make sure all functions are exported
__all__ = [
    'load_pickle_or_excel',
    'load_feed_data',
    'load_lims_data',
    'clean_feed_df',
    'clean_furnace_columns',
    'resample_and_slice',
    'reindex_ffill',
    'build_feature_df',
    'clean_furnace_or_production_df',
    'clean_recycle',
    'process_util_cost',
    'DataPaths',
    'ResampleConfig',
    'DataPipeline'
]