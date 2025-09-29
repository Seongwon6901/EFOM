# ml_predictor.py
# -*- coding: utf-8 -*-
"""
OOP ML predictor for EFOM/HCHEM:
- Adds virtual RCOTs (for chambers 4–6) from real coil tags
- Builds lag features
- Fits LGBM per target with imputation
- Predicts per-row or for a whole DataFrame

Usage (typical):
    from ml_predictor import MLPredictor

    targets = [
        'Ethylene_prod_t+1', 'Propylene_prod_t+1',
        'MixedC4_prod_t+1', 'RPG_prod_t+1'
    ]

    ml = MLPredictor(target_cols=targets)
    ml.fit(X_12h, Y_12h)

    # last plant-state row (make sure virtual RCOTs exist)
    x_last = X_12h.sort_index().iloc[-1]
    preds  = ml.predict_row(x_last)
    print(preds)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Sequence, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import set_config
set_config(transform_output="pandas")

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from lightgbm import LGBMRegressor

try:
    import joblib
except Exception:
    joblib = None



# add near other module-level defaults
GAS_COLS_DEFAULT = [
    'Ethane_gas','Propane_gas','Ethylene_gas',
    'Propylene_gas','n-Butane_gas','i-Butane_gas'
]

# g/mol
GAS_MW = {
    'Ethane_gas': 30.07,
    'Propane_gas': 44.10,
    'Ethylene_gas': 28.05,
    'Propylene_gas': 42.08,
    'n-Butane_gas': 58.12,
    'i-Butane_gas': 58.12,
}

PONA_COLS_DEFAULT = ['Paraffins','Olefins','Naphthenes','Aromatics']

# ──────────────────────────────────────────────────────────────────────────────
# Schema & helper lists
# ──────────────────────────────────────────────────────────────────────────────

# Base RCOT schema for furnaces 1–6 (virtuals for 4–6)
def _build_rcot_schema() -> Dict[int, Dict[str, Dict[str, List[str]]]]:
    schema: Dict[int, Dict[str, Dict[str, List[str]]]] = {
        1: {'virt': ['RCOT_chamber1'], 'real': {'RCOT_chamber1': ['RCOT_chamber1']}},
        2: {'virt': ['RCOT_chamber2'], 'real': {'RCOT_chamber2': ['RCOT_chamber2']}},
        3: {'virt': ['RCOT_chamber3'], 'real': {'RCOT_chamber3': ['RCOT_chamber3']}},
    }
    for ch in (4, 5, 6):
        odds  = [f"RCOT #{i}_naphtha_chamber{ch}" for i in (1, 3, 5, 7)]
        evens = [f"RCOT #{i}_gas_chamber{ch}"    for i in (2, 4, 6, 8)]
        schema[ch] = {
            'virt': [f'RCOT_naphtha_chamber{ch}', f'RCOT_gas_chamber{ch}'],
            'real': {
                f'RCOT_naphtha_chamber{ch}': odds,
                f'RCOT_gas_chamber{ch}':     evens,
            },
        }
    return schema

RCOT_SCHEMA = _build_rcot_schema()

# Feed tags we care about
FEED_COLS_DEFAULT = [
    'Naphtha_chamber1','Naphtha_chamber2','Naphtha_chamber3',
    'Naphtha_chamber4','Naphtha_chamber5','Naphtha_chamber6',
    'Gas Feed_chamber4','Gas Feed_chamber5','Gas Feed_chamber6'
]

# Flatten of *real* RCOT columns (for bookkeeping)
RCOT_ALL_REAL_COLS: List[str] = []
for info in RCOT_SCHEMA.values():
    for real_list in info['real'].values():
        RCOT_ALL_REAL_COLS.extend(real_list)


# ──────────────────────────────────────────────────────────────────────────────
# Public helpers (reused elsewhere in your project)
# ──────────────────────────────────────────────────────────────────────────────

def geometry_from_row(row: pd.Series) -> str:
    """Detect plant geometry from feed pattern."""
    n = sum(float(row.get(f'Naphtha_chamber{ch}', 0.0)) for ch in range(1, 7))
    g = sum(float(row.get(f'Gas Feed_chamber{ch}', 0.0)) for ch in (4, 5, 6))
    if n > 0 and g > 0:
        return 'GF_HYB_NAPH'
    return 'GF_GAS' if g > 0 else 'LF_NAPH'


def compute_ratio(row: pd.Series) -> float:
    """Naphtha ratio = naphtha / (naphtha + gas) over active furnaces."""
    n = sum(float(row.get(c, 0.0)) for c in FEED_COLS_DEFAULT if 'Naphtha' in c)
    g = sum(float(row.get(c, 0.0)) for c in FEED_COLS_DEFAULT if 'Gas Feed' in c)
    s = n + g
    return (n / s) if s > 0 else 0.0


def _mean_over_existing(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    """Row-wise mean over columns that exist (ignores missing). Returns Series aligned to df.index."""
    valid = [c for c in cols if c in df.columns]
    if not valid:
        # all missing → return zeros aligned to index
        return pd.Series(0.0, index=df.index)
    return df[valid].mean(axis=1)


def build_virtual_rcots_inplace(X: pd.DataFrame) -> None:
    """
    Create RCOT_naphtha/gas_chamber4..6 from coil tags.
    For single-feed cases, broadcast all 8 coils to whichever feed is present.
    """
    for ch in (4, 5, 6):
        naph_odds = [f"RCOT #{i}_naphtha_chamber{ch}" for i in (1, 3, 5, 7)]
        gas_evens = [f"RCOT #{i}_gas_chamber{ch}"     for i in (2, 4, 6, 8)]

        has_naph = X.get(f"Naphtha_chamber{ch}", pd.Series(0, index=X.index)) > 0
        has_gas  = X.get(f"Gas Feed_chamber{ch}", pd.Series(0, index=X.index)) > 0

        all8 = _mean_over_existing(X, naph_odds + gas_evens)
        navg = _mean_over_existing(X, naph_odds)
        gavg = _mean_over_existing(X, gas_evens)

        X[f"RCOT_naphtha_chamber{ch}"] = np.where(has_naph & has_gas, navg,
                                                  np.where(has_naph, all8, 0.0))
        X[f"RCOT_gas_chamber{ch}"] = np.where(has_naph & has_gas, gavg,
                                              np.where(has_gas, all8, 0.0))


def active_rcots_for_mean(row: pd.Series,
                          min_feed_tph: float = 0.0,
                          min_rcot_naph: Optional[float] = None,
                          min_rcot_gas: Optional[float] = None) -> Tuple[float, float]:
    """
    Representative RCOTs (mean) for naphtha & gas over *active* furnaces.
    Returns (rc_n, rc_g) which may be np.nan if none active.
    """
    rc_n_list: List[float] = []
    rc_g_list: List[float] = []

    # chambers 1–3 (naphtha only)
    for ch in (1, 2, 3):
        feed = float(row.get(f'Naphtha_chamber{ch}', 0.0))
        rc   = float(row.get(f'RCOT_chamber{ch}', np.nan))
        if feed > min_feed_tph and np.isfinite(rc) and (min_rcot_naph is None or rc >= min_rcot_naph):
            rc_n_list.append(rc)

    # chambers 4–6 (virtuals)
    for ch in (4, 5, 6):
        feed_n = float(row.get(f'Naphtha_chamber{ch}', 0.0))
        rc_n   = float(row.get(f'RCOT_naphtha_chamber{ch}', np.nan))
        if feed_n > min_feed_tph and np.isfinite(rc_n) and (min_rcot_naph is None or rc_n >= min_rcot_naph):
            rc_n_list.append(rc_n)

        feed_g = float(row.get(f'Gas Feed_chamber{ch}', 0.0))
        rc_g   = float(row.get(f'RCOT_gas_chamber{ch}', np.nan))
        if feed_g > min_feed_tph and np.isfinite(rc_g) and (min_rcot_gas is None or rc_g >= min_rcot_gas):
            rc_g_list.append(rc_g)

    rc_n = float(np.mean(rc_n_list)) if rc_n_list else np.nan
    rc_g = float(np.mean(rc_g_list)) if rc_g_list else np.nan
    return rc_n, rc_g


# ──────────────────────────────────────────────────────────────────────────────
# Core ML classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MLPredictorConfig:
    feed_cols: List[str] = None
    add_virtual_rcots: bool = True
    build_lag1_from_targets: bool = True
    lgbm_params: Dict[str, Any] = None
    imputer_strategy: str = "median"
    ds_prefixes: List[str] = None

    # ── NEW ──
    include_comp: bool = True
    comp_gas_cols: List[str] = None       # gas mol%
    comp_naph_cols: List[str] = None      # PONA (currently vol% in your data)
    comp_prefix: str = "comp__"

    def __post_init__(self):
            if self.feed_cols is None:
                self.feed_cols = list(FEED_COLS_DEFAULT)
            if self.lgbm_params is None:
                self.lgbm_params = dict(verbosity=-1, n_jobs=2)
            if self.ds_prefixes is None:
                self.ds_prefixes = ['DS_', 'DSchamber', 'DS_chamber', 'DS ']
            # defaults for composition
            if self.comp_gas_cols is None:
                self.comp_gas_cols = list(GAS_COLS_DEFAULT)
            if self.comp_naph_cols is None:
                self.comp_naph_cols = list(PONA_COLS_DEFAULT)

class MLPredictor:
    """
    1) Prepares features (optional virtual RCOTs + lag1)
    2) Fits LGBM models per target with imputation
    3) Predicts per-row or full-DF
    """

    def __init__(self,
                 target_cols: Sequence[str],
                 feature_cols: Optional[Sequence[str]] = None,
                 cfg: Optional[MLPredictorConfig] = None):
        self.target_cols = list(target_cols)
        self.cfg = cfg or MLPredictorConfig()
        self.feature_cols: Optional[List[str]] = list(feature_cols) if feature_cols else None
        self.models: Dict[str, Any] = {}  # target -> pipeline

    # ── Feature prep ──────────────────────────────────────────────────────────
    @staticmethod
    def _ensure_lag1_columns(X: pd.DataFrame, Y: pd.DataFrame, targets: Sequence[str]) -> pd.DataFrame:
        """Adds `*_lag1` columns to X using Y.shift(1) if missing."""
        X = X.copy()
        for t in targets:
            lag_name = t.replace('_t+1', '_lag1')
            if lag_name not in X.columns and t in Y.columns:
                X[lag_name] = Y[t].shift(1)
        return X

    @staticmethod
    def _default_feature_list(X: pd.DataFrame,
                            cfg: MLPredictorConfig,
                            target_cols: Sequence[str]) -> List[str]:
        # lags
        lag_cols  = [c for c in X.columns if str(c).endswith('_lag1')]

        # DS columns by prefix (configurable)
        def _is_ds(col: str) -> bool:
            s = str(col)
            return any(s.startswith(pfx) for pfx in (cfg.ds_prefixes or []))
        ds_cols = [c for c in X.columns if _is_ds(c)]

        # RCOT columns: any column starting with 'RCOT_' (includes RCOT_chamber1..3 and virtuals)
        rcot_cols = [c for c in X.columns if str(c).startswith('RCOT_')]

        # feed columns from config list (present in X)
        feed_cols = [c for c in (cfg.feed_cols or []) if c in X.columns]

        # # Order: lag → ds → rcot → feed
        # feats = list(dict.fromkeys(lag_cols + ds_cols + rcot_cols + feed_cols))


        comp_cols = [c for c in X.columns if str(c).startswith(cfg.comp_prefix or "comp__")]
        feats = list(dict.fromkeys(lag_cols + ds_cols + rcot_cols + feed_cols + comp_cols))


        if not feats:
            raise ValueError("No features found. Provide feature_cols explicitly or enable lag/virtual RCOT creation.")
        return feats


    def _prepare_X(self, X_raw: pd.DataFrame, Y_raw: Optional[pd.DataFrame]) -> pd.DataFrame:
        X = X_raw.copy()

        # add virtual RCOTs if requested
        if self.cfg.add_virtual_rcots:
            build_virtual_rcots_inplace(X)

        # build lag1 if requested and Y is given
        if self.cfg.build_lag1_from_targets and Y_raw is not None:
            X = self._ensure_lag1_columns(X, Y_raw, self.target_cols)
        # ── NEW: compact composition features
        if self.cfg.include_comp:
            self._add_comp_features_inplace(X)

        return X

    # ── Fit / Predict ────────────────────────────────────────────────────────
    def fit(self, X_raw: pd.DataFrame, Y_raw: pd.DataFrame) -> "MLPredictor":
        """
        Fit one LGBM per target on aligned rows (drops any sample missing target or features).
        """
        if not isinstance(X_raw.index, pd.DatetimeIndex):
            # not required, but common for time-series alignment
            pass

        Xp = self._prepare_X(X_raw, Y_raw)

        # If feature_cols not given, infer a sane default
        if self.feature_cols is None:
            self.feature_cols = self._default_feature_list(Xp, self.cfg, self.target_cols)

        # mask rows that have all targets present and feature cols present
        mask = ~Y_raw[self.target_cols].isnull().any(axis=1)
        for f in self.feature_cols:
            mask &= ~Xp[f].isnull()

        X = Xp.loc[mask, self.feature_cols].copy()
        Y = Y_raw.loc[mask, self.target_cols].astype(float).copy()

        # fit models
        self.models = {}
        for t in self.target_cols:
            pipe = make_pipeline(
                SimpleImputer(strategy=self.cfg.imputer_strategy),
                # StandardScaler(with_mean=False),  # robust for sparse-like col scales
                LGBMRegressor(**self.cfg.lgbm_params)
            )
            pipe.set_output(transform="pandas")

            pipe.fit(X, Y[t])
            self.models[t] = pipe
        return self

    def transform(self, X_raw: pd.DataFrame, Y_raw: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Apply the same feature prep used in training (virtual RCOTs, optional lags),
        and ensure all self.feature_cols exist (missing ones become NaN for imputer).
        """
        Xp = self._prepare_X(X_raw, Y_raw)
        # establish feature list if needed
        feats = self.feature_cols or self._default_feature_list(Xp, self.cfg, self.target_cols)
        # add any missing columns as NaN
        for f in feats:
            if f not in Xp.columns:
                Xp[f] = np.nan
        # keep only features for inference
        return Xp

    # def predict_row(self, row: pd.Series) -> Dict[str, float]:
    #     """Predict for a single plant-state row (Series). Robust to missing virt RCOTs / lags."""
    #     if self.feature_cols is None or not self.models:
    #         raise RuntimeError("Model not fitted. Call fit(...) first.")

    #     # build a 1-row frame to allow in-place transforms
    #     df_row = pd.DataFrame([row])

    #     # add virtual RCOTs if requested
    #     if self.cfg.add_virtual_rcots:
    #         build_virtual_rcots_inplace(df_row)

    #     # ensure all expected features exist (fill missing with NaN for imputer)
    #     for f in self.feature_cols:
    #         if f not in df_row.columns:
    #             df_row[f] = np.nan

    #     arr = df_row[self.feature_cols].to_numpy()
    #     out = {}
    #     for t, pipe in self.models.items():
    #         out[t] = float(pipe.predict(arr)[0])
    #     return out

    def predict_row(self, row: pd.Series, Y_for_lags: pd.DataFrame | None = None) -> Dict[str, float]:
        """Predict for a single plant-state row; auto-fills *_lag1 from Y or *_prod."""
        if self.feature_cols is None or not self.models:
            raise RuntimeError("Model not fitted. Call fit(...) first.")

        # build a 1-row frame to allow in-place transforms
        df_row = pd.DataFrame([row])
        df_row.index = [getattr(row, 'name', pd.NaT)]  # so we can align to Y if provided

        # (A) fill lag1 from Y_for_lags if available
        if Y_for_lags is not None and self.target_cols:
            # lags = Y_for_lags[self.target_cols].shift(1)
            # Be robust to missing targets (e.g., PFO not simulated/cached in closed-loop)
            lags = Y_for_lags.reindex(columns=self.target_cols).shift(1)

            ts = df_row.index[0]
            if ts in lags.index:
                for t in self.target_cols:
                    lag = t.replace('_t+1','_lag1')
                    df_row[lag] = lags.at[ts, t]

        # (B) fallback: if *_prod exists on the row, use it as lag1
        for t in self.target_cols:
            lag = t.replace('_t+1','_lag1')
            if lag not in df_row.columns or pd.isna(df_row[lag]).all():
                base = t.replace('_t+1','_prod')
                if base in df_row.columns:
                    df_row[lag] = df_row[base]

        # add virtual RCOTs if requested
        if self.cfg.add_virtual_rcots:
            build_virtual_rcots_inplace(df_row)
            
        # ── NEW: build comp features on the single row
        if self.cfg.include_comp:
            self._add_comp_features_inplace(df_row)

        # ensure all expected features exist (imputer will handle remaining NaNs)
        for f in self.feature_cols:
            if f not in df_row.columns:
                df_row[f] = np.nan

        # arr = df_row[self.feature_cols].to_numpy()
        # out = {t: float(pipe.predict(arr)[0]) for t, pipe in self.models.items()}
        arr_df = df_row[self.feature_cols]          # keep as DataFrame
        out = {t: float(pipe.predict(arr_df)[0]) for t, pipe in self.models.items()}

        return out
    def _add_comp_features_inplace(self, X: pd.DataFrame) -> None:
        """Build compact composition features from gas mol% and PONA totals."""
        pfx = self.cfg.comp_prefix or "comp__"

        # ----- Gas (mol%) → normalize to 100 + summaries -----
        gas_cols = [c for c in (self.cfg.comp_gas_cols or []) if c in X.columns]
        if gas_cols:
            gas = X[gas_cols].astype(float)
            gsum = gas.sum(axis=1).replace(0, np.nan)
            gfrac = gas.div(gsum, axis=0)  # fractions 0..1

            # normalized mol% copies
            for c in gas_cols:
                X[f"{pfx}{c}"] = gfrac[c] * 100.0

            para_cols = [c for c in gas_cols if c in
                        ['Ethane_gas','Propane_gas','n-Butane_gas','i-Butane_gas']]
            olef_cols = [c for c in gas_cols if c in ['Ethylene_gas','Propylene_gas']]

            if para_cols:
                X[f"{pfx}gas_paraffin_frac"] = gfrac[para_cols].sum(axis=1) * 100.0
            if olef_cols:
                X[f"{pfx}gas_olefin_frac"]   = gfrac[olef_cols].sum(axis=1) * 100.0

            # mixture molecular weight (g/mol)
            mw = pd.Series(GAS_MW).reindex(gas_cols)
            X[f"{pfx}gas_MW"] = gfrac.mul(mw, axis=1).sum(axis=1)

        # ----- Naphtha (PONA totals) — passthrough (vol% today; switch to mass% when ready) -----
        pona_cols = [c for c in (self.cfg.comp_naph_cols or []) if c in X.columns]
        for c in pona_cols:
            X[f"{pfx}{c}"] = X[c].astype(float)

        # Optional: composition age if lims_date exists
        if 'lims_date' in X.columns and isinstance(X.index, pd.DatetimeIndex):
            try:
                X[f"{pfx}age_hours"] = (
                    (X.index.tz_localize(None)
                    - pd.to_datetime(X['lims_date']).dt.tz_localize(None))
                    .dt.total_seconds() / 3600.0
                )
            except Exception:
                pass



    # def predict_row(self, row: pd.Series) -> Dict[str, float]:
    #     """Predict for a single plant-state row (Series)."""
    #     if self.feature_cols is None or not self.models:
    #         raise RuntimeError("Model not fitted. Call fit(...) first.")
    #     arr = row[self.feature_cols].to_numpy().reshape(1, -1)
    #     out = {}
    #     for t, pipe in self.models.items():
    #         out[t] = float(pipe.predict(arr)[0])
    #     return out

    def predict_df(self, X: pd.DataFrame, Y_raw: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Predict for a whole DataFrame; returns DataFrame with same index."""
        if self.feature_cols is None or not self.models:
            raise RuntimeError("Model not fitted. Call fit(...) first.")
        Xp = self.transform(X, Y_raw)
        Xp = Xp[self.feature_cols]
        out = {t: self.models[t].predict(Xp) for t in self.target_cols}
        return pd.DataFrame(out, index=Xp.index)


    # ── Metrics / Persistence ────────────────────────────────────────────────
    @staticmethod
    def metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        return dict(
            r2=r2_score(y_true, y_pred),
            mape=mean_absolute_percentage_error(y_true, y_pred)
        )

    def evaluate(self, X: pd.DataFrame, Y: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Quick eval on provided set; returns {target: {r2, mape}}."""
        preds = self.predict_df(X)
        scores: Dict[str, Dict[str, float]] = {}
        for t in self.target_cols:
            yt = Y[t].reindex(preds.index)
            yp = pd.Series(preds[t], index=preds.index)
            mask = ~(yt.isna() | pd.Series(yp).isna())
            scores[t] = self.metrics(yt[mask], yp[mask])
        return scores

    def save(self, path: Path | str) -> None:
        if joblib is None:
            raise RuntimeError("joblib not available; cannot save model.")
        obj = dict(
            target_cols=self.target_cols,
            feature_cols=self.feature_cols,
            cfg=self.cfg,
            models=self.models,
        )
        joblib.dump(obj, path)

    @classmethod
    def load(cls, path: Path | str) -> "MLPredictor":
        if joblib is None:
            raise RuntimeError("joblib not available; cannot load model.")
        obj = joblib.load(path)
        inst = cls(target_cols=obj['target_cols'],
                   feature_cols=obj['feature_cols'],
                   cfg=obj['cfg'])
        inst.models = obj['models']
        return inst
# spot_coordinator
from dataclasses import dataclass
import pandas as pd

@dataclass
class SpotCoordinator:
    ml: Any            # MLPredictor
    srto: Any          # SRTOPipeline

    def _match_comp_row(self, merged_lims: pd.DataFrame, ts: pd.Timestamp) -> pd.Series:
        # pick the latest composition <= ts; fallback to last available
        if 'date' not in merged_lims.columns:
            raise ValueError("merged_lims must have a 'date' column")
        df = merged_lims.dropna(subset=['date']).sort_values('date')
        df = df[df['date'] <= ts]
        if df.empty:
            df = merged_lims.sort_values('date')
        return df.iloc[-1]

    def predict_now(self,
                    X_12h: pd.DataFrame,
                    Y_12h: pd.DataFrame,
                    merged_lims: pd.DataFrame,
                    prefer_geometry: str | None = None) -> dict:
        ts = X_12h.sort_index().index[-1]
        x_row = X_12h.loc[ts]
        comp_row = self._match_comp_row(merged_lims, ts)

        # ML (uses same plant-state row)
        ml_yields = self.ml.predict_row(x_row, Y_for_lags=Y_12h)

        # SRTO (uses furnace state + matched daily composition)
        srto_spot = self.srto.predict_spot_auto(
            X_row=x_row,
            composition_row=comp_row,
            prefer_geometry=prefer_geometry
        )
        return dict(timestamp=ts, ml=ml_yields, srto=srto_spot)
