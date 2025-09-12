# gp_residuals.py
# -*- coding: utf-8 -*-

# top-level imports (near the others)
from __future__ import annotations

from joblib import Parallel, delayed
import os

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Constants / maps
# ─────────────────────────────────────────────────────────────────────────────
# Map target columns -> SRTO short keys
SHORT_MAP = {
    'Ethylene_prod_t+1':  'Ethylene',
    'Propylene_prod_t+1': 'Propylene',
    'MixedC4_prod_t+1':   'MixedC4',
    'RPG_prod_t+1':       'RPG',
    'Ethane_prod_t+1':    'Ethane',
    'Propane_prod_t+1':   'Propane',
    # If you later include these in PRODUCTS/curves:
    'Hydrogen_prod_t+1':  'Hydrogen',
    'Tail_Gas_prod_t+1':  'Tail_Gas',
}

PRODUCTS = ['Ethylene','Propylene','MixedC4','RPG','Ethane','Propane','Hydrogen','Tail_Gas']
TARGET_MAP = {
    'Ethylene':  'Ethylene_prod_t+1',
    'Propylene': 'Propylene_prod_t+1',
    'MixedC4':   'MixedC4_prod_t+1',
    'RPG':       'RPG_prod_t+1',
    'Ethane':    'Ethane_prod_t+1',
    'Propane':   'Propane_prod_t+1',
    'Hydrogen':  'Hydrogen_prod_t+1',
    'Tail_Gas':  'Tail_Gas_prod_t+1',
}# ...existing code...



# class PriceProviderLike(Protocol):
#     def get(self, ts: pd.Timestamp, item: str, default: float = 0.0) -> float: ...

# def realized_margin_per_h(
#     ts: pd.Timestamp,
#     x_row: pd.Series,
#     Y_12h: pd.DataFrame,
#     price_provider: opt.PriceProvider
# ) -> float:
#     # revenue from ACT t+1
#     price_item_map = {
#         'Ethylene': 'Ethylene',
#         'Propylene': 'Propylene',
#         'MixedC4':  'Mixed C4',
#         'RPG':      'RPG',
#         'Hydrogen': 'Hydrogen',
#         'Tail_Gas': 'Tail Gas',   # product value of tail gas (FG_PRICE)
#     }
#     rev = 0.0
#     for p, item in price_item_map.items():
#         col = f'{p}_prod_t+1'
#         qty = float(Y_12h.at[ts, col]) if (ts in Y_12h.index and col in Y_12h.columns) else 0.0
#         rev += qty * price_provider.get(ts, item, 0.0)

#     # feed costs
#     p_PN   = price_provider.get(ts, 'PN', 0.0)
#     p_LPG  = price_provider.get(ts, 'LPG Fresh', np.nan)
#     p_OFF  = price_provider.get(ts, 'Offgas Fresh', np.nan)
#     p_GF   = price_provider.get(ts, 'Gas Feed', 0.0)

#     naph = sum(float(x_row.get(f'Naphtha_chamber{i}', 0.0)) for i in range(1, 7))
#     lpg  = float(x_row.get('FreshFeed_C3 LPG',      0.0))
#     off  = float(x_row.get('FreshFeed_MX Offgas',   0.0))
#     gas_chambers = sum(float(x_row.get(f'Gas Feed_chamber{i}', 0.0)) for i in (4,5,6))

#     # Prefer explicit fresh feeds if present; otherwise fall back to Gas Feed cost
#     if (lpg > 0 or off > 0) and (not np.isnan(p_LPG) or not np.isnan(p_OFF)):
#         feed_cost = naph * p_PN
#         feed_cost += lpg * (p_LPG if not np.isnan(p_LPG) else p_GF)
#         feed_cost += off * (p_OFF if not np.isnan(p_OFF) else p_GF)
#     else:
#         feed_cost = naph * p_PN + gas_chambers * p_GF

#     return float(rev - feed_cost)

# ─────────────────────────────────────────────────────────────────────────────
# Feature helpers (consistent with your ML feature policy)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GPFeatureConfig:
    ds_prefixes: List[str] = None         # e.g. ['DS_', 'DSchamber', 'DS_chamber']
    rcot_prefixes: List[str] = None       # default ['RCOT_']
    feed_prefixes: List[str] = None       # ['Naphtha_chamber','Gas Feed_chamber']
    extra_prefixes: List[str] = None      # if you want Conv' O2 etc.
    include_ratio_naphtha: bool = True
    include_geometry_flags: bool = True   # is_hybrid / is_naphtha

    def __post_init__(self):
        if self.ds_prefixes is None:
            self.ds_prefixes = ['DS_', 'DSchamber', 'DS_chamber', 'DS ']
        if self.rcot_prefixes is None:
            self.rcot_prefixes = ['RCOT_']
        if self.feed_prefixes is None:
            self.feed_prefixes = ['Naphtha_chamber', 'Gas Feed_chamber']
        if self.extra_prefixes is None:
            self.extra_prefixes = []


def _has_prefix(s: str, prefixes: Sequence[str]) -> bool:
    return any(str(s).startswith(p) for p in prefixes)


def compute_ratio_naphtha(row: pd.Series) -> float:
    n = sum(float(row.get(c, 0.0)) for c in row.index if str(c).startswith('Naphtha_chamber'))
    g = sum(float(row.get(c, 0.0)) for c in row.index if str(c).startswith('Gas Feed_chamber'))
    den = n + g
    return (n / den) if den > 0 else 0.0


def geometry_flags(row: pd.Series) -> Dict[str, int]:
    n = sum(float(row.get(c, 0.0)) for c in row.index if str(c).startswith('Naphtha_chamber'))
    g = sum(float(row.get(c, 0.0)) for c in row.index if str(c).startswith('Gas Feed_chamber'))
    return dict(
        is_hybrid  = 1 if (n > 0 and g > 0) else 0,
        is_naphtha = 1 if (n > 0 and g == 0) else 0,
    )


def select_gp_features(x_row: pd.Series, cfg: GPFeatureConfig) -> Dict[str, float]:
    """Selects feature values from a plant-state row using configured prefixes."""
    out: Dict[str, float] = {}

    # DS / RCOT / Feed / Extra prefixes
    for k, v in x_row.items():
        sk = str(k)
        if _has_prefix(sk, cfg.ds_prefixes) or _has_prefix(sk, cfg.rcot_prefixes) \
           or _has_prefix(sk, cfg.feed_prefixes) or _has_prefix(sk, cfg.extra_prefixes):
            try:
                out[sk] = float(v)
            except Exception:
                continue

    # Add synthetic features
    if cfg.include_ratio_naphtha:
        out['ratio_naphtha'] = float(compute_ratio_naphtha(x_row))
    if cfg.include_geometry_flags:
        out.update(geometry_flags(x_row))
    return out

def rc_grid_with_rc0(x_row: pd.Series, geometry: str,
                     lo: float, hi: float, points: int = 15) -> np.ndarray:
    """
    Build a linspace(lo,hi) and inject the geometry-aware rc0 inferred from x_row.
    """
    base = np.linspace(float(lo), float(hi), int(points))
    rc0  = rc0_guess_for_geom(x_row, geometry, fallback_rc=None)
    if rc0 is None or not np.isfinite(rc0):
        return base
    return np.unique(np.r_[base, float(rc0)])

def tune_alpha_for_bad_pairs(
    *,
    gp: "GPResiduals",
    X_12h: pd.DataFrame,
    merged_lims: pd.DataFrame,
    pipeline,
    ml,
    ts_list: Sequence[pd.Timestamp],
    setter_map: Dict[str, Any],
    rc_bounds_map: Dict[str, tuple[float, float]],
    rc_points: int = 15,
    slope_corr_thresh: float = 0.92,
    alpha_max: float = 0.35,
    inject_rc0: bool = True,
) -> Dict[tuple[str, str], float]:
    """
    Find (product, geometry) pairs with poor slope fidelity and auto-tune α for those pairs.
    Returns: {(geometry, target_col): alpha} overrides that were applied to gp.
    """
    # 1) compute fidelity
    fid, _, _ = batch_curve_fidelity(
        gp=gp, X_12h=X_12h, merged_lims=merged_lims, pipeline=pipeline, ml=ml,
        idx_eval=ts_list, setter_map=setter_map, rc_bounds_map=rc_bounds_map,
        rc_points=rc_points, use_gp_delta=True, alpha=0.2,
        slope_corr_thresh=slope_corr_thresh, anchor_tol=0.5, inject_rc0=inject_rc0
    )
    if fid.empty:
        return {}

    # 2) collect bad pairs
    bad = fid[fid['slope_corr'] < slope_corr_thresh][['product','geometry']].drop_duplicates()

    overrides: Dict[tuple[str, str], float] = {}
    for geom in bad['geometry'].unique():
        lo, hi = rc_bounds_map[geom]
        # choose the first timestamp in ts_list for tuning
        ts0 = ts_list[0]
        x0  = X_12h.loc[ts0]
        comp_row = gp.GPResiduals._comp_row_for_ts(merged_lims, ts0) if hasattr(gp, 'GPResiduals') else gp._comp_row_for_ts(merged_lims, ts0)
        rc_grid = rc_grid_with_rc0(x0, geom, lo, hi, rc_points) if inject_rc0 else np.linspace(lo, hi, rc_points)

        alphas = gp.auto_tune_alpha(
            pipeline=pipeline, geometry=geom,
            base_x_row=x0, comp_row=comp_row,
            rcot_setter=setter_map[geom],
            rc_grid=rc_grid,
            corr_threshold=slope_corr_thresh,
            alpha_max=alpha_max,
        )
        for p in bad.loc[bad['geometry'] == geom, 'product']:
            tcol = TARGET_MAP[p]
            overrides[(geom, tcol)] = float(alphas.get(tcol, 0.0))

    if overrides:
        gp.set_alpha_overrides(overrides)
    return overrides

# ─────────────────────────────────────────────────────────────────────────────
# GP Residual model per product
# ─────────────────────────────────────────────────────────────────────────────

def _default_kernel(n_features: int):
    # ARD via Matern on standardized features + white noise
    return (C(1.0, (1e-3, 1e3))
            * Matern(length_scale=np.ones(n_features), length_scale_bounds=(1e-2, 1e3), nu=1.5)
            + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1)))


class GPResiduals:
    """
    Fits one GP residual model per product on residuals using features available at time t:
        If residual_kind='ml':   e_t = ML_t+1(t) - SRTO_t
        If residual_kind='act':  e_t = Actual_t+1 - SRTO_t
    The GP mean μ(x_t) is added to SRTO in inference: corrected = SRTO + μ.
    """

    def __init__(self,
                 feature_cfg: Optional[GPFeatureConfig] = None,
                 kernel_factory=_default_kernel,
                 n_restarts: int = 2,
                 normalize_y: bool = True):
        self.feature_cfg = feature_cfg or GPFeatureConfig()
        self.kernel_factory = kernel_factory
        self.n_restarts = n_restarts
        self.normalize_y = normalize_y

        self.products_: List[str] = list(PRODUCTS)
        self.feature_names_: List[str] = []
        self.ct_: Optional[ColumnTransformer] = None
        self.models_: Dict[str, Pipeline] = {}  # product -> Pipeline(StandardScaler -> GPR)

    # ——— training table construction ———

    @staticmethod
    def _comp_row_for_ts(merged_lims: pd.DataFrame, ts: pd.Timestamp) -> pd.Series:
        comp = merged_lims.copy()
        comp['date'] = pd.to_datetime(comp['date'], errors='coerce')
        comp = comp.dropna(subset=['date']).sort_values('date')
        arr = comp['date'].to_numpy(dtype='datetime64[ns]')
        if arr.size == 0:
            raise ValueError("merged_lims has no valid 'date'")
        ts64 = np.datetime64(pd.Timestamp(ts), 'ns')
        pos = np.searchsorted(arr, ts64, side='right') - 1
        if pos < 0:
            pos = 0
        return comp.iloc[pos]
    @classmethod
    def build_training_table(cls,
                            X_12h: pd.DataFrame,
                            Y_12h: pd.DataFrame,
                            merged_lims: pd.DataFrame,
                            pipeline,                  # SRTOPipeline
                            start: str | pd.Timestamp,
                            end: str | pd.Timestamp,
                            feed_thr: float = 0.1,
                            feature_cfg: Optional[GPFeatureConfig] = None,
                            residual_kind: str = "ml",   # 'ml' (default) or 'act'
                            ml: Any = None               # prefit MLPredictor when residual_kind=='ml'
                            ) -> pd.DataFrame:
        """
        Build a leakage-free training DataFrame with:
        - features at time t (selected by prefixes + synthetic),
        - srto_tph per product at time t (from predict_spot_plant),
        - ml_tph or act_tph at t+1 (depending on residual_kind),
        - residuals per product:
            if residual_kind='ml' : resid_<p> = ML_t+1 - SRTO_t
            if residual_kind='act': resid_<p> = ACT_t+1 - SRTO_t
        Notes:
        • For residual_kind='ml', pass a prefit ML (trained on history only).
        • All values are absolute t/h.
        """
        if residual_kind not in ("ml", "act"):
            raise ValueError("residual_kind must be 'ml' or 'act'")
        if residual_kind == "ml" and ml is None:
            raise ValueError("residual_kind='ml' requires an ML predictor (ml=...).")

        feature_cfg = feature_cfg or GPFeatureConfig()

        # Precompute ML features once if we are in 'ml' mode
        X_feat = None
        if residual_kind == "ml":
            # uses your MLPredictor.transform to align lags/virtual RCOTs
            X_feat = ml.transform(X_12h, Y_12h)

        idx = X_12h.sort_index().index
        start = pd.Timestamp(start); end = pd.Timestamp(end)
        ts_list = idx[(idx >= start) & (idx <= end)]

        rows: list[dict] = []
        for ts in ts_list:
            x_row = X_12h.loc[ts]

            # composition row (≤ ts)
            comp_row = cls._comp_row_for_ts(merged_lims, ts)

            # SRTO baseline (absolute t/h per product, plant summed)
            plant_spot = pipeline.predict_spot_plant(x_row, comp_row, feed_thr=feed_thr)
            if plant_spot.get('status') == 'error':
                continue
            srto_tot = plant_spot.get('totals_tph', {})  # abs t/h

            # targets at t+1 (ML or Actual)
            y_ml, y_act = {}, {}
            ok = True

            if residual_kind == "ml":
                # ML prediction from features at time t
                row_feat = X_feat.loc[ts]
                # ml_pred = ml.predict_row(row_feat, Y_for_lags=Y_12h)  # dict of target_cols → t/h
                ml_pred = ml.predict_row(X_12h.loc[ts], Y_for_lags=Y_12h)

                for p in PRODUCTS:
                    col = TARGET_MAP[p]
                    v = float(ml_pred.get(col, np.nan))
                    if not np.isfinite(v):
                        ok = False; break
                    y_ml[p] = v
            else:  # 'act'
                for p in PRODUCTS:
                    col = TARGET_MAP[p]
                    v = Y_12h.at[ts, col] if (ts in Y_12h.index and col in Y_12h.columns) else np.nan
                    if not pd.notna(v):
                        ok = False; break
                    y_act[p] = float(v)

            if not ok:
                continue

            # features at time t
            feats = select_gp_features(x_row, feature_cfg)

            row = dict(timestamp=ts, **{f'X__{k}': v for k, v in feats.items()})
            # stash SRTO + ML/ACT + residuals
            for p in PRODUCTS:
                s = float(srto_tot.get(p, np.nan))
                row[f'{p}_SRTO_tph'] = s
                if residual_kind == "ml":
                    m = y_ml[p]
                    row[f'{p}_ML_tph']    = m
                    row[f'{p}_RESID_tph'] = m - s
                else:
                    a = y_act[p]
                    row[f'{p}_ACT_tph']   = a
                    row[f'{p}_RESID_tph'] = a - s
            rows.append(row)

        # return pd.DataFrame(rows).set_index('timestamp').sort_index()
        df = pd.DataFrame(rows)

        # If nothing was built, return an empty, correctly indexed frame
        if df.empty:
            return pd.DataFrame(index=pd.DatetimeIndex([], name='timestamp'))

        # Accept 'timestamp' or 'Timestamp'
        ts_col = next((c for c in df.columns if c.lower() == 'timestamp'), None)
        if ts_col is None:
            raise KeyError(f"No timestamp-like column found in training table. Columns: {list(df.columns)[:20]}")

        df[ts_col] = pd.to_datetime(df[ts_col])
        df = df.set_index(ts_col).rename_axis('timestamp').sort_index()
        return df


    # ——— fitting ———

    def _prep_feature_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str], ColumnTransformer]:
        """Extracts X matrix from columns with 'X__' prefix, builds a ColumnTransformer that standardizes numerics."""
        feat_cols = [c for c in df.columns if str(c).startswith('X__')]
        if not feat_cols:
            raise ValueError("No feature columns (X__) found in training table.")
        # All numeric by construction → single StandardScaler is enough
        ct = ColumnTransformer([('num', StandardScaler(with_mean=True, with_std=True), feat_cols)], remainder='drop')
        X = ct.fit_transform(df[feat_cols])
        return X, feat_cols, ct

    def fit(self, df_train: pd.DataFrame) -> "GPResiduals":
        """
        Fit one GP per product on residuals (RESID_tph).
        Assumes df_train contains 'X__*' features and '*_RESID_tph' targets.
        """
        X, feat_cols, ct = self._prep_feature_matrix(df_train)
        self.feature_names_ = feat_cols
        self.ct_ = ct
        self.models_.clear()

        n_features = X.shape[1]
        for p in PRODUCTS:
            y = df_train[f'{p}_RESID_tph'].to_numpy(float)
            kernel = self.kernel_factory(n_features)
            gp = GaussianProcessRegressor(
                kernel=kernel, normalize_y=self.normalize_y,
                n_restarts_optimizer=self.n_restarts, random_state=0
            )
            pipe = Pipeline([('gp', gp)])  # scaler already in ct_
            pipe.fit(X, y)
            self.models_[p] = pipe
        return self

    # ——— inference on a timestamp (current RCOTs) ———

    def predict_corrected_for_ts(self,
                                 X_12h: pd.DataFrame,
                                 merged_lims: pd.DataFrame,
                                 pipeline,                # SRTOPipeline
                                 ts: pd.Timestamp,
                                 feed_thr: float = 0.1,
                                 return_std: bool = False) -> Dict[str, Any]:
        """
        Returns corrected absolute t/h per product at timestamp ts by:
            corrected = SRTO(t) + GPμ(features at t)
        Also returns SRTO baseline and (optionally) GP std.
        """
        if self.ct_ is None or not self.models_:
            raise RuntimeError("GPResiduals not fitted.")

        x_row = X_12h.loc[ts]
        comp_row = self._comp_row_for_ts(merged_lims, ts)
        plant_spot = pipeline.predict_spot_plant(x_row, comp_row, feed_thr=feed_thr)
        if plant_spot.get('status') == 'error':
            raise RuntimeError("SRTO plant spot failed at ts.")

        # features
        feats = select_gp_features(x_row, self.feature_cfg)
        Xinf = pd.DataFrame([{f'X__{k}': v for k, v in feats.items()}], index=[ts])
        # ensure all feature columns present
        for c in self.feature_names_:
            if c not in Xinf.columns:
                Xinf[c] = 0.0
        Xinf = Xinf[self.feature_names_]
        Xtrf = self.ct_.transform(Xinf)

        srto_tot = plant_spot['totals_tph']
        out = dict(timestamp=ts, srto=srto_tot, corrected={}, gp_std={})
        for p in PRODUCTS:
            base = float(srto_tot.get(p, 0.0))
            pipe = self.models_[p]
            if return_std:
                mu, std = pipe['gp'].predict(Xtrf, return_std=True)
                out['corrected'][p] = float(base + mu[0])
                out['gp_std'][p] = float(std[0])
            else:
                mu = pipe['gp'].predict(Xtrf, return_std=False)
                out['corrected'][p] = float(base + mu[0])
        return out

    # ——— inference on a modified plant-state row (e.g., candidate RCOTs) # inside class GPResiduals

    def fit_parallel(self, df_train: pd.DataFrame, n_jobs: int | None = None) -> "GPResiduals":
        """
        Same as fit(...), but trains one GP per product in parallel.
        Reuses the same preprocessor (ColumnTransformer) and kernel_factory.
        """
        # avoid BLAS oversubscription per worker
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

        X, feat_cols, ct = self._prep_feature_matrix(df_train)
        self.feature_names_ = feat_cols
        self.ct_ = ct
        self.models_.clear()

        n_features = X.shape[1]
        prods = list(self.products_)
        n_jobs = min(len(prods), (os.cpu_count() or 1)) if n_jobs is None else int(n_jobs)

        def _fit_one(p: str):
            y = df_train[f'{p}_RESID_tph'].to_numpy(float)
            kernel = self.kernel_factory(n_features)
            gp = GaussianProcessRegressor(
                kernel=kernel, normalize_y=self.normalize_y,
                n_restarts_optimizer=self.n_restarts, random_state=0
            )
            # scaling is in self.ct_, so pipeline is just the GP
            pipe = Pipeline([('gp', gp)])
            pipe.fit(X, y)
            return p, pipe

        pairs = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(_fit_one)(p) for p in prods
        )
        self.models_ = dict(pairs)
        return self

    def predict_corrected_for_row(self,
                                  X_row: pd.Series,
                                  composition_row: pd.Series,
                                  pipeline,              # SRTOPipeline
                                  feed_thr: float = 0.1,
                                  return_std: bool = False) -> Dict[str, Any]:
        """
        Same as predict_corrected_for_ts but accepts a single X_row (potentially with modified RCOTs).
        """
        if self.ct_ is None or not self.models_:
            raise RuntimeError("GPResiduals not fitted.")

        plant_spot = pipeline.predict_spot_plant(X_row, composition_row, feed_thr=feed_thr)
        if plant_spot.get('status') == 'error':
            raise RuntimeError("SRTO plant spot failed for the provided row.")

        feats = select_gp_features(X_row, self.feature_cfg)
        Xinf = pd.DataFrame([{f'X__{k}': v for k, v in feats.items()}], index=[0])
        for c in self.feature_names_:
            if c not in Xinf.columns:
                Xinf[c] = 0.0
        Xinf = Xinf[self.feature_names_]
        Xtrf = self.ct_.transform(Xinf)

        srto_tot = plant_spot['totals_tph']
        out = dict(srto=srto_tot, corrected={}, gp_std={})
        for p in PRODUCTS:
            base = float(srto_tot.get(p, 0.0))
            pipe = self.models_[p]
            if return_std:
                mu, std = pipe['gp'].predict(Xtrf, return_std=True)
                out['corrected'][p] = float(base + mu[0])
                out['gp_std'][p] = float(std[0])
            else:
                mu = pipe['gp'].predict(Xtrf, return_std=False)
                out['corrected'][p] = float(base + mu[0])
        return out
    # inside class GPResiduals

    # ---- alpha override storage & access ----
    _alpha_overrides: Dict[tuple[str, str], float] = {}  # key=(geometry, target_col)

    def set_alpha_overrides(self, overrides: Dict[tuple[str, str], float]) -> None:
        """Set manual alpha scaling for (geometry, target_col)."""
        self._alpha_overrides = dict(overrides or {})

    def _alpha_for(self, geometry: str | None, target_col: str, default_alpha: float) -> float:
        if geometry is None:
            return default_alpha
        return float(self._alpha_overrides.get((geometry, target_col), default_alpha))

    # ---- internal: GP μ (and σ) for a single row ----
    def _gp_mu_for_row(self, X_row: pd.Series, return_std: bool = False) -> tuple[dict, dict]:
        feats = select_gp_features(X_row, self.feature_cfg)
        Xinf = pd.DataFrame([{f'X__{k}': v for k, v in feats.items()}], index=[0])
        for c in self.feature_names_:
            if c not in Xinf.columns:
                Xinf[c] = 0.0
        Xinf = Xinf[self.feature_names_]
        Xtrf = self.ct_.transform(Xinf)

        mu = {}
        sd = {}
        for p in self.products_:
            pipe = self.models_.get(p)
            if pipe is None:
                mu[p] = 0.0
                sd[p] = 0.0
                continue
            if return_std:
                m, s = pipe['gp'].predict(Xtrf, return_std=True)
                mu[p], sd[p] = float(m[0]), float(s[0])
            else:
                m = pipe['gp'].predict(Xtrf, return_std=False)
                mu[p], sd[p] = float(m[0]), 0.0
        return mu, sd

    def sweep_corrected_curve(self, *,
                            pipeline,
                            base_x_row: pd.Series,
                            comp_row: pd.Series,
                            rcot_setter,
                            rc_grid: np.ndarray,
                            clamp_zero: bool = True,
                            alpha: float = 1.0,
                            geometry: str | None = None,
                            return_std: bool = False) -> pd.DataFrame:
        """
        For each rc in rc_grid:
        X_r = rcot_setter(base_x_row.copy(), rc)   # set the right RCOT tags
        SRTO_r (t/h) = pipeline.predict_spot_plant(X_r, comp_row)
        μ_r          = GP residual mean on features(X_r)
        y_corr       = SRTO_r + α(geom,target) * μ_r
        Returns a DataFrame with SRTO, residual μ, corrected, and (optionally) σ per product.
        """
        rows: list[dict] = []
        for rc in rc_grid:
            Xr = base_x_row.copy()
            rcot_setter(Xr, float(rc))  # user-supplied mutator

            spot = pipeline.predict_spot_plant(Xr, comp_row, feed_thr=0.1)
            if spot.get('status') == 'error':
                continue
            srto = spot['totals_tph']  # absolute t/h per product

            mu, sd = self._gp_mu_for_row(Xr, return_std=return_std)

            out = {'RCOT': float(rc), 'geometry': geometry or spot.get('geometry_used', None)}
            for p in self.products_:
                tcol = TARGET_MAP[p]
                a = self._alpha_for(out['geometry'], tcol, alpha)
                base = float(srto.get(p, 0.0))
                corr = base + a * mu[p]
                if clamp_zero:
                    corr = max(0.0, corr)
                out[f'{p}_SRTO_tph']   = base
                out[f'{p}_RESID_mu']   = a * mu[p]
                out[f'{p}_CORR_tph']   = corr
                if return_std:
                    out[f'{p}_RESID_std'] = sd[p]
            rows.append(out)
        return pd.DataFrame(rows)
    # ──────────────────────────────────────────────────────────────────────
    # ML-anchored corrections: keep the ML point; add SRTO (and optional GP) deltas
    # ──────────────────────────────────────────────────────────────────────
    # def sweep_anchored_curve(self, *,
    #                         pipeline,
    #                         base_x_row: pd.Series,
    #                         comp_row: pd.Series,
    #                         rcot_setter,
    #                         rc_grid: np.ndarray,
    #                         ml_point: dict,                 # keys = TARGET_MAP values
    #                         use_gp_delta: bool = False,     # if True, add α·(μ(rc)-μ(rc0))
    #                         alpha: float = 0.2,             # shrink for GP delta (0..1)
    #                         clamp_zero: bool = True,
    #                         return_std: bool = False) -> pd.DataFrame:
    #     """
    #     Build an RCOT curve anchored to the ML point at current RCOT(s):
    #         y(rc) = ML_point + [SRTO(rc) - SRTO(rc0)] + α * ([μ(rc) - μ(rc0)])
    #     where rc0 is the operating RCOT in base_x_row.
    #     """
    #     # SRTO at operating point (rc0)
    #     spot0 = pipeline.predict_spot_plant(base_x_row, comp_row, feed_thr=0.1)
    #     if spot0.get('status') == 'error':
    #         return pd.DataFrame()
    #     srto0 = spot0['totals_tph']

    #     # optional GP residual mean at rc0
    #     if use_gp_delta:
    #         mu0, sd0 = self._gp_mu_for_row(base_x_row, return_std=return_std)
    #     else:
    #         mu0 = {p: 0.0 for p in self.products_}
    #         sd0 = {p: 0.0 for p in self.products_}

    #     rows = []
    #     for rc in rc_grid:
    #         Xr = base_x_row.copy()
    #         rcot_setter(Xr, float(rc))

    #         spot = pipeline.predict_spot_plant(Xr, comp_row, feed_thr=0.1)
    #         if spot.get('status') == 'error':
    #             continue
    #         srto = spot['totals_tph']

    #         if use_gp_delta:
    #             mu, sd = self._gp_mu_for_row(Xr, return_std=return_std)
    #         else:
    #             mu = {p: 0.0 for p in self.products_}
    #             sd = {p: 0.0 for p in self.products_}

    #         out = {'RCOT': float(rc), 'geometry': spot.get('geometry_used', None)}
    #         for p in self.products_:
    #             tcol   = TARGET_MAP[p]
    #             anchor = float(ml_point.get(tcol, np.nan))  # ML point at rc0
    #             d_phys = float(srto.get(p, 0.0)) - float(srto0.get(p, 0.0))
    #             d_mu   = (mu[p] - mu0[p]) if use_gp_delta else 0.0
    #             y      = anchor + d_phys + alpha * d_mu
    #             if clamp_zero:
    #                 y = max(0.0, y)

    #             out[f'{p}_ANCHOR_tph'] = anchor
    #             out[f'{p}_SRTO_tph']   = float(srto.get(p, 0.0))
    #             out[f'{p}_CORR_tph']   = y
    #             if return_std and use_gp_delta:
    #                 out[f'{p}_RESID_std'] = sd[p]
    #         rows.append(out)
    #     return pd.DataFrame(rows)

    def sweep_anchored_curve(self, *,
                            pipeline,
                            base_x_row: pd.Series,
                            comp_row: pd.Series,
                            rcot_setter,
                            rc_grid: np.ndarray,
                            ml_point: dict,                 # keys = TARGET_MAP values
                            use_gp_delta: bool = False,     # if True, add α·(μ(rc)-μ(rc0))
                            alpha: float = 0.2,             # default; per-product overrides may replace it
                            clamp_zero: bool = True,
                            return_std: bool = False) -> pd.DataFrame:
        """
        Build an RCOT curve anchored to the ML point at current RCOT(s):
            y(rc) = ML_point + [SRTO(rc) - SRTO(rc0)] + α_geom,prod * ([μ(rc) - μ(rc0)])
        where rc0 is the operating RCOT in base_x_row, and α_geom,prod can be overridden
        via set_alpha_overrides({(geometry, TARGET_MAP[prod]): alpha_value}).
        """
        # SRTO at operating point (rc0)
        spot0 = pipeline.predict_spot_plant(base_x_row, comp_row, feed_thr=0.1)
        if spot0.get('status') == 'error':
            return pd.DataFrame()
        srto0 = spot0['totals_tph']

        # optional GP residual mean at rc0
        if use_gp_delta:
            mu0, sd0 = self._gp_mu_for_row(base_x_row, return_std=return_std)
        else:
            mu0 = {p: 0.0 for p in self.products_}
            sd0 = {p: 0.0 for p in self.products_}

        rows = []
        for rc in rc_grid:
            Xr = base_x_row.copy()
            rcot_setter(Xr, float(rc))

            spot = pipeline.predict_spot_plant(Xr, comp_row, feed_thr=0.1)
            if spot.get('status') == 'error':
                continue
            srto = spot['totals_tph']
            geom_used = spot.get('geometry_used', None)
            geom_used = str(geom_used).strip().replace(" ", "_").upper() if geom_used is not None else None  # normalize


            if use_gp_delta:
                mu, sd = self._gp_mu_for_row(Xr, return_std=return_std)
            else:
                mu = {p: 0.0 for p in self.products_}
                sd = {p: 0.0 for p in self.products_}

            out = {'RCOT': float(rc), 'geometry': geom_used}
            for p in self.products_:
                tcol   = TARGET_MAP[p]
                anchor = float(ml_point.get(tcol, np.nan))  # ML point at rc0
                d_phys = float(srto.get(p, 0.0)) - float(srto0.get(p, 0.0))
                d_mu   = (mu[p] - mu0[p]) if use_gp_delta else 0.0

                # 🔸 per-geometry, per-product alpha (falls back to the function's alpha)
                a = self._alpha_for(geom_used, tcol, alpha)

                y = anchor + d_phys + a * d_mu
                if clamp_zero:
                    y = max(0.0, y)

                out[f'{p}_ANCHOR_tph'] = anchor
                out[f'{p}_SRTO_tph']   = float(srto.get(p, 0.0))
                out[f'{p}_CORR_tph']   = y
                # (optional) expose the alpha actually used – uncomment if useful:
                # out[f'{p}_ALPHA_used'] = float(a)

                if return_std and use_gp_delta:
                    out[f'{p}_RESID_std'] = sd[p]
            rows.append(out)
        return pd.DataFrame(rows)

    def predict_anchored_for_ts(self, *,
                                X_12h: pd.DataFrame,
                                merged_lims: pd.DataFrame,
                                pipeline,
                                ml,                     # ml_cached or a dict-like {target: value}
                                ts: pd.Timestamp,
                                alpha: float = 0.0,     # usually 0 for point; curve uses sweep_anchored_curve
                                use_gp_delta: bool = False,
                                feed_thr: float = 0.1,
                                return_std: bool = False) -> dict:
        """
        Return an 'anchored' point for timestamp ts.
        Anchored point = ML_point at rc0 (optionally + α·μ(rc0), typically α=0).
        """
        x_row   = X_12h.loc[ts]
        comp_row= self._comp_row_for_ts(merged_lims, ts)

        spot0 = pipeline.predict_spot_plant(x_row, comp_row, feed_thr=feed_thr)
        if spot0.get('status') == 'error':
            raise RuntimeError("SRTO plant spot failed at ts.")

        # get ML point (works with ml_cached or prebuilt dict)
        ml_point = ml.predict_row(x_row) if hasattr(ml, 'predict_row') else ml

        if use_gp_delta:
            mu0, sd0 = self._gp_mu_for_row(x_row, return_std=return_std)
        else:
            mu0 = {p: 0.0 for p in self.products_}
            sd0 = {p: 0.0 for p in self.products_}

        out = {'timestamp': ts, 'srto': spot0['totals_tph'], 'anchored': {}, 'gp_std': {}}
        for p in self.products_:
            tcol = TARGET_MAP[p]
            anchor = float(ml_point.get(tcol, np.nan))
            y = anchor + alpha * float(mu0.get(p, 0.0))
            out['anchored'][p] = y
            if return_std and use_gp_delta:
                out['gp_std'][p] = float(sd0.get(p, 0.0))
        return out

    # inside class GPResiduals

    def auto_tune_alpha(self, *,
                        pipeline,
                        geometry: str,
                        base_x_row: pd.Series,
                        comp_row: pd.Series,
                        rcot_setter,
                        rc_grid: np.ndarray,
                        targets: Sequence[str] = tuple(TARGET_MAP.values()),
                        h: float = 0.2,
                        corr_threshold: float = 0.92,
                        alpha_max: float = 0.5) -> Dict[str, float]:
        """
        For each target, pick alpha ∈ [0, alpha_max] maximizing corr(d_s/dRCOT, d_h/dRCOT),
        computed with central differences over rc_grid (absolute t/h domain).
        Stores result in self._alpha_overrides[(geometry, target)].
        Returns dict {target_col: alpha}.
        """
        alphas = {}
        # SRTO slope array per rc
        def srto_vals_for_rc(rc: float) -> Dict[str, float]:
            Xr = base_x_row.copy()
            rcot_setter(Xr, float(rc))
            spot = pipeline.predict_spot_plant(Xr, comp_row, feed_thr=0.1)
            return spot['totals_tph'] if spot.get('status') == 'ok' else {}

        def central_diff(v):
            # v: array of vals over rc_grid; central diff with grid spacing
            dr = float(rc_grid[1] - rc_grid[0])
            dv = np.gradient(v, dr)
            return dv

        # precompute SRTO curves
        srto_curves = {t: [] for t in TARGET_MAP}
        for rc in rc_grid:
            vals = srto_vals_for_rc(rc)
            for p in self.products_:
                srto_curves[p].append(float(vals.get(p, 0.0)))
        srto_slopes = {p: central_diff(np.array(srto_curves[p], float)) for p in self.products_}

        for p in self.products_:
            tcol = TARGET_MAP[p]
            # GP μ over grid
            mu = []
            for rc in rc_grid:
                Xr = base_x_row.copy()
                rcot_setter(Xr, float(rc))
                mu_r, _ = self._gp_mu_for_row(Xr, return_std=False)
                mu.append(mu_r[p])
            mu = np.array(mu, float)

            raw_slope = srto_slopes[p]
            best_a, best_c = 0.0, -np.inf
            for a in np.linspace(0.0, alpha_max, 26):
                corr = np.corrcoef(raw_slope, raw_slope + np.gradient(a*mu, float(rc_grid[1]-rc_grid[0])))[0,1]
                if np.isfinite(corr) and corr > best_c:
                    best_c, best_a = corr, a

            if not np.isfinite(best_c):
                best_a = 0.0
            # enforce minimal fidelity if below threshold
            if best_c < corr_threshold:
                # keep whatever is best, even if < threshold (you can also snap to 0)
                pass

            self._alpha_overrides[(geometry, tcol)] = float(best_a)
            alphas[tcol] = float(best_a)
        return alphas


# ─────────────────────────────────────────────────────────────────────────────
# Public utilities: RCOT geometry discovery, setters, anchored sweep wrapper,
# fidelity, optimizers
# ─────────────────────────────────────────────────────────────────────────────

# Module-level storage for discovered RCOT column groups (initialized by user)
_RCOT_GROUPS: dict[str, list[str]] = {
    "RCOT_13": [],
    "RCOT_N456": [],
    "RCOT_G456": [],
}

def set_rcot_groups_from_columns(columns: Sequence[str]) -> dict[str, list[str]]:
    """
    Discover RCOT column groups from a list/Index of column names and store them
    for use by the module's setters. Call this once from your notebook:
        gpmod.set_rcot_groups_from_columns(X_12h.columns)
    Returns the discovered groups dict.
    """
    cols = [str(c) for c in columns]
    groups = {
        "RCOT_13":   [c for c in cols if c.startswith("RCOT_chamber")],
        "RCOT_N456": [c for c in cols if c.startswith("RCOT_naphtha_chamber")],
        "RCOT_G456": [c for c in cols if c.startswith("RCOT_gas_chamber")],
    }
    _RCOT_GROUPS.update(groups)
    return groups


def rcot_setter_lf_naph(row: pd.Series, rc: float) -> pd.Series:
    """LF_NAPH: set ONLY 1–3; do not touch any 4–6 RCOTs."""
    for c in _RCOT_GROUPS.get("RCOT_13", []):
        if c in row.index:
            row[c] = rc
    return row


def rcot_setter_gf_gas(row: pd.Series, rc: float) -> pd.Series:
    """GF_GAS: set ONLY gas-side 4–6; do not touch 1–3 or naphtha 4–6."""
    for c in _RCOT_GROUPS.get("RCOT_G456", []):
        if c in row.index:
            row[c] = rc
    return row


def rcot_setter_hybrid(row: pd.Series, rc: float) -> pd.Series:
    """HYBRID: set BOTH naphtha- and gas-side 4–6; leave 1–3 untouched."""
    for c in (_RCOT_GROUPS.get("RCOT_N456", []) + _RCOT_GROUPS.get("RCOT_G456", [])):
        if c in row.index:
            row[c] = rc
    return row


def rc0_guess_for_geom(x_row: pd.Series, geom: str, fallback_rc: float | None = None) -> float:
    """
    Geometry-aware estimate of rc0 from the appropriate RCOT_* group in x_row.
    """
    if fallback_rc is None:
        fallback_rc = float(
            np.mean([float(v) for k, v in x_row.items() if str(k).startswith("RCOT_")])
        ) if any(str(k).startswith("RCOT_") for k in x_row.index) else 0.0
    if geom == "LF_NAPH":
        cols = _RCOT_GROUPS.get("RCOT_13", [])
    elif geom == "GF_GAS":
        cols = _RCOT_GROUPS.get("RCOT_G456", [])
    else:  # hybrid or other -> both 4..6 sides
        cols = _RCOT_GROUPS.get("RCOT_N456", []) + _RCOT_GROUPS.get("RCOT_G456", [])
    vals = [float(x_row[c]) for c in cols if c in x_row.index and pd.notnull(x_row[c])]
    return float(np.mean(vals)) if vals else float(fallback_rc)


# --- Anchored sweep at a timestamp (wrapper around GPResiduals.sweep_anchored_curve) ---
def anchored_curve_at_ts(
    *,
    gp: "GPResiduals",
    X_12h: pd.DataFrame,
    merged_lims: pd.DataFrame,
    pipeline,
    ml,                              # ml_cached or predictor with .predict_row(...)
    ts: pd.Timestamp,
    rcot_setter,
    rc_grid: np.ndarray,
    use_gp_delta: bool = True,
    alpha: float = 0.2,
    feed_thr: float = 0.1,
    return_std: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build ML-anchored RCOT curve at time ts:
      y(rc) = ML(rc0) + [SRTO(rc)-SRTO(rc0)] + alpha * [mu(rc)-mu(rc0)]
    Returns (curve_df, x_row_used).
    """
    if gp.ct_ is None or not gp.models_:
        raise RuntimeError("GPResiduals not fitted.")
    x_row = X_12h.loc[ts]
    comp_row = gp._comp_row_for_ts(merged_lims, ts)
    # robust ML point
    if hasattr(ml, "predict_row"):
        try:
            ml_point = ml.predict_row(x_row)
        except TypeError:
            ml_point = ml.predict_row(x_row, Y_for_lags=globals().get("Y_12h"))
    else:
        ml_point = ml  # dict-like
    curve = gp.sweep_anchored_curve(
        pipeline=pipeline,
        base_x_row=x_row,
        comp_row=comp_row,
        rcot_setter=rcot_setter,
        rc_grid=rc_grid,
        ml_point=ml_point,
        use_gp_delta=use_gp_delta,
        alpha=alpha,
        clamp_zero=True,
        return_std=return_std,
    )
    return curve, x_row


# --- Fidelity helpers (no plotting here) ---
def _finite_diff(y: np.ndarray, dx: float) -> np.ndarray:
    y = np.asarray(y, float)
    return np.gradient(y, dx)

def _sign_agree(d_ref: np.ndarray, d_test: np.ndarray, eps: float | None = None) -> tuple[float, float]:
    d_ref = np.asarray(d_ref, float); d_test = np.asarray(d_test, float)
    if eps is None:
        eps = 0.01 * (np.nanpercentile(np.abs(d_ref), 95) + 1e-12)
    m = np.abs(d_ref) >= eps
    if not m.any():
        return np.nan, 0.0
    return float((np.sign(d_ref[m]) == np.sign(d_test[m])).mean()), float(m.mean())

def curve_fidelity_for_curve(
    curve: pd.DataFrame,
    x_row: pd.Series,
    product: str,
) -> dict:
    """
    Compare slope shape of SRTO vs anchored CORR for a single product on a prepared curve df.
    Returns dict: slope_corr, sign_agree, sign_cov, anchor_miss.
    """
    if len(curve) < 3:
        return dict(slope_corr=np.nan, sign_agree=np.nan, sign_cov=0.0, anchor_miss=np.nan)

    dx = float(curve['RCOT'].iloc[1] - curve['RCOT'].iloc[0])
    srto = curve[f'{product}_SRTO_tph'].to_numpy(float)
    corr = curve[f'{product}_CORR_tph'].to_numpy(float)

    d_srto = _finite_diff(srto, dx)    # physics slope
    d_corr = _finite_diff(corr, dx)    # anchored corrected slope

    slope_corr = float(np.corrcoef(d_srto, d_corr)[0, 1]) if np.all(np.isfinite([d_srto, d_corr])) else np.nan
    sign_ok, coverage = _sign_agree(d_srto, d_corr)

    # anchor check: nearest grid point to mean RCOT across visible RCOT_* cols
    rc0_candidates = [float(v) for k, v in x_row.items() if str(k).startswith('RCOT_')]
    rc0_guess = float(np.mean(rc0_candidates)) if rc0_candidates else float(curve['RCOT'].iloc[len(curve)//2])
    i0 = int(np.argmin(np.abs(curve['RCOT'].to_numpy() - rc0_guess)))
    anchor_miss = float(abs(curve.iloc[i0][f'{product}_CORR_tph'] - curve.iloc[i0][f'{product}_ANCHOR_tph']))

    return dict(slope_corr=slope_corr, sign_agree=sign_ok, sign_cov=coverage, anchor_miss=anchor_miss)

def batch_curve_fidelity(
    *,
    gp: "GPResiduals",
    X_12h: pd.DataFrame,
    merged_lims: pd.DataFrame,
    pipeline,
    ml,
    idx_eval: Sequence[pd.Timestamp],
    setter_map: Dict[str, Any],
    rc_bounds_map: Dict[str, tuple[float, float]],
    rc_points: int = 15,
    use_gp_delta: bool = True,
    alpha: float = 0.2,
    slope_corr_thresh: float = 0.92,
    anchor_tol: float = 0.5,         # ← practical default in t/h
    inject_rc0: bool = True,         # ← new: ensure anchor grid contains rc0
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, pd.DataFrame]]]:
    """
    Compute fidelity metrics for product×geometry over multiple timestamps.
    Returns: (fidelity_df, summary_df, example_curves)
    """
    products = list(getattr(gp, "products_", PRODUCTS))
    rows: list[dict] = []
    example_curves: Dict[str, Dict[str, pd.DataFrame]] = {}

    for geom, (lo, hi) in rc_bounds_map.items():
        rcot_setter = setter_map.get(geom)
        if rcot_setter is None:
            continue

        for ts in idx_eval:
            # build rc_grid (inject rc0 from the *raw* X row, not from curve)
            x_raw = X_12h.loc[ts]
            rc_grid = (rc_grid_with_rc0(x_raw, geom, lo, hi, rc_points)
                       if inject_rc0 else np.linspace(lo, hi, rc_points))

            curve, x_row = anchored_curve_at_ts(
                gp=gp, X_12h=X_12h, merged_lims=merged_lims, pipeline=pipeline, ml=ml,
                ts=ts, rcot_setter=rcot_setter, rc_grid=rc_grid,
                use_gp_delta=use_gp_delta, alpha=alpha
            )
            if curve.empty:
                continue

            for p in products:
                met = curve_fidelity_for_curve_geom(curve, x_row, p, geometry=geom)
                rows.append({'timestamp': ts, 'geometry': geom, 'product': p, **met})

            if geom not in example_curves:  # save one example per geometry
                example_curves[geom] = {
                    p: curve[['RCOT', f'{p}_SRTO_tph', f'{p}_CORR_tph', f'{p}_ANCHOR_tph']].copy()
                    for p in products
                }

    fidelity_df = pd.DataFrame(rows)
    if not fidelity_df.empty:
        fidelity_df['ok'] = (fidelity_df['slope_corr'] >= slope_corr_thresh) & \
                            (fidelity_df['anchor_miss'] <= anchor_tol)
        summary = (fidelity_df
                   .groupby(['product', 'geometry'])['ok']
                   .agg(pct_ok=lambda s: 100.0 * s.mean(), n='count')
                   .reset_index()
                   .sort_values(['product', 'geometry']))
    else:
        summary = pd.DataFrame(columns=['product', 'geometry', 'pct_ok', 'n'])

    return fidelity_df, summary, example_curves


# def curve_fidelity_for_curve_geom(
#     curve: pd.DataFrame,
#     x_row: pd.Series,
#     product: str,
#     geometry: str,
# ) -> dict:
#     """
#     Geometry-aware fidelity: compares slope shape of SRTO vs anchored CORR for a product,
#     and computes anchor_miss at the rc0 inferred from the geometry's RCOT group.
#     """
#     if len(curve) < 3:
#         return dict(slope_corr=np.nan, sign_agree=np.nan, sign_cov=0.0, anchor_miss=np.nan)

#     dx   = float(curve['RCOT'].iloc[1] - curve['RCOT'].iloc[0])
#     s    = curve[f'{product}_SRTO_tph'].to_numpy(float)
#     c    = curve[f'{product}_CORR_tph'].to_numpy(float)
#     ds   = np.gradient(s, dx)
#     dc   = np.gradient(c, dx)
#     corr = float(np.corrcoef(ds, dc)[0,1]) if np.all(np.isfinite([ds, dc])) else np.nan

#     # sign agreement (ignore near-flat regions)
#     eps = 0.01 * (np.nanpercentile(np.abs(ds), 95) + 1e-12)
#     mask = np.abs(ds) >= eps
#     sign_agree = float((np.sign(ds[mask]) == np.sign(dc[mask])).mean()) if mask.any() else np.nan
#     sign_cov   = float(mask.mean()) if mask.any() else 0.0

#     # anchor at geometry-aware rc0
#     fallback = float(curve['RCOT'].iloc[len(curve)//2])
#     rc0      = rc0_guess_for_geom(x_row, geometry, fallback_rc=fallback)
#     i0       = int(np.argmin(np.abs(curve['RCOT'].to_numpy() - rc0)))
#     anchor_miss = float(abs(curve.iloc[i0][f'{product}_CORR_tph'] - curve.iloc[i0][f'{product}_ANCHOR_tph']))

#     return dict(slope_corr=corr, sign_agree=sign_agree, sign_cov=sign_cov, anchor_miss=anchor_miss)
def curve_fidelity_for_curve_geom(
    curve: pd.DataFrame,
    x_row: pd.Series,
    product: str,
    geometry: str,
) -> dict:
    """
    Robust slope fidelity: compare first differences of SRTO vs CORR so that
    y_corr = anchor + (SRTO - SRTO0) gives slope_corr ≈ 1 even on non-uniform grids.
    Also report anchor_miss at the geometry-aware rc0.
    """
    if len(curve) < 3:
        return dict(slope_corr=np.nan, sign_agree=np.nan, sign_cov=0.0, anchor_miss=np.nan)

    # arrays
    s = curve[f'{product}_SRTO_tph'].to_numpy(float)
    c = curve[f'{product}_CORR_tph'].to_numpy(float)

    # robust slopes via first differences (no dx needed)
    ds = np.diff(s)
    dc = np.diff(c)

    # mask: ignore tiny SRTO slope regions (eps from 95th pct)
    if np.all(~np.isfinite(ds)):
        return dict(slope_corr=np.nan, sign_agree=np.nan, sign_cov=0.0, anchor_miss=np.nan)

    ds_f = ds[np.isfinite(ds)]
    eps = 0.01 * (np.nanpercentile(np.abs(ds_f), 95) + 1e-12)
    mask = np.abs(ds) >= eps

    if mask.sum() < 3 or not np.isfinite(dc[mask]).all():
        slope_corr = np.nan
        sign_agree = np.nan
        sign_cov = float(mask.mean())
    else:
        # correlation of informative segments
        try:
            slope_corr = float(np.corrcoef(ds[mask], dc[mask])[0, 1])
        except Exception:
            slope_corr = np.nan
        sign_agree = float((np.sign(ds[mask]) == np.sign(dc[mask])).mean())
        sign_cov = float(mask.mean())

    # anchor check at geometry-aware rc0
    fallback = float(curve['RCOT'].iloc[len(curve)//2])
    rc0 = rc0_guess_for_geom(x_row, geometry, fallback_rc=fallback)
    i0 = int(np.argmin(np.abs(curve['RCOT'].to_numpy(float) - rc0)))
    anchor_miss = float(abs(curve.iloc[i0][f'{product}_CORR_tph'] - curve.iloc[i0][f'{product}_ANCHOR_tph']))

    return dict(slope_corr=slope_corr, sign_agree=sign_agree, sign_cov=sign_cov, anchor_miss=anchor_miss)


# --- Simple optimizers over the anchored curve ---
def optimize_rcot_anchored_grid(
    *,
    gp: "GPResiduals",
    ts: pd.Timestamp,
    X_12h: pd.DataFrame,
    merged_lims: pd.DataFrame,
    pipeline,
    ml,
    rcot_setter,
    rc_bounds: tuple[float, float] = (800.0, 860.0),
    step: float = 0.5,
    weights: Optional[Dict[str, float]] = None,   # {'Ethylene': 1.0, 'Propylene': 0.4, ...}
    use_gp_delta: bool = True,
    alpha: float = 0.2,
) -> dict:
    """
    Maximize weighted tons over RCOT grid using the ML-anchored curve.
    Returns dict: {'rc_opt','obj','prod_tph','curve'}
    """
    if weights is None:
        weights = {'Ethylene': 1.0}

    rc_grid = np.arange(rc_bounds[0], rc_bounds[1] + 1e-9, step, dtype=float)
    curve, _ = anchored_curve_at_ts(
        gp=gp, X_12h=X_12h, merged_lims=merged_lims, pipeline=pipeline, ml=ml,
        ts=ts, rcot_setter=rcot_setter, rc_grid=rc_grid,
        use_gp_delta=use_gp_delta, alpha=alpha
    )
    if curve.empty:
        raise RuntimeError("Anchored sweep returned empty curve.")

    obj = np.zeros(len(curve), float)
    for p, w in weights.items():
        obj += float(w) * curve[f'{p}_CORR_tph'].to_numpy(float)

    curve = curve.copy()
    curve['OBJ'] = obj
    i_best = int(curve['OBJ'].idxmax())
    best = curve.loc[i_best]

    prod_opt = {p: float(best[f'{p}_CORR_tph']) for p in getattr(gp, "products_", PRODUCTS)}
    return {'rc_opt': float(best['RCOT']), 'obj': float(best['OBJ']), 'prod_tph': prod_opt, 'curve': curve}

def optimize_rcot_margin_at_ts(
    *,
    gp: "GPResiduals",
    ts: pd.Timestamp,
    X_12h: pd.DataFrame,
    merged_lims: pd.DataFrame,
    pipeline,
    ml,
    rcot_setter,
    rc_bounds: tuple[float, float],
    step: float,
    price_per_ton: Dict[str, float],                # {'Ethylene': $, ...}
    fuel_cost_fn: Optional[Any] = None,             # lambda rc -> $/h
    use_gp_delta: bool = True,
    alpha: float = 0.2,
) -> dict:
    """
    Maximize margin(rc) = Σ price[p]*CORR_tph[p](rc) - fuel_cost(rc).
    Returns dict: {'rc_opt','margin','curve'}
    """
    rc_grid = np.arange(rc_bounds[0], rc_bounds[1] + 1e-9, step, dtype=float)
    curve, _ = anchored_curve_at_ts(
        gp=gp, X_12h=X_12h, merged_lims=merged_lims, pipeline=pipeline, ml=ml,
        ts=ts, rcot_setter=rcot_setter, rc_grid=rc_grid,
        use_gp_delta=use_gp_delta, alpha=alpha
    )
    if curve.empty:
        raise RuntimeError("Anchored sweep returned empty curve.")

    rev = np.zeros(len(curve), float)
    for p, pr in price_per_ton.items():
        rev += float(pr) * curve[f'{p}_CORR_tph'].to_numpy(float)

    cost = np.zeros_like(rev)
    if fuel_cost_fn is not None:
        cost = np.array([fuel_cost_fn(rc) for rc in curve['RCOT'].to_numpy(float)], float)

    curve = curve.copy()
    curve['MARGIN'] = rev - cost
    i_best = int(curve['MARGIN'].idxmax())
    best = curve.loc[i_best]
    return {'rc_opt': float(best['RCOT']), 'margin': float(best['MARGIN']), 'curve': curve}


# --- Small helper for your metrics ---
def mape_series(act: pd.Series, pred: pd.Series) -> float:
    mask = act.notna() & pred.notna() & (act != 0)
    return float((pred[mask] - act[mask]).abs().div(act[mask].abs()).mean() * 100) if mask.any() else np.nan


# --- Exported symbols ---
__all__ = [
    # constants
    "PRODUCTS", "TARGET_MAP",
    # config + model
    "GPFeatureConfig", "GPResiduals",
    # setters
    "rcot_setter_lf_naph", "rcot_setter_gf_gas", "rcot_setter_hybrid",
    # anchored sweep + fidelity
    "anchored_curve_at_ts", "curve_fidelity_for_curve", "batch_curve_fidelity",
    # geometry utilities
    "set_rcot_groups_from_columns", "rc0_guess_for_geom",
    # geometry-aware fidelity
    "curve_fidelity_for_curve_geom",
    # optimizers
    "optimize_rcot_anchored_grid", "optimize_rcot_margin_at_ts",
    # util
    "mape_series",
]
