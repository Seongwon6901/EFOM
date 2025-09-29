import pandas as pd
from scipy.interpolate import PchipInterpolator

class SpyroCurveBank:
    def __init__(self, df_results: pd.DataFrame):
        self.bank = {}
        self._build(df_results)

    def _build(self, df):
        df = df.copy()
        df['date'] = pd.to_datetime(df['date']).normalize()
        prods = ['C2H4','C3H6','MixedC4','RPG','Ethane','Propane','Fuel_Gas']
        for geom, gdf in df.groupby('geometry'):
            self.bank[geom] = {}
            for d, ddf in gdf.groupby('date'):
                ddf = ddf.sort_values('target_RCOT')
                rc  = ddf['target_RCOT'].to_numpy(float)
                if rc.size < 2: continue
                funcs = {}
                for p in prods:
                    if p in ddf.columns and ddf[p].notnull().sum() >= 2:
                        y = ddf[p].to_numpy(float)
                        funcs[p] = PchipInterpolator(rc, y, extrapolate=True)
                if funcs:
                    self.bank[geom][d] = funcs

    def percent(self, geom: str, prod: str, rc: float, ts: pd.Timestamp, morning_prev=True) -> float:
        if geom not in self.bank: return 0.0
        d = ts.normalize() - pd.Timedelta(days=1) if (morning_prev and ts.hour < 12) else ts.normalize()
        dated = self.bank[geom]
        if d not in dated:
            if not dated: return 0.0
            d = min(dated.keys(), key=lambda k: abs(k - d))
        f = dated[d].get(prod)
        return float(f(float(rc))) if f is not None else 0.0
