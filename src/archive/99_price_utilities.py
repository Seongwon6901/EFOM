class UtilityModels:
    def __init__(self, models: Dict[str, object], feature_cols: Sequence[str]):
        self.models = models or {}
        self.feature_cols = list(feature_cols or [])
    def predict(self, row: pd.Series):
        if not self.models: return (0.0,0.0,0.0)
        X = row.reindex(self.feature_cols).to_numpy(dtype=float).reshape(1,-1)
        s = float(self.models.get('steam',       lambda x: [0])[0].predict(X)) if 'steam' in self.models else 0.0
        f = float(self.models.get('fuel_gas',    lambda x: [0])[0].predict(X)) if 'fuel_gas' in self.models else 0.0
        e = float(self.models.get('electricity', lambda x: [0])[0].predict(X)) if 'electricity' in self.models else 0.0
        return (s,f,e)
