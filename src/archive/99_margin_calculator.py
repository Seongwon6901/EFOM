class MarginCalculator:
    def __init__(self, price_cache_df: pd.DataFrame, fg_constants: FuelGasConstants):
        self.p = price_cache_df; self.const = fg_constants

    def _lookup(self, ts, col):
        d = pd.Timestamp(ts).to_period('M').to_timestamp()
        s = self.p.loc[:d, col].dropna()
        return float(s.iloc[-1]) if len(s) else 0.0

    def margin(self, ts, row: pd.Series, yields_abs: Dict[str,float], utils=(0,0,0), fg_prod=0.0, fg_cons=0.0):
        p = lambda c: self._lookup(ts, c)
        rev = (yields_abs.get('Ethylene_prod_t+1',0)*p('Ethylene') +
               yields_abs.get('Propylene_prod_t+1',0)*p('Propylene') +
               yields_abs.get('MixedC4_prod_t+1',0)*p('Mixed C4') +
               yields_abs.get('RPG_prod_t+1',0)*p('RPG'))
        naph = sum(float(row.get(f'Naphtha_chamber{ch}',0)) for ch in range(1,7))
        gas  = sum(float(row.get(f'Gas Feed_chamber{ch}',0)) for ch in (4,5,6))
        feed_cost = naph*p('PN') + gas*p('Gas Feed')
        s,f,e = utils
        util_cost = s*p('Steam') + f*p('Fuel Gas') + e*p('Electricity')
        return float(rev - feed_cost - util_cost + (fg_prod - fg_cons)*p('Fuel Gas'))
