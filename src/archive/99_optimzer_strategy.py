class Optimizer:
    def __init__(self, strategy="slsqp", **kwargs):
        self.strategy = strategy
        self.kw = kwargs

    def optimize(self, objective, x0, bounds):
        if self.strategy == "slsqp":
            res = minimize(objective, x0=x0, bounds=bounds, method="SLSQP",
                           options={"maxiter": self.kw.get("maxiter", 120), "ftol":1e-4})
            return res.x, res.success
        elif self.strategy == "de":
            res = differential_evolution(lambda z: objective(np.array(z)),
                                         bounds=bounds, maxiter=self.kw.get("maxiter",40),
                                         popsize=self.kw.get("popsize",12), seed=self.kw.get("seed",0))
            return res.x, True
        # add Bayes if needed
        raise ValueError("Unknown strategy")
