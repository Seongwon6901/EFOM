# Build curves from your df_results
bank   = SpyroCurveBank(df_results)
sabs   = SpyroAbs(bank)

# GP kernel factory
kernel = lambda d: (DotProduct(sigma_0=1.0)**2 + RBF(length_scale=np.ones(d)) + WhiteKernel())

gp     = ResidualGP(feature_cols_gp, kernel, n_restarts=1, seed=0)
mc     = MarginCalculator(_prep_prices(prices_df), FuelGasConstants(
            cp_kcal_per_tonK=411_488.209, dH_eth=1_080_970.0, dH_pro=673_409.0,
            dH_fg=926_147.0, fg_hhv_kcal_per_ton=15_294_088.0, rcot_ref_C=840.0))
opt    = Optimizer(strategy="slsqp", maxiter=120)
cfg    = RollingConfig(window_train=360, window_test=2, min_train=700, min_test=2, start_frac=0.8)

pipe   = RollingPipeline(sabs, gp, mc, opt, cfg, SHORT_MAP)

# pipe.run(X_raw, Y_raw, targets, feature_cols_gp, active_rcot_fn=_active_rcot_vars, geometry_from_row=geometry_from_row, util_models=UtilityModels(um, util_feature_cols), alpha_overrides=ALPHA_OVERRIDES)
