TurbScope learned-model benchmark -- all values computed in this run
numpy 2.4.4
train_default_model(): fit+calibration wall time 3.25 s (506 fit / 169 calibration / 225 test scenarios, x_test shape (675, 4))
fit_report: {'n_rows': 1518.0, 'quantile_crossing_fraction': 0.0032938076416337285, 'conformal_delta_dex': 0.026571592449634807}
conformal delta: 0.026572 dex

==============================================================================
1. Overall held-out error, all test rows (dex = decades of Cn2)
==============================================================================
predictor                                       RMSE       MAE      bias       p95
TurbScope learned model                       0.0714    0.0497   -0.0054    0.1224
Scintillometer weak baseline (mandated)       0.6577    0.3578   -0.3332    1.6216
DIMM-only baseline                            0.0318    0.0249   -0.0020    0.0598
Training mean (learned-nothing floor)         1.6877    1.4449   -0.0434    2.9402

learned/baseline RMSE ratio (mandated comparison): 0.1086

==============================================================================
2. Error broken down by TRUE regime (weak vs saturated scintillometer path)
==============================================================================
n weak rows: 363   n saturated rows: 312
predictor                                    RMSE weak  RMSE saturated
TurbScope learned model                         0.0648          0.0784
Scintillometer weak baseline (mandated)         0.0380          0.9665
DIMM-only baseline                              0.0304          0.0334
Training mean (learned-nothing floor)           1.6053          1.7788

==============================================================================
3. Prediction-interval coverage on held-out test data (nominal 90%)
==============================================================================
interval                         nominal    coverage    mean width (dex)
raw quantile GBR                   0.900      0.7985              0.2869
conformally calibrated             0.900      0.8770              0.3333

==============================================================================
3b. Coverage by regime (calibrated model)
==============================================================================
weak regime      : coverage 0.8512, mean width 0.3464 dex
saturated regime : coverage 0.9071, mean width 0.3181 dex

==============================================================================
4. Reproducibility check
==============================================================================
identical test features on re-run: True
max |prediction difference| across re-runs: 0.000e+00 dex
conformal delta identical: True

==============================================================================
Summary
==============================================================================
total script wall time: 4.97 s
