# SYNTHETIC DEMONSTRATION ARTIFACT -- assuregraph example case

Fabricated for the shipped example. This is not a derivation anyone should use.

## Threshold derivation for the illustrative residual monitor

Decision rule: a one-sided cumulative-sum statistic on the pitch residual,
compared with a constant threshold `h` (dimensionless, residual normalised by
the assumed noise standard deviation).

Target false-alarm rate: 1 in 10 000 samples under the null model of
zero-mean Gaussian residuals with standard deviation 0.0020 rad.

For the normalised one-sided CUSUM with reference value k = 0.5 and the null
model above, the value of `h` that meets the target in this document's
illustrative tabulation is **h = 4.0**.

Validity range: the figure holds only for the null model stated here. It is not
transferable to a different noise model, a different reference value, or a
non-Gaussian residual.
