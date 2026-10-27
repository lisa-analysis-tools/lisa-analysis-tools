# TM proxy parameters for unequal spacecraft noises

With per-MOSA noise amplitudes (`noise_symmetry="asymmetric"`, 6 OMS + 6 TM), the TM amplitudes are strongly degenerate, and the chains converge poorly. The OMS amplitudes are not affected.

This note covers two things:
- which TM combinations the data constrain;
- the proxy parametrization we sample instead of the per-MOSA amplitudes.

## Method

- **Fisher matrix:** computed for the pipeline covariance (`XYZSensitivityBackend`, TDI2 XYZ, averaged closed-form TFs, filter response) on the mojito_lite orbits.
  - Grid: one week, FD, 1e-4–0.1 Hz.
  - The OMS amplitudes are marginalised.
  - Source: `gf_dev/noise/fisher_tm_basis.ipynb` (basis covariances) and `gf_dev/noise/fisher_tm_analysis.ipynb` (Fisher matrix, parametrizations, corner plots of the Fisher covariance against the chain).
- **Linearity:** the covariance is exactly linear in the per-MOSA PSD levels $P_m = b_m^2$ (max relative error 4e-16), so the derivatives are exact.
- **Chain check:** the Fisher numbers are compared with the existing 12-parameter STFT run, `gf_output/unequal_noises_preproc-bias_inv_fmax:1.0e-01/` (one week of data, equal injected noises).

## Result

| TM combination (MOSAs 12, 23, 31, 13, 32, 21) | Fisher σ | Chain std |
|---|---|---|
| Individual amplitudes $b_{ij}$ | unconstrained | 0.27 (rel.) |
| Total level $\sqrt{\langle P\rangle}$ | 1.0% | 1.0% |
| **Arm ASD** $A_k=\sqrt{(P_{ij}+P_{ji})/2}$, arms 12, 23, 31 | **2.2%** | **2.2%** |
| **Arm asymmetry** $\delta_k=(P_{ij}-P_{ji})/(P_{ij}+P_{ji})$ | 0.28 each, from two weak contrasts | 0.48–0.50 |
| Common shift $P_{12,23,31}\mathrel{+}=t$, $P_{13,32,21}\mathrel{-}=t$ | **∞ (exact null)** | prior |

**The null direction is exact.** $\sum_{\rm cw} B^{\rm TM}_m = \sum_{\rm ccw} B^{\rm TM}_m$ holds to 1.5e-14 in all 9 covariance elements and at all frequencies. Here cw are the clockwise MOSAs (12, 23, 31) and ccw the counter-clockwise ones (13, 32, 21). Shifting $P$ along this direction leaves the likelihood unchanged.

**The result does not depend on the epoch.** Weeks 0, 40 and 90 give the same numbers, because the averaged TFs do not change with epoch. A model with time-varying arm lengths might break the null slightly; this note does not test that.

**OMS amplitudes:** 0.5% per MOSA, with no degeneracy.

**Correlations:**
- In the chain, partner MOSAs on the same arm are anti-correlated at about −0.82. This is the degeneracy that makes the TM marginals broad.
- In the proxy basis, the arm ASDs are correlated with each other at about −0.2 and are uncorrelated with the asymmetries ($|\rho|<0.05$).
- In the chain, the three $\delta_k$ are correlated at +0.63. They move together along the null direction.

**Scaling:** σ scales as $1/\sqrt{N_{\rm weeks}}$. For two years of data this gives about 0.2% on $A_k$ and about 0.05 on the $\delta_k$ contrasts.

**Comparison with Adams & Cornish [1]:** they also found that only arm sums $S_{ij}+S_{ji}$ are well determined, and fit sums and differences. Our basis spans the same space. In addition:
- it isolates the exactly null direction from the two weak ones;
- it makes the asymmetry bounded, $\delta_k\in(-1,1)$, so positivity holds automatically.

## Choice of parametrization

A change of variables only rotates and rescales the Fisher matrix. Any basis that keeps all 6 TM degrees of freedom keeps the null direction and the two weak directions. Our proxies are the Adams & Cornish sums and differences, normalized, and give the same sigmas ($s_k = 2\ln A_k$, $D_k = \delta_k$ at the injection). We keep all 6 TM parameters because the model has to describe genuinely unequal noises. Models that fix or drop the asymmetries are compared in `fisher_tm_analysis.ipynb` for reference, but are not used.

The chain agrees with the Fisher covariance in the proxy basis, with the null direction removed:
- σ: 0.022, 0.022, 0.022, 0.43, 0.42 in the chain for $A_k$ and the contrasts $\delta_{12}-\delta_{23}$, $\delta_{23}-\delta_{31}$, against 0.022 and 0.48 from Fisher. The contrasts are truncated by $|\delta_k|<1$.
- Correlations: −0.20 between arm ASDs and −0.49 between the contrasts in the chain, against −0.22 and −0.50 from Fisher.

## Implementation

**Sampling vector:** `[S_oms × 6, A_12, A_23, A_31, δ_12, δ_23, δ_31]`.
- $A_k$ has the same prior as $S_{\rm tm}$.
- $\delta_k \sim U(-1,1)$. The null direction is left to this bounded prior; there is no extra constraint.

**Mapping in the covariance:** $P_{ij}=A_k^2(1+\delta_k)$ and $P_{ji}=A_k^2(1-\delta_k)$, with $ij$ the clockwise MOSA.
- Code: `sensitivity.py`, functions `tm_proxy_to_mosa`, `tm_mosa_to_proxy` and `make_tm_proxy_transform` (an eryn `TransformContainer` acting in place on TM slots 6–11).
- `XYZSensitivityBackend.__call__` now accepts `transform_fn`, which the slow PSD-move path already passed.

**Settings:** `mojito_input/psd_multigpu_settings.py` has `TM_PARAMETRIZATION = "proxy"` (default) or `"mosa"`.
- The injection is given in the sampling basis, via `tm_mosa_to_proxy`. For equal noises the true proxies are $A_k = 3\times10^{-15}$ and $\delta_k=0$, and these are the corner-plot truths.
- The moves, the initial ACs and the postprocessing apply `PSDSettings.transform`, so likelihoods and catalogue outputs stay per-MOSA.

**Checks:**

| Check | Result |
|---|---|
| Proxy ↔ MOSA round trip | 3e-15 |
| `TransformContainer`, batched and single-walker inputs; pickling | 3e-15; OK |
| Per-walker matrix from proxies + transform vs. per-MOSA | 3e-16 |
| Kernel log-likelihood (`compute_log_like`), proxies vs. per-MOSA | identical |
| Covariance change along the null direction ($\delta_k$ += 0.5) | 4e-16 |

## Reference

1. M. R. Adams, N. J. Cornish, *Discriminating between a stochastic gravitational wave background and instrument noise*, Phys. Rev. D 82, 022002 (2010), arXiv:1002.1291.
