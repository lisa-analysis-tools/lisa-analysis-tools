# C_XZ delta-sign fix in the XYZ noise covariance

## Summary

The X–Z cross-spectral transfer function of the symmetric (equal-noise) XYZ TDI-2 noise model applied the
arm-asymmetry phase with the wrong sign: it used `delta_31` where the formula needs `delta_13 = -delta_31`.
This biased `C_XZ` by up to ~6e-5 (relative) while every other element of the covariance was correct to
machine precision; after the one-line fix `C_XZ` agrees with an independent reference to ~1e-15.

## Where

`src/lisatools/cutils/PSD.cu`, `XYZSensitivityMatrix::get_noise_tfs` (X–Z block, after the `Z` auto terms).
It affects `oms_xz` and `tm_xz`, and through them every path that uses the symmetric model:
`compute_sensitivity_matrix`, `compute_log_like`, the averaged transfer functions
(`average_transfer_functions=True`) and per-segment STFT matrices. The asymmetric (per-MOSA) model builds its
responses independently and was never affected.

## The fix

```diff
   index1 = link_to_index(31);
   index2 = link_to_index(32);
   index3 = link_to_index(12);
   ...
+  int index_13 = link_to_index(13);
   *oms_xz = oms_xy_unequal_armlength(f, avg_d[index1], avg_d[index3],
-                                     avg_d[index2], delta_d[index1]);
+                                     avg_d[index2], delta_d[index_13]);
   *tm_xz = tm_xy_unequal_armlength(f, avg_d[index1], avg_d[index3],
-                                   avg_d[index2], delta_d[index1]);
+                                   avg_d[index2], delta_d[index_13]);
```

## Why

Notation: one-way delays are `d_ij = avg_ij + delta_ij / 2`, with `avg_ij = (L_ij + L_ji)/2` and
`delta_ij = L_ij - L_ji`, so `delta_ji = -delta_ij` (`get_averaged_ltts` in `sensitivity.py`).

- **Noise model:** the single-link noise is `eta_ij = n^OMS_ij + D_ij n^TM_ji + n^TM_ij`
  (Hartwig et al., [arXiv:2303.15929](https://arxiv.org/abs/2303.15929), Eq. 2.18). The link CSDs
  (Eqs. 2.21a–b) carry the delay phases `exp(±2πif L_ij)`.
- **TDI variables:** X is defined in Eq. 2.24a and its second-generation form in Eq. 2.23a; Y and Z are
  cyclic permutations of X. The TDI CSDs are `S^UV = C^UV S^eta` (Eqs. 2.27–2.30).
- **The X–Y element:** the closed form `oms_xy_unequal_armlength(d_ij, d_ik, d_jk, delta_ij)` /
  `tm_xy_unequal_armlength` evaluates this for channels with origins i (X) and j (Y). Its phase is
  `exp[-iω(d_ik - d_jk + delta_ij / 2)]`.
- **The X–Z element:** `C_XZ` is `conj(C_ZX)`, and `C_ZX` is the X–Y form with (i, j, k) = (3, 1, 2):
  - Its phase is `exp[-iω(d_32 - d_12 + delta_31 / 2)]`.
  - Conjugating gives `exp[-iω(d_12 - d_32 + delta_13 / 2)]`.
- **What the code did:** it passes `(d_ik, d_jk) = (d_12, d_32)`, which already conjugates the arm-average
  part. The asymmetry term must flip sign as well, but the code kept `delta_31`.
- **Why the error was small:** the error is a phase of `ω·delta_31`. `delta_31` is the light-time difference
  between the two directions of one arm (from the constellation rotation, much smaller than `L`), so the effect is tiny but
  systematic.

## Numerical tests

Test script: `verify_asym.py`, kept in the session scratchpad (not in the repo).

Setup:
- mojito_lite L1 orbits; FD grid with df = 1/day over 1e-4 to 0.1 Hz.
- TDI-2, OMS = 15e-12 and TM = 3e-15 for all links.
- Reference: the independent per-link implementation `asymmetric_noises.py`
  (`/mnt/wd_hdd_6TB/nikos/DATA/global_fit/unequal_noises_prototype/`), which builds C from Eqs. 2.18–2.30
  numerically.

Tolerance: max |ΔC| / max |C| over frequency.

### `C_XZ`, symmetric kernel vs reference

| backend | averaged TFs | before fix | after fix |
|---|---|---|---|
| cpu | no | 5.47e-06 | 4.07e-15 |
| cpu | yes | 5.77e-05 | 1.51e-15 |
| cuda12x | no | 5.47e-06 | 3.91e-15 |
| cuda12x | yes | (not recorded) | 6.04e-16 |

The other five elements (XX, YY, ZZ, XY, YZ) were already at 6e-16 to 8e-15 and are unchanged.

### Log-likelihood, symmetric vs asymmetric model at equal amplitudes

The asymmetric model was never affected, so the two should coincide.

| backend | averaged TFs | before fix | after fix |
|---|---|---|---|
| cpu | no | 1.65e-10 | 5.2e-10 |
| cpu | yes | 1.07e-05 | 1.5e-11 |
| cuda12x | no | 1.46e-09 | 1.2e-09 |
| cuda12x | yes | 7.82e-06 | 2.2e-12 |

- **Before the fix:** the averaged-mode difference (~1e-5) was the bug.
- **After the fix:** all differences are at round-off level for a sum over ~8.6k frequencies. The same
  ~1e-10 to 1e-9 level separates the batched kernel from a plain-Python Whittle sum on identical matrices.
- **Non-averaged runs:** the before/after values are similar because there the `C_XZ` error (5e-6) is below
  the likelihood round-off for this test.

### STFT

On the mojito orbits, with a 7-day span and a 1-day STFT segment (7 × 8632 bins, averaged TFs, GPU),
the symmetric and asymmetric (equal-amplitude) matrices agree to 4.1e-15 in every segment.
