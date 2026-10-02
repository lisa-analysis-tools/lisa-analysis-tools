# Experimental Roulet-inspired spin proposal

Enable with `CD1L_ROULET_SPINS=1 CD1L_SAMPLER=mc` when running
`scripts/mbh/cd1l_pe.py` in the existing WDM environment. Outputs are placed
in a `roulet_spins/` subdirectory of `CD1L_OUT`; the baseline is unchanged.
Use a fresh output directory for each controlled comparison.

The Gaussian move uses chi_eff=(s1z+q*s2z)/(1+q) and chi_diff=s1z-s2z,
where q=m2/m1. Its fixed covariance is the existing Fisher covariance
transported using the full Jacobian at the injection reference (including
the q derivative). The absolute determinant is one. Gaussian translations
therefore need no Hastings/Jacobian correction when mapped back to the
original coordinates. Proposals outside the original prior become
self-transitions; correlated steps are not reflected at prior boundaries.
Timing metadata records attempted and prior-valid spin proposals because
self-transitions can inflate an acceptance statistic.

Chains, waveform inputs, physical phase, distance and priors remain in the
existing MC basis. All waveform harmonics and the WDM likelihood are unchanged.
The existing stretch, sky-partner and prior-draw moves retain their weights;
this work does not validate those inherited moves or implement Roulet folding.
The Fisher reference retains the existing injection-based setup; this is not
yet a blind-search initialization strategy.

At fixed q this is just a linear reparameterization of a full Gaussian;
improved sampling is not guaranteed. Before a campaign, compare independent
baseline/experimental GPU runs using posterior agreement, effective samples
per wall time, mode occupancy and prior-rejection counts. CPU transform and
move-interface tests do not establish end-to-end WDM correctness or speed.
Physical multimode phase folding, LISA-frame sky geometry and arrival-time
coordinates remain separate work requiring explicit branch weights and
verified detector conventions.

CPU checks:
`python -m unittest discover -s scripts/mbh -p test_cd1l_roulet.py`

## Current driver safeguards (2026-09-12)

The MC target now explicitly conditions the mass box on total mass <= 1e8
solar masses, including the normalization constant. Every move therefore
sees the same support as the constrained prior draw. This changes the prior
relative to older chains; those chains must not be appended to with this code.

The ladder and PE share `cd1l_windows.observation_window`: the entry duration
is defined by the original full grid, even when PE allocates a shorter grid.
PE's revised early-epoch window changes likelihoods relative to the earlier
shrunk-grid taper. Use a fresh output directory. Existing ladder files do not
record their taper version; regenerate ladders made before the long-entry
taper was introduced. WDM post-merger pixel cuts remain a distinct mode.

Adaptation occurs at absolute iteration boundaries and freezes at
`CD1L_ADAPT_UNTIL`. Discard adaptive samples when analyzing the posterior.
Each completed segment stores the proposal Cholesky factor, reference,
completed iteration and configuration in HDF5 attributes. Resume restores
the proposal and continues the adaptation schedule. Missing, incompatible,
or incomplete checkpoints are rejected; historical chains are not migrated
automatically. This restores the proposal kernel, not bit-for-bit RNG replay.

Nonfinite saved likelihoods stop the run and prevent resume. Existing NaN
samples are retained for diagnosis, never silently repaired. The Eryn
`np.where` state-update fix is required and remains a separate repository edit.

`CD1L_REGRESS_BASIS=mc|stock` explicitly specifies a reference chain's basis.
Without it, `_mc.h5` indicates MC and other filenames indicate stock.
Regression indices are limited to available iterations.
`CD1L_NANCHECK=1` records the last twelve move diagnostics for a failed
state check.
