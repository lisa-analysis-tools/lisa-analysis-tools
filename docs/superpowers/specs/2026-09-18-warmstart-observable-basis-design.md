# Warm-start GB components in the OBSERVABLE basis

**Status.** Design approved in conversation 2026-09-18 (user: "let's fit the
warmstart in the observed domain again. Just like the fstat fit"; "We should
do this whole process on the observed basis. Then when we sample it, we
convert to the astro basis and make sure the proper jacobians are included";
"When you intake the last run, just convert to the observed basis"; "Rely
heavily on the existing infrastructure around basis conversions and things
like that. They exist").

**One sentence.** `fit_from_store` converts the previous run's cold-chain leaf
table into the observable basis at intake, every downstream stage of the warm
-start pipeline works there, and `WarmStartComponents` converts back to the
sampling basis at draw time with the map's own log Jacobian.

---

## 1. Why

The warm start's job is to birth a source where the previous run found it.
Measured on the 4-GPU 6-month run at iteration 68, it is not doing that:

* `[GB_BIRTH_STALE rj_warm_search]` reports the detected-to-optimal SNR ratio
  of each drawn template at geometric mean **0.19**, with **70.8%** below 0.5
  -- the diagnostic's own label for "peak gone from the residual". Seven in
  ten warm births propose where the signal has already been fitted.
* Cold-chain RJ acceptance for the warm move is **7/6009 (0.12%)**.

It is NOT because the component set is poor. The component nearest the
flagship 20.380377 mHz source (index 5571 of
`gf_prod_3mo_v8_10w_refereed.npz`) is excellent:

| quantity | value |
|---|---|
| f0 offset from truth | **-0.001 uHz** (sigma 0.043 uHz) |
| fdot at the component mean | **1.0302e-13**, truth 1.0245e-13 -> **x1.01** |
| ratio | -0.0018 +/- 0.271 |
| p / n_members / mult | 0.97 / 97 / **1.000** (one source, unblended) |

The three-month run measured that source's chirp to one percent. The problem
is the COORDINATES the component is expressed in, which show the signature
plainly across all 5573 components:

| column | sigma, median | sigma, p90 |
|---|---|---|
| f0 | 0.005 uHz | 0.042 uHz |
| fdot_astro_ratio | **1.529** | **30.0** |

58.8% of components have a ratio sigma above 1, and the fitter's own comments
record ratio means piling at and beyond the +/- 5 rails. One coordinate is
superbly determined and another is prior-dominated.

The cause is structural. The data constrains

```
fdot = (96/5) pi^(8/3) (Msun * Mc)^(5/3) f0^(11/3) (1 + r)
```

which in the sampling basis is a CURVED surface through three columns
(`Mc`, `f0`, `fdot_astro_ratio`). A Gaussian in those coordinates can only
express it as a local linear correlation. Drawing from such a component puts
the birth off the chirp ridge, which is exactly the stale-birth rate above.

A second, independent defect: the clustering feature space is
`FEAT_NAMES = ["f0", "Mc", "ln_dist", "alpha", "sin_delta"]`
(`fit_from_store.py:65`). **The frequency derivative is not in it.** Two
fragments of one source share an f0 and differ in fdot, so the current metric
cannot separate them -- it is left to the match referee's merge pass to clean
up afterwards.

## 2. The map (existing infrastructure -- nothing new is derived here)

`lisatools.sampling.gb_observable_basis.GBObservableFiberBasis`
(`gb_observable_basis.py:461`) is a **9-to-9 bijection**:

| index | sampling (astro) | internal (observable) |
|---|---|---|
| 0 | dist | lnA |
| 1 | f0 | f_mid |
| 2 | Mc | **fdot** |
| 3 | phi0 | phi0 |
| 4 | cos_iota | cos_iota |
| 5 | psi | psi |
| 6 | alpha | alpha |
| 7 | sin_delta | sin_delta |
| 8 | fdot_astro_ratio | Mc (`FIBER_INDEX = 8`) |

Three public methods, all already used by the in-model observable move and
validated there (the VGB work checked `log_jacobian` against both an analytic
expression and a numerical derivative, agreeing to 5e-10):

* `to_internal(coords, leaf_inds=None) -> (n, 9)`
* `from_internal(z, template=None, leaf_inds=None) -> (n, 9)`
* `log_jacobian(coords, leaf_inds=None) -> (n,)`, documented as
  `ln(dist) - ln(fdot_gr(f0, Mc))` (+ `ln(Mc)` for `fiber_coord="lnMc"`),
  correct up to an additive constant that cancels because both ends call the
  same function.

**The GMM machinery is likewise already present** and is used by the F-stat
side today -- see section 4.4: `vec_fit_gmm_min_bic` (`gmm.py:2139`, GPU
batched, min-BIC sweep), `pack_gmm_components` / `unpack_gmm_components`
(`fstat_proposal.py:2533`), `fit_gmm_to_stacked` (`fstat_proposal.py:2506`)
and `FullGaussianMixtureModel` (`prior.py:763`, box-bounded logpdf through a
backend C/CUDA kernel).

**Two properties make this cheap.**

1. **Indices 3 through 7 are identical in both bases** (phi0, cos_iota, psi,
   alpha, sin_delta pass through untouched). So `CIRCULAR_COLS = {3: 2pi,
   5: pi, 6: 2pi}` and `COS_IOTA_COL = 4` in `fit_from_store.py` need NO
   change. Only indices 0, 1, 2, 8 change meaning.
2. The one degenerate direction is isolated on a single named axis
   (`Mc`, the fiber) instead of being smeared across three columns.

Construction convention, taken from the existing in-model caller
(`gbspecialstretch.py:11373`): `Tobs=1.0 / float(self.df)` -- the run's own
frequency resolution, NOT `basis_settings.Tobs` -- with
`shear=GB_INMODEL_OBSERVABLE_SHEAR` (default 0.5) and `fiber_coord="Mc"`.

## 3. Cross-Tobs policy: UNCHANGED

`proposal.py` already documents the v1 policy: components are used **AS
FITTED**, with no Fisher T-rescale; only the f0 candidate windows are
re-derived against the new run's `df = 1/new_tobs`.

This design does not touch that. Intake builds the map at the SOURCE run's
`1/df`, the fit lives in that frame, and `from_internal` returns sampling-basis
columns that are frame-independent. The existing `new_tobs` constructor
argument keeps its present job (candidate windows) and nothing else.

## 4. Design

### 4.1 Intake (`fit_from_store`)

Build ONE `GBObservableFiberBasis` from the source store's transform container
at that run's `1/df`. Immediately after the cold-chain leaf table is read,
map it through `to_internal`. **No stage after this point sees astro columns.**

`COLUMN_NAMES` gains an observable counterpart; the module's existing
`CIRCULAR_COLS` / `COS_IOTA_COL` constants are reused unchanged (section 2,
property 1).

### 4.2 Segmentation

Unchanged algorithm, new coordinate: density-valley segmentation runs on
`f_mid` instead of `f0`, at the same `1/Tobs` bins with the same count floor
and one-bin padding. `f_mid` differs from `f0` by `shear * Tobs * fdot`, which
is sub-bin for all but the fastest chirpers, so island structure is
essentially preserved.

### 4.3 Clustering

```
FEAT_NAMES: ["f0", "Mc", "ln_dist", "alpha", "sin_delta"]
         -> ["f_mid", "fdot", "lnA", "alpha", "sin_delta"]
```

`lnA` is the direct successor to `ln_dist` (it is the measured amplitude,
already logged by the map). `fdot` is new and is the separator the current
metric lacks. `Mc` LEAVES the metric: it is the fiber, a flat direction, and
clustering on a flat direction is what generates the split artifacts the
referee's merge pass exists to repair.

Everything else -- robust MAD whitening, single-linkage on a <=1500-row
subsample cut at 2.0 whitened units, nearest-centroid assignment with junk
radius 6, the satellite-fragment merge pass -- is unchanged.

### 4.4 Component fit: a GAUSSIAN MIXTURE per cluster

**User ruling 2026-09-18: "adjust this to a gaussian mixture model. Let it fit
more components if needed up to some reasonable limit."** A cluster is not
required to be one Gaussian. Its posterior can be genuinely multimodal --
residual sky reflection, the psi/phi0 pairing, an inclination fold -- and a
single Gaussian smeared across two modes puts most of its mass in the empty
space between them, which is a birth proposal aimed where there is nothing.

Fit a `K`-component Gaussian mixture over all nine OBSERVABLE columns, with
full covariances.

**Use the existing GPU fitter. Do not write a BIC sweep.**
`lisatools.sampling.gmm.vec_fit_gmm_min_bic` (`gmm.py:2139`) already is this
algorithm: GPU-batched across groups, sweeps the component count, keeps the
parameters at the BIC minimum, and retires a group once its BIC has risen
twice past its running minimum so the sweep does not run to the cap
needlessly. `return_components=True` yields the seven raw lists
`[weights, means, covs, invcovs, dets, mins, maxs]`.

**Mirror `fit_gmm_to_stacked` exactly** (`fstat_proposal.py:2506`), which is
how the F-stat side already does this and is what "modelled on the F-stat
distributions" means concretely. That function draws a FIXED
`n_samples_per_box` from each box's density and hands the rectangular
`(K, n, d)` block to the fitter.

The warm start needs the same rectangularisation for a different reason:
`vec_fit_gmm_min_bic` wants `(n_groups, n_samples, n_features)` but clusters
have ragged member counts (3 to 97+ in the shipped set). Resample each
cluster's members to a fixed `n_samples_per_cluster`, with replacement for
clusters smaller than that -- the direct analogue of `rvs_per_box`. Default
4096, matching the F-stat call.

**The limit.** `max_comp` default **12**, matching `fit_gmm_to_stacked`
(the fitter's own default is 30; the F-stat call caps it at 12 and the
twice-past-minimum rule means the sweep usually stops far earlier). Guard
small clusters with `max_comp_effective = min(GB_WARMSTART_GMM_MAX_COMP,
max(1, n_members // GB_WARMSTART_GMM_MIN_MEMBERS))`, default
`MIN_MEMBERS = 25`, so a 97-member cluster (the flagship's size) may reach 3
and a 30-member cluster stays at 1 whatever BIC prefers. Resampling with
replacement cannot manufacture structure that the members do not contain, but
it can let BIC believe it has more evidence than it does; that guard is what
prevents it.

**Bounds come from the fitter's own `mins` / `maxs`.** The component lists
carry per-group bounding boxes, which is the mechanism
`FullGaussianMixtureModel` uses to decide which components can contribute at
a query point. Membership of the bounded columns still changes with the
basis, and the observable box is what the fitter is handed:

* `cos_iota` (index 4): bounded `[-1, 1]`, truncated-normal MLE as today.
* `Mc` (index 8, the fiber): bounded below at 0.
* `fdot` (index 2): **unbounded** -- this is the point of the basis change.
  The `+/- ratio_max` rail that produced the p90 sigma of 30 is gone.

**This partially subsumes the referee's split handling** -- see section 4.6,
which has to change so the two do not fight.

### 4.5 Proposal (`WarmStartComponents`)

The class gains the map, reconstructed from parameters stored in the npz.

* `rvs(size)`: pick a cluster by `p`, a component within it by its mixture
  weight, draw the Gaussian, and return `map.from_internal(z)` --
  sampling-basis columns, as every caller expects.
* `logpdf(x)`: `mixture_logpdf(map.to_internal(x)) + map.log_jacobian(x)`.

**The density side should be `FullGaussianMixtureModel`** (`prior.py:763`),
not the current hand-rolled sum. It is built for exactly this shape -- a
concatenation of independently fitted GMMs, each in its own `(mins, maxs)`
box, weights renormalised so the global density integrates to one -- and its
`logpdf` runs a backend C/CUDA kernel that sums only the components whose
bounding box covers the query point.

That kernel subsumes the existing `window_df` candidate-window logic, which
is a Python reimplementation of the same idea keyed on f0. Retiring it in
favour of the bounding boxes is the single largest simplification in this
design. `FullGaussianMixtureModel` has no `rvs`, so the draw path stays in
`WarmStartComponents`.

Circular wrapping is unchanged: the angles pass through the map untouched at
the same indices (section 2, property 1).

### 4.6 Referee and SNR gate

`match_referee` and `opt_snr` build real waveforms and therefore need astro
columns. Each converts at its own boundary via `from_internal`. The referee's
match statistic is basis-independent by construction.

**One contract DOES change, and it is the only place the GMM forces a
behaviour change elsewhere.** The referee auto-merges same-island component
pairs whose centroid cross-match exceeds 0.9, on the reasoning that they are
split artifacts of one source. Under 4.4 a cluster's `K` components are a
DELIBERATE multi-modal description of one source, so they would cross-match
highly and the referee would moment-match them straight back into the single
Gaussian this design exists to avoid.

Fix: **the merge test runs only ACROSS clusters, never within one.** The
grouping needs no new field -- `pack_gmm_components` already stores
`gmm_ncomp`, the per-cluster component count, which partitions the flat
component arrays into clusters exactly. Mixture siblings are exempt; genuine
split artifacts -- which by construction landed in different clusters -- are
still merged as today.

The within-group coherence test is unaffected: it scores members against the
member nearest the component mean, which is still well defined per mixture
component.

## 5. File format and compatibility

**Store the mixture in the EXISTING packed layout**, `pack_gmm_components` /
`unpack_gmm_components` (`fstat_proposal.py:2533`), which is already the
format the F-stat GMM cache uses: `gmm_ncomp`, `gmm_weights`, `gmm_means`,
`gmm_covs`, `gmm_invcovs`, `gmm_dets`, `gmm_mins`, `gmm_maxs`. Same fitter,
same packing, same reader -- which is what makes "modelled on the F-stat
distributions" true at the file level and not just by analogy.

`gmm_ncomp` doubles as the cluster partition of the flat component arrays,
so section 4.6's merge rule needs no new field.

Per-CLUSTER arrays keep their present meaning and length (one entry per
cluster, not per component): `p`, `mult`, `n_members`, `island_id`,
`f0_window_edges`, `blend`, `med_match`, `med_ratio`.

The npz additionally gains:

* `basis`: `"sampling"` (legacy) or `"observable"`.
* `map_params`: `Tobs`, `shear`, `fiber_coord`, `input_basis` -- everything
  needed to reconstruct the map without the source store.

`WarmStartComponents` reads `basis` and handles both. A file without the key
is `"sampling"` with one Gaussian per cluster, i.e. exactly today's layout.
**The current campaign is not blocked**: the shipped
`gf_prod_3mo_v8_10w_refereed.npz` keeps working as it does now, and an
observable-basis set appears when someone reruns the pipeline.

## 6. Testing

1. **Round trip.** `from_internal(to_internal(x)) == x` to 1e-12 over the
   fitted component means and a prior-box sample. (The map has its own tests;
   this pins the pipeline's use of it.)
2. **Jacobian consistency.** `logpdf` in the sampling basis integrates to 1
   over a bounded region by Monte Carlo, and matches a numerical change of
   variables on the existing astro-basis path for a synthetic single
   component.
3. **The flagship, end to end.** Refit the 3-month store, find the component
   nearest 20.380377 mHz, and assert: `mult == 1`, `|d f0| < 0.05 uHz`, and
   drawn samples' fdot within a factor of 1.3 of 1.0245e-13 for >= 68% of
   draws. This is the regression test for the defect that motivated the work.
4. **Fragment separation.** A synthetic island holding two sources with equal
   f0 and fdot differing by 5 sigma must produce TWO clusters under the new
   metric and ONE under the old.
5. **Legacy load.** The shipped refereed npz loads, draws and scores exactly
   as it does today (byte-identical `rvs` under a fixed seed).
6. **No stale-birth regression.** Not a unit test: the cluster read-out is
   `[GB_BIRTH_STALE]` geometric mean and the fraction below 0.5, compared
   against today's 0.19 / 70.8%.

## 7. Risks

* **`f_mid` shear at high fdot.** For the fastest chirpers `shear * Tobs *
  fdot` can exceed a bin, moving a source between islands relative to the
  current segmentation. Mitigated by testing 4; the shear knob already exists
  if it needs turning down for the fitter.
* **`lnA` vs `ln_dist` whitening scale.** The MAD scale floor was tuned for
  `ln_dist`. `lnA` has a different dynamic range; the floor may need
  re-deriving. Cheap to check on the existing store.
* **The fiber's marginal.** Chirp mass leaves the clustering metric but stays
  in the fitted Gaussian. If its marginal is effectively flat the covariance
  may be ill-conditioned along that axis; the truncated-normal fit at the
  lower bound is the existing machinery for exactly that case.
* **Referee thresholds.** The auto-merge cut of 0.9 was tuned against
  sampling-basis clusters. Fewer split artifacts should reach the referee at
  all, so the cut may become less load-bearing rather than more -- worth
  reading the merge counts after the first refit.

## 8. Out of scope

* The VGB warm start (VGB has its own basis work in flight).
* Any change to the F-stat fit itself. This design MIRRORS the F-stat
  distributions' domain; it does not modify them.
* The `GB_LEAF_CAP_MIN_ITERS` / fragmentation work, which is a separate
  ruling already landed.
* Re-tuning the in-model observable move.
