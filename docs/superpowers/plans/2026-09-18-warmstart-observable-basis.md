# Warm-start observable basis + per-cluster GMM — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fit the GB warm-start components in the OBSERVABLE basis with a
per-cluster Gaussian mixture, and draw back into the sampling basis with the
map's log Jacobian.

**Architecture:** `fit_from_store` converts the previous run's cold-chain leaf
table through `GBObservableFiberBasis.to_internal` at intake; segmentation,
clustering and the mixture fit all run there; the mixture is fitted by the
existing GPU `vec_fit_gmm_min_bic` and stored in the existing
`pack_gmm_components` layout; `WarmStartComponents` draws in observable space,
returns `from_internal`, and scores `logpdf = mixture_logpdf(to_internal(x)) +
log_jacobian(x)`.

**Tech Stack:** numpy, scipy (existing clustering), h5py, the LAT
`gb_observable_basis` / `gmm` / `prior` modules, unittest.

**Spec:** `docs/superpowers/specs/2026-09-18-warmstart-observable-basis-design.md`

## Global Constraints

- Python 3.12; run tests with the worktree runner, never bare `python`:
  `.wtenv/wt_run.sh <worktree>/src .wtenv/<name>.log /Users/mkatz/miniconda3/envs/deving/bin/python -m unittest tests.test_<name>`
- **One python process machine-wide.** Before EVERY python invocation:
  `until ! pgrep -x python >/dev/null && ! pgrep -x python3.12 >/dev/null; do sleep 20; done`
  Match by NAME with `-x`. NEVER `pgrep -f` with the interpreter path — it
  self-matches the waiting shell and deadlocks (hit twice in production).
- **NEVER run or import `tests/test_gbspecial_flow.py`** (allocates 10-26 GB).
- Test batches of at most 6 modules.
- Commit trailer EXACTLY:
  `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`
- Do NOT push. Commit on the branch; the parent merges.
- The shipped `gf_prod_3mo_v8_10w_refereed.npz` (sampling basis, one Gaussian
  per cluster, no `basis` key) MUST keep loading and drawing byte-identically
  under a fixed seed. This is a hard back-compat requirement — production is
  armed against that file today.
- Column indices 3..7 (`phi0`, `cos_iota`, `psi`, `alpha`, `sin_delta`) are
  IDENTICAL in both bases. `CIRCULAR_COLS = {3: 2pi, 5: pi, 6: 2pi}` and
  `COS_IOTA_COL = 4` are therefore reused unchanged. Only 0/1/2/8 change
  meaning: `dist->lnA`, `f0->f_mid`, `Mc->fdot`, `fdot_astro_ratio->Mc`.
- Cross-Tobs v1 policy is UNCHANGED: components are used AS FITTED; only the
  f0 candidate windows derive from the new run's `df`. Intake builds the map
  at the SOURCE run's `1/df`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/lisatools/globalfit/warmstart/basis.py` | **NEW.** Owns the observable column names, builds the map from a store or from stored `map_params`, and round-trips those params. The single place every other stage gets a map from. |
| `src/lisatools/globalfit/warmstart/fit_from_store.py` | Intake conversion, observable cluster features, GMM fit, new writer keys. |
| `src/lisatools/globalfit/warmstart/proposal.py` | Dual-format reader; observable `rvs`/`logpdf` with the Jacobian. |
| `src/lisatools/globalfit/warmstart/match_referee.py` | Converts to sampling at its waveform boundary; merge restricted to across-cluster pairs. |
| `src/lisatools/globalfit/warmstart/opt_snr.py` | Converts to sampling at its waveform boundary. |
| `tests/test_warmstart_basis.py` | **NEW.** Task 1. |
| `tests/test_warmstart_observable_fit.py` | **NEW.** Tasks 2-5. |
| `tests/test_warmstart_observable_proposal.py` | **NEW.** Tasks 6-7. |
| `tests/test_warmstart_flagship_regression.py` | **NEW.** Task 9. |

---

### Task 1: The basis adapter module

**Files:**
- Create: `src/lisatools/globalfit/warmstart/basis.py`
- Test: `tests/test_warmstart_basis.py`

**Interfaces:**
- Consumes: `lisatools.sampling.gb_observable_basis.GBObservableFiberBasis`,
  `GB_INTERNAL_BASIS`.
- Produces:
  - `OBSERVABLE_COLUMN_NAMES: list[str]` — `list(GB_INTERNAL_BASIS)`.
  - `map_params_from_map(m) -> dict` with keys `Tobs` (float), `shear`
    (float), `fiber_coord` (str), `input_basis` (list[str]).
  - `build_map(transform_container, *, Tobs, shear=0.5, fiber_coord="Mc") -> GBObservableFiberBasis`
  - `build_map_from_params(transform_container, params: dict) -> GBObservableFiberBasis`

- [ ] **Step 1: Write the failing test**

```python
"""warmstart.basis: one place to build and round-trip the observable map."""
import unittest

import numpy as np

from lisatools.globalfit.warmstart import basis as wb
from lisatools.sampling.gb_observable_basis import GB_INTERNAL_BASIS

SAMPLING_BASIS = ["dist", "f0", "Mc", "phi0", "cos_iota", "psi", "alpha",
                  "sin_delta", "fdot_astro_ratio"]


class _Container:
    """Minimal stand-in for a TransformContainer: only input_basis is read."""
    def __init__(self, basis=None):
        self.input_basis = list(basis if basis is not None else SAMPLING_BASIS)


class BasisAdapterTest(unittest.TestCase):
    def test_observable_column_names_match_the_map(self):
        self.assertEqual(wb.OBSERVABLE_COLUMN_NAMES, list(GB_INTERNAL_BASIS))
        # the four that change meaning, pinned so a reorder is caught here
        self.assertEqual(wb.OBSERVABLE_COLUMN_NAMES[0], "lnA")
        self.assertEqual(wb.OBSERVABLE_COLUMN_NAMES[1], "f_mid")
        self.assertEqual(wb.OBSERVABLE_COLUMN_NAMES[2], "fdot")
        self.assertEqual(wb.OBSERVABLE_COLUMN_NAMES[8], "Mc")
        # and the five that do NOT
        self.assertEqual(wb.OBSERVABLE_COLUMN_NAMES[3:8],
                         ["phi0", "cos_iota", "psi", "alpha", "sin_delta"])

    def test_params_round_trip_rebuilds_an_equivalent_map(self):
        c = _Container()
        m = wb.build_map(c, Tobs=7.776e6, shear=0.5, fiber_coord="Mc")
        params = wb.map_params_from_map(m)
        self.assertEqual(params["Tobs"], 7.776e6)
        self.assertEqual(params["shear"], 0.5)
        self.assertEqual(params["fiber_coord"], "Mc")
        self.assertEqual(params["input_basis"], SAMPLING_BASIS)
        m2 = wb.build_map_from_params(c, params)
        x = np.array([[1.5, 3.0, 0.45, 1.0, 0.2, 0.5, 2.0, -0.3, 0.01]])
        np.testing.assert_allclose(m.to_internal(x), m2.to_internal(x),
                                   rtol=0, atol=0)

    def test_map_round_trip_is_exact(self):
        c = _Container()
        m = wb.build_map(c, Tobs=7.776e6)
        x = np.array([
            [1.5, 3.0, 0.45, 1.0, 0.2, 0.5, 2.0, -0.3, 0.01],
            [9.7, 20.380376, 0.4678, 6.17, -0.9, 1.40, 4.085, -0.7795, -0.0018],
        ])
        back = m.from_internal(m.to_internal(x))
        np.testing.assert_allclose(back, x, rtol=1e-12, atol=1e-12)

    def test_params_reject_a_basis_mismatch(self):
        c = _Container()
        m = wb.build_map(c, Tobs=7.776e6)
        params = wb.map_params_from_map(m)
        params["input_basis"] = SAMPLING_BASIS[:-1] + ["something_else"]
        with self.assertRaises(ValueError):
            wb.build_map_from_params(c, params)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run it and watch it fail**

Run:
```
until ! pgrep -x python >/dev/null && ! pgrep -x python3.12 >/dev/null; do sleep 20; done
.wtenv/wt_run.sh $PWD/src .wtenv/t1.log /Users/mkatz/miniconda3/envs/deving/bin/python -m unittest tests.test_warmstart_basis
```
Expected: `ModuleNotFoundError: No module named 'lisatools.globalfit.warmstart.basis'`

- [ ] **Step 3: Implement**

```python
"""Observable-basis adapter for the warm-start pipeline.

ONE place the fitter, the referee, the SNR gate and the proposal all get
their ``GBObservableFiberBasis`` from, so the four stages can never drift
apart on Tobs, shear or the fiber coordinate.

The map is a 9-to-9 bijection (see the class docstring); indices 3..7 pass
through unchanged, so the pipeline's ``CIRCULAR_COLS`` / ``COS_IOTA_COL``
constants are valid in BOTH bases.
"""

from __future__ import annotations

from lisatools.sampling.gb_observable_basis import (
    GB_INTERNAL_BASIS,
    GBObservableFiberBasis,
)

#: Observable column names, in map order.
OBSERVABLE_COLUMN_NAMES = list(GB_INTERNAL_BASIS)


def build_map(transform_container, *, Tobs, shear=0.5, fiber_coord="Mc"):
    """Build the observable map for a run.

    ``Tobs`` is the run's ``1.0 / df`` -- NOT ``basis_settings.Tobs``, which
    does not exist on ``FDSettings`` and has broken every FD-domain GB flow
    once. This mirrors the in-model caller in
    ``globalfit/moves/gbspecialstretch.py``.
    """
    return GBObservableFiberBasis(
        transform_container, Tobs=float(Tobs), shear=float(shear),
        fiber_coord=str(fiber_coord),
    )


def map_params_from_map(m) -> dict:
    """The parameters needed to rebuild ``m`` without the source store."""
    return {
        "Tobs": float(m.Tobs),
        "shear": float(m.shear),
        "fiber_coord": str(m.fiber_coord),
        "input_basis": list(m.input_basis),
    }


def build_map_from_params(transform_container, params: dict):
    """Rebuild a map from :func:`map_params_from_map` output.

    Refuses when the container's sampling basis differs from the one the
    params were written with -- a silent mismatch would mis-index every
    column rather than fail.
    """
    want = list(params["input_basis"])
    got = list(getattr(transform_container, "input_basis", []) or [])
    if got != want:
        raise ValueError(
            f"map_params were written for input_basis {want} but this "
            f"container has {got}; refusing to build a mis-indexed map."
        )
    return build_map(
        transform_container,
        Tobs=params["Tobs"],
        shear=params.get("shear", 0.5),
        fiber_coord=params.get("fiber_coord", "Mc"),
    )
```

- [ ] **Step 4: Run it and watch it pass**

Run the same command. Expected: `OK`, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add src/lisatools/globalfit/warmstart/basis.py tests/test_warmstart_basis.py
git commit -m "warmstart: observable-basis adapter, one map for every stage

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Intake conversion and the `--basis` flag

**Files:**
- Modify: `src/lisatools/globalfit/warmstart/fit_from_store.py` (`run` at :475,
  `main` at :618, constants at :48-66)
- Test: `tests/test_warmstart_observable_fit.py`

**Interfaces:**
- Consumes: Task 1's `build_map`, `map_params_from_map`,
  `OBSERVABLE_COLUMN_NAMES`.
- Produces: `run(..., basis: str = "observable", transform_container=None)`;
  module constant `OBSERVABLE_BOUNDED_COLS = {4: (-1.0, 1.0)}`.

Read `run` at `fit_from_store.py:475` before starting: `X` is the `(n, 9)`
leaf table from `load_leaf_table`, and `df_mhz = 1.0 / tobs * 1e3` is the
segmentation bin width.

- [ ] **Step 1: Write the failing test**

```python
"""Intake converts the leaf table to the observable basis."""
import unittest

import numpy as np

from lisatools.globalfit.warmstart import basis as wb
from lisatools.globalfit.warmstart import fit_from_store as ffs

SAMPLING_BASIS = ["dist", "f0", "Mc", "phi0", "cos_iota", "psi", "alpha",
                  "sin_delta", "fdot_astro_ratio"]


class _Container:
    def __init__(self):
        self.input_basis = list(SAMPLING_BASIS)


def _rows(n=64, seed=3):
    rng = np.random.default_rng(seed)
    x = np.empty((n, 9))
    x[:, 0] = rng.uniform(0.5, 30.0, n)        # dist
    x[:, 1] = rng.uniform(3.0, 4.0, n)         # f0 mHz
    x[:, 2] = rng.uniform(0.2, 0.8, n)         # Mc
    x[:, 3] = rng.uniform(0, 2 * np.pi, n)
    x[:, 4] = rng.uniform(-1, 1, n)
    x[:, 5] = rng.uniform(0, np.pi, n)
    x[:, 6] = rng.uniform(0, 2 * np.pi, n)
    x[:, 7] = rng.uniform(-1, 1, n)
    x[:, 8] = rng.uniform(-0.5, 0.5, n)        # ratio
    return x


class IntakeConversionTest(unittest.TestCase):
    def test_to_observable_matches_the_map(self):
        x = _rows()
        m = wb.build_map(_Container(), Tobs=7.776e6)
        got = ffs.to_observable(x, m)
        np.testing.assert_allclose(got, m.to_internal(x), rtol=0, atol=0)
        self.assertEqual(got.shape, x.shape)

    def test_observable_columns_are_the_expected_physics(self):
        x = _rows(4)
        m = wb.build_map(_Container(), Tobs=7.776e6)
        z = ffs.to_observable(x, m)
        # fdot is column 2 and must be positive for positive ratio > -1
        self.assertTrue(np.all(z[:, 2] > 0))
        # the five extrinsic columns pass straight through
        np.testing.assert_allclose(z[:, 3:8], x[:, 3:8], rtol=0, atol=0)

    def test_circular_and_cos_iota_constants_are_basis_independent(self):
        self.assertEqual(ffs.COS_IOTA_COL, 4)
        self.assertEqual(sorted(ffs.CIRCULAR_COLS), [3, 5, 6])
        self.assertEqual(
            [wb.OBSERVABLE_COLUMN_NAMES[i] for i in (3, 5, 6)],
            ["phi0", "psi", "alpha"])

    def test_observable_bounded_cols_drop_the_ratio_rail(self):
        # the +/- ratio_max rail is GONE in the observable basis: fdot is
        # unbounded there. cos_iota keeps its physical bound.
        self.assertEqual(ffs.OBSERVABLE_BOUNDED_COLS, {4: (-1.0, 1.0)})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run it and watch it fail**

Run:
```
until ! pgrep -x python >/dev/null && ! pgrep -x python3.12 >/dev/null; do sleep 20; done
.wtenv/wt_run.sh $PWD/src .wtenv/t2.log /Users/mkatz/miniconda3/envs/deving/bin/python -m unittest tests.test_warmstart_observable_fit
```
Expected: `AttributeError: module ... has no attribute 'to_observable'`

- [ ] **Step 3: Implement**

Add near the constants block (`fit_from_store.py:66`):

```python
#: Bounded columns in the OBSERVABLE basis. cos_iota keeps its physical
#: [-1, 1]; the sampling basis's +/- ratio_max rail is GONE because `fdot`
#: is a raw unbounded coordinate there -- that rail is what produced ratio
#: sigmas with a p90 of 30 across the shipped component set. `Mc` (the
#: fiber, col 8) is bounded below by 0, which the GMM's own per-group
#: mins/maxs already carry, so it is not listed here.
OBSERVABLE_BOUNDED_COLS = {COS_IOTA_COL: (-1.0, 1.0)}


def to_observable(x_all: np.ndarray, obs_map) -> np.ndarray:
    """``(n, 9)`` sampling rows -> ``(n, 9)`` observable rows.

    THE intake seam: after this call no stage of the pipeline sees sampling
    columns until the referee, the SNR gate or the proposal converts back.
    """
    return np.asarray(obs_map.to_internal(np.asarray(x_all, dtype=float)),
                      dtype=float)
```

Then in `run` (`:475`), add `basis: str = "observable"` and
`transform_container=None` to the signature, and immediately after
`load_leaf_table` returns `X`:

```python
    obs_map = None
    if basis == "observable":
        if transform_container is None:
            raise ValueError(
                "basis='observable' needs transform_container (the source "
                "run's stock GB transform) to build the map.")
        obs_map = wb.build_map(transform_container, Tobs=tobs)
        X = to_observable(X, obs_map)
    elif basis != "sampling":
        raise ValueError(f"basis must be 'observable' or 'sampling', got {basis!r}")
```

Add `--basis` to `main`'s parser with `default="observable"`,
`choices=("observable", "sampling")`.

- [ ] **Step 4: Run it and watch it pass**

Same command. Expected: `OK`, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add src/lisatools/globalfit/warmstart/fit_from_store.py tests/test_warmstart_observable_fit.py
git commit -m "warmstart: convert the leaf table to the observable basis at intake

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Observable cluster features

**Files:**
- Modify: `src/lisatools/globalfit/warmstart/fit_from_store.py`
  (`FEAT_NAMES` :66, `make_cluster_features` :232)
- Test: `tests/test_warmstart_observable_fit.py` (append)

**Interfaces:**
- Produces: `OBSERVABLE_FEAT_NAMES = ["f_mid", "fdot", "lnA", "alpha", "sin_delta"]`;
  `make_cluster_features(x_all, basis="observable")`.

- [ ] **Step 1: Write the failing test** (append to the same file)

```python
class ClusterFeatureTest(unittest.TestCase):
    def test_observable_features_are_the_measured_ones(self):
        self.assertEqual(ffs.OBSERVABLE_FEAT_NAMES,
                         ["f_mid", "fdot", "lnA", "alpha", "sin_delta"])

    def test_observable_features_read_the_right_columns(self):
        z = np.zeros((3, 9))
        z[:, 0] = [1.0, 2.0, 3.0]        # lnA
        z[:, 1] = [3.0, 3.1, 3.2]        # f_mid
        z[:, 2] = [1e-16, 2e-16, 3e-16]  # fdot
        z[:, 6] = [0.1, 0.2, 0.3]        # alpha
        z[:, 7] = [-0.5, 0.0, 0.5]       # sin_delta
        f = ffs.make_cluster_features(z, basis="observable")
        self.assertEqual(f.shape, (3, 5))
        np.testing.assert_allclose(f[:, 0], z[:, 1])   # f_mid
        np.testing.assert_allclose(f[:, 1], z[:, 2])   # fdot
        np.testing.assert_allclose(f[:, 2], z[:, 0])   # lnA, already logged
        np.testing.assert_allclose(f[:, 4], z[:, 7])   # sin_delta

    def test_fdot_separates_what_f0_alone_cannot(self):
        """Two sources at one frequency, differing only in fdot."""
        z = np.zeros((40, 9))
        z[:, 1] = 3.0                      # identical f_mid
        z[:20, 2] = 1e-16
        z[20:, 2] = 9e-16                  # 9x apart in fdot
        f_obs = ffs.make_cluster_features(z, basis="observable")
        self.assertGreater(np.ptp(f_obs[:, 1]), 0.0)
        # the SAMPLING metric has no fdot column at all, which is the defect
        self.assertNotIn("fdot", ffs.FEAT_NAMES)

    def test_sampling_basis_features_are_unchanged(self):
        x = _rows(8)
        np.testing.assert_allclose(
            ffs.make_cluster_features(x, basis="sampling"),
            ffs.make_cluster_features(x))   # default must stay back-compatible
```

- [ ] **Step 2: Run it and watch it fail**

Expected: `AttributeError: ... 'OBSERVABLE_FEAT_NAMES'`

- [ ] **Step 3: Implement**

```python
#: Cluster feature space in the OBSERVABLE basis. `lnA` succeeds `ln_dist`
#: (it IS the measured amplitude, already logged by the map) and `fdot`
#: is new -- it is the separator the sampling metric lacks, because two
#: fragments of one source share an f0 and differ in fdot. `Mc` LEAVES the
#: metric: it is the fiber, a flat direction, and clustering on a flat
#: direction is what generates the split artifacts the referee then merges.
OBSERVABLE_FEAT_NAMES = ["f_mid", "fdot", "lnA", "alpha", "sin_delta"]
```

Rewrite `make_cluster_features` to take `basis="sampling"` and branch. The
alpha de-wrapping (histogram rotation) is IDENTICAL in both branches because
alpha is column 6 in both:

```python
def make_cluster_features(x_all: np.ndarray, basis: str = "sampling") -> np.ndarray:
    """rows -> (n, 5) cluster features, alpha rotated so the 2pi wrap sits
    in the emptiest region of the island's alpha histogram.

    ``basis="sampling"``   -> (f0,    Mc,   ln dist, alpha, sin_delta)
    ``basis="observable"`` -> (f_mid, fdot, lnA,     alpha, sin_delta)
    Alpha is column 6 in BOTH bases, so the rotation below is shared.
    """
    alpha = x_all[:, 6]
    hist = np.bincount((alpha / (2 * np.pi) * 36).astype(int) % 36,
                       minlength=36)
    shift = (int(hist.argmin()) + 0.5) * (2 * np.pi / 36)
    alpha_rot = (alpha - shift) % (2 * np.pi)
    if basis == "observable":
        return np.column_stack([x_all[:, 1], x_all[:, 2], x_all[:, 0],
                                alpha_rot, x_all[:, 7]])
    return np.column_stack([x_all[:, 1], x_all[:, 2],
                            np.log(np.maximum(x_all[:, 0], 1e-30)),
                            alpha_rot, x_all[:, 7]])
```

Thread `basis` through the one call site in `run`.

- [ ] **Step 4: Run it and watch it pass**

- [ ] **Step 5: Commit**

```bash
git add src/lisatools/globalfit/warmstart/fit_from_store.py tests/test_warmstart_observable_fit.py
git commit -m "warmstart: cluster on (f_mid, fdot, lnA, sky) in the observable basis

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Per-cluster GMM via the existing GPU min-BIC fitter

**Files:**
- Modify: `src/lisatools/globalfit/warmstart/fit_from_store.py`
- Test: `tests/test_warmstart_observable_fit.py` (append)

**Interfaces:**
- Consumes: `lisatools.sampling.gmm.vec_fit_gmm_min_bic(samples, min_comp,
  max_comp, gpu, verbose, return_components)` returning
  `[weights, means, covs, invcovs, dets, mins, maxs]` (ragged per group),
  mirroring `fstat_proposal.fit_gmm_to_stacked` (`fstat_proposal.py:2506`).
- Produces:
  `fit_cluster_gmms(cluster_rows, *, n_samples=4096, max_comp=12, min_members=25, seed, gpu=None) -> list`
  (the seven ragged lists), where `cluster_rows` is a list of `(m_i, 9)`
  observable member arrays.

**Read first:** `fstat_proposal.fit_gmm_to_stacked` — it draws a FIXED
`n_samples_per_box` per group to rectangularise, then calls the fitter. You
do the same, resampling members WITH REPLACEMENT for clusters smaller than
`n_samples`.

- [ ] **Step 1: Write the failing test** (append)

```python
class ClusterGMMTest(unittest.TestCase):
    def _two_mode_cluster(self, n=400, seed=11):
        """One cluster whose fdot marginal is genuinely bimodal."""
        rng = np.random.default_rng(seed)
        z = np.zeros((n, 9))
        z[:, 0] = rng.normal(1.0, 0.05, n)
        z[:, 1] = rng.normal(3.0, 1e-6, n)
        half = n // 2
        z[:half, 2] = rng.normal(1.0e-16, 2e-18, half)
        z[half:, 2] = rng.normal(9.0e-16, 2e-18, n - half)
        for c, sc in ((3, 0.05), (4, 0.05), (5, 0.05), (6, 0.05), (7, 0.05)):
            z[:, c] = rng.normal(0.3, sc, n)
        z[:, 8] = rng.normal(0.45, 0.01, n)
        return z

    def test_bimodal_cluster_gets_more_than_one_component(self):
        comps = ffs.fit_cluster_gmms([self._two_mode_cluster()],
                                     n_samples=2048, max_comp=6, seed=5)
        weights = comps[0]
        self.assertGreaterEqual(len(weights[0]), 2,
                                "BIC should prefer >1 component for a "
                                "genuinely bimodal cluster")

    def test_unimodal_cluster_stays_at_one_component(self):
        rng = np.random.default_rng(2)
        z = np.zeros((400, 9))
        for c in range(9):
            z[:, c] = rng.normal(1.0, 0.05, 400)
        comps = ffs.fit_cluster_gmms([z], n_samples=2048, max_comp=6, seed=5)
        self.assertEqual(len(comps[0][0]), 1)

    def test_small_cluster_is_capped_at_one_component(self):
        """min_members guards against BIC believing resampled evidence."""
        z = self._two_mode_cluster(n=20)
        comps = ffs.fit_cluster_gmms([z], n_samples=2048, max_comp=6,
                                     min_members=25, seed=5)
        self.assertEqual(len(comps[0][0]), 1)

    def test_weights_sum_to_one_per_cluster(self):
        comps = ffs.fit_cluster_gmms(
            [self._two_mode_cluster(), self._two_mode_cluster(seed=99)],
            n_samples=1024, max_comp=4, seed=5)
        for w in comps[0]:
            self.assertAlmostEqual(float(np.sum(w)), 1.0, places=6)

    def test_output_is_the_seven_ragged_lists(self):
        comps = ffs.fit_cluster_gmms([self._two_mode_cluster()],
                                     n_samples=1024, max_comp=4, seed=5)
        self.assertEqual(len(comps), 7)
        weights, means, covs, invcovs, dets, mins, maxs = comps
        k = len(weights[0])
        self.assertEqual(np.shape(means[0]), (k, 9))
        self.assertEqual(np.shape(covs[0]), (k, 9, 9))
```

- [ ] **Step 2: Run it and watch it fail**

Expected: `AttributeError: ... 'fit_cluster_gmms'`

- [ ] **Step 3: Implement**

```python
def fit_cluster_gmms(cluster_rows, *, n_samples: int = 4096,
                     max_comp: int = 12, min_members: int = 25,
                     seed: int = 7, gpu=None, verbose: bool = False):
    """Per-cluster Gaussian mixtures, min-BIC, on the EXISTING GPU fitter.

    Mirrors :func:`lisatools.sampling.fstat_proposal.fit_gmm_to_stacked`:
    ``vec_fit_gmm_min_bic`` wants a RECTANGULAR
    ``(n_groups, n_samples, n_features)`` block, so each cluster's members
    are resampled to a fixed ``n_samples`` (with replacement when the
    cluster is smaller).

    ``min_members`` is the honesty guard. Resampling cannot manufacture
    structure the members do not contain, but it CAN let BIC believe it has
    more evidence than it does, so a cluster's component cap is
    ``min(max_comp, max(1, n_members // min_members))``.

    Returns the seven ragged lists ``[weights, means, covs, invcovs, dets,
    mins, maxs]``, one entry per cluster, ready for
    :func:`lisatools.sampling.fstat_proposal.pack_gmm_components`.
    """
    from lisatools.sampling.gmm import vec_fit_gmm_min_bic

    rng = np.random.default_rng(seed)
    caps = [min(int(max_comp), max(1, len(r) // int(min_members)))
            for r in cluster_rows]
    block = np.empty((len(cluster_rows), int(n_samples),
                      cluster_rows[0].shape[1]), dtype=float)
    for i, rows in enumerate(cluster_rows):
        idx = rng.integers(0, len(rows), size=int(n_samples))
        block[i] = np.asarray(rows, dtype=float)[idx]

    out = [[] for _ in range(7)]
    # Groups sharing a cap are fitted together; the fitter sweeps a single
    # (min_comp, max_comp) range per call, so one call per distinct cap.
    for cap in sorted(set(caps)):
        sel = [i for i, c in enumerate(caps) if c == cap]
        comps = vec_fit_gmm_min_bic(
            block[sel], min_comp=1, max_comp=int(cap), gpu=gpu,
            verbose=verbose, return_components=True,
        )
        for j, i in enumerate(sel):
            for k in range(7):
                out[k].append((i, comps[k][j]))
    # restore cluster order
    return [[v for _, v in sorted(lst, key=lambda t: t[0])] for lst in out]
```

- [ ] **Step 4: Run it and watch it pass**

- [ ] **Step 5: Commit**

```bash
git add src/lisatools/globalfit/warmstart/fit_from_store.py tests/test_warmstart_observable_fit.py
git commit -m "warmstart: per-cluster GMM through vec_fit_gmm_min_bic

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: The writer — packed GMM layout, basis stamp, map params

**Files:**
- Modify: `src/lisatools/globalfit/warmstart/fit_from_store.py` (`run`, the
  `np.savez_compressed` at :592 and the `meta` dict at :576)
- Test: `tests/test_warmstart_observable_fit.py` (append)

**Interfaces:**
- Consumes: `fstat_proposal.pack_gmm_components(comps) -> dict` with keys
  `gmm_ncomp`, `gmm_weights`, `gmm_means`, `gmm_covs`, `gmm_invcovs`,
  `gmm_dets`, `gmm_mins`, `gmm_maxs`.
- Produces: npz additionally carrying `basis` (str array), `map_params`
  (json str) and the eight `gmm_*` arrays. Per-CLUSTER arrays (`p`, `mult`,
  `n_members`, `island_id`, `f0_window_edges`, `blend`) keep length
  `n_clusters`.

- [ ] **Step 1: Write the failing test** (append)

```python
class WriterTest(unittest.TestCase):
    def test_packed_layout_round_trips_and_ncomp_partitions_it(self):
        from lisatools.sampling.fstat_proposal import (
            pack_gmm_components, unpack_gmm_components)
        g = ClusterGMMTest()
        comps = ffs.fit_cluster_gmms(
            [g._two_mode_cluster(), g._two_mode_cluster(seed=77)],
            n_samples=1024, max_comp=4, seed=5)
        d = pack_gmm_components(comps)
        self.assertEqual(len(d["gmm_ncomp"]), 2)
        self.assertEqual(int(np.sum(d["gmm_ncomp"])), len(d["gmm_weights"]))
        back = unpack_gmm_components(d)
        for k in range(7):
            for a, b in zip(comps[k], back[k]):
                np.testing.assert_allclose(np.asarray(a), np.asarray(b))

    def test_gmm_ncomp_is_the_cluster_partition(self):
        """The referee's merge rule keys on this -- no separate cluster_id."""
        from lisatools.sampling.fstat_proposal import pack_gmm_components
        g = ClusterGMMTest()
        comps = ffs.fit_cluster_gmms([g._two_mode_cluster()],
                                     n_samples=1024, max_comp=4, seed=5)
        d = pack_gmm_components(comps)
        splits = np.cumsum(d["gmm_ncomp"])[:-1]
        self.assertEqual(len(np.split(d["gmm_means"], splits, axis=0)), 1)
```

- [ ] **Step 2: Run it and watch it fail**

Expected: FAIL (imports resolve; the helper under test is Task 4's, so this
passes only once Task 4 is in — run it to confirm the packing round trip,
which is the new assertion).

- [ ] **Step 3: Implement**

In `run`, when `basis == "observable"`, replace the per-cluster
`fit_component` call with `fit_cluster_gmms`, then:

```python
    from lisatools.sampling.fstat_proposal import pack_gmm_components

    packed = pack_gmm_components(gmm_comps)
    meta["basis"] = "observable"
    meta["column_names"] = wb.OBSERVABLE_COLUMN_NAMES
    meta["bounded_cols"] = OBSERVABLE_BOUNDED_COLS
    meta["map_params"] = wb.map_params_from_map(obs_map)
    meta["gmm"] = dict(n_samples=int(gmm_samples), max_comp=int(gmm_max_comp),
                       min_members=int(gmm_min_members))
    np.savez_compressed(
        out, p=ps, mult=mults, n_members=ns, island_id=isl_id,
        f0_window_edges=f0_window_edges, meta=json.dumps(meta), **packed)
```

The `basis == "sampling"` path keeps writing `means`/`covs` exactly as today.

Add `--gmm-samples` (4096), `--gmm-max-comp` (12), `--gmm-min-members` (25)
to `main`'s parser and thread them into `run`.

- [ ] **Step 4: Run it and watch it pass**

- [ ] **Step 5: Commit**

```bash
git add src/lisatools/globalfit/warmstart/fit_from_store.py tests/test_warmstart_observable_fit.py
git commit -m "warmstart: write the packed GMM layout with a basis stamp

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: `WarmStartComponents` reads both formats

**Files:**
- Modify: `src/lisatools/globalfit/warmstart/proposal.py`
  (`from_npz` :416, `__init__` :135)
- Test: `tests/test_warmstart_observable_proposal.py`

**Interfaces:**
- Produces: `WarmStartComponents.from_npz` dispatching on `meta["basis"]`;
  instance attributes `basis: str`, `obs_map` (or `None`), `gmm_ncomp`.

- [ ] **Step 1: Write the failing test**

```python
"""WarmStartComponents: both formats, and the Jacobian on the draw path."""
import json
import os
import tempfile
import unittest

import numpy as np

from lisatools.globalfit.warmstart import basis as wb
from lisatools.globalfit.warmstart.proposal import WarmStartComponents

SAMPLING_BASIS = ["dist", "f0", "Mc", "phi0", "cos_iota", "psi", "alpha",
                  "sin_delta", "fdot_astro_ratio"]


class _Container:
    def __init__(self):
        self.input_basis = list(SAMPLING_BASIS)


class LegacyFormatTest(unittest.TestCase):
    def test_a_file_without_a_basis_key_is_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "legacy.npz")
            means = np.array([[1.5, 3.0, 0.45, 1.0, 0.2, 0.5, 2.0, -0.3, 0.01]])
            covs = np.eye(9)[None] * 1e-4
            np.savez(path, means=means, covs=covs, p=np.array([1.0]),
                     mult=np.array([1.0]), n_members=np.array([50]),
                     island_id=np.array([0]),
                     f0_window_edges=np.array([[2.9, 3.1]]),
                     meta=json.dumps({"tobs": 7.776e6,
                                      "column_names": SAMPLING_BASIS}))
            c = WarmStartComponents.from_npz(path, new_tobs=1.5552e7)
            self.assertEqual(c.basis, "sampling")
            self.assertIsNone(c.obs_map)
            x = c.rvs(16)
            self.assertEqual(x.shape, (16, 9))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run it and watch it fail**

Expected: `AttributeError: 'WarmStartComponents' object has no attribute 'basis'`

- [ ] **Step 3: Implement**

In `from_npz`, read `meta.get("basis", "sampling")`. For `"sampling"`, keep
the existing `required` key list and construction verbatim. For
`"observable"`, require the packed keys instead and build:

```python
            basis = meta.get("basis", "sampling")
            if basis == "observable":
                from lisatools.sampling.fstat_proposal import unpack_gmm_components
                required = ("gmm_ncomp", "gmm_weights", "gmm_means",
                            "gmm_covs", "gmm_invcovs", "gmm_dets",
                            "gmm_mins", "gmm_maxs", "p", "mult",
                            "n_members", "island_id", "f0_window_edges",
                            "meta")
                missing = [k for k in required if k not in d]
                if missing:
                    raise ValueError(
                        f"observable-basis warm-start npz {path} is missing "
                        f"keys {missing}.")
                comps = unpack_gmm_components(d)
                ...
```

Set `self.basis`, `self.gmm_ncomp`, and `self.obs_map = None` until Task 7
supplies a container. The `column_names` lockstep check compares against
`wb.OBSERVABLE_COLUMN_NAMES` when the basis is observable, and against
`COLUMN_NAMES` otherwise.

- [ ] **Step 4: Run it and watch it pass**

- [ ] **Step 5: Commit**

```bash
git add src/lisatools/globalfit/warmstart/proposal.py tests/test_warmstart_observable_proposal.py
git commit -m "warmstart: WarmStartComponents reads both component formats

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Observable `rvs` / `logpdf` with the log Jacobian

**Files:**
- Modify: `src/lisatools/globalfit/warmstart/proposal.py` (`rvs` :484,
  `logpdf` :539)
- Test: `tests/test_warmstart_observable_proposal.py` (append)

**Interfaces:**
- Consumes: Task 1's `build_map_from_params`; `obs_map.from_internal`,
  `to_internal`, `log_jacobian`.
- Produces: `WarmStartComponents.attach_transform(container)` which builds
  `self.obs_map` from the stored `map_params`; `rvs` returning SAMPLING
  columns; `logpdf` including `log_jacobian`.

- [ ] **Step 1: Write the failing test** (append)

```python
class ObservableDrawTest(unittest.TestCase):
    def _make(self, tmp, mean_z, cov_z):
        from lisatools.sampling.fstat_proposal import pack_gmm_components
        path = os.path.join(tmp, "obs.npz")
        comps = [[np.array([1.0])], [mean_z[None]], [cov_z[None]],
                 [np.linalg.inv(cov_z)[None]],
                 [np.array([np.linalg.det(cov_z)])],
                 [mean_z - 10.0], [mean_z + 10.0]]
        packed = pack_gmm_components(comps)
        m = wb.build_map(_Container(), Tobs=7.776e6)
        np.savez(path, p=np.array([1.0]), mult=np.array([1.0]),
                 n_members=np.array([90]), island_id=np.array([0]),
                 f0_window_edges=np.array([[2.9, 3.1]]),
                 meta=json.dumps({
                     "tobs": 7.776e6, "basis": "observable",
                     "column_names": wb.OBSERVABLE_COLUMN_NAMES,
                     "map_params": wb.map_params_from_map(m)}),
                 **packed)
        c = WarmStartComponents.from_npz(path, new_tobs=1.5552e7)
        c.attach_transform(_Container())
        return c

    def test_rvs_returns_sampling_columns_near_the_mapped_mean(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = wb.build_map(_Container(), Tobs=7.776e6)
            x0 = np.array([[9.69, 20.380376, 0.4678, 6.17, -0.9, 1.40,
                            4.085, -0.7795, -0.0018]])
            z0 = np.asarray(m.to_internal(x0))[0]
            cov = np.diag(np.maximum(np.abs(z0) * 1e-3, 1e-12) ** 2)
            c = self._make(tmp, z0, cov)
            x = c.rvs(512)
            self.assertEqual(x.shape, (512, 9))
            # f0 (sampling col 1) lands on the source, not scattered
            self.assertLess(abs(np.median(x[:, 1]) - 20.380376), 1e-4)

    def test_logpdf_includes_the_log_jacobian(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = wb.build_map(_Container(), Tobs=7.776e6)
            x0 = np.array([[9.69, 20.380376, 0.4678, 6.17, -0.9, 1.40,
                            4.085, -0.7795, -0.0018]])
            z0 = np.asarray(m.to_internal(x0))[0]
            cov = np.diag(np.maximum(np.abs(z0) * 1e-3, 1e-12) ** 2)
            c = self._make(tmp, z0, cov)
            lp = c.logpdf(x0)
            # the density WITHOUT the jacobian differs by exactly it
            z = np.asarray(m.to_internal(x0))
            gauss = (-0.5 * float((z - z0) @ np.linalg.inv(cov) @ (z - z0).T)
                     - 0.5 * np.log(np.linalg.det(2 * np.pi * cov)))
            jac = float(np.asarray(m.log_jacobian(x0))[0])
            self.assertAlmostEqual(float(lp[0]), gauss + jac, places=6)

    def test_draws_are_self_consistent_between_rvs_and_logpdf(self):
        """logpdf must be finite at every point rvs produces."""
        with tempfile.TemporaryDirectory() as tmp:
            m = wb.build_map(_Container(), Tobs=7.776e6)
            x0 = np.array([[9.69, 20.380376, 0.4678, 6.17, -0.9, 1.40,
                            4.085, -0.7795, -0.0018]])
            z0 = np.asarray(m.to_internal(x0))[0]
            cov = np.diag(np.maximum(np.abs(z0) * 1e-3, 1e-12) ** 2)
            c = self._make(tmp, z0, cov)
            x = c.rvs(64)
            self.assertTrue(np.all(np.isfinite(c.logpdf(x))))
```

- [ ] **Step 2: Run it and watch it fail**

Expected: `AttributeError: ... 'attach_transform'`

- [ ] **Step 3: Implement**

```python
    def attach_transform(self, transform_container):
        """Build the observable map from the stored ``map_params``.

        Separate from ``from_npz`` because the container is a RUNTIME object
        (the stock GB transform), and the pre-build fit must stay
        pickle/deepcopy-safe -- the sprint-wide rule.
        """
        if self.basis != "observable":
            return
        from . import basis as _wb
        self.obs_map = _wb.build_map_from_params(
            transform_container, self._map_params)
```

In `rvs`, when `self.basis == "observable"`, draw `z` from the mixture as
today and `return np.asarray(self.obs_map.from_internal(z))`. Circular
wrapping applies to the returned SAMPLING columns at indices 3/5/6, which are
the same indices in both bases.

In `logpdf`, when observable:

```python
        z = np.asarray(self.obs_map.to_internal(np.atleast_2d(x)))
        return (self._mixture_logpdf(z)
                + np.asarray(self.obs_map.log_jacobian(np.atleast_2d(x))))
```

- [ ] **Step 4: Run it and watch it pass**

- [ ] **Step 5: Commit**

```bash
git add src/lisatools/globalfit/warmstart/proposal.py tests/test_warmstart_observable_proposal.py
git commit -m "warmstart: draw in the observable basis, score with the log Jacobian

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Referee boundary conversion and merge-across-clusters-only

**Files:**
- Modify: `src/lisatools/globalfit/warmstart/match_referee.py`,
  `src/lisatools/globalfit/warmstart/opt_snr.py`,
  `src/lisatools/globalfit/warmstart/referee_apply.py`
- Test: `tests/test_warmstart_observable_proposal.py` (append)

**Interfaces:**
- Consumes: Task 1's `build_map_from_params`, Task 5's `gmm_ncomp`.
- Produces: both judging stages accept an observable npz and convert at their
  waveform boundary; `referee_apply` never merges two components of the SAME
  cluster.

- [ ] **Step 1: Write the failing test** (append)

```python
class MergeScopeTest(unittest.TestCase):
    def test_merge_candidate_pairs_never_cross_within_a_cluster(self):
        from lisatools.globalfit.warmstart.referee_apply import (
            merge_candidate_pairs)
        # three clusters: 2, 1 and 3 components -> flat indices
        ncomp = np.array([2, 1, 3])
        pairs = merge_candidate_pairs(ncomp, island_id=np.array([0, 0, 0]))
        # (0,1) and (3,4),(3,5),(4,5) are WITHIN clusters and must be absent
        for bad in [(0, 1), (3, 4), (3, 5), (4, 5)]:
            self.assertNotIn(bad, pairs)
        # across-cluster pairs in the same island survive
        self.assertIn((0, 2), pairs)
        self.assertIn((2, 3), pairs)

    def test_single_component_clusters_behave_as_before(self):
        from lisatools.globalfit.warmstart.referee_apply import (
            merge_candidate_pairs)
        ncomp = np.array([1, 1, 1])
        pairs = merge_candidate_pairs(ncomp, island_id=np.array([0, 0, 1]))
        self.assertIn((0, 1), pairs)      # same island
        self.assertNotIn((0, 2), pairs)   # different island
```

- [ ] **Step 2: Run it and watch it fail**

Expected: `ImportError: cannot import name 'merge_candidate_pairs'`

- [ ] **Step 3: Implement**

```python
def merge_candidate_pairs(gmm_ncomp, island_id):
    """Flat component-index pairs eligible for the auto-merge test.

    Same island, DIFFERENT cluster. Mixture siblings are exempt: a cluster's
    K components are a deliberate multi-modal description of ONE source, so
    they cross-match highly and the moment-matched merge would collapse them
    straight back into the single Gaussian the observable-basis design
    exists to avoid. Genuine split artifacts land in different clusters and
    are still merged exactly as before.

    ``gmm_ncomp`` is the per-cluster component count from
    ``pack_gmm_components``; it partitions the flat arrays, so no separate
    cluster id is needed. A legacy set is all-ones and reproduces the old
    behaviour exactly.
    """
    ncomp = np.asarray(gmm_ncomp, dtype=int)
    cluster_of = np.repeat(np.arange(len(ncomp)), ncomp)
    isl = np.asarray(island_id)
    isl_of = np.repeat(isl, ncomp) if len(isl) == len(ncomp) else isl
    pairs = []
    n = len(cluster_of)
    for i in range(n):
        for j in range(i + 1, n):
            if cluster_of[i] == cluster_of[j]:
                continue
            if isl_of[i] != isl_of[j]:
                continue
            pairs.append((i, j))
    return pairs
```

Rewire `referee_apply`'s existing same-island pair loop to iterate
`merge_candidate_pairs(...)`. In `match_referee` and `opt_snr`, at the point
each builds waveforms, convert observable means to sampling first:

```python
    if meta.get("basis") == "observable":
        m = wb.build_map_from_params(transform_container, meta["map_params"])
        means_sampling = np.asarray(m.from_internal(means))
    else:
        means_sampling = means
```

- [ ] **Step 4: Run it and watch it pass**

- [ ] **Step 5: Commit**

```bash
git add src/lisatools/globalfit/warmstart/ tests/test_warmstart_observable_proposal.py
git commit -m "warmstart: referee merges across clusters only; judging stages convert at their boundary

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: The flagship regression test

**Files:**
- Create: `tests/test_warmstart_flagship_regression.py`

This is the test that encodes the defect the whole plan exists to fix. It is
a UNIT test over a synthetic cluster built from the real flagship numbers —
it does NOT read the 3-month store (too large for this laptop).

**Interfaces:**
- Consumes: Tasks 1-7.

- [ ] **Step 1: Write the failing test**

```python
"""The flagship 20.380377 mHz source survives the round trip as ONE source.

Measured 2026-09-18 from gf_prod_3mo_v8_10w_refereed.npz component 5571:
f0 offset -0.001 uHz (sigma 0.043 uHz), fdot at the mean 1.0302e-13 against
truth 1.0245e-13 (x1.01), ratio -0.0018 +/- 0.271, p 0.97, 97 members,
mult 1.000. The component set was GOOD; the sampling-basis coordinates were
the problem. This pins that a draw from the observable-basis component lands
on the chirp ridge rather than scattered across it.
"""
import json
import os
import tempfile
import unittest

import numpy as np

from lisatools.globalfit.warmstart import basis as wb
from lisatools.globalfit.warmstart import fit_from_store as ffs
from lisatools.globalfit.warmstart.proposal import WarmStartComponents

SAMPLING_BASIS = ["dist", "f0", "Mc", "phi0", "cos_iota", "psi", "alpha",
                  "sin_delta", "fdot_astro_ratio"]
F0_TRUTH_MHZ = 20.380377
FDOT_TRUTH = 1.0245e-13
MSUN = 4.925490947641267e-6


class _Container:
    def __init__(self):
        self.input_basis = list(SAMPLING_BASIS)


def _phys_fdot(row):
    dist, f0, mc, _, _, _, _, _, r = row
    return ((96 / 5) * np.pi ** (8 / 3) * (MSUN * mc) ** (5 / 3)
            * (f0 * 1e-3) ** (11 / 3) * (1 + r))


def _flagship_members(n=97, seed=17):
    """97 posterior samples matching the measured component-5571 spreads."""
    rng = np.random.default_rng(seed)
    x = np.empty((n, 9))
    x[:, 0] = rng.normal(9.693, 2.397, n).clip(0.1)
    x[:, 1] = rng.normal(20.380376, 4.3e-5, n)
    x[:, 2] = rng.normal(0.4678, 0.0706, n).clip(0.05)
    x[:, 3] = rng.normal(6.174, 1.840, n) % (2 * np.pi)
    x[:, 4] = rng.normal(-0.95, 0.05, n).clip(-1, 1)
    x[:, 5] = rng.normal(1.395, 0.880, n) % np.pi
    x[:, 6] = rng.normal(4.085, 0.0264, n) % (2 * np.pi)
    x[:, 7] = rng.normal(-0.7795, 0.0139, n).clip(-1, 1)
    x[:, 8] = rng.normal(-0.0018, 0.271, n)
    return x


class FlagshipRegressionTest(unittest.TestCase):
    def test_one_component_and_draws_land_on_the_chirp_ridge(self):
        from lisatools.sampling.fstat_proposal import pack_gmm_components
        members = _flagship_members()
        m = wb.build_map(_Container(), Tobs=7.776e6)
        z = ffs.to_observable(members, m)

        comps = ffs.fit_cluster_gmms([z], n_samples=4096, max_comp=12,
                                     min_members=25, seed=5)
        # 97 members // 25 = 3, so up to 3 are ALLOWED; a single source
        # should still be described by one.
        self.assertEqual(len(comps[0][0]), 1,
                         "an unblended single source must stay one component")

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "flagship.npz")
            np.savez(path, p=np.array([0.97]), mult=np.array([1.0]),
                     n_members=np.array([len(members)]),
                     island_id=np.array([0]),
                     f0_window_edges=np.array([[20.37, 20.39]]),
                     meta=json.dumps({
                         "tobs": 7.776e6, "basis": "observable",
                         "column_names": wb.OBSERVABLE_COLUMN_NAMES,
                         "map_params": wb.map_params_from_map(m)}),
                     **pack_gmm_components(comps))
            c = WarmStartComponents.from_npz(path, new_tobs=1.5552e7)
            c.attach_transform(_Container())
            draws = c.rvs(4000)

        # f0 lands on the source
        self.assertLess(abs(np.median(draws[:, 1]) - F0_TRUTH_MHZ), 5e-5)
        # and so does the CHIRP, which is the whole point
        fdot = np.array([_phys_fdot(r) for r in draws])
        ratio = fdot / FDOT_TRUTH
        frac_on_ridge = float(np.mean((ratio > 1 / 1.3) & (ratio < 1.3)))
        self.assertGreater(
            frac_on_ridge, 0.60,
            f"only {frac_on_ridge:.0%} of draws within 1.3x of the true "
            f"fdot; the observable component should keep them on the ridge")
        self.assertLess(float(np.mean(fdot < 0)), 0.02,
                        "negative chirps are unphysical here and were 42% of "
                        "the fragment leaves this design replaces")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run it and watch it fail** (before Tasks 1-7 land it errors on import; after, it must PASS)

- [ ] **Step 3: No new implementation** — this task is the gate on Tasks 1-7.
      If it fails, the defect is in those tasks; fix there.

- [ ] **Step 4: Run the full warm-start batch**

```
until ! pgrep -x python >/dev/null && ! pgrep -x python3.12 >/dev/null; do sleep 20; done
.wtenv/wt_run.sh $PWD/src .wtenv/ws_all.log /Users/mkatz/miniconda3/envs/deving/bin/python -m unittest \
  tests.test_warmstart_basis tests.test_warmstart_observable_fit \
  tests.test_warmstart_observable_proposal tests.test_warmstart_flagship_regression \
  tests.test_warmstart_proposal tests.test_warmstart_fit_from_store
```
Expected: `OK`. (The last two are the EXISTING suites; they must stay green —
they are the back-compat guarantee for the shipped refereed npz.)

- [ ] **Step 5: Commit**

```bash
git add tests/test_warmstart_flagship_regression.py
git commit -m "warmstart: flagship regression -- one component, draws on the chirp ridge

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Self-review notes

**Spec coverage.** 4.1 intake -> Task 2. 4.2 segmentation -> Task 2 (f_mid is
column 1 in the observable basis, so the existing `segment_f0` call operates
on it unchanged once intake has converted; no code change needed, which is
why there is no separate task). 4.3 clustering -> Task 3. 4.4 GMM -> Task 4.
4.5 proposal -> Tasks 6-7. 4.6 referee/SNR -> Task 8. Section 5 file format ->
Task 5. Section 6 testing items 1-5 -> Tasks 1, 6, 7, 9; item 6 (the
stale-birth cluster read-out) is a cluster measurement, not a unit test, and
is listed in the handoff below.

**Deliberately deferred.** The spec's `FullGaussianMixtureModel` swap for the
density side (retiring `window_df`) is NOT in this plan. It is a performance
and simplification change on top of a correctness change, and doing both at
once would make a regression ambiguous. Land these nine tasks, confirm the
flagship gate, then do the swap as its own plan.

**Not covered by any task, by design:** regenerating the production component
set from the 3-month store. That is a cluster job, not a laptop one — the
store is ~30 GB. Run it after this plan lands:
`python -m lisatools.globalfit.warmstart.fit_from_store <3mo store> --basis observable ...`
then the referee and apply stages, then point `GB_WARM_START_COMPONENTS` at
the result.
