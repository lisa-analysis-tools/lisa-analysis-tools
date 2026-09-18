"""Warm-start GB proposal fitter: previous-run store -> Gaussian components npz.

Workstream B (docs/6mo-run-prep.md) real-data builder implementing the
3-stage pipeline of docs/warm-start-gb-proposal.md on a finished run's
cold-chain leaf table (validated prototype: proto_warmstart_cluster.py):

  1. f0 DENSITY-VALLEY segmentation at 1/Tobs bins with a count floor
     (0.5% of posterior samples per bin); islands padded by one bin.
  2. Within-island split (SWAPPABLE strategy, ``split(island_rows) ->
     labels``): robust-MAD-whiten (f0, Mc, ln dist, alpha, sin_delta),
     SINGLE-linkage on a <=1500-row subsample cut at 2.0 whitened units,
     nearest-centroid assignment with junk radius 6 (label -1), plus the
     v1 satellite-fragment merge pass.
  3. Cluster -> component: Gaussian mean/cov over the full 9-col sampled
     basis with CIRCULAR handling for phi0/psi/alpha (v1) and covariance
     eigenvalue floors (v1); inclusion probability
     p = distinct posterior samples containing a member / n_samples,
     and leaf multiplicity mult = members / distinct samples.

Waveform-free by design: numpy/scipy/h5py only, no lisatools import.
CPU budget: run with OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
OPENBLAS_NUM_THREADS=1. Reads the chain iteration-by-iteration (streamed).

Sampled basis (verified against stock/erebor/gb.py init_sampling_info and
the store's column ranges; the store keeps NO column names itself):
  0 dist [kpc], 1 f0 [mHz], 2 Mc [Msol], 3 phi0 [rad, 2pi], 4 cos_iota,
  5 psi [rad, pi], 6 alpha [rad, 2pi], 7 sin_delta, 8 fdot_astro_ratio.

Usage:
  python warmstart_fit_from_store.py --store <h5> [--last-k N]
      [--tobs 7776000] [--out components.npz]
"""
from __future__ import annotations

import argparse
import json
import resource
import subprocess
import time

import h5py
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import cdist, pdist
from scipy.stats import norm as _norm

COLUMN_NAMES = ["dist", "f0", "Mc", "phi0", "cos_iota", "psi", "alpha",
                "sin_delta", "fdot_astro_ratio"]
# circular columns in the sampled basis: column -> period
CIRCULAR_COLS = {3: 2.0 * np.pi, 5: np.pi, 6: 2.0 * np.pi}
# BOUNDED columns (ruling 2026-09-11): fitted by 1-D truncated-normal MLE
# instead of raw sample moments -- benched on the v8 24w gated fit
# (2-fold CV test log-density), the truncated normal beats the plain
# Gaussian on 92% (cos_iota) / 97% (fdot_ratio) of components and beats
# the atanh infinite-basis refit on both. The bounds ride in the npz
# meta ("bounded_cols") and WarmStartComponents truncates the proposal
# mixture accordingly (rectangle normalization + in-box rejection).
# Col 8's bound is +-ratio_max (GBSettings.fdot_astro_ratio_max; CLI
# --ratio-max, default 5.0). Rail piles fit as means AT/BEYOND the box
# edge with honest widths -- that is the truncated law working, not a
# fit failure.
COS_IOTA_COL = 4
RATIO_COL = 8
# cluster-feature space: (f0 [mHz], Mc, ln dist, alpha, sin_delta)
FEAT_NAMES = ["f0", "Mc", "ln_dist", "alpha", "sin_delta"]

# --------------------------------------------------------------------------
# OBSERVABLE basis (2026-09-18). The fit runs in the coordinates the DATA
# constrains rather than the ones the sampler uses:
#
#   0 lnA, 1 f_mid [Hz], 2 fdot [Hz/s], 3 phi0, 4 cos_iota, 5 psi,
#   6 alpha, 7 sin_delta, 8 Mc [Msol] (the fiber)
#
# Indices 3..7 are IDENTICAL in both bases, so CIRCULAR_COLS and
# COS_IOTA_COL above are reused unchanged; only 0/1/2/8 change meaning.
#
# UNITS TRAP: the map returns f_mid and fdot in HZ, not mHz -- the sampling
# basis stores f0 in mHz. Every frequency-scaled quantity downstream (the
# segmentation bin width, the whitening scale floor, the proposal's f0
# candidate window) must therefore be taken in the units of the basis being
# worked in. ``basis_df`` is the single place that choice is made.
#: Bounded columns in the OBSERVABLE basis. cos_iota keeps its physical
#: [-1, 1]; the sampling basis's +/- ratio_max rail is GONE because `fdot`
#: is a raw unbounded coordinate there -- that rail is what produced ratio
#: sigmas with a p90 of 30 across the shipped component set. `Mc` (the
#: fiber, col 8) is bounded below by 0, which the GMM's own per-group
#: mins/maxs already carry, so it is not listed here.
OBSERVABLE_BOUNDED_COLS = {COS_IOTA_COL: (-1.0, 1.0)}

#: Cluster feature space in the OBSERVABLE basis. `lnA` succeeds `ln_dist`
#: (it IS the measured amplitude, already logged by the map) and `fdot`
#: is new -- it is the separator the sampling metric lacks, because two
#: fragments of one source share an f0 and differ in fdot. `Mc` LEAVES the
#: metric: it is the fiber, a flat direction, and clustering on a flat
#: direction is what generates the split artifacts the referee then merges.
OBSERVABLE_FEAT_NAMES = ["f_mid", "fdot", "lnA", "alpha", "sin_delta"]

#: Relative floor for the observable `fdot` whitening scale (see
#: :func:`cluster_scale_floor`).
FDOT_SCALE_FLOOR_FRAC = 1e-3


class _DefaultGBBasisContainer:
    """Stand-in transform container for :func:`build_map`.

    ``GBObservableFiberBasis`` pins nothing per leaf for GB, so it reads
    exactly one attribute off the container: ``input_basis``. This module
    already declares that basis as :data:`COLUMN_NAMES` (verified against
    ``stock/erebor/gb.py``), so the CLI does not need to import and build a
    real stock container -- which would also break this module's
    "waveform-free, numpy/scipy/h5py only" contract.

    An explicit container always wins and is checked against
    :data:`COLUMN_NAMES`, so a genuinely different sampling basis fails
    loudly instead of mis-indexing every column.
    """

    input_basis = COLUMN_NAMES


def basis_df(tobs: float, basis: str) -> float:
    """Frequency bin width ``1/Tobs`` in the units column 1 is stored in.

    ``"sampling"`` -> mHz (the stored ``f0``); ``"observable"`` -> Hz (the
    map's ``f_mid``). Passing the mHz width against Hz data would make the
    segmentation bins 1000x too wide and collapse the whole band into a
    few islands.
    """
    if basis == "observable":
        return 1.0 / float(tobs)
    return 1.0 / float(tobs) * 1e3


def to_observable(x_all: np.ndarray, obs_map) -> np.ndarray:
    """``(n, 9)`` sampling rows -> ``(n, 9)`` observable rows.

    THE intake seam: after this call no stage of the pipeline sees sampling
    columns until the referee, the SNR gate or the proposal converts back.
    """
    return np.asarray(obs_map.to_internal(np.asarray(x_all, dtype=float)),
                      dtype=float)


def build_observable_map(tobs: float, transform_container=None,
                         shear: float = 0.5, fiber_coord: str = "Mc"):
    """The intake map, built at the SOURCE run's ``1/df`` (cross-Tobs v1).

    Lazily imported so the sampling-basis path keeps this module free of
    any lisatools dependency.
    """
    from . import basis as wb

    if transform_container is None:
        transform_container = _DefaultGBBasisContainer()
    else:
        got = list(getattr(transform_container, "input_basis", []) or [])
        if got != COLUMN_NAMES:
            raise ValueError(
                f"transform_container input_basis {got} != this fitter's "
                f"{COLUMN_NAMES}; the leaf-table column meanings would not "
                "match the map's.")
    return wb.build_map(transform_container, Tobs=float(tobs),
                        shear=float(shear), fiber_coord=str(fiber_coord))


# Stage-1 valley split (2026-08-24 fix): at final leaf density (~900
# leaves/walker) the confusion band is CONTINUOUSLY occupied above the
# global count floor, so floor-only segmentation returned ONE island for
# the whole band (and one useless component). Islands are now recursively
# split at genuine density valleys, and islands wider than
# MAX_ISLAND_BINS are force-split at their weakest interior bin — a
# multi-thousand-row blended island is unresolvable by the 5-D subsample
# linkage anyway.
VALLEY_FRAC = 0.35                 # interior min <= frac * smaller flanking peak
MAX_ISLAND_BINS = 64               # force-split wider islands (64 x 1/Tobs)
SUB = 1500                         # linkage subsample size
T_CUT = 2.0                        # single-linkage cut, whitened units
JUNK_RADIUS = 6.0                  # nearest-centroid junk exclusion
SAT_MERGE_CUT = 2.0                # centroid merge distance (any pair)
SAT_FRAC = 0.05                    # satellite: n < frac * n_big ...
SAT_RADIUS = 4.0                   # ... and centroid within this radius
MIN_FRAC = 0.01                    # min members as fraction of n_samples
CORR_EIG_FLOOR = 1e-4              # eigenvalue floor on the correlation mat


def rss_gb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / 1024 ** 3  # macOS: bytes


# --------------------------------------------------------------------------
# stage 0: streamed cold-chain leaf-table extraction
# --------------------------------------------------------------------------
def load_leaf_table(store: str, last_k: int | None, max_iter: int | None = None):
    """Read the cold-chain GB leaf table over the last K stored iterations.

    Returns (X (n, 9) float64, sample_id (n,) int64, info dict).
    sample_id = stored_iteration_index * nwalkers + walker.
    Handles both the gf_format_version-2 layout
    [it, nsamplers, ntemps, nwalkers, nleaves, ndim] and the legacy
    [it, ntemps, nwalkers, nleaves, ndim] one (top group "mcmc").
    """
    with h5py.File(store, "r") as f:
        g = f["global_fit"] if "global_fit" in f else f["mcmc"]
        chain = g["chain"]["gb"]
        inds = g["inds"]["gb"]
        ll = g["log_like"]
        six_d = chain.ndim == 6  # extra nsamplers axis
        cold = (0, 0) if six_d else (0,)
        nwalkers = chain.shape[-3]
        ndim = chain.shape[-1]

        # written iterations: any walker with a nonzero cold-chain log_like
        llv = ll[(slice(None),) + cold + (slice(None),)]
        written = np.flatnonzero(np.any(llv != 0.0, axis=-1))
        if max_iter is not None:
            written = written[written <= max_iter]
        # TORN-TAIL GUARD (replaces the old hardcoded 471 clamp, which
        # silently truncated any store past that iteration): a crash
        # mid-save can leave log_like written while the coord slab is
        # still zeros (the v5 it=472 signature). Drop trailing written
        # iterations whose coords are entirely zero.
        while len(written) and not chain[(written[-1],) + cold][:8].any():
            print(f"  dropping torn tail iteration {written[-1]} "
                  "(log_like written, coord slab empty)")
            written = written[:-1]
        if last_k is not None:
            written = written[-last_k:]

        rows, sids = [], []
        zero_iters = 0
        for it in written:
            ind_it = inds[(it,) + cold]              # (nwalkers, nleaves)
            c_it = chain[(it,) + cold]               # (nwalkers, nleaves, ndim)
            w_idx, l_idx = np.nonzero(ind_it)
            r = c_it[w_idx, l_idx]
            # KEEP-WINDOW GUARD (2026-08-24): snapshot EXTRACT h5s carry
            # FULL inds but coords only for the extractor's keep window
            # (--keep default 3) -- alive-flagged rows outside it read as
            # ZEROS and poisoned a fit with a 1.08M-row f0=0 mega
            # component. A physical GB row can never have f0 == 0 (band
            # floor 0.5556 mHz), so drop zero-coord rows and count the
            # iterations they emptied.
            good = r[:, 1] > 0.0
            if not good.all():
                if not good.any():
                    zero_iters += 1
                    continue
                r, w_idx = r[good], w_idx[good]
            rows.append(r)
            sids.append(np.int64(it) * nwalkers + w_idx)
        if zero_iters:
            print(f"  WARNING: {zero_iters}/{len(written)} requested "
                  "iterations have EMPTY coord slabs (keep-window extract?)"
                  " -- effective window is only the remainder.")
            written = np.array([int(s[0]) // nwalkers for s in sids])
        X = np.concatenate(rows, axis=0)
        sample_id = np.concatenate(sids, axis=0)
        info = dict(
            iterations=[int(written.min()), int(written.max())],
            n_iterations=int(len(written)),
            nwalkers=int(nwalkers), ndim=int(ndim),
            n_samples=int(len(written) * nwalkers),
            leaves_per_walker=float(len(X) / (len(written) * nwalkers)),
            store_iteration_attr=int(g.attrs.get("iteration", -1)),
        )
    return X, sample_id, info


# --------------------------------------------------------------------------
# stage 1: f0 density-valley segmentation
# --------------------------------------------------------------------------
def segment_f0(f0_mhz: np.ndarray, df_mhz: float, n_samples: int):
    """Islands = contiguous 1/Tobs bins above the count floor, padded by 1.

    Returns (bin_index_per_row, island list [(b0, b1) half-open bins],
    f_edge0, floor).
    """
    f_lo = np.floor(f0_mhz.min() / df_mhz) * df_mhz
    idx = ((f0_mhz - f_lo) / df_mhz).astype(np.int64)
    counts = np.bincount(idx)
    floor = max(5, int(0.005 * n_samples))
    hot = counts >= floor
    edges = np.flatnonzero(np.diff(np.concatenate(
        ([0], hot.view(np.int8), [0]))))
    islands = [(max(b0 - 1, 0), min(b1 + 1, len(counts)))     # 1-bin pad
               for b0, b1 in zip(edges[::2], edges[1::2])]
    merged = []
    for b0, b1 in islands:                                    # merge touching
        if merged and b0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b1)
        else:
            merged.append([b0, b1])
    # VALLEY SPLIT (2026-08-24): a global floor is not a valley detector
    # once the confusion band is continuously occupied (final v5 density
    # returned ONE island for the whole band). Recursively split each hot
    # run at interior bins that are genuine density valleys (below
    # VALLEY_FRAC of the smaller flanking peak, or below the floor), and
    # FORCE-split anything still wider than MAX_ISLAND_BINS at its weakest
    # interior bin -- blended-confusion islands beyond that width cannot
    # be resolved by the 5-D subsample linkage downstream anyway.
    final = []
    stack = [tuple(m) for m in merged]
    while stack:
        b0, b1 = stack.pop()
        width = b1 - b0
        seg = counts[b0:b1]
        if width <= 3:
            final.append((b0, b1))
            continue
        interior = seg[1:-1]
        m = int(np.argmin(interior)) + 1          # weakest interior bin
        left_peak = int(seg[:m].max())
        right_peak = int(seg[m + 1:].max())
        is_valley = (seg[m] <= max(floor, VALLEY_FRAC
                                   * min(left_peak, right_peak))
                     and left_peak >= floor and right_peak >= floor)
        if is_valley or width > MAX_ISLAND_BINS:
            stack.append((b0, b0 + m + 1))        # valley bin rides left
            stack.append((b0 + m + 1, b1))
        else:
            final.append((b0, b1))
    final.sort()
    return idx, final, f_lo, floor


# --------------------------------------------------------------------------
# stage 2: within-island split (swappable: split(island_rows) -> labels)
# --------------------------------------------------------------------------
def make_cluster_features(x_all: np.ndarray,
                          basis: str = "sampling") -> np.ndarray:
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


def cluster_scale_floor(feats: np.ndarray, df_seg: float,
                        basis: str = "sampling") -> np.ndarray:
    """Per-feature whitening scale floors ("a zero MAD must not shatter").

    The sampling vector is the historical one. The observable vector CANNOT
    reuse it: entry 1 is ``Mc`` (order 0.5) in the sampling metric but
    ``fdot`` in the observable one, and ``fdot`` runs ~1e-17 at 1 mHz to
    ~1e-12 at 30 mHz. A fixed 1e-4 floor there sits up to 1e13x ABOVE the
    data, whitening every chirp difference to zero -- it would silently
    delete the separator this basis change exists to add, and no fixed
    absolute number works across five decades either. The floor is
    therefore taken RELATIVE to the island's own chirp scale.
    """
    if basis == "observable":
        fdot_scale = float(np.median(np.abs(feats[:, 1])))
        return np.array([0.05 * df_seg,
                         max(FDOT_SCALE_FLOOR_FRAC * fdot_scale, 1e-30),
                         1e-3, 1e-3, 1e-3])
    return np.array([0.05 * df_seg, 1e-4, 1e-3, 1e-3, 1e-3])


def _satellite_merge(labels, zw, stats):
    """v1 refinement: merge satellite fragments in whitened space.

    Any centroid pair closer than SAT_MERGE_CUT merges; a cluster smaller
    than SAT_FRAC of a bigger one merges into it within SAT_RADIUS."""
    for _ in range(5):
        ks = np.unique(labels[labels >= 0])
        if len(ks) < 2:
            break
        cents = np.array([zw[labels == k].mean(0) for k in ks])
        sizes = np.array([(labels == k).sum() for k in ks])
        d = cdist(cents, cents)
        parent = np.arange(len(ks))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        merged_any = False
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                close = d[i, j] < SAT_MERGE_CUT
                sat = (min(sizes[i], sizes[j])
                       < SAT_FRAC * max(sizes[i], sizes[j])
                       and d[i, j] < SAT_RADIUS)
                if close or sat:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[max(ri, rj)] = min(ri, rj)
                        merged_any = True
                        stats["satellite_merges"] += 1
        if not merged_any:
            break
        remap = {k: ks[find(i)] for i, k in enumerate(ks)}
        labels = np.array([remap[k] if k >= 0 else -1 for k in labels])
    # relabel compactly
    ks = np.unique(labels[labels >= 0])
    lut = {k: i for i, k in enumerate(ks)}
    return np.array([lut[k] if k >= 0 else -1 for k in labels])


def split_single_linkage(island_rows: np.ndarray, rng: np.random.Generator,
                         df_seg: float, stats: dict) -> np.ndarray:
    """Default swappable splitter: split(island_rows) -> labels (-1 = junk).

    island_rows: (n, 9) rows of ONE island, in ``stats["basis"]``.
    ``df_seg`` is 1/Tobs in the units of column 1 of THAT basis.
    """
    basis = stats.get("basis", "sampling")
    feats = make_cluster_features(island_rows, basis=basis)
    n = len(feats)
    sub = feats[rng.choice(n, min(n, SUB), replace=False)]
    med = np.median(sub, axis=0)
    mad = 1.4826 * np.median(np.abs(sub - med), axis=0)
    # column-aware scale floors (a zero MAD must not shatter the island)
    scale_floor = cluster_scale_floor(feats, df_seg, basis)
    scale = np.maximum(mad, scale_floor)
    zw_sub = (sub - med) / scale
    if len(zw_sub) > 1:
        lab_sub = fcluster(linkage(pdist(zw_sub), "single"), T_CUT,
                           "distance")
    else:
        lab_sub = np.ones(1, dtype=int)
    cents = np.array([zw_sub[lab_sub == k].mean(0)
                      for k in np.unique(lab_sub)])
    zw_all = (feats - med) / scale
    dmat = cdist(zw_all, cents)
    labels = dmat.argmin(1)
    labels[dmat.min(1) > JUNK_RADIUS] = -1
    return _satellite_merge(labels, zw_all, stats)


# --------------------------------------------------------------------------
# stage 3: cluster -> Gaussian component (circular params + eigval floors)
# --------------------------------------------------------------------------
def circular_wrap(x: np.ndarray, period: float) -> np.ndarray:
    """Wrap samples of a circular parameter around their circular mean."""
    ang = x * (2 * np.pi / period)
    m = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
    m *= period / (2 * np.pi)
    return (x - m + period / 2.0) % period + m - period / 2.0


def _trunc_mle_1d(x: np.ndarray, lo: float, hi: float):
    """Truncated-normal MLE on [lo, hi]; returns (mu, sigma, fitted).

    fitted=False = the fast path (sample >= 3.5 sigma interior on both
    edges, truncation negligible, moments returned)."""
    span = hi - lo
    m0, s0 = float(x.mean()), max(float(x.std()), 1e-4 * span)
    if (lo < m0 - 3.5 * s0) and (m0 + 3.5 * s0 < hi):
        return m0, s0, False

    def nll(th):
        mu, lsg = th
        sg = np.exp(lsg)
        if not (1e-5 * span < sg < 3.0 * span):
            return 1e12
        z = _norm.cdf((hi - mu) / sg) - _norm.cdf((lo - mu) / sg)
        if z < 1e-300:
            return 1e12
        return float(-(_norm.logpdf((x - mu) / sg).sum()
                       - len(x) * (np.log(sg) + np.log(z))))

    r = minimize(nll, [m0, np.log(s0)], method="Nelder-Mead",
                 options=dict(maxiter=200, xatol=1e-5, fatol=1e-6))
    mu = float(np.clip(r.x[0], lo - 2.0 * span, hi + 2.0 * span))
    sg = float(np.clip(np.exp(r.x[1]), 1e-5 * span, 3.0 * span))
    return mu, sg, True


def fit_component(rows: np.ndarray, stats: dict):
    """(m, 9) member rows -> (mean, cov) with circular phi0/psi/alpha,
    eigenvalue-floored covariance (v1 refinements), and truncated-normal
    MLE on the bounded columns (stats["bounded_cols"], 2026-09-11)."""
    x = rows.copy()
    for col, period in CIRCULAR_COLS.items():
        x[:, col] = circular_wrap(x[:, col], period)
    mean = x.mean(0)
    cov = np.atleast_2d(np.cov(x.T))
    # principal-range means for circular columns
    for col, period in CIRCULAR_COLS.items():
        mean[col] = mean[col] % period
    # eigenvalue floors: floor the diagonal, then the correlation spectrum
    # last-resort scales, well below any physical posterior width:
    # dist 1e-4 relative, f0 0.02/Tobs, Mc 1e-5 Msol, angles/unitless 1e-3
    diag_floor = np.array([1e-4 * max(abs(mean[0]), 1e-3), 0.0, 1e-5,
                           1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]) ** 2
    diag_floor[1] = (0.02 * stats["df_mhz"]) ** 2
    d2 = np.diag(cov).copy()
    n_diag = int((d2 < diag_floor).sum())
    d2 = np.maximum(d2, diag_floor)
    d = np.sqrt(d2)
    corr = cov / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    w, v = np.linalg.eigh(corr)
    n_eig = int((w < CORR_EIG_FLOOR).sum())
    if n_diag or n_eig:
        stats["cov_floor_triggers"] += 1
        stats["cov_floor_diag"] += n_diag
        stats["cov_floor_eig"] += n_eig
    w = np.maximum(w, CORR_EIG_FLOOR)
    corr = (v * w) @ v.T
    cov = corr * np.outer(d, d)

    # bounded columns: swap raw moments for truncated-normal MLE, keeping
    # the correlation structure (row/col rescale of a PD matrix by a
    # positive factor stays PD). Rail-piled members legitimately produce
    # means at/beyond the box edge.
    for col, (blo, bhi) in (stats.get("bounded_cols") or {}).items():
        if len(x) < 8:
            continue
        mu, sg, fitted = _trunc_mle_1d(x[:, col], blo, bhi)
        if fitted:
            stats["trunc_mle_fits"] = stats.get("trunc_mle_fits", 0) + 1
            r = sg / np.sqrt(cov[col, col])
            cov[col, :] *= r
            cov[:, col] *= r
            cov[col, col] = sg ** 2
            mean[col] = mu
    return mean, cov


# --------------------------------------------------------------------------
# stage 3 (observable): cluster -> GAUSSIAN MIXTURE
# --------------------------------------------------------------------------
def _unscale_gmm_components(comps):
    """Fitted unit-cube components -> PHYSICAL coordinates, in place of a copy.

    ``GMMFit`` linearly maps each group's samples into ``[-1, 1]^d`` using
    that group's per-feature ``(min, max)`` BEFORE fitting, so the lists
    ``vec_fit_gmm_min_bic`` returns carry CUBE means and covariances, with
    ``mins``/``maxs`` as the affine map back::

        x_phys = (z_cube + 1) / 2 * (max - min) + min

    :class:`~.proposal.WarmStartComponents` draws ``mean + L z`` and scores
    a Mahalanobis distance directly against the stored arrays, so it needs
    PHYSICAL ones; left as cube coordinates every draw would land at a
    completely wrong point (and f_mid ~3 mHz would read as ~0).

    ``mins``/``maxs`` are already physical -- they are the member bounding
    box -- and stay untouched, so they keep working as the box the density
    side culls on.

    NOTE for the deferred ``FullGaussianMixtureModel`` swap (spec 4.5):
    that class applies the affine ITSELF, so it must NOT be handed these
    unscaled arrays. ``meta["gmm"]["components_basis"] = "physical"``
    records which convention a file is in.
    """
    weights, means, covs, invcovs, dets, mins, maxs = comps
    out_means, out_covs, out_invcovs, out_dets = [], [], [], []
    for mu, cv, icv, dt, lo, hi in zip(means, covs, invcovs, dets,
                                       mins, maxs):
        lo = np.asarray(_host_array(lo), dtype=float)
        hi = np.asarray(_host_array(hi), dtype=float)
        s = (hi - lo) / 2.0                      # per-feature half-width
        mu = np.asarray(_host_array(mu), dtype=float)
        cv = np.asarray(_host_array(cv), dtype=float)
        out_means.append((mu + 1.0) / 2.0 * (hi - lo) + lo)
        # cov_phys = S cov_cube S with S = diag(s); invcov and det follow.
        out_covs.append(cv * np.outer(s, s)[None, :, :])
        icv = np.asarray(_host_array(icv), dtype=float)
        out_invcovs.append(icv / np.outer(s, s)[None, :, :])
        out_dets.append(np.asarray(_host_array(dt), dtype=float)
                        * float(np.prod(s)) ** 2)
    return [[np.asarray(_host_array(w), dtype=float) for w in weights],
            out_means, out_covs, out_invcovs, out_dets,
            [np.asarray(_host_array(v), dtype=float) for v in mins],
            [np.asarray(_host_array(v), dtype=float) for v in maxs]]


def _host_array(a):
    """cupy-or-numpy -> numpy (the fitter may run on device)."""
    return a.get() if hasattr(a, "get") else np.asarray(a)


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

    **The cap is the selector, not BIC** (measured 2026-09-18). The shared
    fitter scores BIC on the model's OWN synthetic draws
    (``gmm.bic(gmm.rvs(n))``), which is an entropy estimate rather than a
    fit-to-data criterion: more components always sit tighter, so BIC falls
    essentially monotonically in K and the "risen twice past the running
    minimum" retirement rule almost never fires. On a clean unimodal 9-D
    Gaussian it ran 30556 (K=1) down to 22255 (K=7), with AND without
    resampling. The sweep therefore returns the cap (give or take the noise
    in its own random draws), which makes ``min_members`` the knob that
    actually controls component count. Left as-is deliberately: the
    criterion belongs to ``gmm.py`` and is shared with the F-stat side,
    which is out of scope here (spec section 8).

    Reproducibility note: the underlying EM initialises with
    ``random_state=None``, so refitting the same store does not reproduce
    the same K or the same components bit for bit.

    Returns the seven ragged lists ``[weights, means, covs, invcovs, dets,
    mins, maxs]``, one entry per cluster, in PHYSICAL coordinates (see
    :func:`_unscale_gmm_components`), ready for
    :func:`lisatools.sampling.fstat_proposal.pack_gmm_components`.
    """
    from lisatools.sampling.gmm import vec_fit_gmm_min_bic

    rng = np.random.default_rng(seed)
    caps = [min(int(max_comp), max(1, len(r) // int(min_members)))
            for r in cluster_rows]
    ndim = int(np.asarray(cluster_rows[0]).shape[1])
    block = np.empty((len(cluster_rows), int(n_samples), ndim), dtype=float)
    for i, rows in enumerate(cluster_rows):
        rows = np.asarray(rows, dtype=float)
        idx = rng.integers(0, len(rows), size=int(n_samples))
        block[i] = rows[idx]
        # DEGENERATE-COLUMN GUARD: GMMFit divides by (max - min) per
        # feature, so a column that is constant across the cluster's
        # members produces inf/nan for the whole group. Jitter such a
        # column by a relative epsilon -- far below any posterior width,
        # and it only ever affects columns that carry no information.
        span = block[i].max(0) - block[i].min(0)
        flat = span <= 0.0
        if flat.any():
            ref = np.maximum(np.abs(block[i].mean(0)), 1.0)
            block[i][:, flat] += rng.normal(
                0.0, 1e-12 * ref[flat], size=(int(n_samples), int(flat.sum())))

    out = [[] for _ in range(7)]
    # Groups sharing a cap are fitted together; the fitter sweeps a single
    # (min_comp, max_comp) range per call, so one call per distinct cap.
    for cap in sorted(set(caps)):
        sel = [i for i, c in enumerate(caps) if c == cap]
        comps = _unscale_gmm_components(vec_fit_gmm_min_bic(
            block[sel], min_comp=1, max_comp=int(cap), gpu=gpu,
            verbose=verbose, return_components=True,
        ))
        for j, i in enumerate(sel):
            for k in range(7):
                out[k].append((i, comps[k][j]))
    # restore cluster order
    return [[v for _, v in sorted(lst, key=lambda t: t[0])] for lst in out]


# --------------------------------------------------------------------------
# stage 3.5: wide-blend re-split (ruling 2026-09-08)
# --------------------------------------------------------------------------
# Measured on the v8 deep fits: in dense islands the stage-2 MAD-whitened
# single linkage chains TWO real sources a few bins apart into ONE p~1
# cluster with mult > 2 (e.g. the SNR-153 source at 5.34778 mHz absorbed
# with its ~6-bin neighbor into a mult-2.8 barycenter blob) -- the island
# whitening scale is set by the whole confusion island, so the pair sits
# under T_CUT, and walker-wander bridge rows let single linkage chain
# straight through the gap. The fix re-splits such clusters with WARD
# linkage (bridge-robust, unlike a tighter single-linkage cut) in the
# cluster's OWN whitened frame, recursively, accepting a split only when
# every piece reduces mult (a genuinely inseparable cluster is kept and
# counted; the apply-stage blend flag catches it downstream). A genuine
# same-source double-stack also splits (two near-coincident components)
# -- the deliberate "split upstream" side of the open mult-policy ruling.
def _ward_split(rows, sid, rng, df_seg, basis="sampling"):
    """One ward 2-split in the cluster's own whitened frame.

    Returns [(rows, sid), (rows, sid)] or None when the split is
    degenerate (a tiny piece) or makes no mult progress."""
    feats = make_cluster_features(rows, basis=basis)
    med = np.median(feats, axis=0)
    mad = 1.4826 * np.median(np.abs(feats - med), axis=0)
    scale_floor = cluster_scale_floor(feats, df_seg, basis)
    zw = (feats - med) / np.maximum(mad, scale_floor)
    n = len(zw)
    if n > SUB:
        idx = rng.choice(n, SUB, replace=False)
        lab_sub = fcluster(linkage(zw[idx], "ward"), 2, "maxclust")
        cents = np.array([zw[idx][lab_sub == c].mean(0) for c in (1, 2)])
        lab = cdist(zw, cents).argmin(1) + 1
    else:
        lab = fcluster(linkage(zw, "ward"), 2, "maxclust")
    parts = [(rows[lab == c], sid[lab == c]) for c in (1, 2)]
    if any(len(pr) < 3 for pr, _ in parts):
        return None
    parent_mult = len(rows) / len(np.unique(sid))
    for pr, ps_ in parts:
        if len(pr) / len(np.unique(ps_)) >= parent_mult - 0.1:
            return None
    return parts


def resplit_blends(rows, sid, n_samples, mult_max, rng, df_seg, stats,
                   max_depth=4):
    """Recursively re-split clusters with p > 0.5 and mult > mult_max."""
    out, queue = [], [(rows, sid, 0)]
    while queue:
        r, s, depth = queue.pop()
        ids = np.unique(s)
        p = len(ids) / n_samples
        mult = len(r) / len(ids)
        if depth >= max_depth or p <= 0.5 or mult <= mult_max:
            out.append((r, s))
            continue
        parts = _ward_split(r, s, rng, df_seg,
                            stats.get("basis", "sampling"))
        if parts is None:
            stats["blend_unsplit"] += 1
            out.append((r, s))
            continue
        stats["blend_resplits"] += 1
        queue.extend((pr, ps_, depth + 1) for pr, ps_ in parts)
    return out


# --------------------------------------------------------------------------
def run(store: str, last_k: int | None, tobs: float, out: str,
        split_fn=split_single_linkage, seed: int = 7,
        max_iter: int | None = None, resplit_mult: float = 2.0,
        ratio_max: float = 5.0, basis: str = "sampling",
        transform_container=None):
    """Fit a finished run's cold-chain leaf table into birth components.

    ``basis`` selects the coordinates the WHOLE fit runs in:

    * ``"sampling"`` (default here) -- the historical astro basis, one
      Gaussian per cluster, ``means``/``covs`` in the npz. Kept as the
      library default so every existing caller and stored file is
      bit-identical.
    * ``"observable"`` -- the leaf table is mapped through
      ``GBObservableFiberBasis.to_internal`` at intake and every stage
      after works there; clusters are fitted as Gaussian MIXTURES and
      written in the packed ``gmm_*`` layout. This is the CLI default
      (``--basis``), so a rerun of the pipeline produces an observable set.

    ``transform_container`` supplies the sampling basis the map indexes
    against; ``None`` uses this module's own :data:`COLUMN_NAMES`.
    """
    if basis not in ("observable", "sampling"):
        raise ValueError(
            f"basis must be 'observable' or 'sampling', got {basis!r}")
    rng = np.random.default_rng(seed)
    df_mhz = 1.0 / tobs * 1e3          # 1/Tobs in mHz (stored f0 is mHz)
    # the SAME 1/Tobs, in the units column 1 carries in the working basis
    # (mHz for sampling f0, Hz for observable f_mid) -- see basis_df.
    df_seg = basis_df(tobs, basis)
    walls = {}

    t0 = time.perf_counter()
    X, sample_id, info = load_leaf_table(store, last_k, max_iter=max_iter)
    walls["load"] = time.perf_counter() - t0
    n_samples = info["n_samples"]
    print(f"leaf table: {len(X):,} rows, {n_samples:,} posterior samples "
          f"(its {info['iterations'][0]}..{info['iterations'][1]}, "
          f"{info['nwalkers']} walkers, "
          f"{info['leaves_per_walker']:.1f} leaves/walker) "
          f"[{walls['load']:.1f} s, RSS {rss_gb():.2f} GB]")

    # --- INTAKE SEAM: after this the pipeline is in ONE basis ------------
    obs_map = None
    if basis == "observable":
        obs_map = build_observable_map(tobs, transform_container)
        X = to_observable(X, obs_map)
        print(f"intake: mapped {len(X):,} rows to the OBSERVABLE basis "
              f"(Tobs {tobs:.6g} s, shear {obs_map.shear}, fiber "
              f"{obs_map.fiber_coord}); col 1 is f_mid [Hz], col 2 fdot.")

    t0 = time.perf_counter()
    bin_idx, islands, f_lo, floor = segment_f0(X[:, 1], df_seg, n_samples)
    walls["segment"] = time.perf_counter() - t0
    print(f"stage 1: {len(islands)} islands (floor {floor}/bin, "
          f"df {df_seg:.6g} {'Hz' if basis == 'observable' else 'mHz'}) "
          f"[{walls['segment']:.2f} s]")

    stats = dict(satellite_merges=0, cov_floor_triggers=0,
                 cov_floor_diag=0, cov_floor_eig=0, df_mhz=df_mhz,
                 df_seg=df_seg, basis=basis,
                 junk_rows=0, orphan_rows=0, dropped_fragments=0,
                 dropped_fragment_rows=0, blend_resplits=0,
                 blend_unsplit=0, trunc_mle_fits=0,
                 bounded_cols={COS_IOTA_COL: (-1.0, 1.0),
                               RATIO_COL: (-float(ratio_max),
                                           float(ratio_max))})
    in_island = np.zeros(len(X), dtype=bool)

    means, covs, ps, mults, ns, isl_id = [], [], [], [], [], []
    t0 = time.perf_counter()
    t_split_total = 0.0
    for isl, (b0, b1) in enumerate(islands):
        m = (bin_idx >= b0) & (bin_idx < b1)
        in_island |= m
        x_all, sid = X[m], sample_id[m]
        ts = time.perf_counter()
        labels = split_fn(x_all, rng, df_seg, stats)
        t_split_total += time.perf_counter() - ts
        stats["junk_rows"] += int((labels == -1).sum())
        for k in range(labels.max() + 1 if labels.size else 0):
            mk = labels == k
            nk = int(mk.sum())
            if nk < max(3, MIN_FRAC * n_samples):
                stats["dropped_fragments"] += 1
                stats["dropped_fragment_rows"] += nk
                continue
            pieces = [(x_all[mk], sid[mk])]
            if resplit_mult > 0:
                pieces = resplit_blends(x_all[mk], sid[mk], n_samples,
                                        resplit_mult, rng, df_seg, stats)
            for xr, sr in pieces:
                nr = len(xr)
                if nr < max(3, MIN_FRAC * n_samples):
                    stats["dropped_fragments"] += 1
                    stats["dropped_fragment_rows"] += nr
                    continue
                ids = np.unique(sr)
                mean, cov = fit_component(xr, stats)
                means.append(mean)
                covs.append(cov)
                ps.append(len(ids) / n_samples)
                mults.append(nr / len(ids))
                ns.append(nr)
                isl_id.append(isl)
    walls["split"] = t_split_total
    walls["components"] = time.perf_counter() - t0 - t_split_total
    stats["orphan_rows"] = int((~in_island).sum())

    means = np.array(means)
    covs = np.array(covs)
    ps = np.array(ps)
    mults = np.array(mults)
    ns = np.array(ns, dtype=np.int64)
    isl_id = np.array(isl_id, dtype=np.int64)
    f0_window_edges = np.array(
        [[f_lo + b0 * df_seg, f_lo + b1 * df_seg] for b0, b1 in islands])

    order = np.argsort(means[:, 1])
    means, covs, ps, mults, ns, isl_id = (
        means[order], covs[order], ps[order], mults[order], ns[order],
        isl_id[order])

    try:
        # provenance: the LAT repo this module runs from (editable src
        # layout: <repo>/src/lisatools/globalfit/warmstart/), not a
        # hardcoded checkout path.
        from pathlib import Path

        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parents[4]),
        ).stdout.strip()
    except Exception:
        git_head = "unknown"

    meta = dict(
        store=store, tobs=tobs, df_mhz=df_mhz, last_k=last_k,
        column_names=COLUMN_NAMES, f0_units="mHz",
        circular_cols={str(k): v for k, v in CIRCULAR_COLS.items()},
        sample_id_def="stored_iteration_index * nwalkers + walker",
        git_head=git_head, seed=seed,
        pipeline="density-valley + single-linkage + satellite-merge v1",
        knobs=dict(SUB=SUB, T_CUT=T_CUT, JUNK_RADIUS=JUNK_RADIUS,
                   SAT_MERGE_CUT=SAT_MERGE_CUT, SAT_FRAC=SAT_FRAC,
                   SAT_RADIUS=SAT_RADIUS, MIN_FRAC=MIN_FRAC,
                   CORR_EIG_FLOOR=CORR_EIG_FLOOR,
                   resplit_mult=resplit_mult),
        **info, **{k: v for k, v in stats.items() if k != "df_mhz"},
        walls={k: round(v, 3) for k, v in walls.items()},
    )
    t0 = time.perf_counter()
    np.savez_compressed(
        out, means=means, covs=covs, p=ps, mult=mults, n_members=ns,
        island_id=isl_id, f0_window_edges=f0_window_edges,
        meta=json.dumps(meta))
    walls["write"] = time.perf_counter() - t0

    total_rows = len(X)
    print(f"stage 2: split {walls['split']:.1f} s | stage 3: components "
          f"{walls['components']:.1f} s | write {walls['write']:.2f} s")
    print(f"components: {len(means)} | junk rows "
          f"{stats['junk_rows']:,} ({stats['junk_rows']/total_rows:.2%}) | "
          f"orphan rows (outside islands) {stats['orphan_rows']:,} "
          f"({stats['orphan_rows']/total_rows:.2%}) | dropped fragments "
          f"{stats['dropped_fragments']} ({stats['dropped_fragment_rows']:,}"
          f" rows) | satellite merges {stats['satellite_merges']} | "
          f"blend re-splits {stats['blend_resplits']} "
          f"({stats['blend_unsplit']} inseparable kept) | "
          f"trunc-MLE fits {stats['trunc_mle_fits']} | "
          f"cov floors {stats['cov_floor_triggers']} comps "
          f"(diag {stats['cov_floor_diag']}, eig {stats['cov_floor_eig']})")
    print(f"p: sum {ps.sum():.1f} | >0.9: {(ps > 0.9).sum()} | 0.5-0.9: "
          f"{((ps >= 0.5) & (ps <= 0.9)).sum()} | <0.5: {(ps < 0.5).sum()}")
    print(f"peak RSS {rss_gb():.2f} GB | wrote {out}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--store", required=True, help="previous-run h5 store")
    ap.add_argument("--last-k", type=int, default=None,
                    help="last K stored iterations (default: all written; "
                         "torn trailing iterations auto-dropped)")
    ap.add_argument("--max-iter", type=int, default=None,
                    help="ignore stored iterations beyond this index "
                         "(default: none — the torn-tail guard handles "
                         "crash-truncated stores)")
    ap.add_argument("--tobs", type=float, default=7776000.0,
                    help="observation time [s] (default 3 mo)")
    ap.add_argument("--out", default="warmstart_components.npz")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--resplit-mult", type=float, default=2.0,
                    help="stage-3.5 wide-blend re-split trigger: clusters "
                         "with p > 0.5 and mult above this are ward-"
                         "resplit recursively (default 2.0; <= 0 "
                         "disables, restoring the pre-2026-09-08 fit)")
    ap.add_argument("--ratio-max", type=float, default=5.0,
                    help="fdot_astro_ratio prior half-width (the col-8 "
                         "truncation box; GBSettings.fdot_astro_ratio_max)")
    ap.add_argument("--basis", default="observable",
                    choices=("observable", "sampling"),
                    help="coordinates the WHOLE fit runs in (default "
                         "observable: lnA/f_mid/fdot/.../Mc, per-cluster "
                         "Gaussian MIXTURE in the packed gmm_* layout). "
                         "'sampling' reproduces the pre-2026-09-18 astro-"
                         "basis fit with one Gaussian per cluster.")
    args = ap.parse_args(argv)
    run(args.store, args.last_k, args.tobs, args.out, seed=args.seed,
        max_iter=args.max_iter, resplit_mult=args.resplit_mult,
        ratio_max=args.ratio_max, basis=args.basis)


if __name__ == "__main__":
    main()
