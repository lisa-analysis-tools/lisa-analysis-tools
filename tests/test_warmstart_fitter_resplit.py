"""Tests for the warm-start fitter's blend RE-SPLIT pass (stage 3.5).

The 2026-09-08 gap (measured on the v8 10-walker + 24-walker deep fits):
in dense islands the stage-2 MAD-whitened single linkage chains TWO real
sources a few bins apart into ONE p~1 component with mult > 2 (e.g. the
SNR-153 source at 5.34778 mHz absorbed with its ~6-bin neighbor into a
mult-2.8 barycenter blob). The island-level whitening is the mechanism:
the pair's separation is ~1.4 whitened units < T_CUT 2.0. Re-whitening
the CLUSTER's own rows and cutting tighter separates tight clouds bins
apart. Ruling 2026-09-08: re-split such clusters in the fitter
(``resplit_mult``, default 2.0; <= 0 disables).

Hermetic: a tiny gf_format_version-2 store h5 is synthesized with two
tight sources 3.1 bins apart (valley filled at island whitening -> one
stage-2 cluster, mult 2.2) plus a clean control source in its own island.
"""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

TOBS = 7776000.0
DF = 1e3 / TOBS          # 1/Tobs in mHz
NDIM = 9

def _load_fitter():
    # installed code (2026-09-14 move): no path loading
    from lisatools.globalfit.warmstart import fit_from_store

    return fit_from_store


def _row(rng, f0_mhz, sigma_f0_bins):
    """One sampled-basis leaf row near f0_mhz.

    Non-f0 scatter is kept BELOW the stage-2 MAD scale floors so the
    5-D whitened geometry is f0-dominated: the balanced pair then chains
    at the island whitening (separation ~1.3 whitened units < T_CUT 2.0)
    while staying separable under a tighter local cut -- the measured
    production wide-blend geometry."""
    return np.array([
        8.0 + 0.002 * rng.standard_normal(),                   # dist
        f0_mhz + sigma_f0_bins * DF * rng.standard_normal(),   # f0 [mHz]
        0.60 + 2e-5 * rng.standard_normal(),                   # Mc
        3.00 + 2e-4 * rng.standard_normal(),                   # phi0
        0.30 + 2e-4 * rng.standard_normal(),                   # cos_iota
        1.20 + 2e-4 * rng.standard_normal(),                   # psi
        4.00 + 2e-4 * rng.standard_normal(),                   # alpha
        0.20 + 2e-4 * rng.standard_normal(),                   # sin_delta
        0.05 + 2e-4 * rng.standard_normal(),                   # ratio
    ])


F0_A = 2.0000            # mHz
F0_B = F0_A + 3.1 * DF   # 3.1 bins away -- the wide-blend regime
F0_CTRL = 5.0000
N_ITS, N_WALKERS, N_LEAVES = 3, 10, 8


def _write_store(path, seed=2):
    """30 samples; every sample holds one A + one B core leaf (0.5-bin
    clouds) and the first 16 samples add one BRIDGE leaf with f0 uniform
    in the gap -- the walker-wander rows that fill the stage-1 valley and
    let stage-2 single linkage chain the pair into ONE mult-2.53 cluster
    (the measured production wide-blend formation; calibrated against the
    pre-resplit fitter 2026-09-08). Plus one clean control leaf per
    sample at 5 mHz."""
    rng = np.random.default_rng(seed)
    rows = N_ITS + 1
    chain = np.zeros((rows, 1, 1, N_WALKERS, N_LEAVES, NDIM))
    inds = np.zeros((rows, 1, 1, N_WALKERS, N_LEAVES), dtype=bool)
    ll = np.zeros((rows, 1, 1, N_WALKERS))
    for it in range(N_ITS):
        for w in range(N_WALKERS):
            s = it * N_WALKERS + w
            leaves = [_row(rng, F0_A, 0.5), _row(rng, F0_B, 0.5),
                      _row(rng, F0_CTRL, 0.5)]
            if s < 16:
                f_bridge = F0_A + rng.uniform(0.7, 2.4) * DF
                leaves.append(_row(rng, f_bridge, 0.05))
            for li, leaf in enumerate(leaves):
                chain[it, 0, 0, w, li] = leaf
                inds[it, 0, 0, w, li] = True
            ll[it, 0, 0, w] = 1.0
    with h5py.File(path, "w") as f:
        g = f.create_group("global_fit")
        g.attrs["iteration"] = N_ITS
        g.create_group("chain").create_dataset("gb", data=chain)
        g.create_group("inds").create_dataset("gb", data=inds)
        g.create_dataset("log_like", data=ll)


def _fit(mod, store, out, **kwargs):
    mod.run(store, None, TOBS, out, seed=7, **kwargs)
    z = np.load(out, allow_pickle=False)
    return {k: np.array(z[k]) for k in
            ("means", "covs", "p", "mult", "island_id")}, \
        json.loads(str(z["meta"]))


def _window(fit, lo, hi):
    m = (fit["means"][:, 1] >= lo) & (fit["means"][:, 1] <= hi)
    return {k: (v[m] if k != "covs" else v[m]) for k, v in fit.items()}


class FitterResplitTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = os.path.join(self.tmp.name, "store.h5")
        _write_store(self.store)
        self.mod = _load_fitter()

    def tearDown(self):
        self.tmp.cleanup()

    def _blend_window(self, fit):
        return _window(fit, F0_A - 2 * DF, F0_B + 2 * DF)

    def test_disabled_reproduces_the_blend(self):
        # resplit off: the pair collapses to ONE p~1 mult>1.8 component --
        # the measured production failure this data reproduces.
        fit, meta = _fit(self.mod, self.store,
                         os.path.join(self.tmp.name, "off.npz"),
                         resplit_mult=0.0)
        w = self._blend_window(fit)
        core = w["p"] > 0.8
        self.assertEqual(int(core.sum()), 1,
                         f"expected ONE blended comp, got {w['p']}")
        self.assertGreater(float(w["mult"][core][0]), 1.8)

    def test_default_resplits_the_blend_pair(self):
        fit, meta = _fit(self.mod, self.store,
                         os.path.join(self.tmp.name, "on.npz"))
        w = self._blend_window(fit)
        core = np.flatnonzero(w["p"] > 0.8)
        self.assertEqual(len(core), 2,
                         f"expected TWO resplit comps, got p={w['p']} "
                         f"mult={w['mult']}")
        f0s = np.sort(w["means"][core, 1])
        self.assertLess(abs(f0s[0] - F0_A) / DF, 1.0)
        self.assertLess(abs(f0s[1] - F0_B) / DF, 1.0)
        for m in w["mult"][core]:
            self.assertLess(float(m), 1.5)
        self.assertGreaterEqual(int(meta.get("blend_resplits", 0)), 1)

    def test_resplit_conserves_leaf_mass(self):
        on, _ = _fit(self.mod, self.store,
                     os.path.join(self.tmp.name, "on2.npz"))
        off, _ = _fit(self.mod, self.store,
                      os.path.join(self.tmp.name, "off2.npz"),
                      resplit_mult=0.0)
        for fit in (on, off):
            w = self._blend_window(fit)
            self.assertAlmostEqual(float(np.sum(w["p"] * w["mult"])),
                                   76.0 / 30.0, delta=0.2)

    def test_control_component_untouched(self):
        on, _ = _fit(self.mod, self.store,
                     os.path.join(self.tmp.name, "on3.npz"))
        w = _window(on, F0_CTRL - 2 * DF, F0_CTRL + 2 * DF)
        self.assertEqual(int((w["p"] > 0.8).sum()), 1)
        self.assertAlmostEqual(float(w["mult"][w["p"] > 0.8][0]), 1.0,
                               delta=0.05)


class FitterBoundedColsTest(unittest.TestCase):
    """Stage-3 truncated-normal fit of the BOUNDED columns (ruling
    2026-09-11): cos_iota (col 4, box [-1,1]) and fdot_astro_ratio
    (col 8, box +-ratio_max) are fitted by 1-D truncated-MLE instead of
    raw sample moments, and the bounds ride in the npz meta so
    WarmStartComponents truncates the mixture."""

    def _railed_store(self, path, seed=4):
        """One source whose cos_iota members PILE at the +1 boundary."""
        rng = np.random.default_rng(seed)
        rows = N_ITS + 1
        chain = np.zeros((rows, 1, 1, N_WALKERS, N_LEAVES, NDIM))
        inds = np.zeros((rows, 1, 1, N_WALKERS, N_LEAVES), dtype=bool)
        ll = np.zeros((rows, 1, 1, N_WALKERS))
        for it in range(N_ITS):
            for w in range(N_WALKERS):
                leaf = _row(rng, F0_CTRL, 0.4)
                leaf[4] = 1.0 - abs(0.12 * rng.standard_normal())
                for li in range(3):          # 3 leaves/sample, mult 3?
                    pass
                chain[it, 0, 0, w, 0] = leaf
                inds[it, 0, 0, w, 0] = True
                ll[it, 0, 0, w] = 1.0
        with h5py.File(path, "w") as f:
            g = f.create_group("global_fit")
            g.attrs["iteration"] = N_ITS
            g.create_group("chain").create_dataset("gb", data=chain)
            g.create_group("inds").create_dataset("gb", data=inds)
            g.create_dataset("log_like", data=ll)

    def test_bounded_meta_and_railed_mle(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from lisatools.sampling.warmstart_proposal import WarmStartComponents

        with tempfile.TemporaryDirectory() as td:
            store = os.path.join(td, "s.h5")
            self._railed_store(store)
            mod = _load_fitter()
            out = os.path.join(td, "f.npz")
            mod.run(store, None, TOBS, out, seed=7)
            z = np.load(out, allow_pickle=False)
            meta = json.loads(str(z["meta"]))
            self.assertEqual(meta["bounded_cols"]["4"], [-1.0, 1.0])
            self.assertEqual(meta["bounded_cols"]["8"], [-5.0, 5.0])
            means = np.array(z["means"])
            k = int(np.argmin(np.abs(means[:, 1] - F0_CTRL)))
            # rail pile at +1: the truncated-MLE center sits AT/BEYOND the
            # rail, far above the raw sample mean (~0.90)
            self.assertGreater(means[k, 4], 0.95)
            # the npz loads as a TRUNCATED proposal: draws in-box
            ws = WarmStartComponents.from_npz(out, new_tobs=2 * TOBS,
                                              seed=3)
            self.assertEqual(ws.bounded_cols[4], (-1.0, 1.0))
            dr = np.asarray(ws.rvs(size=20_000))
            self.assertTrue((np.abs(dr[:, 4]) <= 1.0).all())
            self.assertTrue((np.abs(dr[:, 8]) <= 5.0).all())


if __name__ == "__main__":
    unittest.main()
