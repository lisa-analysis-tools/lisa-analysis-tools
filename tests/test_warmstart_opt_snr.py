"""Tests for lisatools.globalfit.warmstart.opt_snr — the optimal-SNR gate on
warm-start clusters (ruling 2026-09-10: limit 8, sensitivity = the
best-logL walker's psd+galfor at the last stored sample).

Hermetic: the gate/annotate logic is tested with an injected per-component
SNR array (the real waveform+sensitivity route reuses
scripts/diagnostics/build_truth.py verbatim and is validated empirically —
matched components must reproduce the truth-set SNRs)."""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lisatools.sampling.warmstart_proposal import (
    CIRCULAR_COLS, COLUMN_NAMES, WarmStartComponents,
)

TOBS = 7776000.0
def _load_gate():
    # installed code (2026-09-14 move): no path loading
    from lisatools.globalfit.warmstart import opt_snr

    return opt_snr


def _make_cov(sigmas, corr_off=0.15):
    d = len(sigmas)
    C = (1.0 - corr_off) * np.eye(d) + corr_off * np.ones((d, d))
    D = np.diag(np.asarray(sigmas, dtype=float))
    return D @ C @ D


def _fit_npz(path, n=4):
    means = np.array([
        [8.0,  2.0000, 0.60, 3.00,  0.30, 1.20, 4.00,  0.20,  0.05],
        [12.0, 2.0008, 0.50, 2.00, -0.40, 0.80, 1.00, -0.50, -0.10],
        [5.0,  5.0000, 0.70, 0.05,  0.00, 0.50, 5.50,  0.60,  0.20],
        [20.0, 8.0000, 0.40, 5.00,  0.70, 2.50, 0.30, -0.80,  0.00],
    ])[:n]
    sig = [0.5, 2.0e-5, 0.01, 0.30, 0.05, 0.15, 0.05, 0.05, 0.02]
    covs = np.stack([_make_cov(sig) for _ in range(n)])
    p = np.array([0.9, 0.3, 0.95, 0.6])[:n]
    meta = dict(
        store="synthetic", tobs=TOBS, df_mhz=1e3 / TOBS, last_k=None,
        column_names=COLUMN_NAMES, f0_units="mHz",
        circular_cols={str(k): v for k, v in CIRCULAR_COLS.items()},
        sample_id_def="stored_iteration_index * nwalkers + walker",
        git_head="test", seed=7,
        pipeline="density-valley + single-linkage + satellite-merge v1",
    )
    np.savez_compressed(
        path, means=means, covs=covs, p=p, mult=np.ones(n),
        n_members=np.full(n, 100, dtype=np.int64),
        island_id=np.array([0, 0, 1, 2][:n], dtype=np.int64),
        f0_window_edges=np.array([[1.99, 2.01], [4.99, 5.01], [7.99, 8.01]]),
        meta=json.dumps(meta))
    return means, p


class OptSnrGateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fit = os.path.join(self.tmp.name, "fit.npz")
        self.means, self.p = _fit_npz(self.fit)
        self.mod = _load_gate()

    def tearDown(self):
        self.tmp.cleanup()

    def test_gate_drops_below_limit_and_annotates(self):
        out = os.path.join(self.tmp.name, "gated.npz")
        snr = np.array([25.0, 3.0, 8.0, 7.99])
        noise = dict(psd_params=[1.5e-11, 3.0e-15],
                     galfor_params=[1, 2, 3, 4, 5],
                     iteration=349, walker=17, log_like=1.234e7)
        self.mod.gate(self.fit, out, 8.0, snr, noise)
        z = np.load(out, allow_pickle=False)
        # comps 1 (snr 3) and 3 (snr 7.99) dropped; >= limit kept
        np.testing.assert_allclose(z["opt_snr"], [25.0, 8.0])
        np.testing.assert_allclose(z["means"][:, 1], [2.0000, 5.0000])
        np.testing.assert_allclose(z["p"], [0.9, 0.95])
        self.assertEqual(len(z["mult"]), 2)
        self.assertEqual(len(z["island_id"]), 2)
        meta = json.loads(str(z["meta"]))
        self.assertEqual(meta["opt_snr_limit"], 8.0)
        self.assertEqual(meta["opt_snr_dropped"], 2)
        self.assertEqual(meta["opt_snr_noise"]["walker"], 17)
        # schema still loads as a proposal
        ws = WarmStartComponents.from_npz(out, new_tobs=2 * TOBS)
        self.assertEqual(ws.n_components, 2)

    def test_boxed_means_clips_bounded_cols(self):
        # truncated-MLE fits legitimately put a rail-piled component's
        # mean AT/BEYOND the box edge; the waveform transform needs
        # physical values (arccos of cos_iota etc.), so the gate clips
        # to the meta bounds before both_transforms.
        means = np.array(self.means, copy=True)
        means[0, 4] = 1.08
        means[1, 8] = -5.7
        meta = {"bounded_cols": {"4": [-1.0, 1.0], "8": [-5.0, 5.0]}}
        out = self.mod.boxed_means(means, meta)
        self.assertAlmostEqual(out[0, 4], 1.0)
        self.assertAlmostEqual(out[1, 8], -5.0)
        # untouched elsewhere; no bounded_cols meta = identity
        self.assertAlmostEqual(out[2, 4], means[2, 4])
        np.testing.assert_array_equal(self.mod.boxed_means(means, {}),
                                      means)

    def test_gate_zero_limit_keeps_all(self):
        out = os.path.join(self.tmp.name, "gated0.npz")
        self.mod.gate(self.fit, out, 0.0, np.array([25.0, 3.0, 8.0, 1.0]),
                      dict(psd_params=[1, 1], galfor_params=[0] * 5,
                           iteration=0, walker=0, log_like=0.0))
        z = np.load(out, allow_pickle=False)
        self.assertEqual(len(z["p"]), 4)
        self.assertEqual(len(z["opt_snr"]), 4)


if __name__ == "__main__":
    unittest.main()
