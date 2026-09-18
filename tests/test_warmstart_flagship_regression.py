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


def _fdot_gr(f0_mhz, mc):
    return ((96 / 5) * np.pi ** (8 / 3) * (MSUN * mc) ** (5 / 3)
            * (f0_mhz * 1e-3) ** (11 / 3))


def _flagship_members(n=97, seed=17, fdot=FDOT_TRUTH, fdot_frac=0.01,
                      mc_sigma=0.0706):
    """97 posterior samples matching the measured component-5571 spreads.

    The members are generated ON THE CHIRP RIDGE, which is what the
    measurement says they are: the 3-month run determined this source's
    fdot to one percent, while Mc is the FIBER -- the flat direction the
    data does not constrain -- so ``ratio = fdot / fdot_gr(f0, Mc) - 1`` is
    whatever the sampled Mc makes it.

    Generating Mc and ratio as INDEPENDENT marginals instead (their
    measured sigmas are 0.0706 and 0.271) would scatter the members ACROSS
    the ridge and no fit of any kind could put draws back on it. That the
    Mc spread of 15% induces a ratio spread of 15% * 5/3 = 25%, against the
    0.271 actually measured, is the check that this is the right geometry:
    the measured ratio width IS the flat Mc direction seen edge-on.
    """
    rng = np.random.default_rng(seed)
    x = np.empty((n, 9))
    x[:, 0] = rng.normal(9.693, 2.397, n).clip(0.1)
    x[:, 1] = rng.normal(20.380376, 4.3e-5, n)
    x[:, 2] = rng.normal(0.4678, mc_sigma, n).clip(0.05)
    x[:, 3] = rng.normal(6.174, 1.840, n) % (2 * np.pi)
    x[:, 4] = rng.normal(-0.95, 0.05, n).clip(-1, 1)
    x[:, 5] = rng.normal(1.395, 0.880, n) % np.pi
    x[:, 6] = rng.normal(4.085, 0.0264, n) % (2 * np.pi)
    x[:, 7] = rng.normal(-0.7795, 0.0139, n).clip(-1, 1)
    fd = rng.normal(fdot, fdot_frac * fdot, n)
    x[:, 8] = fd / _fdot_gr(x[:, 1], x[:, 2]) - 1.0
    return x


def _fit_and_draw(members, n_draw=4000, **fit_kw):
    """Members -> observable GMM -> npz -> WarmStartComponents -> draws."""
    from lisatools.sampling.fstat_proposal import pack_gmm_components

    m = wb.build_map(_Container(), Tobs=7.776e6)
    z = ffs.to_observable(members, m)
    kw = dict(n_samples=4096, max_comp=12, min_members=25, seed=5)
    kw.update(fit_kw)
    comps = ffs.fit_cluster_gmms([z], **kw)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "flagship.npz")
        np.savez(path, p=np.array([0.97]), mult=np.array([1.0]),
                 n_members=np.array([len(members)]),
                 island_id=np.array([0]),
                 f0_window_edges=np.array([[2.037e-2, 2.039e-2]]),
                 meta=json.dumps({
                     "tobs": 7.776e6, "basis": "observable", "f0_units": "Hz",
                     "column_names": wb.OBSERVABLE_COLUMN_NAMES,
                     "map_params": wb.map_params_from_map(m)}),
                 **pack_gmm_components(comps))
        c = WarmStartComponents.from_npz(path, new_tobs=1.5552e7)
        c.attach_transform(_Container())
        return comps, c.rvs(n_draw)


def _ridge_stats(draws):
    fdot = np.array([_phys_fdot(r) for r in draws])
    ratio = fdot / FDOT_TRUTH
    return (float(np.mean((ratio > 1 / 1.3) & (ratio < 1.3))),
            float(np.mean(fdot < 0)))


class FlagshipRegressionTest(unittest.TestCase):
    def test_draws_land_on_the_chirp_ridge(self):
        members = _flagship_members()
        comps, draws = _fit_and_draw(members)

        # 97 members // 25 = 3, so the min_members guard caps K at 3.
        # (It is the cap, not BIC, that decides -- see
        # fit_cluster_gmms; the existing fitter's BIC is monotone in K.)
        self.assertLessEqual(len(comps[0][0]), 3)

        # f0 lands on the source
        self.assertLess(abs(np.median(draws[:, 1]) - F0_TRUTH_MHZ), 5e-5)

        # and so does the CHIRP, which is the whole point
        frac_on_ridge, frac_neg = _ridge_stats(draws)
        print(f"\n[FLAGSHIP] K={len(comps[0][0])} "
              f"on-ridge(1.3x)={frac_on_ridge:.1%} negative={frac_neg:.2%}")
        self.assertGreater(
            frac_on_ridge, 0.60,
            f"only {frac_on_ridge:.0%} of draws within 1.3x of the true "
            f"fdot; the observable component should keep them on the ridge")
        self.assertLess(frac_neg, 0.02,
                        "negative chirps are unphysical here and were 42% of "
                        "the fragment leaves this design replaces")

    def _control_draws(self, members, n=4000, seed=3):
        """The OLD fit: one Gaussian over the sampling columns."""
        stats = dict(cov_floor_triggers=0, cov_floor_diag=0,
                     cov_floor_eig=0, df_mhz=1e3 / 7.776e6,
                     trunc_mle_fits=0,
                     bounded_cols={ffs.COS_IOTA_COL: (-1.0, 1.0),
                                   ffs.RATIO_COL: (-5.0, 5.0)})
        mean, cov = ffs.fit_component(members, stats)
        rng = np.random.default_rng(seed)
        return mean + rng.standard_normal((n, 9)) @ np.linalg.cholesky(cov).T

    def test_flagship_is_the_BEST_case_for_the_old_basis(self):
        """Honest scope limit: the flagship alone does NOT prove the design.

        This component's chirp mass is unusually well constrained (ratio
        sigma 0.271 against a population median of 1.53), so over its
        narrow Mc range the curved chirp surface is nearly linear and a
        single sampling-basis Gaussian tracks it almost as well. Measured
        2026-09-18: observable 100% on-ridge, control 98.6%.

        Pinned so nobody later cites the flagship gate as evidence on its
        own -- the discriminating case is the typical component below.
        """
        members = _flagship_members()
        frac, _ = _ridge_stats(self._control_draws(members))
        print(f"[CONTROL ] flagship (ratio sigma "
              f"{np.std(members[:, 8]):.2f}): on-ridge={frac:.1%}")
        self.assertGreater(frac, 0.9)

    def test_typical_component_is_where_the_old_basis_fails(self):
        """PAIRED CONTROL at the measured population's fiber width.

        Across the shipped 5573 components the ratio sigma has a median of
        1.53 and a p90 of 30 -- the chirp mass is nearly unconstrained, so
        the Gaussian has to span a wide, genuinely CURVED arc of the chirp
        surface. A 32% Mc spread lands this fixture at ratio sigma 3.2,
        between that median and that p90: an ordinary component, not a
        contrived one.

        Measured 2026-09-18: observable 99.8% on-ridge with 0.0% negative
        chirps; sampling-basis single Gaussian 0.4% on-ridge with 18.9%
        NEGATIVE fdot, which is unphysical for an inspiralling binary.
        """
        members = _flagship_members(mc_sigma=0.15)
        _comps, draws = _fit_and_draw(members)
        obs_ridge, obs_neg = _ridge_stats(draws)
        ctrl_ridge, ctrl_neg = _ridge_stats(self._control_draws(members))
        print(f"[TYPICAL ] ratio sigma {np.std(members[:, 8]):.2f} | "
              f"OBS on-ridge={obs_ridge:.1%} neg={obs_neg:.1%} | "
              f"CTRL on-ridge={ctrl_ridge:.1%} neg={ctrl_neg:.1%}")
        self.assertGreater(obs_ridge, 0.90)
        self.assertLess(obs_neg, 0.02)
        self.assertLess(ctrl_ridge, 0.60)
        self.assertGreater(ctrl_neg, 0.02,
                           "the sampling-basis Gaussian is supposed to put "
                           "mass at NEGATIVE chirp here; if it does not, "
                           "this control is not exercising the defect")

    def test_a_blended_pair_is_described_by_more_than_one_component(self):
        """Two sources in one cluster: the mixture must not average them."""
        a = _flagship_members(n=97, seed=17)
        b = _flagship_members(n=97, seed=23, fdot=1.8 * FDOT_TRUTH)
        comps, draws = _fit_and_draw(np.vstack([a, b]), min_members=25)
        self.assertGreater(len(comps[0][0]), 1)
        # draws cover BOTH chirps rather than piling between them
        fdot = np.array([_phys_fdot(r) for r in draws])
        lo = np.median([_phys_fdot(r) for r in a])
        hi = np.median([_phys_fdot(r) for r in b])
        self.assertGreater(np.mean(np.abs(fdot - lo) < 0.25 * (hi - lo)), 0.1)
        self.assertGreater(np.mean(np.abs(fdot - hi) < 0.25 * (hi - lo)), 0.1)


if __name__ == "__main__":
    unittest.main()
