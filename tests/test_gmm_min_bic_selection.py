"""``vec_fit_gmm_min_bic`` must SELECT a component count, not return its cap.

Found 2026-09-18 while moving the warm start onto a per-cluster mixture. The
sweep fitted each candidate ``n_components`` to the data and then scored the
Bayesian information criterion on ``gmm.rvs(...)`` -- the model's OWN
synthetic draws -- rather than on the data it was fitted to. Scoring a model
against samples it generated is an entropy estimate, and entropy falls
monotonically as components are added, so:

* the criterion never had a minimum to find;
* the "retire a group once its BIC has risen twice past its running minimum"
  rule could never fire;
* every group ran to ``max_comp`` and the sweep returned the cap.

Measured on a clean unimodal 9-D Gaussian before the fix: 30556 at one
component falling to 22255 at seven, with and without resampling.

This matters beyond the warm start -- ``gmm.py`` is shared, and
``fstat_proposal.fit_gmm_to_stacked`` calls the same function with
``max_comp=12``.

Second defect pinned here: the underlying expectation-maximisation ran with
``random_state=None``, so refitting the same data gave a different component
count and different components every time.
"""

import unittest

import numpy as np

from lisatools.sampling.gmm import GMMFit, vec_fit_gmm_min_bic


def _unimodal(n=600, ndim=9, seed=4):
    """One well-conditioned Gaussian blob. The honest answer is K=1."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=(n, ndim))


def _bimodal(n=600, ndim=9, sep=12.0, seed=5):
    """Two blobs a long way apart. The honest answer is K>=2."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(n, ndim))
    x[n // 2:, 0] += sep
    return x


class BicIsScoredOnTheDataTest(unittest.TestCase):
    """The criterion must be evaluated on the fitted data, not on draws."""

    def test_bic_has_a_minimum_for_a_unimodal_blob(self):
        data = _unimodal()[None, :, :]
        bics = []
        for k in range(1, 7):
            fit = GMMFit(data, n_components=k, gpu=None, random_state=0)
            bics.append(float(np.asarray(fit.bic(data)).ravel()[0]))
        self.assertEqual(
            int(np.argmin(bics)), 0,
            f"BIC should be minimised at 1 component for one blob; got "
            f"{bics}")
        self.assertLess(bics[0], bics[-1])

    def test_scoring_on_draws_is_NOT_interchangeable_with_scoring_on_data(self):
        """The two are different quantities -- pinned so the old call cannot
        come back on the reasoning that it is "the same thing".

        Scoring a fitted model against samples it generated measures the
        model's own entropy, not how well it explains the data. The
        magnitudes below are not a stable signature (they move with the draw
        count and the seed), which is itself the point: a selection
        criterion cannot depend on how many synthetic samples you happen to
        draw. What IS stable is that the two disagree.
        """
        data = _unimodal()[None, :, :]
        fit = GMMFit(data, n_components=4, gpu=None, random_state=0)
        on_data = float(np.asarray(fit.bic(data)).ravel()[0])
        on_draws = float(np.asarray(fit.bic(fit.rvs(2000))).ravel()[0])
        self.assertNotAlmostEqual(
            on_data, on_draws, delta=1.0,
            msg="BIC on the data and BIC on the model's own draws must not "
                "be treated as the same number")
        # and the draw-based one moves with an arbitrary knob the data-based
        # one does not have
        other = float(np.asarray(fit.bic(fit.rvs(8000))).ravel()[0])
        self.assertNotAlmostEqual(on_draws, other, delta=1.0)


class SelectionTest(unittest.TestCase):
    def test_unimodal_blob_selects_one_component(self):
        comps = vec_fit_gmm_min_bic(
            _unimodal()[None, :, :], min_comp=1, max_comp=6,
            return_components=True, random_state=0)
        self.assertEqual(len(comps[0][0]), 1)

    def test_bimodal_blob_selects_more_than_one(self):
        comps = vec_fit_gmm_min_bic(
            _bimodal()[None, :, :], min_comp=1, max_comp=6,
            return_components=True, random_state=0)
        self.assertGreaterEqual(len(comps[0][0]), 2)

    def test_the_sweep_does_not_simply_return_its_cap(self):
        """The regression that motivated this file."""
        for cap in (4, 8):
            comps = vec_fit_gmm_min_bic(
                _unimodal()[None, :, :], min_comp=1, max_comp=cap,
                return_components=True, random_state=0)
            self.assertLess(
                len(comps[0][0]), cap,
                f"with max_comp={cap} the sweep returned its cap, i.e. it "
                f"never selected")

    def test_groups_are_selected_independently(self):
        both = np.stack([_unimodal(), _bimodal()])
        comps = vec_fit_gmm_min_bic(
            both, min_comp=1, max_comp=6, return_components=True,
            random_state=0)
        self.assertEqual(len(comps[0][0]), 1)
        self.assertGreaterEqual(len(comps[0][1]), 2)


class ReproducibilityTest(unittest.TestCase):
    def test_same_seed_gives_the_same_fit(self):
        data = _bimodal()[None, :, :]
        a = vec_fit_gmm_min_bic(data, min_comp=1, max_comp=5,
                                return_components=True, random_state=7)
        b = vec_fit_gmm_min_bic(data, min_comp=1, max_comp=5,
                                return_components=True, random_state=7)
        self.assertEqual(len(a[0][0]), len(b[0][0]))
        np.testing.assert_allclose(np.asarray(a[1][0]), np.asarray(b[1][0]),
                                   rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(np.asarray(a[0][0]), np.asarray(b[0][0]),
                                   rtol=1e-10, atol=1e-12)

    def test_random_state_none_is_still_allowed(self):
        """Unseeded stays legal -- this is a shared entry point."""
        comps = vec_fit_gmm_min_bic(
            _unimodal()[None, :, :], min_comp=1, max_comp=4,
            return_components=True, random_state=None)
        self.assertGreaterEqual(len(comps[0][0]), 1)


if __name__ == "__main__":
    unittest.main()
