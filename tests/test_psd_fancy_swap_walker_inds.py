"""The PSD ladder never permutes ``walker_inds`` on a fancy swap (pre-port defect)."""

import unittest

import numpy as np
from eryn.moves.tempering import TemperatureControl
from eryn.state import BranchSupplemental

from lisatools.globalfit.recipe import _noise_temperature_control

NT, NW, ND = 3, 6, 2
IDENTITY = np.tile(np.arange(NW), (NT, 1))


def _like(x_here, inds=None, supps=None, branch_supps=None, **kwargs):
    """Every candidate is much better than the incumbent (logl 0) -> every pair accepted."""
    shape = next(iter(x_here.values())).shape[:2]
    return np.full(shape, 50.0), None


def _fancy_swap_once(tc, seed=3):
    np.random.seed(seed)  # temperature_swaps draws iperm/i1perm/raccept from np.random
    x = {"psd": np.random.uniform(size=(NT, NW, 1, ND))}
    zeros = np.zeros((NT, NW))
    supps = BranchSupplemental(
        {"walker_inds": IDENTITY.copy()}, base_shape=(NT, NW), copy=True
    )
    out = tc.temperature_swaps(
        x, zeros.copy(), zeros.copy(), zeros.copy(),
        supps=supps, branch_supps={"psd": None}, compute_log_like=_like,
        fancy_swap=True, permute_here=True,
    )
    # returns (x, logP, logl, logp, inds, blobs, supps, branch_supps); ``supps[...]``
    # (BranchSupplemental.__getitem__) takes a positional/slice index, not a name --
    # named lookup goes through ``.holder`` (the pattern psdmove.py:1881 uses).
    return np.asarray(out[6].holder["walker_inds"])


class PSDFancySwapWalkerIndsTest(unittest.TestCase):
    def test_noise_control_pins_walker_inds(self):
        tc = _noise_temperature_control(ND, NW, betas=None, ntemps=NT, Tmax=None)
        self.assertEqual(list(tc.skip_swap_supp_names), ["walker_inds"])
        self.assertEqual(int(tc.nwalkers), NW)
        self.assertFalse(tc.permute)
        np.testing.assert_array_equal(_fancy_swap_once(tc), IDENTITY)

    def test_default_control_would_permute_walker_inds(self):
        # negative control: the same swap on an unpinned ladder moves the labels,
        # so the positive test above is not vacuous
        loose = TemperatureControl(ND, NW, ntemps=NT, permute=False)
        self.assertFalse(np.array_equal(_fancy_swap_once(loose), IDENTITY))


if __name__ == "__main__":
    unittest.main()
