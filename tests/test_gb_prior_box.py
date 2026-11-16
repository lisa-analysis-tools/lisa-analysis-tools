"""The Fisher proposal's prior box reads the erebor-priors stock GB prior.

``GBSpecialBase._prior_box`` gives each source the uniform box of its prior in the sampling basis,
with f0 narrowed to the source's cell. The stock prior (``mojitolight.GBPrior``) holds f0 and fdot
in erebor-priors' ``F0FdotPrior``, LAT's ``MojitoF0mHzFdotPrior`` holds the same pair, and both give
their fdot bounds through ``get_fdot_bounds``. Both must give the same box, with the Mojito fdot
bounds at each source's f0. The box is computed on a stand-in for the move, on the CPU.
"""

from __future__ import annotations

import types
import unittest

import numpy as np

# * Two sources in the sampling basis: ln A, f0 [mHz], fdot, phi0, cos iota, psi, alpha, sin delta.
SOURCES = np.array(
    [
        [-50.0, 1.0, 1.0e-18, 1.0, 0.3, 2.0, 4.0, -0.2],
        [-49.0, 12.0, 1.0e-16, 5.0, -0.7, 0.5, 1.0, 0.6],
    ]
)
# * The sources' cells in Hz: 0.9 to 1.1 mHz and 11.5 to 12.5 mHz.
CELL_F0_LIMITS_HZ = (np.array([0.9e-3, 11.5e-3]), np.array([1.1e-3, 12.5e-3]))
F0_COLUMN = 1
FDOT_COLUMN = 2


def stock_gb_prior():
    from ereborpriors.stock.mojitolight import GBPrior

    return GBPrior.prior_factory()


def prior_box(prior) -> tuple[np.ndarray, np.ndarray]:
    """The box of ``SOURCES`` under ``prior``, from the move's own ``_prior_box``."""
    from lisatools.globalfit.moves.gbspecialmove import GBSpecialBase

    transform = stock_gb_prior().get_transforms()
    stand_in_move = types.SimpleNamespace(
        xp=np,
        gpu_priors={"gb": prior},
        transform_fn=transform,
        parameter_transforms=transform,
        # ? Without a periodic container the periodic columns keep their prior range, [0, 2 pi].
        periodic=None,
    )
    return GBSpecialBase._prior_box(stand_in_move, SOURCES.copy(), CELL_F0_LIMITS_HZ)


def mojito_fdot_bounds(f0_mhz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The Mojito fdot bounds, power laws in f0 [Hz]."""
    f0_hz = f0_mhz * 1e-3
    return -2e-20 * (f0_hz / 4e-4) ** (16 / 3), 3e-21 * (f0_hz / 1e-4) ** (11 / 3)


class StockGBPriorBoxTest(unittest.TestCase):
    def test_fdot_bounds_are_the_mojito_power_laws(self):
        lower, upper = prior_box(stock_gb_prior())
        fdot_low, fdot_high = mojito_fdot_bounds(SOURCES[:, F0_COLUMN])
        np.testing.assert_allclose(lower[:, FDOT_COLUMN], fdot_low, rtol=1e-14)
        np.testing.assert_allclose(upper[:, FDOT_COLUMN], fdot_high, rtol=1e-14)

    def test_f0_is_narrowed_to_the_cell(self):
        lower, upper = prior_box(stock_gb_prior())
        np.testing.assert_allclose(lower[:, F0_COLUMN], [0.9, 11.5], rtol=1e-9)
        np.testing.assert_allclose(upper[:, F0_COLUMN], [1.1, 12.5], rtol=1e-9)

    def test_same_box_as_the_lat_joint_prior(self):
        from eryn.priors import ProbDistContainer

        from lisatools.globalfit.priors.gbpriors import MojitoF0mHzFdotPrior

        stock_prior = stock_gb_prior()
        # * The stock entries, with LAT's joint (f0, fdot) class over the same f0 range.
        priors_in = dict(stock_prior.priors_in)
        priors_in[("f0", "fdot")] = MojitoF0mHzFdotPrior(f0_min_mHz=0.1, f0_max_mHz=30.0)
        lat_prior = ProbDistContainer(priors_in)
        for stock_side, lat_side in zip(prior_box(stock_prior), prior_box(lat_prior)):
            np.testing.assert_array_equal(stock_side, lat_side)


if __name__ == "__main__":
    unittest.main()
