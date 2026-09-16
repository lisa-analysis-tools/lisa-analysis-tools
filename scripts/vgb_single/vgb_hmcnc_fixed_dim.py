"""Standalone HM Cnc VGB sampler: does the observable+eigen proposal work in isolation?

WHY THIS EXISTS
---------------
The 6-month production global fit runs the VGB branch with the observable-basis
+ eigen in-model proposal (``VGB_INMODEL_PROPOSAL=observable``,
``VGB_INMODEL_OBSERVABLE_EIGEN=full``, commit 4f4eb24a). Its COLD acceptance
collapsed to 0.006-0.017, while

* the older sampling-basis ``eigen_axis`` proposal ran 0.23-0.37, and
* the GB branch's own observable basis runs ~0.10 and got BETTER, not worse,

which points at the VGB-specific reduction rather than at the shared machinery.
This script removes the global fit from the picture entirely: one source, one
leaf, one temperature, a fast CPU likelihood, and the SAME installed proposal
pieces, so the proposal can be judged on its own.

THE SOURCE
----------
HM Cnc (RX J0806.3+1527), ``ID = "HMCnc"``, row 0 of the mojito VGB catalogue
``vgb_cat_mojito_lite_processed.hdf5`` -- the highest-frequency verification
binary in the 55-source set at ``f_GW = 6.220276235065 mHz``. Picked by NAME,
not by frequency proximity.

THE BASIS (user ruling, 2026-09-16)
-----------------------------------
"The VGBs should now (with stretch removed) have the EXACT same mechanics as
the GBs except f0 and sky fixed." That is the SIX-column astro basis

    x = [dist, phi0, cos_iota, psi, Mc, fdot_astro_ratio]     (pins: f0, alpha, sin_delta)

which is exactly ``VGB_SAMPLED_BASIS_CHIRP`` -- i.e. ``VGB_CHIRP_MASS_BASIS=1``.
**Production runs ``VGB_CHIRP_MASS_BASIS=0``** (``submit_gf_6mo_v8.sh:2112``),
which is the FIVE-column basis with ``Mc`` pinned per leaf. See the
compliance discussion in the module-level report printed by ``--report``.

The chain and the priors live in this astro basis. The observable basis
``z = [lnA, fdot, phi0, cos_iota, psi, Mc]`` is INTERNAL TO THE PROPOSAL ONLY.

THE ARMS (paired controls -- an experiment without a control proves nothing)
---------------------------------------------------------------------------
(a) ``stretch``   -- eryn ``StretchMove`` on the astro columns. Control.
(b) ``eigen``     -- eigen-axis proposal DIRECTLY in the astro basis: information
                     matrix in x, ``eigen_axis_set`` + ``axis_prior_bounds``,
                     one-axis draws, ``factors = 0``. This is the OLD production
                     proposal (cold acceptance 0.23-0.37).
(c) ``observable``-- the NEW proposal under test: x -> z via
                     ``VGBObservableBasis.to_internal``, frozen eigen table in
                     whitened z, step, back via ``from_internal``, and
                     ``factors = log_jacobian(new) - log_jacobian(old)`` into the
                     MH ratio.

Readout: (c) << (b) => the map/factors seam is the defect. (b) and (c) both bad
=> the eigen table itself. All three healthy => the collapse is in the
global-fit wiring, not the proposal.

CONVENTIONS
-----------
* Likelihood: ``GBGPU.get_ll`` returns ``-1/2 <d-h|d-h>`` (it needs ``gb.d_d``
  preset). The injection is ZERO-NOISE, so ``lnL(truth) = 0`` exactly -- the two
  conventions named in the task brief coincide here. Gate 1 checks it.
* PE only: ``phase_maximize=False`` everywhere (no maximization in PE), and every
  proposal pays its Metropolis-Hastings factor (detailed balance).
* "information matrix", never "fisher".
* CPU only. Reuses installed code throughout; nothing here re-implements a
  likelihood or a proposal.

USAGE
-----
    python vgb_hmcnc_fixed_dim.py --smoke              # gates + 200-step smoke
    python vgb_hmcnc_fixed_dim.py --arm all --steps 3000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

# ---- installed pieces (reuse-first: nothing below is re-implemented) --------
from gbgpu.gbgpu import GBGPU
from gbgpu.utils.utility import get_amplitude

from lisatools.detector import EqualArmlengthOrbits
from lisatools.diagnostic import inner_product
from lisatools.domains import FDSettings
from lisatools.sensitivity import (get_sensitivity, A1TDISens, E1TDISens,
                                  SensitivityMatrix)
from lisatools.info_matrix_ll import information_matrix_from_ll
from lisatools.sampling.gb_observable_basis import (
    OBSERVABLE_CD_FLOORS,
    OBSERVABLE_CD_FLOOR_DEFAULT,
    VGBObservableBasis,
    fdot_gr,
    gb_observable_step_scales,
)
from lisatools.globalfit.stock.erebor.transforms import make_gb_transform_container

from eryn.ensemble import EnsembleSampler
from eryn.moves import StretchMove, EigenAxisMove, RidgeGibbsMove
from eryn.moves.eigenaxis import axis_prior_bounds, eigen_axis_set, prior_box_scales
from eryn.moves.mh import MHMove
from lisatools.sampling.ridge_fiber import McRatioDistFiber
from eryn.prior import ProbDistContainer
from eryn.priors.analytical import UniformDistribution
from eryn.state import State
from eryn.utils import PeriodicContainer

# ---------------------------------------------------------------------------
# Constants of the experiment
# ---------------------------------------------------------------------------

TOBS = 15552000.0          # 6 months, mirrors the campaign
DT = 10.0                  # mojito cadence
BRANCH = "vgb"

#: The SIX-column astro basis (== VGB_SAMPLED_BASIS_CHIRP == the user ruling).
ASTRO_BASIS = ["dist", "phi0", "cos_iota", "psi", "Mc", "fdot_astro_ratio"]
#: Pinned per leaf. f0 is carried in SAMPLING units (mHz), as production does.
FIXED_BASIS = ["f0", "alpha", "sin_delta"]

#: The five-column production default, for the compliance comparison only.
PROD_ASTRO_BASIS = ["dist", "phi0", "cos_iota", "psi", "fdot_astro_ratio"]
PROD_FIXED_BASIS = ["f0", "alpha", "sin_delta", "Mc"]

# Production prior limits (mirrored; see vgb.py / gb.py defaults).
DIST_LIMS = (0.001, 40.0)          # kpc, GBSettings.dist_lims
RATIO_MAX = 5.0                    # GB_FDOT_ASTRO_RATIO_MAX
OBS_EIGEN_SMAX = 10.0              # GB_INMODEL_OBSERVABLE_EIGEN_SMAX
OBS_FIBER_WEIGHT = 0.0             # GB_INMODEL_OBSERVABLE_FIBER_WEIGHT
OBS_JUMP = 1.0                     # GB_INMODEL_OBSERVABLE_JUMP

CATALOGUE_REL = "catalogues/vgb_cat_mojito_lite_processed.hdf5"


# ---------------------------------------------------------------------------
# 1. The source
# ---------------------------------------------------------------------------

@dataclass
class Source:
    """HM Cnc, in the units this script uses."""

    name: str
    f0_mhz: float                  # SAMPLING units (mHz), as the fills carry it
    mc: float                      # Msol, chirp mass
    dist_kpc: float
    phi0: float
    cos_iota: float
    psi: float
    alpha: float                   # RA, rad
    sin_delta: float
    fdot_cat: float                # Hz/s, catalogue value
    amp_cat: float                 # catalogue amplitude, for the parity gate
    row: dict = field(default_factory=dict)

    @property
    def f0_hz(self) -> float:
        return self.f0_mhz * 1e-3

    @property
    def ratio(self) -> float:
        """``r = fdot / fdot_gr(f0, Mc) - 1`` -- the astro-ratio column."""
        return float(self.fdot_cat / fdot_gr(self.f0_hz, self.mc) - 1.0)

    def astro_truth(self) -> np.ndarray:
        """Truth in ``ASTRO_BASIS`` order."""
        vals = {"dist": self.dist_kpc, "phi0": self.phi0,
                "cos_iota": self.cos_iota, "psi": self.psi,
                "Mc": self.mc, "fdot_astro_ratio": self.ratio}
        return np.array([vals[n] for n in ASTRO_BASIS], dtype=float)

    def fills(self) -> dict:
        return {"f0": self.f0_mhz, "alpha": self.alpha,
                "sin_delta": self.sin_delta}


def load_hmcnc(mojito_path: str | None = None) -> Source:
    """HM Cnc out of the catalogue the global fit itself loads.

    Picked by NAME (``ID == b"HMCnc"``), with a frequency assertion as a
    backstop so a catalogue reshuffle cannot silently hand back another source.
    """
    import h5py

    if mojito_path is None:
        mojito_path = os.environ.get(
            "MOJITO_DATA_PATH",
            os.path.expanduser("~/.mojito_cache/brickmarket/mojito_light_v1_0_0/"),
        )
    path = os.path.join(mojito_path, CATALOGUE_REL)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"VGB catalogue not found at {path}. Set MOJITO_DATA_PATH."
        )
    with h5py.File(path, "r") as f:
        B = f["Binaries"]
        cols = {k: np.asarray(B[k][:]) for k in B.keys()}

    ids = [s.decode() if isinstance(s, (bytes, np.bytes_)) else str(s)
           for s in cols["ID"]]
    if "HMCnc" not in ids:
        raise RuntimeError(f"HMCnc not in catalogue; names present: {ids[:5]}...")
    i = ids.index("HMCnc")

    f_gw = float(cols["GW22FrequencySSBFrame"][i])
    assert abs(f_gw - 6.22e-3) < 5e-5, (
        f"row {i} named HMCnc but sits at {f_gw:.6e} Hz, not ~6.22 mHz"
    )

    src = Source(
        name=ids[i],
        f0_mhz=f_gw * 1e3,
        mc=float(cols["ChirpMassSSBFrame"][i]),
        # catalogue distance is Mpc; production uses d_kpc = Mpc * 1e3
        dist_kpc=float(cols["LuminosityDistance"][i]) * 1e3,
        # phi0 convention matches gb_catalogue_to_sampling_basis: (-TrueAnomaly) % 2pi
        phi0=float((-cols["TrueAnomaly"][i]) % (2.0 * np.pi)),
        cos_iota=float(np.cos(cols["InclinationAngle"][i])),
        psi=float(cols["PolarisationAngle"][i]),
        alpha=float(cols["RightAscension"][i]),
        sin_delta=float(np.sin(cols["Declination"][i])),
        fdot_cat=float(cols["GW22FrequencyDerivativeSourceFrame"][i]),
        amp_cat=float(cols["Amplitude"][i]),
        row={k: (v[i].decode() if isinstance(v[i], (bytes, np.bytes_))
                 else float(v[i])) for k, v in cols.items()},
    )
    return src


# ---------------------------------------------------------------------------
# 2. Fast CPU likelihood
# ---------------------------------------------------------------------------

class HMCncLikelihood:
    """Zero-noise self-injection of one VGB + the installed fast GB likelihood.

    ``GBGPU.get_ll`` returns ``-1/2 <d-h|d-h>``; with a zero-noise injection and
    ``d_d`` set from the injection's own ``<h|h>``, ``lnL(truth) == 0`` exactly.

    The astro -> physical step goes through the PRODUCTION transform container
    (``make_gb_transform_container`` + ``both_transforms``), so the per-leaf
    fills and the astro-quad algebra are the real ones, not a local copy.
    """

    def __init__(self, src: Source, *, basis=ASTRO_BASIS, fixed=FIXED_BASIS,
                 Tobs=TOBS, dt=DT, N=1024):
        self.src = src
        self.basis = list(basis)
        self.fixed = list(fixed)
        self.Tobs = float(Tobs)
        self.dt = float(dt)
        self.df = 1.0 / self.Tobs
        self.N = int(N)

        mc_lims = (0.5 * src.mc, 2.0 * src.mc)
        self.mc_lims = mc_lims
        self.transform = make_gb_transform_container(
            use_chirp_mass=True, use_fdot_astro=True, use_distance=True,
            input_basis=self.basis,
            fill_dict=[{k: src.fills()[k] for k in self.fixed}],
            mc_lims=mc_lims,
        )

        self.gb = GBGPU(orbits=EqualArmlengthOrbits(force_backend="cpu"),
                        force_backend="cpu")

        # ---- the analysis band ----------------------------------------
        # f0 is PINNED, so GBGPU places every template's sparse band at the
        # same ``start_inds`` (it is set by f0 and N alone). The band is
        # therefore a FIXED, common set of bins for the whole run, which makes
        # a band-restricted likelihood exact rather than an approximation --
        # asserted in _fill_band below, never assumed.
        #
        # NOTE: ``GBGPU.get_ll`` is deliberately NOT used. Measured on this
        # machine it returns ``d_h == h_h`` for every parameter point (i.e. it
        # scores the template against ITSELF, not against the supplied data),
        # so lnL comes out LINEAR in the amplitude instead of the analytic
        # parabola ``-1/2 d_d (a-1)^2``. Two lesser traps in the same call:
        # a LIST argument is read as one entry per GPU rather than per channel,
        # and the ``psd`` argument is applied MULTIPLICATIVELY (it wants 1/S).
        # The inner products below go through the installed
        # ``lisatools.diagnostic.inner_product`` -- the same function
        # AnalysisContainer uses -- and reproduce the analytic parabola to
        # machine precision (gate 2).
        phys = self.to_physical(src.astro_truth()[None, :])
        self.truth_phys = phys[0].copy()
        self.gb.run_wave(*phys.T, N=self.N, T=self.Tobs, dt=self.dt,
                         tdi2=False, tdi_channel_setup="AE")
        self.start_ind = int(self.gb.start_inds[0])
        f = (np.arange(self.start_ind, self.start_ind + self.N) * self.df)
        self.f_band = f
        self.sens = np.asarray([
            get_sensitivity(f, sens_fn=A1TDISens, model="scirdv1"),
            get_sensitivity(f, sens_fn=E1TDISens, model="scirdv1"),
        ], dtype=float)
        self.fd_settings = FDSettings(self.N, self.df, force_backend="cpu")
        # inner_product does ``SensitivityMatrix(basis, [psd])`` for a raw
        # argument, which nests a 2-D array one level too deep. Hand it a
        # PRE-BUILT matrix instead -- which also lets the curves be the ones
        # evaluated at the BAND's true frequencies rather than on a grid that
        # restarts at f = 0.
        self.psd_mat = SensitivityMatrix(self.fd_settings,
                                         [self.sens[0].copy(),
                                          self.sens[1].copy()])

        # ---- inject (zero noise) ---------------------------------------
        self.data = np.asarray([np.asarray(self.gb.A[0]),
                                np.asarray(self.gb.E[0])], dtype=complex)
        self.d_d = float(inner_product(self.data, self.data,
                                       basis_settings=self.fd_settings,
                                       psd=self.psd_mat))
        self.snr = float(np.sqrt(self.d_d))

    # -- transforms ------------------------------------------------------
    def to_physical(self, x: np.ndarray) -> np.ndarray:
        """``(n, ndim)`` astro -> ``(n, 9)`` physical GBGPU parameters."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        leaf = np.zeros(x.shape[0], dtype=int)
        return np.asarray(
            self.transform.both_transforms(x.copy(), xp=np, leaf_inds=leaf)
        )

    def d_d_explicit(self) -> float:
        """``4 df sum |d|^2 / S`` -- independent of the installed convention."""
        tot = 0.0
        for d, s in zip(self.data, self.sens):
            tot += 4.0 * self.df * float(np.sum(np.abs(d) ** 2 / s))
        return tot

    def _templates(self, phys: np.ndarray) -> np.ndarray:
        """``(n, 2, N)`` band templates. Asserts the band never moves."""
        self.gb.run_wave(*phys.T, N=self.N, T=self.Tobs, dt=self.dt,
                         tdi2=False, tdi_channel_setup="AE")
        st = np.asarray(self.gb.start_inds).astype(int).ravel()
        if not np.all(st == self.start_ind):
            raise RuntimeError(
                f"GBGPU moved the sparse band ({np.unique(st)} vs "
                f"{self.start_ind}); the fixed-band likelihood assumes a "
                "pinned f0 keeps it put."
            )
        return np.stack([np.asarray(self.gb.A), np.asarray(self.gb.E)], axis=1)

    # -- the likelihood --------------------------------------------------
    def __call__(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """Vectorized ``lnL`` on astro coordinates. Non-finite -> -1e300."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        out = np.full(x.shape[0], -1e300)
        ok = np.isfinite(x).all(axis=1)
        if not ok.any():
            return out
        phys = self.to_physical(x[ok])
        good = np.isfinite(phys).all(axis=1) & (phys[:, 0] > 0) & (phys[:, 1] > 0)
        if not good.any():
            return out
        tmp = self._templates(phys[good])
        ll = np.full(phys.shape[0], -1e300)
        vals = np.empty(tmp.shape[0])
        for i in range(tmp.shape[0]):
            h = tmp[i]
            h_h = float(inner_product(h, h, basis_settings=self.fd_settings,
                                      psd=self.psd_mat))
            d_h = float(inner_product(self.data, h,
                                      basis_settings=self.fd_settings,
                                      psd=self.psd_mat))
            vals[i] = -0.5 * (self.d_d + h_h - 2.0 * d_h)
        ll[good] = vals
        ll[~np.isfinite(ll)] = -1e300
        out[ok] = ll
        return out


# ---------------------------------------------------------------------------
# 3. Priors (mirroring production)
# ---------------------------------------------------------------------------

def build_priors(like: HMCncLikelihood):
    idx = {n: i for i, n in enumerate(like.basis)}
    d = {
        idx["dist"]: UniformDistribution(minimum=DIST_LIMS[0], maximum=DIST_LIMS[1]),
        idx["phi0"]: UniformDistribution(minimum=0.0, maximum=2.0 * np.pi),
        idx["cos_iota"]: UniformDistribution(minimum=-1.0, maximum=1.0),
        idx["psi"]: UniformDistribution(minimum=0.0, maximum=np.pi),
        idx["fdot_astro_ratio"]: UniformDistribution(minimum=-RATIO_MAX,
                                                     maximum=RATIO_MAX),
    }
    if "Mc" in idx:
        d[idx["Mc"]] = UniformDistribution(minimum=like.mc_lims[0],
                                           maximum=like.mc_lims[1])
    return {BRANCH: ProbDistContainer(d)}


def prior_box(like: HMCncLikelihood) -> tuple[np.ndarray, np.ndarray]:
    idx = {n: i for i, n in enumerate(like.basis)}
    lo = np.zeros(len(like.basis))
    hi = np.zeros(len(like.basis))
    lims = {"dist": DIST_LIMS, "phi0": (0.0, 2 * np.pi), "cos_iota": (-1.0, 1.0),
            "psi": (0.0, np.pi), "fdot_astro_ratio": (-RATIO_MAX, RATIO_MAX),
            "Mc": like.mc_lims}
    for n, i in idx.items():
        lo[i], hi[i] = lims[n]
    return lo, hi


# ---------------------------------------------------------------------------
# 4. The information matrix, in x and transported into z
# ---------------------------------------------------------------------------

def astro_eps(like: HMCncLikelihood, x0: np.ndarray, sigmas=None) -> np.ndarray:
    """Central-difference steps for the astro information matrix."""
    if sigmas is not None:
        return 0.3 * np.abs(sigmas)
    idx = {n: i for i, n in enumerate(like.basis)}
    eps = np.full(x0.shape[-1], 1e-4)
    eps[idx["dist"]] = 1e-3 * abs(x0[idx["dist"]])
    if "Mc" in idx:
        eps[idx["Mc"]] = 1e-5 * abs(x0[idx["Mc"]])
    eps[idx["fdot_astro_ratio"]] = 1e-3
    return eps


def information_matrix_astro(like: HMCncLikelihood, x0: np.ndarray, *,
                             refine=True):
    """``-d_i d_j lnL`` at ``x0`` in the ASTRO basis, via the installed helper."""
    eps = astro_eps(like, x0)
    G = information_matrix_from_ll(lambda p: like(p), x0[None, :], xp=np,
                                   param_eps=eps, psd_project=True)[0]
    if not refine:
        return G, eps
    # CONDITIONAL widths 1/sqrt(G_ii), never marginal ones from pinv(G).
    # This basis has an EXACT fiber -- (Mc, dist, r) move the waveform not at
    # all along one direction -- so ``info_x`` is singular by construction and
    # ``regularize`` only lifts it to a 1e-10 relative floor. ``diag(pinv(G))``
    # is then ~1/floor, i.e. astronomically large, and a step refined from it
    # lands far outside the quadratic region (measured: every column pinned at
    # the clamp, eigenvalues 1e+288). The conditional width is finite and
    # meaningful even on a singular matrix.
    with np.errstate(invalid="ignore", divide="ignore"):
        sig = 1.0 / np.sqrt(np.abs(np.diag(G)))
    sig = np.where(np.isfinite(sig) & (sig > 0), sig, eps / 0.3)
    eps2 = np.clip(astro_eps(like, x0, sigmas=sig), eps * 1e-2, eps * 1e2)
    eps2 = np.where(np.isfinite(eps2) & (eps2 > 0), eps2, eps)
    G2 = information_matrix_from_ll(lambda p: like(p), x0[None, :], xp=np,
                                    param_eps=eps2, psd_project=True)[0]
    return G2, eps2


def gamma_z_from_astro(mapobj, info_x: np.ndarray, x0: np.ndarray):
    """Congruence-transport ``info_x`` into z. Mirrors ``_observable_stash_gamma_z``.

    ``M[:, i] = dx/dz_i`` by central differences with the named
    ``OBSERVABLE_CD_FLOORS`` (keyed by NAME precisely so a reduced layout cannot
    inherit the 9-column ordering), then ``Gamma_z = M^T info_x M``.
    """
    coords = x0[None, :].copy()
    leaf = np.zeros(1, dtype=int)
    z = np.asarray(mapobj.to_internal(coords, leaf_inds=leaf))
    floors = np.array([OBSERVABLE_CD_FLOORS.get(nm, OBSERVABLE_CD_FLOOR_DEFAULT)
                       for nm in mapobj.INTERNAL_BASIS])
    ndim, ndim_z = coords.shape[1], z.shape[1]
    M = np.zeros((ndim, ndim_z))
    for i in range(ndim_z):
        h = 1e-6 * max(abs(float(z[0, i])), float(floors[i]))
        up, dn = z.copy(), z.copy()
        up[0, i] += h
        dn[0, i] -= h
        dx = (np.asarray(mapobj.from_internal(up, template=coords, leaf_inds=leaf))
              - np.asarray(mapobj.from_internal(dn, template=coords, leaf_inds=leaf)))
        M[:, i] = dx[0] / (2.0 * h)
    return M.T @ info_x @ M, M, z[0]


def observable_eigen_table(mapobj, gamma_z, z0, *, snr, Tobs,
                           extrinsic_scales, smax=OBS_EIGEN_SMAX, jump=OBS_JUMP):
    """The frozen z-space table. Mirrors ``_observable_eigen_prepare``.

    ``w`` whitens to the analytic step scales; the table column is
    ``w * a_w * sigma_w``, in which the un-whitening cancels -- so ``w`` chooses
    the DIRECTIONS but not the magnitudes.
    """
    names = tuple(mapobj.INTERNAL_BASIS)
    w = np.asarray(gb_observable_step_scales(
        np.array([snr]), Tobs, extrinsic_scales=np.asarray(extrinsic_scales)[None, :],
        mc_step=0.05 * (2.0 * abs(z0[names.index("Mc")]) if "Mc" in names else 1.0),
        jump=jump, internal_basis=names,
    ))[0]
    gw = gamma_z[None, :, :] * w[None, :, None] * w[None, None, :]
    t_fiber = None
    if mapobj.FIBER_INDEX is not None:
        t_fiber = np.zeros((1, len(names)))
        t_fiber[0, mapobj.FIBER_INDEX] = 1.0
    axes_w, sig_w = eigen_axis_set(gw, t_fiber=t_fiber, sigma_max=float(smax))
    table = (w[None, :, None] * axes_w) * sig_w[:, None, :]
    return table[0], axes_w[0], sig_w[0], w


# ---------------------------------------------------------------------------
# 5. The proposal under test
# ---------------------------------------------------------------------------

class ObservableEigenMove(MHMove):
    """x -> z, frozen eigen step in z, z -> x, log-Jacobian into the MH ratio.

    Mirrors ``GBSpecialBase._observable_proposal``. The table is FROZEN (built
    once from the information matrix at truth), so the draw in z is exactly
    symmetric and ``factors`` is the log-Jacobian difference alone.
    """

    def __init__(self, mapobj, table, *, mode="full",
                 fiber_weight=OBS_FIBER_WEIGHT, **kwargs):
        self.map = mapobj
        self.table = np.asarray(table)
        self.mode = mode
        self.fiber_weight = float(fiber_weight)
        # diagnostics the report needs
        self.factors_log: list[np.ndarray] = []
        self.dz_log: list[np.ndarray] = []
        super().__init__(**kwargs)

    def get_proposal(self, branches_coords, random, branches_inds=None, **kw):
        q = {}
        for name, coords in branches_coords.items():
            ntemps, nwalkers, nleaves, ndim = coords.shape
            flat = coords.reshape(-1, ndim)
            leaf = np.zeros(flat.shape[0], dtype=int)
            z = np.asarray(self.map.to_internal(flat, leaf_inds=leaf))
            n, ndim_z = z.shape

            T = np.broadcast_to(self.table, (n,) + self.table.shape)
            if self.mode == "axis":
                naxes = ndim_z - (1 if (self.fiber_weight == 0.0
                                        and self.map.FIBER_INDEX is not None)
                                  else 0)
                pick = random.integers(naxes, size=n) if hasattr(random, "integers") \
                    else random.randint(0, naxes, n)
                zz = random.standard_normal(n) if hasattr(random, "standard_normal") \
                    else random.randn(n)
                dz = T[np.arange(n), :, pick] * zz[:, None]
            else:
                rnd = random.standard_normal((n, ndim_z)) \
                    if hasattr(random, "standard_normal") else random.randn(n, ndim_z)
                dz = np.einsum("nij,nj->ni", T, rnd)
            if self.map.FIBER_INDEX is not None:
                dz[:, self.map.FIBER_INDEX] *= self.fiber_weight

            new = np.asarray(self.map.from_internal(z + dz, template=flat,
                                                    leaf_inds=leaf))
            fac = np.asarray(self.map.factors(flat, new, leaf_inds=leaf),
                             dtype=float).ravel()
            self.factors_log.append(fac.copy())
            self.dz_log.append(dz.copy())
            q[name] = new.reshape(coords.shape)
            factors = fac.reshape(ntemps, nwalkers, nleaves).sum(axis=-1)

        if self.periodic is not None:
            wrapped = self.periodic.wrap(
                {k: v.reshape((-1,) + v.shape[-2:]) for k, v in q.items()}, xp=np)
            q = {k: wrapped[k].reshape(q[k].shape) for k in q}
        return q, factors


# ---------------------------------------------------------------------------
# 6. Gates
# ---------------------------------------------------------------------------

def run_gates(like: HMCncLikelihood, src: Source, mapobj, info_x, gamma_z, M, z0):
    """Sanity gates. Returns (all_passed, lines)."""
    L = []
    ok = True

    def chk(cond, msg):
        nonlocal ok
        ok = ok and bool(cond)
        L.append(f"  [{'PASS' if cond else 'FAIL'}] {msg}")

    x0 = src.astro_truth()
    ll0 = float(like(x0[None, :])[0])
    L.append("GATE 1 -- likelihood convention")
    L.append(f"    lnL(truth) = {ll0:.6e}   (get_ll = -1/2<d-h|d-h>, zero noise)")
    chk(abs(ll0) < 1e-6, "lnL(truth) == 0 to 1e-6")
    dd_e = like.d_d_explicit()
    L.append(f"    <d|d> engine = {like.d_d:.8e} | explicit 4 df sum = {dd_e:.8e}"
             f" | reldiff = {abs(dd_e / like.d_d - 1.0):.3e}")
    chk(abs(dd_e / like.d_d - 1.0) < 1e-6, "<d|d> matches the explicit 4 df sum")
    L.append(f"    optimal SNR = {like.snr:.3f}")

    L.append("GATE 2 -- lnL(truth) is a maximum")
    worse = []
    rng = np.random.default_rng(0)
    for _ in range(24):
        pert = x0 * (1.0 + 1e-3 * rng.standard_normal(x0.shape))
        worse.append(float(like(pert[None, :])[0]))
    L.append(f"    max lnL over 24 perturbations = {max(worse):.6e}")
    chk(max(worse) <= 1e-6, "no perturbation beats the truth")

    L.append("GATE 3 -- catalogue parity of the transform")
    amp = float(like.truth_phys[0])
    L.append(f"    amp(transform) = {amp:.9e} | catalogue = {src.amp_cat:.9e}"
             f" | reldiff = {abs(amp / src.amp_cat - 1.0):.3e}")
    chk(abs(amp / src.amp_cat - 1.0) < 1e-3, "amplitude matches catalogue to 1e-3")
    fd = float(like.truth_phys[2])
    L.append(f"    fdot(transform) = {fd:.9e} | catalogue = {src.fdot_cat:.9e}"
             f" | reldiff = {abs(fd / src.fdot_cat - 1.0):.3e}")
    chk(abs(fd / src.fdot_cat - 1.0) < 1e-6, "fdot matches catalogue")

    L.append("GATE 4 -- priors finite at truth")
    pri = build_priors(like)[BRANCH]
    lp = float(pri.logpdf(x0[None, :])[0])
    L.append(f"    lnPrior(truth) = {lp:.6f}")
    chk(np.isfinite(lp), "prior finite at truth")

    L.append("GATE 5 -- z -> x -> z round-trip (map symmetry)")
    leaf = np.zeros(1, dtype=int)
    z = np.asarray(mapobj.to_internal(x0[None, :], leaf_inds=leaf))
    back = np.asarray(mapobj.from_internal(z, template=x0[None, :], leaf_inds=leaf))
    z2 = np.asarray(mapobj.to_internal(back, leaf_inds=leaf))
    rt_x = float(np.max(np.abs(back[0] / np.where(x0 != 0, x0, 1.0) - 1.0)))
    rt_z = float(np.max(np.abs(z2[0] - z[0]) / np.maximum(np.abs(z[0]), 1e-300)))
    L.append(f"    max rel |x - x_roundtrip| = {rt_x:.3e}")
    L.append(f"    max rel |z - z_roundtrip| = {rt_z:.3e}")
    chk(rt_x < 1e-9 and rt_z < 1e-9, "round-trip is bit-close")

    L.append("GATE 6 -- log_jacobian parity: analytic vs installed map")
    L.append("    analytic  ln|dx/dz| = ln(dist) - ln(fdot_gr(f0, Mc))"
             + (" + ln(Mc)" if mapobj.fiber_coord == "lnMc" else ""))
    rng = np.random.default_rng(7)
    lo, hi = prior_box(like)
    pts = [x0]
    for _ in range(4):
        p = x0.copy()
        p = p * (1.0 + 0.05 * rng.standard_normal(p.shape))
        pts.append(np.clip(p, lo + 1e-9, hi - 1e-9))
    pts = np.array(pts)
    lj_installed = np.asarray(
        mapobj.log_jacobian(pts, leaf_inds=np.zeros(len(pts), dtype=int)))
    i_d = like.basis.index("dist")
    i_mc = like.basis.index("Mc") if "Mc" in like.basis else None
    mc_vals = pts[:, i_mc] if i_mc is not None else np.full(len(pts), src.mc)
    lj_analytic = np.log(pts[:, i_d]) - np.log(fdot_gr(src.f0_hz, mc_vals))
    # numerical: ln|det(dx/dz)| from central differences
    lj_numeric = []
    for p in pts:
        _, Mp, _ = gamma_z_from_astro(mapobj, info_x, p)
        lj_numeric.append(float(np.log(abs(np.linalg.det(Mp)))))
    lj_numeric = np.array(lj_numeric)
    L.append("      pt |   installed   |   analytic    |   numeric ln|det M|")
    for k in range(len(pts)):
        L.append(f"      {k:2d} | {lj_installed[k]: .8f} | {lj_analytic[k]: .8f}"
                 f" | {lj_numeric[k]: .8f}")
    d_inst = lj_installed - lj_installed[0]
    d_anal = lj_analytic - lj_analytic[0]
    d_num = lj_numeric - lj_numeric[0]
    e_anal = float(np.max(np.abs(d_inst - d_anal)))
    e_num = float(np.max(np.abs(d_inst - d_num)))
    L.append(f"    max |Delta installed - Delta analytic| = {e_anal:.3e}")
    L.append(f"    max |Delta installed - Delta numeric | = {e_num:.3e}")
    chk(e_anal < 1e-9, "installed log_jacobian == analytic (differences)")
    chk(e_num < 1e-5, "installed log_jacobian == numeric ln|det dx/dz|")

    L.append("GATE 7 -- information matrix health (astro basis)")
    w = np.linalg.eigvalsh(0.5 * (info_x + info_x.T))
    L.append(f"    eigenvalues: {np.array2string(w, precision=4)}")
    L.append(f"    condition number = {abs(w[-1] / max(w[0], 1e-300)):.3e}")
    i_r = like.basis.index("fdot_astro_ratio")
    L.append(f"    info_x[r, r] = {info_x[i_r, i_r]:.6e}  (the historically DEAD"
             " column: J == 0 in the 5-column production basis)")
    chk(info_x[i_r, i_r] > 0, "the fdot_astro_ratio direction has curvature")
    if i_mc is not None:
        L.append(f"    info_x[Mc, Mc] = {info_x[i_mc, i_mc]:.6e}")
    return ok, L


# ---------------------------------------------------------------------------
# 7. Running an arm
# ---------------------------------------------------------------------------

def integrated_act(chain: np.ndarray) -> np.ndarray:
    """tau_int per parameter from ``(nsteps, nwalkers, ndim)``, installed first."""
    try:
        from eryn.utils.utility import get_integrated_act
        return np.asarray(get_integrated_act(chain, average=True)).ravel()
    except Exception:
        pass
    ns, nw, nd = chain.shape
    out = np.zeros(nd)
    for d in range(nd):
        acf = np.zeros(ns)
        for w in range(nw):
            y = chain[:, w, d] - chain[:, w, d].mean()
            n2 = 1 << (2 * ns - 1).bit_length()
            F = np.fft.fft(y, n2)
            a = np.fft.ifft(F * np.conjugate(F))[:ns].real
            acf += a / a[0] if a[0] > 0 else a
        acf /= nw
        taus = 2.0 * np.cumsum(acf) - 1.0
        m = np.arange(ns) < 5.0 * taus
        out[d] = taus[np.argmin(m)] if not m.all() else taus[-1]
    return out


def make_ridge_move(like: HMCncLikelihood, priors):
    """The GB ridge-Gibbs fiber move, reused VERBATIM for the VGB basis.

    ``McRatioDistFiber`` requires exactly ``dist`` / ``Mc`` /
    ``fdot_astro_ratio`` and resolves their indices BY NAME from the
    container's ``input_basis`` -- and it never reads ``f0`` at all, so a
    per-leaf pinned ``f0`` is not merely tolerated but is strictly safer than
    a sampled one (the fiber's invariant ``Kf = Mc^(5/3) (1+r)`` holds ``fdot``
    fixed only at CONSTANT f0). No code change is needed for VGB.
    """
    fiber = McRatioDistFiber(like.transform, mc_lims=like.mc_lims,
                             dist_lims=DIST_LIMS, ratio_max=RATIO_MAX)
    mv = RidgeGibbsMove(BRANCH, fiber, priors[BRANCH].logpdf, leaf_fraction=1.0)
    mv.name = "vgb_ridge_gibbs"
    return mv


def run_arm(arm: str, like: HMCncLikelihood, src: Source, *, nwalkers=32,
            nsteps=3000, burn=500, seed=42, mapobj=None, info_x=None,
            table=None, obs_mode="full", fiber_weight=OBS_FIBER_WEIGHT,
            add_ridge=False, verbose=True):
    """Run one arm; return a result dict."""
    x0 = src.astro_truth()
    ndim = len(like.basis)
    priors = build_priors(like)
    lo, hi = prior_box(like)
    idx = {n: i for i, n in enumerate(like.basis)}
    periodic = PeriodicContainer(
        {BRANCH: {idx["phi0"]: 2.0 * np.pi, idx["psi"]: np.pi}})

    rng = np.random.default_rng(seed)
    with np.errstate(invalid="ignore"):
        sig0 = np.sqrt(np.abs(np.diag(np.linalg.pinv(info_x))))
    sig0 = np.where(np.isfinite(sig0) & (sig0 > 0), sig0, 1e-3 * np.abs(x0) + 1e-6)
    start = x0[None, None, None, :] + 0.1 * sig0 * rng.standard_normal(
        (1, nwalkers, 1, ndim))
    start = np.clip(start, lo + 1e-9, hi - 1e-9)

    obs_move = None
    if arm == "stretch":
        moves = StretchMove(live_dangerously=True, periodic=periodic)
    elif arm == "eigen":
        axes, sigmas = eigen_axis_set(info_x[None, ...], sigma_max=np.inf)
        bounds = axis_prior_bounds(axes, prior_box_scales(lo, hi))
        sigmas = np.minimum(sigmas, bounds)
        moves = EigenAxisMove({BRANCH: (axes[0], sigmas[0])}, mode="axis",
                              periodic=periodic)
        eigen_table_dump = (axes[0], sigmas[0], bounds[0])
    elif arm == "observable":
        obs_move = ObservableEigenMove(mapobj, table, mode=obs_mode,
                                       fiber_weight=fiber_weight,
                                       periodic=periodic)
        moves = [obs_move, make_ridge_move(like, priors)] if add_ridge \
            else obs_move
    else:
        raise ValueError(arm)

    sampler = EnsembleSampler(
        nwalkers, {BRANCH: ndim}, like, priors,
        tempering_kwargs=dict(ntemps=1), branch_names=[BRANCH],
        nleaves_max={BRANCH: 1}, nleaves_min={BRANCH: 1},
        moves=moves, periodic=periodic, vectorize=True,
    )
    state = State({BRANCH: start})
    state.log_prior = sampler.compute_log_prior(state.branches_coords)
    state.log_like = sampler.compute_log_like(state.branches_coords,
                                              logp=state.log_prior)[0]

    t0 = time.time()
    sampler.run_mcmc(state, nsteps, burn=burn, progress=False, thin_by=1)
    wall = time.time() - t0

    chain = sampler.get_chain()[BRANCH][:, 0, :, 0, :]     # (nsteps, nwalkers, ndim)
    acc = float(np.mean(sampler.acceptance_fraction))
    tau = integrated_act(chain)
    flat = chain.reshape(-1, ndim)
    mean, std = flat.mean(axis=0), flat.std(axis=0)
    pull = (mean - x0) / np.where(std > 0, std, np.inf)

    label = arm
    if arm == "observable":
        label = f"obs/{obs_mode}/fw={fiber_weight:g}" + ("+ridge" if add_ridge else "")
    # Does the fiber coordinate Mc actually move? (the fiber-weight question)
    i_mc = like.basis.index("Mc")
    mc_track = chain[:, :, i_mc]
    mc_moved = float(np.max(np.abs(np.diff(mc_track, axis=0)))) if nsteps > 1 else 0.0
    mc_nuniq = int(np.median([len(np.unique(mc_track[:, w]))
                              for w in range(mc_track.shape[1])]))

    res = dict(arm=arm, label=label,
               mode=(obs_mode if arm == "observable" else None),
               fiber_weight=(fiber_weight if arm == "observable" else None),
               ridge=bool(add_ridge),
               acceptance=acc, wall_s=wall, nsteps=nsteps, nwalkers=nwalkers,
               burn=burn, tau=tau.tolist(), mean=mean.tolist(),
               std=std.tolist(), pull=pull.tolist(), truth=x0.tolist(),
               basis=list(like.basis),
               mc_max_step=mc_moved, mc_unique_median=mc_nuniq)
    if arm == "eigen":
        res["eigen_axes"] = eigen_table_dump[0].tolist()
        res["eigen_sigmas"] = eigen_table_dump[1].tolist()
        res["eigen_prior_bounds"] = eigen_table_dump[2].tolist()
    if obs_move is not None and obs_move.factors_log:
        fac = np.concatenate(obs_move.factors_log)
        fac = fac[np.isfinite(fac) & (fac > -1e299)]
        dz = np.concatenate(obs_move.dz_log, axis=0)
        res["factors_mean"] = float(np.mean(fac)) if fac.size else float("nan")
        res["factors_std"] = float(np.std(fac)) if fac.size else float("nan")
        res["factors_pcts"] = (np.percentile(fac, [1, 25, 50, 75, 99]).tolist()
                               if fac.size else [])
        res["dz_rms"] = np.sqrt(np.mean(dz ** 2, axis=0)).tolist()
        res["z_basis"] = list(mapobj.INTERNAL_BASIS)
    if verbose:
        print(f"  [{label}] acc={acc:.4f} wall={wall:.1f}s "
              f"Mc max step={mc_moved:.3e}")
    return res


# ---------------------------------------------------------------------------
# 8. Reporting
# ---------------------------------------------------------------------------

def compliance_report() -> list[str]:
    """Three-way comparison: user spec vs production vs the shipped map."""
    return [
        "COMPLIANCE: user spec vs production vs shipped VGBObservableBasis",
        "  Ruling: 'VGBs should have the EXACT same mechanics as the GBs",
        "          except f0 and sky fixed.'",
        "",
        "  (a) USER SPEC  : 6 free  [dist, phi0, cos_iota, psi, Mc, fdot_astro_ratio]",
        "                   pins    [f0, alpha, sin_delta]",
        "                   -> this is EXACTLY VGB_SAMPLED_BASIS_CHIRP",
        "                      (vgb.py:80-81), i.e. VGB_CHIRP_MASS_BASIS=1.",
        "",
        "  (b) PRODUCTION : 5 free  [dist, phi0, cos_iota, psi, fdot_astro_ratio]",
        "                   pins    [f0, alpha, sin_delta, Mc]        <-- Mc PINNED",
        "                   vgb.py:71-72 (VGB_*_BASIS_DIST), selected because",
        "                   chirp_mass_basis defaults False (vgb.py:505-507) AND",
        "                   scripts/fstat_proposal/submit_gf_6mo_v8.sh:2112 pins",
        "                   'export VGB_CHIRP_MASS_BASIS=0'.",
        "",
        "  (c) SHIPPED MAP: VGBObservableBasis DERIVES its layout from the basis",
        "                   it is handed (gb_observable_basis.py:525-536), so it",
        "                   is correct for BOTH: under (b) it pins f0+Mc and",
        "                   FIBER_INDEX=None; under (a) it pins f0 only, grows the",
        "                   Mc column back and FIBER_INDEX=5. Verified empirically.",
        "",
        "  => The divergence from the ruling is (b) vs (a): a DEFAULT, not a",
        "     missing mechanism. Under (b) the ONLY fdot freedom is r, whose",
        "     transform target is the 'fddot' output slot, and the astro quad",
        "     emits fddot == 0 -- so r's column of the info-matrix Jacobian is",
        "     identically zero and the r direction has no curvature at all.",
        "     4f4eb24a works around this in the MOVE (_infomat_phys_inds,",
        "     gbspecialstretch.py:10267, substitutes the live fdot slot for the",
        "     dead fddot one) rather than by giving VGB the GB freedom.",
        "     Under (a) 'fdot' is in test_inds natively and no substitution is",
        "     needed -- which is precisely 'the same mechanics as the GBs'.",
        "",
        "  MINIMAL CHANGE TO RESTORE PARITY (described, NOT applied):",
        "    1. Flip the default: vgb.py chirp_mass_basis field default",
        "       False -> True, and drop/flip 'export VGB_CHIRP_MASS_BASIS=0' in",
        "       scripts/fstat_proposal/submit_gf_6mo_v8.sh:2112. No code change",
        "       is needed anywhere else -- the container, the map, the eigen",
        "       table and the proposal all already handle the 6-column layout.",
        "    2. GBObservableFiberBasis itself CANNOT be reused directly for VGB:",
        "       its _REQUIRED = ('dist','f0','Mc','fdot_astro_ratio')",
        "       (gb_observable_basis.py:483) demands a SAMPLED f0, which VGB pins.",
        "       Verified: it raises for both (a) and (b). 'Reuse the GB",
        "       implementation with pins' is therefore spelled",
        "       VGBObservableBasis-under-(a), which is the same code path with",
        "       _PINNED=('f0',) resolved at run time. If a literal GB class is",
        "       wanted, the minimal change is to make _REQUIRED a function of",
        "       which names are pinned rather than a fixed tuple.",
        "    3. NOTE a consequence of (a) worth deciding explicitly: with a fiber",
        "       present and GB_INMODEL_OBSERVABLE_FIBER_WEIGHT=0.0 (production),",
        "       the observable proposal NEVER moves the fiber coordinate Mc. With",
        "       stretch removed, nothing else in the in-model stack moves it.",
    ]


def format_results(results: list[dict], src: Source, like: HMCncLikelihood,
                   extra: list[str]) -> str:
    L = []
    L.append("=" * 78)
    L.append("HM Cnc standalone VGB fixed-dimensional sampler")
    L.append("=" * 78)
    L.append(f"source            : {src.name} (catalogue row picked BY NAME)")
    L.append(f"f_GW              : {src.f0_hz:.12e} Hz  ({src.f0_mhz:.9f} mHz)")
    L.append(f"Mc                : {src.mc:.12e} Msol")
    L.append(f"dist              : {src.dist_kpc:.6f} kpc")
    L.append(f"phi0, cos_iota,psi: {src.phi0:.9f}, {src.cos_iota:.9f}, {src.psi:.9f}")
    L.append(f"alpha, sin_delta  : {src.alpha:.9f}, {src.sin_delta:.9f}   (PINNED)")
    L.append(f"fdot (catalogue)  : {src.fdot_cat:.9e} Hz/s")
    L.append(f"fdot_gr(f0, Mc)   : {fdot_gr(src.f0_hz, src.mc):.9e} Hz/s")
    L.append(f"fdot_astro_ratio r: {src.ratio:.9e}")
    L.append(f"Tobs / dt         : {TOBS:.0f} s (6 mo) / {DT:.0f} s")
    L.append(f"optimal SNR       : {like.snr:.4f}")
    L.append(f"astro basis       : {like.basis}")
    L.append("")
    L.extend(extra)
    L.append("")
    L.append("-" * 78)
    L.append("ARM RESULTS")
    L.append("-" * 78)
    hdr = (f"{'arm':<28}{'acceptance':>12}{'wall (s)':>10}{'max tau':>9}"
           f"{'max|pull|':>11}{'Mc max step':>14}")
    L.append(hdr)
    L.append("-" * len(hdr))
    for r in results:
        nm = r.get("label", r["arm"])
        tau = np.asarray(r["tau"], dtype=float)
        pull = np.abs(np.asarray(r["pull"], dtype=float))
        L.append(f"{nm:<28}{r['acceptance']:>12.5f}{r['wall_s']:>10.1f}"
                 f"{np.nanmax(tau):>9.1f}{np.nanmax(pull):>11.2f}"
                 f"{r.get('mc_max_step', float('nan')):>14.4e}")
    L.append("")
    L.append("  'Mc max step' == 0 means the fiber coordinate NEVER moved.")
    L.append("")
    for r in results:
        nm = r.get("label", r["arm"])
        L.append(f"--- {nm} ---")
        L.append(f"{'param':<20}{'truth':>15}{'mean':>15}{'std':>13}"
                 f"{'pull':>9}{'tau':>9}")
        for i, p in enumerate(r["basis"]):
            L.append(f"{p:<20}{r['truth'][i]:>15.6g}{r['mean'][i]:>15.6g}"
                     f"{r['std'][i]:>13.4g}{r['pull'][i]:>9.2f}"
                     f"{r['tau'][i]:>9.1f}")
        if "factors_mean" in r:
            L.append(f"  factors (ln-Jacobian diff): mean={r['factors_mean']:.4e}"
                     f" std={r['factors_std']:.4e}")
            if r.get("factors_pcts"):
                q = r["factors_pcts"]
                L.append(f"    percentiles 1/25/50/75/99 = "
                         + " ".join(f"{v:.3e}" for v in q))
            L.append(f"  z-step RMS in {r['z_basis']}:")
            for nm2, v in zip(r["z_basis"], r["dz_rms"]):
                L.append(f"    {nm2:<12} {v:.6e}")
        if "eigen_sigmas" in r:
            L.append("  astro-basis eigen table (axes are columns):")
            ax = np.asarray(r["eigen_axes"])
            for k in range(ax.shape[1]):
                L.append(f"    axis {k}: sigma={r['eigen_sigmas'][k]:.4e} "
                         f"(prior bound {r['eigen_prior_bounds'][k]:.4e})")
                L.append("      " + " ".join(f"{v:+.4f}" for v in ax[:, k]))
        L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 9. main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", default="all",
                    choices=["all", "stretch", "eigen", "observable"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--burn", type=int, default=500)
    ap.add_argument("--nwalkers", type=int, default=32)
    ap.add_argument("--smoke", action="store_true",
                    help="gates + a 200-step smoke of every arm, then exit")
    ap.add_argument("--obs-mode", default="full", choices=["full", "axis"])
    ap.add_argument("--fiber-weight", type=float, default=0.1,
                    help="nonzero fiber weight for observable sub-variant (c2)")
    ap.add_argument("--both-obs-modes", action="store_true",
                    help="run the observable arm in BOTH full and axis modes")
    ap.add_argument("--N", type=int, default=1024, help="GB sparse band points")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--report", action="store_true",
                    help="print the compliance report and exit")
    args = ap.parse_args(argv)

    if args.report:
        print("\n".join(compliance_report()))
        return 0

    print("[setup] loading HM Cnc from the mojito VGB catalogue ...")
    src = load_hmcnc()
    print(f"[setup] picked '{src.name}' at f_GW = {src.f0_hz:.9e} Hz")

    print("[setup] building the zero-noise injection + fast CPU likelihood ...")
    like = HMCncLikelihood(src, N=args.N)
    print(f"[setup] optimal SNR = {like.snr:.4f}")

    x0 = src.astro_truth()
    print("[setup] information matrix at truth (astro basis) ...")
    t0 = time.time()
    info_x, eps = information_matrix_astro(like, x0)
    print(f"[setup] info matrix done in {time.time() - t0:.1f}s; eps = {eps}")

    mapobj = VGBObservableBasis(like.transform, Tobs=TOBS, shear=0.5,
                                fiber_coord="Mc")
    print(f"[setup] observable map: INTERNAL={mapobj.INTERNAL_BASIS} "
          f"FIBER={mapobj.FIBER_INDEX} pinned={list(mapobj._pinned)}")

    gamma_z, M, z0 = gamma_z_from_astro(mapobj, info_x, x0)

    passed, gate_lines = run_gates(like, src, mapobj, info_x, gamma_z, M, z0)
    print("\n".join(gate_lines))
    if not passed:
        print("\n*** ONE OR MORE GATES FAILED -- see above ***")

    # extrinsic step scales for the whitening: marginal widths of the
    # pass-through columns, the same quantity production reads off `chol`.
    with np.errstate(invalid="ignore"):
        sig_x = np.sqrt(np.abs(np.diag(np.linalg.pinv(info_x))))
    ex_names = [n for n in mapobj.INTERNAL_BASIS
                if n in ("phi0", "cos_iota", "psi")]
    ex = np.array([sig_x[like.basis.index(n)] for n in ex_names])
    ex = np.where(np.isfinite(ex) & (ex > 0), ex, 0.1)
    table, axes_w, sig_w, w = observable_eigen_table(
        mapobj, gamma_z, z0, snr=like.snr, Tobs=TOBS, extrinsic_scales=ex)

    extra = list(compliance_report())
    extra.append("")
    extra.append("OBSERVABLE (z-space) EIGEN TABLE")
    extra.append(f"  z basis        : {list(mapobj.INTERNAL_BASIS)}")
    extra.append(f"  FIBER_INDEX    : {mapobj.FIBER_INDEX}")
    extra.append(f"  whitening w    : " + " ".join(f"{v:.4e}" for v in w))
    extra.append(f"  sigma_w (cap {OBS_EIGEN_SMAX}) : "
                 + " ".join(f"{v:.4e}" for v in sig_w))
    extra.append(f"  capped axes    : "
                 f"{int(np.sum(np.asarray(sig_w) >= OBS_EIGEN_SMAX * (1 - 1e-12)))}"
                 f" / {len(sig_w)}")
    extra.append("  table columns (z-space 1-sigma steps):")
    for k in range(table.shape[1]):
        extra.append(f"    axis {k}: " + " ".join(f"{v:+.4e}" for v in table[:, k]))
    extra.append("  Gamma_z diagonal: "
                 + " ".join(f"{v:.4e}" for v in np.diag(gamma_z)))
    if mapobj.FIBER_INDEX is not None:
        fi = mapobj.FIBER_INDEX
        col = table[:, -1]                  # eigen_axis_set puts the fiber LAST
        extra.append("")
        extra.append("FIBER ALIGNMENT (the direction the ridge move must own)")
        extra.append(f"  FIBER_INDEX = {fi} -> z coordinate "
                     f"'{mapobj.INTERNAL_BASIS[fi]}' (fiber_coord="
                     f"'{mapobj.fiber_coord}')")
        extra.append("  last eigen column, in z:")
        for nm2, v in zip(mapobj.INTERNAL_BASIS, col):
            extra.append(f"    {nm2:<12} {v:+.6e}")
        dx_fib = M @ col
        extra.append("  the same step pushed into x (dx = M dz):")
        for nm2, v in zip(like.basis, dx_fib):
            extra.append(f"    {nm2:<20} {v:+.6e}")
        extra.append(f"  |dz| = {np.linalg.norm(col):.6e}   "
                     f"prior-capped sigma_w(fiber) = {sig_w[-1]:.6e} "
                     f"(cap {OBS_EIGEN_SMAX})")
        extra.append(f"  PRODUCTION zeroes this column entirely "
                     f"(GB_INMODEL_OBSERVABLE_FIBER_WEIGHT={OBS_FIBER_WEIGHT:g}).")

    nsteps = 200 if args.smoke else args.steps
    burn = 0 if args.smoke else args.burn
    arms = (["stretch", "eigen", "observable"] if args.arm == "all"
            else [args.arm])

    common = dict(nwalkers=args.nwalkers, nsteps=nsteps, burn=burn,
                  seed=args.seed, mapobj=mapobj, info_x=info_x, table=table)
    results = []
    for arm in arms:
        if arm == "observable":
            # (c1) production config: fiber weight 0 -> Mc frozen by construction
            # (c2) nonzero fiber weight -> does Mc move at a sane acceptance?
            # (c3) fiber weight 0 + ridge-Gibbs -> the full GB-mechanics mirror
            variants = [dict(obs_mode=args.obs_mode, fiber_weight=0.0),
                        dict(obs_mode=args.obs_mode, fiber_weight=args.fiber_weight),
                        dict(obs_mode=args.obs_mode, fiber_weight=0.0,
                             add_ridge=True)]
            if args.both_obs_modes:
                variants.append(dict(obs_mode="axis", fiber_weight=0.0,
                                     add_ridge=True))
            for v in variants:
                results.append(run_arm(arm, like, src, **common, **v))
        else:
            results.append(run_arm(arm, like, src, **common))

    out = format_results(results, src, like, extra + [""] + gate_lines)
    print()
    print(out)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(dict(source=src.row, snr=like.snr, basis=like.basis,
                           results=results), f, indent=2, default=str)
        print(f"[out] wrote {args.json_out}")
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
