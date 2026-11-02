"""Spline deviations of the instrument-noise PSDs from a reference spectrum, on
combinations of MOSAs (scripts/notes/SPLINE_NOISE_MODEL.md).

Each MOSA ``m`` (``MOSA_NAMES`` order: 12, 23, 31, 13, 32, 21) has an OMS and a TM PSD

    S_m(f) = A_m^2 S_nom(f) 10^{x_m(f)},

with ``A_m`` the reference level (the injected ASD, fixed), ``S_nom`` the analytic shape
and ``x_m`` a log10 deviation. The deviations are not fitted per MOSA: they are built from
spline curves ``g_c(f)`` on fixed linear combinations of the 6 MOSAs
(:data:`COMBINATIONS`), ``x(f) = M g(f)`` with ``M = R^{-1}`` (:func:`combination_matrix`).
The data constrain these combinations very differently, so each one gets its own knots.

Each curve has knots at

* ``f_min`` and ``f_max`` (the data band): fixed positions, free amplitudes (the "edges");
* up to ``n_interior`` free knots ``(log10 f, amplitude)``, sampled as the leaves of the
  component's own eryn branch (``psd_<noise>_<combination>``). With a fixed number of
  knots every leaf is active; with reversible jump (``n_interior_min`` set) the active
  leaves (``inds``) are the knots in use.

A constant curve (the TM arm differences) has a single amplitude and no interior knots.

The curves are interpolated linearly in log10 f between the knots (``interpolation``;
cubic Akima overshoots by several dex when knots crowd together). Beyond
``[f_min, f_max]`` (e.g. on the downsampling alias grid) they stay at their edge values.

Adjacent knots, edges included, must be at least ``min_log10_separation`` apart. This is
part of the prior: :meth:`SplineNoiseModel.knot_log_prior` rejects closer knots and
renormalises the truncated position prior, so that the prior on the number of knots stays
the one eryn's reversible-jump moves assume (uniform).

Flat parameter row consumed by :class:`~lisatools.sensitivity.XYZSensitivityBackend`::

    [levels (2 n_par) | edges (e_min, e_max per curve) | interior knots (log10 f, amp) ...]

where the interior knots follow the component order and inactive knots are NaN.
:meth:`SplineNoiseModel.flatten` builds it from the ``psd`` branch (the edges) and the
spline branches, and prepends the fixed reference levels.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Optional, Sequence

import numpy as np
from scipy.stats import norm

MOSA_NAMES = ["12", "23", "31", "13", "32", "21"]

# Arms as (clockwise MOSA, counter-clockwise MOSA) indices into MOSA_NAMES: 12/21, 23/32, 31/13.
ARM_MOSAS = [(0, 5), (1, 4), (2, 3)]

COMBINATIONS = (
    "common",
    "armsum_12_23",
    "armsum_23_31",
    "armdiff_12_23",
    "armdiff_23_31",
    "cw_ccw",
)

NOISE_TYPES = ("oms", "tm")

# Defaults of :func:`make_spline_noise_model`.
DEFAULT_AMPLITUDE_SIGMA = 0.3  # prior std of every knot amplitude [dex]
DEFAULT_MIN_SEPARATION = 0.2  # minimum distance between adjacent knots, edges included [dex]
DEFAULT_N_INTERIOR = 3  # interior knots per curve, fixed number
DEFAULT_RJ_N_MIN = 0  # fewest interior knots per curve, reversible jump
DEFAULT_RJ_N_MAX = 8  # most interior knots per curve, reversible jump
DEFAULT_INTERPOLATION = "linear"  # between knots, in log10 f


def combination_matrix() -> np.ndarray:
    """Rows of ``R``: the 6 combinations of per-MOSA log deviations, ``g = R x``.

    With ``s_k`` / ``d_k`` the half sum / half difference of the clockwise and
    counter-clockwise MOSAs on arm ``k``:

    * ``common``: mean of all 6 MOSAs;
    * ``armsum_12_23``, ``armsum_23_31``: ``s_12 - s_23``, ``s_23 - s_31``;
    * ``armdiff_12_23``, ``armdiff_23_31``: ``d_12 - d_23``, ``d_23 - d_31``;
    * ``cw_ccw``: ``(d_12 + d_23 + d_31) / 3``, all clockwise against all
      counter-clockwise MOSAs (an exact null of the TM noise).
    """
    s = np.zeros((3, 6))
    d = np.zeros((3, 6))
    for k, (cw, ccw) in enumerate(ARM_MOSAS):
        s[k, cw] = s[k, ccw] = 0.5
        d[k, cw], d[k, ccw] = 0.5, -0.5
    return np.array([np.full(6, 1 / 6), s[0] - s[1], s[1] - s[2], d[0] - d[1], d[1] - d[2], d.sum(0) / 3])


def mixing_vectors() -> dict:
    """Per-MOSA deviation pattern of a unit curve on each combination (columns of ``R^{-1}``)."""
    M = np.linalg.inv(combination_matrix())
    return {c: M[:, i] for i, c in enumerate(COMBINATIONS)}


@dataclasses.dataclass(frozen=True)
class SplineComponent:
    """One spline curve of the noise model.

    Args:
        noise: ``"oms"`` or ``"tm"``.
        combination: One of :data:`COMBINATIONS`. The symmetric model only has ``"common"``.
        n_interior: Number of free interior knots (leaves of the component's branch); the
            maximum number with reversible jump. ``0``: only the edge knots, and no branch.
        n_interior_min: ``None`` for a fixed number of knots (every leaf active), or the
            minimum number of knots for reversible jump.
        edge_sigma: Prior standard deviation of the edge amplitudes [log10 PSD].
        knot_sigma: Prior standard deviation of the interior knot amplitudes [log10 PSD].
        amplitude_bound: If given, the edge and interior knot amplitudes have the prior
            ``U(-amplitude_bound, amplitude_bound)`` [log10 PSD] instead of the normal
            priors above.
        constant: A frequency-independent deviation: one amplitude, shared by both edge
            knots, and no interior knots.
    """

    noise: str
    combination: str = "common"
    n_interior: int = 0
    n_interior_min: Optional[int] = None
    edge_sigma: float = DEFAULT_AMPLITUDE_SIGMA
    knot_sigma: float = DEFAULT_AMPLITUDE_SIGMA
    amplitude_bound: Optional[float] = None
    constant: bool = False

    @property
    def name(self) -> str:
        return f"{self.noise}_{self.combination}"

    @property
    def n_edge(self) -> int:
        """Edge amplitudes in the ``psd`` branch: 1 for a constant curve, else 2."""
        return 1 if self.constant else 2

    @property
    def branch_name(self) -> Optional[str]:
        """Eryn branch holding the interior knots, ``None`` without interior knots."""
        return f"psd_{self.name}" if self.n_interior > 0 else None

    @property
    def nleaves_min(self) -> int:
        """Fewest active knots: ``n_interior_min``, or ``n_interior`` for a fixed number."""
        return self.n_interior if self.n_interior_min is None else self.n_interior_min

    @property
    def varies(self) -> bool:
        """Whether reversible jump changes the number of knots of this curve."""
        return self.nleaves_min < self.n_interior

    def amplitude_prior(self, sigma: float):
        """Prior of one knot amplitude: uniform within ``amplitude_bound``, else ``N(0, sigma)``."""
        if self.amplitude_bound is not None:
            from eryn.prior import uniform_dist

            return uniform_dist(-self.amplitude_bound, self.amplitude_bound)
        return norm(0.0, sigma)

    def draw_amplitudes(self, rng, size, sigma: float):
        """Prior draws of knot amplitudes (see :meth:`amplitude_prior`)."""
        if self.amplitude_bound is not None:
            return rng.uniform(-self.amplitude_bound, self.amplitude_bound, size)
        return rng.normal(0.0, sigma, size)


class SplineNoiseModel:
    """Spline deviations from a reference spectrum on MOSA combinations, for the OMS and
    TM noise.

    Args:
        components: The spline curves. At most one per (noise, combination).
        f_min, f_max: Data band [Hz]; the edge knots sit here.
        reference_levels: The fixed amplitudes ``[OMS | TM]``, ``n_par`` each (the injected
            ASDs), around which the curves describe every deviation.
        noise_symmetry: ``"asymmetric"`` (per-MOSA amplitudes, any combination) or
            ``"symmetric"`` (one amplitude per noise type, ``"common"`` curves only).
        min_log10_separation: Minimum distance between adjacent knots in log10 f, edges
            included (a prior constraint, see :meth:`knot_log_prior`). The default 0.2 dex
            is about the frequency resolution of the noise curves.
        interpolation: ``"linear"`` (default) or ``"cubic"`` (Akima), passed to the
            ``cudakima`` interpolant of the sensitivity backend.
    """

    def __init__(
        self,
        components: Sequence[SplineComponent],
        f_min: float,
        f_max: float,
        reference_levels: Sequence[float],
        noise_symmetry: str = "asymmetric",
        min_log10_separation: float = DEFAULT_MIN_SEPARATION,
        interpolation: str = DEFAULT_INTERPOLATION,
    ):
        if noise_symmetry not in ("symmetric", "asymmetric"):
            raise ValueError(f"noise_symmetry must be 'symmetric' or 'asymmetric', got {noise_symmetry!r}.")
        if not 0 < f_min < f_max:
            raise ValueError(f"Need 0 < f_min < f_max, got {f_min}, {f_max}.")
        self.components = list(components)
        self.f_min, self.f_max = float(f_min), float(f_max)
        self.log10_f_min = np.log10(self.f_min)
        self.log10_f_max = np.log10(self.f_max)
        self.noise_symmetry = noise_symmetry
        self.min_log10_separation = float(min_log10_separation)
        if interpolation not in ("linear", "cubic"):
            raise ValueError(f"interpolation must be 'linear' or 'cubic', got {interpolation!r}.")
        self.interpolation = interpolation
        reference_levels = np.asarray(reference_levels, dtype=np.float64).ravel()
        if reference_levels.size != 2 * self.n_par or not np.all(reference_levels > 0):
            raise ValueError(
                f"reference_levels must be {2 * self.n_par} positive amplitudes [OMS | TM], "
                f"got {reference_levels}."
            )
        self.reference_levels = reference_levels

        names = [c.name for c in self.components]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate spline components: {names}.")
        width = self.log10_f_max - self.log10_f_min
        for c in self.components:
            if c.noise not in NOISE_TYPES:
                raise ValueError(f"Unknown noise type {c.noise!r}; use one of {NOISE_TYPES}.")
            if c.combination not in COMBINATIONS:
                raise ValueError(f"Unknown combination {c.combination!r}; use one of {COMBINATIONS}.")
            if noise_symmetry == "symmetric" and c.combination != "common":
                raise ValueError("The symmetric noise model only has 'common' spline components.")
            if c.noise == "tm" and c.combination == "cw_ccw":
                raise ValueError(
                    "tm_cw_ccw is an exact null of the data: any curve on it is set by the prior alone."
                )
            if c.constant and c.n_interior > 0:
                raise ValueError(f"{c.name}: a constant curve has no interior knots.")
            if not 0 <= c.nleaves_min <= c.n_interior:
                raise ValueError(f"{c.name}: need 0 <= n_interior_min <= n_interior.")
            if (c.n_interior + 1) * self.min_log10_separation >= width:
                raise ValueError(
                    f"{c.name}: {c.n_interior} knots {self.min_log10_separation} dex apart do not "
                    f"fit in {self.f_min:g}-{self.f_max:g} Hz; lower min_log10_separation or n_interior."
                )
            if c.noise == "tm" and c.combination.startswith("armdiff") and not c.constant:
                warnings.warn(
                    f"{c.name}: the TM arm differences are not constrained within any frequency "
                    "band (see SPLINE_NOISE_MODEL.md); this curve is mostly set by its prior."
                )

    # ------------------------------------------------------------------ layout

    @property
    def n_par(self) -> int:
        """Amplitudes per noise type: 1 (symmetric) or 6 (asymmetric)."""
        return 1 if self.noise_symmetry == "symmetric" else 6

    @property
    def n_curves(self) -> int:
        return len(self.components)

    @property
    def n_knots_max(self) -> int:
        """Knots per curve after padding: the 2 edges + the most interior knots."""
        return 2 + max((c.n_interior for c in self.components), default=0)

    @property
    def mixing(self) -> np.ndarray:
        """``(2 n_par, n_curves)`` map from curves to per-MOSA deviations, rows [OMS | TM]."""
        M = np.zeros((2 * self.n_par, self.n_curves))
        vec = mixing_vectors()
        for j, c in enumerate(self.components):
            rows = slice(0, self.n_par) if c.noise == "oms" else slice(self.n_par, 2 * self.n_par)
            M[rows, j] = 1.0 if self.n_par == 1 else vec[c.combination]
        return M

    @property
    def edge_param_names(self) -> list:
        """LaTeX names of the edge amplitudes, in the order they sit in the ``psd`` branch."""
        out = []
        for c in self.components:
            label = rf"{{\rm {c.noise}, {c.combination.replace('_', ' ')}}}"
            if c.constant:
                out.append(rf"$c_{label}$")
            else:
                out += [rf"$e^{{\rm min}}_{label}$", rf"$e^{{\rm max}}_{label}$"]
        return out

    @property
    def edge_param_keys(self) -> list:
        """Plain names of the edge amplitudes (same order): ``<curve>_edge_min``/``_edge_max``,
        or ``<curve>_constant``."""
        return [
            f"{c.name}_{s}" for c in self.components for s in (("constant",) if c.constant else ("edge_min", "edge_max"))
        ]

    @property
    def n_edge_params(self) -> int:
        return sum(c.n_edge for c in self.components)

    @property
    def edge_columns(self) -> tuple:
        """``(i_min, i_max)``: column of each curve's ``f_min`` / ``f_max`` amplitude among the
        edge parameters (the same column for a constant curve)."""
        start = np.cumsum([0] + [c.n_edge for c in self.components])[:-1]
        stop = start + np.array([c.n_edge - 1 for c in self.components], dtype=int)
        return start, stop

    @property
    def branches(self) -> dict:
        """``{branch name: component}`` for the components with interior knots."""
        return {c.branch_name: c for c in self.components if c.n_interior > 0}

    @property
    def branch_names(self) -> list:
        return list(self.branches)

    @property
    def rj(self) -> bool:
        """Whether reversible jump changes the number of knots of any curve."""
        return any(c.varies for c in self.branches.values())

    @property
    def rj_branch_names(self) -> list:
        """Branches whose number of knots changes under reversible jump."""
        return [name for name, c in self.branches.items() if c.varies]

    @property
    def nleaves_max(self) -> dict:
        return {name: c.n_interior for name, c in self.branches.items()}

    @property
    def nleaves_min(self) -> dict:
        return {name: c.nleaves_min for name, c in self.branches.items()}

    @property
    def n_spline_params(self) -> int:
        """Length of the spline part of the flat row (edges + interior knots)."""
        return self.n_edge_params + sum(2 * c.n_interior for c in self.components)

    @property
    def n_sampled_params(self) -> int:
        """Sampled parameters with every knot active: the edges and all interior knots
        (e.g. for the dimension of the temperature ladder)."""
        return self.n_spline_params

    @property
    def injection(self) -> np.ndarray:
        """``psd`` branch value of the reference spectrum itself: all edges at zero."""
        return np.zeros(self.n_edge_params)

    def psd_priors(self):
        """``ProbDistContainer`` of the ``psd`` branch (the edge amplitudes)."""
        from eryn.prior import ProbDistContainer

        return ProbDistContainer(self.edge_priors())

    def to_dict(self) -> dict:
        """JSON-friendly description; :meth:`from_dict` rebuilds the model."""
        return dict(
            components=[dataclasses.asdict(c) for c in self.components],
            f_min=self.f_min,
            f_max=self.f_max,
            reference_levels=self.reference_levels.tolist(),
            noise_symmetry=self.noise_symmetry,
            min_log10_separation=self.min_log10_separation,
            interpolation=self.interpolation,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "SplineNoiseModel":
        """Inverse of :meth:`to_dict`."""
        d = dict(d)
        comps = [SplineComponent(**c) for c in d.pop("components")]
        return cls(comps, **d)

    def metadata(self) -> dict:
        """Run metadata of the noise model (``model_config`` of the PSD source)."""
        return dict(
            reference_levels=self.reference_levels.tolist(),
            spline_model=self.describe(),
            num_params=self.n_sampled_params,
            reversible_jump=self.rj,
            min_log10_separation=self.min_log10_separation,
            interpolation=self.interpolation,
        )

    # ------------------------------------------------------------------ priors

    def edge_priors(self) -> dict:
        """``{name: distribution}`` for the edge amplitudes: the ``psd`` branch priors."""
        out = {}
        names = iter(self.edge_param_names)
        for c in self.components:
            for _ in range(c.n_edge):
                out[next(names)] = c.amplitude_prior(c.edge_sigma)
        return out

    def branch_priors(self) -> dict:
        """``{branch: ProbDistContainer}`` for one interior knot ``(log10 f, amplitude)``.

        The positions are uniform over the band; the minimum separation and its
        normalisation are added by :meth:`knot_log_prior`.
        """
        from eryn.prior import ProbDistContainer, uniform_dist

        return {
            name: ProbDistContainer(
                {0: uniform_dist(self.log10_f_min, self.log10_f_max), 1: c.amplitude_prior(c.knot_sigma)}
            )
            for name, c in self.branches.items()
        }

    def log_separation_probability(self, n):
        """``log P_ok(n)``: probability that ``n`` knots uniform over the band keep every gap,
        edges included, at least ``min_log10_separation``.

        The ``n + 1`` gaps between the edges are uniform on a simplex, so
        ``P_ok(n) = (1 - (n + 1) s / W)^n`` with ``W`` the band width in log10 f (``-inf``
        when the knots do not fit).
        """
        n = np.asarray(n, dtype=np.float64)
        width = self.log10_f_max - self.log10_f_min
        room = 1.0 - (n + 1.0) * self.min_log10_separation / width
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(room > 0, n * np.log(np.where(room > 0, room, 1.0)), -np.inf)
        return np.where(n == 0, 0.0, out)

    def knot_log_prior(self, branch: str, coords, inds=None):
        """Separation part of the knot prior of ``branch``, per walker.

        The knot positions are uniform over the band and truncated to configurations with
        all gaps (edges included) at least ``min_log10_separation``. The per-knot uniform
        and amplitude priors are in :meth:`branch_priors`; this adds ``-inf`` for knots
        that are too close and ``-log P_ok(n)``, the normalisation of the truncated
        position prior for ``n`` active knots. Without it, reversible jump would see the
        prior on ``n`` multiplied by ``P_ok(n)`` (fewer knots favoured); with a fixed
        number of knots it is a constant.

        Args:
            branch: Spline branch name.
            coords: Knots, shape ``(..., nleaves, 2)``.
            inds: Active knots, shape ``(..., nleaves)``; all active if ``None``.

        Returns:
            Shape ``coords.shape[:-2]``.
        """
        coords = np.asarray(coords)
        pos = coords[..., 0]
        inds = np.ones(pos.shape, dtype=bool) if inds is None else np.asarray(inds, dtype=bool)
        lead = pos.shape[:-1]
        allpos = np.concatenate(
            [
                np.broadcast_to(self.log10_f_min, lead + (1,)),
                np.where(inds, pos, np.inf),
                np.broadcast_to(self.log10_f_max, lead + (1,)),
            ],
            axis=-1,
        )
        allpos = np.sort(allpos, axis=-1)
        with np.errstate(invalid="ignore"):  # inf - inf between inactive knots
            gaps = np.diff(allpos, axis=-1)
        too_close = np.any(np.isfinite(gaps) & (gaps < self.min_log10_separation), axis=-1)
        log_p_ok = self.log_separation_probability(inds.sum(axis=-1))
        return np.where(too_close | ~np.isfinite(log_p_ok), -np.inf, -log_p_ok)

    def draw_interior_knots(self, branch: str, size: tuple, n_active=None, rng=None):
        """Exact prior draws of the interior knots of ``branch``.

        ``n`` positions uniform on the band and at least ``s = min_log10_separation`` apart
        (edges included) are ``n`` sorted uniforms on the reduced width ``W - (n + 1) s``,
        with ``(i + 1) s`` added to the ``i``-th: this maps the uniform distribution onto the
        allowed set one to one, so no draw is rejected.

        Args:
            branch: Spline branch name.
            size: Leading shape, e.g. ``(ntemps, nwalkers)``.
            n_active: Number of active knots per row: an int, an array of shape ``size``, or
                ``None`` for every leaf (fixed number of knots). Inactive leaves get
                independent draws of the per-knot prior.
            rng: ``numpy.random.Generator``.

        Returns:
            ``(coords, inds)`` with shapes ``size + (n_interior, 2)`` and
            ``size + (n_interior,)``; the active knots are the first leaves, in random order.
        """
        rng = np.random.default_rng() if rng is None else rng
        c = self.branches[branch]
        L = c.n_interior
        num = int(np.prod(size))
        n_active = np.broadcast_to(L if n_active is None else np.asarray(n_active), tuple(size)).reshape(num)
        if np.any((n_active < c.nleaves_min) | (n_active > L)):
            raise ValueError(f"{branch}: n_active must be within [{c.nleaves_min}, {L}].")
        inds = np.arange(L)[None, :] < n_active[:, None]
        s = self.min_log10_separation
        width = self.log10_f_max - self.log10_f_min
        out = np.empty((num, L, 2))
        out[:, :, 0] = rng.uniform(self.log10_f_min, self.log10_f_max, (num, L))  # inactive leaves
        for n in np.unique(n_active):
            rows = np.flatnonzero(n_active == n)
            if n == 0:
                continue
            y = np.sort(rng.uniform(0.0, width - (n + 1) * s, (rows.size, n)), axis=1)
            pos = self.log10_f_min + y + s * np.arange(1, n + 1)
            out[rows, :n, 0] = rng.permuted(pos, axis=1)
        out[:, :, 1] = c.draw_amplitudes(rng, (num, L), c.knot_sigma)
        return out.reshape(tuple(size) + (L, 2)), inds.reshape(tuple(size) + (L,))

    def draw_n_active(self, branch: str, size: tuple, rng=None):
        """Number of active knots per row: ``n_interior`` for a fixed number, else a draw
        from the uniform prior on ``[n_interior_min, n_interior]``."""
        rng = np.random.default_rng() if rng is None else rng
        c = self.branches[branch]
        return rng.integers(c.nleaves_min, c.n_interior + 1, size=size)

    def gibbs_sampling_setup(self, ndim_psd: Optional[int] = None) -> list:
        """Eryn Gibbs blocks: OMS edges, TM edges, then one block per spline branch.

        Each block is small enough for the stretch move with a modest number of walkers.

        Args:
            ndim_psd: Width of the ``psd`` branch, checked against the edge count if given.
        """
        if ndim_psd is None:
            ndim_psd = self.n_edge_params
        if ndim_psd != self.n_edge_params:
            raise ValueError(f"psd branch has {ndim_psd} parameters, expected the {self.n_edge_params} edges.")
        blocks = []
        start = np.cumsum([0] + [c.n_edge for c in self.components])
        for noise in NOISE_TYPES:
            cols = [k for j, c in enumerate(self.components) if c.noise == noise for k in range(start[j], start[j + 1])]
            if cols:
                mask = np.zeros((1, ndim_psd), dtype=bool)
                mask[0, cols] = True
                blocks.append(("psd", mask))
        blocks += self.branch_names
        return blocks

    # ------------------------------------------------------------------ rows

    def flatten(self, psd_coords, branch_coords: dict, branch_inds: Optional[dict] = None):
        """Flat rows ``[levels | edges | interior knots]`` for the sensitivity backend.

        Args:
            psd_coords: ``psd`` branch coordinates (the edge amplitudes), shape
                ``(..., n_edge_params)``. The fixed reference levels are prepended here.
            branch_coords: ``{branch: coords}`` with coords of shape ``(..., n_interior, 2)``.
            branch_inds: ``{branch: inds}`` with shape ``(..., n_interior)``; inactive knots
                become NaN. ``None``: every knot is active.
        """
        xp = _array_module(psd_coords)
        lead = psd_coords.shape[:-1]
        if psd_coords.shape[-1] != self.n_edge_params:
            raise ValueError(
                f"The psd branch holds the {self.n_edge_params} edge amplitudes, "
                f"got {psd_coords.shape[-1]} parameters."
            )
        levels = xp.asarray(self.reference_levels, dtype=psd_coords.dtype)
        parts = [xp.broadcast_to(levels, lead + levels.shape), psd_coords]
        for name, c in self.branches.items():
            knots = xp.asarray(branch_coords[name], dtype=psd_coords.dtype)
            if branch_inds is not None:
                active = xp.asarray(branch_inds[name], dtype=bool)
                knots = xp.where(active[..., None], knots, xp.nan)
            parts.append(knots.reshape(lead + (2 * c.n_interior,)))
        return xp.concatenate(parts, axis=-1)

    def knots(self, spline_params, xp=None):
        """Knot positions and amplitudes from the spline part of the flat rows.

        Args:
            spline_params: Shape ``(n_spline_params,)`` or ``(num, n_spline_params)``.
                Inactive interior knots are NaN.

        Returns:
            ``(positions, amplitudes)``, each ``(n_curves, num, n_knots_max)``: log10 f and
            log10 deviation, sorted by position, NaN-padded at the end (the interpolant
            skips trailing NaNs).
        """
        if xp is None:
            xp = _array_module(spline_params)
        sp = xp.atleast_2d(xp.asarray(spline_params, dtype=xp.float64))
        if sp.shape[-1] != self.n_spline_params:
            raise ValueError(
                f"Expected {self.n_spline_params} spline parameters (edges + interior knots), "
                f"got {sp.shape[-1]}."
            )
        num, K = sp.shape[0], self.n_knots_max
        pos = xp.full((num, self.n_curves, K), xp.inf)
        amp = xp.full((num, self.n_curves, K), xp.nan)

        i_min, i_max = self.edge_columns
        pos[:, :, 0], amp[:, :, 0] = self.log10_f_min, sp[:, xp.asarray(i_min)]
        pos[:, :, 1], amp[:, :, 1] = self.log10_f_max, sp[:, xp.asarray(i_max)]

        off = self.n_edge_params
        for j, c in enumerate(self.components):
            n = c.n_interior
            if n == 0:
                continue
            block = sp[:, off : off + 2 * n].reshape(num, n, 2)
            pos[:, j, 2 : 2 + n] = block[:, :, 0]
            amp[:, j, 2 : 2 + n] = block[:, :, 1]
            off += 2 * n

        # sort by position; inactive knots (NaN) and padding (inf) go last and become NaN
        pos = xp.where(xp.isnan(pos), xp.inf, pos)
        order = xp.argsort(pos, axis=-1)
        pos = xp.take_along_axis(pos, order, axis=-1)
        amp = xp.take_along_axis(amp, order, axis=-1)
        pos[~xp.isfinite(pos)] = xp.nan
        amp[xp.isnan(pos)] = xp.nan
        return xp.moveaxis(pos, 1, 0).copy(), xp.moveaxis(amp, 1, 0).copy()

    def invalid_knots(self, positions):
        """``(num,)`` mask of rows whose knots are closer than ``min_log10_separation``
        or whose interior knots fall outside the band (already excluded by the prior;
        kept as a guard in the likelihood)."""
        xp = _array_module(positions)
        p = xp.where(xp.isnan(positions), xp.inf, positions)
        gaps = xp.diff(p, axis=-1)
        too_close = xp.any(xp.isfinite(gaps) & (gaps < self.min_log10_separation), axis=(0, 2))
        finite = xp.isfinite(p)
        outside = xp.any(finite & ((p < self.log10_f_min) | (p > self.log10_f_max)), axis=(0, 2))
        return too_close | outside

    def describe(self) -> str:
        """One line per component, for logs and notebooks."""
        levels = np.array2string(self.reference_levels, precision=2, separator=",")
        lines = [
            f"SplineNoiseModel({self.noise_symmetry}, reference levels {levels}, band "
            f"{self.f_min:g}-{self.f_max:g} Hz, {self.n_spline_params} spline parameters (max), "
            f"{self.interpolation} interpolation, knots >= {self.min_log10_separation:g} dex apart"
            + (", reversible jump)" if self.rj else ")")
        ]
        for c in self.components:
            prior = f"U(+-{c.amplitude_bound:g})" if c.amplitude_bound is not None else f"N(0,{c.knot_sigma:g})"
            if c.constant:
                knots = "constant"
            elif c.varies:
                knots = f"{c.nleaves_min}-{c.n_interior} interior knots"
            else:
                knots = f"{c.n_interior} interior knots"
            lines.append(
                f"  {c.name:18s} {knots}, amplitudes {prior}" + (f" (branch {c.branch_name})" if c.branch_name else "")
            )
        return "\n".join(lines)


def _array_module(arr):
    try:
        import cupy as cp

        if isinstance(arr, cp.ndarray):
            return cp
    except (ImportError, ModuleNotFoundError):
        pass
    return np


def default_spline_components(
    noise_symmetry: str = "asymmetric",
    n_interior: int = DEFAULT_N_INTERIOR,
    n_interior_min: Optional[int] = None,
    sigma: float = DEFAULT_AMPLITUDE_SIGMA,
    amplitude_bound: Optional[float] = None,
) -> list:
    """Curves of the spline noise model.

    The symmetric model has one OMS and one TM common curve. The asymmetric one has curves
    on all 6 OMS combinations and on the TM common and arm sums, plus a constant on each
    TM arm difference. TM cw-ccw is an exact null and stays at the reference.

    Args:
        noise_symmetry: ``"symmetric"`` or ``"asymmetric"``.
        n_interior: Interior knots per curve; the maximum with reversible jump.
        n_interior_min: ``None`` for a fixed number of knots, else the minimum number of
            knots per curve (reversible jump).
        sigma: Prior standard deviation of every amplitude [log10 PSD].
        amplitude_bound: If given, ``U(-amplitude_bound, amplitude_bound)`` amplitude priors
            instead of ``N(0, sigma)``.
    """
    common = dict(edge_sigma=sigma, knot_sigma=sigma, amplitude_bound=amplitude_bound)

    def curve(noise, comb):
        return SplineComponent(noise, comb, n_interior=n_interior, n_interior_min=n_interior_min, **common)

    if noise_symmetry == "symmetric":
        return [curve("oms", "common"), curve("tm", "common")]
    comps = [curve("oms", c) for c in COMBINATIONS]
    comps += [curve("tm", c) for c in ("common", "armsum_12_23", "armsum_23_31")]
    comps += [SplineComponent("tm", c, constant=True, **common) for c in ("armdiff_12_23", "armdiff_23_31")]
    return comps


def make_spline_noise_model(
    f_min: float,
    f_max: float,
    oms_level,
    tm_level,
    noise_symmetry: str = "asymmetric",
    rj: bool = False,
    n_interior: int = DEFAULT_N_INTERIOR,
    rj_n_min: int = DEFAULT_RJ_N_MIN,
    rj_n_max: int = DEFAULT_RJ_N_MAX,
    sigma: float = DEFAULT_AMPLITUDE_SIGMA,
    min_separation: float = DEFAULT_MIN_SEPARATION,
    interpolation: str = DEFAULT_INTERPOLATION,
    amplitude_bound: Optional[float] = None,
) -> SplineNoiseModel:
    """The spline noise model of a run, with the default curves
    (:func:`default_spline_components`).

    Args:
        f_min, f_max: Data band [Hz].
        oms_level, tm_level: Reference (injected) OMS and TM ASDs: one value, or one per
            MOSA (``MOSA_NAMES`` order; all equal for the symmetric model).
        noise_symmetry: ``"symmetric"`` or ``"asymmetric"``.
        rj: Reversible jump on the number of interior knots, in ``[rj_n_min, rj_n_max]``;
            otherwise ``n_interior`` knots per curve, all active.
        sigma: Prior standard deviation of every amplitude [dex].
        min_separation: Minimum distance between adjacent knots, edges included [dex].
        interpolation: ``"linear"`` or ``"cubic"``.
        amplitude_bound: If given, ``U(-amplitude_bound, amplitude_bound)`` amplitude priors
            instead of ``N(0, sigma)``.
    """
    n_par = 1 if noise_symmetry == "symmetric" else len(MOSA_NAMES)
    levels = []
    for name, level in (("oms_level", oms_level), ("tm_level", tm_level)):
        level = np.asarray(level, dtype=np.float64).ravel()
        if n_par == 1 and np.all(level == level[0]):
            level = level[:1]  # equal per-MOSA levels for the symmetric model
        if level.size == 1:
            level = np.full(n_par, level[0])
        if level.size != n_par:
            raise ValueError(f"{name}: need 1 or {n_par} values, got {level.size}.")
        levels.append(level)
    comps = default_spline_components(
        noise_symmetry,
        n_interior=rj_n_max if rj else n_interior,
        n_interior_min=rj_n_min if rj else None,
        sigma=sigma,
        amplitude_bound=amplitude_bound,
    )
    return SplineNoiseModel(
        comps,
        f_min,
        f_max,
        np.concatenate(levels),
        noise_symmetry,
        min_log10_separation=min_separation,
        interpolation=interpolation,
    )
