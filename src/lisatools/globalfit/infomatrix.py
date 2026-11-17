"""Information-matrix covariances for one source branch of the global fit.

:class:`InfoMatrixGenerator` is shared by the two places that need a local covariance of a source: the
starting scatter around a search point and the information-matrix proposal of the sampler. It holds
what persists between computations (the prior precision that regularises the inverse, and the
derivative steps of each leaf); everything that differs between the two uses is a call argument.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Hashable, Optional

import numpy as np
from eryn.utils import TransformContainer

from .. import domains
from ..diagnostic import _place_on_grid, covariance, estimate_eps, info_matrix, inner_product

# the waveform-distance cap of :class:`InfoMatrixGenerator`
CAP_TOLERANCE = 1.5  # a direction within ``CAP_TOLERANCE * distance_cap`` at one sigma is left alone
CAP_FLOOR = 1e-4  # smallest extent a capped direction can get
CAP_HALVINGS = 8  # bisection steps in log(extent) between the floor and 1
CAP_EDGE_HALVINGS = 30  # bisection steps for the last point inside the prior along an axis


def _to_host(arr) -> np.ndarray:
    return np.asarray(arr.get() if hasattr(arr, "get") else arr, dtype=float)


@dataclass(frozen=True)
class ProposalBasis:
    """A basis other than the sampling one in which the information matrix (and its Gaussian) is taken.

    Attributes:
        name: Name of the basis (logs, diagnostics).
        transform: eryn ``TransformContainer`` with ``both_transforms`` from the sampling basis to this one
            and ``both_inverse_transforms`` back. A bijection: same number of columns, no fill values.
        log_jacobian: ``log_jacobian(x) -> log|det d(this basis)/d(sampling basis)|`` for rows ``x`` of
            sampling coordinates, shape ``(N, ndim) -> (N,)``.
    """

    name: str
    transform: Any
    log_jacobian: Callable

    def from_sampling(self, x: np.ndarray) -> np.ndarray:
        """Sampling coordinates (1D or rows) in this basis."""
        x = np.asarray(x, dtype=float)
        return self.transform.both_transforms(np.atleast_2d(x)).reshape(x.shape)

    def to_sampling(self, y: np.ndarray) -> np.ndarray:
        """Coordinates of this basis (1D or rows) in the sampling basis."""
        y = np.asarray(y, dtype=float)
        return self.transform.both_inverse_transforms(np.atleast_2d(y)).reshape(y.shape)


def _chirp_forward(logM, q, s1z, s2z):
    return logM + 0.6 * np.log(q) - 1.2 * np.log1p(q), np.log(q), (q * s1z + s2z) / (1.0 + q), 0.5 * (s1z - s2z)


def _chirp_inverse(log_mc, log_q, chi_eff, chi_a):
    q = np.exp(log_q)
    return (
        log_mc - 0.6 * log_q + 1.2 * np.log1p(q),
        q,
        chi_eff + 2.0 * chi_a / (1.0 + q),
        chi_eff - 2.0 * q * chi_a / (1.0 + q),
    )


def _chirp_log_jacobian(x, q_index):
    return -np.log(2.0 * np.asarray(x, dtype=float)[:, q_index])


CHIRP_BASIS_COLUMNS = ("logM", "q", "s1z", "s2z")


def chirp_basis(names: list[str]) -> ProposalBasis:
    """Chirp-mass / effective-spin basis of a binary sampled in ``(logM, q, s1z, s2z, ...)``, ``q = m1 / m2``.

    The four columns become ``log Mc = logM + 0.6 log q - 1.2 log(1 + q)``, ``log q``,
    ``chi_eff = (q s1z + s2z) / (1 + q)`` and ``chi_a = (s1z - s2z) / 2``, in the places of ``logM``, ``q``,
    ``s1z`` and ``s2z``; every other column is unchanged. ``log|det| = -log(2 q)``.

    Args:
        names: Names of the sampling basis, in order.

    Raises:
        ValueError: If one of ``logM``, ``q``, ``s1z``, ``s2z`` is not in ``names``.
    """
    names = list(names)
    missing = [name for name in CHIRP_BASIS_COLUMNS if name not in names]
    if missing:
        raise ValueError(f"The chirp basis needs the columns {missing}, which the basis {names} lacks.")
    transform = TransformContainer(
        input_basis=names,
        output_basis=names,
        parameter_transforms={CHIRP_BASIS_COLUMNS: _chirp_forward},
        inverse_parameter_transforms={CHIRP_BASIS_COLUMNS: _chirp_inverse},
    )
    return ProposalBasis("chirp", transform, partial(_chirp_log_jacobian, q_index=names.index("q")))


# name -> builder ``fn(names of the sampling basis) -> ProposalBasis`` (the l2d ``info_matrix.proposal_basis``)
PROPOSAL_BASES = {"chirp": chirp_basis}


class _PriorSupport:
    """The support of a prior, as the ``prior`` of :func:`lisatools.diagnostic.estimate_eps`.

    ``logpdf`` takes points of ``basis`` (the sampling basis if ``None``), maps them to the sampling basis and
    wraps the periodic columns before asking the prior: a derivative stencil that crosses the edge of a
    periodic parameter is still inside its support. The periodic columns are read from the prior
    (``get_periodic`` and ``get_input_basis``, as the erebor priors have them; periods start at 0); a prior
    without them has none.
    """

    def __init__(self, prior: Any, basis: Optional[ProposalBasis] = None):
        self.prior, self.basis = prior, basis
        self.periodic = {}
        try:
            names = list(prior.get_input_basis())
            self.periodic = {names.index(name): float(period) for name, period in (prior.get_periodic() or {}).items()}
        except (AttributeError, TypeError, ValueError):
            pass

    def logpdf(self, points: np.ndarray) -> np.ndarray:
        points = np.array(points, dtype=float)
        if self.basis is not None:
            points = self.basis.to_sampling(points)
        for column, period in self.periodic.items():
            points[:, column] = np.mod(points[:, column], period)
        return self.prior.logpdf(points)


class InfoMatrixGenerator:
    """``gen(point, waveform, psd, basis_settings) -> covariance`` from the information matrix at ``point``.

    The covariance is the inverse of ``M + diag(prior_precision)`` (see
    :func:`lisatools.diagnostic.covariance`): finite also where the information matrix ``M`` is singular,
    and no parameter is wider than its prior.

    Cached on the object:

    * ``prior_precision``: ``1 / variance`` of ``num_prior_draws`` prior draws per parameter, computed
      once (on first use). This works for any prior with an ``rvs``, not only uniform ones.
    * the derivative steps of each leaf (from :func:`lisatools.diagnostic.estimate_eps`, the expensive
      part: ``len(ladder)`` derivatives per parameter), with the number of calls since they were
      estimated. ``leaf=None`` is a leaf like any other. The ladder steps whose stencil leaves the support
      of ``prior`` are skipped (near a parameter edge the transform to the waveform parameters gives NaN,
      which can crash a GPU waveform instead of raising).

    The waveform, the PSD and the grid are call arguments and are never stored: they differ between the
    starting scatter and the proposal (and between calls of the proposal). The object therefore survives
    the one deep copy the global fit makes of its settings with cache and prior precision intact, as
    long as ``prior`` itself can be deep-copied.

    **One prior per leaf.** ``prior`` may be a list with one prior per leaf, indexed by the integer ``leaf``
    of the call (the per-source priors of a block with ``proposal_priors``). The prior precision, the support
    the derivative steps must stay in and the support of the distance cap are then those of the leaf's own
    prior. With a single prior every leaf shares it.

    **Proposal basis.** With a ``proposal_basis`` (:class:`ProposalBasis`), a call with
    ``in_proposal_basis=True`` takes the matrix in that basis: the point is mapped to it, the derivatives are
    taken along its coordinates (with their own cached steps) and the prior precision is that of the prior
    draws mapped to it. The covariance returned is in the proposal basis. Calls without the flag are in the
    sampling basis, whatever ``proposal_basis`` is.

    **Distance cap.** A Gaussian from the information matrix is too wide along a direction in which the
    waveform is not linear over one sigma. With a ``distance_cap`` (the target, 1 is the natural value), a
    call with ``cap=True`` measures the waveform distance ``<dh|dh>``, ``dh = h(point + x) - h(point)``, at
    both one-sigma ends ``x = +-delta_k`` of every principal axis of ``M + diag(prior_precision)`` and
    shortens the axes on which it exceeds the target (or on which one sigma leaves the prior) to the extent
    where it meets it. The covariances then come from the capped information matrix (see :meth:`_cap`).
    Calls without the flag, and every call of a generator without ``distance_cap``, are not capped.

    Args:
        prior: Prior of the branch in the basis of ``point``: an object with
            ``rvs(size=n) -> (n, ndim)`` array (e.g. an eryn ``ProbDistContainer``) and, for the skipped
            steps and the distance cap, ``logpdf``.
        more_accurate: Passed to :func:`~lisatools.diagnostic.estimate_eps` and
            :func:`~lisatools.diagnostic.info_matrix` (five-point stencil if ``True``).
        steps: Step ladder(s) passed to :func:`~lisatools.diagnostic.estimate_eps`.
        eps_every: Re-estimate the steps of a leaf once they have been used for this many calls
            (``eps_every=2``: estimated on calls 1, 3, 5, ... of that leaf). ``None``: estimate once per
            leaf and never again (until :meth:`clear`).
        num_prior_draws: Number of prior draws for the prior precision.
        seed: Seed of those draws.
        proposal_basis: :class:`ProposalBasis` of the calls with ``in_proposal_basis=True``. ``None``: there
            is none and those calls raise.
        distance_cap: Target waveform distance ``<dh|dh>`` at one sigma of the calls with ``cap=True``.
            ``None``: there is no cap, whatever the call asks.

    Attributes:
        last: Diagnostics of the last successful call (empty before it): ``leaf``, ``basis`` (name of the
            proposal basis, ``None`` for the sampling basis), ``point`` (in that basis), ``eps``
            (steps used), ``estimated`` (whether the steps were estimated in that call),
            ``info_matrix`` (``M``), ``condition`` (largest / smallest eigenvalue of
            ``M + diag(prior_precision)`` scaled to unit diagonal), ``time`` (seconds) and, of the
            distance cap, ``extents`` (the fraction ``r_k`` of its one sigma each principal axis keeps, in
            ascending order of the eigenvalues; all ones when the cap did not run), ``capped`` (number of
            ``r_k < 1``) and ``cap_waveforms`` (waveform calls the cap spent). ``info_matrix`` and
            ``condition`` are those of the matrix before the cap.
    """

    def __init__(
        self,
        prior: Any,
        more_accurate: bool = True,
        steps: Optional[np.ndarray | list] = None,
        eps_every: Optional[int] = None,
        num_prior_draws: int = 10000,
        seed: Optional[int] = 0,
        proposal_basis: Optional[ProposalBasis] = None,
        distance_cap: Optional[float] = None,
    ):
        if distance_cap is not None and not distance_cap > 0.0:
            raise ValueError(f"InfoMatrixGenerator: distance_cap must be positive or None; got {distance_cap}.")
        self.distance_cap = distance_cap
        self.prior = prior
        self.more_accurate = more_accurate
        self.steps = steps
        self.eps_every = eps_every
        self.num_prior_draws = num_prior_draws
        self.seed = seed
        self.proposal_basis = proposal_basis
        # leaf (None: the shared prior) -> (precision in the sampling basis, in the proposal basis or None)
        self._prior_precisions: dict = {}
        # leaf -> [steps, number of calls since they were estimated]; one cache per basis
        self._steps_cache: dict = {}
        self._steps_cache_proposal: dict = {}
        self.last: dict = {}

    @staticmethod
    def _precision_of(draws: np.ndarray, basis: str) -> np.ndarray:
        variance = draws.var(axis=0)
        if not np.all(np.isfinite(variance)) or np.any(variance <= 0.0):
            raise ValueError(
                f"The prior draws have no finite, positive variance per parameter in the {basis} basis: {variance}."
            )
        return 1.0 / variance

    @property
    def per_leaf_priors(self) -> bool:
        """Whether ``prior`` is a list with one prior per leaf."""
        return isinstance(self.prior, (list, tuple))

    def prior_of(self, leaf: Optional[Hashable] = None) -> Any:
        """The prior of ``leaf``: its own with one prior per leaf, the shared one otherwise."""
        if not self.per_leaf_priors:
            return self.prior
        try:
            return self.prior[leaf]
        except (TypeError, IndexError):
            raise ValueError(
                f"InfoMatrixGenerator: with one prior per leaf ({len(self.prior)} priors) the leaf must be the "
                f"index of its prior; got {leaf!r}."
            ) from None

    @property
    def prior_precision(self) -> np.ndarray:
        """``1 / variance`` of the prior draws, one entry per parameter (computed on first access).

        Of the shared prior; with one prior per leaf use :meth:`prior_precisions`.
        """
        return self.prior_precisions()[0]

    @property
    def prior_precision_proposal(self) -> np.ndarray:
        """``1 / variance`` of the same prior draws mapped to the proposal basis (shared prior)."""
        if self.proposal_basis is None:
            raise ValueError("InfoMatrixGenerator: there is no proposal_basis.")
        return self.prior_precisions()[1]

    def prior_precisions(self, leaf: Optional[Hashable] = None) -> tuple:
        """``(sampling basis, proposal basis)`` prior precision of ``leaf``'s prior, cached.

        The second entry is ``None`` without a ``proposal_basis``. With a single prior ``leaf`` is ignored.
        """
        key = leaf if self.per_leaf_priors else None
        if key not in self._prior_precisions:
            prior = self.prior_of(leaf)
            # eryn's distributions draw from the global numpy generator: seed it for a reproducible
            # draw, then put the caller's state back so the run's random stream is not disturbed.
            # (A prior drawing on the GPU uses cupy's generator, which is neither seeded nor restored.)
            state = np.random.get_state()
            try:
                np.random.seed(self.seed)
                draws = prior.rvs(size=(self.num_prior_draws,))
            finally:
                np.random.set_state(state)
            draws = _to_host(draws)
            draws = draws.reshape(-1, draws.shape[-1])
            precision = self._precision_of(draws, "sampling")
            proposal = None
            if self.proposal_basis is not None:
                proposal = self._precision_of(self.proposal_basis.from_sampling(draws), self.proposal_basis.name)
            self._prior_precisions[key] = (precision, proposal)
        return self._prior_precisions[key]

    def clear(self, leaf: Optional[Hashable] = None) -> None:
        """Drop the cached steps of ``leaf`` (in both bases); of all leaves when called without argument.

        (So the steps cached under the key ``None`` are dropped with all the others.)
        """
        for cache in (self._steps_cache, self._steps_cache_proposal):
            if leaf is None:
                cache.clear()
            else:
                cache.pop(leaf, None)

    def __call__(
        self,
        point: np.ndarray | list,
        waveform: callable,
        psd: Any,
        basis_settings: Any,
        leaf: Optional[Hashable] = None,
        betas: Optional[np.ndarray | list] = None,
        parameter_transforms: Any = None,
        in_proposal_basis: bool = False,
        cap: bool = False,
    ) -> np.ndarray:
        """Covariance at ``point``.

        Args:
            point: Parameters of the source, 1D, in the basis of the prior (the sampling basis), also with
                ``in_proposal_basis``.
            waveform: ``waveform(*params) -> DomainBase`` (after ``parameter_transforms``, if given).
            psd: Sensitivity matrix for the inner products, on ``basis_settings``.
            basis_settings: Common grid for the stencil points (see
                :func:`~lisatools.diagnostic.dh_dlambda`). ``None``: the waveform's own settings.
            leaf: Key of the step cache (any hashable).
            betas: Inverse temperatures. If given, one covariance per entry, from ``beta * M``.
            parameter_transforms: eryn ``TransformContainer`` from the basis of ``point`` to the
                waveform's parameters.
            in_proposal_basis: Take the matrix, and return the covariance, in ``proposal_basis`` (the point
                in that basis is ``proposal_basis.from_sampling(point)``).
            cap: Cap the directions that are too wide by waveform distance (see the class and :meth:`_cap`).
                Without a ``distance_cap`` on the generator this does nothing.

        Returns:
            Covariance with shape ``(ndim, ndim)``; ``(len(betas), ndim, ndim)`` with ``betas``. In the
            sampling basis, or in the proposal basis with ``in_proposal_basis``.

        Raises:
            ValueError: If the steps cannot be estimated or the information matrix is not finite. The
                message names the leaf and the point; no steps are cached for that call. Also if
                ``in_proposal_basis`` is asked of a generator without ``proposal_basis``, and if the
                cap cannot evaluate the waveform at ``point``.
        """
        start = time.perf_counter()
        point = np.array(_to_host(point), dtype=float)

        if in_proposal_basis:
            basis = self.proposal_basis
            if basis is None:
                raise ValueError("InfoMatrixGenerator: in_proposal_basis needs a proposal_basis.")
            prior_precision, steps_cache = self.prior_precisions(leaf)[1], self._steps_cache_proposal
            point = basis.from_sampling(point)
            sampling_waveform, to_waveform = waveform, parameter_transforms

            def waveform_parameters(proposal_params):
                params = basis.to_sampling(np.array(proposal_params, dtype=float))
                if to_waveform is not None:
                    params = to_waveform.both_transforms(params[None, :])[0]
                return params

            def waveform(*proposal_params):
                return sampling_waveform(*waveform_parameters(proposal_params))

            parameter_transforms = None
        else:
            basis = None
            prior_precision, steps_cache = self.prior_precisions(leaf)[0], self._steps_cache
            sampling_waveform, to_waveform = waveform, parameter_transforms

            def waveform_parameters(params):
                return params if to_waveform is None else to_waveform.both_transforms(params[None, :])[0]

        kwargs = dict(inner_product_kwargs=dict(psd=psd), more_accurate=self.more_accurate)
        if basis_settings is not None:
            kwargs["basis_settings"] = basis_settings
        if parameter_transforms is not None:
            kwargs["parameter_transforms"] = parameter_transforms

        cached = steps_cache.get(leaf)
        estimated = cached is None or (self.eps_every is not None and cached[1] >= self.eps_every)
        if estimated:
            try:
                support = _PriorSupport(self.prior_of(leaf), basis)
                eps = estimate_eps(waveform, point, steps=self.steps, prior=support, **kwargs)
            except ValueError as error:
                raise ValueError(
                    f"InfoMatrixGenerator: the steps of leaf {leaf} could not be estimated at point "
                    f"{point.tolist()}: {error}"
                ) from error
        else:
            eps = cached[0]

        info = info_matrix(eps, waveform, point, **kwargs)
        if not np.all(np.isfinite(info)):
            raise ValueError(
                f"InfoMatrixGenerator: the information matrix of leaf {leaf} is not finite at point "
                f"{point.tolist()} (steps {np.asarray(eps).tolist()}):\n{info}"
            )

        if estimated:
            cached = steps_cache[leaf] = [eps, 0]
        cached[1] += 1

        capped_info, extents, cap_waveforms = info, np.ones(len(point)), 0
        if cap and self.distance_cap is not None:
            capped_info, extents, cap_waveforms = self._cap(
                info, prior_precision, point, sampling_waveform, waveform_parameters, psd, basis_settings, basis, leaf
            )

        if betas is None:
            cov = covariance(info_mat=capped_info, prior_precision=prior_precision)
        else:
            cov = np.array(
                [
                    covariance(info_mat=beta * capped_info, prior_precision=prior_precision)
                    for beta in np.atleast_1d(_to_host(betas))
                ]
            )

        regularised = info + np.diag(prior_precision)
        scale = np.sqrt(np.diag(regularised))
        eigenvalues = np.linalg.eigvalsh(regularised / np.outer(scale, scale))
        condition = float(eigenvalues[-1] / eigenvalues[0]) if eigenvalues[0] > 0.0 else np.inf

        self.last = dict(
            leaf=leaf,
            basis=None if basis is None else basis.name,
            point=point,
            eps=eps.copy(),
            estimated=estimated,
            info_matrix=info,
            condition=condition,
            time=time.perf_counter() - start,
            extents=extents,
            capped=int(np.sum(extents < 1.0)),
            cap_waveforms=cap_waveforms,
        )
        return cov

    def _cap(
        self,
        info: np.ndarray,
        prior_precision: np.ndarray,
        point: np.ndarray,
        waveform: Callable,
        waveform_parameters: Callable,
        psd: Any,
        basis_settings: Any,
        basis: Optional[ProposalBasis],
        leaf: Optional[Hashable],
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Shorten the principal axes along which one sigma changes the waveform by more than ``distance_cap``.

        With ``A = info + diag(prior_precision)``, ``s = sqrt(diag A)`` and the eigen-decomposition
        ``A / (s s^T) = V diag(lambda) V^T``, the one-sigma step along axis ``k`` is
        ``delta_k = v_k / s / sqrt(lambda_k)``. The waveform distance is
        ``d2(x) = <h(point + x) - h(point) | h(point + x) - h(point)>``; it is infinite where ``point + x``
        has a waveform parameter that is not finite, where the waveform raises or where the distance is not
        finite.

        A prior edge does not shorten an axis: the seeds that land outside the prior are redrawn anyway. An
        end ``+-r delta_k`` that is outside the support of the leaf's prior is measured at the last point
        inside it along the axis (found by bisection on the prior alone, without waveform calls); a side on
        which the prior ends at the point itself is not tested and the other side decides. An axis with no
        testable side is left alone.

        Per axis: if ``d2(+delta_k)`` and ``d2(-delta_k)`` are both at most ``CAP_TOLERANCE * distance_cap``
        the axis is left alone (extent ``r_k = 1``, two waveforms). Otherwise ``r_k`` is the largest extent a
        bisection in ``log r`` between ``CAP_FLOOR`` and 1 (``CAP_HALVINGS`` halvings) finds with
        ``max(d2(+r delta_k), d2(-r delta_k)) <= distance_cap``; the floor itself is not measured and is
        what an axis gets on which no measured extent passes. An axis with ``lambda_k <= 0`` has no one sigma
        and is left alone, with a warning.

        The capped precision is ``A_c = S V diag(lambda_k / r_k^2) V^T S``, ``S = diag(s)``: the same axes,
        the capped ones shortened to ``r_k`` sigma. The capped information matrix is
        ``A_c - diag(prior_precision)``, symmetrised; it is ``info`` itself when no axis is capped.

        Args:
            info: Information matrix ``M`` at ``point``.
            prior_precision: Prior precision of the basis of ``point``.
            point: Point in the basis in use (sampling or proposal).
            waveform: The caller's waveform, ``waveform(*waveform parameters)``.
            waveform_parameters: ``fn(point of the basis in use, 1D) -> waveform parameters``.
            psd: Sensitivity matrix of the inner product.
            basis_settings: Common grid of the waveforms, or ``None``.
            basis: Proposal basis of ``point`` (``None``: the sampling basis), for the prior support.
            leaf: Leaf of the call, for the error message.

        Returns:
            ``(capped information matrix, extents r_k, number of waveform calls)``. One call is the
            waveform at ``point``.

        Raises:
            ValueError: If the waveform at ``point`` fails or is not finite.
        """
        target = float(self.distance_cap)
        support = _PriorSupport(self.prior_of(leaf), basis)
        calls = 0

        def signal(x):
            nonlocal calls
            params = waveform_parameters(x)
            if not np.all(np.isfinite(np.asarray(params, dtype=float))):
                raise ValueError("a waveform parameter is not finite")
            calls += 1
            h = waveform(*params)
            if isinstance(h, domains.DomainBase):
                return h if basis_settings is None else _place_on_grid(h, basis_settings)
            return h[None, :] if h.ndim == 1 else h  # a plain array, as info_matrix takes them too

        def squared_norm(a, b=None):
            if b is not None and isinstance(a, domains.DomainBase):
                a = a.settings.associated_class(a.arr - b.arr, a.settings)
            elif b is not None:
                length = min(a.shape[1], b.shape[1])
                a = a[:, :length] - b[:, :length]
            return float(_to_host(inner_product(a, a, psd=psd)))

        try:
            reference = signal(point)
            if not np.isfinite(squared_norm(reference)):
                raise ValueError("the waveform is not finite")
        except Exception as error:
            raise ValueError(
                f"InfoMatrixGenerator: the distance cap of leaf {leaf} has no waveform at point "
                f"{point.tolist()}: {error}"
            ) from error

        def inside(x):
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return bool(np.all(np.isfinite(_to_host(support.logpdf(x[None, :])))))

        def end(offset):
            """``offset`` if ``point + offset`` is inside the prior, else the last offset along it that is
            (``None`` if the prior ends at the point on that side)."""
            if inside(point + offset):
                return offset
            low, high = 0.0, 1.0
            for _ in range(CAP_EDGE_HALVINGS):
                middle = 0.5 * (low + high)
                low, high = (middle, high) if inside(point + middle * offset) else (low, middle)
            return low * offset if low > 0.0 else None

        def distance(offset):
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    d2 = squared_norm(signal(point + offset), reference)
                except Exception:
                    return np.inf
            return d2 if np.isfinite(d2) else np.inf

        def within(offset, limit):
            # a side that leaves the prior is measured at its last point inside; a side with no point inside
            # is not tested (the other side decides; an axis with neither is left alone)
            for side in (offset, -offset):
                side = end(side)
                if side is not None and distance(side) > limit:
                    return False
            return True

        regularised = info + np.diag(prior_precision)
        scale = np.sqrt(np.diag(regularised))
        eigenvalues, eigenvectors = np.linalg.eigh(regularised / np.outer(scale, scale))
        extents = np.ones(len(point))
        for k, eigenvalue in enumerate(eigenvalues):
            if not eigenvalue > 0.0:
                warnings.warn(
                    f"InfoMatrixGenerator: leaf {leaf}: direction {k} of the distance cap has the eigenvalue "
                    f"{eigenvalue}, not positive; it is not capped."
                )
                continue
            delta = eigenvectors[:, k] / scale / np.sqrt(eigenvalue)
            if within(delta, CAP_TOLERANCE * target):
                continue
            low, high = np.log(CAP_FLOOR), 0.0  # low: the floor, or an extent that was measured and passed
            for _ in range(CAP_HALVINGS):
                middle = 0.5 * (low + high)
                if within(np.exp(middle) * delta, target):
                    low = middle
                else:
                    high = middle
            extents[k] = np.exp(low)

        if np.all(extents == 1.0):
            return info, extents, calls
        capped = (eigenvectors * (eigenvalues / extents**2)) @ eigenvectors.T * np.outer(scale, scale)
        capped -= np.diag(prior_precision)
        return 0.5 * (capped + capped.T), extents, calls
