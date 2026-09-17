# -*- coding: utf-8 -*-
"""Per-leaf eigen proposal tables for the single-source PE moves.

Builds the ``(axes, sigmas)`` tables that :class:`eryn.moves.EigenAxisMove`
consumes, from an information matrix in the SAMPLING basis. Two builders,
one shared post-processing pipeline:

* :func:`eigen_table_from_ll` — likelihood second differences via
  :func:`lisatools.info_matrix_ll.information_matrix_from_ll`. The right
  route when the likelihood is cheap and batched (SOBBH chunked scoring).
* :func:`eigen_table_from_waveform` — the Gram form
  ``<d_i h | d_j h>`` via the general waveform machinery
  :func:`lisatools.diagnostic.info_matrix` (positive semi-definite by
  construction, ``parameter_transforms`` keeps the derivatives in the
  sampling basis). The right route when one likelihood row is expensive
  but waveform builds are not (MBH, EMRI).

Pipeline (the GB lesson, ``gb_prior_box_scales``: a relative eigen floor is
not scale invariant): whiten the matrix by the prior box widths, run the
generic :func:`eryn.moves.eigenaxis.eigen_axis_set` (whitened
``sigma_max=1`` = one prior width), map the axes back to the sampling
basis with their curvature widths, and cap each width by
:func:`eryn.moves.eigenaxis.axis_prior_bounds`.

Any failure — a non-finite likelihood, a raising engine, a broken prior —
degrades to the identity-axes / 1%-of-prior-width fallback with a logged
warning. A refresh must never crash the sampler.
"""

import logging

import numpy as np

from eryn.moves.eigenaxis import (
    axis_prior_bounds,
    eigen_axis_set,
    prior_box_scales,
)

from ...diagnostic import info_matrix
from ...info_matrix_ll import information_matrix_from_ll

__all__ = [
    "prior_box_widths",
    "eigen_table_from_ll",
    "eigen_tables_from_ll_batch",
    "eigen_table_from_waveform",
]

logger = logging.getLogger(__name__)


def _prior_entries(prob_dist_container):
    """``[(column indices, distribution), ...]``, key-spelling agnostic.

    Read the container's PARSED ``priors`` list — eryn's
    :class:`~eryn.priors.probdist.ProbDistContainer` normalises every key
    spelling it accepts (``int``, ``str``, and tuples of either) into
    ``[column_index_array, dist]`` entries there, so this is the only view
    that sees a string-keyed container's columns at all.

    ``priors_in`` is the fallback for duck-typed containers that carry only
    the raw mapping; it can resolve INTEGER keys only, because a string key
    carries no column index outside the parser.
    """
    parsed = getattr(prob_dist_container, "priors", None)
    if parsed is not None:
        return [(np.atleast_1d(np.asarray(inds)), dist) for inds, dist in parsed]
    return [
        (np.atleast_1d(np.asarray(key)), dist)
        for key, dist in prob_dist_container.priors_in.items()
        if isinstance(key, (int, np.integer))
    ]


def prior_box_widths(prob_dist_container, ndim):
    """Per-column prior box widths from an eryn prior container.

    Mirrors the GB move's ``_eigen_axis_widths`` reader: eryn's uniform
    exposes ``minimum``/``maximum`` (the ``min_val``/``max_val`` spelling
    belongs to other distributions — try both rather than silently falling
    back to unit widths). Columns without a scalar distribution, and any
    reader failure, fall back to width 1.0 so the table build degrades
    instead of crashing.

    **Read the columns through** :func:`_prior_entries`, never off
    ``priors_in`` keys. Half the stock branches spell their prior dict with
    LABELS rather than column indices — ``psd`` (``psd_prior_dict``:
    ``r"$S_{\\rm oms}$"``), ``mbh`` (``"logM"``, ...), ``emri`` and ``sobbh``
    (their ``input_basis`` names) — and an index-keyed reader silently
    resolves NONE of those columns. Unit widths there are not a harmless
    default: the psd levels live at ~1e-11 / ~1e-14 in boxes 1.9e-10 /
    2.0e-13 wide, so width 1.0 overstates the box by 5e9x / 5e12x, which
    puts every finite-difference corner AND every capped eigen step far
    outside the prior (2026-09-16 one-walker defect: psd in-model
    acceptance exactly 0, "All points entering likelihood have a log prior
    of minus inf" on every repeat).

    A column the container DOES cover but whose distribution exposes no
    FINITE, positive box (eryn distributions default ``minimum`` /
    ``maximum`` to -inf / +inf, and :func:`prior_box_scales` maps such a
    width back to 1.0) is reported once per ``(ndim, columns)`` rather than
    defaulting silently — that silence is what hid the defect above.
    Columns the container does not cover at all (fixed / per-leaf-filled
    parameters) keep width 1.0 quietly, as before. A reader failure keeps
    the columns read up to that point (unit widths elsewhere) and warns.
    """
    lo = np.zeros(ndim)
    hi = np.ones(ndim)
    covered = []
    unread = []
    try:
        for cols, dist in _prior_entries(prob_dist_container):
            cols_all = [int(c) for c in cols.ravel()]
            keep = np.array([0 <= c < ndim for c in cols_all], dtype=bool)
            if not keep.any():
                continue
            cols_in = [c for c, k in zip(cols_all, keep) if k]
            _mn = getattr(dist, "minimum", getattr(dist, "min_val", None))
            _mx = getattr(dist, "maximum", getattr(dist, "max_val", None))
            if _mn is None or _mx is None:
                unread.extend(cols_in)
                continue
            try:
                # a scalar bound on a multi-column (tuple-keyed) entry
                # applies to each of its columns; a per-column array must
                # match the entry's columns one for one — masked TOGETHER
                # with the columns, so an out-of-range column cannot shift
                # the others' bounds
                mn = np.broadcast_to(np.asarray(_mn, dtype=float), (len(cols_all),))[keep]
                mx = np.broadcast_to(np.asarray(_mx, dtype=float), (len(cols_all),))[keep]
            except (TypeError, ValueError):
                unread.extend(cols_in)
                continue
            lo[cols_in] = mn
            hi[cols_in] = mx
            covered.extend(cols_in)
    except Exception as exc:  # never break the sampler on an exotic prior
        logger.warning(
            "[eigen_refresh] prior box unavailable (%r); keeping the columns "
            "read so far, unit widths elsewhere", exc,
        )
    width = hi - lo
    unread.extend(c for c in covered if not np.isfinite(width[c]) or width[c] <= 0)
    if unread:
        _warn_unread(ndim, tuple(sorted(set(unread))))
    return prior_box_scales(lo, hi)


# (ndim, columns) already reported by _warn_unread: addremove calls
# prior_box_widths once per leaf per refresh, so an unbounded column would
# otherwise log one identical line per leaf.
_UNREAD_WARNED = set()


def _warn_unread(ndim, cols):
    key = (int(ndim), tuple(cols))
    if key in _UNREAD_WARNED:
        return
    _UNREAD_WARNED.add(key)
    logger.warning(
        "[eigen_refresh] prior columns %s expose no finite positive "
        "(minimum, maximum) box; using unit width there — the eigen steps "
        "and the prior-box cap on those columns are NOT in the parameter's "
        "own units.", list(cols),
    )


def _fallback_table(widths):
    """Identity axes with 1%-of-prior-width steps — always usable."""
    widths = np.asarray(widths, dtype=float)
    return np.eye(widths.size), 1e-2 * widths


def _identity_tables(widths, n):
    """Batched identity fallback: ``n`` copies of :func:`_fallback_table`."""
    widths = np.asarray(widths, dtype=float)
    d = widths.size
    return (np.broadcast_to(np.eye(d), (n, d, d)).copy(),
            np.broadcast_to(1e-2 * widths, (n, d)).copy())


def _tables_from_info_batch(info, widths, sigma_max_frac=1.0):
    """Whiten -> eigen axes -> un-whiten -> prior cap, batched ``(n, d, d)``.

    Raises on non-finite input or output (callers catch and fall back).
    """
    widths = np.asarray(widths, dtype=float)
    info = np.asarray(info, dtype=float)
    if not np.all(np.isfinite(info)):
        raise ValueError("non-finite information matrix")
    # diag(w) @ info @ diag(w): the whitened matrix whose spectrum reflects
    # curvature, not unit choice
    info_y = info * widths[None, :, None] * widths[None, None, :]
    # whitened sigma_max=1.0 == one prior width along any axis
    axes_y, sig_y = eigen_axis_set(info_y, sigma_max=1.0)
    a_x = widths[None, :, None] * axes_y
    norms = np.linalg.norm(a_x, axis=1)
    norms = np.where(norms > 0, norms, 1.0)
    axes = a_x / norms[:, None, :]
    sigmas = sig_y * norms
    bounds = axis_prior_bounds(axes, widths)
    sigmas = np.minimum(sigmas, float(sigma_max_frac) * bounds)
    if not (np.all(np.isfinite(axes)) and np.all(np.isfinite(sigmas))):
        raise ValueError("non-finite eigen table")
    return axes, sigmas


def _table_from_info(info, widths, sigma_max_frac=1.0):
    """Single-matrix wrapper over :func:`_tables_from_info_batch`."""
    axes, sigmas = _tables_from_info_batch(
        np.asarray(info, dtype=float)[None], widths,
        sigma_max_frac=sigma_max_frac,
    )
    return axes[0], sigmas[0]


def eigen_table_from_ll(call_ll, x0, widths, *, eps_rel=1e-4,
                        sigma_max_frac=1.0, xp=np):
    """``(axes, sigmas)`` from likelihood second differences at ``x0``.

    ``call_ll(params_2d) -> ll_1d`` scores rows in the SAMPLING basis (wrap
    any transform inside it) and must not mutate the residual. The
    per-parameter step is ``eps_rel`` of the prior box width, so the
    corners stay well inside the prior for any reasonable ``x0``.
    """
    widths = np.asarray(widths, dtype=float)
    try:
        x0 = np.asarray(x0, dtype=float)
        param_eps = float(eps_rel) * widths
        info = information_matrix_from_ll(
            call_ll, x0[None, :], xp=xp, param_eps=param_eps
        )
        info = np.asarray(info)[0]
        return _table_from_info(info, widths, sigma_max_frac=sigma_max_frac)
    except Exception as exc:
        logger.warning(
            "[eigen_refresh] information-matrix build from the likelihood "
            "failed (%r); using the identity fallback table", exc,
        )
        return _fallback_table(widths)


def eigen_tables_from_ll_batch(call_ll, x0s, widths, *, eps_rel=1e-4,
                               sigma_max_frac=1.0, xp=np):
    """One ``(axes, sigmas)`` table per row of ``x0s``, ONE batched sweep.

    ``x0s`` is ``(n, ndim)`` — e.g. every (temperature, walker) point of a
    leaf. All corners of all points go through ``call_ll`` together, so a
    batched scorer pays one dispatch. Rows reach ``call_ll`` as whole
    n-point blocks in the ``x0s`` row order (the
    :func:`~lisatools.info_matrix_ll.information_matrix_from_ll` batching
    invariant), so per-point metadata — a per-walker ``data_index`` —
    must be TILED by ``rows // n``.

    Returns ``axes (n, ndim, ndim)``, ``sigmas (n, ndim)``; any failure
    degrades to identity tables for every point with a logged warning.
    """
    widths = np.asarray(widths, dtype=float)
    x0s = np.atleast_2d(np.asarray(x0s, dtype=float))
    try:
        info = information_matrix_from_ll(
            call_ll, x0s, xp=xp, param_eps=float(eps_rel) * widths
        )
        return _tables_from_info_batch(
            np.asarray(info), widths, sigma_max_frac=sigma_max_frac
        )
    except Exception as exc:
        logger.warning(
            "[eigen_refresh] batched information-matrix build failed (%r); "
            "using identity fallback tables", exc,
        )
        return _identity_tables(widths, x0s.shape[0])


def eigen_table_from_waveform(waveform_model, params, widths, *,
                              eps_rel=1e-4, sigma_max_frac=1.0,
                              deriv_inds=None, parameter_transforms=None,
                              inner_product_kwargs=None,
                              waveform_kwargs=None, more_accurate=False):
    """``(axes, sigmas)`` from the Gram-form waveform information matrix.

    Delegates to :func:`lisatools.diagnostic.info_matrix` — ``params`` and
    the ``eps_rel``-of-prior-width derivative steps are in the SAMPLING
    basis, with ``parameter_transforms`` carrying the map to the waveform
    basis (fills included). ``inner_product_kwargs`` must weight by the
    PSD the run is actually using.
    """
    widths = np.asarray(widths, dtype=float)
    try:
        eps = float(eps_rel) * widths
        info = info_matrix(
            eps,
            waveform_model,
            np.asarray(params, dtype=float).copy(),
            deriv_inds=deriv_inds,
            inner_product_kwargs=inner_product_kwargs or {},
            parameter_transforms=parameter_transforms,
            waveform_kwargs=waveform_kwargs or {},
            more_accurate=more_accurate,
        )
        return _table_from_info(
            np.asarray(info), widths, sigma_max_frac=sigma_max_frac
        )
    except Exception as exc:
        logger.warning(
            "[eigen_refresh] information-matrix build from the waveform "
            "failed (%r); using the identity fallback table", exc,
        )
        return _fallback_table(widths)
