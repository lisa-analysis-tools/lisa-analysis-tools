"""Host-side cache of GB in-model proposal factors (``GB_CHOL_CACHE``).

What is cached
--------------
Every GB in-model block draws its jumps from a per-source factor ``B`` of the
inverse information matrix in the sampling basis (``B B^T = Gamma^-1``, or the
eigen-axis table when ``GB_INMODEL_EIGEN_AXIS`` is ready), built by
:meth:`GBSpecialBase._compute_proposal_cholesky` and held fixed for the whole
block. Computing it for every source of every block dominated
``inmodel_cholesky``. With ``GB_CHOL_CACHE=1`` the factor of each living
source is computed once per refresh and served to every later block that
contains the source, until the next refresh.

With the observable eigen proposal on (``GB_INMODEL_OBSERVABLE_EIGEN``), the
same computation also stashes the observable-basis information matrix
``Gamma_z`` on the move (``move._obs_gamma_z``, indexed by sorter position).
It is cached beside the factor and written back into that stash on every hit,
so ``_observable_eigen_prepare`` reads what a direct computation would have
left there.

Residency
---------
Entries live in HOST (numpy) memory, one append-only table per walker
(:class:`_WalkerTable`). A block gathers its rows on the host and uploads a
single ``(n, ndim, ndim)`` array to the move's device (``move.xp``); nothing
is held on the GPU between blocks.

Keying
------
Sorter positions are rebuilt every propose, so an entry is found by what the
source is: its walker and its coordinates. A source is matched only within
its own walker and never by its temperature rung -- a vertical (all-rungs)
swap only relabels the rung a source sits on, so the source must keep its
entry. The candidates are the ``GB_CHOL_CACHE_WINDOW`` entries on each side
of the source in f0 (wide enough to span every rung's copy of the same
source); the best candidate is the one with the smallest worst-column
distance over sampling columns 0-2 (amplitude or distance, f0, fdot or Mc),
each column measured in that entry's own marginal width
``sqrt(diag(B B^T))``. It is a hit when that distance is at most
``GB_CHOL_CACHE_TOL``. A hit moves the entry's key to the source's current
coordinates, so the key follows the source as it random-walks.

Refresh (global ticker)
-----------------------
Each cache keeps one ticker: the largest propose index (``move.time``) seen
from any move that shares it, so a slower move never rewinds it. When the
ticker enters a new multiple of ``GB_CHOL_CACHE_EVERY``, the first in-model
block of the next proposal (:meth:`GBSpecialBase._ensure_proposal_tables`)
rebuilds the whole cache from every alive source of the sorter (all walkers,
all rungs) in ``GB_CHOL_CACHE_BATCH``-sized calls. A refresh belongs to no block and has no
buffer slots, so it takes the engine's slot-free route: the lookup Gram
``<dh_a|dh_b>`` with ``SIGHET_INFOMAT_ENGINE=lookup`` (production), otherwise
the chunked delegate (slow; ``_infomat_route_check`` reports the cost).

Births and misses
-----------------
Rows of a block that find no entry (accepted births, sources that moved
beyond tolerance, a walker with no entries) are computed in one call per
block, before the repeats, through the block's own route -- with their
buffer slots, so the sig-het fast route applies -- and appended to the cache.
The factor is fixed for the whole block exactly as with direct factors, so
the proposal stays symmetric: a borrowed or slightly stale factor costs
acceptance rate, never detailed balance.

End-of-block retrack
--------------------
The ~25 repeats of a block can carry a source several tolerances away from
where the block found it (Gram steps are large). After the final write-back,
``GBSpecialBase._run_in_model_repeats`` calls :meth:`_CholCache.retrack` with
the final coordinates, so the next block finds the entry.

Failure
-------
Any exception in a refresh, a take or a retrack disables that cache for the
rest of the run (one WARNING with the traceback) and the move falls back to
computing every block's factors directly, as if the cache were off.

The lookup Gram engine came in together with the cache, so the failure path
also unsets ``SIGHET_INFOMAT_ENGINE`` and the direct factors go back to the
validated route (sig-het second differences with slots). This has to be an
environment change: GBGPU's ``GBSignalHetComputations.information_matrix``
reads ``SIGHET_INFOMAT_ENGINE`` from ``os.environ`` on every call and has no
argument or attribute that selects the route. The change is process-wide, so
it is logged once at WARNING; it also applies to every other move and every
other cache on the rank (their later refreshes go through the chunked
delegate).

Knobs (environment)
-------------------
``GB_CHOL_CACHE`` (default ``0``)
    ``1`` turns the cache on; read by the move's gate on every call.
``GB_CHOL_CACHE_EVERY`` (default ``40``)
    Refresh period in proposes (40 proposes x ~25 repeats is ~1000 in-model
    steps). Minimum 1.
``GB_CHOL_CACHE_TOL`` (default ``5``)
    Match tolerance, in the entry's marginal widths.
``GB_CHOL_CACHE_BATCH`` (default ``4096``)
    Sources per factor call during a refresh (bounds peak device memory).
    Minimum 1.
``GB_CHOL_CACHE_WINDOW`` (default ``32``)
    Candidate entries examined on each side of a source in f0. Minimum 1.

All but ``GB_CHOL_CACHE`` are read once, when a cache is created.

Logging
-------
Every line reads ``<move name>: [GB_CHOL_CACHE] ...`` (the branch name when
no move is in hand):

* ``refreshed N sources in T s (...)`` -- INFO, each refresh (ticker, period,
  info-matrix engine, host MB);
* ``H hits / M misses over the last 500 blocks (...)`` -- INFO; the counts
  cover every move that shares the cache;
* ``DISABLED after an error in <refresh|take|retrack> ...`` -- WARNING with
  the traceback, at most once per cache;
* ``unset SIGHET_INFOMAT_ENGINE=... for the rest of the run`` -- WARNING, once
  per process.

Registry, deepcopy and pickle
-----------------------------
There is one cache per process (one MPI rank) per key ``(branch, observable
eigen mode, eigen-axis flag)``, held in the module-level ``_CHOL_CACHES`` and
reached through :func:`get_chol_cache`. The mode and flag are in the key
because they change what ``_compute_proposal_cholesky`` returns (a joint
factor or an eigen-axis table, with or without ``Gamma_z``), so those factors
are not interchangeable. Every GB move of a branch shares one map, so a
refresh is paid once per ticker period, not once per move.

The registry is module-level on purpose, rather than an attribute of a move.
The cache is per-process runtime state whose host arrays scale with the live
source count, so a move that is deep-copied or pickled carries no reference
to it and stays light (settings-tree objects must survive
``pickle.loads(pickle.dumps(copy.deepcopy(obj)))``). The cache itself holds
only numpy arrays and plain Python values -- never an array module; the
device module comes from ``move.xp`` at call time -- so it can be deep-copied
and pickled too.
"""

from __future__ import annotations

import logging
import os
import time
from typing import NamedTuple, Optional

import numpy as np

from ...utils.utility import asnumpy, get_array_module

logger = logging.getLogger(__name__)

__all__ = ["chol_cache_enabled", "get_chol_cache"]

#: Defaults of the ``GB_CHOL_CACHE_*`` knobs (see the module docstring).
DEFAULT_EVERY = 40
DEFAULT_TOL = 5.0
DEFAULT_BATCH = 4096
DEFAULT_WINDOW = 32

#: Sampling column the per-walker tables are sorted on (f0).
_F0_COL = 1
#: Sampling columns compared against the tolerance (amplitude or distance, f0,
#: fdot or Mc).
_MATCH_COLS = np.array([0, 1, 2])
#: Smallest row capacity a walker table is allocated with.
_MIN_CAPACITY = 1024
#: Blocks between two hits/misses report lines.
_REPORT_EVERY_BLOCKS = 500
#: GBGPU's info-matrix route selector, unset when a cache fails.
_INFOMAT_ENGINE_ENV = "SIGHET_INFOMAT_ENGINE"

#: Per-process registry: one :class:`_CholCache` per key (see the module
#: docstring for why this is module-level rather than on the move).
_CHOL_CACHES: dict = {}


def chol_cache_enabled() -> bool:
    """True when ``GB_CHOL_CACHE=1`` (default off)."""
    return os.environ.get("GB_CHOL_CACHE", "0") == "1"


def get_chol_cache(key) -> "_CholCache":
    """Return this process's cache for ``key``, creating it on first use.

    Args:
        key: ``(branch name, observable eigen mode, eigen-axis flag)``.
    """
    cache = _CHOL_CACHES.get(key)
    if cache is None:
        cache = _CHOL_CACHES[key] = _CholCache(key)
    return cache


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    """Integer knob; empty or unset means ``default``; clamped to ``minimum``."""
    return max(int(os.environ.get(name, str(default)) or default), minimum)


def _env_float(name: str, default: float) -> float:
    """Float knob; empty or unset means ``default``."""
    return float(os.environ.get(name, str(default)) or default)


def _take_rows(a, rows):
    """``a[rows]`` for a host (numpy) row index, whether ``a`` is numpy or cupy."""
    if isinstance(a, np.ndarray):
        return a[rows]
    return a[get_array_module(a).asarray(rows)]


def _regrow(old, n_filled: int, capacity: int, template, fill=None):
    """A new ``capacity``-row buffer holding the first ``n_filled`` rows of ``old``.

    Row shape and dtype come from ``old``, or from ``template`` when there is
    no ``old`` yet. The other rows are uninitialised, or ``fill`` if given.
    """
    like = template if old is None else old
    shape = (capacity,) + like.shape[1:]
    if fill is None:
        buf = np.empty(shape, dtype=like.dtype)
    else:
        buf = np.full(shape, fill, dtype=like.dtype)
    if old is not None:
        buf[:n_filled] = old[:n_filled]
    return buf


class _WalkerTable:
    """One walker's cache entries: append-only host arrays plus an f0 index.

    Rows ``[0, n)`` are filled. They are appended with amortized doubling and
    never move, so an entry number recorded for a block stays valid until a
    refresh replaces the table. ``coords`` holds each entry's key (the
    coordinates it was last matched or retracked at) and is the only array
    rewritten in place.

    Attributes:
        n: Number of filled rows.
        coords: ``(capacity, ndim)`` key coordinates, sampling basis.
        factors: ``(capacity, ndim, ndim)`` proposal factors ``B``.
        widths: ``(capacity, 3)`` marginal widths of columns ``_MATCH_COLS``
            in coordinate units.
        gamma_z: ``(capacity, nz, nz)`` observable information matrices, or
            ``None`` while no row carries one. Rows stored without one are NaN
            (the move treats a non-finite ``Gamma_z`` as "no eigen table").
        f0_order: Filled rows sorted by key f0.
        f0_sorted: The key f0 values in that order.
    """

    def __init__(self):
        self.n = 0
        self.coords = None
        self.factors = None
        self.widths = None
        self.gamma_z = None
        self.f0_order = None
        self.f0_sorted = None

    def append(self, coords, factors, widths, gamma_z) -> np.ndarray:
        """Append rows and return their entry numbers.

        Args:
            coords: ``(k, ndim)`` key coordinates.
            factors: ``(k, ndim, ndim)`` proposal factors.
            widths: ``(k, 3)`` marginal widths.
            gamma_z: ``(k, nz, nz)`` observable information matrices, or
                ``None`` (stored as NaN if the table carries ``Gamma_z``).
        """
        n, k = self.n, len(coords)
        if self.coords is None or n + k > len(self.coords):
            capacity = max(2 * (n + k), _MIN_CAPACITY)
            self.coords = _regrow(self.coords, n, capacity, coords)
            self.factors = _regrow(self.factors, n, capacity, factors)
            self.widths = _regrow(self.widths, n, capacity, widths)
            if self.gamma_z is not None:
                self.gamma_z = _regrow(self.gamma_z, n, capacity, None, fill=np.nan)
        self.coords[n:n + k] = coords
        self.factors[n:n + k] = factors
        self.widths[n:n + k] = widths
        if gamma_z is not None:
            if self.gamma_z is None:
                # the first rows with Gamma_z: every earlier row reads as NaN
                self.gamma_z = _regrow(None, n, len(self.coords), gamma_z, fill=np.nan)
            self.gamma_z[n:n + k] = gamma_z
        # rows appended without Gamma_z keep the NaN fill
        self.n = n + k
        self.reindex()
        return np.arange(n, n + k)

    def reindex(self):
        """Rebuild the f0 sort index after keys were added or moved."""
        f0 = self.coords[:self.n, _F0_COL]
        self.f0_order = np.argsort(f0, kind="stable")
        self.f0_sorted = f0[self.f0_order]

    def nbytes(self) -> int:
        """Host bytes allocated by the table's arrays (capacity, not fill)."""
        return sum(a.nbytes for a in (self.coords, self.factors, self.widths,
                                      self.gamma_z) if a is not None)


class _Lookup(NamedTuple):
    """Result of :meth:`_CholCache.match` for one block's rows.

    Attributes:
        hit: ``(n,)`` bool, row found an entry.
        walkers: ``(n,)`` walker of each row (the table it was looked up in).
        entries: ``(n,)`` entry number within that table, ``-1`` on a miss.
        factors: ``(n, ndim, ndim)`` host factors; zeros on misses.
        gamma_z: ``(n, nz, nz)`` host ``Gamma_z`` (NaN where absent), or
            ``None`` when no hit came from a table that carries one.
    """

    hit: np.ndarray
    walkers: np.ndarray
    entries: np.ndarray
    factors: np.ndarray
    gamma_z: Optional[np.ndarray]


class _CholCache:
    """Host-resident in-model proposal factors, mapped to living sources.

    See the module docstring for the design (keying, refresh ticker, misses,
    retrack, failure) and the knobs. Instances come from
    :func:`get_chol_cache`; the GB move's ``_chol_cache`` gate decides
    whether one is used at all.

    Attributes:
        key: ``(branch, observable eigen mode, eigen-axis flag)``.
        every: Refresh period in proposes (``GB_CHOL_CACHE_EVERY``).
        tol: Match tolerance in marginal widths (``GB_CHOL_CACHE_TOL``).
        batch: Sources per factor call in a refresh (``GB_CHOL_CACHE_BATCH``).
        window: Candidates on each side in f0 (``GB_CHOL_CACHE_WINDOW``).
        tick: Largest propose index seen (the global ticker).
        epoch: ``tick // every`` at the last refresh; ``None`` before the
            first one (the cache serves nothing until then).
        tables: ``{walker: _WalkerTable}``.
        disabled: Set by the first failure; permanent.
    """

    def __init__(self, key):
        self.key = key
        self.every = _env_int("GB_CHOL_CACHE_EVERY", DEFAULT_EVERY)
        self.tol = _env_float("GB_CHOL_CACHE_TOL", DEFAULT_TOL)
        self.batch = _env_int("GB_CHOL_CACHE_BATCH", DEFAULT_BATCH)
        self.window = _env_int("GB_CHOL_CACHE_WINDOW", DEFAULT_WINDOW)
        self.tick = -1
        self.epoch = None
        self.tables = {}
        self.disabled = False
        # (ids, walkers, entries) of the block the last take served; consumed
        # by retrack at the end of that block.
        self._served = None
        self._hits_since_report = 0
        self._misses_since_report = 0
        self._blocks_served = 0

    # ---- ticker and refresh ------------------------------------------------

    def due(self, propose_index) -> bool:
        """Advance the ticker to ``propose_index`` and say if a refresh is due.

        The ticker only moves forward (moves sharing the cache count proposes
        separately). A refresh is due when the cache has never been refreshed
        or the ticker has entered a new ``every``-period since the last one;
        never once the cache is disabled.
        """
        self.tick = max(self.tick, int(propose_index))
        return (not self.disabled
                and (self.epoch is None or self.tick // self.every != self.epoch))

    def refresh(self, move, model, band_sorter):
        """Rebuild the whole cache from every alive source of ``band_sorter``.

        Computes the factors in ``batch``-sized calls through the move's
        slot-free route and replaces all tables. On error the cache is
        disabled (see :meth:`_fail`).
        """
        if self.disabled:
            return
        t0 = time.perf_counter()
        try:
            alive = move.xp.where(band_sorter.inds)[0]
            self.tables = {}
            self._served = None
            n_alive = int(alive.shape[0])
            for start in range(0, n_alive, self.batch):
                ids = alive[start:start + self.batch]
                factors = move._compute_proposal_cholesky(model, band_sorter, ids)
                self._store(move, band_sorter, ids, factors)
        except Exception as exc:  # noqa: BLE001 -- see _fail
            self._fail(move, "refresh", exc)
            return
        self.epoch = self.tick // self.every
        logger.info(
            "%s: [GB_CHOL_CACHE] refreshed %d sources in %.1f s (propose %d, "
            "every %d proposes, %s=%s, %.1f MB host)", move.name, n_alive,
            time.perf_counter() - t0, self.tick, self.every, _INFOMAT_ENGINE_ENV,
            os.environ.get(_INFOMAT_ENGINE_ENV, "") or "default",
            self.nbytes() / 2**20)

    # ---- serving a block ---------------------------------------------------

    def take(self, move, model, band_sorter, ids, slots, buffer_obj):
        """Proposal factors for one block's sources ``ids`` (on ``move.xp``).

        Hits are served from the cache; misses are computed in one call
        through the block's own route (``slots`` and ``buffer_obj`` as for
        :meth:`GBSpecialBase._compute_proposal_cholesky`) and stored. Also
        leaves ``move._proposal_param_scales`` set and, for hits, writes the
        cached ``Gamma_z`` into ``move._obs_gamma_z`` -- the two side effects
        the direct computation has. Before the first refresh, once disabled,
        or on error (which disables), the whole block is computed directly.
        """
        self._served = None
        if self.disabled or self.epoch is None:
            return move._compute_proposal_cholesky(
                model, band_sorter, ids, slots=slots, buffer_obj=buffer_obj)
        xp = move.xp
        try:
            found = self.match(band_sorter, ids)
            chol = xp.asarray(found.factors)
            miss = np.nonzero(~found.hit)[0]
            if miss.size:
                # Computed before the Gamma_z write-back below: the direct
                # computation may replace move._obs_gamma_z with a new array.
                miss_ids = _take_rows(ids, miss)
                miss_factors = move._compute_proposal_cholesky(
                    model, band_sorter, miss_ids,
                    slots=None if slots is None else _take_rows(slots, miss),
                    buffer_obj=buffer_obj)
                found.entries[miss] = self._store(
                    move, band_sorter, miss_ids, miss_factors)
                chol[xp.asarray(miss)] = miss_factors
            else:
                # no direct computation ran: set its scale side effect here
                move._proposal_param_scales = self._param_scales(move, band_sorter)
            if found.gamma_z is not None and found.hit.any():
                self._write_gamma_z(move, band_sorter, ids, found)
            self._served = (asnumpy(ids).copy(), found.walkers, found.entries)
        except Exception as exc:  # noqa: BLE001 -- see _fail
            self._fail(move, "take", exc)
            return move._compute_proposal_cholesky(
                model, band_sorter, ids, slots=slots, buffer_obj=buffer_obj)
        self._count(move, int(found.hit.sum()), int(miss.size))
        return chol

    def match(self, band_sorter, ids) -> _Lookup:
        """Find each source's entry; move the keys of hits to the sources.

        For every row, looks in the table of the row's walker at the
        ``2 * window`` entries nearest in f0 and picks the one with the
        smallest worst-column distance over ``_MATCH_COLS`` in that entry's
        marginal widths; a hit is a distance ``<= tol``. Each hit entry's key
        is set to the row's current coordinates.
        """
        coords = asnumpy(band_sorter.coords[ids])
        n_rows, ndim = coords.shape
        hit = np.zeros(n_rows, dtype=bool)
        entries = np.full(n_rows, -1, dtype=np.int64)
        factors = np.zeros((n_rows, ndim, ndim))
        gamma_z = None
        walkers = self._walkers(band_sorter, ids)
        offsets = np.arange(-self.window, self.window)
        for walker in np.unique(walkers):
            table = self.tables.get(int(walker))
            if table is None or table.n == 0:
                continue
            rows = np.nonzero(walkers == walker)[0]
            query = coords[rows]
            # the 2 * window table positions around each row's f0 insertion
            # point, clipped to the filled range
            pos = np.searchsorted(table.f0_sorted, query[:, _F0_COL])
            cand = table.f0_order[
                np.clip(pos[:, None] + offsets[None, :], 0, table.n - 1)]
            with np.errstate(divide="ignore", invalid="ignore"):
                dist = (np.abs(query[:, None, _MATCH_COLS]
                               - table.coords[cand][:, :, _MATCH_COLS])
                        / table.widths[cand])
            # a zero width gives inf or NaN: never a match
            score = np.nan_to_num(dist.max(-1), nan=np.inf)
            best = score.argmin(1)
            r = np.arange(len(rows))
            best_entry = cand[r, best]
            ok = score[r, best] <= self.tol
            if not ok.any():
                continue
            hit_rows, hit_entries = rows[ok], best_entry[ok]
            hit[hit_rows] = True
            entries[hit_rows] = hit_entries
            factors[hit_rows] = table.factors[hit_entries]
            if table.gamma_z is not None:
                if gamma_z is None:
                    gamma_z = np.full((n_rows,) + table.gamma_z.shape[1:], np.nan)
                gamma_z[hit_rows] = table.gamma_z[hit_entries]
            table.coords[hit_entries] = query[ok]
            table.reindex()
        return _Lookup(hit, walkers, entries, factors, gamma_z)

    def retrack(self, ids, coords):
        """End of block: move the served entries' keys to the final coordinates.

        The key is set when the block starts (:meth:`match`); the block's
        repeats then move the source, often beyond tolerance. Applies only
        to the block the last :meth:`take` served: ``ids`` must be that
        block's ids in the same order, else nothing changes. ``coords`` are
        the final sampling coordinates, row-aligned with ``ids``. On error
        the cache is disabled (see :meth:`_fail`).
        """
        served, self._served = self._served, None
        if served is None or self.disabled:
            return
        served_ids, walkers, entries = served
        ids_now = asnumpy(ids)
        if ids_now.shape != served_ids.shape or not np.array_equal(ids_now, served_ids):
            return
        try:
            final = asnumpy(coords)
            for walker in np.unique(walkers):
                table = self.tables.get(int(walker))
                rows = (walkers == walker) & (entries >= 0)
                if table is None or not rows.any():
                    continue
                table.coords[entries[rows]] = final[rows]
                table.reindex()
        except Exception as exc:  # noqa: BLE001 -- see _fail
            self._fail(None, "retrack", exc)

    # ---- internals ---------------------------------------------------------

    def _store(self, move, band_sorter, ids, factors) -> np.ndarray:
        """Append freshly computed ``factors`` for ``ids``; return entry numbers.

        Must run right after ``move._compute_proposal_cholesky`` for the same
        ``ids``: the marginal widths use the scales and the ``Gamma_z`` comes
        from the stash that call left on the move.
        """
        coords = asnumpy(band_sorter.coords[ids])
        factors = asnumpy(factors)
        gamma_z = self._gamma_z_of(move, ids)
        widths = self._marginal_widths(move, factors)
        walkers = self._walkers(band_sorter, ids)
        entries = np.empty(len(walkers), dtype=np.int64)
        for walker in np.unique(walkers):
            rows = walkers == walker
            table = self.tables.get(int(walker))
            if table is None:
                table = self.tables[int(walker)] = _WalkerTable()
            entries[rows] = table.append(
                coords[rows], factors[rows], widths[rows],
                None if gamma_z is None else gamma_z[rows])
        return entries

    @staticmethod
    def _walkers(band_sorter, ids) -> np.ndarray:
        """Host walker index of each source: the lookup table it belongs to.

        Deliberately not the rung, which a vertical swap relabels.
        """
        return asnumpy(band_sorter.walker_inds[ids]).astype(np.int64)

    @staticmethod
    def _marginal_widths(move, factors) -> np.ndarray:
        """``sqrt(diag(B B^T))`` on ``_MATCH_COLS``, in coordinate units.

        ``B`` is in the conditioned basis ``y = x / s``
        (``move._proposal_param_scales``), so the widths are scaled by ``s``.
        """
        scales = asnumpy(move._proposal_param_scales)[_MATCH_COLS]
        return np.sqrt((factors[:, _MATCH_COLS, :] ** 2).sum(-1)) * scales[None, :]

    @staticmethod
    def _gamma_z_of(move, ids):
        """Host ``Gamma_z`` the move just stashed for ``ids``, or ``None``."""
        stash = getattr(move, "_obs_gamma_z", None)
        if stash is None or move._obs_eigen_mode() == "off":
            return None
        return asnumpy(stash[ids])

    @staticmethod
    def _param_scales(move, band_sorter):
        """The conditioning scales ``_compute_proposal_cholesky`` would set."""
        scales = move.xp.ones(band_sorter.coords.shape[1])
        if move._fdot_col is not None:
            scales[move._fdot_col] = move._fdot_scale
        return scales

    @staticmethod
    def _write_gamma_z(move, band_sorter, ids, found: _Lookup):
        """Write the hits' cached ``Gamma_z`` into ``move._obs_gamma_z``.

        The stash is indexed by sorter position; it is replaced by a NaN array
        of the current sorter size when missing or mis-shaped, as the direct
        computation does.
        """
        xp = move.xp
        n_src = int(band_sorter.inds.shape[0])
        nz = int(found.gamma_z.shape[-1])
        stash = getattr(move, "_obs_gamma_z", None)
        if (stash is None or int(stash.shape[0]) != n_src
                or int(stash.shape[-1]) != nz):
            stash = xp.full((n_src, nz, nz), xp.nan)
        hit_rows = np.nonzero(found.hit)[0]
        stash[xp.asarray(_take_rows(ids, hit_rows))] = xp.asarray(found.gamma_z[found.hit])
        move._obs_gamma_z = stash

    def _count(self, move, hits: int, misses: int):
        """Tally a served block; log hits/misses every ``_REPORT_EVERY_BLOCKS``."""
        self._hits_since_report += hits
        self._misses_since_report += misses
        self._blocks_served += 1
        if self._blocks_served % _REPORT_EVERY_BLOCKS:
            return
        total = self._hits_since_report + self._misses_since_report
        logger.info(
            "%s: [GB_CHOL_CACHE] %d hits / %d misses over the last %d blocks "
            "(%.1f %% hits, %.1f MB host)", move.name, self._hits_since_report,
            self._misses_since_report, _REPORT_EVERY_BLOCKS,
            100.0 * self._hits_since_report / max(total, 1), self.nbytes() / 2**20)
        self._hits_since_report = self._misses_since_report = 0

    def _fail(self, move, where: str, exc: BaseException):
        """Disable the cache for the rest of the run after an error.

        The run must not stop for a cache fault: the fallback is the direct
        per-block computation the cache replaced, hence the broad ``except``
        at the call sites. Also unsets ``SIGHET_INFOMAT_ENGINE`` (process
        wide, logged once; see the module docstring for why it has to be the
        environment). ``move`` is ``None`` when called from :meth:`retrack`.
        """
        self.disabled = True
        self._served = None
        # retrack has no move in hand: name the cache's branch instead
        name = move.name if move is not None else str(self.key[0])
        logger.warning(
            "%s: [GB_CHOL_CACHE] DISABLED after an error in %s (%r); every "
            "block computes its own factors from here on (cache %r).",
            name, where, exc, self.key, exc_info=exc)
        engine = os.environ.pop(_INFOMAT_ENGINE_ENV, None)
        if engine is not None:
            logger.warning(
                "%s: [GB_CHOL_CACHE] unset %s=%r for the rest of the run, "
                "process-wide: direct factors go back to the sig-het "
                "second-difference route (slot-free calls to the chunked "
                "delegate).", name, _INFOMAT_ENGINE_ENV, engine)

    def nbytes(self) -> int:
        """Host bytes allocated by all tables."""
        return sum(t.nbytes() for t in self.tables.values())
