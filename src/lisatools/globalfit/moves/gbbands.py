"""Band-level infrastructure for the GB special moves.

This module owns the sub-band machinery that the GB proposal moves in
:mod:`gbspecialstretch` drive:

* :func:`pack_special_index` / :func:`unpack_special_index` -- the single
  home of the ``(temp * nwalkers + walker) * 1e6 + band`` encoding used to
  identify a (temperature, walker, band) cell of the ensemble.
* :class:`SubBandBuffer` -- the per-cell residual / PSD / template scratch
  buffer. It **is** an :class:`~lisatools.analysiscontainer.AnalysisContainerArray`
  (one :class:`~lisatools.analysiscontainer.AnalysisContainer` per active
  cell) extended with fast GB source injection/removal through a
  :class:`~lisatools.globalfit.moves.gb_likelihood.BandLikelihoodEngine`.
  ``Buffer`` is kept as a back-compat alias.
* :class:`BandSorter` -- flat per-source view of the eryn GB branch
  (coords / inds / temp / walker / leaf / band index arrays) with subset
  and RJ pre-draw machinery.
"""

from __future__ import annotations

import logging
import warnings
import copy as copy_module
from copy import deepcopy
from types import ModuleType
from typing import List, Optional, Tuple, Union

import numpy as np
import numpy

try:
    import cupy as cp
    import cupy

    gpu_available = True
except ModuleNotFoundError:
    import numpy as cp

    gpu_available = False

from eryn.state import Branch
from eryn.utils import TransformContainer

from ...analysiscontainer import (
    AnalysisContainer,
    AnalysisContainerArray,
    BandView,
    band_gpu_assignment,
)
from ...domains import DomainSettingsBase, FDSettings, STFTSettings, WDMSettings
from ...sensitivity import SensitivityMatrixBase
from ...utils.constants import AU_SI, C_SI, YRSID_SI
from ...utils.device import assert_peer_access, device_context
from ...utils.devicereplicas import device_local_gb_comp, device_local_orbits
from ...utils.parallelbase import LISAToolsParallelModule
from ...utils.utility import asnumpy, get_array_module
from .gbbandstructure import STFT_F0_LIMIT_FRACTION

__all__ = [
    "pack_special_index",
    "unpack_special_index",
    "SubBandBuffer",
    "Buffer",
    "BandSorter",
    "BandScheduler",
    "make_routed_band_engine",
    "estimate_buffer_preload_limits",
]

logger = logging.getLogger(__name__)

_to_numpy = asnumpy

# Encoding base for the band part of a special index. Bands live in
# ``[0, _SPECIAL_INDEX_BASE)``; everything above encodes (temp, walker).
_SPECIAL_INDEX_BASE = int(1e6)


def estimate_buffer_preload_limits(
    basis_settings: DomainSettingsBase,
    nchannels: int = 3,
    tdi_setup: str = "XYZ",
    max_data_store_size: int = 6000,
    ntemps: int = 1,
    xp=np,
    gpus: Optional[List[int]] = None,
    max_budget_fraction: float = 0.10,
    max_budget_bytes: int = 4 * 1024**3,
) -> Tuple[int, int]:
    """Estimate safe (num_band_preload, num_bands_preload_temp) from device memory and basis grid.

    Calculates the memory footprint per buffer cell (data residual, inverse-PSD,
    ACA linear buffers, and twin template buffers), queries available/total device
    memory, and determines maximum safe preload limits so that proposal moves and
    tempering passes never exceed the allocated memory budget.

    Parameters
    ----------
    basis_settings : DomainSettingsBase
        Active domain settings (FDSettings, STFTSettings, WDMSettings).
    nchannels : int
        Number of data channels (default 3 for XYZ).
    tdi_setup : str
        Channel setup (e.g. 'XYZ', 'AE', 'AET').
    max_data_store_size : int
        Window length for FD buffers.
    ntemps : int
        Number of temperature rungs (tempering chunks have ntemps cells per row).
    xp : module
        Active array module (numpy or cupy).
    gpus : list of int, optional
        List of configured GPU device IDs.
    max_budget_fraction : float
        Fraction of total device memory budgeted for sub-band buffers (default 0.10).
    max_budget_bytes : int
        Absolute cap on buffer memory budget in bytes (default 4 GB).

    Returns
    -------
    num_band_preload : int
        Suggested preload capacity for in-model / RJ proposal moves.
    num_bands_preload_temp : int
        Suggested number of (band, walker) units preloaded per tempering chunk.
    """
    if isinstance(basis_settings, FDSettings):
        length = int(max_data_store_size)
        data_bytes = nchannels * length * 16  # complex128
        sens_channels = nchannels * nchannels if tdi_setup == "XYZ" else nchannels
        sens_bytes = sens_channels * length * 8  # float64
    elif isinstance(basis_settings, WDMSettings):
        length = int(basis_settings.Nf_active * basis_settings.Nt_active)
        data_bytes = nchannels * length * 8   # float64
        sens_channels = nchannels * nchannels if tdi_setup == "XYZ" else nchannels
        sens_bytes = sens_channels * length * 8  # float64
    elif isinstance(basis_settings, STFTSettings):
        length = int(basis_settings.NT * basis_settings.NF_active)
        data_bytes = nchannels * length * 16  # complex128
        sens_channels = nchannels * nchannels if tdi_setup == "XYZ" else nchannels
        sens_bytes = sens_channels * length * 16  # complex128
    else:
        length = int(max_data_store_size)
        data_bytes = nchannels * length * 16
        sens_bytes = (nchannels * nchannels) * length * 16

    # In SubBandBuffer with template twin:
    # ~4 * data_bytes + 4 * sens_bytes
    bytes_per_cell = max(1, 4 * data_bytes + 4 * sens_bytes)

    device_mem = None
    if getattr(xp, "__name__", "") == "cupy":
        try:
            device_id = int(gpus[0]) if (gpus is not None and len(gpus) > 0) else int(xp.cuda.runtime.getDevice())
            props = xp.cuda.runtime.getDeviceProperties(device_id)
            device_mem = int(props["totalGlobalMem"])
        except Exception:
            device_mem = None

    if device_mem is None:
        try:
            import psutil
            device_mem = int(psutil.virtual_memory().total)
        except Exception:
            device_mem = 16 * 1024**3  # fallback 16 GB

    budget = min(int(device_mem * max_budget_fraction), int(max_budget_bytes))
    max_cells = max(1, budget // bytes_per_cell)

    if isinstance(basis_settings, FDSettings):
        num_band_preload = min(20000, max(200, max_cells))
        num_bands_preload_temp = min(200, max(1, max_cells // max(1, ntemps)))
    else:
        num_band_preload = max(1, max_cells)
        num_bands_preload_temp = max(1, max_cells // max(1, ntemps))

    return int(num_band_preload), int(num_bands_preload_temp)


def pack_special_index(temp_inds, walker_inds, band_inds, nwalkers: int):
    """Pack ``(temp, walker, band)`` triplets into scalar special indices.

    ``special = (temp * nwalkers + walker) * 1e6 + band``. Works
    elementwise on array inputs of any matching shape.
    """
    return (temp_inds * nwalkers + walker_inds) * _SPECIAL_INDEX_BASE + band_inds


def unpack_special_index(special_band_inds, nwalkers: int) -> tuple:
    """Recover ``(temp, walker, band)`` arrays from packed special indices."""
    # input-driven array module (NOT the module-level cp): this helper runs
    # on numpy inputs during CPU-resolved runs on cupy-installed machines.
    xp = get_array_module(special_band_inds)
    temp_walker_inds = xp.floor(special_band_inds / _SPECIAL_INDEX_BASE).astype(int)
    temp_inds = temp_walker_inds // nwalkers
    walker_inds = temp_walker_inds % nwalkers
    band_inds = (special_band_inds - temp_walker_inds * _SPECIAL_INDEX_BASE).astype(int)
    return (temp_inds, walker_inds, band_inds)


def return_x(x):
    """Identity helper used as a no-op replacement for :func:`copy.deepcopy`."""
    return x


class StoreWindowTrackCounter:
    """Kernel rows whose predicted carrier span leaves their cell's store window.

    Counts only: nothing is rejected, so the posterior is unchanged. Rows are kept on the device
    until :meth:`summary`, which synchronises once.
    """

    def __init__(self):
        self.rows_evaluated = 0
        self.rows_outside = 0
        self._cells_outside = set()
        self._pending = []

    def record(self, outside, cell_specials) -> None:
        """Queue one launch: ``outside`` per row, and the special index of each row's cell."""
        xp = get_array_module(outside)
        self.rows_evaluated += int(outside.shape[0])
        # ? -1 marks rows inside their window; special indices are never negative.
        self._pending.append(xp.where(outside, cell_specials, -1))

    def _flush(self) -> None:
        if not self._pending:
            return
        xp = get_array_module(self._pending[0])
        marked = xp.concatenate(self._pending)
        self._pending = []
        outside = marked[marked >= 0]
        self.rows_outside += int(outside.shape[0])
        self._cells_outside.update(asnumpy(xp.unique(outside)).tolist())

    def summary(self) -> str:
        self._flush()
        return (
            f"store-window track check: {self.rows_outside} of {self.rows_evaluated} rows "
            f"outside their window, {len(self._cells_outside)} cells affected"
        )


def _stft_window_copy_kernel():
    """CuPy kernel copying ``nf_dst`` bins per row block from ``src`` into ``dst``, both flat.

    Cell ``k`` writes ``dst[dst_rows[k], l, f] = src[src_rows[k], l, starts[k] + f]``, where ``l``
    runs over the ``lead`` (channel x time) entries of one cell.
    """
    kernel = getattr(_stft_window_copy_kernel, "_kernel", None)
    if kernel is None:
        kernel = cp.ElementwiseKernel(
            "raw T src, raw int64 src_rows, raw int64 dst_rows, raw int64 starts, "
            "int64 lead, int64 nf_src, int64 nf_dst",
            "raw T dst",
            """
            const long long block = lead * nf_dst;
            const long long cell = i / block;
            const long long row_offset = (i % block) / nf_dst;
            const long long bin = (i % block) % nf_dst;
            dst[(dst_rows[cell] * lead + row_offset) * nf_dst + bin] =
                src[(src_rows[cell] * lead + row_offset) * nf_src + starts[cell] + bin];
            """,
            "lat_stft_window_copy",
        )
        _stft_window_copy_kernel._kernel = kernel
    return kernel


def _stride_or_index(xp, rows: np.ndarray):
    """A slice when ``rows`` is one ascending stride, so indexing a shard makes a view; else an index array."""
    if rows.shape[0] == 1:
        return slice(int(rows[0]), int(rows[0]) + 1)
    steps = np.diff(rows)
    if steps[0] > 0 and bool((steps == steps[0]).all()):
        return slice(int(rows[0]), int(rows[-1]) + 1, int(steps[0]))
    return xp.asarray(rows)


class BandScheduler:
    """Staged loading of (temp, walker, band) cells through the sub-band buffer.

    The proposal loop runs one source-pick per active cell per round. This
    object owns the bookkeeping the loop needs:

    * which cells (packed special indices) currently occupy the buffer's
      ``n_subbands`` slots,
    * how many of each cell's sources have been consumed (a cell is finished
      when every one of its sources has been picked exactly once),
    * which buffer slots to swap out for pending cells when their cell
      finishes (:meth:`advance` returns the ``(inds_fill, new_specials)``
      pair that :meth:`SubBandBuffer` loading consumes).

    Cells are ordered by ascending source count so short cells retire early
    and the buffer stays densely packed with work.
    """

    @property
    def xp(self):
        """Array module derived from a stored flag — never the module itself
        (raw module attributes break deepcopy/pickle of containing graphs)."""
        return cp if self._uses_cupy else np

    def __init__(self, special_band_inds, n_subbands, xp=np):
        # Store a flag, not the module (see the ``xp`` property).
        self._uses_cupy = (getattr(xp, "__name__", "numpy") == "cupy")
        uni, counts = xp.unique(special_band_inds, return_counts=True)
        order = xp.argsort(counts)[::-1]
        self.cell_specials = uni[order]
        self.cell_counts = counts[order]
        self.cell_run = xp.zeros_like(self.cell_counts)
        self.n_cells = int(len(uni))
        # lookup table: special index -> cell position (cell_specials order)
        self._lookup_order = xp.argsort(self.cell_specials)
        self._specials_sorted = self.cell_specials[self._lookup_order]

        n_slots = min(int(n_subbands), self.n_cells)
        self.slot_cell = xp.arange(n_slots)
        self.slot_active = xp.ones(n_slots, dtype=bool)
        self._next_cell = n_slots

    def _cells_of(self, specials):
        """Map special indices to cell positions."""
        pos = self.xp.searchsorted(self._specials_sorted, specials, side="left")
        return self._lookup_order[pos]

    @property
    def n_slots(self) -> int:
        return len(self.slot_cell)

    @property
    def slot_specials(self):
        """Special index per buffer slot (retired slots keep their last cell)."""
        return self.cell_specials[self.slot_cell]

    @property
    def active_slot_specials(self):
        """Special indices of the slots still doing work."""
        return self.slot_specials[self.slot_active]

    def any_active(self) -> bool:
        return bool(self.xp.any(self.slot_active))

    def record_picks(self, picked_specials) -> None:
        """Count one consumed source for each cell that got a pick."""
        self.cell_run[self._cells_of(picked_specials)] += 1

    def advance(self):
        """Retire finished slots and stage pending cells into them.

        Returns ``(inds_fill, new_specials)``: the buffer slot positions to
        repack and the special indices of the cells to load there. Slots
        with no pending replacement are deactivated.
        """
        finished = self.slot_active & (
            self.cell_run[self.slot_cell] >= self.cell_counts[self.slot_cell]
        )
        n_finished = int(finished.sum())
        if n_finished == 0:
            return self.xp.zeros(0, dtype=int), self.xp.zeros(0, dtype=int)

        n_pending = self.n_cells - self._next_cell
        n_replace = min(n_finished, n_pending)
        finished_slots = self.xp.arange(self.n_slots)[finished]

        inds_fill = finished_slots[:n_replace]
        new_cells = self._next_cell + self.xp.arange(n_replace)
        self.slot_cell[inds_fill] = new_cells
        self._next_cell += n_replace

        # slots beyond the replacements retire
        self.slot_active[finished_slots[n_replace:]] = False
        return inds_fill, self.cell_specials[new_cells]


class _ShardHolderView:
    """Single-shard holder view over one GPU split of a multi-shard ACA.

    The gbgpu FD and WDM band engines are single-shard by contract: they
    consume ``holder.linear_data_arr[0]`` / ``linear_psd_arr[0]``, index rows
    by an intra-buffer ``data_index``, and cache pointer-bound bindings on the
    holder. This view presents ONE split of a multi-shard
    :class:`~lisatools.analysiscontainer.AnalysisContainerArray` (a
    :class:`SubBandBuffer` or the parent residual ACA) through exactly that
    protocol:

    * ``linear_data_arr`` / ``linear_psd_arr`` are one-element lists holding
      the owning split's live buffer (zero-copy);
    * ``acs_total_entries`` is the split's row count -- engine row indices
      are INTRA-shard (:class:`_RoutedBandEngine` translates);
    * ``min_freq_inds`` / ``start_freq_ind`` / ``slab_min_f`` are persistent
      per-shard stores refreshed IN PLACE from the parent
      (:meth:`refresh_row_metadata`) so the engines' pointer-binding contract
      survives cell swaps;
    * everything else (settings, df, xp, ...) delegates to the parent.

    Engine bindings (``_gb_fd_binding``) cache on this object, so a view must
    live exactly as long as its shard buffers: the router stores views on the
    holder itself (``holder._shard_holder_views``), which dies with the holder
    at proposal teardown (memory-lifecycle rule).

    .. note::
       The STFT engine does NOT use this view -- it shards internally via
       ``STFTBandLikelihoodEngine._split_plan`` and needs the real
       multi-shard ACA (it indexes ``cpp_splits`` alongside
       ``linear_data_arr``). See :func:`make_routed_band_engine`.
    """

    def __init__(self, parent, split_index: int):
        self._parent = parent
        self._split = int(split_index)
        rows = np.asarray(asnumpy(parent.gpu_splits[self._split]), dtype=int)
        self._rows = rows
        self.acs_total_entries = int(rows.shape[0])
        self.device = (
            None if parent.gpus is None else int(parent.gpus[self._split])
        )
        self.gpus = None if parent.gpus is None else [self.device]
        self.gpu_splits = [np.arange(rows.shape[0])]
        self.split_map = np.zeros(rows.shape[0], dtype=int)
        self.gpu_map = (
            np.zeros(rows.shape[0], dtype=int)
            if self.device is None
            else np.full(rows.shape[0], self.device, dtype=int)
        )
        # Intra-shard row id per row, i.e. the identity on a single-shard
        # view. Mirrors AnalysisContainerArray.ac_to_intra so anything that
        # resolves intra-shard positions off the holder keeps working.
        self.ac_to_intra = np.arange(rows.shape[0], dtype=np.int32)
        # * Both tables on this shard's own device, as the parent ACA carries them: a consumer
        # * gathers its kernel index arrays from them inside the shard's context.
        with device_context(parent.xp, self.device):
            self.split_map_by_split = [parent.xp.asarray(self.split_map)]
            self.ac_to_intra_by_split = [parent.xp.asarray(self.ac_to_intra)]
        self._min_freq_inds_view = None
        self._start_freq_ind_view = None
        self._slab_min_f_view = None
        self.refresh_row_metadata()

    @property
    def rows(self):
        """Global row ids owned by this shard (ascending)."""
        return self._rows

    @property
    def linear_data_arr(self):
        return [self._parent.linear_data_arr[self._split]]

    @property
    def linear_psd_arr(self):
        return [self._parent.linear_psd_arr[self._split]]

    @property
    def data_shaped(self):
        return [self._parent.data_shaped[self._split]]

    @property
    def psd_shaped(self):
        return [self._parent.psd_shaped[self._split]]

    @property
    def cpp_splits(self):
        """This shard's computation group, as a ONE-element list.

        MUST be explicit for the same reason as :attr:`slab_min_f`:
        ``__getattr__`` would forward the parent's FULL split list, so a view
        that is single-shard in every other respect would advertise
        ``len(cpp_splits) > 1``. The live consumer is
        ``STFTGBComputations._resolve_info_group``, which reads
        ``len(cpp_splits)`` to decide whether it is looking at a single split
        and raises ``NotImplementedError`` otherwise -- so the delegated form
        makes :meth:`_RoutedBandEngine.route_information_matrix` fail on
        exactly the multi-shard holders it exists to serve.

        Absent on the parent (FD / WDM holders carry no computation groups)
        -> ``None``, matching ``getattr(holder, "cpp_splits", None)`` at the
        call sites.
        """
        splits = getattr(self._parent, "cpp_splits", None)
        return None if splits is None else [splits[self._split]]

    @property
    def xp(self):
        return self._parent.xp

    @property
    def min_freq_inds(self):
        return self._min_freq_inds_view

    @property
    def start_freq_ind(self):
        return self._start_freq_ind_view

    @property
    def slab_min_f(self):
        """Per-slot narrow-slab layer origins SLICED to this shard's rows.

        MUST be an explicit property: ``__getattr__`` delegation would hand
        back the parent's **global-slot** array while every index the engines
        pass is **intra-shard** -- so a source in shard-1 row 0 would be
        folded against buffer slot 0's slab origin instead of its own.

        ``None`` on this branch in practice: the per-band narrow-slab
        machinery (``band_slab_Nf`` / ``recommend_band_slab_layers``, dev's
        ``dd23c5b``) is not ported here, so no holder carries ``slab_min_f``.
        Kept because it costs nothing and because getting this wrong is
        silent -- when the slab work lands, the correct slice is already in
        place rather than needing to be rediscovered.

        ``band_slab_Nf`` needs no such override: it is a scalar extent shared
        by every slab, hence shard-invariant, and keeps delegating.
        """
        return self._slab_min_f_view

    def refresh_row_metadata(self) -> None:
        """Re-slice per-row metadata from the parent.

        Updates the persistent per-shard ``min_freq_inds`` / ``slab_min_f``
        stores IN PLACE (the FD binding holds a pointer to the former)
        instead of rebinding.
        """
        xp = self._parent.xp
        starts = getattr(self._parent, "min_freq_inds", None)
        if starts is None:
            self._min_freq_inds_view = None
        else:
            vals_host = np.ascontiguousarray(
                np.asarray(asnumpy(starts))[self._rows].astype(np.int32)
            )
            with device_context(xp, self.device):
                if (
                    self._min_freq_inds_view is not None
                    and self._min_freq_inds_view.shape == vals_host.shape
                ):
                    self._min_freq_inds_view[...] = xp.asarray(vals_host)
                else:
                    self._min_freq_inds_view = xp.ascontiguousarray(
                        xp.asarray(vals_host)
                    )
        sfi = getattr(self._parent, "start_freq_ind", None)
        if sfi is None:
            self._start_freq_ind_view = None
        else:
            arr = np.asarray(asnumpy(sfi))
            self._start_freq_ind_view = arr[self._rows] if arr.ndim else arr
        # Narrow per-band slab origins: one value PER BUFFER SLOT, so the
        # shard's view must carry its own rows' values in intra-shard order
        # (see the ``slab_min_f`` property). Refreshed in place like
        # ``min_freq_inds`` so a cell swap on the parent reaches every view.
        slab = getattr(self._parent, "slab_min_f", None)
        if slab is None:
            self._slab_min_f_view = None
        else:
            slab_host = np.ascontiguousarray(
                np.asarray(asnumpy(slab))[self._rows].astype(np.int32)
            )
            with device_context(xp, self.device):
                if (
                    self._slab_min_f_view is not None
                    and self._slab_min_f_view.shape == slab_host.shape
                ):
                    self._slab_min_f_view[...] = xp.asarray(slab_host)
                else:
                    self._slab_min_f_view = xp.ascontiguousarray(
                        xp.asarray(slab_host)
                    )

    def __len__(self) -> int:
        # The engines read ``len(holder)`` as the shard's row (cell) count
        # -> ``num_data``/``num_noise`` for the flat single-shard buffer.
        # MUST be an explicit dunder: ``len()`` resolves ``__len__`` on the
        # TYPE, bypassing ``__getattr__`` delegation, so a parent-forwarded
        # ``__len__`` is never seen. Mirrors AnalysisContainerArray.__len__
        # (== the number of containers on this split).
        return int(self.acs_total_entries)

    def __getattr__(self, name):
        # Guard dunder/underscore probing (deepcopy/pickle safety rule);
        # delegate the public long tail (settings, df, nchannels, ...).
        # NOTE: implicitly-invoked dunders (len(), iter(), ...) resolve on
        # the type and never reach here -- define each one explicitly above.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._parent, name)


class _RoutedBandEngine:
    """Multi-shard router in front of a single-shard band likelihood engine.

    Wraps a :func:`gbgpu.gb_likelihood.make_band_likelihood_engine` product.
    Single-shard holders pass straight through (no overhead). For multi-shard
    holders each call's rows are partitioned by owning split
    (``holder.split_map``), run per shard inside the owning device context
    against a persistent :class:`_ShardHolderView`, and the outputs are
    reassembled full-length on the caller's device. Cross-shard movement is
    host-routed (no P2P), matching the ACA conventions. Per-launch
    host->device upload of wrapper structs (LAT-wide convention) keeps kernel
    config device-local under each context.

    The GB comps a shard's kernels read (chunk geometry, WDM window,
    ``OrbitsWrap`` / ``TDIConfigWrap`` pointer fields, and under sig-het the
    whole heterodyne reference stash) are allocated once, on the device
    current at variant-build time. A shard launching on a different device
    would dereference them across the PCIe link -- a silent peer-access tax
    with P2P, an illegal access without it -- and, for sig-het, would share
    ONE reference stash, ONE slot->reference map and ONE ``_in_model`` flag
    between shards whose slot ids both start at zero. Both are closed by
    per-device replicas: ``engine_factory`` (supplied by
    :func:`make_routed_band_engine`) rebuilds the whole engine around
    device-local comps for any shard whose device differs from the
    prototype's, and the raw-comp class methods resolve the same replica
    through :meth:`_comp_for`. The prototype's own device reuses the existing
    engine object, so single-shard / primary-shard behaviour is unchanged and
    allocates nothing.
    """

    #: fill_template kwargs holding one value PER BUFFER SLOT -- sliced to the
    #: shard's rows so intra-shard indexing stays aligned.
    _PER_SLOT_KWARGS = ("slab_min_f",)

    def __init__(self, engine, engine_factory=None):
        self._engine = engine
        # device -> engine replica, populated lazily on the first multi-shard
        # call that lands on a non-prototype device. The prototype's device
        # maps to ``engine`` ITSELF (never a copy), so ``len(gpus) <= 1``
        # returns the same object and allocates nothing.
        self._engine_factory = engine_factory
        self._engine_by_device = {}

    @property
    def wrapped_engine(self):
        """The underlying single-shard engine."""
        return self._engine

    @property
    def device_engines(self) -> dict:
        """``{device: engine replica}`` built so far (diagnostics/tests)."""
        return self._engine_by_device

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._engine, name)

    # ---------------- per-device comp / engine replicas ----------------

    @staticmethod
    def _comp_build_device(comp):
        """The CUDA device a GB comp's buffers were allocated on, or None.

        ``_build_device`` is recorded by ``WDMComputationsBase.__init__``,
        ``GBFDComputations.__init__`` and ``STFTGBComputations.__init__``. The
        sig-het wrapper records none of its own -- ``for_band_engine`` runs in
        the same device context as its chunked delegate -- so it reports the
        delegate's. The final fallback reads residency straight off a known
        device buffer, which keeps the answer meaningful for a comp built
        before the recording existed.
        """
        dev = getattr(comp, "_build_device", None)
        if dev is None:
            dev = getattr(getattr(comp, "chunked", None), "_build_device", None)
        if dev is None:
            dev = getattr(getattr(comp, "wdm_window", None), "device", None)
            dev = getattr(dev, "id", None)
        return None if dev is None else int(dev)

    @classmethod
    def _engine_comp(cls, engine):
        """The GB comp an engine scores through (WDM, FD or STFT), or None."""
        for attr in ("gb_comps", "gb_fd_comp", "gb_stft_comp"):
            comp = getattr(engine, attr, None)
            if comp is not None:
                return comp
        return None

    @classmethod
    def _primary_device(cls, comp, holder):
        """Device whose shard reuses ``comp`` unchanged.

        The comp's OWN build device when it recorded one -- not blindly
        ``gpus[0]``: a comp constructed before the run pinned its main device
        lives on device 0 even when ``gpus=[2, 3]``, and keying on the
        recorded value replicates for every shard (correct) instead of
        handing device-0 pointers to the ``gpus[0]`` shard (wrong).
        """
        dev = None if comp is None else cls._comp_build_device(comp)
        if dev is not None:
            return dev
        gpus = getattr(holder, "gpus", None)
        return None if not gpus else int(gpus[0])

    @classmethod
    def _assert_comp_device(cls, comp, view):
        """Fail loudly when a shard is about to launch on foreign buffers.

        Cheap permanent guard: the moment a new comp-level device buffer is
        added without a matching replica path, this turns "mysteriously slow,
        or an illegal access on a non-P2P node" into a message that names the
        fix. A comp that records nothing (CPU, or an unrecognised comp type)
        is skipped rather than guessed at.
        """
        dev = getattr(view, "device", None)
        if comp is None or dev is None:
            return
        build_dev = cls._comp_build_device(comp)
        if build_dev is not None and build_dev != int(dev):
            raise RuntimeError(
                f"GB comp {type(comp).__name__} holds buffers on device "
                f"{build_dev} but this shard launches on device {dev}: the "
                "kernel would read across devices (a silent P2P tax, or an "
                "illegal access on a node without peer access). A per-device "
                "comp replica is needed -- see device_local_gb_comp in "
                "lisatools.utils.devicereplicas."
            )

    @classmethod
    def _comp_for(cls, comp, holder, view):
        """The device-local replica of ``comp`` for this shard's device."""
        dev = getattr(view, "device", None)
        if comp is None or dev is None:
            return comp
        out = device_local_gb_comp(
            comp, holder.xp, int(dev), cls._primary_device(comp, holder)
        )
        cls._assert_comp_device(out, view)
        return out

    def _engine_for(self, holder, view):
        """The likelihood engine this shard must run on.

        The prototype's device (and any holder/engine without the metadata to
        do better) gets ``self._engine`` itself. Every other device gets one
        cached replica built through ``engine_factory`` -- which rebuilds the
        engine around device-local comps -- so the shard's kernels, its
        coefficient stash and its ``_in_model`` state are all its own.
        """
        dev = getattr(view, "device", None)
        if dev is None or self._engine_factory is None:
            return self._engine
        primary = self._primary_device(self._engine_comp(self._engine), holder)
        if primary is not None and int(dev) == int(primary):
            return self._engine
        engine = self._engine_by_device.get(int(dev))
        if engine is None:
            engine = self._engine_factory(int(dev), primary)
            self._engine_by_device[int(dev)] = engine
        self._assert_comp_device(self._engine_comp(engine), view)
        return engine

    # ---------------- shard bookkeeping ----------------

    @staticmethod
    def _is_multi(holder) -> bool:
        return len(holder.linear_data_arr) > 1

    @staticmethod
    def _shard_views(holder):
        views = getattr(holder, "_shard_holder_views", None)
        n = len(holder.linear_data_arr)
        if views is None or len(views) != n:
            views = [_ShardHolderView(holder, s) for s in range(n)]
            holder._shard_holder_views = views
        else:
            for v in views:
                v.refresh_row_metadata()
        return views

    @staticmethod
    def _partition(holder, data_index, noise_index=None):
        """Per-shard ``(positions, intra_data, intra_noise)`` partition.

        ``positions`` index into the call's row batch; ``intra_*`` are the
        corresponding intra-shard buffer rows. ``data_index`` and
        ``noise_index`` rows must be co-located on the same shard.
        """
        idx = np.asarray(asnumpy(data_index), dtype=int)
        split_map = np.asarray(asnumpy(holder.split_map), dtype=int)
        intra = np.empty(int(holder.acs_total_entries), dtype=int)
        for rows in holder.gpu_splits:
            rr = np.asarray(asnumpy(rows), dtype=int)
            intra[rr] = np.arange(rr.shape[0])
        nidx = None
        if noise_index is not None:
            nidx = np.asarray(asnumpy(noise_index), dtype=int)
            if not np.array_equal(split_map[idx], split_map[nidx]):
                raise ValueError(
                    "data_index and noise_index rows must live on the same "
                    "shard (cross-shard noise rows are unsupported)."
                )
        parts = []
        for s in range(len(holder.linear_data_arr)):
            pos = np.where(split_map[idx] == s)[0]
            parts.append((
                pos,
                intra[idx[pos]],
                None if nidx is None else intra[nidx[pos]],
            ))
        return parts

    @staticmethod
    def _assemble(num, pieces, default, xp):
        """Host-assemble per-shard outputs into one xp array (row order).

        ``pieces`` is ``[(positions, host_values_or_None), ...]``. Returns
        None when every shard produced None (e.g. ``phase_angle`` without
        phase maximisation).
        """
        first = next(
            (np.asarray(v) for _, v in pieces if v is not None), None
        )
        if first is None:
            return None
        out = np.full(
            (num,) + tuple(first.shape[1:]), default, dtype=first.dtype
        )
        for pos, vals in pieces:
            if vals is None or len(pos) == 0:
                continue
            out[pos] = np.asarray(vals)
        return xp.asarray(out)

    def _mirror_engine_outputs(self):
        """Refresh routed-output attrs from the wrapped engine after a
        passthrough call so stale routed values never shadow them."""
        for name in ("d_h_out", "h_h_out", "phase_angle", "kept_out"):
            if hasattr(self._engine, name):
                setattr(self, name, getattr(self._engine, name))

    # ---------------- routed engine protocol ----------------

    def fill_template(self, holder, params_phys, params_index, N_vals, *,
                      factor, waveform_kwargs, **kwargs):
        if not self._is_multi(holder):
            return self._engine.fill_template(
                holder, params_phys, params_index, N_vals,
                factor=factor, waveform_kwargs=waveform_kwargs, **kwargs)
        xp = holder.xp
        views = self._shard_views(holder)
        parts = self._partition(holder, params_index)
        params_host = asnumpy(params_phys)
        N_host = None if N_vals is None else asnumpy(N_vals)
        slot_kwargs_host = {
            k: np.asarray(asnumpy(kwargs[k]))
            for k in self._PER_SLOT_KWARGS
            if kwargs.get(k) is not None
        }
        for view, (pos, intra, _) in zip(views, parts):
            if pos.shape[0] == 0:
                continue
            kw_s = dict(kwargs)
            engine = self._engine_for(holder, view)
            with device_context(xp, view.device):
                for k, host_vals in slot_kwargs_host.items():
                    kw_s[k] = xp.asarray(host_vals[view.rows])
                engine.fill_template(
                    view, xp.asarray(params_host[pos]), intra,
                    None if N_host is None else xp.asarray(N_host[pos]),
                    factor=factor, waveform_kwargs=waveform_kwargs, **kw_s)

    def get_ll(self, holder, params_phys, *, data_index, noise_index,
               N_vals, phase_maximize=False, waveform_kwargs, **kwargs):
        if not self._is_multi(holder):
            out = self._engine.get_ll(
                holder, params_phys, data_index=data_index,
                noise_index=noise_index, N_vals=N_vals,
                phase_maximize=phase_maximize,
                waveform_kwargs=waveform_kwargs, **kwargs)
            self._mirror_engine_outputs()
            return out
        xp = holder.xp
        views = self._shard_views(holder)
        parts = self._partition(holder, data_index, noise_index)
        num = int(params_phys.shape[0])
        params_host = asnumpy(params_phys)
        N_host = None if N_vals is None else asnumpy(N_vals)
        ll_p, dh_p, hh_p, ang_p, kept_p = [], [], [], [], []
        for view, (pos, intra, intra_noise) in zip(views, parts):
            if pos.shape[0] == 0:
                continue
            engine = self._engine_for(holder, view)
            with device_context(xp, view.device):
                ll_s = engine.get_ll(
                    view, xp.asarray(params_host[pos]),
                    data_index=intra,
                    noise_index=intra if intra_noise is None else intra_noise,
                    N_vals=None if N_host is None else xp.asarray(N_host[pos]),
                    phase_maximize=phase_maximize,
                    waveform_kwargs=waveform_kwargs, **kwargs)
                ll_p.append((pos, asnumpy(ll_s)))
                dh_p.append((pos, asnumpy(engine.d_h_out)))
                hh_p.append((pos, asnumpy(engine.h_h_out)))
                ang = getattr(engine, "phase_angle", None)
                ang_p.append((pos, None if ang is None else asnumpy(ang)))
                kept = getattr(engine, "kept_out", None)
                kept_p.append((pos, None if kept is None else asnumpy(kept)))
        ll = self._assemble(num, ll_p, -1e300, xp)
        if ll is None:
            ll = xp.full(num, -1e300)
        self.d_h_out = self._assemble(num, dh_p, 0.0, xp)
        self.h_h_out = self._assemble(num, hh_p, 0.0, xp)
        self.phase_angle = self._assemble(num, ang_p, 0.0, xp)
        kept_arr = self._assemble(num, kept_p, False, xp)
        self.kept_out = (
            xp.ones(num, dtype=bool) if kept_arr is None else kept_arr
        )
        return ll

    def get_swap_ll(self, holder, params_remove_phys, params_add_phys, *,
                    data_index, noise_index, N_vals, phase_maximize=False,
                    waveform_kwargs, **kwargs):
        if not self._is_multi(holder):
            return self._engine.get_swap_ll(
                holder, params_remove_phys, params_add_phys,
                data_index=data_index, noise_index=noise_index,
                N_vals=N_vals, phase_maximize=phase_maximize,
                waveform_kwargs=waveform_kwargs, **kwargs)
        from gbgpu.gb_likelihood import SwapLLResult

        xp = holder.xp
        views = self._shard_views(holder)
        parts = self._partition(holder, data_index, noise_index)
        num = int(params_add_phys.shape[0])
        rem_host = asnumpy(params_remove_phys)
        add_host = asnumpy(params_add_phys)
        N_host = None if N_vals is None else asnumpy(N_vals)
        fields = ("ll_diff", "d_h_add", "d_h_remove", "hh_add",
                  "hh_remove", "hh_cross", "opt_snr_add", "phase_angle",
                  "kept")
        pieces = {f: [] for f in fields}
        for view, (pos, intra, intra_noise) in zip(views, parts):
            if pos.shape[0] == 0:
                continue
            engine = self._engine_for(holder, view)
            with device_context(xp, view.device):
                res = engine.get_swap_ll(
                    view, xp.asarray(rem_host[pos]), xp.asarray(add_host[pos]),
                    data_index=intra,
                    noise_index=intra if intra_noise is None else intra_noise,
                    N_vals=None if N_host is None else xp.asarray(N_host[pos]),
                    phase_maximize=phase_maximize,
                    waveform_kwargs=waveform_kwargs, **kwargs)
                for f in fields:
                    v = getattr(res, f)
                    pieces[f].append((pos, None if v is None else asnumpy(v)))
        defaults = dict(ll_diff=-1e300, opt_snr_add=0.0, kept=False)
        out = {}
        for f in fields:
            out[f] = self._assemble(num, pieces[f], defaults.get(f, 0.0), xp)
        if out["ll_diff"] is None:
            out["ll_diff"] = xp.full(num, -1e300)
        if out["opt_snr_add"] is None:
            out["opt_snr_add"] = xp.zeros(num)
        if out["kept"] is None:
            out["kept"] = xp.zeros(num, dtype=bool)
        return SwapLLResult(**out)

    def setup_in_model(self, holder, params_phys, data_index, N_vals=None):
        """Route the per-shard in-model reference build (sig-het).

        A truthy return means the comp built heterodyne references: a
        coefficient stash, a slot->reference map and an ``_in_model`` flag,
        all held on the comp and all indexed by INTRA-shard slot ids. Two
        shards therefore need two comps -- their slot ids both start at zero,
        so a shared comp would have the second shard's build silently PATCH
        the first shard's references (the ``_in_model`` flag makes the second
        call take the mid-block patch branch), and every subsequent
        ``get_ll`` would resolve references through the wrong map. With
        ``engine_factory`` supplied each shard resolves to its own engine and
        its own comp, so each takes the fresh-build branch against its own
        residual slabs. Without one, the collision is refused loudly rather
        than computed wrongly.
        """
        if not self._is_multi(holder):
            return self._engine.setup_in_model(
                holder, params_phys, data_index, N_vals=N_vals)
        xp = holder.xp
        views = self._shard_views(holder)
        parts = self._partition(holder, data_index)
        params_host = asnumpy(params_phys)
        N_host = None if N_vals is None else asnumpy(N_vals)
        built_on = set()
        for view, (pos, intra, _) in zip(views, parts):
            if pos.shape[0] == 0:
                continue
            engine = self._engine_for(holder, view)
            with device_context(xp, view.device):
                ret = engine.setup_in_model(
                    view, xp.asarray(params_host[pos]), intra,
                    N_vals=None if N_host is None else xp.asarray(N_host[pos]))
            if not ret:
                continue
            if id(engine) in built_on:
                self.clear_in_model()
                raise NotImplementedError(
                    "sig-het in-model references are per-comp state, but two "
                    "shards resolved to the SAME likelihood engine: the "
                    "second shard's build would patch the first's references "
                    "(intra-shard slot ids collide by construction). Build "
                    "the router with engine_factory= so every shard device "
                    "gets its own comp replica, or run the sig-het in-model "
                    "path on a single GPU."
                )
            built_on.add(id(engine))
        return None

    def clear_in_model(self):
        """Clear the in-model reference on EVERY per-device engine.

        Missed fan-out is a silent bug, not an error: a replica that keeps
        ``_in_model`` set makes the next block's first ``setup_in_model`` on
        that device take the mid-block patch branch against a stale slot map.
        """
        for engine in self._engine_by_device.values():
            engine.clear_in_model()
        return self._engine.clear_in_model()

    def _route_matrix(self, method_name, holder, params_phys, *, data_index,
                      noise_index, N_vals, **kwargs):
        """Shared row-wise routing for matrix-valued outputs (grad/hessian)."""
        if not self._is_multi(holder):
            return getattr(self._engine, method_name)(
                holder, params_phys, data_index=data_index,
                noise_index=noise_index, N_vals=N_vals, **kwargs)
        xp = holder.xp
        views = self._shard_views(holder)
        parts = self._partition(holder, data_index, noise_index)
        num = int(params_phys.shape[0])
        params_host = asnumpy(params_phys)
        N_host = None if N_vals is None else asnumpy(N_vals)
        pieces = []
        for view, (pos, intra, intra_noise) in zip(views, parts):
            if pos.shape[0] == 0:
                continue
            method = getattr(self._engine_for(holder, view), method_name)
            with device_context(xp, view.device):
                out_s = method(
                    view, xp.asarray(params_host[pos]),
                    data_index=intra,
                    noise_index=intra if intra_noise is None else intra_noise,
                    N_vals=None if N_host is None else xp.asarray(N_host[pos]),
                    **kwargs)
                pieces.append((pos, asnumpy(out_s)))
        return self._assemble(num, pieces, 0.0, xp)

    def get_ll_grad(self, holder, params_phys, *, data_index, noise_index,
                    N_vals, **kwargs):
        return self._route_matrix(
            "get_ll_grad", holder, params_phys, data_index=data_index,
            noise_index=noise_index, N_vals=N_vals, **kwargs)

    def hessian(self, holder, params_phys, *, data_index, noise_index,
                N_vals, **kwargs):
        return self._route_matrix(
            "hessian", holder, params_phys, data_index=data_index,
            noise_index=noise_index, N_vals=N_vals, **kwargs)

    @classmethod
    def route_information_matrix(cls, comp, holder, params_phys, *, inds,
                                 noise_index, **swap_kwargs):
        """Route ``comp.information_matrix`` per shard.

        The Fisher/information matrix is computed on the RAW GB comp
        (``gb_wdm_comp`` / ``gb_fd_comp`` / ``gb_stft_comp``) for the proposal
        Cholesky, not on the wrapped likelihood engine -- so it can't go
        through the instance router and needs its own entry point. Each
        binary's matrix depends only on its walker's PSD (``noise_index``; the
        data slab is irrelevant, per ``information_matrix``), so partition
        binaries by the owning shard of their walker, compute per shard
        against a persistent :class:`_ShardHolderView` inside the owning
        device context, and reassemble the ``(num_bin, nd, nd)`` stack on the
        caller's device. Single-shard holders pass straight through.

        Each shard runs against its own device-local comp replica
        (:meth:`_comp_for`), so the kernel never dereferences another
        device's chunk geometry / window / wrap pointers.
        """
        if not cls._is_multi(holder):
            return comp.information_matrix(
                params_phys, holder, inds=inds,
                noise_index=noise_index, **swap_kwargs)
        xp = holder.xp
        views = cls._shard_views(holder)
        # info matrix weights by noise only -> data_index == noise_index.
        parts = cls._partition(holder, noise_index, noise_index)
        params_host = np.atleast_2d(asnumpy(params_phys))
        num = int(params_host.shape[0])
        # * Launch every shard before collecting any, so the shard kernels overlap. Safe because a
        # * shard reads only host arrays uploaded inside its own device context, never caller memory.
        launched = []
        for view, (pos, intra, intra_noise) in zip(views, parts):
            if pos.shape[0] == 0:
                continue
            comp_s = cls._comp_for(comp, holder, view)
            with device_context(xp, view.device):
                out_s = comp_s.information_matrix(
                    xp.asarray(params_host[pos]), view, inds=inds,
                    noise_index=intra if intra_noise is None else intra_noise,
                    **swap_kwargs)
            launched.append((pos, view.device, out_s))
        pieces = []
        for pos, device, out_s in launched:
            with device_context(xp, device):
                pieces.append((pos, asnumpy(out_s)))
        return cls._assemble(num, pieces, 0.0, xp)


def make_routed_band_engine(basis_settings, *, xp, gb_wdm_comp=None,
                            gb_fd_comp=None, gb_stft_comp=None,
                            **engine_kwargs):
    """Build the shard-safe band likelihood engine for a holder.

    Returns exactly the
    :func:`gbgpu.gb_likelihood.make_band_likelihood_engine` product the two
    construction sites (:class:`SubBandBuffer` and the move-level parent-ACA
    engine in ``gbspecialstretch``) built before, wrapped in a
    :class:`_RoutedBandEngine` for the domains that need one. Single-GPU
    behaviour is unchanged either way: the router's fast path is a straight
    passthrough on single-shard holders.

    **The STFT engine is returned unwrapped.** ``STFTBandLikelihoodEngine``
    already shards internally -- ``_split_plan`` partitions rows by
    ``split_map``, enters each owning ``device_context`` and indexes
    ``linear_data_arr[s]`` with intra-shard ids -- so it needs the REAL
    multi-shard ACA. Wrapping it would hand it a :class:`_ShardHolderView`
    whose ``linear_data_arr`` is 1-element while ``cpp_splits`` still
    delegates to the parent's full list, and its own per-split loop would
    then index past the end. Its device-locality comes from the per-split
    comp replica resolved inside that loop, not from this wrapper.

    The ``engine_factory`` closure rebuilds the SAME engine around per-device
    comp replicas (:func:`lisatools.utils.devicereplicas.device_local_gb_comp`)
    the first time a shard lands on a device other than the comps' own --
    giving that shard device-local chunk geometry / window / orbit + TDI
    wraps, and, under sig-het, its own heterodyne reference stash.

    Replicas are module-cached and allocate-once; they deliberately do NOT
    follow the holder's proposal lifetime (rebuilding a ``GBTDIonTheFly`` per
    proposal would be ruinous) and never reach the settings tree.
    """
    from gbgpu.gb_likelihood import make_band_likelihood_engine

    prototype = make_band_likelihood_engine(
        basis_settings, gb_wdm_comp=gb_wdm_comp, gb_fd_comp=gb_fd_comp,
        gb_stft_comp=gb_stft_comp, **engine_kwargs)

    if isinstance(basis_settings, STFTSettings):
        return prototype

    def _engine_factory(device, primary):
        with device_context(xp, device):
            return make_band_likelihood_engine(
                basis_settings,
                gb_wdm_comp=device_local_gb_comp(
                    gb_wdm_comp, xp, device, primary),
                gb_fd_comp=device_local_gb_comp(
                    gb_fd_comp, xp, device, primary),
                gb_stft_comp=gb_stft_comp,
                **engine_kwargs,
            )

    return _RoutedBandEngine(prototype, engine_factory=_engine_factory)


class SubBandBuffer(AnalysisContainerArray, LISAToolsParallelModule):
    """Per-(temp, walker, band) scratch buffers for the GB special moves.

    One :class:`AnalysisContainer` per active cell, all owned directly by
    this object (it *is* the :class:`AnalysisContainerArray`): the linear
    data buffer holds each cell's residual window, the linear PSD buffer the
    matching inverse-PSD slice. GB sources are written into / removed from
    the residual through the domain-aware likelihood engine
    (:meth:`add_sources_to_band_buffer` / :meth:`remove_sources_from_band_buffer`),
    so the inner MCMC loop avoids reallocating large arrays each iteration.

    The most-used members are:

    - ``special_indices_unique`` / ``special_indices_unique_sort``: lookup
      tables that map a per-source ``special_index`` back into the buffer
      ordering.
    - ``params_interest``: parameters of GBs that participate in the move.

    Sign convention: each cell buffer holds the **residual**
    ``data - sum(templates)``. Removing a source from the model therefore
    means *adding* its template back into the buffer (``factor=+1``);
    adding a source to the model subtracts it (``factor=-1``).
    """

    stft_store_windows = None
    track_counter = None
    _stft_cell_width_value = None

    @property
    def xp(self) -> Union[ModuleType, numpy, cupy]:
        """Active array module (NumPy or CuPy) for this buffer.

        Overrides the ``gpus``-based :class:`AnalysisContainerArray`
        property: the buffer's backend decides the array module, so a
        CUDA-backend run on the current device (``gpus=None``) still
        allocates on the GPU.
        """
        return self.backend.xp

    @property
    def df(self):
        """Frequency spacing used for band-index math.

        FD: the FD bin width. WDM: ``layer_df`` so ``band_edges / df``
        yields WDM layer indices. Overrides the read-only
        :class:`AnalysisContainerArray` property (which derives ``df``
        from ``f_arr``).
        """
        return self._df

    @classmethod
    def supported_backends(cls):
        """List the GPU backend names this buffer supports."""
        return ["lisatools_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    def get_index(self, special_inds_test):
        """Map a special-index test value to its position inside the buffer."""
        xp = get_array_module(self.special_indices_unique)
        now_index = (
            self.special_indices_unique_sort[
                xp.searchsorted(
                    self.special_indices_unique[self.special_indices_unique_sort],
                    special_inds_test,
                    side="right",
                )
                - 1
            ]
        ).astype(xp.int32)
        return now_index

    def __init__(
        self,
        is_rj,
        nwalkers,
        gb,
        band_edges,
        band_N_vals,
        unique_band_combos,
        params_interest,
        num_bands_now,
        nchannels,
        data_length,
        special_indices_unique,
        transform_fn,
        waveform_kwargs,
        df,
        sources_now_map,
        sources_inject_now_map,
        special_band_inds,
        opt_snr_rej_samp_limit=5.0,
        force_backend="gpu",
        use_template_arr=False,
        basis_settings: Optional[DomainSettingsBase] = None,
        gb_wdm_comp=None,
        gb_fd_comp=None,
        gb_stft_comp=None,
        stft_store_windows=None,
        track_counter: Optional[StoreWindowTrackCounter] = None,
        *args,
        **kwargs,
    ):
        self.force_backend = force_backend
        LISAToolsParallelModule.__init__(self, force_backend=force_backend)
        assert self.backend.name.split("_")[-1] == gb.backend.name.split("_")[-1]
        self.gb = gb
        # Domain computation objects (gbgpu.gbcomps.GBWDMComputations /
        # GBFDComputations prototype / STFTGBComputations). The one matching
        # ``basis_settings`` is required; the others stay None. The legacy
        # ``gb`` handle is kept only for the info-matrix proposal shaping.
        self.gb_wdm_comp = gb_wdm_comp
        self.gb_fd_comp = gb_fd_comp
        self.gb_stft_comp = gb_stft_comp
        self._df = df
        self.nwalkers = nwalkers
        self.sources_now_map, self.sources_inject_now_map = (
            sources_now_map,
            sources_inject_now_map,
        )
        self.band_edges, self.unique_band_combos = band_edges, unique_band_combos
        self.num_bands = len(self.band_edges) - 1
        self.params_interest = params_interest
        self.num_bands_now, self.nchannels, self.data_length = (
            num_bands_now,
            nchannels,
            data_length,
        )
        # FD store length of one cell window; kept distinct from the ACA's
        # ``data_length`` layout attribute (see :attr:`_fd_store_length`).
        self._fd_store_length_value = data_length
        self.band_N_vals = self.xp.asarray(band_N_vals) if band_N_vals is not None else None
        # TODO: adjust this
        self.edge_buffer = 2000
        self.is_rj = is_rj
        
        if basis_settings is None:
            basis_settings = FDSettings(
                N=self.data_length,
                df=float(self.df) if not hasattr(self.df, "item") else self.df.item(),
            )
        self._basis_settings = basis_settings

        # * STFT cells hold a store window of the parent grid; without windows a cell is the full grid.
        self.stft_store_windows = None
        self.track_counter = None
        if isinstance(basis_settings, STFTSettings):
            self.stft_store_windows = stft_store_windows
            self.track_counter = track_counter if stft_store_windows is not None else None
            if stft_store_windows is None:
                self._stft_cell_width_value = int(basis_settings.NF_active)
            else:
                if gb_stft_comp is not None and stft_store_windows.n_side_bins != gb_stft_comp.n_side_bins:
                    raise ValueError(
                        f"store windows were sized for n_side_bins={stft_store_windows.n_side_bins}, "
                        f"the STFT comp evaluates {gb_stft_comp.n_side_bins}."
                    )
                if stft_store_windows.n_grid_bins != int(basis_settings.NF_active):
                    raise ValueError(
                        f"store windows index a {stft_store_windows.n_grid_bins}-bin grid, the parent "
                        f"has NF_active={int(basis_settings.NF_active)}."
                    )
                self._stft_cell_width_value = stft_store_windows.buffer_width

        self._parent_ind_min = None
        self._parent_stored_len = None
        if (
            isinstance(basis_settings, FDSettings)
            and getattr(basis_settings, "ind_min", None) is not None
            and getattr(basis_settings, "ind_max", None) is not None
        ):
            self._parent_ind_min = int(basis_settings.ind_min)
            self._parent_stored_len = (
                int(basis_settings.ind_max) - int(basis_settings.ind_min) + 1
            )
            if self._fd_store_length_value > self._parent_stored_len:
                logger.warning(
                    "FD cell window (%d bins) exceeds the clipped parent domain "
                    "(%d stored bins); clamping each cell window to the full "
                    "stored band.",
                    self._fd_store_length_value,
                    self._parent_stored_len,
                )
                self._fd_store_length_value = self._parent_stored_len

        self.special_indices_unique = special_indices_unique
        self.transform_fn = transform_fn
        self.waveform_kwargs = waveform_kwargs
        self.opt_snr_rej_samp_limit = opt_snr_rej_samp_limit
        self.use_template_arr = use_template_arr

        self.tdi_channel_setup = self.waveform_kwargs.get("tdi_channel_setup")
        if self.tdi_channel_setup == "XYZ":
            assert self.nchannels == 3
        else:
            assert "A" in self.tdi_channel_setup and "E" in self.tdi_channel_setup
            logger.warning("using AE(T) channels where we assume ortogonality. This may not be sufficient for realistic orbtis.")

        # Build the per-cell AnalysisContainers and initialise *ourselves* as
        # the AnalysisContainerArray that owns them. On the WDM path the ACA
        # layout metadata (``data_length = Nf_active * Nt_active``,
        # ``end_shape``) replaces the FD-style ctor value -- the inherited
        # linear-buffer indexing needs the ACA meaning; nothing on the WDM
        # move path consumes the FD-style value.
        ac_list, aca_kwargs = self._build_band_ac_list()
        AnalysisContainerArray.__init__(self, ac_list, **aca_kwargs)
        self._skip_eager_dd = True
        if self.use_template_arr:
            # Templates mirror the band-buffer layout in a twin ACA so they
            # share the same managed memory region. The per-band sensitivity
            # slot on the template ACA is unused but keeps construction
            # symmetric across the two buffers.
            template_ac_list, template_aca_kwargs = self._build_band_ac_list()
            self._acs_template_buffer = AnalysisContainerArray(
                template_ac_list, **template_aca_kwargs
            )
            self._acs_template_buffer._skip_eager_dd = True
            self._share_template_psd_blocks()

        for _acs in (self, self._acs_template_buffer) if self.use_template_arr else (self,):
            for _ac in _acs.acs.flatten():
                _ac.sens_mat._sens_mat = _ac.sens_mat.invC

        if isinstance(self._basis_settings, STFTSettings):
            self._refresh_stft_split_start_inds()
            if self.use_template_arr:
                # * Same list object: an in-place refill of the starts reaches the twin's kernels too.
                self._acs_template_buffer.stft_split_start_inds = self.stft_split_start_inds

        # psd_shape is exposed for back-compat with downstream consumers that
        # inspect it; it tracks the shape of the per-band PSD view.
        self.psd_shape = (self.num_bands_now,) + self._per_band_sens_shape

        # Build the domain-aware likelihood engine. Dispatch is on
        # ``isinstance(basis_settings, ...)`` -- no string-level mode flag.
        # The engine takes an AnalysisContainerArray at call time, so the
        # buffer's get_swap_ll / get_ll / adjust_sources_in_band_buffer
        # methods don't reach into self.gb (or self.gb_wdm_comp) directly.
        # Persistent per-slot window-start store. Bound BY POINTER into the
        # FD computations clone (FDDomain.start_inds), so it must be updated
        # in place on cell swaps -- never rebound (see the
        # special_indices_unique setter).
        if isinstance(self._basis_settings, FDSettings):
            self._min_freq_inds_store = self.xp.ascontiguousarray(
                self.xp.asarray(self.start_freq_inds), dtype=self.xp.int32
            ).copy()
            if self.use_template_arr:
                # The template twin shares the buffer's per-slot window
                # starts (same array object: in-place updates on cell swaps
                # reach both FD comps clones).
                self._acs_template_buffer.min_freq_inds = self._min_freq_inds_store

        self._likelihood_engine = make_routed_band_engine(
            self._basis_settings,
            xp=self.xp,
            gb=self.gb,
            gb_fd_comp=self.gb_fd_comp,
            gb_wdm_comp=self.gb_wdm_comp,
            gb_stft_comp=self.gb_stft_comp,
            nchannels=self.nchannels,
            tdi_channel_setup=self.tdi_channel_setup,
            df=float(self.df) if not hasattr(self.df, "item") else self.df.item(),
            start_freq_inds=getattr(self, "start_freq_inds", None),
            data_length=self.data_length,
            opt_snr_rej_samp_limit=self.opt_snr_rej_samp_limit,
        )

        # TODO: fix this 4????
        self.special_band_inds = special_band_inds
        assert special_band_inds.shape[0] == self.params_interest.shape[0]
        self.now_index = self.get_index(special_band_inds)

    # ------------------------------------------------------------------
    # Views into the AnalysisContainerArray-backed scratch buffers
    # ------------------------------------------------------------------

    @property
    def acs_buffer(self) -> AnalysisContainerArray:
        """The :class:`AnalysisContainerArray` backing the per-band residual buffers.

        Post Buffer/ACA merge this *is* the buffer object itself; kept for
        back-compat with callers that used ``buffer.acs_buffer``.
        """
        return self

    # ------------------------------------------------------------------
    # Per-band buffer accessors
    # ------------------------------------------------------------------
    # In multi-GPU mode the buffer holds the bands sharded across GPUs
    # (striped by default; see ``_build_band_ac_list``). The shaped
    # accessors below return a :class:`BandView` that lets callers
    # index by global band number; reads/writes route to the owning
    # shard. In single-GPU mode they return the underlying ndarray
    # view directly (no overhead). The ``*_tmp`` flat accessors stay
    # single-GPU-only -- with multi-shard there is no single flat
    # ndarray, so callers should use the engine path (gb_likelihood
    # passes the list-of-shards through ``buffer_aca.linear_data_arr``
    # / ``linear_psd_arr`` directly).

    def _shaped_or_view(self, acs, kind: str):
        """Return either the single-shard reshape (single-GPU) or a BandView (multi-GPU)."""
        if len(acs.linear_data_arr) == 1:
            return acs.data_shaped[0] if kind == "data" else acs.psd_shaped[0]
        return acs.data_shaped_view() if kind == "data" else acs.psd_shaped_view()

    def _flat_or_raise(self, acs, kind: str):
        if len(acs.linear_data_arr) == 1:
            return (
                acs.linear_data_arr[0] if kind == "data" else acs.linear_psd_arr[0]
            )
        raise RuntimeError(
            f"{kind}_buffer_tmp is only valid in single-GPU mode "
            "(multi-GPU buffers are a list of per-GPU shards). Use the "
            "engine path or BandView accessors instead."
        )

    def _share_template_psd_blocks(self) -> None:
        """Point the template twin's inverse-CSD views at THIS buffer's blocks.

        Nothing reads the twin's inverse CSD: :meth:`likelihood` takes ``self.psd_buffer``, and
        the twin only ever receives templates. Its blocks are needed as a valid pointer for the
        twin's C++ domain, so they are shared rather than dropped. That saves one psd-shaped
        block per cell, which on the tempering path is 3 of the 5 blocks a cell held.

        Both ACAs are built from the same cell list, so a cell sits at the same split and the
        same offset in each. Must run BEFORE the twin's ``cpp_splits`` are built, because a
        computation group captures ``linear_psd_arr[split]`` when it is constructed.
        """
        twin = self._acs_template_buffer
        block = int(np.prod(self.shape_sens) * self.data_length)
        twin.linear_psd_arr = self.linear_psd_arr
        for i, ac in enumerate(twin.acs.flatten()):
            split = int(twin.split_map[i])
            intra = int(np.where(twin.gpu_splits[split] == i)[0][0])
            ac.sens_mat.invC = self.linear_psd_arr[split][
                intra * block:(intra + 1) * block
            ].reshape(self.shape_sens + self.end_shape)

    @property
    def band_buffer_tmp(self):
        """Flat per-GPU residual buffer (1D view; single-GPU only)."""
        return self._flat_or_raise(self, "data")

    @property
    def band_buffer(self):
        """Per-band residual buffer indexable by global band id.

        Single-GPU: returns the ``(num_bands_now, nchannels, data_length)``
        reshape directly. Multi-GPU: returns a :class:`BandView` that
        routes per-band reads/writes through the owning shard.
        """
        return self._shaped_or_view(self, "data")

    @property
    def psd_buffer_tmp(self):
        """Flat per-GPU inverse-PSD buffer (1D view; single-GPU only)."""
        return self._flat_or_raise(self, "psd")

    @property
    def psd_buffer(self):
        """Per-band inverse-PSD buffer indexable by global band id.

        Same single-GPU / multi-GPU behaviour as :attr:`band_buffer`.
        """
        return self._shaped_or_view(self, "psd")

    @property
    def template_buffer_tmp(self):
        """Flat per-GPU template buffer (single-GPU only; ``use_template_arr`` True)."""
        return self._flat_or_raise(self._acs_template_buffer, "data")

    @property
    def template_buffer(self):
        """Per-band template buffer indexable by global band id.

        Same single-GPU / multi-GPU behaviour as :attr:`band_buffer`.
        """
        return self._shaped_or_view(self._acs_template_buffer, "data")

    # ------------------------------------------------------------------
    # Domain-aware allocation helpers
    # ------------------------------------------------------------------

    @property
    def basis_settings(self) -> DomainSettingsBase:
        """Parent basis-domain settings driving per-band buffer geometry."""
        return self._basis_settings

    @property
    def _stft_cell_width(self) -> int:
        """Frequency bins one STFT cell holds: its store window, or the whole active grid."""
        if self._stft_cell_width_value is None:
            return int(self._basis_settings.NF_active)
        return int(self._stft_cell_width_value)

    @property
    def _per_band_data_shape(self) -> tuple:
        """Shape of a single band's residual buffer (one AC's data_res_arr)."""
        if isinstance(self._basis_settings, FDSettings):
            return (self.nchannels, self._fd_store_length)
        elif isinstance(self._basis_settings, WDMSettings):
            # First-cut: each per-band buffer covers the FULL WDM active grid
            # (Nf_active layers x Nt_active time pixels). The WDM kernel
            # currently uses a single global [ind_min_f, ind_max_f] rather
            # than per-band offsets, so per-band slicing on the layer axis is
            # a follow-on once the kernel takes per-band layer offsets.
            Nf_active = self._basis_settings.Nf_active
            Nt_active = self._basis_settings.Nt_active
            return (self.nchannels, Nf_active, Nt_active)
        elif isinstance(self._basis_settings, STFTSettings):
            # * A cell holds its store window; the kernels place pixels through the per-cell starts.
            return (
                self.nchannels,
                self._basis_settings.NT,
                self._stft_cell_width,
            )
        else:
            raise NotImplementedError(
                f"Buffer does not support basis domain {type(self._basis_settings).__name__}."
            )

    @property
    def _per_band_sens_shape(self) -> tuple:
        """Shape of a single band's inverse-PSD buffer (one AC's sens_mat.invC)."""
        if isinstance(self._basis_settings, FDSettings):
            if self.tdi_channel_setup == "XYZ":
                return (self.nchannels, self.nchannels, self._fd_store_length)
            return (self.nchannels, self._fd_store_length)
        elif isinstance(self._basis_settings, WDMSettings):
            Nf_active = self._basis_settings.Nf_active
            Nt_active = self._basis_settings.Nt_active
            if self.tdi_channel_setup == "XYZ":
                return (self.nchannels, self.nchannels, Nf_active, Nt_active)
            return (self.nchannels, Nf_active, Nt_active)
        elif isinstance(self._basis_settings, STFTSettings):
            NT = self._basis_settings.NT
            if self.tdi_channel_setup == "XYZ":
                return (self.nchannels, self.nchannels, NT, self._stft_cell_width)
            return (self.nchannels, NT, self._stft_cell_width)
        else:
            raise NotImplementedError(
                f"Buffer does not support basis domain {type(self._basis_settings).__name__}."
            )

    @property
    def _per_band_data_dtype(self):
        """Element dtype for the per-band residual buffer."""
        if isinstance(self._basis_settings, FDSettings):
            return self.xp.complex128
        elif isinstance(self._basis_settings, WDMSettings):
            return self.xp.float64
        elif isinstance(self._basis_settings, STFTSettings):
            return self.xp.complex128
        else:
            raise NotImplementedError(
                f"Buffer does not support basis domain {type(self._basis_settings).__name__}."
            )

    @property
    def _per_band_sens_dtype(self):
        """Element dtype for the per-band inverse-PSD buffer.

        FD is REAL: the gb_fd kernels consume the real part of the inverse
        covariance (Hermitian; the imaginary parts cancel in the quadratic
        forms), matching the FDDomain double* layout.
        """
        if isinstance(self._basis_settings, FDSettings):
            return self.xp.float64
        elif isinstance(self._basis_settings, WDMSettings):
            return self.xp.float64
        elif isinstance(self._basis_settings, STFTSettings):
            # Complex inverse covariance (STFT cross-terms are complex; the
            # C++ STFTDomain wrap takes complex128 invC).
            return self.xp.complex128
        else:
            raise NotImplementedError(
                f"Buffer does not support basis domain {type(self._basis_settings).__name__}."
            )

    def _build_per_band_basis_settings(self) -> DomainSettingsBase:
        """Construct the per-band domain settings used by each per-band AC.

        Each per-band AC's data domain needs a settings object whose
        ``basis_shape_active`` matches the per-band data shape. For FD this
        is a fresh FDSettings sized to the FD store length. For WDM the
        per-band settings share the parent's active grid.
        """
        if isinstance(self._basis_settings, FDSettings):
            return FDSettings(
                N=self._fd_store_length,
                df=float(self.df) if not hasattr(self.df, "item") else self.df.item(),
                force_backend=self._basis_settings.backend_name.split("_", 1)[1],
            )
        elif isinstance(self._basis_settings, WDMSettings):
            # First-cut: per-band WDMSettings matches the parent grid (full
            # WDM active band). A true per-band sliced WDMSettings becomes
            # possible once the WDM kernel takes per-band
            # [ind_min_f, ind_max_f] arrays; until then we share the parent.
            parent = self._basis_settings
            return WDMSettings(
                Nf=parent.Nf,
                Nt=parent.Nt,
                dt=parent.data_dt,
                t0=parent.t0,
                oversample=parent.oversample,
                window=parent.window,
                omega=parent.omega,
                min_freq=parent.ind_min_f * parent.layer_df,
                max_freq=parent.ind_max_f * parent.layer_df,
                min_time=parent.ind_min_t * parent.layer_dt,
                max_time=parent.ind_max_t * parent.layer_dt,
            )
        elif isinstance(self._basis_settings, STFTSettings):
            # The args/kwargs round-trip reconstructs from the RAW min/max_freq
            # inputs, keeping ind_min/ind_max identical to the parent's.
            parent = self._basis_settings
            if self.stft_store_windows is None:
                return type(parent)(*parent.args, **parent.kwargs)
            width = self._stft_cell_width
            cell_settings = type(parent)(
                *parent.args,
                min_freq=max(parent.ind_min - 0.5, 0.0) * parent.df,
                max_freq=(parent.ind_min + width - 0.5) * parent.df,
                force_backend=parent.kwargs["force_backend"],
            )
            if cell_settings.NF_active != width or cell_settings.min_freq != parent.min_freq:
                raise RuntimeError(
                    f"windowed cell settings have NF_active={cell_settings.NF_active} and "
                    f"min_freq={cell_settings.min_freq}; expected {width} and {parent.min_freq}."
                )
            return cell_settings
        else:
            raise NotImplementedError(
                f"Buffer does not support basis domain {type(self._basis_settings).__name__}."
            )

    def _build_band_ac_list(self) -> tuple:
        """Allocate one :class:`AnalysisContainer` per active cell.

        Returns ``(ac_list, aca_kwargs)`` ready to be fed into
        ``AnalysisContainerArray.__init__`` (either on ``self`` or on the
        template twin). Branches on the parent basis domain: FD buffers are
        complex with ``complex_psd=True`` (XYZ CSD support); WDM buffers are
        real-valued slabs over the full active grid.
        """
        per_band_settings = self._build_per_band_basis_settings()
        data_shape = self._per_band_data_shape
        sens_shape = self._per_band_sens_shape
        data_dtype = self._per_band_data_dtype
        sens_dtype = self._per_band_sens_dtype

        is_stft = isinstance(self._basis_settings, STFTSettings)
        parent_group = None
        if is_stft:
            if self.gb_stft_comp is None:
                raise ValueError(
                    "SubBandBuffer(basis_settings=STFTSettings) requires "
                    "gb_stft_comp (a gbgpu.gbcomps.STFTGBComputations "
                    "instance)."
                )
            parent_group = self.gb_stft_comp.stft_comps

        # Multi-GPU at the GB band-tree level: a striped band assignment so
        # consecutive bands land on different GPUs. The BandSorter even/odd
        # within-pass invariant keeps bands in one pass non-overlapping in
        # time-frequency support, so striping is safe. The per-band accessors
        # (band_buffer / psd_buffer / template_buffer) automatically fall back
        # to a single ndarray view for single-GPU runs and return a BandView
        # (multi-shard router) otherwise -- see the accessor block above.
        #
        # Resolved BEFORE the allocation loop because each band's sensitivity
        # backend has to be built for its OWN device (see below).
        gpus_in = getattr(self.gb, "gpus", None) if self.backend.uses_cupy else None
        if gpus_in:
            gpus_in = list(gpus_in)[: max(1, int(self.num_bands_now))]
        gpu_assignment = (
            band_gpu_assignment(self.num_bands_now, list(gpus_in))
            if gpus_in else None
        )
        primary_device = int(gpus_in[0]) if gpus_in else None

        sens_prototypes = {}
        zero_sens_blocks = {}

        ac_list = []
        for _b in range(self.num_bands_now):
            band_device = (
                None if gpu_assignment is None else int(gpu_assignment[_b])
            )
            # Allocate this band's buffers, and build its sensitivity backend,
            # ON THE DEVICE THAT WILL OWN THE BAND. The ACA repacks the data
            # arrays into its per-shard linear buffers, but the sensitivity
            # backend object is kept as-is -- its C++ orbit tables are never
            # migrated.
            with device_context(self.xp, band_device):
                res_data = self.xp.zeros(data_shape, dtype=data_dtype)
                data_domain = per_band_settings.associated_class(
                    res_data, per_band_settings)
                if is_stft:
                    # Unlike the FD/WDM band ACAs (whose engines never touch
                    # cpp_splits), the STFT engine drives the band ACA's own
                    # per-split STFTComputationGroup, and that group's
                    # ``build_cpp_objects`` reconstructs a sensitivity backend
                    # from the split's first AC -- requiring a REAL backend
                    # (``orbits`` + ``kwargs``), not a bare
                    # SensitivityMatrixBase. Clone the parent group's backend
                    # per band and overwrite the buffers with band-local
                    # zeros.
                    #
                    # ``parent_sb.kwargs`` carries the run's SHARED ``orbits``
                    # object, whose C++ tables live on the device it was built
                    # on. Handing that to a band striped onto another device
                    # makes ``build_cpp_objects`` -- and every kernel reached
                    # through this group -- dereference foreign pointers:
                    # measured as an illegal memory access in
                    # ``Detector.cu``, on a node where P2P is ENABLED (the
                    # orbit pointers are not peer-mapped). Swap in the
                    # device-local orbits replica.
                    prototype = sens_prototypes.get(band_device)
                    if prototype is None:
                        parent_sb = parent_group.sensitivity_backend
                        sb_kwargs = dict(parent_sb.kwargs)
                        if band_device is not None and sb_kwargs.get("orbits") is not None:
                            sb_kwargs["orbits"] = device_local_orbits(
                                sb_kwargs["orbits"], self.xp, primary_device)
                        prototype = sens_prototypes[band_device] = type(parent_sb)(**sb_kwargs)
                    sm = copy_module.copy(prototype)
                    sm.data_shape = per_band_settings.basis_shape_active
                else:
                    sm = SensitivityMatrixBase(per_band_settings, skip_inv_det=True)
                zero_sens = zero_sens_blocks.get(band_device)
                if zero_sens is None:
                    zero_sens = zero_sens_blocks[band_device] = self.xp.zeros(
                        sens_shape, dtype=sens_dtype)
                sm.sens_mat = zero_sens
                sm.invC = zero_sens
                sm.channel_shape = sens_shape[
                    : -len(per_band_settings.basis_shape_active)]
                ac_list.append(AnalysisContainer(data_domain, sm))
        aca_kwargs = dict(
            gpus=list(gpus_in) if gpus_in else None,
            # STFT invC is complex128 (see _per_band_sens_dtype); FD/WDM
            # buffers hold the real inverse covariance.
            complex_psd=is_stft,
            gpu_assignment=gpu_assignment,
        )
        if is_stft:
            # The band ACA's per-split STFTComputationGroups (cpp_splits)
            # must carry the same Fresnel configuration as the parent group
            # -- the STFT engine rebinds gb_stft_comp.stft_comps onto exactly
            # those groups per call.
            aca_kwargs["domain_group_kwargs"] = dict(
                tdi_type=parent_group.tdi_type,
                window_alpha=parent_group.window_alpha,
                use_midpoint=parent_group.use_midpoint,
                linear_envelope=parent_group.linear_envelope,
            )
        return ac_list, aca_kwargs

    def update_special_indices(self, new_special_indices, inds_fill=None):
        if inds_fill is None:
            inds_fill = self.xp.arange(self.num_bands_now)

        assert inds_fill.shape[0] == new_special_indices.shape[0]
        _tmp_indices = self.special_indices_unique.copy()
        _tmp_indices[inds_fill] = new_special_indices
        self.special_indices_unique = _tmp_indices

    @property
    def special_indices_unique(self):
        return self._special_indices_unique

    @special_indices_unique.setter
    def special_indices_unique(self, special_indices_unique):
        self._special_indices_unique_sort = self.xp.argsort(special_indices_unique)
        self._special_indices_unique = special_indices_unique

        _temp_inds, _walker_inds, _band_inds = self.get_separate_inds_from_special_index(
            special_indices_unique
        )

        self.unique_band_combos = self.xp.array([_temp_inds, _walker_inds, _band_inds]).T

        if isinstance(self._basis_settings, FDSettings):
            if self.num_bands == 1:
                tmp_buffer_start_index = (self.band_edges[0] / self.df).astype(
                    np.int32
                ) - self.edge_buffer
                if getattr(self, "_parent_ind_min", None) is None:
                    # Legacy full-grid parent: the single window must cover the
                    # whole band plus both edge buffers. (With a frequency-
                    # clipped parent this is un-satisfiable by construction --
                    # the clamp below shifts/shrinks the window instead.)
                    assert tmp_buffer_start_index + self._fd_store_length >= (
                        (self.band_edges[-1] / self.df).astype(np.int32) + self.edge_buffer
                    )
                self.buffer_start_index = self.xp.repeat(
                    tmp_buffer_start_index, self.unique_band_combos.shape[0]
                )

            else:
                self.buffer_start_index = (
                    self.band_edges[self.unique_band_combos[:, 2] - 1] / self.df
                ).astype(np.int32)
                self.buffer_start_index[self.unique_band_combos[:, 2] == 0] = (
                    self.band_edges[0] / self.df
                ).astype(np.int32) - self.edge_buffer
                # Clamp so buffer end never overflows the data range (band_edges[-1])
                max_start = int(self.band_edges[-1] / self.df) - self._fd_store_length
                self.buffer_start_index = np.minimum(self.buffer_start_index, max_start)

            if getattr(self, "_parent_ind_min", None) is not None:
                # Frequency-clipped parent: clamp every window into the stored
                # bin range [ind_min, ind_min + stored_len - window]. Edge cells
                # lose their out-of-domain guard margin (the parent stores
                # nothing there); the engines read placement from
                # ``start_freq_inds`` so a shifted window stays consistent.
                lo = self._parent_ind_min
                hi = max(lo, lo + self._parent_stored_len - self._fd_store_length)
                self.buffer_start_index = self.xp.clip(self.buffer_start_index, lo, hi)

            self.start_freq_inds = self.xp.asarray(self.buffer_start_index.copy().astype(np.int32))
            if hasattr(self, "_min_freq_inds_store"):
                # in-place: the FD comps clone holds a pointer to this array
                self._min_freq_inds_store[:] = self.start_freq_inds
        elif isinstance(self._basis_settings, WDMSettings):
            self.buffer_start_index = self.xp.full(
                self.num_bands_now, self._basis_settings.ind_min_f, dtype=self.xp.int32
            )
            self.start_freq_inds = self.buffer_start_index
        elif isinstance(self._basis_settings, STFTSettings):
            # * Active-bin window start of each slot; 0 everywhere is the full grid.
            if self.stft_store_windows is None:
                starts_host = np.zeros(self.num_bands_now, dtype=np.int32)
            else:
                starts_host = self.stft_store_windows.window_starts(_band_inds, self._stft_cell_width)
            self._stft_start_inds_host = starts_host
            self.buffer_start_index = self.xp.asarray(starts_host)
            self.start_freq_inds = self.buffer_start_index
            if getattr(self, "stft_split_start_inds", None) is not None:
                self._refresh_stft_split_start_inds()
        else:
            raise NotImplementedError(
                f"Buffer does not support basis domain {type(self._basis_settings).__name__}."
            )

        self.frequency_lims = self._compute_frequency_lims()

    def _count_window_tracks(self, params_phys, data_index) -> None:
        """Record the kernel rows whose predicted carrier span leaves their slot's store window.

        The span is f0 + fdot (t - t_ref) over the data, widened by the largest Doppler amplitude and
        the stencil. A window edge that is also a grid edge does not count: the full grid drops those
        pixels too. Carrier jumps near antenna-pattern nulls are not predicted.
        """
        if self.track_counter is None:
            return
        xp = self.xp
        settings = self._basis_settings
        params_phys = xp.atleast_2d(xp.asarray(params_phys))
        data_index = xp.asarray(data_index).astype(xp.int64)
        t_ref = float(self.gb_stft_comp.t_ref)
        time_first = float(settings.t0) - t_ref
        time_last = float(settings.t0) + int(settings.NT) * float(settings.dt) - t_ref
        f0, fdot = params_phys[:, 1], params_phys[:, 2]
        f_first = f0 + fdot * time_first
        f_last = f0 + fdot * time_last
        doppler = 2.0 * np.pi * f0 * (AU_SI / C_SI) / YRSID_SI
        df, f_min = float(settings.df), float(settings.min_freq)
        n_side = int(self.gb_stft_comp.n_side_bins)
        bin_lo = xp.rint((xp.minimum(f_first, f_last) - doppler - f_min) / df) - n_side
        bin_hi = xp.rint((xp.maximum(f_first, f_last) + doppler - f_min) / df) + n_side
        window_lo = self.buffer_start_index[data_index].astype(xp.int64)
        window_hi = window_lo + self._stft_cell_width - 1
        grid_hi = int(settings.NF_active) - 1
        outside = ((bin_lo < window_lo) & (window_lo > 0)) | ((bin_hi > window_hi) & (window_hi < grid_hi))
        self.track_counter.record(outside, self.special_indices_unique[data_index])

    def _refresh_stft_split_start_inds(self) -> None:
        """Copy the slot window starts into the per-split int32 stores the STFT kernels read.

        One store per split, on the split's device and in ``gpu_splits[s]`` order, so a kernel indexes
        it with the intra-split slot. Updated in place after the first build, so the pointer a kernel
        receives never changes over the buffer's life.
        """
        stores = getattr(self, "stft_split_start_inds", None)
        first_build = stores is None
        if first_build:
            stores = []
        for split, rows in enumerate(self.gpu_splits):
            values = np.ascontiguousarray(
                self._stft_start_inds_host[np.asarray(asnumpy(rows), dtype=int)], dtype=np.int32
            )
            device = None if self.gpus is None else int(self.gpus[split])
            with device_context(self.xp, device):
                if first_build:
                    stores.append(self.xp.array(values, dtype=self.xp.int32))
                else:
                    stores[split][...] = self.xp.asarray(values)
        self.stft_split_start_inds = stores

    def _compute_frequency_lims(self):
        lower_f_lim = self.band_edges[
            self.unique_band_combos[:, 2]
        ].copy()
        higher_f_lim = self.band_edges[
            self.unique_band_combos[:, 2] + 1
        ].copy()

        # allow to move over band edge when proposing in-model
        if isinstance(self._basis_settings, FDSettings):
            if self.is_rj and self.band_N_vals is not None:
                lower_f_lim -= self.band_N_vals[self.unique_band_combos[:, 2]] * self.df / 4
                higher_f_lim += self.band_N_vals[self.unique_band_combos[:, 2]] * self.df / 4
        else:
            # WDM and STFT: frequency limits are 1/4th of the current band padded to the band's edges
            # * The STFT store windows are sized for exactly this padding.
            band_width = higher_f_lim - lower_f_lim
            lower_f_lim -= band_width * STFT_F0_LIMIT_FRACTION
            higher_f_lim += band_width * STFT_F0_LIMIT_FRACTION
        return [lower_f_lim, higher_f_lim]

    @property
    def _fd_store_length(self) -> int:
        """FD frequency-bin count of one cell's residual window.

        This is the FD-specific store size handed in at construction. It is
        deliberately kept separate from the inherited ACA ``data_length``
        (which on the WDM path becomes ``Nf_active * Nt_active`` -- the
        linear-buffer stride the inherited packing methods need).
        """
        return self._fd_store_length_value

    @property
    def min_freq_inds(self):
        """Per-slot minimum-frequency index, unified across domains.

        FD: absolute FD bin where each cell's residual window starts
        (``buffer_start_index``). WDM: the start layer of each cell's slab --
        currently the parent grid's ``ind_min_f`` for every slot, until the
        WDM kernel takes per-band layer offsets. The likelihood engines read
        this off the buffer at every call, so it never goes stale across
        band swap-outs.
        """
        if isinstance(self._basis_settings, WDMSettings):
            return self.xp.full(
                self.num_bands_now, self._basis_settings.ind_min_f, dtype=self.xp.int32
            )
        if isinstance(self._basis_settings, STFTSettings):
            return self.buffer_start_index
        return self._min_freq_inds_store

    @property
    def special_indices_unique_sort(self):
        return self._special_indices_unique_sort

    @staticmethod
    def _materialize(buf):
        """Return a single ndarray view for ``buf``.

        Single-shard runs already expose ``buf`` as a reshape view of the
        underlying ndarray, so this is a no-op (returns ``buf`` itself).
        Multi-shard runs expose ``buf`` as a :class:`BandView` -- gather it
        to a single ndarray on ``gpus[0]`` so downstream einsum / boolean
        indexing / sum kernels see a contiguous array.
        """
        if isinstance(buf, BandView):
            return buf.gather()
        return buf

    def likelihood(self, source_only: bool = False, noise_only: bool = False, cells=None) -> float:
        """Band-level log-likelihood over the cells in the buffer.

        Overrides the inherited per-AC ``AnalysisContainerArray.likelihood``
        dispatch: the buffer computes its cell likelihoods directly from the
        shaped residual / PSD views (vectorized over cells).

        ``cells`` picks the cells to score with ``source_only``, as global cell ids of any shape;
        the result has the same shape, and ``None`` scores every cell.
        """
        assert not (source_only and noise_only)
        assert cells is None or source_only, "cells selects source terms only"

        if source_only:
            return self._source_terms(cells)

        psd_buffer = self._materialize(self.psd_buffer)
        if noise_only:
            if self.tdi_channel_setup == "XYZ":
                raise NotImplementedError("Noise-only likelihood requires log=determinant over frequency for XYZ CSD.")
            return -self.xp.sum(self.xp.log(self.xp.abs(1 / psd_buffer[psd_buffer != 0.0])))

        source_term = self._source_terms()

        # Diagonal noise_term fall_back # TODO check if this is sufficient not used currently anyway
        psd_term = -self.xp.sum(self.xp.log(self.xp.abs(psd_buffer[psd_buffer != 0.0])))
        if self.tdi_channel_setup == "XYZ":
            warnings.warn("The current psd ll calculation is not correct for XYZ CSD channel setup.")

        return source_term + psd_term

    def _source_terms(self, cells=None):
        """Per-cell source term of ``cells`` (global ids, any shape), on the caller's device.

        Each shard contracts its own cells and only the per-cell values cross devices. The
        contraction couples no two cells, so this equals contracting a gathered buffer, which
        copied the residual, template and inverse CSD of every cell onto ``gpus[0]`` per call.
        """
        xp = self.xp
        cells = (np.arange(int(self.num_bands_now)) if cells is None
                 else np.atleast_1d(np.asarray(asnumpy(cells), dtype=np.int64)))
        # * Each row is one block, so a row that is a stride of the buffer reads shards as views.
        blocks = cells.reshape(-1, cells.shape[-1]) if cells.ndim > 1 else cells[None, :]
        split_map = np.asarray(self.split_map)
        ac_to_intra = np.asarray(self.ac_to_intra)
        data_shards = self.data_shaped
        psd_shards = self.psd_shaped
        template_shards = self._acs_template_buffer.data_shaped if self.use_template_arr else None

        launched = []
        for block_i, block in enumerate(blocks):
            block_split = split_map[block]
            for split in np.unique(block_split):
                where = np.nonzero(block_split == split)[0]
                device = None if self.gpus is None else int(self.gpus[split])
                with device_context(xp, device):
                    select = _stride_or_index(xp, ac_to_intra[block[where]])
                    residual = data_shards[split][select]
                    if isinstance(select, slice):
                        # ! A slice is a view: copy it, or subtracting the template changes the residual buffer.
                        residual = residual.copy()
                    if template_shards is not None:
                        residual -= template_shards[split][select]
                    terms = self._contract_source_term(residual, psd_shards[split][select])
                    ready = None
                    if device is not None:
                        ready = xp.cuda.Event(block=False, disable_timing=True)
                        ready.record()
                launched.append((block_i * blocks.shape[1] + where, terms, ready))

        # * Collect only after every shard has launched, so the contractions on different devices overlap.
        with device_context(xp, None if self.gpus is None else int(self.gpus[0])):
            out = xp.empty(cells.size, dtype=xp.float64)
            for positions, terms, ready in launched:
                if ready is not None:
                    xp.cuda.get_current_stream().wait_event(ready)
                out[xp.asarray(positions)] = xp.asarray(terms)
        return out.reshape(cells.shape)

    def _contract_source_term(self, residual, psd):
        """``-2 dc <r|r>`` per cell of ``residual``, against the matching inverse CSD rows ``psd``."""
        # Domain-generic inner product: <a|b> = 4 sum(a* invC b) * dc where
        # dc is the basis measure (FD: df; WDM: the pixel measure) -- the
        # same convention as lisatools.diagnostic.inner_product. Trailing
        # basis axes (FD: k; WDM: (Nf, Nt)) are flattened.
        nb = residual.shape[0]
        nc = self.nchannels
        num_flat = residual.reshape(nb, nc, -1)
        dc = float(self.settings.differential_component)

        if self.tdi_channel_setup == "XYZ":
            psd_flat = psd.reshape(nb, nc, nc, -1)
            # b=bands, i/j=channels, k=flattened basis
            return (
                - (1.0 / 2.0) * 4.0 * dc
                * self.xp.einsum(
                    "bik,bijk,bjk->b", num_flat.conj(), psd_flat, num_flat
                ).real
            )
        psd_flat = psd.reshape(nb, nc, -1)
        return (
            - (1.0 / 2.0) * 4.0 * dc
            * self.xp.sum((num_flat.conj() * num_flat) * psd_flat, axis=(1, 2)).real
        )

    # Explicit alias while callers migrate off the ``likelihood`` name (which
    # shadows the inherited per-AC ACA dispatch).
    band_likelihoods = likelihood

    def get_swap_ll(self, params_remove, params_add, data_index, N_vals, phase_maximize=False):
        """Per-proposal swap log-likelihood difference.

        Domain-agnostic: dispatches to ``self._likelihood_engine.get_swap_ll``,
        which is either :class:`FDBandLikelihoodEngine` or
        :class:`WDMBandLikelihoodEngine` depending on the buffer's
        ``basis_settings``. Both engines take the per-band ACA (``self``)
        and the physical params, and return a :class:`SwapLLResult`. The
        rejection-sampling clamp and the phase-maximisation correction live
        here so the engine stays a thin wrapper around the kernel.
        """
        params_remove_phys = self.transform_fn.both_transforms(params_remove, xp=self.xp)
        params_add_phys = self.transform_fn.both_transforms(params_add, xp=self.xp)
        self._count_window_tracks(params_add_phys, data_index)

        result = self._likelihood_engine.get_swap_ll(
            self,
            params_remove_phys,
            params_add_phys,
            data_index=data_index,
            noise_index=data_index,
            N_vals=N_vals,
            phase_maximize=phase_maximize,
            waveform_kwargs=self.waveform_kwargs,
        )

        ll_diff = result.ll_diff
        kept = result.kept

        if np.any(~kept):
            logger.info(f"NOT KEEPING: {(~kept).sum()}")

        if phase_maximize and result.phase_angle is not None:
            # Engine returns the per-proposal phase rotation applied during
            # phase-maximisation; subtract it from phi0 so the accepted
            # parameters reflect the maximised draw.
            params_add[kept, 3] = params_add[kept, 3] - result.phase_angle

        # Rejection sampling on SNR: only applied to *add* proposals (the
        # remove side's opt_snr is meaningless when amp_add is tiny).
        reject = self.xp.zeros(kept.shape[0], dtype=bool)
        reject[kept] = (result.opt_snr_add[kept] < self.opt_snr_rej_samp_limit) & (
            params_add_phys[kept, 0] > 1e-30
        )
        ll_diff[reject] = -1e300

        return ll_diff

    def get_ll(self, params, data_index, noise_index, N_vals, phase_maximize=False,
               return_inner_products=False):
        """Per-source log-likelihood against the cell residuals.

        Domain-agnostic dispatch like :meth:`get_swap_ll`. Returns the
        log-likelihood array ``-0.5 * (d_d + h_h - 2 d_h)`` on the engine's
        xp module (``d_d`` per the underlying computation object's
        convention; 0 unless configured). With
        ``return_inner_products=True`` returns ``(ll, d_h, h_h,
        phase_angle)`` instead. The raw inner products also land on
        :attr:`d_h_out` / :attr:`h_h_out`, and :attr:`phase_angle` carries
        the maximising rotation when ``phase_maximize=True``.
        """
        params_phys = self.transform_fn.both_transforms(params, xp=self.xp)
        self._count_window_tracks(params_phys, data_index)
        ll = self._likelihood_engine.get_ll(
            self,
            params_phys,
            data_index=data_index,
            noise_index=noise_index,
            N_vals=N_vals,
            phase_maximize=phase_maximize,
            waveform_kwargs=self.waveform_kwargs,
        )
        self.d_h_out = self._likelihood_engine.d_h_out
        self.h_h_out = self._likelihood_engine.h_h_out
        self.phase_angle = self._likelihood_engine.phase_angle
        self.kept_out = getattr(
            self._likelihood_engine, "kept_out",
            self.xp.ones(params.shape[0], dtype=bool),
        )
        if return_inner_products:
            return ll, self.d_h_out, self.h_h_out, self.phase_angle
        return ll

    def setup_in_model_likelihood(self, params, data_index, N_vals=None) -> None:
        """Per-source in-model likelihood setup (once per repeat block).

        Forwards the picked sources' CURRENT sampling-basis params
        (transformed to physical) plus their buffer slots to the engine's
        ``setup_in_model`` hook. Chunked-het / FD engines no-op; a sig-het
        computation builds its heterodyne reference against the
        source-free cell residuals here and holds it constant until
        :meth:`clear_in_model_likelihood`. Call AFTER the sources are
        removed from the residual, BEFORE the reference ll of the repeat
        block is computed.

        Returns the engine hook's value: truthy when a sig-het reference
        is now active (the move uses this to arm its mid-block drift
        refresh), ``None`` from the no-op hooks."""
        params_phys = self.transform_fn.both_transforms(params, xp=self.xp)
        return self._likelihood_engine.setup_in_model(
            self, params_phys, data_index, N_vals=N_vals)

    def clear_in_model_likelihood(self) -> None:
        """Deactivate the per-source in-model setup (no-op engines ignore)."""
        self._likelihood_engine.clear_in_model()

    def get_add_ll(self, params, data_index, noise_index, N_vals, phase_maximize=False):
        """Log-likelihood delta of ADDING a source to the model.

        ``ll(r - h) - ll(r) = <r|h> - 0.5 <h|h>`` where ``r`` is the current
        cell residual (which does not contain ``h``). This is a delta, not a
        singular log-likelihood -- the ``d_d`` term cancels. Computed from
        the :attr:`d_h_out` / :attr:`h_h_out` stashed by :meth:`get_ll`;
        :attr:`phase_angle` is available after the call when
        ``phase_maximize=True``. Sources rejected by the engine's bounds
        check (:attr:`kept_out`) come back as ``-1e300``.
        """
        self.get_ll(params, data_index, noise_index, N_vals, phase_maximize=phase_maximize)
        delta = self.d_h_out.real - 0.5 * self.h_h_out.real
        delta[~self.kept_out] = -1e300
        return delta

    def get_removal_ll(self, params, data_index, noise_index, N_vals):
        """Log-likelihood delta of REMOVING a source that is in the residual.

        For a residual ``r`` that still *contains* the subtracted template
        ``h`` (i.e. the source is part of the model), the delta of taking it
        out is ``ll(r + h) - ll(r) = -<r|h> - 0.5 <h|h>``. Computed from one
        normal :meth:`get_ll` call by flipping the sign of ``d_h`` -- the
        arithmetic equivalent of evaluating the template with its reference
        phase flipped (``phi -> -phi``, i.e. ``h -> -h``). Delta only; the
        ``d_d`` term cancels. Bounds-rejected sources come back as
        ``-1e300`` (see :attr:`kept_out`).
        """
        self.get_ll(params, data_index, noise_index, N_vals)
        delta = -self.d_h_out.real - 0.5 * self.h_h_out.real
        delta[~self.kept_out] = -1e300
        return delta

    def get_ll_grad(self, params, data_index, noise_index, N_vals,
                     *, param_eps=None, chunk=None):
        """Per-source gradient of ``L = <d|h> - 0.5 <h|h>`` w.r.t. params.

        Dispatches to ``self._likelihood_engine.get_ll_grad`` -- only
        the chunked-het backend implements this; the legacy FD path
        raises NotImplementedError. Returns ``(num_proposals, nparams)``
        on the engine's xp module.

        Used by the in-model NUTS / gradient move (the chunked-het
        replacement for the legacy info-matrix Cholesky proposal). The
        buffer must hold the source-of-interest's *clean* residual --
        i.e. ``remove_sources_from_band_buffer`` has been called for
        that source already -- before invoking this.

        The compute backend (C++ central-FD or JAX autograd) is fixed
        on the ``GBWDMComputations`` instance passed in at buffer
        construction time via ``gb_wdm_comp``. Per the sprint-wide
        rule there is no runtime ``backend=`` kwarg; build a JAX-
        backed ``gb_wdm_comp`` if you need the autograd path.
        """
        params_phys = self.transform_fn.both_transforms(params, xp=self.xp)
        self._count_window_tracks(params_phys, data_index)
        return self._likelihood_engine.get_ll_grad(
            self,
            params_phys,
            data_index=data_index,
            noise_index=noise_index,
            N_vals=N_vals,
            param_eps=param_eps,
            chunk=chunk,
            waveform_kwargs=self.waveform_kwargs,
        )

    def hessian(self, params, data_index, noise_index, N_vals,
                 *, chunk=None,
                 psd_fix=False, psd_floor_rel=1e-30):
        """Per-source Hessian of ``L = <d|h> - 0.5 <h|h>``.

        Dispatches to ``self._likelihood_engine.hessian``. Returns
        ``(num_proposals, nparams, nparams)``. With ``psd_fix=True``,
        returns ``M = |-H|`` (eigendecompose-then-abs, with a relative
        floor) -- ready to feed to ``NUTSSampler(metric=M)`` as a
        per-leaf mass matrix.

        Same buffer-state precondition as :meth:`get_ll_grad`: the
        active source must have been removed from the band buffer
        before calling.

        Currently only the JAX-backed chunked-het generator
        implements ``hessian_wdm``; the C++ chunked-het backend
        raises until the native Hessian kernel lands. Per the
        sprint-wide rule the backend is fixed on the underlying
        ``gb_wdm_comp`` instance -- no runtime ``backend=`` kwarg.
        """
        params_phys = self.transform_fn.both_transforms(params, xp=self.xp)
        return self._likelihood_engine.hessian(
            self,
            params_phys,
            data_index=data_index,
            noise_index=noise_index,
            N_vals=N_vals,
            chunk=chunk,
            psd_fix=psd_fix,
            psd_floor_rel=psd_floor_rel,
            waveform_kwargs=self.waveform_kwargs,
        )

    def reset_residual_buffers(self, inds_fill=None):
        if inds_fill is None:
            inds_fill = self.xp.arange(self.num_bands_now)
        self.band_buffer[inds_fill] = 0.0

    def reset_psd_buffers(self, inds_fill=None):
        if inds_fill is None:
            inds_fill = self.xp.arange(self.num_bands_now)
        self.psd_buffer[inds_fill] = 0.0

    def fill_buffer_residual_and_psd_from_acs(
        self, acs: AnalysisContainerArray, inds_fill: Optional[cp.ndarray] = None
    ) -> None:
        # The outer ``acs`` is accessed via tuple-fancy indexing
        # ``data_shaped[0][inds1, inds2, inds3]`` (3-tuple for AET, 5-tuple
        # for XYZ CSD). BandView routes the tuple-fancy index through the
        # owning shard at the right intra-shard band position; on
        # single-shard ACAs the reshape view is touched directly. No
        # outer-buffer materialisation needed.
        if inds_fill is None:
            inds_fill = self.xp.arange(self.num_bands_now)

        if isinstance(self._basis_settings, STFTSettings):
            self._fill_stft_windows(acs, inds_fill)
            return

        outer_data_view = acs.data_shaped_view()
        outer_psd_view = acs.psd_shaped_view()

        inds_get_data = self._get_fill_buffer_ind_map(acs, inds_fill=inds_fill, is_psd=False)

        # load rest of data into buffer (has current sources removed)
        self.reset_residual_buffers(inds_fill=inds_fill)

        # By removing `.flatten()` during indexing, broadcasting gives us the exact shape natively.
        self.band_buffer[inds_fill] += outer_data_view[inds_get_data]
        del inds_get_data

        inds_get_psd = self._get_fill_buffer_ind_map(acs, inds_fill=inds_fill, is_psd=True)
        self.reset_psd_buffers(inds_fill=inds_fill)

        psd_vals = outer_psd_view[inds_get_psd]
        if self.xp.iscomplexobj(psd_vals) and not self.xp.iscomplexobj(
            self.psd_buffer if not isinstance(self.psd_buffer, BandView) else psd_vals
        ):
            # FD buffers store the REAL inverse covariance (gb_fd kernel
            # convention); the parent XYZ CSD invC may be complex.
            psd_vals = psd_vals.real
        self.psd_buffer[inds_fill] = psd_vals
        del inds_get_psd

    def _fill_stft_windows(self, acs: AnalysisContainerArray, inds_fill) -> None:
        """Copy each slot's store window of its walker's residual and inverse CSD from the parent.

        Writes straight into the buffer rows, with no full-grid temporary and no index array larger
        than one entry per cell. Cells are grouped by (parent split, buffer split).
        """
        slots = np.asarray(asnumpy(inds_fill), dtype=int)
        walkers = np.asarray(asnumpy(self.unique_band_combos[inds_fill, 1]), dtype=int)
        starts = self._stft_start_inds_host[slots].astype(np.int64)
        nf_parent = int(self._basis_settings.NF_active)
        nf_cell = int(self._stft_cell_width)
        num_times = int(self._basis_settings.NT)
        # * Entries of one cell ahead of the frequency axis: channels x times, or channel pairs x times.
        lead_data = int(self.nchannels) * num_times
        lead_psd = int(np.prod(self.shape_sens)) * num_times

        parent_split = np.asarray(acs.split_map)[walkers]
        cell_split = np.asarray(self.split_map)[slots]
        parent_rows = np.asarray(acs.ac_to_intra)[walkers].astype(np.int64)
        cell_rows = np.asarray(self.ac_to_intra)[slots].astype(np.int64)
        for split_src in np.unique(parent_split):
            for split_dst in np.unique(cell_split):
                cells = np.where((parent_split == split_src) & (cell_split == split_dst))[0]
                if cells.size == 0:
                    continue
                device_src = None if acs.gpus is None else int(acs.gpus[split_src])
                device_dst = None if self.gpus is None else int(self.gpus[split_dst])
                for src, dst, lead in (
                    (acs.linear_data_arr[split_src], self.linear_data_arr[split_dst], lead_data),
                    (acs.linear_psd_arr[split_src], self.linear_psd_arr[split_dst], lead_psd),
                ):
                    self._copy_stft_windows(
                        src, dst, parent_rows[cells], cell_rows[cells], starts[cells],
                        lead, nf_parent, nf_cell, device_src, device_dst,
                    )

    def _copy_stft_windows(self, src, dst, src_rows, dst_rows, starts, lead, nf_src, nf_dst,
                           device_src, device_dst) -> None:
        """``dst[dst_rows[k], l, f] = src[src_rows[k], l, starts[k] + f]`` on flat buffers."""
        if not self.backend.uses_cupy:
            src_shaped = src.reshape(-1, lead, nf_src)
            dst_shaped = dst.reshape(-1, lead, nf_dst)
            for src_row, dst_row, start in zip(src_rows, dst_rows, starts):
                dst_shaped[dst_row] = src_shaped[src_row, :, start:start + nf_dst]
            return

        num_cells = int(src_rows.shape[0])
        kernel = _stft_window_copy_kernel()
        same_device = device_src == device_dst
        with device_context(self.xp, device_src):
            if same_device:
                target, target_rows = dst, cp.asarray(dst_rows)
            else:
                target = cp.empty(num_cells * lead * nf_dst, dtype=dst.dtype)
                target_rows = cp.arange(num_cells, dtype=cp.int64)
            kernel(src, cp.asarray(src_rows), target_rows, cp.asarray(starts),
                   lead, nf_src, nf_dst, target, size=num_cells * lead * nf_dst)
            if same_device:
                return
            target_ready = cp.cuda.Event(block=False, disable_timing=True)
            target_ready.record()
        # * Peer copy: the previous route through pageable host memory was 8x slower on gpu04.
        assert_peer_access(self.xp, [device_src, device_dst], context="SubBandBuffer window copy")
        with device_context(self.xp, device_dst):
            cp.cuda.get_current_stream().wait_event(target_ready)
            block = cp.asarray(target).reshape(num_cells, lead * nf_dst)
            dst.reshape(-1, lead * nf_dst)[cp.asarray(dst_rows)] = block

    def verify_against_parent(self, acs: AnalysisContainerArray, inds_fill) -> None:
        """Check a freshly filled buffer against the parent ACA. ``verify_buffer`` only.

        Every cell's residual and inverse CSD must equal the parent's for that walker, bit for
        bit, and each cell's ``sens_mat`` must be its own ``invC`` view. This is what the
        construction path is allowed to affect, so it isolates construction changes from the
        run-to-run scatter of the sampler itself.
        """
        # ! The comparison reads other devices' cells from the caller's device, which orders nothing.
        if self.gpus is not None and self.backend.uses_cupy:
            for device in self.gpus:
                with device_context(self.xp, int(device)):
                    self.xp.cuda.runtime.deviceSynchronize()
        outer_data = acs.data_shaped_view()
        outer_psd = acs.psd_shaped_view()
        walkers = self.unique_band_combos[inds_fill, 1]
        data_bad = psd_bad = alias_bad = 0
        is_stft = isinstance(self._basis_settings, STFTSettings)
        for slot, walker in zip(_to_numpy(inds_fill).tolist(), _to_numpy(walkers).tolist()):
            window = slice(None)
            if is_stft:
                start = int(self._stft_start_inds_host[slot])
                window = slice(start, start + self._stft_cell_width)
            if not bool((self._materialize(self.band_buffer[slot]) == outer_data[walker][..., window]).all()):
                data_bad += 1
            if not bool((self._materialize(self.psd_buffer[slot]) == outer_psd[walker][..., window]).all()):
                psd_bad += 1
        for acs_here in (self, self._acs_template_buffer) if self.use_template_arr else (self,):
            for ac in acs_here.acs.flatten():
                if ac.sens_mat.sens_mat is not ac.sens_mat.invC:
                    alias_bad += 1
        logger.info(
            "[VERIFY_BUFFER] %d cells: residual mismatches %d, inverse CSD mismatches %d, "
            "sens_mat not aliasing invC %d",
            len(walkers), data_bad, psd_bad, alias_bad,
        )

    def _get_fill_buffer_ind_map(
        self, acs: AnalysisContainerArray, inds_fill: Optional[cp.ndarray] = None, is_psd: bool = False
    ) -> Union[Tuple[Any, ...], Any]:

        if isinstance(self._basis_settings, (WDMSettings, STFTSettings)):
            # Per-band buffers cover the full active grid (WDM: (Nf_active, Nt_active);
            # STFT: (NT, NF_active)), so each band takes its entire (channel, ...) slab
            # out of the parent ACA. Returning the 1D parent walker indices allows
            # BandView.__getitem__ to route directly to _gather (a direct slice along
            # axis 0), avoiding multi-gigabyte 4D/5D Cartesian broadcast indexing arrays.
            if inds_fill is None:
                inds_fill = self.xp.arange(self.num_bands_now)

            return self.unique_band_combos[inds_fill, 1]

        if not isinstance(self._basis_settings, FDSettings):
            raise NotImplementedError(
                f"Buffer does not support basis domain {type(self._basis_settings).__name__}."
            )

        if inds_fill is None:
            inds_fill = self.xp.arange(self.num_bands_now)

        assert np.all(acs.start_freq_ind[0] == acs.start_freq_ind)
        start_freq_ind = acs.start_freq_ind[0]

        assert np.all((self.buffer_start_index[inds_fill] - start_freq_ind) >= 0), (
            "Buffer start indices fall below the parent data start index."
        )

        assert np.all(
            (self.buffer_start_index[inds_fill] - start_freq_ind + self._fd_store_length)
            <= acs.data_length
        ), f"Buffer indexing exceeds available data length in AnalysisContainerArray. Start indices: {self.buffer_start_index[inds_fill]}, start_freq_ind: {start_freq_ind}, data_length: {self._fd_store_length}, acs data_length: {acs.data_length}"

        start_inds = self.buffer_start_index[inds_fill] - start_freq_ind

        if is_psd and self.tdi_channel_setup == "XYZ":
            # Target output shape:
            # (len(inds_fill), nchannels, nchannels, window_bins).
            # The parent FD ACA's psd_shaped has shape
            # (num_walkers, nchan, nchan, n_bins) -- axis 0 is the raw
            # walker index (mirrors the WDM XYZ 5-tuple below).
            inds1 = self.unique_band_combos[inds_fill, 1][:, None, None, None]
            inds2 = self.xp.arange(self.nchannels)[None, :, None, None]
            inds3 = self.xp.arange(self.nchannels)[None, None, :, None]
            inds4 = start_inds[:, None, None, None] + self.xp.arange(
                self.band_buffer.shape[-1]
            )[None, None, None, :]
            return inds1, inds2, inds3, inds4

        else:
            # Target output shape: (len(inds_fill), self.nchannels, self.band_buffer.shape[-1])
            inds1 = self.unique_band_combos[inds_fill, 1][:, None, None]
            inds2 = self.xp.arange(self.nchannels)[None, :, None]
            inds3 = start_inds[:, None, None] + self.xp.arange(self.band_buffer.shape[-1])[None, None, :]

        return inds1, inds2, inds3

    def remove_sources_from_template_buffer(self, *args, **kwargs) -> None:
        self._adjust_via_engine(-1, self._acs_template_buffer, *args, **kwargs)

    def add_sources_to_template_buffer(self, *args, **kwargs) -> None:
        self._adjust_via_engine(+1, self._acs_template_buffer, *args, **kwargs)

    def swap_template_slots(self, slots_a, slots_b) -> None:
        """Exchange the template-buffer contents of slot sets ``a`` and ``b``.

        Used by the tempering stage to swap a temperature pair's per-cell
        templates -- and, called again with the rejected subset, to revert
        the swaps that failed the acceptance draw.
        """
        tmp = self.template_buffer[slots_a].copy()
        self.template_buffer[slots_a] = self.template_buffer[slots_b]
        self.template_buffer[slots_b] = tmp[:]

    def _adjust_via_engine(
        self, factor, target_aca, params, params_index, N_vals, *args, **kwargs
    ) -> None:
        """Domain-agnostic dispatch into ``self._likelihood_engine.fill_template``.

        ``factor`` is +1 (write source into the template) or -1 (subtract it).
        ``target_aca`` selects which AnalysisContainerArray to write into
        (the buffer itself for residuals, or the template twin). Both share
        the same per-band geometry, so the engine doesn't need to know which
        one it's filling.
        """
        assert isinstance(factor, int) and (factor == -1 or factor == +1)
        params_phys = self.transform_fn.both_transforms(params, xp=self.xp)
        self._count_window_tracks(params_phys, params_index)
        self._likelihood_engine.fill_template(
            target_aca,
            params_phys,
            params_index,
            N_vals,
            factor=factor,
            waveform_kwargs=self.waveform_kwargs,
        )

    def adjust_sources_in_band_buffer(
        self, factor, input_array, params, params_index, N_vals, *args, **kwargs
    ) -> None:
        """Backwards-compatible shim around :meth:`_adjust_via_engine`.

        Routes ``input_array`` (a flat buffer pointer the legacy code passed
        through) back to whichever ACA owns it. New code should call
        :meth:`_adjust_via_engine` directly.
        """
        if input_array is self.band_buffer_tmp:
            target_aca = self
        elif self.use_template_arr and input_array is self.template_buffer_tmp:
            target_aca = self._acs_template_buffer
        else:
            raise ValueError(
                "adjust_sources_in_band_buffer received an input_array that "
                "is neither the band-residual nor the template buffer."
            )
        self._adjust_via_engine(factor, target_aca, params, params_index, N_vals, *args, **kwargs)

    def remove_sources_from_band_buffer(self, *args, **kwargs) -> None:
        # NOTE: sign is +1 because band_buffer holds the residual
        # (= data - sum(templates)). Removing a source from the model means
        # ADDING it back to the residual, hence factor=+1 here.
        self._adjust_via_engine(+1, self, *args, **kwargs)

    def add_sources_to_band_buffer(self, *args, **kwargs) -> None:
        # See remove_sources_from_band_buffer note; sign is flipped for the
        # residual-tracking band_buffer.
        self._adjust_via_engine(-1, self, *args, **kwargs)

    def get_special_band_index(
        self, temp_inds: np.ndarray, walker_inds: np.ndarray, band_inds: np.ndarray
    ) -> np.ndarray:
        return pack_special_index(temp_inds, walker_inds, band_inds, self.nwalkers)

    def get_separate_inds_from_special_index(self, special_band_inds: np.ndarray) -> tuple:
        return unpack_special_index(special_band_inds, self.nwalkers)


# Back-compat alias: the pre-merge class name.
Buffer = SubBandBuffer


class BandSorter(LISAToolsParallelModule):
    """GPU helper that sorts/ungroups GB samples by frequency band.

    Flattens the eryn GB branch ``(ntemps, nwalkers, nleaves, ndim)`` into
    per-source arrays (``coords`` / ``inds`` / ``temp_inds`` /
    ``walker_inds`` / ``leaf_inds`` / ``band_inds``), assigns each source to
    its frequency band (``searchsorted`` on the *source* frequency -- not the
    domain settings), pre-draws RJ proposals for the ``inds=False`` slots
    when ``rj_prop`` is given, and provides subset / packing machinery for
    the per-band proposal loop and band-temperature swaps.
    """

    @property
    def xp(self) -> Union[ModuleType, numpy, cupy]:
        return self.backend.xp

    @classmethod
    def supported_backends(cls):
        return ["lisatools_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    def __init__(
        self,
        gb_branch: Branch,
        band_edges: Optional[np.ndarray] = None,
        band_N_vals: Optional[np.ndarray] = None,
        force_backend: Optional[str] = None,
        transform_fn: Optional[TransformContainer] = None,
        copy: bool = True,
        inds_subset: Optional[np.ndarray] = None,
        inds_main_band_sorter: Optional[np.ndarray] = None,
        gb=None,
        gb_wdm_comp=None,
        gb_fd_comp=None,
        gb_stft_comp=None,
        waveform_kwargs={},
        main_band_sorter=None,
        max_data_store_size: int = 6000,
        rj_prop=None,
        keep_all_inds=True,
        stft_store_windows=None,
        verify_buffer: bool = False,
    ):

        LISAToolsParallelModule.__init__(self, force_backend=force_backend)
        self.force_backend = force_backend

        dc = deepcopy if copy else return_x
        if hasattr(gb_branch, "num_sources"):
            _band_sorter = gb_branch
            self.force_backend = _band_sorter.force_backend
            for key, value in _band_sorter.__dict__.items():
                if key[:2] != "__":
                    if key in [
                        "main_band_sorter",
                        "inds_main_band_sorter",
                        "gb",
                        "gb_wdm_comp",
                        "gb_fd_comp",
                        "gb_stft_comp",
                        "rj_prop",
                        "stft_store_windows",
                    ]:
                        continue

                    elif (
                        isinstance(value, self.xp.ndarray)
                        and value.shape[0] == _band_sorter.num_sources
                    ):
                        if inds_subset is None:
                            inds_subset = self.xp.arange(_band_sorter.num_sources)
                        else:
                            assert (
                                isinstance(inds_subset, self.xp.ndarray)
                                and inds_subset.dtype == int
                            )
                            assert inds_subset.max() < (_band_sorter.num_sources)
                        set_value = dc(value[inds_subset])

                    else:
                        set_value = dc(value)

                    setattr(self, key, set_value)

            self.rj_prop = _band_sorter.rj_prop
            self.gb = _band_sorter.gb
            # Forward the computation objects explicitly (skipped in the
            # copy loop so we don't deepcopy GPU-resident objects).
            self.gb_wdm_comp = getattr(_band_sorter, "gb_wdm_comp", None)
            self.gb_fd_comp = getattr(_band_sorter, "gb_fd_comp", None)
            self.gb_stft_comp = getattr(_band_sorter, "gb_stft_comp", None)
            self.stft_store_windows = getattr(_band_sorter, "stft_store_windows", None)
            # need to make sure is not mixed up in loop
            self.set_main_band_sorter_info(main_band_sorter, inds_main_band_sorter)
            return

        assert band_edges is not None
        self.force_backend = force_backend
        self.gb = gb
        # Domain computation objects, forwarded to the buffer in
        # :meth:`get_buffer` so the engine selection matches the parent
        # ACA's basis (WDMSettings -> gb_wdm_comp, FDSettings -> gb_fd_comp,
        # STFTSettings -> gb_stft_comp).
        self.gb_wdm_comp = gb_wdm_comp
        self.gb_fd_comp = gb_fd_comp
        self.gb_stft_comp = gb_stft_comp
        self.stft_store_windows = stft_store_windows
        self.verify_buffer = bool(verify_buffer)
        self.waveform_kwargs = waveform_kwargs
        self.gb_branch_orig = gb_branch
        self.num_bands = len(band_edges) - 1
        self.band_edges = self.xp.asarray(band_edges)
        self.band_N_vals = self.xp.asarray(band_N_vals) if band_N_vals is not None else None
        self.ntemps, self.nwalkers, self.nleaves_max, self.ndim = gb_branch.shape
        self.orig_inds = self.xp.asarray(gb_branch.inds)
        self.keep_all_inds = keep_all_inds
        self.rj_prop = rj_prop

        if rj_prop is not None:
            if keep_all_inds:
                self.coords = self.xp.asarray(gb_branch.coords.reshape(-1, 8))
                self.inds = self.orig_inds.flatten()
            else:
                self.coords = self.xp.asarray(gb_branch.coords[gb_branch.inds])
                self.inds = self.xp.ones(self.coords.shape[:-1], dtype=bool)

            if self.xp.any(~self.inds):
                # self.xp (run backend), NOT the module-level cp: on a
                # CPU-resolved run on a cupy-installed machine cp is cupy
                # while self.coords is numpy -- mixing them crashes the
                # assignment below (module-cp-vs-force_backend trap).
                new_sources = self.xp.full_like(self.coords[~self.inds], np.nan)
                fix = self.xp.full(new_sources.shape[0], True)
                while self.xp.any(fix):
                    new_sources[fix] = self.xp.asarray(
                        rj_prop.rvs(size=fix.sum().item())
                    )
                    fix = self.xp.any(self.xp.isnan(new_sources), axis=-1)

                self.coords[~self.inds] = new_sources

            proposal_logpdf = self.xp.zeros(self.coords.shape[0])

            batch_here = int(1e6)
            inds_splitting = np.arange(0, self.coords.shape[0], batch_here)
            if inds_splitting[-1] != self.coords.shape[0] - 1:
                inds_splitting = np.concatenate(
                    [inds_splitting, np.array([self.coords.shape[0] - 1])]
                )

            for stind, eind in zip(inds_splitting[:-1], inds_splitting[1:]):
                proposal_logpdf[stind:eind] = self.xp.asarray(
                    rj_prop.logpdf(self.coords[stind:eind])
                )
            if self.backend.uses_cupy:
                self.xp.get_default_memory_pool().free_all_blocks()

            if keep_all_inds:
                self.factors = (self.xp.asarray(proposal_logpdf) * -1) * (~self.orig_inds).flatten() + (
                    self.xp.asarray(proposal_logpdf) * +1
                ) * (self.orig_inds).flatten()
                tmp_inds_shaped = self.xp.full_like(self.orig_inds, True)
            else:
                assert self.xp.all(self.inds)
                self.factors = self.xp.asarray(proposal_logpdf) * +1
                tmp_inds_shaped = self.orig_inds.copy()

        else:
            self.coords = self.xp.asarray(gb_branch.coords[gb_branch.inds])
            self.inds = self.xp.ones(self.coords.shape[:-1], dtype=bool)
            self.factors = self.xp.ones_like(self.inds)
            tmp_inds_shaped = self.orig_inds.copy()

        self.has_run_rj = self.xp.zeros_like(self.inds)
        self.num_sources = self.coords.shape[0]
        self.set_main_band_sorter_info(main_band_sorter, inds_main_band_sorter)

        self.freqs = self.coords[:, 1] / 1e3
        self.band_inds = (
            self.xp.searchsorted(self.band_edges, self.freqs, side="right") - 1
        )
        self.max_data_store_size = max_data_store_size

        self.temp_inds = self.xp.repeat(
            self.xp.arange(self.ntemps), self.nwalkers * self.nleaves_max
        ).reshape(self.ntemps, self.nwalkers, self.nleaves_max)[tmp_inds_shaped]
        self.walker_inds = self.xp.tile(
            self.xp.arange(self.nwalkers), (self.ntemps, self.nleaves_max, 1)
        ).transpose((0, 2, 1))[tmp_inds_shaped]
        self.leaf_inds = self.xp.tile(
            self.xp.arange(self.nleaves_max), ((self.ntemps, self.nwalkers, 1))
        )[tmp_inds_shaped]
        self.special_band_inds = self.get_special_band_index(
            self.temp_inds, self.walker_inds, self.band_inds
        )

        self.orig_temp_inds = self.temp_inds.copy()
        self.orig_walker_inds = self.walker_inds.copy()
        self.orig_leaf_inds = self.leaf_inds.copy()
        self.orig_special_band_inds = self.special_band_inds.copy()
        self.orig_band_inds = self.band_inds.copy()
        self.transform_fn = transform_fn

    def set_main_band_sorter_info(self, main_band_sorter, inds_main_band_sorter):
        if main_band_sorter is None:
            self.inds_main_band_sorter = self.xp.arange(self.num_sources)
        else:
            self.inds_main_band_sorter = inds_main_band_sorter

        self.main_band_sorter = main_band_sorter

    @property
    def coords_in(self) -> np.ndarray:
        return self.transform_fn.both_transforms(self.coords, xp=self.xp)

    def get_special_band_index(
        self, temp_inds: np.ndarray, walker_inds: np.ndarray, band_inds: np.ndarray
    ) -> np.ndarray:
        return pack_special_index(temp_inds, walker_inds, band_inds, self.nwalkers)

    def get_separate_inds_from_special_index(self, special_band_inds: np.ndarray) -> tuple:
        return unpack_special_index(special_band_inds, self.nwalkers)

    @property
    def special_index_check(self) -> bool:
        return self.xp.all(
            self.special_band_inds
            == self.get_special_band_index(self.temp_inds, self.walker_inds, self.band_inds)
        )

    def exchange_cell_labels(self, specials_a, temp_a, walkers_a,
                             specials_b, temp_b, walkers_b, bands=None) -> None:
        """Pairwise-swap the (temp, walker) labels of the sources in cell
        sets ``a`` and ``b``.

        ``specials_a[k]`` exchanges with ``specials_b[k]``: every source in
        cell ``a_k`` is relabelled to ``(temp_b, walkers_b[k])`` and vice
        versa (band indices are unchanged -- tempering swaps stay within a
        band; pass ``bands`` to assert that). Both membership maps are
        computed BEFORE any mutation so the two directions cannot see each
        other's relabelled sources.
        """
        xp = self.xp

        def _map(specials_from):
            order = xp.argsort(specials_from.flatten())
            keep = xp.isin(self.special_band_inds, specials_from)
            take = order[xp.searchsorted(
                specials_from[order], self.special_band_inds[keep], side="left"
            )]
            return keep, take

        keep_a, take_a = _map(specials_a)   # take_* indexes the OTHER set's rows
        keep_b, take_b = _map(specials_b)

        if bands is not None:
            assert xp.all(self.band_inds[keep_a] == bands[take_a])
            assert xp.all(self.band_inds[keep_b] == bands[take_b])

        self.special_band_inds[keep_a] = specials_b[take_a]
        self.temp_inds[keep_a] = temp_b
        self.walker_inds[keep_a] = walkers_b[take_a]

        self.special_band_inds[keep_b] = specials_a[take_b]
        self.temp_inds[keep_b] = temp_a
        self.walker_inds[keep_b] = walkers_a[take_b]

    @property
    def N_vals(self) -> Optional[np.ndarray]:
        if self.band_N_vals is None:
            return None
        return self.band_N_vals[self.band_inds]

    @property
    def unique_N(self) -> Optional[np.ndarray]:
        if self.band_N_vals is None:
            return None
        return self.xp.unique(self.N_vals)

    def get_subset(self, *args, **kwargs):
        subset_inds = self.get_subset_inds(*args, **kwargs)

        if len(subset_inds) == 0:
            return None

        # source information
        subset = BandSorter(
            self,
            inds_subset=subset_inds,
            main_band_sorter=self.main_band_sorter,
            inds_main_band_sorter=self.inds_main_band_sorter[subset_inds],
        )
        # band information
        return subset

    def get_subset_inds(self, *args, **kwargs):
        subset_bool = self.get_subset_bool(*args, **kwargs)
        return self.xp.arange(len(subset_bool))[subset_bool]

    def get_subset_bool(
        self,
        units: Optional[int] = None,
        remainder: Optional[int] = None,
        temp: Optional[int] = None,
        walker: Optional[int] = None,
        leaf: Optional[int] = None,
        band: Optional[int] = None,
        apply_inds: Optional[bool] = False,
        special_band_inds: Optional[int | np.ndarray] = None,
        extra_bool: Optional[np.ndarray] = None,
        full_bool: Optional[np.ndarray] = None,
    ) -> np.ndarray:

        inds_keep = self.xp.ones_like(self.band_inds, dtype=bool)

        if full_bool is None:
            if band is not None:
                assert isinstance(band, int)
                inds_keep &= self.band_inds == band
            elif units is not None or remainder is not None:
                assert units is not None and remainder is not None
                inds_keep &= self.band_inds % units == remainder

            if temp is not None:
                assert isinstance(temp, int)
                inds_keep &= self.temp_inds == temp
            if walker is not None:
                assert isinstance(walker, int)
                inds_keep &= self.walker_inds == walker
            if leaf is not None:
                assert isinstance(leaf, int)
                inds_keep &= self.leaf_inds == leaf

            if extra_bool is not None:
                assert isinstance(extra_bool, self.xp.ndarray)
                assert extra_bool.shape == (self.num_sources,)
                inds_keep &= extra_bool

            if apply_inds:
                inds_keep &= self.inds

            if special_band_inds is not None:
                if isinstance(special_band_inds, int):
                    inds_keep &= self.special_band_inds == special_band_inds

                elif isinstance(special_band_inds, self.xp.ndarray):
                    inds_keep &= self.xp.isin(self.special_band_inds, special_band_inds)

        else:
            assert full_bool.shape[0] == self.num_sources
            inds_keep = full_bool

        return inds_keep

    @property
    def main_band_sorter(self):
        main_band_sorter = self if self._main_band_sorter is None else self._main_band_sorter
        return main_band_sorter

    @main_band_sorter.setter
    def main_band_sorter(self, main_band_sorter):
        self._main_band_sorter = main_band_sorter

    def get_buffer(
        self, acs, special_indices_unique, inds_fill=None, buffer_obj=None, **kwargs
    ) -> SubBandBuffer:

        num_band_preload = len(special_indices_unique)

        # Array module from the SORTER's arrays, not the module-level ``cp``:
        # on a CPU run on a machine where cupy imports (cluster), ``cp`` is
        # cupy while the sorter holds numpy -- cp.isin then dies with
        # ``TypeError: Unsupported type <class 'numpy.ndarray'>`` (and
        # cp.arange/cp.asarray would silently UPLOAD host data instead).
        xp = get_array_module(self.main_band_sorter.special_band_inds)

        # CAN USE main_band_sorter TO GET SOURCES IN BANDS OF INTEREST THAT ARE NOT CURRENTLY OF INTEREST THEMSELVES

        sources_now_map = xp.arange(self.main_band_sorter.special_band_inds.shape[0])[
            xp.isin(self.main_band_sorter.special_band_inds, special_indices_unique)
        ]

        # NOTE: self.main_band_sorter.inds needed to only inject real sources
        # inject sources must include sources that have been turned off in these bands
        sources_inject_now_map = xp.arange(self.main_band_sorter.special_band_inds.shape[0])[
            xp.isin(self.main_band_sorter.special_band_inds, special_indices_unique)
            & self.main_band_sorter.inds
        ]

        # separate out inds
        temp_inds_now, walker_inds_now, band_inds_now = self.get_separate_inds_from_special_index(
            special_indices_unique
        )

        all_unique_band_combos = xp.asarray([temp_inds_now, walker_inds_now, band_inds_now]).T
        num_bands_here_total = all_unique_band_combos.shape[0]
        num_bands_now = special_indices_unique.shape[0]

        points_curr_tmp = self.main_band_sorter.coords[sources_now_map].copy()
        curr_special_band_inds = self.main_band_sorter.special_band_inds[sources_now_map].copy()

        # sort these sources by band
        if inds_fill is None:
            inds_fill = xp.arange(num_band_preload)
            assert buffer_obj is None
            buffer_obj = SubBandBuffer(
                self.rj_prop,
                self.nwalkers,
                self.gb,
                self.band_edges,
                self.band_N_vals,
                all_unique_band_combos,
                points_curr_tmp,
                num_bands_now,
                acs.nchannels,
                self.max_data_store_size,
                special_indices_unique,
                self.transform_fn,
                self.waveform_kwargs,
                (
                    acs.settings.layer_df 
                    if isinstance(acs.settings, WDMSettings)
                    else acs.settings.df
                ),
                sources_now_map,
                sources_inject_now_map,
                self.main_band_sorter.special_band_inds[sources_now_map],
                basis_settings=acs.settings,
                gb_wdm_comp=self.gb_wdm_comp,
                gb_fd_comp=self.gb_fd_comp,
                gb_stft_comp=self.gb_stft_comp,
                stft_store_windows=self.stft_store_windows,
                force_backend=self.force_backend,
                **kwargs,
            )

        else:
            assert isinstance(buffer_obj, SubBandBuffer)
            assert inds_fill.max() <= buffer_obj.num_bands_now
            # THIS NEEDS TO HAPPEN before updating data
            buffer_obj.update_special_indices(special_indices_unique, inds_fill=inds_fill)

        buffer_obj.fill_buffer_residual_and_psd_from_acs(acs, inds_fill=inds_fill)
        buffer_obj.parent_acs = acs
        if self.verify_buffer:
            buffer_obj.verify_against_parent(acs, inds_fill)
        # includes sources in these sub-bands that are no longer getting proposals
        coords_to_inject = self.main_band_sorter.coords[sources_inject_now_map].copy()
        inj_special_indices_now = self.main_band_sorter.special_band_inds[
            sources_inject_now_map
        ].copy()

        inject_index = buffer_obj.get_index(inj_special_indices_now)
        inject_N_vals = (
            self.band_N_vals[
                self.main_band_sorter.band_inds[sources_inject_now_map]
            ].copy()
            if self.band_N_vals is not None
            else None
        )

        assert len(inject_index) == len(coords_to_inject)

        inj_args = (coords_to_inject, inject_index, inject_N_vals)
        if buffer_obj.use_template_arr:
            buffer_obj.add_sources_to_template_buffer(*inj_args)
        else:
            buffer_obj.add_sources_to_band_buffer(*inj_args)

        return buffer_obj

    # ------------------------------------------------------------------
    # Group-stretch friends
    # ------------------------------------------------------------------

    def build_friend_index(self, nfriends: int) -> bool:
        """Build the sorted cold-chain frequency table + per-source friend windows.

        Friends are cold-chain (``temp == 0``, ``inds == True``) sources close
        in frequency: for every source (any temperature) we store the start of
        an ``nfriends``-wide window into the frequency-sorted cold-chain
        coordinate table, centred on the source's own frequency (clamped at
        the table edges). :meth:`draw_friends` then draws one friend uniformly
        from that window per source.

        Returns ``False`` (and clears the table) when there are too few
        cold-chain sources to form a window.
        """
        self.nfriends = int(nfriends)
        cold = self.inds & (self.temp_inds == 0)
        n_cold = int(cold.sum())
        if n_cold < max(2, self.nfriends):
            self.friend_start_inds = None
            return False

        cold_coords = self.coords[cold]
        order = self.xp.argsort(cold_coords[:, 1])
        self.friends_coords_sorted = cold_coords[order].copy()
        self.friends_freqs_sorted = self.friends_coords_sorted[:, 1].copy()

        starts = (
            self.xp.searchsorted(self.friends_freqs_sorted, self.coords[:, 1], side="right")
            - self.nfriends // 2
        )
        self.friend_start_inds = self.xp.clip(starts, 0, n_cold - self.nfriends).astype(
            self.xp.int32
        )
        return True

    def draw_friends(self, source_ids):
        """One random friend (coordinate row) per source in ``source_ids``.

        Requires :meth:`build_friend_index` to have been run this proposal.
        """
        starts = self.friend_start_inds[source_ids]
        deviation = self.xp.random.randint(0, self.nfriends, size=len(starts))
        take = self.xp.clip(starts + deviation, 0, len(self.friends_freqs_sorted) - 1)
        return self.friends_coords_sorted[take]

    def get_band_info(self):

        uni_special, uni_special_counts = self.xp.unique(
            self.special_band_inds[self.inds], return_counts=True
        )
        uni_temp_inds, uni_walker_inds, uni_band_inds = self.get_separate_inds_from_special_index(
            uni_special
        )

        num_bands = len(self.band_edges) - 1
        band_counts = np.zeros((self.ntemps, self.nwalkers, num_bands), dtype=int)
        band_counts[_to_numpy(uni_temp_inds), _to_numpy(uni_walker_inds), _to_numpy(uni_band_inds)] = (
            _to_numpy(uni_special_counts)
        )

        return {"band_counts": band_counts}
