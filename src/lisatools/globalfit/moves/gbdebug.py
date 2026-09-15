"""Galactic Binary sampler verification, diagnostic checks, and debug plotting."""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from copy import deepcopy
import dataclasses
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypedDict,
    Union,
)

import numpy as np

from ...domains import WDMSettings
from ...domains import STFTSettings, WDMSettings
from ...utils.typing import NDArrayLike
from ...utils.utility import asnumpy

if TYPE_CHECKING:
    from eryn.model import Model
    from ...analysiscontainer import AnalysisContainerArray
    from .gbbands import BandScheduler, BandSorter, SubBandBuffer
    from .gbspecialmove import GBSpecialBase
else:
    Model = Any
    AnalysisContainerArray = Any
    BandScheduler = Any
    BandSorter = Any
    SubBandBuffer = Any
    GBSpecialBase = Any

__all__ = [
    "GBDebugSettings",
    "GBSamplerDebugger",
    "DomainGeometry",
    "PickedSourcesDict",
    "InModelSequenceDict",
    "RJSequenceDict",
    "UnitSnapshot",
    "CoordinateMapCPU",
    "ProposalStage",
    "SeqPickStrategy",
]

logger = logging.getLogger(__name__)

# * Type aliases for structured quantities
UnitSnapshot = Tuple[np.ndarray, np.ndarray]
CoordinateMapCPU = Tuple[np.ndarray, np.ndarray, np.ndarray]
ProposalStage = Literal["rj", "in-model"]
SeqPickStrategy = Union[Literal["first", "loudest"], str]


@dataclass
class DomainGeometry:
    """Unified container for domain time-frequency lattice geometry."""

    frequency_step: float
    min_frequency_index: int
    num_frequencies: int
    num_times: int
    time_step: float
    start_time: float
    is_stft: bool
    basis_name: str


class PickedSourcesDict(TypedDict):
    """Vectorized source picks across cells for RJ and in-model proposals."""

    ids: NDArrayLike
    specials: NDArrayLike
    slot_index: NDArrayLike
    temp_inds: NDArrayLike
    walker_inds: NDArrayLike
    band_inds: NDArrayLike
    N_vals: NDArrayLike


class InModelSequenceDict(TypedDict, total=False):
    """Traced in-model repeat block state for diagnostic sequence figures."""

    idx: int
    slot: int
    temp: int
    walker: int
    band: int
    f0_old: float
    f0_new: float
    snaps: Dict[str, np.ndarray]
    data_const: Optional[np.ndarray]
    t_tot: Optional[np.ndarray]
    ll_ref_final: Optional[float]


class RJSequenceDict(TypedDict, total=False):
    """Traced RJ proposal state for before/template/after figures."""

    idx: int
    slot: int
    temp: int
    walker: int
    band: int
    before: np.ndarray
    accepted: bool


# * Runtime quantity validation helpers for debug mode


def _validate_coordinate_array(coords: NDArrayLike, name: str) -> None:
    """Ensure coordinates array has 2 dimensions (num_sources, ndim)."""
    if hasattr(coords, "ndim") and coords.ndim != 2:
        raise ValueError(
            f"Expected 2D coordinate array (num_sources, ndim) for '{name}', "
            f"got shape {getattr(coords, 'shape', None)}"
        )


def _validate_1d_array(
    array_val: NDArrayLike,
    name: str,
    expected_length: Optional[int] = None,
) -> None:
    """Ensure array is 1D with the expected element count."""
    if hasattr(array_val, "ndim") and array_val.ndim != 1:
        raise ValueError(
            f"Expected 1D array for '{name}', got shape {getattr(array_val, 'shape', None)}"
        )
    if expected_length is not None and len(array_val) != expected_length:
        raise ValueError(
            f"Length mismatch for '{name}': expected {expected_length}, got {len(array_val)}"
        )


def _validate_picked_dict(picked_dict: Mapping[str, Any]) -> None:
    """Ensure picked dictionary contains all required vectorized keys."""
    required_keys = (
        "ids",
        "specials",
        "slot_index",
        "temp_inds",
        "walker_inds",
        "band_inds",
        "N_vals",
    )
    missing_keys = [key_name for key_name in required_keys if key_name not in picked_dict]
    if missing_keys:
        raise KeyError(
            f"Picked sources dictionary is missing required keys: {missing_keys}"
        )


def _validate_coordinate_map_cpu(
    map_cpu_tuple: Any,
    name: str,
    expected_length: Optional[int] = None,
) -> None:
    """Ensure CPU coordinate map is a 3-tuple (temps, walkers, bands) of matching 1D arrays."""
    if not isinstance(map_cpu_tuple, (tuple, list)) or len(map_cpu_tuple) != 3:
        raise ValueError(
            f"Expected 3-tuple (temps, walkers, bands) for '{name}', "
            f"got {type(map_cpu_tuple).__name__}"
        )
    temps_arr, walkers_arr, bands_arr = map_cpu_tuple
    _validate_1d_array(temps_arr, f"{name}[0] (temps)", expected_length=expected_length)
    _validate_1d_array(walkers_arr, f"{name}[1] (walkers)", expected_length=len(temps_arr))
    _validate_1d_array(bands_arr, f"{name}[2] (bands)", expected_length=len(temps_arr))


def _validate_stage(stage_val: str) -> None:
    """Ensure proposal stage is recognized."""
    if stage_val not in ("rj", "in-model"):
        raise ValueError(
            f"Invalid proposal stage '{stage_val}'; must be 'rj' or 'in-model' or 'full'"
        )


@dataclass
class GBDebugSettings:
    """Settings for Galactic Binary sampler verification and diagnostics.

    Controls debug verification hooks, consistency checks, and diagnostic plots
    generated during proposal iterations.
    """

    enabled: bool = False
    plot_dir: Optional[str] = None
    plot_walker: int = 0
    plot_band: Optional[int] = None
    seq_pick: SeqPickStrategy = "first"
    verify_buffer: bool = False
    mem_probe: bool = False
    mem_probe_file: Optional[str] = None
    mem_probe_site_bytes: int = 1024**2
    mem_probe_site_depth: int = 3

    # * Compatibility property alias for the debug boolean flag
    @property
    def debug(self) -> bool:
        return self.enabled

    @debug.setter
    def debug(self, value: bool) -> None:
        self.enabled = bool(value)


class GBSamplerDebugger:
    """Galactic Binary sampler verification, diagnostic checks, and plotting.

    Decouples debug verification hooks and diagnostic plotting from GBSpecialBase.
    When ``settings.enabled`` is False, all methods early-return with negligible overhead.
    """

    def __init__(
        self,
        move: GBSpecialBase,
        debug_settings: Optional[GBDebugSettings] = None,
    ) -> None:
        self.move: GBSpecialBase = move
        self.settings: GBDebugSettings = (
            deepcopy(debug_settings)
            if debug_settings is not None
            else GBDebugSettings()
        )
        if self.settings.plot_dir is None:
            self.settings.plot_dir = "./gf_output/gb_debug/"

        # * Transient state for tracing and plotting
        self.plot_counter: int = 0
        self.plotted_stages: Set[ProposalStage] = set()
        self.null_logged: bool = False
        self.seq_done: bool = False
        self.rj_done: bool = False
        self.rj_seq: Optional[RJSequenceDict] = None

    def _get_domain_geometry(self) -> DomainGeometry:
        """Extract domain time-frequency dimensions and scaling metrics."""
        basis_settings = getattr(self.move, "_basis_settings", None)
        if isinstance(basis_settings, STFTSettings):
            return DomainGeometry(
                frequency_step=float(basis_settings.df),
                min_frequency_index=int(basis_settings.ind_min),
                num_frequencies=int(basis_settings.NF_active),
                num_times=int(basis_settings.NT),
                time_step=float(basis_settings.dt),
                start_time=float(basis_settings.t0),
                is_stft=True,
                basis_name="STFT",
            )
        # Default / WDM
        layer_df_val = float(getattr(basis_settings, "layer_df", 1.0))
        ind_min_f_val = int(getattr(basis_settings, "ind_min_f", 0))
        nf_active_val = int(
            getattr(basis_settings, "Nf_active", None)
            or getattr(basis_settings, "Nf", 1)
        )
        nt_active_val = int(
            getattr(basis_settings, "Nt_active", None)
            or getattr(basis_settings, "Nt", 1)
        )
        layer_dt_val = float(
            getattr(basis_settings, "layer_dt", None)
            or getattr(basis_settings, "dt", 1.0)
        )
        start_time_val = float(getattr(basis_settings, "t0", 0.0))
        return DomainGeometry(
            frequency_step=layer_df_val,
            min_frequency_index=ind_min_f_val,
            num_frequencies=nf_active_val,
            num_times=nt_active_val,
            time_step=layer_dt_val,
            start_time=start_time_val,
            is_stft=False,
            basis_name="WDM",
        )

    def _cell_geometry(self, buffer_obj: SubBandBuffer, slot: Optional[int] = None) -> DomainGeometry:
        """Geometry of one buffer cell: its own width and the absolute bin of its first column.

        A windowed STFT cell holds ``[start_j, start_j + W)`` of the parent grid, so slab reshapes
        and every bin-to-frequency conversion have to run on the cell's own origin. Full-grid cells
        and WDM return the parent geometry unchanged.
        """
        geometry = self._get_domain_geometry()
        width = getattr(buffer_obj, "_stft_cell_width", None)
        if not geometry.is_stft or width is None:
            return geometry
        starts = getattr(buffer_obj, "_stft_start_inds_host", None)
        start = 0 if (starts is None or slot is None) else int(starts[int(slot)])
        return dataclasses.replace(
            geometry,
            num_frequencies=int(width),
            min_frequency_index=geometry.min_frequency_index + start,
        )

    def _format_slab(self, raw_slab: np.ndarray,
                     geometry: Optional[DomainGeometry] = None) -> np.ndarray:
        """Format raw buffer slab to (nchannels, num_frequencies, num_times).

        For WDM, the buffer slab is already (nchannels, Nf_active, Nt_active).
        For STFT, the buffer slab is (nchannels, NT, NF_active) and is transposed
        to (nchannels, NF_active, NT) so that axis 1 is frequency and axis 2 is time.
        """
        geometry = self._get_domain_geometry() if geometry is None else geometry
        nc_count = raw_slab.shape[0] if raw_slab.ndim >= 2 else 1
        if geometry.is_stft:
            reshaped_slab = raw_slab.reshape(
                nc_count, geometry.num_times, geometry.num_frequencies
            )
            return np.transpose(reshaped_slab, (0, 2, 1))
        return raw_slab.reshape(
            nc_count, geometry.num_frequencies, geometry.num_times
        )

    @property
    def tiles_active(self) -> bool:
        """Whether the per-cell slab snapshots and band-tile plots run.

        Active on both WDMSettings and STFTSettings where 2-D time-frequency
        slabs exist. On flat frequency window domains (FD), 2-D tile figures
        are skipped while numeric consistency checks still run.
        """
        basis_settings = getattr(self.move, "_basis_settings", None)
        return self.settings.enabled and isinstance(
            basis_settings, (WDMSettings, STFTSettings)
        )

    def start_proposal_step(self) -> None:
        """Reset per-step transient trace state at the beginning of run_proposal."""
        self.plotted_stages.clear()
        self.null_logged = False
        self.seq_done = False
        self.rj_done = False
        self.rj_seq = None

    def snapshot_unit_start(
        self,
        model: Model,
        ll_change_log: NDArrayLike,
    ) -> Optional[UnitSnapshot]:
        """Capture cold-chain baseline log-likelihood at the start of a parity unit.

        Parameters
        ----------
        model : Model
            Global fit sampler model containing ``analysis_container_arr``.
        ll_change_log : NDArrayLike
            Array of accumulated log-likelihood changes.

        Returns
        -------
        Optional[UnitSnapshot]
            Tuple of (unit_start_ll, unit_start_change) copied to host NumPy arrays,
            or None if disabled or on error.
        """
        if not self.settings.enabled:
            return None
        try:
            unit_start_ll = asnumpy(
                model.analysis_container_arr.likelihood()
            ).copy()
            unit_start_change = asnumpy(ll_change_log[0].sum(axis=-1)).copy()
            return unit_start_ll, unit_start_change
        except Exception as exc:
            logger.warning("[GB_DEBUG %s] snapshot_unit_start failed: %r", self.move.name, exc)
            return None

    def reconcile_unit_end(
        self,
        model: Model,
        ll_change_log: NDArrayLike,
        start_snapshot: Optional[UnitSnapshot],
        unit_idx: int,
        remainder: int,
    ) -> None:
        """Log parent likelihood reconciliation between direct and tracked delta.

        Parameters
        ----------
        model : Model
            Global fit sampler model containing ``analysis_container_arr``.
        ll_change_log : NDArrayLike
            Array of accumulated log-likelihood changes.
        start_snapshot : Optional[UnitSnapshot]
            Baseline snapshot captured by :meth:`snapshot_unit_start`.
        unit_idx : int
            Index of the current parity unit within the proposal pass.
        remainder : int
            Parity remainder identifying the current central band subset.
        """
        if not self.settings.enabled or start_snapshot is None:
            return
        try:
            start_ll, start_change = start_snapshot
            end_ll = asnumpy(model.analysis_container_arr.likelihood())
            end_change = asnumpy(ll_change_log[0].sum(axis=-1))
            direct_diff = end_ll - start_ll
            tracked_diff = end_change - start_change
            logger.info(
                "[GB_DEBUG %s] unit %d (remainder %d) parent-ll reconcile: "
                "direct per-walker %s vs tracked %s (max abs diff %.3e)",
                self.move.name,
                unit_idx,
                remainder,
                np.array2string(direct_diff, precision=3),
                np.array2string(tracked_diff, precision=3),
                float(np.abs(direct_diff - tracked_diff).max()),
            )
        except Exception as exc:
            logger.warning("[GB_DEBUG %s] reconcile_unit_end failed: %r", self.move.name, exc)

    def cold_chain_residual_loaded(self, model: Model, remainder: int) -> None:
        """Log the neighbour cold-chain residual baseline likelihood for this band unit.

        Parameters
        ----------
        model : Model
            Global fit sampler model containing ``analysis_container_arr``.
        remainder : int
            Parity remainder of the active band unit.
        """
        if not self.settings.enabled:
            return
        try:
            ll_np = asnumpy(model.analysis_container_arr.likelihood())
            logger.info(
                "[GB_DEBUG %s] cold-chain residual loaded (remainder=%s): "
                "sum ll = %.6e over %d walker(s)",
                self.move.name,
                remainder,
                float(np.sum(ll_np.real)),
                ll_np.size,
            )
        except Exception as exc:  # debug-only: never break the sampler
            logger.warning(
                "[GB_DEBUG %s] cold-chain snapshot skipped: %r",
                self.move.name,
                exc,
            )

    def verify_rj_step(
        self,
        buffer_obj: SubBandBuffer,
        params: NDArrayLike,
        alive: NDArrayLike,
        slots: NDArrayLike,
        N_vals: NDArrayLike,
        delta_ll: NDArrayLike,
        keep: NDArrayLike,
        picked: PickedSourcesDict,
        round_idx: int,
        scheduler: Optional[BandScheduler] = None,
    ) -> None:
        """Re-verify RJ deltas through get_add_ll/get_removal_ll and test removal identity.

        Parameters
        ----------
        buffer_obj : SubBandBuffer
            The active sub-band scratch buffer with residual and PSD slabs.
        params : NDArrayLike
            Coordinate array of proposed sources, shaped ``(num_sources, ndim)``.
        alive : NDArrayLike
            Boolean mask indicating which picked sources were already modeled in the state.
        slots : NDArrayLike
            Integer array of buffer slot indices corresponding to the picked sources.
        N_vals : NDArrayLike
            Integer array of sample count multipliers per source band.
        delta_ll : NDArrayLike
            Float array of proposed log-likelihood shifts from the sampler proposal.
        keep : NDArrayLike
            Boolean mask of accepted or evaluated proposals in the current round.
        picked : PickedSourcesDict
            Dictionary containing vectorized pick metadata (ids, specials, slot_index,
            temp_inds, walker_inds, band_inds, N_vals).
        round_idx : int
            Current RJ proposal round index within the band unit.
        scheduler : Optional[BandScheduler], default=None
            Active band scheduler managing source pick orders.
        """
        if not self.settings.enabled:
            return
        _validate_coordinate_array(params, "params")
        num_sources = len(params)
        _validate_1d_array(alive, "alive", expected_length=num_sources)
        _validate_1d_array(slots, "slots", expected_length=num_sources)
        if N_vals is not None:
            _validate_1d_array(N_vals, "N_vals", expected_length=num_sources)
        _validate_1d_array(delta_ll, "delta_ll", expected_length=num_sources)
        _validate_1d_array(keep, "keep", expected_length=num_sources)
        _validate_picked_dict(picked)

        try:
            xp = self.move.xp
            keep_bool = asnumpy(keep).astype(bool)
            if not keep_bool.any():
                return
            births = keep_bool & ~asnumpy(alive)
            deaths = keep_bool & asnumpy(alive)

            if births.any():
                birth_idx = xp.asarray(np.where(births)[0])
                n_vals_b = N_vals[birth_idx] if N_vals is not None else None
                check = buffer_obj.get_add_ll(
                    params[birth_idx], slots[birth_idx], slots[birth_idx], n_vals_b
                )
                lhs = delta_ll[birth_idx]
                finite_mask = xp.isfinite(check) & (lhs > -1e290)
                if bool(finite_mask.any()):
                    relmax = float(
                        xp.max(
                            xp.abs(lhs[finite_mask] - check[finite_mask])
                            / xp.maximum(xp.abs(check[finite_mask]), 1.0)
                        )
                    )
                    logger.info(
                        "[GB_DEBUG %s] RJ birth delta vs get_add_ll: max rel %.3e",
                        self.move.name,
                        relmax,
                    )
                d_h_b = buffer_obj.d_h_out.copy()
                h_h_b = buffer_obj.h_h_out.copy()
                self.residual_round_trip(
                    buffer_obj,
                    params[birth_idx],
                    slots[birth_idx],
                    n_vals_b,
                    d_h_b,
                    h_h_b,
                )

            if deaths.any():
                death_idx = xp.asarray(np.where(deaths)[0])
                n_vals_d = N_vals[death_idx] if N_vals is not None else None
                check = buffer_obj.get_removal_ll(
                    params[death_idx], slots[death_idx], slots[death_idx], n_vals_d
                )
                lhs = delta_ll[death_idx]
                finite_mask = xp.isfinite(check) & (lhs > -1e290)
                if bool(finite_mask.any()):
                    relmax = float(
                        xp.max(
                            xp.abs(lhs[finite_mask] - check[finite_mask])
                            / xp.maximum(xp.abs(check[finite_mask]), 1.0)
                        )
                    )
                    logger.info(
                        "[GB_DEBUG %s] RJ death delta vs get_removal_ll: max rel %.3e",
                        self.move.name,
                        relmax,
                    )
                # Removal identity: restore the template (factor +1), then
                # <r+h|h> must equal <r|h> + <h|h>. Restored in finally.
                d_h_1 = buffer_obj.d_h_out.copy()
                h_h_1 = buffer_obj.h_h_out.copy()
                engine = buffer_obj._likelihood_engine
                params_phys = self.move.transform_fn.both_transforms(
                    params[death_idx], xp=self.move.xp
                )
                data_idx = slots[death_idx].astype(xp.int32)
                engine.fill_template(
                    buffer_obj,
                    params_phys,
                    data_idx,
                    n_vals_d,
                    factor=+1,
                    waveform_kwargs=self.move.waveform_kwargs,
                )
                try:
                    buffer_obj.get_ll(
                        params[death_idx],
                        slots[death_idx],
                        slots[death_idx],
                        n_vals_d,
                    )
                    expected = (d_h_1 + h_h_1).real
                    relmax = float(
                        xp.max(
                            xp.abs(buffer_obj.d_h_out.real - expected)
                            / xp.maximum(xp.abs(expected), 1.0)
                        )
                    )
                    logger.info(
                        "[GB_DEBUG %s] removal identity <r+h|h>=<r|h>+<h|h>: max rel %.3e",
                        self.move.name,
                        relmax,
                    )
                finally:
                    engine.fill_template(
                        buffer_obj,
                        params_phys,
                        data_idx,
                        n_vals_d,
                        factor=-1,
                        waveform_kwargs=self.move.waveform_kwargs,
                    )

            if round_idx in (0, 1):
                map_cpu = (
                    asnumpy(picked["temp_inds"]),
                    asnumpy(picked["walker_inds"]),
                    asnumpy(picked["band_inds"]),
                )
                self.plot_band(
                    buffer_obj,
                    params,
                    slots,
                    N_vals,
                    delta_ll,
                    map_cpu,
                    keep,
                    round_idx,
                    stage="rj",
                )
        except Exception as exc:  # debug-only: never break the sampler
            logger.warning("[GB_DEBUG %s] verify_rj_step skipped: %r", self.move.name, exc)

    def verify_in_model(
        self,
        buffer_obj: SubBandBuffer,
        curr: NDArrayLike,
        new: NDArrayLike,
        slots: NDArrayLike,
        N_vals: NDArrayLike,
        delta_ll: NDArrayLike,
        keep: NDArrayLike,
        map_cpu: CoordinateMapCPU,
        move_idx: int,
    ) -> None:
        """Run residual round-trip and diagnostic band plot during in-model repeats.

        Parameters
        ----------
        buffer_obj : SubBandBuffer
            The active sub-band scratch buffer.
        curr : NDArrayLike
            Current source coordinates before repeat update, shaped ``(num_sources, ndim)``.
        new : NDArrayLike
            Proposed source coordinates, shaped ``(num_sources, ndim)``.
        slots : NDArrayLike
            Buffer slot indices, shaped ``(num_sources,)``.
        N_vals : NDArrayLike
            Per-source sample multipliers, shaped ``(num_sources,)``.
        delta_ll : NDArrayLike
            Proposed log-likelihood differences.
        keep : NDArrayLike
            Boolean mask of proposals to retain.
        map_cpu : CoordinateMapCPU
            3-tuple of (temp_indices, walker_indices, band_indices) on CPU.
        move_idx : int
            Current repeat proposal iteration index.
        """
        if not self.settings.enabled:
            return
        _validate_coordinate_array(curr, "curr")
        _validate_coordinate_array(new, "new")
        num_sources = len(curr)
        if len(new) != num_sources:
            raise ValueError(
                f"Shape mismatch: curr has {num_sources} rows but new has {len(new)}"
            )
        _validate_1d_array(slots, "slots", expected_length=num_sources)
        if N_vals is not None:
            _validate_1d_array(N_vals, "N_vals", expected_length=num_sources)
        _validate_1d_array(delta_ll, "delta_ll", expected_length=num_sources)
        _validate_1d_array(keep, "keep", expected_length=num_sources)
        _validate_coordinate_map_cpu(map_cpu, "map_cpu", expected_length=num_sources)

        try:

            if move_idx == 0:
                buffer_obj.get_ll(curr, slots, slots, N_vals)
                d_h_c = buffer_obj.d_h_out.copy()
                h_h_c = buffer_obj.h_h_out.copy()
                self.residual_round_trip(
                    buffer_obj, curr, slots, N_vals, d_h_c, h_h_c
                )
            num_repeats = self.move.num_repeat_proposals
            if move_idx in (0, num_repeats // 2, max(num_repeats - 1, 0)):
                self.plot_band(
                    buffer_obj,
                    new,
                    slots,
                    N_vals,
                    delta_ll,
                    map_cpu,
                    keep,
                    move_idx,
                    stage="in-model",
                )
        except Exception as exc:  # debug-only: never break the sampler
            logger.warning("[GB_DEBUG %s] verify_in_model skipped: %r", self.move.name, exc)

    def residual_round_trip(
        self,
        buffer_obj: SubBandBuffer,
        params_add: NDArrayLike,
        data_index: NDArrayLike,
        swap_N_vals: NDArrayLike,
        d_h_arr: NDArrayLike,
        h_h_arr: NDArrayLike,
    ) -> None:
        """Add proposed template to residual, confirm get_ll shift equals -<h|h>, and remove it.

        Parameters
        ----------
        buffer_obj : SubBandBuffer
            The active sub-band scratch buffer.
        params_add : NDArrayLike
            Coordinates of the sources to inject, shaped ``(num_sources, ndim)``.
        data_index : NDArrayLike
            Buffer slot indices, shaped ``(num_sources,)``.
        swap_N_vals : NDArrayLike
            Sample count multipliers per source.
        d_h_arr : NDArrayLike
            Inner product <d|h> prior to template injection.
        h_h_arr : NDArrayLike
            Template self-inner product <h|h>.
        """
        if not self.settings.enabled:
            return
        _validate_coordinate_array(params_add, "params_add")
        num_sources = len(params_add)
        _validate_1d_array(data_index, "data_index", expected_length=num_sources)
        if swap_N_vals is not None:
            _validate_1d_array(swap_N_vals, "swap_N_vals", expected_length=num_sources)
        _validate_1d_array(d_h_arr, "d_h_arr", expected_length=num_sources)
        _validate_1d_array(h_h_arr, "h_h_arr", expected_length=num_sources)

        xp = self.move.xp
        engine = buffer_obj._likelihood_engine
        params_phys = self.move.transform_fn.both_transforms(params_add, xp=self.move.xp)
        di = data_index.astype(xp.int32)
        engine.fill_template(
            buffer_obj.acs_buffer,
            params_phys,
            di,
            swap_N_vals,
            factor=-1,
            waveform_kwargs=self.move.waveform_kwargs,
        )
        try:
            buffer_obj.get_ll(params_add, data_index, data_index, swap_N_vals)
            d_h2 = xp.asarray(buffer_obj.d_h_out).real
            # residual r' = r - h_add  =>  <r'|h_add> = d_h_arr - h_h_arr
            expected = (xp.asarray(d_h_arr) - xp.asarray(h_h_arr)).real
            finite_mask = xp.isfinite(d_h2) & xp.isfinite(expected)
            if bool(xp.any(finite_mask)):
                diff = xp.abs(d_h2[finite_mask] - expected[finite_mask])
                scale = xp.maximum(xp.abs(expected[finite_mask]), 1.0)
                relmax = float(xp.max(diff / scale))
                logger.info(
                    "[GB_DEBUG %s] residual add/remove round-trip: max rel diff = %.3e",
                    self.move.name,
                    relmax,
                )
        finally:
            engine.fill_template(
                buffer_obj.acs_buffer,
                params_phys,
                di,
                swap_N_vals,
                factor=+1,
                waveform_kwargs=self.move.waveform_kwargs,
            )

    def seq_select(
        self,
        buffer_obj: SubBandBuffer,
        band_sorter: BandSorter,
        ids: NDArrayLike,
        temp_inds: NDArrayLike,
        walker_inds: NDArrayLike,
        band_inds: NDArrayLike,
        slots: NDArrayLike,
        curr: NDArrayLike,
    ) -> Optional[InModelSequenceDict]:
        """Pick the entry of this repeat batch to trace with 3x3 sequence figures.

        Parameters
        ----------
        buffer_obj : SubBandBuffer
            The active sub-band scratch buffer.
        band_sorter : BandSorter
            Active band sorter maintaining source catalog state.
        ids : NDArrayLike
            Global source IDs of candidate proposals.
        temp_inds : NDArrayLike
            Temperature index per candidate.
        walker_inds : NDArrayLike
            Walker index per candidate.
        band_inds : NDArrayLike
            Band index per candidate.
        slots : NDArrayLike
            Buffer slot index per candidate.
        curr : NDArrayLike
            Current coordinate array for candidates.

        Returns
        -------
        Optional[InModelSequenceDict]
            Traced sequence dictionary if a matching candidate is armed, else None.
        """
        if not self.tiles_active or self.seq_done:
            return None
        _validate_coordinate_array(curr, "curr")
        num_candidates = len(curr)
        _validate_1d_array(ids, "ids", expected_length=num_candidates)
        _validate_1d_array(temp_inds, "temp_inds", expected_length=num_candidates)
        _validate_1d_array(walker_inds, "walker_inds", expected_length=num_candidates)
        _validate_1d_array(band_inds, "band_inds", expected_length=num_candidates)
        _validate_1d_array(slots, "slots", expected_length=num_candidates)

        try:

            sel_w = self.settings.plot_walker
            sel_b = (
                self.settings.plot_band
                if self.settings.plot_band is not None
                else (len(self.move.band_edges) - 1) // 2
            )
            w_np = asnumpy(walker_inds)
            b_np = asnumpy(band_inds)
            t_np = asnumpy(temp_inds)
            match = np.where((w_np == sel_w) & (b_np == sel_b))[0]
            if match.size == 0:
                return None
            idx = int(match[np.argmin(t_np[match])])

            if self.settings.seq_pick != "first":
                cell_mask = (
                    (band_sorter.temp_inds == int(t_np[idx]))
                    & (band_sorter.walker_inds == sel_w)
                    & (band_sorter.band_inds == sel_b)
                    & band_sorter.inds
                )
                cell_ids = asnumpy(
                    self.move.xp.arange(band_sorter.num_sources)[cell_mask]
                )
                if cell_ids.size == 0:
                    return None
                cell_coords = asnumpy(band_sorter.coords)[cell_ids]
                if self.settings.seq_pick == "loudest":
                    c2 = cell_coords[:, 4] ** 2
                    snr_proxy = np.exp(cell_coords[:, 0]) * np.sqrt(
                        ((1.0 + c2) / 2.0) ** 2 + c2
                    )
                    target_id = int(cell_ids[np.argmax(snr_proxy)])
                else:
                    f0_target = float(self.settings.seq_pick)
                    target_id = int(
                        cell_ids[np.argmin(np.abs(cell_coords[:, 1] - f0_target))]
                    )
                if int(asnumpy(ids)[idx]) != target_id:
                    return None

            self.seq_done = True
            f0_old = float(
                asnumpy(
                    self.move.transform_fn.both_transforms(
                        curr[idx : idx + 1], xp=self.move.xp
                    )[0, 1]
                )
            )
            return dict(
                idx=idx,
                slot=int(asnumpy(slots)[idx]),
                temp=int(t_np[idx]),
                walker=sel_w,
                band=sel_b,
                f0_old=f0_old,
                f0_new=f0_old,
                snaps={},
            )
        except Exception as exc:
            logger.warning("[GB_DEBUG %s] seq select skipped: %r", self.move.name, exc)
            return None

    def slab_snapshot(
        self, buffer_obj: SubBandBuffer, slot: int
    ) -> Optional[np.ndarray]:
        """Snapshot the buffer's residual slab as an (nchannels, num_frequencies, num_times) array.

        Parameters
        ----------
        buffer_obj : SubBandBuffer
            The active sub-band scratch buffer.
        slot : int
            Buffer slot index to snapshot.

        Returns
        -------
        Optional[np.ndarray]
            Host copy of the 3D residual slab, shaped ``(nchannels, num_frequencies, num_times)``,
            or None if disabled or on error.
        """
        if not self.settings.enabled:
            return None
        try:
            raw_arr = asnumpy(buffer_obj.band_buffer[slot]).copy()
            return self._format_slab(raw_arr, self._cell_geometry(buffer_obj, slot))
        except Exception as exc:
            logger.warning(
                "[GB_DEBUG %s] slab snapshot skipped: %r", self.move.name, exc
            )
            return None

    def cell_total_template(
        self,
        buffer_obj: SubBandBuffer,
        band_sorter: BandSorter,
        seq: InModelSequenceDict,
    ) -> Optional[np.ndarray]:
        """Sum of ALL modeled templates of the traced cell (scratch fill).

        Parameters
        ----------
        buffer_obj : SubBandBuffer
            The active sub-band scratch buffer.
        band_sorter : BandSorter
            Active band sorter containing modeled source coordinates.
        seq : InModelSequenceDict
            Sequence tracking record identifying the traced cell.

        Returns
        -------
        Optional[np.ndarray]
            Host 3D array of the combined template, shaped ``(nchannels, num_frequencies, num_times)``,
            or None if skipped or on error.
        """
        try:
            temp_val = seq["temp"]
            walker_val = seq["walker"]
            band_val = seq["band"]
            mask = (
                (band_sorter.temp_inds == temp_val)
                & (band_sorter.walker_inds == walker_val)
                & (band_sorter.band_inds == band_val)
                & band_sorter.inds
            )
            geometry = self._cell_geometry(buffer_obj, seq["slot"])
            nc_count = buffer_obj.nchannels
            num_src = int(mask.sum())
            if num_src == 0:
                dtype_val = complex if geometry.is_stft else float
                return np.zeros(
                    (nc_count, geometry.num_frequencies, geometry.num_times),
                    dtype=dtype_val,
                )

            coords = band_sorter.coords[mask]
            params_phys = self.move.transform_fn.both_transforms(
                coords, xp=self.move.xp
            )

            data_shape = buffer_obj._per_band_data_shape
            data_dtype = buffer_obj._per_band_data_dtype
            total_elements = int(np.prod(data_shape))
            scratch = self.move.xp.zeros(total_elements, dtype=data_dtype)

            parent_aca = buffer_obj.acs_buffer
            scratch_starts = None
            if getattr(buffer_obj, "stft_split_start_inds", None) is not None:
                scratch_starts = [self.move.xp.asarray(
                    np.asarray([int(buffer_obj._stft_start_inds_host[int(seq["slot"])])],
                               dtype=np.int32))]

            scratch_split_map = self.move.xp.zeros(1, dtype=int)
            scratch_intra = self.move.xp.zeros(1, dtype=self.move.xp.int32)

            class _Scratch:
                linear_data_arr = [scratch]
                split_map = np.zeros(1, dtype=int)
                ac_to_intra = np.zeros(1, dtype=int)
                # * The engine gathers its index arrays from these, on the shard's own device.
                split_map_by_split = [scratch_split_map]
                ac_to_intra_by_split = [scratch_intra]
                cpp_splits = getattr(parent_aca, "cpp_splits", [None])
                gpus = getattr(parent_aca, "gpus", None)
                stft_split_start_inds = scratch_starts

                def device_context(self, device):
                    if hasattr(parent_aca, "device_context"):
                        return parent_aca.device_context(device)
                    return nullcontext()

                def __len__(self) -> int:
                    return 1

            n_vals_fill = (
                band_sorter.band_N_vals[
                    self.move.xp.full(num_src, band_val, dtype=int)
                ]
                if band_sorter.band_N_vals is not None
                else None
            )
            buffer_obj._likelihood_engine.fill_template(
                _Scratch(),
                params_phys,
                self.move.xp.zeros(num_src, dtype=self.move.xp.int32),
                n_vals_fill,
                factor=+1,
                waveform_kwargs=self.move.waveform_kwargs,
            )
            raw_result = asnumpy(scratch).reshape(data_shape)
            return self._format_slab(raw_result, geometry)
        except Exception as exc:  # debug-only: never break the sampler
            logger.warning(
                "[GB_DEBUG %s] cell total-template fill skipped: %r",
                self.move.name,
                exc,
            )
            return None

    def walker_true_data(
        self,
        acs: AnalysisContainerArray,
        walker: int,
        buffer_obj: Optional[SubBandBuffer] = None,
        slot: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """The traced walker's true data slab (injection minus non-GB models).

        Parameters
        ----------
        acs : AnalysisContainerArray
            Parent analysis container array holding full-instrument data.
        walker : int
            Index of the target walker.

        Returns
        -------
        Optional[np.ndarray]
            Host 3D array of the true data slab, shaped ``(nchannels, num_frequencies, num_times)``,
            or None if unavailable.
        """
        try:
            snap = getattr(self.move, "reset_non_gb_linear_data_arr", None)
            if snap is None:
                return None
            geometry = self._get_domain_geometry()
            # * The parent slab is the FULL grid; a windowed cell only compares against its window.
            cell = geometry if buffer_obj is None else self._cell_geometry(buffer_obj, slot)
            window = slice(
                cell.min_frequency_index - geometry.min_frequency_index,
                cell.min_frequency_index - geometry.min_frequency_index + cell.num_frequencies,
            )
            nc_count = int(acs.nchannels)
            for split_idx, split in enumerate(acs.gpu_splits):
                loc = np.where(np.asarray(split) == int(walker))[0]
                if loc.size:
                    if geometry.is_stft:
                        raw_arr = asnumpy(snap[split_idx]).reshape(
                            -1, nc_count, geometry.num_times, geometry.num_frequencies
                        )[int(loc[0])].copy()
                        return np.transpose(raw_arr, (0, 2, 1))[:, window]
                    raw_arr = asnumpy(snap[split_idx]).reshape(
                        -1, nc_count, geometry.num_frequencies, geometry.num_times
                    )[int(loc[0])].copy()
                    return raw_arr
            return None
        except Exception as exc:  # debug-only: never break the sampler
            logger.warning(
                "[GB_DEBUG %s] true-data slice skipped: %r", self.move.name, exc
            )
            return None

    def band_source_only_ll(
        self,
        buffer_obj: SubBandBuffer,
        arr: np.ndarray,
        slot: Optional[int],
        band: Optional[int],
    ) -> float:
        """Source-only log-likelihood -1/2 <a|a> of arr, sliced to band layers.

        Parameters
        ----------
        buffer_obj : SubBandBuffer
            The active sub-band scratch buffer providing PSD buffers.
        arr : np.ndarray
            3D residual slab array on host, shaped ``(nchannels, num_frequencies, num_times)``.
        slot : Optional[int]
            Slot index in the PSD buffer to evaluate against.
        band : Optional[int]
            Band index whose layer boundaries restrict the evaluation, or None for full slab.

        Returns
        -------
        float
            Calculated log-likelihood value.
        """
        geometry = self._cell_geometry(buffer_obj, slot)
        freq_step = geometry.frequency_step
        ind_min_val = geometry.min_frequency_index
        num_freq = arr.shape[1]
        if band is None:
            bin_start, bin_end = 0, num_freq
        else:
            band_edges_arr = asnumpy(self.move.band_edges)
            bin_start = max(
                int(np.ceil(band_edges_arr[band] / freq_step - 1e-9)) - ind_min_val, 0
            )
            bin_end = min(
                int(np.floor(band_edges_arr[band + 1] / freq_step + 1e-9)) + 1 - ind_min_val,
                num_freq,
            )
        diff_comp = float(buffer_obj.settings.differential_component)
        nc_count = buffer_obj.nchannels
        sub_arr = arr[:, bin_start:bin_end]
        psd_np = asnumpy(buffer_obj._materialize(buffer_obj.psd_buffer)[slot])

        if geometry.is_stft:
            if buffer_obj.tdi_channel_setup == "XYZ":
                inv_c = np.transpose(
                    psd_np.reshape(nc_count, nc_count, geometry.num_times, geometry.num_frequencies),
                    (0, 1, 3, 2),
                )[:, :, bin_start:bin_end]
                return -0.5 * 4.0 * diff_comp * float(
                    np.einsum("ifk,ijfk,jfk->", sub_arr.conj(), inv_c, sub_arr).real
                )
            inv_c = np.transpose(
                psd_np.reshape(nc_count, geometry.num_times, geometry.num_frequencies),
                (0, 2, 1),
            )[:, bin_start:bin_end]
            return -0.5 * 4.0 * diff_comp * float(
                np.sum((sub_arr.conj() * sub_arr).real * inv_c.real)
            )

        if buffer_obj.tdi_channel_setup == "XYZ":
            inv_c = psd_np.reshape(nc_count, nc_count, num_freq, -1)[:, :, bin_start:bin_end]
            return -0.5 * 4.0 * diff_comp * float(
                np.einsum("ifk,ijfk,jfk->", sub_arr, inv_c.real, sub_arr)
            )
        inv_c = psd_np.reshape(nc_count, num_freq, -1)[:, bin_start:bin_end]
        return -0.5 * 4.0 * diff_comp * float(np.sum(sub_arr * inv_c.real * sub_arr))

    def plot_band_sequence(
        self,
        buffer_obj: SubBandBuffer,
        seq: InModelSequenceDict,
    ) -> None:
        """Save four 3x3 figures at the four buffer moments of one in-model repeat block.

        Parameters
        ----------
        buffer_obj : SubBandBuffer
            The active sub-band scratch buffer.
        seq : InModelSequenceDict
            Sequence tracking record containing the snapshot slabs.
        """
        if not self.settings.enabled:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            snaps_dict = seq["snaps"]
            need_snaps = (
                "before_removal",
                "after_removal",
                "before_addback",
                "after_addback",
            )
            if any(key not in snaps_dict for key in need_snaps):
                return
            geometry = self._cell_geometry(buffer_obj, seq["slot"])
            freq_step = geometry.frequency_step
            ind_min_val = geometry.min_frequency_index

            data_const = seq.get("data_const")
            total_template = seq.get("t_tot")
            if data_const is None or total_template is None:
                data_const = snaps_dict["after_removal"]
                total_template = (
                    snaps_dict["after_removal"] - snaps_dict["before_removal"]
                )
            lls = {
                key: self.band_source_only_ll(
                    buffer_obj, snaps_dict[key], seq["slot"], seq["band"]
                )
                for key in need_snaps
            }

            ll_ref_final = seq.get("ll_ref_final")
            dll_addback = (
                self.band_source_only_ll(
                    buffer_obj, snaps_dict["after_addback"], seq["slot"], None
                )
                - self.band_source_only_ll(
                    buffer_obj, snaps_dict["before_addback"], seq["slot"], None
                )
            )
            if ll_ref_final is not None:
                logger.info(
                    "[GB_DEBUG %s] addback delta-ll (full slab) = %.6e vs "
                    "final get_ll = %.6e (diff %.3e)",
                    self.move.name,
                    dll_addback,
                    ll_ref_final,
                    abs(dll_addback - ll_ref_final),
                )

            def _t_at(key_name: str) -> np.ndarray:
                return total_template - (
                    snaps_dict[key_name] - snaps_dict["before_removal"]
                )

            figures = [
                (
                    "1_before_removal",
                    _t_at("before_removal"),
                    data_const,
                    snaps_dict["before_removal"],
                    seq["f0_old"],
                ),
                (
                    "2_after_removal",
                    _t_at("after_removal"),
                    data_const,
                    snaps_dict["after_removal"],
                    seq["f0_old"],
                ),
                (
                    "3_before_addback",
                    _t_at("before_addback"),
                    data_const,
                    snaps_dict["before_addback"],
                    seq["f0_new"],
                ),
                (
                    "4_after_addback",
                    _t_at("after_addback"),
                    data_const,
                    snaps_dict["after_addback"],
                    seq["f0_new"],
                ),
            ]

            nc_count = data_const.shape[0]
            ch_names = ["X", "Y", "Z"][:nc_count]

            if geometry.is_stft:
                band_idx = seq["band"]
                band_edges_arr = asnumpy(self.move.band_edges)
                band_start = max(
                    int(np.ceil(band_edges_arr[band_idx] / freq_step - 1e-9)) - ind_min_val,
                    0,
                )
                band_end = min(
                    int(np.floor(band_edges_arr[band_idx + 1] / freq_step + 1e-9)) + 1 - ind_min_val,
                    data_const.shape[1],
                )
                lo_bin = max(band_start - 2, 0)
                hi_bin = min(band_end + 2, data_const.shape[1])
                total_time_span = geometry.num_times * geometry.time_step
                if total_time_span >= 86400.0 * 365.25:
                    time_scale = 86400.0 * 365.25
                    time_label = "time [years]"
                else:
                    time_scale = 86400.0
                    time_label = "time [days]"
                x_extent = [
                    geometry.start_time / time_scale,
                    (geometry.start_time + total_time_span) / time_scale,
                ]
            else:
                local = int(round(seq["f0_old"] / freq_step)) - ind_min_val
                lo_bin = max(local - 2, 0)
                hi_bin = min(local + 3, data_const.shape[1])
                time_label = "WDM time pixel"
                x_extent = [0, data_const.shape[2]]

            y_lo = (ind_min_val + lo_bin - 0.5) * freq_step * 1e3
            y_hi = (ind_min_val + hi_bin - 0.5) * freq_step * 1e3

            os.makedirs(self.settings.plot_dir, exist_ok=True)
            vmax_row = [
                max(
                    float(np.abs(arr_item[row_idx, lo_bin:hi_bin]).max())
                    for _tag, _template, _data, _residual, _freq in figures
                    for arr_item in (_template, _data, _residual)
                )
                for row_idx in range(nc_count)
            ]
            for fig_tag, templ_arr, data_arr, resid_arr, freq_0 in figures:
                ll_state = lls[fig_tag[2:]]
                fig, axes = plt.subplots(
                    nc_count,
                    3,
                    figsize=(13.5, 3.2 * nc_count),
                    squeeze=False,
                    sharex=True,
                    sharey=True,
                )
                for row_idx in range(nc_count):
                    for col_idx, (panel_name, arr_item) in enumerate(
                        [
                            ("total template", templ_arr),
                            ("total data", data_arr),
                            ("buffer residual", resid_arr),
                        ]
                    ):
                        ax_sub = axes[row_idx][col_idx]
                        im_plot = ax_sub.imshow(
                            np.abs(arr_item[row_idx, lo_bin:hi_bin]),
                            aspect="auto",
                            origin="lower",
                            extent=[x_extent[0], x_extent[1], y_lo, y_hi],
                            vmin=0.0,
                            vmax=vmax_row[row_idx],
                        )
                        ax_sub.axhline(freq_0 * 1e3, color="r", ls="--", lw=1.0)
                        if row_idx == 0:
                            ax_sub.set_title(f"|{panel_name}|", fontsize=11)
                        if col_idx == 0:
                            ax_sub.set_ylabel(
                                f"{ch_names[row_idx]}\nfrequency [mHz]",
                                fontsize=10,
                            )
                        if row_idx == nc_count - 1:
                            ax_sub.set_xlabel(time_label, fontsize=10)
                        ax_sub.tick_params(labelsize=8)
                        cbar = fig.colorbar(im_plot, ax=ax_sub)
                        cbar.ax.tick_params(labelsize=7)
                extra = ""
                if fig_tag.startswith("4_") and ll_ref_final is not None:
                    extra = (
                        f"  |  addback $\\Delta$ll = {dll_addback:.4e} "
                        f"vs final get_ll = {ll_ref_final:.4e}"
                    )
                fig.suptitle(
                    f"GB {geometry.basis_name} in-model sequence {fig_tag.replace('_', ' ')} -- "
                    f"band {seq['band']} | walker {seq['walker']} | "
                    f"T{seq['temp']} | f0 = {freq_0 * 1e3:.4f} mHz\n"
                    f"band SOURCE-ONLY ll of buffer residual = "
                    f"{ll_state:.4e}{extra}",
                    fontsize=13,
                )
                fname = os.path.join(
                    self.settings.plot_dir,
                    f"gb_debug_seq{fig_tag}_band{seq['band']}_w{seq['walker']}"
                    f"_t{seq['temp']}_{self.plot_counter:04d}.png",
                )
                fig.savefig(fname, dpi=120, bbox_inches="tight")
                plt.close(fig)
                self.plot_counter += 1
                logger.info(
                    "[GB_DEBUG %s] saved sequence plot -> %s",
                    self.move.name,
                    fname,
                )
        except Exception as exc:
            logger.warning(
                "[GB_DEBUG %s] sequence plots skipped: %r",
                self.move.name,
                exc,
            )

    def rj_select(
        self,
        buffer_obj: SubBandBuffer,
        picked: PickedSourcesDict,
    ) -> Optional[RJSequenceDict]:
        """Arm RJ before/after trace for the chosen (walker, band) cell.

        Parameters
        ----------
        buffer_obj : SubBandBuffer
            The active sub-band scratch buffer.
        picked : PickedSourcesDict
            Dictionary of vectorized source picks.

        Returns
        -------
        Optional[RJSequenceDict]
            Armed RJ sequence record, or None if cell not picked or tracing inactive.
        """
        self.rj_seq = None
        if not self.tiles_active or self.rj_done:
            return None
        _validate_picked_dict(picked)

        try:
            sel_w = self.settings.plot_walker
            sel_b = (
                self.settings.plot_band
                if self.settings.plot_band is not None
                else (len(self.move.band_edges) - 1) // 2
            )
            w_np = asnumpy(picked["walker_inds"])
            b_np = asnumpy(picked["band_inds"])
            t_np = asnumpy(picked["temp_inds"])
            match = np.where((w_np == sel_w) & (b_np == sel_b))[0]
            if match.size == 0:
                return None
            idx = int(match[np.argmin(t_np[match])])
            slot = int(asnumpy(picked["slot_index"])[idx])
            seq_record: RJSequenceDict = dict(
                idx=idx,
                slot=slot,
                temp=int(t_np[idx]),
                walker=sel_w,
                band=sel_b,
                before=self.slab_snapshot(buffer_obj, slot),
                accepted=False,
            )
            self.rj_seq = seq_record
            return seq_record
        except Exception as exc:
            logger.warning("[GB_DEBUG %s] rj select skipped: %r", self.move.name, exc)
            return None

    def plot_rj_pair(
        self,
        buffer_obj: SubBandBuffer,
        rj_seq: Optional[RJSequenceDict],
    ) -> None:
        """Save one 3x3 figure if the traced cell's RJ proposal was accepted.

        Parameters
        ----------
        buffer_obj : SubBandBuffer
            The active sub-band scratch buffer.
        rj_seq : Optional[RJSequenceDict]
            RJ sequence record captured by :meth:`rj_select`.
        """
        if rj_seq is None or not self.settings.enabled:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            after = self.slab_snapshot(buffer_obj, rj_seq["slot"])
            before = rj_seq["before"]
            diff = after - before
            self.rj_seq = None
            if not rj_seq.get("accepted", False):
                return
            self.rj_done = True

            geometry = self._cell_geometry(buffer_obj, rj_seq["slot"])
            freq_step = geometry.frequency_step
            ind_min_val = geometry.min_frequency_index
            ll_b = self.band_source_only_ll(
                buffer_obj, before, rj_seq["slot"], rj_seq["band"]
            )
            ll_a = self.band_source_only_ll(
                buffer_obj, after, rj_seq["slot"], rj_seq["band"]
            )

            prof = np.abs(diff).sum(axis=(0, 2))
            if geometry.is_stft:
                band_idx = rj_seq["band"]
                band_edges_arr = asnumpy(self.move.band_edges)
                band_start = max(
                    int(np.ceil(band_edges_arr[band_idx] / freq_step - 1e-9)) - ind_min_val,
                    0,
                )
                band_end = min(
                    int(np.floor(band_edges_arr[band_idx + 1] / freq_step + 1e-9)) + 1 - ind_min_val,
                    diff.shape[1],
                )
                lo_bin = max(band_start - 2, 0)
                hi_bin = min(band_end + 2, diff.shape[1])
                total_time_span = geometry.num_times * geometry.time_step
                if total_time_span >= 86400.0 * 365.25:
                    time_scale = 86400.0 * 365.25
                    time_label = "time [years]"
                else:
                    time_scale = 86400.0
                    time_label = "time [days]"
                x_extent = [
                    geometry.start_time / time_scale,
                    (geometry.start_time + total_time_span) / time_scale,
                ]
            else:
                local = int(np.argmax(prof))
                lo_bin = max(local - 2, 0)
                hi_bin = min(local + 3, diff.shape[1])
                time_label = "WDM time pixel"
                x_extent = [0, diff.shape[2]]

            y_lo = (ind_min_val + lo_bin - 0.5) * freq_step * 1e3
            y_hi = (ind_min_val + hi_bin - 0.5) * freq_step * 1e3

            nc_count = diff.shape[0]
            ch_names = ["X", "Y", "Z"][:nc_count]
            vmax_row = [
                max(
                    float(np.abs(arr_item[row_idx, lo_bin:hi_bin]).max())
                    for arr_item in (diff, before, after)
                )
                for row_idx in range(nc_count)
            ]
            fig, axes = plt.subplots(
                nc_count,
                3,
                figsize=(13.5, 3.2 * nc_count),
                squeeze=False,
                sharex=True,
                sharey=True,
            )
            for row_idx in range(nc_count):
                for col_idx, (panel_name, arr_item) in enumerate(
                    [
                        ("accepted template", diff),
                        ("buffer before RJ", before),
                        ("buffer after RJ", after),
                    ]
                ):
                    ax_sub = axes[row_idx][col_idx]
                    im_plot = ax_sub.imshow(
                        np.abs(arr_item[row_idx, lo_bin:hi_bin]),
                        aspect="auto",
                        origin="lower",
                        extent=[x_extent[0], x_extent[1], y_lo, y_hi],
                        vmin=0.0,
                        vmax=vmax_row[row_idx],
                    )
                    if row_idx == 0:
                        ax_sub.set_title(f"|{panel_name}|", fontsize=11)
                    if col_idx == 0:
                        ax_sub.set_ylabel(
                            f"{ch_names[row_idx]}\nfrequency [mHz]",
                            fontsize=10,
                        )
                    if row_idx == nc_count - 1:
                        ax_sub.set_xlabel(time_label, fontsize=10)
                    ax_sub.tick_params(labelsize=8)
                    cbar = fig.colorbar(im_plot, ax=ax_sub)
                    cbar.ax.tick_params(labelsize=7)
            fig.suptitle(
                f"GB {geometry.basis_name} rj ACCEPTED -- band {rj_seq['band']} | "
                f"walker {rj_seq['walker']} | T{rj_seq['temp']}\n"
                f"band SOURCE-ONLY ll: before = {ll_b:.4e}, "
                f"after = {ll_a:.4e} (Delta = {ll_a - ll_b:+.4e})",
                fontsize=13,
            )
            os.makedirs(self.settings.plot_dir, exist_ok=True)
            fname = os.path.join(
                self.settings.plot_dir,
                f"gb_debug_seq0_rj_accepted_band{rj_seq['band']}"
                f"_w{rj_seq['walker']}_t{rj_seq['temp']}"
                f"_{self.plot_counter:04d}.png",
            )
            fig.savefig(fname, dpi=120, bbox_inches="tight")
            plt.close(fig)
            self.plot_counter += 1
            logger.info(
                "[GB_DEBUG %s] saved rj plot -> %s", self.move.name, fname
            )
        except Exception as exc:
            logger.warning("[GB_DEBUG %s] rj plot skipped: %r", self.move.name, exc)

    def log_band_null(
        self,
        buffer_obj: SubBandBuffer,
    ) -> None:
        """Log the sub-band source-only residual likelihood across temperatures once per step.

        Parameters
        ----------
        buffer_obj : SubBandBuffer
            The active sub-band scratch buffer.
        """
        if not self.tiles_active or self.null_logged:
            return
        try:
            geometry = self._get_domain_geometry()
            freq_step = geometry.frequency_step

            sel_w = self.settings.plot_walker
            sel_b = (
                self.settings.plot_band
                if self.settings.plot_band is not None
                else (len(self.move.band_edges) - 1) // 2
            )
            band_edges_arr = asnumpy(self.move.band_edges)

            combos = asnumpy(buffer_obj.unique_band_combos)
            rows = [
                row_idx
                for row_idx, combo in enumerate(combos)
                if int(combo[1]) == sel_w and int(combo[2]) == sel_b
            ]
            if not rows:
                return
            self.null_logged = True

            # * Every row here is the same band, so one cell geometry (window) covers them all.
            cell = self._cell_geometry(buffer_obj, rows[0])
            ind_min_val = cell.min_frequency_index
            bin_start = max(
                int(np.ceil(band_edges_arr[sel_b] / freq_step - 1e-9)) - ind_min_val, 0
            )
            bin_end = min(
                int(np.floor(band_edges_arr[sel_b + 1] / freq_step + 1e-9)) + 1 - ind_min_val,
                cell.num_frequencies,
            )

            diff_comp = float(buffer_obj.settings.differential_component)
            nc_count = buffer_obj.nchannels

            band_np = asnumpy(buffer_obj._materialize(buffer_obj.band_buffer))
            psd_np = asnumpy(buffer_obj._materialize(buffer_obj.psd_buffer))
            msgs = []
            for row_idx in sorted(rows, key=lambda idx_val: int(combos[idx_val, 0])):
                temp_idx = int(combos[row_idx, 0])
                formatted_slab = self._format_slab(band_np[row_idx], cell)
                resid_slab = formatted_slab[:, bin_start:bin_end]

                if geometry.is_stft:
                    if buffer_obj.tdi_channel_setup == "XYZ":
                        inv_c = np.transpose(
                            psd_np[row_idx].reshape(nc_count, nc_count, cell.num_times, cell.num_frequencies),
                            (0, 1, 3, 2),
                        )[:, :, bin_start:bin_end]
                        ll_val = -0.5 * 4.0 * diff_comp * float(
                            np.einsum("ifk,ijfk,jfk->", resid_slab.conj(), inv_c, resid_slab).real
                        )
                    else:
                        inv_c = np.transpose(
                            psd_np[row_idx].reshape(nc_count, cell.num_times, cell.num_frequencies),
                            (0, 2, 1),
                        )[:, bin_start:bin_end]
                        ll_val = -0.5 * 4.0 * diff_comp * float(
                            np.sum((resid_slab.conj() * resid_slab).real * inv_c.real)
                        )
                else:
                    if buffer_obj.tdi_channel_setup == "XYZ":
                        inv_c = psd_np[row_idx].reshape(nc_count, nc_count, cell.num_frequencies, -1)[:, :, bin_start:bin_end]
                        ll_val = -0.5 * 4.0 * diff_comp * float(
                            np.einsum("ifk,ijfk,jfk->", resid_slab, inv_c.real, resid_slab)
                        )
                    else:
                        inv_c = psd_np[row_idx].reshape(nc_count, cell.num_frequencies, -1)[:, bin_start:bin_end]
                        ll_val = -0.5 * 4.0 * diff_comp * float(
                            np.sum(resid_slab * inv_c.real * resid_slab)
                        )
                msgs.append(f"T{temp_idx}: {ll_val:.6e}")
            logger.info(
                "[GB_DEBUG %s] sub-band SOURCE-ONLY residual ll "
                "(band %d, walker %d, bins %d:%d): %s "
                "(cold chain at injection should be ~0)",
                self.move.name,
                sel_b,
                sel_w,
                ind_min_val + bin_start,
                ind_min_val + bin_end,
                "; ".join(msgs),
            )
        except Exception as exc:
            logger.warning(
                "[GB_DEBUG %s] band-null log skipped: %r", self.move.name, exc
            )

    def plot_band(
        self,
        buffer_obj: SubBandBuffer,
        params_add: NDArrayLike,
        data_index: NDArrayLike,
        swap_N_vals: NDArrayLike,
        ll_diff_kept: NDArrayLike,
        map_to_update_cpu: CoordinateMapCPU,
        keep_mask: NDArrayLike,
        move_idx: int,
        stage: ProposalStage = "in-model",
    ) -> None:
        """Save one time-frequency figure for the chosen cell across temperatures.

        Parameters
        ----------
        buffer_obj : SubBandBuffer
            The active sub-band scratch buffer.
        params_add : NDArrayLike
            Coordinates of proposed sources, shaped ``(num_kept, ndim)``.
        data_index : NDArrayLike
            Buffer slot indices, shaped ``(num_kept,)``.
        swap_N_vals : NDArrayLike
            Sample count multipliers per source, shaped ``(num_kept,)``.
        ll_diff_kept : NDArrayLike
            Proposed log-likelihood differences for kept proposals.
        map_to_update_cpu : CoordinateMapCPU
            3-tuple of (temps_all, walkers_all, bands_all) indexing the full batch.
        keep_mask : NDArrayLike
            Boolean mask selecting the kept proposals from the full batch.
        move_idx : int
            Repeat proposal or RJ proposal round index.
        stage : ProposalStage, default="in-model"
            Proposal context identifier (``"rj"`` or ``"in-model"``).
        """
        if not self.tiles_active:
            return
        _validate_stage(stage)
        _validate_coordinate_array(params_add, "params_add")
        num_kept = len(params_add)
        _validate_1d_array(data_index, "data_index", expected_length=num_kept)
        if swap_N_vals is not None:
            _validate_1d_array(swap_N_vals, "swap_N_vals", expected_length=num_kept)
        _validate_1d_array(ll_diff_kept, "ll_diff_kept", expected_length=num_kept)
        _validate_coordinate_map_cpu(map_to_update_cpu, "map_to_update_cpu")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            geometry = self._get_domain_geometry()
            freq_step = geometry.frequency_step

            sel_w = self.settings.plot_walker
            sel_b = (
                self.settings.plot_band
                if self.settings.plot_band is not None
                else (len(self.move.band_edges) - 1) // 2
            )

            orig = np.where(asnumpy(keep_mask).astype(bool))[0]
            if orig.size == 0:
                return
            temps_all, walkers_all, bands_all = map_to_update_cpu
            pos = [
                pos_idx
                for pos_idx, j_idx in enumerate(orig)
                if int(walkers_all[j_idx]) == sel_w and int(bands_all[j_idx]) == sel_b
            ]
            if not pos:
                return

            if stage in self.plotted_stages:
                return
            self.plotted_stages.add(stage)

            pos.sort(key=lambda pos_idx: int(temps_all[orig[pos_idx]]))

            params_phys = self.move.transform_fn.both_transforms(
                params_add, xp=self.move.xp
            )
            di_np = asnumpy(data_index)
            ll_np = asnumpy(self.move.xp.asarray(ll_diff_kept))

            num_panels = len(pos)
            ncols = min(num_panels, 4)
            nrows = (num_panels + ncols - 1) // ncols
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(5.6 * ncols, 4.4 * nrows),
                squeeze=False,
                sharey=True,
            )
            for ax_unused in axes.flat[num_panels:]:
                ax_unused.set_visible(False)

            cell = self._cell_geometry(buffer_obj, int(asnumpy(data_index)[pos[0]]))
            ind_min_val = cell.min_frequency_index
            if geometry.is_stft:
                band_edges_arr = asnumpy(self.move.band_edges)
                band_start = max(
                    int(np.ceil(band_edges_arr[sel_b] / freq_step - 1e-9)) - ind_min_val, 0
                )
                band_end = min(
                    int(np.floor(band_edges_arr[sel_b + 1] / freq_step + 1e-9)) + 1 - ind_min_val,
                    cell.num_frequencies,
                )
                lo_bin = max(band_start - 2, 0)
                hi_bin = min(band_end + 2, cell.num_frequencies)
                total_time_span = geometry.num_times * geometry.time_step
                if total_time_span >= 86400.0 * 365.25:
                    time_scale = 86400.0 * 365.25
                    time_label = "time [years]"
                else:
                    time_scale = 86400.0
                    time_label = "time [days]"
                x_extent = [
                    geometry.start_time / time_scale,
                    (geometry.start_time + total_time_span) / time_scale,
                ]
            else:
                lo_bin = None
                hi_bin = None
                time_label = "WDM time pixel (X)"
                x_extent = [0, geometry.num_times]

            for panel_idx, pos_item in enumerate(pos):
                ax_panel = axes.flat[panel_idx]
                temp_val = int(temps_all[orig[pos_item]])
                slab_idx = int(di_np[pos_item])
                freq_0 = float(asnumpy(params_phys[pos_item, 1]))
                ll_val = float(ll_np[pos_item])

                raw_slab = asnumpy(buffer_obj.band_buffer[slab_idx][0])
                if geometry.is_stft:
                    reshaped = raw_slab.reshape(cell.num_times, cell.num_frequencies)
                    tile = np.abs(np.transpose(reshaped, (1, 0)))
                    sub_lo = lo_bin
                    sub_hi = hi_bin
                else:
                    tile = np.abs(raw_slab.reshape(geometry.num_frequencies, geometry.num_times))
                    local = int(round(freq_0 / freq_step)) - ind_min_val
                    sub_lo = max(local - 2, 0)
                    sub_hi = min(local + 3, tile.shape[0])

                sub_tile = tile[sub_lo:sub_hi]
                y_lo = (ind_min_val + sub_lo - 0.5) * freq_step * 1e3
                y_hi = (ind_min_val + sub_hi - 0.5) * freq_step * 1e3

                im_tile = ax_panel.imshow(
                    sub_tile,
                    aspect="auto",
                    origin="lower",
                    extent=[
                        x_extent[0],
                        x_extent[1],
                        y_lo,
                        y_hi,
                    ],
                )
                ax_panel.axhline(
                    freq_0 * 1e3,
                    color="r",
                    ls="--",
                    lw=1.2,
                    label=f"f0 = {freq_0 * 1e3:.4f} mHz",
                )
                ll_txt = (
                    "forbidden proposal"
                    if ll_val < -1e290
                    else f"$\\Delta$logL = {ll_val:.3e}"
                )
                ax_panel.set_title(f"T{temp_val}  {ll_txt}", fontsize=11)
                ax_panel.set_xlabel(time_label, fontsize=10)
                if panel_idx % ncols == 0:
                    ax_panel.set_ylabel("frequency [mHz]", fontsize=10)
                ax_panel.tick_params(labelsize=9)
                ax_panel.legend(loc="upper right", fontsize=8, framealpha=0.9)
                cbar = fig.colorbar(im_tile, ax=ax_panel)
                cbar.ax.tick_params(labelsize=8)

            fig.suptitle(
                f"GB {geometry.basis_name} {stage} proposal -- |residual| around the source | "
                f"band {sel_b} | walker {sel_w} | repeat {move_idx} | "
                f"all temperatures",
                fontsize=13,
            )
            os.makedirs(self.settings.plot_dir, exist_ok=True)
            fname = os.path.join(
                self.settings.plot_dir,
                f"gb_debug_{stage.replace('-', '')}_band{sel_b}_w{sel_w}"
                f"_move{move_idx}_{self.plot_counter:04d}.png",
            )
            fig.savefig(fname, dpi=130, bbox_inches="tight")
            plt.close(fig)
            self.plot_counter += 1
            logger.info("[GB_DEBUG %s] saved band plot -> %s", self.move.name, fname)
        except Exception as exc:
            logger.warning(
                "[GB_DEBUG %s] band plot skipped: %r", self.move.name, exc
            )
