"""Galactic-binary specialized stretch / RJ moves and supporting infrastructure."""

from __future__ import annotations

import os
import time
import logging
import warnings
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from inspect import Attribute
from types import ModuleType
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

import numpy as np
import numpy

from eryn.state import BranchSupplemental
from gbgpu.gbgpu import GBGPU
from gbgpu.utils.utility import get_N, get_fdot
from scipy import stats
from scipy.special import gammaln

from ... import sensitivity
from ...detector import sangria
from ...utils.constants import *
from ...analysiscontainer import (
    AnalysisContainer,
    AnalysisContainerArray,
    BandView,
    band_gpu_assignment,
)

from ...domains import DomainSettingsBase, FDSettings, STFTSettings, WDMSettings
from ...sensitivity import SensitivityMatrixBase
from ...utils.parallelbase import LISAToolsParallelModule
from ...utils.utility import asnumpy
from ..galaxyglobal import fit_each_leaf, make_gmm, run_gb_bulk_search
from gbgpu.gb_likelihood import (
    BandLikelihoodEngine,
    FDBandLikelihoodEngine,
    SwapLLResult,
    WDMBandLikelihoodEngine,
    make_band_likelihood_engine,
)
from .globalfitmove import GFCombineMove, GlobalFitMove
from ..priors.gbpriors import get_fdot_mojito

try:
    import cupy as cp
    import cupy
    gpu_available = True
except ModuleNotFoundError:
    import numpy as cp

    gpu_available = False

from eryn.moves import GroupStretchMove, Move, StretchMove
from eryn.moves.multipletry import get_mt_computations, logsumexp
from eryn.paraensemble import ParaEnsembleSampler
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.utils import PeriodicContainer
from eryn.utils.utility import groups_from_inds

from ...diagnostic import inner_product
from ...sampling.prior import FullGaussianMixtureModel, GBPriorWrap
from ...utils.utility import get_array_module, get_groups_from_band_structure, searchsorted2d_vec
from ..state import GFState, ensure_leaf_cap_fields

from .gbdebug import (
    CoordinateMapCPU,
    GBDebugSettings,
    GBSamplerDebugger,
    InModelSequenceDict,
    PickedSourcesDict,
    ProposalStage,
    RJSequenceDict,
    SeqPickStrategy,
    UnitSnapshot,
)
from .gbbands import (
    BandScheduler,
    BandSorter,
    StoreWindowTrackCounter,
    SubBandBuffer,
    estimate_buffer_preload_limits,
)
from ...utils.typing import NDArrayLike

if TYPE_CHECKING:
    from eryn.model import Model
else:
    Model = Any

__all__ = [
    "GBSpecialStretchMove",
    "GBDebugSettings",
    "GBSamplerDebugger",
    "PickedSourcesDict",
    "InModelSequenceDict",
    "RJSequenceDict",
    "UnitSnapshot",
    "CoordinateMapCPU",
    "ProposalStage",
    "SeqPickStrategy",
    "GBSigHetSettings",
    "GBLeafCapSettings",
]

logger = logging.getLogger(__name__)

class _NoOpMempool:
    """CPU stand-in for ``cupy.get_default_memory_pool()`` — calls become no-ops."""

    def free_all_blocks(self):
        return


from dataclasses import dataclass

from eryn.state import Branch
from eryn.utils import TransformContainer

from .gbbands import (
    BandScheduler,
    BandSorter,
    Buffer,
    SubBandBuffer,
    free_device_bytes,
    split_rows_by_budget,
    stft_batch_budget_bytes,
    _RoutedBandEngine,
    make_routed_band_engine,
    pack_special_index,
    return_x,
    unpack_special_index,
)
from .gbmemprobe import make_probe
from ...utils.devicereplicas import release_gb_comp_groups



class _ProposeTimer:
    """Accumulating wall-clock stage timer for one GB ``propose()`` call.

    Localizes where a proposal spends its time (GPU-efficiency diagnosis):
    stages are accumulated with :meth:`span` and reported as a single
    sorted INFO line at the end of the propose. Overhead is a pair of
    ``perf_counter`` calls per span, so it stays on by default.

    On a CuPy backend the numbers are HOST wall time per stage. Because
    kernel launches are asynchronous, device work is attributed to the
    stage that *forces* the sync (the next ``asnumpy`` / ``.item()`` /
    explicit synchronize). Set ``GBDebugSettings.prop_timing_sync`` to synchronize the
    device at every span boundary instead — slightly slower overall, but
    each stage then carries exactly its own kernel time. Either view is
    diagnostic: if host time dominates in stages with tiny kernels
    (``inmodel_repeats`` with few cells), the run is launch-overhead-bound
    (too few sub-bands/cells per launch to keep the GPU busy).
    """

    __slots__ = ("stages", "counts", "_sync", "_probe")

    def __init__(self, sync_fn=None, probe=None):
        self.stages: dict = {}
        self.counts: dict = {}
        self._sync = sync_fn
        self._probe = probe

    @contextmanager
    def span(self, name: str):
        # * The memory probe wraps the timed region, so its pool queries are not timed.
        with (self._probe.span(name) if self._probe is not None else nullcontext()):
            if self._sync is not None:
                self._sync()
            t0 = time.perf_counter()
            try:
                yield
            finally:
                if self._sync is not None:
                    self._sync()
                self.stages[name] = self.stages.get(name, 0.0) + (
                    time.perf_counter() - t0
                )

    def count(self, name: str, n: int = 1) -> None:
        self.counts[name] = self.counts.get(name, 0) + int(n)

    def report(self, total: float) -> str:
        top = (
            "sorter_build", "friend_index", "resid_open_close", "ll_checks",
            "run_proposal", "run_tempering", "write_back", "sorter_rebuild",
            "band_info", "ll_inject_final", "ll_inject_drift", "mempool_free",
        )
        items = sorted(self.stages.items(), key=lambda kv: -kv[1])
        tracked = sum(v for k, v in self.stages.items() if k in top)
        parts = [f"{k}={v:.3f}s" for k, v in items]
        cparts = [f"{k}={v}" for k, v in sorted(self.counts.items())]
        return (
            f"total={total:.3f}s tracked={tracked:.3f}s "
            f"untracked={max(total - tracked, 0.0):.3f}s | "
            + " ".join(parts)
            + (" | " + " ".join(cparts) if cparts else "")
        )


def _tspan(tm, name: str):
    """Timer span or no-op when the propose-level timer is absent."""
    return tm.span(name) if tm is not None else nullcontext()


# * Currently only using uniform priors for which the variance is w^2 / 12 for width w
PRIOR_PRECISION_FACTOR = 12.0

# * Step of the numerical Jacobian of the transform container: 1e-6 of |x| with a floor of 1e-3 in
# * |x|. It is exact for the linear f0 and fdot transforms, also for fdot sampled as fdot x Tobs^2.
JACOBIAN_STEP_RELATIVE = 1e-6
JACOBIAN_STEP_FLOOR = 1e-3

# * Cached Fisher Information matrices in the gb branch supplemental, per leaf slot:
# * FIM in the sampling basis, the prior widths, the blocks served, and whether the slot holds one.
_FIM_STORE_KEYS = ("fim_matrix", "fim_widths", "fim_age", "fim_valid")


def fisher_proposal_factor(info_sampling, prior_widths, beta, xp=np):
    """Factor L with L L^T = (beta F + P)^-1 per source, P = diag(12 / w^2), in correlation form.

    ``info_sampling`` (n_src, ndim, ndim) is the Fisher matrix F in the sampling basis,
    ``prior_widths`` (n_src, ndim) or (ndim,) the widths w of the uniform prior box in the same
    units and ``beta`` a scalar or (n_src,) inverse temperatures. The sum is whitened by its own
    diagonal d: C = (beta F + P) / (d d^T) = V diag(lam) V^T and L = V diag(lam^-1/2) / d. No
    eigenvalue floor is needed, since P holds every direction the data leave open. A source whose
    matrix is not finite or not positive definite gets the prior factor diag(w / sqrt(12)).

    Returns ``(factor, diagnostics)``. ``diagnostics`` holds per source ``fallback`` (the prior
    factor was used), ``condition`` (largest over smallest eigenvalue of C, NaN for a fallback) and
    ``prior_dominated`` (an eigenvector of C whose eigenvalue is more than half prior).
    """
    info = xp.asarray(info_sampling, dtype=xp.float64)
    n_src, ndim = info.shape[0], info.shape[-1]
    widths = xp.broadcast_to(xp.asarray(prior_widths, dtype=xp.float64), (n_src, ndim))
    betas = xp.broadcast_to(xp.asarray(beta, dtype=xp.float64).reshape(-1), (n_src,))
    precision = PRIOR_PRECISION_FACTOR / widths ** 2
    identity = xp.eye(ndim)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        total = betas[:, None, None] * info + precision[:, :, None] * identity[None]
        scale = xp.sqrt(xp.diagonal(total, axis1=1, axis2=2))
        finite = xp.all(xp.isfinite(total), axis=(1, 2)) & xp.all(scale > 0.0, axis=1)
        safe_scale = xp.where(finite[:, None], scale, 1.0)
        # ! one non-finite matrix would make the batched eigh fail (numpy) or return NaN (cupy) for
        # ! the whole batch, so it is replaced by the identity here and gets the prior factor below
        corr = xp.where(
            finite[:, None, None],
            total / safe_scale[:, :, None] / safe_scale[:, None, :],
            identity[None],
        )
        evals, evecs = xp.linalg.eigh(corr)
        usable = finite & xp.all(evals > 0.0, axis=1)
        safe_evals = xp.where(usable[:, None], evals, 1.0)
        factor = evecs / xp.sqrt(safe_evals)[:, None, :] / safe_scale[:, :, None]
        prior_factor = (widths / np.sqrt(PRIOR_PRECISION_FACTOR))[:, :, None] * identity[None]
        factor = xp.where(usable[:, None, None], factor, prior_factor)
        # * prior share of eigenvector v: v^T (d^-1 P d^-1) v / lam, between 0 and 1
        prior_share = xp.einsum("nik,ni->nk", evecs ** 2, precision / safe_scale ** 2) / safe_evals
        diagnostics = {
            "fallback": ~usable,
            "condition": xp.where(usable, safe_evals.max(axis=1) / safe_evals.min(axis=1), xp.nan),
            # more than half of the eigenvalue from the prior
            "prior_dominated": usable & xp.any(prior_share > 0.5, axis=1),
        }

    return factor, diagnostics



@dataclass
class GBSigHetSettings:
    """Settings for signal-heterodyne trust region and reference drift refreshes.

    Controls mid-block drift refresh thresholds, trust region excursion bounds,
    and anchor and drift diagnostic checks for signal heterodyne likelihood evaluation.
    """

    refresh_every: int = 20
    refresh_dphase: float = 0.5
    refresh_min_beta: float = 0.1
    trust_dlna: float = 1.5
    trust_dphase: float = 0.5
    trust_snr_c: float = 30.0
    trust_dlna_min: float = 0.3
    anchor_check: bool = False
    drift_check: bool = False


@dataclass
class GBLeafCapSettings:
    """Settings for per-band progressive leaf capacity limits during search mode.

    Truncates the prior on the number of active leaves per frequency band to
    encourage progressive convergence before adding more leaves.
    """

    start: Optional[int] = None
    min_iters: int = 50
    ll_nsigma: float = 3.0
    require_occupancy: bool = True
    update: bool = True

    # * Helper property to determine if the leaf cap is active
    @property
    def enabled(self) -> bool:
        return self.start is not None


def sweep_tempering_rows(values, betas, log_uniforms, bands, walkers, source_counts,
                         walker_totals, nleaves_max, log_factorial):
    """Swap decisions of the tempering loop, made on the host from each grid row's value table.

    Row ``r`` holds one cell of band ``bands[r]`` per temperature: column ``t`` is walker
    ``walkers[r, t]``, whose cell holds ``source_counts[r, t]`` sources. ``values[r, t, j]`` is the
    ln L of column ``t`` holding the template of column ``j``, up to a term of column ``t`` alone.
    The decisions go band after band (ascending), pair after pair from the top, the rows of a band
    together: the order of the swap loop, whose chunks never mix bands. The rows of one band touch
    distinct walker totals, so their decisions do not interact.

    The rule is the swap loop's. For the pair ``(t, t - 1)``, ``betas[r, t]`` and ``betas[r, t - 1]``
    weight the ln L changes of the two columns; the change of ln k! of both walkers is added, with
    ``walker_totals[t, w]`` the sources of walker ``w`` at temperature ``t`` over all bands (updated
    in place on acceptance); a swap that gives a walker more than ``nleaves_max`` sources is refused;
    ``log_uniforms[r, t - 1]`` is the log of the uniform draw. A template keeps moving down the
    ladder when the next pair accepts it again.

    Returns:
        ``template_column[r, t]``, the column whose template ends at column ``t``, and
        ``accepted[r, t - 1]``, whether the swap of the pair ``(t, t - 1)`` was accepted.
    """
    num_rows, num_temps = betas.shape
    template_column = np.tile(np.arange(num_temps), (num_rows, 1))
    accepted = np.zeros((num_rows, num_temps - 1), dtype=bool)
    for band in np.unique(bands):
        rows = np.flatnonzero(bands == band)
        for hot in range(num_temps - 1, 0, -1):
            cold = hot - 1
            from_hot = template_column[rows, hot]
            from_cold = template_column[rows, cold]
            paccept = (
                betas[rows, hot] * (values[rows, hot, from_cold] - values[rows, hot, from_hot])
                + betas[rows, cold] * (values[rows, cold, from_hot] - values[rows, cold, from_cold])
            )
            hot_walkers = walkers[rows, hot]
            cold_walkers = walkers[rows, cold]
            sources_hot = source_counts[rows, from_hot]
            sources_cold = source_counts[rows, from_cold]
            k_hot = walker_totals[hot, hot_walkers]
            k_cold = walker_totals[cold, cold_walkers]
            k_hot_new = k_hot - sources_hot + sources_cold
            k_cold_new = k_cold - sources_cold + sources_hot
            fits = (k_hot_new <= nleaves_max) & (k_cold_new <= nleaves_max)
            log_weight_change = (
                log_factorial[np.minimum(k_hot_new, nleaves_max)]
                + log_factorial[np.minimum(k_cold_new, nleaves_max)]
                - log_factorial[k_hot]
                - log_factorial[k_cold]
            )
            take = fits & (paccept + log_weight_change > log_uniforms[rows, cold])
            walker_totals[hot, hot_walkers[take]] = k_hot_new[take]
            walker_totals[cold, cold_walkers[take]] = k_cold_new[take]
            template_column[rows[take], hot] = from_cold[take]
            template_column[rows[take], cold] = from_hot[take]
            accepted[rows, cold] = take
    return template_column, accepted


# MHMove needs to be to the left here to overwrite GBBruteRejectionRJ RJ proposal method
class GBSpecialBase(GlobalFitMove, GroupStretchMove, Move, LISAToolsParallelModule):
    """Base class for GB-specific stretch / reversible-jump moves.

    Combines :class:`GlobalFitMove`, :class:`eryn.moves.GroupStretchMove`,
    :class:`Move`, and :class:`LISAToolsParallelModule` so each GB move can
    use try-force rejection, optional phase maximization, and GPU-resident
    band-aware buffers (:class:`Buffer`, :class:`BandSorter`).

    # TODO/DOCS: full argument list: many constructor kwargs are passed
    through unmodified to ``GroupStretchMove``. Intended use is via the
    concrete subclasses :class:`GBSpecialStretchMove`,
    :class:`GBSpecialRJPriorMove`, :class:`GBSpecialRJSearchMove`,
    :class:`GBSpecialRJSerialSearchMCMC`, and
    :class:`GBSpecialRJRefitMove`.

    Args:
        gb: :class:`gbgpu.GBGPU` instance.
        priors: :class:`ProbDistContainer` for in-model GB parameters.
        start_freq_ind: Inclusive starting index into the global ``f_arr``
            (FD path); ignored on the WDM path.
        data_length: Length of the data array per channel (FD path); the
            WDM path sizes its per-band buffers from
            ``WDMSettings.Nf_active`` / ``Nt_active`` instead.
        acs: :class:`AnalysisContainerArray` used for SETUP ONLY: the
            basis-domain settings on ``acs.settings`` drive every
            domain-dependent choice (FD vs WDM) and the initial parent
            binding. It is not stored -- at run time everything reads
            from / fills into the ACA that arrives with the model in
            :meth:`propose` (re-binding if it changed).
        band_edges: Frequency-band edges.
        band_N_vals: Per-band waveform sample counts.
        gpu_priors: Branch-keyed GPU-resident priors.
        waveform_kwargs: Forwarded to ``gb`` waveform calls.
        parameter_transforms: :class:`TransformContainer` for GBs.
        snr_lim: Optional SNR cut.
        rj_proposal_distribution: Distribution used to draw RJ proposals.
        is_rj_prop: Marks this move as a reversible-jump proposal.
        num_repeat_proposals: Inner repeat count per call.
        name: Move name (used for logging and bookkeeping).
        use_prior_removal: If ``True``, draw RJ proposals from the prior.
        phase_maximize: If ``True``, marginalize over phase in the
            likelihood.
        ranks_needed / gpus: MPI / GPU resource requests.
        num_band_preload: Number of bands preloaded per call.
        run_swaps: Whether to run band-temperature swaps.
        max_data_store_size: Cap on the per-iteration data store size.
        force_backend: Optional backend override.
        gb_wdm_comp: Optional :class:`gbgpu.gbcomps.GBWDMComputations`
            instance. Required when ``acs.settings`` is a
            :class:`~lisatools.domains.WDMSettings`; ignored otherwise.
        gb_stft_comp: Optional :class:`gbgpu.gbcomps.STFTGBComputations`
            instance. Required when ``mgh.settings`` is an
            :class:`~lisatools.domains.STFTSettings`; ignored otherwise.
        num_stationary_fim_steps: Blocks of in-model repeats a source's Fisher
            matrix serves before it is recomputed at the source's position. 1
            recomputes it at every block.
    """

    # * STFT tempering decides its swaps from the cross terms of each grid row; False runs the swap
    # * loop.
    TEMPERING_CROSS_TERMS = True
    # Sum the DFFR cross terms over each band's store window (True) or over the whole cell. TDI on
    # the fly always takes the whole cell: its carrier can spike out of the window near antenna nulls.
    TEMPERING_BAND_WINDOW = True
    # * DFFR proposal cells take the width of their band's class (a power of two) instead of the
    # * widest window, and the classes of a band unit run one after another.
    STFT_WIDTH_CLASSES = True
    # * The same for the tempering cells of DFFR rows. A switch of its own, so that the classes of
    # * the two phases can be measured apart.
    TEMPERING_WIDTH_CLASSES = True
    # * Empty the pool just before a proposal phase. Its band sorter subsets live for the whole propose: in a
    # * free block they pin it, 1.6 GiB of a 2.6 GiB block from the second propose on.
    EMPTY_POOL_BEFORE_PROPOSAL = True
    # * True keeps the parent closed during a pass: a cell adds back only its own band's cold sources, and the
    # * other bands of its class stay subtracted at their values from the start of the pass. The cells of a
    # * class still run in parallel. False first adds the class back to the parent, as before.
    PARALLEL_GIBBS = True

    @property
    def xp(self) -> Union[ModuleType, numpy , cupy]:
        """Active array module (NumPy or CuPy) for this move."""
        return self.backend.xp

    def __init__(
        self,
        gb: GBGPU,
        priors,
        start_freq_ind,
        data_length,
        acs,
        band_edges,
        band_N_vals,
        gpu_priors,
        *args,
        waveform_kwargs={},
        parameter_transforms: Optional[TransformContainer] = None,
        snr_lim=1e-10,
        rj_proposal_distribution=None,
        is_rj_prop=False,
        num_repeat_proposals=100,
        name=None,
        use_prior_removal=False,
        phase_maximize=False,
        ranks_needed=0,
        num_band_preload=None,
        num_bands_preload_temp=None,
        run_swaps=True,
        max_data_store_size=6000,
        force_backend=None,
        gb_wdm_comp=None,
        gb_fd_comp=None,
        gb_stft_comp=None,
        orbits=None,
        tdi_config=None,
        t_ref=0.0,
        search_kwargs=None,
        stretch_probability=0.5,
        band_units=2,
        gamma_base=0.84, # = 2.38 / sqrt(D) where D=8
        num_stationary_fim_steps: int = 1,
        debug_settings: Optional[GBDebugSettings] = None,
        sighet_settings: Optional[GBSigHetSettings] = None,
        leaf_cap_settings: Optional[GBLeafCapSettings] = None,
        stft_store_windows=None,
        **kwargs,
    ):
        LISAToolsParallelModule.__init__(self, force_backend=force_backend)
        GlobalFitMove.__init__(self, name=name)
        Move.__init__(self, *args, return_gpu=True, **kwargs)

        self.a = float(kwargs.get("a", 2.0))
        self.nfriends = int(kwargs.get("nfriends", 32))
        self.lift_scale = float(kwargs.get("lift_scale", 2.0))
        self.lift_kmax = int(kwargs.get("lift_kmax", 2))
        self._lift = None
        self._inds_run = {}
        self._last_log_norm = None
        self._turns = self.xp.arange(-self.lift_kmax, self.lift_kmax + 1)
        self._periodic_cache = {}
        self.return_gpu = True
        self.use_gpu = self.backend.uses_cupy

        self.force_backend = force_backend
        self.ranks_needed = ranks_needed
        self.gpu_priors = gpu_priors
        self.num_repeat_proposals = num_repeat_proposals

        # Buffer preload sizing: use estimate_buffer_preload_limits if not explicitly set
        est_preload, est_preload_temp = estimate_buffer_preload_limits(
            basis_settings=getattr(acs, "settings", None),
            nchannels=getattr(acs, "nchannels", 3),
            tdi_setup=getattr(acs, "tdi_channel_setup", "XYZ"),
            max_data_store_size=max_data_store_size,
            ntemps=int(kwargs.get("ntemps", 4) or 4),
            xp=self.xp,
            gpus=kwargs.get("gpus", None),
        )

        # * With store windows, a setting left at None is sized from the memory free at each build,
        # * where the cell width is known. A given setting runs as it is; the estimate serves the rest.
        self._size_proposal_batches = num_band_preload is None and stft_store_windows is not None
        self._size_tempering_batches = num_bands_preload_temp is None and stft_store_windows is not None
        if stft_store_windows is not None:
            self._stft_bytes_per_bin = int(getattr(acs, "nchannels", 3)) * int(acs.settings.NT) * 16
            if not SubBandBuffer.READ_PARENT_NOISE_ROWS:
                # A cell then owns its noise window, and a new twin holds one until it is built.
                self._stft_bytes_per_bin += (
                    int(np.prod(acs.shape_sens)) * int(acs.settings.NT) * np.dtype(acs.noise_dtype).itemsize
                )

        # Both arrive from GBSettings through recipe.py; the estimate is the fallback.
        if num_band_preload is None:
            num_band_preload = est_preload
        if num_bands_preload_temp is None:
            num_bands_preload_temp = est_preload_temp

        self.num_band_preload = int(num_band_preload)
        self.num_bands_preload_temp = int(num_bands_preload_temp)
        self.band_preload_size = self.max_data_store_size = max_data_store_size
        self.use_prior_removal = use_prior_removal
        self.has_setup_group = False

        # * Initialize grouped settings dataclasses and debugger
        self.debugger = GBSamplerDebugger(self, debug_settings=debug_settings)
        self.debug_settings = self.debugger.settings

        self.leaf_cap_settings = (
            deepcopy(leaf_cap_settings)
            if leaf_cap_settings is not None
            else GBLeafCapSettings()
        )
        self._band_leaf_cap = None

        self.sighet_settings = (
            deepcopy(sighet_settings)
            if sighet_settings is not None
            else GBSigHetSettings()
        )

        self.priors = priors
        self.gb = gb
        self.gb_wdm_comp = gb_wdm_comp
        self.gb_stft_comp = gb_stft_comp
        # * Per-band STFT store windows; None keeps full-grid cells.
        self.stft_store_windows = stft_store_windows
        self._track_counter = None
        self.run_swaps = run_swaps

        if self.backend.uses_cupy:
            self.mempool = self.xp.get_default_memory_pool()
        else:
            # TODO: add a NoOpMempool setup to the backend itself
            self.mempool = _NoOpMempool()

        self.band_edges = band_edges
        self.num_bands = len(band_edges) - 1
        self.start_freq_ind = start_freq_ind
        self.data_length = data_length
        self.waveform_kwargs = waveform_kwargs
        self.parameter_transforms = parameter_transforms

        self._configure_domain(acs)
        self.phase_maximize = phase_maximize

        self.snr_lim = snr_lim

        self.band_edges = self.xp.asarray(self.band_edges)

        self.rj_proposal_distribution = rj_proposal_distribution
        self.is_rj_prop = is_rj_prop or (self.rj_proposal_distribution is not None)

        # if self.is_rj_prop:
        #     if (self.num_repeat_proposals != 1):
        #         print("Adjusting repeat proposals to 1 for RJ.")

        #     self.num_repeat_proposals = 1

        # setup N vals for bands
        self.band_N_vals = self.xp.asarray(band_N_vals) if band_N_vals is not None else None

        self.num_proposals = 0
        self.search_kwargs = search_kwargs
        self.stretch_probability = float(stretch_probability)
        self.band_units = max(1, int(band_units))
        self.jump_scales = float(gamma_base) * self.xp.array([0.2, 1.0, 2.5])
        self.jump_scales_probs = self.xp.array([0.2, 0.6, 0.2])
        fim_interval = num_stationary_fim_steps
        if int(fim_interval) != fim_interval or fim_interval < 1:
            raise ValueError(
                f"num_stationary_fim_steps must be a positive integer, got {fim_interval}"
            )
        self.num_stationary_fim_steps = int(num_stationary_fim_steps)
        self._held_fim_rows = None

        # Parent binding: config-only FD comp + move-level engine. A user-
        # supplied gb_fd_comp is honored as-is; otherwise one is built from
        # the ACA's FDSettings -- data holders are passed to it at call
        # time, so ACA changes only rewire the engine (see _bind_parent_acs,
        # re-invoked from propose()).
        self.transform_fn = self.parameter_transforms
        self._gb_fd_comp_user_supplied = gb_fd_comp is not None
        self.gb_fd_comp = gb_fd_comp
        self._proposal_orbits = orbits
        self._proposal_tdi_config = tdi_config
        self._t_ref = float(t_ref)
        self._bind_parent_acs(acs)


    def _configure_domain(self, acs) -> None:
        """Derive ``self.fd`` / ``self.df`` / ``self._basis_settings`` from an ACA.

        The band-index math uses ``df = 1 / Tobs`` consistently across FD
        and WDM (FD: equals ``acs.df``; WDM: ``acs.df == layer_df`` differs,
        so we recompute). ``self.fd`` is only meaningful in the FD path.
        """

        # TODO: make this more generic
        if isinstance(acs.settings, FDSettings):
            self.fd = acs.f_arr.copy()
            self.df = float(self.fd[1] - self.fd[0])
        elif isinstance(acs.settings, WDMSettings):
            self.fd = None
            self.df = 1.0 / acs.settings.Tobs
        elif isinstance(acs.settings, STFTSettings):
            # STFT's .df IS the FD bin width (1 / Tobs), so the band-index
            # math matches the FD/WDM convention directly.
            self.fd = None
            self.df = float(acs.settings.df)
        else:
            raise NotImplementedError(
                f"GBSpecialBase does not support basis domain "
                f"{type(acs.settings).__name__}."
            )
        self._basis_settings = acs.settings

    def _bind_parent_acs(self, acs) -> None:
        """(Re)bind the parent-level engine to ``acs``.

        Every run-time fill / likelihood targets the ACA that arrives with
        the model in :meth:`propose`; the constructor ACA only provides the
        initial binding. Post-2026-07 the FD comp is CONFIG-ONLY
        (``GBFDComputations(fd_settings, t_ref, ...)``) -- data holders are
        passed at get_ll/fill_global time, so an ACA change only refreshes
        the engine wiring here, never any C-side data pointers.
        """
        self._parent_acs = acs

        token = (id(acs), id(acs.linear_data_arr[0]))
        if getattr(self, "_parent_acs_token", None) == token:
            return

        # The parent is itself a window of the global rfft grid starting at
        # ``start_freq_ind``; the engine's bounds mask uses this when the
        # holder itself can't resolve per-row starts. Only relevant for FD.
        if isinstance(acs.settings, FDSettings) and self.start_freq_ind is not None:
            self._parent_start_inds = self.xp.full(
                int(acs.acs_total_entries), int(self.start_freq_ind),
                dtype=self.xp.int32,
            )
        else:
            self._parent_start_inds = None

        if isinstance(acs.settings, FDSettings):
            assert isinstance(self._basis_settings, FDSettings)
            if self._gb_fd_comp_user_supplied:
                if getattr(self, "_parent_acs_token", None) is not None:
                    logger.warning(
                        f"{self.name}: parent ACA changed but gb_fd_comp was "
                        "user-supplied; keeping the user's comp."
                    )
            elif (
                self.gb_fd_comp is None
                # Rebuild if the grid config drifted (band re-slice etc.);
                # cheap, no data pointers involved.
                or self.gb_fd_comp.df != float(self._basis_settings.df)
                or self.gb_fd_comp.ind_min != int(self._basis_settings.ind_min)
                or self.gb_fd_comp.ind_max != int(self._basis_settings.ind_max)
            ):
                # TODO: check this a little further
                from gbgpu.gbcomps import GBFDComputations

                if self._proposal_tdi_config is None:
                    raise ValueError(
                        "FD-basis GB moves need tdi_config= (and orbits=) to "
                        "build the GBFDComputations comp, or pass "
                        "gb_fd_comp= directly."
                    )
                orbits_in = (
                    self._proposal_orbits
                    if self._proposal_orbits is not None
                    else getattr(self.gb, "orbits", None)
                )
                self.gb_fd_comp = GBFDComputations(
                    self._basis_settings,
                    self._t_ref,
                    N_sparse=int(self.band_N_vals.max()),
                    orbits=orbits_in, tdi_config=self._proposal_tdi_config,
                    force_backend=self.force_backend,
                    d_d=0.0,
                    tdi_type=self.waveform_kwargs.get(
                        "tdi_channel_setup", "XYZ"),
                    nchannels=acs.nchannels,
                )

        # Move-level engine for parent-residual fills (cold-chain
        # open/close). Same domain dispatch as the sub-band buffer's engine;
        # the parent ACA has no per-slot ``min_freq_inds``, so the FD engine
        # falls back to ``start_freq_inds`` (one shared window start per
        # walker row).
        self._likelihood_engine = make_routed_band_engine(
            self._basis_settings,
            xp=self.xp,
            gb=self.gb,
            gb_fd_comp=self.gb_fd_comp,
            gb_wdm_comp=self.gb_wdm_comp,
            gb_stft_comp=self.gb_stft_comp,
            nchannels=acs.nchannels,
            tdi_channel_setup=self.waveform_kwargs.get("tdi_channel_setup"),
            df=self.df,
            start_freq_inds=self._parent_start_inds,
            data_length=acs.data_length,
        )
        self._parent_acs_token = token

    def setup(self, model, branches):
        return

    @classmethod
    def supported_backends(cls):
        return ["lisatools_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    def find_friends(self, name, s, s_inds=None, branch_supps=None):
        """Complement points for the band-aware group stretch.

        The friends were drawn (one per proposed source) by
        :meth:`BandSorter.draw_friends` immediately before the
        ``GroupStretchMove.get_proposal`` call in :meth:`in_model_proposal`;
        this override just reshapes them to match ``s``
        ``(1, n_sources, 1, ndim)``.
        """
        friends = self._friends_for_stretch
        assert friends.shape[0] == s.shape[1]
        return friends[None, :, None, :]

    def adjust_sources_in_residual_buffer(
        self, factor, model, band_sorter: BandSorter, *args, **kwargs
    ) -> None:
        """Add or remove sources from the main residual buffer.

        Domain-agnostic: uses ``self._likelihood_engine.fill_template`` to
        work with both FD and WDM domains. The ``factor`` controls whether
        sources are added (``-1``, reducing the residual) or removed (``+1``,
        restoring them to the residual).
        """
        assert isinstance(factor, int) and (factor == -1 or factor == +1)

        subset = band_sorter.get_subset(*args, **kwargs)

        if subset is None or subset.inds.sum().item() == 0:
            return

        params_in = subset.coords_in[subset.inds]
        walkers_in = subset.walker_inds[subset.inds].astype(self.xp.int32)
        N_vals_in = subset.N_vals[subset.inds] if subset.N_vals is not None else None

        # FD-specific bounds checks (only meaningful when basis is FD).
        # WDM and STFT bounds are checked internally by the engine / kernel.
        if isinstance(self._basis_settings, FDSettings):
            assert N_vals_in is not None
            f_bin = (params_in[:, 1] / self.df).astype(int) - int(self.start_freq_ind)
            assert not np.any(
                f_bin + (N_vals_in / 2) > model.analysis_container_arr.data_length
            ), "cold-chain source window exceeds the parent data range"
            assert not np.any(f_bin - (N_vals_in / 2) < 0), (
                "cold-chain source window falls below the parent data range"
            )

        # ``coords_in`` is already in physical units; dispatch via the engine.
        self._likelihood_engine.fill_template(
            model.analysis_container_arr,
            params_in,
            walkers_in,
            N_vals_in,
            factor=factor,
            waveform_kwargs=self.waveform_kwargs,
        )

    def remove_cold_chain_sources_from_residual(self, *args, **kwargs) -> None:
        kwargs["temp"] = 0
        kwargs["apply_inds"] = True
        if self.PARALLEL_GIBBS and ("remainder" in kwargs or "extra_bool" in kwargs):
            # Nothing goes into the parent. The cells and the close read the class as it is now.
            band_sorter = args[1]
            band_sorter.cold_class_at_open = band_sorter.get_subset(**kwargs)
            return
        self.remove_sources_from_residual(*args, **kwargs)

    def remove_sources_from_residual(self, *args, **kwargs) -> None:
        self.adjust_sources_in_residual_buffer(+1, *args, **kwargs)

    def add_cold_chain_sources_to_residual(self, *args, **kwargs) -> None:
        kwargs["temp"] = 0
        kwargs["apply_inds"] = True
        if self.PARALLEL_GIBBS and ("remainder" in kwargs or "extra_bool" in kwargs):
            # The parent still holds the class as it was at the open: those sources go back, the new ones come off below.
            model, band_sorter = args
            cold_class = band_sorter.cold_class_at_open
            band_sorter.cold_class_at_open = None
            if cold_class is not None:
                self._likelihood_engine.fill_template(
                    model.analysis_container_arr,
                    cold_class.coords_in,
                    cold_class.walker_inds.astype(self.xp.int32),
                    cold_class.N_vals,
                    factor=+1,
                    waveform_kwargs=self.waveform_kwargs,
                )
        self.add_sources_to_residual(*args, **kwargs)

    def add_sources_to_residual(self, *args, **kwargs) -> None:
        self.adjust_sources_in_residual_buffer(-1, *args, **kwargs)

    # * ================= GB-sampler verification (debug mode) =================
    # * Delegated to GBSamplerDebugger (see gbdebug.py). Backward-compatible shims
    # * forward calls and properties to self.debugger.

    @property
    def _debug_tiles(self) -> bool:
        return self.debugger.tiles_active

    @property
    def _dbg_plotted_stages(self):
        return self.debugger.plotted_stages

    @_dbg_plotted_stages.setter
    def _dbg_plotted_stages(self, value):
        self.debugger.plotted_stages = value

    @property
    def _dbg_null_logged(self):
        return self.debugger.null_logged

    @_dbg_null_logged.setter
    def _dbg_null_logged(self, value):
        self.debugger.null_logged = value

    @property
    def _dbg_seq_done(self):
        return self.debugger.seq_done

    @_dbg_seq_done.setter
    def _dbg_seq_done(self, value):
        self.debugger.seq_done = value

    @property
    def _dbg_rj_done(self):
        return self.debugger.rj_done

    @_dbg_rj_done.setter
    def _dbg_rj_done(self, value):
        self.debugger.rj_done = value

    @property
    def _dbg_plot_counter(self):
        return self.debugger.plot_counter

    @_dbg_plot_counter.setter
    def _dbg_plot_counter(self, value):
        self.debugger.plot_counter = value

    @property
    def _dbg_rj_seq(self):
        return self.debugger.rj_seq

    @_dbg_rj_seq.setter
    def _dbg_rj_seq(self, value):
        self.debugger.rj_seq = value

    def _debug_cold_chain_residual_loaded(
        self,
        model: Model,
        remainder: int,
    ) -> None:
        self.debugger.cold_chain_residual_loaded(model, remainder)

    def _debug_verify_rj_step(
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
        self.debugger.verify_rj_step(
            buffer_obj,
            params,
            alive,
            slots,
            N_vals,
            delta_ll,
            keep,
            picked,
            round_idx,
            scheduler=scheduler,
        )

    def _debug_verify_in_model(
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
        self.debugger.verify_in_model(
            buffer_obj,
            curr,
            new,
            slots,
            N_vals,
            delta_ll,
            keep,
            map_cpu,
            move_idx,
        )

    def _debug_residual_round_trip(
        self,
        buffer_obj: SubBandBuffer,
        params_add: NDArrayLike,
        data_index: NDArrayLike,
        swap_N_vals: NDArrayLike,
        d_h_arr: NDArrayLike,
        h_h_arr: NDArrayLike,
    ) -> None:
        self.debugger.residual_round_trip(
            buffer_obj,
            params_add,
            data_index,
            swap_N_vals,
            d_h_arr,
            h_h_arr,
        )

    def _debug_seq_select(
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
        return self.debugger.seq_select(
            buffer_obj,
            band_sorter,
            ids,
            temp_inds,
            walker_inds,
            band_inds,
            slots,
            curr,
        )

    def _debug_slab_snapshot(
        self,
        buffer_obj: SubBandBuffer,
        slot_idx: int,
    ) -> np.ndarray:
        return self.debugger.slab_snapshot(buffer_obj, slot_idx)

    def _debug_cell_total_template(
        self,
        buffer_obj: SubBandBuffer,
        band_sorter: BandSorter,
        seq: InModelSequenceDict,
    ) -> Optional[np.ndarray]:
        return self.debugger.cell_total_template(buffer_obj, band_sorter, seq)

    def _debug_walker_true_data(
        self,
        acs: AnalysisContainerArray,
        walker: int,
        buffer_obj: Optional[SubBandBuffer] = None,
        slot_idx: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        return self.debugger.walker_true_data(acs, walker, buffer_obj, slot_idx)

    def _debug_band_source_only_ll(
        self,
        buffer_obj: SubBandBuffer,
        arr: np.ndarray,
        slot_idx: Optional[int],
        band_idx: Optional[int],
    ) -> float:
        return self.debugger.band_source_only_ll(buffer_obj, arr, slot_idx, band_idx)

    def _debug_plot_band_sequence(
        self,
        buffer_obj: SubBandBuffer,
        seq: InModelSequenceDict,
    ) -> None:
        self.debugger.plot_band_sequence(buffer_obj, seq)

    def _debug_rj_select(
        self,
        buffer_obj: SubBandBuffer,
        picked: PickedSourcesDict,
    ) -> Optional[RJSequenceDict]:
        return self.debugger.rj_select(buffer_obj, picked)

    def _debug_plot_rj_pair(
        self,
        buffer_obj: SubBandBuffer,
        rj_seq: Optional[RJSequenceDict],
    ) -> None:
        self.debugger.plot_rj_pair(buffer_obj, rj_seq)

    def _debug_log_band_null(
        self,
        buffer_obj: SubBandBuffer,
    ) -> None:
        self.debugger.log_band_null(buffer_obj)

    def _debug_plot_band(
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
        self.debugger.plot_band(
            buffer_obj,
            params_add,
            data_index,
            swap_N_vals,
            ll_diff_kept,
            map_to_update_cpu,
            keep_mask,
            move_idx,
            stage=stage,
        )

    # * ========================================================================
    # * The main proposal loop: per-band RJ + in-model repeats.
    # * ========================================================================
    
    def run_proposal(self, model, state, band_sorter, band_temps):
        """The proposal step for the GBs is done according to the following steps:
        
        A 'cell' is a combination of a temperature (t), a walker (w), and a band (b).
        Each cell is associated with a (t, w, b) combo and contains N_cell alive sources.
        
        
        run_proposal:
            -> select band unit (parity class)
            for each band unit:
                -> add cold-chain sources of unit to residual (t=0, b in unit)
                _run_band_unit:
                    -> build bandbuffer:
                        -> Create data for each cell (t, w, b)
                        -> subtract all sources in active cells from data
                        -> now we have residuals for each (t, w, b) combination build 
                           from cold chain residual.
                           
                    -> all cells are active and are processed in descending order of N_cell
                    while a cell is still active:
                        -> pick a source in currently considered cell out of nleaves_max
                        _run_rj_step:
                            if source is alive:
                                -> propose death (against full cell residual)
                                -> compute delta_ll = - < r| h > - 0.5 < h | h >
                            if source is dead:
                                -> propose birth from pre-drawn coords
                                if source within band-freq-limits:
                                    -> with phase_maximization = True
                                    -> compute delta_ll = - < r| h > - 0.5 < h | h > 
                                else (outside freq limits):
                                    -> curr_logp = -inf (fast rejection)
                            -> compute acceptance ratio (the picks of one walker in turn):
                               accept = beta * delta_ll + delta_logp + factors + k_term >= log(rand)
                            if accept:
                                kill/birth source and update residual
                            else (rejected):
                                nothing
                        if source alive:
                            _run_inmodel_repeats:
                                -> add source back into cell residual d_c = r_c + h_0
                                -> pre-compute information cholesky
                                for step in num_repeat_steps:
                                    in_model_proposal:
                                        if rand < stretch_probability:
                                            -> get friends for stretch
                                            -> get_stretch_proposal
                                        else:
                                            -> informaton cholesky jump
                                    -> compute delta_ll of new source (current is w.r.t. starting params)
                                    -> compute acceptance ratio
                                       accept = beta * delta_ll + delta_logp + factors >= log(rand)
                                    -> refresh ll_ref is necessary (sighet method)
                                -> clean bandbuffer attributes (sighet method)
                                -> subtract source from cell residual
                        else (source dead):
                            nothing...
                        -> record that source has been run
                        -> pick new cell if all sources have been run
                -> subtract cold-chain sources of unit from residual (t=0, b in unit)
                
                
        Returns ``(ll_change_log, prop_counts, acc_counts)``; the count
        arrays have shape ``(2, ntemps, nwalkers, num_bands)`` with row 0 =
        RJ proposals and row 1 = in-model proposals.
        """
        ll_change_log = self.xp.zeros((self.ntemps, self.nwalkers, self.num_bands))
        prop_counts = self.xp.zeros((2, self.ntemps, self.nwalkers, self.num_bands), dtype=int)
        acc_counts = self.xp.zeros_like(prop_counts)

        self._jump_scale_counts = self.xp.zeros((2, len(self.jump_scales), self.ntemps), dtype=int)

        # * Alive sources per (temp, walker) for the k prior term; _run_rj_step keeps it current
        # * (it is the only step that flips ``inds`` in this pass).
        if self.is_rj_prop:
            walker_cell_all = band_sorter.temp_inds * self.nwalkers + band_sorter.walker_inds
            alive_cells = walker_cell_all[band_sorter.inds]
            # ! CuPy's bincount raises on an empty array (zero-leaf start)
            self._k_walker = (
                self.xp.bincount(alive_cells, minlength=self.ntemps * self.nwalkers)
                if alive_cells.shape[0] > 0
                else self.xp.zeros(self.ntemps * self.nwalkers, dtype=self.xp.int64)
            )

        # Reset per-step debug state
        self.debugger.start_proposal_step()

        units = self.band_units if self.num_bands > 1 else 1
        start_unit = model.random.randint(units)

        for unit_i in range(units):
            remainder = (start_unit + unit_i) % units

            _dbg_snapshot = self.debugger.snapshot_unit_start(model, ll_change_log)

            # Open this parity class in the parent residual.
            with _tspan(getattr(self, "_prop_timer", None), "unit_open_close"):
                self.remove_cold_chain_sources_from_residual(
                    model, band_sorter, units=units, remainder=remainder
                )
            self._debug_cold_chain_residual_loaded(model, remainder)

            apply_inds = not self.is_rj_prop
            
            # TODO Do we still need this for STFT?
            extra_bool = (
                (band_sorter.band_inds < self.num_bands - 1) & (band_sorter.band_inds > 0)
            ) if self.num_bands > 1 else None

            subset = band_sorter.get_subset(
                units=units,
                remainder=remainder,
                apply_inds=apply_inds,
                extra_bool=extra_bool,
            )
            # Read with getattr: run_proposal also runs on stand-in moves without these attributes.
            class_widths = getattr(self, "_stft_band_class_widths", lambda switched_on: None)(
                getattr(self, "STFT_WIDTH_CLASSES", False)
            )
            if subset is not None and class_widths is not None:
                self._run_width_classes(
                    model, band_sorter, subset, band_temps, ll_change_log, prop_counts, acc_counts,
                    class_widths, dict(units=units, remainder=remainder, apply_inds=apply_inds,
                                       extra_bool=extra_bool),
                )
            elif subset is not None:
                self._run_band_unit(
                    model, band_sorter, subset, band_temps,
                    ll_change_log, prop_counts, acc_counts,
                    **({"num_slots": self._proposal_num_slots()}
                       if getattr(self, "_size_proposal_batches", False) else {}),
                )

            # Close: re-subtract with (possibly updated) cold-chain coords.
            with _tspan(getattr(self, "_prop_timer", None), "unit_open_close"):
                self.add_cold_chain_sources_to_residual(
                    model, band_sorter, units=units, remainder=remainder
                )

            self.debugger.reconcile_unit_end(
                model, ll_change_log, _dbg_snapshot, unit_i, remainder
            )

            with _tspan(getattr(self, "_prop_timer", None), "mempool_free"):
                if self.backend.uses_cupy:
                    self.xp.cuda.runtime.deviceSynchronize()
                self.mempool.free_all_blocks()

        return ll_change_log, prop_counts, acc_counts

    def _stft_band_class_widths(self, switched_on: bool):
        """Cell width of each band's class when the caller's phase uses classes, else None (one width).

        DFFR only: a TDI-on-the-fly carrier can spike out of the store window, and a class cell
        leaves it a smaller margin than the widest window does.
        """
        store_windows = getattr(self, "stft_store_windows", None)
        if (not switched_on or store_windows is None
                or getattr(getattr(self, "gb_stft_comp", None), "response", None) != "dffr"):
            return None
        return store_windows.class_width

    def _class_num_slots(self, setting: int, cell_width: int) -> int:
        """Cells of a class buffer for a batch setting, which counts cells of the widest window.

        The buffer then takes the bytes the setting means with one width.
        """
        return max(1, int(setting) * int(self.stft_store_windows.buffer_width) // int(cell_width))

    def _batch_budget_bytes(self) -> tuple:
        """Bytes one device may give to buffers now, and the number of devices."""
        gpus = getattr(self.gb, "gpus", None) if self.backend.uses_cupy else None
        return stft_batch_budget_bytes(free_device_bytes(self.xp, gpus)), len(gpus) if gpus else 1

    def _proposal_num_slots(self, cell_width=None) -> int:
        """Cells of a proposal buffer: from the free memory, or from the setting when it is given.

        ``None`` is the one width. A buffer shares its cells evenly between the devices.
        """
        if not self._size_proposal_batches:
            if cell_width is None:
                return self.num_band_preload
            return self._class_num_slots(self.num_band_preload, cell_width)
        width = int(self.stft_store_windows.buffer_width) if cell_width is None else int(cell_width)
        budget_bytes, num_devices = self._batch_budget_bytes()
        num_slots = max(1, num_devices * budget_bytes // (self._stft_bytes_per_bin * width))
        logger.info(
            f"{self.name}: {num_slots} proposal cells of {width} bins, from a budget of "
            f"{budget_bytes / 1024**3:.2f} GiB per device."
        )
        return num_slots

    def _sized_tempering_rows(self, class_widths, num_splits: int) -> dict:
        """Rows per chunk over all splits for each tempering buffer, from the free memory.

        Keyed by cell width, or by ``None`` alone for one width, which also serves when one row of
        every class does not fit: a class buffer stays for the whole propose.
        """
        budget_bytes, _ = self._batch_budget_bytes()
        # A row is one cell and its twin per temperature.
        budget_bins = budget_bytes // (2 * int(self.ntemps) * self._stft_bytes_per_bin)
        if class_widths is not None:
            bands = np.arange(1) if self.num_bands == 1 else np.arange(1, self.num_bands - 1)
            widths, band_counts = np.unique(class_widths[bands], return_counts=True)
            if int(widths.sum()) <= budget_bins:
                rows = split_rows_by_budget(
                    band_counts * int(self.nwalkers) // num_splits, widths, budget_bins
                )
                logger.info(
                    f"{self.name}: tempering rows per chunk {dict(zip(widths.tolist(), rows.tolist()))} "
                    f"by class width, from a budget of {budget_bytes / 1024**3:.2f} GiB per device."
                )
                return {width: row * num_splits for width, row in zip(widths.tolist(), rows.tolist())}
            logger.warning(
                f"{self.name}: one tempering row of every class needs {int(widths.sum())} bins, the "
                f"budget holds {budget_bins}; tempering runs at one width."
            )
        rows = max(1, budget_bins // int(self.stft_store_windows.buffer_width))
        logger.info(
            f"{self.name}: {rows} tempering rows per chunk at one width, from a budget of "
            f"{budget_bytes / 1024**3:.2f} GiB per device."
        )
        return {None: rows * num_splits}

    def _run_width_classes(self, model, band_sorter, subset, band_temps, ll_change_log, prop_counts,
                           acc_counts, class_widths, unit_selection):
        """Drive one parity unit's cells class by class, each with its own scheduler and buffer."""
        xp = self.xp
        # ! Cut from the main sorter: a subset of a subset deep-copies the main sorter it points at.
        source_widths = xp.asarray(class_widths)[xp.clip(band_sorter.band_inds, 0, self.num_bands - 1)]
        unit_widths = np.unique(class_widths[np.unique(asnumpy(subset.band_inds))])
        extra_bool = unit_selection.pop("extra_bool")
        for cell_width in unit_widths.tolist():
            class_bool = source_widths == cell_width
            class_subset = band_sorter.get_subset(
                **unit_selection,
                extra_bool=class_bool if extra_bool is None else class_bool & extra_bool,
            )
            if class_subset is not None:
                self._run_band_unit(
                    model, band_sorter, class_subset, band_temps, ll_change_log, prop_counts,
                    acc_counts, cell_width=int(cell_width),
                    num_slots=self._proposal_num_slots(cell_width),
                )
                # The next build's first small allocation would split this class's freed block, and
                # its own block would then take new device memory.
                with _tspan(getattr(self, "_prop_timer", None), "mempool_free"):
                    if self.backend.uses_cupy:
                        self.xp.cuda.runtime.deviceSynchronize()
                    self.mempool.free_all_blocks()

    def _run_band_unit(self, model, band_sorter, subset, band_temps,
                       ll_change_log, prop_counts, acc_counts, cell_width=None, num_slots=None):
        """Drive one parity unit's cells through the sub-band buffer.

        ``cell_width`` and ``num_slots`` size the buffer of one width class; by default it takes the
        widest window and ``num_band_preload`` cells.
        """
        tm = getattr(self, "_prop_timer", None)
        scheduler = BandScheduler(
            subset.special_band_inds,
            self.num_band_preload if num_slots is None else num_slots,
            xp=self.xp,
        )
        probe = getattr(self, "_mem_probe", None)
        if probe is not None:
            probe.check_released("inmodel_buffer")
        with _tspan(tm, "buffer_build"):
            buffer_obj = subset.get_buffer(
                model.analysis_container_arr, scheduler.slot_specials.copy(),
                track_counter=self._track_counter,
                **({} if cell_width is None else {"stft_cell_width": cell_width}),
            )
        if tm is not None:
            tm.count("cells", int(scheduler.n_cells))
        self._debug_log_band_null(buffer_obj)

        # Pick eligibility lives on the MAIN sorter: only sources inside this
        # unit's subset are candidates (for in-model moves the subset already
        # applied ``inds``; for RJ it includes the freshly-drawn dead ones).
        eligible = self.xp.zeros(band_sorter.num_sources, dtype=bool)
        eligible[subset.inds_main_band_sorter] = True

        round_i = 0
        while scheduler.any_active():
            with _tspan(tm, "pick"):
                picked = self._pick_sources(band_sorter, buffer_obj, scheduler, eligible)
            if picked is None:
                break
            if tm is not None:
                # Batch size per repeat round: on GPU, small batches mean the
                # 100-repeat in-model loop is kernel-launch-overhead-bound.
                tm.count("picked_sources", int(len(picked["specials"])))

            #? Why rj_prop before in_model repeats? eryn/ensemble.py does the reverse
            if self.is_rj_prop:
                # RJ before/after trace of the chosen cell: snapshots
                # bracket the RJ step; figures save only when the cell's RJ
                # proposal was ACCEPTED (buffer changed). Chronologically
                # BEFORE the in-model sequence figures.
                rj_seq = self._debug_rj_select(buffer_obj, picked)
                with _tspan(tm, "rj_step"):
                    self._run_rj_step(
                        model, band_sorter, buffer_obj, band_temps, picked,
                        ll_change_log, prop_counts, acc_counts, round_i, scheduler,
                    )
                self._debug_plot_rj_pair(buffer_obj, rj_seq)

            with _tspan(tm, "inmodel_repeats"):
                self._run_in_model_repeats(
                    model, band_sorter, buffer_obj, band_temps, picked,
                    ll_change_log, prop_counts, acc_counts,
                )

            scheduler.record_picks(picked["specials"])
            inds_fill, new_specials = scheduler.advance()
            if len(inds_fill):
                with _tspan(tm, "buffer_build"):
                    subset.get_buffer(
                        model.analysis_container_arr, new_specials,
                        inds_fill=inds_fill, buffer_obj=buffer_obj,
                    )
                self._debug_log_band_null(buffer_obj)
            round_i += 1
            # NOTE (GPU efficiency): freeing the WHOLE CuPy pool every pick
            # round forces cudaFree/cudaMalloc churn for every allocation in
            # the next round. The mempool_free stage time quantifies it.
            with _tspan(tm, "mempool_free"):
                self.mempool.free_all_blocks()
        if tm is not None:
            tm.count("pick_rounds", round_i)
        if probe is not None:
            probe.watch("inmodel_buffer", buffer_obj, span_name="buffer_build")
        
        self._release_stft_comp_groups()

        logger.info(
            f"{self.name}: band unit complete after {round_i} pick rounds "
            f"({scheduler.n_cells} cells)."
        )

    def _release_stft_comp_groups(self) -> None:
        """Point the shared STFT comp back at the parent's groups.

        A finished buffer stays reachable through the comp otherwise, so its arrays are still
        allocated while the next buffer is built. The binding is restored rather than cleared
        because ``SubBandBuffer._build_band_ac_list`` reads it to find the parent group.
        """
        parent_acs = getattr(self, "_parent_acs", None)
        if self.gb_stft_comp is not None and parent_acs is not None:
            release_gb_comp_groups(self.gb_stft_comp, parent_acs)

    def _pick_sources(self, band_sorter, buffer_obj, scheduler, eligible):
        """One not-yet-visited source per active cell, without replacement.

        Vectorized on ``self.xp``: candidates are gathered through the
        special-index maps, randomly ranked within each cell, and the first
        per cell wins. ``band_sorter.has_run_rj`` marks consumed sources for
        the remainder of this proposal, so every source is visited exactly
        once per pass.
        """
        xp = self.xp
        cand = (
            eligible
            & (~band_sorter.has_run_rj)
            & band_sorter.get_subset_bool(
                special_band_inds=scheduler.active_slot_specials
            )
        )
        cand_ids = xp.arange(band_sorter.num_sources)[cand]
        if len(cand_ids) == 0:
            return None

        # Random rank within each cell: specials are integers spaced >= 1,
        # so adding U[0, 0.5) keeps the cell blocks intact while randomizing
        # the within-cell order (robust to non-stable argsort).
        specials = band_sorter.special_band_inds[cand_ids]
        key = specials.astype(xp.float64) + xp.random.rand(len(specials)) * 0.5
        order = xp.argsort(key)
        ids_sorted = cand_ids[order]
        _, first = xp.unique(specials[order], return_index=True)
        ids = ids_sorted[first]

        band_sorter.has_run_rj[ids] = True

        specials_picked = band_sorter.special_band_inds[ids]
        band_inds = band_sorter.band_inds[ids]
        return {
            "ids": ids,
            "specials": specials_picked,
            "slot_index": buffer_obj.get_index(specials_picked).astype(xp.int32),
            "temp_inds": band_sorter.temp_inds[ids],
            "walker_inds": band_sorter.walker_inds[ids],
            "band_inds": band_inds,
            "N_vals": (
                band_sorter.band_N_vals[band_inds].copy()
                if band_sorter.band_N_vals is not None
                else None
            ),
        }

    def _run_rj_step(self, model, band_sorter, buffer_obj, band_temps, picked,
                     ll_change_log, prop_counts, acc_counts, round_i, scheduler):
        """Birth/death proposal for each picked source (vectorized over cells).

        Births (``inds == False``; coordinates pre-drawn from the RJ proposal
        distribution in the BandSorter) score the add delta
        ``<r|h> - 0.5<h|h>``; deaths (``inds == True``) score the removal
        delta ``-<r|h> - 0.5<h|h>`` (see
        :meth:`SubBandBuffer.get_removal_ll`). Detailed-balance factors are
        the pre-computed ±logpdf of the RJ proposal distribution. On accept,
        ``inds`` flips and the cell residual is updated through
        ``fill_template`` with the appropriate sign.

        The prior on the number of sources k of a walker is uniform on
        [0, N], N = nleaves_max. The slot flip alone (one birth proposal per
        dead slot, one death per source) would weigh k by C(N, k), so a birth
        adds ln((k + 1) / (N - k)) and a death ln((N - k + 1) / k), with k the
        walker's count just before the flip. A round holds picks in several
        bands of one walker, so those are decided in turn (random order) with
        k updated after each accepted flip; the likelihood terms stay
        vectorized, since a band's delta does not depend on flips in the other
        bands of the unit.
        """
        xp = self.xp
        ids = picked["ids"]
        slots = picked["slot_index"]
        N_vals = picked["N_vals"]
        alive = band_sorter.inds[ids].copy()   # True -> death proposal
        temp_idx, walker_idx, band_idx = picked["temp_inds"], picked["walker_inds"], picked["band_inds"]

        params = band_sorter.coords[ids].copy()
        params[:] = self.periodic.wrap({"gb": params[:, None, :]}, xp=xp)["gb"][:, 0]

        logp = self.xp.asarray(self.gpu_priors["gb"].logpdf(params))
        prev_logp = self.xp.zeros_like(logp)
        curr_logp = self.xp.zeros_like(logp)
        prev_logp[alive] = logp[alive]
        curr_logp[~alive] = logp[~alive]

        # Births outside this cell's frequency window are unphysical.
        f_hz = params[:, 1] / 1e3
        out_of_band = (
            (f_hz < buffer_obj.frequency_lims[0][slots])
            | (f_hz > buffer_obj.frequency_lims[1][slots])
        )
        curr_logp[(~alive) & out_of_band] = -np.inf

        # Per-band progressive leaf cap (search mode): a birth into a band
        # already holding ``cap[b]`` alive sources is prior-forbidden --
        # a truncation of the prior on the per-band leaf count. The cap is
        # judged on the cold chain (``_update_band_leaf_caps``) but enforced
        # at EVERY temperature so tempering swaps stay within a common prior
        # support. Setting -inf here routes the birth through the existing
        # ``keep`` machinery: it never reaches the likelihood kernel and the
        # bad-accept guard force-rejects it at beta > 0.
        if self._band_leaf_cap is not None:
            num_bands = self.num_bands
            cap_xp = xp.asarray(self._band_leaf_cap)
            flat_all = (
                (band_sorter.temp_inds.astype(xp.int64) * self.nwalkers
                 + band_sorter.walker_inds) * num_bands
                + band_sorter.band_inds
            )
            # Guard empty input: numpy.bincount([]) returns zeros, but
            # CuPy's bincount computes max(x) first and raises on a
            # zero-size array (the zero-leaf search start hits this on GPU).
            _alive_cells = flat_all[band_sorter.inds]
            _nbins = self.ntemps * self.nwalkers * num_bands
            if _alive_cells.shape[0] == 0:
                cell_counts = xp.zeros(_nbins, dtype=xp.int64)
            else:
                cell_counts = xp.bincount(_alive_cells, minlength=_nbins)
            cell_flat = (
                (picked["temp_inds"].astype(xp.int64) * self.nwalkers
                 + picked["walker_inds"]) * num_bands
                + picked["band_inds"]
            )
            over_cap = (
                cell_counts[cell_flat] >= cap_xp[picked["band_inds"]]
            )
            curr_logp[(~alive) & over_cap] = -np.inf

        delta_ll = self.xp.full_like(logp, -1e300)
        d_h = self.xp.zeros_like(logp)
        h_h = self.xp.zeros_like(logp)
        keep = ~self.xp.isinf(curr_logp)

        if bool(keep.any()):
            k_ids = xp.arange(len(ids))[keep]
            birth_k = k_ids[~alive[keep]]
            death_k = k_ids[alive[keep]]

            def _eval(rows, phase_maximize):
                buffer_obj.get_ll(
                    params[rows], slots[rows], slots[rows],
                    N_vals[rows] if N_vals is not None else None,
                    phase_maximize=phase_maximize,
                )
                # use d_h and h_h to determine birth and death logls by trick with 
                # phase -> -phase
                d_h[rows] = buffer_obj.d_h_out.real
                h_h[rows] = buffer_obj.h_h_out.real
                bad_rows = rows[~buffer_obj.kept_out]
                return bad_rows

            oob_rows = xp.zeros(0, dtype=int)
            if self.phase_maximize and len(birth_k):
                # Maximise the birth phase; deaths keep the true phase.
                oob_rows = _eval(birth_k, True)
                if buffer_obj.phase_angle is not None:
                    params[birth_k, 3] = params[birth_k, 3] - buffer_obj.phase_angle
                if len(death_k):
                    oob_rows = xp.concatenate([oob_rows, _eval(death_k, False)])
            else:
                oob_rows = _eval(k_ids, False)

            delta_all = xp.where(alive, -d_h - 0.5 * h_h, d_h - 0.5 * h_h)
            delta_ll[keep] = delta_all[keep]
            delta_ll[oob_rows] = -1e300

            # SNR rejection-sampling clamp on births.
            opt_snr = xp.sqrt(xp.maximum(h_h, 0.0))
            reject = (~alive) & keep & (opt_snr < buffer_obj.opt_snr_rej_samp_limit)
            delta_ll[reject] = -1e300

            self._debug_verify_rj_step(
                buffer_obj, params, alive, slots, N_vals, delta_ll, keep,
                picked, round_i, scheduler,
            )

        beta = band_temps[picked["band_inds"], picked["temp_inds"]]
        factors = band_sorter.factors[ids]
        lnpdiff = beta * delta_ll + (curr_logp - prev_logp) + factors
        log_u = xp.log(xp.random.rand(*lnpdiff.shape))

        # Coordinates outside the prior can only be accepted at beta == 0;
        # everything else is a bug -> warn and reject.
        bad_mask = (delta_ll <= -1e299) | (curr_logp <= -1e229)

        # * Uniform prior on k: each flip carries C(N, k) / C(N, k +- 1) at the walker's current k, so the
        # * picks of one walker are decided in turn (rank = position within the walker, random order).
        n_slots = band_sorter.nleaves_max
        walker_cell = temp_idx * self.nwalkers + walker_idx
        order = xp.lexsort(xp.stack([xp.random.rand(len(ids)), walker_cell]))
        _, first, group = xp.unique(walker_cell[order], return_index=True, return_inverse=True)
        rank = xp.empty(len(ids), dtype=xp.int64)
        rank[order] = xp.arange(len(ids)) - first[group]
        accept = xp.zeros(len(ids), dtype=bool)
        for position in range(int(rank.max()) + 1):
            rows = xp.flatnonzero(rank == position)
            k_now = self._k_walker[walker_cell[rows]].astype(xp.float64)
            # a birth has N - k >= 1 and a death k >= 1; the clamps only touch the unused branch
            lnpdiff[rows] += xp.where(
                alive[rows],
                xp.log((n_slots - k_now + 1) / xp.maximum(k_now, 1)),
                xp.log((k_now + 1) / xp.maximum(n_slots - k_now, 1)),
            )
            accept[rows] = lnpdiff[rows] >= log_u[rows]
            flips = rows[accept[rows] & ~bad_mask[rows]]
            self._k_walker[walker_cell[flips]] += xp.where(alive[flips], -1, 1)

        bad_accepts = accept & bad_mask
        if bool(xp.any(bad_accepts)):
            if bool(xp.any(beta[bad_accepts] != 0.0)) and not (
                "fstat" in self.name or "refit" in self.name
            ):
                logger.warning(
                    f"{self.name}: accepted an out-of-prior RJ coordinate at beta > 0."
                )
            accept[bad_accepts] = False

        rj_seq = self.debugger.rj_seq
        if rj_seq is not None:
            rj_seq["accepted"] = bool(accept[rj_seq["idx"]])

        t_i, w_i, b_i = picked["temp_inds"], picked["walker_inds"], picked["band_inds"]
        prop_counts[0][t_i, w_i, b_i] += 1

        if getattr(self.debug_settings, "rj_trace", False):
            # Trace every cold-chain DEATH proposal (accepted or not) plus
            # every accepted cold-chain move. The death delta is
            # -<r|h> - 0.5<h|h>: for a well-fit bright source it must sit
            # near -0.5*SNR^2 and essentially never be accepted at beta=1,
            # so a cold-chain leaf loss means either h_h came back ~0 from
            # the kernel (template dropped: band-edge layer gating /
            # sub-band slab window / stale device index arrays) or the
            # accept bookkeeping raced the residual update. d_h/h_h at
            # proposal time distinguish the two -- compare GPU vs CPU runs
            # of the same seed/config.
            for _k in range(len(ids)):
                if t_i[_k] != 0:
                    continue
                _acc = bool(accept[_k])
                if not (_acc or bool(alive[_k])):
                    continue
                logger.warning(
                    "RJTRACE %s t=%d w=%d b=%d slot=%d f0=%.9e mHz N=%s "
                    "d_h=%.6e h_h=%.6e delta=%.6e beta=%.3e lnp=%.6e "
                    "factors=%.4e curr_lp=%.4e prev_lp=%.4e accept=%d",
                    "DEATH" if alive[_k] else "BIRTH",
                    int(t_i[_k]), int(w_i[_k]), int(b_i[_k]), int(slots[_k]),
                    float(params[_k, 1]),
                    str(int(N_vals[_k])) if N_vals is not None else "None",
                    float(d_h[_k]), float(h_h[_k]), float(delta_ll[_k]),
                    float(beta[_k]), float(lnpdiff[_k]), float(factors[_k]),
                    float(curr_logp[_k]), float(prev_logp[_k]), int(_acc),
                )

        if bool(accept.any()):
            acc_ids = ids[accept]
            band_sorter.inds[acc_ids] = ~band_sorter.inds[acc_ids]

            # a born source has no Fisher matrix yet and a dead one loses its own
            rows = getattr(self, "_held_fim_rows", None)
            if rows is not None:
                row_ids = asnumpy(acc_ids)
                rows["fim_valid"][row_ids] = False
                rows["fim_age"][row_ids] = 0

            # Phase-maximised births carry the rotated phi0 forward.
            band_sorter.coords[acc_ids] = self.periodic.wrap(
                {"gb": params[accept][:, None, :]}, xp=xp
            )["gb"][:, 0]

            ll_change_log[t_i[accept], w_i[accept], b_i[accept]] += delta_ll[accept]
            acc_counts[0][t_i[accept], w_i[accept], b_i[accept]] += 1

            birth_acc = accept & (~alive)
            death_acc = accept & alive
            if bool(birth_acc.any()):
                buffer_obj.add_sources_to_band_buffer(
                    band_sorter.coords[ids[birth_acc]],
                    slots[birth_acc],
                    N_vals[birth_acc] if N_vals is not None else None,
                )
            if bool(death_acc.any()):
                buffer_obj.remove_sources_from_band_buffer(
                    band_sorter.coords[ids[death_acc]],
                    slots[death_acc],
                    N_vals[death_acc] if N_vals is not None else None,
                )

    def _prior_box(self, coords, cell_f0_lims):
        """Uniform prior box (lower, upper) of every source, sampling basis, each (n_src, ndim).

        A one-column prior gives its ``minimum`` and ``maximum``. A joint (f0, fdot) prior gives f0
        its ``f0_min`` and ``f0_max`` and fdot its bounds at each source's f0 (``fdot_bounds``).
        f0 is narrowed to the cell's frequency range, which the in-model gate enforces:
        ``cell_f0_lims`` holds the physical limits per source, mapped into the sampling basis
        through the transform container. Periodic columns span their period, [0, period), the
        range the periodic container wraps into.
        """
        xp = self.xp
        coords = xp.asarray(coords)
        n_src, ndim = coords.shape
        lower = xp.full((n_src, ndim), xp.nan)
        upper = xp.full((n_src, ndim), xp.nan)
        prior_entries = getattr(self.gpu_priors["gb"], "priors", None)
        if not isinstance(prior_entries, list):
            raise TypeError(
                "the prior box needs a ProbDistContainer, whose priors are "
                "[columns, distribution] pairs"
            )
        for columns, dist in prior_entries:
            columns = [int(column) for column in np.asarray(columns).reshape(-1)]
            if len(columns) == 1 and hasattr(dist, "minimum") and hasattr(dist, "maximum"):
                minimum, maximum = float(dist.minimum), float(dist.maximum)
                if not (np.isfinite(minimum) and np.isfinite(maximum) and maximum > minimum):
                    raise ValueError(
                        f"the prior of column {columns[0]} ({type(dist).__name__}) has the range "
                        f"[{minimum}, {maximum}]; the prior box needs finite bounds")
                lower[:, columns[0]], upper[:, columns[0]] = minimum, maximum
            elif len(columns) == 2 and hasattr(dist, "fdot_bounds"):
                # * joint (f0, fdot) prior: f0 uniform, fdot uniform between f0-dependent bounds
                f0_column, fdot_column = columns
                lower[:, f0_column], upper[:, f0_column] = float(dist.f0_min), float(dist.f0_max)
                fdot_low, fdot_high = dist.fdot_bounds(coords[:, f0_column])
                lower[:, fdot_column] = xp.asarray(fdot_low)
                upper[:, fdot_column] = xp.asarray(fdot_high)
            else:
                raise TypeError(
                    f"the prior box cannot read {type(dist).__name__} over columns {columns}: "
                    "it needs a one-column prior with minimum and maximum, or a joint (f0, fdot) "
                    "prior with f0_min, f0_max and fdot_bounds(f0)")

        # * f0: the cell's physical limits mapped through the local slope of the transform
        input_basis = list(getattr(self.transform_fn, "input_basis", []))
        if "f0" not in input_basis:
            raise ValueError("the prior box needs a sampling column named 'f0' for the cell limits")
        f0_column = input_basis.index("f0")
        f0_output = int(np.asarray(self.parameter_transforms.fill_dict["test_inds"])[f0_column])
        step = JACOBIAN_STEP_RELATIVE * xp.maximum(
            xp.abs(coords[:, f0_column]), JACOBIAN_STEP_FLOOR
        )
        shifted_up, shifted_down = coords.copy(), coords.copy()
        shifted_up[:, f0_column] += step
        shifted_down[:, f0_column] -= step
        f0_here = self.transform_fn.both_transforms(coords, xp=xp)[:, f0_output]
        f0_up = self.transform_fn.both_transforms(shifted_up, xp=xp)[:, f0_output]
        f0_down = self.transform_fn.both_transforms(shifted_down, xp=xp)[:, f0_output]
        slope = (f0_up - f0_down) / (2.0 * step)
        cell_low = coords[:, f0_column] + (xp.asarray(cell_f0_lims[0]) - f0_here) / slope
        cell_high = coords[:, f0_column] + (xp.asarray(cell_f0_lims[1]) - f0_here) / slope
        lower[:, f0_column] = xp.maximum(lower[:, f0_column], xp.minimum(cell_low, cell_high))
        upper[:, f0_column] = xp.minimum(upper[:, f0_column], xp.maximum(cell_low, cell_high))

        periodic = getattr(self, "periodic", None)
        if periodic is not None and "gb" in periodic.inds_periodic:
            for column, period in zip(periodic.inds_periodic["gb"], periodic.periods["gb"]):
                lower[:, int(column)] = 0.0
                upper[:, int(column)] = float(period)
        return lower, upper

    def _sampling_jacobian(self, coords, lower, upper):
        """Diagonal Jacobian d phys_i / d x_i of the transform container, and where it is taken.

        Central differences with steps h = 1e-6 max(|x|, 1e-3). Within 2h of a prior edge the step
        could leave the domain of arccos or arcsin (cos iota and sin delta reach +-1 in doubles),
        so each column is clipped into [lower + 2h, upper - 2h] where the box is wider than 4h;
        periodic columns are not clipped. The Fisher matrix is evaluated at the same point: a
        Fisher matrix at the edge and a Jacobian 2h inside it would not belong together, since
        dh/diota vanishes at iota = 0 while diota/dcos(iota) diverges there.

        Returns ``(jacobian, point, clipped)``, clipped (n_src,) bool.
        """
        xp = self.xp
        coords = xp.asarray(coords)
        n_src, ndim = coords.shape
        test_inds = np.asarray(self.parameter_transforms.fill_dict["test_inds"])
        steps = JACOBIAN_STEP_RELATIVE * xp.maximum(xp.abs(coords), JACOBIAN_STEP_FLOOR)
        inner_low, inner_high = lower + 2.0 * steps, upper - 2.0 * steps
        clip_columns = inner_low < inner_high
        periodic = getattr(self, "periodic", None)
        if periodic is not None and "gb" in periodic.inds_periodic:
            clip_columns[:, xp.asarray(periodic.inds_periodic["gb"])] = False
        point = xp.where(clip_columns, xp.clip(coords, inner_low, inner_high), coords)
        clipped = xp.any(point != coords, axis=1)

        steps = JACOBIAN_STEP_RELATIVE * xp.maximum(xp.abs(point), JACOBIAN_STEP_FLOOR)
        jacobian = xp.zeros((n_src, ndim))
        for column in range(ndim):
            shifted_up, shifted_down = point.copy(), point.copy()
            shifted_up[:, column] += steps[:, column]
            shifted_down[:, column] -= steps[:, column]
            output = int(test_inds[column])
            phys_up = self.transform_fn.both_transforms(shifted_up, xp=xp)[:, output]
            phys_down = self.transform_fn.both_transforms(shifted_down, xp=xp)[:, output]
            jacobian[:, column] = (phys_up - phys_down) / (2.0 * steps[:, column])
        return jacobian, point, clipped

    def _information_matrix_phys(self, model, params_phys, walker_inds):
        """Fisher matrix (n_src, ndim, ndim) in the physical basis from the domain's computation.

        FD -> GBFDComputations, WDM -> GBWDMComputations, STFT -> STFTGBComputations, against the
        parent inverse-covariance rows keyed by walker. The parent container is sharded across GPUs
        on a multi-GPU run while every information_matrix is single-shard, so
        route_information_matrix partitions the sources by the shard of their walker.
        """
        xp = self.xp
        if isinstance(self._basis_settings, STFTSettings):
            info_comp = self.gb_stft_comp
        elif isinstance(self._basis_settings, FDSettings):
            info_comp = self.gb_fd_comp
        else:
            info_comp = self.gb_wdm_comp
        test_inds = np.asarray(self.parameter_transforms.fill_dict["test_inds"])
        return _RoutedBandEngine.route_information_matrix(
            info_comp, model.analysis_container_arr, params_phys,
            inds=test_inds, noise_index=xp.asarray(walker_inds).astype(xp.int32),
        )

    def _compute_fisher_sampling(self, model, band_sorter, ids, cell_f0_lims):
        """Fisher matrix in the sampling basis and the prior widths for the sources ``ids``.

        F is the Fisher matrix of the domain's computation object at the point of
        _sampling_jacobian, mapped to the sampling basis with that Jacobian. The widths are those of
        each source's prior box (_prior_box, with the cell limits ``cell_f0_lims``). Both are
        independent of the cell's temperature, so a held pair can be turned into a factor at any
        beta with fisher_proposal_factor.

        Returns ``(info_sampling, prior_widths, clipped)``: (n_src, ndim, ndim), (n_src, ndim) and
        (n_src,) bool for sources evaluated at a clipped point.
        """
        xp = self.xp
        coords = band_sorter.coords[ids]
        n_src = coords.shape[0]
        timer = getattr(self, "_prop_timer", None)
        if timer is not None:
            timer.count("fisher_sources", int(n_src))

        with _tspan(timer, "fisher_jacobian"):
            lower, upper = self._prior_box(coords, cell_f0_lims)
            jacobian, point, clipped = self._sampling_jacobian(coords, lower, upper)
        with _tspan(timer, "fisher_matrix"):
            info_phys = self._information_matrix_phys(
                model,
                self.transform_fn.both_transforms(point, xp=xp),
                band_sorter.walker_inds[ids],
            )
        with _tspan(timer, "fisher_mempool"):
            self.mempool.free_all_blocks()
        info_sampling = info_phys * jacobian[:, :, None] * jacobian[:, None, :]
        return info_sampling, upper - lower, clipped

    def _ensure_fim_store(self, branch):
        """The held Fisher matrices of the gb ``branch``, created empty when missing.

        Four host arrays over the leaf slots (ntemps, nwalkers, nleaves_max) in the branch
        supplemental: the matrix in the sampling basis (..., ndim, ndim), the prior widths
        (..., ndim), the blocks served and a flag for a held matrix. The HDF5 backends do not save
        branch supplementals, so after a restart the store starts empty and every source gets a
        matrix at its next block. A supplemental with other keys gets the four keys added.
        """
        ntemps, nwalkers, nleaves_max, ndim = branch.shape
        base_shape = (ntemps, nwalkers, nleaves_max)
        supplemental = branch.branch_supplemental
        if supplemental is not None and all(key in supplemental for key in _FIM_STORE_KEYS):
            return supplemental
        empty = {
            "fim_matrix": np.zeros(base_shape + (ndim, ndim)),
            "fim_widths": np.zeros(base_shape + (ndim,)),
            "fim_age": np.zeros(base_shape, dtype=np.int64),
            "fim_valid": np.zeros(base_shape, dtype=bool),
        }
        if supplemental is None:
            branch.branch_supplemental = BranchSupplemental(empty, base_shape=base_shape)
        else:
            supplemental.add_objects(
                {key: value for key, value in empty.items() if key not in supplemental}
            )
        return branch.branch_supplemental

    def _load_fim_rows(self, branch, band_sorter):
        """Held Fisher matrices of the sorter rows, copied from the rows' slots at the start.

        Accepted tempering swaps exchange the (temperature, walker) labels of the rows, so the
        per-row copy follows them; _write_back_state puts the rows back at their new slots.
        """
        store = self._ensure_fim_store(branch)
        start_slots = (
            band_sorter.orig_temp_inds, band_sorter.orig_walker_inds, band_sorter.orig_leaf_inds
        )
        slots = tuple(asnumpy(index) for index in start_slots)
        rows = {key: store.holder[key][slots] for key in _FIM_STORE_KEYS}
        # a dead slot holds no matrix, so a birth always starts without one
        rows["fim_valid"] &= asnumpy(band_sorter.inds)
        return rows

    def _compute_proposal_cholesky(self, model, band_sorter, ids, beta, cell_f0_lims):
        """Factor of the proposal covariance (beta F + P)^-1 for the sources ``ids`` (sorter rows).

        Each source holds its Fisher matrix in the sampling basis and its prior widths for
        ``num_stationary_fim_steps`` blocks: a source without a held pair, or whose pair has served
        that many blocks, gets a new pair from _compute_fisher_sampling at its current position.
        The factor is rebuilt every block from the held pair with the cell's current ``beta``, so a
        source that changed temperature keeps its matrix and takes the new beta. Between refreshes
        the factor does not depend on the current position, so the proposal is symmetric.
        """
        xp = self.xp
        rows = self._held_fim_rows
        if rows is None:
            raise RuntimeError("the held Fisher matrices are loaded by propose (_load_fim_rows)")
        row_ids = asnumpy(ids)
        due = ~rows["fim_valid"][row_ids] | (
            rows["fim_age"][row_ids] >= self.num_stationary_fim_steps
        )
        clipped_count = 0
        if due.any():
            due_device = xp.asarray(due)
            info_sampling, prior_widths, clipped = self._compute_fisher_sampling(
                model, band_sorter, ids[due_device],
                (cell_f0_lims[0][due_device], cell_f0_lims[1][due_device]),
            )
            due_rows = row_ids[due]
            rows["fim_matrix"][due_rows] = asnumpy(info_sampling)
            rows["fim_widths"][due_rows] = asnumpy(prior_widths)
            rows["fim_age"][due_rows] = 0
            rows["fim_valid"][due_rows] = True
            clipped_count = int(clipped.sum())
        timer = getattr(self, "_prop_timer", None)
        with _tspan(timer, "fisher_eigh"):
            chol, diagnostics = fisher_proposal_factor(
                xp.asarray(rows["fim_matrix"][row_ids]), xp.asarray(rows["fim_widths"][row_ids]),
                beta, xp=xp,
            )
        rows["fim_age"][row_ids] += 1

        if logger.isEnabledFor(logging.DEBUG):
            n_src = len(row_ids)
            condition = diagnostics["condition"]
            largest = (
                float(xp.max(xp.where(xp.isfinite(condition), condition, 0.0)))
                if n_src else float("nan")
            )
            logger.debug(
                "%s: Fisher factor of %d sources (%d new matrices, %d at a clipped point): %d with "
                "a prior-dominated direction, largest condition number of C %.3e, %d given the "
                "prior factor.",
                self.name, n_src, int(due.sum()), clipped_count,
                int(diagnostics["prior_dominated"].sum()), largest,
                int(diagnostics["fallback"].sum()),
            )
        return chol

    def in_model_proposal(self, coords, chol, band_sorter, source_ids, model):
        """Default in-model proposal: group-stretch / info-matrix mix.

        Overridable hook: subclasses provide other proposal components with
        the same ``(new_coords, factors)`` contract (``factors`` are the
        detailed-balance log-factors, zero for symmetric proposals).

        Group stretch is only allowed once the move has completed at least
        one full pass (``self.time >= 1``) and the cold-chain friend table
        exists; it is then drawn with probability ``stretch_probability``
        per repeat round (the info-matrix Cholesky jump otherwise).
        """
        xp = self.xp
        use_stretch = (
            self.stretch_probability > 0.0
            and self.time >= 1
            and getattr(band_sorter, "friend_start_inds", None) is not None
            and bool(np.random.rand() < self.stretch_probability)
        )

        if use_stretch:
            # Friends drawn per source; Eryn's GroupStretchMove supplies the
            # stretch math + (ndim-1)*log(zz) factors through find_friends.
            self._friends_for_stretch = band_sorter.draw_friends(source_ids)
            q, factors = self.get_proposal(
                {"gb": coords[None, :, None, :]},
                model.random,
                s_inds_all={"gb": xp.ones((1, coords.shape[0], 1), dtype=bool)},
            )
            new_coords = q["gb"][0, :, 0, :]
            factors = factors.reshape(-1)
        else:
            scale_inds = xp.random.choice(
                len(self.jump_scales), size=coords.shape[0], p=self.jump_scales_probs
            )
            self._last_jump_scale_inds = scale_inds
            _rand = xp.random.randn(*coords.shape)
            dx = xp.einsum("...ij,...j->...i", chol, _rand)
            new_coords = coords + self.jump_scales[scale_inds][:, None] * dx
            factors = xp.zeros(coords.shape[0])   # symmetric draw

        return new_coords, factors

    def _sighet_drift_metrics(self, curr, ref_track):
        """Per-source drift vs the heterodyne expansion point, PHYSICAL basis.

        Returns ``(drift, damp)``: accumulated carrier-phase drift
        ``2*pi*|df0|*Tobs + pi*|dfdot|*Tobs**2`` (rad) and the amplitude
        ratio ``|ln(A/A_ref)|``. Computed by transforming BOTH coordinate
        sets to physical parameters, so it is branch-agnostic: reduced
        sampling bases get exactly-zero contributions from the fixed
        parameters. Reading sampling columns directly (the old form) would
        silently misread a reduced basis whose column 1 is not f0.

        Callers must be inside a ``sighet_active`` branch -- ``Tobs``
        exists only on the time-domain / WDM settings, and sig-het is only
        ever active on the WDM arm (see :meth:`_run_in_model_repeats`).
        """
        xp = self.xp
        pc = self.transform_fn.both_transforms(curr, xp=xp)
        pr = self.transform_fn.both_transforms(ref_track, xp=xp)
        Tobs = float(self._basis_settings.Tobs)
        drift = (2.0 * np.pi * xp.abs(pc[:, 1] - pr[:, 1]) * Tobs
                 + np.pi * xp.abs(pc[:, 2] - pr[:, 2]) * Tobs**2)
        damp = xp.abs(xp.log(xp.abs(pc[:, 0]) / xp.abs(pr[:, 0])))
        return drift, damp

    def _sighet_trust_dlna_vec(self, buffer_obj, num_sources):
        """Per-source amplitude gate ``clip(C/snr_ref, dlna_min, dlna_cap)``.

        Reads the reference template power ``h_h_out`` stashed on the
        buffer by the most recent ``get_ll``/``get_add_ll`` call (for example the
        block's ``ll_ref`` evaluation, or a refresh re-basing call for
        the refreshed subset). ``C = 0`` returns the uniform cap."""
        xp = self.xp
        trust_snr_c = self.sighet_settings.trust_snr_c
        trust_dlna = self.sighet_settings.trust_dlna
        trust_dlna_min = self.sighet_settings.trust_dlna_min

        if trust_snr_c <= 0.0:
            return xp.full(num_sources, trust_dlna)
        power_template = xp.asarray(buffer_obj.h_h_out).real
        snr_ref = xp.sqrt(xp.clip(power_template, 0.0, None))
        return xp.clip(
            trust_snr_c / xp.maximum(snr_ref, 1e-30),
            trust_dlna_min,
            trust_dlna,
        )

    def _sighet_anchor_phys(self, ref_track):
        """Anchor-side physical quantities for the trust-region gate.

        Returns ``(|A|, f0, fdot)`` of the heterodyne expansion points so
        the per-repeat gate only transforms the CANDIDATES (the anchor side
        is fixed for the block, modulo mid-block refresh)."""
        xp = self.xp
        pr = self.transform_fn.both_transforms(ref_track, xp=xp)
        return xp.abs(pr[:, 0]), pr[:, 1].copy(), pr[:, 2].copy()

    def _run_in_model_repeats(self, model, band_sorter, buffer_obj, band_temps,
                              picked, ll_change_log, prop_counts, acc_counts):
        """``num_repeat_proposals`` in-model rounds on the picked live sources.

        The picked source is first taken OUT of its cell residual, so every
        repeat scores through a plain ``get_add_ll`` against the
        source-free residual (the buffer is not touched between repeats --
        only the tracked coordinates and counters move). After the repeats,
        the final coordinates are written back into the residual and into
        the BandSorter.
        """
        xp = self.xp
        # Read ``inds`` from the MAIN sorter AFTER the RJ step: _run_rj_step
        # flips band_sorter.inds on accepted births/deaths, so this mask
        # includes freshly-born sources (they get the repeat block) and drops
        # freshly-killed ones (their template is already out of the residual).
        alive = band_sorter.inds[picked["ids"]]
        if not bool(alive.any()):
            return

        ids = picked["ids"][alive]
        slots = picked["slot_index"][alive]
        N_vals = picked["N_vals"][alive] if picked["N_vals"] is not None else None
        t_i = picked["temp_inds"][alive]
        w_i = picked["walker_inds"][alive]
        b_i = picked["band_inds"][alive]
        beta = band_temps[b_i, t_i]

        curr = band_sorter.coords[ids].copy()
        curr[:] = self.periodic.wrap({"gb": curr[:, None, :]}, xp=xp)["gb"][:, 0]

        # Debug 3x3 sequence figures (channels x template/data/residual) at
        # the four buffer moments of this repeat block, for the chosen
        # (walker, band) cell only, once per sampler step.
        seq = self._debug_seq_select(
            buffer_obj, band_sorter, ids, t_i, w_i, b_i, slots, curr)
        if seq is not None:
            seq["snaps"]["before_removal"] = self._debug_slab_snapshot(
                buffer_obj, seq["slot"])
            # Cell TOTAL DATA (constant across the block): residual +
            # sum of ALL modeled templates of the cell. The figures show
            # total template = data_const - residual, so removing the one
            # picked source appears as +1 signal in the residual and a
            # small dent in the total template.
            _t_tot = self._debug_cell_total_template(
                buffer_obj, band_sorter, seq)
            seq["t_tot"] = _t_tot
            seq["data_const"] = (
                None if _t_tot is None
                else seq["snaps"]["before_removal"] + _t_tot
            )
            # TRUE data guard: prefer the injection-data slab (minus
            # non-GB models) over the residual+templates reconstruction --
            # the two coincide unless GB sources are modeled OUTSIDE the
            # traced band (the reconstruction cannot undo those
            # subtractions; the snapshot slice can).
            _true = self._debug_walker_true_data(
                model.analysis_container_arr, seq["walker"], buffer_obj, seq["slot"])
            if _true is not None:
                seq["data_const"] = _true

        # Take the source out of the cell residual for the whole repeat block.
        with _tspan(getattr(self, "_prop_timer", None), "inmodel_resid"):
            buffer_obj.remove_sources_from_band_buffer(curr, slots, N_vals)

        if seq is not None:
            seq["snaps"]["after_removal"] = self._debug_slab_snapshot(
                buffer_obj, seq["slot"])

        with _tspan(getattr(self, "_prop_timer", None), "inmodel_fisher"):
            # * the cell's beta and f0 range enter the proposal factor
            chol = self._compute_proposal_cholesky(
                model, band_sorter, ids, beta,
                (buffer_obj.frequency_lims[0][slots], buffer_obj.frequency_lims[1][slots]),
            )
        # Per-source likelihood setup for the repeat block (same stage as
        # the proposal cholesky / friend table). Chunked-het / FD engines
        # no-op; a sig-het computation builds its heterodyne reference
        # against the source-free residual HERE and holds it CONSTANT for
        # the whole repeat block, so ll_ref below and every repeat's
        # get_add_ll score through the same likelihood.
        sighet_active = bool(
            buffer_obj.setup_in_model_likelihood(curr, slots, N_vals)
        )
        # Drift-refresh anchor: the sampling-basis coords each source's
        # sig-het reference was built at (see the refresh block below).
        ref_track = curr.copy() if sighet_active else None

        # Trust-region setup (runs only when sighet is active). Bounds the
        # candidate displacement |dlnA| and carrier-phase drift against the
        # block anchor where the reference was linearized.
        anchor_phys = None
        trust_dlna = None
        if sighet_active:
            anchor_phys = self._sighet_anchor_phys(curr)
            trust_dlna = self._sighet_trust_dlna_vec(buffer_obj, len(ids))
            trust_Tobs = (
                getattr(self._basis_settings, "Tobs", None)
                or getattr(self._basis_settings, "T", None)
                or getattr(self.gb_wdm_comp, "T_full", 0.0)
            )

        # Baseline ll: evaluate the proposed source(s) against the residual
        # from which they were just removed.
        ll_ref = buffer_obj.get_add_ll(curr, slots, slots, N_vals)
        curr_prior = self.xp.asarray(self.gpu_priors["gb"].logpdf(curr))

        # Exact-ll anchor cross-check: when GB_SIGHET_ANCHOR_CHECK=1,
        # evaluate the reference point under the exact likelihood to measure
        # the expansion error right at the linearization anchor (where it
        # should be near 0).
        if sighet_active and self.sighet_settings.anchor_check:
            _ll_ex0 = buffer_obj.get_add_ll(curr, slots, slots, N_vals)
            buffer_obj.setup_in_model_likelihood(curr, slots, N_vals)
            _e0 = self.xp.abs(ll_ref - _ll_ex0)
            _i0 = int(self.xp.argmax(_e0))
            _f0_0 = float(asnumpy(
                self.transform_fn.both_transforms(
                    curr[_i0:_i0 + 1], xp=self.xp)[0, 1]))
            logger.info(
                f"{self.name}: sig-het ANCHOR check ({len(ids)} sources): "
                f"|dll@anchor| max={float(_e0.max()):.3e} "
                f"median={float(self.xp.median(_e0)):.3e}; worst: "
                f"temp={int(t_i[_i0])} walker={int(w_i[_i0])} "
                f"band={int(b_i[_i0])} f0={_f0_0:.6e} Hz "
                f"ll_het={float(ll_ref[_i0]):.3e} "
                f"ll_exact={float(_ll_ex0[_i0]):.3e}"
            )

        if isinstance(self._basis_settings, FDSettings) and N_vals is not None:
            n4 = (N_vals / 4).astype(int)
            lo_bin = (buffer_obj.frequency_lims[0][slots] / self.df).astype(int)
            hi_bin = (buffer_obj.frequency_lims[1][slots] / self.df).astype(int)

        for move_i in range(self.num_repeat_proposals):
            # * a Fisher jump records its scale indices here; a stretch jump leaves None
            self._last_jump_scale_inds = None
            new, factors = self.in_model_proposal(curr, chol, band_sorter, ids, model)
            new[:] = self.periodic.wrap({"gb": new[:, None, :]}, xp=xp)["gb"][:, 0]

            new_logp = self.xp.asarray(self.gpu_priors["gb"].logpdf(new))
            if isinstance(self._basis_settings, FDSettings) and N_vals is not None:
                # In-model steps stay within +- N/4 bins of the current source
                # and inside the band window (widened by N/4).
                new_bin = self.xp.abs(new[:, 1] / 1e3 / self.df).astype(int)
                new_logp[
                    (self.xp.abs(new[:, 1] / 1e3 - curr[:, 1] / 1e3) / self.df).astype(int) > n4
                ] = -np.inf
                new_logp[new_bin < lo_bin - n4] = -np.inf
                new_logp[new_bin > hi_bin + n4] = -np.inf
            else:
                # Non-FD domains (STFT / WDM) do not use N_vals windowing:
                # bound candidates directly by the cell frequency limits.
                new_f_hz = new[:, 1] / 1e3
                new_logp[new_f_hz < buffer_obj.frequency_lims[0][slots]] = -np.inf
                new_logp[new_f_hz > buffer_obj.frequency_lims[1][slots]] = -np.inf

            # Sig-het TRUST REGION: reject candidates outside the
            # expansion's validity region around the block anchor
            # (physical |dlnA| / carrier-phase gates; see the ctor
            # comment for thresholds + MH-validity). Gated rows drop out
            # of ``keep`` below, so they also skip the ll kernel.
            if anchor_phys is not None:
                _pc = self.transform_fn.both_transforms(new, xp=self.xp)
                _damp_n = self.xp.abs(self.xp.log(
                    self.xp.abs(_pc[:, 0]) / anchor_phys[0]))
                _drift_n = (
                    2.0 * np.pi * self.xp.abs(_pc[:, 1] - anchor_phys[1])
                    * trust_Tobs
                    + np.pi * self.xp.abs(_pc[:, 2] - anchor_phys[2])
                    * trust_Tobs**2
                )
                new_logp[
                    (_damp_n > trust_dlna)
                    | (_drift_n > self.sighet_settings.trust_dphase)
                ] = -np.inf

            keep = ~self.xp.isinf(new_logp)
            new_ll = self.xp.full(len(ids), -1e300)
            if bool(keep.any()):
                with _tspan(getattr(self, "_prop_timer", None), "inmodel_add_ll"):
                    new_ll[keep] = buffer_obj.get_add_ll(
                        new[keep], slots[keep], slots[keep],
                        N_vals[keep] if N_vals is not None else None,
                        phase_maximize=self.phase_maximize,
                    )
                if self.phase_maximize and buffer_obj.phase_angle is not None:
                    new[keep, 3] = new[keep, 3] - buffer_obj.phase_angle
                    new[keep] = self.periodic.wrap(
                        {"gb": new[keep][:, None, :]}, xp=xp
                    )["gb"][:, 0]

            delta_ll = new_ll - ll_ref
            lnpdiff = beta * delta_ll + (new_logp - curr_prior) + factors
            accept = lnpdiff >= self.xp.log(self.xp.random.rand(*lnpdiff.shape))

            bad_mask = (new_ll <= -1e299) | (new_logp <= -1e229)
            bad_accepts = accept & bad_mask
            if bool(xp.any(bad_accepts)):
                if bool(xp.any(beta[bad_accepts] != 0.0)):
                    logger.warning(
                        f"{self.name}: accepted an out-of-prior in-model "
                        "coordinate at beta > 0."
                    )
                accept[bad_accepts] = False

            prop_counts[1][t_i, w_i, b_i] += 1
            if bool(accept.any()):
                curr[accept] = new[accept]
                ll_ref[accept] = new_ll[accept]
                curr_prior[accept] = new_logp[accept]
                ll_change_log[t_i[accept], w_i[accept], b_i[accept]] += delta_ll[accept]
                acc_counts[1][t_i[accept], w_i[accept], b_i[accept]] += 1

            # * To log Fisher jumps proposed and accepted per (scale, temperature)
            scale_inds = getattr(self, "_last_jump_scale_inds", None)
            scale_counts = getattr(self, "_jump_scale_counts", None)
            if scale_inds is not None and scale_counts is not None:
                n_temps = scale_counts.shape[2]
                cells = scale_inds * n_temps + t_i
                scale_counts[0] += xp.bincount(
                    cells, minlength=scale_counts[0].size
                ).reshape(scale_counts[0].shape)
                # weights instead of cells[accept]: older CuPy raises on an empty bincount input
                scale_counts[1] += xp.bincount(
                    cells, weights=accept.astype(xp.float64), minlength=scale_counts[1].size
                ).astype(scale_counts.dtype).reshape(scale_counts[1].shape)

            self._debug_verify_in_model(
                buffer_obj, curr, new, slots, N_vals, delta_ll, keep,
                (asnumpy(t_i), asnumpy(w_i), asnumpy(b_i)), move_i,
            )

            # Sig-het drift refresh: every ``sighet_refresh_every``
            # repeats, re-anchor the references of the sources that
            # walked too far from their expansion point. The test is
            # pure parameter arithmetic (no kernel call): accumulated
            # carrier-phase drift 2*pi*|df0|*Tobs + pi*|dfdot|*Tobs^2
            # plus an amplitude-ratio guard. Refreshed sources get their
            # reference PATCHED in place (only those coefficient blocks
            # rebuild) and their ll_ref re-based against the new
            # reference so the MH deltas never mix references.
            if (
                sighet_active
                and self.sighet_settings.refresh_every > 0
                and (move_i + 1) % self.sighet_settings.refresh_every == 0
                and move_i + 1 < self.num_repeat_proposals
            ):
                drift, damp = self._sighet_drift_metrics(curr, ref_track)
                far = (drift > self.sighet_settings.refresh_dphase) | (damp > np.log(2.0))
                # Hot cells keep their stale reference: the ll error is
                # beta-suppressed and each refresh is a full setup.
                far = far & (beta >= self.sighet_settings.refresh_min_beta)
                if bool(far.any()):
                    buffer_obj.setup_in_model_likelihood(
                        curr[far], slots[far], N_vals[far] if N_vals is not None else None
                    )
                    ll_ref[far] = buffer_obj.get_add_ll(
                        curr[far], slots[far], slots[far], N_vals[far] if N_vals is not None else None
                    )
                    ref_track[far] = curr[far]
                    # Re-anchor the trust-region cache with the refreshed
                    # references (refresh is rare; full recompute is cheap).
                    # The re-basing get_add_ll above stashed h_h for the
                    # refreshed subset, so the SNR-scaled gate re-anchors
                    # from the same call.
                    if anchor_phys is not None:
                        anchor_phys = self._sighet_anchor_phys(ref_track)
                        trust_dlna[far] = self._sighet_trust_dlna_vec(
                            buffer_obj, int(far.sum()))
                    logger.debug(
                        f"{self.name}: sig-het reference refresh for "
                        f"{int(far.sum())}/{len(ids)} sources at repeat "
                        f"{move_i + 1}."
                    )

        # End-of-block sig-het DRIFT audit (debug knob): report exactly
        # how far each source walked from its heterodyne expansion point
        # over the block -- same parameter-space metric the refresh gates
        # on -- WITHOUT changing the sampling. Pure arithmetic, no kernel.
        _audit = sighet_active and self.sighet_settings.drift_check
        if _audit:
            drift, damp = self._sighet_drift_metrics(curr, ref_track)
            n_over = int((drift > self.sighet_settings.refresh_dphase).sum())
            _gate = (
                f" gate=[{float(trust_dlna.min()):.2f}"
                f"..{float(trust_dlna.max()):.2f}]"
                if trust_dlna is not None else ""
            )
            logger.info(
                f"{self.name}: sig-het end-of-block drift ({len(ids)} sources, "
                f"{self.num_repeat_proposals} repeats): phase max="
                f"{float(drift.max()):.3e} median="
                f"{float(self.xp.median(drift)):.3e} rad, "
                f"{n_over} over dphase={self.sighet_settings.refresh_dphase}; "
                f"|dlnA| max={float(damp.max()):.3e}.{_gate}"
            )
            # The sig-het delta the CHAIN actually used at the final coords
            # (tracked ll_ref), captured before the engine reverts.
            _ll_het_final = ll_ref.copy()

        # Repeat block over: deactivate the per-source likelihood setup so
        # everything outside the block (RJ, removal, fills) scores through
        # the standard engine path again.
        buffer_obj.clear_in_model_likelihood()

        # End-of-block LIKELIHOOD accuracy AUDIT (same knob as the drift
        # audit): re-score the block's FINAL coordinates through the EXACT
        # engine -- after clear_in_model the buffer routes to the chunked
        # delegate natively -- and compare against the sig-het value the MH
        # chain used. This is the accuracy tracker for the fixed-reference
        # policy at the chain's actual operating points: |dll| is directly
        # comparable to the het budget (dlnL ~ SNR^2 * mm), reported for all
        # temps and for the COLD chain separately (hot walkers legitimately
        # roam where the linearization is worst and beta suppresses the
        # error's effect there). Costs ONE exact batched call per block.
        if _audit:
            tm = getattr(self, "_prop_timer", None)
            with _tspan(tm, "inmodel_accuracy_check"):
                _ll_exact = buffer_obj.get_add_ll(
                    curr, slots, slots, N_vals,
                    phase_maximize=self.phase_maximize)
            _err = self.xp.abs(_ll_het_final - _ll_exact)
            _cold = beta > 0.999
            _n_c = int(_cold.sum())
            _cmax = float(_err[_cold].max()) if _n_c else float("nan")
            _cmed = float(self.xp.median(_err[_cold])) if _n_c else float("nan")
            logger.info(
                f"{self.name}: sig-het end-of-block ll AUDIT vs exact "
                f"({len(ids)} sources): |dll| max={float(_err.max()):.3e} "
                f"median={float(self.xp.median(_err)):.3e}; COLD ({_n_c}): "
                f"max={_cmax:.3e} median={_cmed:.3e}."
            )

        # Final coordinates back into the residual and the sorter.
        band_sorter.coords[ids] = curr
        if seq is not None:
            seq["snaps"]["before_addback"] = self._debug_slab_snapshot(
                buffer_obj, seq["slot"])
            # Final get_ll value for the traced source (post-repeats), for
            # the addback delta-ll cross-check in the sequence figures.
            seq["ll_ref_final"] = float(asnumpy(ll_ref)[seq["idx"]])
        buffer_obj.add_sources_to_band_buffer(curr, slots, N_vals)
        if seq is not None:
            seq["snaps"]["after_addback"] = self._debug_slab_snapshot(
                buffer_obj, seq["slot"])
            seq["f0_new"] = float(asnumpy(
                self.transform_fn.both_transforms(
                    curr[seq["idx"]:seq["idx"] + 1], xp=self.xp)[0, 1]))
            self._debug_plot_band_sequence(buffer_obj, seq)

    def _tempering_swap_grid(self, band_sorter, start):
        """Permuted (band, walker, temp) cell grid for one tempering parity.

        Interior bands only (the edge bands host no swaps), every
        temperature, and an independent random walker permutation per
        (band, temp) -- adjacent temperature columns of a grid row are the
        cells whose templates may exchange. Only the ``start``-parity
        interior bands are kept.

        Returns ``(band_index, temp_index, walkers_permuted, special_index,
        num_bands_unit)``; the first four are shaped
        ``(bands_this_parity, nwalkers, ntemps)``.
        """
        if self.num_bands == 1:
            num_bands_tempered = 1
            band_index_arr = self.xp.arange(1)
        else:
            num_bands_tempered = self.num_bands - 2
            band_index_arr = self.xp.arange(1, self.num_bands - 1)

        num_bands_unit = np.arange(num_bands_tempered)[start::2].shape[0]

        walkers_permuted = (
            self.xp.asarray(
                [
                    self.xp.random.permutation(self.xp.arange(self.nwalkers))
                    for _ in range(self.ntemps * num_bands_tempered)
                ]
            )
            .reshape(num_bands_tempered, self.ntemps, self.nwalkers)
            .transpose(0, 2, 1)[start::2]
        )
        temp_index = (
            self.xp.repeat(self.xp.arange(self.ntemps), num_bands_tempered * self.nwalkers)
            .reshape(self.ntemps, num_bands_tempered, self.nwalkers)
            .transpose(1, 2, 0)[start::2]
        )
        band_index = (
            self.xp.repeat(band_index_arr, self.ntemps * self.nwalkers)
            .reshape(num_bands_tempered, self.ntemps, self.nwalkers)
            .transpose(0, 2, 1)[start::2]
        )
        special_index = band_sorter.get_special_band_index(
            temp_index, walkers_permuted, band_index
        )
        return band_index, temp_index, walkers_permuted, special_index, num_bands_unit

    def _tempering_cross_term_grid(self, start, walker_groups):
        """Grid rows of one tempering parity pass and their uniforms, drawn with NumPy.

        The bands are those of :meth:`_tempering_swap_grid`. For every band, temperature and walker
        group, the group's walkers are permuted by sorting uniform numbers, and row ``k`` of a band
        and group holds the ``k``-th walker of each temperature's permutation; swaps therefore pair
        walkers of one group. Then one uniform per row and temperature pair. The draws depend on
        the global seed and the groups only, not on the chunks or the devices.

        Returns:
            ``(bands, walkers, log_uniforms)`` of shapes ``(rows,)``, ``(rows, ntemps)`` and
            ``(rows, ntemps - 1)``, rows ordered by band, then group, then ``k``. Column ``p`` of
            ``log_uniforms`` serves the pair ``(p + 1, p)``.
        """
        band_ids = np.arange(1) if self.num_bands == 1 else np.arange(1, self.num_bands - 1)
        band_ids = band_ids[start::2]
        rows_per_group = []
        for group in walker_groups:
            keys = np.random.random((band_ids.size, self.ntemps, group.size))
            permuted = group[np.argsort(keys, axis=-1, kind="stable")]
            rows_per_group.append(permuted.transpose(0, 2, 1))
        walkers = np.concatenate(rows_per_group, axis=1).reshape(-1, self.ntemps)
        bands = np.repeat(band_ids, walkers.shape[0] // max(band_ids.size, 1))
        log_uniforms = np.log(np.random.random((walkers.shape[0], self.ntemps - 1)))
        return bands, walkers, log_uniforms

    def _adapt_band_temps(self, band_temps, band_swaps_accepted, band_swaps_proposed):
        """Per-band temperature-ladder adaptation, in place on ``band_temps``.

        Hyperbolic-decay adjustment of the inverse-temperature ladder from
        the just-collected swap acceptance ratios (hottest and coldest
        chains pinned) -- the standard eryn/ptemcee adaptation applied
        band-by-band. No-op on the first proposal (``self.time == 0``).

        TODO: change temperature adaptation.
        """
        if self.time <= 0:
            return
        # Edge bands never receive swap proposals (interior-bands-only grid),
        # so guard the 0/0: an unproposed (band, pair) adapts with ratio 0
        # instead of propagating NaN into the ladder (at ntemps > 2 a NaN
        # here corrupts band_temps for the edge bands' middle temps, which
        # then NaN-poisons every acceptance in those bands).
        _prop_safe = self.xp.maximum(band_swaps_proposed, 1)
        ratios = (band_swaps_accepted / _prop_safe).T
        betas0 = band_temps.copy().T
        betas1 = betas0.copy()

        # Modulate temperature adjustments with a hyperbolic decay.
        decay = self.temperature_control.adaptation_lag / (
            self.time + self.temperature_control.adaptation_lag
        )
        kappa = decay / self.temperature_control.adaptation_time

        # Construct temperature adjustments.
        dSs = kappa * (ratios[:-1] - ratios[1:])

        # Compute new ladder (hottest and coldest chains don't move).
        deltaTs = self.xp.diff(1 / betas1[:-1], axis=0)

        deltaTs *= self.xp.exp(dSs)
        betas1[1:-1] = 1 / (self.xp.cumsum(deltaTs, axis=0) + 1 / betas1[0])

        dbetas = betas1 - betas0
        band_temps += self.xp.asarray(dbetas.T)

    def run_tempering(self, model, state, band_sorter, band_temps):
        """Band-wise swaps between adjacent temperatures.

        A swap exchanges the sources of one band between a walker at
        temperature t and one at t - 1, so the number of sources k of both
        walkers changes. Under the uniform prior on k a walker with k sources
        carries the weight k! against its set of sources (the write-back
        repacks the leaves, so the slot labels carry no information), so a
        swap adds the change of ln k! of both walkers, and a swap that would
        give a walker more than nleaves_max sources is rejected. The bands of
        a chunk are decided in turn, so that the totals stay current.
        """
        # Read with getattr: harnesses borrow this method onto stand-ins without the new attributes.
        if getattr(self, "TEMPERING_CROSS_TERMS", False) and isinstance(
            getattr(self, "_basis_settings", None), STFTSettings
        ):
            return self._run_tempering_cross_terms(model, state, band_sorter, band_temps)
        ll_change_log_temp = self.xp.zeros((self.ntemps, self.nwalkers, self.num_bands))

        band_swaps_accepted = self.xp.zeros((len(self.band_edges) - 1, self.ntemps - 1), dtype=int)
        band_swaps_proposed = self.xp.zeros((len(self.band_edges) - 1, self.ntemps - 1), dtype=int)

        # * Alive sources per (temp, walker, band) cell and per walker, kept current through the swaps.
        # * Sources outside the band range count for their walker but sit in no cell.
        n_slots = band_sorter.nleaves_max
        num_walker_cells = self.ntemps * self.nwalkers
        walker_cell_all = band_sorter.temp_inds * self.nwalkers + band_sorter.walker_inds
        in_band = (
            band_sorter.inds & (band_sorter.band_inds >= 0) & (band_sorter.band_inds < self.num_bands)
        )
        band_cell_all = walker_cell_all * self.num_bands + band_sorter.band_inds
        # ! CuPy's bincount raises on an empty array
        cell_alive = (
            self.xp.bincount(band_cell_all[in_band], minlength=num_walker_cells * self.num_bands)
            if bool(in_band.any())
            else self.xp.zeros(num_walker_cells * self.num_bands, dtype=self.xp.int64)
        ).reshape(self.ntemps, self.nwalkers, self.num_bands)
        walker_alive = (
            self.xp.bincount(walker_cell_all[band_sorter.inds], minlength=num_walker_cells)
            if bool(band_sorter.inds.any())
            else self.xp.zeros(num_walker_cells, dtype=self.xp.int64)
        ).reshape(self.ntemps, self.nwalkers)
        log_factorial = self.xp.asarray(gammaln(np.arange(n_slots + 1) + 1.0))   # ln k!

        units = 2
        tmp_start = np.random.randint(units)
        for tmp in range(units):
            remainder = (tmp_start + tmp) % units
            start = remainder
            # start == 0 pairs with bool_remainder 1 because tempering
            # begins at band 1 (the interior bands).
            bool_remainder = 1 if start == 0 else 0

            with _tspan(getattr(self, "_prop_timer", None), "temper_open_close"):
                self.remove_cold_chain_sources_from_residual(
                    model,
                    band_sorter,
                    extra_bool=(band_sorter.band_inds % 2 == bool_remainder),
                )

            (band_index, temp_index, walkers_permuted, special_index,
             num_bands_unit) = self._tempering_swap_grid(band_sorter, start)

            num_bands_preload_temp = self.num_bands_preload_temp
            num_bands_run = 0
            while num_bands_run < self.nwalkers * num_bands_unit:
                start_ind = num_bands_run
                end_ind = start_ind + num_bands_preload_temp

                band_inds_now = band_index.reshape(-1, self.ntemps)[start_ind:end_ind].copy()
                walker_inds_now = walkers_permuted.reshape(-1, self.ntemps)[
                    start_ind:end_ind
                ].copy()
                special_inds_now = special_index.reshape(-1, self.ntemps)[start_ind:end_ind].copy()
                special_inds_now_flat = special_inds_now.flatten()

                probe = getattr(self, "_mem_probe", None)
                if probe is not None:
                    probe.check_released("temper_buffer")
                with _tspan(getattr(self, "_prop_timer", None), "temper_buffer"):
                    buffer_obj = band_sorter.get_buffer(
                        model.analysis_container_arr,
                        special_inds_now_flat,
                        use_template_arr=True,
                        track_counter=self._track_counter,
                    )

                with _tspan(getattr(self, "_prop_timer", None), "temper_swap_score"):
                    current_lls = buffer_obj.band_likelihoods(source_only=True).reshape(-1, self.ntemps)
                current_lls_orig = current_lls.copy()
                for t in range(self.ntemps)[1:][::-1]:
                    i1 = t
                    i2 = t - 1

                    # Buffer slots interleave temperatures: column t of a
                    # grid row is slot (row * ntemps + t).
                    buffer_i1 = self.xp.arange(buffer_obj.num_bands_now)[i1 :: self.ntemps]
                    buffer_i2 = self.xp.arange(buffer_obj.num_bands_now)[i2 :: self.ntemps]

                    buffer_obj.swap_template_slots(buffer_i1, buffer_i2)

                    # TODO: C-side vectorized temperature-pair swap kernel
                    # (batch the template exchange + per-cell likelihoods).
                    # * Only columns i2 and i1 hold swapped templates, so only their cells are rescored.
                    pair_cells = np.arange(buffer_obj.num_bands_now).reshape(-1, self.ntemps)[:, i2 : i1 + 1].T
                    with _tspan(getattr(self, "_prop_timer", None), "temper_swap_score"):
                        new_lls = buffer_obj.band_likelihoods(source_only=True, cells=pair_cells).T
                    old_lls = current_lls[:, i2 : i1 + 1]

                    beta1 = band_temps[(band_inds_now[:, 0], i1)]
                    beta2 = band_temps[(band_inds_now[:, 0], i2)]

                    paccept = beta1 * (new_lls[:, 1] - old_lls[:, 1]) + beta2 * (
                        new_lls[:, 0] - old_lls[:, 0]
                    ) # ! this is changed because it think this was wrong, below is the previous paccept (comparing with paccept in paper, it should now be good)
                    # paccept = bi * (band_here_i1->swapped_like - band_here_i->current_like) + bi1 * (band_here_i->swapped_like - band_here_i1->current_like);

                    raccept = self.xp.log(self.xp.random.uniform(size=paccept.shape))

                    # * Uniform prior on k: add the change of ln k! of both walkers, reject swaps past
                    # * nleaves_max. Rows of one band pair distinct walkers; bands go in turn.
                    hot_walkers, cold_walkers = walker_inds_now[:, i1], walker_inds_now[:, i2]
                    row_bands = band_inds_now[:, i1]
                    n_hot = cell_alive[i1, hot_walkers, row_bands]
                    n_cold = cell_alive[i2, cold_walkers, row_bands]
                    sel = self.xp.zeros(paccept.shape, dtype=bool)
                    for band in self.xp.unique(row_bands).tolist():
                        rows = self.xp.flatnonzero(row_bands == band)
                        k_hot = walker_alive[i1, hot_walkers[rows]]
                        k_cold = walker_alive[i2, cold_walkers[rows]]
                        k_hot_new = k_hot - n_hot[rows] + n_cold[rows]
                        k_cold_new = k_cold - n_cold[rows] + n_hot[rows]
                        fits = (k_hot_new <= n_slots) & (k_cold_new <= n_slots)
                        log_weight_change = (
                            log_factorial[self.xp.minimum(k_hot_new, n_slots)]
                            + log_factorial[self.xp.minimum(k_cold_new, n_slots)]
                            - log_factorial[k_hot]
                            - log_factorial[k_cold]
                        )
                        sel[rows] = fits & (paccept[rows] + log_weight_change > raccept[rows])
                        taken = rows[sel[rows]]
                        walker_alive[i1, hot_walkers[taken]] = k_hot_new[sel[rows]]
                        walker_alive[i2, cold_walkers[taken]] = k_cold_new[sel[rows]]
                    # * the accepted cells now hold each other's sources
                    cell_alive[i1, hot_walkers[sel], row_bands[sel]] = n_cold[sel]
                    cell_alive[i2, cold_walkers[sel], row_bands[sel]] = n_hot[sel]

                    current_lls[sel, i2 : i1 + 1] = new_lls[sel]

                    # Reverse the swaps that were not accepted.
                    buffer_obj.swap_template_slots(buffer_i1[~sel], buffer_i2[~sel])

                    # bincount accumulation: several grid rows share a band
                    # (one per walker), and fancy-index ``+=`` collapses
                    # duplicate indices (arr[[1,1,1]] += 1 increments ONCE).
                    # The counters must add one per row, not one per band.
                    # NOTE: guard empty inputs — numpy.bincount([]) returns
                    # zeros, but CuPy's bincount computes max(x) first and
                    # raises on a zero-size array (no accepted swaps in a
                    # chunk is the common case).
                    _nb_tot = band_swaps_accepted.shape[0]
                    _acc_bands = band_inds_now[sel, 0]
                    if _acc_bands.size:
                        band_swaps_accepted[:, i2] += self.xp.bincount(
                            _acc_bands, minlength=_nb_tot
                        ).astype(band_swaps_accepted.dtype)
                    if band_inds_now.size:
                        band_swaps_proposed[:, i2] += self.xp.bincount(
                            band_inds_now[:, 0], minlength=_nb_tot
                        ).astype(band_swaps_proposed.dtype)

                    # Accepted cells trade their (temp, walker) labels in the
                    # sorter so the sources follow their templates.
                    specials_i1 = band_sorter.get_special_band_index(
                        i1, walker_inds_now[sel, i1], band_inds_now[sel, i1]
                    )
                    specials_i2 = band_sorter.get_special_band_index(
                        i2, walker_inds_now[sel, i2], band_inds_now[sel, i2]
                    )
                    band_sorter.exchange_cell_labels(
                        specials_i1, i1, walker_inds_now[sel, i1],
                        specials_i2, i2, walker_inds_now[sel, i2],
                        bands=band_inds_now[sel, i2],
                    )

                diffs = current_lls - current_lls_orig
                # ``=`` (not ``+=``): each (temp, walker, band) cell is
                # visited exactly once per tempering pass.
                ll_change_log_temp[
                    (
                        buffer_obj.unique_band_combos[:, 0],
                        buffer_obj.unique_band_combos[:, 1],
                        buffer_obj.unique_band_combos[:, 2],
                    )
                ] = diffs.flatten()
                num_bands_run += num_bands_preload_temp
                if probe is not None:
                    probe.watch("temper_buffer", buffer_obj, span_name="temper_buffer")
                
                self._release_stft_comp_groups()
                del buffer_obj

            # ll_before3 = model.analysis_container_arr.likelihood()
            with _tspan(getattr(self, "_prop_timer", None), "temper_open_close"):
                self.add_cold_chain_sources_to_residual(
                    model,
                    band_sorter,
                    extra_bool=(band_sorter.band_inds % 2 == bool_remainder),
                )
            # ll_after3 = model.analysis_container_arr.likelihood()

        self._adapt_band_temps(band_temps, band_swaps_accepted, band_swaps_proposed)

        # TODO Ask michael what this is about print("NEED TO FIX ANALYSIS CONTAINER extra factor")
        ll_change_sum_temp = ll_change_log_temp.sum(axis=-1)

        return ll_change_sum_temp, band_swaps_accepted, band_swaps_proposed

    def _run_tempering_cross_terms(self, model, state, band_sorter, band_temps):
        """The swaps of :meth:`run_tempering`, decided from the cross terms of each grid row (STFT).

        Per parity pass, every grid row (the cells of one band, one per temperature) gets the table
        ``values[t, j] = (r_t|t_j) - (t_j|t_j) / 2`` from :meth:`SubBandBuffer.cross_terms`: column
        ``t`` taking the template of column ``j`` changes its ln L by ``values[t, j] - values[t, t]``.
        No ln L depends on a decision (each cell sits in one row and the residuals stay fixed), so
        every row of the pass is scored first and :func:`sweep_tempering_rows` then makes the swap
        loop's decisions on the host; the sources move once per pass. One buffer with its template
        twin serves the whole propose and is refilled per chunk of rows. The draws come from NumPy at
        the start of each pass and swaps pair walkers of one parent split, so the results repeat with
        the seed for any chunk size. DFFR rows sum over their band's store window and TDI-on-the-fly
        rows over the whole cell (``TEMPERING_BAND_WINDOW``).
        """
        acs = model.analysis_container_arr
        xp = self.xp
        ntemps, nwalkers, num_bands = self.ntemps, self.nwalkers, self.num_bands
        timer = getattr(self, "_prop_timer", None)
        probe = getattr(self, "_mem_probe", None)
        temps_range = np.arange(ntemps)

        ll_change_log_temp = np.zeros((ntemps, nwalkers, num_bands))
        band_swaps_accepted = np.zeros((len(self.band_edges) - 1, ntemps - 1), dtype=int)
        band_swaps_proposed = np.zeros((len(self.band_edges) - 1, ntemps - 1), dtype=int)

        # * Alive sources per (temp, walker, band) cell and per walker, as in the swap loop. A pass
        # * moves sources between cells of its own bands only, so the other pass's counts stay valid.
        nleaves_max = int(band_sorter.nleaves_max)
        temp_all = asnumpy(band_sorter.temp_inds).astype(np.int64)
        walker_all = asnumpy(band_sorter.walker_inds).astype(np.int64)
        band_all = asnumpy(band_sorter.band_inds).astype(np.int64)
        alive_all = asnumpy(band_sorter.inds).astype(bool)
        walker_cell_all = temp_all * nwalkers + walker_all
        in_band = alive_all & (band_all >= 0) & (band_all < num_bands)
        cell_alive = np.bincount(
            (walker_cell_all * num_bands + band_all)[in_band],
            minlength=ntemps * nwalkers * num_bands,
        ).reshape(ntemps, nwalkers, num_bands)
        walker_alive = np.bincount(
            walker_cell_all[alive_all], minlength=ntemps * nwalkers
        ).reshape(ntemps, nwalkers)
        log_factorial = gammaln(np.arange(nleaves_max + 1) + 1.0)
        betas_all = asnumpy(band_temps)

        # Swaps pair walkers of one parent split (one group on one device). A buffer with one split
        # takes every group's rows; one with a split per group takes each group's rows on its device.
        walker_groups = [np.asarray(asnumpy(walkers), dtype=np.int64) for walkers in acs.gpu_splits]
        group_of_walker = np.empty(nwalkers, dtype=np.int64)
        for group, walkers in enumerate(walker_groups):
            group_of_walker[walkers] = group
        buffer_gpus = getattr(self.gb, "gpus", None) if self.backend.uses_cupy else None
        num_buffer_splits = len(buffer_gpus) if buffer_gpus else 1
        if num_buffer_splits > 1 and (
            num_buffer_splits != len(walker_groups) or list(buffer_gpus) != list(acs.gpus)
        ):
            raise ValueError(
                f"tempering buffer devices {list(buffer_gpus)} must equal the parent's walker "
                f"splits on {acs.gpus}."
            )
        # Verify runs measure what a band-window sum would drop.
        window_totals = None
        if (getattr(getattr(self, "debug_settings", None), "verify_buffer", False)
                and getattr(self, "stft_store_windows", None) is not None):
            window_totals = dict(rows=0, rows_outside=0, max_energy=0.0, max_swap_change=0.0)
        store_windows = getattr(self, "stft_store_windows", None)
        # Only DFFR places its pixels around the kinematic carrier, which the store windows bound.
        band_window = (
            getattr(self, "TEMPERING_BAND_WINDOW", False) and store_windows is not None
            and getattr(getattr(self, "gb_stft_comp", None), "response", None) == "dffr"
        )

        # One buffer and chunk size per width class, kept for the whole propose; None is the one width.
        # Read with getattr: run_tempering also runs on stand-in moves without this method.
        class_widths = getattr(self, "_stft_band_class_widths", lambda switched_on: None)(
            getattr(self, "TEMPERING_WIDTH_CLASSES", False)
        )
        # Sized once per propose, because the class buffers stay for all of it.
        sized_rows = None
        if getattr(self, "_size_tempering_batches", False):
            sized_rows = self._sized_tempering_rows(class_widths, num_buffer_splits)
            if None in sized_rows:
                class_widths = None
        class_buffers = {}
        class_rows_per_split = {}
        buffer_obj = None
        units = 2
        tmp_start = np.random.randint(units)
        for tmp in range(units):
            start = (tmp_start + tmp) % units
            # start == 0 pairs with bool_remainder 1 because tempering begins at band 1.
            bool_remainder = 1 if start == 0 else 0
            in_pass_bands = band_sorter.band_inds % 2 == bool_remainder

            with _tspan(timer, "temper_open_close"):
                self.remove_cold_chain_sources_from_residual(
                    model, band_sorter, extra_bool=in_pass_bands
                )

            row_bands, row_walkers, log_uniforms = self._tempering_cross_term_grid(
                start, walker_groups
            )
            num_rows = row_bands.size
            row_specials = band_sorter.get_special_band_index(
                temps_range[None, :], row_walkers, row_bands[:, None]
            )
            if num_buffer_splits == 1:
                queues = [np.arange(num_rows)]
            else:
                row_group = group_of_walker[row_walkers[:, 0]]
                queues = [np.flatnonzero(row_group == split) for split in range(num_buffer_splits)]
            values = np.empty((num_rows, ntemps, ntemps))
            # * The rows of a pass are all scored before the sweep, so their order does not matter.
            cell_widths = [None] if class_widths is None else np.unique(class_widths[row_bands]).tolist()
            for cell_width in cell_widths:
                class_queues = queues if cell_width is None else [
                    queue[class_widths[row_bands[queue]] == cell_width] for queue in queues
                ]
                if cell_width not in class_rows_per_split and num_rows > 0:
                    # * The first chunk must fill every split, so the contiguous blocks hold whole rows.
                    rows_setting = (
                        sized_rows[cell_width] if sized_rows is not None
                        else self.num_bands_preload_temp if cell_width is None
                        else self._class_num_slots(self.num_bands_preload_temp, cell_width)
                    )
                    class_rows_per_split[cell_width] = max(1, min(
                        rows_setting // num_buffer_splits,
                        min(queue.size for queue in class_queues),
                    ))
                rows_per_split = class_rows_per_split.get(cell_width)
                buffer_obj = class_buffers.get(cell_width)

                positions = [0] * len(class_queues)
                while any(position < queue.size for position, queue in zip(positions, class_queues)):
                    chunk_rows, chunk_row_slots = [], []
                    for split, queue in enumerate(class_queues):
                        taken = queue[positions[split]:positions[split] + rows_per_split]
                        positions[split] += taken.size
                        chunk_rows.append(taken)
                        chunk_row_slots.append(split * rows_per_split + np.arange(taken.size))
                    chunk_rows = np.concatenate(chunk_rows)
                    # Row k of split s owns the ntemps slots from (s * rows_per_split + k) * ntemps.
                    chunk_slots = (
                        np.concatenate(chunk_row_slots)[:, None] * ntemps + temps_range[None, :]
                    )
                    chunk_specials = xp.asarray(row_specials[chunk_rows].reshape(-1))

                    with _tspan(timer, "temper_buffer"):
                        if buffer_obj is None:
                            if probe is not None:
                                probe.check_released("temper_buffer")
                            if cell_width is not None:
                                # ! A new buffer reads its parent group from the comp, which still
                                # ! points at the group of the class scored last.
                                self._release_stft_comp_groups()
                            # The first chunk fills every slot, in slot order.
                            buffer_obj = band_sorter.get_buffer(
                                acs, chunk_specials, use_template_arr=True,
                                track_counter=self._track_counter, contiguous_splits=True,
                                **({} if cell_width is None else {"stft_cell_width": cell_width}),
                            )
                            class_buffers[cell_width] = buffer_obj
                        else:
                            band_sorter.get_buffer(
                                acs, chunk_specials, inds_fill=xp.asarray(chunk_slots.reshape(-1)),
                                buffer_obj=buffer_obj,
                            )

                    starts = buffer_obj._stft_start_inds_host[chunk_slots[:, 0]].astype(np.int64)
                    if band_window:
                        window_lo = store_windows.j_lo[row_bands[chunk_rows]]
                        window_hi = store_windows.j_hi[row_bands[chunk_rows]] + 1
                    else:
                        window_lo = starts
                        window_hi = starts + int(buffer_obj._stft_cell_width)
                    with _tspan(timer, "temper_swap_score"):
                        cross, template, _ = buffer_obj.cross_terms(chunk_slots, window_lo, window_hi)
                    values[chunk_rows] = cross - 0.5 * template
                    if window_totals is not None:
                        with _tspan(timer, "temper_verify_window"):
                            self._check_tempering_band_window(
                                buffer_obj, chunk_slots, starts, row_bands[chunk_rows], store_windows,
                                window_totals,
                            )

            with _tspan(timer, "temper_sweep"):
                source_counts = cell_alive[temps_range[None, :], row_walkers, row_bands[:, None]]
                template_column, accepted = sweep_tempering_rows(
                    values, betas_all[row_bands], log_uniforms, row_bands, row_walkers,
                    source_counts, walker_alive, nleaves_max, log_factorial,
                )
                # Every row proposes each temperature pair once.
                np.add.at(band_swaps_proposed, row_bands, 1)
                np.add.at(band_swaps_accepted, row_bands, accepted.astype(int))
                row_ids = np.arange(num_rows)[:, None]
                ll_change_log_temp[temps_range[None, :], row_walkers, row_bands[:, None]] = (
                    values[row_ids, temps_range[None, :], template_column]
                    - values[row_ids, temps_range[None, :], temps_range[None, :]]
                )

            with _tspan(timer, "temper_relabel"):
                # * The sources of the cell whose template ends at column t take column t's label.
                moved_rows, moved_columns = np.nonzero(template_column != temps_range[None, :])
                if moved_rows.size:
                    band_sorter.move_cell_labels(
                        row_specials[moved_rows, template_column[moved_rows, moved_columns]],
                        moved_columns,
                        row_walkers[moved_rows, moved_columns],
                        row_specials[moved_rows, moved_columns],
                    )

            with _tspan(timer, "temper_open_close"):
                self.add_cold_chain_sources_to_residual(
                    model, band_sorter, extra_bool=in_pass_bands
                )

        if class_buffers:
            if probe is not None:
                # The tag holds one buffer; with width classes that is the class built last.
                probe.watch("temper_buffer", buffer_obj, span_name="temper_buffer")
            del buffer_obj
            class_buffers.clear()
            self._release_stft_comp_groups()
            if probe is not None:
                probe.check_released("temper_buffer")

        if window_totals is not None:
            logger.info(
                "[VERIFY_TEMPER_WINDOW] scores summed over %s. %d rows: %d with template signal "
                "outside the band window (largest (t|t) outside %.3e); largest change of a swap "
                "value between the whole cell and the band window %.3e",
                "the band windows" if band_window else "the whole cells",
                window_totals["rows"], window_totals["rows_outside"], window_totals["max_energy"],
                window_totals["max_swap_change"],
            )

        band_swaps_accepted = xp.asarray(band_swaps_accepted)
        band_swaps_proposed = xp.asarray(band_swaps_proposed)
        self._adapt_band_temps(band_temps, band_swaps_accepted, band_swaps_proposed)
        ll_change_sum_temp = xp.asarray(ll_change_log_temp.sum(axis=-1))
        return ll_change_sum_temp, band_swaps_accepted, band_swaps_proposed

    def _check_tempering_band_window(self, buffer_obj, chunk_slots, starts, bands, store_windows,
                                     totals):
        """What a band-window sum drops from the whole cell, for one chunk of rows (verify runs).

        Two cross-term calls over the cell bins left and right of each band's store window give
        exactly the template signal outside the window, as ``(t_j|t_j)`` under the noise of the
        template's own column, and the part of every swap value that the band window leaves out.
        ``totals`` accumulates over the propose.
        """
        band_lo = store_windows.j_lo[bands]
        band_hi = store_windows.j_hi[bands] + 1
        cell_hi = starts + int(buffer_obj._stft_cell_width)
        outside_values = 0.0
        outside_energy = 0.0
        for window_lo, window_hi in ((starts, band_lo), (band_hi, cell_hi)):
            cross, template, _ = buffer_obj.cross_terms(chunk_slots, window_lo, window_hi)
            outside_values = outside_values + cross - 0.5 * template
            outside_energy = outside_energy + np.einsum("rjj->rj", template)
        swap_change = outside_values - np.einsum("rii->ri", outside_values)[:, :, None]
        totals["rows"] += int(chunk_slots.shape[0])
        totals["rows_outside"] += int((outside_energy > 0.0).any(axis=1).sum())
        totals["max_energy"] = max(totals["max_energy"], float(outside_energy.max(initial=0.0)))
        totals["max_swap_change"] = max(
            totals["max_swap_change"], float(np.abs(swap_change).max(initial=0.0))
        )

    def _write_back_state(self, new_state, band_sorter) -> None:
        """Repack the sorter's live sources into ``new_state.branches['gb']``.

        Leaves are re-indexed densely per (temp, walker) in frequency order:
        live sources are ranked by the composite key
        ``(temp * nwalkers + walker) * 1e6 + f0`` and numbered ``0..n-1``
        within each (temp, walker) block. ``inds`` is rebuilt from scratch
        (all ``False``, then ``True`` at the repacked leaves), so RJ
        births/deaths and tempering walker reassignments all land here.

        The held Fisher matrices of the live rows (``_held_fim_rows``) move to
        the same new slots of the gb branch supplemental; every other slot ends
        without one.
        """
        alive = band_sorter.inds
        special_indices_finish = (
            band_sorter.temp_inds[alive] * self.nwalkers
            + band_sorter.walker_inds[alive]
        ) * int(1e6) + band_sorter.coords[alive, 1]
        special_inds_temp_walker = (
            band_sorter.temp_inds[alive] * self.nwalkers
            + band_sorter.walker_inds[alive]
        )
        sorted_inds = self.xp.argsort(special_indices_finish)

        uni, uni_inds, uni_inverse, uni_counts = self.xp.unique(
            special_inds_temp_walker[sorted_inds],
            return_index=True,
            return_counts=True,
            return_inverse=True,
        )

        leaf_inds_new_tmp = self.xp.arange(special_indices_finish.shape[0]) - uni_inds[uni_inverse]
        leaf_inds_new = self.xp.zeros_like(leaf_inds_new_tmp)
        leaf_inds_new[sorted_inds] = leaf_inds_new_tmp

        inds_new = (
            asnumpy(band_sorter.temp_inds[alive]),
            asnumpy(band_sorter.walker_inds[alive]),
            asnumpy(leaf_inds_new),
        )
        inds_old = (
            asnumpy(band_sorter.orig_temp_inds[alive]),
            asnumpy(band_sorter.orig_walker_inds[alive]),
            asnumpy(band_sorter.orig_leaf_inds[alive]),
        )
        new_state.branches["gb"].coords[inds_new] = asnumpy(band_sorter.coords[alive])
        new_state.branches["gb"].inds[:] = False
        # turn on all the ones that are there
        new_state.branches["gb"].inds[inds_new] = True
        # * held Fisher matrices: an empty store, then the live rows at their new slots
        rows = getattr(self, "_held_fim_rows", None)
        if rows is not None:
            store = self._ensure_fim_store(new_state.branches["gb"]).holder
            alive_rows = asnumpy(alive)
            for key in _FIM_STORE_KEYS:
                store[key][...] = 0
                store[key][inds_new] = rows[key][alive_rows]
            self._held_fim_rows = None

    def _band_residual_lls(self, acs):
        """Per-band cold-walker residual ll ``-1/2 <r|r>`` from the parent ACA.

        Returns a host ``(nwalkers, num_bands)`` array. The parent ACA holds
        one AC per COLD-chain walker, so this is exactly the per-band null
        the leaf-cap convergence test needs. Shard-aware: each per-GPU (or
        per-CPU-split) slab is reduced on its owning device via a per-bin ll
        followed by a cumulative-sum band reduction (no per-band kernel
        loop). Also stores ``self._band_dof`` -- the real-dof count per band
        used to scale the convergence tolerance.
        """
        xp = self.xp
        bs = self._basis_settings
        be = asnumpy(self.band_edges)
        num_bands = self.num_bands
        norm = 4.0 * float(bs.differential_component)
        is_wdm = isinstance(bs, WDMSettings)
        is_stft = isinstance(bs, STFTSettings)
        nchannels = int(acs.nchannels)
        # XYZ runs carry the full cross-channel inverse covariance
        # (shape_sens == (nc, nc)); AET/AE runs a per-channel diagonal.
        is_xyz = len(acs.shape_sens) == 2

        if is_wdm:
            layer_df = float(bs.layer_df)
            ind_min_f = int(bs.ind_min_f)
            Nf = int(bs.Nf_active)
            Nt = int(bs.Nt_active)
            k0 = np.clip(
                np.ceil(be[:-1] / layer_df - 1e-9).astype(int) - ind_min_f, 0, Nf
            )
            k1 = np.clip(
                np.floor(be[1:] / layer_df + 1e-9).astype(int) + 1 - ind_min_f,
                0, Nf,
            )
            # real WDM coefficients: 1 dof per (channel, layer, time) pixel
            dof_per_bin = nchannels * Nt
        elif is_stft:
            df = float(bs.df)
            ind_min = int(bs.ind_min)
            Nf = int(bs.NF_active)
            Nt = int(bs.NT)
            k0 = np.clip(
                np.ceil(be[:-1] / df - 1e-9).astype(int) - ind_min, 0, Nf
            )
            k1 = np.clip(
                np.floor(be[1:] / df + 1e-9).astype(int) + 1 - ind_min,
                0, Nf,
            )
            # complex STFT coefficients: 2 real dof per (channel, frequency, time) pixel
            dof_per_bin = 2 * nchannels * Nt
        else:
            df = float(bs.df)
            start_bin = int(acs.start_freq_ind[0])
            Nf = int(acs.data_length)
            k0 = np.clip(np.rint(be[:-1] / df).astype(int) - start_bin, 0, Nf)
            k1 = np.clip(np.rint(be[1:] / df).astype(int) - start_bin, 0, Nf)
            # complex FD bins: 2 real dof per (channel, bin)
            dof_per_bin = 2 * nchannels
        k1 = np.maximum(k1, k0)
        self._band_dof = (k1 - k0) * dof_per_bin

        def _shard_band_lls(r, ic):
            # r: (nw, nc, Nf[, Nt]); ic: (nw, nc[, nc], Nf[, Nt]) -> (nw, Nf)
            # STFT: r: (nw, nc, Nt, Nf); ic: (nw, nc[, nc], Nt, Nf) -> (nw, Nf)
            if is_xyz:
                if is_wdm:
                    per_bin = xp.einsum("wifk,wijfk,wjfk->wf", r, ic.real, r)
                elif is_stft:
                    per_bin = xp.einsum("witf,wijtf,wjtf->wf", r.conj(), ic, r).real
                else:
                    per_bin = xp.einsum(
                        "wif,wijf,wjf->wf", r.conj(), ic, r
                    ).real
            else:
                if is_wdm:
                    per_bin = xp.einsum("wifk,wifk,wifk->wf", r, ic.real, r)
                elif is_stft:
                    per_bin = xp.einsum("witf,witf,witf->wf", r.conj(), ic, r).real
                else:
                    per_bin = ((r.conj() * r).real * ic.real).sum(axis=1)
            cs = xp.zeros((per_bin.shape[0], Nf + 1))
            cs[:, 1:] = xp.cumsum(per_bin, axis=1)
            k0_xp, k1_xp = xp.asarray(k0), xp.asarray(k1)
            return -0.5 * norm * (cs[:, k1_xp] - cs[:, k0_xp])

        out = np.zeros((int(acs.acs_total_entries), num_bands))
        data_shaped, psd_shaped = acs.data_shaped, acs.psd_shaped
        for i, split in enumerate(acs.gpu_splits):
            if acs.gpus is not None:
                with xp.cuda.Device(acs.gpus[i]):
                    out[np.asarray(split)] = asnumpy(
                        _shard_band_lls(data_shaped[i], psd_shaped[i])
                    )
            else:
                out[np.asarray(split)] = asnumpy(
                    _shard_band_lls(data_shaped[i], psd_shaped[i])
                )
        return out

    def _update_band_leaf_caps(self, model, new_state, band_counts) -> None:
        """Advance the per-band progressive leaf caps (once per iteration).

        Runs at the very end of ``propose`` (after the final
        ``check_ll_inject`` rebuild, so the parent residual reflects the
        accepted state). Per band ``b``, the cap increments when ALL of:

        1. ``band_cap_iters[b] >= leaf_cap_settings.min_iters`` at the current cap;
        2. every cold walker's band residual ll sits within
           ``leaf_cap_settings.ll_nsigma * sqrt(N_b / 2)`` of the running best
           (``N_b`` = real dof in the band -- for a whitened residual
           ``-1/2<r|r>`` fluctuates with sigma ~ sqrt(N_b/2), so this is the
           "converged up to statistical change" test);
        3. (optional, ``leaf_cap_settings.require_occupancy``) at least one cold
           walker actually holds ``cap[b]`` leaves in the band -- an
           exhausted band with free headroom keeps its cap.

        Bands increment independently; nothing waits on other bands. On
        increment the iteration counter and running best reset, so the next
        level must re-converge on its own evidence.
        """
        bi = new_state.sub_states["gb"].band_info
        cap = bi["band_leaf_cap"]
        iters = bi["band_cap_iters"]
        best = bi["band_best_ll"]

        lls = self._band_residual_lls(model.analysis_container_arr)
        best[:] = np.maximum(best, lls.max(axis=0))
        iters += 1

        ll_nsigma = self.leaf_cap_settings.ll_nsigma
        min_iters = self.leaf_cap_settings.min_iters
        require_occupancy = self.leaf_cap_settings.require_occupancy

        tolerance = ll_nsigma * np.sqrt(self._band_dof / 2.0)
        converged = (iters >= min_iters) & (
            (best - lls.min(axis=0)) <= tolerance
        )
        if require_occupancy:
            cold_counts = asnumpy(band_counts[0])  # (nwalkers, num_bands)
            converged &= cold_counts.max(axis=0) >= cap
        nleaves_max = new_state.branches["gb"].shape[2]
        converged &= cap < nleaves_max

        if np.any(converged):
            inc_bands = np.where(converged)[0]
            cap[converged] += 1
            iters[converged] = 0
            best[converged] = -np.inf
            logger.info(
                f"{self.name}: leaf cap incremented for bands "
                f"{inc_bands.tolist()} -> {cap[inc_bands].tolist()}."
            )
        logger.info(
            f"{self.name}: leaf caps min/max = {int(cap.min())}/{int(cap.max())}; "
            f"bands at min-iters gate: {int((iters < min_iters).sum())}."
        )

    def propose(self, model, state):
        """Use the move to generate a proposal and compute the acceptance

        Args:
            model (:class:`eryn.model.Model`): Carrier of sampler information.
            state (:class:`GFState`): Current state of the sampler.

        Returns:
            :class:`GFState`: GFState of sampler after proposal is complete.

        """

        st_all = time.perf_counter()

        # Per-propose stage timing (GPU-efficiency diagnosis): one INFO line
        # per propose with the sorted stage breakdown. debug_settings.prop_timing_sync
        # synchronizes the device at every span boundary so device work is
        # attributed to the launching stage (see _ProposeTimer docstring).
        _tm_sync = None
        if self.backend.uses_cupy and getattr(self.debug_settings, "prop_timing_sync", False):
            _tm_sync = self.xp.cuda.runtime.deviceSynchronize
        self._mem_probe = make_probe(
            self.xp,
            self.name,
            context=dict(
                num_proposals=int(self.num_proposals),
                num_band_preload="sized" if self._size_proposal_batches else int(self.num_band_preload),
                num_bands_preload_temp=(
                    "sized" if self._size_tempering_batches else int(self.num_bands_preload_temp)
                ),
            ),
            settings=self.debug_settings,
        )
        self._prop_timer = tm = _ProposeTimer(sync_fn=_tm_sync, probe=self._mem_probe)
        self._track_counter = (
            StoreWindowTrackCounter() if self.stft_store_windows is not None else None
        )

        if self.backend.uses_cupy:
            self.xp.cuda.runtime.setDevice(model.analysis_container_arr.gpus[0])
        # Run-time source of truth is the ACA that arrives with the model:
        # refresh the domain quantities and re-bind the parent engine (FD
        # prototype comp + move-level engine) if this ACA differs from the
        # one currently bound. All fills / likelihoods below go through
        # model.analysis_container_arr.
        self._configure_domain(model.analysis_container_arr)
        self._bind_parent_acs(model.analysis_container_arr)
        self.current_state = state
        # np.random.seed(10)
        # print("start stretch")

        # Check that the dimensions are compatible.
        ntemps, nwalkers, nleaves_max, ndim = state.branches_coords["gb"].shape

        if not self.is_rj_prop and not np.any(state.branches["gb"].inds):
            return state, np.zeros((ntemps, nwalkers), dtype=bool)

        self.nwalkers = nwalkers
        self.ntemps = ntemps

        # Arm the per-band progressive leaf cap (search mode). The cap array
        # lives in ``state.band_info`` (HDF5-persisted); a fresh state
        # carries the -1 sentinel and is armed to ``leaf_cap_settings.start`` here.
        # ``self._band_leaf_cap`` is a live numpy reference: the birth gate
        # in ``_run_rj_step`` reads it and ``_update_band_leaf_caps``
        # mutates it in place.
        self._band_leaf_cap = None
        if self.leaf_cap_settings.enabled and self.is_rj_prop:
            bi = state.sub_states["gb"].band_info
            ensure_leaf_cap_fields(bi, self.num_bands)
            if np.all(bi["band_leaf_cap"] < 0):
                bi["band_leaf_cap"][:] = int(self.leaf_cap_settings.start)
                logger.info(
                    f"{self.name}: armed per-band leaf cap at "
                    f"{int(self.leaf_cap_settings.start)} for {self.num_bands} bands."
                )
            self._band_leaf_cap = bi["band_leaf_cap"]

        # Run any move-specific setup.
        self.setup(model, state.branches)
        self.num_proposals += 1

        # An RJ move without a proposal distribution (e.g. search/refit
        # variants whose setup() has not produced one yet) cannot run. Pure
        # in-model moves don't need one.
        if self.is_rj_prop and self.rj_proposal_distribution is None:
            return state, np.zeros((ntemps, nwalkers), dtype=bool)

        new_state = GFState(state, copy=True)
        assert new_state.log_like is not None

        band_temps = self.xp.asarray(state.sub_states["gb"].band_info["band_temps"].copy())

        self.mempool.free_all_blocks()

        rj_prop = None if not self.is_rj_prop else self.rj_proposal_distribution["gb"]

        # make sure all periodic parameters have been put into their range
        new_state.branches["gb"].coords[:] = self.periodic.wrap(
            {"gb": new_state.branches["gb"].coords[:].reshape(ntemps * nwalkers, nleaves_max, ndim)}
        )["gb"].reshape(ntemps, nwalkers, nleaves_max, ndim)

        # TODO Ask Michael about this print("is this okay for rj? I do not think so, check with below use of gb_inds_in")
        if self.use_prior_removal:  # TODO: make this stronger?
            keep_all_inds = False
        else:
            keep_all_inds = True

        with tm.span("sorter_build"):
            band_sorter = BandSorter(
                new_state.branches["gb"],
                self.band_edges,
                self.band_N_vals,
                force_backend=self.force_backend,
                transform_fn=self.parameter_transforms,
                max_data_store_size=self.max_data_store_size,
                gb=self.gb,
                gb_wdm_comp=self.gb_wdm_comp,
                gb_fd_comp=self.gb_fd_comp,
                gb_stft_comp=self.gb_stft_comp,
                waveform_kwargs=self.waveform_kwargs,
                rj_prop=rj_prop,
                keep_all_inds=keep_all_inds,
                stft_store_windows=self.stft_store_windows,
                verify_buffer=self.debug_settings.verify_buffer,
            )
        # * held Fisher matrices of the sorter rows, read at each row's start-of-propose slot
        self._held_fim_rows = self._load_fim_rows(new_state.branches["gb"], band_sorter)

        # Cold-chain friend table for the group-stretch half of the in-model
        # mix (rebuilt every proposal; cheap sort of the cold-chain f0s).
        if self.stretch_probability > 0.0:
            with tm.span("friend_index"):
                band_sorter.build_friend_index(self.nfriends)

        # get non-gb contribution
        with tm.span("resid_open_close"):
            self.remove_cold_chain_sources_from_residual(model, band_sorter, apply_inds=True)
            # Multi-GPU: snapshot every per-GPU shard of linear_data_arr inside its
            # owning device context so the copies live on the right device. Restored
            # in check_ll_inject() symmetrically.
            self.reset_non_gb_linear_data_arr = self._snapshot_linear_data_arr(
                model.analysis_container_arr
            )
            self.add_cold_chain_sources_to_residual(model, band_sorter, apply_inds=True)
        # NOTE: no explicit source_only here — follow the run-level container
        # default so this baseline stays in the same convention as the
        # incremental checks below and check_ll_inject().
        with tm.span("ll_checks"):
            ll_after = model.analysis_container_arr.likelihood()  #  - self.xp.sum(self.xp.log(self.xp.asarray(psd[:2])), axis=(0, 2))).get()

        # print(np.abs(new_state.log_like - ll_after).max())        
        # store_max_diff = np.abs(new_state.log_like[0] - ll_after).max()
        start_diffs = np.abs(new_state.log_like[0] - ll_after)

        check = ll_after - new_state.log_like[0] - start_diffs

        logger.debug(f"Start check: {start_diffs=}, {check=}")
        if not np.abs(check).max() < 1e-4:
            # assert np.abs(check).max() < 1.0
            new_state.log_like[0] = self.check_ll_inject(model, band_sorter)
            #? update start diffs
            start_diffs = np.abs(new_state.log_like[0] - ll_after)

        # print("CHECKING 0:", store_max_diff, self.is_rj_prop)
        # self.check_ll_inject(new_state, verbose=True)
        # assert np.all(start_diffs < 2.0)
        num_active_leaves = new_state.branches["gb"].inds[0].sum(axis=-1) # cold chain only
        logger.info(f"Number of active leaves before proposal: {num_active_leaves}")

        # ! Every budget of this propose must see the copies of the parent noise rows that the
        # ! buffers read on several devices, so they are made before the first budget reading.
        with tm.span("parent_noise_copies"):
            band_sorter.make_parent_noise_copies(model.analysis_container_arr)

        if self.EMPTY_POOL_BEFORE_PROPOSAL:
            with tm.span("mempool_free"):
                if self.backend.uses_cupy:
                    self.xp.cuda.runtime.deviceSynchronize()
                self.mempool.free_all_blocks()

        st_prop = time.perf_counter()
        with tm.span("run_proposal"):
            ll_change_log, prop_counts, acc_counts = self.run_proposal(
                model, new_state, band_sorter, band_temps
            )
        et_prop = time.perf_counter()
        
        # Diagnostic: per-temperature alive source counts after run_proposal
        _alive_per_temp_post_prop = [
            int(band_sorter.inds[band_sorter.temp_inds == _t].sum()) for _t in range(ntemps)
        ]
        logger.info(f"Alive sources per temp after run_proposal: {_alive_per_temp_post_prop}")
        logger.info(f"Runtime of {self.name} proposal is {round(et_prop - st_prop,3)} seconds.")

        ll_change_sum = ll_change_log.sum(axis=-1)
        new_state.log_like[0] += asnumpy(ll_change_sum[0])

        # ! THIS HAS CHANGED, WHY and should we be more thourough with checking (domain dependent)?
        with tm.span("ll_checks"):
            ll_after = model.analysis_container_arr.likelihood()
        check = ll_after - new_state.log_like[0] - start_diffs

        logger.debug(f"After proposal check: {start_diffs=}, {check=}")
        drift = float(np.abs(check).max())
        if drift >= 1e-4:
            # Incremental per-accept bookkeeping drifted from the true
            # residual likelihood (the narrow-band inner product reads a few
            # layers beyond the central band, whose context differs between
            # the cell buffer and the closed parent). The rebuild below is
            # exact, so the sampler stays correct; the warning tracks how
            # large the incremental drift got.
            logger.warning(
                f"{self.name}: incremental ll drift {drift:.3e} after "
                f"proposal; rebuilding log_like from the residual.{self._track_note()}"
            )
            with tm.span("ll_inject_drift"):
                new_state.log_like[0] = self.check_ll_inject(model, band_sorter)

        # TEMPERING
        self.temperature_control.swaps_accepted = np.zeros(ntemps - 1)
        self.temperature_control.swaps_proposed = np.zeros(ntemps - 1)

        band_swaps_accepted = self.xp.zeros((len(self.band_edges) - 1, self.ntemps - 1), dtype=int)
        band_swaps_proposed = self.xp.zeros((len(self.band_edges) - 1, self.ntemps - 1), dtype=int)

        if (
            self.temperature_control is not None
            and self.time % 1 == 0
            and self.ntemps > 1
            and self.is_rj_prop
            and self.run_swaps
            # and False
        ):
            st_temp = time.perf_counter()
            # with tm.span("ll_checks"):
            #     ll_before1 = model.analysis_container_arr.likelihood()

            with tm.span("run_tempering"):
                ll_change_sum_temp, band_swaps_accepted, band_swaps_proposed = self.run_tempering(
                    model, new_state, band_sorter, band_temps
                )

            new_state.log_like[0] += asnumpy(ll_change_sum_temp[0])

            with tm.span("ll_checks"):
                ll_after = model.analysis_container_arr.likelihood()
            check = ll_after - new_state.log_like[0] - start_diffs

            logger.debug(f"After tempering check: {start_diffs=}, {check=}")
            drift = float(np.abs(check).max())
            if drift >= 1e-4:
                logger.warning(
                    f"{self.name}: incremental ll drift {drift:.3e} after "
                    f"tempering; rebuilding log_like from the residual.{self._track_note()}"
                )
                with tm.span("ll_inject_drift"):
                    new_state.log_like[0] = self.check_ll_inject(model, band_sorter)

            with tm.span("mempool_free"):
                self.mempool.free_all_blocks()
            et_temp = time.perf_counter()
            logger.info(f"Runtime of {self.name} tempering is {round(et_temp - st_temp,3)} seconds.")
            # Diagnostic: per-temperature alive source counts after run_tempering
            # _alive_per_temp_post_temp = [
            #     int(band_sorter.inds[band_sorter.temp_inds == _t].sum()) for _t in range(ntemps)
            # ]
            # logger.info(f"Alive sources per temp after run_tempering: {_alive_per_temp_post_temp}")

        with tm.span("write_back"):
            self._write_back_state(new_state, band_sorter)

        et_all = time.perf_counter()
        logger.info(f"Full runtime of {self.name} is {round(et_all - st_all, 3)} seconds.")
        num_active_leaves = new_state.branches["gb"].inds[0].sum(axis=-1)
        logger.info(f"Number of active leaves in cold chain after proposal: {num_active_leaves}")

        new_inds = self.xp.asarray(new_state.branches_inds["gb"])
        # ! The copies of the parent noise rows leave the devices with this propose, even if
        # ! something still holds the sorter.
        band_sorter.parent_noise_copies.clear()
        del band_sorter
        with tm.span("mempool_free"):
            self.mempool.free_all_blocks()
        with tm.span("sorter_rebuild"):
            new_band_sorter = BandSorter(
                new_state.branches["gb"],
                self.band_edges,
                self.band_N_vals,
                force_backend=self.force_backend,
                transform_fn=self.parameter_transforms,
                max_data_store_size=self.max_data_store_size,
                gb=self.gb,
                gb_wdm_comp=self.gb_wdm_comp,
                gb_fd_comp=self.gb_fd_comp,
                gb_stft_comp=self.gb_stft_comp,
                waveform_kwargs=self.waveform_kwargs,
                stft_store_windows=self.stft_store_windows,
                verify_buffer=self.debug_settings.verify_buffer,
            )

        # in-model inds will not change
        tmp_freqs_find_bands = self.xp.asarray(new_state.branches_coords["gb"][:, :, :, 1])

        # calculate current band counts
        band_here = (
            self.xp.searchsorted(self.band_edges, tmp_freqs_find_bands.flatten() / 1e3, side="right") - 1
        ).reshape(tmp_freqs_find_bands.shape)

        group_temp_finder = [
            self.xp.repeat(self.xp.arange(ntemps), nwalkers * nleaves_max).reshape(
                ntemps, nwalkers, nleaves_max
            ),
            self.xp.tile(self.xp.arange(nwalkers), (ntemps, nleaves_max, 1)).transpose((0, 2, 1)),
            self.xp.tile(self.xp.arange(nleaves_max), ((ntemps, nwalkers, 1))),
        ]

        # TEMPERING
        self.temperature_control.swaps_accepted = np.zeros(ntemps - 1)
        self.temperature_control.swaps_proposed = np.zeros(ntemps - 1)

        with tm.span("mempool_free"):
            self.mempool.free_all_blocks()

        self.time += 1
        # self.xp.cuda.runtime.deviceSynchronize()

        with tm.span("band_info"):
            band_info = new_band_sorter.get_band_info()

        # prop/acc counts: row 0 = RJ, row 1 = in-model; band_info wants
        # (num_bands, ntemps) summed over walkers. The two families are
        # recorded separately (one propose produces both kinds).
        sub = new_state.sub_states["gb"]
        sub.band_info["band_temps"][:] = asnumpy(band_temps)
        sub.band_info["band_num_binaries"][:] = band_info["band_counts"]
        sub.accumulate_proposals(
            asnumpy(prop_counts[0].sum(axis=1).T),
            asnumpy(acc_counts[0].sum(axis=1).T),
            is_rj=True,
        )
        sub.accumulate_proposals(
            asnumpy(prop_counts[1].sum(axis=1).T),
            asnumpy(acc_counts[1].sum(axis=1).T),
            is_rj=False,
        )
        sub.accumulate_swaps(
            asnumpy(band_swaps_proposed), asnumpy(band_swaps_accepted)
        )
        # TODO: check rj numbers

        # new_state.log_like[:] = self.check_ll_inject(new_state)

        with tm.span("mempool_free"):
            self.mempool.free_all_blocks()
        with tm.span("ll_inject_final"):
            new_state.log_like[:] = self.check_ll_inject(model, new_band_sorter)

        # Per-band progressive leaf caps advance AFTER the final residual
        # rebuild so the convergence metric sees the accepted state. Only
        # the designated updater move (one RJ move per iteration) advances
        # the counters; every cap-enabled RJ move enforces the gate.
        if self._band_leaf_cap is not None and self.leaf_cap_settings.update:
            self._update_band_leaf_caps(model, new_state, band_info["band_counts"])

        # if self.is_rj_prop:
        #     pass  # print(self.name, "2nd count check:", new_state.branches["gb"].inds.sum(axis=-1).mean(axis=-1), "\nll:", new_state.log_like[0] - orig_store, new_state.log_like[0])

        # new_state.log_prior[:] = model.compute_log_prior_fn(new_state.branches_coords, inds=new_state.branches_inds, supps=new_state.supplemental)
        accepted = np.zeros((ntemps, nwalkers), dtype=bool)

        num_active_sources = new_state.branches["gb"].inds.sum(axis=-1)[0]
        logger.info(f"Current number of active sources in cold chain is {num_active_sources}")

        # ACCEPTANCE RATES for this propose, split RJ vs in-model, cold chain
        # (temp 0) and all-temperature. Nothing logged them before, which made
        # every proposal-machinery A/B unjudgeable -- whether the pure in_model
        # move still earns its place, a changed jump scale, the trust-region
        # gate's cost in rejected candidates. Pure logging; counters already
        # exist.
        try:
            _pc, _ac = asnumpy(prop_counts), asnumpy(acc_counts)

            def _rate(row, cold_only):
                pr = _pc[row][0] if cold_only else _pc[row]
                ac = _ac[row][0] if cold_only else _ac[row]
                tot = float(pr.sum())
                return (float(ac.sum()) / tot if tot > 0 else float("nan"), tot)

            (rj_c, rj_cn), (rj_a, rj_an) = _rate(0, True), _rate(0, False)
            (im_c, im_cn), (im_a, im_an) = _rate(1, True), _rate(1, False)
            logger.info(
                "[GB_ACCEPT %s] rj cold %.4f (n=%.0f) all %.4f (n=%.0f) | "
                "in-model cold %.4f (n=%.0f) all %.4f (n=%.0f)",
                self.name, rj_c, rj_cn, rj_a, rj_an, im_c, im_cn, im_a, im_an,
            )
            # * acceptance of the Fisher jumps per jump scale, cold chain and all temperatures
            scale_counts = asnumpy(self._jump_scale_counts)
            scale_parts = []
            for scale_index, scale in enumerate(asnumpy(self.jump_scales)):
                proposed = scale_counts[0, scale_index]
                accepted_jumps = scale_counts[1, scale_index]
                cold = accepted_jumps[0] / proposed[0] if proposed[0] else float("nan")
                total = accepted_jumps.sum() / proposed.sum() if proposed.sum() else float("nan")
                scale_parts.append(
                    f"{scale:.3f}: cold {cold:.4f} (n={proposed[0]}) "
                    f"all {total:.4f} (n={proposed.sum()})"
                )
            logger.info(
                "[GB_ACCEPT_SCALES %s] Fisher jump scale %s", self.name, "; ".join(scale_parts)
            )
        except Exception as exc:  # never break a propose for a log line
            logger.debug("[GB_ACCEPT %s] skipped: %r", self.name, exc)

        if self._track_counter is not None:
            logger.info(f"{self.name}: {self._track_counter.summary()}")

        # Stage-timing breakdown for this propose (see _ProposeTimer).
        logger.info(
            "[GB_TIMING %s] %s",
            self.name,
            tm.report(time.perf_counter() - st_all),
        )
        if self._mem_probe is not None:
            self._mem_probe.finish()

        self._release_stft_comp_groups()

        return new_state, accepted

    def _track_note(self) -> str:
        """Store-window track count so far, appended to the drift warnings; empty without windows."""
        if self._track_counter is None:
            return ""
        return f" So far this propose, {self._track_counter.summary()}."

    def check_ll_inject(self, model, band_sorter, verbose=False):
        # breakpoint()
        init_like = model.analysis_container_arr.likelihood()
        model.analysis_container_arr.zero_out_data_arr()
        # Restore the non-GB residual snapshot per-shard inside each owning
        # device context. Matches the snapshot loop in propose().
        self._restore_linear_data_arr(
            model.analysis_container_arr, self.reset_non_gb_linear_data_arr
        )
        self.add_cold_chain_sources_to_residual(model, band_sorter, apply_inds=True)
        final_like = model.analysis_container_arr.likelihood()
        return final_like

    @staticmethod
    def _snapshot_linear_data_arr(aca):
        """Per-GPU shard copy of ``aca.linear_data_arr``. Returns a list of
        device-local buffers (one per entry in ``aca.linear_data_arr``).
        """
        if aca.gpus is None:
            return [b.copy() for b in aca.linear_data_arr]
        main_gpu = cp.cuda.runtime.getDevice()
        try:
            out = []
            for i, gpu in enumerate(aca.gpus):
                with cp.cuda.Device(int(gpu)):
                    out.append(aca.linear_data_arr[i].copy())
            return out
        finally:
            cp.cuda.runtime.setDevice(main_gpu)

    @staticmethod
    def _restore_linear_data_arr(aca, snapshot):
        """In-place restore of every shard from the matching snapshot entry."""
        if aca.gpus is None:
            for buf, snap in zip(aca.linear_data_arr, snapshot):
                buf[:] = snap[:]
            return
        main_gpu = cp.cuda.runtime.getDevice()
        try:
            for i, gpu in enumerate(aca.gpus):
                with cp.cuda.Device(int(gpu)):
                    aca.linear_data_arr[i][:] = snapshot[i][:]
        finally:
            cp.cuda.runtime.setDevice(main_gpu)

    @property
    def ranks_needed(self):
        if not hasattr(self, "_ranks_needed"):
            raise ValueError("Need to set ranks needed for this class.")

        return self._ranks_needed

    @ranks_needed.setter
    def ranks_needed(self, ranks_needed):
        assert isinstance(ranks_needed, int)
        self._ranks_needed = ranks_needed


class GBSpecialStretchMove(GBSpecialBase):
    """In-model GB move with the band-aware group-stretch / info-matrix mix.

    All machinery lives in :class:`GBSpecialBase`; the cold-chain friend
    table for the group stretch is rebuilt at the top of every ``propose``
    call (see ``build_friend_index``), so no per-iteration setup is needed
    here.
    """

    pass


class GBSpecialRJPriorMove(GBSpecialBase):
    """Reversible-jump GB move that draws proposals from the prior distribution."""
    pass


def gb_search_func(comm, curr, main_rank, class_extra_gpus, class_ranks_list):
    """Worker entry point for ranks dedicated to the GB bulk search.

    # TODO/DOCS: full handshake protocol with ``main_rank``; the body is the
    canonical reference. Worker ranks pin themselves to a GPU, receive
    band-split assignments, run :func:`run_gb_bulk_search`, and report
    back.

    Args:
        comm: MPI communicator.
        curr: Global-fit info object.
        main_rank: Rank that orchestrates the search.
        class_extra_gpus: GPU indices owned by this move class.
        class_ranks_list: Ranks owned by this move class.
    """
    assert comm is not None

    # get current rank and get index into class_ranks_list
    logger.info(f"INSIDE GB search, RANK: {comm.Get_rank()}")
    rank = comm.Get_rank()
    rank_index = class_ranks_list.index(rank)
    if rank_index == 0:
        comm_info = {"process_ranks_for_fit": class_ranks_list}
        logger.info("waiting to send process ranks")
        comm.send(comm_info, dest=main_rank, tag=232342)
        logger.info("sent process ranks")

    fit_each_leaf(rank, curr, main_rank, comm)

def fit_gmm(samples, comm, comm_info):
    """Fit a Gaussian mixture model to per-leaf GB chain samples.

    # TODO/DOCS: cross-rank protocol; mirrors the helper in
    :mod:`lisatools.globalfit.galaxyglobal` but is invoked by worker ranks
    of the GB special move set.
    """

    if len(samples) == 0:
        return None

    keep = np.arange(8)  # array([0, 1, 2, 4, 6, 7])

    if samples.ndim == 4:
        num_keep, num_samp, nwalkers_keep, ndim = samples.shape

        args = []
        for band in range(num_keep):
            args.append(samples[band].reshape(-1, ndim)[:, keep])

    elif samples.ndim == 2:
        max_groups = samples[:, 0].astype(int).max()

        args = []
        for group in np.unique(samples[:, 0].astype(int)):
            keep_samp = samples[:, 0].astype(int) == group
            if keep_samp.sum() > 0:
                if np.any(np.isnan(samples[keep_samp, 3:])) or np.any(
                    np.isinf(samples[keep_samp, 3:])
                ):
                    breakpoint()
                args.append(samples[keep_samp, 3:][:, keep])

    else:
        raise ValueError

    # for debugging
    # args = args[:1000]

    batch = 10000
    breaks = np.arange(0, len(args) + batch, batch)
    logger.info("BREAKS", breaks)
    if len(breaks) == 1:
        breakpoint()
    process_ranks_for_fit = comm_info["process_ranks_for_fit"]
    gmm_info_all = []
    for i in range(len(breaks) - 1):
        start = breaks[i]
        end = breaks[i + 1]
        args_tmp = args[start:end]
        gmm_info = [None for tmp in args_tmp]
        gmm_complete = np.zeros(len(gmm_info), dtype=bool)

        # OPPOSITE
        # send_tags = comm_info["rec_tags"]
        # rec_tags = comm_info["send_tags"]
        outer_iteration = 0
        current_send_arg_index = 0
        current_status = [False for _ in process_ranks_for_fit]

        while np.any(~gmm_complete):
            time.sleep(0.1)
            if current_send_arg_index >= len(args_tmp) and np.all(~np.asarray(current_status)):
                current_send_arg_index = 0

            outer_iteration += 1
            if outer_iteration % 500 == 0:
                logger.info(
                    f"ITERATION: {outer_iteration}, need:",
                    np.sum(~gmm_complete),
                    current_status,
                )

            for proc_i, proc_rank in enumerate(process_ranks_for_fit):
                # time.sleep(0.6)
                if current_status[proc_i]:
                    rec_tag = int(str(proc_rank) + "4545")
                    check_output = comm.irecv(source=proc_rank)

                    if not check_output.get_status():
                        check_output.cancel()
                    else:
                        # first two give some delay for the processor that messes up
                        try:
                            output_info = check_output.wait()
                        except (
                            pickle.UnpicklingError,
                            UnicodeDecodeError,
                            ValueError,
                            OverflowError,
                        ) as e:
                            current_status[proc_i] = False
                            logger.warning("BAD error on return")
                            continue
                        if "BAD" in output_info:
                            current_status[proc_i] = False
                            logger.warning("BAD", output_info["BAD"])
                            continue
                        # print(output_info)

                        arg_index = output_info["arg"]
                        rank_recv = output_info["rank"]
                        output_list = output_info["output"]

                        gmm_info[arg_index] = output_list
                        gmm_complete[arg_index] = True
                        current_status[proc_i] = False

                        if gmm_complete.sum() + 25 > len(args):
                            print(proc_i, current_status)

                if not current_status[proc_i]:
                    while (
                        current_send_arg_index < len(args_tmp)
                        and gmm_complete[current_send_arg_index]
                    ):
                        current_send_arg_index += 1

                    if current_send_arg_index < len(args_tmp):
                        send_info = {
                            "samples": args_tmp[current_send_arg_index],
                            "arg": current_send_arg_index,
                        }
                        # print("sending", process_ranks_for_fit[index_add])
                        send_tag = int(str(proc_rank) + "67676")
                        comm.send(send_info, dest=proc_rank, tag=send_tag)
                        current_status[proc_i] = True

                        current_send_arg_index += 1

        gmm_info_all.append(gmm_info)

    weights = [tmp[0] for tmp in gmm_info]
    means = [tmp[1] for tmp in gmm_info]
    covs = [tmp[2] for tmp in gmm_info]
    invcovs = [tmp[3] for tmp in gmm_info]
    dets = [tmp[4] for tmp in gmm_info]
    mins = [tmp[5] for tmp in gmm_info]
    maxs = [tmp[6] for tmp in gmm_info]

    output = [weights, means, covs, invcovs, dets, mins, maxs]

    return output


def fit_each_leaf(rank, curr, gather_rank, comm):
    """Worker-side helper that fits one GMM per assigned GB leaf."""

    run_process = True

    while run_process:
        try:
            check = comm.recv(source=gather_rank)
        except (
            pickle.UnpicklingError,
            UnicodeDecodeError,
            ValueError,
            OverflowError,
        ) as e:
            # print("BAD BAD ", rank)
            comm.send({"BAD": "receiving issue"}, dest=gather_rank, tag=send_tag)
            continue

        if isinstance(check, str):
            if check == "end":
                run_process = False
            continue

        assert isinstance(check, dict)

        try:
            arg_index = check["arg"]

            # print("INSIDE", rank, arg_index)
            samples = check["samples"]
        except KeyError:
            comm.send({"BAD": "KeyError"}, dest=gather_rank, tag=send_tag)
            continue

        assert isinstance(samples, np.ndarray)

        gmm = GMMFit(samples)
        output_list = [
            gmm.keep_mix.weights_,
            gmm.keep_mix.means_,
            gmm.keep_mix.covariances_,
            np.array(
                [
                    np.linalg.inv(gmm.keep_mix.covariances_[i])
                    for i in range(len(gmm.keep_mix.weights_))
                ]
            ),
            np.array(
                [
                    np.linalg.det(gmm.keep_mix.covariances_[i])
                    for i in range(len(gmm.keep_mix.weights_))
                ]
            ),
            gmm.sample_mins,
            gmm.sample_maxs,
        ]
        comm.send({"output": output_list, "rank": rank, "arg": arg_index}, dest=gather_rank)
    return


def gb_refit_func(comm, curr, main_rank, class_extra_gpus, class_ranks_list):
    """Worker entry point for ranks dedicated to GMM-based GB refits.

    # TODO/DOCS: full protocol; coordinates the per-leaf GMM refits used to
    refresh ``rj_proposal_distribution`` on the GB RJ moves.
    """
    assert comm is not None

    # get current rank and get index into class_ranks_list
    logger.info(f"INSIDE GB refit, RANK: {comm.Get_rank()}")
    rank = comm.Get_rank()
    rank_index = class_ranks_list.index(rank)
    gather_rank = class_ranks_list[0]
    if rank_index == 0:
        split_remainder = 0  # will fix this setup in the future
        num_search = 2
        gpu = class_extra_gpus[0]
        comm_info = {"process_ranks_for_fit": class_ranks_list[1:]}
        # run search here
        # run_gb_bulk_search(gpu, curr, comm, comm_info, main_rank, num_search, split_remainder)
        pass

    else:
        # run GMM fit here
        fit_each_leaf(rank, curr, gather_rank, comm)
        pass

def para_log_like(
    x,
    gb,
    acs,
    walker_max,
    transform_fn,
    phase_maximize,
    waveform_kwargs,
    fstat=True,
    return_snr=False,
):
    """Vectorized GB log-likelihood used by serial-search and refit moves.

    Args:
        x: GB parameter rows (untransformed).
        gb: :class:`gbgpu.GBGPU` instance.
        acs: :class:`AnalysisContainerArray`.
        walker_max: Index of the walker whose data the proposals are scored
            against.
        transform_fn: :class:`TransformContainer` for GB parameters.
        phase_maximize: If ``True``, marginalize over phase.
        waveform_kwargs: Forwarded to ``gb.get_fstat_ll`` / ``gb.get_ll``.
        fstat: If ``True``, use the F-statistic likelihood (``get_fstat_ll``)
            and overwrite the amplitude / phase / iota / polarization
            entries of ``x`` with their maximized values.
        return_snr: If ``True`` and ``fstat`` is ``False``, also return the
            optimal SNR per row.

    Returns:
        Per-row log-likelihood (or ``(ll, snr)`` tuple).
    """
    xp = gb.backend.xp

    x_tmp = transform_fn.both_transforms(x, xp=xp)
    # need to get just f, fdot, fddot, alpha, delta
    data_index = xp.full(x.shape[0], walker_max, dtype=xp.int32)
    if fstat:
        x_in = x_tmp[:, xp.array([1, 2, 3, 7, 8])]
        # breakpoint()
        # TODO: fix for N>256?
        ll = gb.get_fstat_ll(
            x_in,
            acs.linear_data_arr,
            acs.linear_psd_arr,
            data_index=data_index,
            noise_index=data_index,
            data_length=acs.end_shape[0],
            data_splits=np.array([gb.gpus[0]]),
            phase_maximize=phase_maximize,
            return_cupy=True,
            N=512,  
            **waveform_kwargs,
        )

        x[:, 0] = np.log(gb.A_max)
        x[:, 3] = gb.phi0_max % (2 * np.pi)
        x[:, 4] = np.cos(gb.iota_max % (np.pi))
        x[:, 5] = gb.psi_max % (np.pi)

    else:
        # breakpoint()
        x_in = x_tmp[:]
        ll = gb.get_ll(
            x_in,
            acs.linear_data_arr,
            acs.linear_psd_arr,
            data_index=data_index,
            noise_index=data_index,
            data_length=acs.end_shape[0],
            data_splits=np.array([gb.gpus[0]]),
            phase_maximize=phase_maximize,
            return_cupy=True,
            # N=512,
            **waveform_kwargs,
        )
        # breakpoint()

        # params_remove_in = x_in.copy()
        # params_add_in = x_in.copy()

        # params_remove_in[:, 0] *= 1e-50
        # breakpoint()
        # ll_diff_2 = gb.swap_likelihood_difference(
        #     params_remove_in,
        #     params_add_in,
        #     acs.linear_data_arr,
        #     acs.linear_psd_arr,
        #     # start_freq_ind=self.xp.asarray(self.acs.start_freq_ind).astype(np.int32),
        #     data_index=data_index,
        #     noise_index=data_index,
        #     # N=N_vals,
        #     data_length=acs.data_length,
        #     data_splits=np.array([gb.gpus[0]]),
        #     phase_maximize=phase_maximize,
        #     return_cupy=True,
        #     N=256,
        #     **waveform_kwargs,
        # )
        # breakpoint()

        if phase_maximize:
            x[:, 3] = (x[:, 3] - xp.angle(xp.asarray(gb.non_marg_d_h))) % (2 * np.pi)

        if return_snr:
            opt_snr = gb.h_h.real ** (1 / 2)
            return (ll, opt_snr)

    return ll


class PriorTransformFn:
    """Transform between unit-cube prior coordinates and GB :math:`(f, \\dot f)`.

    Used by the serial-search move to draw uniform-in-band proposals while
    keeping the rest of the GB prior unchanged.

    Args:
        f_min: Minimum frequency (Hz).
        f_max: Maximum frequency (Hz).
        fdot_min: Minimum frequency derivative.
        fdot_max: Maximum frequency derivative.
    """

    def __init__(self, f_min: float, f_max: float, fdot_min: float, fdot_max: float):
        self.f_min, self.f_max, self.fdot_min, self.fdot_max = (
            f_min,
            f_max,
            fdot_min,
            fdot_max,
        )

    def adjust_logp(self, logp, groups_running):
        """Add the (uniform) ``f`` and ``fdot`` log-density to ``logp``."""
        xp = get_array_module(self.f_min)

        if groups_running is None:
            groups_running = xp.arange(len(self.f_min))

        f_min_here = self.f_min[groups_running]
        f_max_here = self.f_max[groups_running]
        f_logpdf = np.log(1.0 / (f_max_here - f_min_here))

        fdot_min_here = self.fdot_min[groups_running]
        fdot_max_here = self.fdot_max[groups_running]
        fdot_logpdf = np.log(1.0 / (fdot_max_here - fdot_min_here))

        logp[:] += f_logpdf[:, None, None]
        logp[:] += fdot_logpdf[:, None, None]

        return logp

    def transform_to_prior_basis(self, coords, groups_running):
        """Map ``f`` / ``fdot`` columns of ``coords`` to the unit-cube basis."""
        xp = get_array_module(self.f_min)

        if groups_running is None:
            groups_running = xp.arange(len(self.f_min))

        f_min_here = self.f_min[groups_running]
        f_max_here = self.f_max[groups_running]
        try:
            coords[:, :, :, 1] = (coords[:, :, :, 1] - f_min_here[:, None, None]) / (
                f_max_here[:, None, None] - f_min_here[:, None, None]
            )
        except:
            breakpoint()

        fdot_min_here = self.fdot_min[groups_running]
        fdot_max_here = self.fdot_max[groups_running]
        coords[:, :, :, 2] = (coords[:, :, :, 2] - fdot_min_here[:, None, None]) / (
            fdot_max_here[:, None, None] - fdot_min_here[:, None, None]
        )

        return

    def transform_from_prior_basis(self, coords, groups_running):
        """Map ``f`` / ``fdot`` columns of ``coords`` from unit cube back to physical."""
        if groups_running is None:
            groups_running = xp.arange(len(self.f_min))

        assert groups_running.shape[0] == coords.shape[0]
        f_min_here = self.f_min[groups_running]
        f_max_here = self.f_max[groups_running]
        coords[:, :, :, 1] = (
            coords[:, :, :, 1] * (f_max_here[:, None, None] - f_min_here[:, None, None])
        ) + f_min_here[:, None, None]

        fdot_min_here = self.fdot_min[groups_running]
        fdot_max_here = self.fdot_max[groups_running]
        coords[:, :, :, 2] = (
            coords[:, :, :, 2] * (fdot_max_here[:, None, None] - fdot_min_here[:, None, None])
        ) + fdot_min_here[:, None, None]

        return


class BayesGMMFit:
    """Variational Bayesian GMM fit to per-leaf GB samples (sklearn ``BayesianGaussianMixture``).

    Stores the per-feature min/max so samples can be transformed in/out of
    a ``[-1, 1]`` GMM basis.

    Args:
        samples_in: 2D NumPy array of GB samples to fit.
    """

    def __init__(self, samples_in):

        assert isinstance(samples_in, np.ndarray)

        run = True
        min_bic = np.inf
        self.sample_mins = sample_mins = samples_in.min(axis=0)
        self.sample_maxs = sample_maxs = samples_in.max(axis=0)

        samples = self.transform_to_gmm_basis(samples_in)

        mixture = BayesianGaussianMixture(
            weight_concentration_prior_type="dirichlet_distribution",
            n_components=60,
            # reg_covar=0,
            # init_params="random",
            max_iter=5000,
            # mean_precision_prior=0.8,
            # random_state=random_state,
        )
        mixture.fit(samples)

        self.keep_mix = mixture

    def transform_to_gmm_basis(self, samples):
        """Map samples from physical to ``[-1, 1]`` GMM basis."""
        return (
            (samples - self.sample_mins[None, :])
            / (self.sample_maxs[None, :] - self.sample_mins[None, :])
        ) * 2 - 1

    def transform_from_gmm_basis(self, samples):
        """Map samples from ``[-1, 1]`` GMM basis back to physical."""
        return (samples + 1.0) / 2.0 * (
            self.sample_maxs[None, :] - self.sample_mins[None, :]
        ) + self.sample_mins[None, :]


from sklearn.mixture import GaussianMixture


class GMMFit:
    """Plain GMM fit to per-leaf GB samples (sklearn ``GaussianMixture``).

    Args:
        samples_in: 2D NumPy array of GB samples to fit.
    """

    def __init__(self, samples_in):

        assert isinstance(samples_in, np.ndarray)

        run = True
        min_bic = np.inf
        self.sample_mins = sample_mins = samples_in.min(axis=0)
        self.sample_maxs = sample_maxs = samples_in.max(axis=0)

        samples = self.transform_to_gmm_basis(samples_in)

        mixture = GaussianMixture(n_components=30, verbose=False, verbose_interval=2)

        mixture.fit(samples)

        # bad = False
        # for n_components in range(1, 31)[-1:]:
        #     if not run:
        #         continue
        #     #fit_gaussian_mixture_model(n_components, samples)
        #     #breakpoint()
        #     try:
        #         mixture = GaussianMixture(n_components=n_components, verbose=False, verbose_interval=2)

        #         mixture.fit(samples)
        #         test_bic = mixture.bic(samples)
        #     except ValueError:
        #         # print("ValueError", samples)
        #         run = False
        #         bad = True
        #         continue
        #     # print(n_components, test_bic)
        #     if test_bic < min_bic:
        #         min_bic = test_bic
        #         keep_mix = mixture
        #         keep_components = n_components

        #     else:
        #         run = False

        #         # print(leaf, n_components - 1, et - st)

        #     """if keep_components >= 9:
        #         new_samples = keep_mix.sample(n_samples=100000)[0]
        #         old_samples = samples
        #         fig = corner.corner(old_samples, hist_kwargs=dict(density=True, color="r"), color="r", plot_datapoints=False, plot_density=False)
        #         corner.corner(new_samples, hist_kwargs=dict(density=True, color="b"), color="b", plot_datapoints=False, plot_contours=True, plot_density=False, fig=fig)
        #         fig.savefig("mix_check.png")
        #         plt.close()
        #         breakpoint()"""

        # if bad:
        #     print("BAD")
        # if keep_components >= 19:
        #     print(keep_components)
        # # output_list = [keep_mix.weights_, keep_mix.means_, keep_mix.covariances_, np.array([np.linalg.inv(keep_mix.covariances_[i]) for i in range(len(keep_mix.weights_))]), np.array([np.linalg.det(keep_mix.covariances_[i]) for i in range(len(keep_mix.weights_))]), sample_mins, sample_maxs]

        self.keep_mix = mixture

    def transform_to_gmm_basis(self, samples):
        """Map samples from physical to ``[-1, 1]`` GMM basis."""
        return (
            (samples - self.sample_mins[None, :])
            / (self.sample_maxs[None, :] - self.sample_mins[None, :])
        ) * 2 - 1

    def transform_from_gmm_basis(self, samples):
        """Map samples from ``[-1, 1]`` GMM basis back to physical."""
        return (samples + 1.0) / 2.0 * (
            self.sample_maxs[None, :] - self.sample_mins[None, :]
        ) + self.sample_mins[None, :]


def gather_gmms(gmms):
    """Pack a list of GMM fits into the dict format expected by :func:`make_gmm`."""
    weights = []
    means = []
    covs = []
    inv_covs = []
    dets = []
    sample_mins = []
    sample_maxs = []

    for gmm in gmms:
        weights.append(gmm.keep_mix.weights_)
        means.append(gmm.keep_mix.means_)
        covs.append(gmm.keep_mix.covariances_)
        inv_covs.append(
            np.array(
                [
                    np.linalg.inv(gmm.keep_mix.covariances_[i])
                    for i in range(len(gmm.keep_mix.weights_))
                ]
            )
        )
        dets.append(
            np.array(
                [
                    np.linalg.det(gmm.keep_mix.covariances_[i])
                    for i in range(len(gmm.keep_mix.weights_))
                ]
            )
        )
        sample_mins.append(gmm.sample_mins)
        sample_maxs.append(gmm.sample_maxs)

    return (weights, means, covs, inv_covs, dets, sample_mins, sample_maxs)


from lisatools.sampling.gmm import vec_fit_gmm_min_bic

class GBSpecialRJSerialSearchMCMC(GBSpecialBase):
    """Reversible-jump GB move that runs a serial F-statistic MCMC search per band.

    Each band proposes one new GB at a time using a parallel ensemble
    sampler driven by :func:`para_log_like`, with proposals drawn from a
    band-restricted prior via :class:`PriorTransformFn`.
    """
    comm_info = None

    def get_rank_function(self):
        return gb_search_func

    def setup(self, model, branches):
        assert isinstance(self.search_kwargs, dict)
        nwalkers: int = self.search_kwargs["nwalkers"]
        ntemps: int = self.search_kwargs["ntemps"]
        shutoff_band_iteration: int = self.search_kwargs["shutoff_band_iteration"]
        shutoff_frequency_threshold: float = self.search_kwargs["shutoff_frequency_threshold"]
        burn_1: int = self.search_kwargs["burn_1"]
        nsteps_1: int = self.search_kwargs["nsteps_1"]
        snr_threshold: float = self.search_kwargs["snr_threshold"]
        burn_2: int = self.search_kwargs["burn_2"]
        nsteps_2: int = self.search_kwargs["nsteps_2"]

        # FOR FAST TESTING/DEBUGGING
        # import pickle
        # with open("gmm_tmp.pickle", "rb") as fp:
        #     full_gmm = pickle.load(fp)

        # rj_dist = ProbDistContainer(
        #     {
        #         ("A", "f0", "fdot", "cos_iota", "alpha", "sin_delta"): full_gmm,
        #         "phi0": uniform_dist(0.0, 2 * np.pi),
        #         "psi": uniform_dist(0.0, np.pi),
        #     },
        #     use_cupy=True,
        # )
        # rj_dist.reset_key_order(["A", "f0", "fdot", "phi0", "cos_iota", "psi", "alpha", "sin_delta"])
        # return

        # run paraensemble MCMC.
        max_logl_walker = np.argmax(model.analysis_container_arr.likelihood()).item()
        self.gb.d_d = model.analysis_container_arr.inner_product()[max_logl_walker] # 0.0
        ndim = branches["gb"].ndim
        priors_global = self.priors if not self.backend.uses_cuda else self.gpu_priors            

        if self.num_bands == 1:
            f0_max = self.band_edges[1:]
            f0_min = self.band_edges[:-1]
        else:
            f0_max = self.band_edges[2:-1]
            f0_min = self.band_edges[1:-2]

        # logic to shutoff bands #? Think about how this should change when we change SNR_thresh and with changing noise
        if self.num_proposals >= shutoff_band_iteration:
            bands_to_shutoff = np.all(~self.found_source_in_band[-shutoff_band_iteration:, :], axis=0)

            if shutoff_frequency_threshold is not None:
                min_freqs = getattr(f0_min, "get")() if hasattr(f0_min, "get") else f0_min
                freq_mask = min_freqs >= shutoff_frequency_threshold
                bands_to_shutoff = bands_to_shutoff & freq_mask

            if np.all(bands_to_shutoff):
                logger.info(f"No sources found across all bands for {shutoff_band_iteration} iterations, reverting to priors")
                self.rj_proposal_distribution = priors_global
                return

            else:
                shutoff_mask = ~bands_to_shutoff
                f0_max = f0_max[shutoff_mask]
                f0_min = f0_min[shutoff_mask]
                ngroups = np.sum(shutoff_mask)

        else:
            ngroups = max(1, self.num_bands - 2)
            assert f0_max.shape[0] == ngroups
            assert f0_min.shape[0] == ngroups
            bands_to_shutoff = None

        logger.info(f"The current number of active bands is {ngroups}")

        fdot_max = get_fdot_mojito(f0_max, sign="+")
        fdot_min = get_fdot_mojito(f0_max, sign="-")

        # * The search samples f0 and fdot on the unit cube. The run prior may hold them as one
        # * joint entry, so the dictionary is rebuilt in order rather than updated in place.
        priors_in = {}
        for prior_key, prior_here in deepcopy(priors_global)["gb"].priors_in.items():
            if prior_key in ("f0", ("f0", "fdot")):
                priors_in["f0"] = uniform_dist(0.0, 1.0, use_cupy=self.backend.uses_cupy)
                priors_in["fdot"] = uniform_dist(0.0, 1.0, use_cupy=self.backend.uses_cupy)
            elif prior_key != "fdot":
                priors_in[prior_key] = prior_here
        priors = {
            "gb": ProbDistContainer(priors_in, return_gpu=True, use_cupy=self.backend.uses_cupy)
        }
        start_params = priors["gb"].rvs(size=(ngroups, ntemps, nwalkers))
        prior_transform_fn = PriorTransformFn(f0_min * 1e3, f0_max * 1e3, fdot_min, fdot_max)
        prior_transform_fn.transform_from_prior_basis(start_params, self.xp.arange(ngroups))

        #? print("phase maximizing here right now (?)")
        ll_args = (
            self.gb,
            model.analysis_container_arr,
            max_logl_walker,
            self.parameter_transforms,
            True,  # self.phase_maximize,
            self.waveform_kwargs,
        )

        ll_args_2 = (
            self.gb,
            model.analysis_container_arr,
            max_logl_walker,
            self.parameter_transforms,
            self.phase_maximize, # False, #
            self.waveform_kwargs,
        )

        # test_ll = para_log_like(
        #     test_params,
        #     *ll_args
        # )

        gibbs_sampling_setup = np.ones(8, dtype=bool)
        gibbs_sampling_setup[np.array([0, 3, 4, 5])] = False
        para_sampler = ParaEnsembleSampler(
            ndim,
            nwalkers,
            ngroups,
            para_log_like,
            priors,
            tempering_kwargs=dict(ntemps=ntemps, Tmax=np.inf),
            args=ll_args,
            # kwargs: dict = {},
            gpu=self.gb.gpus[0],
            periodic=self.periodic,
            # backend: ParaBackend = None,  # add ParaHDFBackend
            # update_fn: Callable = None,
            # update_iterations=-1,
            # stopping_fn: Callable = None,
            # stopping_iterations: int=-1,
            prior_transform_fn=prior_transform_fn,
            name="gb",
            gibbs_sampling_setup=gibbs_sampling_setup,
            # provide_supplemental=False,
        )

        from eryn.state import ParaState

        state = ParaState({"gb": start_params}, groups_running=self.xp.ones(ngroups, dtype=bool))
        state.log_prior = para_sampler.compute_log_prior(state.branches_coords)
        state.log_like = para_sampler.compute_log_like(state.branches_coords, logp=state.log_prior)

        para_sampler.run_mcmc(state, nsteps_1, burn=burn_1, progress=True)

        samples = self.xp.asarray(para_sampler.get_chain()[:, :, 0])
        # Diagnostics disabled (stft_tof): computed-but-unused and each costs
        # a full likelihood sweep over all samples.
        # check_ll = para_sampler.get_log_like()[:, :, 0]
        # sample_ll = asnumpy(
        #     para_log_like(samples.reshape(-1, 8), *ll_args).reshape(samples.shape[:-1])
        # )

        # check_real_ll_phase_maximized = asnumpy(
        #     para_log_like(samples.reshape(-1, 8), *ll_args, fstat=False)
        #     .reshape(samples.shape[:-1])
        # )
        check_real_ll, opt_snr = para_log_like(
            samples.reshape(-1, 8), *ll_args_2, fstat=False, return_snr=True
        )
        check_real_ll = asnumpy(check_real_ll.reshape(samples.shape[:-1]))
        opt_snr = asnumpy(opt_snr.reshape(samples.shape[:-1]))

        # np.save("opt_snr_from_ldc_parasampler_check.npy", opt_snr)
        # np.save("samples_from_ldc_parasampler_check.npy", samples)

        # TODO: make cut adjustable
        groups_running_now = opt_snr.min(axis=(0, 2)) > snr_threshold

        if self.num_proposals == 0:
            self.found_source_in_band = groups_running_now
        else:
            if bands_to_shutoff is None:
                self.found_source_in_band = np.vstack([self.found_source_in_band, groups_running_now])
            else:
                shutoff_temp = np.zeros(self.found_source_in_band.shape[1], dtype=bool)
                shutoff_temp[~bands_to_shutoff] = groups_running_now
                self.found_source_in_band = np.vstack([self.found_source_in_band, shutoff_temp])

        logger.info(f"Found a source in {groups_running_now.sum()} out of {groups_running_now.shape[0]} active bands")
        if not np.any(groups_running_now):
            logger.info("Did not find any new sources.")
            return

        start_params_2 = np.tile(samples[-1][groups_running_now, None], (1, ntemps, 1, 1))
        # Maybe not start from maximized values?
        gibbs_sampling_setup_2 = np.ones(8, dtype=bool)
        if ll_args_2[4]: # phase_maximization
            gibbs_sampling_setup_2[np.array([3])] = False

        prior_transform_fn_2 = PriorTransformFn(
            f0_min[groups_running_now] * 1e3,
            f0_max[groups_running_now] * 1e3,
            fdot_min[groups_running_now],
            fdot_max[groups_running_now],
        )
        ngroups_2 = groups_running_now.sum().item()
        # prior_transform_fn_2.transform_from_prior_basis(start_params_2, self.xp.arange(ngroups_2))

        para_sampler_2 = ParaEnsembleSampler(
            ndim,
            nwalkers,
            ngroups_2,
            para_log_like,
            priors,
            tempering_kwargs=dict(ntemps=ntemps, Tmax=np.inf),
            args=ll_args_2,
            kwargs=dict(fstat=False),
            gpu=self.gb.gpus[0],
            periodic=self.periodic,
            # backend: ParaBackend = None,  # add ParaHDFBackend
            # update_fn: Callable = None,
            # update_iterations=-1,
            # stopping_fn: Callable = None,
            # stopping_iterations: int=-1,
            prior_transform_fn=prior_transform_fn_2,
            name="gb",
            gibbs_sampling_setup=gibbs_sampling_setup_2,
            # provide_supplemental=False,
        )

        new_state = ParaState(
            {"gb": start_params_2}, groups_running=self.xp.ones(ngroups_2, dtype=bool)
        )
        new_state.log_prior = para_sampler_2.compute_log_prior(new_state.branches_coords)
        new_state.log_like = para_sampler_2.compute_log_like(
            new_state.branches_coords, logp=new_state.log_prior
        )

        if np.any(np.isinf(new_state.log_prior)):
            breakpoint()

        para_sampler_2.run_mcmc(new_state, nsteps_2, burn=burn_2, progress=True)

        samples_2 = self.xp.asarray(para_sampler_2.get_chain()[:, :, 0])
        # check_ll_2 = para_sampler_2.get_log_like()[:, :, 0]

        # Diagnostics disabled (stft_tof): computed-but-unused likelihood sweeps.
        # check_real_ll_phase_maximized_2 = asnumpy(
        #     para_log_like(samples_2.reshape(-1, 8), *ll_args, fstat=False)
        #     .reshape(samples_2.shape[:-1])
        # )
        # check_real_ll_2 = asnumpy(
        #     para_log_like(samples_2.reshape(-1, 8), *ll_args_2, fstat=False)
        #     .reshape(samples_2.shape[:-1])
        # )

        samples_2 = samples_2.transpose(1, 0, 2, 3)
        # np.save("/workspace/rrondeel/erebor/testing/highf_gb/search2_samples_check.npy", samples_2)

        st = time.perf_counter()
        samples_2_tmp = samples_2.reshape(samples_2.shape[0], -1, samples_2.shape[-1])[
            :, :, np.array([0, 1, 2, 4, 6, 7])
        ]

        if self.xp.isnan(samples_2_tmp).any() or self.xp.isinf(samples_2_tmp).any():
            logger.warning(
                f"samples_2_tmp contains NaN or Inf before GMM fitting. \
                NaN count: {self.xp.isnan(samples_2_tmp).sum()}. \
                Inf count: {self.xp.isinf(samples_2_tmp).sum()}. \
                Skipping search..."
            )
            return
            # breakpoint()
            # raise ValueError(
            #     f"samples_2_tmp contains NaN or Inf before GMM fitting. "
            #     f"NaN count: {self.xp.isnan(samples_2_tmp).sum()}, "
            #     f"Inf count: {self.xp.isinf(samples_2_tmp).sum()}"
            # )

        ranges = samples_2_tmp.max(axis=1) - samples_2_tmp.min(axis=1)  # (n_groups, n_features)
        degenerate = (ranges == 0)
        if degenerate.any():
            bad_groups, bad_feats = self.xp.where(degenerate)
            logger.warning(
                f"Degenerate features (zero range) in groups {bad_groups} \
                for features {bad_feats}. transform_to_gmm_basis will produce NaN. \
                Skipping search..."
            )
            return
            # breakpoint()
            # raise ValueError(
            #     f"Degenerate features (zero range) in groups {bad_groups} "
            #     f"for features {bad_feats}. transform_to_gmm_basis will produce NaN."
            # )

        full_gmm = vec_fit_gmm_min_bic(
            samples_2_tmp,
            min_comp=1,
            max_comp=30,
            n_samp_bic_test=5000,
            gpu=self.xp.cuda.runtime.getDevice(),
            verbose=False,
        )
        # import pickle
        # with open("gmm_tmp.pickle", "wb") as fp:
        #     pickle.dump(full_gmm, fp, pickle.HIGHEST_PROTOCOL)

        et = time.perf_counter()
        logger.info(f"Runtime of GPU GMM FIT: {round(et - st,3)} seconds")

        rj_dist = ProbDistContainer(
            {
                ("A", "f0", "fdot", "cos_iota", "alpha", "sin_delta"): full_gmm,
                "phi0": uniform_dist(0.0, 2 * np.pi),
                "psi": uniform_dist(0.0, np.pi),
            },
            use_cupy=True,
        )
        rj_dist.reset_key_order(["A", "f0", "fdot", "phi0", "cos_iota", "psi", "alpha", "sin_delta"])
        # if self.ranks_needed == 0:
        #     gmms = [GMMFit(samples_2[i].get().reshape(-1, 8)) for i in range(samples_2.shape[0])[:10]]
        #     gmm_info = gather_gmms(gmms)

        # else:
        #     if self.comm_info is None:
        #         # this only happens the first time through
        #         self.comm_info = self.comm.recv(tag=232342)

        #     gmm_info = fit_gmm(samples_2.get(), self.comm, self.comm_info)

        # full_gmm = FullGaussianMixtureModel(*gmm_info, use_cupy=self.use_gpu)
        # breakpoint()

        # gen_samp = self.xp.asarray(rj_dist.rvs(1000))
        # gen_ll, gen_opt_snr = para_log_like(gen_samp, *ll_args, fstat=False, return_snr=True)
        # print(gen_ll, self.gb.d_h / gen_opt_snr, gen_opt_snr)
        # breakpoint()
        self.rj_proposal_distribution = {"gb": rj_dist}


class GBSpecialRJSearchMove(GBSpecialBase): #? only needed for mutli GPU usage
    """Reversible-jump GB search move that delegates work to extra GPU/MPI ranks.

    # TODO/DOCS: full multi-GPU coordination protocol with the worker
    function :func:`gb_search_func` — used when sufficient ranks are
    available so the bulk search runs concurrently with PE.
    """
    def get_rank_function(self):
        return gb_search_func

    def setup(self, model, branches):
        self.interact_with_search()
        super(GBSpecialRJSearchMove, self).setup(branches) # should be serial move?

    def interact_with_search(self):
        search_rank = self.ranks[0]

        search_ch = self.comm.irecv(source=search_rank)
        if search_ch.get_status():
            search_req = search_ch.wait()

            if "receive" in search_req and search_req["receive"]:
                search_dict = self.comm.recv(source=search_rank)
                self.rj_proposal_distribution["gb"] = make_gmm(self.gb, search_dict["search"])

            if "send" in search_req and search_req["send"]:
                # DEPRECATED (2026-07 rework): this legacy MPI hand-off read
                # from a stored ``self.mgh`` with a two-shard layout and a
                # ``lisasens_shaped`` attribute that no longer exist. When
                # the multi-GPU search path is revived, read the residual /
                # psd from the model ACA at propose time instead.
                raise NotImplementedError(
                    "GBSpecialRJSearchMove residual hand-off needs re-wiring "
                    "to the model's AnalysisContainerArray (legacy self.mgh "
                    "path removed in the 2026-07 rework)."
                )

        else:
            search_ch.cancel()

        # TODO print("CHECK INSIDE PROP")


from lisatools.globalfit.gathergalaxy import gather_gb_samples
from lisatools.globalfit.hdfbackend import GBHDFBackend, GFHDFBackend, MBHHDFBackend
from lisatools.globalfit.state import GBState


class GBSpecialRJRefitMove(GBSpecialBase):
    """Reversible-jump GB move that uses GMM-refitted proposals.

    Loads per-leaf GMM proposals (refit by :func:`gb_refit_func`) and uses
    them as the RJ proposal distribution. This is typically alternated
    with :class:`GBSpecialRJPriorMove` to sharpen accepted GB candidates.
    """
    def __init__(self, *args, fp=None, **kwargs):
        assert fp is not None and isinstance(fp, str)
        assert os.path.exists(fp)
        self.fp = fp
        GBSpecialBase.__init__(self, *args, **kwargs)

    def setup(self, model, branches):
        samples_keep = self.search_kwargs["refit_start_iteration"]
        nwalkers = self.search_kwargs["nwalkers"]
        num_compare_samples = 1
        # FOR FAST TESTING/DEBUGGING
        # import pickle
        # with open("gmm_tmp.pickle", "rb") as fp:
        #     full_gmm = pickle.load(fp)

        # rj_dist = ProbDistContainer(
        #     {
        #         ("A", "f0", "fdot", "cos_iota", "alpha", "sin_delta"): full_gmm,
        #         "phi0": uniform_dist(0.0, 2 * np.pi),
        #         "psi": uniform_dist(0.0, np.pi),
        #     },
        #     use_cupy=True,
        # )
        # rj_dist.key_order = ["A", "f0", "fdot", "phi0", "cos_iota", "psi", "alpha", "sin_delta"]
        # self.rj_proposal_distribution = {"gb": rj_dist}
        # return
        # run paraensemble MCMC.

        max_logl_walker = np.argmax(model.analysis_container_arr.likelihood()).item()
        self.gb.d_d = 0.0  # model.analysis_container_arr.inner_product()[max_logl_walker]
        reader = GFHDFBackend(
            self.fp, sub_state_bases={"gb": GBState}, sub_backend={"gb": GBHDFBackend}
        )

        st = time.perf_counter()
        sens_mat = model.analysis_container_arr[max_logl_walker].sens_mat
        if reader.iteration < 2 * samples_keep:
            logger.info("Not enough samples to perform refitting, reverting to priors.")
            self.rj_proposal_distribution = {"gb": self.priors if not self.backend.uses_cuda else self.gpu_priors}
            return

        num_compare_samples = 1
        nwalkers = 30
        gpu = self.xp.cuda.runtime.getDevice() if self.backend.uses_cupy else -1
        # ``gather_gb_samples`` builds FD waveforms for the GMM refit;
        # the WDM equivalent has not been wired in yet.
        if not isinstance(model.analysis_container_arr.settings, FDSettings):
            raise NotImplementedError(
                "GBSpecialRJRefitMove currently requires the FD basis "
                "(GMM refit fits FD waveforms)."
            )
        fd = model.analysis_container_arr.f_arr.copy()
        groups = gather_gb_samples(
            fd,
            self.parameter_transforms,
            self.gb,
            self.waveform_kwargs.copy(),
            self.band_edges,
            self.band_N_vals,
            reader,
            sens_mat,
            gpu,
            num_compare_samples=num_compare_samples,
            samples_keep=samples_keep,
            thin_by=1,
        )

        num_in_groups = np.asarray([len(tmp) for tmp in groups])
        keep = num_in_groups > nwalkers * samples_keep / 2

        logger.info(
            f"Groups passing sample count filter: {keep.sum()} / {len(keep)}. "
            f"num_in_groups: {num_in_groups}"
        )

        if not keep.any():
            logger.warning(
                f"No groups have enough samples (threshold={nwalkers * samples_keep / 2:.0f}). "
                f"Max samples in any group: {num_in_groups.max()}. "
                f"Reverting to priors."
            )
            self.rj_proposal_distribution = {
                "gb": self.priors if not self.backend.uses_cuda else self.gpu_priors
            }
            return

        max_num_source = max([tmp.shape[0] for tmp in groups])
        samples = np.full((len(groups), max_num_source, groups[0].shape[-1]), np.nan)
        for i, group in enumerate(groups):
            samples[i, : len(group)] = group

        samples_fin = samples[keep]
        num_in_groups_fin = num_in_groups[keep]

        if len(num_in_groups_fin) == 0 or num_in_groups_fin.min() == num_in_groups_fin.max():
            logger.warning(
                f"Cannot construct step range from num_in_groups_fin={num_in_groups_fin}. "
                f"Reverting to priors..."
            )
            self.rj_proposal_distribution = {
                "gb": self.priors if not self.backend.uses_cuda else self.gpu_priors
            }
            return

        cp.cuda.runtime.setDevice(gpu)
        output_info = []
        step = 5
        steps = np.arange(num_in_groups_fin.min(), num_in_groups_fin.max(), step)
        if steps[-1] < num_in_groups_fin.max().item():
            steps = np.concatenate([steps, np.array([num_in_groups_fin.max().item()])])

        weights_all = []
        means_all = []
        covs_all = []
        invcovs_all = []
        dets_all = []
        mins_all = []
        maxs_all = []
        for start, end in zip(steps[:-1], steps[1:]):
            here = (num_in_groups_fin >= start) & (num_in_groups_fin < end)
            # this randomly throughs away ~step amount of samples to make gmm work
            samples_here = samples_fin[here][:, :start, np.array([0, 1, 2, 4, 6, 7])].copy()

            if np.isnan(samples_here).any():
                nan_groups = np.where(np.isnan(samples_here).any(axis=(1, 2)))[0]
                logger.warning(
                    f"NaN padding leaked into samples_here at start={start}. \
                    Affected groups (local indices): {nan_groups}. \
                    num_in_groups for those groups: {num_in_groups_fin[here][nan_groups]} \
                    Skipping Refit..."
                )
                return
                # raise ValueError(
                #     f"NaN padding leaked into samples_here at start={start}. "
                #     f"Affected groups (local indices): {nan_groups}. "
                #     f"num_in_groups for those groups: {num_in_groups_fin[here][nan_groups]}"
                # )

            ranges = samples_here.max(axis=1) - samples_here.min(axis=1)
            if (ranges == 0).any():
                bad = np.where((ranges == 0))
                logger.warning(
                    f"Degenerate features at start={start}: groups={bad[0]}, features={bad[1]} \
                    Skipping Refit..."
                )
                # raise ValueError(
                #     f"Degenerate features at start={start}: groups={bad[0]}, features={bad[1]}"
                # )

            weights, means, covs, invcovs, dets, mins, maxs = vec_fit_gmm_min_bic(
                self.xp.asarray(samples_here),
                min_comp=1,
                max_comp=30,
                n_samp_bic_test=5000,
                gpu=gpu,
                verbose=False,
                return_components=True,
            )
            weights_all += weights
            means_all += means
            covs_all += covs
            invcovs_all += invcovs
            dets_all += dets
            mins_all += mins
            maxs_all += maxs
            # logger.info(start, end)

        full_gmm = FullGaussianMixtureModel(
            weights_all,
            means_all,
            covs_all,
            invcovs_all,
            dets_all,
            mins_all,
            maxs_all,
            use_cupy=True,
        )

        logger.info(f"Runtime GMM Refit: {round(time.perf_counter() - st)}")
        rj_dist = ProbDistContainer(
            {
                ("A", "f0", "fdot", "cos_iota", "alpha", "sin_delta"): full_gmm,
                "phi0": uniform_dist(0.0, 2 * np.pi),
                "psi": uniform_dist(0.0, np.pi),
            },
            use_cupy=True,
        )
        rj_dist.key_order = ["A", "f0", "fdot", "phi0", "cos_iota", "psi", "alpha", "sin_delta"]
        # if self.ranks_needed == 0:
        #     gmms = [GMMFit(samples_2[i].get().reshape(-1, 8)) for i in range(samples_2.shape[0])[:10]]
        #     gmm_info = gather_gmms(gmms)

        # else:
        #     if self.comm_info is None:
        #         # this only happens the first time through
        #         self.comm_info = self.comm.recv(tag=232342)

        #     gmm_info = fit_gmm(samples_2.get(), self.comm, self.comm_info)

        # full_gmm = FullGaussianMixtureModel(*gmm_info, use_cupy=self.use_gpu)
        # breakpoint()

        # gen_samp = self.xp.asarray(rj_dist.rvs(1000))
        # gen_ll, gen_opt_snr = para_log_like(gen_samp, *ll_args, fstat=False, return_snr=True)
        # print(gen_ll, self.gb.d_h / gen_opt_snr, gen_opt_snr)
        # breakpoint()

        self.rj_proposal_distribution = {"gb": rj_dist}


def get_param_limits(array): # can be used for debugging of coordinate values
    """Return per-column min/max of ``array`` (debug helper for GB coordinates)."""
    num_params = array.shape[-1]

    if num_params == 8:
        param_labels = ["A", "f0", "fdot", "phi0", "cos_iota", "psi", "alpha", "sin_delta"]
    elif num_params == 9:
        param_labels = ["A", "f0", "fdot", "fddot", "phi0", "cos_iota", "psi", "alpha", "sin_delta"]
    else:
        param_labels = num_params * [""]
    for i, param_label in enumerate(param_labels):
        param_values = array[..., i]
        min_array_i = param_values.min()
        max_array_i = param_values.max()
        print(f"For parameter {param_label}, the minimun value is {min_array_i}, the maximum value is {max_array_i}")