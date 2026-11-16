"""Stock Erebor recipe: per-source :class:`Setup` / :class:`Settings` definitions."""

from __future__ import annotations

import dataclasses
import typing
from typing import Any, Optional

import h5py
import numpy as np

try:
    import cupy as cp

except (ModuleNotFoundError, ImportError) as e:
    import numpy as cp

import logging

from eryn.backends import HDFBackend as eryn_Backend
from eryn.moves.tempering import make_ladder
from eryn.prior import uniform_dist, log_uniform
from eryn.priors import ProbDistContainer
from eryn.state import State as ErynState
from eryn.state import Branch as ErynBranch
from eryn.utils import TransformContainer
from gbgpu.utils.utility import get_fdot, get_N

from lisatools.sources.utils import ecliptic_to_icrs
from lisatools.utils.utility import AET, detrend, tukey
from lisatools.utils.constants import YRSID_SI, PC_SI

from ...domains import DomainSettingsBase, FDSettings, WDMSettings, STFTSettings
from ..engine import Settings, Setup, GeneralSetup
from ..moves.gbbandstructure import stft_band_structure
from ..moves.gbdebug import GBDebugSettings
from ..loginfo import init_logger
from ereborpriors.priors import F0FdotPrior
from ..searches import MBHBSearchInterface, GBSearchInterface, EMRISearchInterface, SOBHBSearchInterface, NoiseSearchInterface, GalacticForegroundSearchInterface

if typing.TYPE_CHECKING:
    from ..moves.gbspecialmove import GBLeafCapSettings, GBSigHetSettings
    from ..moves.gbspecialmove import GBFStatSettings
    from ..moves.gbspecialmove import GBFStatSplitMergeSettings


# * The GB code reads the sampling columns by position: the amplitude (ln A) first, then these.
GB_POSITIONAL_COLUMNS = ("f0", "fdot", "phi0", "cos_iota", "psi", "alpha", "sin_delta")


@dataclasses.dataclass
class GBWaveformSettings:
    """The GBGPU arguments: its constructor arguments and its waveform call arguments.

    The run chooses the first four fields; :meth:`GBSettings.from_general_setup` fills the others
    from the run's :class:`GeneralSetup`.
    """

    oversample: int = 4
    # ! The STFT computation in build_gb_moves is fixed to XYZ, second-generation TDI.
    tdi_channel_setup: str = "XYZ"
    tdi2: bool = True
    use_c_implementation: bool = True
    # * Filled from the GeneralSetup.
    dt: Optional[float] = None
    T: Optional[float] = None
    start_freq_ind: Optional[int] = None
    window: Optional[str] = None
    window_alpha: Optional[float] = None
    orbits: Any = None
    t0: Optional[float] = None
    force_backend: Optional[str] = None

    @property
    def initialize_kwargs(self) -> dict[str, Any]:
        """The GBGPU constructor arguments."""
        return dict(orbits=self.orbits, t0=self.t0, force_backend=self.force_backend)

    @property
    def waveform_kwargs(self) -> dict[str, Any]:
        """The GBGPU waveform call arguments."""
        return dict(
            dt=self.dt,
            T=self.T,
            use_c_implementation=self.use_c_implementation,
            start_freq_ind=self.start_freq_ind,
            tdi_channel_setup=self.tdi_channel_setup,
            tdi2=self.tdi2,
            oversample=self.oversample,
            window=self.window,
            window_alpha=self.window_alpha,
        )


@dataclasses.dataclass
class GBSTFTSettings:
    """What the STFT band structure and the STFT GB computation are built from.

    The defaults are those of the full-band DFFR run. :meth:`GBSettings.from_general_setup` fills
    the STFT domain and the band taper from the run's :class:`GeneralSetup`.
    """

    n_side_bins: int = 16
    response: str = "dffr"
    # * First-order delays, as in every DFFR run of the GB memory effort (user, 2026-10-30).
    first_order_delays: bool = True
    use_midpoint: bool = True
    linear_envelope: bool = True
    # DFFR does not read it; the TDI-on-the-fly response does.
    freq_from_tdi_phase: bool = True
    # Forwarded to stft_band_structure (k_safety_factor, q, floor_correction, convention).
    band_kwargs: Optional[dict[str, Any]] = None
    # * Filled from the GeneralSetup: the STFT grid and the taper as a fraction of a segment.
    domain_settings: Optional[DomainSettingsBase] = None
    band_alpha: Optional[float] = None

    @property
    def computation_kwargs(self) -> dict[str, Any]:
        """The STFTGBComputations arguments that the run chooses."""
        return dict(
            n_side_bins=self.n_side_bins,
            freq_from_tdi_phase=self.freq_from_tdi_phase,
            use_midpoint=self.use_midpoint,
            linear_envelope=self.linear_envelope,
            response=self.response,
            first_order_delays=self.first_order_delays,
        )


@dataclasses.dataclass
class GBMoveSettings:
    """What build_gb_moves needs: the moves to build and the settings every GB move gets.

    The defaults are those of the full-band DFFR run.
    """

    pe_move_names: typing.Sequence[str] = ("rj_prior",)
    search_move_names: typing.Sequence[str] = ()
    include_refit: bool = False
    Tmax: float = 1e6
    num_repeat_proposals: int = 50
    # * 0: every in-model proposal is the Fisher (FIM) proposal.
    stretch_probability: float = 0.0
    num_stationary_fim_steps: int = 1
    # Bands, and (band, walker) units per tempering chunk, preloaded per proposal; None sizes them
    # from the free memory.
    num_band_preload: Optional[int] = None
    num_bands_preload_temp: Optional[int] = None
    debug_settings: GBDebugSettings = dataclasses.field(default_factory=GBDebugSettings)
    leaf_cap_settings: Optional[GBLeafCapSettings] = None
    sighet_settings: Optional[GBSigHetSettings] = None
    # ! Read by the refit move only, which is FD only; FD GB setups come at a later stage.
    search_kwargs: Optional[dict[str, Any]] = None
    # Read by the F-statistic moves (rj_fstat_mcmc, rj_fstat_mcmc_search) only; None takes their defaults.
    fstat_settings: Optional[GBFStatSettings] = None
    # Read by the rj_fstat_split_merge move only; None takes its defaults.
    fstat_split_merge_settings: Optional[GBFStatSplitMergeSettings] = None


@dataclasses.dataclass
class GBSettings(Settings):
    """Settings dataclass describing the GB branch in an Erebor-style recipe.

    The GBGPU, STFT and move settings sit in three groups, and :meth:`from_general_setup` fills
    what the run's :class:`GeneralSetup` owns. The prior comes from erebor-priors
    (``priors={"gb": mojitolight.GBPrior.prior_factory()}``); the transform and the periodic map
    follow from it, and :class:`GBSetup` sets its f0 range to the proposed bands.
    """

    # ! FD and WDM GB setups come at a later stage; their fields are commented out until then.
    # start_freq: float = 0.0001  # this might get adjusted ?
    # end_freq: float = 0.025
    # extra_buffer: int = 5
    # # Optional WDM-domain likelihood object (a ``gbgpu.gbcomps.GBWDMComputations`` instance).
    # gb_wdm_comp: typing.Any = None
    waveform: GBWaveformSettings = dataclasses.field(default_factory=GBWaveformSettings)
    stft: GBSTFTSettings = dataclasses.field(default_factory=GBSTFTSettings)
    moves: GBMoveSettings = dataclasses.field(default_factory=GBMoveSettings)
    search_interface: Optional[GBSearchInterface] = None

    @classmethod
    def from_general_setup(
        cls,
        general_setup: GeneralSetup,
        *,
        priors: dict,
        nleaves_max: int,
        nleaves_min: int = 0,
        betas: Optional[np.ndarray] = None,
        waveform: Optional[GBWaveformSettings] = None,
        stft: Optional[GBSTFTSettings] = None,
        moves: Optional[GBMoveSettings] = None,
        search_interface: Optional[GBSearchInterface] = None,
    ) -> GBSettings:
        """GB settings of a run, with what the run's GeneralSetup owns filled in.

        ``waveform``, ``stft`` and ``moves`` hold the run's choices (by default those of the
        full-band DFFR run); the time grid, the window, the STFT domain, the orbits and the backend
        come from ``general_setup``. Without ``betas`` the ladder is :class:`GBSetup`'s default.
        """
        domain_settings = general_setup.domain_settings
        on_gpu = general_setup.gpus is not None
        waveform = dataclasses.replace(
            waveform or GBWaveformSettings(),
            dt=general_setup.dt,
            T=general_setup.Tobs,
            # * The first bin of the snapped STFT grid (min_freq, not the requested value).
            start_freq_ind=int(np.round(float(domain_settings.min_freq) * general_setup.Tobs)),
            window=general_setup.window_type,
            window_alpha=general_setup.window_alpha,
            orbits=general_setup.gpu_orbits if on_gpu else general_setup.orbits,
            t0=general_setup.data_t0,
            force_backend=general_setup.force_backend,
        )
        stft = dataclasses.replace(
            stft or GBSTFTSettings(),
            domain_settings=domain_settings,
            band_alpha=general_setup.window_alpha,
        )
        return cls(
            Tobs=general_setup.Tobs,
            dt=general_setup.dt,
            log_dir=general_setup.file_store_dir,
            priors=priors,
            nleaves_max=nleaves_max,
            nleaves_min=nleaves_min,
            ntemps=general_setup.ntemps,
            betas=betas,
            waveform=waveform,
            stft=stft,
            moves=moves or GBMoveSettings(),
            search_interface=search_interface,
        )

# basic transform functions for pickling
def f_ms_to_s(x):
    """Convert frequencies from mHz to Hz (named for pickling support)."""
    return x * 1e-3


def f_s_to_ms(x):
    """Convert frequencies from Hz to mHz (inverse of :func:`f_ms_to_s`)."""
    return x * 1e3


def ten_to_the_x(x):
    """Return ``10 ** x`` (named 1-arg transform for pickling support).

    Used as a per-parameter :class:`~eryn.utils.TransformContainer` transform for
    PSD / foreground amplitudes sampled in ``log10`` space.
    """
    return 10.0 ** x


def make_gb_transform_container():
    """Build the stock GB :class:`TransformContainer` (forward + inverse).

    Sampling basis: ``A (ln), f0 (mHz), fdot, phi0, cos_iota, psi, lam,
    sin_beta``; output basis adds the filled ``fddot``. The inverse
    transforms enable ``both_inverse_transforms`` (full basis -> sampling
    basis).
    """
    input_basis = ["A", "f0", "fdot", "phi0", "cos_iota", "psi", "lam", "sin_beta"]

    gb_transform_fn_in = {
        "A": np.exp,
        "f0": f_ms_to_s,
        "cos_iota": np.arccos,
        "sin_beta": np.arcsin,
    }

    gb_inverse_transform_fn_in = {
        "A": np.log,
        "f0": f_s_to_ms,
        "cos_iota": np.cos,
        "sin_beta": np.sin,
    }

    output_basis = [
        "A",
        "f0",
        "fdot",
        "fddot",
        "phi0",
        "cos_iota",
        "psi",
        "lam",
        "sin_beta",
    ]
    gb_fill_dict = {"fddot": 0.0}

    # gb_fill_dict = {"fill_inds": np.array([3]), "ndim_full": 9, "fill_values": np.array([0.0])}

    return TransformContainer(
        input_basis=input_basis,
        output_basis=output_basis,
        parameter_transforms=gb_transform_fn_in,
        fill_dict=gb_fill_dict,
        inverse_parameter_transforms=gb_inverse_transform_fn_in,
    )


from ..hdfbackend import GBHDFBackend
from ..state import GBState


class GBSetup(Setup, GBSettings):
    """:class:`Setup` for galactic binaries in the Erebor recipe.

    Args:
        gb_settings: the GB settings: the prior and the waveform, STFT and move groups (see
            :meth:`GBSettings.from_general_setup`).
    """

    def __init__(self, gb_settings: GBSettings):

        # had a better way to do this but it stopped allowing for pickle
        Setup.__init__(self, gb_settings)

        level = logging.DEBUG
        name = "GBSetup"
        self.logger = init_logger(filename="gb_setup.log", level=level, name=name, log_dir=getattr(self, 'log_dir', None))

        self.init_setup()

    def init_sampling_info(self):
        """Take the GB prior, check its columns, and set the default ladder.

        The prior is an erebor-priors GB prior under the key ``"gb"``
        (``mojitolight.GBPrior.prior_factory()``). The transform and the periodic map come from it
        unless they are given, and its (f0, fdot) entry is rebuilt as erebor-priors' ``F0FdotPrior``
        over the bands the moves propose in. The sampling basis is ln A, f0 [mHz], fdot, phi0,
        cos iota, psi, alpha, sin delta, with ICRS sky angles; the waveform basis adds fddot as its
        fourth column.
        """
        if not isinstance(self.priors, dict) or "gb" not in self.priors:
            raise ValueError(
                "GBSetup needs priors={'gb': prior}, for example "
                "mojitolight.GBPrior.prior_factory() from erebor-priors."
            )
        gb_prior = self.priors["gb"]
        if getattr(gb_prior, "branch_name", "gb") != "gb":
            raise ValueError(
                f"the GB prior belongs to the {gb_prior.branch_name!r} branch, not 'gb'."
            )

        if self.transform is None:
            self.transform = gb_prior.get_transforms()
        if self.periodic is None:
            self.periodic = {"gb": gb_prior.get_periodic()}

        # ! The GB code reads the sampling and waveform columns by position: their order is fixed.
        input_basis = list(self.transform.input_basis)
        output_basis = list(self.transform.output_basis)
        if (
            tuple(input_basis[1:]) != GB_POSITIONAL_COLUMNS
            or len(output_basis) != len(input_basis) + 1
            or output_basis[3] != "fddot"
        ):
            raise ValueError(
                "the GB sampling basis must be an amplitude column, then "
                f"{list(GB_POSITIONAL_COLUMNS)}, and the waveform basis must add fddot as its "
                f"fourth column; got {input_basis} -> {output_basis}."
            )
        if self.ndim is None:
            self.ndim = len(input_basis)
        elif self.ndim != len(input_basis):
            raise ValueError(
                f"ndim is {self.ndim}, but the GB prior has {len(input_basis)} columns."
            )

        # * The (f0, fdot) entry becomes erebor-priors' F0FdotPrior over f0_lims, the prior's own range
        # * clipped to the bands the moves propose in, so no prior mass sits in the first and the last band.
        priors_in = dict(gb_prior.priors_in)
        if ("f0", "fdot") not in priors_in:
            raise ValueError("the GB prior needs a joint ('f0', 'fdot') entry.")
        priors_in[("f0", "fdot")] = F0FdotPrior(
            f0_min=float(self.f0_lims[0]) * 1e3, f0_max=float(self.f0_lims[1]) * 1e3, f0_units="mHz"
        )
        # ? Rebuilt as the same class, so the prior keeps its YAML dump and labels.
        self.priors = {
            "gb": type(gb_prior)(
                priors_in=priors_in, branch_name="gb", include_fddot=gb_prior.include_fddot
            )
        }
        self.logger.info(
            f"GB prior f0 range, clipped to the proposed bands: [{self.f0_lims[0] * 1e3:.7f}, "
            f"{self.f0_lims[1] * 1e3:.7f}] mHz"
        )

        if self.betas is None:
            # snrs_ladder = np.array(
            #     [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0,
            #      15.0, 20.0, 35.0, 50.0, 75.0, 125.0, 250.0, 5e2]
            # )
            if self.ntemps is None:
                ntemps_pe = 24 # len(snrs_ladder)
            else:
                ntemps_pe = self.ntemps
            # betas =  1 / snrs_ladder ** 2  # make_ladder(ndim * 10, Tmax=5e6, ntemps=ntemps_pe)
            betas = 1 / 1.2 ** np.arange(ntemps_pe)
            betas[-1] = 0.0001
            self.betas = betas

        if self.other_tempering_kwargs is None:
            self.other_tempering_kwargs = dict(adaptation_time=2, permute=True)

    # def __getattr__(self, attr: str) -> typing.Any:
    #     if hasattr(self.gb_settings, attr):
    #         return getattr(self.gb_settings, attr)

    def init_setup(self):
        """Run the band-structure, sampling-info, and state-backend init helpers."""
        self.init_band_structure()
        self.init_sampling_info()
        self.init_state_backend_info()

    def init_state_backend_info(self):
        """Default :attr:`branch_state` to :class:`GBState` and :attr:`branch_backend` to :class:`GBHDFBackend`."""
        if self.branch_state is None:
            self.branch_state = GBState

        if self.branch_backend is None:
            self.branch_backend = GBHDFBackend

    def make_source_metadata(
        self,
        waveform_model_code_link: str = "",
        prior_model_code_link: str = "",
        prior_model_config_file: str = "",
    ):
        """The GB branch's :class:`SourceMetadata`: per-band frequency ranges, GBGPU arguments.

        The metadata is written to disk, so the GBGPU arguments hold no orbits and name the CPU
        backend.
        """
        from ..postprocessing import SourceMetadata

        band_edges = np.asarray(self.band_edges)
        init_kwargs = self.waveform.initialize_kwargs
        init_kwargs.pop("orbits")
        init_kwargs["force_backend"] = "cpu"
        return SourceMetadata(
            source_type="gb",
            frequency_ranges=list(zip(band_edges[:-1], band_edges[1:])),
            waveform_model="GBGPU",
            waveform_model_code_link=waveform_model_code_link,
            prior_model_code_link=prior_model_code_link,
            prior_model_config_file=prior_model_config_file,
            waveform_model_config=dict(
                init_kwargs=init_kwargs, runtime_kwargs=self.waveform.waveform_kwargs
            ),
        )

    def prior_f0_range_hz(self) -> tuple[float, float]:
        """The f0 range of the GB prior's joint (f0, fdot) entry, in Hz."""
        if not isinstance(self.priors, dict) or "gb" not in self.priors:
            raise ValueError(
                "GBSetup needs priors={'gb': prior}, for example "
                "mojitolight.GBPrior.prior_factory() from erebor-priors."
            )
        joint_prior = self.priors["gb"].priors_in.get(("f0", "fdot"))
        if joint_prior is None:
            raise ValueError("the GB prior needs a joint ('f0', 'fdot') entry.")
        # ? F0FdotPrior holds its f0 range in its own units.
        scale = 1e-3 if joint_prior.f0_units == "mHz" else 1.0
        return float(joint_prior.f0_min) * scale, float(joint_prior.f0_max) * scale

    def init_band_structure(self):
        """Compute :attr:`band_edges` and :attr:`band_N_vals` from the STFT domain.

        STFT uses the sub-bin containment recipe from
        :mod:`lisatools.globalfit.moves.gbbandstructure`, and additionally
        sets :attr:`band_structure` so the store-window helper is reachable.
        The FD and WDM layouts are commented out until those GB setups return.
        """
        #: Only the STFT path builds one; FD and WDM leave it None so a reader
        #: gets None rather than an AttributeError.
        self.band_structure = None

        # band separation setup
        if self.waveform.oversample is None and self.Tobs < YRSID_SI / 2.0:
            self.waveform.oversample = 2
        elif self.waveform.oversample is None:
            self.waveform.oversample = 4

        assert self.waveform.oversample >= 1

        # ! FD and WDM GB setups come at a later stage. Their band code is commented out below until
        # ! then, and any domain other than STFT raises.
        # FD and WDM use a frequency-banded layout whose edges are derived from
        # ``df = 1/Tobs`` (always meaningful for either basis) and the
        # FD-oversampled per-band N. For WDM runs the per-band ``band_N_vals``
        # is unused by the WDM likelihood engine (which sizes its buffers from
        # ``WDMSettings.Nf_active`` / ``Nt_active``) but is preserved for shape
        # parity.
        #
        # # Clamp the GB band to the WDM active band when running on a WDM
        # # grid. The WDM kernel only carries layers in [ind_min_f, ind_max_f];
        # # putting band edges outside that range silently produces zero-fill
        # # behaviour, so we trim to stay inside.
        # if isinstance(self.domain_settings, WDMSettings):
        #     wdm = self.domain_settings
        #     wdm_min = float(wdm.ind_min_f * wdm.layer_df)
        #     wdm_max = float(wdm.ind_max_f * wdm.layer_df)
        #     self.start_freq = max(self.start_freq, wdm_min)
        #     self.end_freq = min(self.end_freq, wdm_max)
        #     if self.start_freq >= self.end_freq:
        #         raise ValueError(
        #             "WDM active band [{:.4e}, {:.4e}] does not overlap GB "
        #             "frequency range; widen WDMSettings.min_freq/max_freq "
        #             "or the GB start/end_freq.".format(wdm_min, wdm_max)
        #         )
        #
        # if isinstance(self.domain_settings, WDMSettings):
        #     # WDM path: one band per WDM frequency layer. Edges land on
        #     # ``k * layer_df`` boundaries so the band grid aligns with the
        #     # WDM grid; ``band_N_vals`` is unused by the WDM likelihood
        #     # engine but is preserved with one entry per band for shape
        #     # parity with the FD layout.
        #     wdm = self.domain_settings
        #     layer_df = float(wdm.layer_df)
        #     k_lo = int(np.ceil(self.start_freq / layer_df))
        #     k_hi = int(np.floor(self.end_freq / layer_df))
        #     if k_hi <= k_lo:
        #         raise ValueError(
        #             "GB frequency range [{:.4e}, {:.4e}] spans fewer than "
        #             "one WDM layer (layer_df={:.4e}).".format(
        #                 self.start_freq, self.end_freq, layer_df
        #             )
        #         )
        #     self.band_edges = np.asarray(
        #         [k * layer_df for k in range(k_lo, k_hi + 1)]
        #     )
        #     self.band_N_vals = np.asarray(
        #         [
        #             get_N(1e-30, edge, self.Tobs, oversample=self.oversample).item()
        #             for edge in self.band_edges[:-1]
        #         ]
        #     )
        # elif isinstance(self.domain_settings, FDSettings):
        #     # FD path: bands sized in multiples of ``df = 1/Tobs`` using the
        #     # FD-oversampled per-band N. Walks down from ``end_freq``
        #     # (stft_tof refinements: half-bin start, min_N stop guard, and
        #     # edge trim against out-of-bound indexing).
        #     # TODO: assign to binned f or leave general? probably better to be general
        #     band_edges_in_reverse_order = [self.end_freq]
        #     current_N = get_N(1e-30, self.end_freq, self.Tobs, oversample=self.oversample).item()
        #     min_N = get_N(1e-30, self.start_freq, self.Tobs, oversample=self.oversample).item()
        #     band_N_vals_reverse_order = [current_N]
        #
        #     current_freq = self.end_freq - self.df / 2
        #     last_freq = self.end_freq
        #     while current_freq > self.start_freq + min_N * self.df:
        #         current_freq = last_freq - (current_N * 2 + self.extra_buffer) * self.df
        #         band_edges_in_reverse_order.append(current_freq)
        #         current_N = get_N(1e-30, current_freq, self.Tobs, oversample=self.oversample).item()
        #         band_N_vals_reverse_order.append(current_N)
        #         last_freq = current_freq
        #
        #     band_edges = np.asarray(band_edges_in_reverse_order)[::-1]
        #     band_N_vals = np.asarray(band_N_vals_reverse_order)[::-1]
        #
        #     # trim edges to avoid out of bound indexing
        #     self.band_edges = band_edges[2:-1]
        #     self.band_N_vals = band_N_vals[2:-1]

        if isinstance(self.stft.domain_settings, STFTSettings):
            # STFT path: the sub-bin containment recipe, W_band = k * B_q.
            if self.stft.band_alpha is None:
                raise ValueError(
                    "STFT band structure needs stft.band_alpha (the Tukey taper fraction); "
                    "GBSettings.from_general_setup fills it from general_setup.window_alpha."
                )
            band_structure = stft_band_structure(
                self.stft.domain_settings, self.stft.band_alpha,
                **(self.stft.band_kwargs or {})
            )
            # * Only the bands the prior reaches, plus one guard band on each side (the moves never propose
            # * in the first and the last band): the GB block holds and runs no band outside the prior.
            prior_f0_min_hz, prior_f0_max_hz = self.prior_f0_range_hz()
            first_band, last_band = (
                int(band) for band in band_structure.assign(np.array([prior_f0_min_hz, prior_f0_max_hz]))
            )
            first_band = max(first_band - 1, 0)
            last_band = min(last_band + 1, band_structure.n_bands - 1)
            band_structure = dataclasses.replace(
                band_structure, edges_hz=band_structure.edges_hz[first_band:last_band + 2]
            )
            self.band_edges = band_structure.edges_hz
            self.band_N_vals = band_structure.band_N_vals(
                oversample=self.waveform.oversample
            )
            self.band_structure = band_structure
            self.logger.info(
                "STFT band structure: %d bands, alpha=%.5f, median width "
                "%.4g coherent elements (%.4g STFT pixels)",
                band_structure.n_bands, self.stft.band_alpha,
                float(np.median(band_structure.width_coherent_elements)),
                float(np.median(band_structure.width_grid_pixels)),
            )

        else:
            raise NotImplementedError(
                "GBSetup builds STFT bands only for now; FD and WDM GB setups come at a later "
                f"stage (got {type(self.stft.domain_settings).__name__})."
            )
        
        # * The moves propose only in the inner bands, between the second and the second-last edge. The f0 range
        # * is the prior's, clipped to them: the bands follow the prior, never the other way round.
        prior_f0_min_hz, prior_f0_max_hz = self.prior_f0_range_hz()
        self.f0_lims = [
            max(prior_f0_min_hz, float(self.band_edges[1])), min(prior_f0_max_hz, float(self.band_edges[-2]))
        ]
        if not self.f0_lims[0] < self.f0_lims[1]:
            raise ValueError(
                f"the GB prior's f0 range [{prior_f0_min_hz}, {prior_f0_max_hz}] Hz misses the bands the moves "
                f"propose in, [{float(self.band_edges[1])}, {float(self.band_edges[-2])}] Hz."
            )

        self.num_sub_bands = len(self.band_edges) - 1

        self.logger.info(
            f"GB f0 range of the proposed bands: {round(self.f0_lims[0], 7)} to "
            f"{round(self.f0_lims[1], 7)} Hz"
        )
        self.logger.info(f"The number of subbands is {self.num_sub_bands}")
        self.logger.info(f"Min freq of subbands is {self.band_edges.min()}")
        self.logger.info(f"Max freq of subbands is {self.band_edges.max()}")

        
def mbh_dist_trans(x):
    """Transform an MBH ``ln total mass`` parameter (named for pickling support)."""
    return x * PC_SI * 1e9  # Gpc


def gpc_to_mpc(x):
    """
    Transform from Gpc to Mpc, for distance prior.
    """
    return x * 1e3

def mT_Q(M, Q):
    """
    Transform from total mass and mass ratio m1/m2 to m1 and m2.
    """
    m2 = M / (1 + Q)
    m1 = Q * m2
    assert np.all(m1 >= m2), "m1 should be the larger mass"
    return m1, m2

def mpc_to_gpc(x):
    """
    Transform from Mpc to Gpc (inverse of :func:`gpc_to_mpc`).
    """
    return x * 1e-3


def m1_m2_to_mT_q(m1, m2):
    """Convert m1, m2 to total mass and mass ratio (inverse of ``mT_q``)."""
    return (m1 + m2, m2 / m1)


def m1_m2_to_mT_Q(m1, m2):
    """Convert m1, m2 to total mass and ``Q = m1/m2 >= 1`` (inverse of :func:`mT_Q`)."""
    return (m1 + m2, m1 / m2)


# Frame helpers kept for settings files; the stock MBH basis is direct ICRS.
# bbhx is optional (only MBH branches need it); stubs raise on first use.
try:
    from bbhx.utils.transform import LISA_to_SSB, SSB_to_LISA, mT_q
except ImportError as _bbhx_err:
    def _bbhx_missing(name, err):
        def _stub(*args, **kwargs):
            raise ImportError(f"{name} requires bbhx, which is not installed.") from err
        _stub.__name__ = name
        return _stub

    LISA_to_SSB = _bbhx_missing("LISA_to_SSB", _bbhx_err)
    SSB_to_LISA = _bbhx_missing("SSB_to_LISA", _bbhx_err)
    mT_q = _bbhx_missing("mT_q", _bbhx_err)
from eryn.moves import Move


def make_mbh_transform_container():
    """Build the stock MBH :class:`TransformContainer` (forward + inverse).

    Sampling basis (ICRS — the 2026-06 run frame, matching
    :func:`lisatools.globalfit.recipe.mbh_catalogue_to_sampling_basis`):
    ``logM, Q (= m1/m2 >= 1), s1z, s2z, dist (Gpc), phi_ref, cos_iota,
    psi (ICRS), alpha (RA), sin_delta, t_plunge (SSB)``. The output basis
    holds ``m1, m2 (via mT_Q), dist (Mpc), iota, delta`` with sky /
    polarization passed through unchanged in ICRS — no frame
    conversion; the orbits must be loaded with ``frame='icrs'`` so the
    response consumes ``(alpha, delta)`` directly. The inverse
    transforms enable ``both_inverse_transforms`` (full basis ->
    sampling basis).
    """
    input_basis = [
        "logM",
        "Q",
        "s1z",
        "s2z",
        "dist",
        "phi_ref",
        "cos_iota",
        "psi",
        "alpha",
        "sin_delta",
        "t_plunge",
    ]

    output_basis = list(input_basis)

    mbh_transform_fn_in = {
        "logM": np.exp,
        "dist": gpc_to_mpc,
        "cos_iota": np.arccos,
        "sin_delta": np.arcsin,
        ("logM", "Q"): mT_Q,
    }

    # keyed identically to mbh_transform_fn_in; same insertion order so
    # the multi-parameter inverses unwind in reverse of the forward order
    mbh_inverse_transform_fn_in = {
        "logM": np.log,
        "dist": mpc_to_gpc,
        "cos_iota": np.cos,
        "sin_delta": np.sin,
        ("logM", "Q"): m1_m2_to_mT_Q,
    }

    # for transforms
    # mbh_fill_dict = {"f_ref": 0.0}
    mbh_fill_dict = {}

    return TransformContainer(
        input_basis=input_basis,
        output_basis=output_basis,
        parameter_transforms=mbh_transform_fn_in,
        fill_dict=mbh_fill_dict,
        inverse_parameter_transforms=mbh_inverse_transform_fn_in,
    )

from ..hdfbackend import MBHHDFBackend
from ..state import MBHState




@dataclasses.dataclass
class MBHSettings(Settings):
    """Settings dataclass describing the MBH branch in an Erebor-style recipe."""

    waveform_kwargs: Optional[dict] = None
    betas: Optional[np.ndarray] = None
    inner_moves: Optional[typing.List[Move]] = None
    num_prop_repeats: Optional[int] = 200
    mbh_search_file_key: Optional[str] = "_mbh_search_tmp_file"
    injection: Optional[np.ndarray] = None
    search_interface: Optional[MBHBSearchInterface] = None

class MBHSetup(Setup):
    """:class:`Setup` for massive black-hole binaries in the Erebor recipe.

    Args:
        mbh_settings: Settings dataclass with MBH waveform / prior config.
    """

    def __init__(self, mbh_settings: MBHSettings):

        # had a better way to do this but it stopped allowing for pickle
        super().__init__(mbh_settings)

        level = logging.DEBUG
        name = "MBHSetup"
        self.logger = init_logger(filename="mbh_setup.log", level=level, name=name, log_dir=getattr(self, 'log_dir', None))

        self.init_setup()

    def init_sampling_info(self):
        """Build the MBH :class:`TransformContainer`, prior, periodicity, and waveform kwargs."""

        # ICRS sampling basis (2026-06 run-frame directive): sky and
        # polarization are sampled directly in ICRS (alpha = RA,
        # sin_delta, psi = ICRS polarization), matching
        # ``mbh_catalogue_to_sampling_basis``. No LISA->SSB->ICRS chain;
        # the orbits must be loaded with ``frame='icrs'`` so the response
        # consumes ``(alpha, delta)`` directly.
        input_basis = [
            "logM",
            "Q",
            "s1z",
            "s2z",
            "dist",
            "phi_ref",
            "cos_iota",
            "psi",
            "alpha",
            "sin_delta",
            "t_plunge",
        ]

        if self.transform is None:
            # Stock forward + inverse container (direct ICRS; same basis as
            # ``input_basis`` above). The inverse transforms enable
            # ``both_inverse_transforms`` (full basis -> sampling basis)
            # for injection/diagnostic round trips.
            self.transform = make_mbh_transform_container()

        if self.periodic is None:
            self.periodic = {"mbh": {"phi_ref": 2 * np.pi, "alpha": 2 * np.pi, "psi": np.pi}}

        self.logger.debug("Decide how to treat fdot prior")
        if self.priors is None:
            priors_mbh = {
                "logM": uniform_dist(np.log(1e5), np.log(1e8)),
                "Q": log_uniform(1., 10.),
                "s1z": uniform_dist(-0.99999999, +0.99999999),
                "s2z": uniform_dist(-0.99999999, +0.99999999),
                "dist": uniform_dist(1, 150.0), # uniform_dist(0.01, 1000.0),
                "phi_ref": uniform_dist(0.0, 2 * np.pi),
                "cos_iota": uniform_dist(-1.0 + 1e-6, 1.0 - 1e-6),
                "psi": uniform_dist(0.0, np.pi), #is this right?
                "alpha": uniform_dist(0.0, 2 * np.pi),
                "sin_delta": uniform_dist(-1.0 + 1e-6, 1.0 - 1e-6),
                "t_plunge": uniform_dist(0.0, self.Tobs + 3600.0),
            }

            self.priors = {"mbh": ProbDistContainer(priors_mbh)}

        if self.betas is None:
            snrs_ladder = np.array(
                [
                    1.0,
                    1.5,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    7.5,
                    10.0,
                    15.0,
                    20.0,
                    35.0,
                    50.0,
                    75.0,
                    125.0,
                    250.0,
                    5e2,
                ]
            )
            ntemps_pe = 24  # len(snrs_ladder)
            # betas =  1 / snrs_ladder ** 2  # make_ladder(ndim * 10, Tmax=5e6, ntemps=ntemps_pe)
            betas = 1 / 1.2 ** np.arange(ntemps_pe)
            betas[-1] = 0.0001
            self.betas = betas

        # TODO: maybe combine this into Setup
        if self.other_tempering_kwargs is None:
            self.other_tempering_kwargs = dict(permute=False)

        if "permute" not in self.other_tempering_kwargs:
            self.other_tempering_kwargs["permute"] = False

        assert not self.other_tempering_kwargs["permute"]

        if self.initialize_kwargs is None:
            self.initialize_kwargs = {}

        if self.waveform_kwargs is None:
            self.logger.warning(
                "No waveform kwargs provided for MBHSetup, using defaults. These are the legacy BBHx settings"
            )
            self.waveform_kwargs = dict(
                modes=[(2, 2)],
                length=1024,
            )

        if self.inner_moves is None:
            from eryn.moves import StretchMove

            # TODO(post-merge): re-enable SkyMove hops once the move
            # supports the ICRS sampling basis — the stock MBH basis is
            # now direct ICRS (alpha, sin_delta, psi ICRS) while the
            # existing SkyMove geometry assumes SSB ecliptic. The index
            # map is unchanged (cos_iota=6, psi=7, alpha=8, sin_delta=9):
            #
            #   from lisatools.sampling.moves.skymodehop import SkyMove
            #   angles_map = dict(cosinc=6, psi=7, lam=8, sinbeta=9)
            #   ... (SkyMove(ind_map=angles_map, which=...), w) ...
            self.inner_moves = [
                (StretchMove(), 1.0),
            ]

    def init_setup(self):
        """Run sampling-info and state-backend initialization."""
        self.init_sampling_info()
        self.init_state_backend_info()

    def init_state_backend_info(self):
        """Default the MBH state and backend to :class:`MBHState` / :class:`MBHHDFBackend`."""
        if self.branch_state is None:
            self.branch_state = MBHState

        if self.branch_backend is None:
            self.branch_backend = MBHHDFBackend


def make_emri_transform_container(fill_values):
    """Build the stock EMRI :class:`TransformContainer` (forward + inverse).

    Sampling basis (12 params): ``logm1, m2, a, p0, e0, dist, qS (cos),
    phiS, qK (cos), phiK, Phi_phi0, Phi_r0``; the output basis adds the
    filled ``xI0`` and ``Phi_theta0``. The inverse transforms enable
    ``both_inverse_transforms`` (full basis -> sampling basis).

    Args:
        fill_values: Length-2 sequence with the fill values for ``xI0``
            (inclination) and ``Phi_theta0``.
    """
    input_basis = [
        "logm1",
        "m2",
        "a",
        "p0",
        "e0",
        "dist",
        "cosqS",
        "phiS",
        "cosqK",
        "phiK",
        "Phi_phi0",
        "Phi_r0",
    ]

    output_basis = [
        "m1",
        "m2",
        "a",
        "p0",
        "e0",
        "xI0",
        "dist",
        "qS",
        "phiS",
        "qK",
        "phiK",
        "Phi_phi0",
        "Phi_theta0",
        "Phi_r0",
    ]

    # for transforms

    emri_fill_dict = {
        "xI0": fill_values[0],  # inclination
        "Phi_theta0": fill_values[1],  # Phi_theta
    }

    emri_transform_fn_in = {
        output_basis[0]: np.exp,  # M
        output_basis[7]: np.arccos,  # qS
        output_basis[9]: np.arccos,  # qK
    }

    emri_inverse_transform_fn_in = {
        output_basis[0]: np.log,  # M
        output_basis[7]: np.cos,  # qS
        output_basis[9]: np.cos,  # qK
    }
    key_map = {
        "logm1": "m1",
        "cosqK": "qK",
        "cosqS": "qS",
    }

    return TransformContainer(
        input_basis=input_basis,
        output_basis=output_basis,
        parameter_transforms=emri_transform_fn_in,
        fill_dict=emri_fill_dict,
        inverse_parameter_transforms=emri_inverse_transform_fn_in,
        key_map=key_map
    )


from ..hdfbackend import EMRIHDFBackend
from ..state import EMRIState


@dataclasses.dataclass
class EMRISettings(Settings):
    """Settings dataclass describing the EMRI branch in an Erebor-style recipe.

    Holds parameter ranges for the intrinsic EMRI parameters (primary mass,
    secondary mass, spin, semi-latus rectum, eccentricity), the waveform
    configuration, and search/sampling helpers consumed by :class:`EMRISetup`.

    # TODO/DOCS: confirm the role of ``info_matrix_gen`` and ``fill_values``
    # in coordinating EMRI starts with the Fisher / search pipeline.
    """

    logm1_lims: typing.List[float] = dataclasses.field(default_factory=list)
    m2_lims: typing.List[float] = dataclasses.field(default_factory=list)
    a_lims: typing.List[float] = dataclasses.field(default_factory=list)
    p0_lims: typing.List[float] = dataclasses.field(default_factory=list)
    e0_lims: typing.List[float] = dataclasses.field(default_factory=list)
    waveform_kwargs: Optional[dict] = None
    injection: Optional[np.ndarray] = None  # AS here only for the starting state
    info_matrix_gen: Optional[Any] = None  # todo change name to info matrix or smth
    fill_values: np.ndarray = dataclasses.field(default_factory=lambda: np.array([1.0, 0.0]))
    betas: Optional[np.ndarray] = None
    inner_moves: Optional[typing.List[Move]] = None
    num_prop_repeats: Optional[int] = 10
    emri_search_file_key: Optional[str] = "_emri_search_tmp_file"
    search_interface: Optional[EMRISearchInterface] = None

class EMRISetup(Setup):
    """:class:`Setup` for extreme mass-ratio inspirals in the Erebor recipe.

    Args:
        emri_settings: Settings dataclass with EMRI parameter ranges,
            waveform configuration, and search helpers.
    """

    def __init__(self, emri_settings: EMRISettings):

        # had a better way to do this but it stopped allowing for pickle
        super().__init__(emri_settings)

        level = logging.DEBUG
        name = "EMRISetup"
        self.logger = init_logger(filename="emri_setup.log", level=level, name=name, log_dir=getattr(self, 'log_dir', None))

        self.init_setup()

    def init_sampling_info(self):
        """Build the EMRI :class:`TransformContainer`, prior, periodicity, and tempering ladder."""
        input_basis = [
            "logm1",
            "m2",
            "a",
            "p0",
            "e0",
            "dist",
            "qS",
            "phiS",
            "qK",
            "phiK",
            "Phi_phi0",
            "Phi_r0",
        ]

        if self.transform is None:
            self.transform = make_emri_transform_container(self.fill_values)

        if self.periodic is None:
            self.periodic = {
                "emri": {
                    "phiS": 2 * np.pi,
                    "phiK": 2 * np.pi,
                    "Phi_phi0": 2 * np.pi,
                    "Phi_r0": 2 * np.pi,
                }
            }

        if self.priors is None:
            self.setup_priors(input_basis)

        if self.betas is None:
            snrs_ladder = np.array(
                [
                    1.0,
                    1.5,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    7.5,
                    10.0,
                    15.0,
                    20.0,
                    35.0,
                    50.0,
                    75.0,
                    125.0,
                    250.0,
                    5e2,
                ]
            )
            ntemps_pe = 24  # len(snrs_ladder)
            # betas =  1 / snrs_ladder ** 2  # make_ladder(ndim * 10, Tmax=5e6, ntemps=ntemps_pe)
            betas = 1 / 1.2 ** np.arange(ntemps_pe)
            # betas[-1] = 0.0001
            self.betas = betas

        self.logger.info(f"Using betas: {self.betas} in EMRI branch")

        # TODO: maybe combine this into Setup
        if self.other_tempering_kwargs is None:
            self.other_tempering_kwargs = dict(permute=False)

        if "permute" not in self.other_tempering_kwargs:
            self.other_tempering_kwargs["permute"] = False

        assert not self.other_tempering_kwargs["permute"]

        if self.initialize_kwargs is None:
            self.initialize_kwargs = {}

        if self.inner_moves is None:
            from eryn.moves import StretchMove

            self.inner_moves = [(StretchMove(), 1.0)]

    def setup_priors(self, input_basis):
        """Build the EMRI prior dictionary, overriding intrinsic ranges from settings.

        Starts from a default uniform prior on each input-basis parameter and
        overrides the intrinsic parameter ranges (``logm1``, ``m2``, ``a``,
        ``p0``, ``e0``) with the corresponding ``*_lims`` attributes when
        those have been set.

        Args:
            input_basis: Ordered list of EMRI parameter names matching the
                sampler's input basis.
        """

        priors_emri = {
            input_basis[0]: uniform_dist(np.log(5e5), np.log(5e6)),  # log m1
            input_basis[1]: uniform_dist(1, 100),  # m2
            input_basis[2]: uniform_dist(0.01, 0.999),  # a
            input_basis[3]: uniform_dist(5.0, 100.0),  # p0
            input_basis[4]: uniform_dist(0.001, 0.8),  # e0
            input_basis[5]: uniform_dist(0.01, 100.0),  # dist in Gpc
            input_basis[6]: uniform_dist(-0.99999, 0.99999),  # qS
            input_basis[7]: uniform_dist(0.0, 2 * np.pi),  # phiS
            input_basis[8]: uniform_dist(-0.99999, 0.99999),  # qK
            input_basis[9]: uniform_dist(0.0, 2 * np.pi),  # phiK
            input_basis[10]: uniform_dist(0.0, 2 * np.pi),  # Phi_phi0
            input_basis[11]: uniform_dist(0.0, 2 * np.pi),  # Phi_r0
        }

        limits = ["logm1_lims", "m2_lims", "a_lims", "p0_lims", "e0_lims"]
        for i, lims in enumerate(limits):
            if getattr(self, lims) is not None:
                self.logger.info(
                    f"Setting prior for parameter {i} using limits {getattr(self, lims)}"
                )
                priors_emri[input_basis[i]] = uniform_dist(*getattr(self, lims))

        self.priors = {"emri": ProbDistContainer(priors_emri)}

    def init_setup(self):
        """Run sampling-info and state-backend initialization."""
        self.init_sampling_info()
        self.init_state_backend_info()

    def init_state_backend_info(self):
        """Default the EMRI state and backend to :class:`EMRIState` / :class:`EMRIHDFBackend`."""
        if self.branch_state is None:
            self.branch_state = EMRIState

        if self.branch_backend is None:
            self.branch_backend = EMRIHDFBackend


def cosqS_to_beta(x):
    """Transform ``cosqS`` to ecliptic latitude ``beta = pi/2 - arccos(cosqS)``."""
    return np.pi / 2.0 - np.arccos(x)


def beta_to_cosqS(x):
    """Transform ecliptic latitude ``beta`` to ``cosqS = sin(beta)`` (inverse of :func:`cosqS_to_beta`)."""
    return np.sin(x)


def make_sobbh_transform_container():
    """Build the stock SOBBH :class:`TransformContainer` (forward + inverse).

    Sampling basis (11 params): ``logm1, logm2, s1, s2, dist, cosinc,
    f_low, phiS, cosqS, psi, phi0``; output basis: ``m1, m2, s1, s2,
    dist, inc, f_low, lam, beta, psi, phi0``. The inverse transforms
    enable ``both_inverse_transforms`` (full basis -> sampling basis).
    """
    input_basis = [
        "logm1",
        "logm2",
        "s1",
        "s2",
        "dist",
        "cosinc",
        "f_low",
        "phiS",
        "cosqS",
        "psi",
        "phi0",
    ]

    output_basis = [
        "m1",
        "m2",
        "s1",
        "s2",
        "dist",
        "inc",
        "f_low",
        "lam",
        "beta",
        "psi",
        "phi0",
    ]

    sobbh_transform_fn_in = {
        output_basis[0]: np.exp,  # m1
        output_basis[1]: np.exp,  # m2
        output_basis[5]: np.arccos,  # inc <- cosinc
        output_basis[8]: cosqS_to_beta,  # beta <- cosqS
    }

    sobbh_inverse_transform_fn_in = {
        output_basis[0]: np.log,  # logm1 <- m1
        output_basis[1]: np.log,  # logm2 <- m2
        output_basis[5]: np.cos,  # cosinc <- inc
        output_basis[8]: beta_to_cosqS,  # cosqS <- beta
    }

    # Every input-basis name not already present in output_basis
    # needs a key_map entry pointing to its renamed counterpart.
    return TransformContainer(
        input_basis=input_basis,
        output_basis=output_basis,
        parameter_transforms=sobbh_transform_fn_in,
        key_map={
            "logm1": "m1",
            "logm2": "m2",
            "cosinc": "inc",
            "cosqS": "beta",
            "phiS": "lam",
        },
        inverse_parameter_transforms=sobbh_inverse_transform_fn_in,
    )


from ..hdfbackend import SOBBHHDFBackend
from ..state import SOBBHState


@dataclasses.dataclass
class SOBBHSettings(Settings):
    """Settings dataclass describing the SOBBH branch in an Erebor-style recipe.

    Mirrors :class:`EMRISettings` for stellar-origin BBHs: parameter
    ranges for the intrinsic SOBBH parameters (component masses, aligned
    spins, distance, starting GW frequency) plus the waveform / sampling
    knobs consumed by :class:`SOBBHSetup`.

    The sampling basis is 11 params (no ``fill_values`` by default — the
    SOBBH waveform consumes everything sampled).
    """

    logm1_lims: typing.List[float] = dataclasses.field(default_factory=list)
    logm2_lims: typing.List[float] = dataclasses.field(default_factory=list)
    s1_lims: typing.List[float] = dataclasses.field(default_factory=list)
    s2_lims: typing.List[float] = dataclasses.field(default_factory=list)
    f_low_lims: typing.List[float] = dataclasses.field(default_factory=list)
    waveform_kwargs: Optional[dict] = None
    injection: Optional[np.ndarray] = None
    info_matrix_gen: Optional[Any] = None
    # No fill_values used by default; keep the dataclass field for parity
    # with EMRISettings so settings files can opt-in without breakage.
    fill_values: np.ndarray = dataclasses.field(default_factory=lambda: np.array([]))
    betas: Optional[np.ndarray] = None
    inner_moves: Optional[typing.List[Move]] = None
    num_prop_repeats: Optional[int] = 10
    sobbh_search_file_key: Optional[str] = "_sobbh_search_tmp_file"


class SOBBHSetup(Setup):
    """:class:`Setup` for stellar-origin BBHs in the Erebor recipe.

    Args:
        sobbh_settings: Settings dataclass with SOBBH parameter ranges,
            waveform configuration, and search helpers.
    """

    def __init__(self, sobbh_settings: SOBBHSettings):
        super().__init__(sobbh_settings)

        level = logging.DEBUG
        name = "SOBBHSetup"
        self.logger = init_logger(
            filename="sobbh_setup.log",
            level=level,
            name=name,
            log_dir=getattr(self, "log_dir", None),
        )

        self.init_setup()

    def init_sampling_info(self):
        """Build the SOBBH :class:`TransformContainer`, prior, periodicity, and tempering ladder.

        Sampling basis (11 params): ``logm1, logm2, s1, s2, dist, cosinc,
        f_low, phiS, cosqS, psi, phi0``. ``M`` (==logm1.exp) and ``m2``
        are exponentiated; ``cosinc, cosqS`` are inverted to ``inc, qS``;
        ``phiS`` is mapped to the ``lambda`` argument that ResponseWrapper
        uses, ``qS -> beta = pi/2 - qS``.
        """
        input_basis = [
            "logm1",
            "logm2",
            "s1",
            "s2",
            "dist",
            "cosinc",
            "f_low",
            "phiS",
            "cosqS",
            "psi",
            "phi0",
        ]

        if self.transform is None:
            self.transform = make_sobbh_transform_container()

        if self.periodic is None:
            self.periodic = {
                "sobbh": {
                    "phiS": 2 * np.pi,
                    "psi": np.pi,
                    "phi0": 2 * np.pi,
                }
            }

        self.setup_priors(input_basis)

        if self.betas is None:
            ntemps_pe = 24
            betas = 1 / 1.2 ** np.arange(ntemps_pe)
            self.betas = betas

        self.logger.info(f"Using betas: {self.betas} in SOBBH branch")

        if self.other_tempering_kwargs is None:
            self.other_tempering_kwargs = dict(permute=False)

        if "permute" not in self.other_tempering_kwargs:
            self.other_tempering_kwargs["permute"] = False

        assert not self.other_tempering_kwargs["permute"]

        if self.initialize_kwargs is None:
            self.initialize_kwargs = {}

        if self.inner_moves is None:
            from eryn.moves import StretchMove

            self.inner_moves = [(StretchMove(), 1.0)]

    def setup_priors(self, input_basis):
        """Build the SOBBH prior dictionary, overriding intrinsic ranges from settings."""
        priors_sobbh = {
            input_basis[0]: uniform_dist(np.log(2.0), np.log(100.0)),     # logm1 (M_sun)
            input_basis[1]: uniform_dist(np.log(2.0), np.log(100.0)),     # logm2
            input_basis[2]: uniform_dist(-0.99, 0.99),                    # s1z
            input_basis[3]: uniform_dist(-0.99, 0.99),                    # s2z
            input_basis[4]: uniform_dist(0.01, 10.0),                     # dist (Gpc)
            input_basis[5]: uniform_dist(-1.0, 1.0),                      # cosinc
            input_basis[6]: uniform_dist(1.0e-3, 1.0e-1),                 # f_low (Hz)
            input_basis[7]: uniform_dist(0.0, 2 * np.pi),                 # phiS
            input_basis[8]: uniform_dist(-1.0, 1.0),                      # cosqS
            input_basis[9]: uniform_dist(0.0, np.pi),                     # psi
            input_basis[10]: uniform_dist(0.0, 2 * np.pi),                # phi0
        }

        limits = ["logm1_lims", "logm2_lims", "s1_lims", "s2_lims", "f_low_lims"]
        target_idx = {"logm1_lims": 0, "logm2_lims": 1, "s1_lims": 2, "s2_lims": 3, "f_low_lims": 6}
        for lims in limits:
            cfg = getattr(self, lims)
            if cfg is not None and len(cfg) == 2:
                i = target_idx[lims]
                self.logger.info(
                    f"Setting prior for parameter {i} ({input_basis[i]}) using limits {cfg}"
                )
                priors_sobbh[input_basis[i]] = uniform_dist(*cfg)

        self.priors = {"sobbh": ProbDistContainer(priors_sobbh)}

    def init_setup(self):
        self.init_sampling_info()
        self.init_state_backend_info()

    def init_state_backend_info(self):
        """Default the SOBBH state and backend to :class:`SOBBHState` / :class:`SOBBHHDFBackend`."""
        if self.branch_state is None:
            self.branch_state = SOBBHState

        if self.branch_backend is None:
            self.branch_backend = SOBBHHDFBackend


@dataclasses.dataclass
class SOBBHTFSettings(Settings):
    """Settings dataclass for the gwtf (time-frequency) SOBBH branch.

    The prior, transform, periodicity and ``ndim`` come from erebor-priors
    ``mojitolight.SOBHBPrior``: sampling basis ``Mc, eta, s1, s2, d_L [Gpc], cos_iota,
    f0 [Hz, at data start], phi_0 [orbital phase at data start], psi, alpha, sin_delta``, transformed to the
    ``GWTF_generator`` / ``SOBBHTaylorT3TDIWaveform`` basis ``m1, m2, s1, s2, dist [Gpc],
    inc, f0, phi_0, psi, ra, dec`` (sky/polarisation in the frame of the orbits; ``phi_0`` passes
    through, the waveforms take it with ``phase_at_coalescence=False``).

    Injected ``priors`` (e.g. from an l2d YAML block) take precedence and then need
    ``transform``. Otherwise ``SOBHBPrior.prior_factory`` is built from ``f0_lims`` and
    ``Mc_lims`` when given (its defaults otherwise).

    ``branch_name`` is the key used for ``priors`` / ``periodic``; it must
    match the branch name the recipe and ``GlobalFitSettings.source_info`` use.
    """

    branch_name: str = "sobbh"
    Mc_lims: typing.List[float] = dataclasses.field(default_factory=list)
    f0_lims: typing.List[float] = dataclasses.field(default_factory=list)
    # GWTF_generator.get_signals_for_residuals takes no kwargs, and
    # subtract_initial_signal splats these into it -> must stay a (empty) dict.
    waveform_kwargs: dict = dataclasses.field(default_factory=dict)
    injection: Optional[np.ndarray] = None
    fill_values: np.ndarray = dataclasses.field(default_factory=lambda: np.array([]))
    betas: Optional[np.ndarray] = None
    inner_moves: Optional[typing.List[Move]] = None
    num_prop_repeats: Optional[int] = 10
    search_interface: Optional[SOBHBSearchInterface] = None


class SOBBHTFSetup(Setup):
    """:class:`Setup` for the gwtf (time-frequency) SOBBH branch.

    Args:
        sobbhtf_settings: Settings dataclass with gwtf SOBBH prior ranges
            and sampling knobs.
    """

    def __init__(self, sobbhtf_settings: SOBBHTFSettings):

        super().__init__(sobbhtf_settings)

        level = logging.DEBUG
        name = "SOBBHTFSetup"
        self.logger = init_logger(
            filename="sobbhtf_setup.log",
            level=level,
            name=name,
            log_dir=getattr(self, "log_dir", None),
        )

        self.init_setup()

    def init_sampling_info(self):
        """Default the prior, transform, periodicity and ``ndim`` from ``SOBHBPrior``.

        Each is only filled when not given. ``betas`` is left as given: ``None`` lets the
        recipe build a ladder matching the general ``ntemps`` (``make_ladder``).
        """
        if self.waveform_kwargs is None:
            self.waveform_kwargs = {}

        if self.priors is None:
            prior = self.make_default_prior()
            self.priors = {self.branch_name: prior}
            if self.transform is None:
                self.transform = prior.get_transforms()
            if self.periodic is None:
                self.periodic = {self.branch_name: prior.get_periodic()}
        else:
            self.logger.info("Using custom priors for the SOBBHTF branch")
            if self.transform is None:
                raise ValueError("SOBBHTFSettings: custom priors need a matching transform.")

        if self.ndim is None:
            self.ndim = self.priors[self.branch_name].ndim

        if self.betas is not None:
            self.logger.info(f"Using betas: {self.betas} in SOBBHTF branch")

        if self.other_tempering_kwargs is None:
            self.other_tempering_kwargs = dict(permute=False)

        if "permute" not in self.other_tempering_kwargs:
            self.other_tempering_kwargs["permute"] = False

        assert not self.other_tempering_kwargs["permute"]

        if self.initialize_kwargs is None:
            self.initialize_kwargs = {}

        if self.inner_moves is None:
            from eryn.moves import StretchMove

            self.inner_moves = [(StretchMove(), 1.0)]

    def make_default_prior(self):
        """``SOBHBPrior.prior_factory`` with ``f0_lims`` / ``Mc_lims`` when given."""
        from ereborpriors.stock.mojitolight import SOBHBPrior

        kwargs = {}
        for lims, (lo, hi) in ((self.f0_lims, ("f0_min", "f0_max")), (self.Mc_lims, ("chirp_mass_min", "chirp_mass_max"))):
            if lims is not None and len(lims) == 2:
                kwargs[lo], kwargs[hi] = (float(v) for v in lims)
        self.logger.info(f"Default SOBHBPrior with {kwargs or 'its default limits'}")
        prior = SOBHBPrior.prior_factory(**kwargs)
        if prior.branch_name != self.branch_name:
            raise ValueError(f"SOBHBPrior branch {prior.branch_name!r} != settings branch {self.branch_name!r}.")
        return prior

    def init_setup(self):
        self.init_sampling_info()
        self.init_state_backend_info()

    def init_state_backend_info(self):
        """Default the state and backend to :class:`SOBBHState` / :class:`SOBBHHDFBackend`."""
        if self.branch_state is None:
            self.branch_state = SOBBHState

        if self.branch_backend is None:
            self.branch_backend = SOBBHHDFBackend


from lisatools.detector import EqualArmlengthOrbits


@dataclasses.dataclass
class PSDSettings(Settings):
    """Settings dataclass describing the PSD branch in an Erebor-style recipe.

    Configures the PSD model parameters that are sampled jointly with the
    other source branches (e.g. instrumental noise levels) and feeds into
    :class:`PSDSetup`.

    # TODO/DOCS: confirm coordination between ``nknots`` and ``ndim`` for
    # spline-style PSD parameterizations vs. the default 4-parameter setup.
    """

    psd_kwargs: typing.Dict = dataclasses.field(default_factory=dict)
    nleaves_max: int = 1
    nleaves_min: int = 1
    ndim: int = 4
    transform: Optional[TransformContainer] = None
    injection: Optional[np.ndarray] = None 
    nknots: Optional[int] = None
    num_prop_repeats: int = 50
    search_interface: Optional[NoiseSearchInterface] = None

class PSDSetup(Setup):
    """:class:`Setup` for the instrumental PSD branch in the Erebor recipe.

    Args:
        psd_settings: Settings dataclass with PSD kwargs, prior, and
            tempering configuration.
    """

    def __init__(self, psd_settings: PSDSettings):

        # had a better way to do this but it stopped allowing for pickle
        super().__init__(psd_settings)

        level = logging.DEBUG
        name = "PSDSetup"
        self.logger = init_logger(filename="psd_setup.log", level=level, name=name, log_dir=getattr(self, 'log_dir', None))

        self.init_setup()

    def init_sampling_info(self):
        """Build the PSD prior, tempering ladder, and default ``psd_kwargs``."""
        if self.psd_kwargs is None:
            self.psd_kwargs = dict(sens_fn="A1TDISens")

        if self.initialize_kwargs is None:
            self.initialize_kwargs = {}

        if self.priors is None:
            # TODO: change to scaled linear in amplitude!?!
            priors_psd = {
                r"$S_{\rm oms}$": uniform_dist(6.0e-12, 20.0e-11),  # Soms_d
                r"$S_{\rm tm}$": uniform_dist(1.0e-15, 20.0e-14),  # Sa_a
                # 2: uniform_dist(6.0e-12, 20.0e-12),  # Soms_d
                # 3: uniform_dist(1.0e-15, 20.0e-15),  # Sa_a
            }

            # TODO: orbits check against sangria/sangria_hm
            self.priors = {"psd": ProbDistContainer(priors_psd)}

        else:
            self.logger.info("Using custom priors for PSD branch")

        if self.betas is None:
            # TODO: fix this to be generic
            ntemps_pe = 24  # len(snrs_ladder)
            # betas =  1 / snrs_ladder ** 2  #

            betas = make_ladder(self.ndim * 10, Tmax=np.inf, ntemps=ntemps_pe)
            self.betas = betas

        if self.other_tempering_kwargs is None:
            self.other_tempering_kwargs = dict(permute=False)

        if "permute" not in self.other_tempering_kwargs:
            self.other_tempering_kwargs["permute"] = False

        assert not self.other_tempering_kwargs["permute"]

    def init_setup(self):
        """Run sampling-info initialization for the PSD branch."""
        self.init_sampling_info()


@dataclasses.dataclass
class GalForSettings(Settings):
    """Settings dataclass describing the galactic-foreground branch in an Erebor-style recipe.

    Holds the foreground-model kwargs and the leaf / dimension counts used
    by :class:`GalForSetup` to build the prior on the stochastic galactic
    foreground (amplitude, knee, slopes, etc.).
    """

    galfor_kwargs: typing.Dict = dataclasses.field(default_factory=dict)
    transform: Optional[TransformContainer] = None
    nleaves_max: int = 1
    nleaves_min: int = 1
    ndim: int = 5
    search_interface: Optional[GalacticForegroundSearchInterface] = None


class GalForSetup(Setup):
    """:class:`Setup` for the galactic-foreground branch in the Erebor recipe.

    Args:
        galfor_settings: Settings dataclass with foreground kwargs, prior,
            and tempering configuration.
    """

    def __init__(self, galfor_settings: GalForSettings):

        # had a better way to do this but it stopped allowing for pickle
        super().__init__(galfor_settings)

        level = logging.DEBUG
        name = "GalForSetup"
        self.logger = init_logger(filename="galfor_setup.log", level=level, name=name, log_dir=getattr(self, 'log_dir', None))

        self.init_setup()

    def init_sampling_info(self):
        """Build the galactic-foreground prior, tempering kwargs, and default ``galfor_kwargs``."""
        if self.galfor_kwargs is None:
            self.galfor_kwargs = dict(sens_fn="A1TDISens")

        if self.initialize_kwargs is None:
            self.initialize_kwargs = {}

        if self.priors is None:
            # TODO: change to scaled linear in amplitude!?!
            priors_galfor = {
                0: uniform_dist(1e-45, 2e-43),  # amp
                1: uniform_dist(1e-4, 5e-2),  # knee
                2: uniform_dist(0.01, 3.0),  # alpha
                3: uniform_dist(1e0, 1e7),  # Slope1
                4: uniform_dist(5e1, 8e3),  # Slope2
            }

            # TODO: orbits check against sangria/sangria_hm
            self.priors = {"galfor": ProbDistContainer(priors_galfor)}

        # if self.betas is None:
        #     # TODO: fix this to be generic
        #     ntemps_pe = 24  # len(snrs_ladder)
        #     # betas =  1 / snrs_ladder ** 2  #

        #     betas = make_ladder(self.ndim * 10, Tmax=np.inf, ntemps=ntemps_pe)
        #     self.betas = betas

        if self.other_tempering_kwargs is None:
            self.other_tempering_kwargs = dict(permute=False)

        if "permute" not in self.other_tempering_kwargs:
            self.other_tempering_kwargs["permute"] = False

        assert not self.other_tempering_kwargs["permute"]

    def init_setup(self):
        """Run sampling-info initialization for the galactic-foreground branch."""
        self.init_sampling_info()


def get_galfor_erebor_settings(general_set: GeneralSetup) -> GalForSetup:
    """Construct the default :class:`GalForSetup` for an Erebor run.

    Builds a :class:`GalForSettings` from the run-wide ``Tobs`` / ``dt``
    carried on ``general_set`` and wraps it in a :class:`GalForSetup`.

    # TODO/DOCS: the local ``Tobs = YRSID_SI`` and ``dt = 10.0`` are dead
    # code (overridden by ``general_set``); confirm whether they should be
    # used as fallbacks.

    Args:
        general_set: Run-wide :class:`GeneralSetup` providing ``Tobs`` and
            ``dt``.

    Returns:
        Configured :class:`GalForSetup` ready for use in the Erebor
        pipeline.
    """
    from lisatools.detector import EqualArmlengthOrbits
    from lisatools.utils.constants import YRSID_SI

    Tobs = YRSID_SI
    dt = 10.0

    galfor_settings = GalForSettings(
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        initialize_kwargs={},
    )

    return GalForSetup(galfor_settings)


if __name__ == "__main__":
    # general_set = get_general_erebor_settings()
    # gb_set = get_gb_erebor_settings(general_set)
    # mbh_set = get_mbh_erebor_settings(general_set)
    # psd_set = get_psd_erebor_settings(general_set)
    # galfor_set = get_galfor_erebor_settings(general_set)
    breakpoint()

    # # mcmc info for main run
    # gb_main_run_mcmc_info = dict(
    #     branch_names=["gb"],
    #     nleaves_max=15000,
    #     ndim=8,
    #     ntemps=len(betas),
    #     betas=betas,
    #     nwalkers=nwalkers,

    #     pe_waveform_kwargs=pe_gb_waveform_kwargs,
    #     ,

    #     use_prior_removal=False,
    #     rj_refit_fraction=0.2,
    #     rj_search_fraction=0.2,
    #     rj_prior_fraction=0.6,
    #     nsteps=10000,
    #     update_iterations=1,
    #     thin_by=3,
    #     progress=True,
    #     rho_star=rho_star,
    #     stop_kwargs=stopping_kwargs,
    #     stop_search_kwargs=dict(convergence_iter=5, verbose=True),  # really 5 * thin_by
    #     stopping_iterations=1,
    #     in_model_phase_maximize=False,
    #     rj_phase_maximize=False,
    # )

    # # mcmc info for search runs
    # gb_search_run_mcmc_info = dict(
    #     ndim=8,
    #     ntemps=10,
    #     nwalkers=100,
    #     pe_waveform_kwargs=pe_gb_waveform_kwargs,
    #     m_chirp_lims=[0.001, 1.2],
    #     snr_lim=5.0,
    #     # stop_kwargs=dict(newly_added_limit=1, verbose=True),
    #     stopping_iterations=1,
    # )

    # # template generator
    # get_gb_templates = GetGBTemplates(
    #     gb_initialize_kwargs,
    #     gb_waveform_kwargs
    # )

    # all_gb_info = dict(
    #     band_edges=band_edges,
    #     band_N_vals=band_N_vals,
    #     periodic=gb_periodic,
    #     priors=priors_gb_fin,
    #     transform=gb_transform_fn,
    #     waveform_kwargs=gb_waveform_kwargs,
    #     initialize_kwargs=gb_initialize_kwargs,
    #     pe_info=gb_main_run_mcmc_info,
    #     search_info=gb_search_run_mcmc_info,
    #     get_templates=get_gb_templates,
    # )
