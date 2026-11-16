from __future__ import annotations

import h5py
import numpy as np
import os
import shutil
import sys
import logging

try:
    import cupy as cp

    gpu_available = bool(cp.cuda.is_available())
    if not gpu_available:
        import numpy as cp
except (ModuleNotFoundError, ImportError) as e:
    import numpy as cp

    gpu_available = False

from typing import TYPE_CHECKING, Any

from lisatools.detector import L1Orbits
from lisatools.domains import FDSettings, STFTSettings, WDMSettings, DomainSettingsBase
from lisatools.domaincomputation import DomainComputationGroupArray
from lisatools.utils.constants import *
from lisatools.sensitivity import XYZSensitivityBackend
from eryn.state import BranchSupplemental
from lisatools.globalfit.run import CurrentInfoGlobalFit
from lisatools.globalfit.stock.erebor import PSDSetup, PSDSettings, MBHSetup, MBHSettings, GBSetup, GBSettings
from lisatools.globalfit.stock.erebor import GBMoveSettings, GBSTFTSettings
from ereborpriors.stock.mojitolight import GBPrior


from eryn.prior import uniform_dist, log_uniform
from eryn.utils import TransformContainer
from eryn.prior import ProbDistContainer

from eryn.moves import StretchMove, TemperatureControl
from eryn.moves.tempering import make_ladder
from lisatools.globalfit.moves import GFCombineMove, MultiGPUPSDMove, TDMBHSpecialMove
from lisatools.globalfit.engine import GlobalFitSettings, GeneralSetup, GeneralSettings, RankInfo
from lisatools.globalfit.recipe import subtract_initial_signal
from lisatools.utils.constants import YRSID_SI

from eryn.utils.updates import Update

from lisatools.globalfit.preprocessing import L1ProcessingStep
from lisatools.globalfit.recipe import (
    SearchRecipeStep,
    PERecipeStep,
    RJRecipeStep,
    build_psd_moves,
    build_gb_moves,
    build_mbh_moves_phenom,
    scatter_around_injection,
    mbh_catalogue_to_sampling_basis,
    setup_state_for_injection,
)

from lisatools.globalfit.postprocessing import (
    StochasticMetadata,
    SourceMetadata
)

from gbgpu.gbcomps import STFTGBComputations
from lisatools.globalfit.moves.gbdebug import GBDebugSettings
from lisatools.globalfit.moves.gbspecialmove import GBSigHetSettings, GBLeafCapSettings

if TYPE_CHECKING:
    from lisatools.globalfit.recipe import Recipe
    from lisatools.analysiscontainer import AnalysisContainerArray
    from lisatools.datacontainer import DataResidualArray
    from lisatools.globalfit.engine import EngineInfo


MOJITO_REFERENCE_TIME = 97729089.327664
# * The stock GB prior of erebor-priors, at the commit this run installs.
GB_PRIOR_CODE_LINK = (
    "https://github.com/Erebor-L2D/erebor-priors/blob/fb65c565aea06c62a41ddbe95bb6b40a1a9c04b6/"
    "src/ereborpriors/stock/mojitolight/gb.py"
)

logger = logging.getLogger(__name__)


################

### DEFINE RECIPE

#############


def setup_recipe(
    recipe: Recipe, 
    engine_info, 
    curr: CurrentInfoGlobalFit, 
    acs: AnalysisContainerArray, 
    priors: dict[str, ProbDistContainer], 
    state
):
    general_info = curr.general_info
    assert general_info.nwalkers and general_info.ntemps, "nwalkers and ntemps must be set in general_info"
    nwalkers: int = general_info.nwalkers
    ntemps: int = general_info.ntemps
    
    if gpu_available:
        gpus = general_info.gpus
        assert gpus is not None and len(gpus) > 0, "No GPUs specified in general_info.gpus"
        getattr(cp, "cuda").runtime.setDevice(gpus[0])
        
    #* =============================== INJECT SOURCES =================================
    # Sampling basis: ``[logA, f0 [mHz], fdot, phi0, cos_iota, psi, alpha, sin_delta]``
    spread_gb = 0.0 
    
    data_input_path = str(general_info.input_data_link)
    iteratively_resolved_population_path = os.path.join(data_input_path + "/catalogues", "iteratively_resolved_gbs_075yrs_snr7.npy")
    iteratively_resolved_population = np.load(iteratively_resolved_population_path, allow_pickle=True)

    frequencies = iteratively_resolved_population["Frequency"]
    in_band = (frequencies > curr.source_info["gb"].f0_lims[0]) & (frequencies < curr.source_info["gb"].f0_lims[1])
    iteratively_resolved_population = iteratively_resolved_population[in_band]
    logger.info(f"Keeping {np.sum(in_band)} out of {len(iteratively_resolved_population)} iteratively resolved GB sources within the band limits {curr.source_info['gb'].f0_lims[0]} - {curr.source_info['gb'].f0_lims[1]}")
    
    subset_inds = np.array([int(name.split('_')[1]) for name in iteratively_resolved_population["Name"]])
    logger.info(f"Injecting {len(subset_inds)} GB sources from iteratively resolved population.")
    
    # subset_inds = None
    setup_state_for_injection(curr, state, "GB", "gb", spread=spread_gb, subset_inds=subset_inds, priors=priors)
    
        
    # ! WDM GB setups come at a later stage; GBSettings has no gb_wdm_comp until then.
    # #* ============== Chunked-heterodyne (WDM) GB likelihood ==============
    # # When the run domain is WDM, the GB branch uses the chunked-het
    # # likelihood (``gb_wdm_het.GBWDMHeterodyne``). Built lazily here, not on
    # # the settings dataclass: ``CurrentInfoGlobalFit.__init__`` deepcopies
    # # the settings tree and the underlying C++ orbit wrap is not picklable.
    # # ``N_cp_sig=48`` / ``N_cp_orbit=32`` are the validated defaults
    # # (median mm5 ~1e-9; see sprint-root CLAUDE.md). Mirrors the
    # # ``gb_and_foreground_global_fit_settings.py`` wiring.
    # gb_info = curr.source_info["gb"]
    # if (
    #     isinstance(general_info.domain_settings, WDMSettings)
    #     and gb_info.gb_wdm_comp is None
    # ):
    #     _gb_wdm_het_dir = os.path.abspath(
    #         os.path.join(
    #             os.path.dirname(os.path.abspath(__file__)),
    #             "..", "scripts", "gb_chunked_het",
    #         )
    #     )
    #     if _gb_wdm_het_dir not in sys.path:
    #         sys.path.insert(0, _gb_wdm_het_dir)
    #     from gb_wdm_het import GBWDMHeterodyne

    #     _wdm = general_info.domain_settings
    #     # Anchor the chunked-het grid at the runtime data start
    #     # (``times[0]`` after the preprocess trim), not at 0.
    #     _t_obs_start = float(getattr(general_info, "data_t0", 0.0))
    #     _jax_chunk_env = os.environ.get("CHUNKED_JAX_CHUNK")
    #     _jax_chunk = int(_jax_chunk_env) if _jax_chunk_env else None
    #     gb_info.gb_wdm_comp = GBWDMHeterodyne(
    #         Nf=_wdm.Nf, Nt=_wdm.Nt, dt=general_info.dt,
    #         T_full=general_info.Tobs, t_ref_full=gb_info.t0,
    #         Nt_sub=int(os.environ.get("CHUNKED_NT_SUB", 256)),
    #         n_pad=int(os.environ.get("CHUNKED_N_PAD", 32)),
    #         N_sparse=int(os.environ.get("CHUNKED_N_SPARSE", 256)),
    #         nchannels=3,
    #         force_backend=general_info.force_backend,
    #         tdi_gen="2nd generation",
    #         orbits=general_info.gpu_orbits,
    #         t_obs_start=_t_obs_start,
    #         N_cp_sig=int(os.environ.get("CHUNKED_N_CP_SIG", 48)),
    #         N_cp_orbit=int(os.environ.get("CHUNKED_N_CP_ORBIT", 32)),
    #         jax_chunk=_jax_chunk,
    #     )
    #     logger.info(
    #         "Chunked-het GB likelihood: Nf=%d Nt=%d Nt_sub=%d N_sparse=%d "
    #         "N_cp_sig=%d N_cp_orbit=%d (t_obs_start=%.3e t_ref=%.3e)",
    #         _wdm.Nf, _wdm.Nt,
    #         gb_info.gb_wdm_comp.Nt_sub, gb_info.gb_wdm_comp.N_sparse,
    #         gb_info.gb_wdm_comp.N_cp_sig, gb_info.gb_wdm_comp.N_cp_orbit,
    #         _t_obs_start, gb_info.t0,
    #     )

    #* ================================= BUILD MOVES ==================================
    _, gb_pe_moves = build_gb_moves(
        engine_info, curr, acs, priors, state
    )

    #* ================================= SETUP SEARCH ================================= 
    # search_weights = [0.8, 0.2]
    # recipe.add_recipe_component(RJRecipeStep(moves=gb_search_moves, weights=search_weights, convergence_iter=10), name="gb search")
    
    #* ========================== SETUP PARAMETER ESTIMATION ========================== 
    pe_weights = None # [0.8, 0.16, 0.04] # [0.05, 0.45, 0.5] # 
    recipe.add_recipe_component(PERecipeStep(moves=gb_pe_moves, weights=pe_weights), name="gb_pe")
    
    # moves_info = "".join([f"Move {all_pe_moves[i].name} has weight {w}, " for i, w in enumerate(pe_weights)])
    # logger.info(f"For PE: {moves_info}")


#######################
##### SETTINGS ###########
###############

def get_gb_erebor_settings(general_set: GeneralSetup) -> tuple[GBSetup, SourceMetadata]:
    # ! Grids without min_freq (FD, WDM) come at a later stage: the factory reads the STFT grid.
    assert general_set.preprocess_kwargs
    assert general_set.ntemps is not None, "ntemps must be set in general_info" # TODO add post init method of settings

    # ! The FD serial search reads search_kwargs; FD GB setups come at a later stage.
    # search_kwargs = dict(
    #     nwalkers = 32,
    #     ntemps = 24,
    #     shutoff_band_iteration = 2,
    #     shutoff_frequency_threshold = None, # 4e-3
    #     burn_1 = 700,
    #     nsteps_1 = 300,
    #     snr_threshold = 8.0,
    #     burn_2 = 500,
    #     nsteps_2 = 500,
    #     refit_start_iteration = 5
    # )

    leaf_cap_settings = None
    # leaf_cap_settings = GBLeafCapSettings(
    #     start=None,
    #     min_iters=int(os.environ.get("GB_LEAF_CAP_MIN_ITERS", "50")),
    #     ll_nsigma=float(os.environ.get("GB_LEAF_CAP_LL_NSIGMA", "3.0")),
    #     require_occupancy=bool(int(os.environ.get("GB_LEAF_CAP_OCCUPANCY", "1"))),
    # )
    sighet_settings = None
    # sighet_settings = GBSigHetSettings(
    #     refresh_every=int(os.environ.get("GB_SIGHET_REFRESH_EVERY", "20")),
    #     refresh_dphase=float(os.environ.get("GB_SIGHET_REFRESH_DPHASE", "0.5")),
    #     refresh_min_beta=float(os.environ.get("GB_SIGHET_REFRESH_MIN_BETA", "0.1")),
    #     trust_dlna=float(os.environ.get("GB_SIGHET_TRUST_DLNA", "1.5")),
    #     trust_dphase=float(os.environ.get("GB_SIGHET_TRUST_DPHASE", "0.5")),
    #     trust_snr_c=float(os.environ.get("GB_SIGHET_TRUST_SNR_C", "30")),
    #     trust_dlna_min=float(os.environ.get("GB_SIGHET_TRUST_DLNA_MIN", "0.3")),
    #     anchor_check=os.environ.get("GB_SIGHET_ANCHOR_CHECK", "0") == "1",
    #     drift_check=os.environ.get("GB_SIGHET_DRIFT_CHECK", "0") == "1",
    # )

    # * The time grid, window, STFT domain, orbits, backend and ladder come from general_set.
    gb_settings = GBSettings.from_general_setup(
        general_set,
        # * The shared erebor-priors stock GB prior; GBSetup limits its f0 to the proposed bands.
        priors={"gb": GBPrior.prior_factory()},
        nleaves_max=20,
        nleaves_min=0,
        # ! The chunked-het (WDM) GB likelihood comes at a later stage: no gb_wdm_comp until then.
        stft=GBSTFTSettings(n_side_bins=10),
        moves=GBMoveSettings(
            num_repeat_proposals=50,
            stretch_probability=0.0,
            num_band_preload=512,
            num_bands_preload_temp=128,
            leaf_cap_settings=leaf_cap_settings,
            sighet_settings=sighet_settings,
        ),
    )
    gb_setup = GBSetup(gb_settings)
    gb_metadata = gb_setup.make_source_metadata(
        waveform_model_code_link="https://github.com/Erebor-L2D/GBGPU/tree/cd1l-run0",
        prior_model_code_link=GB_PRIOR_CODE_LINK,
    )
    return gb_setup, gb_metadata


def get_general_erebor_settings() -> GeneralSetup:

    global_fit_codename = "erebor"
    global_fit_version = "highf_gb_stft_2_gpu_run2"
    global_fit_contact = "ereborl2d@googlegroups.com"
    global_fit_code_link = "https://github.com/Erebor-L2D"
    global_fit_input_data_link = "/workspace/ggfitlisa/ldc/mojito_light/"
    global_fit_input_reference = "mojito light"
    global_fit_noise_model = "parametric"
    global_fit_noise_model_code_link = "https://github.com/Erebor-L2D"
    comment = "Testing STFT multi-gpu"

    submission_folder = None
    
    num_iterations = 100
    
    Tobs = 9.0 * YRSID_SI / 12.0
    dt = 5.0
    start_freq, end_freq = [0.014, 0.022]

    head_dir = "/workspace/rrondeel/dev_gb_stft/_results/"
    data_input_path = "/workspace/ggfitlisa/ldc/mojito_light/"
    base_file_name = global_fit_version #"test_mbh_18_with_covariance"
    file_store_dir = head_dir + "gb_test_runs/"

    gpus = [0, 1] if gpu_available else None
    if gpu_available:
        cp.cuda.runtime.setDevice(gpus[0])
        # Restrict JAX to only see the target GPU — must be set before JAX backend init
        import jax

        jax.config.update("jax_cuda_visible_devices", ",".join(str(gpu) for gpu in gpus))

    backend = "cuda12x" if gpus is not None else "cpu"
    nwalkers = 8
    ntemps = 4
    
    #* =============================== DOMAIN SETTINGS =================================
    # ! GBSetup builds STFT bands only for now; the FD and WDM choices below come at a later stage.
    domain_cls = STFTSettings

    if domain_cls == STFTSettings:
        stft_dt = 1 * 24 * 3600.0  # hours
        domain_settings = STFTSettings.make_factory(
            big_dt=stft_dt, min_freq=start_freq, max_freq=end_freq
        )
    if domain_cls == FDSettings:
        domain_settings = FDSettings.make_factory(
            min_freq=start_freq, max_freq=end_freq
        )
    if domain_cls == WDMSettings:
        # WDM grid: ~1-hour wavelets via adjust_to_even_bins, which also snaps
        # Tobs to an exact Nf*Nt*dt span.
        WAVELET_DUR_BOUNDS = (
            float(os.environ.get("WAVELET_DUR_MIN", 3600.0)),
            float(os.environ.get("WAVELET_DUR_MAX", 4400.0)),
        )
        NF, NT, WAVELET_DURATION = WDMSettings.adjust_to_even_bins(
            t_min=WAVELET_DUR_BOUNDS[0], t_max=WAVELET_DUR_BOUNDS[1], dt=dt, Tobs=Tobs,
        )
        Tobs = NF * NT * dt  # exact WDM span
        logger.info(
            "WDM grid: Nf=%d Nt=%d wavelet_duration=%.1f s Tobs=%.6e s",
            NF, NT, WAVELET_DURATION, Tobs,
        )
        domain_settings = domain_cls.make_factory(
            Nf=NF, Nt=NT, min_freq=start_freq, max_freq=end_freq,
            min_time=20 * WAVELET_DURATION, max_time=(NT - 20) * WAVELET_DURATION,
        ) # type: ignore

    # base_file_name += f"_{domain_cls.__name__}"

    # * =============================== DATA PROCESSOR =================================    
    window_type = "tukey"
    window_taper_duration = 1 / start_freq
    normalize_window = True
    
    processor_init_kwargs = dict(
        L1_folder=data_input_path,
        source_types=["gb", "noise"],  #'vgb', 'gb', "mbhb",
        source_ids=dict(), # mbhb=source_ids
        verbose=True,
        do_plots=True,
        orbits_class=L1Orbits,
        orbits_kwargs=dict(
            force_backend=backend, 
            frame="icrs", 
            armlength=2493162305.42235
        ), 
    )

    downsample_kwargs = {
        "target_fs": 1 / dt,  # Hz — target sampling rate (None = no downsampling).
        "window": (
            "kaiser",
            31.0,
        ),  # Kaiser window beta parameter (higher = more aggressive anti-aliasing)
    }

    highpass_kwargs = {
        "cutoff": 1e-5,  # Hz — highpass cutoff frequency
        "order": 2,  # Butterworth filter order
        "zero_phase": True,
    }

    lowpass_kwargs = {
        "cutoff": 1e-1,  # Hz — lowpass cutoff frequency
        "order": 2,  # Butterworth filter order
        "zero_phase": True,
    }

    trim_kwargs = {
        "duration": 200 * 3600,  # seconds — duration to trim from each end
        "is_percent": False,  # If True, 'duration' is interpreted as a percentage of the total signal length
        "trimming_type": "from_each_end",  # "from_each_end" or "from_start"
    }

    preprocess_kwargs = dict(
        highpass_kwargs=highpass_kwargs,
        lowpass_kwargs=lowpass_kwargs,
        trim_kwargs=trim_kwargs,
        downsample_kwargs=downsample_kwargs,
        Tobs=Tobs,
    )

    sensitivity_init_kwargs = dict(
        tdi_generation=2, 
        mask_percentage=0.02, 
        average_transfer_functions=True
    )

    general_settings = GeneralSettings(
        Tobs=Tobs,
        dt=dt,
        file_store_dir=file_store_dir,
        base_file_name=base_file_name,
        domain_settings=domain_settings,
        random_seed=5701424,
        backup_iter=5,
        nwalkers=nwalkers,
        ntemps=ntemps,
        num_iterations=num_iterations,
        window_type=window_type,
        window_taper_duration=window_taper_duration,
        gpus=gpus,
        gpu_backend=backend,
        data_processor_class=L1ProcessingStep,
        processor_init_kwargs=processor_init_kwargs,
        preprocess_kwargs=preprocess_kwargs,
        normalize_window=normalize_window,
        sensitivity_backend_class=XYZSensitivityBackend,
        sensitivity_init_kwargs=sensitivity_init_kwargs,
        global_fit_codename=global_fit_codename,
        global_fit_version=global_fit_version,
        global_fit_contact=global_fit_contact,
        global_fit_code_link=global_fit_code_link,
        input_data_link=global_fit_input_data_link,
        input_reference=global_fit_input_reference,
        noise_model=global_fit_noise_model,
        noise_model_code_link=global_fit_noise_model_code_link,
        submission_parent_folder=submission_folder,
        comment=comment
    )

    general_setup = GeneralSetup(general_settings)
    
    #? do we still need this?
    # Band/STFT metadata consumed by the per-source setup functions
    # (no longer GeneralSettings fields post-merge; the analysis band
    # itself lives on domain_settings).
    # general_setup.start_freq = start_freq
    # general_setup.end_freq = end_freq
    # general_setup.stft_dt = stft_dt
    
    return general_setup


def get_global_fit_settings(copy_settings_file=False):

    general_setup = get_general_erebor_settings()

    if copy_settings_file:
        shutil.copy(
            __file__,
            general_setup.file_store_dir
            + general_setup.base_file_name
            + "_"
            + __file__.split("/")[-1],
        )

    ###############################
    ###############################
    ######    Rank/GPU setup  #####
    ###############################
    ###############################

    head_rank = 1

    main_rank = 0

    # run results rank will be next available rank if used
    # gmm_ranks will be all other ranks

    rank_info = RankInfo(head_rank=head_rank, main_rank=main_rank)

    ##################################
    ##################################
    ###  PSD Settings  ###############
    ##################################
    ##################################

    # psd_setup, psd_metadata = get_psd_erebor_settings(general_setup)

    ##################################
    ##################################
    ###  MBH Settings  ###############
    ##################################
    ##################################

    # mbh_setup, mbh_metadata = get_mbh_erebor_settings(general_setup)

    ##################################
    ##################################
    ###  GB Settings  ###############
    ##################################
    ##################################

    gb_setup, gb_metadata = get_gb_erebor_settings(general_setup)

    ##############
    ## READ OUT ##
    ##############

    global_settings = GlobalFitSettings(
        source_info={
            # "mbh": mbh_setup,
            "gb": gb_setup,
            # "psd": psd_setup,
        },
        general_info=general_setup,
        rank_info=rank_info,
        setup_function=setup_recipe,
        source_metadata={
            # "mbh": mbh_metadata,
            "gb": gb_metadata,
            # "psd": psd_metadata,
        }
    )

    curr_info = CurrentInfoGlobalFit(global_settings)

    return curr_info


if __name__ == "__main__":
    settings = get_global_fit_settings()
    breakpoint()