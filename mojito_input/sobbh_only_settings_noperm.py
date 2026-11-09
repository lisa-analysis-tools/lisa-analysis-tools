import h5py
import numpy as np
import os
import shutil
import logging

from functools import partial

try:
    import cupy as cp

    gpu_available = True
except (ModuleNotFoundError, ImportError) as e:
    import numpy as cp

    gpu_available = False


from lisatools.detector import L1Orbits
from lisatools.utils.constants import *
from lisatools.globalfit.run import CurrentInfoGlobalFit
from lisatools.globalfit.stock.erebor import PSDSetup, PSDSettings, EMRISetup, EMRISettings, MBHSetup, MBHSettings, SOBBHTFSettings,SOBBHTFSetup

from eryn.prior import uniform_dist, log_uniform
from eryn.utils import TransformContainer
from eryn.prior import ProbDistContainer
from eryn.moves import StretchMove, TemperatureControl, DEMove
from eryn.moves.tempering import make_ladder

from lisatools.domains import STFTSettings, FDSettings
from lisatools.sensitivity import XYZSensitivityBackend
from lisatools.globalfit.moves import GFCombineMove, MultiGPUPSDMove, EMRISpecialMove, SOBBHSpecialMove
from lisatools.globalfit.engine import GlobalFitSettings, GeneralSetup, GeneralSettings, RankInfo
from lisatools.utils.constants import YRSID_SI
from lisatools.globalfit.preprocessing import L1ProcessingStep
from lisatools.globalfit.recipe import (
    SearchRecipeStep,
    PERecipeStep,
    IterationCountRecipeStep,
    build_psd_moves,
    build_mbh_moves_phenom,
    scatter_around_injection,
    mbh_catalogue_to_sampling_basis,
    emri_catalogue_to_sampling_basis,
    sobhb_catalogue_to_sampling_basis,
    subtract_initial_signal
)

from lisatools.globalfit.postprocessing import (
    StochasticMetadata,
    SourceMetadata
)

logger = logging.getLogger(__name__)

MOJITO_REFERENCE_TIME = 97729089.327664


def setup_recipe(recipe, engine_info, curr, acs, priors, state):

    # PSD is held fixed at general_info.fixed_psd_kwargs (no "psd" branch), so the
    # AnalysisContainers are built once with that sensitivity and only SOBBHs are sampled.

    cp.cuda.runtime.setDevice(curr.general_info.gpus[0])

    general_info = curr.general_info
    nwalkers: int = general_info.nwalkers
    ntemps: int = general_info.ntemps

    sobhb_info = curr.source_info["sobbh"]

    #* ========================= *#
    
    # Initialize SOBBH walkers from catalogue injection parameters
    catalogue = getattr(curr.general_info, "catalogue", {})
    sobhb_catalgue = catalogue.get("SOBHB", {})  # L1ProcessingStep upper-cases source types
    if sobhb_catalgue:
        sobhb_info = curr.source_info["sobbh"]
        injection_params_list = []
        for source_id in sorted(sobhb_catalgue.keys()):
            entry = sobhb_catalgue[source_id]
            sampling_params = sobhb_catalogue_to_sampling_basis(entry,trim_plus_shift = curr.general_info.data_t0 - MOJITO_REFERENCE_TIME)   # = 850.5 + 200*3600
            injection_params_list.append(sampling_params)

        injection_params = np.array(injection_params_list)

        # Store injection truths for diagnostic plots
        curr.source_info["sobbh"].injection = injection_params

        # Per-parameter spread for the Gaussian scatter (absolute, sampling basis).
        # Order: [Mc, eta, cosinc, dist, f0, s1, s2, phi_coal, psi, ra, dec]
        spread = np.array([
            1e-5,   # Mc
            1e-2,   # eta
            0.1,    # cosinc
            10.0,   # dist [Mpc]
            1e-9,   # f0 [Hz]
            0.1,    # s1
            0.1,    # s2
            0.0,    # phi_coal -> drawn from its prior below
            0.3,    # psi
            5e-3,   # ra
            5e-3,   # dec
        ])

        scatter_around_injection(
            state,
            "sobbh",
            injection_params,
            spread,
            priors=priors,
        )

        # phi_coal is unconstrained and slow-mixing: seed it from its prior rather than at the
        # catalogue value (fixed to 0, not the gwtf convention), which would freeze it.
        sobbh_coords = state.branches_coords["sobbh"]
        sobbh_coords[..., 7] = np.random.uniform(0.0, 2 * np.pi, size=sobbh_coords.shape[:-1])
        
    from lisatools.sources.sobbh.waveform_gwtf import GWTF_generator

    sobhb_wave_gen = GWTF_generator(**sobhb_info.initialize_kwargs)

    subtract_initial_signal(acs, state, sobhb_wave_gen.get_signal_for_residuals, "sobbh", sobhb_info)

    # If ladder is not supplied, then make a default geometric ladder with the given number of temperatures. 
    if sobhb_info.betas is None:
        sobhb_info.betas = make_ladder(sobhb_info.ndim, ntemps=ntemps)
    
    betas_all = np.tile(sobhb_info.betas, (sobhb_info.nleaves_max, 1))
    state.sub_states["sobbh"].betas_all = betas_all
    logger.debug(f"SOBHB betas: {sobhb_info.betas}")

    coords_shape_sobhb = (betas_all.shape[1], nwalkers, sobhb_info.nleaves_max, sobhb_info.ndim)


    # We dont have the normalising flow for what we are doing 
    # burnin_inner_moves_sobhb = attach_flow_trainers(sobhb_info, "sobbh")

    sobhb_move_kwargs = dict(
        dcga=acs,
        waveform_gen=sobhb_wave_gen,
        branch_name="sobbh",
        coords_shape=coords_shape_sobhb,
        waveform_gen_kwargs={},
        waveform_like_kwargs={},
        num_repeats=sobhb_info.num_prop_repeats,
        transform_fn=sobhb_info.transform,# I think None for now?
        priors=priors,
        inner_moves=sobhb_info.inner_moves,
        betas_all=betas_all,# *starting* inverse temperatures
        permute_every=0,# NOPERM test: fancy (walker-permuting) swap disabled; ordinary swaps still run
        pad_out_of_prior=True, # Out of prior points are replaced by a copy of a random in-prior point.
        run_async=True, #unused in sobbh block. 
        run_threaded=True, # threaded allowing for threaded execution of the likelihoods over GPUs. 
        randomize_split=True, # eryn setting 
        batch_size_per_gpu=None, # Maximum number of walkers per GPU to run in parallel. Setting None 
        tc_min_days=60, # Reject proposals merging less than tc_min_days after data start (None = off).
    )

    sobhb_pe_move = SOBBHSpecialMove(**sobhb_move_kwargs)
    sobhb_pe_move.accepted = np.zeros((ntemps, nwalkers))

    recipe.add_recipe_component(PERecipeStep(moves=[sobhb_pe_move]), name="sobbh pe")

#######################
##### SETTINGS ########
#######################

def get_sobbh_erebor_settings(general_set: GeneralSetup) -> SOBBHTFSetup:

    waveform_model = "GWTF_generator (pygwtf TaylorT3Spin, Fresnel TF)"
    waveform_model_code_link = "https://github.com/cchapmanbird/gwtf"  # todo populate repositories
    prior_model_code_link = "" # todo 
    frequency_ranges = [(general_set.start_freq, general_set.end_freq)]

    # GWTF_generator(settings, orbits, T_obs, fresnel_kernel_width, use_GPU).
    # GeneralSetup is built before the source settings, so domain_settings is
    # already the concrete (resolved) STFTSettings on general_set.force_backend
    # and gpu_orbits exists. MultiGPUResidualAddRemoveMove rebuilds one replica
    # per GPU from GWTF_generator.kwargs with "orbits" swapped per device.
    use_GPU = gpu_available and general_set.force_backend != "cpu"
    fresnel_kernel_width = 16  

    waveform_init_kwargs = dict(
        settings=general_set.domain_settings,
        orbits=general_set.gpu_orbits if use_GPU else general_set.orbits,
        T_obs=general_set.Tobs,  # post-trim duration (GeneralSetup resets Tobs = Nt * dt)
        fresnel_kernel_width=fresnel_kernel_width,
        use_GPU=use_GPU,
    )

    # GWTF_generator.get_signal_for_residuals takes no kwargs.
    waveform_runtime_kwargs = dict()

    # None -> setup_recipe builds make_ladder(ndim, ntemps=general ntemps).
    betas = [1/1.0, 1/1.1, 1/1.22, 1/1.4, 1/1.65, 1/2.0, 1/2.6, 1/3.5]


    # The catalogue source-type key is "sobhb" (L1ProcessingStep), the branch is "sobbh".
    # Number of leaves is the number of sources in the catalogue. The catalogue is read in setup_recipe.
    nleaves_max_sobbh = len(general_set.processor_init_kwargs["source_ids"]["sobhb"])

    sobbh_settings = SOBBHTFSettings(
        log_dir=general_set.artifacts_file_dir,
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        initialize_kwargs=waveform_init_kwargs,
        waveform_kwargs=waveform_runtime_kwargs,
        nleaves_max=nleaves_max_sobbh,# Fixed dimensionality for the sobbh
        nleaves_min=nleaves_max_sobbh,# Fixed dimensionality for the sobbh
        ndim=11,
        num_prop_repeats=300,
        betas=betas,
        inner_moves=[(StretchMove(), 1.0)],
        # transform / periodic / priors default inside SOBBHTFSetup (identity,
        # {phi_coal, psi, ra}, wide uniform box). Narrow any prior with <name>_lims=[lo, hi].
        f0_lims=[general_set.start_freq, general_set.end_freq],
    )

    wf_metadata = dict(
        model_class="TaylorT3Spin",
        tdi_type=2,
        fresnel_kernel_width=fresnel_kernel_width,
        T_obs=general_set.Tobs,
        sampling_basis=["Mc", "eta", "cosinc", "dist", "f0", "s1", "s2", "phi_coal", "psi", "ra", "dec"],
        f0_reference="data start (data_t0)",
        sky_frame="icrs",
    )

    sobbh_metadata = SourceMetadata(
        source_type="SOBBH",
        frequency_ranges=frequency_ranges,
        waveform_model=waveform_model,
        waveform_model_code_link=waveform_model_code_link,
        waveform_model_config=wf_metadata,
        prior_model_code_link=prior_model_code_link,
    )

    return SOBBHTFSetup(sobbh_settings), sobbh_metadata


def get_general_erebor_settings() -> GeneralSetup:

    global_fit_codename = "erebor"
    global_fit_version = "CDL1run0_v0"
    global_fit_contact = "ereborl2d@googlegroups.com"
    global_fit_code_link = "https://github.com/Erebor-L2D/LISAanalysistools/releases/tag/cdl1-run_0"
    global_fit_input_data_link = ""
    global_fit_input_reference = "mojito light"
    global_fit_noise_model = "parametric"
    global_fit_noise_model_code_link = "https://github.com/Erebor-L2D/LISAanalysistools/blob/9d63bb1e63e7b8f640d3780551d9421df5245992/src/lisatools/sensitivity.py#L1797" #todo populate repositories
    comment = ""

    submission_folder = None #"/work/asantini/globalfit/erebor_org_setup/mojito_runs/"

    num_iterations = 250

    source_ids = dict(
        sobhb=[0, 1, 2, 3, 4, 5]
        )

    # Mojito light is 2 yr; keep everything left after trimming trim_duration from each end.
    trim_duration = 200 * 3600  # s, trimmed from each end of the data
    Tobs = 2 * YRSID_SI - 2 * trim_duration
    dt = 5.
    start_freq = 1.e-4
    end_freq = 1.e-1

    head_dir = "/data/diganta/Mojito_Search/Integration_GF/Run/"  # trailing slash: paths are built by string concatenation
    data_input_path = "/data/asantini/globalfit/MOJITO_DATA/mojito_light_2p5s/"
    base_file_name = "SOBBH_only_MOJITO_light_2p5s_GF_setup_tc_prior_noperm"
    file_store_dir = head_dir

    gpus = [2]
    cp.cuda.runtime.setDevice(gpus[0])
    # Restrict JAX to only see the target GPU — must be set before JAX backend init
    import jax

    jax.config.update("jax_cuda_visible_devices", ",".join(str(gpu) for gpu in gpus))

    backend = "cuda12x" if gpus is not None else "cpu"
    nwalkers = 10
    ntemps = 8

    window_type = "tukey"
    window_taper_duration = 864 # s 
    normalize_window = True

    basis_domain = "stft"
    stft_dt = 1 * 24 * 3600.0 if basis_domain == "stft" else None  # hours

    if basis_domain == "stft":
        domain_settings = STFTSettings.make_factory(
            big_dt=stft_dt, min_freq=start_freq, max_freq=end_freq
        )
    else:
        domain_settings = FDSettings.make_factory(min_freq=start_freq, max_freq=end_freq)

    base_file_name += f"_{basis_domain}"

    processor_init_kwargs = dict(
        L1_folder=data_input_path,
        source_types=["noise", "sobhb"],  #'vgb', 'gb'
        source_ids=source_ids,
        verbose=True,
        do_plots=True,
        orbits_class=L1Orbits,
        store_individual_timeseries=True,
        orbits_kwargs=dict(force_backend="cpu", frame="icrs"),  # icrs
    )

    downsample_kwargs = {
        "target_fs": 1 / dt,  # Hz — target sampling rate (None = no downsampling).
        "window": (
            "kaiser",
            31.0,
        ),  # Kaiser window beta parameter (higher = more aggressive anti-aliasing)
    }

    highpass_kwargs = {
        "cutoff": 1.e-5,  # Hz — highpass cutoff frequency
        "order": 2,  # Butterworth filter order
        "zero_phase": True,
    }

    lowpass_kwargs = {
        "cutoff": 0.12,  # Hz — lowpass cutoff frequency
        "order": 2,  # Butterworth filter order
        "zero_phase": True,
    }

    trim_kwargs = {
        "duration": trim_duration,  # seconds — duration to trim from each end
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

    # PSD held at the injected noise levels (Soms_d, Sa_a amplitudes; squared in the kernel)
    fixed_psd_kwargs = dict(
        psd_params=[15e-12, 3e-15],
        galfor_params=None,
    )

    sensitivity_init_kwargs = dict(tdi_generation=2, mask_percentage=0.02, average_transfer_functions=True)

    general_settings = GeneralSettings(
        num_iterations=num_iterations,
        Tobs=Tobs,
        dt=dt,
        file_store_dir=file_store_dir,
        base_file_name=base_file_name,
        domain_settings=domain_settings,
        random_seed=103209,
        backup_iter=5,
        nwalkers=nwalkers,
        ntemps=ntemps,
        window_type=window_type,
        window_taper_duration=window_taper_duration,
        gpus=gpus,
        data_processor_class=L1ProcessingStep,
        processor_init_kwargs=processor_init_kwargs,
        preprocess_kwargs=preprocess_kwargs,
        normalize_window=normalize_window,
        sensitivity_backend_class=XYZSensitivityBackend,
        sensitivity_init_kwargs=sensitivity_init_kwargs,
        fixed_psd_kwargs=fixed_psd_kwargs,
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
    ###  SOBBH Settings  ##############
    ##################################
    ##################################

    sobbh_setup, sobbh_metadata = get_sobbh_erebor_settings(general_setup)

    ##############
    ## READ OUT ##
    ##############

    global_settings = GlobalFitSettings(
        source_info={
            "sobbh": sobbh_setup,
        },
        general_info=general_setup,
        rank_info=rank_info,
        setup_function=setup_recipe,
        source_metadata={
            "sobbh": sobbh_metadata,
        }
    )

    curr_info = CurrentInfoGlobalFit(global_settings)

    return curr_info


if __name__ == "__main__":
    settings = get_global_fit_settings()
    breakpoint()
