"""Full-band global fit of the resolved galaxy, the instrument noise and the galactic foreground.

STFT domain of 713 one-day segments over 0.1 to 30 mHz, DFFR response. The data hold the WD-WD
galaxy, the verification binaries and the instrument noise. The GB branch starts at the iteratively
resolved sources of the 714-day subtraction run (SNR 7) inside the amplitude prior and at every
verification binary, drawn with a small spread around their catalogue values. The sources below the
prior's lower limit are left out; the run log and
global_fit_dev/plans/memory/optimisation_test_1_removed_sources.md list them. The recipe first
searches the PSD and foreground at maximum likelihood with the GBs fixed, then runs
NUM_PE_ITERATIONS iterations of the GB block (RJ
from the prior, Fisher in-model proposals) combined with the PSD and foreground move.

    GLOBAL_FIT_RUNS_DIR=<folder> python scripts/run_global.py \\
        -sfp mojito_input/galaxy_psd_dffr_global_fit_settings.py -sff get_global_fit_settings

``get_global_fit_settings`` runs without any probe, for timings. ``get_global_fit_probe_settings``
runs the same with the GB memory probe on, for memory, under its own run name.
``get_global_fit_calibration_settings`` runs one iteration of that, also under its own run name.
``get_optimisation_test_2_settings`` and ``get_optimisation_test_3_settings`` run all iterations with the
probe, with tempering width classes off and on.
Environment: GLOBAL_FIT_RUNS_DIR (default: global_fit_runs of the workspace), MOJITO_FOLDER (default the
Mojito-light data on Spider), RESOLVED_GB_FILE (default the 714-day resolved-source file).
"""

from __future__ import annotations

import logging
import os
import pathlib
import shutil

import numpy as np

try:
    import cupy as cp

    gpu_available = bool(cp.cuda.is_available())
except (ModuleNotFoundError, ImportError):
    gpu_available = False

from eryn.moves import TemperatureControl
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.utils import TransformContainer
from gbgpu.gbcomps import STFTGBComputations

from lisatools.detector import L1Orbits
from lisatools.domains import STFTSettings
from lisatools.globalfit.engine import GeneralSettings, GeneralSetup, GlobalFitSettings, RankInfo
from lisatools.globalfit.moves import GFCombineMove, MultiGPUPSDMove
from lisatools.globalfit.moves.gbdebug import GBDebugSettings
from lisatools.globalfit.moves.gbspecialmove import GBSpecialBase
from lisatools.globalfit.postprocessing import SourceMetadata, StochasticMetadata
from lisatools.globalfit.preprocessing import L1ProcessingStep
from lisatools.globalfit.recipe import (
    MOJITO_REFERENCE_TIME,
    IterationCountRecipeStep,
    SearchRecipeStep,
    build_gb_moves,
    gb_catalogue_to_sampling_basis,
    scatter_around_injection,
)
from lisatools.globalfit.run import CurrentInfoGlobalFit
from lisatools.globalfit.stock.erebor import (
    GalForSettings,
    GalForSetup,
    GBSettings,
    GBSetup,
    PSDSettings,
    PSDSetup,
    get_fdot_mojito,
)
from lisatools.sensitivity import XYZSensitivityBackend

# Under "lisatools", whose level and run-log handler the global fit sets; __name__ is not.
logger = logging.getLogger("lisatools.settings.galaxy_psd_dffr_global_fit_settings")

RUN_NAME = "optimisation_test_1"
_WORKSPACE = pathlib.Path.home() / "projects" / "erebor"
DEFAULT_RUNS_DIR = _WORKSPACE / "global_fit_runs"
DEFAULT_RESOLVED_GB_FILE = _WORKSPACE / "data" / "iteratively_resolved_gbs_714d_snr7_named.npy"

STFT_SEGMENT_S = 86400.0
N_SEGMENTS = 713
START_FREQ_HZ = 1.0e-4
END_FREQ_HZ = 30.0e-3
WINDOW_TAPER_S = 1.0 / START_FREQ_HZ
N_SIDE_BINS = 16
NWALKERS = 32
NTEMPS = 24
# * 7319 resolved sources and 55 verification binaries; the rest is room for RJ births.
NLEAVES_MAX = 8000
# * None: the moves size their buffers from the memory free at each build.
NUM_BAND_PRELOAD = None
NUM_BANDS_PRELOAD_TEMP = None
NUM_PE_ITERATIONS = 5
# An upper bound only: the recipe stops after the search and NUM_PE_ITERATIONS iterations.
MAX_ITERATIONS = 1 + NUM_PE_ITERATIONS + 4
PSD_PROP_REPEATS = 100

# * Start spread of the GB walkers, one standard deviation per sampling parameter
# * (ln A, f0 [mHz], fdot [Hz/s], phi0, cos iota, psi, alpha, sin delta). f0 is about 1e-4 of its
# * resolution 1/Tobs, being the most precisely measured, and fdot about 1e-3 of 1/Tobs^2;
# * scatter_around_injection widens hotter chains.
GB_START_SPREAD = np.array([1e-3, 1e-9, 1e-19, 1e-3, 1e-3, 1e-3, 1e-4, 1e-4])
PROBE_FILE_NAME = "mem_probe.jsonl"

LOG10_TM_ASD_RANGE = (-16.0, -13.0)
LOG10_OMS_ASD_RANGE = (-12.0, -10.0)
LOG10_AMP_RANGE = (-48.0, -40.0)
ALPHA_RANGE = (1.0, 10.0)
LOG10_FREQ1_RANGE = (np.log10(1e-4), np.log10(1e-2))
LOG10_FREQ2_RANGE = (np.log10(1e-4), np.log10(1e-2))
LOG10_FKNEE_RANGE = (np.log10(1e-3), np.log10(1e-1))


def ten_to_the_x(x):
    return 10.0**x


def check_resolved_rows(resolved, galaxy_entry: dict, epoch_lag_s: float) -> None:
    """Stop unless every resolved source is the catalogue row its ``CatalogueIndex`` names.

    The resolved file holds its parameters at data_t0, the catalogue at its reference epoch, so
    the amplitudes must agree bit for bit and the file's f0 must be the catalogue f0 evolved by
    fdot over the lag (exact for the 714-day file at a lag of 720850.5 s, checked 2026-10-30).
    """
    rows = resolved["CatalogueIndex"]
    amplitude = np.asarray(galaxy_entry["Amplitude"])[rows]
    f0_hz = np.asarray(galaxy_entry["GW22FrequencySSBFrame"])[rows]
    fdot = np.asarray(galaxy_entry["GW22FrequencyDerivativeSourceFrame"])[rows]
    evolved_f0_hz = f0_hz + fdot * epoch_lag_s
    if not np.array_equal(amplitude, resolved["Amplitude"]):
        raise ValueError("the resolved-source file does not index the loaded WD-WD catalogue.")
    # 1e-12 Hz is 6e-5 of a frequency bin; it allows rounding, not a wrong epoch.
    if not np.allclose(evolved_f0_hz, resolved["Frequency"], rtol=0.0, atol=1e-12):
        raise ValueError(
            f"the resolved-source frequencies do not sit {epoch_lag_s} s after the catalogue epoch; "
            "the file and this run use different data start times."
        )


def setup_recipe(recipe, engine_info, curr, acs, priors, state):
    general_info = curr.general_info
    if gpu_available:
        cp.cuda.runtime.setDevice(general_info.gpus[0])

    # * =============================== START STATE ===================================
    resolved = np.load(
        os.environ.get("RESOLVED_GB_FILE", str(DEFAULT_RESOLVED_GB_FILE)), allow_pickle=False
    )
    rows = resolved["CatalogueIndex"].astype(np.int64)
    galaxy_entry = general_info.catalogue["GB"][0]
    check_resolved_rows(resolved, galaxy_entry, general_info.data_t0 - MOJITO_REFERENCE_TIME)
    # * Sources below the amplitude prior's lower limit cannot start: the recipe refuses a start state
    # * outside the prior. They stay in the data and out of the model (user, 2026-10-30).
    below_amplitude_prior = resolved["Amplitude"] < curr.source_info["gb"].A_lims[0]
    logger.info(
        f"Left out {below_amplitude_prior.sum()} resolved sources below the amplitude prior's lower "
        f"limit, CatalogueIndex {rows[below_amplitude_prior].tolist()}"
    )
    rows = rows[~below_amplitude_prior]
    # The start state takes the catalogue's own values, as the validated conversion expects.
    catalogue_f0_hz = np.asarray(galaxy_entry["GW22FrequencySSBFrame"])[rows]
    gb_info = curr.source_info["gb"]
    f0_lo_hz, f0_hi_hz = (float(limit) for limit in gb_info.f0_lims)
    outside = (catalogue_f0_hz < f0_lo_hz) | (catalogue_f0_hz >= f0_hi_hz)
    if np.any(outside):
        raise ValueError(
            f"{outside.sum()} resolved sources lie outside the sampled f0 range "
            f"[{f0_lo_hz * 1e3:.4f}, {f0_hi_hz * 1e3:.4f}] mHz."
        )
    # * Both populations go into the one gb branch in a single scatter call: the scatter fills the
    # * leaves from index 0, so a second call would overwrite the first population.
    galaxy_params = gb_catalogue_to_sampling_basis(galaxy_entry)[rows]
    verification_params = np.concatenate(
        [
            np.atleast_2d(gb_catalogue_to_sampling_basis(general_info.catalogue["VGB"][source_id]))
            for source_id in sorted(general_info.catalogue["VGB"])
        ]
    )
    start_params = np.concatenate([galaxy_params, verification_params])
    # The sampling basis holds f0 in mHz.
    vgb_f0_hz = verification_params[:, 1] * 1e-3
    if np.any((vgb_f0_hz < f0_lo_hz) | (vgb_f0_hz >= f0_hi_hz)):
        raise ValueError("a verification binary lies outside the sampled f0 range.")
    if start_params.shape[0] > gb_info.nleaves_max:
        raise ValueError(
            f"{start_params.shape[0]} start sources exceed nleaves_max {gb_info.nleaves_max}."
        )
    logger.info(
        f"Starting {rows.size} resolved GB sources (f0 in [{catalogue_f0_hz.min() * 1e3:.4f}, "
        f"{catalogue_f0_hz.max() * 1e3:.4f}] mHz) and {verification_params.shape[0]} verification "
        f"binaries at their catalogue values."
    )
    gb_info.injection = start_params
    scatter_around_injection(
        state, "gb", start_params, GB_START_SPREAD, betas=gb_info.betas, priors=priors,
        # Near 0.35 mHz the hottest chain's fdot scatter is about 30 times wider than the fdot prior,
        # so only about 1 draw in 80 lands inside it.
        max_resample_tries=2000,
    )

    # * ================================= MOVES =======================================
    # ! Build the GB moves before the PSD moves: this call resets acs.domain_group_kwargs, which
    # ! drops the cached computation groups, and MultiGPUPSDMove binds split 0's sensitivity
    # ! backend once, at construction.
    _, gb_pe_moves = build_gb_moves(
        engine_info, curr, acs, priors, state,
        search_move_names=[],
        pe_move_names=["rj_prior"],
        include_refit=False,
    )

    psd_info = curr.source_info["psd"]
    galfor_info = curr.source_info["galfor"]
    # The PSD and foreground parameters share one move, so the ladder uses their joint dimension.
    temperature_control = TemperatureControl(
        engine_info.ndims["psd"] + engine_info.ndims["galfor"],
        NWALKERS, ntemps=NTEMPS, Tmax=1.0e6, permute=False,
    )
    psd_move_kwargs = dict(
        num_repeats=psd_info.num_prop_repeats,
        permute_every=200,
        live_dangerously=True,
        psd_transform_fn=psd_info.transform,
        galfor_transform_fn=galfor_info.transform,
        temperature_control=temperature_control,
        use_gpu=gpu_available,
        run_async=True,
        run_threaded=False,
    )
    psd_search_move = MultiGPUPSDMove(
        acs, priors, max_logl_mode=True, name="psd search move", **psd_move_kwargs
    )
    psd_pe_move = MultiGPUPSDMove(
        acs, priors, max_logl_mode=False, name="psd pe move", **psd_move_kwargs
    )
    psd_search_move.accepted = np.zeros((NTEMPS, NWALKERS))
    psd_pe_move.accepted = np.zeros((NTEMPS, NWALKERS))

    # * ================================= RECIPE ======================================
    # The search is one iteration (the move runs to its maximum likelihood) and is not counted.
    recipe.add_recipe_component(SearchRecipeStep(moves=[psd_search_move]), name="psd+galfor search")
    gb_psd_move = GFCombineMove(
        moves=[gb_pe_moves[0], psd_pe_move], share_temperature_control=False
    )
    recipe.add_recipe_component(
        IterationCountRecipeStep(moves=[gb_psd_move], num_iters=NUM_PE_ITERATIONS),
        name="gb+psd+galfor pe",
    )


def get_psd_settings(general_set: GeneralSetup) -> tuple:
    frequency_ranges = [(general_set.start_freq, general_set.end_freq)]
    psd_basis = [r"$S_{\rm oms}$", r"$S_{\rm tm}$"]
    prior_model_config = {
        r"$S_{\rm oms}$": LOG10_OMS_ASD_RANGE,
        r"$S_{\rm tm}$": LOG10_TM_ASD_RANGE,
    }
    # The two noise levels are sampled in log10 and mapped back for the kernel.
    psd_transform = TransformContainer(
        input_basis=psd_basis,
        output_basis=psd_basis,
        parameter_transforms={name: ten_to_the_x for name in psd_basis},
    )
    priors = {
        "psd": ProbDistContainer(
            {name: uniform_dist(*prior_model_config[name]) for name in psd_basis}
        )
    }
    psd_settings = PSDSettings(
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        initialize_kwargs=dict(),
        priors=priors,
        ndim=2,
        # * The Mojito-light noise levels, for the diagnostic plots only.
        injection=np.array([np.log10(15e-12), np.log10(3e-15)]),
        log_dir=general_set.artifacts_file_dir,
        num_prop_repeats=PSD_PROP_REPEATS,
        transform=psd_transform,
    )
    psd_metadata = StochasticMetadata(
        model_config=dict(use_splines=False, num_params=2),
        frequency_ranges=frequency_ranges,
        prior_model="uniform",
        prior_model_code_link="",
        prior_model_config=prior_model_config,
    )
    return PSDSetup(psd_settings), psd_metadata


def get_galfor_settings(general_set: GeneralSetup) -> tuple:
    frequency_ranges = [(general_set.start_freq, general_set.end_freq)]
    galfor_basis = [
        r"$\log_{10} A_{\rm gal}$",
        r"$\alpha_{\rm gal}$",
        r"$\log_{10} f_1$",
        r"$\log_{10} f_{\rm knee}$",
        r"$\log_{10} f_2$",
    ]
    prior_model_config = dict(
        zip(galfor_basis, (LOG10_AMP_RANGE, ALPHA_RANGE, LOG10_FREQ1_RANGE, LOG10_FKNEE_RANGE,
                           LOG10_FREQ2_RANGE))
    )
    galfor_transform = TransformContainer(
        input_basis=galfor_basis,
        output_basis=galfor_basis,
        parameter_transforms={
            name: ten_to_the_x for name in galfor_basis if name != r"$\alpha_{\rm gal}$"
        },
    )
    # Only the spectral envelope is inferred; the sky geometry of the galactic grid is fixed.
    galfor_settings = GalForSettings(
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        initialize_kwargs={},
        transform=galfor_transform,
        priors={
            "galfor": ProbDistContainer(
                {name: uniform_dist(*prior_model_config[name]) for name in galfor_basis}
            )
        },
        ndim=5,
    )
    galfor_metadata = StochasticMetadata(
        model_config=dict(
            num_params=5,
            galactic_grid_kwargs=general_set.sensitivity_init_kwargs.get(
                "galactic_grid_kwargs", {}
            ),
        ),
        frequency_ranges=frequency_ranges,
        prior_model="uniform",
        prior_model_code_link="",
        prior_model_config=prior_model_config,
    )
    return GalForSetup(galfor_settings), galfor_metadata


def get_gb_settings(general_set: GeneralSetup, mem_probe_file: str | None) -> tuple:
    delta_safe = 1e-9
    # * f0 and fdot limits are reset from the band edges by GBSetup.
    f0_lims = [START_FREQ_HZ, END_FREQ_HZ]
    domain_settings = general_set.domain_settings
    start_freq = float(domain_settings.min_freq)
    end_freq = float(domain_settings.max_freq)
    Tobs = general_set.Tobs
    oversample = 4
    start_freq_ind = int(np.round(start_freq * Tobs))

    initialize_kwargs = dict(
        orbits=general_set.gpu_orbits if gpu_available else general_set.orbits,
        t0=general_set.data_t0,
        force_backend=general_set.gpu_backend,
    )
    # * Geometric ladder; the hottest chain samples nearly the prior.
    betas = 1 / 1.2 ** np.arange(general_set.ntemps)
    betas[-1] = 0.0001

    waveform_kwargs = dict(
        dt=general_set.dt,
        T=Tobs,
        use_c_implementation=True,
        start_freq_ind=start_freq_ind,
        tdi_channel_setup="XYZ",
        tdi2=True,
        oversample=oversample,
        window=general_set.window_type,
        window_alpha=general_set.window_alpha,
    )
    # * first_order_delays=True as in every DFFR run of the GB memory effort (user, 2026-10-30);
    # * DFFR's own default is the full delays.
    gb_stft_kwargs = dict(
        n_side_bins=N_SIDE_BINS,
        # The recipe needs the key; DFFR does not read the value.
        freq_from_tdi_phase=True,
        use_midpoint=True,
        linear_envelope=True,
        response="dffr",
        first_order_delays=True,
    )
    debug_settings = GBDebugSettings(
        enabled=False,
        plot_walker=0,
        plot_band=None,
        seq_pick="first",
        rj_trace=False,
        prop_timing_sync=False,
        # The probe runs Python on every CuPy allocation (about 10 % of an iteration).
        mem_probe=mem_probe_file is not None,
        mem_probe_file=mem_probe_file,
    )
    gb_settings = GBSettings(
        Tobs=Tobs,
        dt=general_set.dt,
        initialize_kwargs=initialize_kwargs,
        nleaves_max=NLEAVES_MAX,
        nleaves_min=0,
        ndim=8,
        betas=betas,
        log_dir=general_set.file_store_dir,
        A_lims=[10 ** (-23.2), 1e-20],
        f0_lims=f0_lims,
        m_chirp_lims=[0.03, 1.34],
        fdot_lims=[get_fdot_mojito(f0_lims[-1], sign="-"), get_fdot_mojito(f0_lims[-1], sign="+")],
        phi0_lims=[0.0, 2 * np.pi],
        iota_lims=[0.0 + delta_safe, np.pi - delta_safe],
        psi_lims=[0.0, np.pi],
        alpha_lims=[0.0, 2 * np.pi],
        delta_lims=[-np.pi / 2.0 + delta_safe, np.pi / 2.0 - delta_safe],
        start_freq=start_freq,
        end_freq=end_freq,
        oversample=oversample,
        extra_buffer=5,
        # None takes GBSettings' defaults; only the GB search moves, which are off, read them.
        search_kwargs=None,
        # * 0: every in-model proposal is the Fisher (FIM) proposal.
        stretch_probability=0.0,
        start_freq_ind=start_freq_ind,
        num_repeat_proposals=50,
        num_band_preload=NUM_BAND_PRELOAD,
        num_bands_preload_temp=NUM_BANDS_PRELOAD_TEMP,
        t0=general_set.data_t0,
        waveform_kwargs=waveform_kwargs,
        domain_settings=domain_settings,
        gb_wdm_comp=None,
        gb_stft_comp=STFTGBComputations,
        gb_stft_kwargs=gb_stft_kwargs,
        stft_band_alpha=general_set.window_alpha,
        debug_settings=debug_settings,
        leaf_cap_settings=None,
        sighet_settings=None,
    )
    gb_setup = GBSetup(gb_settings)

    band_edges = gb_setup.band_edges.copy()
    metadata_initialize_kwargs = initialize_kwargs.copy()
    metadata_initialize_kwargs.pop("orbits", None)
    # CPU backend in the metadata, which is serialised.
    metadata_initialize_kwargs["force_backend"] = "cpu"
    gb_metadata = SourceMetadata(
        source_type="gb",
        frequency_ranges=list(zip(band_edges[:-1], band_edges[1:])),
        waveform_model="GBGPU",
        waveform_model_code_link="https://github.com/Erebor-L2D/GBGPU/tree/cd1l-run0",
        prior_model_code_link="https://priors-database-f0027f.gitlab.io/mojito_light_1a.html#massive-black-hole-binaries-mbhb",
        waveform_model_config=dict(
            init_kwargs=metadata_initialize_kwargs, runtime_kwargs=waveform_kwargs
        ),
    )
    return gb_setup, gb_metadata


def get_general_settings(name: str) -> GeneralSetup:
    runs_dir = os.environ.get("GLOBAL_FIT_RUNS_DIR", str(DEFAULT_RUNS_DIR))
    data_input_path = os.environ.get("MOJITO_FOLDER", "/project/lisa_nlddpc/Data/mojito_light")
    # The setup classes join paths by string concatenation, so the folder ends in a slash.
    file_store_dir = os.path.join(os.path.abspath(runs_dir), name) + "/"

    dt = 5.0
    Tobs = N_SEGMENTS * STFT_SEGMENT_S
    gpus = [0] if gpu_available else None
    if gpu_available:
        cp.cuda.runtime.setDevice(gpus[0])
        # ! Restrict JAX to the same GPU before its backend starts.
        import jax

        jax.config.update("jax_cuda_visible_devices", ",".join(str(gpu) for gpu in gpus))
    # ! The effort's environment builds CUDA 13 backends only.
    backend = "cuda13x" if gpus is not None else "cpu"

    domain_settings = STFTSettings.make_factory(
        big_dt=STFT_SEGMENT_S, min_freq=START_FREQ_HZ, max_freq=END_FREQ_HZ
    )
    processor_init_kwargs = dict(
        L1_folder=data_input_path,
        source_types=["gb", "vgb", "noise"],
        source_ids=dict(),
        verbose=True,
        do_plots=True,
        orbits_class=L1Orbits,
        # On the host: the setup rebuilds these orbits on the device as gpu_orbits and never reads this
        # copy's arrays during the run, and on the device it would take 1.26 GiB.
        orbits_kwargs=dict(force_backend="cpu", frame="icrs", armlength=2493162305.42235),
    )
    # * After the trims, keep the first Tobs: the whole segments, so no partial day is left over.
    preprocess_kwargs = dict(
        highpass_kwargs={"cutoff": 1e-5, "order": 2, "zero_phase": True},
        lowpass_kwargs={"cutoff": 1e-1, "order": 2, "zero_phase": True},
        trim_kwargs={"duration": 200 * 3600, "is_percent": False, "trimming_type": "from_each_end"},
        downsample_kwargs={"target_fs": 1 / dt, "window": ("kaiser", 31.0)},
        Tobs=Tobs,
    )
    # The galactic grid gives the foreground's sky-averaged geometry at every segment centre.
    galactic_grid_kwargs = dict(R_d=2.18, z_d=0.48, N_lambda=90, N_beta=60)
    sensitivity_init_kwargs = dict(
        tdi_generation=2,
        mask_percentage=0.02,
        galactic_grid_kwargs=galactic_grid_kwargs,
        average_transfer_functions=True,
    )

    general_settings = GeneralSettings(
        Tobs=Tobs,
        dt=dt,
        file_store_dir=file_store_dir,
        base_file_name=name,
        domain_settings=domain_settings,
        random_seed=5701424,
        backup_iter=1,
        nwalkers=NWALKERS,
        ntemps=NTEMPS,
        num_iterations=MAX_ITERATIONS,
        window_type="tukey",
        window_taper_duration=WINDOW_TAPER_S,
        gpus=gpus,
        gpu_backend=backend,
        data_processor_class=L1ProcessingStep,
        processor_init_kwargs=processor_init_kwargs,
        preprocess_kwargs=preprocess_kwargs,
        normalize_window=True,
        sensitivity_backend_class=XYZSensitivityBackend,
        sensitivity_init_kwargs=sensitivity_init_kwargs,
        global_fit_codename="erebor",
        global_fit_version=name,
        global_fit_contact="ereborl2d@googlegroups.com",
        global_fit_code_link="https://github.com/Erebor-L2D",
        input_data_link=data_input_path,
        input_reference="mojito light",
        noise_model="parametric",
        noise_model_code_link="https://github.com/Erebor-L2D",
        submission_parent_folder=None,
        comment=(
            "Full-band GB, PSD and galactic foreground fit, DFFR response, from the resolved "
            "sources."
        ),
    )
    return GeneralSetup(general_settings)


def build_settings(name: str, mem_probe: bool) -> CurrentInfoGlobalFit:
    general_setup = get_general_settings(name)
    shutil.copy(__file__, general_setup.file_store_dir + name + "_" + os.path.basename(__file__))

    psd_setup, psd_metadata = get_psd_settings(general_setup)
    galfor_setup, galfor_metadata = get_galfor_settings(general_setup)
    mem_probe_file = general_setup.file_store_dir + PROBE_FILE_NAME if mem_probe else None
    gb_setup, gb_metadata = get_gb_settings(general_setup, mem_probe_file)
    global_settings = GlobalFitSettings(
        source_info={"gb": gb_setup, "psd": psd_setup, "galfor": galfor_setup},
        general_info=general_setup,
        rank_info=RankInfo(head_rank=1, main_rank=0),
        setup_function=setup_recipe,
        source_metadata={"gb": gb_metadata, "psd": psd_metadata, "galfor": galfor_metadata},
    )
    return CurrentInfoGlobalFit(global_settings)


def get_global_fit_settings():
    return build_settings(RUN_NAME, mem_probe=False)


def get_global_fit_probe_settings():
    return build_settings(RUN_NAME + "_probe", mem_probe=True)


def get_global_fit_calibration_settings():
    """One iteration with the probe, under its own run name, to calibrate the batch sizing."""
    global NUM_PE_ITERATIONS
    NUM_PE_ITERATIONS = 1
    return build_settings(RUN_NAME + "_calibration", mem_probe=True)


def get_optimisation_test_2_settings():
    """The full run with the probe, batches sized from the free memory, tempering at one width."""
    GBSpecialBase.TEMPERING_WIDTH_CLASSES = False
    return build_settings("optimisation_test_2", mem_probe=True)


def get_optimisation_test_3_settings():
    """Test 2 with tempering width classes on, the one difference between the two."""
    GBSpecialBase.TEMPERING_WIDTH_CLASSES = True
    return build_settings("optimisation_test_3", mem_probe=True)


def get_optimisation_test_4_settings():
    """Two iterations with tempering width classes on and the pool not emptied before the proposal."""
    global NUM_PE_ITERATIONS
    NUM_PE_ITERATIONS = 2
    GBSpecialBase.TEMPERING_WIDTH_CLASSES = True
    GBSpecialBase.EMPTY_POOL_BEFORE_PROPOSAL = False
    GBSpecialBase.PARALLEL_GIBBS = False
    return build_settings("optimisation_test_4", mem_probe=True)


def get_optimisation_test_5_settings():
    """Two iterations at one tempering width with the pool emptied before the proposal."""
    global NUM_PE_ITERATIONS
    NUM_PE_ITERATIONS = 2
    GBSpecialBase.TEMPERING_WIDTH_CLASSES = False
    GBSpecialBase.EMPTY_POOL_BEFORE_PROPOSAL = True
    GBSpecialBase.PARALLEL_GIBBS = False
    return build_settings("optimisation_test_5", mem_probe=True)


def get_parallel_gibbs_test_settings():
    """Two iterations at the defaults, the full-band test of the parent closed during a pass."""
    global NUM_PE_ITERATIONS
    NUM_PE_ITERATIONS = 2
    GBSpecialBase.PARALLEL_GIBBS = True
    return build_settings("parallel_gibbs_test", mem_probe=True)


def get_optimisation_test_6_settings():
    """The full run with the probe at the defaults, with every switch of the GB moves on."""
    return build_settings("optimisation_test_6", mem_probe=True)
