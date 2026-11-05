"""Global-fit settings: instrument noise + galactic foreground parameter estimation on 1 year of Mojito light noise
plus a simulated galactic confusion foreground (Quentin's galaxy, TD XYZ at 5 s, upsampled and added to the 2.5 s
noise before preprocessing), with the galaxy sky modulation FIXED.

Built from psd_multigpu_settings.py (PSD branch, MultiGPUPSDMove) and galfor_psd_multigpu_stft_settings.py
(galfor branch), without the GB branch:
  - 1 year, dt = 5 s, 1-day STFT, 0.1-99.99 mHz, Tukey window with taper 1 / f_min;
  - covariance convolved with the window spectrum along frequency, filter response + decimation alias + upstream
    Mojito anti-aliasing filter (upstream_aafilter_kwargs), TDI dips kept (mask_percentage = 0), no smoothing;
  - closed-form galaxy response (response="analytic"), 180 x 120 sky grid, disk + bulge with refined line-of-sight
    quadrature; modulation (R_d, z_d, f_b, r_b) fixed at the MAP of a direct 1-year MCMC with free modulation;
  - sampled: log10 S_oms, log10 S_tm (symmetric noise) and log10 A, alpha, log10 f_1, log10 f_knee, log10 f_2.

Run (single process, one GPU):  python scripts/run_global.py -sfp mojito_input/galfor_psd_quentin_galaxy_settings.py
Optional environment variables: GF_NUM_ITERATIONS (default 400), GF_NWALKERS (default 24; every walker keeps a full
3 x 3 covariance on the GPU, ~0.45 GB here: 12 walkers fit next to another job on a 40 GB A100, 24 need more room),
GF_PSD_BATCH (PSDs per likelihood call, default all at once). L1_FOLDER, GAL_FILE and OUT_DIR below are paths on
the AEI cluster.
"""

import logging
import os
import shutil

import numpy as np
import cupy as cp
from scipy import signal

from eryn.moves import TemperatureControl
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.utils import TransformContainer

from lisatools.detector import L1Orbits
from lisatools.domains import STFTSettings
from lisatools.globalfit.engine import GeneralSettings, GeneralSetup, GlobalFitSettings, RankInfo
from lisatools.globalfit.moves import MultiGPUPSDMove
from lisatools.globalfit.postprocessing import StochasticMetadata
from lisatools.globalfit.preprocessing import L1ProcessingStep
from lisatools.globalfit.recipe import PERecipeStep, SearchRecipeStep
from lisatools.globalfit.run import CurrentInfoGlobalFit
from lisatools.globalfit.stock.erebor import GalForSettings, GalForSetup, PSDSettings, PSDSetup
from lisatools.sensitivity import XYZSensitivityBackend
from lisatools.utils.constants import YRSID_SI

logger = logging.getLogger("lisatools.settings." + __name__)

MOJITO_AVERAGE_ARMLENGTH = 2493162305.42235
L1_FOLDER = "/data/asantini/globalfit/MOJITO_DATA/mojito_light_2p5s/"
GAL_FILE = "/work/fpozzoli/foreground_quentin/confusion_noise_tditd_mojito_2yrs.npy"  # galaxy only, TD XYZ, dt = 5 s
OUT_DIR = "/work/fpozzoli/foreground_quentin/globalfit_1yr/output/"
# Kaiser FIR fitted to the anti-aliasing filter of the Mojito L0 simulation (4 Hz -> 0.4 Hz decimation)
MOJITO_UPSTREAM_AAFILTER = dict(attenuation=186.2, freq1=0.09432, freq2=0.20585, fs=4.0)
# MAP of the direct 1-year MCMC (symmetric noise, same covariance model) with free modulation
MODULATION = dict(R_d=1.99506849, z_d=0.16776102, f_b=0.64907336, r_b=1.71475300)

LOG10_OMS_ASD_RANGE = (-12.0, -10.0)
LOG10_TM_ASD_RANGE = (-16.0, -13.0)
LOG10_AMP_RANGE = (-48.0, -40.0)
ALPHA_RANGE = (1.0, 20.0)  # widened to the right as in the direct MCMC (FIT_ALPHA_MAX=20)
LOG10_FREQ1_RANGE = (-4.0, -2.0)
LOG10_FKNEE_RANGE = (-3.0, -1.0)
LOG10_FREQ2_RANGE = (-4.0, -2.0)


def ten_to_the_x(x):
    return 10.0 ** x


class MojitoNoisePlusQuentinGalaxy(L1ProcessingStep):
    """Mojito light noise (L1, 2.5 s) plus Quentin's galaxy, upsampled 5 s -> 2.5 s and added before preprocessing."""

    def __init__(self, galaxy_file: str, **kwargs):
        super().__init__(**kwargs)
        gal = np.load(galaxy_file, mmap_mode="r")
        gal_up = signal.resample_poly(np.vstack([np.asarray(gal[c]) for c in "XYZ"]), up=2, down=1, axis=-1,
                                      window=("kaiser", 31.0))[:, : self.data.shape[1]]
        self.data = self.data + gal_up
        logger.info(f"Added the galaxy of {galaxy_file} to the noise ({gal_up.shape[1]} samples at 2.5 s).")


def setup_recipe(recipe, engine_info, curr, acs, priors, state):
    general_info = curr.general_info
    nwalkers, ntemps = general_info.nwalkers, general_info.ntemps
    cp.cuda.runtime.setDevice(general_info.gpus[0])

    psd_info = curr.source_info["psd"]
    galfor_info = curr.source_info["galfor"]
    # PSD and foreground parameters share one branch pair in the move: ladder sized on the combined dimension
    effective_ndim = engine_info.ndims["psd"] + engine_info.ndims["galfor"]
    temperature_control = TemperatureControl(effective_ndim, nwalkers, ntemps=ntemps, Tmax=1.0e6, permute=False)
    psd_move_kwargs = dict(
        num_repeats=psd_info.num_prop_repeats,
        permute_every=200,
        live_dangerously=True,
        psd_transform_fn=psd_info.transform,
        galfor_transform_fn=galfor_info.transform,
        temperature_control=temperature_control,
        use_gpu=True,
        run_async=True,
        run_threaded=False,
    )
    psd_search_move = MultiGPUPSDMove(acs, priors, max_logl_mode=True, name="psd search move", **psd_move_kwargs)
    psd_pe_move = MultiGPUPSDMove(acs, priors, max_logl_mode=False, name="psd pe move", **psd_move_kwargs)
    psd_search_move.accepted = np.zeros((ntemps, nwalkers))
    psd_pe_move.accepted = np.zeros((ntemps, nwalkers))
    recipe.add_recipe_component(SearchRecipeStep(moves=[psd_search_move]), name="psd+galfor search")
    recipe.add_recipe_component(PERecipeStep(moves=[psd_pe_move]), name="psd+galfor pe")


def get_psd_erebor_settings(general_set: GeneralSetup):
    names = [r"$\log_{10} S_{\rm oms}$", r"$\log_{10} S_{\rm tm}$"]
    prior_model_config = dict(zip(names, (LOG10_OMS_ASD_RANGE, LOG10_TM_ASD_RANGE)))
    psd_settings = PSDSettings(
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        initialize_kwargs=dict(),
        priors={"psd": ProbDistContainer({n: uniform_dist(*prior_model_config[n]) for n in names})},
        ndim=2,
        injection=np.array([np.log10(15e-12), np.log10(3e-15)]),  # for diagnostic plots
        log_dir=general_set.artifacts_file_dir,
        num_prop_repeats=50,
        transform=TransformContainer(input_basis=names, output_basis=names,
                                     parameter_transforms={n: ten_to_the_x for n in names}),
    )
    psd_metadata = StochasticMetadata(
        model_config=dict(use_splines=False, num_params=2, noise_symmetry="symmetric"),
        frequency_ranges=[(general_set.start_freq, general_set.end_freq)],
        prior_model="uniform",
        prior_model_code_link="",
        prior_model_config=prior_model_config,
    )
    return PSDSetup(psd_settings), psd_metadata


def get_galfor_erebor_settings(general_set: GeneralSetup):
    names = [r"$\log_{10} A_{\rm gal}$", r"$\alpha_{\rm gal}$", r"$\log_{10} f_1$", r"$\log_{10} f_{\rm knee}$",
             r"$\log_{10} f_2$"]
    prior_model_config = dict(zip(names, (LOG10_AMP_RANGE, ALPHA_RANGE, LOG10_FREQ1_RANGE, LOG10_FKNEE_RANGE,
                                          LOG10_FREQ2_RANGE)))
    galfor_settings = GalForSettings(
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        initialize_kwargs={},
        transform=TransformContainer(input_basis=names, output_basis=names,
                                     parameter_transforms={n: ten_to_the_x for n in names if "alpha" not in n}),
        priors={"galfor": ProbDistContainer({n: uniform_dist(*prior_model_config[n]) for n in names})},
        ndim=5,
    )
    galfor_metadata = StochasticMetadata(
        model_config=dict(num_params=5,
                          galactic_grid_kwargs=general_set.sensitivity_init_kwargs.get("galactic_grid_kwargs", {})),
        frequency_ranges=[(general_set.start_freq, general_set.end_freq)],
        prior_model="uniform",
        prior_model_code_link="",
        prior_model_config=prior_model_config,
    )
    return GalForSetup(galfor_settings), galfor_metadata


def get_general_erebor_settings() -> GeneralSetup:
    Tobs, dt = YRSID_SI, 5.0
    start_freq, end_freq = 1e-4, 0.09999  # last bin below the 0.1 Hz Nyquist of the 5 s data
    gpus = [0]  # first visible device (set CUDA_VISIBLE_DEVICES; under condor: the device assigned to the job)
    cp.cuda.runtime.setDevice(gpus[0])
    backend = "cuda12x"

    processor_init_kwargs = dict(
        galaxy_file=GAL_FILE,
        L1_folder=L1_FOLDER,
        source_types=["noise"],
        source_ids=dict(),
        verbose=True,
        do_plots=True,
        orbits_class=L1Orbits,
        orbits_kwargs=dict(force_backend=backend, frame="icrs", armlength=MOJITO_AVERAGE_ARMLENGTH),
    )
    preprocess_kwargs = dict(
        highpass_kwargs={"cutoff": 1e-5, "order": 2, "zero_phase": True},
        lowpass_kwargs={"cutoff": 1e-1, "order": 2, "zero_phase": True},
        trim_kwargs={"duration": 200 * 3600, "is_percent": False, "trimming_type": "from_each_end"},
        downsample_kwargs={"target_fs": 1 / dt, "window": ("kaiser", 31.0)},
        Tobs=Tobs,
        alias_correction=True,
        upstream_aafilter_kwargs=MOJITO_UPSTREAM_AAFILTER,
    )
    galactic_grid_kwargs = dict(
        **MODULATION,
        N_lambda=180,
        N_beta=120,
        los_center_half_width=2.5,
        los_center_sub=4,
        response="analytic",
    )
    sensitivity_init_kwargs = dict(
        tdi_generation=2,
        mask_percentage=0.0,
        smoothing_sigma=0.0,
        average_transfer_functions=True,
        noise_symmetry="symmetric",
        convolve_window=True,
        galactic_grid_kwargs=galactic_grid_kwargs,
        # PSDs per backend call (one 3x3 x 365 x 8641 complex covariance is ~0.45 GB); None = all at once
        psd_batch_size=int(os.environ["GF_PSD_BATCH"]) if os.environ.get("GF_PSD_BATCH") else None,
    )
    general_settings = GeneralSettings(
        num_iterations=int(os.environ.get("GF_NUM_ITERATIONS", 400)),
        Tobs=Tobs,
        dt=dt,
        file_store_dir=OUT_DIR,
        base_file_name="galfor_psd_quentin_1yr",
        domain_settings=STFTSettings.make_factory(big_dt=86400.0, min_freq=start_freq, max_freq=end_freq),
        random_seed=20261030,
        backup_iter=5,
        nwalkers=int(os.environ.get("GF_NWALKERS", 24)),  # one full covariance per walker is kept on the GPU
        ntemps=4,
        window_type="tukey",
        window_taper_duration=1 / start_freq,
        gpus=gpus,
        gpu_backend=backend,
        data_processor_class=MojitoNoisePlusQuentinGalaxy,
        processor_init_kwargs=processor_init_kwargs,
        preprocess_kwargs=preprocess_kwargs,
        normalize_window=True,
        sensitivity_backend_class=XYZSensitivityBackend,
        sensitivity_init_kwargs=sensitivity_init_kwargs,
        global_fit_codename="erebor",
        global_fit_version="galfor_psd_quentin_1yr",
        global_fit_contact="",
        global_fit_code_link="",
        input_data_link=L1_FOLDER,
        input_reference="mojito light noise + Quentin galaxy",
        noise_model="parametric-symmetric",
        noise_model_code_link="",
        submission_parent_folder=None,
        comment="PSD + galactic foreground PE, modulation fixed at the MAP of the direct 1-year MCMC.",
    )
    return GeneralSetup(general_settings)


def get_global_fit_settings(copy_settings_file=True):
    general_setup = get_general_erebor_settings()
    if copy_settings_file:
        shutil.copy(__file__, general_setup.file_store_dir + general_setup.base_file_name + "_" + os.path.basename(__file__))
    psd_setup, psd_metadata = get_psd_erebor_settings(general_setup)
    galfor_setup, galfor_metadata = get_galfor_erebor_settings(general_setup)
    global_settings = GlobalFitSettings(
        source_info={"psd": psd_setup, "galfor": galfor_setup},
        general_info=general_setup,
        rank_info=RankInfo(head_rank=1, main_rank=0),
        setup_function=setup_recipe,
        source_metadata={"psd": psd_metadata, "galfor": galfor_metadata},
    )
    return CurrentInfoGlobalFit(global_settings)
