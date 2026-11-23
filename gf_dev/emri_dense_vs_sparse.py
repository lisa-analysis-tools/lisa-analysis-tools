import argparse
import logging
import sys
import os
from copy import deepcopy
import gc
import time


GPUS = None

import jax
if GPUS is not None:
    try:
        import cupy as cp
        cp.cuda.runtime.setDevice(GPUS[0])
    except (ImportError, ModuleNotFoundError):
        raise ImportError("CuPy is required for GPU support. Please install it with `pip install cupy`.")
    # Restrict JAX to only see the target GPU — must be set before JAX backend init
    jax.config.update("jax_cuda_visible_devices", ",".join(str(gpu) for gpu in GPUS))
else:
    jax.config.update("jax_platform_name", "cpu")

import h5py
import numpy as np

from lisatools.detector import L1Orbits
from lisatools.globalfit.preprocessing import L1ProcessingStep
from lisatools.sources.emri.waveform import EMRITDIWaveform, EMRISparseSTFTWaveform
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.domains import FDSignal, STFTSettings, TDSignal, FDSettings, TDSettings
from lisatools.utils.constants import YRSID_SI
from lisatools.utils.utility import tukey, asnumpy
from lisatools.sensitivity import XYZSensitivityBackend

from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")  # headless backend: safe for non-interactive / parallel batch runs
import matplotlib.pyplot as plt


FIGSIZE = (10, 6)
BACKEND = "cpu" if GPUS is None else "gpu"
print(f"Using {BACKEND} backend")

PATHS = {
    'gpu': "/data/asantini/globalfit/MOJITO_DATA/mojito_light_2p5s/",
    'cpu': "/Volumes/as_storage/mojito/mojito_light_2p5s"
}

CATALOG_RELATIVE_PATH = "catalogues/emri_cat_mojito_lite_processed_MT.hdf5"

STFT_DT = 1 * 24 * 3600
FREQ_MAX = 1e-1
FREQ_MIN = 1e-4
DT = 5. #s
TOBS = 24. * YRSID_SI / 12.0 # seconds

MOJITO_REFERENCE_TIME = 97729089.327664
DOMAIN = "stft"
TAPER_LENGTH = 1 / (FREQ_MIN) # seconds, set to 1 cycle of the minimum frequency to ensure good frequency resolution in the STFT

domain_factory = STFTSettings.make_factory(STFT_DT, FREQ_MIN, FREQ_MAX)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

EPS = 1e-4
ERROR_TRAJ = 1e-8

CUBIC_TOL = 1e-2
FALLBACK_VIA_POLARIZATIONS = True

path = PATHS[BACKEND]
savedir = f"./dev_emri_output/eps_{EPS:.0e}_{TOBS / YRSID_SI:.2f}/stft_traj_{ERROR_TRAJ:.0e}/"
os.makedirs(savedir, exist_ok=True)

def get(x):
    return x.get() if hasattr(x, 'get') else x

source_types = ['emri']
SOURCE_IDS = [0, 1, 2, 3, 4, 5, 6, 7]

downsample_kwargs = {
        "target_fs": 1. / DT,  # Hz — target sampling rate (None = no downsampling).
        "window": ("kaiser", 31.0)  # Kaiser window beta parameter (higher = more aggressive anti-aliasing)
    }

preprocess_kwargs = dict(
        highpass_kwargs=dict(cutoff=1e-5, order=2, zero_phase=True),
        lowpass_kwargs=dict(cutoff=0.8 * downsample_kwargs['target_fs'], order=2, zero_phase=True),
        trim_kwargs=dict(
            duration=200 * 3600, is_percent=False, trimming_type="from_each_end"
        ),
        downsample_kwargs=downsample_kwargs,
        Tobs=TOBS,
)


def prepare_data_orbits(source_id: int):
    """"""
    source_ids = dict(
        emri = [source_id]
    )

    logger.info(f"Preparing data and orbits for source ID {source_id}...")

    loader = L1ProcessingStep(
        L1_folder=path,
        source_types=source_types,
        source_ids=source_ids,
        orbits_class=L1Orbits,
        orbits_kwargs=dict(force_backend=BACKEND, frame="icrs"),
        verbose=True,
        do_plots=False,
    )

    times, _ = loader.process(**preprocess_kwargs)

    domain_settings = domain_factory(times=times, dt=loader.td_signal.settings.dt, force_backend=BACKEND)
    
    if DOMAIN == "stft":
        nperseg = domain_settings.get_nperseg(loader.td_signal.settings.dt)
    else:
        nperseg = len(times)
    
    tukey_alpha = TAPER_LENGTH / (nperseg * loader.td_signal.settings.dt)
    logger.info(f"Using Tukey window with alpha={tukey_alpha:.3f}")
    window = tukey(nperseg, alpha=tukey_alpha)

    input_data_array, orbits = loader.pour(
            settings=domain_settings, window=window, return_orbits=True
        )
    
    input_data_array.plot(filename=f"{savedir}data_{source_id}.png")

    return loader, tukey_alpha, window, input_data_array, orbits

def get_source_params(loader, source_id: int):

    binary_params = loader.catalogue['EMRI'][source_id]

    m1 = float(binary_params['PrimaryMassSSBFrame'][()].squeeze())
    m2 = float(binary_params['SecondaryMassSSBFrame'][()].squeeze())
    a = float(binary_params['PrimarySpinParameter'][()].squeeze())
    p0 = float(binary_params['SemiLatusRectum'][()].squeeze())
    e0 = float(binary_params['Eccentricity'][()].squeeze())
    xI0= float(np.cos(binary_params['InclinationAngle'][()].squeeze()))
    dist = float(binary_params['LuminosityDistance'][()].squeeze() * 1e-3) # convert to Gpc
    ra = float(binary_params['RightAscension'][()].squeeze())
    dec = float(binary_params['Declination'][()].squeeze())
    merger_time = float(binary_params["TimeCoalescenceSSBFrame"][()].squeeze())

    qK = float(binary_params['PolarAnglePrimarySpin'][()].squeeze())
    phiK = float(binary_params['AzimuthalAnglePrimarySpin'][()].squeeze())

    Phi_phi0 = float(binary_params['AzimuthalPhase'][()].squeeze())
    Phi_theta0 = float(binary_params['PolarPhase'][()].squeeze())
    Phi_r0 = float(binary_params['RadialPhase'][()].squeeze())

    snr = float(binary_params['EstimatedSNR'][()].squeeze())

    logger.info(f"snr: {snr}")


    return np.array([m1, m2, a, p0, e0, xI0, dist, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, ra, dec]), snr

def get_waveform_kwargs(orbits, td_domain_settings, window_alpha):
    """Constructor arguments shared by the dense and the sparse generator (same FEW, same response)."""
    wave_kwargs = dict(
            sum_kwargs = {"pad_output": True,},
            inspiral_kwargs = {"DENSE_STEPPING": 0,  # sparsely sampled trajectory
                "buffer_length": int(1e3),  # length of trajectories well under 1000
                "err": ERROR_TRAJ
            },
            amplitude_kwargs = {},
        )

    return dict(
            waveform_class="FastKerrEccentricEquatorialFlux",
            waveform_kwargs=wave_kwargs,
            waveform_t0=MOJITO_REFERENCE_TIME,
            data_td_settings=td_domain_settings,
            sampling_frequency=1.0 / td_domain_settings.dt,
            orbits=orbits,
            tdi_channels="XYZ",
            tdi_generation="2nd generation",
            order=30,
            force_backend=BACKEND,
            tukey_alpha=window_alpha,
            buffer_time=15000,
            stft_dt=STFT_DT,
            freq_min=FREQ_MIN,
            freq_max=FREQ_MAX,
        )


def get_dense_waveform_generator(orbits, td_domain_settings, window_alpha):
    """The production dense generator."""
    return EMRITDIWaveform(**get_waveform_kwargs(orbits, td_domain_settings, window_alpha))


def get_sparse_waveform_generator(orbits, td_domain_settings, window_alpha):
    """The sparse-track generator, on the same settings as the dense one."""
    return EMRISparseSTFTWaveform(
        **get_waveform_kwargs(orbits, td_domain_settings, window_alpha),
        cubic_tol=CUBIC_TOL, fallback_via_polarizations=FALLBACK_VIA_POLARIZATIONS,
    )


def get_templates(wave_gen, wave_params, wave_kwargs):
    """``(likelihood/residual template in the analysis domain, time-domain template, seconds each)``."""
    tic = time.perf_counter()
    template = wave_gen.get_signals_for_residuals(*wave_params, **wave_kwargs)
    t_template = time.perf_counter() - tic
    tic = time.perf_counter()
    td_template = wave_gen.get_td_signals(*wave_params, **wave_kwargs)
    t_td = time.perf_counter() - tic
    return template, td_template, (t_template, t_td)

def plot_time_domain(td_data: TDSignal, td_templates: list[TDSignal], source_id: int, labels: list[str]) -> None:
    """Plot the time-domain data, template, """
    # Restrict the data and the templates to their common time window.
    #
    # The data grid (loader dt) and the template grid (dt recovered as
    # _times[1]-_times[0]) share a nominal dt but are NOT bit-identical, so the
    # overlap span is not an exact integer multiple of either spacing. A
    # half-open boolean mask `(t_arr >= t0) & (t_arr < t1)` evaluated
    # independently on each grid can then select counts that differ by one
    # sample, which breaks the residual subtraction below. Slice the array AND
    # its settings with the SAME integer index range (compute_slice_indices),
    # and trim both to the common length so the subtraction is always defined.
    td_templates_sliced = []
    residual_templates = []
    for td_template in td_templates:
        t0 = max(td_data.settings.t0, td_template.settings.t0)
        t1 = min(td_data.settings.t0 + td_data.settings.N * td_data.settings.dt, td_template.settings.t0 + td_template.settings.N * td_template.settings.dt)

        data_slice = td_data.settings.compute_slice_indices(t0, t1)
        template_slice = td_template.settings.compute_slice_indices(t0, t1)

        n_common = min(data_slice.stop - data_slice.start, template_slice.stop - template_slice.start)
        data_slice = slice(data_slice.start, data_slice.start + n_common)
        template_slice = slice(template_slice.start, template_slice.start + n_common)

        td_data_sliced = TDSignal(arr=td_data.arr[:, data_slice], settings=td_data.get_slice(index=data_slice))
        _td_template_sliced = TDSignal(arr=td_template.arr[:, template_slice], settings=td_template.get_slice(index=template_slice))

        _residual_template = TDSignal(arr=td_data_sliced.arr - _td_template_sliced.arr, settings=td_data_sliced.settings)

        td_templates_sliced.append(_td_template_sliced)
        residual_templates.append(_residual_template)

    gridspec_kwargs = dict(height_ratios=[2, 1])
    fig, axs = plt.subplots(2, 3, figsize=(15, 9), sharex=True, gridspec_kw=gridspec_kwargs)

    for i, channel in enumerate(['X', 'Y', 'Z']):
        axs[0, i].plot(asnumpy(td_data_sliced.settings.t_arr), asnumpy(td_data_sliced.arr[i]), label='Data', color='black')
        for k, td_template_sliced in enumerate(td_templates_sliced):
            axs[0, i].plot(asnumpy(td_template_sliced.settings.t_arr), asnumpy(td_template_sliced.arr[i]), ls=':', label=labels[k], alpha=0.5)
        axs[0, i].set_title(f'Channel {channel}')

        for k, residual_template in enumerate(residual_templates):
            axs[1, i].plot(asnumpy(residual_template.settings.t_arr), asnumpy(residual_template.arr[i]), label=f'Residual {labels[k]}', alpha=0.7)

        axs[1, i].set_xlabel('Time [s]')
        
    axs[0, 0].set_ylabel('Strain')
    axs[1, 0].set_ylabel('Residual Strain')
    
    axs[0, 0].legend(loc='upper left')
    axs[1, 0].legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(f"{savedir}/time_domain_plot_{source_id}.png")

    plt.close(fig)
    return

def plot_frequency_domain(data: FDSignal, fd_templates: list[FDSignal], source_id: int) -> None:
    """Plot the frequency-domain data, template, """
    # keep only the the data within the frequency range of the template
    residual_templates = []
    for fd_template in fd_templates:
        residual_template = data.add_signal(sign=-1, template=fd_template, copy=True)
        residual_templates.append(residual_template)

    gridspec_kwargs = dict(height_ratios=[2, 1])
    fig, axs = plt.subplots(2, 3, figsize=(15, 9), sharex=True, gridspec_kw=gridspec_kwargs)

    for i, channel in enumerate(['X', 'Y', 'Z']):
        
        data.plot(channel=i, ax=axs[0, i], label='Data', color='black')
        for k, fd_template in enumerate(fd_templates):
            fd_template.plot(channel=i, ax=axs[0, i], ls='--', label=f'Template {k}')

        axs[0, i].set_title(f'Channel {channel}')
        axs[0, i].set_xlabel('')
        axs[0, i].set_ylabel('')

        for k, residual_template in enumerate(residual_templates):
            residual_template.plot(channel=i, ax=axs[1, i], ls='-', label=f'Residual Template {k}')

        axs[1, i].set_xlabel('Frequency [Hz]')
        axs[1, i].set_ylabel('')
    
    axs[0, 0].set_ylabel('Magnitude')
    axs[1, 0].set_ylabel('Residual Magnitude')

    axs[0, 0].legend(loc='lower left')
    axs[1, 0].legend(loc='lower left')

    plt.tight_layout()
    plt.savefig(f"{savedir}/frequency_domain_plot_{source_id}.png")

    plt.close(fig)
    return

def template_metrics(ac, template):
    """"""
    opt_snr, det_snr = ac.template_snr(template=template)
    d_h = ac.template_inner_product(template=template)
    d_d = ac.inner_product()

    mismatch = 1 - d_h / (np.sqrt(d_d) * opt_snr)

    indistinguishable = mismatch <= 1 / (2 * opt_snr**2)

    return {
        "optimal_snr": opt_snr,
        "detected_snr": det_snr,
        "mismatch": mismatch,
        "indistinguishable": indistinguishable,
    }

def run_one_source(source_id: int):
    
    loader, tukey_alpha, window, input_data_array, orbits = prepare_data_orbits(source_id)
    domain_settings = input_data_array.settings.__class__(*input_data_array.settings.args, **input_data_array.settings.kwargs)

    xp = orbits.backend.xp

    sensitivity_backend = XYZSensitivityBackend(
        orbits=orbits,
        settings=domain_settings,
        force_backend=BACKEND,
        tdi_generation=2,
        mask_percentage=0.05,
        window_values=window,
        average_transfer_functions=True,
    )
    logger.info("Sensitivity backend initialized.")

    Soms = 15e-12
    Sa =  3e-15

    sensitivity_backend.set_sensitivity_matrix(Soms, Sa)

    logger.info("Sensitivity matrix computed.")

    device_data_array = input_data_array.settings.associated_class(
        arr = xp.asarray(input_data_array.arr),
        settings = domain_settings
    )

    ac = AnalysisContainer(
        data=device_data_array,
        sens_mat=sensitivity_backend
    )

    logger.info("Analysis container created.")
    try:
        data_snr = ac.snr()
    except Exception as e:
        logger.error(f"Error computing SNR for source ID {source_id}: {e}")
        
        breakpoint()  # Drop into a debugger for inspection

    logger.info(f"computed SNR: {data_snr:.3f}")

    new_td_settings = TDSettings(*loader.td_signal.settings.args, t0=loader.td_signal.settings.t0, force_backend=BACKEND)

    wave_params, catalog_snr = get_source_params(loader, source_id)

    wave_kwargs = dict(mode_selection_threshold=EPS)

    # one generator at a time: each FEW generator holds several GB
    templates, td_templates, all_metrics = {}, {}, {}
    for label, builder in (("dense", get_dense_waveform_generator), ("sparse", get_sparse_waveform_generator)):
        logger.info(f"Generating {label} template for source ID {source_id}...")
        wave_gen = builder(orbits, new_td_settings, tukey_alpha)
        # the template the likelihood scores and the residual subtracts (analysis domain), and its time series
        templates[label], td_templates[label], timings = get_templates(wave_gen, wave_params, wave_kwargs)
        logger.info(f"{label}: template {timings[0]:.2f} s, time domain {timings[1]:.2f} s")
        if label == "sparse":
            mode = wave_gen.last_segment_mode[0]
            logger.info(f"sparse segments: {int((mode == 1).sum())} tracks, {int((mode == 2).sum())} exact, {int((mode == 0).sum())} empty")
        del wave_gen
        gc.collect()

        all_metrics[label] = template_metrics(ac, templates[label])
        all_metrics[label]['catalog_snr'] = catalog_snr

    # sparse against dense, noise weighted, on the dense template's segments
    dense_ac = AnalysisContainer(data=deepcopy(ac.data) * 0.0, sens_mat=sensitivity_backend)
    dense_ac.data = dense_ac.data.add_signal(sign=1, template=templates["dense"])
    all_metrics["sparse"]["mismatch_to_dense"] = template_metrics(dense_ac, templates["sparse"])["mismatch"]
    all_metrics["dense"]["mismatch_to_dense"] = 0.0

    device_td_signal = TDSignal(arr=xp.asarray(loader.td_signal.arr), settings=new_td_settings)

    plot_time_domain(td_data=device_td_signal, td_templates=list(td_templates.values()), source_id=source_id,
                     labels=list(td_templates.keys()))

    residuals = {}
    for label, template in templates.items():
        residual_ac = deepcopy(ac)
        residual_ac.data = residual_ac.data.add_signal(sign=-1, template=template)
        residuals[label] = residual_ac

    if DOMAIN == "fd":
        raise ValueError("Sparse waveform generator is only implemented for STFT domain. Please set DOMAIN='stft' in the script.")
    else:
        for label, residual_ac in residuals.items():
            residual_ac.data.plot(filename=f"{savedir}residual_{source_id}_{label}.png")

            all_metrics[label]['residual_snr'] = residual_ac.snr()

    logger.info(f"template metrics: {all_metrics}")

    return data_snr, all_metrics

def main():
    # Write one metrics file per source so that parallel jobs (e.g. Snakemake
    # fanning out over sources) never write to the same file.
    metrics_file = f"{savedir}/metrics.txt"
    with open(metrics_file, 'w') as f:
        f.write("Source ID | Catalog SNR | Data SNR | template optimal SNR | residual SNR | mismatch | indistinguishable | mismatch to dense |\n")
        f.write("-" * 120 + "\n")

    for source_id in tqdm(SOURCE_IDS, desc="Processing sources", unit="source"):
        data_snr, all_metrics = run_one_source(source_id)
        for label, metrics in all_metrics.items():
            string_to_write = f"{source_id} | {metrics['catalog_snr']:.3f} | {data_snr:.3f} | {metrics['optimal_snr']:.3f} | {metrics['residual_snr']:.3f} | {metrics['mismatch']:.5e} | {metrics['indistinguishable']} | {metrics['mismatch_to_dense']:.5e}\n"
            with open(metrics_file, 'a') as f:
                f.write(label)
                f.write("-" * 120 + "\n")
                f.write(string_to_write)
                f.write("-" * 120 + "\n")

        # clear gpu memory after each source to avoid memory leaks
        if BACKEND == "gpu":
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()

if __name__ == "__main__":

    main()