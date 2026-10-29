"""Small but realistic FD fixtures for ``XYZSensitivityBackend(convolve_window=True)``.

Usage (always import the worktree shim first)::

    import _use_worktree  # noqa: F401
    from fixtures import CONFIGS, make_backend, make_inputs

    cfg = CONFIGS["gal1_spl0_sym"]
    sm = make_backend(cfg, "cuda12x")
    inp = make_inputs(sm, num_psds=5, seed=0)      # numpy arrays (host)
    ll = sm.compute_log_like(**to_xp(inp, sm.xp)["log_like_kwargs"], convolve_window=True)

Setup (``BASE``): 2 days at dt = 5 s (N = 34560 samples, df = 5.8e-6 Hz), ESA trailing
orbits, TDI 2, a Tukey window with the production taper ``1 / min_freq`` (2e4 s ->
alpha = 0.116 on the main band) used with ``convolve_window=True``, a
filter power response ``|H(f)|^2`` from an 8th-order Butterworth low-pass at 0.08 Hz
designed at the pre-decimation rate fs = 0.4 Hz, and the D = 2 decimation alias
``f_a = -(fs' - f)`` with ``|H(|f_a|)|^2`` -- exactly what
``BaseProcessingStep.get_alias_response`` returns for ``resample_down=2`` with a
``sos_lowpass`` (p_lowpass = 2) and no FIR.

Config axes (``CONFIGS``): galaxy {off, on} x splines {off, on} x noise {sym, asym}
on the band [5e-5, 0.09] Hz, plus two band variants (galaxy on, no splines, sym):
``full_*`` (min_freq = 0, f[0] = 0 is in the grid -> exercises the f[0] guard) and
``narrow_*`` ([1e-3, 1.5e-3] Hz, the production galaxy-search band).
Aliasing is dropped (``alias=False``) for the spline configs: ``XYZSensitivityBackend``
raises ``NotImplementedError("Downsampling aliasing is not supported with use_splines")``.

All randomness is drawn on the host with ``numpy.random.default_rng(seed)``; the data
are generated from a CPU-evaluated fiducial covariance (``make_inputs`` always builds
the fiducial on a CPU twin of the backend), so CPU and GPU goldens share identical inputs.
"""

from __future__ import annotations

import copy
import itertools

import numpy as np

import _use_worktree  # noqa: F401  (must precede any lisatools import)

# --------------------------------------------------------------------------- configs

BASE = dict(
    dt=5.0,
    T_days=2.0,
    min_freq=5e-5,
    max_freq=0.09,
    window_type="tukey",
    # production rule (mojito_input/*settings.py): window_taper_duration = 1 / start_freq,
    # engine: alpha = taper / (Nt * dt). None -> derived from min_freq (5e-5 if min_freq == 0)
    window_taper_s=None,
    orbits="ESAOrbits",
    tdi_generation=2,
    mask_percentage=0.02,
    galaxy=False,
    galactic_grid_kwargs=dict(R_d=2.18, z_d=0.48, t0=0.0, N_lambda=90, N_beta=60),
    use_splines=False,
    spline_order="cubic",
    n_knots=8,
    noise_symmetry="symmetric",
    filters=True,
    lowpass=dict(order=8, fc=0.08, fs_orig=0.4, p=2),
    alias=True,
    fft_batch_size=512,
    # golden / per-agent tests compare against the full per-walker convolution
    window_convolution_mode="legacy",
    average_transfer_functions=False,
    smoothing_sigma=1.0,
)


def _build_configs() -> dict:
    cfgs = {}
    for gal, spl, sym in itertools.product((0, 1), (0, 1), ("sym", "asym")):
        name = f"gal{gal}_spl{spl}_{sym}"
        c = copy.deepcopy(BASE)
        c.update(
            name=name,
            galaxy=bool(gal),
            use_splines=bool(spl),
            noise_symmetry="symmetric" if sym == "sym" else "asymmetric",
        )
        if spl:
            c["alias"] = False
            c["dropped"] = "alias (NotImplementedError: aliasing not supported with use_splines)"
        cfgs[name] = c
    for band, (lo, hi) in {"full": (0.0, None), "narrow": (1e-3, 1.5e-3)}.items():
        name = f"{band}_gal1_spl0_sym"
        c = copy.deepcopy(BASE)
        c.update(name=name, galaxy=True, min_freq=lo, max_freq=hi)
        cfgs[name] = c
    return cfgs


CONFIGS = _build_configs()
BACKENDS = ("cpu", "cuda12x")

# --------------------------------------------------------------------------- builders


def grid_size(cfg) -> tuple[int, float]:
    """(N time samples, dt)."""
    return int(round(cfg["T_days"] * 86400.0 / cfg["dt"])), float(cfg["dt"])


def make_settings(cfg, force_backend):
    from lisatools.domains import FDSettings

    Nt, dt = grid_size(cfg)
    # == FDSettings.make_factory(min_freq, max_freq)(times, dt, force_backend)
    return FDSettings(
        N=Nt // 2 + 1, df=1.0 / (Nt * dt), min_freq=cfg["min_freq"], max_freq=cfg["max_freq"],
        force_backend=force_backend,
    )


def window_taper_s(cfg) -> float:
    if cfg.get("window_taper_s") is not None:
        return float(cfg["window_taper_s"])
    return 1.0 / (cfg["min_freq"] if cfg["min_freq"] else 5e-5)


def window_alpha(cfg) -> float:
    Nt, dt = grid_size(cfg)
    return min(1.0, window_taper_s(cfg) / (Nt * dt))


def make_window(cfg) -> np.ndarray:
    from lisatools.utils.utility import windowfun

    Nt, _ = grid_size(cfg)
    w, _ = windowfun(cfg["window_type"], Nt, alpha=window_alpha(cfg))
    return np.asarray(w, dtype=np.float64)


def lowpass_power_response(freqs, lp) -> np.ndarray:
    """|H(f)|^p of the Butterworth low-pass, as ``BaseProcessingStep.get_filter_response``."""
    from scipy.signal import butter, freqz_sos

    sos = butter(lp["order"], lp["fc"], btype="low", fs=lp["fs_orig"], output="sos")
    _, h = freqz_sos(sos, worN=np.abs(np.asarray(freqs, dtype=np.float64)), fs=lp["fs_orig"])
    return np.abs(h) ** lp["p"]


def make_filters(cfg, f_arr):
    """``(filters_response, alias_frequencies, alias_filters_response)`` on the active grid.

    Mirrors ``BaseProcessingStep.get_total_response`` / ``get_alias_response`` for an
    integer decimation D = fs_orig * dt (here 2).
    """
    f_arr = asnumpy(f_arr).astype(np.float64)
    if not cfg["filters"]:
        return None, None, None
    lp = cfg["lowpass"]
    filters_response = lowpass_power_response(f_arr, lp)
    if not cfg["alias"]:
        return filters_response, None, None
    D = int(round(lp["fs_orig"] * cfg["dt"]))
    fs_new = lp["fs_orig"] / D
    f = np.abs(f_arr)
    alias = [-(k * fs_new - f) for k in range(1, D // 2 + 1)]
    alias += [k * fs_new + f for k in range(1, (D - 1) // 2 + 1)]
    alias_frequencies = np.array(alias)
    alias_filters_response = lowpass_power_response(np.abs(alias_frequencies).ravel(), lp).reshape(
        alias_frequencies.shape
    )
    return filters_response, alias_frequencies, alias_filters_response


def make_orbits(cfg, force_backend):
    from lisatools import detector

    return getattr(detector, cfg["orbits"])(force_backend=force_backend)


def backend_kwargs(cfg, force_backend) -> dict:
    """Everything ``XYZSensitivityBackend.__init__`` gets (host arrays), minus orbits."""
    settings = make_settings(cfg, force_backend)
    fr, af, afr = make_filters(cfg, settings.f_arr)
    return dict(
        settings=settings,
        tdi_generation=cfg["tdi_generation"],
        use_splines=cfg["use_splines"],
        spline_order=cfg["spline_order"],
        force_backend=force_backend,
        mask_percentage=cfg["mask_percentage"],
        galactic_grid_kwargs=dict(cfg["galactic_grid_kwargs"]) if cfg["galaxy"] else None,
        window_values=make_window(cfg),
        convolve_window=True,
        filters_response=fr,
        average_transfer_functions=cfg["average_transfer_functions"],
        smoothing_sigma=cfg["smoothing_sigma"],
        noise_symmetry=cfg["noise_symmetry"],
        fft_batch_size=cfg["fft_batch_size"],
        window_convolution_mode=cfg["window_convolution_mode"],
        alias_frequencies=af,
        alias_filters_response=afr,
    )


def make_backend(config, force_backend: str = "cpu", **overrides):
    """Build the ``XYZSensitivityBackend`` for ``config`` (a name or a CONFIGS dict).

    ``overrides`` are passed straight to ``XYZSensitivityBackend`` (e.g. new kwargs
    such as ``window_convolution_mode`` / ``psd_batch_size``).
    """
    from lisatools.sensitivity import XYZSensitivityBackend

    cfg = CONFIGS[config] if isinstance(config, str) else config
    kw = backend_kwargs(cfg, force_backend)
    kw.update(overrides)
    sm = XYZSensitivityBackend(orbits=make_orbits(cfg, force_backend), **kw)
    sm._fixture_config = cfg
    return sm


# --------------------------------------------------------------------------- inputs

FIDUCIAL = dict(Soms=15e-12, Sa=3e-15, Amp=10 ** -43.7, alpha=1.8, f_1=2e-3, kn=3e-3, f_2=8e-4)


def _draw_params(rng, cfg, num_psds, n_freq_lo, n_freq_hi):
    n_par = 6 if cfg["noise_symmetry"] == "asymmetric" else 1
    shape = (num_psds, n_par) if n_par > 1 else (num_psds,)
    p = dict(
        Soms=FIDUCIAL["Soms"] * rng.uniform(0.8, 1.2, shape),
        Sa=FIDUCIAL["Sa"] * rng.uniform(0.8, 1.2, shape),
    )
    if cfg["galaxy"]:
        p.update(
            Amp=10 ** rng.uniform(-44.2, -43.2, num_psds),
            alpha=rng.uniform(1.2, 2.5, num_psds),
            f_1=10 ** rng.uniform(-3.0, -2.4, num_psds),
            kn=10 ** rng.uniform(-2.8, -2.2, num_psds),
            f_2=10 ** rng.uniform(-3.4, -2.8, num_psds),
        )
    else:
        p.update({k: np.zeros(num_psds) for k in ("Amp", "alpha", "f_1", "kn", "f_2")})
    if cfg["use_splines"]:
        # same construction as tests/test_noise_covariance_batched.py (per-walker knots,
        # end knots pinned just outside the band), amplitudes = log10 multiplicative residual
        k = cfg["n_knots"]
        lo, hi = n_freq_lo - 0.1, n_freq_hi + 0.1
        inner = np.sort(rng.uniform(lo, hi, (2, num_psds, k - 2)), axis=-1)
        p["knots_position_all"] = np.concatenate(
            [np.full((2, num_psds, 1), lo), inner, np.full((2, num_psds, 1), hi)], axis=-1
        )
        p["knots_amplitude_all"] = 0.1 * rng.standard_normal((2, num_psds, k))
    else:
        p["knots_position_all"] = None
        p["knots_amplitude_all"] = None
    return p


def _fiducial_cov(cfg, f_pos):
    """Host (nf, 3, 3) fiducial covariance (convolve_window=False) from a CPU twin."""
    sm = make_backend(cfg, "cpu")
    n_par = sm.n_noise_par
    Soms = np.full(n_par, FIDUCIAL["Soms"]) if n_par > 1 else FIDUCIAL["Soms"]
    Sa = np.full(n_par, FIDUCIAL["Sa"]) if n_par > 1 else FIDUCIAL["Sa"]
    gal = [FIDUCIAL[k] for k in ("Amp", "alpha", "f_1", "kn", "f_2")] if cfg["galaxy"] else [0.0] * 5
    kp = ka = None
    if cfg["use_splines"]:
        kp = np.stack([np.linspace(f_pos[0] - 0.1, f_pos[1] + 0.1, cfg["n_knots"])] * 2)
        ka = np.zeros_like(kp)
    C = np.asarray(
        sm.compute_sensitivity_matrix(Soms, Sa, *gal, kp, ka, convolve_window=False)
    )  # (3, 3, nf)
    return np.moveaxis(C, (0, 1), (-2, -1)), sm


def _sqrtm_psd(C):
    """Hermitian square root per bin, robust to (near-)singular bins (dips, f = 0)."""
    C = np.where(np.isfinite(C), C, 0.0)
    C = 0.5 * (C + np.conj(np.swapaxes(C, -1, -2)))
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)[..., None, :]) @ np.conj(np.swapaxes(V, -1, -2))


def make_inputs(backend, num_psds: int = 5, seed: int = 0) -> dict:
    """Random-but-physical walker parameters + Gaussian-noise data for ``backend``.

    Returns host (numpy) arrays; use :func:`to_xp` to move them to the backend.
    ``log_like_kwargs`` holds exactly the keyword arguments of
    ``XYZSensitivityBackend.compute_log_like`` (minus ``convolve_window``).
    Data: ``num_psds`` independent noise realisations from the fiducial covariance
    (the binding requires ``len(data) == num_psds * 3 * num_freqs``), layout
    ``(num_psds, 3, num_freqs)`` flattened walker-major / channel-major, scaled so that
    ``2 df d^H C^-1 d ~ 3`` per bin (as in tests/test_noise_covariance_batched.py).
    ``data_index_all`` is a non-trivial map walker -> data block (a shuffled draw with
    repeats, so the kernel's indexing is exercised).
    """
    cfg = backend._fixture_config
    rng = np.random.default_rng(seed)
    f_arr = asnumpy(backend.basis_settings.f_arr)
    f_pos = f_arr[f_arr > 0]
    logf = (np.log10(f_pos[0]), np.log10(f_pos[-1]))

    params = _draw_params(rng, cfg, num_psds, *logf)

    Cfid, _ = _fiducial_cov(cfg, logf)
    L = _sqrtm_psd(Cfid)  # (nf, 3, 3)
    nf = f_arr.size
    num_data = num_psds
    z = (rng.standard_normal((num_data, nf, 3)) + 1j * rng.standard_normal((num_data, nf, 3))) / np.sqrt(2)
    d = np.einsum("fij,nfj->nfi", L, z) / np.sqrt(2.0 * backend.basis_settings.differential_component)
    data = np.ascontiguousarray(np.swapaxes(d, -1, -2))  # (num_data, 3, nf)

    data_index_all = rng.integers(0, num_data, num_psds).astype(np.int32)
    data_index_all[0] = num_data - 1 if data_index_all[0] == 0 else 0  # never the identity

    llk = dict(
        data_in_all=data.ravel(),
        data_index_all=data_index_all,
        Soms_in_all=params["Soms"],
        Sa_in_all=params["Sa"],
        Amp_in_all=params["Amp"],
        alpha_in_all=params["alpha"],
        f_1_in_all=params["f_1"],
        kn_in_all=params["kn"],
        f_2_in_all=params["f_2"],
        knots_position_all=params["knots_position_all"],
        knots_amplitude_all=params["knots_amplitude_all"],
    )
    return dict(log_like_kwargs=llk, num_psds=num_psds, num_data=num_data, seed=seed)


def sens_mat_kwargs(inputs: dict, n: int = 2) -> dict:
    """``compute_sensitivity_matrix`` kwargs for the first ``n`` walkers of ``inputs``."""
    llk = inputs["log_like_kwargs"]
    kw = dict(
        Soms_d_in=llk["Soms_in_all"][:n],
        Sa_a_in=llk["Sa_in_all"][:n],
        Amp=llk["Amp_in_all"][:n],
        alpha=llk["alpha_in_all"][:n],
        f_1=llk["f_1_in_all"][:n],
        kn=llk["kn_in_all"][:n],
        f_2=llk["f_2_in_all"][:n],
        knots_position_all=None,
        knots_amplitude_all=None,
    )
    if llk["knots_position_all"] is not None:
        kw["knots_position_all"] = np.ascontiguousarray(llk["knots_position_all"][:, :n])
        kw["knots_amplitude_all"] = np.ascontiguousarray(llk["knots_amplitude_all"][:, :n])
    return kw


def to_xp(obj, xp):
    """Recursively move numpy arrays in ``obj`` (dict/list/array) to ``xp``."""
    if isinstance(obj, dict):
        return {k: to_xp(v, xp) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_xp(v, xp) for v in obj)
    if isinstance(obj, np.ndarray):
        return xp.asarray(obj)
    return obj


def asnumpy(x):
    if x is None:
        return None
    try:
        import cupy as cp

        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    except ImportError:
        pass
    return np.asarray(x)
