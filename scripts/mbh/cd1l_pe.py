#!/usr/bin/env python
"""Bayesian PE for one CD1-L MBHB source, WDM likelihood, instrument noise only.

Basis is the VALIDATED one -- `mbh_catalogue_to_sampling_basis` + MBH_TRANSFORM,
the same path cd1l_match_check.py reproduces the data with at mm=1.5e-6:

  [logM, Q=m1/m2, s1z, s2z, dist(Gpc), phi_ref, cos_iota, psi, ra, sin_dec, t_plunge]

Deliberately NOT the (mT, q, lam, sinbeta) basis of mbh_test_script_td_wave.py:
switching bases would break the chain of custody from that mismatch number.

vectorize=False is a MEASURED choice, not an oversight. Profiling on an A6000
(job 13908029) gives per-row cost 0.508 s at B=1 rising monotonically to
0.642 s at B=16, because PR#81 batches the response but LOOPS the per-source
WDM transform (48% of the cost). Feeding Eryn's 16-walker half-blocks would
cost ~26% more, so rows are evaluated one at a time.

Env: MBHB_ID, CD1L_NWALKERS, CD1L_NTEMPS, CD1L_NSTEPS, CD1L_INIT_SCALE, CD1L_OUT
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import h5py
import cd1l_match_check as M

from lisatools.domains import (TDSettings, WDMSettings, TDSignal,
                               place_td_signal_on_grid)
from lisatools.sources.bbh.gridaligned import GridAlignedPhenomTHMTDIWaveform
from lisatools.sources.batching import BatchedDomainSignalGen
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix

from eryn.ensemble import EnsembleSampler
from eryn.prior import ProbDistContainer
try:                                    # eryn.prior.uniform_dist is deprecated
    from eryn.priors.analytical import UniformDistribution as uniform_dist
except ImportError:                     # older eryn
    from eryn.prior import uniform_dist
from eryn.state import State
from eryn.backends import HDFBackend
from eryn.utils import PeriodicContainer
from eryn.moves import StretchMove

ID        = int(os.environ.get("MBHB_ID", "0"))
NWALKERS  = int(os.environ.get("CD1L_NWALKERS", "32"))
NTEMPS    = int(os.environ.get("CD1L_NTEMPS", "1"))
NSTEPS    = int(os.environ.get("CD1L_NSTEPS", "200"))
INIT_SC   = float(os.environ.get("CD1L_INIT_SCALE", "1e-4"))
OUT       = os.environ.get("CD1L_OUT", "/data/nbody/majoburo/cd1l_pe/runs")
# Inputs shared by every run (SNR ladder, data cache) live in the base runs
# directory; OUT may be a per-experiment subdirectory (e.g. roulet_spins/).
RUNS_BASE = os.environ.get("CD1L_RUNS", "/data/nbody/majoburo/cd1l_pe/runs")
ROULET_SPINS = os.environ.get("CD1L_ROULET_SPINS", "0") == "1"
ROULET_STEP_SCALE = float(os.environ.get("CD1L_ROULET_STEP_SCALE", "1.0"))
# CD1L_TDET=1: sample the arrival time at the constellation centre instead of
# t_SSB (see S.TDetTransform). Removes 90-98% of the t_plunge excess over the
# Fisher -- the part that is the LISA light delay -- which is the only reason a
# Gaussian proposal cannot span the sky/t ridge.
TDET = os.environ.get("CD1L_TDET", "0") == "1"
if ROULET_SPINS:
    # Isolate experiments while retaining downstream MC chain conventions.
    OUT = os.path.join(OUT, "roulet_spins")
EPOCH_K   = int(os.environ.get("CD1L_EPOCH_K", "0"))   # 0 = full window, -1 = merger+1d
# Optimised defaults, each verified against the mm=1.53e-06 baseline:
#   order 30->8   1.59x, mm 1.5252e-06 (BETTER than order=30; mm is flat 30..4)
#   dt 2.5->10 s  2.66x, mm 1.93e-06 (L/c=8.3 s sub-sample is harmless here)
#   batching      2.09x at B>=16 (see vectorize below)
ORDER     = int(os.environ.get("CD1L_ORDER", "8"))
NTAP      = 400                                        # overwritten below
DECIMATE  = int(os.environ.get("CD1L_DECIMATE", "4"))  # dt = 2.5 * DECIMATE
# Taper width is a DURATION, not a sample count. The ladder defines its epochs
# with 400 samples at dt=2.5 s = 1000 s; at dt=10 s the same 400 samples would
# be 4000 s and eat extra signal near merger (measured: realised SNR 168.8 vs a
# 183.1 target, job 13918172). Scale it so the physical taper is invariant.
NTAP_BASE = int(os.environ.get("CD1L_NTAP", "400"))   # samples at dt=2.5 s
os.makedirs(OUT, exist_ok=True)

# SAMPLER: "mc" (default) = chirp-mass basis + Fisher-Gaussian / sky-partner
# Gibbs / prior-draw moves ported from osearch run_pe_v0.py; "stock" = the
# original (ln M, Q) basis with StretchMove only (kept for the comparison
# chains). See cd1l_sampling.py for why.
SAMPLER   = os.environ.get("CD1L_SAMPLER", "mc")
if ROULET_SPINS and SAMPLER != "mc":
    raise ValueError("CD1L_ROULET_SPINS=1 requires CD1L_SAMPLER=mc")
import cd1l_sampling as S
STOCK_BASIS = ["logM", "Q", "s1z", "s2z", "dist", "phi_ref",
               "cos_iota", "psi", "ra", "sin_dec", "t_plunge"]
BASIS = S.BASIS_MC if SAMPLER == "mc" else STOCK_BASIS
NDIM = len(BASIS)
TRANSFORM = S.make_mc_transform_container() if SAMPLER == "mc" else M.MBH_TRANSFORM

# ---------------------------------------------------------------- data + gen
data_td, window_t0, _, cat, abs_merger = M.load_data_cached()
if TDET:
    # Built here rather than later only so the vector is read before
    # get_wave_gen() drops the shared base orbit; get_orbits memoizes, so
    # the generator below reuses this same object at no extra cost.
    DVEC = S.delay_vector(M.get_orbits(window_t0), abs_merger)
    TRANSFORM = S.TDetTransform(TRANSFORM, DVEC)
    print(f"  CD1L_TDET on: r_centre/c (ICRS, s) = "
          f"[{DVEC[0]:.2f}, {DVEC[1]:.2f}, {DVEC[2]:.2f}]  |d| = "
          f"{np.linalg.norm(DVEC):.2f} s = {np.linalg.norm(DVEC)/S.AU_LIGHT_S:.4f} AU", flush=True)
DT = M.DT
NF, NT, N_WIN = M.NF, M.NT, M.N_WIN

if DECIMATE > 1:
    # Anti-aliased decimation (NOT the naive subsampling the loader refuses):
    # zero-phase FIR, and the template is generated natively at the new dt, so
    # both sides are band-limited consistently.
    from scipy.signal import decimate as _dec
    data_td = _dec(data_td, DECIMATE, axis=-1, ftype="fir", zero_phase=True)
    DT = M.DT * DECIMATE
    NF, NT, _wd = WDMSettings.adjust_to_even_bins(
        0.5 * 86400.0, 0.75 * 86400.0, DT, data_td.shape[-1] * DT)
    N_WIN = NF * NT
    if N_WIN > data_td.shape[-1]:
        NT -= 2
        N_WIN = NF * NT
    data_td = np.ascontiguousarray(data_td[:, :N_WIN])
NTAP = max(4, NTAP_BASE // DECIMATE)      # same 1000 s taper at any dt
TAG = "full"
TAG_SUFFIX = "_mc" if SAMPLER == "mc" else ""
TUKEY_A = M.TUKEY_ALPHA
N_KEEP = N_WIN          # samples actually observed; == N_WIN for the full run
_epoch = None

if EPOCH_K != 0:
    # WHY NOT SHORTEN THE WDM WINDOW: one WDM time layer is NF*dt =
    # 17280*2.5 = 43200 s = 12 h, so an NF x NT window can only be truncated in
    # 12 h steps. The SNR-halving epochs sit 10 min - 2 h before merger, where
    # SNR doubles every ~14 min. Measured (job 13908791): asking for
    # t_merger-1.8 h gave a window at t_merger-13.4 h and SNR 80.4 against a
    # target of 183.1 -- essentially the k=2 rung. Going to a finer WDM grid
    # would fix time resolution but destroy frequency resolution (NF~240 gives
    # df 8.3e-4 Hz, ~30 bins across the band) and would no longer be the CD1-L
    # WDM resolution the task specifies.
    #
    # So: keep the CD1-L grid EXACTLY and truncate in the TIME domain, where
    # the resolution is dt = 2.5 s. The same window multiplies the data and the
    # template, so <d-h|d-h> compares like with like.
    import json as _json
    _lad = _json.load(open(os.path.join(RUNS_BASE, "cd1l_snr_ladder.json")))
    _m = [e for e in _lad[str(ID)]["epochs"] if e["k"] == EPOCH_K]
    if not _m:
        raise SystemExit(f"id{ID} has no epoch k={EPOCH_K} in the ladder")
    _epoch = _m[0]
    # ladder n_keep is in ORIGINAL dt=2.5 s samples; rescale to this grid
    N_KEEP = int(_epoch["n_keep"] // DECIMATE)
    # EPOCH MECHANISM. "wdm" (default for the post-merger anchor): keep the
    # validated k=0 machinery (batched grid-aligned generator, full-grid Tukey
    # window) and express "observation ends at t_cut" as the WDM active time
    # band (WDMSettings.max_time; pixel = NF*dt = 12 h). Data and template are
    # both restricted to the same pixels through the sensitivity/active slice,
    # nothing is truncated in the time domain. Measured on id19 (SNR 3589):
    # the single-row TD-truncation path has O(1) jaggedness in log L on the
    # 1e-5 rad scale in the sky (grows with response order: 2 smooth, 4 mild,
    # 8 O(1), 30 O(5)) while the k=0 path is smooth. 12 h pixels cannot
    # express the SNR-halving rungs (10 min - 2 h before merger), so those
    # keep the TD path ("td"); at their lower SNR the jaggedness is SNR^2
    # smaller.
    EPOCH_MODE = os.environ.get("CD1L_TRUNC", "wdm" if EPOCH_K == -1 else "td")
    # k=-1 is the merger+1 day post-merger rung (SNR saturated, = full), kept
    # as its own epoch to match the previous "merger+1day" reference convention.
    TAG = {-1: "post1d", 99: "snr10"}.get(EPOCH_K, f"k{EPOCH_K}")
    # Shrink the GRID to the smallest whole number of 12 h WDM layers that still
    # contains N_KEEP. The truncation itself stays at N_KEEP exactly (dt=2.5 s)
    # via the window, so this changes cost, not the analysed segment: samples
    # beyond N_KEEP are zero anyway, and carrying the full 48 d grid made every
    # epoch run cost MORE than the k=0 run it is a subset of.
    TD_SHRINK = os.environ.get("CD1L_TD_SHRINK", "1") == "1"     # diagnostic knobs
    TD_WINDOW = os.environ.get("CD1L_TD_WINDOW", "trunc")        # trunc | tukey
    if EPOCH_MODE == "td" and TD_SHRINK:
        NT = int(np.ceil(N_KEEP / NF))
        if NT % 2:            # the WDM transform is built on EVEN bin counts
            NT += 1           # (cf. WDMSettings.adjust_to_even_bins); NT=69 gave
                              # "operands could not be broadcast (2152,69) (1,68)"
        N_WIN = NF * NT
        assert N_WIN >= N_KEEP, (N_WIN, N_KEEP)
        data_td = data_td[:, :N_WIN]
else:
    EPOCH_MODE = "full"


def truncation_window(n_keep, n_total, ntap=NTAP, alpha=None):
    """Observe until n_keep: the full-grid Tukey ENTRY taper (as the k=0 path)
    followed by a FIXED 1000 s ramp at the observation cut.

    Why not a 1000 s ramp at the window start too (the previous version):
    measured on id19 (SNR 3589, merger+1d) that ramp made log L jagged at
    the O(1) level on 1e-5 rad scales in the sky while the k=0 Tukey path was
    smooth; toggling ONLY the window (same shrunk grid) removed it. The window
    opens on a loud ~1e-4 Hz inspiral (period 1e4 s) -- a 1000 s edge there is
    sharper than one cycle. The 1000 s ramp at the CUT stays: it is what
    defines the epoch at dt resolution (the SNR ladder is built on it).
    """
    # ENTRY ramp only (the rising half of the Tukey); the Tukey's own end taper
    # would start alpha*N/2 (~21 h on a 35 d grid) before the cut and eat the
    # merger (measured: k=1 realised SNR 1007 vs 1794 with the full Tukey).
    from cd1l_windows import observation_window
    a = TUKEY_A if alpha is None else alpha
    return observation_window(n_keep, n_total, DT,
                              int(a * M.N_WIN / 2) * M.DT,
                              NTAP_BASE * M.DT)


# Snap waveform_t0 onto the DATA lattice so PR#81's grid-aligned path accepts
# mojito timing; t_plunge absorbs the same shift, so abs_merger is unchanged.
# Verified physics-neutral: mm=1.0654e-06 with grid_align both False and True.
off   = M.MBH_WAVEFORM_T0 - window_t0
WF_T0 = float(window_t0 + np.rint(off / DT) * DT)
DELTA = WF_T0 - M.MBH_WAVEFORM_T0

inj = np.asarray(M.mbh_catalogue_to_sampling_basis(cat), dtype=float).copy()
inj[STOCK_BASIS.index("t_plunge")] -= DELTA
if SAMPLER == "mc":
    inj = S.stock_to_mc(inj)          # (logM, Q) -> (lnMc, q=m2/m1)

# Persist the injection in the SAMPLING basis (with the t_plunge snap already
# applied) so post-processing scores against exactly what was sampled.
np.save(os.path.join(OUT, f"injection_id{ID}{TAG_SUFFIX}.npy"), inj)

print(f"=== CD1-L PE: MBHB id={ID} ===")
print(f"  SNR(cat)={cat.get('EstimatedSNR', float('nan')):.1f}  "
      f"nwalkers={NWALKERS} ntemps={NTEMPS} nsteps={NSTEPS}")
print(f"  dt={DT} (decimate x{DECIMATE})  N_WIN={N_WIN:,}  WDM {NF}x{NT}  "
      f"order={ORDER}  vectorize=True")
if _epoch is not None:
    print(f"  EPOCH k={EPOCH_K}: SNR target {_epoch['snr_target']:.1f}, "
          f"t-t_merger = {_epoch['days_from_merger']:+.3f} d, "
          f"observe {N_KEEP*DT/86400:.3f} d of the {N_WIN*DT/86400:.2f} d grid")
print(f"  waveform_t0 snap {DELTA:+.3f} s; t_plunge -> {inj[-1]!r}")
print("  injection: " + "  ".join(f"{k}={v:.6g}" for k, v in zip(BASIS, inj)))
if TDET:
    # Every downstream consumer takes the injection as the origin -- the
    # t_plunge prior box, the init ball, the numerical Fisher -- so one
    # conversion here puts the whole run in t_det.
    _jt = BASIS.index("t_plunge")
    _t_ssb = float(inj[_jt])
    inj = S.ssb_to_tdet(inj, DVEC)
    # The saved injection above stays t_SSB; post-processing needs the delay
    # vector to map this chain's t_det column back (S.tdet_to_ssb).
    np.save(os.path.join(OUT, f"dvec_id{ID}{TAG_SUFFIX}.npy"), np.asarray(DVEC))
    print(f"  CD1L_TDET: injection t_plunge {_t_ssb!r} (SSB) -> "
          f"{float(inj[_jt])!r} (t_det), delay "
          f"{_t_ssb - float(inj[_jt]):+.3f} s", flush=True)

_wdm_kw = dict(min_freq=M.F_MIN, max_freq=M.F_MAX, force_backend=M.BACKEND)
if EPOCH_MODE == "wdm":
    _layer_dt = NF * DT
    _t_cut = N_KEEP * DT                       # seconds from window_t0
    # post-merger anchor: keep one extra pixel past merger+1d so the last
    # pixel's window tail does not clip the merger (measured: 3553 vs 3589
    # without it); a pre-merger cut needs the opposite margin, max_time - layer_dt.
    _max_time = _t_cut + _layer_dt if EPOCH_K == -1 else max(_t_cut - _layer_dt, 0.0)
    _wdm_kw["max_time"] = _max_time
wdm_full = WDMSettings(NF, NT, DT, t0=window_t0, **_wdm_kw)
if EPOCH_MODE == "wdm":
    print(f"  EPOCH via WDM active time band: max_time={_max_time/86400:.3f} d -> pixels "
          f"0..{wdm_full.ind_max_t} of {NT} kept (pixel {NF*DT/3600:.1f} h; kept box ends at "
          f"{(wdm_full.ind_max_t+1)*NF*DT/86400:.3f} d, merger at {(abs_merger-window_t0)/86400:.3f} d)",
          flush=True)
td_set = TDSettings(N=N_WIN, dt=DT, t0=window_t0, force_backend=M.BACKEND)

gen = GridAlignedPhenomTHMTDIWaveform(
    waveform_kwargs=dict(higher_modes=list(M.MBH_HIGHER_MODES),
                         include_negative_modes=True, t_low_fit=True,
                         coarse_grain=False, atol=M.MBH_PHENOM_TOL,
                         rtol=M.MBH_PHENOM_TOL),
    Tobs=M.MBH_WAVEFORM_DURATION, start_freq=M.MBH_START_FREQ,
    use_reference_time=True, waveform_t0=WF_T0, data_td_settings=td_set,
    tdi_generation=M.TDI_GEN_STR, tdi_channels=M.TDI_CHAN,
    sampling_frequency=1.0 / DT, orbits=M.get_orbits(window_t0),
    order=ORDER, tukey_alpha=TUKEY_A, stft_dt=None,
    freq_min=M.F_MIN, freq_max=M.F_MAX, fft_batch_size=2,
    buffer_time=M.MBH_BUFFER_TIME, output_domain_settings=wdm_full,
    force_backend=M.BACKEND)
gen._check_alignable()

win   = (truncation_window(N_KEEP, N_WIN) if (EPOCH_MODE == "td" and TD_WINDOW == "trunc")
         else M.tukey(N_WIN, alpha=TUKEY_A))
if EPOCH_MODE == "td":
    print(f"  TD epoch path: shrink={TD_SHRINK} (NT={NT}), window={TD_WINDOW}", flush=True)
d_wdm = TDSignal(M._on(td_set, data_td), td_set).transform(
    wdm_full, window=M._on(td_set, win))
# Instrument noise only: the analytic instrument sensitivity, no galactic
# foreground term. Time evolution of the foreground is out of scope per the
# task, so results are qualitative by construction.
sens = XYZ2SensitivityMatrix(wdm_full, model=M.SENS_MODEL)
# The raw generator's __call__ returns (signal, start_freqs) and its bin-edge
# path supports FD/STFT only -- a WDM target must go through
# get_signals_for_residuals(). BatchedDomainSignalGen is the sanctioned
# adapter that does exactly that, so it is required here even though
# vectorize=False means every call carries a single row.
class TruncatedWDMSignalGen(BatchedDomainSignalGen):
    """Batched signal_gen that applies the OBSERVATION WINDOW to the template
    in the time domain before the WDM transform (same window as the data).

    Mirrors BatchedDomainSignalGen (batched rows -> stacked WDM template with a
    leading source axis, so the epoch runs keep the 2x batching of the k=0
    path) but replaces the generator's full-grid Tukey with ``win``. Truncating
    only the data would leave the template carrying a merger the data no
    longer contains, and the residual would be dominated by it.
    """

    def __init__(self, gen, win_dev, grid, td_set, wdm, nch):
        super().__init__(gen)
        self.win, self.grid = win_dev, grid
        self.td_set, self.wdm, self.nch = td_set, wdm, nch

    def _to_domain(self, times, channels):
        h = place_td_signal_on_grid(
            np.atleast_2d(channels)[:self.nch], self.grid, times=times).arr
        return TDSignal(h * self.win, self.td_set).transform(self.wdm)

    def __call__(self, *params, **kw):
        times, channels = self.wave_gen.compute_tdi_channels(*params, **kw)
        if getattr(times, "ndim", 1) == 1:
            return self._to_domain(times, channels)
        n_src = int(times.shape[0])              # see BatchedDomainSignalGen
        squeezed = channels.ndim == times.ndim
        return self._stack([self._to_domain(times[i], channels if squeezed else channels[i])
                            for i in range(n_src)])


if EPOCH_MODE == "td":
    _grid = TDSettings(N=N_WIN, dt=DT, t0=window_t0, force_backend=M.BACKEND)
    _sig = TruncatedWDMSignalGen(gen, M._on(td_set, win), _grid, td_set,
                                 wdm_full, M.NCHANNELS)
else:
    _sig = BatchedDomainSignalGen(gen)
ac   = AnalysisContainer(d_wdm, sens, signal_gen=_sig)

_dd = np.asarray(M._host(ac.inner_product()))
print(f"  realised SNR of analysed segment = "
      f"{float(np.sqrt(max(float(np.real(_dd)), 0.0))):.1f}"
      + (f"   (ladder target {_epoch['snr_target']:.1f})" if _epoch is not None else ""))

_like_raw = getattr(ac, "eryn_likelihood_function", None) or ac.eryn_likelihood_wrap

def like_fn(params, *args, **kwargs):
    """Batch likelihood with a per-row fallback: at low SNR a walker can reach
    a corner where phentax's coefficient root-find (compute_coeffs_22) does not
    converge and the whole vmapped batch raises. Re-evaluate that batch row by
    row and hand the offending rows -inf instead of killing the run."""
    try:
        out = np.asarray(_like_raw(params, *args, **kwargs), dtype=float)
        # NaN/+inf from a degenerate waveform must not enter a walker state: a
        # NaN log L never accepts again (lnpdiff is NaN) and freezes the walker.
        return np.where(np.isfinite(out), out, -np.inf)
    except Exception as e:                      # equinox/jax runtime error
        rows = np.atleast_2d(np.asarray(params))
        out = np.full(rows.shape[0], -np.inf)
        nbad = 0
        for r in range(rows.shape[0]):
            try:
                v = float(np.asarray(_like_raw(rows[r:r+1], *args, **kwargs)).ravel()[0])
                out[r] = v if np.isfinite(v) else -np.inf
            except Exception as e2:
                nbad += 1
                print("  [like_fn] FAILED row: " + "  ".join(f"{k}={v:.6g}" for k, v in zip(BASIS, rows[r]))
                      + f"   ({str(e2).strip().splitlines()[-1][:80]})", flush=True)
        print(f"  [like_fn] batch raised ({type(e).__name__}); {nbad}/{rows.shape[0]} "
              f"rows -> -inf", flush=True)
        return out
like_fn.__name__ = "batched+rowfallback(" + _like_raw.__name__ + ")"
print(f"  likelihood fn: {like_fn.__name__}")

# ------------------------------------------------------------------- priors
i = BASIS.index
# Chirp-mass box: Mc in [3e3, 4e7] Msun. The stock box capped the TOTAL mass at
# 1e8; a lnMc box up to ln(1e8) let PriorDrawMove propose Mc~1e8 (M_tot 3-5e8,
# merger ~1e-5 Hz) where phentax's coefficient root-find fails (debug3 rows).
# Heaviest CD1-L source is id19 at Mc ~ 8e6 (ln 15.9).
_mass_priors = ({i("lnMc"): uniform_dist(np.log(3e3), np.log(4e7)),
                 i("q"):    uniform_dist(0.05, 1.0)}
                if SAMPLER == "mc" else
                {i("logM"): uniform_dist(np.log(1e4), np.log(1e8)),
                 i("Q"):    uniform_dist(1.0, 20.0)})
priors = {"mbh": (S.MassConstrainedPrior if SAMPLER == "mc" else ProbDistContainer)({
    **_mass_priors,
    i("s1z"):      uniform_dist(-0.999999, 0.999999),
    i("s2z"):      uniform_dist(-0.999999, 0.999999),
    i("dist"):     uniform_dist(0.1, 200.0),          # Gpc
    i("phi_ref"):  uniform_dist(0.0, 2 * np.pi),
    i("cos_iota"): uniform_dist(-1.0, 1.0),
    i("psi"):      uniform_dist(0.0, np.pi),
    i("ra"):       uniform_dist(0.0, 2 * np.pi),
    i("sin_dec"):  uniform_dist(-1.0, 1.0),
    # +/- 1 hour around the injected plunge: wider than any plausible posterior
    # (timing at SNR 366 is O(s)) but far narrower than the 48 d window, which
    # would otherwise let walkers wander into regions with no signal at all.
    i("t_plunge"): uniform_dist(inj[i("t_plunge")] - 3600.0,
                                inj[i("t_plunge")] + 3600.0),
})}
periodic = PeriodicContainer({"mbh": {i("phi_ref"): 2 * np.pi,
                                      i("psi"): np.pi,
                                      i("ra"): 2 * np.pi}})

# --------------------------------------------------------------- init state
# NOTE ON INITIALISATION: ISSUE_DRAFT_mbh_gpu_run.md documents a run that began
# AT the injection and monotonically expanded for 100 iterations -- burn-in
# width mistaken for a posterior. Starting from a ball of width INIT_SCALE
# (relative) means the chain must CONTRACT to the posterior, so stationarity is
# detectable rather than assumed. Convergence is judged from the chain, never
# from the fact that it started at truth.
rng = np.random.default_rng(1234 + ID)

# ABSOLUTE per-parameter widths, not a fraction of |injection|. A relative
# scale is meaningless for a parameter whose value is large in absolute terms:
# at t_plunge = 2.6e7 s a 1e-4 relative width is 2595 s, which (a) exceeds the
# +/-3600 s prior at 1.4 sigma, giving log_prior = -inf, and (b) displaces the
# merger far enough that the template misses the signal entirely -- the
# -1e308 log_like seen in job 13908082.
#
# These are chosen ~10x wider than the expected posterior so the ball must
# CONTRACT: an expanding chain is then a visible failure rather than something
# mistaken for a posterior (see ISSUE_DRAFT_mbh_gpu_run.md).
INIT_WIDTH = {
    "logM":     1e-4,      # ln M, i.e. ~1e-4 fractional in total mass
    "Q":        1e-3,
    "lnMc":     1e-4,      # chirp-mass basis
    "q":        1e-3,
    "s1z":      1e-3,
    "s2z":      1e-3,
    "dist":     1e-3,      # FRACTIONAL (id19 is 4.6 Gpc, id0 is 54 Gpc)
    "phi_ref":  5e-3,      # rad -- posterior std ~0.57 on id0; was expanding at 1e-3
    "cos_iota": 1e-3,
    "psi":      1e-3,      # rad
    "ra":       1e-3,      # rad
    "sin_dec":  1e-3,
    "t_plunge": 1.0,       # s -- timing at SNR 366 is O(s); prior is +/-3600 s
}
scale = np.array([INIT_WIDTH[k] for k in BASIS]) * (INIT_SC / 1e-4)
scale[i("dist")] *= inj[i("dist")]                      # fractional -> Gpc
start = inj[None, None, :] + rng.normal(size=(NTEMPS, NWALKERS, NDIM)) * scale
for k, period in (("phi_ref", 2*np.pi), ("ra", 2*np.pi), ("psi", np.pi)):
    start[..., i(k)] = np.mod(start[..., i(k)], period)   # periodic: wrap
start[..., i("cos_iota")] = np.clip(start[..., i("cos_iota")], -1, 1)
start[..., i("sin_dec")]  = np.clip(start[..., i("sin_dec")],  -1, 1)
if SAMPLER == "mc":
    start[..., i("q")]    = np.clip(start[..., i("q")], 0.05, 1.0)
else:
    start[..., i("Q")]    = np.clip(start[..., i("Q")], 1.0, 20.0)
start[..., i("psi")]      = np.clip(start[..., i("psi")], 0.0, np.pi)

# Fail loudly here rather than inside Eryn: a -inf in the initial prior means
# the ball straddles a prior edge, which is a setup bug, not a sampling event.
_lp = priors["mbh"].logpdf(start.reshape(-1, NDIM))
_bad = int(np.sum(~np.isfinite(np.asarray(M._host(_lp)))))
if _bad and SAMPLER != "mc":
    raise SystemExit(f"{_bad}/{start.shape[0]*start.shape[1]} initial walkers "
                     f"fall outside the prior -- widen the prior or narrow "
                     f"INIT_WIDTH before running.")
print(f"  init ball OK: all {NTEMPS*NWALKERS} walkers inside the prior")

fp = os.path.join(OUT, f"cd1l_mbh_id{ID}_{TAG}{TAG_SUFFIX}.h5")

# A crashed run leaves an .h5 with no stored iterations; "file exists" is
# therefore NOT the same as "resumable", and treating it as such makes every
# retry fail on get_last_sample. Probe it, and move anything unreadable aside
# rather than deleting -- a backend that raises could be a damaged real chain,
# not merely an empty one.
resume = False
if os.path.exists(fp):
    try:
        _probe = HDFBackend(fp).get_last_sample()
        resume = True
    except Exception as _e:
        stale = fp + f".stale-{int(time.time())}"
        os.rename(fp, stale)
        print(f"  {os.path.basename(fp)} has no readable samples "
              f"({type(_e).__name__}); moved to {os.path.basename(stale)}")
LIKE_KW = dict(transform_fn=TRANSFORM, source_only=True)
if os.environ.get("CD1L_REGRESS"):
    # LIKELIHOOD REGRESSION TEST against a stored chain: re-evaluate the
    # exact walker coordinates of an old run with today's code and compare
    # to the log L that run stored. Unchanged physics => |dlogL| ~ 1e-8.
    _b = HDFBackend(os.environ["CD1L_REGRESS"])
    _ch = np.asarray(_b.get_chain()["mbh"]); _llo = np.asarray(_b.get_log_like())
    _n = _ch.shape[0]
    print(f"  regression vs {os.path.basename(os.environ['CD1L_REGRESS'])} ({_n} steps)")
    _ref_basis = os.environ.get("CD1L_REGRESS_BASIS")
    if _ref_basis is None:
        _ref_basis = "mc" if os.environ["CD1L_REGRESS"].endswith("_mc.h5") else "stock"
    if _ref_basis not in ("mc", "stock"):
        raise ValueError("CD1L_REGRESS_BASIS must be mc or stock")
    if _n == 0:
        raise ValueError("Regression chain has no samples")
    for _it in sorted(j for j in {0, 1, 100, 1000, _n // 2, _n - 1} if 0 <= j < _n):
        _x = _ch[_it, 0, :, 0, :]; _old = _llo[_it, 0, :]
        if _ref_basis != SAMPLER:
            _x = S.stock_to_mc(_x) if SAMPLER == "mc" else S.mc_to_stock(_x)
        _new = np.asarray(like_fn(np.asarray(_x), **LIKE_KW)).ravel()
        _d = _new - _old
        print(f"    step {_it:5d}: max|dlogL| = {np.max(np.abs(_d)):.3e}   "
              f"logL range [{_old.min():.1f}, {_old.max():.1f}]   "
              f"worst walker old {_old[np.argmax(np.abs(_d))]:.6f} new {_new[np.argmax(np.abs(_d))]:.6f}", flush=True)
    raise SystemExit(0)
if SAMPLER == "mc":
    # prior box (for reflection / prior draws) from the ProbDistContainer
    _lo = np.array([priors["mbh"].priors_in[j].minimum for j in range(NDIM)])
    _hi = np.array([priors["mbh"].priors_in[j].maximum for j in range(NDIM)])
    _ll = lambda x: like_fn(np.asarray(x), **LIKE_KW)
    _eps0 = np.array([1e-5, 1e-3, 1e-2, 1e-2, 5e-3 * inj[i("dist")], 1e-2, 1e-2,
                      1e-2, 1e-3, 1e-3, 1.0])
    if os.environ.get("CD1L_PROPOSAL_PROBE"):
        # The one-step proposal operator applied to EXACT posterior samples from
        # a finished chain: acceptance and expected squared jump distance for
        # the best possible global Gaussian proposal in whatever parametrisation
        # this run uses.  No burn-in, no chain length, no walker collapse, and
        # ~2300 likelihoods instead of a 2000-step run.  Jumps are always
        # reported in the t_SSB basis normalised by the SSB posterior sd, so
        # CD1L_TDET=0 and =1 are directly comparable.
        import cd1l_ridge_accounting as RA
        _rc = os.environ.get("CD1L_PROBE_CHAIN",
                             f"{RA.ROOT}/runs/roulet_spins/cd1l_mbh_id{ID}_post1d_mc.h5")
        _cch, _cit = RA.load_chain(_rc)
        _rinj = RA.injection_mc(ID, os.path.relpath(os.path.dirname(os.path.dirname(_rc)), RA.ROOT))
        _xs = RA.unwrap(_cch[_cit // 2:].reshape(-1, NDIM), _rinj)
        _msk, _, _ = RA.truth_mode(_xs, _rinj)
        _xs = _xs[_msk]                                    # exact draws, t_SSB
        # Same branch-cut repair the warm covariance gets (cd1l_warm_inputs.py):
        # unwrapping at the injection splits id19's phi_ref lobe and inflates its
        # sd 3.4x, which would poison _sd, _ridge and _Sig alike.  No-op for
        # id2/id14, so it does not move their numbers.
        if os.environ.get("CD1L_PROBE_RECENTRE", "1") != "0":
            _xs = RA.recentre_periodic(_xs)
        _sd = _xs.std(axis=0)
        _w, _V = np.linalg.eigh(np.corrcoef((_xs / _sd).T)); _ridge = _V[:, -1]
        _xp = S.ssb_to_tdet(_xs, DVEC) if TDET else _xs    # likelihood basis
        _Sig = np.cov(_xp.T)
        _rng2 = np.random.default_rng(12345)
        _N = int(os.environ.get("CD1L_PROBE_N", "384"))
        _base = _xp[_rng2.choice(len(_xp), _N, replace=False)]

        def _batched(rows):
            if len(rows) == 0:
                return np.zeros(0)
            return np.concatenate([np.asarray(_ll(rows[k:k + 64])).ravel()
                                   for k in range(0, len(rows), 64)])

        _lb = _batched(_base)
        _PER = ((i("phi_ref"), 2 * np.pi), (i("ra"), 2 * np.pi), (i("psi"), np.pi))
        print(f"\n=== proposal probe: {'t_det' if TDET else 't_SSB'} parametrisation, "
              f"{_N} exact posterior samples from {os.path.basename(_rc)} ===", flush=True)
        print(f"  logL there: median {np.median(_lb):.2f}  [{_lb.min():.2f}, {_lb.max():.2f}]")
        print(f"  {'f':>6} {'acc':>7} {'ESJD_ridge':>11} {'+-MC':>10} {'ESJD_t':>9} "
              f"{'ESJD_ra':>9} {'ESJD_sdec':>10} {'2/ESJD':>8}", flush=True)
        _FGRID = [float(v) for v in os.environ.get(
            "CD1L_PROBE_F", "0.01,0.03,0.1,0.3,1.0").split(",")]
        for _f in _FGRID:
            _L = np.linalg.cholesky(S.ensure_pd((2.38 ** 2 / NDIM) * _f * _Sig, rel_floor=1e-12))
            _prop = _base + _rng2.standard_normal((_N, NDIM)) @ _L.T
            for _j, _per in _PER:
                _prop[:, _j] %= _per
            _ok = np.isfinite(np.asarray(M._host(priors["mbh"].logpdf(_prop))))
            _lp = np.full(_N, -np.inf)
            _lp[_ok] = _batched(_prop[_ok])
            _a = np.clip(np.exp(np.minimum(_lp - _lb, 0.0)), 0.0, 1.0)
            _d = (S.tdet_to_ssb(_prop, DVEC) if TDET else _prop) - \
                 (S.tdet_to_ssb(_base, DVEC) if TDET else _base)
            for _j, _per in _PER:
                _d[:, _j] = (_d[:, _j] + _per / 2) % _per - _per / 2
            _z = _d / _sd
            _e = (_a[:, None] * _z ** 2).mean(axis=0)
            _rs = _a * (_z @ _ridge) ** 2          # per-sample ridge jump
            _er = float(_rs.mean())
            _se = float(_rs.std(ddof=1) / np.sqrt(len(_rs)))   # MC error on ESJD
            print(f"  {_f:6.2f} {_a.mean():7.4f} {_er:11.5f} {_se:10.5f} "
                  f"{_e[i('t_plunge')]:9.5f} "
                  f"{_e[i('ra')]:9.5f} {_e[i('sin_dec')]:10.5f} "
                  f"{(2 / _er if _er > 0 else float('inf')):8.0f}", flush=True)
        raise SystemExit(0)
    if os.environ.get("CD1L_BATCHTEST"):
        # Is the batched likelihood a function of the ROW ONLY, or of the batch?
        e = np.eye(NDIM)
        def row(j, d): return inj + d * e[j]
        probes = {"x0": inj, "q+1e-3": row(i("q"), 1e-3), "dist+0.27": row(i("dist"), 0.27),
                  "t+1": row(i("t_plunge"), 1.0)}
        contexts = {
            "alone":            lambda x: x[None, :],
            "x16 copies":       lambda x: np.repeat(x[None, :], 16, 0),
            "+lnMc+-1e-5":      lambda x: np.array([x, row(i("lnMc"), 1e-5), row(i("lnMc"), -1e-5)]),
            "+lnMc+-1e-3":      lambda x: np.array([x, row(i("lnMc"), 1e-3), row(i("lnMc"), -1e-3)]),
            "+t+-100 s":        lambda x: np.array([x, row(i("t_plunge"), 100.), row(i("t_plunge"), -100.)]),
            "+t+-3000 s":       lambda x: np.array([x, row(i("t_plunge"), 3000.), row(i("t_plunge"), -3000.)]),
            "+q 0.05":          lambda x: np.array([x, row(i("q"), 0.05 - inj[i("q")])]),
            "last of 3":        lambda x: np.array([row(i("lnMc"), 1e-5), row(i("t_plunge"), 100.), x]),
        }
        for pname, x in probes.items():
            print(f"  probe {pname}:")
            ref = None
            for cname, mk in contexts.items():
                b = mk(x); k = 0 if cname != "last of 3" else 2
                v = float(np.asarray(_ll(b)).ravel()[k])
                ref = v if ref is None else ref
                print(f"    {cname:14s} logL = {v:+.6f}   diff vs alone = {v-ref:+.3e}", flush=True)
        raise SystemExit(0)
    if os.environ.get("CD1L_PROFILE"):
        # 1-d log L profiles around the injection: curvature + noise floor per axis
        L0 = float(np.asarray(_ll(inj[None, :])).ravel()[0])
        print(f"  log L(inj) = {L0:.6f}")
        for j, name in enumerate(BASIS):
            base = _eps0[j]
            steps = base * np.array([1e-3, 1e-2, 1e-1, 1, 10, 100])
            pts = np.array([inj + s_ * np.eye(NDIM)[j] for s_ in np.concatenate([-steps[::-1], steps])])
            ll = np.asarray(_ll(pts)).ravel() - L0
            print(f"  {name:>9}: " + " ".join(f"{d:+.3g}" for d in ll) + f"   (steps +-{base:.2g}*[1e-3..100])", flush=True)
        raise SystemExit(0)
    _t = time.time()
    if os.environ.get("CD1L_FISHER", "templates") == "hessian":
        print("  computing Fisher covariance at the injection (log L Hessian, 2 passes)...", flush=True)
        FCOV = S.fisher_cov(_ll, inj, _eps0)
    else:
        print("  computing Fisher covariance at the injection (template derivatives)...", flush=True)
        FCOV = S.fisher_from_templates(ac, TRANSFORM, inj)
    print(f"  Fisher done in {time.time()-_t:.1f} s", flush=True)
    print(f"  log L at injection = {float(np.asarray(_ll(inj[None, :])).ravel()[0]):.4f}"
          "   (noise-free data: -0.5 * mismatch * SNR^2 if the template were exact)", flush=True)
    if os.environ.get("CD1L_FISHER_ONLY"):
        _ref = os.environ.get("CD1L_REF_CHAIN")
        if _ref and os.path.exists(_ref):
            _c = np.asarray(HDFBackend(_ref).get_chain()["mbh"])[-1000:, 0, :, 0, :].reshape(-1, NDIM)
            if SAMPLER == "mc": _c = S.stock_to_mc(_c)
            np.set_printoptions(precision=3, linewidth=200)
            print("  [chain ] empirical sigma (last 1000 steps of ref):", _c.std(axis=0))
            _cc = np.corrcoef(_c.T); print("  [chain ] |corr| max off-diag:", np.max(np.abs(_cc - np.eye(NDIM))))
        raise SystemExit(0)
    np.save(os.path.join(OUT, f"fisher_cov_id{ID}_{TAG}{TAG_SUFFIX}.npy"), FCOV)
    # WARM RESTART (2026-09-21, ridge-accounting result): the posterior's sky/t
    # excess over the Fisher is a LINEAR ridge the numerical Fisher misses by
    # 10-100x, and the adaptive kernel freezes at step 1000 on an ensemble that
    # has not yet spread along it. CD1L_PROPOSAL_COV replaces the Fisher by a
    # stored covariance (e.g. a previous chain's truth-mode second half, saved
    # by cd1l_warm_inputs.py) for BOTH the proposal and the init ball.
    _ovr = os.environ.get("CD1L_PROPOSAL_COV")
    if _ovr:
        FCOV = np.load(_ovr)
        if FCOV.shape != (NDIM, NDIM):
            raise ValueError(f"CD1L_PROPOSAL_COV {_ovr}: shape {FCOV.shape} != ({NDIM},{NDIM})")
        print(f"  proposal covariance OVERRIDDEN from {_ovr}: sigma "
              + " ".join(f"{k}={v:.2g}" for k, v in zip(BASIS, np.sqrt(np.diag(FCOV)))), flush=True)
    # START AT THE INJECTION (user decision 2026-09-11): a 0.1-sigma Fisher
    # jitter around truth -- just enough that StretchMove is not degenerate
    # (identical walkers propose zero-length steps). The chain then EXPANDS
    # to the posterior; cd1l_stationarity.py's windowed std tells when the
    # expansion has plateaued. The hand-set INIT_WIDTH ball at INIT_SCALE=1e-2
    # was ~100 Fisher sigmas wide and cost 500-1000 steps of burn-in.
    _nsig = float(os.environ.get("CD1L_INIT_NSIGMA", "0.02"))
    _Lf = np.linalg.cholesky(S.ensure_pd(FCOV, rel_floor=1e-12))
    start = inj[None, None, :] + _nsig * (rng.standard_normal((NTEMPS, NWALKERS, NDIM)) @ _Lf.T)
    for k, period in (("phi_ref", 2*np.pi), ("ra", 2*np.pi), ("psi", np.pi)):
        start[..., i(k)] = np.mod(start[..., i(k)], period)
    _out = ~np.isfinite(priors["mbh"].logpdf(start.reshape(-1, NDIM))).reshape(NTEMPS, NWALKERS)
    for _ in range(20):
        if not _out.any(): break
        start[_out] = inj + _nsig * (rng.standard_normal((int(_out.sum()), NDIM)) @ _Lf.T)
        for k, period in (("phi_ref", 2*np.pi), ("ra", 2*np.pi), ("psi", np.pi)):
            start[..., i(k)] = np.mod(start[..., i(k)], period)
        _out = ~np.isfinite(priors["mbh"].logpdf(start.reshape(-1, NDIM))).reshape(NTEMPS, NWALKERS)
    print(f"  init: {_nsig:g}-sigma Fisher ball; {int(_out.sum())} walkers outside prior", flush=True)
    if _out.any():
        raise ValueError("Fisher initialization exhausted prior-valid redraws")
    # WARM RESTART init: walkers = random COLD samples from the second half of a
    # previous chain (already spread along the ridge), every temperature drawn
    # from the same pool. Only for fresh runs; a resume keeps its own state.
    _ic = os.environ.get("CD1L_INIT_CHAIN")
    if _ic and not resume:
        with h5py.File(_ic, "r") as _hf:
            _it = int(_hf["mcmc"].attrs["iteration"])
            _pool = _hf["mcmc/chain/mbh"][_it // 2:_it, 0, 0, :, 0, :].reshape(-1, NDIM)
        if not _ic.endswith("_mc.h5"):
            _pool = S.stock_to_mc(_pool)
        # Previous chains are stored in t_SSB. Unconverted, a t_det run would put
        # every walker ~the light delay (hundreds of s) off in t_plunge -- still
        # inside the +-3600 s prior box, so nothing downstream would notice.
        if TDET:
            _pool = S.ssb_to_tdet(_pool, DVEC)
        _pool = _pool[np.isfinite(_pool).all(axis=1)]
        _pool = _pool[np.isfinite(np.asarray(M._host(priors["mbh"].logpdf(_pool))))]
        if len(_pool) < NTEMPS * NWALKERS:
            raise ValueError(f"CD1L_INIT_CHAIN {_ic}: only {len(_pool)} usable samples")
        start = _pool[rng.choice(len(_pool), size=NTEMPS * NWALKERS, replace=False)].reshape(NTEMPS, NWALKERS, NDIM)
        print(f"  init OVERRIDDEN: {NTEMPS*NWALKERS} cold samples from {_ic} (pool {len(_pool)})", flush=True)
    _partners = S.sky_partners(inj)[0]
    _pll = np.asarray(_ll(_partners)).ravel()
    print("  sky partners dLL: " + " ".join(f"{v - _pll[0]:+.1f}" for v in _pll), flush=True)
    gaussian_move = (S.RouletSpinGaussianMove(
                         FCOV, inj, _lo, _hi, proposal_scale=ROULET_STEP_SCALE)
                     if ROULET_SPINS else S.FisherGaussianMove(FCOV, _lo, _hi))
    # The sky-partner Gibbs move only does work while some partner carries
    # weight. Post-merger every partner sits ~1e5 in log L below the truth
    # (id0 post1d: -77e3 .. -134e3), so the draw returns the current mode with
    # probability 1 at every temperature and the move is 8 wasted likelihoods
    # per walker on 15% of steps. Decide from the partner check above: keep it
    # only if some partner is within SKY_GIBBS_DEAD of the truth (a partner
    # 1000 below is dead even at beta=0.05: e^-50). CD1L_SKY_GIBBS=1 forces it on.
    SKY_GIBBS_DEAD = 1000.0
    _partner_alive = bool(np.any(_pll[1:] - _pll[0] > -SKY_GIBBS_DEAD))
    _use_gibbs = _partner_alive or os.environ.get("CD1L_SKY_GIBBS", "0") == "1"
    if resume:
        # Eryn (track_moves=True) refuses to continue a backend whose move list
        # changed, so a resumed chain keeps whatever it was started with.
        _use_gibbs = any("SkyPartnerGibbs" in k for k in HDFBackend(fp).move_keys)
    # PriorDrawMove: acceptance was exactly 0.00 on all 40 finished chains
    # (runs/ and runs_dt25_o8) -- a uniform draw never lands in a posterior this
    # narrow -- so it only burned 5% of steps. Off by default since 2026-09-24;
    # CD1L_PRIOR_DRAW=1 restores it, and a resumed chain keeps its own move list.
    _use_prior = os.environ.get("CD1L_PRIOR_DRAW", "0") == "1"
    if resume:
        _use_prior = any("PriorDraw" in k for k in HDFBackend(fp).move_keys)
    # CD1L_STRETCH_WEIGHT: per-source override (2026-09-24). Where the Gaussian
    # is dead (acc ~0.00 on id11/id12: runaway phi_ref in the adapted
    # covariance) StretchMove is the only move that moves, so give it more.
    _w_stretch = float(os.environ.get("CD1L_STRETCH_WEIGHT", "0.15"))
    MOVES = [(StretchMove(), _w_stretch)]
    if _use_gibbs:
        MOVES.append((S.SkyPartnerGibbsMove(_ll, _lo, _hi), 0.15))
    if _use_prior:
        MOVES.append((S.PriorDrawMove(_lo, _hi, constraint=S.mtot_ok), 0.05))
    # the Gaussian takes the remainder: 0.65 / 0.80 / 0.70 / 0.85 exactly as before
    _w_gauss = round(1.0 - sum(w for _, w in MOVES), 10)
    if _w_gauss < 0:
        raise SystemExit(f"move weights exceed 1: {MOVES}")
    MOVES.insert(0, (gaussian_move, _w_gauss))
    MOVES = [(m, w) for m, w in MOVES if w > 0]      # weight 0 = move switched off
    print(f"  moves: {', '.join(f'{type(m).__name__} {w:.2f}' for m, w in MOVES)}", flush=True)
    print(f"  moves: {'with' if _use_gibbs else 'WITHOUT'} SkyPartnerGibbs "
          f"(closest partner dLL = {np.max(_pll[1:] - _pll[0]):+.1f}"
          f"{'' if _partner_alive else f' < -{SKY_GIBBS_DEAD:g}: all partners dead'})", flush=True)
else:
    MOVES = StretchMove()
sampler_kw = dict(
    kwargs=LIKE_KW,
    moves=MOVES,
    branch_names=["mbh"], periodic=periodic,
    # MEASURED REVERSAL: with the plan-cache fix + order=8 + dt=10 s each
    # waveform is ~8x cheaper, the GPU no longer saturates at B=1, and PR#81
    # batching gives 2.09x (0.1174 -> 0.0562 s/row), saturating at B=16-32.
    # Eryn's red-blue move hands 16-walker half-blocks at nwalkers=32 -- exactly
    # the batch size where this saturates.
    vectorize=True,
    backend=fp,
)
# Eryn calls TemperatureControl(**tempering_kwargs) unconditionally, so at
# ntemps=1 the key must be ABSENT rather than None.
if NTEMPS > 1:
    if SAMPLER == "mc":   # fixed ladder to 1e-4, as in run_pe_v0.py
        sampler_kw["tempering_kwargs"] = dict(
            ntemps=NTEMPS, betas=np.geomspace(1.0, 1e-4, NTEMPS), adaptive=False)
    else:
        sampler_kw["tempering_kwargs"] = dict(ntemps=NTEMPS)
sampler = EnsembleSampler(NWALKERS, {"mbh": NDIM}, like_fn, priors, **sampler_kw)
_MOVE_LOG = []
if os.environ.get("CD1L_NANCHECK") and SAMPLER == "mc":
    for _mv, _weight in MOVES:
        _orig = _mv.get_proposal
        def _wrapped(branches_coords, random, *args, _orig=_orig, _mv=_mv, **kwargs):
            proposal, factors = _orig(branches_coords, random, *args, **kwargs)
            _MOVE_LOG.append((type(_mv).__name__, bool(np.isnan(proposal["mbh"]).any()),
                              bool(np.isnan(factors).any()), bool(np.isinf(factors).any())))
            del _MOVE_LOG[:-12]
            return proposal, factors
        _mv.get_proposal = _wrapped
ADAPT_EVERY = int(os.environ.get("CD1L_ADAPT_EVERY", "25"))
ADAPT_UNTIL = int(os.environ.get("CD1L_ADAPT_UNTIL", "1000"))
_config = json.dumps(dict(version=2, sampler=SAMPLER, roulet=ROULET_SPINS,
    scale=ROULET_STEP_SCALE, epoch=EPOCH_K, epoch_mode=EPOCH_MODE,
    dt=DT, nf=NF, nt=NT, n_keep=N_KEEP, window_t0=window_t0,
    window_sha=__import__('hashlib').sha256(np.asarray(win).tobytes()).hexdigest(),
    order=ORDER, modes=M.MBH_HIGHER_MODES, start_freq=M.MBH_START_FREQ,
    injection=inj.tolist(), ntemps=NTEMPS, nwalkers=NWALKERS,
    adapt_every=ADAPT_EVERY, adapt_until=ADAPT_UNTIL,
    # warm-restart knobs enter the hash ONLY when set, so every existing
    # chain's stored config still matches and can be resumed
    **{k: os.environ[e] for k, e in (("proposal_cov", "CD1L_PROPOSAL_COV"),
                                      ("init_chain", "CD1L_INIT_CHAIN")) if os.environ.get(e)}),
    sort_keys=True)
_completed = 0
if resume:
    with h5py.File(fp, "r") as _hf:
        if _hf.attrs.get("cd1l_config") != _config:
            raise ValueError("Resume configuration missing or changed; use a fresh CD1L_OUT")
        _completed = int(_hf.attrs["cd1l_completed"])
        _stored = len(HDFBackend(fp).get_log_like())
        if _completed != _stored:
            # A wall-time kill mid-segment leaves a few iterations past the last
            # checkpoint. Once the proposal is frozen (past ADAPT_UNTIL) those
            # steps used the very same kernel, so the stored chain is exact and
            # the checkpoint can simply be advanced. Before the freeze it cannot.
            if _stored > _completed >= int(os.environ.get("CD1L_ADAPT_UNTIL", "1000")):
                print(f"  resume: advancing checkpoint {_completed} -> {_stored} (frozen kernel, "
                      f"wall-time kill mid-segment)", flush=True)
                _completed = _stored
            else:
                raise ValueError("Incomplete proposal checkpoint; use a fresh CD1L_OUT")
        if SAMPLER == "mc":
            gaussian_move._L = np.asarray(_hf.attrs["cd1l_proposal_L"])
            if ROULET_SPINS:
                gaussian_move.reference = np.asarray(_hf.attrs["cd1l_reference"])
    if not np.isfinite(HDFBackend(fp).get_log_like()).all():
        raise ValueError("Stored chain contains nonfinite likelihoods; historical samples need audit")
    st = HDFBackend(fp).get_last_sample()
    st.log_prior = sampler.compute_log_prior(st.branches_coords)
    if not np.isfinite(st.log_prior).all():
        raise ValueError("Resume state violates current prior")
    print(f"  RESUMING from {fp}")
else:
    st = State({"mbh": start[..., None, :]})
    st.log_prior = sampler.compute_log_prior(st.branches_coords)
    st.log_like  = sampler.compute_log_like(st.branches_coords, logp=st.log_prior)[0]
    print(f"  start log_like: max={np.max(M._host(st.log_like)):.4f}  "
          f"min={np.min(M._host(st.log_like)):.4f}")

if not np.isfinite(st.log_like).all():
    raise ValueError("Initial state has nonfinite likelihoods")
if NTEMPS == 1:
    st.betas = None
t0 = time.time()
if SAMPLER == "mc":
    # ADAPTIVE BURN-IN: refresh the Gaussian proposal with the empirical
    # covariance of the cold ensemble (last ADAPT_EVERY steps x walkers), then
    # freeze at ADAPT_UNTIL so the remainder is a fixed-kernel chain.
    done = 0
    while done < NSTEPS:
        # segments of ADAPT_EVERY for the WHOLE run (the NaN guard below needs
        # boundaries); the covariance refresh itself stops at ADAPT_UNTIL.
        interval = ADAPT_EVERY if ADAPT_EVERY > 0 else 25
        n = min(interval - ((_completed + done) % interval), NSTEPS - done)
        sampler.run_mcmc(st, n, progress=False, thin_by=1)
        st = sampler.get_last_sample(); done += n
        # Eryn must reject -inf proposals without poisoning state. A remaining
        # nonfinite state is an error: retain the chain for diagnosis and stop,
        # since repairing only the live state would leave corrupt stored samples.
        _llst = np.asarray(M._host(st.log_like))
        _bad = np.argwhere(~np.isfinite(_llst))
        if len(_bad):
            print("  recent moves (name, coords NaN, factors NaN, factors inf):", _MOVE_LOG, flush=True)
            raise RuntimeError(f"Nonfinite walker state at step {_completed + done}: {_bad.tolist()}; "
                               "chain retained for diagnosis, no proposal checkpoint written")
        if not np.isfinite(np.asarray(sampler.get_log_like())[-n:]).all():
            raise RuntimeError("Nonfinite stored samples; chain retained for diagnosis")
        if NTEMPS == 1:
            # Eryn: a state carrying betas is rejected when there is no
            # temperature_control (ntemps=1). Strip them for the next segment.
            st.betas = None
        if ADAPT_EVERY > 0 and _completed + done <= ADAPT_UNTIL and (_completed + done) % ADAPT_EVERY == 0:
            ch = np.asarray(sampler.get_chain()["mbh"])[-ADAPT_EVERY:, 0, :, 0, :].reshape(-1, NDIM)
            ll_now = np.asarray(M._host(sampler.get_log_like()))[-1, 0, :]
            # Unwrap periodic coordinates locally before estimating covariance.
            ch = ch.copy()
            for j, period in S.PERIODIC.items():
                center = np.angle(np.mean(np.exp(2j*np.pi*ch[:, j]/period))) * period/(2*np.pi)
                ch[:, j] = center + (ch[:, j] - center + period/2) % period - period/2
            ecov = np.cov(ch.T)
            # blend with the Fisher only while the ensemble is too small to
            # estimate 66 covariance entries on its own
            w = min(1.0, ch.shape[0] / (10.0 * NDIM))
            gaussian_move.set_cov(w * ecov + (1 - w) * FCOV)
            print(f"  [adapt] step {done}: cold log L max {np.max(ll_now):.2f} median {np.median(ll_now):.2f}; "
                  "sigma " + " ".join(f"{k}={v:.2g}" for k, v in zip(BASIS, np.sqrt(np.diag(ecov)))), flush=True)
        with h5py.File(fp, "a") as _hf:
            _hf.attrs["cd1l_config"] = _config
            _hf.attrs["cd1l_proposal_L"] = gaussian_move._L
            if ROULET_SPINS:
                _hf.attrs["cd1l_reference"] = gaussian_move.reference
            _hf.attrs["cd1l_completed"] = _completed + done
else:
    sampler.run_mcmc(st, NSTEPS, progress=False, thin_by=1)
    if not np.isfinite(sampler.get_log_like()).all():
        raise RuntimeError("Nonfinite stored likelihoods")
    with h5py.File(fp, "a") as _hf:
        _hf.attrs["cd1l_config"] = _config
        _hf.attrs["cd1l_completed"] = _completed + NSTEPS
el = time.time() - t0
n_ev = NSTEPS * NWALKERS * max(NTEMPS, 1)
print(f"\n=== timing ===")
print(f"  {NSTEPS} steps in {el:.1f} s  =  {el/NSTEPS:.3f} s/step  "
      f"({el/n_ev:.4f} s/likelihood, {n_ev} evals)")
print(f"  chain -> {fp}")
json.dump({"id": ID, "epoch_k": EPOCH_K, "tag": TAG, "nwalkers": NWALKERS, "ntemps": NTEMPS, "nsteps": NSTEPS,
           "roulet_spins": ROULET_SPINS,
           "spin_proposal_attempted": gaussian_move.attempted if ROULET_SPINS else None,
           "spin_proposal_prior_valid": gaussian_move.prior_valid if ROULET_SPINS else None,
           "seconds": el, "s_per_step": el/NSTEPS, "s_per_like": el/n_ev,
           "snr_cat": float(cat.get("EstimatedSNR", np.nan))},
          open(os.path.join(OUT, f"timing_id{ID}_{TAG}.json"), "w"), indent=2)
