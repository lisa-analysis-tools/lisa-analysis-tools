# Sparse-track STFT templates for the EMRI likelihood

`EMRISparseSTFTWaveform` builds the short-time Fourier transform (STFT) likelihood template of an
extreme mass-ratio inspiral (EMRI) without ever building the dense time series of the inspiral.
FEW's sparse holder goes through a sparse TDI response, each harmonic becomes an amplitude/phase
track, and each track is added to the STFT grid as a linear chirp per segment through the
`STFTFresnel` kernel. The few segments a linear chirp cannot represent (the fast chirp before the
plunge, the plunge itself) are filled from an exact time series. It is a drop-in replacement for
the production `EMRITDIWaveform` in the STFT domain: same constructor, same 14 parameters, same
output.

The work lives on the branch `feat-emri-sparse-stft` (worktree `wt-emri-sparse-stft`), on top of
`l2d-dev`, from 2026-11-16 to 2026-11-23. Nothing is pushed.

| Document | Content |
| --- | --- |
| `README.md` | What the waveform computes, how each step works, how to use it, the branch, and the open points. |
| [`results.md`](results.md) | The measurements: accuracy against production, the two dense fallbacks, `cubic_tol`, the CPU profile, the edge cases of the fallback, the heterodyne study, tests, and how to reproduce them. |

Goal of the branch: a drop-in EMRI template that matches the production dense template to about
1e-5 mismatch, beats the production 300 ms per likelihood on 2 years of data on the GPU, and keeps
everything except a thin adapter reusable for other sources (massive black hole binaries next).
The GB path is hand-optimised separately and is not touched.

## Where it stands

| | |
| --- | --- |
| Accuracy, no plunge (8 d) | mismatch **2.1e-8** against production, every segment on tracks |
| Accuracy, plunge in the data (36 d) | mismatch **1.7e-6** white, **7.6e-6** noise-weighted (production-response fallback, `cubic_tol` 0.03); **9.5e-8 / 1.0e-6** with `cubic_tol` 0.003 |
| Segments filled from the exact series | equal to production to 1e-16 (production-response fallback) |
| CPU time, 36 d with plunge | 1.0 s (production 1.5 s); 8 d without plunge: 0.12 s (production 0.34 s) |
| GPU | **never compiled** (no `nvcc` on the laptop); parity tests written and skipped |
| Tests | 98 passed, 27 skipped (all GPU-only), 95 subtests |

## What it computes

The STFT likelihood of `l2d-dev` works on contiguous segments of `nperseg = stft_dt / dt`
samples. In segment `j` (start `t_j`), channel `c`, bin `k` the template pixel is

```
H[c, j, k] = dt * rfft(w * x_c(t_j + n dt))[k],      k in [f_min, f_max] / df,   df = 1 / stft_dt
```

with `w` the symmetric Tukey window of `nperseg` points (the data's window, `tukey_alpha`), the
same quantity `EMRITDIWaveform.__call__` computes from its dense channels. The production path
builds `x_c` densely: FEW dense h+, hx -> `pyResponseTDI` -> per-segment window + rfft. The sparse
path writes each channel as a sum of harmonics,

```
x_c(t) = sum_s amp_{s,c}(t) cos(phase_{s,c}(t)),
```

and adds each harmonic's contribution to `H` per segment, from its amplitude and phase at the
segment midpoint, with an exact formula for a linear chirp.

## How the waveform works, step by step

The pieces, in the order a call runs them. Everything in `stft_tracks.py`,
`lat_stft_track_sum.{hh,cu}` and `sources/windowed_response.py` is source-agnostic; the EMRI
adapter is `EMRITrackSource`, `EMRISparseResponse`, `polarizations_from_item` and the waveform
class.

```
EMRISparseSTFTWaveform.__call__(m1, m2, a, p0, e0, x0, dist, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, ra, dec)
 1. EMRISparseResponse.template_inputs      FEW sparse holder: kept modes, DOPR853 phases, mode amplitudes
 2. EMRITrackSource(...)                    sparse TDI response -> ExactPhaseTDIOutput (per harmonic, channel)
 3. eval_tracks(segment midpoints)          amp, phase, f, fdot of every harmonic and channel
 4. stft_templates_from_tracks              classify every (template, segment): zero / tracks / dense
 5.   tracks -> STFTTrackAccumulator        STFTFresnelWrap.sum_tracks kernel (linear chirp per segment)
 6.   dense  -> td_channels(row, t)         exact channels -> window -> dt * rfft -> band
 -> (template (B, 3, NT, NF), start_freqs (B,), start_times (B,))
```

### 1. FEW sparse holder (`sources/emri/sparse_tracks.py`, `EMRISparseResponse`)

One FEW call per template on the integrator's own knots
(`FastKerrEccentricEquatorialFlux(..., return_sparse_holder=True)`, FEW `gpu_backend`) gives the
kept modes, their amplitudes on the knots, and the trajectory. `dense_inputs_from_holder` reads
the DOPR853 dense-output coefficients of the three fundamental phases, so the harmonic phase
`Phi_s = m Phi_phi + k Phi_theta + n Phi_r` is exact at any time. The item handed on is
`((t_k, C, mkn, amp_re, amp_im), (psi, lam, beta))`: knots, phase coefficients `(K-1, 3, 8)`,
harmonic indices, monomial cubics of the complex amplitudes, and the sky (`lam, beta` = `ra, dec`
in the orbits' frame; `psi` from FEW's `get_polarization_angle` of the ecliptic angles). The sky
convention is the production one: `(ra, dec)` in the orbits' frame for the response, converted to
the ecliptic for FEW. The trajectory is integrated `traj_pad` (2000 s) past the data window.

### 2. Sparse TDI response (`EMRISparseResponse.tracks`, `ExactPhaseTDIOutput`)

dev's dense TDI-on-the-fly kernel (`TDDenseTDIWaveform`, `cutils/lat_spline_tdi_waveform.cu`,
kept byte-identical to dev) evaluates the TDI response of every harmonic on a **sparse** grid:
the trajectory's knots plus a point every `sparse_dt` (12 h), kept `DELAY_MARGIN` (600 s) inside
the trajectory and the orbit tables. `ExactPhaseTDIOutput` writes the channel of harmonic `s` as
`amp(t) exp(-i (Phi_s(t) + r(t)))`: the carrier `Phi_s` exact from the DOPR853 coefficients,
only the amplitude and the slow residual phase `r` (Doppler delay, polarisation, transfer
function, the mode amplitude's own phase) splined on the sparse grid. A `-m` partner (negative
frequency) is mirrored. A template whose trajectory is too short for the response splines
(fewer than `MIN_RESPONSE_POINTS` = 4) has no tracks; its row is filled from the exact series.

### 3. Tracks at the segment midpoints (`EMRITrackSource.eval_tracks`)

`eval_tracer` evaluates `amp`, `phase`, `f = phase'/2pi` and `fdot = phase''/2pi` with analytic
derivatives (the carrier's exact ones plus the residual spline's), per (harmonic, channel), at the
midpoint of every segment. Nothing else of the harmonic is used on the track path.

### 4. Classification (`lisatools/stft_tracks.py::stft_templates_from_tracks`)

Every (template row, segment) is filled one of three ways:

* **zero**: the segment lies outside the row's signal span `[trajectory start, trajectory end +
  DELAY_MARGIN]`;
* **tracks**: every harmonic of the row passes, in that segment: its response grid covers the
  whole segment, its values are finite, the phase the linear chirp leaves out at the segment edge
  `cubic_phase_error = (pi/3) |fddot| (T/2)^3` is at most `cubic_tol` (0.03 rad; `fddot` from the
  differences of `fdot` to the neighbouring midpoints, worst channel), and its band half-width
  `0.5 |fdot| T^2 + n_side` is at most `max_half_width` (no cap by default);
* **dense**: otherwise; the whole (row, segment) comes from the row's exact channel series.

`last_segment_mode` keeps the decision of the last call (0 zero, 1 tracks, 2 dense).

### 5. Track segments: the Fresnel accumulator (`STFTTrackAccumulator`, `lat_stft_track_sum.{hh,cu}`)

The kernel `STFTFresnelWrap.sum_tracks` (bound in `binding_detector.cxx`) takes, per (component,
channel, segment), `(amp, phase, f, fdot)` at the midpoint and adds
`0.5 * STFTFresnel::get_fourier_value(...)` over the `half_width` bins around the mid-segment
frequency into `(row, channel, segment, bin)`. That is the windowed transform of a linear chirp
`amp cos(phase + 2 pi f tau + pi fdot tau^2)`, exact for any sweep width; the amplitude is constant
across the segment (the kernel accepts a slope; the EMRI path does not pass one). Conventions,
those of the STFT likelihood: pixel = `dt * rfft(w * x)`, positive-frequency half (the
negative-frequency image is neglected), Fourier origin at the segment start. A negative amplitude
becomes `(|amp|, phase + pi)`, a negative frequency is mirrored, `|fdot|` is floored at 1e-22.
The kernel knows nothing about segment times, so any hop between segments works.

The window is the data's **symmetric** Tukey window: `STFTFresnel(window_samples=N,
sample_cadence=T/N)` (`l2d-dev` `7db0fc80`); `EMRISparseSTFTWaveform` passes
`window_samples = nperseg`. The periodic window used before left an amplitude offset of about
0.5/N (2.3e-4 at N = 2160), the largest error of the first prototype.

Because every harmonic is added into one dense template, the inner product of the result carries
every cross term between harmonics: there are many, the harmonic bands overlap about four times
over on the grid (see `results.md`).

### 6. Dense segments: two exact fallbacks (`EMRITrackSource.td_channels`)

The driver asks the source for the exact channels of one row at all samples of its dense segments
(possibly non-contiguous), windows each segment and takes `dt * rfft`. Two sources of the exact
series, chosen by `fallback_via_polarizations`:

* **`False` (default): TDI-on-the-fly per harmonic.** `EMRISparseResponse.td_channels` runs dev's
  dense kernel at every sample, summed over harmonics, with the emission tapered over the last
  `stop_taper` (120 s) of the trajectory. Equal to production away from the plunge (1e-16); at the
  plunge segment the taper differs from production's ending (5e-4 in that segment). About 86 us
  per sample on the CPU.
* **`True`: the production response.** The holder's modes are summed into h+, hx at every sample
  (`polarizations_from_item`, as `TDDenseTDIWaveform::phases/strain_term`: `h+ = Re(A_p Z)`,
  `hx = Re(A_c Z)`, FEW's `Re`/`-Im`) and production's own `pyResponseTDI` is applied on windows
  of the data (`sources/windowed_response.py`). Those segments then **are** production's, plunge
  included (1e-16). About 4 us per sample on the CPU (20x cheaper).

How the production-response fallback reproduces production exactly:

* **Production's sample counts** (`EMRISparseSTFTWaveform._production_samples`). The
  polarizations production feeds the response are FEW's samples `k dt` after `waveform_t0`, the
  first `n_signal` of them summed, then FEW's zero padding up to `n_prod`. Rounding moves either
  end by one sample, so production's own float expressions are reproduced literally: `wave_gen`
  passes `T = int(merger_time/dt) dt` in years; the trajectory ends at `tmax_dimensionless * Msec`
  (`few.trajectory.integrate`) or at the plunge, whichever comes first; FEW's summation
  recomputes `n_pts = int(T YRSID_SI / dt)` (which loses one sample for about 2.5% of lengths at
  dt = 10 s) and pads only with `pad_output` (one more zero with `odd_len`). The tests check the
  result against FEW's own `num_pts`/`num_pts_pad` and the last nonzero sample of `wave_gen`.
* **Margins of the windowed response** (`response_margins`). An output sample needs
  `int(buffer_time/dt)` input samples before it (structural: the TDI kernel starts there; 1000 at
  dt = 10 s), of which only the last `reach_back` (about 95) are read, and reads `reach_fwd`
  (about 89) after it; both from the Lagrange stencil (order 35), the light-travel times and
  `R/c` (about 501 s), worst case over sky and epoch. `window_input_range` turns a window of data
  samples into production input indices; `response_on_window` runs `wf._apply_response` on that
  window and raises if a margin is short. The sub-sample shift `t0_shift_to_data` is recomputed
  exactly as production does. Lagrange projection only: `use_spline=True` is refused (its quintic
  spline is global over the input).
* **Runs** (`_fallback_channels`). The requested samples are sorted, runs closer than
  `left + reach_fwd` samples are merged (computing the gap is cheaper than a second window), each
  run costs its own length plus the margins, a window that reads only padding is skipped (exactly
  zero), and the result is gathered back in the caller's order.

`windowed_response.py` is source-agnostic: any `TDPyResponseWaveformBase` works; the caller gives
production's input span `(t_w0, n_prod)`, the only source-specific piece.

### Output

`(template, start_freqs, start_times)`: `template` complex128 `(B, 3, NT, NF)` on the waveform's
backend, every segment of the data grid (`NT = N // nperseg`, contiguous) and the bins of
`TDWaveformBase.fft` (`rfftfreq(nperseg, dt)` between `freq_min` and `freq_max`); `start_freqs` the
first bin; `start_times` the data start for every source. `merger_time` is ignored, as in
`EMRITDIWaveform`.

### Residual and time-domain signals

Everything the waveform returns is built from the same track source (`_track_source`: holder,
sparse response, `EMRITrackSource`), so what a move subtracts is what the likelihood scored.

- `get_signals_for_residuals(...)` in the analysis domain returns the templates of `__call__`
  themselves, as `STFTSignal`s on the STFT grid of the data (bit-identical arrays, no time
  series).
- `compute_tdi_channels(...)` is overridden: it returns the exact time series that fills the
  dense segments (`EMRITrackSource.td_channels`), on every data sample of the segments the signal
  touches. It follows `fallback_via_polarizations`: production's sum-then-respond (about
  4 us/sample on the CPU), or the TDI-on-the-fly kernel with `stop_taper` (about 86 us/sample).
  `get_td_signals` and the other output domains of `get_signals_for_residuals` (FD, WDM, explicit
  `domain_kwargs`) are the parent's, fed with these channels.

The windowed FFT of that time series equals the template in the dense segments (1e-12 with the
production fallback, bit for bit with the kernel) and is its linear-chirp model in the track
segments, so the two agree there to the track error set by `cubic_tol`: no time series has the
track template as its STFT exactly. Two differences from `EMRITDIWaveform.compute_tdi_channels`:
there is no sample outside the data, and the first `buffer_time` after `waveform_t0` is not
zeroed (the template does not zero it either). The production dense path is reached through a
separate `EMRITDIWaveform`.

## How to use it

Python:

```python
from lisatools.detector import EqualArmlengthOrbits
from lisatools.domains import TDSettings
from lisatools.sources.emri.response import EMRI_INSPIRAL_KWARGS, EMRI_SUM_KWARGS
from lisatools.sources.emri.waveform import EMRISparseSTFTWaveform

wf = EMRISparseSTFTWaveform(
    waveform_class="FastKerrEccentricEquatorialFlux",
    waveform_kwargs=dict(inspiral_kwargs=dict(EMRI_INSPIRAL_KWARGS), sum_kwargs=dict(EMRI_SUM_KWARGS)),
    waveform_t0=t0 - 86400.0, data_td_settings=TDSettings(N=N, dt=10.0, t0=t0),
    sampling_frequency=0.1, tukey_alpha=0.5, stft_dt=21600.0, freq_min=1e-4, freq_max=3e-2,
    orbits=EqualArmlengthOrbits(force_backend="cpu"), force_backend="cpu",
    cubic_tol=0.03, fallback_via_polarizations=True,
)
template, start_freqs, start_times = wf(m1, m2, a, p0, e0, x0, dist, qK, phiK,
                                        Phi_phi0, Phi_theta0, Phi_r0, ra, dec,
                                        mode_selection_threshold=1e-3)
```

Batches: pass 1-D arrays for the parameters (one FEW call per source, one accumulator launch for
the batch). The l2d EMRI block resolves waveform classes in `lisatools.sources.emri.waveform`, so
in the run configuration:

```yaml
waveform:
  class: EMRISparseSTFTWaveform
  kwargs: {stft_dt: 21600.0, cubic_tol: 0.03, fallback_via_polarizations: true, ...}
```

Settings beyond `EMRITDIWaveform`'s (all in `kwargs`, so a waveform can be rebuilt from them):

| Setting | Default | What it sets |
| --- | --- | --- |
| `cubic_tol` | 0.03 rad | Edge phase error a linear chirp may leave before a segment goes dense. The main accuracy/speed knob: 0.003 cuts the plunging mismatch 7x for +25% CPU time (`results.md`). |
| `fallback_via_polarizations` | `False` | Which exact series fills dense segments: TDI-on-the-fly with `stop_taper` (`False`) or the production response (`True`, exact and 20x cheaper per sample). |
| `n_side` | 30 | Bins kept beyond half a harmonic's sweep, each side, on track segments. |
| `max_half_width` | `None` | Cap on a track segment's band half-width; wider goes dense. |
| `sparse_dt` | 43200 s | Largest spacing of the sparse response grid. |
| `stop_taper` | 120 s | Emission taper before the end of the inspiral, TDI-on-the-fly fallback only. |

Requirements: an STFT analysis domain (`stft_dt` set; otherwise `ValueError`), FEW `gpu_backend`
(the sparse holder), and for the production-response fallback the Lagrange projection
(`use_spline=False`, the default).

Memory: every FEW `GenerateEMRIWaveform("FastKerrEccentricEquatorialFlux")` holds about 5.5 GB
while alive, and each waveform object builds its own (a production + sparse pair is about 11 GB on
the CPU). Run FEW-heavy jobs one at a time on a 24 GB machine; the tests release their waveforms
between classes.

## The branch

`feat-emri-sparse-stft` on `l2d-dev` (`0ecf0cde`), oldest first:

| Commit | What it adds |
| --- | --- |
| `619f7315`, `4460b26b` | Cache the configured `Orbits` in the TDI-on-the-fly response instead of deep-copying it per evaluation, keyed by backend and device. |
| `b9bf31bf`, `de2a4d5a`, `62375473` | dev's `TDDenseTDIWaveform` (template-batched TD TDI-on-the-fly, exact dense-output phases, raw channels, stop taper) and `wdm_lookup_kernels.hh`, byte-identical to dev. |
| `51ecb9e2` | `sources/emri/sparse_tracks.py`: dev's sparse front end lifted out of its WDM module, independent of the output domain; `EMRISparseResponse`. |
| `110119fd` | `scripts/emri/sparse_stft_fresnel_prototype.py`: Python accuracy prototype. |
| `d637460f` | `lat_stft_track_sum.{hh,cu}`: the source-agnostic multi-component track accumulator. |
| `6443e263` | `stft_tracks.py`: `SparseTrackSource` protocol, `STFTTrackAccumulator`, `stft_templates_from_tracks` (tracks / dense / zero); `EMRITrackSource`. |
| `0e2171a7` | `EMRISparseSTFTWaveform`, the drop-in template. |
| `bce0201a` | The data's symmetric Tukey window in the accumulator (`window_samples`). |
| `b03412ca` | The production-response fallback: `windowed_response.py`, `polarizations_from_item`, `_production_samples`, `_fallback_channels`, `fallback_via_polarizations`; with the fixes of its review (FEW's `pad_output=False`, the sample lost in the years round trip, `use_spline` refused). |

Uncommitted: `scripts/emri/compare_sparse_stft_vs_dense.py` (the comparison against production
used throughout `results.md`) and the worktree's `HANDOFF.md`. Backup of the branch before the
rebase onto `l2d-dev`: `feat-emri-sparse-stft-pre-l2d`.

## How dev computes EMRI likelihoods (for comparison)

dev (`e25ff72f`) scores EMRIs in the global fit with `EMRIDirectLikeMove`
(`globalfit/moves/emridirectmove.py`, `EMRI_LIKELIHOOD=direct`): per chunk of 8 rows, one FEW call
per row, one dense TDI-on-the-fly launch, and the `wdm_lookup_sum` kernel scatter-adds every
(harmonic, pixel) of the n_ref WDM lookup table into a full-grid `(3, Nf, Nt)` accumulator per
row (about 150 MB at Nf 1440 x Nt 4320), cropped to the active box and scored with the generic
`inner_product`; the plunge tail is an exact dense chunk. The template is therefore built in full
in the WDM domain. dev's only fused lookup + inner-product kernel is for SOBBHs
(`cutils/sobbh_lookup_kernel.cu`, mode 0 accumulates `<d|h>`, `<h|h>` per row with nothing per
pixel in memory), which works because a SOBBH row is one harmonic: all its channels sit on the
same pixel. An EMRI row is many overlapping harmonics, so `<h|h>` needs the summed template per
pixel, which is why dev materialises it. In LAT, `cutils/lat_stft_kernels.hh` is the fused STFT
likelihood of the GBs (one source per thread block, nothing per pixel in memory).

## Open points

In order of what they decide.

1. **GPU.** Compile the CUDA path (never done), run the skipped parity classes of
   `tests/test_stft_track_sum.py`, and time single and batched templates, with and without a
   plunge, against the production 300 ms per likelihood on 2 years. The GPU profile decides what
   to optimise next: on the CPU the per-pixel Fresnel fill is 97% of the time (2.2 s of 2.3 s for
   180 days), which the GPU parallelises.
2. **Defaults.** Make `fallback_via_polarizations=True` the default (exact, faster), and lower
   `cubic_tol` to about 0.003 (mismatch 7.6e-6 -> 1.0e-6 noise-weighted on the 36-d plunging
   case, SNR^2 x mismatch 7.3 -> 1.0 at SNR 980, +25% CPU time). Both are one-line changes,
   waiting for a decision.
3. **Likelihood-level validation.** `<d|h>`, `<h|h>` and the log-likelihood through the STFT
   likelihood of the l2d EMRI block on the `gf_dev/interface/emri_mbh_psd` settings (the
   residual templates are now the scoring templates themselves).
4. **A realistic 2-year EMRI.** Fraction of dense segments and the cost of the fallback for
   catalogue sources (the presets used so far are 8 and 36 days).
5. **Accuracy floor of the track segments.** With the exact fallback the remaining error is in
   the Fresnel segments (1.2e-5 white / 1.4e-5 noise-weighted over them at `cubic_tol` 0.03).
   The tracks themselves are better than the linear-chirp model built on them: summed densely
   they match the exact channels to 6e-6 - 2e-5 in most segments. Candidates: the amplitude
   slope the kernel already accepts (`linear_envelope`), or sampling each track inside the
   segment and taking a small FFT (the het-FFT design of the GB STFT,
   `docs/specs/2026-07-10-stft-gb-fresnel-accuracy-and-het-fft-successor.md`).
6. **Track accuracy anomalies.** On the 36-d plunging preset the tracks are off by 9e-4 in
   segment 100 and 1.4e-4 in segment 103 (a track segment), about 8 days before the plunge, far
   larger than in the segments around them, and by up to 1.4e-3 in the last 1.5 days. Not
   understood yet (sparse grid spacing, amplitude spline?). It sets the accuracy of the track
   segments there and of any heterodyne built on tracks.
7. **Speed of the track segments.** If the GPU profile shows the Fresnel fill and the full-grid
   inner product dominate: a fused STFT inner product over harmonics (one thread per pixel,
   looping over the harmonics whose band covers it, so `<h|h>` stays local), or an inner-product
   heterodyne against a reference for parameter estimation (relative binning in frequency within
   a segment and in time across segments; per-walker data sums rebuilt when the residual
   changes). Parked until the GPU numbers exist.
8. **Heterodyning the dense end steps: not worth it** (studied 2026-11-20). With the exact
   fallback at about 4 us per sample those segments are exact and cheap (81 dense segments cost
   0.25 s more than 31 on the CPU); a heterodyne against a reference (template = sum of ratio
   coefficients times exact reference basis spectra) would only hold within about 0.1 bin of the
   reference (plunge-time offsets of 500 s eight days before the plunge down to 30 s in the last
   segment) and could only match or lose against exact. Details in `results.md`.
9. **Smaller items.**
   * The negative-frequency image of each track is neglected on track segments (as in the
     Fresnel kernel); bound it for the lowest-frequency harmonics.
   * `waveform_t0 == data_t0`: production zeroes its first `buffer_time` (10000 s) after
     `waveform_t0`, the sparse path does not, so the first segment differs (mismatch 0.25 there);
     decide whether to reproduce it. The comparisons start the waveform a day before the data.
   * When production's trajectory is cut at its `T` before a plunge, the holder's trajectory runs
     on and their last DOPR853 steps differ: h differs by 2.5e-8 to 2.4e-5 relative there
     (integrator tolerance, a phase error of that order and a mismatch below 1e-9; accepted).
   * Overlapping segments (`origin/feat/stft-overlap`): the driver accepts any segment starts;
     `EMRISparseSTFTWaveform` uses the contiguous grid; the side-bin count needs re-deriving.
   * Each waveform object builds its own FEW generator (5.5 GB); share it between a production
     and a sparse waveform, and watch it with one replica per GPU in the global fit.
   * Housekeeping: commit `scripts/emri/compare_sparse_stft_vs_dense.py`; delete the backup
     branch `feat-emri-sparse-stft-pre-l2d`; re-check `tests/test_l2d_sources.py` (36 failures
     from a missing `bbhx` before the rebase) and
     `tests/test_gbspecial_flow_stft.py::STFTProposeFlowTest::test_in_model_propose_two_passes`;
     the branch's commits carry the git-config email with one "s".
