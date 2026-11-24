"""Compare the sparse-track EMRI STFT template with the dense production template.

Builds the same source with

* ``EMRITDIWaveform`` (``sources/emri/waveform.py``): FEW dense h+/hx -> dense TDI response ->
  per-segment Tukey window + rfft (production), and
* ``EMRISparseSTFTWaveform``: FEW sparse holder -> sparse TDI response -> per-harmonic linear
  chirps through the STFT Fresnel kernel, dense fallback where a harmonic chirps too fast or
  the inspiral ends,

on one STFT grid, then reports mismatches (white noise and, with ``--psd``, noise-weighted in
A/E/T), timings and the fill of every segment, and saves three figures:

* ``spectrograms.png``: |h| of both and |difference| for one channel;
* ``segments.png``: per-segment mismatch, power ratio and weight in the inner product;
* ``slice.png``: |h(f)| and the phase difference in one segment (default: the worst).

Examples::

    python scripts/emri/compare_sparse_stft_vs_dense.py                       # 8 d, no plunge
    python scripts/emri/compare_sparse_stft_vs_dense.py --preset plunging --psd
    python scripts/emri/compare_sparse_stft_vs_dense.py --days 30 --p0 8.5 --cubic-tol 0.01 --no-show

FEW must find its data files, e.g. ``FEW_FILE_ALLOW_DOWNLOAD=0
FEW_FILE_EXTRA_PATHS="$HOME/Library/Application Support/few/v2.1.0/download"``.

Conventions to keep in mind when reading the numbers:

* The production path zeroes the first ``buffer_time`` (10000 s) after ``waveform_t0``; the
  sparse path does not. Start the waveform before the data (``--lead-days``, default 1) or the
  first segment differs by construction.
* Segments the tracks cannot represent (fast chirp, plunge) are filled by default with the
  per-harmonic TDI-on-the-fly response and a ``--stop-taper`` [s] at the plunge (the plunge
  segment then differs from production at 1e-3); ``--fallback-via-polarizations`` sums the
  modes into h+, hx and applies the production response instead, so those segments are the
  production template's.
* The noise-weighted numbers use a diagonal A/E/T noise (analytic ``A/E/T{1,2}TDISens``) with the
  STFT-pixel variance (S/2) T mean(w^2); the SNR is an estimate, the mismatches do not depend on
  the normalisation.
"""

import argparse
import os
import time

import numpy as np

DAY = 86400.0
MODE_NAMES = {0: "zero", 1: "tracks", 2: "dense"}
MODE_COLOURS = {0: "0.6", 1: "tab:blue", 2: "tab:red"}

PRESETS = {
    # non-plunging: the whole window on tracks
    "nonplunging": dict(m1=1e6, m2=10.0, a=0.9, p0=9.5, e0=0.3, days=8.0),
    # plunges at about day 32.4: fast-chirp segments go to the dense fallback
    "plunging": dict(m1=1e6, m2=100.0, a=0.9, p0=7.0, e0=0.4, days=36.0),
}
PARAM_NAMES = ("m1", "m2", "a", "p0", "e0", "x0", "dist", "qK", "phiK",
               "Phi_phi0", "Phi_theta0", "Phi_r0", "ra", "dec")
PARAM_DEFAULTS = dict(m1=1e6, m2=10.0, a=0.9, p0=9.5, e0=0.3, x0=1.0, dist=1.0, qK=0.5,
                      phiK=2.0, Phi_phi0=0.3, Phi_theta0=0.0, Phi_r0=1.2, ra=1.1, dec=0.4)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--preset", choices=sorted(PRESETS), default="nonplunging",
                   help="source and duration; explicit flags below override it")
    src = p.add_argument_group("source (FEW parameters; ra, dec in the orbits' frame)")
    for name in PARAM_NAMES:
        src.add_argument(f"--{name}", type=float, default=None,
                         help=f"default: preset, else {PARAM_DEFAULTS[name]}")
    grid = p.add_argument_group("data and STFT grid")
    grid.add_argument("--days", type=float, default=None, help="data length [d] (default: preset)")
    grid.add_argument("--lead-days", type=float, default=1.0,
                      help="waveform start before the data start [d]")
    grid.add_argument("--t0", type=float, default=2.0e7, help="data start [s]")
    grid.add_argument("--dt", type=float, default=10.0, help="data sample spacing [s]")
    grid.add_argument("--stft-dt", type=float, default=21600.0, help="segment length [s]")
    grid.add_argument("--tukey-alpha", type=float, default=0.5)
    grid.add_argument("--freq-min", type=float, default=1e-4)
    grid.add_argument("--freq-max", type=float, default=3e-2)
    grid.add_argument("--tdi-generation", choices=("1st", "2nd"), default="2nd")
    grid.add_argument("--orbits", choices=("equal", "esa"), default="equal")
    grid.add_argument("--eps", type=float, default=1e-3, help="FEW mode_selection_threshold")
    sp = p.add_argument_group("sparse path")
    sp.add_argument("--n-side", type=int, default=30)
    sp.add_argument("--cubic-tol", type=float, default=0.03)
    sp.add_argument("--max-half-width", type=int, default=None)
    sp.add_argument("--sparse-dt", type=float, default=43200.0)
    sp.add_argument("--stop-taper", type=float, default=120.0,
                    help="emission taper at the plunge [s] (TDI-on-the-fly fallback only)")
    sp.add_argument("--fallback-via-polarizations", action="store_true",
                    help="fill the segments the tracks cannot represent with modes -> h+, hx -> "
                         "production response instead of the per-harmonic TDI-on-the-fly response")
    out = p.add_argument_group("output")
    out.add_argument("--psd", action="store_true",
                     help="also noise-weighted mismatches and an SNR estimate (A/E/T)")
    out.add_argument("--channel", default=None,
                     help="channel of the spectrogram/slice figures: X, Y, Z (or A, E, T with --psd)")
    out.add_argument("--segment", type=int, default=None,
                     help="segment of slice.png (default: the worst one)")
    out.add_argument("--repeat", type=int, default=3, help="timed calls per path (after a warm-up)")
    out.add_argument("--backend", default="cpu")
    out.add_argument("--outdir", default=None, help="default: ./emri_sparse_vs_dense_<preset>")
    out.add_argument("--save-npz", action="store_true", help="also save both templates")
    out.add_argument("--no-show", action="store_true", help="only save the figures")
    args = p.parse_args()

    preset = PRESETS[args.preset]
    for name in PARAM_NAMES:
        if getattr(args, name) is None:
            setattr(args, name, preset.get(name, PARAM_DEFAULTS[name]))
    if args.days is None:
        args.days = preset["days"]
    if args.outdir is None:
        args.outdir = f"emri_sparse_vs_dense_{args.preset}"
    return args


def host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def build_waveforms(args):
    from lisatools.detector import EqualArmlengthOrbits, ESAOrbits
    from lisatools.domains import TDSettings
    from lisatools.sources.emri.response import EMRI_INSPIRAL_KWARGS, EMRI_SUM_KWARGS
    from lisatools.sources.emri.waveform import EMRISparseSTFTWaveform, EMRITDIWaveform

    orbits_class = EqualArmlengthOrbits if args.orbits == "equal" else ESAOrbits
    num_samples = int(round(args.days * DAY / args.dt))
    common = dict(
        waveform_class="FastKerrEccentricEquatorialFlux",
        waveform_kwargs=dict(inspiral_kwargs=dict(EMRI_INSPIRAL_KWARGS),
                             sum_kwargs=dict(EMRI_SUM_KWARGS)),
        waveform_t0=args.t0 - args.lead_days * DAY,
        data_td_settings=TDSettings(N=num_samples, dt=args.dt, t0=args.t0),
        sampling_frequency=1.0 / args.dt,
        tdi_generation=f"{args.tdi_generation} generation",
        tdi_channels="XYZ",   # the sparse path returns XYZ
        tukey_alpha=args.tukey_alpha, stft_dt=args.stft_dt,
        freq_min=args.freq_min, freq_max=args.freq_max, force_backend=args.backend)
    dense = EMRITDIWaveform(orbits=orbits_class(force_backend=args.backend), **common)
    sparse = EMRISparseSTFTWaveform(
        orbits=orbits_class(force_backend=args.backend), n_side=args.n_side,
        cubic_tol=args.cubic_tol, max_half_width=args.max_half_width, sparse_dt=args.sparse_dt,
        stop_taper=args.stop_taper, fallback_via_polarizations=args.fallback_via_polarizations,
        **common)
    return dense, sparse


def timed(fn, params, kwargs, repeat):
    """First call (setup + warm-up), then the best of ``repeat`` calls."""
    t = time.perf_counter()
    out = fn(*params, **kwargs)
    first = time.perf_counter() - t
    best = np.inf
    for _ in range(repeat):
        t = time.perf_counter()
        out = fn(*params, **kwargs)
        best = min(best, time.perf_counter() - t)
    return out, first, (best if repeat > 0 else first)


def on_common_grid(out_dense, out_sparse, data_t0, stft_dt, df):
    """Both templates as (3, NT, NF) on the sparse (full data) grid."""
    h_d, sf_d, st_d = (host(v) for v in out_dense)
    h_s, sf_s, st_s = (host(v) for v in out_sparse)
    h_d, h_s = h_d[0], h_s[0]
    if float(st_s[0]) != data_t0:
        raise RuntimeError("the sparse template does not start at the data start.")
    j0 = int(round((float(st_d[0]) - data_t0) / stft_dt))
    k0 = int(round((float(sf_d[0]) - float(sf_s[0])) / df))
    _, nt_d, nf_d = h_d.shape
    if j0 < 0 or k0 < 0 or j0 + nt_d > h_s.shape[1] or k0 + nf_d > h_s.shape[2]:
        raise RuntimeError(f"the dense template ({nt_d} x {nf_d} from segment {j0}, bin {k0}) "
                           f"is not inside the sparse one {h_s.shape[1:]}.")
    full = np.zeros_like(h_s)
    full[:, j0:j0 + nt_d, k0:k0 + nf_d] = h_d
    return full, h_s, (j0, nt_d, k0, nf_d)


def to_aet(h):
    from lisatools.utils.utility import AET

    return np.stack(AET(h[0], h[1], h[2]))


def noise_weights(args, freqs, window):
    """1 / variance of an STFT pixel, (3, 1, NF), for A, E, T."""
    from lisatools import sensitivity as sens

    names = ("A", "E", "T")
    gen = "1" if args.tdi_generation == "1st" else "2"
    psd = np.stack([getattr(sens, f"{c}{gen}TDISens").get_Sn(freqs) for c in names])
    var = 0.5 * psd * args.stft_dt * np.mean(window ** 2)
    return (1.0 / var)[:, None, :]


def inner(a, b, w=None, axis=None):
    """Re <a|b> with pixel weights w (white: 1); the factor 2 of a complex Gaussian pixel."""
    prod = np.conj(a) * b if w is None else np.conj(a) * b * w
    return 2.0 * np.real(prod).sum(axis=axis)


def mismatch(a, b, w=None, axis=None):
    ab, aa, bb = inner(a, b, w, axis), inner(a, a, w, axis), inner(b, b, w, axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 1.0 - ab / np.sqrt(aa * bb)


def report(label, h_ref, h_new, w, mode, channel_names):
    """Print the numbers of one weighting; return the per-segment mismatch and weight."""
    aa, bb = inner(h_new, h_new, w), inner(h_ref, h_ref, w)
    overlap = np.sum(np.conj(h_new) * h_ref * (1.0 if w is None else w))
    print(f"\n== {label}")
    print(f"  mismatch          {mismatch(h_new, h_ref, w):.3e}")
    print(f"  |h_sparse|/|h_dense| {np.sqrt(aa / bb):.8f}   phase of <sparse|dense> "
          f"{np.angle(overlap):+.2e} rad")
    for c, name in enumerate(channel_names):
        wc = None if w is None else w[c]
        print(f"  channel {name}: mismatch {mismatch(h_new[c], h_ref[c], wc):.3e}, "
              f"share of <h|h> {inner(h_ref[c], h_ref[c], wc) / bb:.3g}")
    per_seg = mismatch(h_new, h_ref, w, axis=(0, 2))
    weight = inner(h_ref, h_ref, w, axis=(0, 2)) / bb
    has = weight > 1e-12
    print(f"  per segment (segments with signal): median {np.median(per_seg[has]):.2e}, "
          f"max {np.max(per_seg[has]):.2e} at segment {int(np.argmax(np.where(has, per_seg, -1)))}")
    for m, name in MODE_NAMES.items():
        sel = mode == m
        if sel.any() and m != 0:
            print(f"  {name:6s} segments: {int(sel.sum()):4d}, share of <h|h> {weight[sel].sum():.3f}, "
                  f"mismatch over them {mismatch(h_new[:, sel], h_ref[:, sel], w):.3e}")
    outside = mode == 0
    if outside.any():
        print(f"  zero segments: {int(outside.sum())}, dense power there "
              f"{weight[outside].sum():.2e} of <h|h>")
    return per_seg, weight


def plot(args, h_ref, h_new, mode, per_seg, weight, channel_names, channel, freqs, t_days, label):
    import matplotlib.pyplot as plt

    c = channel_names.index(channel)
    diff = h_new - h_ref
    os.makedirs(args.outdir, exist_ok=True)
    edges_t = np.append(t_days, t_days[-1] + args.stft_dt / DAY)
    df = freqs[1] - freqs[0]
    edges_f = np.append(freqs - 0.5 * df, freqs[-1] + 0.5 * df)
    edges_f[0] = max(edges_f[0], 0.5 * edges_f[1])   # log axis
    peak = np.abs(h_ref[c]).max()
    floor = peak * 1e-8

    # --- spectrograms
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True, sharey=True,
                             constrained_layout=True)
    panels = ((h_ref[c], f"dense (production) |{channel}|"), (h_new[c], f"sparse |{channel}|"),
              (diff[c], f"|sparse - dense| {channel}"))
    for ax, (z, title) in zip(axes, panels):
        im = ax.pcolormesh(edges_t, edges_f, np.log10(np.maximum(np.abs(z), floor) / peak).T,
                           vmin=-8, vmax=0, cmap="viridis", shading="flat")
        ax.set_title(title)
        ax.set_xlabel("time since data start [d]")
        ax.set_yscale("log")
    axes[0].set_ylabel("frequency [Hz]")
    for j in np.flatnonzero(mode == 2):   # dense-fallback segments
        axes[1].axvspan(edges_t[j], edges_t[j + 1], color="tab:red", alpha=0.08, lw=0)
    fig.colorbar(im, ax=axes, label="log10 |h| / max |h_dense|")
    fig.suptitle(f"{args.preset}: spectrograms (red band on the sparse panel: dense fallback)")
    fig.savefig(os.path.join(args.outdir, "spectrograms.png"), dpi=130)

    # --- per segment
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True, constrained_layout=True)
    mid = t_days + 0.5 * args.stft_dt / DAY
    has = weight > 1e-12
    power_ref = np.where(has, inner(h_ref, h_ref, None, axis=(0, 2)), 1.0)
    ratio = np.sqrt(inner(h_new, h_new, None, axis=(0, 2)) / power_ref) - 1.0
    for m, name in MODE_NAMES.items():
        sel = (mode == m) & has
        if sel.any():
            axes[0].semilogy(mid[sel], np.maximum(per_seg[sel], 1e-16), "o", ms=4,
                             color=MODE_COLOURS[m], label=name)
            axes[1].plot(mid[sel], ratio[sel], "o", ms=4, color=MODE_COLOURS[m])
            axes[2].semilogy(mid[sel], weight[sel], "o", ms=4, color=MODE_COLOURS[m])
    axes[0].set_ylabel(f"mismatch ({label})")
    axes[0].legend(title="sparse fill")
    axes[1].axhline(0.0, color="k", lw=0.5)
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1].set_ylabel("|h_sparse| / |h_dense| - 1\n(white)")
    axes[2].set_ylabel(f"share of <h|h>\n({label})")
    axes[2].set_xlabel("segment midpoint, time since data start [d]")
    fig.suptitle(f"{args.preset}: per-segment comparison")
    fig.savefig(os.path.join(args.outdir, "segments.png"), dpi=130)

    # --- one segment
    j = args.segment if args.segment is not None else int(np.argmax(np.where(has, per_seg, -1)))
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    a, b = h_new[c, j], h_ref[c, j]
    axes[0].semilogy(freqs, np.abs(b), lw=1.5, label="dense")
    axes[0].semilogy(freqs, np.abs(a), "--", lw=1.0, label="sparse")
    axes[0].semilogy(freqs, np.abs(a - b), lw=0.8, color="k", label="|difference|")
    axes[0].set_ylim(max(np.abs(b).max(), 1e-300) * 1e-10, None)
    axes[0].set_ylabel(f"|{channel}|")
    axes[0].legend()
    sig = np.abs(b) > 1e-3 * np.abs(b).max()
    axes[1].plot(freqs[sig], np.angle(a[sig] * np.conj(b[sig])), ".", ms=3)
    axes[1].set_ylabel("arg(sparse) - arg(dense) [rad]\n(bins above 1e-3 of the peak)")
    axes[1].set_xlabel("frequency [Hz]")
    axes[1].set_xscale("log")
    fig.suptitle(f"{args.preset}: segment {j} (t = {t_days[j]:.2f} d, fill "
                 f"{MODE_NAMES[int(mode[j])]}, mismatch {per_seg[j]:.2e} {label})")
    fig.savefig(os.path.join(args.outdir, "slice.png"), dpi=130)

    print(f"\nfigures in {os.path.abspath(args.outdir)}")
    if not args.no_show:
        plt.show()


def main():
    args = parse_args()
    from lisatools.utils.utility import tukey

    dense, sparse = build_waveforms(args)
    params = tuple(getattr(args, n) for n in PARAM_NAMES)
    kwargs = dict(mode_selection_threshold=args.eps)
    print("source:", ", ".join(f"{n}={v:g}" for n, v in zip(PARAM_NAMES, params)))
    print(f"data: {args.days:g} d at {args.dt:g} s from t0 = {args.t0:g} s; waveform starts "
          f"{args.lead_days:g} d earlier; segments {args.stft_dt:g} s, Tukey {args.tukey_alpha:g}; "
          f"TDI {args.tdi_generation} generation, {args.orbits} orbits; eps {args.eps:g}")

    out_d, first_d, best_d = timed(dense, params, kwargs, args.repeat)
    out_s, first_s, best_s = timed(sparse, params, kwargs, args.repeat)
    print(f"\ntiming ({args.backend}): dense first {first_d:.2f} s, best {best_d:.3f} s; "
          f"sparse first {first_s:.2f} s, best {best_s:.3f} s")

    df = 1.0 / args.stft_dt
    h_ref, h_new, (j0, nt_d, k0, nf_d) = on_common_grid(out_d, out_s, args.t0, args.stft_dt, df)
    mode = host(sparse.last_segment_mode)[0]
    num_seg, num_freq = h_new.shape[1:]
    freqs = float(host(out_s[1])[0]) + df * np.arange(num_freq)
    t_days = np.arange(num_seg) * args.stft_dt / DAY
    print(f"grid: {num_seg} segments x {num_freq} bins from {freqs[0]:.4e} Hz; dense template "
          f"covers segments {j0}..{j0 + nt_d - 1}, bins {k0}..{k0 + nf_d - 1}")
    print("sparse fill: " + ", ".join(f"{MODE_NAMES[m]} {int((mode == m).sum())}" for m in MODE_NAMES))

    per_seg, weight = report("white noise, XYZ", h_ref, h_new, None, mode, ["X", "Y", "Z"])
    label, names, ref_plot, new_plot = "white", ["X", "Y", "Z"], h_ref, h_new
    if args.psd:
        window = host(tukey(sparse.nperseg, alpha=args.tukey_alpha)) if args.tukey_alpha else np.ones(sparse.nperseg)
        w = noise_weights(args, freqs, window)
        ref_aet, new_aet = to_aet(h_ref), to_aet(h_new)
        snr = np.sqrt(inner(ref_aet, ref_aet, w))
        per_seg, weight = report(f"noise-weighted, AET (TDI{args.tdi_generation[0]} analytic PSD)",
                                 ref_aet, new_aet, w, mode, ["A", "E", "T"])
        mm = mismatch(new_aet, ref_aet, w)
        print(f"  SNR of the dense template (estimate) {snr:.1f};  SNR^2 x mismatch {snr ** 2 * mm:.2e} "
              f"(order of the log-likelihood error at the true parameters)")
        label, names, ref_plot, new_plot = "noise-weighted", ["A", "E", "T"], ref_aet, new_aet

    if args.save_npz:
        os.makedirs(args.outdir, exist_ok=True)
        path = os.path.join(args.outdir, "templates.npz")
        np.savez(path, dense=h_ref, sparse=h_new, mode=mode, freqs=freqs, t_days=t_days,
                 params=np.array(params), param_names=np.array(PARAM_NAMES))
        print(f"templates saved to {os.path.abspath(path)}")

    channel = args.channel or names[0]
    if channel not in names:
        raise SystemExit(f"--channel must be one of {names} here.")
    plot(args, ref_plot, new_plot, mode, per_seg, weight, names, channel, freqs, t_days, label)


if __name__ == "__main__":
    main()
