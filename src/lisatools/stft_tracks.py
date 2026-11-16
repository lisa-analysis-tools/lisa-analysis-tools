"""STFT templates of signals given as sums of slowly chirping tracks (source-agnostic).

A template row is a sum of *components*; per TDI channel a component is the real signal
``amp(t) * cos(phase(t))``. Within one STFT segment each component is modelled as a linear chirp
about the segment anchor and its windowed transform is evaluated in closed form by the backend's
``STFTFresnel`` (``cutils/lat_stft_track_sum.{hh,cu}``), then added into a dense template
``(row, channel, segment, bin)``. The dense inner product of that template carries all cross
terms between the components of a row.

Nothing here depends on the kind of source. A source takes part by implementing
:class:`SparseTrackSource`: harmonics of an EMRI, modes of a massive black hole binary, or one
track of a quasi-monochromatic source are all "components".

Conventions (those of ``STFTFresnel`` and the STFT likelihood): a pixel is ``dt * rfft(w * x)``
at ``f_min + k * df`` with ``df = 1 / window_length``, Fourier origin at the segment start, ``w``
the periodic Tukey window. The anchor of a segment is its midpoint. Segment start times never
enter the kernel, so the hop between segments need not equal the window length.
"""

from __future__ import annotations

from typing import Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from . import get_backend

__all__ = [
    "SparseTrackSource",
    "STFTTrackAccumulator",
    "stft_templates_from_tracks",
    "track_half_width",
    "cubic_phase_error",
    "periodic_tukey",
]


@runtime_checkable
class SparseTrackSource(Protocol):
    """What a source provides to be turned into STFT templates by track accumulation.

    Shapes: ``C`` components in total over all template rows, ``nch`` TDI channels. Arrays live
    on the array module of the accumulator's backend, except the spans (host).
    """

    #: Number of template rows.
    num_rows: int
    #: (C,) host int array, template row of each component.
    component_row: np.ndarray

    def eval_tracks(self, t_anchor) -> Tuple:
        """``(amp, phase, f, fdot)`` at the times ``t_anchor`` (NT,), each ``(C, nch, NT)``.

        ``amp * cos(phase)`` is the channel's time series, ``f = phase' / 2 pi`` and
        ``fdot = phase'' / 2 pi``. The amplitude may be signed. Must accept any time; values
        outside :meth:`track_span` are ignored.
        """
        ...

    def track_span(self) -> Tuple:
        """``(t_lo, t_hi)``, each ``(C,)``: the times between which the tracks are defined."""
        ...

    def signal_span(self) -> Tuple:
        """``(t_lo, t_hi)``, each ``(num_rows,)``: outside it the row's signal is zero."""
        ...

    def td_channels(self, row: int, t):
        """Exact time series ``(nch, len(t))`` of one row at the times ``t``, all components.

        Used for the segments the linear chirp cannot represent. A signal that stops inside
        ``t`` must be tapered to zero by the source.
        """
        ...


def track_half_width(fdot, window_length: float, n_side: int, xp=np):
    """Band half-width in bins, ``(C, NT)``, from ``fdot`` ``(C, nch, NT)``.

    Half the sweep of the widest channel, ``|fdot| * window_length**2 / 2`` bins, plus ``n_side``.
    """
    half_sweep = 0.5 * xp.abs(fdot).max(axis=1) * window_length**2
    half_sweep = xp.where(xp.isfinite(half_sweep), half_sweep, 0.0)
    return (xp.ceil(half_sweep) + n_side).astype(xp.int32)


def cubic_phase_error(fdot, t_anchor, window_length: float, valid=None, xp=np):
    """Phase the linear chirp leaves out at a segment edge, ``(C, NT)`` [rad].

    ``(pi / 3) * |fddot| * (window_length / 2)**3``, with ``|fddot|`` the larger of the two
    differences of ``fdot`` ``(C, nch, NT)`` to the neighbouring anchors ``t_anchor`` (NT,), and
    the largest channel. ``valid`` ``(C, NT)`` marks the anchors where ``fdot`` is defined: a
    difference to an anchor that is not valid is left out, and an anchor without any valid
    neighbour gets ``inf`` (its chirp rate cannot be checked).
    """
    num_comp, _, num_seg = fdot.shape
    if valid is None:
        valid = xp.ones((num_comp, num_seg), dtype=bool)
    valid = xp.asarray(valid, dtype=bool)
    fddot = xp.full((num_comp, num_seg), -1.0)
    if num_seg > 1:
        t_anchor = xp.asarray(t_anchor, dtype=float)
        step = xp.abs(xp.diff(fdot, axis=-1) / xp.diff(t_anchor)).max(axis=1)   # (C, NT - 1)
        step = xp.where(valid[:, 1:] & valid[:, :-1] & xp.isfinite(step), step, -1.0)
        fddot[:, :-1] = step
        fddot[:, 1:] = xp.maximum(fddot[:, 1:], step)
    fddot = xp.where(fddot < 0.0, xp.inf, fddot)
    return (np.pi / 3.0) * fddot * (0.5 * window_length) ** 3


def periodic_tukey(num: int, alpha: float, xp=np):
    """Periodic Tukey window of ``num`` samples and taper fraction ``alpha``: the window of
    ``STFTFresnel`` sampled at ``n / num`` of the segment (``scipy`` ``tukey(num, alpha,
    sym=False)``)."""
    u = xp.arange(num) / num
    if alpha <= 0.0:
        return xp.ones(num)
    half = 0.5 * alpha
    w = xp.ones(num)
    w = xp.where(u < half, 0.5 * (1.0 - xp.cos(np.pi * u / half)), w)
    w = xp.where(u > 1.0 - half, 0.5 * (1.0 - xp.cos(np.pi * (1.0 - u) / half)), w)
    return w


class STFTTrackAccumulator:
    """Adds chirping tracks into dense STFT templates on one backend.

    Args:
        window_length: Length of one STFT segment [s]; the bin spacing is its inverse.
        f_min: Frequency of bin 0 of the templates [Hz].
        num_freqs: Number of bins of the templates.
        window_alpha: Taper fraction of the Tukey window (0 is a boxcar).
        linear_envelope: Use the amplitude slope of each track (``slope`` in :meth:`__call__`).
        force_backend: Backend name (``"cpu"``, ``"cuda12x"``, ...).
    """

    def __init__(self, window_length: float, f_min: float, num_freqs: int, window_alpha: float = 0.0,
                 *, linear_envelope: bool = False, force_backend: Optional[str] = None):
        self.backend = get_backend(force_backend or "cpu")
        self.xp = self.backend.xp
        self.window_length = float(window_length)
        self.df = 1.0 / self.window_length
        self.f_min = float(f_min)
        self.num_freqs = int(num_freqs)
        self.window_alpha = float(window_alpha)
        self.linear_envelope = bool(linear_envelope)
        f_max = self.f_min + (self.num_freqs - 1) * self.df
        # num_times, num_channels and t0 of the settings are not read by the track kernel
        self._fresnel = self.backend.STFTFresnelWrap(
            1, self.num_freqs, 3, 0.0, self.f_min, f_max, self.window_length, self.df,
            self.window_alpha, True, self.linear_envelope)

    def __call__(self, amp, phase, f, fdot, *, component_row=None, num_rows: Optional[int] = None,
                 half_width=None, n_side: int = 30, slope=None, out=None, scale: float = 0.5):
        """Add the tracks to ``out`` and return it.

        Args:
            amp, phase, f, fdot: ``(C, nch, NT)`` at the segment midpoints.
            component_row: ``(C,)`` row of each component (default: all in row 0).
            num_rows: Number of rows (default: ``max(component_row) + 1``).
            half_width: ``(C, NT)`` bins each side of the mid-segment frequency; negative skips
                the (component, segment). Default: :func:`track_half_width` with ``n_side``.
            slope: ``(C, nch, NT)`` fractional amplitude slope ``amp' / amp`` [1/s]; needs
                ``linear_envelope``.
            out: ``(num_rows, nch, NT, num_freqs)`` complex128, accumulated; allocated (zero)
                when omitted.
            scale: Multiplies every value; 0.5 is the transform of ``amp * cos(phase)``.
        """
        xp = self.xp
        amp = xp.ascontiguousarray(xp.asarray(amp, dtype=xp.float64))
        if amp.ndim != 3:
            raise ValueError("amp, phase, f, fdot must have shape (components, channels, segments).")
        num_comp, nch, num_seg = amp.shape

        def track(a, name):
            a = xp.ascontiguousarray(xp.asarray(a, dtype=xp.float64))
            if a.shape != amp.shape:
                raise ValueError(f"{name} has shape {a.shape}, expected {amp.shape}.")
            return a

        phase, f, fdot = track(phase, "phase"), track(f, "f"), track(fdot, "fdot")
        if slope is None:
            slope = xp.zeros(0, dtype=xp.float64)
        else:
            if not self.linear_envelope:
                raise ValueError("slope needs an accumulator built with linear_envelope=True.")
            slope = track(slope, "slope")
        if component_row is None:
            component_row = xp.zeros(num_comp, dtype=xp.int32)
        component_row = xp.ascontiguousarray(xp.asarray(component_row, dtype=xp.int32))
        if component_row.shape != (num_comp,):
            raise ValueError(f"component_row has shape {component_row.shape}, expected ({num_comp},).")
        if num_rows is None:
            num_rows = int(component_row.max()) + 1 if num_comp else 1
        if num_comp and (int(component_row.min()) < 0 or int(component_row.max()) >= num_rows):
            raise ValueError("component_row outside [0, num_rows).")
        if half_width is None:
            half_width = track_half_width(fdot, self.window_length, n_side, xp=xp)
        half_width = xp.ascontiguousarray(xp.asarray(half_width, dtype=xp.int32))
        if half_width.shape != (num_comp, num_seg):
            raise ValueError(f"half_width has shape {half_width.shape}, expected {(num_comp, num_seg)}.")
        shape = (num_rows, nch, num_seg, self.num_freqs)
        if out is None:
            out = xp.zeros(shape, dtype=xp.complex128)
        elif out.shape != shape or out.dtype != xp.complex128 or not out.flags.c_contiguous:
            raise ValueError(f"out must be a C-contiguous complex128 array of shape {shape}.")
        if num_comp == 0 or num_seg == 0:
            return out
        self._fresnel.sum_tracks(out.reshape(-1), num_rows, nch, num_seg, self.num_freqs, self.f_min,
                                 float(scale), num_comp, component_row, half_width.reshape(-1),
                                 amp.reshape(-1), phase.reshape(-1), f.reshape(-1), fdot.reshape(-1),
                                 slope.reshape(-1))
        return out


def stft_templates_from_tracks(source: SparseTrackSource, accumulator: STFTTrackAccumulator,
                               segment_start, data_dt: float, *, n_side: int = 30,
                               cubic_tol: float = 0.03, max_half_width: Optional[int] = None,
                               window=None, fallback_block: int = 20_000_000,
                               return_info: bool = False):
    """STFT templates of a :class:`SparseTrackSource`, any kind of source.

    Every (row, segment) is filled in one of three ways:

    * *tracks*: each component's linear chirp about the segment midpoint, through
      ``accumulator``;
    * *dense*: the window and FFT of the row's exact time series
      (:meth:`SparseTrackSource.td_channels`), where the linear chirp is not good enough for
      some component of the row: the phase it leaves out at the segment edge
      (:func:`cubic_phase_error`) exceeds ``cubic_tol``, the band needed is wider than
      ``max_half_width``, or the component's track does not cover the whole segment. A row
      without any component is dense wherever its signal is;
    * *zero*: the segment lies outside the row's signal span.

    Args:
        source: The source.
        accumulator: Grid, window and backend of the templates.
        segment_start: ``(NT,)`` start times of the segments [s], in the clock of the source.
            Any hop; each segment is ``accumulator.window_length`` long.
        data_dt: Sampling interval of the data [s]; the window length is a whole number of
            samples.
        n_side: Bins kept beyond half the sweep on each side (track segments).
        cubic_tol: Largest edge phase error of the linear chirp [rad].
        max_half_width: Largest band half-width of a track segment [bins]; ``None``: no cap.
        window: ``(samples per segment,)`` window of the dense segments; default the periodic
            Tukey window of the accumulator.
        fallback_block: Largest number of (channel, sample) values of one dense evaluation.
        return_info: Also return a dict with ``mode`` ``(num_rows, NT)`` (0 zero, 1 tracks,
            2 dense), ``cubic`` and ``half_width`` ``(C, NT)``.

    Returns:
        ``(num_rows, nch, NT, accumulator.num_freqs)`` complex128 on the accumulator's backend
        (and the info dict with ``return_info``).
    """
    xp = accumulator.xp
    T = accumulator.window_length
    nper = int(round(T / data_dt))
    if nper < 2 or abs(nper * data_dt - T) > 1e-9 * T:
        raise ValueError("the window length must be a whole number of samples of data_dt.")
    bin_lo = int(round(accumulator.f_min / accumulator.df))
    if abs(bin_lo * accumulator.df - accumulator.f_min) > 1e-6 * accumulator.df:
        raise ValueError("f_min of the accumulator must be a multiple of 1 / window_length.")
    if bin_lo + accumulator.num_freqs > nper // 2 + 1:
        raise ValueError("the accumulator's band exceeds the Nyquist frequency of data_dt.")

    def host(a):
        return a.get() if hasattr(a, "get") else np.asarray(a)

    seg = np.asarray(host(segment_start), dtype=float).reshape(-1)
    num_seg = seg.size
    rows = np.asarray(host(source.component_row), dtype=np.int64).reshape(-1)
    num_rows = int(source.num_rows)
    t_mid = seg + 0.5 * T
    amp, phase, f, fdot = source.eval_tracks(xp.asarray(t_mid))

    # --- classification, on the host: (C, NT) and (num_rows, NT) are small
    t_lo, t_hi = (np.asarray(host(v), dtype=float).reshape(-1, 1) for v in source.track_span())
    s_lo, s_hi = (np.asarray(host(v), dtype=float).reshape(-1, 1) for v in source.signal_span())
    in_track = (seg[None, :] >= t_lo) & (seg[None, :] + T <= t_hi)                # (C, NT)
    in_signal = (seg[None, :] < s_hi) & (seg[None, :] + T > s_lo)                 # (R, NT)
    half_width = host(track_half_width(fdot, T, n_side, xp=xp)).astype(np.int64)
    cubic = host(cubic_phase_error(fdot, t_mid, T, valid=xp.asarray(in_track), xp=xp))
    finite = host(xp.isfinite(amp).all(axis=1) & xp.isfinite(phase).all(axis=1)
                  & xp.isfinite(f).all(axis=1) & xp.isfinite(fdot).all(axis=1))
    bad = ~in_track | ~finite | ~(cubic <= cubic_tol)
    if max_half_width is not None:
        bad |= half_width > int(max_half_width)
    row_bad = np.zeros((num_rows, num_seg), dtype=bool)
    np.logical_or.at(row_bad, rows, bad)
    row_bad[np.bincount(rows, minlength=num_rows) == 0] = True   # a row without tracks is dense
    dense = in_signal & row_bad
    use_tracks = in_signal & ~row_bad
    half_width = np.where(use_tracks[rows], half_width, -1).astype(np.int32)

    out = accumulator(amp, phase, f, fdot, component_row=rows, num_rows=num_rows,
                      half_width=half_width)

    # --- dense segments: window + FFT of the exact time series, per row
    if dense.any():
        w = periodic_tukey(nper, accumulator.window_alpha, xp=xp) if window is None else xp.asarray(window)
        if w.shape != (nper,):
            raise ValueError(f"window has shape {w.shape}, expected ({nper},).")
        offsets = xp.arange(nper) * data_dt
        nch = out.shape[1]
        per_call = max(1, int(fallback_block) // (nch * nper))
        for r in np.flatnonzero(dense.any(axis=1)):
            js = np.flatnonzero(dense[r])
            for i in range(0, js.size, per_call):
                jb = js[i:i + per_call]
                t = (xp.asarray(seg[jb])[:, None] + offsets[None, :]).reshape(-1)
                x = xp.asarray(source.td_channels(int(r), t)).reshape(nch, jb.size, nper)
                spec = data_dt * xp.fft.rfft(x * w, axis=-1)
                out[int(r)][:, xp.asarray(jb)] = spec[..., bin_lo:bin_lo + accumulator.num_freqs]
    if return_info:
        mode = np.where(dense, 2, np.where(use_tracks, 1, 0)).astype(np.int8)
        return out, dict(mode=mode, cubic=cubic, half_width=half_width)
    return out
