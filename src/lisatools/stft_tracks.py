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
    "track_half_width",
    "cubic_phase_error",
]


@runtime_checkable
class SparseTrackSource(Protocol):
    """What a source provides to be turned into STFT templates by track accumulation.

    Shapes: ``C`` components in total over all template rows, ``nch`` TDI channels.
    """

    #: (C,) template row of each component.
    component_row: np.ndarray
    #: Number of template rows.
    num_rows: int

    def eval_tracks(self, t_anchor) -> Tuple:
        """``(amp, phase, f, fdot)`` at the times ``t_anchor`` (NT,), each ``(C, nch, NT)``.

        ``amp * cos(phase)`` is the channel's time series, ``f = phase' / 2 pi`` and
        ``fdot = phase'' / 2 pi``. The amplitude may be signed.
        """
        ...

    def track_span(self) -> Tuple:
        """``(t_lo, t_hi)``, each ``(C,)``: the times between which ``eval_tracks`` is defined."""
        ...

    def td_channels(self, t, rows=None):
        """Exact time series ``(num_rows, nch, len(t))``, summed over the components of each row.

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


def cubic_phase_error(fdot, t_anchor, window_length: float, xp=np):
    """Phase the linear chirp leaves out at a segment edge, ``(C, NT)`` [rad].

    ``(pi / 3) * |fddot| * (window_length / 2)**3`` with ``fddot`` from differencing ``fdot``
    ``(C, nch, NT)`` across the anchors ``t_anchor`` (NT,); the largest channel is returned.
    With fewer than two anchors the estimate is zero.
    """
    if fdot.shape[-1] < 2:
        return xp.zeros((fdot.shape[0], fdot.shape[-1]))
    fddot = xp.gradient(fdot, xp.asarray(t_anchor), axis=-1)
    return (np.pi / 3.0) * xp.abs(fddot).max(axis=1) * (0.5 * window_length) ** 3


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
