"""The production ``pyResponseTDI`` response of a time-domain waveform on a short window of data.

Source-agnostic: works with any :class:`~lisatools.sources.waveformbase.TDPyResponseWaveformBase`
through its own ``_apply_response``. The caller says where production's input series lives
(``span = (t_w0, n_prod)``: production feeds the polarizations at ``t_w0 + k dt`` for
``0 <= k < n_prod`` and zeros elsewhere), the only source-specific piece. Fed production's own
``h_plus``, ``h_cross`` on :func:`window_input_range`, :func:`response_on_window` returns the
channels of ``compute_tdi_channels`` at the requested data samples, to the float floor of
production itself.

Where the margins come from (``pyResponseTDI`` and ``LISAResponse.cu``):

* Lagrange interpolation at a fractional index ``x`` reads ``input[d - (H - 1) .. d + H]``,
  ``d = ceil(x) - 1``, ``H = (order + 1) // 2``.
* The projection ``y_gw[j]`` interpolates h at ``t_j + s - k.x_rec / c`` and
  ``t_j + s - L - k.x_em / c``: h within ``R / c + L`` of ``t_j`` (``R`` the largest spacecraft
  distance from the origin, ``R / c`` about 501 s), plus or minus ``H`` samples. The sign of
  ``k.x`` depends on the sky and the epoch, so the reach is taken on both sides.
* The TDI value ``X[i]`` interpolates ``y_gw`` at ``t_i`` minus up to ``max(unit_lengths)``
  light travel times, plus or minus ``H`` samples; it is computed only from input sample
  ``int(buffer_time / dt)`` on (``tdi_start_ind``).
* ``_apply_response`` pads ``buffer_time`` of zeros on the right and zeroes the first
  ``int(buffer_time / dt)`` output samples.

Hence an output sample needs ``int(buffer_time / dt)`` input samples before it (structural; of
those only the last :func:`response_margins` ``reach_back`` are read, the earlier ones may be
zeros) and reads ``reach_fwd`` samples after it. Lagrange projection only: the spline projection
(``use_spline``) fits one spline over the whole input and is refused.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

C_SI = 299792458.0  # m/s, C_inv of LISAResponse
#: slack on each reach [samples]: the ceil(x) - 1 stencil index, the sub-sample shift
#: ``t0_shift_to_data`` (``|s| <= dt / 2``) and the rint tie at ``|s| = dt / 2``
ROUNDING_SAMPLES = 2


def _host(v):
    return v.get() if hasattr(v, "get") else np.asarray(v)


def response_margins(wf) -> Tuple[int, int, int]:
    """Margins of the production response of ``wf``, in samples.

    Returns:
        ``(left, reach_back, reach_fwd)``: input samples needed before the first output sample
        (structural, ``int(buffer_time / dt)``), the last ``reach_back`` of which are read; input
        samples read after the last output sample. Worst case over sky and epoch.

    Raises:
        ValueError: The response uses the spline projection (``use_spline``): its quintic
            spline is global over the input, so a window's edges reach further than these
            stencil margins.
    """
    resp = wf.response
    if getattr(resp, "use_spline", False):
        raise ValueError("the windowed response needs the Lagrange projection (use_spline=False): "
                         "the spline projection is global over the input series.")
    cache = wf.__dict__.setdefault("_response_margins_cache", {})
    key = (resp.order, wf.buffer_time, float(wf.dt), id(resp.response_orbits))
    if key not in cache:
        orbits = resp.response_orbits
        dt = float(wf.dt)
        half = (resp.order + 1) // 2                                     # Lagrange half stencil
        x = _host(orbits.x_base)
        r_c = float(np.sqrt(np.sum(x * x, axis=-1)).max()) / C_SI        # R / c
        l_max = float(_host(orbits.ltt).max())                           # longest arm [s]
        n_delay = int(np.max(_host(wf.tdi_config.unit_lengths)))         # delays per TDI term
        proj = math.ceil((r_c + l_max) / dt)     # projection: h at t - k.x / c [- L]
        tdi = math.ceil(n_delay * l_max / dt)    # TDI: y_gw at t - sum(L)
        reach_back = tdi + proj + 2 * half + ROUNDING_SAMPLES
        reach_fwd = math.ceil(r_c / dt) + 2 * half + ROUNDING_SAMPLES
        cache[key] = (int(wf.buffer_time / dt), reach_back, reach_fwd)
    return cache[key]


def _grid_shift(wf, t_first: float) -> float:
    """``t0_shift_to_data`` of an input series starting at ``t_first``, as ``_apply_response``."""
    delta = float(wf.data_t0) - float(t_first)
    return delta - float(np.rint(delta / wf.dt)) * float(wf.dt)


def window_input_range(wf, span, t_out_first: float, t_out_last: float) -> Tuple[int, int, int]:
    """Production input samples needed for the output samples ``[t_out_first, t_out_last]``.

    Args:
        wf: The waveform.
        span: ``(t_w0, n_prod)``, production's input series (see the module docstring).
        t_out_first, t_out_last: First and last data sample of the window [s].

    Returns:
        ``(k0, n_in, n_zero)``: provide the polarizations at ``t_w0 + k dt`` for
        ``k0 <= k < k0 + n_in``; the first ``n_zero`` of them are never read into the window and
        may be zeros. ``n_in == 0`` when the window sees none of production's input.
    """
    dt = float(wf.dt)
    t_w0, n_prod = float(span[0]), int(span[1])
    left, back, fwd = response_margins(wf)
    s = _grid_shift(wf, t_w0)
    # production input index of the output samples (output time = input time + s); +-1 for ties
    k_a = int(np.rint((t_out_first - s - t_w0) / dt))
    k_b = int(np.rint((t_out_last - s - t_w0) / dt))
    k0 = max(0, k_a - left - 1)
    k1 = min(n_prod - 1, k_b + fwd + 1)
    n_in = max(0, k1 - k0 + 1)
    n_zero = min(n_in, max(0, (k_a - back - 1) - k0))
    return k0, n_in, n_zero


def response_on_window(wf, span, k0: int, h_plus, h_cross, ra: float, dec: float, t_out):
    """Production TDI channels ``(nchan, len(t_out))`` at the data samples ``t_out``.

    Args:
        wf: The waveform; its ``_apply_response`` is used as in production.
        span: ``(t_w0, n_prod)``, production's input series.
        k0 (int): Production input index of ``h_plus[0]`` (time ``t_w0 + k0 dt``), from
            :func:`window_input_range`.
        h_plus, h_cross: Polarizations at ``t_w0 + (k0 + k) dt``. Samples outside
            ``[0, n_prod)`` are dropped (production has zeros there).
        ra, dec: Sky position, as for ``compute_tdi_channels``.
        t_out: Contiguous data samples ``data_t0 + j dt`` (increasing, step ``dt``).

    Returns:
        The channels on ``t_out`` (zero where production has no output), on ``wf.xp``.

    Raises:
        ValueError: A margin shorter than :func:`response_margins` asks for.
    """
    xp = wf.xp
    dt = float(wf.dt)
    t_out = np.asarray(_host(t_out), dtype=float).reshape(-1)
    t_w0, n_prod = float(span[0]), int(span[1])
    k0 = int(k0)
    lo, hi = max(0, -k0), min(int(h_plus.shape[-1]), n_prod - k0)
    out = xp.zeros((len(wf.tdi_channels), t_out.size), dtype=xp.float64)
    if lo >= hi or t_out.size == 0:
        return out
    h_plus, h_cross = h_plus[lo:hi], h_cross[lo:hi]
    k0, n_in = k0 + lo, hi - lo

    left, _back, fwd = response_margins(wf)
    s = _grid_shift(wf, t_w0 + k0 * dt)
    i_a = int(np.rint((t_out[0] - s - t_w0) / dt)) - k0      # window index of t_out[0]
    i_b = int(np.rint((t_out[-1] - s - t_w0) / dt)) - k0
    # outputs before int(buffer_time / dt) are not computed; production zeroes the same span
    # after its first input sample, so a window starting at production's first sample is exact
    if i_a < left and k0 > 0:
        raise ValueError(f"left margin {i_a} < {left} samples (int(buffer_time / dt)).")
    if i_b + fwd > n_in - 1 and k0 + n_in < n_prod:
        raise ValueError(f"right margin {n_in - 1 - i_b} < {fwd} samples before production's end.")

    # production's time labels: input sample k at t_w0 + k dt (merger_time 0 adds nothing)
    t_arr = (t_w0 - float(wf.waveform_t0)) + xp.arange(k0, k0 + n_in, dtype=xp.float64) * dt
    times, chans = wf._apply_response(t_arr, xp.asarray(h_plus), xp.asarray(h_cross),
                                      float(ra), float(dec), 0.0)
    j0 = int(np.rint((t_out[0] - float(times[0])) / dt))
    a, b = max(0, -j0), min(t_out.size, int(chans.shape[-1]) - j0)
    if a < b:
        out[:, a:b] = chans[:, j0 + a:j0 + b]
    return out
