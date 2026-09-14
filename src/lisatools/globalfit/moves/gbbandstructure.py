"""Frequency-band layouts for the GB special moves.

A band structure says which interval of source ``f0`` each band owns. That is
all it is: it is NOT the residual pixel window a band cell reserves (see
:func:`stft_store_windows`) and NOT the per-source likelihood stencil. Keeping
the three apart is what lets the STFT layout carry bands narrower than one
STFT pixel.

:class:`GBBandStructure` is the shape every recipe returns, so they are
interchangeable at the :class:`~lisatools.globalfit.moves.gbbands.BandSorter`
boundary. Only STFT is implemented; :func:`gb_band_structure` names the FD and
WDM recipes at the point where they would land.

Device policy: the edge walk is a sequential scalar recurrence (edge ``b + 1``
depends on edge ``b``), so it always runs on the host. The builder takes an
``xp`` argument and moves the finished edge array once, so a GPU run holds its
edges on the device and :meth:`GBBandStructure.assign` never transfers.
"""

from __future__ import annotations

import dataclasses
import logging
from functools import lru_cache
from typing import Optional

import numpy as np

from ...domains import DomainSettingsBase, FDSettings, STFTSettings, WDMSettings
from ...utils.constants import AU_SI, C_SI, MTSUN_SI, YRSID_SI
from ...utils.utility import asnumpy, get_array_module

__all__ = [
    "GBBandStructure",
    "STFTStoreWindows",
    "STFT_F0_LIMIT_FRACTION",
    "STFT_TIER_SPLIT_HZ",
    "gb_band_structure",
    "stft_band_structure",
    "stft_band_width_hz",
    "stft_reach_hz",
    "stft_store_window_layout",
    "stft_store_windows",
    "tukey_floor_bins",
    "worst_case_fdot_hz_per_s",
]

logger = logging.getLogger(__name__)

# Fitted mass-transfer constants
_MT_AMPLITUDE_HZ_PER_S = 1.0e-20
_MT_PIVOT_HZ = 4.0e-4
_MT_SLOPE = 16.0 / 3.0
#: Above this no stably mass-transferring binary exists (weak tidal coupling)
_MT_CUTOFF_HZ = 0.08283142425665438

# Fitted correction on the combined sideband-plus-window floor at 99% containment over (alpha, Tobs).
_FC_ALPHA = np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0])
_FC_TOBS_YR = np.array(
    [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0]
)
_FC_Q99 = np.array([
    [7.284838, 6.593734, 3.972973, 3.908216, 3.769391, 3.695825, 3.343005,
     3.349803, 3.204785, 3.230142, 3.014878, 3.001266, 2.989124],
    [4.608019, 4.356643, 3.090078, 3.034302, 2.900433, 2.801593, 2.507389,
     2.505119, 2.415561, 2.310141, 2.328233, 2.308801, 2.336183],
    [3.741777, 3.594743, 2.705787, 2.700217, 2.511206, 2.509043, 2.337590,
     2.253953, 2.073735, 2.083660, 2.062340, 2.099543, 2.144673],
    [2.716983, 2.679213, 2.157377, 2.157459, 1.935877, 2.046584, 1.888874,
     1.902526, 1.807681, 1.855711, 1.914731, 1.995303, 2.038176],
    [2.300695, 2.277518, 1.880309, 1.874425, 1.686871, 1.818883, 1.774486,
     1.848716, 1.920765, 1.995188, 2.042814, 2.096859, 2.124055],
    [1.952049, 1.898708, 1.611780, 1.748987, 1.911974, 1.979149, 2.024255,
     2.049704, 2.062356, 2.098355, 2.116322, 2.133612, 2.140402],
    [1.757750, 1.789756, 1.773039, 1.941525, 2.072444, 2.122105, 2.122733,
     2.126567, 2.121056, 2.131689, 2.134696, 2.135040, 2.133453],
    [1.966425, 1.955195, 2.133726, 2.331018, 2.163059, 2.156750, 2.142419,
     2.127527, 2.112488, 2.103711, 2.096448, 2.085756, 2.078485],
])


def _floor_correction_q99(alpha: float, tobs_yr: float) -> float:
    """Bilinear lookup of :data:`_FC_Q99`, interpolated on ``log C``.

    Interpolating the log matches the study's own prescription, so the two
    agree to round-off. Off the grid the edge value is held, which is the
    conservative direction for a width.
    """
    per_alpha = np.array([np.interp(tobs_yr, _FC_TOBS_YR, np.log(row))
                          for row in _FC_Q99])
    return float(np.exp(np.interp(alpha, _FC_ALPHA, per_alpha)))


@lru_cache(maxsize=32)
def tukey_floor_bins(q: float = 0.99, alpha: float = 0.11574,
                     n_window: int = 4096, oversample: int = 64) -> float:
    """Containment width of a Tukey window's own spectrum, in bins of ``1/T``.

    The width a strictly monochromatic source gets from the window alone:
    anything narrower than this is unmeasurable. Computed rather than
    tabulated, so any ``q`` and ``alpha`` work; it reproduces the study's
    measured table to 2% at every alpha below 0.35.

    ``alpha`` is the Tukey taper fraction counting BOTH tapers. Which window
    it belongs to is the caller's choice and the two readings differ by orders
    of magnitude: the STFT segment taper gives ``1/(f_start * stft_DT)``
    (0.116 for a run starting at 0.1 mHz), the full-record taper gives
    ``1/(f_start * Tobs)`` (about 6e-4).
    """
    sample = np.arange(n_window)
    taper_samples = 0.5 * float(alpha) * n_window
    window = np.ones(n_window)
    if taper_samples >= 1.0:
        rising = sample < taper_samples
        window[rising] = 0.5 * (
            1.0 - np.cos(np.pi * sample[rising] / taper_samples))
        falling = sample >= n_window - taper_samples
        window[falling] = 0.5 * (
            1.0 - np.cos(np.pi * (n_window - sample[falling]) / taper_samples))
    padded = np.zeros(n_window * oversample)
    padded[:n_window] = window
    power = np.abs(np.fft.rfft(padded)) ** 2

    weight = np.full(power.size, 2.0)
    weight[0] = 1.0
    cumulative = np.cumsum(power * weight)
    index = int(np.searchsorted(cumulative / cumulative[-1], q))
    return 2.0 * min(index, power.size - 1) / oversample


def worst_case_fdot_hz_per_s(
    f0_hz,
    MT_fmax: Optional[float] = _MT_CUTOFF_HZ,
    Mc_max_msun: float = 1.4
) -> np.ndarray:
    """Largest ``|fdot|`` either population branch allows, in Hz/s."""
    f0_hz = np.asarray(f0_hz, dtype=float)
    
    if MT_fmax is not None:
        mass_transfer = np.where(
            f0_hz <= MT_fmax,
            _MT_AMPLITUDE_HZ_PER_S * (f0_hz / _MT_PIVOT_HZ) ** _MT_SLOPE, 0.0)
    else:
        mass_transfer = _MT_AMPLITUDE_HZ_PER_S * (f0_hz / _MT_PIVOT_HZ) ** _MT_SLOPE

    radiation = ((96.0 / 5.0) * np.pi ** (8.0 / 3.0)
                 * (Mc_max_msun * MTSUN_SI) ** (5.0 / 3.0)
                 * f0_hz ** (11.0 / 3.0))
    
    return np.maximum(mass_transfer, radiation)


def stft_band_width_hz(f0_hz, tobs_s: float, alpha: float, *,
                       k_safety_factor: float = 4.0, q: float = 0.99,
                       floor_correction: Optional[float] = None,
                       sin_ecliptic_colatitude: float = 1.0):
    r"""``W_band(f0) = k * B_q(f0)``, in Hz. q is the containment fraction.
    k_safety_factor is the recipe's safety factor.

    The band width ``B_q`` is the plain SUM (not the quadrature) of three terms:

    * ``4 pi f0 (R/c) f_mod sin(theta) sin(q Theta)`` the orbital-Doppler swing, with
      ``Theta`` saturating at ``pi/2`` once the record covers half a year;
    * ``|fdot|_max Tobs``  the chirp drift at the worst-case rate;
    * ``hypot(2 n f_mod, W(alpha, q)/Tobs) * C`` the sideband-plus-window floor, 
      with the Tukey window's own spectrum width ``W(alpha, q)`` and a fitted 
      correction ``C`` for the sideband floor and the window's own leakage.

    Args:
        alpha: The tukey alpha parameter, counting both tapers. 
        The STFT segment taper gives ``1/(f_start * stft_DT)``.
        floor_correction: ``None`` looks up the embedded q = 0.99 grid; a float
            overrides it. Required for any ``q`` other than 0.99.
    """
    f0_hz = np.atleast_1d(np.asarray(f0_hz, dtype=float))
    tobs_s, alpha, q = float(tobs_s), float(alpha), float(q)
    if floor_correction is None:
        if not np.isclose(q, 0.99):
            raise ValueError(
                f"the embedded floor-correction grid covers q = 0.99 only; "
                f"pass floor_correction explicitly for q = {q}."
            )
        floor_correction = _floor_correction_q99(alpha, tobs_s / YRSID_SI)

    f_mod_hz = 1.0 / YRSID_SI
    doppler_amplitude_hz = (2.0 * np.pi * f0_hz * (AU_SI / C_SI) / YRSID_SI
                            * float(sin_ecliptic_colatitude))
    swing_phase = min(np.pi * tobs_s / YRSID_SI, 0.5 * np.pi)
    doppler_hz = 2.0 * doppler_amplitude_hz * np.sin(q * swing_phase)
    chirp_hz = worst_case_fdot_hz_per_s(f0_hz) * tobs_s

    floor_hz = np.hypot(4.0 / YRSID_SI,
                        tukey_floor_bins(q, alpha) / tobs_s) * floor_correction
    return float(k_safety_factor) * (doppler_hz + chirp_hz + floor_hz)


# ======================================================================
# The layout
# ======================================================================
@dataclasses.dataclass(frozen=True)
class GBBandStructure:
    """Band ``b`` owns ``[edges_hz[b], edges_hz[b + 1])`` in source ``f0``.

    ``edges_hz`` lives in whichever array module its builder was given, so a
    GPU run holds it on the device and no call here transfers. Everything else
    is a scalar. The recipe fields are ``None`` for domains that do not use
    them.
    """

    domain: str
    edges_hz: np.ndarray
    tobs_s: float
    grid_df_hz: float
    k_safety_factor: Optional[float] = None
    q_containment: Optional[float] = None
    alpha_taper: Optional[float] = None
    floor_correction: Optional[float] = None

    @property
    def xp(self):
        """Array module the edges live in, taken from the edges themselves."""
        return get_array_module(self.edges_hz)

    @property
    def n_bands(self) -> int:
        return int(self.edges_hz.size - 1)

    @property
    def band_lo_hz(self):
        return self.edges_hz[:-1]

    @property
    def band_hi_hz(self):
        return self.edges_hz[1:]

    @property
    def band_centre_hz(self):
        return 0.5 * (self.band_lo_hz + self.band_hi_hz)

    @property
    def width_hz(self):
        return self.xp.diff(self.edges_hz)

    @property
    def width_coherent_elements(self):
        """Width in coherent resolution elements, each ``1/Tobs`` wide."""
        return self.width_hz * self.tobs_s

    @property
    def width_grid_pixels(self):
        """Width in pixels of the domain's own frequency grid."""
        return self.width_hz / self.grid_df_hz

    def assign(self, freqs_hz):
        """Band index of each source frequency.

        Same convention as ``BandSorter``: a frequency below the first edge
        gives ``-1`` and one at or above the last gives ``n_bands``.

        """
        return self.xp.searchsorted(self.edges_hz, freqs_hz, side="right") - 1

    def band_N_vals(self, oversample: int = 4) -> np.ndarray | None:
        """Per-band FastGB sample count, which ``BandSorter`` requires.

        The STFT and WDM engines size their buffers from the domain settings
        and never read it, so there it exists for shape parity, exactly as the
        stock WDM recipe builds it.
        """
        from gbgpu.utils.utility import get_N

        if self.domain in ("stft",):
            return None
        else:
            return np.asarray([get_N(1e-30, edge, self.tobs_s,
                                    oversample=oversample).item()
                            for edge in asnumpy(self.band_lo_hz)])

    def band_sorter_kwargs(self, oversample: int = 4) -> dict:
        """``band_edges`` / ``band_N_vals`` ready to hand to ``BandSorter``."""
        return {"band_edges": self.edges_hz,
                "band_N_vals": self.band_N_vals(oversample=oversample)}

    def __repr__(self) -> str:
        return (f"GBBandStructure({self.domain}, {self.n_bands} bands, "
                f"{float(self.edges_hz[0]) * 1e3:.4f}-"
                f"{float(self.edges_hz[-1]) * 1e3:.4f} mHz, "
                f"Tobs={self.tobs_s / 86400.0:.1f} d)")


def stft_band_structure(domain_settings: STFTSettings, alpha: float, *,
                        k_safety_factor: float = 4.0, q: float = 0.99,
                        floor_correction: Optional[float] = None,
                        convention: str = "midpoint", xp=np,
                        max_bands: int = 200000) -> GBBandStructure:
    """Walk the STFT layout upward from ``min_freq``, each band its own width.

    Every grid quantity comes from ``domain_settings``: ``min_freq`` and
    ``max_freq`` (already snapped to the grid by their setters), ``df`` as the
    pixel spacing, and ``NT * dt`` as the span.

    ``STFTSettings.dt`` IS the segment length DT_STFT, not the data
    cadence, and ``df`` IS DF_STFT = 1/dt. There is no ``Tobs`` attribute,
    and ``get_stft_settings`` FLOORS ``NT``, so ``NT * dt`` can be shorter
    than the record the settings were built from. That shorter span is the
    right one: it is what the layout actually sees.

    ``k = 4``, ``q = 0.99`` and the midpoint convention are the decided
    recipe. ``max_over_band`` (the maximum over the trial span, iterated to a
    fixed point) moves the 365 d count by 4 bands in 1555, so the choice
    carries no conclusion.

    The walk takes a full width at its last step with no clipping, so the top
    band overshoots ``max_freq``.

    Sequential scalar recurrence, so it runs on the host whatever ``xp``
    is; only the finished edge array is moved, once.
    """
    if convention not in ("midpoint", "max_over_band"):
        raise ValueError(f"convention must be 'midpoint' or 'max_over_band'; "
                         f"got {convention!r}")
    min_freq = float(domain_settings.min_freq)
    max_freq = float(domain_settings.max_freq)
    tobs_s = float(domain_settings.NT) * float(domain_settings.dt)
    if not max_freq > min_freq > 0.0:
        raise ValueError(f"need 0 < min_freq < max_freq; got {min_freq}, {max_freq}")

    def width_at(f_hz):
        return stft_band_width_hz(f_hz, tobs_s, alpha,
                                  k_safety_factor=k_safety_factor, q=q,
                                  floor_correction=floor_correction)

    edges_hz = [min_freq]
    while edges_hz[-1] < max_freq:
        edge_hz = edges_hz[-1]
        if convention == "midpoint":
            half_width_hz = 0.5 * float(width_at(edge_hz)[0])
            width_hz = float(width_at(edge_hz + half_width_hz)[0])
        else:
            width_hz = float(width_at(edge_hz)[0])
            for _iteration in range(20):
                span_hi_hz = min(edge_hz + width_hz, max_freq)
                samples_hz = np.linspace(edge_hz, span_hi_hz, 16)
                new_width_hz = float(np.max(width_at(samples_hz)))
                converged = abs(new_width_hz - width_hz) <= 1e-6 * new_width_hz
                width_hz = new_width_hz
                if converged:
                    break
        if not np.isfinite(width_hz) or width_hz <= 0.0:
            raise RuntimeError(f"band width {width_hz:g} Hz is not finite and "
                               f"positive at {edge_hz * 1e3:.6g} mHz")
        edges_hz.append(edge_hz + width_hz)
        if len(edges_hz) > max_bands:
            raise RuntimeError(f"band walk exceeded max_bands={max_bands}; the "
                               f"width recipe is returning widths far too small")

    return GBBandStructure(
        domain="stft", edges_hz=xp.asarray(np.asarray(edges_hz, dtype=float)),
        tobs_s=tobs_s, grid_df_hz=float(domain_settings.df),
        k_safety_factor=float(k_safety_factor), q_containment=float(q),
        alpha_taper=float(alpha), floor_correction=floor_correction)


# ======================================================================
# Store windows: the pixel window a band cell reserves
# ======================================================================
STFT_F0_LIMIT_FRACTION = 1/4
#: Bands whose lower edge is below this frequency are Tier 1, the others Tier 2.
STFT_TIER_SPLIT_HZ = 1.0e-2


def stft_reach_hz(f0_hz, tobs_s: float, epoch_offset_s: float = 0.0):
    """Worst-case per-side excursion of a carrier track, in Hz.

    The ridge itself rather than its containment, so no ``q`` enters here,
    unlike :func:`stft_band_width_hz`. Symmetric: the two chirp branches are
    collapsed to the larger magnitude, which is the conservative choice and
    means one array serves both sides.

    ``epoch_offset_s`` is ``t_ref - data_t0``. The sampled f0 belongs to ``t_ref``, so the chirp
    runs over the offset as well as over the record.
    """
    f0_hz = np.atleast_1d(np.asarray(f0_hz, dtype=float))
    doppler_hz = (2.0 * np.pi * f0_hz * (AU_SI / C_SI) / YRSID_SI
                  * np.sin(min(np.pi * float(tobs_s / YRSID_SI), 0.5 * np.pi)))
    chirp_span_s = float(tobs_s) + abs(float(epoch_offset_s))
    return doppler_hz + worst_case_fdot_hz_per_s(f0_hz) * chirp_span_s


def stft_store_windows(structure: GBBandStructure,
                       domain_settings: STFTSettings, *,
                       n_side_bins: int, epoch_offset_s: float,
                       guard_low: int = 5, guard_high: int = 3,
                       f0_limit_fraction: float = STFT_F0_LIMIT_FRACTION) -> tuple:
    """Inclusive ``(j_lo, j_hi)`` grid bins each band cell must hold.

    Indices are relative to active-grid bin 0, whose frequency is the
    settings' snapped ``min_freq``; they are clipped to ``NF_active``, the
    bins the kernel actually carries.

    Rounded strictly OUTWARD, so a source's worst-case reach can never fall
    outside the pixels its own band cell owns.

    The guard absorbs error in the reach model, not window leakage. It is
    larger below 1 mHz because the carrier track spikes near antenna-pattern
    nulls there.

    ``n_side_bins`` and ``epoch_offset_s`` have no default, so a window cannot drift from the
    comp's stencil or its ``t_ref``. The f0 range is the band widened by ``f0_limit_fraction``
    of its width on each side, the range an in-model move may visit.

    Returned on the host: these index buffer allocations, which is host-side
    bookkeeping, and there is one entry per band rather than per source.
    """
    f_grid_min_hz = float(domain_settings.min_freq)
    n_grid_bins = int(domain_settings.NF_active)
    band_lo_hz = asnumpy(structure.band_lo_hz)
    band_hi_hz = asnumpy(structure.band_hi_hz)
    f0_margin_hz = float(f0_limit_fraction) * (band_hi_hz - band_lo_hz)
    f0_lo_hz = band_lo_hz - f0_margin_hz
    f0_hi_hz = band_hi_hz + f0_margin_hz

    reach_hz = np.maximum(stft_reach_hz(f0_lo_hz, structure.tobs_s, epoch_offset_s),
                          stft_reach_hz(f0_hi_hz, structure.tobs_s, epoch_offset_s))

    guard = np.where(band_lo_hz < 1.0e-3, guard_low, guard_high)

    pad_hz = (n_side_bins + guard) * structure.grid_df_hz

    j_lo = np.floor((f0_lo_hz - reach_hz - pad_hz - f_grid_min_hz)
                    / structure.grid_df_hz)
    j_hi = np.ceil((f0_hi_hz + reach_hz + pad_hz - f_grid_min_hz)
                   / structure.grid_df_hz)
    return (np.clip(j_lo, 0, n_grid_bins - 1).astype(int),
            np.clip(j_hi, 0, n_grid_bins - 1).astype(int))


@dataclasses.dataclass(frozen=True)
class STFTStoreWindows:
    """Per-band store windows in active bins, and the widths band buffers are sized from.

    ``j_lo`` and ``j_hi`` are each band's inclusive window, host int arrays. ``tier_width`` maps a
    tier to the largest window width among its interior bands: the moves never build cells for
    the first and last band.
    """

    j_lo: np.ndarray
    j_hi: np.ndarray
    tier: np.ndarray
    tier_width: dict
    n_grid_bins: int
    n_side_bins: int

    @property
    def buffer_width(self) -> int:
        """Width of every band buffer until the schedulers split by tier."""
        return int(max(self.tier_width.values()))

    def window_starts(self, band_inds, width: int) -> np.ndarray:
        """First active bin of the window of each cell in ``band_inds``, for ``width``-bin cells."""
        band_inds = np.asarray(asnumpy(band_inds), dtype=int)
        starts = np.clip(self.j_lo[band_inds], 0, self.n_grid_bins - int(width))
        truncated = self.j_hi[band_inds] - starts >= int(width)
        if np.any(truncated):
            raise ValueError(
                f"store window of band(s) {np.unique(band_inds[truncated]).tolist()} is wider "
                f"than the {int(width)}-bin cells; the window would be truncated."
            )
        return starts.astype(np.int32)


def stft_store_window_layout(structure: GBBandStructure,
                             domain_settings: STFTSettings, *,
                             n_side_bins: int, epoch_offset_s: float,
                             tier_split_hz: float = STFT_TIER_SPLIT_HZ,
                             **window_kwargs) -> STFTStoreWindows:
    """Store windows of every band, their tiers, and the per-tier buffer widths."""
    j_lo, j_hi = stft_store_windows(structure, domain_settings, n_side_bins=n_side_bins,
                                    epoch_offset_s=epoch_offset_s, **window_kwargs)
    tier = np.where(asnumpy(structure.band_lo_hz) < tier_split_hz, 1, 2)
    interior = np.ones(tier.size, dtype=bool)
    if tier.size >= 3:
        interior[[0, -1]] = False
    width = j_hi - j_lo + 1
    tier_width = {int(t): int(width[interior & (tier == t)].max())
                  for t in np.unique(tier[interior])}
    return STFTStoreWindows(j_lo=j_lo, j_hi=j_hi, tier=tier, tier_width=tier_width,
                            n_grid_bins=int(domain_settings.NF_active),
                            n_side_bins=int(n_side_bins))


# ======================================================================
def gb_band_structure(domain_settings: DomainSettingsBase,
                      **kwargs) -> GBBandStructure:
    """Build the band structure matching ``domain_settings``' basis.

    Dispatches on the settings class, the way every other domain-aware branch
    in the GB move machinery does, so a domain is never selected by string.
    The frequency range and the span come from the settings, so there is
    nothing to pass here but the recipe's own parameters.
    """
    if isinstance(domain_settings, STFTSettings):
        return stft_band_structure(domain_settings, **kwargs)
    if isinstance(domain_settings, FDSettings):
        raise NotImplementedError(
            "FD band structure not ported. Recipe, from "
            "GBSetup.init_band_structure: width (2N + extra_buffer)/Tobs with "
            "N = get_N(1e-30, f, Tobs, oversample), walked DOWN from f_hi_hz, "
            "then the [2:-1] out-of-bounds edge trim."
        )
    if isinstance(domain_settings, WDMSettings):
        raise NotImplementedError(
            "WDM band structure not ported. Recipe: one band per wavelet "
            "frequency layer, edges at k * domain_settings.layer_df over "
            "[ceil(f_lo/layer_df), floor(f_hi/layer_df)]."
        )
    raise NotImplementedError(
        f"no band structure for domain {type(domain_settings).__name__}."
    )
