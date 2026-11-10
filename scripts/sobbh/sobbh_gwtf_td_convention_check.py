"""Cross-check the TD SOBBH waveform (coalescence-phase mode) against the gwtf STFT waveform.

Both generators get the same 11 parameters ``m1, m2, s1, s2, dist[Gpc], inc, f0, phi_coal, psi, ra, dec``
with ``waveform_t0 = settings.t0`` (f0 at the data start in both). The TD waveform is placed on the data
grid and STFT'd onto the same ``STFTSettings`` gwtf uses; per source we report the white-noise overlap of
the XYZ STFTs, its argument (a GW-phase offset), and the amplitude ratio, for a few convention variants:
``phi -> -phi`` (reflection) and ``psi -> psi + pi/2`` (overall sign).

The two codes place the chirp differently in time (gwtf's tc is the unrefined ``time_to_merger``, the TD
tc is Newton-refined): gwtf runs late by dtc = tc_gwtf - tc_TD, so arg <gwtf|TD> = phase_TD - phase_gwtf is expected
near +2*pi*f0*dtc (printed as ``pred``), on top of any convention offset.

pygwtf uses the mojito polarisation convention (``POL_CONVENTION = -1`` in pygwtf/response/transfer.py) and
``GWTF_generator`` passes phi_coal through unchanged, so the "as is" variant should have arg ~ pred and the
other variants should not.

Usage: python scripts/sobbh/sobbh_gwtf_td_convention_check.py [--days 32] [--backend cuda12x]
"""

import sys

sys.modules["scienceplots"] = None
import builtins
import typing

builtins.typing = typing

import argparse

import numpy as np

from lisatools.detector import EqualArmlengthOrbits
from lisatools.domains import TDSettings, get_stft_settings
from lisatools.sources.sobbh import SOBBHTaylorT3TDIWaveform
from lisatools.sources.sobbh.waveform import _pn_intrinsics
from lisatools.sources.sobbh.waveform_gwtf import GWTF_generator
from lisatools.utils.utility import asnumpy
from pygwtf.models.taylort3spin import common as gw

ap = argparse.ArgumentParser()
ap.add_argument("--days", type=float, default=32.0)
ap.add_argument("--backend", default="cuda12x")
args = ap.parse_args()

DT, BIG_DT, T0 = 5.0, 86400.0, 1.0e5
N = int(round(args.days * 86400.0 / DT))
times = T0 + np.arange(N) * DT
settings = get_stft_settings(times, big_dt=BIG_DT, min_freq=4e-3, max_freq=4e-2, force_backend=args.backend)
N = int(round(settings.NT * settings.dt / DT))  # whole segments only

orbits = EqualArmlengthOrbits(force_backend=args.backend)
gwtf = GWTF_generator(settings=settings, orbits=orbits, T_obs=N * DT, fresnel_kernel_width=16,
                      use_GPU=args.backend != "cpu", phase_at_coalescence=True)  # same phi_coal as the TD below
td = SOBBHTaylorT3TDIWaveform(
    waveform_t0=settings.t0,
    data_td_settings=TDSettings(N=N, dt=DT, t0=settings.t0, force_backend=args.backend),
    tdi_generation="2nd generation",
    tdi_channels="XYZ",
    sampling_frequency=1.0 / DT,
    orbits=EqualArmlengthOrbits(force_backend=args.backend),
    order=25,
    buffer_time=1.0e4,
    force_backend=args.backend,
    phase_at_coalescence=True,
)
data_grid = TDSettings(N=N, dt=DT, t0=settings.t0, force_backend=args.backend)


def td_stft(p):
    sig = td.get_td_signals(*p, td_settings=data_grid)
    return asnumpy(sig.transform(settings).arr)


def gwtf_stft(p):
    return asnumpy(gwtf.get_signals_for_residuals(*p).arr)


def inner(a, b):
    return np.sum(np.conj(a) * b)


def tc_offset(m1, m2, s1, s2, f0):
    """tc_gwtf - tc_TD (s): unrefined time_to_merger vs the Newton-refined TD tc."""
    M, eta, sigma, s, delta, tc_td = (float(v) for v in _pn_intrinsics(m1, m2, f0, s1, s2))
    tc_g = gw.time_to_merger((np.pi * M * f0) ** (2 / 3), sigma, delta, eta, s) * M
    return tc_g - tc_td


SOURCES = np.array([
    # m1,  m2,   s1,   s2, dist,  inc,    f0,  phi,  psi,  ra,  dec
    [36.0, 29.0, 0.3, -0.2, 0.05, 0.6, 0.012, 1.1, 0.4, 2.0, 0.3],
    [25.0, 18.0, -0.5, 0.4, 0.05, 2.1, 0.018, 0.3, 1.3, 4.5, -0.8],
    [50.0, 45.0, 0.1, 0.1, 0.05, 1.2, 0.009, 2.7, 2.6, 0.7, 1.1],
])
VARIANTS = {
    "as is": lambda p: p,
    "-phi": lambda p: np.r_[p[:7], -p[7], p[8:]],
    "psi+pi/2": lambda p: np.r_[p[:8], p[8] + np.pi / 2, p[9:]],
    "-phi,psi+pi/2": lambda p: np.r_[p[:7], -p[7], p[8] + np.pi / 2, p[9:]],
}

print(settings)
for p in SOURCES:
    g = gwtf_stft(p)
    pred = 2 * np.pi * p[6] * tc_offset(*p[[0, 1, 2, 3, 6]])
    print(f"\nsource m1={p[0]:.0f} m2={p[1]:.0f} f0={p[6]}  (gwtf {g.shape}, pred arg {pred:+.3f} rad)")
    for name, f in VARIANTS.items():
        t = td_stft(f(p))
        assert t.shape == g.shape, (t.shape, g.shape)
        ov = inner(g, t) / np.sqrt(inner(g, g).real * inner(t, t).real)
        print(f"  {name:>14}: |overlap|={abs(ov):.6f}  arg={np.angle(ov):+.4f} rad  "
              f"<t|t>/<g|g>={inner(t, t).real / inner(g, g).real:.4f}")

# ---------------------------------------------------------------- catalogue phase mapping
# TD in its default (reference-epoch, mojito) convention with phi0 = TrueAnomaly, vs gwtf seeded with
# sobhb_catalogue_to_gwtf_basis (coalescence phase, as this gwtf takes it). The catalogue reference
# epoch is the data start here (delta_t = 0), so only the tc offset should remain in arg.
from lisatools.globalfit.searches import sobhb_catalogue_to_gwtf_basis  # noqa: E402

td_ref = SOBBHTaylorT3TDIWaveform(**{**td.kwargs, "phase_at_coalescence": False})
print("\ncatalogue mapping (TD reference mode, phi0 = TrueAnomaly) vs gwtf(sobhb_catalogue_to_gwtf_basis):")
for p, ta in zip(SOURCES, (0.4, 2.2, 5.1)):
    m1, m2, s1, s2, dist, inc, f0, _, psi, ra, dec = p
    entry = dict(PrimaryMassSSBFrame=m1, SecondaryMassSSBFrame=m2, PrimarySpinCompZ=s1, SecondarySpinCompZ=s2,
                 LuminosityDistance=dist * 1e3, InclinationAngle=inc, GW22FrequencySSBFrame=f0, TrueAnomaly=ta,
                 PolarisationAngle=psi, RightAscension=ra, Declination=dec)
    g = gwtf_stft(sobhb_catalogue_to_gwtf_basis(entry, phase_at_coalescence=True))
    t = asnumpy(td_ref.get_td_signals(m1, m2, s1, s2, dist, inc, f0, ta, psi, ra, dec,
                                      td_settings=data_grid).transform(settings).arr)
    ov = inner(g, t) / np.sqrt(inner(g, g).real * inner(t, t).real)
    pred = 2 * np.pi * f0 * tc_offset(m1, m2, s1, s2, f0)
    print(f"  m1={m1:.0f} TrueAnomaly={ta}: |overlap|={abs(ov):.6f}  arg={np.angle(ov):+.4f} rad (pred {pred:+.3f})")
