"""Oracle for the overlapping STFT grid of the GB likelihood path (overlap = 3, round 2a).

An overlapping grid with hop T/3 and one-day Tukey alpha = 1 (Hann) windows is three contiguous
one-day grids, offset by 0, T/3 and 2T/3: segment k = 3 j + o of the overlapping grid is segment
j of the offset-o grid. With the data and inverse covariance of the overlapping grid interleaved
from the three offset grids, for the 24 production sources and both responses:

* ``fill_global_stft`` on the overlapping grid equals, segment by segment, the templates of the
  offset grids;
* every inner product the C++ kernels return (d|h), (h|h), the five swap terms, the log-likelihood
  gradient and the Fisher matrix equals 8/9 (the weight df / window_power) times the sum over the
  three offset grids.

The overlapping grid runs through a real ``STFTComputationGroup`` (seg_duration, diff_comp and the
kinematics table from the group); the offset grids run through today's contiguous wraps. The grid
start is a whole second and the hop 28800 s, so every segment start and midpoint is the same
double on both sides.

    python stft_overlap_oracle.py [cpu|gpu] [--nt 8]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stft_fixtures import (  # noqa: E402
    DATA_START_OFFSET, SEGMENT_DURATION, ShimGroup, array_module, draw_cell_arrays, load_orbits,
    production_settings, production_sources, to_host,
)

OVERLAP = 3
HANN = 1.0
N_SIDE_BINS = 4
TEMPLATE_TOLERANCE = 1e-12  # of the peak pixel of each source
SCALAR_TOLERANCE = 1e-12  # of the largest magnitude of the quantity over the sources
# * The gradient is a central difference of the log-likelihood with relative steps near 1e-6, so a
# * rounding change of 1e-15 in ll (the overlapping grid sums in another order) moves it by up to
# * about 1e-15 |ll| / (eps |dll/dtheta|); the 8/9 linearity itself is checked exactly on the terms.
GRADIENT_TOLERANCE = 1e-5


def make_comp(orbits, device, response, nt):
    from gbgpu.gbcomps import STFTGBComputations

    orbits_object, orbits_start = orbits
    return STFTGBComputations(
        stft_comps=None, T=nt * SEGMENT_DURATION, t_ref=orbits_start, orbits=orbits_object,
        tdi_config="2nd generation", force_backend=device, n_side_bins=N_SIDE_BINS,
        window_factor=1.0, freq_from_tdi_phase=True, window_alpha=HANN, use_midpoint=True,
        linear_envelope=True, response=response, first_order_delays=True,
    )


def offset_group(comp, band, data, invc, response):
    """Today's contiguous wraps on one offset grid, as the Phase 0 script builds them."""
    from lisatools.domaincomputation import KINEMATICS_TABLE_FIELDS

    backend, xp = comp.backend, comp.xp
    data_d, invc_d = xp.asarray(data[None]), xp.asarray(invc[None])
    grid = (band.NT, band.NF_active, 3, band.t0, band.min_freq, band.max_freq, band.dt, band.df)
    domain = backend.STFTDomainWrap(*grid, data_d.reshape(-1), invc_d.reshape(-1), 1, 1,
                                    backend.TDITypeDict["XYZ"])
    fresnel = backend.STFTFresnelWrap(*grid, window_alpha=HANN, use_midpoint=True,
                                      linear_envelope=True)
    kinematics, arrays = None, {}
    if response == "dffr":
        arrays = {n: xp.zeros(int(np.prod(s)) * band.NT) for n, s in KINEMATICS_TABLE_FIELDS}
        kinematics = backend.KinematicsTableWrap(band.t0 + 0.5 * band.dt, band.dt, band.NT,
                                                 *arrays.values())
        kinematics.build(comp.cpp_orbits)
    # ! The wraps hold raw pointers into these buffers, the kinematics arrays included.
    return ShimGroup(fresnel, domain, num_noise=1, buffers=(data_d, invc_d, arrays),
                     linear_envelope=True if response == "dffr" else None, kinematics=kinematics)


def overlapping_group(band, data, invc, device):
    """A real STFTComputationGroup on the overlapping grid, one analysis container."""
    from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
    from lisatools.detector import EqualArmlengthOrbits
    from lisatools.sensitivity import XYZSensitivityBackend

    xp = array_module(device)
    sensitivity = XYZSensitivityBackend(
        orbits=EqualArmlengthOrbits(force_backend=device), settings=band, force_backend=device
    )
    sensitivity.sens_mat = xp.zeros(invc.shape, dtype=np.complex128)
    sensitivity.invC = xp.asarray(invc)
    sensitivity.channel_shape = invc.shape[: -len(band.basis_shape_active)]
    container = AnalysisContainer(band.associated_class(xp.asarray(data), band), sensitivity)
    array = AnalysisContainerArray(
        [container], gpus=None if device == "cpu" else [0],
        domain_group_kwargs=dict(tdi_type="XYZ", window_alpha=HANN, use_midpoint=True,
                                 linear_envelope=True),
    )
    return array, array.cpp_splits[0]


def evaluate(comp, params, swap_partner, band):
    """Templates and every inner-product quantity of the kernels, on the host."""
    xp = comp.xp
    n = params.shape[0]
    num_times, num_freqs = band.NT, band.NF_active
    templates = xp.zeros((n, 3, num_times, num_freqs), dtype=np.complex128)
    comp.fill_global_stft(params, templates, data_index=xp.arange(n, dtype=np.int32),
                          factors=xp.ones(n))
    zero = xp.zeros(n, dtype=np.int32)
    comp.get_ll_stft(params, data_index=zero, noise_index=zero)
    out = {"templates": to_host(templates), "d_h": to_host(comp.d_h_out),
           "h_h": to_host(comp.h_h_out)}
    swap = comp.get_swap_ll_stft(params, swap_partner, data_index=zero, noise_index=zero)
    for name, value in zip(("d_h_add", "d_h_remove", "add_add", "remove_remove", "add_remove"),
                           swap[2:]):
        out["swap " + name] = to_host(value)
    out["grad"] = to_host(comp.get_ll_grad_stft(params, data_index=zero, noise_index=zero))
    out["fisher"] = to_host(comp.information_matrix(params, noise_index=zero))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("device", nargs="?", default="gpu", choices=("cpu", "gpu"))
    parser.add_argument("--nt", type=int, default=8, help="one-day segments per offset grid")
    args = parser.parse_args()
    device, nt = args.device, args.nt

    import lisatools
    import gbgpu  # noqa: F401
    from lisatools.domains import STFTSettings

    native = lisatools.get_backend("cpu" if device == "cpu" else "gpu")
    print(f"lisatools {Path(lisatools.__file__).parent}")
    print(f"backend   {native.__class__.__name__}")

    orbits = load_orbits("esa", device)
    # * A whole second, so that every segment start and midpoint is exact on both sides.
    t0 = float(np.ceil(orbits[1] + DATA_START_OFFSET))
    hop = SEGMENT_DURATION / OVERLAP
    assert hop == 28800.0

    offset_bands = [production_settings(t0 + o * hop, nt, device)[0] for o in range(OVERLAP)]
    band0 = offset_bands[0]
    overlap_band = STFTSettings(
        t0=t0, dt=hop, df=band0.df, NT=OVERLAP * nt, NF=band0.NF, min_freq=band0.min_freq,
        max_freq=band0.max_freq, overlap=OVERLAP, force_backend=device,
    )
    assert overlap_band.NF_active == band0.NF_active
    assert overlap_band.differential_component == band0.df * 8.0 / 9.0

    params, _, _ = production_sources(band0)
    partner = params[::-1].copy()
    rng = np.random.default_rng(20261114)
    draws = [draw_cell_arrays(rng, nt, band0.NF_active, False) for _ in range(OVERLAP)]
    # * Segment k = 3 j + o of the overlapping grid is segment j of the offset-o grid.
    data = np.empty((3, OVERLAP * nt, band0.NF_active), dtype=np.complex128)
    invc = np.empty((3, 3, OVERLAP * nt, band0.NF_active), dtype=np.complex128)
    for o, (d, c) in enumerate(draws):
        data[:, o::OVERLAP] = d
        invc[:, :, o::OVERLAP] = c

    print(f"{params.shape[0]} sources, {nt} one-day segments per offset grid, "
          f"{OVERLAP * nt} overlapping segments, {band0.NF_active} bins, device {device}")
    failures = 0
    for response in ("tdi_on_the_fly", "dffr"):
        offsets = []
        for o, band in enumerate(offset_bands):
            comp = make_comp(orbits, device, response, nt)
            comp.stft_comps = offset_group(comp, band, *draws[o], response)
            offsets.append(evaluate(comp, params, partner, band))

        _keep, group = overlapping_group(overlap_band, data, invc, device)
        assert group.cpp_fresnel.seg_duration == SEGMENT_DURATION
        assert group.cpp_domain.seg_duration == SEGMENT_DURATION
        assert group.cpp_domain.diff_comp == overlap_band.differential_component
        comp = make_comp(orbits, device, response, nt)
        comp.stft_comps = group
        overlapping = evaluate(comp, params, partner, overlap_band)

        print(f"\n[{response}]")
        templates = overlapping["templates"]
        peak = np.abs(templates).reshape(len(templates), -1).max(axis=1)
        error = max(
            (np.abs(templates[:, :, o::OVERLAP] - offsets[o]["templates"]).reshape(len(peak), -1)
             .max(axis=1) / peak).max()
            for o in range(OVERLAP)
        )
        ok = error <= TEMPLATE_TOLERANCE
        failures += not ok
        print(f"  templates (k = 3j + o) vs offset grids: max {error:.2e} of peak  "
              f"{'OK' if ok else 'FAIL'}")
        for name in overlapping:
            if name == "templates":
                continue
            expected = 8.0 / 9.0 * sum(offsets[o][name] for o in range(OVERLAP))
            scale = np.abs(expected).max()
            error = np.abs(overlapping[name] - expected).max() / scale
            ok = error <= (GRADIENT_TOLERANCE if name == "grad" else SCALAR_TOLERANCE)
            failures += not ok
            print(f"  {name:20s} = 8/9 sum over offsets: max {error:.2e} of max  "
                  f"{'OK' if ok else 'FAIL'}")
    print(f"\n{'ALL OK' if failures == 0 else f'{failures} FAILED'}")
    return failures


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
