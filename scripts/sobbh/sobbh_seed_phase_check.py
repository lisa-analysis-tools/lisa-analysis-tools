"""Does the catalogue phase mapping land on the data? (gwtf SOBBH, L2D run folder)

Builds the run folder's GlobalFit up to the recipe (data, sensitivity, seeded state), then on the raw data
(no templates subtracted) scans, per catalogue source, the phase parameter (phi_0 or phi_coal, the run's phase_at_coalescence) around the
mapped truth:

    log L(phi) = Re<d|h(phi)> - <h|h>/2   (gwtf inner products on walker 0's data and PSD, as SOBBHSpecialMove)

and reports where it peaks relative to the mapped phase (``sobhb_catalogue_to_gwtf_basis``), with the SNR
``sqrt(<h|h>)`` and the matched-filter SNR ``Re<d|h>/sqrt(<h|h>)`` at the truth. The peak should sit within a few
hundredths of a radian of the mapped phase (gwtf's coalescence-time offset), not pi/2 away (a sign error).

Usage: python scripts/sobbh/sobbh_seed_phase_check.py RUN_FOLDER [--nphi 72]
Writes the backend/artifacts under the folder's run.output_dir; use a scratch copy of the run folder.
"""
import sys

sys.modules["scienceplots"] = None
import builtins
import typing

builtins.typing = typing

import argparse
import importlib.util
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--nphi", type=int, default=72)
args = ap.parse_args()

spec = importlib.util.spec_from_file_location("run_global", Path(__file__).parents[1] / "run_global.py")
run_global = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_global)

from mpi4py import MPI  # noqa: E402

from lisatools.globalfit.run import GlobalFit  # noqa: E402
from lisatools.globalfit.searches import sobhb_catalogue_to_gwtf_basis  # noqa: E402
from lisatools.utils.utility import asnumpy  # noqa: E402


class _Captured(Exception):
    pass


cfg, curr, _ = run_global.curr_info_from_run_config(args.run, 0)
gf = GlobalFit(curr, MPI.COMM_WORLD)
captured = {}


def capture(recipe, engine_info, curr_, acs, priors, state):
    captured.update(state=state)
    raise _Captured


curr.settings_dict.setup_function = capture
try:
    gf.run_global_fit()
except _Captured:
    pass

acs = gf.setup_acs(captured["state"], rebuild_residuals=False)  # the raw data, no template subtracted
gen = curr.source_info["sobbh"].signal_gen.waveform
n = len(acs.gpu_splits[0])
with acs.device_context(acs.gpus[0] if acs.gpus else None):
    data = gen.to_gwtf_layout_from_xyz(acs.linear_data_arr[0].reshape(n, acs.nchannels, *acs.end_shape))[:1]
    invC = acs.linear_psd_arr[0].reshape(n, *acs.shape_sens, *acs.end_shape)[:1]
    psd = gen.psd_aet_from_invC(invC)

data_t0 = curr.general_info.data_t0
catalogue = curr.general_info.catalogue["SOBHB"]
offsets = np.linspace(-np.pi / 2, np.pi / 2, args.nphi, endpoint=False)  # the phase enters as 2*phi: period pi
print(f"{'id':>3} {'f0 [Hz]':>9} {'SNR':>7} {'MF SNR':>7} {'peak - mapped [rad]':>20} {'dlogL(mapped)':>14}")
for sid, entry in sorted(catalogue.items()):
    truth = sobhb_catalogue_to_gwtf_basis(
        entry, data_t0 - float(entry["TimeReferenceSSBFrame"]), phase_at_coalescence=gen.phase_at_coalescence
    )
    params = np.tile(truth, (args.nphi, 1))
    params[:, 7] = truth[7] + offsets
    idx = gen.xp.zeros(args.nphi, dtype=gen.xp.int32)
    d_h, h_h = gen.compute_inner_products(gen.xp.asarray(params), idx, data, psd)
    d_h, h_h = asnumpy(d_h), asnumpy(h_h).real
    logl = d_h.real - 0.5 * h_h
    i0 = int(np.argmin(np.abs(offsets)))
    # parabolic refinement of the peak on the periodic grid
    k = int(np.argmax(logl))
    ym, y0, yp = logl[k - 1], logl[k], logl[(k + 1) % args.nphi]
    peak = offsets[k] + 0.5 * (ym - yp) / (ym - 2 * y0 + yp) * (offsets[1] - offsets[0])
    snr = np.sqrt(h_h[i0])
    print(f"{sid:>3} {truth[6]:9.5f} {snr:7.2f} {d_h[i0].real / snr:7.2f} {peak:+20.4f} {logl[i0] - logl.max():14.4f}")
