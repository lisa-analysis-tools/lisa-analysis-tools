"""Shared run box, noise model and mojito brick access for the SOBBH speed / accuracy scripts.

Aligned with the EMRI and MBH 1-GPU test setups (2026-10-02):

* box: the 6-month launcher's band (MIN_FREQ 2.5e-4, MAX_FREQ 2.5e-2 Hz) and its edge crop
  (EDGE_CROP_WAVELETS = 60 layers at each end of the time axis), Nf 1440 at dt 2.5 s
  (3600-s layers), Nt = days * 24;
* noise for every SNR and data score: ``XYZ2SensitivityMatrix(dom, model="scirdv1",
  stochastic_params=(Tobs,))`` -- scirdv1 plus the fitted hyperbolic-tangent galactic
  foreground (``FittedHyperbolicTangentGalacticForeground``, the default stochastic function)
  at the WINDOW's Tobs; ``foreground="off"`` gives instrument-only;
* data: the source's own mojito L1 brick through the fit loader's accessor
  (``MojitoL1File(fp).tdis.xyz_doppler``), its ``L1Orbits`` (ICRS), the window starting
  ``START_OFFSET_S`` after the brick start.
"""

from __future__ import annotations

import glob
import os

#: the 6-month launcher's run box
MIN_FREQ, MAX_FREQ = 2.5e-4, 2.5e-2
EDGE_CROP_WAVELETS = 60
#: window start after the brick start [s] (as the EMRI test setup)
START_OFFSET_S = 5.0e4
SOBHB_CATALOGUE = "sobhb_cat_mojito_lite_processed_MT.hdf5"


def run_box(
    nf,
    nt,
    dt,
    t0,
    *,
    edge=EDGE_CROP_WAVELETS,
    min_freq=MIN_FREQ,
    max_freq=MAX_FREQ,
    force_backend="cpu",
    max_time=None,
):
    """The run's WDM box: band [min_freq, max_freq], ``edge`` layers cropped at each end (and
    the end pulled in to ``max_time`` [s from t0] when given, e.g. the orbit tables' end)."""
    from lisatools.domains import WDMSettings

    layer_dt = nf * dt
    if 2 * edge >= nt:
        raise ValueError(f"edge crop {edge} leaves nothing of Nt = {nt}")
    hi = (nt - edge) * layer_dt
    if max_time is not None:
        hi = min(hi, float(max_time))
    return WDMSettings(
        nf,
        nt,
        dt,
        t0=t0,
        min_freq=min_freq,
        max_freq=max_freq,
        min_time=edge * layer_dt,
        max_time=hi,
        force_backend=force_backend,
    )


def noise(wdm, tobs, foreground="on"):
    """scirdv1 (+ the fitted tanh galactic foreground at ``tobs`` [s] unless ``foreground`` is
    ``"off"``) on ``wdm``; returns ``(sensitivity matrix, label)``."""
    from lisatools.sensitivity import XYZ2SensitivityMatrix

    if foreground == "off":
        return XYZ2SensitivityMatrix(wdm, model="scirdv1"), "scirdv1"
    sens = XYZ2SensitivityMatrix(wdm, model="scirdv1", stochastic_params=(float(tobs),))
    return sens, f"scirdv1+tanh-foreground(Tobs={float(tobs) / 86400.0:.0f} d)"


def _brick_dirs(l1_dir=None):
    dirs = []
    if l1_dir:
        dirs.append(l1_dir)
    light = os.environ.get("MOJITO_LIGHT_PATH")
    if light:
        dirs.append(os.path.join(light, "data", "SOBHB", "L1"))
    for root in (
        os.environ.get("MOJITO_DATA_PATH"),
        os.environ.get("MOJITO_INFO_PATH"),
        "/shared/data/mojito_cache",
        os.path.expanduser("~/.mojito_cache"),
    ):
        if root and os.path.isdir(root):
            # bounded recursive search: .../data/SOBHB/L1 up to 4 levels below the root
            for depth in range(0, 5):
                pattern = os.path.join(root, *(["*"] * depth), "data", "SOBHB", "L1")
                dirs.extend(sorted(glob.glob(pattern)))
    out = []
    for d in dirs:
        if d and os.path.isdir(d) and d not in out:
            out.append(d)
    return out


def find_sobhb_bricks(l1_dir=None):
    """``{source id: brick path}`` of every SOBHB L1 brick found (first directory wins)."""
    import re

    found = {}
    for d in _brick_dirs(l1_dir):
        for name in sorted(os.listdir(d)):
            m = re.search(r"_source(\d+)_", name)
            if name.startswith("SOBHB_") and name.endswith(".h5") and m:
                found.setdefault(int(m.group(1)), os.path.join(d, name))
    return found


def find_catalogue(brick, catalogue=None):
    """The SOBHB catalogue: ``catalogue`` if given, else ``<root>/catalogues/<name>`` of the
    brick's mojito tree (``<root>/data/SOBHB/L1/<brick>``)."""
    if catalogue:
        return catalogue
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(brick))))
    path = os.path.join(root, "catalogues", SOBHB_CATALOGUE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no SOBHB catalogue next to {brick} (looked for {path})")
    return path


def catalogue_row(catalogue, src):
    """The catalogue row of SOBHB id ``src`` as a dict of floats."""
    import h5py
    import numpy as np

    with h5py.File(catalogue, "r") as f:
        b = f["Binaries"]
        ids = np.asarray(b["ID"][:]).astype(int)
        hit = np.flatnonzero(ids == int(src))
        if hit.size != 1:
            raise KeyError(f"SOBHB id {src} not (uniquely) in {catalogue}")
        i = int(hit[0])
        return {k: float(np.asarray(b[k][i])) for k in b if np.asarray(b[k][i]).size == 1}


def load_l1_window(brick, nobs, dt, *, start_offset=START_OFFSET_S):
    """``(xyz (3, nobs), t0 of the window, brick dt)`` from the brick, decimated to ``dt`` by
    plain slicing (the bricks are noiseless single-source streams)."""
    import numpy as np
    from mojito import MojitoL1File

    with MojitoL1File(brick) as f:
        ts = f.tdis.time_sampling
        dt_b = float(ts.dt)
        deci = int(round(dt / dt_b))
        if deci < 1 or abs(deci * dt_b - dt) > 1e-9:
            raise ValueError(f"grid dt {dt} is not a multiple of the brick's {dt_b} s")
        t_first = float(np.asarray(ts.t()[0]))
        i0 = int(round(start_offset / dt_b))
        i1 = i0 + nobs * deci
        xyz = np.asarray(f.tdis.xyz_doppler[i0:i1])
    xyz = np.ascontiguousarray(xyz[::deci].T)
    if xyz.shape[1] < nobs:
        raise ValueError(f"brick {os.path.basename(brick)} is shorter than the window")
    return xyz[:, :nobs], t_first + i0 * dt_b, dt_b
