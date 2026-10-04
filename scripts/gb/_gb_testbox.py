"""Shared run box, noise, engines and mojito access for the GB / VGB speed + accuracy scripts.

Mirrors ``scripts/sobbh/_sobbh_testbox.py`` (EMRI / MBH / SOBBH test setups, 2026-10-02):

* box: the 6-month launcher's band (MIN_FREQ 2.5e-4, MAX_FREQ 2.5e-2 Hz) and edge crop
  (EDGE_CROP_WAVELETS = 60 layers each end), Nf 1440 at dt 2.5 s (3600-s layers), Nt = days*24
  (``--laptop``: Nf 180 at dt 20 s, the same layer);
* noise for every SNR and data score: scirdv1 + the fitted tanh galactic foreground at the
  window's Tobs (``foreground="off"``: instrument only), full XYZ 3x3 per pixel;
* data: the class brick (GB = the whole galaxy, VGB = the 55 verification binaries; one brick
  per class, ``*_source0_*``) read through ``MojitoL1File(...).tdis.xyz_doppler``, window starting
  ``START_OFFSET_S`` after the brick start; catalogue parameters at REF in the validated injection
  convention (``gb_mojito_match.py`` / ``vgb_mojito_match.py``): ``f0 = GW22FrequencySSBFrame``,
  ``phi0 = +TrueAnomaly``, RA/Dec with ICRS orbits, ``t_ref = REF``.

Engines (the arms of every step):

* ``chunked``       -- chunked-het (the production exact scorer and filler);
* ``sighet_reim``   -- sig-het v5, Re/Im control points + Re/Im node ratio (GBGPU dev c81d8bb+,
                       the default);
* ``sighet_ampph``  -- sig-het v5, amplitude/phase control points + log-polar node ratio (the
                       pre-c81d8bb behaviour, ``SIGHET_CP_REPR=ampph``);
* ``lookup``        -- the reference-free direct-to-WDM lookup (``gb_lookup_scorer.py``, Python
                       prototype: common carrier + amplitude-slope term).
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np

MIN_FREQ, MAX_FREQ = 2.5e-4, 2.5e-2
EDGE_CROP_WAVELETS = 60
START_OFFSET_S = 5.0e4
REF = 97729089.327664
SLAB_W = 5
CATALOGUES = {"GB": "wdwd_cat_mojito_lite_processed.hdf5",
              "VGB": "vgb_cat_mojito_lite_processed.hdf5"}
#: catalogue field -> UCB param column (fddot = 0 inserted at index 3)
FIELDS = ["Amplitude", "GW22FrequencySSBFrame", "GW22FrequencyDerivativeSourceFrame",
          "TrueAnomaly", "InclinationAngle", "PolarisationAngle", "RightAscension", "Declination"]
ENGINES = ("chunked", "sighet_reim", "sighet_ampph", "lookup")
#: chunked-het settings of the production GB engine (gb_sighet_bfold_gpu_probe.py)
CHUNKED_KW = dict(Nt_sub=256, n_pad=32, N_sparse=256, N_cp_sig=48, N_cp_orbit=32)
SIGHET_KW = dict(n_sparse_fd=1024, m_active_half_width=2, max_r=0.0, n_cp_build=256,
                 v3_n_nodes=64, v4_knots=128, v4_band=16, v5=1)
TABLE_DEFAULT = os.environ.get(
    "TABLE", "/Users/mkatz/Research/lisa_sprint_2026/"
    "wdm_lookup_emri_cx_NF180_DT20_TL32_fd8x0p01_nld2.h5")


def grid_args(days, laptop=False):
    """``(nf, nt, dt)`` of the run grid (3600-s layers)."""
    nf, dt = (180, 20.0) if laptop else (1440, 2.5)
    return nf, int(round(days * 24)), dt


def run_box(nf, nt, dt, t0, *, edge=EDGE_CROP_WAVELETS, min_freq=MIN_FREQ,
            max_freq=MAX_FREQ, force_backend="cpu"):
    from lisatools.domains import WDMSettings

    layer_dt = nf * dt
    if 2 * edge >= nt:
        raise ValueError(f"edge crop {edge} leaves nothing of Nt = {nt}")
    return WDMSettings(nf, nt, dt, t0=t0, min_freq=min_freq, max_freq=min(max_freq, 0.5 / dt),
                       min_time=edge * layer_dt, max_time=(nt - edge) * layer_dt,
                       force_backend=force_backend)


def noise(wdm, tobs, foreground="on"):
    from lisatools.sensitivity import XYZ2SensitivityMatrix

    if foreground == "off":
        return XYZ2SensitivityMatrix(wdm, model="scirdv1"), "scirdv1"
    sens = XYZ2SensitivityMatrix(wdm, model="scirdv1", stochastic_params=(float(tobs),))
    return sens, f"scirdv1+tanh-foreground(Tobs={float(tobs) / 86400.0:.0f} d)"


def slab_invc(sens, wdm, slab_lo, W=SLAB_W):
    """``(n, 3, 3, W, Nt_active)`` inverse-covariance slabs at absolute layers ``slab_lo``."""
    from lisatools.utils.utility import asnumpy

    invC = np.asarray(asnumpy(sens.invC))                    # (3, 3, Nf_active, Nt_active)
    lo = np.asarray(slab_lo) - int(wdm.ind_min_f)
    return np.stack([invC[:, :, a:a + W, :] for a in lo])


def slab_lo_for(f0, wdm, W=SLAB_W):
    m = np.floor(np.asarray(f0) / wdm.layer_df).astype(int) - W // 2
    return np.clip(m, int(wdm.ind_min_f), int(wdm.ind_max_f) - W + 1).astype(np.int32)


def tukey_for(nt, edge=EDGE_CROP_WAVELETS):
    """Sig-het build taper: 0.01 or less, always subsumed by the crop (EC >= taper)."""
    return float(min(0.01, 0.9 * 2.0 * edge / nt))


class SlabHolder:
    """Narrow per-slot slab holder (the production SubBandBuffer layout)."""

    def __init__(self, data, invc, slab_lo, W=SLAB_W, xp=np):
        self.linear_data_arr = [xp.ascontiguousarray(xp.asarray(data, dtype=xp.float64)).ravel()]
        self.linear_psd_arr = [xp.ascontiguousarray(xp.asarray(invc, dtype=xp.float64)).ravel()]
        self.band_slab_Nf = int(W)
        self.slab_min_f = xp.asarray(np.asarray(slab_lo, dtype=np.int32))
        self.min_freq_inds = self.slab_min_f
        self._n = int(len(slab_lo))

    def __len__(self):
        return self._n


def build_engines(wdm, orbits, *, names=ENGINES, backend="cpu", nt_layer=-1, table=None):
    """``{name: engine}`` -- band-likelihood engines (chunked / sig-het) and the lookup."""
    from gbgpu.gbcomps import GBWDMComputations
    from gbgpu.gbsignalhetcomputations import GBSignalHetComputations
    from gbgpu.gb_likelihood import make_band_likelihood_engine

    chunked = GBWDMComputations(wdm, t_ref=REF, orbits=orbits, tdi_config="2nd generation",
                                force_backend=backend, d_d=0.0, tdi_type="XYZ", **CHUNKED_KW)
    chunked.convert_to_ra_dec = False
    out = {}
    for name in names:
        if name == "chunked":
            out[name] = make_band_likelihood_engine(wdm, gb_wdm_comp=chunked, nchannels=3,
                                                    tdi_channel_setup="XYZ")
        elif name.startswith("sighet_"):
            sig = GBSignalHetComputations.for_band_engine(
                chunked, nt_layer=nt_layer, tukey_alpha=tukey_for(int(wdm.Nt)),
                cp_repr=name.split("_", 1)[1], **SIGHET_KW)
            out[name] = make_band_likelihood_engine(wdm, gb_wdm_comp=sig, nchannels=3,
                                                    tdi_channel_setup="XYZ")
        elif name == "lookup":
            import sys

            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from gb_lookup_scorer import GBLookupTemplate
            from lisatools.domains import WDMLookupTable
            from lisatools.response.tdiconfig import TDIConfig
            from lisatools.wdm_lookup_eval import WDMLookupEvaluator

            tab = WDMLookupTable.from_file(table or TABLE_DEFAULT, force_backend="cpu")
            ev = WDMLookupEvaluator(tab, interp="spline", force_backend=backend)
            out[name] = GBLookupTemplate(
                ev, wdm, orbits=orbits, tdi_config=TDIConfig("2nd generation", force_backend="cpu"),
                t_ref=REF, force_backend=backend)
        else:
            raise ValueError(f"unknown engine {name!r}; choose from {ENGINES}")
    return out


def lookup_inner(tpl, data_slab, invc_slab):
    """Kernel-convention ``(d_h, h_h)`` per row for lookup template slabs ``(N, 3, W, P)``."""
    xp = np if isinstance(tpl, np.ndarray) else __import__("cupy")
    d_h = xp.einsum("ncwt,ncdwt,ndwt->n", data_slab, invc_slab, tpl)
    h_h = xp.einsum("ncwt,ncdwt,ndwt->n", tpl, invc_slab, tpl)
    return d_h, h_h


# ---- mojito -------------------------------------------------------------------------------
def _brick_dirs(kind, l1_dir=None):
    dirs = [l1_dir] if l1_dir else []
    light = os.environ.get("MOJITO_LIGHT_PATH")
    if light:
        dirs.append(os.path.join(light, "data", kind, "L1"))
    for root in (os.environ.get("MOJITO_DATA_PATH"), os.environ.get("MOJITO_INFO_PATH"),
                 "/shared/data/mojito_cache", os.path.expanduser("~/.mojito_cache")):
        if root and os.path.isdir(root):
            for depth in range(0, 5):
                dirs.extend(sorted(glob.glob(os.path.join(root, *(["*"] * depth), "data", kind, "L1"))))
    out = []
    for d in dirs:
        if d and os.path.isdir(d) and d not in out:
            out.append(d)
    return out


def find_brick(kind, l1_dir=None):
    """The class brick (``GB`` or ``VGB``; one per class, ``*_source0_*``) or ``None``."""
    for d in _brick_dirs(kind, l1_dir):
        for name in sorted(os.listdir(d)):
            if name.startswith(f"{kind}_") and name.endswith(".h5") and re.search(r"_source0_", name):
                return os.path.join(d, name)
    return None


def find_catalogue(kind, brick, catalogue=None):
    if catalogue:
        return catalogue
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(brick))))
    path = os.path.join(root, "catalogues", CATALOGUES[kind])
    if not os.path.exists(path):
        raise FileNotFoundError(f"no {kind} catalogue next to {brick} (looked for {path})")
    return path


def catalogue_params(catalogue, idx=None, *, top_f=None, fmin=None, fmax=None):
    """``(params (n, 9), ids, rows)``: the 9 UCB parameters at REF of the selected rows --
    ``idx`` explicit rows, or the ``top_f`` highest-frequency rows, or every row with
    ``fmin <= f0 <= fmax``."""
    import h5py

    with h5py.File(catalogue, "r") as f:
        b = f["Binaries"]
        f0 = np.asarray(b["GW22FrequencySSBFrame"][:])
        if idx is None:
            if top_f is not None:
                idx = np.argsort(f0)[::-1][: int(top_f)]
            else:
                idx = np.flatnonzero((f0 >= fmin) & (f0 <= fmax))
        idx = np.sort(np.asarray(idx, dtype=int))
        cols = [np.asarray(b[k][:])[idx] if k != "GW22FrequencySSBFrame" else f0[idx]
                for k in FIELDS]
        ids = np.asarray(b["ID"][:])[idx]
    A, f0s, fdot, phi0, inc, psi, ra, dec = cols
    params = np.column_stack([A, f0s, fdot, np.zeros_like(A), phi0, inc, psi, ra, dec])
    ids = [i.decode() if isinstance(i, bytes) else str(i) for i in ids]
    return params, ids, idx


def load_l1_window(brick, nobs, dt, *, start_offset=START_OFFSET_S):
    """``(xyz (3, nobs), t0 of the window, brick dt)``, decimated to ``dt`` by slicing
    (the bricks are noiseless signal streams)."""
    from mojito import MojitoL1File

    with MojitoL1File(brick) as f:
        ts = f.tdis.time_sampling
        dt_b = float(ts.dt)
        deci = int(round(dt / dt_b))
        if deci < 1 or abs(deci * dt_b - dt) > 1e-9:
            raise ValueError(f"grid dt {dt} is not a multiple of the brick's {dt_b} s")
        t_first = float(np.asarray(ts.t()[0]))
        i0 = int(round(start_offset / dt_b))
        xyz = np.asarray(f.tdis.xyz_doppler[i0:i0 + nobs * deci])
    xyz = np.ascontiguousarray(xyz[::deci].T)
    if xyz.shape[1] < nobs:
        raise ValueError(f"brick {os.path.basename(brick)} is shorter than the window")
    return xyz[:, :nobs], t_first + i0 * dt_b, dt_b


def l1_orbits(brick, t_lo, t_hi, backend="cpu", pad=1.0e5):
    """The brick's ICRS L1 orbits with the light-travel tables trimmed to the window (memory)."""
    from lisatools.detector import L1Orbits

    orb = L1Orbits(brick, force_backend=backend, frame="icrs")
    lo, hi = max(t_lo - pad, float(orb.sc_t0)), min(t_hi + pad, float(orb._sc_t_base[-1]))
    lt = np.asarray(orb.ltt_t)
    mk = (lt >= lo) & (lt <= hi)
    orb.ltt = np.asarray(orb.ltt)[mk].copy()
    orb.ltt_t = lt[mk].copy()
    orb.ltt_t0 = float(orb.ltt_t[0])
    orb.configure(linear_interp_setup=True)
    return orb
