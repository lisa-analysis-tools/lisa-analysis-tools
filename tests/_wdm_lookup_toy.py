# tests/_wdm_lookup_toy.py
"""Shared toy pieces for the WDM lookup tests: a tiny n_ref table, a chirp truth, a mismatch."""

import os

import numpy as np


def build_tiny_table(
    dirname,
    *,
    nf=64,
    dt=56.25,
    nt=128,
    eps_freq=0.01,
    eps_fdot=0.1,
    fdot_max_factor=1.0,
    m_ref=20,
    time_layers=64,
):
    """A small ``n_ref_complex`` table with layer duration ``nf * dt`` (3600 s by default).

    Offsets cover [-3, 3) layers (``num_layers_diff=2``); the fdot axis spans
    ``+-fdot_max_factor`` layer units in steps of ``eps_fdot``. Builds in a few seconds.
    """
    from lisatools.domains import WDMLookupTable, WDMSettings

    wdm = WDMSettings(Nf=nf, Nt=nt, dt=dt, force_backend="cpu")
    norm_f, m_diffs, m_ref = WDMLookupTable.apply_eps_frequency(
        eps_freq, wdm, m_ref=m_ref, num_layers_diff=2
    )
    fdot_vals = WDMLookupTable.apply_eps_fdot(eps_fdot, wdm, fdot_max_factor=fdot_max_factor)
    table = WDMLookupTable(
        wdm,
        1,
        m_ref=m_ref,
        norm_freq_single_layer=norm_f,
        m_diffs=m_diffs,
        fdot_vals=fdot_vals,
        store_path=os.path.join(dirname, f"lookup_nf{nf}_dt{dt:g}.h5"),
        batch_size_gen=64,
        build_kind="n_ref_complex",
        time_layers=time_layers,
    )
    return wdm, table


def chirp_truth(wdm, f0, fdot, phi0):
    """TD->WDM of cos(2 pi (f0 t + fdot t^2 / 2) + phi0) on ``wdm``'s grid: (Nf_active, Nt)."""
    from lisatools.domains import TDSettings, TDSignal

    N = wdm.Nf * wdm.Nt
    t = np.arange(N) * wdm.data_dt
    y = np.cos(2 * np.pi * (f0 * t + 0.5 * fdot * t**2) + phi0)[None, :]
    return np.asarray(
        TDSignal(y, TDSettings(N, wdm.data_dt, force_backend="cpu")).transform(wdm).arr
    )[0]


def td_mismatch(a, b):
    """(1 - <a,b>/sqrt(<a,a><b,b>), sqrt(<a,a>/<b,b>)) for two real arrays of the same shape."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    aa, bb, ab = float(a @ a), float(b @ b), float(a @ b)
    return 1.0 - ab / np.sqrt(aa * bb), np.sqrt(aa / bb)
