"""Error budget of the factorized (approximate) window-convolution mode (agent 5).

For each config and parameter family, compares on the SAME backend instance
  exact  = legacy ``_compute_matrix_elements(..., convolve_window=True)`` (12 FFTs/walker)
  fact   = unit bases x 10^spl (splines) + factorized galaxy rows
and reports per-bin relative errors of the 6 covariance entries and delta log L
(both evaluated by the same compiled kernel, ``psd_likelihood_wrap`` with an
external matrix, so the difference isolates the approximation).

Usage::

    CUDA_VISIBLE_DEVICES=3 .venv/bin/python agent5_error_budget.py --backend cuda12x \
        --configs gal1_spl1_sym gal1_spl0_sym --out agent5_results.json [--T-days 30]
"""

from __future__ import annotations

import argparse
import copy
import json
import time
import zlib

import numpy as np

import _use_worktree  # noqa: F401
from fixtures import CONFIGS, asnumpy, make_backend, make_inputs, to_xp
from lisatools import sensitivity_window as sw

ENTRY_NAMES = ("00", "11", "22", "01", "02", "12")
MIN_KNOT_GAP = 0.05  # dex


# ------------------------------------------------------------------ parameter families


def draw_family(rng, sm, fam, n):
    """Walker params ``dict`` for family ``fam`` (host arrays)."""
    cfg = sm._fixture_config
    n_par = sm.n_noise_par
    shape = (n, n_par) if n_par > 1 else (n,)
    p = dict(Soms=15e-12 * rng.uniform(0.8, 1.2, shape), Sa=3e-15 * rng.uniform(0.8, 1.2, shape))
    # galaxy
    if fam.startswith("sharp"):
        # sharp knee inside the band: small f_2
        p.update(Amp=10 ** rng.uniform(-44.2, -43.2, n), alpha=rng.uniform(1.2, 2.5, n),
                 f_1=10 ** rng.uniform(-3.0, -2.4, n), kn=10 ** rng.uniform(-3.0, -2.0, n),
                 f_2=10 ** rng.uniform(-4.0, -3.4, n))
    else:
        p.update(Amp=10 ** rng.uniform(-44.2, -43.2, n), alpha=rng.uniform(1.2, 2.5, n),
                 f_1=10 ** rng.uniform(-3.0, -2.4, n), kn=10 ** rng.uniform(-2.8, -2.2, n),
                 f_2=10 ** rng.uniform(-3.4, -2.8, n))
    # splines
    if sm.use_splines:
        sig = {"spl01": 0.1, "spl03": 0.3, "spl05": 0.5}.get(fam.split("_")[-1], 0.1)
        f = asnumpy(sm.f_arr)
        f = f[f > 0]
        lo, hi = np.log10(f[0]) - 0.1, np.log10(f[-1]) + 0.1
        k = cfg["n_knots"]
        inner = np.sort(rng.uniform(lo, hi, (2, n, k - 2)), axis=-1)
        # reject draws with near-coincident knots (Akima then overshoots to 10^20+,
        # making even the legacy convolved C non-positive-definite): min gap 0.05 dex
        for _ in range(1000):
            allk = np.concatenate([np.full((2, n, 1), lo), inner, np.full((2, n, 1), hi)], axis=-1)
            bad = (np.diff(allk, axis=-1) < MIN_KNOT_GAP).any(axis=-1)
            if not bad.any():
                break
            inner[bad] = np.sort(rng.uniform(lo, hi, (int(bad.sum()), k - 2)), axis=-1)
        p["kp"] = np.concatenate([np.full((2, n, 1), lo), inner, np.full((2, n, 1), hi)], axis=-1)
        p["ka"] = sig * rng.standard_normal((2, n, k))
    else:
        p["kp"] = p["ka"] = None
    return p


# ------------------------------------------------------------------ assembly / likelihood


def exact_entries(sm, p, Amp_on=True):
    xp = sm.xp
    a = {k: (xp.asarray(v) if v is not None else None) for k, v in p.items()}
    Amp = a["Amp"] if Amp_on else xp.zeros_like(a["Amp"])
    return sm._compute_matrix_elements(
        sm.f_arr, a["Soms"], a["Sa"], Amp, a["alpha"], a["f_1"], a["kn"], a["f_2"], a["kp"], a["ka"],
        convolve_window=True,
    )


def fact_instrument(sm, p, order=0):
    """Factorized instrument entries (in-band factorized, edge padding exact)."""
    xp = sm.xp
    coeff = sw.basis_coeffs(sm, xp.asarray(p["Soms"]), xp.asarray(p["Sa"]))  # (n, nb)
    if not sm.use_splines:
        return [coeff @ b for b in sw.get_instrument_bases(sm)]
    Bj, (lo_v, hi_v, P_lo, P_hi) = sw.factorized_instrument_moment_bases(sm, order)
    w = sw.factorized_spline_weights(sm, xp.asarray(p["kp"]), xp.asarray(p["ka"]), order=order)
    if order == 0:
        w = w[:, :, None, :]
    nt, nf, nb = sm.num_times, sm.num_freqs, coeff.shape[1]
    typ = xp.arange(nb) // sm.n_noise_par  # 0 = OMS, 1 = TM
    out = [0.0] * 6
    for j, B in enumerate(Bj):
        W = xp.tile(w[:, :, j, :], (1, 1, nt))[:, typ, :]  # (n, nb, total)
        cw = coeff[:, :, None] * W
        for k in range(6):
            out[k] = out[k] + xp.einsum("nbf,bf->nf", cw, B[k])
    w0 = w[:, :, 0, :][:, typ, :]  # (n, nb, nf)
    for k in range(6):
        a_lo = xp.einsum("nb,bt->nt", coeff * w0[:, :, 0], lo_v[k])  # (n, nt)
        a_hi = xp.einsum("nb,bt->nt", coeff * w0[:, :, -1], hi_v[k])
        edge = a_lo[:, :, None] * P_lo[None, None, :] + a_hi[:, :, None] * P_hi[None, None, :]
        out[k] = out[k] + edge.reshape(edge.shape[0], -1)
    return out


def gal_entries(sm, gal_conv, ratios):
    """6 entries (c00, c11, c22, c01, c02, c12) from (gal_conv, ratios[k in 00,01,02,11,12,22])."""
    n = gal_conv.shape[0]
    g = gal_conv.reshape(n, -1)
    nt, nf = sm.num_times, sm.num_freqs
    r = sm.xp.repeat(ratios, nf, axis=0)  # (nt*nf, 6)
    idx = {"00": 0, "01": 1, "02": 2, "11": 3, "12": 4, "22": 5}
    out = [g * r[None, :, idx[k]] for k in ENTRY_NAMES]
    return [o if i < 3 else o.astype(sm.xp.complex128) for i, o in enumerate(out)]


def ll_external(sm, inp, p, entries):
    xp = sm.xp
    llk = inp["log_like_kwargs"]
    n = entries[0].shape[0]
    out = xp.zeros(n, dtype=xp.float64)
    e = xp.empty(0, dtype=xp.float64)
    f64 = lambda v: xp.ascontiguousarray(xp.asarray(v, dtype=xp.float64)).ravel()
    sm.pycpp_sensitivity_matrix.psd_likelihood_wrap(
        out, sm.f_arr, xp.asarray(llk["data_in_all"]).ravel(),
        xp.asarray(llk["data_index_all"], dtype=xp.int32).ravel(), xp.asarray(sm.time_indices),
        f64(p["Soms"]), f64(p["Sa"]), f64(p["Amp"]), f64(p["alpha"]), f64(p["f_1"]), f64(p["kn"]),
        f64(p["f_2"]), e, e, sm.basis_settings.differential_component, sm.num_freqs, sm.num_times,
        sm.dips_mask, n,
        *[xp.ascontiguousarray(c).ravel() for c in entries], True, False,
    )
    return asnumpy(out)


BAND_EDGES = (1e-4, 1e-3)  # dlogL split: f < 1e-4 | 1e-4..1e-3 | >= 1e-3 Hz


def ll_bands(sm, inp, entries):
    """Per-walker log-likelihood summed in BAND_EDGES bands; per-bin algebra of
    psd_likelihood_xyz_kernel (dip mask, skip det == 0 / non-finite, Q non-finite / < 0),
    in xp, walker by walker. Returns (n, 3) host array."""
    xp = sm.xp
    llk = inp["log_like_kwargs"]
    nf, nt = sm.num_freqs, sm.num_times
    data = xp.asarray(llk["data_in_all"]).reshape(-1, 3, nt * nf)
    didx = asnumpy(llk["data_index_all"])
    fk = xp.tile(xp.asarray(sm.f_arr), nt)
    band = xp.searchsorted(xp.asarray(BAND_EDGES), fk, side="right")
    dips = xp.asarray(sm.dips_mask)
    df = sm.basis_settings.differential_component
    out = np.zeros((entries[0].shape[0], 3))
    for w in range(entries[0].shape[0]):
        c00, c11, c22, c01, c02, c12 = [e[w] for e in entries]
        c00, c11, c22 = c00.real, c11.real, c22.real
        C00 = c11 * c22 - xp.abs(c12) ** 2
        C11 = c00 * c22 - xp.abs(c02) ** 2
        C22 = c00 * c11 - xp.abs(c01) ** 2
        C01 = c12 * xp.conj(c02) - xp.conj(c01) * c22
        C02 = xp.conj(c01) * xp.conj(c12) - c11 * xp.conj(c02)
        C12 = c01 * xp.conj(c02) - c00 * xp.conj(c12)
        det = c00 * C00 + (c01 * C01).real + (c02 * C02).real
        X, Y, Z = data[int(didx[w])]
        # adjugate quadratic form: q = d^H adj(C) d / det
        num = (C00 * xp.abs(X) ** 2 + C11 * xp.abs(Y) ** 2 + C22 * xp.abs(Z) ** 2
               + 2 * (xp.conj(X) * xp.conj(C01) * Y).real + 2 * (xp.conj(X) * xp.conj(C02) * Z).real
               + 2 * (xp.conj(Y) * xp.conj(C12) * Z).real)
        q = num / det
        term = -0.5 * 4 * df * q - xp.log(xp.abs(det))
        skip = dips | (det == 0) | ~xp.isfinite(det) | ~xp.isfinite(q) | (q < 0)
        term = xp.where(skip, 0.0, term)
        for b in range(3):
            out[w, b] = float(term[band == b].sum())
    return out


def _herm(e):
    """(n, nf, 3, 3) Hermitian matrices from the 6 entries."""
    c00, c11, c22, c01, c02, c12 = e
    xp = c01.__class__ is np.ndarray and np or __import__("cupy")
    r0 = xp.stack([c00 + 0j, c01, c02], -1)
    r1 = xp.stack([c01.conj(), c11 + 0j, c12], -1)
    r2 = xp.stack([c02.conj(), c12.conj(), c22 + 0j], -1)
    return xp.stack([r0, r1, r2], -2)


def entry_errors(sm, ex, ap):
    """Per-bin relative errors; diag / C_ii, offdiag / sqrt(C_ii C_jj); dips excluded."""
    xp = sm.xp
    keep = ~xp.asarray(sm.dips_mask)
    diag = [ex[0].real, ex[1].real, ex[2].real]
    pair = {3: (0, 1), 4: (0, 2), 5: (1, 2)}
    errs = []
    for k in range(6):
        d = xp.abs(ap[k] - ex[k])
        den = diag[k] if k < 3 else xp.sqrt(diag[pair[k][0]] * diag[pair[k][1]])
        errs.append(d / xp.abs(den))
    # whitened error: eps(f) = sqrt(tr[(C^-1 dC)^2]) = ||L^-1 dC L^-H||_F (what drives dlogL)
    epsw = []
    for w in range(ex[0].shape[0]):  # per walker: bounds memory at 2 yr
        M = _herm([e[w : w + 1] for e in ex])
        D = _herm([a[w : w + 1] - b[w : w + 1] for a, b in zip(ap, ex)])
        P = xp.linalg.solve(M, D)
        epsw.append(asnumpy(xp.sqrt(xp.abs(xp.einsum("nfij,nfji->nf", P, P)))[:, keep]))
        del M, D, P
    epsw = np.concatenate(epsw)
    E = xp.stack(errs, axis=1)[:, :, keep]  # (n, 6, nkeep)
    E = asnumpy(E)
    fk = asnumpy(xp.tile(sm.f_arr, sm.num_times)[keep])
    flat = E.max(axis=1)  # worst entry per bin
    imax = np.unravel_index(np.argmax(E), E.shape)
    iw = np.unravel_index(np.nanargmax(epsw), epsw.shape)
    return dict(
        w_median=float(np.nanmedian(epsw)), w_p99=float(np.nanpercentile(epsw, 99)),
        w_max=float(np.nanmax(epsw)), w_f_at_max=float(fk[iw[1]]),
        w_sum=float(np.nanmax(np.nansum(epsw, axis=1))), w_rss=float(np.nanmax(np.sqrt(np.nansum(epsw**2, axis=1)))),
        median=float(np.median(flat)), p99=float(np.percentile(flat, 99)), max=float(E.max()),
        f_at_max=float(fk[imax[2]]), entry_at_max=ENTRY_NAMES[imax[1]],
    )


# ------------------------------------------------------------------ main loop


def run_config(name, backend, families, n, seed, T_days=None, gal_variants=None, lean=False, floor=False):
    cfg = copy.deepcopy(CONFIGS[name])
    if T_days is not None:
        cfg["T_days"] = T_days
        cfg["name"] = f"{name}_T{T_days:g}d"
    t0 = time.time()
    sm = make_backend(cfg, backend)
    xp = sm.xp
    inp = to_xp(make_inputs(sm, num_psds=n, seed=seed), xp)
    print(f"[{cfg['name']}/{backend}] built in {time.time() - t0:.1f}s, nf={sm.num_freqs}", flush=True)
    cpu = inp_cpu = None
    if floor:  # legacy CPU twin, same inputs: the legacy round-off floor for these draws
        cpu = make_backend(cfg, "cpu")
        inp_cpu = make_inputs(cpu, num_psds=n, seed=seed)
    res = dict(config=cfg["name"], backend=backend, nf=sm.num_freqs, N=len(sm.window_values), rows=[])

    # --- validation of the building blocks
    B = sw.get_instrument_bases(sm)
    raw = sw.get_instrument_bases(sm, convolve_window=False)
    v = max(float(xp.max(xp.abs(sw._fact_convolve(sm, r) - b)) / xp.max(xp.abs(b))) for r, b in zip(raw, B))
    res["moment0_helper_vs_legacy_band"] = v
    print(f"  _fact_convolve(moment 0) on raw instrument bases vs legacy-convolved bases: {v:.2e}")
    if cfg["galaxy"]:
        st = sw._fact_galaxy_setup(sm, "full", 0)
        res["R_mirror_mismatch"] = st["R_mismatch"]
        # exact galaxy via my real convolution of the mirrored galaxy row vs legacy
        p0 = draw_family(np.random.default_rng(1), sm, "prior", 2)
        gex = [a - b for a, b in zip(exact_entries(sm, p0), exact_entries(sm, p0, Amp_on=False))]
        G = 0.0
        for _, fabs, norm in sw._fact_alias_grids(sm):
            G = G + sw._fact_galaxy_shape(
                xp, fabs[None, :], *[xp.asarray(p0[k])[:, None] for k in ("Amp", "alpha", "f_1", "kn", "f_2")],
                split="full") * sw._fact_common_tdi(sm, fabs) * norm
        gconv = sw._fact_convolve(sm, G) * st["R00"][0]
        r = float(xp.max(xp.abs(gconv - gex[0].real)) / xp.max(xp.abs(gex[0])))
        res["gal_conv_mirror_vs_legacy_band"] = r
        print(f"  R mirror mismatch {st['R_mismatch']:.2e}; mirrored exact galaxy conv vs legacy {r:.2e}")

    for fam in families:
        rng = np.random.default_rng(seed + zlib.crc32(fam.encode()) % 1000)
        p = draw_family(rng, sm, fam, n)
        ex = exact_entries(sm, p)
        ll_ex = ll_external(sm, inp, p, ex)
        # sanity: same as compute_log_like on this instance
        llk = dict(inp["log_like_kwargs"])
        llk.update(Soms_in_all=xp.asarray(p["Soms"]), Sa_in_all=xp.asarray(p["Sa"]),
                   Amp_in_all=xp.asarray(p["Amp"]), alpha_in_all=xp.asarray(p["alpha"]),
                   f_1_in_all=xp.asarray(p["f_1"]), kn_in_all=xp.asarray(p["kn"]),
                   f_2_in_all=xp.asarray(p["f_2"]),
                   knots_position_all=None if p["kp"] is None else xp.asarray(p["kp"]),
                   knots_amplitude_all=None if p["ka"] is None else xp.asarray(p["ka"]))
        ll_ref = asnumpy(sm.compute_log_like(**llk, convolve_window=True))
        base = dict(family=fam, ll_abs=float(np.median(np.abs(ll_ex))),
                    ll_self_check=float(np.max(np.abs(ll_ref - ll_ex))))

        llb_ex = ll_bands(sm, inp, ex)
        base["ll_band_check"] = float(np.max(np.abs(llb_ex.sum(1) - ll_ex)))
        if cpu is not None:
            ex_c = exact_entries(cpu, p)
            ll_c = ll_external(cpu, inp_cpu, p, ex_c)
            erf = entry_errors(sm, ex, [xp.asarray(e) for e in ex_c])
            dll = np.abs(ll_c - ll_ex)
            dband = np.abs(ll_bands(sm, inp, [xp.asarray(e) for e in ex_c]) - llb_ex).max(axis=0)
            res["rows"].append(dict(base, variant="LEGACY_cpu_vs_gpu", **erf, dll_median=float(np.median(dll)),
                                    dll_max=float(dll.max()), dll_band_max=dband.tolist()))
            print(f"  {fam:10s} {'LEGACY cpu vs gpu':20s} med {erf['median']:.1e} p99 {erf['p99']:.1e} max {erf['max']:.1e}"
                  f" | white med {erf['w_median']:.1e} p99 {erf['w_p99']:.1e} max {erf['w_max']:.1e} @f={erf['w_f_at_max']:.3e}"
                  f" | dll med {np.median(dll):.2e} max {dll.max():.2e} | bands {np.array2string(dband, precision=2)}", flush=True)
            del ex_c
        inst_ex = exact_entries(sm, p, Amp_on=False) if cfg["galaxy"] else ex
        gal_ex = [a - b for a, b in zip(ex, inst_ex)] if cfg["galaxy"] else None
        variants = {}
        if sm.use_splines:
            inst_fa = fact_instrument(sm, p)
            variants["spline_fact"] = (inst_fa, gal_ex)
            # what fused_log_like(spline_weights=10^spl) computes today: legacy edge-padded
            # bases x w(f) (edge padding approximated too)
            wl = sw.factorized_spline_weights(sm, xp.asarray(p["kp"]), xp.asarray(p["ka"]))
            typ = xp.arange(sm.n_noise_par * 2) // sm.n_noise_par
            cw = sw.basis_coeffs(sm, xp.asarray(p["Soms"]), xp.asarray(p["Sa"]))[:, :, None] * \
                xp.tile(wl, (1, 1, sm.num_times))[:, typ, :]
            variants["spline_fact_fusedif"] = ([xp.einsum("nbf,bf->nf", cw, b) for b in sw.get_instrument_bases(sm)],
                                               gal_ex)
            if not lean:
                variants["spline_fact_o2"] = (fact_instrument(sm, p, order=2), gal_ex)
        if cfg["galaxy"]:
            for split, order in gal_variants:
                gc, rat = sw.factorized_galaxy(sm, p["Amp"], p["alpha"], p["f_1"], p["kn"], p["f_2"],
                                               order=order, split=split)
                variants[f"gal_{split}_o{order}"] = (inst_ex, gal_entries(sm, gc, rat))
            if sm.use_splines and not lean:
                gc, rat = sw.factorized_galaxy(sm, p["Amp"], p["alpha"], p["f_1"], p["kn"], p["f_2"],
                                               order=2, split="powerlaw")
                variants["both_powerlaw_o2"] = (inst_fa, gal_entries(sm, gc, rat))
        if not variants:  # gal0_spl0: instrument bases only (exact)
            variants["bases_only"] = (fact_instrument(sm, p), None)
        # end-to-end: agent 2's fused kernel fed with the factorized outputs vs the
        # compiled kernel on the same assembled entries
        if (sm.use_splines or cfg["galaxy"]) and not lean:
            gc = rat = None
            ie = fact_instrument(sm, p) if sm.use_splines else inst_ex
            ge = None
            if cfg["galaxy"]:
                gc, rat = sw.factorized_galaxy(sm, p["Amp"], p["alpha"], p["f_1"], p["kn"], p["f_2"],
                                               order=2, split="powerlaw")
                ge = gal_entries(sm, gc, rat)
            spl = (sw.factorized_spline_weights(sm, xp.asarray(p["kp"]), xp.asarray(p["ka"]))
                   if sm.use_splines else None)
            ap_ = ie if ge is None else [a + b for a, b in zip(ie, ge)]
            try:
                ll_f = asnumpy(sw.fused_log_like(sm, llk["data_in_all"], llk["data_index_all"],
                                             sw.basis_coeffs(sm, p["Soms"], p["Sa"]), gal_conv=gc,
                                             gal_ratios=rat, spline_weights=spl))
                base["fused_vs_compiled"] = float(np.max(np.abs(ll_f - ll_external(sm, inp, p, ap_))))
            except Exception as exc:  # agent 2's kernel is in flight
                print(f"  fused_log_like unavailable: {type(exc).__name__}: {str(exc)[:120]}")
                base["fused_vs_compiled"] = float("nan")
            print(f"  {fam}: fused_log_like(fact outputs) vs compiled kernel on same C: "
                  f"max|dll| {base['fused_vs_compiled']:.2e}", flush=True)
        for vname, (ie, ge) in variants.items():
            ap = ie if ge is None else [a + b for a, b in zip(ie, ge)]
            er = entry_errors(sm, ex, ap)
            ll_ap = ll_external(sm, inp, p, ap)
            dll = np.abs(ll_ap - ll_ex)
            dband = np.abs(ll_bands(sm, inp, ap) - llb_ex).max(axis=0)
            row = dict(base, variant=vname, **er, dll_median=float(np.median(dll)), dll_max=float(dll.max()),
                       dll_band_max=dband.tolist())
            res["rows"].append(row)
            print(f"  {fam:10s} {vname:20s} med {er['median']:.1e} p99 {er['p99']:.1e} max {er['max']:.1e}"
                  f" @f={er['f_at_max']:.3e} ({er['entry_at_max']}) | white med {er['w_median']:.1e} p99 {er['w_p99']:.1e}"
                  f" max {er['w_max']:.1e} @f={er['w_f_at_max']:.3e} rss {er['w_rss']:.1e} | dll med {row['dll_median']:.2e}"
                  f" max {row['dll_max']:.2e} | bands {np.array2string(dband, precision=2)} | selfchk {base['ll_self_check']:.1e}"
                  f" bandchk {base['ll_band_check']:.1e}", flush=True)
        del ex, inst_ex, gal_ex, variants
        if xp is not np:
            xp.get_default_memory_pool().free_all_blocks()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cuda12x")
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--families", nargs="+", default=None)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--T-days", type=float, default=None)
    ap.add_argument("--gal-variants", default="full:0,powerlaw:0,powerlaw:2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lean", action="store_true", help="fewer variants (2 yr memory)")
    ap.add_argument("--floor", action="store_true", help="also legacy CPU vs GPU on the same draws")
    a = ap.parse_args()
    gv = [(s.split(":")[0], int(s.split(":")[1])) for s in a.gal_variants.split(",")]
    allres = []
    for c in a.configs:
        cfg = CONFIGS[c]
        fams = a.families
        if fams is None:
            fams = []
            if cfg["galaxy"]:
                fams += ["prior", "sharp"]
            if cfg["use_splines"]:
                fams = [f"{g}_{s}" for g in (fams or ["prior"]) for s in ("spl01", "spl03", "spl05")]
            if not fams:
                fams = ["prior"]
        allres.append(run_config(c, a.backend, fams, a.n, a.seed, a.T_days, gv, a.lean, a.floor))
        with open(a.out, "w") as fh:
            json.dump(allres, fh, indent=1)


if __name__ == "__main__":
    main()
