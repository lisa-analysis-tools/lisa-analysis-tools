"""Agent 3: exact galaxy window convolution (``convolution_helpers.galaxy_convolved_rows``).

    CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/window_conv/test_agent3_galaxy.py [cpu|cuda12x ...] [--e2e]\n\n(``--e2e``: only the end-to-end ll check 7 -- fused_log_like + galaxy rows vs legacy.)

Checks (per backend, every galaxy config):
1. prefix evaluation of the compiled kernel == full-band evaluation (bitwise); the compiled
   ``get_galactic_row_wrap`` XX row == kernel XX row (band-normalised <= 1e-14);
2. synthetic-row oracle: ``_gal_convolve`` vs legacy ``_convolve_matrix_components`` on a
   random real row (non-zero at column 0, zero above ``n_sup``), short circle forced;
3. galaxy-only legacy ``_compute_matrix_elements(Soms=Sa=0, convolve_window=True)`` vs
   ``gal_ratios * gal_conv`` for all 6 entries, ``galaxy_fmax=None`` (full band) and a
   forced short circle (``galaxy_fmax=0.045``); band_reldiff <= 1e-14;
4. total: legacy instrument-only convolution + ours vs legacy total convolution
   (normalised by max|total| per entry);
5. tail bound: ``max|G[n_sup:]| <= bound`` on the full-band evaluation;
6. fallback: a walker with a high knee triggers the full-band plan (warning, exact).
"""

from __future__ import annotations

import sys
import warnings

import numpy as np

import _use_worktree  # noqa: F401
from compare import band_reldiff, golden_inputs
from fixtures import CONFIGS, make_backend

from lisatools import convolution_helpers as sw

GAL_CONFIGS = [n for n, c in CONFIGS.items() if c["galaxy"]]
TOL = 1e-14
FMAX_SHORT = 0.045  # fixture prior: kn <= 6.3e-3, f_2 <= 1.6e-3 -> tail ~exp(-48)


def host(x):
    return x.get() if hasattr(x, "get") else np.asarray(x)


def gal_params(sm, n=5, seed=1):
    rng = np.random.default_rng(seed)
    p = dict(
        Amp=10 ** rng.uniform(-44.2, -43.2, n),
        alpha=rng.uniform(1.2, 2.5, n),
        f_1=10 ** rng.uniform(-3.0, -2.4, n),
        kn=10 ** rng.uniform(-2.8, -2.2, n),
        f_2=10 ** rng.uniform(-3.4, -2.8, n),
    )
    return {k: sm.xp.asarray(v) for k, v in p.items()}


def legacy_entries(sm, p, Soms, Sa, convolve=True):
    """Legacy 6 entries in GAL_RATIO_ORDER, each (num_psds, nf) real (cross: real part)."""
    xp = sm.xp
    n = p["Amp"].shape[0]
    kp = ka = None
    if sm.use_splines:
        f = host(sm.f_arr)
        f = f[f > 0]
        kp = xp.asarray(np.broadcast_to(np.linspace(np.log10(f[0]) - .1, np.log10(f[-1]) + .1, 8), (2, n, 8)).copy())
        ka = xp.zeros_like(kp)
    c00, c11, c22, c01, c02, c12 = sm._compute_matrix_elements(
        sm.f_arr, Soms, Sa, p["Amp"], p["alpha"], p["f_1"], p["kn"], p["f_2"], kp, ka,
        convolve_window=convolve,
    )
    by = {"00": c00, "11": c11, "22": c22, "01": c01, "02": c02, "12": c12}
    return [by[k] for k in sw.GAL_RATIO_ORDER]


def ours_entries(sm, p, galaxy_fmax):
    st = sw.get_galaxy_setup(sm, galaxy_fmax=galaxy_fmax)
    gal_conv, ratios = sw.galaxy_convolved_rows(sm, **p, setup=st)
    gal_conv = host(gal_conv)[:, 0, :]  # num_times == 1
    nf = int(sm.num_freqs)
    full = np.zeros((gal_conv.shape[0], nf))
    full[:, : gal_conv.shape[1]] = gal_conv
    return [host(ratios)[0, k] * full for k in range(6)], st, gal_conv.shape[1]


def zeros_amp(sm, n):
    return sm.xp.zeros((n, sm.n_noise_par)) if sm.n_noise_par > 1 else sm.xp.zeros(n)


def check(name, val, tol, results):
    ok = val <= tol
    results.append((name, val, ok))
    print(f"  {'PASS' if ok else 'FAIL'} {name}: {val:.3e} (tol {tol:g})")


def run(backend_name):
    results = []
    for cfg_name in GAL_CONFIGS:
        sm = make_backend(cfg_name, backend_name)
        xp = sm.xp
        print(f"[{cfg_name}/{backend_name}] nf={sm.num_freqs} slice={sm.active_slice} N={len(sm.window_values)}")
        p = gal_params(sm)
        n = p["Amp"].shape[0]

        # 1. prefix evaluation == full-band evaluation (bitwise)
        f = xp.asarray(sm.f_arr)
        pl = [xp.ascontiguousarray(p[k]) for k in ("Amp", "alpha", "f_1", "kn", "f_2")]
        full = sw._gal_eval(sm, f, pl, sm.time_indices)
        m = max(2, int(sm.num_freqs) // 3)
        pre = sw._gal_eval(sm, f[:m], pl, sm.time_indices)
        d = max(float(np.nanmax(np.abs(host(a)[..., :m] - host(b)))) for a, b in zip(full, pre))
        check("prefix eval bitwise", d, 0.0, results)

        # 1b. compiled XX row (get_galactic_row_wrap) vs the covariance kernel, whole band
        mine = host(sw._gal_row(sm, int(sm.num_freqs), pl))
        ref = host(full[0])[:, 0, :]
        fin = np.isfinite(ref)
        check("galaxy row finite pattern == kernel", float(np.any(fin != np.isfinite(mine))), 0.0, results)
        check("galaxy row vs kernel (band)", band_reldiff(np.where(fin, mine, 0), np.where(fin, ref, 0))[0], 1e-14, results)

        # 2. synthetic-row oracle vs legacy _convolve_matrix_components (short circle forced)
        st = sw.get_galaxy_setup(sm, galaxy_fmax=None)
        nsup = max(2, int(sm.num_freqs) // 4)
        for label, plan in (
            ("full plan", st["plan"]),
            ("short plan", sw._gal_plan(sm, nsup, st["K"], st["lo"], st["n_active"], st["N"])),
            # whole band (constant G_hi term) on a short circle: small K forces M < N
            ("full-band short plan K=1500", sw._gal_plan(sm, st["n_active"], 1500, st["lo"], st["n_active"], st["N"])),
        ):
            rng = np.random.default_rng(3)
            G = np.zeros((3, int(sm.num_freqs)))
            G[:, : plan["n_sup"]] = rng.standard_normal((3, plan["n_sup"])) + 2.0  # non-zero at both edges
            Gx = xp.asarray(G)
            mine = host(sw._gal_convolve(sm, st, plan, xp.ascontiguousarray(Gx[:, : plan["n_sup"]])))
            z = xp.zeros_like(Gx)
            leg = host(sm._convolve_matrix_components(Gx, z, z, z.astype(complex), z.astype(complex), z.astype(complex), num_psds=3)[0])
            ref = leg[:, : plan["n_gal"]]
            r, _ = band_reldiff(mine, ref)
            beyond = float(np.max(np.abs(leg[:, plan["n_gal"]:]), initial=0.0)) / float(np.max(np.abs(leg)))
            print(f"    {label}: n_sup={plan['n_sup']} M={plan['M']} (N={st['N']}) K={st['K']} n_gal={plan['n_gal']}")
            kh = host(sm.window_kernel)
            Kp = st["K"] if "K=" not in label else 1500
            tail = float(2 * kh[Kp + 1 : len(kh) // 2 + 1].sum() / kh.sum()) * float(np.max(np.abs(G))) / float(np.max(np.abs(ref)))
            check(f"synthetic oracle {label} (M={plan['M']})", r, max(TOL, 10 * tail), results)
            check(f"synthetic oracle {label}: legacy beyond n_gal (round-off)", beyond, 1e-15, results)

        # 3. galaxy-only legacy convolution vs ratios * rows
        leg = [host(e) for e in legacy_entries(sm, p, zeros_amp(sm, n), zeros_amp(sm, n))]
        for fmax in (None, FMAX_SHORT):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                ours, st, n_gal = ours_entries(sm, p, fmax)
            plan = st["plan"]
            print(f"    galaxy_fmax={fmax}: n_sup={plan['n_sup']} M={plan['M']} K={st['K']} n_gal={n_gal} "
                  f"guard={st['guard']} ratio_spread={st['ratio_spread']:.1e}")
            r = max(band_reldiff(o, np.real(l))[0] for o, l in zip(ours, leg))
            check(f"galaxy-only vs legacy (fmax={fmax})", r, TOL, results)
            imag = max(float(np.max(np.abs(np.imag(l)))) / float(np.max(np.abs(l))) for l in leg[1:3] + leg[4:5])
            check("legacy galaxy cross-term imag / max", imag, 1e-15, results)

        # 4. total: instrument-only + ours vs total (normalised by max|total|)
        g = golden_inputs(cfg_name, xp)["log_like_kwargs"]
        S, A = g["Soms_in_all"][:n], g["Sa_in_all"][:n]
        tot = [host(e) for e in legacy_entries(sm, p, S, A)]
        ins = [host(e) for e in legacy_entries(sm, {k: xp.zeros(n) for k in p}, S, A)]
        ours, _, _ = ours_entries(sm, p, FMAX_SHORT)
        r = max(band_reldiff(i + o, t)[0] for i, o, t in zip(ins, ours, tot))
        check("instrument + ours vs legacy total", r, TOL, results)

        # 5. tail bound validity on the full-band evaluation
        st = sw.get_galaxy_setup(sm, galaxy_fmax=FMAX_SHORT)
        if st["bound"] is not None:
            G = host(full[0])[:, 0, :]
            nsup = st["plan"]["n_sup"]
            tail = np.max(np.abs(G[:, nsup:]), axis=-1)
            bound = host(sw._gal_tail_bound(sm, st["bound"], pl))
            gmax = np.nanmax(np.abs(G[:, :nsup]), axis=-1)
            ratio = float(np.max(tail / bound))
            check("true tail / analytic bound", ratio, 1.0, results)
            print(f"    bound/gmax max {float(np.max(bound / gmax)):.2e}, true tail/gmax max {float(np.max(tail / gmax)):.2e}")

            # 6. fallback: a walker whose knee sits above galaxy_fmax
            p2 = {k: v.copy() for k, v in p.items()}
            p2["kn"][1] = 0.06
            p2["f_2"][1] = 5e-3
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                ours2, _, n_gal2 = ours_entries(sm, p2, FMAX_SHORT)
            check("fallback warned", 0.0 if any("full band" in str(x.message) for x in w) else 1.0, 0.0, results)
            check("fallback n_gal == n_active", float(n_gal2 != sm.num_freqs), 0.0, results)
            leg2 = [host(e) for e in legacy_entries(sm, p2, zeros_amp(sm, n), zeros_amp(sm, n))]
            r = max(band_reldiff(o, np.real(l))[0] for o, l in zip(ours2, leg2))
            check("fallback galaxy-only vs legacy", r, TOL, results)
        del sm
    return results


def run_e2e(backend_name):
    """End-to-end ll (agent 2's fused_log_like + our galaxy rows) vs legacy on the SAME
    backend instance and vs the golden (spline-free galaxy configs)."""
    from compare import load_golden, reldiff

    results = []
    for cfg_name in [n for n in GAL_CONFIGS if not CONFIGS[n]["use_splines"]]:
        sm = make_backend(cfg_name, backend_name)
        inp = golden_inputs(cfg_name, sm.xp)["log_like_kwargs"]
        legacy = host(sm.compute_log_like(**inp, convolve_window=True))
        coeffs = sw.basis_coeffs(sm, inp["Soms_in_all"], inp["Sa_in_all"])
        gp = [inp[k] for k in ("Amp_in_all", "alpha_in_all", "f_1_in_all", "kn_in_all", "f_2_in_all")]
        z = zeros_amp(sm, gp[0].shape[0])
        leg_gal = sm._compute_matrix_elements(sm.f_arr, z, z, *gp, convolve_window=True)[0]  # XX galaxy-only
        for fmax in (None, FMAX_SHORT):
            st = sw.get_galaxy_setup(sm, galaxy_fmax=fmax)
            gal_conv, ratios = sw.galaxy_convolved_rows(
                sm, inp["Amp_in_all"], inp["alpha_in_all"], inp["f_1_in_all"], inp["kn_in_all"],
                inp["f_2_in_all"], setup=st,
            )
            ll = host(sw.fused_log_like(sm, inp["data_in_all"], inp["data_index_all"], coeffs,
                                        gal_conv=gal_conv, gal_ratios=ratios))
            # same fused instrument path, galaxy rows from the legacy galaxy-only convolution:
            # isolates the galaxy (the fused-vs-legacy ll gap is instrument/conditioning)
            ll_lg = host(sw.fused_log_like(sm, inp["data_in_all"], inp["data_index_all"], coeffs,
                                           gal_conv=leg_gal[:, None, :], gal_ratios=ratios))
            r_gal = reldiff(ll, ll_lg)[0]
            r_leg = reldiff(ll, legacy)[0]
            r_gold = reldiff(ll, load_golden(cfg_name, backend_name)["ll_conv"])[0]
            print(f"[{cfg_name}/{backend_name}] fmax={fmax}: ll(ours) vs ll(legacy galaxy rows) {r_gal:.3e}; "
                  f"vs legacy compute_log_like (same instance) {r_leg:.3e} (|dll| {np.max(np.abs(ll - legacy)):.3e}), "
                  f"vs golden {r_gold:.3e}")
            check(f"e2e ll, ours vs legacy galaxy rows {cfg_name} fmax={fmax}", r_gal, 1e-12, results)
        del sm
    return results


if __name__ == "__main__":
    backends = [a for a in sys.argv[1:] if a != "--e2e"] or ["cpu", "cuda12x"]
    allres = []
    for b in backends:
        if "--e2e" not in sys.argv:
            allres += [(b,) + r for r in run(b)]
        allres += [(b,) + r for r in run_e2e(b)]
    bad = [r for r in allres if not r[3]]
    print(f"\n{len(allres) - len(bad)}/{len(allres)} checks passed")
    for r in bad:
        print("FAILED:", r)
    sys.exit(1 if bad else 0)
