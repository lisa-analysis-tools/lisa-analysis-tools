"""GB / VGB speed + accuracy test (one script, three steps), mirroring the SOBBH / EMRI / MBH
speed-durations setups. Driver: ``gb_speed_durations.sh``.

Steps (``--step``):

* ``speed``  -- per engine, warm wall time of one scoring call vs rows per call on the run grid
  (Nf 1440 / dt 2.5 at ``--days``; ``--laptop`` Nf 180 / dt 20). Rows are jittered copies of
  mojito-like GB / VGB parameters, each slot's slab holding its reference source. Sig-het also
  reports its per-reference setup (``setup_in_model``) cost. One line per (engine, rows) to
  ``speed.jsonl``.
* ``gate``   -- accuracy vs a DENSE truth (GBTDIonTheFly at dt 20 -> Tukey inside the crop ->
  TD->WDM, the 3600-s layer), on the Nf 180 / dt 20 grid. Per synthetic source (f0 ladder x
  inclinations, full XYZ 3x3 noise, SNR ``--rho``):
    - template level at the reference: noise-weighted mm and norm ratio of chunked / lookup;
    - delta-vs-delta at posterior-scale steps (``--step-scales``): eps = |D_engine - D_truth|,
      T = |D_truth|, tier pass = eps <= max(0.1, T/100), for every engine;
  one line per (source, candidate) to ``gate.jsonl`` plus a summary line.
* ``mojito`` -- the mojito data: the top-frequency GBs (whole-galaxy brick) and VGBs (VGB brick),
  catalogue parameters, window ``START_OFFSET_S`` after the brick start, in-slab catalogue
  neighbours subtracted (chunked fill), scirdv1 + fitted tanh foreground. Per (class, source,
  days) and engine: snr (optimal, the engine's own h_h), snr_det, logL, mm vs data, and
  mm_vs_production (vs chunked). One line per (class, source, days) to ``mojito.jsonl``.

Field names follow the SOBBH / EMRI scripts where a quantity is shared (rows, nf, nt, dt, edge,
foreground, tobs_s, backend, snr, data.snr, data.snr_det, mm_vs_production, days).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _gb_testbox as tb  # noqa: E402

import gbgpu  # noqa: E402,F401  (registers the gbgpu backends)


def _xp(backend):
    if backend == "cpu":
        return np
    import cupy

    return cupy


def _sync(xp):
    if xp is not np:
        xp.cuda.runtime.deviceSynchronize()


def _orbits(backend):
    from lisatools.detector import DefaultOrbits

    return DefaultOrbits(force_backend=backend, frame="icrs")


def _base_sources(n, rng):
    """Mojito-like GB / VGB parameter rows: high-f galaxy GBs and mid-f VGB-like sources,
    generic and edge-on inclinations (``n`` rows, cycled)."""
    f0 = np.array([20.38e-3, 16.72e-3, 15.65e-3, 6.22e-3, 3.51e-3, 2.61e-3, 1.4e-3, 0.62e-3])
    cosi = np.array([0.8, 0.3, 0.05, 0.6, 0.9, 0.02, 0.4, 0.7])
    k = np.arange(n) % f0.size
    return np.column_stack([
        np.full(n, 1e-22), f0[k], 1e-18 * (f0[k] / 1e-3) ** (11 / 3), np.zeros(n),
        rng.uniform(0, 2 * np.pi, n), np.arccos(cosi[k]), rng.uniform(0, np.pi, n),
        rng.uniform(0, 2 * np.pi, n), np.arcsin(rng.uniform(-1, 1, n))])


def _jitter(p, rng, scale):
    """Posterior-scale steps (x ``scale``) around rows ``p``."""
    n = len(p)
    q = p.copy()
    q[:, 0] *= np.exp(0.05 * scale * rng.standard_normal(n))
    q[:, 4] += 0.1 * scale * rng.standard_normal(n)
    q[:, 5] = np.clip(q[:, 5] + 0.03 * scale * rng.standard_normal(n), 1e-3, np.pi - 1e-3)
    q[:, 6] += 0.03 * scale * rng.standard_normal(n)
    q[:, 7] += 0.01 * scale * rng.standard_normal(n)
    q[:, 8] = np.clip(q[:, 8] + 0.01 * scale * rng.standard_normal(n), -1.5, 1.5)
    return q


def _timed(fn, xp, reps):
    t = time.perf_counter()
    fn()
    _sync(xp)
    first = time.perf_counter() - t
    warm = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        _sync(xp)
        warm.append(time.perf_counter() - t)
    return first, float(np.median(warm))


def _write(path, rec):
    with open(path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


# ===================================================================== speed
def step_speed(a):
    xp = _xp(a.backend)
    rng = np.random.default_rng(a.seed)
    nf, nt, dt = tb.grid_args(a.days, a.laptop)
    t0 = 0.5 * 365.25 * 86400.0 + tb.REF
    wdm = tb.run_box(nf, nt, dt, t0, edge=a.edge, force_backend=a.backend)
    orbits = _orbits(a.backend)
    engines = tb.build_engines(wdm, orbits, names=a.engines, backend=a.backend, table=a.table)
    sens, noise_label = tb.noise(wdm, wdm.Tobs, a.foreground)
    T = int(wdm.ind_max_t - wdm.ind_min_t + 1)
    for rows in a.rows:
        n_slots = min(rows, a.max_slots)
        p0 = _base_sources(n_slots, rng)
        slab_lo = tb.slab_lo_for(p0[:, 1], wdm)
        invc = tb.slab_invc(sens, wdm, slab_lo)
        di = np.arange(rows) % n_slots
        cand = _jitter(p0[di], rng, a.scale)
        holder = tb.SlabHolder(np.zeros((n_slots, 3, tb.SLAB_W, T)), invc, slab_lo, xp=xp)
        ch = engines.get("chunked") or tb.build_engines(wdm, orbits, names=["chunked"],
                                                        backend=a.backend)["chunked"]
        NV = np.full(n_slots, 1024)
        ch.fill_template(holder, p0, np.arange(n_slots), NV, factor=+1, waveform_kwargs={},
                         band_slab_Nf=tb.SLAB_W, slab_min_f=slab_lo)
        NVr = np.full(rows, 1024)
        rec_base = dict(rows=int(rows), slots=int(n_slots), backend=a.backend, nf=nf, nt=nt, dt=dt,
                        edge=a.edge, foreground=a.foreground, noise=noise_label,
                        tobs_s=float(wdm.Tobs), days=float(a.days), tag=a.tag, scale=a.scale)
        ref_ll = None
        for name, eng in engines.items():
            rec = dict(rec_base, engine=name)
            if name == "lookup":
                dslab = xp.asarray(holder.linear_data_arr[0]).reshape(n_slots, 3, tb.SLAB_W, T)[di]
                islab = xp.asarray(invc)[di]

                def call():
                    tpl = eng.slab(cand, slab_lo[di], tb.SLAB_W)
                    return tb.lookup_inner(tpl, dslab, islab)

                first, warm = _timed(call, xp, a.reps)
                d_h, h_h = call()
                ll = d_h - 0.5 * h_h
                ll = np.asarray(ll.get() if hasattr(ll, "get") else ll, dtype=float).ravel()
            else:
                setup_s = None
                if name.startswith("sighet_"):
                    t = time.perf_counter()
                    eng.setup_in_model(holder, p0, np.arange(n_slots))
                    _sync(xp)
                    setup_s = time.perf_counter() - t

                def call():
                    return eng.get_ll(holder, cand, data_index=di, noise_index=di, N_vals=NVr,
                                      waveform_kwargs={})

                first, warm = _timed(call, xp, a.reps)
                res = call()
                ll = np.asarray(res.get() if hasattr(res, "get") else res, dtype=float).ravel()
                if name.startswith("sighet_"):
                    eng.clear_in_model()
                rec["setup_s"] = setup_s
                rec["setup_s_per_ref"] = None if setup_s is None else setup_s / n_slots
            rec.update(first_s=first, warm_s=warm, us_per_row=1e6 * warm / rows)
            if name == "chunked":
                ref_ll = ll
            elif ref_ll is not None:
                d = np.abs(ll - ref_ll)
                rec.update(max_abs_dll_vs_chunked=float(d.max()),
                           median_abs_dll_vs_chunked=float(np.median(d)))
            _write(a.out, rec)


# ===================================================================== gate
def _gate_cases(a, rng, wdm):
    """``(params (n, 9) at unit-ish amplitude, rho (n,) target SNR or nan = keep, labels)``.

    synthetic: ``--gate-f0`` x ``--gate-cosi`` x ``--gate-skies`` random skies / phases, each at
    every SNR of ``--rho-list``; catalogue: the ``--gate-catalogue-top`` highest-SNR galaxy GBs and
    ``--gate-vgb-top`` VGBs (exact SNR on this run's box) at their catalogue amplitude."""
    rows, rho, lab = [], [], []
    if "synthetic" in a.gate_sources:
        for f in np.array(a.gate_f0) * 1e-3:
            for c in a.gate_cosi:
                for _ in range(a.gate_skies):
                    base = [1e-22, f, 1e-18 * (f / 1e-3) ** (11 / 3), 0.0,
                            rng.uniform(0, 2 * np.pi), np.arccos(c), rng.uniform(0, np.pi),
                            rng.uniform(0, 2 * np.pi), np.arcsin(rng.uniform(-1, 1))]
                    for r in a.rho_list:
                        rows.append(list(base))
                        rho.append(float(r))
                        lab.append(dict(kind="synthetic", id=f"syn_{f*1e3:.2f}mHz_c{c:+.2f}"))
    if "catalogue" in a.gate_sources:
        import gb_catalogue_snr as cs

        for kind, top in (("GB", a.gate_catalogue_top), ("VGB", a.gate_vgb_top)):
            if top <= 0:
                continue
            cat = a.catalogue if kind == "GB" else a.vgb_catalogue
            if cat is None:
                brick = tb.find_brick(kind, a.l1_dir)
                if brick is None:
                    print(f"[gate] no {kind} catalogue -- skipped", flush=True)
                    continue
                cat = tb.find_catalogue(kind, brick)
            snr_p, _, _ = cs.proxy_snr(cat, wdm.Tobs)
            order = np.argsort(snr_p)[::-1][: max(5 * top, 50)]
            params, ids, _ = tb.catalogue_params(cat, idx=order)
            snr_e = cs.exact_snr(params, a.days)
            best = np.argsort(snr_e)[::-1][:top]
            for k in best:
                rows.append(list(params[k]))
                rho.append(np.nan)
                lab.append(dict(kind=kind.lower(), id=ids[k]))
    return np.array(rows), np.array(rho), lab


def step_gate(a):
    from scipy.signal.windows import tukey
    from lisatools.domains import TDSettings, TDSignal, WDMSettings
    from lisatools.response.tdionfly import GBTDIonTheFly

    rng = np.random.default_rng(a.seed)
    nf, nt, dt = 180, int(round(a.days * 24)), 20.0
    t0 = 0.5 * 365.25 * 86400.0 + tb.REF
    wdm = tb.run_box(nf, nt, dt, t0, edge=a.edge)
    wdm_full = WDMSettings(nf, nt, dt, t0=t0, min_freq=1e-4, max_freq=0.5 / dt,
                           force_backend="cpu")
    orbits = _orbits("cpu")
    engines = tb.build_engines(wdm, orbits, names=a.engines, backend="cpu", table=a.table)
    sens, noise_label = tb.noise(wdm, wdm.Tobs, a.foreground)
    T = int(wdm.ind_max_t - wdm.ind_min_t + 1)
    n_lo = int(wdm.ind_min_t)
    p0, rho, lab = _gate_cases(a, rng, wdm)
    n = len(p0)
    slab_lo = tb.slab_lo_for(p0[:, 1], wdm)
    invc = tb.slab_invc(sens, wdm, slab_lo)
    N2 = nf * nt
    win = tukey(N2, 2.0 * min(20, a.edge - 10) / nt)
    gen_d = GBTDIonTheFly(t0 + np.arange(N2) * dt, nt * wdm.layer_dt, tb.REF, 1.0, 1,
                          tdi_config="2nd generation", orbits=orbits, force_backend="cpu")

    def dense(i, p):
        out = gen_d(*[np.array([v]) for v in p], convert_to_ra_dec=False)
        y = np.asarray(out.tdi_amp)[0] * np.cos(np.asarray(out.tdi_phase)[0]
                                                + np.asarray(out.phase_ref)[0][None, :])
        arr = np.asarray(TDSignal(y * win[None, :], TDSettings(N2, dt, force_backend="cpu"))
                         .transform(wdm_full).arr)
        r0 = slab_lo[i] - int(wdm_full.ind_min_f)
        return arr[:, r0:r0 + tb.SLAB_W, n_lo:n_lo + T]

    def ip(i, x, y):
        return float(np.einsum("cwt,cdwt,dwt->", x, invc[i], y))

    # references: dense truth once per distinct base row, amplitude set to the target SNR
    d0, cache = [], {}
    snr0 = np.zeros(n)
    for i in range(n):
        key = tuple(np.round(p0[i, 1:], 12))
        if key not in cache:
            cache[key] = (p0[i, 0], dense(i, p0[i]))
        a_ref, s = cache[key]
        s = s * (p0[i, 0] / a_ref)
        snr_i = np.sqrt(ip(i, s, s))
        if np.isfinite(rho[i]):
            sc = rho[i] / snr_i
            p0[i, 0] *= sc
            s = s * sc
            snr_i = rho[i]
        snr0[i] = snr_i
        d0.append(s)
    d0 = np.array(d0)
    holder = tb.SlabHolder(d0, invc, slab_lo)
    idx = np.arange(n)
    NV = np.full(n, 1024)
    # posterior-scale steps shrink as 1/SNR: scale x 100 / snr keeps T comparable across SNR
    stepk = 100.0 / snr0
    cands = []
    for s in a.step_scales:
        for _ in range(a.n_cand):
            q = p0.copy()
            for i in range(n):
                q[i] = _jitter(p0[i:i + 1], rng, s * stepk[i])[0]
            cands.append((s, q))

    def ll_engine(name, eng, P):
        if name == "lookup":
            tpl = eng.slab(P, slab_lo, tb.SLAB_W)
            d_h, h_h = tb.lookup_inner(tpl, d0, invc)
            return np.asarray(d_h - 0.5 * h_h), tpl
        out = eng.get_ll(holder, P, data_index=idx, noise_index=idx, N_vals=NV,
                         waveform_kwargs={})
        return np.asarray(out, dtype=float).ravel()[:n], None

    D, anchor, tmpl, t_tr = {}, {}, {}, {}
    for name, eng in engines.items():
        t = time.perf_counter()
        if name.startswith("sighet_"):
            eng.setup_in_model(holder, p0, idx)
        l0, tpl0 = ll_engine(name, eng, p0)
        D[name] = [ll_engine(name, eng, p1)[0] - l0 for _, p1 in cands]
        if name.startswith("sighet_"):
            eng.clear_in_model()
        anchor[name], tmpl[name], t_tr[name] = l0, tpl0, time.perf_counter() - t
    if "chunked" in engines:
        z = tb.SlabHolder(np.zeros_like(d0), invc, slab_lo)
        engines["chunked"].fill_template(z, p0, idx, NV, factor=+1, waveform_kwargs={},
                                         band_slab_Nf=tb.SLAB_W, slab_min_f=slab_lo)
        tmpl["chunked"] = z.linear_data_arr[0].reshape(n, 3, tb.SLAB_W, T)
    ll_true0 = np.array([0.5 * ip(i, d0[i], d0[i]) for i in range(n)])
    head = dict(days=float(a.days), nf=nf, nt=nt, dt=dt, edge=a.edge, foreground=a.foreground,
                noise=noise_label)
    rec_rows = []
    for k, (s, p1) in enumerate(cands):
        for i in range(n):
            h1 = dense(i, p1[i])
            Dt = ip(i, d0[i], h1) - 0.5 * ip(i, h1, h1) - ll_true0[i]
            rec = dict(head, kind="delta", source=lab[i]["kind"], id=lab[i]["id"],
                       snr=float(snr0[i]), scale=float(s), f0=float(p0[i, 1]),
                       cosi=float(np.cos(p0[i, 5])), T=float(abs(Dt)))
            for name in engines:
                e = float(abs(D[name][k][i] - Dt))
                rec[name] = dict(eps=e, eps_over_T=e / max(abs(Dt), 1e-30),
                                 tier_pass=bool(e <= max(0.1, abs(Dt) / 100.0)),
                                 anchor=float(anchor[name][i] - ll_true0[i]))
            rec_rows.append(rec)
            _write(a.out, rec)
    for i in range(n):
        rec = dict(head, kind="template", source=lab[i]["kind"], id=lab[i]["id"],
                   f0=float(p0[i, 1]), cosi=float(np.cos(p0[i, 5])), snr=float(snr0[i]))
        for name, s_ in tmpl.items():
            if s_ is None:
                continue
            ab, aa, bb = ip(i, s_[i], d0[i]), ip(i, s_[i], s_[i]), ip(i, d0[i], d0[i])
            rec[name] = dict(mm=1 - ab / np.sqrt(aa * bb), ratio=float(np.sqrt(aa / bb)))
        _write(a.out, rec)
    summ = {}
    for name in engines:
        e = np.array([r[name]["eps"] for r in rec_rows])
        Tt = np.array([r["T"] for r in rec_rows])
        ok = np.array([r[name]["tier_pass"] for r in rec_rows])
        summ[name] = dict(n=int(e.size), tier_pass_frac=float(ok.mean()),
                          median_eps=float(np.median(e)), p90_eps=float(np.percentile(e, 90)),
                          max_eps=float(e.max()), max_eps_over_T=float((e / Tt).max()),
                          max_abs_anchor=float(np.abs(anchor[name] - ll_true0).max()),
                          t_total_s=t_tr[name])
    _write(a.out, dict(head, kind="summary", engines=summ, n_sources=int(n),
                       sources=sorted({l["kind"] for l in lab})))


# ===================================================================== mojito
def step_mojito(a):
    from lisatools.domains import TDSettings, TDSignal

    nf, nt, dt = tb.grid_args(a.days, a.laptop)
    for kind, topn in (("VGB", a.vgb_top), ("GB", a.gb_top)):
        if topn <= 0:
            continue
        brick = tb.find_brick(kind, a.l1_dir)
        if brick is None:
            print(f"[mojito] no {kind} brick found -- skipped", flush=True)
            continue
        cat = tb.find_catalogue(kind, brick, a.catalogue if kind == "GB" else a.vgb_catalogue)
        p_top, ids, _ = tb.catalogue_params(cat, top_f=topn)
        xyz, t0w, dt_b = tb.load_l1_window(brick, nf * nt, dt, start_offset=a.start_offset)
        wdm = tb.run_box(nf, nt, dt, t0w, edge=a.edge, force_backend="cpu")
        orbits = tb.l1_orbits(brick, t0w, t0w + nf * nt * dt)
        engines = tb.build_engines(wdm, orbits, names=a.engines, backend="cpu", table=a.table)
        sens, noise_label = tb.noise(wdm, wdm.Tobs, a.foreground)
        win_alpha = 2.0 * min(20, a.edge - 10) / nt
        from scipy.signal.windows import tukey

        dwdm = np.asarray(TDSignal(xyz * tukey(xyz.shape[1], win_alpha)[None, :],
                                   TDSettings(xyz.shape[1], dt, force_backend="cpu"))
                          .transform(wdm).arr)                      # active band
        del xyz
        T = int(wdm.ind_max_t - wdm.ind_min_t + 1)
        if dwdm.shape[-1] != T:          # active time slice
            dwdm = dwdm[..., int(wdm.ind_min_t):int(wdm.ind_max_t) + 1]
        for j in range(len(p_top)):
            p = p_top[j:j + 1]
            lo = tb.slab_lo_for(p[:, 1], wdm)
            invc = tb.slab_invc(sens, wdm, lo)
            r0 = int(lo[0] - wdm.ind_min_f)
            d = dwdm[:, r0:r0 + tb.SLAB_W, :][None].copy()
            # subtract in-slab catalogue neighbours (chunked fill at their catalogue params)
            f_lo, f_hi = (lo[0] - 1) * wdm.layer_df, (lo[0] + tb.SLAB_W + 1) * wdm.layer_df
            nb, nb_ids, _ = tb.catalogue_params(cat, fmin=f_lo, fmax=f_hi)
            keep = np.array([i != ids[j] for i in nb_ids], dtype=bool)
            nb = nb[keep]
            holder = tb.SlabHolder(d, invc, lo)
            if len(nb):
                engines["chunked"].fill_template(
                    holder, nb, np.zeros(len(nb), dtype=np.int64), np.full(len(nb), 1024),
                    factor=-1, waveform_kwargs={}, band_slab_Nf=tb.SLAB_W, slab_min_f=lo)
            dres = holder.linear_data_arr[0].reshape(1, 3, tb.SLAB_W, T)
            d_d = float(np.einsum("ncwt,ncdwt,ndwt->", dres, invc, dres))
            rec = dict(source=kind.lower(), src=ids[j], brick=os.path.basename(brick),
                       days=float(a.days), nt=nt, nf=nf, dt=dt, tobs_s=float(wdm.Tobs),
                       edge=a.edge, start_offset=a.start_offset, foreground=a.foreground,
                       noise=noise_label, f0=float(p[0, 1]), neighbours_subtracted=int(len(nb)),
                       data_snr=float(np.sqrt(d_d)))
            hh_prod = dh_prod = None
            for name, eng in engines.items():
                t = time.perf_counter()
                if name == "lookup":
                    tpl = eng.slab(p, lo, tb.SLAB_W)
                    d_h, h_h = (float(np.asarray(x).ravel()[0]) for x in tb.lookup_inner(tpl, dres, invc))
                else:
                    if name.startswith("sighet_"):
                        eng.setup_in_model(holder, p, np.zeros(1, dtype=np.int64))
                    eng.get_ll(holder, p, data_index=np.zeros(1, dtype=np.int64),
                               noise_index=np.zeros(1, dtype=np.int64), N_vals=np.full(1, 1024),
                               waveform_kwargs={})
                    d_h = float(np.asarray(eng.d_h_out).real.ravel()[0])
                    h_h = float(np.asarray(eng.h_h_out).real.ravel()[0])
                    if name.startswith("sighet_"):
                        eng.clear_in_model()
                ms = 1e3 * (time.perf_counter() - t)
                snr = float(np.sqrt(max(h_h, 0.0)))
                rec[name] = dict(snr=snr, t_get_ll_ms=ms,
                                 data=dict(snr=snr, snr_det=d_h / max(snr, 1e-300),
                                           logL=d_h - 0.5 * h_h,
                                           mm=1 - d_h / max(np.sqrt(d_d * h_h), 1e-300)))
                if name == "chunked":
                    hh_prod, dh_prod = h_h, d_h
            if hh_prod is not None:
                for name in engines:
                    if name != "chunked":
                        rec[name]["dlogL_vs_production"] = (rec[name]["data"]["logL"]
                                                            - (dh_prod - 0.5 * hh_prod))
            _write(a.out, rec)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", choices=("speed", "gate", "mojito"), required=True)
    ap.add_argument("--days", type=float, default=180.0)
    ap.add_argument("--laptop", action="store_true", help="Nf 180 / dt 20 (same 3600-s layer)")
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--engines", default=",".join(tb.ENGINES))
    ap.add_argument("--rows", default="8,64,512,4096")
    ap.add_argument("--max-slots", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--scale", type=float, default=1.0, help="speed-step candidate jitter")
    ap.add_argument("--edge", type=int, default=tb.EDGE_CROP_WAVELETS)
    ap.add_argument("--foreground", choices=("on", "off"), default="on")
    ap.add_argument("--table", default=None)
    ap.add_argument("--gate-sources", default="synthetic,catalogue")
    ap.add_argument("--gate-f0", default="0.3,0.6,1.0,2.0,4.0,8.0,16.0", help="mHz")
    ap.add_argument("--gate-cosi", default="0.0,0.02,0.05,0.1,0.3,0.8")
    ap.add_argument("--gate-skies", type=int, default=1)
    ap.add_argument("--rho-list", default="100,1000", help="synthetic SNR ladder")
    ap.add_argument("--gate-catalogue-top", type=int, default=20)
    ap.add_argument("--gate-vgb-top", type=int, default=10)
    ap.add_argument("--step-scales", default="0.1,1,3", help="x 100/SNR posterior-scale steps")
    ap.add_argument("--n-cand", type=int, default=1)
    ap.add_argument("--vgb-top", type=int, default=6)
    ap.add_argument("--gb-top", type=int, default=6)
    ap.add_argument("--l1-dir", default=None)
    ap.add_argument("--catalogue", default=None)
    ap.add_argument("--vgb-catalogue", default=None)
    ap.add_argument("--start-offset", type=float, default=tb.START_OFFSET_S)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True, help="jsonl to append to")
    a = ap.parse_args()
    a.engines = [e for e in a.engines.split(",") if e]
    a.rows = [int(x) for x in a.rows.split(",")]
    a.gate_f0 = [float(x) for x in a.gate_f0.split(",")]
    a.gate_cosi = [float(x) for x in a.gate_cosi.split(",")]
    a.step_scales = [float(x) for x in a.step_scales.split(",")]
    a.gate_sources = [x for x in a.gate_sources.split(",") if x]
    a.rho_list = [float(x) for x in a.rho_list.split(",")]
    {"speed": step_speed, "gate": step_gate, "mojito": step_mojito}[a.step](a)


if __name__ == "__main__":
    main()
