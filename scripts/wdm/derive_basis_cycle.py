"""Derive the per-pixel map from WDMLookupTable values to true WDM pixels (Task A3).

For a target pixel (m, n) and a unit carrier at frequency offset ``delta`` from
layer m, with local chirp ``fdot`` and phase referenced at the pixel time
``t_n = n * layer_dt`` (the builder's convention; the half-pixel centre breaks it),

    y(t) = cos(2 pi [(m df + delta) tau + fdot tau^2 / 2] + phi),  tau = t - t_n,

the TD->WDM pixel is  w = c_mn cos(phi) + s_mn sin(phi)  (c from phi = 0, s from
phi = -pi/2). The table stores, per (delta, fdot), one (cos, sin) pair measured at
its single reference pixel. We fit, for each class

    (dm, dn, block) = ((m - m_ref) % 4, (n - n_ref) % 4, floor(delta / df)),

the signed 2x2 map T (8 candidates: 4 quarter-turn rotations x optional reflection)
with  [c_mn, s_mn] = T @ [tab_cos, tab_sin],  and print its worst relative error.
delta and fdot sit on table nodes so interpolation error does not blur the fit.
Everything uses the installed lisatools TD->WDM transform.
"""

import itertools
import os
import tempfile

import numpy as np

from lisatools.domains import TDSettings, TDSignal, WDMLookupTable, WDMSettings

NF, NT, DT = 64, 128, 56.25           # layer_dt = 3600 s like production
TIME_LAYERS = 64                      # long build grid: short ones carry edge artefacts
EPS_F = float(os.environ.get("EPS_F", "0.02"))
EPS_FD = float(os.environ.get("EPS_FD", "0.25"))
FD_FACTOR = 1.0
BATCH = int(os.environ.get("BATCH", "16"))

wdm = WDMSettings(Nf=NF, Nt=NT, dt=DT, force_backend="cpu")
df, ldt = wdm.layer_df, wdm.layer_dt
N = NF * NT
t = np.arange(N) * DT
td_set = TDSettings(N, DT, force_backend="cpu")

norm_f, m_diffs, m_ref = WDMLookupTable.apply_eps_frequency(EPS_F, wdm, m_ref=int(os.environ.get("M_REF", "20")), num_layers_diff=2)
fdot_vals = WDMLookupTable.apply_eps_fdot(EPS_FD, wdm, fdot_max_factor=FD_FACTOR)
table = WDMLookupTable(wdm, 1, m_ref=m_ref, norm_freq_single_layer=norm_f, m_diffs=m_diffs,
                       fdot_vals=fdot_vals, store_path=os.path.join(tempfile.mkdtemp(), "t.h5"),
                       batch_size_gen=BATCH, build_kind="n_ref_complex", time_layers=TIME_LAYERS)
n_ref = table.n_ref
print(f"m_ref={m_ref} n_ref(table)={n_ref} eval Nt//2={NT // 2}  fdot nodes={len(fdot_vals)} "
      f"f support=[{float(table.f_vals_norm.min()) / df:+.2f}, {float(table.f_vals_norm.max()) / df:+.2f}] df")

deltas = np.array([-2.7, -2.3, -1.64, -1.2, -0.8, -0.5, 0.1, 0.34, 0.5, 0.9, 1.3, 1.62, 2.44, 2.9]) * df  # blocks -3..2, multiples of EPS_F
ms = range(m_ref - 2, m_ref + 2) if os.environ.get('QUICK') else None
fdots = np.array([float(x) for x in os.environ.get("FDOTS", "0.0,0.5,-0.75").split(",")]) * FD_FACTOR * df / ldt  # on EPS_FD nodes
ms = ms or range(m_ref - 2, m_ref + 6)
ns = range(NT // 2 - 4, NT // 2 + 4)

CANDS = []
for k in range(4):
    th = k * np.pi / 2
    R = np.round(np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]))
    CANDS.append((f"rot{k}", R))
    CANDS.append((f"rot{k}*refl", R @ np.diag([1.0, -1.0])))


def pixel_pair(m, n, delta, fdot):
    tau = t - n * ldt
    out = []
    for phi in (0.0, -np.pi / 2):
        y = np.cos(2 * np.pi * ((m * df + delta) * tau + 0.5 * fdot * tau ** 2) + phi)[None, :]
        w = np.asarray(TDSignal(y, td_set).transform(wdm).arr)[0]
        out.append(w[m - int(wdm.ind_min_f), n])   # arr is (nch, Nf_active, Nt_active)
    return np.array(out)


samples = {}
for m, n, delta, fdot in itertools.product(ms, ns, deltas, fdots):
    s_c, c_c = table.get_table_coeffs(np.array([delta]), np.array([fdot]), np.array([n]))
    tab = np.array([float(np.asarray(c_c)[0]), float(np.asarray(s_c)[0])])
    true = pixel_pair(m, n, delta, fdot)
    key = ((m - m_ref) % 4, (n - n_ref) % 4, int(np.floor(delta / df)))
    samples.setdefault(key, []).append((tab, true))

print("\nclass (dm%4, dn%4, block) -> best map, worst rel err over its samples, runner-up err")
rows = []
for key in sorted(samples):
    errs = []
    for name, T in CANDS:
        e = max(np.linalg.norm(T @ tab - true) / max(np.linalg.norm(true), 1e-300) for tab, true in samples[key])
        errs.append((e, name))
    errs.sort()
    rows.append((key, errs[0][1], errs[0][0], errs[1][0]))
    print(f"  {key}: {errs[0][1]:10s} worst {errs[0][0]:.2e}   (next {errs[1][1]} {errs[1][0]:.2e})")
worst = max(r[2] for r in rows)
print(f"\nWORST best-fit error over all classes: {worst:.2e}  -> {'PURE SIGNED-PERMUTATION RULE' if worst < 1e-4 else 'NOT a pure rule'}")

if os.environ.get("DERIVE_DUMP"):
    for key in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]:
        for tab, true in samples[key][:6]:
            print(f"  DUMP {key}: tab(c,s)=({tab[0]:+.4e},{tab[1]:+.4e})  true(c,s)=({true[0]:+.4e},{true[1]:+.4e})  |tab|/|true|={np.linalg.norm(tab)/np.linalg.norm(true):.3e}")

if os.environ.get("DERIVE_WORST"):
    # per class: error of the parity rule (rot0 if dm+dn even else rot3) and the worst sample
    R0, R3 = CANDS[0][1], CANDS[6][1]
    for key in sorted(samples):
        T = R0 if (key[0] + key[1]) % 2 == 0 else R3
        errs = [(np.linalg.norm(T @ tab - true) / np.linalg.norm(tab), tab, true) for tab, true in samples[key]]
        e, tab, true = max(errs, key=lambda x: x[0])
        print(f"  WORST {key}: err={e:.2e} tab=({tab[0]:+.3e},{tab[1]:+.3e}) true=({true[0]:+.3e},{true[1]:+.3e})")


def rule(dm, dn, block, tab):
    """General rule: odd (m_ref + n_ref) tables first map (c, s) -> (-c, s); then the
    (-1)^block bake on s is undone; then pixels of odd ABSOLUTE parity (m + n) turn
    (c, s) -> (s, -c)."""
    c, sn = tab[0], tab[1]
    if (m_ref + n_ref) % 2:
        c = -c
    sn = ((-1.0) ** block) * sn
    odd = (dm + dn + m_ref + n_ref) % 2 != 0
    return np.array([sn, -c]) if odd else np.array([c, sn])


peak = max(np.linalg.norm(tab) for v in samples.values() for tab, _ in v)
worst_rule = 0.0
per_block = {}
for (dm, dn, b), v in samples.items():
    for tab, true in v:
        e = np.linalg.norm(rule(dm, dn, b, tab) - true) / peak
        worst_rule = max(worst_rule, e)
        per_block[b] = max(per_block.get(b, 0.0), e)
print("RULE per block (max |rule - true| / peak):", {k: f"{per_block[k]:.2e}" for k in sorted(per_block)})
print(f"RULE worst over all {sum(len(v) for v in samples.values())} samples: {worst_rule:.2e} of peak")

