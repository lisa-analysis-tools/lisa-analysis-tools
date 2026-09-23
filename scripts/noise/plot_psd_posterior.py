"""Corner plot of a noise-only PSD posterior, truth marked.

    python scripts/noise/plot_psd_posterior.py <store_dir> <out.png> [burnin]

Colors are the data-viz reference palette (categorical slots 1-2 + the
blue sequential ramp). TRUTH comes from ``psd_truth_levels``: PSD_TRUTH if
set, else the brick's tabulated estimates refitted with the UNEQUAL-arm model,
else mojito-light's round injection. Do NOT paste a literal back in -- the old
1.496182e-11 / 2.982412e-15 was the EQUAL-arm fit, 0.26% / 0.59% low, and the
truth marker landed off the posterior (2026-09-23).
"""
import os, sys, h5py, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

STORE = sys.argv[1]
OUT = sys.argv[2]
BURN = int(sys.argv[3]) if len(sys.argv) > 3 else 50

# truth: resolved, never a literal (see the module docstring)
from lisatools.globalfit.stock.erebor.noise import psd_truth_levels

TRUTH = np.array(psd_truth_levels(
    mojito_data_path=os.environ.get("MOJITO_DATA_PATH")))
SCALE = np.array([1e-12, 1e-15])          # -> pm , fm/s^2 : readable axis numbers
LAB = [r"$S_{\rm oms,d}$  [$10^{-12}$ m]", r"$S_{\rm a,a}$  [$10^{-15}$ m s$^{-2}$]"]

# reference palette, slots 1 & 2 + blue sequential ramp (validated defaults)
SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
BLUE, ORANGE = "#2a78d6", "#eb6834"
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
GRID = "#e4e3e0"

with h5py.File(f"{STORE}/gf_noise_6mo_testing.h5", "r") as f:
    ch = f["global_fit/sub_backend/psd/chain"][:]      # (it, temp, walker, leaf, 2)
    ll = f["global_fit/sub_backend/psd/log_like"][:]   # (it, temp, walker)
live = ~np.all(ch.reshape(ch.shape[0], -1) == 0, axis=1)
n = int(live.sum())
cold, llc = ch[:n, 0, :, 0, :], ll[:n, 0, :]          # beta = 1
s = cold[BURN:].reshape(-1, 2) / SCALE                # posterior samples
it, w = np.unravel_index(np.nanargmax(llc), llc.shape)
ml = cold[it, w] / SCALE
tr = TRUTH / SCALE

fig = plt.figure(figsize=(7.4, 7.0), facecolor=SURF)
gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.62], height_ratios=[0.62, 1],
                      wspace=0.07, hspace=0.07,
                      left=0.13, right=0.97, top=0.88, bottom=0.10)
ax_j = fig.add_subplot(gs[1, 0])
ax_x = fig.add_subplot(gs[0, 0], sharex=ax_j)
ax_y = fig.add_subplot(gs[1, 1], sharey=ax_j)
ax_t = fig.add_subplot(gs[0, 1]); ax_t.axis("off")

for a in (ax_j, ax_x, ax_y):
    a.set_facecolor(SURF)
    for sp in a.spines.values():
        sp.set_color(GRID); sp.set_linewidth(0.8)
    a.tick_params(colors=INK2, labelsize=9, width=0.8, color=GRID)

# ---- joint: filled density + 68/95 contours -------------------------------
H, xe, ye = np.histogram2d(s[:, 0], s[:, 1], bins=40)
from scipy.ndimage import gaussian_filter
Hs = gaussian_filter(H.T.astype(float), 1.1)
Hs /= Hs.sum()
flat = np.sort(Hs.ravel())[::-1]
csum = np.cumsum(flat)
lv = [flat[np.searchsorted(csum, q)] for q in (0.95, 0.68)]
xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
ax_j.contourf(xc, yc, Hs, levels=[lv[0], lv[1], Hs.max()],
              colors=[RAMP[1], RAMP[3]], alpha=0.95)
ax_j.contour(xc, yc, Hs, levels=lv, colors=[RAMP[4], RAMP[5]], linewidths=1.2)

# truth: 2px, orange, on every panel
for a, v in ((ax_j, tr[0]), (ax_x, tr[0])):
    a.axvline(v, color=ORANGE, lw=2.0, zorder=5)
for a, v in ((ax_j, tr[1]), (ax_y, tr[1])):
    a.axhline(v, color=ORANGE, lw=2.0, zorder=5)
ax_j.plot(tr[0], tr[1], "o", ms=9, mfc=ORANGE, mec=SURF, mew=2.0, zorder=7)
ax_j.plot(ml[0], ml[1], "X", ms=11, mfc=INK, mec=SURF, mew=2.0, zorder=6)

# ---- marginals ------------------------------------------------------------
for a, col, vert in ((ax_x, 0, True), (ax_y, 1, False)):
    h, e = np.histogram(s[:, col], bins=55, density=True)
    c = 0.5 * (e[1:] + e[:-1])
    if vert:
        a.fill_between(c, h, color=RAMP[1], step="mid")
        a.step(c, h, where="mid", color=BLUE, lw=2.0)
        a.set_ylim(0, h.max() * 1.25)
    else:
        a.fill_betweenx(c, h, color=RAMP[1], step="mid")
        a.step(h, c, where="mid", color=BLUE, lw=2.0)
        a.set_xlim(0, h.max() * 1.25)
    a.set_yticks([]) if vert else a.set_xticks([])
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
plt.setp(ax_x.get_xticklabels(), visible=False)
plt.setp(ax_y.get_yticklabels(), visible=False)
ax_x.spines["left"].set_visible(False)
ax_y.spines["bottom"].set_visible(False)

ax_j.set_xlabel(LAB[0], color=INK, fontsize=11)
ax_j.set_ylabel(LAB[1], color=INK, fontsize=11)

# direct labels instead of a legend box
ax_j.annotate("truth", xy=(tr[0], ax_j.get_ylim()[0]), xytext=(4, 6),
              textcoords="offset points", color=ORANGE, fontsize=10,
              fontweight="bold", ha="left", va="bottom")
ax_j.annotate("max-likelihood", xy=(ml[0], ml[1]), xytext=(14, 10),
              textcoords="offset points", color=INK, fontsize=10, ha="left")

# ---- headline + numbers ---------------------------------------------------
err = 100 * (ml / tr - 1)
fig.text(0.13, 0.955, "6-month noise-only PSD posterior",
         color=INK, fontsize=14, fontweight="bold", ha="left")
fig.text(0.13, 0.925,
         f"mojito instrument noise, XYZ TDI-2, 180 d  ·  {s.shape[0]:,} cold samples "
         f"(8 walkers, burn-in {BURN})",
         color=INK2, fontsize=9.5, ha="left")
rows = [("", "fit", "truth", "error"),
        ("Soms_d", f"{ml[0]:.3f}", f"{tr[0]:.3f}", f"{err[0]:+.2f}%"),
        ("Sa_a", f"{ml[1]:.3f}", f"{tr[1]:.3f}", f"{err[1]:+.2f}%")]
for r, row in enumerate(rows):
    for c, cell in enumerate(row):
        ax_t.text(0.00 + c * 0.265, 0.72 - r * 0.20, cell,
                  transform=ax_t.transAxes, fontsize=9.5,
                  color=INK2 if r == 0 else INK,
                  fontweight="bold" if r == 0 else "normal",
                  ha="left", va="center", family="monospace")
sig = (ml - tr) / s.std(axis=0)
ax_t.text(0.02, 0.10, f"68% / 95% credible regions\n"
                     f"fit sits {sig[0]:+.1f}$\\sigma$ / {sig[1]:+.1f}$\\sigma$ from truth", transform=ax_t.transAxes,
          fontsize=9, color=INK2, ha="left", va="center")

fig.savefig(OUT, dpi=170, facecolor=SURF)
print(f"wrote {OUT}")
print(f"samples {s.shape[0]}  ML {ml}  truth {tr}  err {err}")
