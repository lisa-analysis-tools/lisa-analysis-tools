"""Plots for one CD1-L chain: corner, sky, traces."""
import os, sys, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from eryn.backends import HDFBackend

BASIS = ["logM","Q","s1z","s2z","dist","phi_ref","cos_iota","psi","ra","sin_dec","t_plunge"]
LABELS = [r"$\ln M$", r"$q=m_1/m_2$", r"$\chi_{1z}$", r"$\chi_{2z}$", r"$d_L$ [Gpc]",
          r"$\phi_{\rm ref}$", r"$\cos\iota$", r"$\psi$", r"RA", r"$\sin\delta$",
          r"$t_c$ [s]"]
fp = os.environ["CD1L_CHAIN"]
OUT = os.environ.get("CD1L_PLOTS", "/data/nbody/majoburo/cd1l_pe/plots"); os.makedirs(OUT, exist_ok=True)
m = re.search(r"_id(\d+)_(\w+)\.h5$", os.path.basename(fp)); ID, TAG = int(m.group(1)), m.group(2)
MC = TAG.endswith("_mc")              # chirp-mass-basis chain -> convert to (logM, Q) for plotting
if MC:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import cd1l_sampling as S
    inj = S.mc_to_stock(np.load(os.path.join(os.path.dirname(fp), f"injection_id{ID}_mc.npy")))
else:
    inj = np.load(f"/data/nbody/majoburo/cd1l_pe/runs/injection_id{ID}.npy")

b = HDFBackend(fp, read_only=True)
stop = min(b.iteration, int(os.environ.get("CD1L_PLOT_STOP", b.iteration)))
ch = np.asarray(b.get_chain()["mbh"])[:stop, 0, :, 0, :]      # (nsteps, nwalkers, ndim) cold chain
ll = np.asarray(b.get_log_like())[:stop, 0, :]
if MC:
    # A CD1L_TDET=1 run samples t_det in the t_plunge slot and leaves its delay
    # vector beside the chain; map back to t_SSB so the (t_SSB) injection line
    # and every other campaign's plots share one convention.
    _dv = os.path.join(os.path.dirname(fp), f"dvec_id{ID}_mc.npy")
    if os.path.exists(_dv):
        ch = S.tdet_to_ssb(ch, np.load(_dv))
        print(f"  t_det chain: t_plunge mapped back to t_SSB with {_dv}")
    ch = S.mc_to_stock(ch)
# Public backend iteration excludes the preallocated tail. Never silently
# discard genuine stored failures or legitimate zero log likelihoods.
if len(ch) < 8 or not np.isfinite(ch).all() or not np.isfinite(ll).all():
    raise ValueError('Insufficient samples or nonfinite stored cold-chain values')
ns, nw, nd = ch.shape
# Display periodic coordinates and truths in the same local branch. This
# removes the artificial 0/2pi split without shifting the physical posterior.
ch = ch.copy(); inj = inj.copy()
for j, period in ((5, 2*np.pi), (7, np.pi), (8, 2*np.pi)):
    center = float(np.angle(np.exp(2j*np.pi*ch[ns//2:, :, j]/period).mean()))*period/(2*np.pi)
    ch[..., j] = center+(ch[..., j]-center+period/2)%period-period/2
    inj[j] = center+(inj[j]-center+period/2)%period-period/2
burn = ns // 2
flat = ch[burn:].reshape(-1, nd)
print(f"id{ID} [{TAG}]: {ns} steps, {nw} walkers, {flat.shape[0]:,} post-burn samples")

# ---- 1. traces
fig, axes = plt.subplots(6, 1, figsize=(11, 14), sharex=True)
axes[0].plot(ll[:, ::4], lw=0.4, alpha=0.6); axes[0].set_ylabel("log L (T=1)")
axes[0].axvline(burn, color="k", ls="--", lw=0.8)
for ax, j in zip(axes[1:], [BASIS.index("logM"), BASIS.index("dist"), BASIS.index("phi_ref"), BASIS.index("ra"), BASIS.index("sin_dec")]):
    ax.plot(ch[:, ::4, j], lw=0.4, alpha=0.6); ax.axhline(inj[j], color="r", lw=1)
    ax.set_ylabel(LABELS[j]); ax.axvline(burn, color="k", ls="--", lw=0.8)
axes[-1].set_xlabel("step"); fig.suptitle(f"CD1-L MBHB id{ID} [{TAG}] traces (red = injection)")
fig.tight_layout(); fig.savefig(f"{OUT}/id{ID}_{TAG}_traces.png", dpi=130); plt.close(fig)

# ---- 2. corner
try:
    import corner
    fig = corner.corner(flat, labels=LABELS, truths=inj, truth_color="r",
                        quantiles=[0.05, 0.5, 0.95], show_titles=True,
                        title_fmt=".4g", title_kwargs=dict(fontsize=8),
                        label_kwargs=dict(fontsize=9))
    fig.suptitle(f"CD1-L MBHB id{ID} [{TAG}] posterior (red = injection)", y=1.01)
    fig.savefig(f"{OUT}/id{ID}_{TAG}_corner.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    print("  corner OK")
except Exception as e:
    print("  corner failed:", type(e).__name__, e)

# ---- 3. sky: scatter + equal-area 90% contour, in (RA, sin dec) and in degrees
ra = flat[:, BASIS.index("ra")] % (2*np.pi); sd = np.clip(flat[:, BASIS.index("sin_dec")], -1, 1)
dec = np.degrees(np.arcsin(sd))
ra0, sd0 = inj[BASIS.index("ra")] % (2*np.pi), inj[BASIS.index("sin_dec")]
STER2DEG2 = (180/np.pi)**2
# ONE RA convention for scatter, star, contour and limits: unwrapped around the
# sample mean (so a posterior straddling RA=0 is not split), then mapped back
# so the mean sits in [0, 360).
ra_c = float(np.angle(np.mean(np.exp(1j * ra)))) % (2 * np.pi)
ra_u = (ra - ra_c + np.pi) % (2 * np.pi) - np.pi
rad = np.degrees(ra_u + ra_c)
ra0d = np.degrees(((ra0 - ra_c + np.pi) % (2 * np.pi) - np.pi) + ra_c)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.5))
a1.scatter(rad, dec, s=1, alpha=0.15, c="C0"); a1.plot(ra0d, np.degrees(np.arcsin(sd0)), "r*", ms=14)
a1.set_xlabel("RA [deg]"); a1.set_ylabel("Dec [deg]"); a1.set_title("posterior samples (red star = injection)")
# credible contours from an equal-area histogram in (ra, sin dec) on a grid
# ADAPTED to the sample spread. A fixed 60x60 all-sky grid has 34.4 deg^2
# cells, larger than any post-merger localisation (areas came out as integer
# multiples of one cell, and single-cell posteriors crashed contourf).
def sky_hist(nb):
    rlo, rhi = np.percentile(ra_u, [0.2, 99.8]); slo, shi = np.percentile(sd, [0.2, 99.8])
    pr, ps = 0.15 * (rhi - rlo) + 1e-9, 0.15 * (shi - slo) + 1e-9
    Hh, re_, se_ = np.histogram2d(ra_u, sd, bins=nb, range=[[rlo - pr, rhi + pr], [slo - ps, shi + ps]])
    cell = (re_[1] - re_[0]) * (se_[1] - se_[0])
    order = np.sort(Hh.ravel())[::-1]; csum = np.cumsum(order) / order.sum()
    lev = {p: order[min(np.searchsorted(csum, p), order.size - 1)] for p in (0.5, 0.9)}
    a90 = int(np.sum(Hh >= lev[0.9])) * cell * STER2DEG2; a50 = int(np.sum(Hh >= lev[0.5])) * cell * STER2DEG2
    return Hh, re_, se_, lev, a50, a90
nb = int(np.clip(np.sqrt(flat.shape[0] / 25.0), 20, 80))
H, re_, se_, lev, area50, area90 = sky_hist(nb)
_, _, _, _, a50h, a90h = sky_hist(max(nb // 2, 10))           # resolution check
print(f"  sky grid {nb}x{nb} (cell {(re_[1]-re_[0])*(se_[1]-se_[0])*STER2DEG2:.3g} deg^2); "
      f"at half resolution 50/90% = {a50h:.3g}/{a90h:.3g} deg^2")
X, Y = np.meshgrid(0.5 * (re_[1:] + re_[:-1]) + ra_c, 0.5 * (se_[1:] + se_[:-1]), indexing="ij")
levels = sorted({float(lev[0.9]), float(lev[0.5]), float(H.max()) + 1})
a2.contourf(np.degrees(X), np.degrees(np.arcsin(Y)), H, levels=levels,
            colors=["#9ecae1", "#3182bd"][:len(levels) - 1], alpha=0.8)
a2.plot(ra0d, np.degrees(np.arcsin(sd0)), "r*", ms=14)
a2.set_xlabel("RA [deg]"); a2.set_ylabel("Dec [deg]")
a2.set_title(f"50% / 90% credible: {area50:.3g} / {area90:.3g} deg$^2$")
lo, hi = np.degrees(np.percentile(ra_u, [0.2, 99.8]) + ra_c); pad = max(0.15 * (hi - lo), 0.02)
a2.set_xlim(lo - pad, hi + pad); a1.set_xlim(lo - pad, hi + pad)
lo, hi = np.percentile(dec, [0.2, 99.8]); pad = max(0.15 * (hi - lo), 0.02)
a2.set_ylim(lo - pad, hi + pad); a1.set_ylim(lo - pad, hi + pad)
fig.suptitle(f"CD1-L MBHB id{ID} [{TAG}] sky localisation"); fig.tight_layout()
fig.savefig(f"{OUT}/id{ID}_{TAG}_sky.png", dpi=130); plt.close(fig)
print(f"  sky OK: 50%={area50:.3g} 90%={area90:.3g} deg^2")
print("wrote", OUT)
