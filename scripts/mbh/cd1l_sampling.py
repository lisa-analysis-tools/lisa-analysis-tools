"""Sampling machinery for the CD1-L MBHB PE, ported from the user's
osearch_a6000/run_pe_v0.py (BBHx) to the WDM/phentax likelihood.

Three things that made those chains converge and the (lnM, Q) + StretchMove
driver not:

  1. COORDINATES: chirp mass and q = m2/m1 <= 1 instead of (ln M_tot, Q=m1/m2).
     The signal measures ln Mc to ~1e-4 and q only weakly; in (ln M, Q) that is
     a long diagonal ridge, the slow direction of every chain so far.
  2. MOVES: a Gaussian proposal from the Fisher covariance at the injection
     (Cholesky steps, scale {0.3, 1}/sqrt(beta)), a Gibbs move over the eight
     LISA sky-symmetry partners, and a 5% prior draw, alongside StretchMove.
  3. (phi_ref marginalisation is NOT ported: the WDM likelihood has no FD
     phase trick. phi_ref stays sampled and periodic.)

Sampling basis (11):  lnMc, q, s1z, s2z, dist[Gpc], phi_ref, cos_iota, psi,
                      ra, sin_dec, t_plunge     (ICRS sky, as the stock basis)
"""
import numpy as np
from eryn.utils.transform import TransformContainer
from eryn.moves import MHMove
import astropy.units as u
from astropy.coordinates import SkyCoord, ICRS, BarycentricMeanEcliptic

BASIS_MC = ["lnMc", "q", "s1z", "s2z", "dist", "phi_ref", "cos_iota", "psi",
            "ra", "sin_dec", "t_plunge"]
NDIM = len(BASIS_MC)
I = {k: j for j, k in enumerate(BASIS_MC)}
PERIODIC = {I["phi_ref"]: 2 * np.pi, I["psi"]: np.pi, I["ra"]: 2 * np.pi}


# ------------------------------------------------------------- transform
def mc_q_to_m1_m2(lnMc, q):
    """(ln chirp mass, q = m2/m1 <= 1) -> (m1, m2)."""
    Mc = np.exp(lnMc)
    M = Mc * (1.0 + q) ** 1.2 / q ** 0.6
    m1 = M / (1.0 + q)
    return m1, q * m1


def m1_m2_to_mc_q(m1, m2):
    Mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    return np.log(Mc), m2 / m1


def gpc_to_mpc(x): return x * 1e3
def mpc_to_gpc(x): return x / 1e3


def make_mc_transform_container():
    """Same output basis as the stock erebor MBH container (m1, m2, s1z, s2z,
    dist[Mpc], phi_ref, iota, psi, alpha, delta, t_plunge), new input basis."""
    input_basis = ["lnMc", "q", "s1z", "s2z", "dist", "phi_ref", "cos_iota",
                   "psi", "alpha", "sin_delta", "t_plunge"]
    return TransformContainer(
        input_basis=input_basis,
        output_basis=list(input_basis),
        parameter_transforms={
            "dist": gpc_to_mpc,
            "cos_iota": np.arccos,
            "sin_delta": np.arcsin,
            ("lnMc", "q"): mc_q_to_m1_m2,
        },
        fill_dict={},
        inverse_parameter_transforms={
            "dist": mpc_to_gpc,
            "cos_iota": np.cos,
            "sin_delta": np.sin,
            ("lnMc", "q"): m1_m2_to_mc_q,
        },
    )


def stock_to_mc(x):
    """Stock sampling row (logM, Q=m1/m2, ...) -> chirp-mass row."""
    x = np.asarray(x, dtype=float); out = x.copy()
    M = np.exp(x[..., 0]); Q = x[..., 1]
    m2 = M / (1.0 + Q); m1 = Q * m2
    out[..., 0], out[..., 1] = m1_m2_to_mc_q(m1, m2)
    return out


def mc_to_stock(x):
    x = np.asarray(x, dtype=float); out = x.copy()
    m1, m2 = mc_q_to_m1_m2(x[..., 0], x[..., 1])
    out[..., 0] = np.log(m1 + m2); out[..., 1] = m1 / m2
    return out


# ------------------------------------------------ t_det (Doppler) coordinate
C_LIGHT_MS = 299792458.0
AU_LIGHT_S = 499.00478384


def delay_vector(orbits, t_abs):
    """r_centre(t_abs)/c in ICRS light-seconds.

    ``orbits`` is the L1Orbits built with frame='icrs', so no frame rotation is
    needed anywhere: the sky is ICRS (ra, sin_dec) too.  ``orbits.x`` is
    (n_t, spacecraft, coord) on the uniform (sc_t0, sc_dt) grid and may live on
    the GPU.  Verified against the delay vector fitted from the id2 chain:
    |d| = 498.89 s = 1.0000 AU and |d - c_fit|/|c_fit| = 0.0175, which is the
    fit's own residual."""
    x = orbits.x
    x = np.asarray(x.get() if hasattr(x, "get") else x)
    t = getattr(orbits, "_sc_t", None)
    if t is not None:
        t = np.asarray(t.get() if hasattr(t, "get") else t)
        t0, dt = float(t[0]), float(t[1] - t[0])
    else:                                   # orbits.dt is broken on L1Orbits
        t0, dt = float(orbits.sc_t0), float(orbits._sc_dt)
    k = (float(t_abs) - t0) / dt
    k0 = int(np.floor(k)); f = k - k0
    if not 0 <= k0 < len(x) - 1:
        raise ValueError(f"t_abs {t_abs} outside the orbit grid "
                         f"[{t0}, {t0 + dt * (len(x) - 1)}]")
    X = (1.0 - f) * x[k0] + f * x[k0 + 1]            # (3 spacecraft, 3 coords)
    return X.mean(axis=0) / C_LIGHT_MS


def _n_hat_icrs(x):
    ra = x[..., I["ra"]]
    dec = np.arcsin(np.clip(x[..., I["sin_dec"]], -1.0, 1.0))
    cd = np.cos(dec)
    return np.stack([cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)], axis=-1)


def tdet_to_ssb(x, d):
    """Sampling row(s) carrying t_det in the t_plunge slot -> t_SSB."""
    x = np.asarray(x, dtype=float); out = x.copy()
    out[..., I["t_plunge"]] = x[..., I["t_plunge"]] + _n_hat_icrs(x) @ d
    return out


def ssb_to_tdet(x, d):
    x = np.asarray(x, dtype=float); out = x.copy()
    out[..., I["t_plunge"]] = x[..., I["t_plunge"]] - _n_hat_icrs(x) @ d
    return out


class TDetTransform:
    """Wrap a TransformContainer so the sampler's 11th coordinate is the arrival
    time at the constellation centre rather than t_SSB:

        t_SSB = t_det + n_hat(ra, dec) . r_centre / c

    The delay is EXACTLY linear in n_hat but trigonometric in (ra, sin_dec),
    which is why no Gaussian proposal in (ra, sin_dec, t_SSB) can follow the
    ridge: over a 0.07 rad sky patch the quadratic part of the angle -> n_hat
    map is ~1.2 s against a ~0.04 s ridge thickness.  cd1l_ridge_accounting
    basis B says this delay is 90-98% of the whole t_plunge excess on every
    source that has one (post/F median 0.29 afterwards, max 1.02).

    r_centre is evaluated ONCE at the injection's absolute merger time, so the
    Jacobian is exactly 1: over the <30 s posterior spread in t the centre moves
    ~800 km = 3 ms of delay, against a 0.04 s thickness."""

    def __init__(self, inner, dvec):
        self.inner = inner
        self.dvec = np.asarray(dvec, dtype=float)

    def both_transforms(self, x, *a, **kw):
        return self.inner.both_transforms(tdet_to_ssb(x, self.dvec), *a, **kw)

    def __getattr__(self, name):
        if name in ("inner", "dvec"):
            raise AttributeError(name)
        return getattr(self.inner, name)


# --------------------------------------------------------------- Fisher
def fisher_cov(loglike, x0, eps0, nbatch=64, verbose=True):
    """Fisher covariance from the numerical Hessian of log L at x0 (noise-free
    data: log L = -1/2 dx^T F dx near the peak, so F = -H). Two passes: eps0
    guesses, then 0.5 sigma from pass 1.  Batched: ~2n + 4 n(n-1)/2 = 242
    likelihood rows per pass at n=11."""
    n = len(x0)

    def hessian(eps):
        pts = [x0.copy()]
        for a in range(n):
            for s in (+1, -1):
                p = x0.copy(); p[a] += s * eps[a]; pts.append(p)
        for a in range(n):
            for b in range(a + 1, n):
                for sa, sb in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    p = x0.copy(); p[a] += sa * eps[a]; p[b] += sb * eps[b]; pts.append(p)
        pts = np.asarray(pts)
        ll = np.concatenate([np.asarray(loglike(pts[k:k + nbatch])).ravel()
                             for k in range(0, len(pts), nbatch)])
        L0 = ll[0]; H = np.zeros((n, n)); k = 1
        for a in range(n):
            H[a, a] = (ll[k] + ll[k + 1] - 2 * L0) / eps[a] ** 2; k += 2
        for a in range(n):
            for b in range(a + 1, n):
                H[a, b] = H[b, a] = (ll[k] - ll[k + 1] - ll[k + 2] + ll[k + 3]) / (4 * eps[a] * eps[b]); k += 4
        return H, L0

    H1, L0 = hessian(np.asarray(eps0, dtype=float))
    F1 = -H1
    d = np.diag(F1)
    sig1 = np.where(d > 0, 1.0 / np.sqrt(np.abs(d) + 1e-300), np.asarray(eps0))
    H2, _ = hessian(0.5 * sig1)
    F2raw = -H2
    F2 = ensure_pd(F2raw)
    # invert in scaled coordinates too: pinv on a matrix with a 1e12 dynamic
    # range would drop the small (= widest) directions with rcond=1e-12
    dsc = np.sqrt(np.diag(F2))
    cov = np.linalg.inv(F2 / np.outer(dsc, dsc)) / np.outer(dsc, dsc)
    if verbose:
        np.set_printoptions(precision=3, linewidth=200)
        print("  [fisher] pass1 diag F   :", d)
        print("  [fisher] pass1 1/sqrt(F):", sig1)
        print("  [fisher] pass2 eps      :", 0.5 * sig1)
        print("  [fisher] pass2 diag F   :", np.diag(F2raw))
        print("  [fisher] pass2 1/sqrt(F):", 1.0 / np.sqrt(np.abs(np.diag(F2raw))))
        ev = np.linalg.eigvalsh(0.5 * (F2raw + F2raw.T))
        print("  [fisher] pass2 eigenvalues (raw):", ev)
        print("  [fisher] eigen-floor applied:", not np.allclose(F2, F2raw))
        dd = np.sqrt(np.abs(np.diag(F2raw))); Cs = F2raw / np.outer(dd, dd)
        print("  [fisher] scaled spectrum:", np.linalg.eigvalsh(0.5 * (Cs + Cs.T)))
        # floor 1e-6 chosen on id0 against the 5000-step chain: sky/dist/time
        # widths within 2x; the Fisher is near-singular in 4-5 directions
        # (scaled eigenvalues < 1e-4 are below finite-difference resolution).
        corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
        print("  [fisher] |corr| max off-diag:", np.max(np.abs(corr - np.eye(len(corr)))))
        sig = np.sqrt(np.diag(cov))
        print("  Fisher (pass 2) 1-sigma: " + "  ".join(f"{k}={s:.3g}" for k, s in zip(BASIS_MC, sig)),
              flush=True)
        print(f"  log L at injection = {L0:.4f}", flush=True)
    return cov


# Template-derivative Fisher, ported from the user's osearch fisher_forecast.py
# (Marsat/lisabeta construction): F_ij = <d_i h | d_j h> from CENTRAL finite
# differences of the template itself. A Gram matrix is positive semi-definite
# by construction, so unlike the log L Hessian above it needs no eigenvalue
# floor and its small eigenvalues (the sky/psi/phi/t/dist degeneracies) are
# measured, not invented. 2n template evaluations per pass.
FISHER_STEPS = {"lnMc": 1e-7, "q": 1e-7, "s1z": 1e-7, "s2z": 1e-7, "dist": 1e-7,   # dist: FRACTIONAL
                "phi_ref": 1e-7, "cos_iota": 1e-7, "psi": 1e-7, "ra": 1e-7,
                "sin_dec": 1e-7, "t_plunge": 1e-4}
FISHER_FRACTIONAL = {"dist"}


def fisher_from_templates(ac, transform, x0, steps=None, factors=(0.5, 1.0, 2.0),
                          verbose=True):
    """Fisher covariance in the chirp-mass sampling basis from template
    derivatives. `factors` rescale all steps for a convergence check (the
    factor-1 result is returned; the spread across factors is reported)."""
    from lisatools.diagnostic import inner_product
    x0 = np.asarray(x0, dtype=float); n = len(x0)
    steps = dict(FISHER_STEPS if steps is None else steps)
    _h = lambda a: a.get() if hasattr(a, "get") else np.asarray(a)

    def template(x):
        p_in = transform.both_transforms(np.asarray(x, dtype=float))
        return ac._build_template(np.asarray(p_in, dtype=float))

    def fisher(fac):
        dh, sc = [], np.ones(n)
        for j, name in enumerate(BASIS_MC):
            eps = steps[name] * fac
            e = np.zeros(n)
            if name in FISHER_FRACTIONAL:
                e[j] = eps * x0[j]; sc[j] = x0[j]          # step-space: d/d(eps)
            else:
                e[j] = eps
            hp, hm = template(x0 + e), template(x0 - e)
            d = (hp.arr - hm.arr) / (2.0 * eps)              # derivative wrt step variable
            dh.append(hp.settings.associated_class(d, hp.settings))
        _, _, sens = ac._slice_to_template(dh[0])
        F = np.zeros((n, n))
        for a in range(n):
            for b in range(a, n):
                F[a, b] = F[b, a] = float(np.real(_h(inner_product(dh[a], dh[b], psd=sens))))
        # invert in step-space (diag-normalised), then rescale to physical
        d = np.sqrt(np.diag(F)); d = np.where(d > 0, d, 1.0)
        C = F / np.outer(d, d)
        Cinv = np.linalg.pinv(0.5 * (C + C.T), rcond=1e-14, hermitian=True)
        cov_step = Cinv / np.outer(d, d)
        S = np.diag(sc)                                     # physical = step * sc
        return F / np.outer(sc, sc), S @ cov_step @ S, np.linalg.cond(C)

    res = {fac: fisher(fac) for fac in factors}
    F, cov, cond = res[1.0] if 1.0 in res else res[factors[0]]
    if verbose:
        sig = {fac: np.sqrt(np.diag(r[1])) for fac, r in res.items()}
        med = np.median(np.array(list(sig.values())), axis=0)
        spread = (np.max(list(sig.values()), axis=0) - np.min(list(sig.values()), axis=0)) / med
        np.set_printoptions(precision=3, linewidth=200)
        print(f"  [fisher-T] cond(scaled F) = {cond:.3g}; step-convergence spread per param: "
              + " ".join(f"{k}={v:.2f}" for k, v in zip(BASIS_MC, spread)), flush=True)
        print("  Fisher-T 1-sigma: " + "  ".join(f"{k}={s:.3g}" for k, s in zip(BASIS_MC, np.sqrt(np.diag(cov)))),
              flush=True)
    return cov


def ensure_pd(F, rel_floor=1e-6):
    """Make a Fisher/covariance matrix positive definite WITHOUT distorting it.
    The MBH Fisher diagonal spans ~15 orders of magnitude (lnMc ~6e13, dist
    ~50), so adding alpha*diag jitter (the usual trick) injects 1e-10*6e13 =
    6e3 into every direction and wipes out the small eigenvalues that ARE the
    marginal widths. Instead: scale to unit diagonal, floor the eigenvalues
    at rel_floor * max, and scale back."""
    F = 0.5 * (F + F.T)
    d = np.sqrt(np.abs(np.diag(F)))
    d = np.where(d > 0, d, 1.0)
    C = F / np.outer(d, d)
    ev, V = np.linalg.eigh(C)
    floor = rel_floor * ev.max()
    if ev.min() < floor:
        ev = np.maximum(ev, floor)
        C = (V * ev) @ V.T
    return C * np.outer(d, d)


# ---------------------------------------------------------- sky partners
def _to_ecliptic(ra, dec):
    """Vectorised: (lam, beta) of ICRS directions and the local rotation angle
    between ICRS north and ecliptic north there (for the polarisation angle)."""
    ra = np.atleast_1d(ra); dec = np.atleast_1d(dec)
    c = SkyCoord(ra=ra * u.rad, dec=dec * u.rad, frame=ICRS)
    e = c.transform_to(BarycentricMeanEcliptic)
    lam, beta = e.lon.rad, e.lat.rad
    north = SkyCoord(lon=lam * u.rad, lat=np.minimum(beta + 1e-4, np.pi / 2 - 1e-9) * u.rad,
                     frame=BarycentricMeanEcliptic).transform_to(ICRS)
    return lam, beta, c.position_angle(north).rad     # ecliptic north, E of ICRS north


def _from_ecliptic(lam, beta):
    c = SkyCoord(lon=np.atleast_1d(lam) * u.rad, lat=np.atleast_1d(beta) * u.rad,
                 frame=BarycentricMeanEcliptic).transform_to(ICRS)
    return c.ra.rad, c.dec.rad


def sky_partners(rows):
    """The 8 LISA sky-symmetry partners of chirp-basis rows (ICRS), shape
    (N, 8, ndim): rotate ecliptic longitude by k pi/2 (psi follows),
    optionally flip latitude (cos_iota -> -cos_iota, psi -> pi - psi).
    An exact group action, so the Gibbs move is valid whatever the psi
    convention; a wrong psi sign only costs acceptance."""
    rows = np.atleast_2d(np.asarray(rows, dtype=float))
    N = rows.shape[0]
    lam, beta, off = _to_ecliptic(rows[:, I["ra"]], np.arcsin(np.clip(rows[:, I["sin_dec"]], -1, 1)))
    psi_e = (rows[:, I["psi"]] + off) % np.pi
    out = np.empty((N, 8, rows.shape[1]))
    for k in range(4):
        for f in range(2):
            x = rows.copy()
            l2 = (lam + k * np.pi / 2) % (2 * np.pi); b2 = beta.copy()
            p2 = (psi_e + k * np.pi / 2) % np.pi
            if f:
                b2 = -b2; x[:, I["cos_iota"]] = -x[:, I["cos_iota"]]; p2 = (np.pi - p2) % np.pi
            ra2, dec2 = _from_ecliptic(l2, b2)
            _, _, off2 = _to_ecliptic(ra2, dec2)
            x[:, I["ra"]] = ra2 % (2 * np.pi); x[:, I["sin_dec"]] = np.sin(dec2)
            x[:, I["psi"]] = (p2 - off2) % np.pi
            out[:, 2 * k + f] = x
    return out


# ---------------------------------------------------------------- moves
def reflect_into_prior(x, lo, hi):
    for j in range(x.shape[-1]):
        if j in PERIODIC:
            continue
        w = hi[j] - lo[j]
        v = (x[..., j] - lo[j]) % (2 * w)
        x[..., j] = lo[j] + np.where(v > w, 2 * w - v, v)
    return x


class FisherGaussianMove(MHMove):
    """Gaussian proposal with the Fisher covariance (Cholesky factor), scale in
    {0.3, 1.0} x 1/sqrt(beta). Symmetric -> factors = 0."""
    SCALES = (0.3, 1.0)

    def __init__(self, cov, lo, hi, branch="mbh", **kw):
        super().__init__(**kw)
        self.branch, self.lo, self.hi = branch, lo, hi
        self.set_cov(cov)

    def set_cov(self, cov):
        """(Re)set the proposal covariance (sampling basis). Used for adaptive
        burn-in: the Fisher's floored degenerate directions can be 40x too wide
        (id19 sky), so the empirical ensemble covariance replaces it."""
        self.cov = np.asarray(cov, dtype=float)
        self._L = np.linalg.cholesky(ensure_pd((2.38 ** 2 / NDIM) * self.cov, rel_floor=1e-12))

    def get_proposal(self, branches_coords, random, branches_inds=None, **_):
        coords = branches_coords[self.branch]
        ntemps, nwalkers, nleaves, ndim = coords.shape
        betas = (self.temperature_control.betas
                 if self.temperature_control is not None else np.ones(ntemps))
        tscale = 1.0 / np.sqrt(np.maximum(betas, 1e-10))
        proposed = coords.copy()
        for t in range(ntemps):
            sc = random.choice(self.SCALES, size=nwalkers) * tscale[t]
            step = (random.standard_normal((nwalkers, ndim)) @ self._L.T) * sc[:, None]
            proposed[t, :, 0, :] = coords[t, :, 0, :] + step
        # Periodic wrap only. Do NOT reflect at the prior box: reflecting a
        # correlated Gaussian at a rectangular boundary is not a symmetric
        # proposal in general (see cd1l_roulet.propose), so factors=0 would be
        # wrong there. Out-of-prior proposals are rejected by the prior.
        for j, per in PERIODIC.items():
            proposed[..., j] %= per
        return {self.branch: proposed}, np.zeros((ntemps, nwalkers))


class RouletSpinGaussianMove(FisherGaussianMove):
    """Fixed Gaussian in unit-Jacobian spin coordinates; chains remain MC.

    Prior violations are self-transitions, not reflected Gaussian draws.
    This is a spin proposal only, not sky/phase folding.
    """

    def __init__(self, cov, reference, lo, hi, proposal_scale=1.0, **kw):
        self.reference = np.asarray(reference, dtype=float)
        super().__init__(cov, lo, hi, **kw)
        self.proposal_scale = float(proposal_scale)
        self.attempted = self.prior_valid = 0

    def set_cov(self, cov):
        """Covariance is given in the sampling basis; transport it to the
        Roulet spin coordinates at the reference point."""
        from cd1l_roulet import transport_covariance
        super().set_cov(transport_covariance(cov, self.reference))

    def get_proposal(self, branches_coords, random, branches_inds=None, **_):
        from cd1l_roulet import propose
        coords = branches_coords[self.branch]
        ntemps, nwalkers, nleaves, ndim = coords.shape
        if nleaves != 1 or ndim != NDIM:
            raise ValueError("Roulet spin move requires one 11-D leaf")
        betas = (self.temperature_control.betas
                 if self.temperature_control is not None else np.ones(ntemps))
        proposed = coords.copy()
        for t in range(ntemps):
            scale = random.choice(self.SCALES, size=nwalkers) / np.sqrt(max(betas[t], 1e-10))
            steps = (random.standard_normal((nwalkers, ndim)) @ self._L.T) * scale[:, None]
            steps *= self.proposal_scale
            proposed[t, :, 0, :], valid = propose(
                coords[t, :, 0, :], steps, self.lo, self.hi, PERIODIC)
            self.attempted += nwalkers
            self.prior_valid += int(valid.sum())
        return {self.branch: proposed}, np.zeros((ntemps, nwalkers))


class SkyPartnerGibbsMove(MHMove):
    """Exact Gibbs draw over the 8 sky partners of each walker: one batched
    likelihood call on 8N rows, resample the partner index prop. to
    exp(beta LL); factors cancel Eryn's beta dlogp so detailed balance holds."""

    def __init__(self, loglike, lo, hi, branch="mbh", **kw):
        super().__init__(**kw)
        self.loglike, self.lo, self.hi, self.branch = loglike, lo, hi, branch

    def get_proposal(self, branches_coords, random, branches_inds=None, **_):
        coords = branches_coords[self.branch]
        ntemps, nwalkers, nleaves, ndim = coords.shape
        flat = coords[:, :, 0, :].reshape(-1, ndim)
        N = flat.shape[0]
        partners = sky_partners(flat)                              # (N, 8, ndim)
        inside = np.all((partners >= self.lo) & (partners <= self.hi), axis=-1)
        rows = partners.reshape(-1, ndim)
        ll = np.concatenate([np.asarray(self.loglike(rows[k:k + 64])).ravel()
                             for k in range(0, len(rows), 64)]).reshape(N, 8)
        ll = np.where(inside, ll, -np.inf)
        betas = (self.temperature_control.betas
                 if self.temperature_control is not None else np.ones(ntemps))
        bvec = np.repeat(betas, nwalkers)
        logw = bvec[:, None] * ll
        logw = np.where(np.isfinite(logw), logw, -np.inf)
        mx = np.max(logw, axis=1, keepdims=True); mx = np.where(np.isfinite(mx), mx, 0.0)
        w = np.exp(logw - mx)
        bad = w.sum(axis=1) <= 0
        w[bad] = 0.0; w[bad, 0] = 1.0
        w /= w.sum(axis=1, keepdims=True)
        chosen = np.array([random.choice(8, p=w[r]) for r in range(N)])
        new = partners[np.arange(N), chosen]
        factors = bvec * (ll[:, 0] - ll[np.arange(N), chosen])
        # A walker whose current log L is -inf (e.g. a phentax-rejected prior
        # draw) gives (-inf) - (-inf) = NaN here and poisons its state. Any
        # finite partner is an improvement from -inf: let Eryn accept it.
        factors = np.where(np.isfinite(factors), factors, 0.0).reshape(ntemps, nwalkers)
        return {self.branch: new.reshape(ntemps, nwalkers, nleaves, ndim)}, factors


MTOT_MAX = 1e8      # stock prior bound on the TOTAL mass; phentax fails well above it


from eryn.priors.probdist import ProbDistContainer


class MassConstrainedPrior(ProbDistContainer):
    """Normalized box prior conditioned on total mass <= MTOT_MAX."""

    def __init__(self, *args, **kwargs):
        from scipy.integrate import quad
        super().__init__(*args, **kwargs)
        lm, q = self.priors_in[0], self.priors_in[1]
        def fraction(r):
            upper = np.log(MTOT_MAX) + .6 * np.log(r) - 1.2 * np.log1p(r)
            return np.clip((upper - lm.minimum) / (lm.maximum - lm.minimum), 0., 1.)
        self.log_fraction = np.log(quad(fraction, q.minimum, q.maximum,
                                       epsabs=1e-11)[0] / (q.maximum - q.minimum))

    def logpdf(self, x, keys=None, **kwargs):
        if keys is not None:
            raise ValueError("MassConstrainedPrior requires the full joint prior")
        value = super().logpdf(x, **kwargs)
        ok = mtot_ok(x)
        if np.asarray(x).ndim == 1:
            ok = ok[0]
        return np.where(ok, value - self.log_fraction, -np.inf)

    def rvs(self, size=1, keys=None, **kwargs):
        if keys is not None:
            raise ValueError("MassConstrainedPrior requires joint draws")
        result = super().rvs(size=size, **kwargs)
        flat = result.reshape(-1, result.shape[-1])
        for _ in range(100):
            bad = ~mtot_ok(flat)
            if not bad.any():
                return result
            flat[bad] = super().rvs(size=int(bad.sum()), **kwargs)
        raise RuntimeError("Mass prior rejection sampling exhausted attempts")


def mtot_ok(rows):
    """Rows whose total mass is below MTOT_MAX (chirp-mass basis)."""
    rows = np.atleast_2d(rows)
    m1, m2 = mc_q_to_m1_m2(rows[..., I["lnMc"]], rows[..., I["q"]])
    return (m1 + m2) <= MTOT_MAX


class PriorDrawMove(MHMove):
    """Independent uniform draw inside [lo, hi] (optionally within a constraint
    region, e.g. M_tot <= 1e8, redrawn until satisfied). The proposal is a fixed
    distribution, uniform on the region, so factors = 0 between in-region
    states; a current state outside the region is unphysical anyway."""

    def __init__(self, lo, hi, branch="mbh", constraint=None, **kw):
        super().__init__(**kw)
        self.lo, self.hi, self.branch, self.constraint = lo, hi, branch, constraint

    def get_proposal(self, branches_coords, random, branches_inds=None, **_):
        coords = branches_coords[self.branch]
        ntemps, nwalkers, nleaves, ndim = coords.shape
        new = self.lo + (self.hi - self.lo) * random.uniform(size=coords.shape)
        if self.constraint is not None:
            flat = new.reshape(-1, ndim)
            bad = ~self.constraint(flat)
            for _ in range(50):
                if not bad.any(): break
                flat[bad] = self.lo + (self.hi - self.lo) * random.uniform(size=(int(bad.sum()), ndim))
                bad = ~self.constraint(flat)
            new = flat.reshape(coords.shape)
            if bad.any():
                raise RuntimeError("Constrained prior draw exhausted rejection attempts")
        return {self.branch: new}, np.zeros((ntemps, nwalkers))
