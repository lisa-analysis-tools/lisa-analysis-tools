"""Fast window convolution for :class:`~lisatools.sensitivity.XYZSensitivityBackend`.

With ``convolve_window=True`` the backend convolves every covariance entry with the
window's spectral kernel ``|W|^2``. Done naively that is 12 length-N FFTs and six
full-grid arrays per walker. The covariance is however structured (``PSD.cu``,
``get_noise_covariance_single``)::

    C_k(f, t) = Soms^2 T^oms_k(f, t) s_oms(f) 10^spl_oms(f)
              + Sa^2   T^tm_k(f, t)  s_tm(f)  10^spl_tm(f)
              + R_k(t) G(f; theta_gal)
    (all times noise_normalization, plus alias copies of each term)

and the convolution is linear, which gives:

* instrument noise, no splines -- exact: ``conv(C) = sum_b coeff_b B_b`` with
  walker-independent convolved bases ``B_b`` built once (2 bases, 12 if asymmetric);
* galactic foreground -- exact: ``conv(R_k G) = R_k conv(G)``, one real convolution of
  a band-limited row per walker, done on a short circle with a truncated kernel;
* splines -- no exact separation. ``mode="exact"`` falls back to the full per-walker
  convolution; ``mode="factorized"`` uses ``conv(m T) ~= m conv(T)`` for the smooth
  per-walker factors, with a measured error budget.

All functions take the backend as first argument and keep no state of their own;
the walker-independent arrays are cached on the backend (``_wconv_*`` attributes)
and shared by reference across walker copies.
"""

from __future__ import annotations

import numpy as np

WINDOW_CONVOLUTION_MODES = ("exact", "factorized", "legacy")


# --- instrument bases + fused likelihood ---


#: order of the 6 covariance entries in the bases tuple
INST_ENTRY_ORDER = ("00", "11", "22", "01", "02", "12")
#: order of the 6 entries in ``gal_ratios`` (= the C++ ``gal_R_avg[t * 6 + k]`` layout)
GAL_RATIO_ORDER = ("00", "01", "02", "11", "12", "22")
# position in GAL_RATIO_ORDER of each entry of INST_ENTRY_ORDER
_FUSED_GAL_K = tuple(GAL_RATIO_ORDER.index(e) for e in INST_ENTRY_ORDER)


def _inst_one_hot(backend):
    """One-hot amplitude rows ``(Soms, Sa)``, one row per basis, in basis order."""
    xp = backend.xp
    n = backend.n_noise_par
    if n == 1:
        return xp.array([1.0, 0.0]), xp.array([0.0, 1.0])
    eye, zero = xp.eye(n), xp.zeros((n, n))
    return xp.concatenate([eye, zero]), xp.concatenate([zero, eye])


def _inst_zero_knots(backend, n_knots: int = 8):
    """Spline knots for one PSD with zero amplitude everywhere: ``10**spline == 1``."""
    xp = backend.xp
    f = backend.f_arr
    fpos = f[f > 0]
    lo = float(xp.log10(fpos[0])) - 0.1
    hi = float(xp.log10(fpos[-1])) + 0.1
    pos = xp.ascontiguousarray(xp.broadcast_to(xp.linspace(lo, hi, n_knots), (2, 1, n_knots)))
    return pos, xp.zeros_like(pos)


def _inst_build_bases(backend, convolve_window: bool):
    xp = backend.xp
    Soms, Sa = _inst_one_hot(backend)
    n_basis = Soms.shape[0]
    total = backend.basis_settings.total_terms
    freqs = backend.f_arr
    if not convolve_window and len(freqs) > 1 and float(freqs[0]) == 0.0:
        # unconvolved bases mirror psd_likelihood_xyz_kernel's inline path, which
        # evaluates the f = 0 bin at f_arr[1] (grid index / normalization stay 0)
        freqs = freqs.copy()
        freqs[0] = freqs[1]
    out = [
        xp.empty((n_basis, total), dtype=xp.float64 if i < 3 else xp.complex128)
        for i in range(6)
    ]
    kp = None
    if backend.use_splines:
        kp, _ = _inst_zero_knots(backend)
    # one basis per call: bounds the one-off FFT scratch to a single row (batching the
    # bases would grab up to half the free device memory, which the cupy pool keeps)
    for b0 in range(n_basis):
        b1 = b0 + 1
        n = b1 - b0
        z = xp.zeros(n, dtype=xp.float64)  # Amp = 0: no galaxy in the bases
        kpb = kab = None
        if kp is not None:
            kpb = xp.ascontiguousarray(xp.broadcast_to(kp, (2, n, kp.shape[-1])))
            kab = xp.zeros_like(kpb)
        els = backend._compute_matrix_elements(
            freqs, Soms[b0:b1], Sa[b0:b1], z, z, z, z, z,
            kpb, kab, run_async=False, convolve_window=convolve_window,
        )
        for i, e in enumerate(els):
            e = e.reshape(n, total)
            out[i][b0:b1] = e.real if i < 3 else e
        del els
    return tuple(out)


def get_instrument_bases(backend, convolve_window: bool = True):
    """Instrument-noise bases, built once and cached on ``backend``.

    Returns ``(B00, B11, B22, B01, B02, B12)`` (:data:`INST_ENTRY_ORDER`), each a
    contiguous ``(n_basis, total_terms)`` array (float64 for the diagonal, complex128
    otherwise), such that for coefficients ``coeffs`` ``(num_psds, n_basis)`` from
    :func:`basis_coeffs` the (convolved) instrument covariance of PSD ``i`` is
    ``coeffs[i] @ B``. This is exact: the model is linear in ``Soms**2`` / ``Sa**2``
    (``PSD.cu`` ``get_noise_covariance_single`` + alias loop) and so is
    ``_convolve_matrix_components`` (edge padding, f[0] guard).

    Basis order: symmetric ``[OMS, TM]`` (``n_basis = 2``); asymmetric
    ``[OMS_m for m in MOSAs] + [TM_m for m in MOSAs]`` (``n_basis = 12``), i.e. basis
    ``b`` belongs to noise type ``b // n_noise_par`` (0 = OMS, 1 = TM).

    The bases are built through ``backend._compute_matrix_elements`` with one-hot
    amplitude rows and ``Amp = alpha = f_1 = kn = f_2 = 0`` (the galactic term is then
    exactly 0 for f > 0; at f = 0 it is NaN like the instrument term, and the f[0]
    guard of the convolution replaces that bin as in the legacy path). With
    ``use_splines`` the bases carry a zero spline (``10**0 = 1``).

    Cached as ``backend._wconv_inst_bases`` (convolved) or
    ``backend._wconv_inst_bases_raw`` (``convolve_window=False``; there the f = 0 bin is
    evaluated at ``f_arr[1]`` like the kernel's inline path). Memory: ``n_basis * 72 B``
    per bin (2 yr / 5 s: 0.9 GB symmetric, 5.4 GB asymmetric). Both attributes must be
    shared by reference in ``XYZSensitivityBackend.__deepcopy__``.
    """
    attr = "_wconv_inst_bases" if convolve_window else "_wconv_inst_bases_raw"
    bases = backend.__dict__.get(attr)
    if bases is None:
        bases = _inst_build_bases(backend, convolve_window)
        setattr(backend, attr, bases)
    return bases


def basis_coeffs(backend, Soms, Sa):
    """Basis coefficients ``(num_psds, n_basis)``: squared amplitudes in basis order.

    ``Soms`` / ``Sa`` as for ``compute_log_like``: ``(num_psds,)`` (or scalar) for the
    symmetric model, ``(num_psds, 6)`` (or ``(6,)``) for the asymmetric one.
    """
    xp = backend.xp
    n = backend.n_noise_par
    Soms = xp.asarray(Soms, dtype=xp.float64)
    Sa = xp.asarray(Sa, dtype=xp.float64)
    if n == 1:
        Soms, Sa = xp.broadcast_arrays(Soms.reshape(-1), Sa.reshape(-1))
        return xp.ascontiguousarray(xp.stack([Soms * Soms, Sa * Sa], axis=1))
    Soms, Sa = xp.broadcast_arrays(Soms.reshape(-1, n), Sa.reshape(-1, n))
    return xp.ascontiguousarray(xp.concatenate([Soms * Soms, Sa * Sa], axis=1))


def _fused_prepare(backend, data, data_index, coeffs, gal_conv, gal_ratios, spline_weights, bases):
    xp = backend.xp
    if bases is None:
        bases = get_instrument_bases(backend)
    n_basis, total = bases[0].shape
    num_freqs, num_times = int(backend.num_freqs), int(backend.num_times)
    if total != num_freqs * num_times:
        raise ValueError(f"bases have {total} terms, expected num_times*num_freqs={num_times * num_freqs}")
    coeffs = xp.ascontiguousarray(xp.asarray(coeffs, dtype=xp.float64))
    if coeffs.ndim != 2 or coeffs.shape[1] != n_basis:
        raise ValueError(f"coeffs must be (num_psds, {n_basis}); got {coeffs.shape}")
    num_psds = coeffs.shape[0]
    data = xp.ascontiguousarray(xp.asarray(data, dtype=xp.complex128)).reshape(-1)
    if data.size % (3 * total):
        raise ValueError(f"data size {data.size} is not a multiple of 3 * {total}")
    data_index = xp.ascontiguousarray(xp.asarray(data_index, dtype=xp.int32)).reshape(-1)
    if data_index.size != num_psds:
        raise ValueError(f"data_index has {data_index.size} entries for {num_psds} PSDs")
    if gal_conv is not None:
        gal_conv = xp.asarray(gal_conv, dtype=xp.float64)
        if gal_conv.ndim == 2:  # (num_psds, n_gal): same row for every time
            gal_conv = gal_conv[:, None, :]
        if gal_conv.shape[0] != num_psds or gal_conv.shape[1] not in (1, num_times):
            raise ValueError(f"gal_conv must be (num_psds, [num_times,] n_gal); got {gal_conv.shape}")
        if gal_conv.shape[-1] > num_freqs:
            raise ValueError("gal_conv is longer than the frequency grid")
        if gal_ratios is None:
            raise ValueError("gal_conv needs gal_ratios (num_times, 6)")
        gal_ratios = xp.ascontiguousarray(xp.asarray(gal_ratios, dtype=xp.float64)).reshape(num_times, 6)
    if spline_weights is not None:
        spline_weights = xp.asarray(spline_weights, dtype=xp.float64)
        if spline_weights.shape != (num_psds, 2, num_freqs):
            raise ValueError(f"spline_weights must be (num_psds, 2, num_freqs); got {spline_weights.shape}")
    return bases, coeffs, data, data_index, gal_conv, gal_ratios, spline_weights, num_freqs, num_times


def _require_compiled(backend, name: str):
    """The compiled method ``name`` of the backend's sensitivity matrix (``PSD.cu``)."""
    fn = getattr(getattr(backend, "pycpp_sensitivity_matrix", None), name, None)
    if fn is None:
        raise RuntimeError(
            f"the lisatools binary has no XYZSensitivityMatrix.{name}: rebuild lisatools "
            "(the window convolution needs the PSD.cu kernels)"
        )
    return fn


def _fused_ll_compiled(backend, bases, coeffs, data, data_index, gal_conv, gal_ratios, spl, num_freqs, num_times):
    """``XYZSensitivityMatrix::psd_likelihood_basis_wrap`` (``PSD.cu``, CPU and GPU)."""
    xp = backend.xp
    num_psds, n_basis = coeffs.shape
    total = num_freqs * num_times
    empty = xp.empty(0, dtype=xp.float64)
    gal_args = (False, empty, empty, 0, 0, 0)
    if gal_conv is not None:
        es = gal_conv.itemsize
        if gal_conv.strides[-1] != es:
            gal_conv = xp.ascontiguousarray(gal_conv)
        gal_args = (
            True, gal_conv, gal_ratios, int(gal_conv.shape[-1]), gal_conv.strides[0] // es,
            0 if gal_conv.shape[1] == 1 else gal_conv.strides[1] // es,
        )
    spl_args = (False, empty, 0, 0, 0)
    if spl is not None:
        es = spl.itemsize
        spl_args = (True, spl, *(st // es for st in spl.strides))
    bases = [xp.ascontiguousarray(B) for B in bases]  # no-op for get_instrument_bases
    out = xp.empty(num_psds, dtype=xp.float64)
    _require_compiled(backend, "psd_likelihood_basis_wrap")(
        out, data, data_index, coeffs, *bases, *gal_args, *spl_args,
        xp.ascontiguousarray(backend.dips_mask, dtype=bool).reshape(-1),
        float(backend.basis_settings.differential_component),
        num_freqs, num_times, num_psds, data.size // (3 * total), n_basis,
    )
    return out


def fused_log_like(
    backend, data, data_index, coeffs, gal_conv=None, gal_ratios=None, spline_weights=None,
    bases=None,
):
    """Log-likelihood per PSD from the instrument bases (+ galaxy rows, + spline weights).

    Per PSD ``p`` and bin ``idx = t * num_freqs + f``::

        c_k = sum_b coeffs[p, b] * w_b(p, f) * B_k[b, idx]
              + gal_ratios[t, k] * gal_conv[p, t, f]        (f < gal_conv.shape[-1])

    then exactly the algebra of ``psd_likelihood_xyz_kernel`` (``PSD.cu``): dip-mask
    skip, ``invert_3x3_hermitian``, skip on ``det == 0`` / non-finite, quadratic form,
    skip on non-finite / negative ``Q``, ``term = -0.5 * 4 df Q - log|det|``, Kahan sums;
    computed by ``XYZSensitivityMatrix::psd_likelihood_basis_wrap`` (CPU and GPU) without
    per-PSD full-length temporaries.

    Args:
        data: complex data, flat layout ``(num_data, 3, num_times * num_freqs)``, as
            ``compute_log_like``'s ``data_in_all``.
        data_index: ``(num_psds,)`` PSD -> data block.
        coeffs: ``(num_psds, n_basis)`` from :func:`basis_coeffs`.
        gal_conv: optional ``(num_psds, num_times, n_gal)`` or ``(num_psds, n_gal)``
            real galaxy rows on the active grid (index 0 = ``f_arr[0]``), in units of
            the XX entry; bins ``f >= n_gal`` get no galaxy.
        gal_ratios: ``(num_times, 6)`` real ratios ``R_k / R_00`` in
            :data:`GAL_RATIO_ORDER` (``[00, 01, 02, 11, 12, 22]``, the ``gal_R_avg``
            layout); required with ``gal_conv``. Off-diagonal galaxy terms are real.
        spline_weights: optional ``(num_psds, 2, num_freqs)`` MULTIPLICATIVE weights
            ``10**spline`` (row 0 OMS, row 1 TM; any strides, no copy), applied to the
            bases of noise type ``b // n_noise_par``. Per frequency only (as the C++).
        bases: override the bases (default: :func:`get_instrument_bases`, convolved).

    Returns:
        ``(num_psds,)`` float64 on the backend's array module.
    """
    (bases, coeffs, data, data_index, gal_conv, gal_ratios, spl,
     num_freqs, num_times) = _fused_prepare(
        backend, data, data_index, coeffs, gal_conv, gal_ratios, spline_weights, bases
    )
    return _fused_ll_compiled(
        backend, bases, coeffs, data, data_index, gal_conv, gal_ratios, spl, num_freqs, num_times
    )


# --- exact galactic foreground ---


def get_galaxy_setup(backend, galaxy_fmax=None, kernel_tol=None, tail_tol=None, batch_size=None):
    """Walker-independent data of the exact galaxy convolution, cached on
    ``backend._wconv_gal_setup`` (rebuilt when any argument or the window kernel changes).

    The galactic entries are ``C_k(f, t) = R_k(t) G_w(f)`` with one real per-walker row
    ``G_w`` (``S_gal * common_tdi * noise_normalization`` + alias copies, ``PSD.cu``), so
    the convolved entries are ``(R_k / R_00) * conv(R_00 G_w)``. Built here:

    * ``ratios`` ``(num_times, 6)``: ``R_k(t) / R_00(t)`` in :data:`GAL_RATIO_ORDER`
      (the ``gal_R_avg`` layout), from a galaxy-only fiducial evaluation of the compiled
      kernel (checked f-independent to round-off); ``rho`` ``(num_times,)``:
      ``R_00(t) / R_00(t_0)``;
    * ``guard``: the legacy f[0] guard decision (``_convolve_matrix_components`` copies
      column 1 into column 0 of every entry when any column-0 value of the TOTAL
      covariance is non-finite; decided from a fiducial total evaluation on ``f_arr[:2]``
      -- it fires whenever ``f_arr[0] == 0``);
    * ``K``: kernel half-width such that the two-sided tail mass of ``window_kernel``
      beyond ``|m| > K`` is ``<= kernel_tol`` (default ``1e-17``) of its total;
    * the convolution plan (``n_sup``, ``M``, ``n_gal``, kernel FFT on the circle).

    Support. ``galaxy_fmax`` (default: ``getattr(backend, "galaxy_fmax", None)``):
    ``None`` -> ``n_sup = n_active``: the row is evaluated on the whole active band and
    the result is exact by construction (padding above with the last value as the
    legacy, handled as a constant ``G_hi * sum(kernel)``). A float -> the row is
    evaluated only on the active bins below ``galaxy_fmax`` and treated as zero above
    (padding included). Every call checks, per walker, an analytic upper bound of
    ``|G_w|`` over the dropped bins (direct and alias terms, see
    :func:`_gal_tail_bound`) against ``tail_tol * max|G_w|`` (default ``1e-17``); if any
    walker fails, the whole call falls back to the full-band plan (warning).

    Circle. The padded row minus its constant is supported on two-sided ``|j| <= J =
    lo + n_sup - 1`` (``lo`` = first active bin of the full one-sided grid); with the
    kernel truncated to ``|m| <= K``, a circular convolution of length
    ``M >= 2 (J + K) + 1`` reproduces the linear one on the output columns. ``M`` is the
    next 5-smooth size (fixed, so a single FFT plan per batch size); if ``M >= N`` the
    full circle and untruncated kernel are used (still 1 real FFT pair per walker instead
    of 12). Output columns: active ``[0, n_gal)``, ``n_gal = min(n_active, n_sup + K)``;
    beyond ``n_gal`` the convolved galaxy is ``<= kernel_tol * max|G|``.

    Row evaluation. ``G_w`` comes from ``XYZSensitivityMatrix::get_galactic_row_wrap``
    (``PSD.cu``): bitwise the XX entry of a galaxy-only covariance evaluation, without
    the instrument transfer functions the full covariance kernel would compute (and
    discard) on every bin.

    Walkers are processed in fixed-size batches (``batch_size``, default 8; smaller
    calls are rounded up to a power of two), the last one padded with copies of a real
    walker, so cuFFT plans are reused. Scratch per walker in a batch ~ ``8 n_sup +
    24 M`` bytes (+ the output ``8 n_gal`` per walker).

    Returned dict keys: ``N, lo, hi, n_active, num_times, K, kernel_tol, tail_tol,
    batch_size, galaxy_fmax, guard, ratios, rho, ratio_spread, ksum, plan`` (``n_sup,
    M, n_gal, kfft, full``), ``full_plan`` (lazy fallback), ``bound``.
    """
    xp = backend.xp
    if galaxy_fmax is None:
        galaxy_fmax = getattr(backend, "galaxy_fmax", None)
    kernel_tol = _GAL_KERNEL_TOL if kernel_tol is None else float(kernel_tol)
    tail_tol = _GAL_TAIL_TOL if tail_tol is None else float(tail_tol)
    batch_size = _GAL_BATCH if batch_size is None else int(batch_size)
    kern = backend.window_kernel
    key = (galaxy_fmax, kernel_tol, tail_tol, batch_size, id(kern), int(kern.shape[0]))
    st = getattr(backend, "_wconv_gal_setup", None)
    if st is not None and st["key"] == key:
        return st
    if getattr(backend, "_galactic_grid", None) is None:
        raise ValueError("galaxy_convolved_rows needs the galactic grid (galactic_grid_kwargs)")

    N = int(kern.shape[0])
    nf_full = N // 2 + 1
    sl = backend.active_slice
    lo, hi = sl.start or 0, nf_full if sl.stop is None else sl.stop
    n_active = int(backend.basis_settings.basis_shape_active[-1])
    if hi - lo != n_active:
        raise ValueError(f"active slice [{lo}, {hi}) does not match nf_active={n_active}")
    num_times = len(backend.time_indices)
    if int(backend.basis_settings.total_terms) != num_times * n_active:
        raise ValueError("total_terms != num_times * nf_active: unsupported row layout")
    f_arr = xp.asarray(backend.f_arr, dtype=xp.float64)

    # kernel truncation from the two-sided tail mass (kernel is real and even)
    half = _gal_host(kern[:nf_full]).astype(np.float64)
    tot = float(_gal_host(kern).sum())
    above = np.cumsum(half[::-1])[::-1]  # above[m] = sum_{j >= m} half[j]
    tail = 2.0 * np.concatenate([above[1:], [0.0]]) / tot  # mass beyond |m| > K
    K = int(np.argmax(tail <= kernel_tol))

    # fiducial galaxy-only evaluation: ratios R_k / R_00 (and R_00(t) / R_00(t0))
    n_fid = min(n_active, 256)
    fid = [xp.full(1, v, dtype=xp.float64) for v in _GAL_FIDUCIAL]
    ent = _gal_eval(backend, f_arr[:n_fid], fid, backend.time_indices)  # 6 x (1, nt, n_fid)
    ent = [_gal_host(e)[0] for e in ent]  # (nt, n_fid) each, GAL_RATIO_ORDER
    c00 = ent[0]
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(c00).all(axis=0) & (np.abs(c00) > 0).all(axis=0)
        if ok.any():  # bins where the galaxy is well above its own round-off
            ok &= np.abs(c00).min(axis=0) > 1e-3 * np.abs(c00[:, ok]).max()
    if not ok.any():
        raise RuntimeError("galaxy setup: no usable fiducial bin to measure R_k / R_00")
    ratios = np.empty((num_times, 6))
    spread = 0.0
    for k, e in enumerate(ent):
        r = e[:, ok] / c00[:, ok]
        ratios[:, k] = np.median(r, axis=1)
        spread = max(spread, float(np.max(np.abs(r - ratios[:, k : k + 1]))))
    if spread > 1e-13 * max(1.0, float(np.abs(ratios).max())):
        raise RuntimeError(f"galaxy entries are not R_k(t) * G(f): ratio spread {spread:.3e}")
    rho = np.median(c00[:, ok] / c00[:1, ok], axis=1)

    # legacy f[0] guard decision, from the TOTAL covariance (instrument + galaxy)
    tot_ent = _gal_eval(
        backend, f_arr[: min(2, n_active)], fid, backend.time_indices[:1], instrument=True
    )
    guard = not all(bool(np.isfinite(_gal_host(e)[..., 0]).all()) for e in tot_ent)

    # support
    if galaxy_fmax is None:
        n_sup = n_active
    else:
        n_sup = int(xp.searchsorted(f_arr, float(galaxy_fmax), side="left"))
        n_sup = min(max(n_sup, min(2, n_active)), n_active)
    st = dict(
        key=key, N=N, lo=lo, hi=hi, n_active=n_active, num_times=num_times, K=K,
        kernel_tol=kernel_tol, tail_tol=tail_tol, batch_size=batch_size,
        galaxy_fmax=galaxy_fmax, guard=guard, ratio_spread=spread,
        ratios_host=ratios, rho_host=rho,
        ratios=xp.ascontiguousarray(xp.asarray(ratios)), rho=xp.asarray(rho),
        trivial_rho=bool(num_times == 1),
        ksum=float(_gal_host(backend._window_kernel_fft_real[:1])[0]),
        plan=_gal_plan(backend, n_sup, K, lo, n_active, N),
        full_plan=None,
        bound=_gal_bound_setup(backend, n_sup, f_arr) if n_sup < n_active else None,
    )
    backend._wconv_gal_setup = st
    return st


def galaxy_convolved_rows(backend, Amp, alpha, f_1, kn, f_2, setup=None):
    """Exactly window-convolved galactic foreground, one real row per walker.

    Returns ``(gal_conv, gal_ratios)``:

    * ``gal_conv`` ``(num_psds, num_times, n_gal)`` float64 (backend array module): the
      convolved XX galactic entry on the active columns ``[0, n_gal)`` (index 0 =
      ``f_arr[0]``); zero (to ``kernel_tol`` of max) beyond. **``n_gal`` is
      ``gal_conv.shape[-1]``** -- normally ``setup["plan"]["n_gal"]``, but
      ``n_active`` when a walker failed the support bound and the call fell back to
      the full-band plan.
    * ``gal_ratios`` ``(num_times, 6)`` float64: ``R_k(t) / R_00(t)`` in
      :data:`GAL_RATIO_ORDER` (``00, 01, 02, 11, 12, 22``: the ``gal_R_avg`` layout).
      Convolved galactic entry ``k`` = ``gal_ratios[t, k] * gal_conv[p, t]`` (cross
      terms real).

    This equals the galaxy part of ``_convolve_matrix_components`` (same padding with
    the edge values, same f[0] guard, same full-circle kernel up to the ``kernel_tol``
    tail) to round-off: one short real FFT pair per walker instead of 12 length-N FFTs.
    Parameters are ``(num_psds,)`` arrays or scalars. See :func:`get_galaxy_setup`.
    """
    xp = backend.xp
    st = get_galaxy_setup(backend) if setup is None else setup
    params = [xp.atleast_1d(xp.asarray(p, dtype=xp.float64)).ravel() for p in (Amp, alpha, f_1, kn, f_2)]
    num_psds = max(int(p.shape[0]) for p in params)
    params = [xp.ascontiguousarray(xp.broadcast_to(p, (num_psds,))) for p in params]

    rows, gmax = _gal_rows(backend, st, st["plan"], params, num_psds)
    if st["bound"] is not None:
        # rows were only evaluated below n_sup: check the analytic tail bound
        bound = _gal_tail_bound(backend, st["bound"], params)
        if not bool(xp.all(bound <= st["tail_tol"] * gmax)):
            import warnings

            warnings.warn(
                "galaxy_convolved_rows: galactic foreground not negligible above "
                f"galaxy_fmax={st['galaxy_fmax']} for some walkers; using the full band",
                RuntimeWarning,
            )
            if st["full_plan"] is None:
                st["full_plan"] = _gal_plan(backend, st["n_active"], st["K"], st["lo"], st["n_active"], st["N"])
            rows, _ = _gal_rows(backend, st, st["full_plan"], params, num_psds)

    if st["trivial_rho"]:
        gal_conv = rows[:, None, :]
    else:
        gal_conv = rows[:, None, :] * st["rho"][None, :, None]
    return gal_conv, st["ratios"]


# fiducial galaxy parameters (Amp, alpha, f_1, kn, f_2) for the walker-independent setup
_GAL_FIDUCIAL = (10 ** -43.7, 1.8, 2e-3, 3e-3, 8e-4)
_GAL_KERNEL_TOL = 1e-17
_GAL_TAIL_TOL = 1e-17
_GAL_BATCH = 8


def _gal_host(x):
    """Host numpy copy."""
    get = getattr(x, "get", None)
    return np.asarray(get() if get is not None else x)


def _gal_next_smooth(n: int) -> int:
    """Smallest 2^a 3^b 5^c >= n (FFT-friendly for pocketfft and cuFFT)."""
    best = 1 << max(0, int(n - 1).bit_length())
    p5 = 1
    while p5 < best:
        p35 = p5
        while p35 < best:
            p = p35
            while p < n:
                p *= 2
            best = min(best, p)
            p35 *= 3
        p5 *= 5
    return best


def _gal_eval(backend, freqs, params, time_indices, instrument=False):
    """Covariance entries from the compiled kernel on ``freqs`` (a PREFIX of ``f_arr``:
    the kernel indexes noise_normalization / alias grids by the prefix index), in
    :data:`GAL_RATIO_ORDER`, each ``(num_psds, len(time_indices), len(freqs))``.

    Galaxy only (``Soms = Sa = 0``, zero spline weights) unless ``instrument`` (then
    the fiducial instrument amplitudes are added -- used for the guard decision).
    """
    xp = backend.xp
    n = int(freqs.shape[0])
    nt = int(len(time_indices))
    num_psds = int(params[0].shape[0])
    n_par = backend.n_noise_par
    if instrument:
        Soms = xp.full(num_psds * n_par, 15e-12)
        Sa = xp.full(num_psds * n_par, 3e-15)
    else:
        Soms = xp.zeros(num_psds * n_par)
        Sa = xp.zeros(num_psds * n_par)
    spl0 = xp.zeros(num_psds * n)
    spl1 = xp.zeros(num_psds * n)
    size = num_psds * nt * n
    c00, c11, c22 = (xp.empty(size, dtype=xp.float64) for _ in range(3))
    c01, c02, c12 = (xp.empty(size, dtype=xp.complex128) for _ in range(3))
    backend.pycpp_sensitivity_matrix.get_noise_covariance_wrap(
        xp.ascontiguousarray(freqs), xp.ascontiguousarray(xp.asarray(time_indices, dtype=xp.int32)),
        Soms, Sa, *[xp.ascontiguousarray(p) for p in params], spl0, spl1,
        c00, c01, c02, c11, c12, c22, n, nt, num_psds, False,
    )
    shape = (num_psds, nt, n)
    by_name = {"00": c00, "11": c11, "22": c22, "01": c01.real, "02": c02.real, "12": c12.real}
    return [by_name[k].reshape(shape) for k in GAL_RATIO_ORDER]


def _gal_row(backend, n, params):
    """Unconvolved XX galactic rows ``(B, n)`` on the first ``n`` active bins at the
    first time index (``get_galactic_row_wrap``; NaN at f = 0, as the covariance)."""
    xp = backend.xp
    B = int(params[0].shape[0])
    out = xp.empty((B, n), dtype=xp.float64)
    _require_compiled(backend, "get_galactic_row_wrap")(
        out, xp.ascontiguousarray(xp.asarray(backend.f_arr, dtype=xp.float64)[:n]),
        *[xp.ascontiguousarray(q, dtype=xp.float64) for q in params],
        n, B, int(backend.time_indices[0]),
    )
    return out


def _gal_plan(backend, n_sup, K, lo, n_active, N):
    """Convolution plan for a row supported on active columns ``[0, n_sup)``."""
    xp = backend.xp
    J = lo + n_sup - 1  # two-sided support |j| <= J of (padded row - constant)
    M = _gal_next_smooth(2 * (J + K) + 1)
    n_gal = min(n_active, n_sup + K)
    if M >= N:
        M, kfft = N, backend._window_kernel_fft_real
    else:
        kern = backend.window_kernel
        ks = xp.zeros(M, dtype=xp.float64)
        ks[: K + 1] = kern[: K + 1]
        ks[M - K :] = kern[N - K :]
        kfft = xp.ascontiguousarray(xp.fft.fft(ks).real)
    return dict(n_sup=int(n_sup), M=int(M), n_gal=int(n_gal), kfft=kfft, full=bool(n_sup == n_active))


def _gal_rows(backend, st, plan, params, num_psds):
    """Convolved XX rows ``(num_psds, n_gal)`` at the first time index (batched) and
    ``max_f |G_w|`` ``(num_psds,)`` of the evaluated (unconvolved) rows."""
    xp = backend.xp
    n_sup, M, n_gal = plan["n_sup"], plan["M"], plan["n_gal"]
    B = st["batch_size"]
    if num_psds < B:
        B = 1 << max(0, int(num_psds - 1).bit_length())
    out = xp.empty((num_psds, n_gal), dtype=xp.float64)
    gmax = xp.empty((num_psds,), dtype=xp.float64)
    one = xp.zeros((B, M // 2 + 1), dtype=xp.complex128)
    for s in range(0, num_psds, B):
        n = min(B, num_psds - s)
        idx = xp.minimum(xp.arange(s, s + B), num_psds - 1)  # pad with the last real walker
        p = [q[idx] for q in params]
        G = _gal_row(backend, n_sup, p)  # (B, n_sup) XX at t0
        if st["guard"]:
            G[:, 0] = G[:, 1]
        bad = ~xp.isfinite(G[:, 0])
        if bool(bad.any()):  # legacy would fire the batch-wide guard here as well
            G[:, 0] = xp.where(bad, G[:, 1], G[:, 0])
        gmax[s : s + n] = xp.max(xp.abs(G[:n]), axis=-1)
        out[s : s + n] = _gal_convolve(backend, st, plan, G, one)[:n]
    return out, gmax


def _gal_convolve(backend, st, plan, G, one=None):
    """Window convolution of real rows ``G`` ``(B, n_sup)`` on the active columns
    ``[0, n_sup)`` (already guarded), embedded and padded as the legacy
    ``_convolve_matrix_components``; returns active columns ``[0, n_gal)``.

    padded row = c + D, ``c`` = last active value if the row spans the whole band
    (constant above the band; it convolves to ``c * sum(kernel)``), else 0; ``D`` is
    supported on the two-sided ``|j| < lo + n_sup`` and convolved on the length-``M``
    circle with the (truncated) kernel via ``rfft(irfft(D) * FFT(kernel))``."""
    xp = backend.xp
    lo, n_sup, M, n_gal = st["lo"], plan["n_sup"], plan["M"], plan["n_gal"]
    B = G.shape[0]
    if one is None:
        one = xp.zeros((B, M // 2 + 1), dtype=xp.complex128)
    c = G[:, -1:] if plan["full"] else xp.zeros((B, 1))
    one[:] = 0.0
    one[:, :lo] = G[:, :1] - c
    one[:, lo : lo + n_sup] = G - c
    x = xp.fft.irfft(one, n=M, axis=-1)
    x *= plan["kfft"]
    res = xp.fft.rfft(x, axis=-1)[:, lo : lo + n_gal].real
    if plan["full"]:
        res += c * st["ksum"]  # in place on the (temporary) rfft output
    return res


def _gal_bound_setup(backend, n_sup, f_arr):
    """Walker-independent pieces of the tail bound over the dropped bins ``[n_sup, n_active)``."""
    xp = backend.xp
    n_active = int(f_arr.shape[0])
    f_s = float(f_arr[n_sup])
    norm = backend.noise_normalization
    norm_dir = float(xp.max(xp.abs(norm[n_sup:n_active])))
    alias = []
    n_alias = backend.n_alias
    if n_alias:
        fa = xp.abs(xp.asarray(backend._alias_freqs_flat)).reshape(n_alias, -1)[:, n_sup:]
        na = xp.abs(norm[n_active:]).reshape(n_alias, -1)[:, n_sup:]
        for a in range(n_alias):
            alias.append((float(xp.min(fa[a])), float(xp.max(na[a]))))
    L_over_c = 1.01 * float(backend.orbits.armlength) / 299792458.0  # 1 % margin
    ctdi_c = 16.0 if int(backend.tdi_generation) == 2 else 4.0
    return dict(terms=[(f_s, norm_dir)] + alias, L_over_c=L_over_c, ctdi_c=ctdi_c)


def _gal_tail_bound(backend, bd, params):
    """Per-walker upper bound of ``|G_w(f)|`` over the dropped bins (``PSD.cu``):
    ``S_gal = Amp exp(-(f/f_1)^alpha) f^{-7/3} / (1 + e^{2 (f - kn) / f_2})`` with
    ``exp(-(f/f_1)^alpha) <= 1``, ``1/(1+e^{2u}) <= e^{-2u}``,
    ``common_tdi <= c x^2`` (``x = 2 pi f L / c``; ``c = 16`` TDI 2, ``4`` TDI 1),
    and ``f^{-7/3}`` and ``x^2 e^{-2 (f - kn) / f_2}`` decreasing for ``f > f_2``, so the
    bound at the lowest dropped frequency ``f_s`` (per term: direct, each alias grid)
    times the max normalisation holds over all dropped bins when ``f_s > f_2``;
    otherwise the bound is ``inf``."""
    xp = backend.xp
    Amp, alpha, f_1, kn, f_2 = params
    total = xp.zeros_like(Amp)
    for f_s, nrm in bd["terms"]:
        x = 2.0 * np.pi * f_s * bd["L_over_c"]
        with np.errstate(over="ignore", invalid="ignore"):
            b = xp.abs(Amp) * f_s ** (-7.0 / 3.0) * bd["ctdi_c"] * x * x * nrm * xp.exp(-2.0 * (f_s - kn) / f_2)
        b = xp.where((f_s > f_2) & (f_2 > 0), b, xp.inf)
        total = total + b
    return total


# --- factorized approximation ---


def factorized_spline_weights(backend, knots_position_all, knots_amplitude_all):
    """Per-walker multiplicative spline weights ``10^spl(f)`` for the factorized mode,
    shape ``(num_psds, 2, num_freqs)`` (row 0: OMS, row 1: TM), for
    ``fused_log_like(..., spline_weights=...)``; the mode then uses
    ``C ~= Soms^2 10^spl_oms B_oms + Sa^2 10^spl_tm B_tm`` (approximate; error budget in
    ``scripts/window_conv/agent5_error_budget.md``).
    """
    xp = backend.xp
    nf = backend.num_freqs
    spl = backend.spline_interpolant(xp.log10(backend.f_arr), knots_position_all, knots_amplitude_all)
    w = xp.stack([xp.asarray(x, dtype=xp.float64).reshape(-1, nf) for x in spl[:2]], axis=1)
    return xp.ascontiguousarray(10.0**w)
