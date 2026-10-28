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

from typing import Optional

import numpy as np

WINDOW_CONVOLUTION_MODES = ("exact", "legacy")


# --- instrument bases + fused likelihood (agent 2) ---------------------------------


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


_FUSED_NT = 256  # = NUM_THREADS_LIKE in PSD.cu

_FUSED_SRC = r"""
#include <cupy/complex.cuh>
typedef complex<double> cmplx;
#define NT 256

extern "C" __global__ void fused_psd_like(
    double* like_contrib, const cmplx* __restrict__ data, const int* __restrict__ data_index,
    const double* __restrict__ coeffs, const int n_basis, const int n_noise_par,
    const double* __restrict__ B00, const double* __restrict__ B11, const double* __restrict__ B22,
    const cmplx* __restrict__ B01, const cmplx* __restrict__ B02, const cmplx* __restrict__ B12,
    const int has_gal, const double* __restrict__ gal, const int n_gal,
    const long long gal_psd_stride, const long long gal_t_stride,
    const double* __restrict__ gal_ratios,
    const int has_spl, const double* __restrict__ spl,
    const long long spl_psd_stride, const long long spl_noise_stride, const long long spl_f_stride,
    const bool* __restrict__ dips_mask, const double differential_component,
    const int num_freqs, const int num_times, const int num_psds, const int num_blocks_tf)
{
    __shared__ double like_vals[NT];
    __shared__ double compensation[NT];

    const int tid = threadIdx.x;
    // psd is the fast block index: blocks working on the same (t, f) chunk for
    // different PSDs run together, so the bases are read from DRAM ~once per chunk
    const int psd_i = blockIdx.x % num_psds;
    const int bx = blockIdx.x / num_psds;
    const long long total = (long long)num_times * num_freqs;

    like_vals[tid] = 0.0;
    compensation[tid] = 0.0;

    const long long base_idx = (long long)data_index[psd_i] * 3 * total;
    const double* cf = coeffs + (long long)psd_i * n_basis;

    for (long long idx = (long long)bx * NT + tid; idx < total; idx += (long long)num_blocks_tf * NT)
    {
        if (dips_mask[idx]) continue;
        const int t_idx = (int)(idx / num_freqs);
        const int f_idx = (int)(idx % num_freqs);

        double c00 = 0.0, c11 = 0.0, c22 = 0.0;
        cmplx c01(0.0, 0.0), c02(0.0, 0.0), c12(0.0, 0.0);
        for (int b = 0; b < n_basis; b++) {
            double a = cf[b];
            if (has_spl)
                a *= spl[psd_i * spl_psd_stride + (b / n_noise_par) * spl_noise_stride + f_idx * spl_f_stride];
            const long long o = (long long)b * total + idx;
            c00 += a * B00[o];
            c11 += a * B11[o];
            c22 += a * B22[o];
            c01 += a * B01[o];
            c02 += a * B02[o];
            c12 += a * B12[o];
        }
        if (has_gal && f_idx < n_gal) {
            const double g = gal[psd_i * gal_psd_stride + t_idx * gal_t_stride + f_idx];
            const double* r = gal_ratios + 6 * t_idx;  // [00, 01, 02, 11, 12, 22]
            c00 += r[0] * g;
            c01 += cmplx(r[1] * g, 0.0);
            c02 += cmplx(r[2] * g, 0.0);
            c11 += r[3] * g;
            c12 += cmplx(r[4] * g, 0.0);
            c22 += r[5] * g;
        }

        // invert_3x3_hermitian (PSD.cu)
        const double C00 = c11 * c22 - norm(c12);
        const double C11 = c00 * c22 - norm(c02);
        const double C22 = c00 * c11 - norm(c01);
        const cmplx C01 = c12 * conj(c02) - conj(c01) * c22;
        const cmplx C02 = conj(c01) * conj(c12) - c11 * conj(c02);
        const cmplx C12 = c01 * conj(c02) - c00 * conj(c12);
        const double det = c00 * C00 + (c01 * C01).real() + (c02 * C02).real();
        if (det == 0.0 || !isfinite(det)) continue;
        const double inv_det = 1.0 / det;
        const double i00 = C00 * inv_det, i11 = C11 * inv_det, i22 = C22 * inv_det;
        const cmplx i01 = conj(C01) * inv_det, i02 = conj(C02) * inv_det, i12 = conj(C12) * inv_det;

        const cmplx dX = data[base_idx + idx];
        const cmplx dY = data[base_idx + total + idx];
        const cmplx dZ = data[base_idx + 2 * total + idx];

        // quadratic_form (PSD.cu)
        const cmplx inv_X = i00 * dX + i01 * dY + i02 * dZ;
        const cmplx inv_Y = conj(i01) * dX + i11 * dY + i12 * dZ;
        const cmplx inv_Z = conj(i02) * dX + conj(i12) * dY + i22 * dZ;
        const double Q = (conj(dX) * inv_X + conj(dY) * inv_Y + conj(dZ) * inv_Z).real();
        if (!isfinite(Q) || Q < 0.0) continue;

        const double term = -0.5 * (4.0 * differential_component * Q) - log(fabs(det));
        const double y = term - compensation[tid];
        const double t = like_vals[tid] + y;
        compensation[tid] = (t - like_vals[tid]) - y;
        like_vals[tid] = t;
    }
    __syncthreads();
    for (unsigned int s = NT / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const double y = like_vals[tid + s] - compensation[tid + s] - compensation[tid];
            const double t = like_vals[tid] + y;
            compensation[tid] = (t - like_vals[tid]) - y;
            like_vals[tid] = t;
        }
        __syncthreads();
    }
    if (tid == 0)
        like_contrib[(long long)psd_i * num_blocks_tf + bx] = like_vals[0] - compensation[0];
}

// like_sum_from_contrib (PSD.cu): one block per PSD
extern "C" __global__ void fused_like_sum(double* out, const double* like_contrib,
                                          const int num_blocks_tf, const int num_psds)
{
    __shared__ double shared_sum[NT];
    const int tid = threadIdx.x;
    for (int p = blockIdx.x; p < num_psds; p += gridDim.x) {
        double sum = 0.0;
        for (int i = tid; i < num_blocks_tf; i += blockDim.x)
            sum += like_contrib[(long long)p * num_blocks_tf + i];
        shared_sum[tid] = sum;
        __syncthreads();
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) shared_sum[tid] += shared_sum[tid + s];
            __syncthreads();
        }
        if (tid == 0) out[p] = shared_sum[0];
        __syncthreads();
    }
}
"""

_FUSED_MODULE = None


def _fused_kernels():
    global _FUSED_MODULE
    if _FUSED_MODULE is None:
        import cupy as cp

        _FUSED_MODULE = cp.RawModule(code=_FUSED_SRC, options=("--std=c++17",))
    return _FUSED_MODULE.get_function("fused_psd_like"), _FUSED_MODULE.get_function("fused_like_sum")


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


def _fused_ll_cupy(backend, bases, coeffs, data, data_index, gal_conv, gal_ratios, spl, num_freqs, num_times):
    import cupy as cp

    k_like, k_sum = _fused_kernels()
    num_psds, n_basis = coeffs.shape
    total = num_freqs * num_times
    nblk = -(-total // _FUSED_NT)
    nblk = max(1, min(nblk, (2**31 - 1) // num_psds))
    dummy = cp.zeros(1, dtype=cp.float64)
    if gal_conv is not None:
        es = gal_conv.itemsize
        if gal_conv.strides[-1] != es:
            gal_conv = cp.ascontiguousarray(gal_conv)
        gal_args = (np.int32(1), gal_conv, np.int32(gal_conv.shape[-1]),
                    np.int64(gal_conv.strides[0] // es),
                    np.int64(0 if gal_conv.shape[1] == 1 else gal_conv.strides[1] // es), gal_ratios)
    else:
        gal_args = (np.int32(0), dummy, np.int32(0), np.int64(0), np.int64(0), dummy)
    if spl is not None:
        es = spl.itemsize
        spl_args = (np.int32(1), spl, *(np.int64(s // es) for s in spl.strides))
    else:
        spl_args = (np.int32(0), dummy, np.int64(0), np.int64(0), np.int64(0))
    dips = cp.ascontiguousarray(backend.dips_mask, dtype=cp.bool_)
    contrib = cp.empty(num_psds * nblk, dtype=cp.float64)
    k_like(
        (num_psds * nblk,), (_FUSED_NT,),
        (contrib, data, data_index, coeffs, np.int32(n_basis), np.int32(backend.n_noise_par),
         *bases, *gal_args, *spl_args, dips,
         np.float64(backend.basis_settings.differential_component),
         np.int32(num_freqs), np.int32(num_times), np.int32(num_psds), np.int32(nblk)),
    )
    out = cp.empty(num_psds, dtype=cp.float64)
    k_sum((min(num_psds, 65535),), (_FUSED_NT,), (out, contrib, np.int32(nblk), np.int32(num_psds)))
    return out


def _fused_ll_numpy(backend, bases, coeffs, data, data_index, gal_conv, gal_ratios, spl, num_freqs, num_times):
    import math

    num_psds, n_basis = coeffs.shape
    total = num_freqs * num_times
    data3 = data.reshape(-1, 3, total)
    dips = np.asarray(backend.dips_mask, dtype=bool).reshape(-1)
    dc = float(backend.basis_settings.differential_component)
    which = np.arange(n_basis) // backend.n_noise_par
    partial = [[] for _ in range(num_psds)]
    chunk = max(4096, (1 << 21) // max(num_psds, 1))
    n_gal = 0 if gal_conv is None else gal_conv.shape[-1]
    with np.errstate(all="ignore"):
        for s in range(0, total, chunk):
            sl = slice(s, min(total, s + chunk))
            idx = np.arange(sl.start, sl.stop)
            t_idx, f_idx = idx // num_freqs, idx % num_freqs
            if spl is None:
                c = [coeffs @ B[:, sl] for B in bases]  # (num_psds, n)
            else:
                w = coeffs[:, :, None] * spl[:, which][:, :, f_idx]  # (num_psds, n_basis, n)
                c = [np.einsum("pbn,bn->pn", w, B[:, sl]) for B in bases]
            if n_gal:
                m = f_idx < n_gal
                if m.any():
                    tg = np.minimum(t_idx[m], gal_conv.shape[1] - 1)
                    g = gal_conv[:, tg, f_idx[m]]  # (num_psds, nm)
                    for e, k in enumerate(_FUSED_GAL_K):
                        c[e][:, m] += gal_ratios[t_idx[m], k] * g
            c00, c11, c22, c01, c02, c12 = c
            C00 = c11 * c22 - (c12.real**2 + c12.imag**2)
            C11 = c00 * c22 - (c02.real**2 + c02.imag**2)
            C22 = c00 * c11 - (c01.real**2 + c01.imag**2)
            C01 = c12 * np.conj(c02) - np.conj(c01) * c22
            C02 = np.conj(c01) * np.conj(c12) - c11 * np.conj(c02)
            C12 = c01 * np.conj(c02) - c00 * np.conj(c12)
            det = c00 * C00 + (c01 * C01).real + (c02 * C02).real
            inv_det = 1.0 / det
            i00, i11, i22 = C00 * inv_det, C11 * inv_det, C22 * inv_det
            i01, i02, i12 = np.conj(C01) * inv_det, np.conj(C02) * inv_det, np.conj(C12) * inv_det
            d = data3[:, :, sl][data_index]  # (num_psds, 3, n); copies the chunk only
            dX, dY, dZ = d[:, 0], d[:, 1], d[:, 2]
            Q = (
                np.conj(dX) * (i00 * dX + i01 * dY + i02 * dZ)
                + np.conj(dY) * (np.conj(i01) * dX + i11 * dY + i12 * dZ)
                + np.conj(dZ) * (np.conj(i02) * dX + np.conj(i12) * dY + i22 * dZ)
            ).real
            term = -0.5 * (4.0 * dc * Q) - np.log(np.abs(det))
            ok = (~dips[sl])[None, :] & (det != 0.0) & np.isfinite(det) & np.isfinite(Q) & (Q >= 0.0)
            term = np.where(ok, term, 0.0)
            for p in range(num_psds):
                partial[p].append(float(np.sum(term[p])))  # pairwise within a chunk
    return np.array([math.fsum(v) for v in partial], dtype=np.float64)


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
    skip on non-finite / negative ``Q``, ``term = -0.5 * 4 df Q - log|det|``. GPU: one
    ``cupy.RawKernel`` (per-thread + per-block Kahan sums, then a per-PSD reduction as
    ``like_sum_from_contrib``); no per-PSD full-length temporaries. CPU: numpy over
    frequency chunks (pairwise sums per chunk, ``math.fsum`` across chunks).

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
    if backend.xp is np:
        return _fused_ll_numpy(
            backend, bases, coeffs, data, data_index, gal_conv, gal_ratios, spl, num_freqs, num_times
        )
    return _fused_ll_cupy(
        backend, bases, coeffs, data, data_index, gal_conv, gal_ratios, spl, num_freqs, num_times
    )


# --- exact galactic foreground (agent 3) -------------------------------------------


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

    Row evaluation. ``G_w`` is evaluated directly (``direct=True``): an elementwise
    mirror of the ``PSD.cu`` galaxy formula (``get_galactic_foreground`` x
    ``galaxy_common_tdi_factor`` x ``noise_normalization``, + alias grids;
    ``cupy.ElementwiseKernel`` on GPU) times ``R00`` = ``R_00(t_0)`` (least-squares fit
    to the compiled kernel). The compiled covariance kernel would also compute the
    instrument transfer functions on every bin (~95 % of its cost, then discarded).
    The mirror is validated against the kernel at setup on three parameter sets
    (``direct_mismatch``, band-normalised; measured 0-1.3e-15); above
    ``_GAL_DIRECT_TOL = 1e-13`` the kernel path is used instead (warning).

    Walkers are processed in fixed-size batches (``batch_size``, default 8; smaller
    calls are rounded up to a power of two), the last one padded with copies of a real
    walker, so cuFFT plans are reused. Scratch per walker in a batch ~ ``8 n_sup +
    24 M`` bytes (+ the output ``8 n_gal`` per walker).

    Returned dict keys: ``N, lo, hi, n_active, num_times, K, kernel_tol, tail_tol,
    batch_size, galaxy_fmax, guard, ratios, rho, ratio_spread, ksum, plan`` (``n_sup,
    M, n_gal, kfft, full``), ``full_plan`` (lazy fallback), ``bound``, ``R00``,
    ``direct``, ``direct_mismatch``.
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
    st["R00"], st["direct_mismatch"] = _gal_direct_setup(backend, st)
    st["direct"] = st["direct_mismatch"] <= _GAL_DIRECT_TOL
    if not st["direct"]:
        import warnings

        warnings.warn(
            f"galaxy rows: direct evaluation mismatch {st['direct_mismatch']:.2e} vs the compiled "
            "kernel; falling back to the (slower) kernel evaluation", RuntimeWarning,
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
_GAL_DIRECT_TOL = 1e-13  # direct-vs-kernel row mismatch (band-normalised) accepted at setup
_GAL_DIRECT_SRC = r"""
    // PSD.cu: get_galactic_foreground * galaxy_common_tdi_factor, then
    // c00 = R_00 * S_gal; c00 *= noise_normalization; alias copies added
    double S = Amp * exp(-pow(fa / f_1, alpha)) * pow(fa, -7.0 / 3.0)
               * 0.5 * (1.0 + tanh(-(fa - kn) / f_2));
    double x = (2.0 * M_PI * fa) * L / Clight;
    double t = 4.0 * (x * sin(x)) * (x * sin(x));
    if (gen == 2) t *= 4.0 * sin(2.0 * x) * sin(2.0 * x);
    S *= t;
    double v = (R * S) * nrm;
    out = first ? v : out + v;
"""
_GAL_DIRECT_KERNEL = None


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


def _gal_direct_terms(backend, n):
    """``[(|f|, normalisation)]`` on the first ``n`` active bins: direct grid, then each
    alias grid (``noise_normalization[a * n_grid + f_idx]``), as ``get_noise_covariance``."""
    xp = backend.xp
    n_grid = int(backend.basis_settings.basis_shape_active[-1])
    norm = backend.noise_normalization
    terms = [(xp.asarray(backend.f_arr, dtype=xp.float64)[:n], norm[:n])]
    for a in range(backend.n_alias):
        fa = backend._alias_freqs_flat[a * n_grid : a * n_grid + n]
        terms.append((xp.abs(fa), norm[(a + 1) * n_grid : (a + 1) * n_grid + n]))
    return terms


def _gal_eval_direct(backend, st, n, params, R00=None):
    """XX galactic entry at the first time index, ``(B, n)``, evaluated directly
    (elementwise, the ``PSD.cu`` formula; ``cupy.ElementwiseKernel`` on GPU) instead of
    the full 6-entry covariance kernel -- which spends ~all its time on the instrument
    transfer functions that a galaxy-only call discards. Validated against the kernel
    at setup (``st["direct_mismatch"]``)."""
    global _GAL_DIRECT_KERNEL
    xp = backend.xp
    R = st["R00"] if R00 is None else R00
    L = float(backend.orbits.armlength)
    gen = int(backend.tdi_generation)
    Amp, alpha, f_1, kn, f_2 = [q[:, None] for q in params]
    out = xp.empty((params[0].shape[0], n), dtype=xp.float64)
    for i, (fa, nrm) in enumerate(_gal_direct_terms(backend, n)):
        if xp is np:
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                S = Amp * np.exp(-np.power(fa / f_1, alpha)) * np.power(fa, -7.0 / 3.0) * (
                    0.5 * (1.0 + np.tanh(-(fa - kn) / f_2)))
                x = (2.0 * np.pi * fa) * L / 299792458.0
                t = 4.0 * (x * np.sin(x)) * (x * np.sin(x))
                if gen == 2:
                    t = t * (4.0 * np.sin(2.0 * x) * np.sin(2.0 * x))
                v = (R * (S * t)) * nrm
            out = v if i == 0 else out + v
        else:
            if _GAL_DIRECT_KERNEL is None:
                _GAL_DIRECT_KERNEL = xp.ElementwiseKernel(
                    "float64 fa, float64 nrm, float64 Amp, float64 alpha, float64 f_1, float64 kn, "
                    "float64 f_2, float64 R, float64 L, int32 gen, bool first",
                    "float64 out", _GAL_DIRECT_SRC, "lisatools_wconv_gal_direct",
                    preamble="#define Clight 299792458.\n#ifndef M_PI\n#define M_PI 3.14159265358979323846\n#endif\n",
                )
            _GAL_DIRECT_KERNEL(fa, nrm, Amp, alpha, f_1, kn, f_2, float(R), L, np.int32(gen), i == 0, out)
    return out


def _gal_direct_setup(backend, st):
    """``R_00(t_0)`` (least squares of the compiled kernel's galaxy-only XX against the
    direct shape with ``R = 1``) and the band-normalised max mismatch of the direct
    evaluation vs the kernel on the first ``min(n_active, 4096)`` bins, for the fiducial
    and two more parameter sets."""
    xp = backend.xp
    n = min(st["n_active"], 4096)
    f = xp.asarray(backend.f_arr, dtype=xp.float64)[:n]
    sets = [_GAL_FIDUCIAL, (1e-44, 1.2, 1e-3, 6e-3, 1.5e-3), (1e-43, 2.5, 4e-3, 1.6e-3, 4e-4)]
    params = [xp.asarray([s[k] for s in sets], dtype=xp.float64) for k in range(5)]
    ref = _gal_host(_gal_eval(backend, f, params, backend.time_indices[:1])[0][:, 0, :])
    unit = _gal_host(_gal_eval_direct(backend, st, n, params, R00=1.0))
    ok = np.isfinite(ref) & np.isfinite(unit)
    R00 = float(np.sum(np.where(ok, ref * unit, 0.0)) / np.sum(np.where(ok, unit * unit, 0.0)))
    mine = _gal_host(_gal_eval_direct(backend, st, n, params, R00=R00))
    both_nan = ~np.isfinite(ref) & ~np.isfinite(mine)
    scale = np.max(np.where(ok, np.abs(ref), 0.0), axis=-1, keepdims=True)
    with np.errstate(invalid="ignore"):
        d = np.where(both_nan, 0.0, np.abs(mine - ref) / scale)
    return R00, float(np.max(np.where(np.isnan(d), np.inf, d)))


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
    f_sup = xp.asarray(backend.f_arr, dtype=xp.float64)[:n_sup]
    t0 = backend.time_indices[:1]
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
        if st["direct"]:
            G = _gal_eval_direct(backend, st, n_sup, p)  # (B, n_sup) XX at t0
        else:
            G = _gal_eval(backend, f_sup, p, t0)[0][:, 0, :]
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


# --- factorized approximation (agent 5) --------------------------------------------


#: galaxy split m(f; theta) x T(f) for the factorized mode:
#: "powerlaw" (default) -- the walker-independent f^{-7/3} goes into T, m = Amp exp() tanh();
#: "full"               -- m = S_gal (whole galactic PSD), T = common_tdi * norm.
FACT_GALAXY_SPLITS = ("powerlaw", "full")

_FACT_CLIGHT = 299792458.0  # PSD.hpp Clight


def _fact_galaxy_shape(xp, f, Amp, alpha, f_1, kn, f_2, split="powerlaw"):
    """Per-walker galactic factor ``m`` at frequencies ``f`` (broadcast with the params).

    Mirrors ``NoiseLevels::get_galactic_foreground`` (``PSD.cu``)::

        S_gal = Amp exp(-(f/f_1)^alpha) f^{-7/3} 0.5 (1 + tanh(-(f - kn)/f_2))

    ``split="full"`` returns ``S_gal``; ``split="powerlaw"`` drops the ``f^{-7/3}``.
    """
    m = Amp * xp.exp(-((f / f_1) ** alpha)) * (0.5 * (1.0 + xp.tanh(-(f - kn) / f_2)))
    if split == "full":
        with np.errstate(divide="ignore", invalid="ignore"):
            m = m * f ** (-7.0 / 3.0)
    elif split != "powerlaw":
        raise ValueError(f"unknown galaxy split {split!r}; expected one of {FACT_GALAXY_SPLITS}")
    return m


def _fact_common_tdi(backend, f):
    """``XYZSensitivityMatrix::galaxy_common_tdi_factor`` (PSD.cu) in Python."""
    xp = backend.xp
    x = 2.0 * np.pi * f * backend.orbits.armlength / _FACT_CLIGHT
    t = 4.0 * (x * xp.sin(x)) ** 2
    if backend.tdi_generation == 2:
        t = t * 4.0 * xp.sin(2.0 * x) ** 2
    return t


def _fact_alias_grids(backend):
    """``[(f_a, |f_a|, norm_a)]`` for a = 0 (data grid) and each alias term; ``f_a`` is
    affine in ``f`` with slope +1 (``-(k fs' - f)`` or ``k fs' + f``)."""
    xp = backend.xp
    f = xp.asarray(backend.f_arr, dtype=xp.float64)
    nf = f.size
    norm = xp.asarray(backend.noise_normalization, dtype=xp.float64)
    out = [(f, xp.abs(f), norm[:nf])]
    for a in range(1, backend.n_alias + 1):
        fa = backend._alias_freqs_flat[(a - 1) * nf : a * nf]
        out.append((fa, xp.abs(fa), norm[a * nf : (a + 1) * nf]))
    return out


def _fact_guard_f0(x):
    """Legacy f[0] guard (``_convolve_matrix_components``): copy bin 1 into a
    non-finite bin 0 (f = 0 in the grid)."""
    if x.shape[-1] > 1 and not bool((x[..., 0] == x[..., 0]).all() and (abs(x[..., 0]) < np.inf).all()):
        x = x.copy()
        x[..., 0] = x[..., 1]
    return x


def _fact_zero_knots(backend, num_psds):
    """Knot arrays giving a zero spline (10^0 = 1), or ``(None, None)`` without splines."""
    if not backend.use_splines:
        return None, None
    xp = backend.xp
    f = backend.f_arr[backend.f_arr > 0]
    lo, hi = float(xp.log10(f[0])) - 0.1, float(xp.log10(f[-1])) + 0.1
    pos = xp.tile(xp.linspace(lo, hi, 4)[None, None, :], (2, num_psds, 1))
    return xp.ascontiguousarray(pos), xp.zeros_like(pos)


def _fact_convolve(backend, rows, moment: int = 0, pad: str = "edge"):
    """Convolve one-sided rows ``(n, nf_active)`` (real, or complex = Hermitian two-sided
    spectrum) exactly as ``_convolve_matrix_components`` does (edge padding, length-N
    circle, DC/Nyquist imaginary fix), with kernel ``delta^moment * |W|^2_delta``
    (signed circular lag, in bins). Returns ``B_j(f) = sum_delta delta^j K_delta
    S(f + delta)`` on the active band (real for real input). ``pad="zero"`` leaves the
    out-of-band bins at zero instead of the legacy edge values (see
    :func:`_fact_edge_rows`)."""
    xp = backend.xp
    rows = xp.asarray(rows)
    is_real = not xp.iscomplexobj(rows)
    rows = _fact_guard_f0(rows.astype(xp.float64 if is_real else xp.complex128))
    N = len(backend.window_values)
    nf_full = N // 2 + 1
    sl = backend.active_slice
    lo, hi = sl.start or 0, nf_full if sl.stop is None else sl.stop
    full = xp.zeros((rows.shape[0], nf_full), dtype=rows.dtype)
    full[:, lo:hi] = rows
    if pad == "edge":
        full[:, :lo] = rows[:, 0:1]
        full[:, hi:] = rows[:, -1:]
    elif pad != "zero":
        raise ValueError(f"pad must be 'edge' or 'zero', got {pad!r}")
    return _fact_convolve_full(backend, full, moment, is_real)


def _fact_convolve_full(backend, full, moment, is_real):
    """Core of :func:`_fact_convolve` on full one-sided rows ``(n, N//2 + 1)``."""
    xp = backend.xp
    N = len(backend.window_values)
    nf_full = N // 2 + 1
    sl = backend.active_slice
    lo, hi = sl.start or 0, nf_full if sl.stop is None else sl.stop
    x = xp.fft.irfft(full, n=N, axis=-1)
    if moment == 0:
        y = xp.fft.rfft(x * backend._window_kernel_fft_real, axis=-1)
        if not is_real:
            y += 1j * (full[:, 0:1].imag * backend._window_kernel_dc)
            if backend._window_kernel_nyq is not None:
                y += 1j * (full[:, N // 2 : N // 2 + 1].imag * backend._window_kernel_nyq)
    else:
        n = xp.arange(N, dtype=xp.float64)
        lag = xp.where(n < N / 2, n, n - N)
        kj = backend.window_kernel * lag**moment
        y = xp.fft.fft(x * xp.fft.fft(kj), axis=-1)[:, :nf_full]
        if not is_real:  # DC term S(0) reached at lag -f
            fpos = xp.arange(nf_full, dtype=xp.float64)
            y += 1j * (full[:, 0:1].imag * (backend.window_kernel[:nf_full] * (-fpos) ** moment))
    y = y[:, lo:hi]
    return xp.ascontiguousarray(y.real if is_real else y)


def _fact_edge_rows(backend):
    """Convolved out-of-band indicator rows ``(P_lo, P_hi)`` on the active band, cached.

    The legacy convolution pads below / above the active band with the first / last
    in-band value, so for any row ``c``:
    ``conv_edge(c) = conv_zero(c) + c[0] P_lo + c[-1] P_hi`` exactly. Factorizing only
    the in-band part (``m conv_zero(T)``) and adding the per-walker edge values
    ``m(f_lo) T(f_lo) P_lo + m(f_hi) T(f_hi) P_hi`` removes the band-edge error of the
    factorization (dominant in narrow bands)."""
    cache = backend.__dict__.get("_wconv_fact_edge")
    if cache is not None:
        return cache
    xp = backend.xp
    N = len(backend.window_values)
    nf_full = N // 2 + 1
    sl = backend.active_slice
    lo, hi = sl.start or 0, nf_full if sl.stop is None else sl.stop
    full = xp.zeros((2, nf_full), dtype=xp.float64)
    full[0, :lo] = 1.0
    full[1, hi:] = 1.0
    P = _fact_convolve_full(backend, full, 0, True)
    backend._wconv_fact_edge = (P[0], P[1])
    return backend._wconv_fact_edge


def _fact_galaxy_R(backend):
    """Sky-averaged galactic response ``R_k(t)``, ``(num_times, 6)`` in C++ entry order
    (00, 01, 02, 11, 12, 22), extracted from a galaxy-only evaluation of the compiled
    kernel against the Python mirror (validates the mirror: returns its max mismatch,
    band-normalised per entry)."""
    xp = backend.xp
    ref = dict(Amp=1.0, alpha=1.8, f_1=2e-3, kn=3e-3, f_2=8e-4)
    n_par = backend.n_noise_par
    zeros = xp.zeros((1, n_par)) if n_par > 1 else xp.zeros(1)
    kp, ka = _fact_zero_knots(backend, 1)
    c = backend._compute_matrix_elements(
        backend.f_arr, zeros, zeros, *[xp.asarray([v]) for v in ref.values()], kp, ka,
        convolve_window=False,
    )
    nt, nf = backend.num_times, backend.num_freqs
    # _compute_matrix_elements order: c00, c11, c22, c01, c02, c12 -> gal_R_avg order
    ent = [c[0].real, c[3].real, c[4].real, c[1].real, c[5].real, c[2].real]
    ent = xp.stack([e.reshape(nt, nf) for e in ent], axis=-1)  # (nt, nf, 6)
    G = 0.0
    for _, fabs, norm in _fact_alias_grids(backend):
        with np.errstate(divide="ignore", invalid="ignore"):
            G = G + _fact_galaxy_shape(xp, fabs, split="full", **ref) * _fact_common_tdi(backend, fabs) * norm
    ok = xp.isfinite(G) & (G > 0) & xp.isfinite(ent).all(axis=(0, 2))
    Gm = xp.where(ok, G, 0.0)[None, :, None]
    entm = xp.where(ok[None, :, None], ent, 0.0)
    R = (entm * Gm).sum(axis=1) / (Gm**2).sum(axis=1)  # (nt, 6)
    scale = xp.abs(entm).max(axis=1, keepdims=True)
    mismatch = float(xp.max(xp.abs(entm - R[:, None, :] * Gm) / xp.where(scale > 0, scale, 1.0)))
    return R, mismatch


def _fact_galaxy_setup(backend, split="powerlaw", order=0):
    """Walker-independent factorized-galaxy data, cached on ``backend``.

    For each alias term ``a`` (a = 0: data grid) and moment ``j <= order``:
    ``B[j][a] = conv_j(T_a)`` with ``T_a(f) = common_tdi(|f_a|) norm_a(f) [|f_a|^{-7/3}]``
    (the bracket only for ``split="powerlaw"``), plus ``R`` (num_times, 6).
    """
    if split not in FACT_GALAXY_SPLITS:
        raise ValueError(f"unknown galaxy split {split!r}; expected one of {FACT_GALAXY_SPLITS}")
    cache = backend.__dict__.setdefault("_wconv_fact_gal", {})
    key = (split, int(order))
    if key in cache:
        return cache[key]
    xp = backend.xp
    if "R" not in cache:
        cache["R"], cache["R_mismatch"] = _fact_galaxy_R(backend)
    T = []
    for _, fabs, norm in _fact_alias_grids(backend):
        t = _fact_common_tdi(backend, fabs) * norm
        if split == "powerlaw":
            with np.errstate(divide="ignore", invalid="ignore"):
                t = t * fabs ** (-7.0 / 3.0)
        T.append(t)
    T = _fact_guard_f0(xp.stack(T))  # (1 + n_alias, nf)
    # in-band part only; the out-of-band (edge) padding is added exactly per walker
    B = [_fact_convolve(backend, T, moment=j, pad="zero") for j in range(order + 1)]  # each (n_a, nf)
    P_lo, P_hi = _fact_edge_rows(backend)
    R = cache["R"]
    R00 = R[:, 0]
    setup = dict(
        split=split, order=int(order), B=B, R=R, R00=R00,
        T_lo=T[:, 0], T_hi=T[:, -1], P_lo=P_lo, P_hi=P_hi,
        ratios=xp.ascontiguousarray(R / R00[:, None]),
        df=float(backend.f_arr[1] - backend.f_arr[0]),
        R_mismatch=cache["R_mismatch"],
    )
    cache[key] = setup
    return setup


def factorized_galaxy(backend, Amp, alpha, f_1, kn, f_2, order=0, split="powerlaw", n_gal=None):
    """Approximate convolved galaxy rows without per-walker FFTs (``mode="factorized"``).

    Uses ``conv(m T) ~= m conv(T) [+ m' conv(delta K, T) + m''/2 conv(delta^2 K, T)]`` per
    alias term, with ``m`` the per-walker galactic factor (see ``split``) at ``|f_a(f)|``
    and ``T`` walker-independent. ``order`` = 0 (plain factorization), 1 or 2 (Taylor
    correction; bin-unit derivatives from exact evaluations of ``m`` at ``f +- df``).

    Returns ``(gal_conv, gal_ratios)`` in the same form as :func:`galaxy_convolved_rows`:
    ``gal_conv`` ``(num_psds, num_times, n_gal)`` in units of the XX entry and
    ``gal_ratios`` ``(num_times, 6)`` = ``R_k / R_00`` in C++ entry order
    (00, 01, 02, 11, 12, 22). Entry ``k`` of the convolved covariance is
    ``gal_ratios[t, k] * gal_conv[psd, t, f]``. ``n_gal`` truncates the band
    (default: the whole active grid). The legacy out-of-band edge padding is handled
    exactly (:func:`_fact_edge_rows`); only the in-band product is approximated.

    Measured error budget (``scripts/window_conv/agent5_error_budget.md``): use the
    default ``split="powerlaw"`` with ``order=2``.
    """
    if order not in (0, 1, 2):
        raise ValueError("order must be 0, 1 or 2")
    xp = backend.xp
    setup = _fact_galaxy_setup(backend, split, order)
    p = [xp.atleast_1d(xp.asarray(v, dtype=xp.float64)).ravel()[:, None] for v in (Amp, alpha, f_1, kn, f_2)]
    df = setup["df"]
    row = 0.0
    for a, (fa, _, _) in enumerate(_fact_alias_grids(backend)):
        m0 = _fact_guard_f0(_fact_galaxy_shape(xp, xp.abs(fa)[None, :], *p, split=split))
        row = row + m0 * setup["B"][0][a]
        # exact contribution of the legacy edge padding (constant out-of-band values)
        row = row + (m0[:, 0:1] * setup["T_lo"][a]) * setup["P_lo"][None, :]
        row = row + (m0[:, -1:] * setup["T_hi"][a]) * setup["P_hi"][None, :]
        if order >= 1:
            mp = _fact_galaxy_shape(xp, xp.abs(fa + df)[None, :], *p, split=split)
            mm = _fact_galaxy_shape(xp, xp.abs(fa - df)[None, :], *p, split=split)
            row = row + _fact_guard_f0(0.5 * (mp - mm)) * setup["B"][1][a]
            if order >= 2:
                row = row + _fact_guard_f0(0.5 * (mp - 2.0 * m0 + mm)) * setup["B"][2][a]
    gal_conv = setup["R00"][None, :, None] * row[:, None, :]
    if n_gal is not None:
        gal_conv = gal_conv[..., :n_gal]
    return xp.ascontiguousarray(gal_conv), setup["ratios"]


def factorized_spline_weights(backend, knots_position_all, knots_amplitude_all, order: int = 0):
    """Per-walker multiplicative spline weights for the factorized mode.

    ``order=0``: ``10^spl(f)``, shape ``(num_psds, 2, num_freqs)`` (row 0: OMS, row 1:
    TM), for ``fused_log_like(..., spline_weights=...)``; the mode then uses
    ``C ~= Soms^2 10^spl_oms B_oms + Sa^2 10^spl_tm B_tm`` (approximate).

    ``order=2``: shape ``(num_psds, 2, 3, num_freqs)`` = ``[w, w', w''/2]`` with bin-unit
    central differences of ``w = 10^spl`` (from the spline evaluated at ``f +- df``),
    to be paired with :func:`factorized_instrument_moment_bases` moments 0, 1, 2:
    ``C ~= sum_b coeff_b sum_j W_j B^(j)_b``.
    """
    if order not in (0, 2):
        raise ValueError("order must be 0 or 2")
    xp = backend.xp
    nf = backend.num_freqs

    def w_at(f):
        spl = backend.spline_interpolant(xp.log10(f), knots_position_all, knots_amplitude_all)
        w = xp.stack([xp.asarray(s, dtype=xp.float64).reshape(-1, nf) for s in spl[:2]], axis=1)
        return 10.0**w

    w0 = w_at(backend.f_arr)
    if order == 0:
        return xp.ascontiguousarray(w0)
    df = float(backend.f_arr[1] - backend.f_arr[0])
    f = xp.asarray(backend.f_arr, dtype=xp.float64)
    wp, wm = w_at(f + df), w_at(xp.where(f - df > 0, f - df, f))
    if float(f[0]) - df <= 0:  # one-sided difference at a bin with no f - df
        wm[..., 0] = 2 * w0[..., 0] - wp[..., 0]
    return xp.ascontiguousarray(xp.stack([w0, 0.5 * (wp - wm), 0.5 * (wp - 2 * w0 + wm)], axis=2))


def factorized_instrument_moment_bases(backend, order: int = 2):
    """Instrument bases for the factorized spline mode, with exact edge padding.

    Returns ``(Bj, edge)``: ``Bj`` is a list over ``j = 0..order`` of 6-tuples
    ``(c00, c11, c22, c01, c02, c12)`` each ``(n_basis, total_terms)`` in
    :func:`get_instrument_bases` order, convolved with ``delta^j |W|^2`` with the
    out-of-band bins left at zero; ``edge = (lo_vals, hi_vals, P_lo, P_hi)`` with
    ``lo_vals`` / ``hi_vals`` 6-tuples ``(n_basis, num_times)`` of the unconvolved basis
    values at the first / last active bin. Then, per PSD with coefficients ``c_b`` and
    spline weights ``w_b`` (and ``W_j`` = ``[w, w', w''/2]``)::

        C_k(f) ~= sum_b c_b [ sum_j W_j(b, f) Bj[j][k][b, f]
                              + w_b(f_lo) lo_vals[k][b] P_lo(f) + w_b(f_hi) hi_vals[k][b] P_hi(f) ]

    (the edge terms are exact). Cached on ``backend`` (``_wconv_fact_inst_moments``).
    """
    cache = backend.__dict__.setdefault("_wconv_fact_inst_moments", {})
    raw = get_instrument_bases(backend, convolve_window=False)
    nt, nf = backend.num_times, backend.num_freqs
    for j in range(order + 1):
        if j not in cache:
            cache[j] = tuple(
                _fact_convolve(backend, b.reshape(-1, nf), moment=j, pad="zero").reshape(b.shape)
                for b in raw
            )
    if "edge" not in cache:
        r = [b.reshape(b.shape[0], nt, nf) for b in raw]
        r = [_fact_guard_f0(x) for x in r]
        cache["edge"] = (
            tuple(x[..., 0] for x in r), tuple(x[..., -1] for x in r), *_fact_edge_rows(backend)
        )
    return [cache[j] for j in range(order + 1)], cache["edge"]
