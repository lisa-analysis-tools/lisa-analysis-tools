#ifndef __WDM_LOOKUP_KERNELS_HH__
#define __WDM_LOOKUP_KERNELS_HH__

// Device helpers (plain inline functions on the CPU build) for the fused WDM
// lookup-sum kernels: from a sparse response (amplitude/phase splines + an
// optional exact dense-output carrier) straight to WDM pixel coefficients through
// the n_ref lookup table. Header-only so other source families' kernels (SOBBH)
// can include them; the EMRI kernel itself lives in lat_spline_tdi_waveform.cu
// (wdm_lookup_sum).
//
// Semantics mirrored from the Python reference:
//   * the table: lisatools.domains._UniformCubicSpline (scipy/cupyx ndimage
//     map_coordinates, order 3, mode "mirror", prefilter=False on coefficients
//     prefiltered by spline_filter), 0 outside the grid (index tolerance 1e-9);
//   * the per-layer rule: WDMLookupTable.get_wdm_coeffs, BASIS_CYCLE
//     "quarter_turn" (n_ref_complex / n_ref_only builds);
//   * response splines: gpubackendtools CubicSplineInterpolant (segment =
//     searchsorted(x, t, "right") - 1 clamped to [0, N - 2]; y + c1 dx + c2 dx^2
//     + c3 dx^3);
//   * the carrier: FEW's DOPR853 dense output (wdm_direct.dense_eval_derivs).

#include "gbt_global.h"
#include <math.h>

// scipy.ndimage "mirror" reflection of an integer tap into [0, n)
CUDA_DEVICE inline int wdm_mirror_index(int i, int n)
{
    if (n <= 1) return 0;
    int s2 = 2 * n - 2;
    if (i < 0)
    {
        i = s2 * (-i / s2) + i;
        i = (i <= 1 - n) ? i + s2 : -i;
    }
    else if (i >= n)
    {
        i -= s2 * (i / s2);
        if (i >= n) i = s2 - i;
    }
    return i;
}

// cubic B-spline taps and weights at the continuous index x on an axis of n
// points; returns false when x is outside [-1e-9, n - 1 + 1e-9] (value 0 there)
CUDA_DEVICE inline bool wdm_bspline3_axis(double x, int n, int *taps, double *w)
{
    if (!(x >= -1e-9) || !(x <= (double)(n - 1) + 1e-9)) return false;
    if (x < 0.0) x = -x;                                  // mirror the coordinate itself
    if (x > (double)(n - 1)) x = 2.0 * (double)(n - 1) - x;
    double fl = floor(x);
    double t = x - fl;
    double u = 1.0 - t;
    double t2 = t * t;
    double t3 = t2 * t;
    w[0] = u * u * u / 6.0;
    w[1] = (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0;
    w[2] = (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0;
    w[3] = t3 / 6.0;
    int start = (int)fl - 1;
    for (int k = 0; k < 4; k += 1) taps[k] = wdm_mirror_index(start + k, n);
    return true;
}

// Prefiltered table geometry: coefficients (FD, FF) row-major over (fdot, f_norm),
// FD == 1 for a table without an fdot axis (1-D in f_norm).
struct WDMLookupTableView {
    const double *coeff_c;   // B-spline coefficients of Re(table_cx) / table_cos
    const double *coeff_s;   // ... of the sin table with the build's (-1)^block bake undone
    int FD;
    int FF;
    double fdot0;
    double dfdot;
    double f0;
    double df;
    double f_lo;             // [f_lo, f_hi]: the f_norm support (out_of_support="zero")
    double f_hi;
    int ref_odd;             // (m_ref + n_ref) & 1
};

// fdot-axis taps/weights of a table (shared by every layer of one (harmonic, channel, pixel));
// false outside the axis (every layer is 0 there). A table without an fdot axis: always true.
CUDA_DEVICE inline bool wdm_table_fdot_axis(const WDMLookupTableView &tab, double fdot, int *td, double *wd)
{
    if (tab.FD <= 1) return true;
    return wdm_bspline3_axis((fdot - tab.fdot0) / tab.dfdot, tab.FD, td, wd);
}

// (c, s) = the two tables at f_norm and the fdot taps (td, wd) of wdm_table_fdot_axis; false
// (and zeros) outside the f_norm axis
CUDA_DEVICE inline bool wdm_table_cs_at(const WDMLookupTableView &tab, const int *td, const double *wd,
    double f_norm, double *c, double *s)
{
    int tf[4];
    double wf[4];
    *c = 0.0;
    *s = 0.0;
    if (!wdm_bspline3_axis((f_norm - tab.f0) / tab.df, tab.FF, tf, wf)) return false;
    if (tab.FD <= 1)
    {
        for (int b = 0; b < 4; b += 1)
        {
            *c += wf[b] * tab.coeff_c[tf[b]];
            *s += wf[b] * tab.coeff_s[tf[b]];
        }
        return true;
    }
    for (int a = 0; a < 4; a += 1)
    {
        const double *rc = &tab.coeff_c[(size_t)td[a] * tab.FF];
        const double *rs = &tab.coeff_s[(size_t)td[a] * tab.FF];
        double ac = 0.0;
        double as = 0.0;
        for (int b = 0; b < 4; b += 1)
        {
            ac += wf[b] * rc[tf[b]];
            as += wf[b] * rs[tf[b]];
        }
        *c += wd[a] * ac;
        *s += wd[a] * as;
    }
    return true;
}

// (c, s) = the two tables at (fdot, f_norm); false (and zeros) outside the grid
CUDA_DEVICE inline bool wdm_table_cs(const WDMLookupTableView &tab, double fdot, double f_norm, double *c, double *s)
{
    int td[4];
    double wd[4];
    *c = 0.0;
    *s = 0.0;
    if (!wdm_table_fdot_axis(tab, fdot, td, wd)) return false;
    return wdm_table_cs_at(tab, td, wd, f_norm, c, s);
}

// WDMLookupTable.get_wdm_coeffs, quarter_turn: the pixel (m, n) value of
// amp cos(x + phi) from the table pair (c, s) at its (f_norm, fdot)
CUDA_DEVICE inline double wdm_quarter_turn_value(double c, double s, int ref_odd, int pix_odd,
    double amp, double cos_phi, double sin_phi)
{
    double c_eff = ref_odd ? -c : c;
    double cosc = pix_odd ? s : c_eff;
    double sinc = pix_odd ? -c_eff : s;
    return amp * (cosc * cos_phi - sinc * sin_phi);
}

// gpubackendtools CubicSplineInterpolant segment of t on the knots x[0..N-1]
CUDA_DEVICE inline int wdm_spline_segment(const double *x, int N, double t)
{
    if (t < x[0]) return 0;
    if (t >= x[N - 1]) return N - 2;
    int lo = 0;
    int hi = N - 1;          // x[lo] <= t < x[hi]
    while (hi - lo > 1)
    {
        int mid = (lo + hi) / 2;
        if (x[mid] <= t) lo = mid;
        else hi = mid;
    }
    return lo;
}

// value and first two derivatives of one spline (flats offset to the interp's row)
CUDA_DEVICE inline void wdm_spline_derivs(const double *y, const double *c1, const double *c2, const double *c3,
    int seg, double dx, double *v, double *d1, double *d2)
{
    *v = y[seg] + dx * (c1[seg] + dx * (c2[seg] + dx * c3[seg]));
    *d1 = c1[seg] + dx * (2.0 * c2[seg] + 3.0 * dx * c3[seg]);
    *d2 = 2.0 * c2[seg] + 6.0 * dx * c3[seg];
}

// FEW DOPR853 dense output r1 + s(r2 + s1(r3 + s(r4 + s1(r5 + s(r6 + s1(r7 + s r8)))))),
// s1 = 1 - s, and its first two s-derivatives (forward-mode Horner; see
// wdm_direct.dense_eval_derivs)
CUDA_DEVICE inline void wdm_dense_phase_derivs(const double *c, double s, double *v, double *d1, double *d2)
{
    double s1 = 1.0 - s;
    double q = c[7];
    double q1 = 0.0;
    double q2 = 0.0;
    for (int j = 6; j >= 0; j -= 1)
    {
        if ((j & 1) == 0)    // j = 6, 4, 2, 0: q = c + s p
        {
            q2 = 2.0 * q1 + s * q2;
            q1 = q + s * q1;
            q = c[j] + s * q;
        }
        else                 // j = 5, 3, 1: q = c + s1 p
        {
            q2 = -2.0 * q1 + s1 * q2;
            q1 = -q + s1 * q1;
            q = c[j] + s1 * q;
        }
    }
    *v = q;
    *d1 = q1;
    *d2 = q2;
}

#endif // __WDM_LOOKUP_KERNELS_HH__
