// sobbh_lookup -- fused SOBBH direct-to-WDM lookup (see sobbh_lookup_kernel.hpp).
// The CPU build compiles this same file (copied to sobbh_lookup_kernel.cxx by CMake).

#include "sobbh_lookup_kernel.hpp"
#include "gbt_global.h"

#include <cmath>
#include <cstdint>

#define SOBBH_LOOKUP_MAX_NCH 3
#define SOBBH_LOOKUP_THREADS 256

static CUDA_DEVICE inline void sobbh_lookup_atomic_add(double *p, double v)
{
#ifdef __CUDA_ARCH__   // device pass only (atomicAdd is __device__)
    atomicAdd(p, v);
#else
    *p += v;
#endif
}

// One (row r, pixel n): adds this pixel's <d|h>, <h|h> into *dh, *hh (mode 0) or
// writes factor * w into the fill buffer (mode 1); counts dropped channel-pixels.
static CUDA_DEVICE void sobbh_lookup_point(const SOBBHLookupArgs &A, int r, int n,
    double *dh, double *hh, unsigned long long *n_drop, unsigned long long *n_look)
{
    const double two_pi = 6.283185307179586476925286766559;
    const int nch = A.nch;
    double t = A.t0 + (double)n * A.layer_dt;

    // spline segment: every channel and the reference phase share row r's knots
    const double *xs = &A.x[(size_t)r * nch * A.N];
    int seg = wdm_spline_segment(xs, A.N, t);
    double dx = t - xs[seg];

    const double *ry = &A.ref_y[(size_t)r * A.N];
    double ref, ref1, ref2;
    wdm_spline_derivs(ry, &A.ref_c1[(size_t)r * A.N], &A.ref_c2[(size_t)r * A.N],
                      &A.ref_c3[(size_t)r * A.N], seg, dx, &ref, &ref1, &ref2);

    double amp[SOBBH_LOOKUP_MAX_NCH], f[SOBBH_LOOKUP_MAX_NCH], fdot[SOBBH_LOOKUP_MAX_NCH];
    double cph[SOBBH_LOOKUP_MAX_NCH], sph[SOBBH_LOOKUP_MAX_NCH];
    bool live[SOBBH_LOOKUP_MAX_NCH], fd_ok[SOBBH_LOOKUP_MAX_NCH];
    int td[SOBBH_LOOKUP_MAX_NCH][4];
    double wd[SOBBH_LOOKUP_MAX_NCH][4];
    bool past_tc = (t >= A.tc[r]);
    double f_sum = 0.0;
    int n_live = 0;
    for (int c = 0; c < nch; c += 1)
    {
        size_t off = ((size_t)r * nch + c) * A.N;
        double a = A.amp_y[off + seg] + dx * (A.amp_c1[off + seg] + dx * (A.amp_c2[off + seg] + dx * A.amp_c3[off + seg]));
        if (past_tc) a = 0.0;
        double p, p1, p2;
        wdm_spline_derivs(&A.ph_y[off], &A.ph_c1[off], &A.ph_c2[off], &A.ph_c3[off], seg, dx, &p, &p1, &p2);
        double ph = p + ref;
        double fc = (p1 + ref1) / two_pi;
        double fdc = (p2 + ref2) / two_pi;
        if (fc < 0.0)        // cos is even: mirror to positive frequency
        {
            ph = -ph;
            fc = -fc;
            fdc = -fdc;
        }
        amp[c] = a;
        f[c] = fc;
        fdot[c] = fdc;
        cph[c] = cos(ph);
        sph[c] = sin(ph);
        live[c] = (a != 0.0);
        if (live[c])
        {
            f_sum += fc;
            n_live += 1;
        }
        // the fdot axis: exact support first (the Python mask), then the B-spline taps
        fd_ok[c] = true;
        if (A.tab.FD > 1)
        {
            fd_ok[c] = (fdc >= A.fdot_lo) && (fdc <= A.fdot_hi);
            if (live[c] && !fd_ok[c]) *n_drop += 1;
            if (fd_ok[c]) fd_ok[c] = wdm_table_fdot_axis(A.tab, fdc, td[c], wd[c]);
        }
    }
    if ((n_live < nch) && (A.row_dead != nullptr)) A.row_dead[r] = 1;   // benign race: all write 1
    if (n_live == 0) return;
    *n_look += 1;
    double f_ref = f_sum / (double)n_live;
    int m0 = (int)floor(f_ref / A.layer_df);
    int ind_max_f = A.ind_min_f + A.Nf_active - 1;
    int na = n - A.ind_min_t;
    size_t plane = (size_t)A.Nf_active * A.Nt_active;

    for (int dm = -A.num_m_layers; dm <= A.num_m_layers; dm += 1)
    {
        int m = m0 + dm;
        if ((m < A.ind_min_f) || (m > ind_max_f)) continue;
        int pix_odd = (m + n) & 1;
        double w[SOBBH_LOOKUP_MAX_NCH];
        bool any = false;
        for (int c = 0; c < nch; c += 1)
        {
            w[c] = 0.0;
            if (!live[c] || !fd_ok[c]) continue;
            double f_norm = f[c] - (double)m * A.layer_df;
            if ((f_norm < A.tab.f_lo) || (f_norm > A.tab.f_hi)) continue;   // the exact support
            double cc, ss;
            if (!wdm_table_cs_at(A.tab, td[c], wd[c], f_norm, &cc, &ss)) continue;
            w[c] = wdm_quarter_turn_value(cc, ss, A.tab.ref_odd, pix_odd, amp[c], cph[c], sph[c]);
            any = true;
        }
        if (!any) continue;
        size_t pix = (size_t)(m - A.ind_min_f) * A.Nt_active + na;
        if (A.mode == 1)
        {
            double fac = A.factors[r];
            double *b = &A.buf[(size_t)A.data_index[r] * nch * plane + pix];
            for (int c = 0; c < nch; c += 1)
                if (w[c] != 0.0) sobbh_lookup_atomic_add(&b[(size_t)c * plane], fac * w[c]);
            continue;
        }
        const double *d = &A.data[(size_t)A.data_index[r] * nch * plane + pix];
        if (A.full_invC)
        {
            const double *C = &A.invC[(size_t)A.noise_index[r] * nch * nch * plane + pix];
            for (int c = 0; c < nch; c += 1)
            {
                double dc = d[(size_t)c * plane];
                for (int c2 = 0; c2 < nch; c2 += 1)
                {
                    double icc = C[(size_t)(c * nch + c2) * plane];
                    *dh += dc * icc * w[c2];
                    *hh += w[c] * icc * w[c2];
                }
            }
        }
        else
        {
            const double *C = &A.invC[(size_t)A.noise_index[r] * nch * plane + pix];
            for (int c = 0; c < nch; c += 1)
            {
                double icc = C[(size_t)c * plane];
                *dh += d[(size_t)c * plane] * icc * w[c];
                *hh += w[c] * icc * w[c];
            }
        }
    }
}

#ifdef __CUDACC__
// grid (pixel blocks, rows); one block reduction per (block, row)
CUDA_KERNEL
void sobbh_lookup_kernel(SOBBHLookupArgs A)
{
    CUDA_SHARED double s_dh[SOBBH_LOOKUP_THREADS];
    CUDA_SHARED double s_hh[SOBBH_LOOKUP_THREADS];
    CUDA_SHARED unsigned long long s_drop;
    CUDA_SHARED unsigned long long s_look;
    int r = A.row0 + blockIdx.y;
    if (threadIdx.x == 0)
    {
        s_drop = 0;
        s_look = 0;
    }
    double dh = 0.0;
    double hh = 0.0;
    unsigned long long n_drop = 0;
    unsigned long long n_look = 0;
    int P = A.n_hi - A.n_lo;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < P; j += gridDim.x * blockDim.x)
    {
        sobbh_lookup_point(A, r, A.n_lo + j, &dh, &hh, &n_drop, &n_look);
    }
    s_dh[threadIdx.x] = dh;
    s_hh[threadIdx.x] = hh;
    CUDA_SYNC_THREADS;
    if (A.counts != nullptr)
    {
        if (n_drop) atomicAdd(&s_drop, n_drop);
        if (n_look) atomicAdd(&s_look, n_look);
    }
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            s_dh[threadIdx.x] += s_dh[threadIdx.x + stride];
            s_hh[threadIdx.x] += s_hh[threadIdx.x + stride];
        }
        CUDA_SYNC_THREADS;
    }
    if (threadIdx.x == 0)
    {
        if (A.mode == 0)
        {
            atomicAdd(&A.d_h[r], s_dh[0]);
            atomicAdd(&A.h_h[r], s_hh[0]);
        }
        if (A.counts != nullptr)
        {
            if (s_drop) atomicAdd(&A.counts[0], s_drop);
            if (s_look) atomicAdd(&A.counts[1], s_look);
        }
    }
}
#endif

void sobbh_lookup_wrap(SOBBHLookupArgs args)
{
    int P = args.n_hi - args.n_lo;
    if ((P <= 0) || (args.num_rows <= 0)) return;
#ifdef __CUDACC__
    int threads = SOBBH_LOOKUP_THREADS;
    int bx = (P + threads - 1) / threads;
    for (int r0 = 0; r0 < args.num_rows; r0 += 65535)     // grid.y limit
    {
        int rows_here = (args.num_rows - r0 < 65535) ? (args.num_rows - r0) : 65535;
        SOBBHLookupArgs a = args;
        a.row0 = r0;
        dim3 grid(bx, rows_here);
        sobbh_lookup_kernel<<<grid, threads>>>(a);
        gpuErrchk(cudaGetLastError());
    }
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
#else
    unsigned long long n_drop = 0;
    unsigned long long n_look = 0;
    for (int r = 0; r < args.num_rows; r += 1)
    {
        double dh = 0.0;
        double hh = 0.0;
        for (int n = args.n_lo; n < args.n_hi; n += 1)
        {
            sobbh_lookup_point(args, r, n, &dh, &hh, &n_drop, &n_look);
        }
        if (args.mode == 0)
        {
            args.d_h[r] += dh;
            args.h_h[r] += hh;
        }
    }
    if (args.counts != nullptr)
    {
        args.counts[0] += n_drop;
        args.counts[1] += n_look;
    }
#endif
}
