#ifndef __SOBBH_LOOKUP_KERNEL_HPP__
#define __SOBBH_LOOKUP_KERNEL_HPP__

// ============================================================================
// sobbh_lookup -- the fused SOBBH direct-to-WDM lookup (2026-10-02).
//
// From the batched SOBBH TDI-on-the-fly response on its SPARSE grid (per row and
// channel: amplitude and TDI-phase cubic splines, per row: the reference-phase
// spline; gpubackendtools CubicSplineInterpolant flats) straight to either
//   mode 0 (ACCUMULATE): <d|h> and <h|h> per row against the walker slabs, or
//   mode 1 (FILL):       factor[row] * h added into the slab buffer,
// with NOTHING per pixel in global memory. One thread per (row r, pixel n):
//   1. per channel c: amp (0 at or past tc[r]), phase = tdi_phase + phase_ref,
//      f = phase' / 2 pi, fdot = phase'' / 2 pi (analytic spline derivatives),
//      mirror f < 0 (cos is even);
//   2. live_c = amp != 0 (the TDI amplitude is SIGNED); ONE layer window for all
//      channels from the channel-mean f of the live channels:
//      m = floor(f_ref / layer_df) + [-L .. L] (every channel sits on the same
//      (m, n), so w^T invC w is thread-local -- a SOBBH row is ONE harmonic);
//   3. per (m, c): the n_ref table's uniform cubic B-spline at (fdot_c, f_c - m
//      layer_df) and the quarter-turn rule (wdm_lookup_kernels.hh, shared with
//      the EMRI wdm_lookup_sum kernel); valid iff in the table support, live
//      and ind_min_f <= m <= ind_max_f;
//   4. ACCUMULATE: d_h += sum_cc' d_c C_cc' w_c', h_h += sum_cc' w_c C_cc' w_c'
//      (XYZ: full 3x3 per pixel; AET / AE: the diagonal), block reduction per
//      row + one atomicAdd per block; FILL: atomicAdd(buf[di[r], c, ma, na],
//      factor[r] w_c).
// Python reference: lisatools.sources.sobbh.wdm_direct (SOBBHDirectWDM.sparse_from
// + sparse_inner_products / scatter_add, WDMLookupEvaluator(interp="spline")).
//
// BACKEND CONTRACT: the CUDA build leads; the CPU build compiles this SAME source
// (CMake copies sobbh_lookup_kernel.cu to .cxx) as a serial loop and agrees to
// roundoff (only the summation order differs). Free functions only -- no class
// escapes this TU, so the CPU/GPU class-name aliasing rule does not apply.
// ============================================================================

#include "gbt_global.h"
#include "wdm_lookup_kernels.hh"

struct SOBBHLookupArgs {
    int mode;                  // 0 = accumulate (d_h, h_h), 1 = fill
    // ---- outputs ----
    double *d_h;               // (num_rows) accumulated (zero it first), mode 0
    double *h_h;               // (num_rows) accumulated (zero it first), mode 0
    double *buf;               // (n_slots_d, nch, Nf_active, Nt_active) accumulated, mode 1
    unsigned long long *counts;// (2,): [dropped channel-pixels (live, fdot off the table
                               //       axis), lookup pixels (any channel live)]; nullptr: off
    int *row_dead;             // (num_rows): set to 1 where any channel-pixel is dead (amp == 0,
                               //       i.e. at or past merger); nullptr: off
    // ---- walker slabs (mode 0) ----
    const double *data;        // (n_slots_d, nch, Nf_active, Nt_active)
    const double *invC;        // full: (n_slots_c, nch, nch, Nfa, Nta); diag: (n_slots_c, nch, Nfa, Nta)
    int full_invC;             // 1: XYZ full channel matrix, 0: AET / AE diagonal
    const int *data_index;     // (num_rows) slab of each row (data / fill target)
    const int *noise_index;    // (num_rows) slab of each row (invC), mode 0
    const double *factors;     // (num_rows), mode 1
    // ---- response splines (GBT CubicSplineInterpolant flats) ----
    int num_rows;
    int row0;                  // first row of this launch (set by the wrapper: grid.y <= 65535)
    int nch;
    int N;                     // spline length (response evaluation points)
    const double *x;           // (num_rows * nch, N) knots; row r * nch is read
    const double *amp_y;       // (num_rows * nch, N) each
    const double *amp_c1;
    const double *amp_c2;
    const double *amp_c3;
    const double *ph_y;        // TDI phase (num_rows * nch, N) each
    const double *ph_c1;
    const double *ph_c2;
    const double *ph_c3;
    const double *ref_y;       // reference phase (num_rows, N) each, same knots
    const double *ref_c1;
    const double *ref_c2;
    const double *ref_c3;
    const double *tc;          // (num_rows) absolute merger times (inf: never)
    // ---- grid ----
    int n_lo;                  // absolute pixel range [n_lo, n_hi) = [ind_min_t, ind_max_t]
    int n_hi;
    double t0;                 // absolute time of pixel 0
    double layer_dt;
    double layer_df;
    int ind_min_f;             // active band [ind_min_f, ind_min_f + Nf_active)
    int Nf_active;
    int ind_min_t;
    int Nt_active;
    int num_m_layers;
    // ---- table (prefiltered B-spline coefficients) + its exact support ----
    WDMLookupTableView tab;
    double fdot_lo;            // [fdot_lo, fdot_hi]: the fdot axis (FD > 1)
    double fdot_hi;
};

void sobbh_lookup_wrap(SOBBHLookupArgs args);

#endif // __SOBBH_LOOKUP_KERNEL_HPP__
