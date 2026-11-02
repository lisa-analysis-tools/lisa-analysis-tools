// stft_cross_terms.cu: kernels and host wrapper of stft_cross_terms.hpp.
//
// Pass 1 sums every (table, pixel range, residual, template) over the pixels of the range, one
// pixel after the other (segments outer, bins inner). Pass 2 adds the ranges of a table in order.
// That order does not depend on the other tables of the call, on the tiling of the pairs or on the
// device. The CPU build runs the same loops through the GBT axis macros and differs from the GPU
// only where a compiler fuses multiply-adds.
//
// Per pixel, a residual is reduced to y = C^H r (6 reals) and the 9 real noise terms of Re(t^H C t),
// and a template to its values (6 reals) and its 9 real products p. A pair then takes 15
// multiply-adds per pixel:
//   Re(r^H C t) = sum_l [Re y_l Re t_l + Im y_l Im t_l],
//   Re(t^H C t) = sum_k Re C_kk |t_k|^2
//                 + sum_{k<l} [(Re C_kl + Re C_lk) Re p_kl + (Im C_lk - Im C_kl) Im p_kl],
// with p_kl = conj(t_k) t_l. Both hold for any C; nothing assumes C is Hermitian.

#include <stdexcept>
#include <vector>
#include "stft_cross_terms.hpp"

#ifdef __CUDACC__
#define CROSS_THREADS 128
#else
#define CROSS_THREADS 1
#endif

// Residuals and templates of one pair tile: a thread block computes up to CROSS_TILE^2 pairs.
#define CROSS_TILE 32
// Pixels of one partial sum. Larger ranges mean fewer partial sums and fewer thread blocks.
#define CROSS_RANGE_PIXELS 2048
// Pixels staged in shared memory at once: (2 x 15 + 1) x 32 x 4 doubles, 31 KiB.
#define CROSS_STAGE_PIXELS 4
// Staged reals per residual or template and pixel: 6 values, then 9 noise terms or products.
#define CROSS_TERMS 15
#define CROSS_VALUE_TERMS 6
#define CROSS_PAIRS_PER_THREAD (CROSS_TILE * CROSS_TILE / CROSS_THREADS)
#define CROSS_RESIDUALS_PER_THREAD ((CROSS_TILE + CROSS_THREADS - 1) / CROSS_THREADS)

static_assert((CROSS_TILE * CROSS_TILE) % CROSS_THREADS == 0,
              "every thread must own the same number of pairs of a tile");

/**
 * Staged terms of one residual at one pixel, written with stride CROSS_TILE:
 * y = C^H r (Re, Im per channel), then Re C_kk (k = 0, 1, 2), then the pair terms of
 * (0, 1), (0, 2), (1, 2). Zero outside the residual's data window.
 */
CUDA_DEVICE
static void cross_residual_terms(STFTDomain& domain, int time_i, int freq_j, int data_row,
                                 int noise_row, int start_j, double* terms, double* self_term) {
  for (int term = 0; term < CROSS_TERMS; term++) terms[term * CROSS_TILE] = 0.0;
  *self_term = 0.0;
  int freq_local = freq_j - start_j;
  if ((freq_j < 0) || (freq_local < 0) || (freq_local >= domain.num_freqs)) return;
  if (domain.noise_full_grid && (freq_j >= domain.num_freqs_noise)) return;
  int noise_j = domain.noise_bin(freq_j, freq_local);

  cmplx residual_vals[3] = {cmplx(0.0, 0.0), cmplx(0.0, 0.0), cmplx(0.0, 0.0)};
  cmplx weighted[3] = {cmplx(0.0, 0.0), cmplx(0.0, 0.0), cmplx(0.0, 0.0)};
  if (domain.tdi_type == TDI_XYZ) {
    cmplx inv_cov[3][3];
    for (int ch_i = 0; ch_i < 3; ch_i++) {
      residual_vals[ch_i] = domain.get_data_value(time_i, freq_local, ch_i, data_row);
      for (int ch_j = 0; ch_j < 3; ch_j++)
        inv_cov[ch_i][ch_j] = domain.get_invC_cross_value(time_i, noise_j, ch_i, ch_j, noise_row);
    }
    for (int ch_j = 0; ch_j < 3; ch_j++)
      for (int ch_i = 0; ch_i < 3; ch_i++)
        weighted[ch_j] += gcmplx::conj(inv_cov[ch_i][ch_j]) * residual_vals[ch_i];
    for (int ch = 0; ch < 3; ch++)
      terms[(CROSS_VALUE_TERMS + ch) * CROSS_TILE] = inv_cov[ch][ch].real();
    const int pair_first[3] = {0, 0, 1};
    const int pair_second[3] = {1, 2, 2};
    for (int pair = 0; pair < 3; pair++) {
      cmplx upper = inv_cov[pair_first[pair]][pair_second[pair]];
      cmplx lower = inv_cov[pair_second[pair]][pair_first[pair]];
      terms[(CROSS_VALUE_TERMS + 3 + 2 * pair) * CROSS_TILE] = upper.real() + lower.real();
      terms[(CROSS_VALUE_TERMS + 4 + 2 * pair) * CROSS_TILE] = lower.imag() - upper.imag();
    }
  } else {
    for (int ch = 0; ch < domain.num_channels; ch++) {
      cmplx inv_cov_ch = domain.get_invC_value(time_i, noise_j, ch, noise_row);
      residual_vals[ch] = domain.get_data_value(time_i, freq_local, ch, data_row);
      weighted[ch] = gcmplx::conj(inv_cov_ch) * residual_vals[ch];
      terms[(CROSS_VALUE_TERMS + ch) * CROSS_TILE] = inv_cov_ch.real();
    }
  }
  double self_sum = 0.0;
  for (int ch = 0; ch < 3; ch++) {
    terms[(2 * ch) * CROSS_TILE] = weighted[ch].real();
    terms[(2 * ch + 1) * CROSS_TILE] = weighted[ch].imag();
    self_sum += weighted[ch].real() * residual_vals[ch].real() +
                weighted[ch].imag() * residual_vals[ch].imag();
  }
  *self_term = self_sum;
}

/**
 * Staged terms of one template at one pixel, written with stride CROSS_TILE: its values
 * (Re, Im per channel), then |t_k|^2 (k = 0, 1, 2), then Re and Im of conj(t_k) t_l for
 * (0, 1), (0, 2), (1, 2). Zero outside the template's window.
 */
CUDA_DEVICE
static void cross_template_terms(cmplx* template_vals, int num_template_rows, int num_channels,
                                 int num_times, int num_freqs_template, int time_i, int freq_j,
                                 int row, int start_j, double* terms) {
  for (int term = 0; term < CROSS_TERMS; term++) terms[term * CROSS_TILE] = 0.0;
#ifndef __CUDACC__
  if ((row < 0) || (row >= num_template_rows))
    throw std::invalid_argument("stft_cross_terms: template_index is outside the template rows.");
#endif
  int freq_local = freq_j - start_j;
  if ((freq_local < 0) || (freq_local >= num_freqs_template)) return;
  cmplx vals[3] = {cmplx(0.0, 0.0), cmplx(0.0, 0.0), cmplx(0.0, 0.0)};
  for (int ch = 0; ch < num_channels; ch++)
    vals[ch] = template_vals[(((size_t)row * num_channels + ch) * num_times + time_i) *
                                 num_freqs_template + freq_local];
  for (int ch = 0; ch < 3; ch++) {
    terms[(2 * ch) * CROSS_TILE] = vals[ch].real();
    terms[(2 * ch + 1) * CROSS_TILE] = vals[ch].imag();
    terms[(CROSS_VALUE_TERMS + ch) * CROSS_TILE] =
        vals[ch].real() * vals[ch].real() + vals[ch].imag() * vals[ch].imag();
  }
  const int pair_first[3] = {0, 0, 1};
  const int pair_second[3] = {1, 2, 2};
  for (int pair = 0; pair < 3; pair++) {
    cmplx first = vals[pair_first[pair]];
    cmplx second = vals[pair_second[pair]];
    terms[(CROSS_VALUE_TERMS + 3 + 2 * pair) * CROSS_TILE] =
        first.real() * second.real() + first.imag() * second.imag();
    terms[(CROSS_VALUE_TERMS + 4 + 2 * pair) * CROSS_TILE] =
        first.real() * second.imag() - first.imag() * second.real();
  }
}

/**
 * Pass 1: one partial sum per (table, pixel range, residual, template) and per
 * (table, pixel range, residual).
 *
 * Grid (GPU): x runs over table * num_ranges + range, y over the pair tiles. A thread owns the
 * pairs THREAD_START_X + CROSS_THREADS * q of its tile, so a warp shares one residual and reads 32
 * consecutive templates. Partial layout: [num_tables, num_ranges, num_residuals, num_templates]
 * and [num_tables, num_ranges, num_residuals].
 */
CUDA_KERNEL
static void compute_cross_terms_contributions_kernel(
    double* cross_partial, double* template_partial, double* residual_partial,
    STFTDomain domain, cmplx* template_vals, int num_template_rows, int num_freqs_template,
    int* residual_index_all, int* noise_index_all, int* residual_start_all,
    int* template_index_all, int* template_start_all, int* window_lo_all, int* window_hi_all,
    int num_tables, int num_residuals, int num_templates, int num_ranges) {
  CUDA_SHARED double residual_stage[CROSS_STAGE_PIXELS][CROSS_TERMS * CROSS_TILE];
  CUDA_SHARED double template_stage[CROSS_STAGE_PIXELS][CROSS_TERMS * CROSS_TILE];
  CUDA_SHARED double residual_self_stage[CROSS_STAGE_PIXELS][CROSS_TILE];

  int num_tiles_template = (num_templates + CROSS_TILE - 1) / CROSS_TILE;
  int num_tiles = ((num_residuals + CROSS_TILE - 1) / CROSS_TILE) * num_tiles_template;
  size_t num_pairs = (size_t)num_residuals * num_templates;

  for (int range_block = BLOCK_START_X; range_block < num_tables * num_ranges;
       range_block += GRID_INCR_X) {
    int table = range_block / num_ranges;
    int range = range_block % num_ranges;
    int window_lo = window_lo_all[table];
    int window_width = window_hi_all[table] - window_lo;
    int num_pixels = (window_width > 0) ? domain.num_times * window_width : 0;
#ifndef __CUDACC__
    if (num_pixels > num_ranges * CROSS_RANGE_PIXELS)
      throw std::invalid_argument(
          "stft_cross_terms: a table's window is wider than max_window_width.");
#endif
    int pixel_begin = range * CROSS_RANGE_PIXELS;
    int pixel_end = pixel_begin + CROSS_RANGE_PIXELS;
    if (pixel_end > num_pixels) pixel_end = num_pixels;

    for (int tile = BLOCK_START_Y; tile < num_tiles; tile += GRID_INCR_Y) {
      int residual_base = (tile / num_tiles_template) * CROSS_TILE;
      int template_base = (tile % num_tiles_template) * CROSS_TILE;
      // The residual terms do not depend on the templates: the first template tile sums them.
      bool sums_residual_terms = (template_base == 0);

      double cross_acc[CROSS_PAIRS_PER_THREAD];
      double template_acc[CROSS_PAIRS_PER_THREAD];
      double residual_acc[CROSS_RESIDUALS_PER_THREAD];
      for (int slot = 0; slot < CROSS_PAIRS_PER_THREAD; slot++) {
        cross_acc[slot] = 0.0;
        template_acc[slot] = 0.0;
      }
      for (int slot = 0; slot < CROSS_RESIDUALS_PER_THREAD; slot++) residual_acc[slot] = 0.0;

      for (int stage_begin = pixel_begin; stage_begin < pixel_end;
           stage_begin += CROSS_STAGE_PIXELS) {
        int stage_count = pixel_end - stage_begin;
        if (stage_count > CROSS_STAGE_PIXELS) stage_count = CROSS_STAGE_PIXELS;

        // Consecutive entries are consecutive pixels of one row, so the loads share segments.
        for (int entry = THREAD_START_X; entry < 2 * CROSS_TILE * CROSS_STAGE_PIXELS;
             entry += BLOCK_INCR_X) {
          bool is_template = entry >= CROSS_TILE * CROSS_STAGE_PIXELS;
          int within = entry % (CROSS_TILE * CROSS_STAGE_PIXELS);
          int slot = within / CROSS_STAGE_PIXELS;
          int stage_pixel = within % CROSS_STAGE_PIXELS;
          if (stage_pixel >= stage_count) continue;
          int pixel = stage_begin + stage_pixel;
          int time_i = pixel / window_width;
          int freq_j = window_lo + pixel % window_width;
          if (!is_template) {
            int residual = residual_base + slot;
            if (residual >= num_residuals) continue;
            size_t at = (size_t)table * num_residuals + residual;
            cross_residual_terms(domain, time_i, freq_j, residual_index_all[at],
                                 noise_index_all[at], residual_start_all[at],
                                 &residual_stage[stage_pixel][slot],
                                 &residual_self_stage[stage_pixel][slot]);
          } else {
            int template_j = template_base + slot;
            if (template_j >= num_templates) continue;
            size_t at = (size_t)table * num_templates + template_j;
            cross_template_terms(template_vals, num_template_rows, domain.num_channels,
                                 domain.num_times, num_freqs_template, time_i, freq_j,
                                 template_index_all[at], template_start_all[at],
                                 &template_stage[stage_pixel][slot]);
          }
        }
        CUDA_SYNC_THREADS;

#ifdef __CUDACC__
#pragma unroll
#endif
        for (int slot = 0; slot < CROSS_PAIRS_PER_THREAD; slot++) {
          int pair = THREAD_START_X + BLOCK_INCR_X * slot;
          int residual_slot = pair / CROSS_TILE;
          int template_slot = pair % CROSS_TILE;
          if ((residual_base + residual_slot >= num_residuals) ||
              (template_base + template_slot >= num_templates))
            continue;
          for (int stage_pixel = 0; stage_pixel < stage_count; stage_pixel++) {
            double* residual_terms = &residual_stage[stage_pixel][residual_slot];
            double* template_terms = &template_stage[stage_pixel][template_slot];
            double cross_term = residual_terms[0] * template_terms[0];
            for (int term = 1; term < CROSS_VALUE_TERMS; term++)
              cross_term += residual_terms[term * CROSS_TILE] * template_terms[term * CROSS_TILE];
            double template_term = residual_terms[CROSS_VALUE_TERMS * CROSS_TILE] *
                                   template_terms[CROSS_VALUE_TERMS * CROSS_TILE];
            for (int term = CROSS_VALUE_TERMS + 1; term < CROSS_TERMS; term++)
              template_term +=
                  residual_terms[term * CROSS_TILE] * template_terms[term * CROSS_TILE];
            cross_acc[slot] += cross_term;
            template_acc[slot] += template_term;
          }
        }
        if (sums_residual_terms) {
          for (int slot = 0; slot < CROSS_RESIDUALS_PER_THREAD; slot++) {
            int residual_slot = THREAD_START_X + BLOCK_INCR_X * slot;
            if ((residual_slot >= CROSS_TILE) || (residual_base + residual_slot >= num_residuals))
              continue;
            for (int stage_pixel = 0; stage_pixel < stage_count; stage_pixel++)
              residual_acc[slot] += residual_self_stage[stage_pixel][residual_slot];
          }
        }
        // ! The next stage overwrites the shared terms: every thread must be done reading them.
        CUDA_SYNC_THREADS;
      }

      for (int slot = 0; slot < CROSS_PAIRS_PER_THREAD; slot++) {
        int pair = THREAD_START_X + BLOCK_INCR_X * slot;
        int residual = residual_base + pair / CROSS_TILE;
        int template_j = template_base + pair % CROSS_TILE;
        if ((residual >= num_residuals) || (template_j >= num_templates)) continue;
        size_t at = (size_t)range_block * num_pairs + (size_t)residual * num_templates + template_j;
        cross_partial[at] = cross_acc[slot];
        template_partial[at] = template_acc[slot];
      }
      if (sums_residual_terms) {
        for (int slot = 0; slot < CROSS_RESIDUALS_PER_THREAD; slot++) {
          int residual_slot = THREAD_START_X + BLOCK_INCR_X * slot;
          int residual = residual_base + residual_slot;
          if ((residual_slot >= CROSS_TILE) || (residual >= num_residuals)) continue;
          residual_partial[(size_t)range_block * num_residuals + residual] = residual_acc[slot];
        }
      }
    }
  }
}

/**
 * Pass 2: add the range partial sums of each table in range order and apply 4 df.
 * Grid (GPU): x runs over the tables; threads over the pairs and the residuals.
 */
CUDA_KERNEL
static void cross_terms_sum_from_contrib(double* cross_out, double* template_out,
                                         double* residual_out, double* cross_partial,
                                         double* template_partial, double* residual_partial,
                                         double prefactor, int num_tables, int num_residuals,
                                         int num_templates, int num_ranges) {
  size_t num_pairs = (size_t)num_residuals * num_templates;
  for (int table = BLOCK_START_X; table < num_tables; table += GRID_INCR_X) {
    for (size_t pair = THREAD_START_X; pair < num_pairs; pair += BLOCK_INCR_X) {
      double cross_sum = 0.0;
      double template_sum = 0.0;
      for (int range = 0; range < num_ranges; range++) {
        size_t at = ((size_t)table * num_ranges + range) * num_pairs + pair;
        cross_sum += cross_partial[at];
        template_sum += template_partial[at];
      }
      cross_out[(size_t)table * num_pairs + pair] = prefactor * cross_sum;
      template_out[(size_t)table * num_pairs + pair] = prefactor * template_sum;
    }
    for (int residual = THREAD_START_X; residual < num_residuals; residual += BLOCK_INCR_X) {
      double residual_sum = 0.0;
      for (int range = 0; range < num_ranges; range++)
        residual_sum +=
            residual_partial[((size_t)table * num_ranges + range) * num_residuals + residual];
      residual_out[(size_t)table * num_residuals + residual] = prefactor * residual_sum;
    }
  }
}

void stft_cross_terms_wrap(STFTDomain& domain, double* cross_out, double* template_out,
                           double* residual_out, cmplx* template_vals, int num_template_rows,
                           int num_freqs_template, int* residual_index, int* noise_index,
                           int* residual_start, int* template_index, int* template_start,
                           int* window_lo, int* window_hi, int num_tables, int num_residuals,
                           int num_templates, int max_window_width, bool run_async) {
  if (num_tables <= 0) return;
  if ((num_residuals <= 0) || (num_templates <= 0) || (num_freqs_template <= 0))
    throw std::invalid_argument(
        "stft_cross_terms: a table needs at least one residual, one template and one bin.");

  long long num_pixels_max =
      (long long)domain.num_times * (max_window_width > 0 ? max_window_width : 0);
  long long num_ranges_ll = (num_pixels_max + CROSS_RANGE_PIXELS - 1) / CROSS_RANGE_PIXELS;
  if (num_ranges_ll < 1) num_ranges_ll = 1;
  if ((long long)num_tables * num_ranges_ll > 2147483647LL)
    throw std::invalid_argument("stft_cross_terms: too many tables and ranges for one call.");
  int num_ranges = (int)num_ranges_ll;
  size_t num_pair_partials = (size_t)num_tables * num_ranges * num_residuals * num_templates;
  size_t num_residual_partials = (size_t)num_tables * num_ranges * num_residuals;
  double prefactor = 4.0 * domain.diff_comp;

#ifdef __CUDACC__
  double* cross_partial;
  double* template_partial;
  double* residual_partial;
  if (run_async) {
    gpuErrchk(cudaMallocAsync(&cross_partial, num_pair_partials * sizeof(double),
                              cudaStreamDefault));
    gpuErrchk(cudaMallocAsync(&template_partial, num_pair_partials * sizeof(double),
                              cudaStreamDefault));
    gpuErrchk(cudaMallocAsync(&residual_partial, num_residual_partials * sizeof(double),
                              cudaStreamDefault));
  } else {
    gpuErrchk(cudaMalloc(&cross_partial, num_pair_partials * sizeof(double)));
    gpuErrchk(cudaMalloc(&template_partial, num_pair_partials * sizeof(double)));
    gpuErrchk(cudaMalloc(&residual_partial, num_residual_partials * sizeof(double)));
  }

  int num_tiles = ((num_residuals + CROSS_TILE - 1) / CROSS_TILE) *
                  ((num_templates + CROSS_TILE - 1) / CROSS_TILE);
  // The tiles beyond the grid's y limit are taken by the kernel's own loop.
  dim3 grid_dim(num_tables * num_ranges, num_tiles < 65535 ? num_tiles : 65535);
  compute_cross_terms_contributions_kernel<<<grid_dim, CROSS_THREADS>>>(
      cross_partial, template_partial, residual_partial, domain, template_vals,
      num_template_rows, num_freqs_template, residual_index, noise_index, residual_start,
      template_index, template_start, window_lo, window_hi, num_tables, num_residuals,
      num_templates, num_ranges);
  cross_terms_sum_from_contrib<<<num_tables, CROSS_THREADS>>>(
      cross_out, template_out, residual_out, cross_partial, template_partial, residual_partial,
      prefactor, num_tables, num_residuals, num_templates, num_ranges);
  gpuErrchk(cudaGetLastError());

  if (run_async) {
    gpuErrchk(cudaFreeAsync(cross_partial, cudaStreamDefault));
    gpuErrchk(cudaFreeAsync(template_partial, cudaStreamDefault));
    gpuErrchk(cudaFreeAsync(residual_partial, cudaStreamDefault));
  } else {
    gpuErrchk(cudaFree(cross_partial));
    gpuErrchk(cudaFree(template_partial));
    gpuErrchk(cudaFree(residual_partial));
    cudaDeviceSynchronize();
  }
#else
  (void)run_async;  // The CPU path is synchronous.
  std::vector<double> cross_partial(num_pair_partials);
  std::vector<double> template_partial(num_pair_partials);
  std::vector<double> residual_partial(num_residual_partials);
  compute_cross_terms_contributions_kernel(
      cross_partial.data(), template_partial.data(), residual_partial.data(), domain,
      template_vals, num_template_rows, num_freqs_template, residual_index, noise_index,
      residual_start, template_index, template_start, window_lo, window_hi, num_tables,
      num_residuals, num_templates, num_ranges);
  cross_terms_sum_from_contrib(cross_out, template_out, residual_out, cross_partial.data(),
                               template_partial.data(), residual_partial.data(), prefactor,
                               num_tables, num_residuals, num_templates, num_ranges);
#endif
}
