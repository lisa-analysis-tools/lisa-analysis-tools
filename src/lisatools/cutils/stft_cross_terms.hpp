#ifndef __STFT_CROSS_TERMS_HPP__
#define __STFT_CROSS_TERMS_HPP__

// stft_cross_terms.{hpp,cu}: inner products of the data rows of an STFTDomain with the rows of a
// template array, grouped in tables.
//
// Table b holds num_residuals data rows of the domain (residual i is data row
// residual_index[b, i], read with noise row noise_index[b, i]) and num_templates rows of the
// template array (template j is row template_index[b, j]). With
// (a | c) = 4 df Re sum a^H C_i^-1 c over the pixels of the table and C_i^-1 the noise row of
// residual i:
//
//   cross[b, i, j]    = (r_i | t_j)
//   template[b, i, j] = (t_j | t_j)
//   residual[b, i]    = (r_i | r_i)
//
// The pixels of table b are every segment and the active bins [window_lo[b], window_hi[b]). A
// pixel outside a residual's data window adds nothing to that residual's terms, as in the
// likelihood kernels of domains.cu; a template is zero outside its own window. Noise rows are read
// through STFTDomain::noise_bin, so windowed and full-grid noise rows both work.
//
// Every value is a sum in one fixed order that does not depend on the other tables of a call or
// on the device, so tables split over calls or devices give identical results.
//
// Only LAT compiles this file. Downstream packages compile domains.cu and binding_domains.hpp
// without it, so the binding registers it in binding_detector.cxx.

#include "domains.hpp"

/**
 * @brief Cross terms of residual and template rows for @p num_tables tables (see the file note).
 *
 * @param domain             Residual side: data rows (cell windows), noise rows and grid.
 * @param cross_out          (r_i|t_j), shape [num_tables, num_residuals, num_templates].
 * @param template_out       (t_j|t_j) under the noise row of residual i, same shape.
 * @param residual_out       (r_i|r_i), shape [num_tables, num_residuals].
 * @param template_vals      Template rows, shape [num_template_rows, num_channels, num_times,
 *                           num_freqs_template].
 * @param residual_index     Data row of each residual, shape [num_tables, num_residuals].
 * @param noise_index        Noise row of each residual, same shape.
 * @param residual_start     Active bin of the first bin of each residual's data row, same shape.
 * @param template_index     Template row of each template, shape [num_tables, num_templates].
 * @param template_start     Active bin of the first bin of each template row, same shape.
 * @param window_lo          First active bin summed per table, shape [num_tables].
 * @param window_hi          One past the last active bin summed per table, same shape.
 * @param max_window_width   At least the largest window_hi - window_lo; sizes the launch.
 */
void stft_cross_terms_wrap(STFTDomain& domain, double* cross_out, double* template_out,
                           double* residual_out, cmplx* template_vals, int num_template_rows,
                           int num_freqs_template, int* residual_index, int* noise_index,
                           int* residual_start, int* template_index, int* template_start,
                           int* window_lo, int* window_hi, int num_tables, int num_residuals,
                           int num_templates, int max_window_width, bool run_async = false);

#endif  // __STFT_CROSS_TERMS_HPP__
