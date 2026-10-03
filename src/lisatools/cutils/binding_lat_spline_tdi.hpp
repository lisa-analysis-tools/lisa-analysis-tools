#ifndef __BINDING_LAT_SPLINE_TDI_HPP__
#define __BINDING_LAT_SPLINE_TDI_HPP__

// pybind11 wrappers for the LAT-owned LISATDIonTheFly subclasses:
//   LISATDIonTheFlyWrap     -- base holder for OrbitsWrap + TDIConfigWrap
//                              (used as a common base for all source-class
//                              TDIonTheFly wrappers)
//   TDSplineTDIWaveformWrap -- pybind11 holder for TDSplineTDIWaveform
//   FDSplineTDIWaveformWrap -- pybind11 holder for FDSplineTDIWaveform
//
// Phase 3L.6 (2026-06-03): moved from
//   lisa-on-gpu/src/fastlisaresponse/cutils/binding_tof.hpp:45-105
// to LISAanalysistools. The pybind11 registrations live in LAT's
// binding_flr.cxx (response_part(m)) so the single-registrant rule
// owns these classes.
//
// GBTDIonTheFlyWrap and SOBBHTDIonTheFlyWrap (still in lisa-on-gpu)
// inherit from LISATDIonTheFlyWrap via the LAT include; that's the
// same pattern the underlying GBTDIonTheFly / SOBBHTDIonTheFly use to
// inherit from the LAT-owned LISATDIonTheFly base.

#include "binding_flr.hpp"     // ReturnPointerBase, TDIConfigWrap,
                               // CubicSplineWrap
#include "binding_detector.hpp"         // OrbitsWrap, array_type<T>
#include "lat_spline_tdi_waveform.hh"

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#define LISATDIonTheFlyWrap     LISATDIonTheFlyWrapGPU
#define TDSplineTDIWaveformWrap TDSplineTDIWaveformWrapGPU
#define FDSplineTDIWaveformWrap FDSplineTDIWaveformWrapGPU
#define TDDenseTDIWaveformWrap TDDenseTDIWaveformWrapGPU
#else
#define LISATDIonTheFlyWrap     LISATDIonTheFlyWrapCPU
#define TDSplineTDIWaveformWrap TDSplineTDIWaveformWrapCPU
#define FDSplineTDIWaveformWrap FDSplineTDIWaveformWrapCPU
#define TDDenseTDIWaveformWrap TDDenseTDIWaveformWrapCPU
#endif

// Common base for all LISATDIonTheFly Wrap subclasses. Pure data holder
// for the OrbitsWrap + TDIConfigWrap pair.
class LISATDIonTheFlyWrap : public ReturnPointerBase {
  public:
    OrbitsWrap *orbits;
    TDIConfigWrap *tdi_config;
    LISATDIonTheFlyWrap(OrbitsWrap *orbits_, TDIConfigWrap *tdi_config_){
        orbits = orbits_;
        tdi_config = tdi_config_;
    };
};

class FDSplineTDIWaveformWrap : public LISATDIonTheFlyWrap {
  public:
    CubicSplineWrap *amp_spline;
    CubicSplineWrap *freq_spline;
    FDSplineTDIWaveform *waveform;
    FDSplineTDIWaveformWrap(OrbitsWrap *orbits_, TDIConfigWrap *tdi_config_, CubicSplineWrap *amp_spline_, CubicSplineWrap *freq_spline_): LISATDIonTheFlyWrap(orbits_, tdi_config_)
    {
        amp_spline = amp_spline_;
        freq_spline = freq_spline_;
        waveform = new FDSplineTDIWaveform(orbits_->orbits, tdi_config_->tdi_config, amp_spline_->spline, freq_spline_->spline);
    };
    ~FDSplineTDIWaveformWrap(){
        delete waveform;
    };

    inline void run_wave_tdi_wrap(
        array_type<std::complex<double>> tdi_channels_arr,
        array_type<double> tdi_amp, array_type<double> tdi_phase, array_type<double> phi_ref,
        array_type<double> params, array_type<double> t_arr, int N, int num_bin, int n_params, int nchannels)
    {
        fd_spline_run_wave_tdi_wrap(
            waveform,
            (cmplx*)return_pointer_and_check_length(tdi_channels_arr, "tdi_channels_arr", N, num_bin * nchannels),
            return_pointer_and_check_length(tdi_amp, "tdi_amp", N, num_bin * nchannels),
            return_pointer_and_check_length(tdi_phase, "tdi_phase", N, num_bin * nchannels),
            return_pointer_and_check_length(phi_ref, "phi_ref", N, num_bin),
            return_pointer_and_check_length(params, "params", n_params, num_bin),
            return_pointer_and_check_length(t_arr, "t_arr", N, num_bin),
            N, num_bin, n_params, nchannels
        );
    }

    inline int get_buffer_size(int N){return waveform->get_fd_spline_buffer_size(N);}
};


class TDSplineTDIWaveformWrap : public LISATDIonTheFlyWrap {
  public:
    CubicSplineWrap *amp_spline;
    CubicSplineWrap *phase_spline;
    TDSplineTDIWaveform *waveform;
    TDSplineTDIWaveformWrap(OrbitsWrap *orbits_, TDIConfigWrap *tdi_config_, CubicSplineWrap *amp_spline_, CubicSplineWrap *phase_spline_): LISATDIonTheFlyWrap(orbits_, tdi_config_)
    {
        amp_spline = amp_spline_;
        phase_spline = phase_spline_;
        waveform = new TDSplineTDIWaveform(orbits_->orbits, tdi_config_->tdi_config, amp_spline_->spline, phase_spline_->spline);
    };
    ~TDSplineTDIWaveformWrap(){
        delete waveform;
    };

    inline void run_wave_tdi_wrap(
        array_type<std::complex<double>> tdi_channels_arr,
        array_type<double> tdi_amp, array_type<double> tdi_phase, array_type<double> phi_ref,
        array_type<double> params, array_type<double> t_arr, int N, int num_bin, int n_params, int nchannels)
    {
        td_spline_run_wave_tdi_wrap(
            waveform,
            (cmplx*)return_pointer_and_check_length(tdi_channels_arr, "tdi_channels_arr", N, num_bin * nchannels),
            return_pointer_and_check_length(tdi_amp, "tdi_amp", N, num_bin * nchannels),
            return_pointer_and_check_length(tdi_phase, "tdi_phase", N, num_bin * nchannels),
            return_pointer_and_check_length(phi_ref, "phi_ref", N, num_bin),
            return_pointer_and_check_length(params, "params", n_params, num_bin),
            return_pointer_and_check_length(t_arr, "t_arr", N, num_bin),
            N, num_bin, n_params, nchannels
        );
    }

    inline int get_buffer_size(int N){return waveform->get_td_spline_buffer_size(N);}
};

// Template-batched TD TDI-on-the-fly with exact dense-output phases (see
// TDDenseTDIWaveform). The data arrays are NOT copied: the Python side keeps
// them alive for the life of the wrap.
class TDDenseTDIWaveformWrap : public LISATDIonTheFlyWrap {
  public:
    TDDenseTDIWaveform *waveform;
    TDDenseTDIWaveformWrap(OrbitsWrap *orbits_, TDIConfigWrap *tdi_config_, int n_temp, int K, int num_sub,
        double amp_factor, array_type<int> sub_temp, array_type<int> sub_mkn, array_type<int> n_knots,
        array_type<double> t_knots, array_type<double> phase_coeffs, array_type<double> amp_re,
        array_type<double> amp_im): LISATDIonTheFlyWrap(orbits_, tdi_config_)
    {
        waveform = new TDDenseTDIWaveform(orbits_->orbits, tdi_config_->tdi_config, n_temp, K, num_sub, amp_factor,
            return_pointer_and_check_length(sub_temp, "sub_temp", num_sub, 1),
            return_pointer_and_check_length(sub_mkn, "sub_mkn", num_sub, 3),
            return_pointer_and_check_length(n_knots, "n_knots", n_temp, 1),
            return_pointer_and_check_length(t_knots, "t_knots", K, n_temp),
            return_pointer_and_check_length(phase_coeffs, "phase_coeffs", (K - 1) * 3 * 8, n_temp),
            return_pointer_and_check_length(amp_re, "amp_re", (K - 1) * 4, num_sub),
            return_pointer_and_check_length(amp_im, "amp_im", (K - 1) * 4, num_sub));
    };
    ~TDDenseTDIWaveformWrap(){
        delete waveform;
    };

    inline void run_wave_tdi_wrap(
        array_type<std::complex<double>> tdi_channels_arr,
        array_type<double> tdi_amp, array_type<double> tdi_phase, array_type<double> phi_ref,
        array_type<double> params, array_type<double> t_arr, array_type<int> sub_offsets,
        int N, int n_params, int nchannels)
    {
        int num_sub = waveform->num_sub;
        int n_temp = waveform->n_temp;
        td_dense_run_wave_tdi_wrap(
            waveform,
            (cmplx*)return_pointer_and_check_length(tdi_channels_arr, "tdi_channels_arr", N, num_sub * nchannels),
            return_pointer_and_check_length(tdi_amp, "tdi_amp", N, num_sub * nchannels),
            return_pointer_and_check_length(tdi_phase, "tdi_phase", N, num_sub * nchannels),
            return_pointer_and_check_length(phi_ref, "phi_ref", N, num_sub),
            return_pointer_and_check_length(params, "params", n_params, n_temp),
            return_pointer_and_check_length(t_arr, "t_arr", N, n_temp),
            return_pointer_and_check_length(sub_offsets, "sub_offsets", n_temp + 1, 1),
            N, n_params, nchannels
        );
    }

    // the raw complex channels only: (num_sub, nch, N), or with sum_subs (n_temp, nch, N)
    inline void run_channels_wrap(array_type<std::complex<double>> tdi_channels_arr,
        array_type<double> params, array_type<double> t_arr, array_type<int> sub_offsets,
        int N, int n_params, int nchannels, int sum_subs)
    {
        int rows = sum_subs ? waveform->n_temp : waveform->num_sub;
        td_dense_run_channels_wrap(
            waveform,
            (cmplx*)return_pointer_and_check_length(tdi_channels_arr, "tdi_channels_arr", N, rows * nchannels),
            return_pointer_and_check_length(params, "params", n_params, waveform->n_temp),
            return_pointer_and_check_length(t_arr, "t_arr", N, waveform->n_temp),
            return_pointer_and_check_length(sub_offsets, "sub_offsets", waveform->n_temp + 1, 1),
            N, n_params, sum_subs
        );
    }

    inline int get_buffer_size(int N){return waveform->get_td_dense_buffer_size(N);}
};

// wdm_lookup_sum (fused sparse-response -> WDM lookup; lat_spline_tdi_waveform.hh).
// Optional arrays (counts; the carrier arrays when carrier == 0) may be size 0.
template<typename T>
static T* wdm_lookup_array(array_type<T> a, const char *name, size_t n, bool optional)
{
    if (optional && a.size() == 0) return nullptr;
#if !(defined(__CUDA_COMPILATION__) || defined(__CUDACC__))
    if (a.size() != n)
    {
        throw std::invalid_argument(std::string("wdm_lookup_sum: ") + name + " has length " +
            std::to_string(a.size()) + ", expected " + std::to_string(n) + ".");
    }
#else
    (void)name; (void)n;
#endif
    return a.data();
}

inline void wdm_lookup_sum_binding(
    array_type<double> out, int n_rows, array_type<uint64_t> counts,
    int carrier, int n_temp, int K, array_type<double> t_knots, array_type<int> n_knots,
    array_type<double> phase_coeffs,
    int num_sub, array_type<int> sub_temp, array_type<int> sub_row, array_type<int> sub_mkn,
    array_type<int> n_stop,
    int N, int nch, array_type<double> x,
    array_type<double> amp_y, array_type<double> amp_c1, array_type<double> amp_c2, array_type<double> amp_c3,
    array_type<double> res_y, array_type<double> res_c1, array_type<double> res_c2, array_type<double> res_c3,
    int n_lo, int n_hi, double t0, double layer_dt, double layer_df, int Nf, int Nt, int m_lo, int m_hi,
    int num_m_layers, double fdot_max, double f_min,
    array_type<double> coeff_c, array_type<double> coeff_s, int FD, int FF, double fdot0, double dfdot,
    double f0, double df, double f_lo, double f_hi, int ref_odd)
{
    if ((m_hi <= m_lo) || (n_lo < 0) || (n_hi > Nt) || (nch <= 0) || (N < 2) || (FF < 2) || (FD < 1))
        throw std::invalid_argument("wdm_lookup_sum: inconsistent grid/band/table sizes");
    if (carrier && (K < 2 || n_temp < 1))
        throw std::invalid_argument("wdm_lookup_sum: carrier needs K >= 2 knots and n_temp >= 1");
    size_t ns = (size_t)num_sub;
    size_t spl = ns * nch * N;
    WDMLookupSumArgs a;
    a.out = wdm_lookup_array(out, "out", (size_t)n_rows * nch * (m_hi - m_lo) * Nt, false);
    a.counts = reinterpret_cast<unsigned long long *>(wdm_lookup_array(counts, "counts", 2, true));
    a.carrier = carrier;
    a.K = K;
    a.t_knots = wdm_lookup_array(t_knots, "t_knots", (size_t)n_temp * K, !carrier);
    a.n_knots = wdm_lookup_array(n_knots, "n_knots", (size_t)n_temp, !carrier);
    a.phase_coeffs = wdm_lookup_array(phase_coeffs, "phase_coeffs", (size_t)n_temp * (K - 1) * 24, !carrier);
    a.num_sub = num_sub;
    a.sub_temp = wdm_lookup_array(sub_temp, "sub_temp", ns, !carrier);
    a.sub_row = wdm_lookup_array(sub_row, "sub_row", ns, false);
    a.sub_mkn = wdm_lookup_array(sub_mkn, "sub_mkn", 3 * ns, !carrier);
    a.n_stop = wdm_lookup_array(n_stop, "n_stop", ns, false);
    a.N = N;
    a.nch = nch;
    a.x = wdm_lookup_array(x, "x", spl, false);
    a.amp_y = wdm_lookup_array(amp_y, "amp_y", spl, false);
    a.amp_c1 = wdm_lookup_array(amp_c1, "amp_c1", spl, false);
    a.amp_c2 = wdm_lookup_array(amp_c2, "amp_c2", spl, false);
    a.amp_c3 = wdm_lookup_array(amp_c3, "amp_c3", spl, false);
    a.res_y = wdm_lookup_array(res_y, "res_y", spl, false);
    a.res_c1 = wdm_lookup_array(res_c1, "res_c1", spl, false);
    a.res_c2 = wdm_lookup_array(res_c2, "res_c2", spl, false);
    a.res_c3 = wdm_lookup_array(res_c3, "res_c3", spl, false);
    a.n_lo = n_lo;
    a.n_hi = n_hi;
    a.t0 = t0;
    a.layer_dt = layer_dt;
    a.layer_df = layer_df;
    a.Nf = Nf;
    a.Nt = Nt;
    a.m_lo = m_lo;
    a.m_hi = m_hi;
    a.num_m_layers = num_m_layers;
    a.fdot_max = fdot_max;
    a.f_min = f_min;
    a.tab.coeff_c = wdm_lookup_array(coeff_c, "coeff_c", (size_t)FD * FF, false);
    a.tab.coeff_s = wdm_lookup_array(coeff_s, "coeff_s", (size_t)FD * FF, false);
    a.tab.FD = FD;
    a.tab.FF = FF;
    a.tab.fdot0 = fdot0;
    a.tab.dfdot = dfdot;
    a.tab.f0 = f0;
    a.tab.df = df;
    a.tab.f_lo = f_lo;
    a.tab.f_hi = f_hi;
    a.tab.ref_odd = ref_odd;
    wdm_lookup_sum_wrap(a);
}

#endif // __BINDING_LAT_SPLINE_TDI_HPP__
