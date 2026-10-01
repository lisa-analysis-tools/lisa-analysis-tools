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

    inline int get_buffer_size(int N){return waveform->get_td_dense_buffer_size(N);}
};

#endif // __BINDING_LAT_SPLINE_TDI_HPP__
