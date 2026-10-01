#ifndef __LAT_SPLINE_TDI_WAVEFORM_HH__
#define __LAT_SPLINE_TDI_WAVEFORM_HH__

// LISATDIonTheFly subclasses for spline-fed time-domain (TDSpline) and
// frequency-domain (FDSpline) on-the-fly TDI templates. These exist
// primarily as glue classes that plug a CubicSpline-driven intrinsic
// waveform (amp/phase or amp/freq) into the LAT-owned
// LISATDIonTheFly::run_wave_tdi pipeline.
//
// Phase 3L.6 (2026-06-03): moved from
//   lisa-on-gpu/src/fastlisaresponse/cutils/TDIonTheFly.hh:377-453
//   lisa-on-gpu/src/fastlisaresponse/cutils/TDIonTheFly.cu:9659-9774
//                                                          :9841-9986
// to LISAanalysistools. lat_spline_tdi_waveform.cu compiles the method
// bodies + the kernel/wrap host launchers. LAT's own CPU (.cxx copy) and
// GPU static libs are the only compilers of this TU -- lisa-on-gpu retired
// its cutils/ copy-compile at Phase 3L.7n (2026-06-04).

#include "gbt_global.h"
#include "Detector.hpp"        // Orbits, Vec
#include "LISAResponse.hh"     // TDIConfig
#include "Interpolate.hh"      // CubicSpline + NLINKS
#include "lat_tdi_on_the_fly.hh"

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#define TDSplineTDIWaveform TDSplineTDIWaveformGPU
#define FDSplineTDIWaveform FDSplineTDIWaveformGPU
#define TDDenseTDIWaveform TDDenseTDIWaveformGPU
#else
#define TDSplineTDIWaveform TDSplineTDIWaveformCPU
#define FDSplineTDIWaveform FDSplineTDIWaveformCPU
#define TDDenseTDIWaveform TDDenseTDIWaveformCPU
#endif

class TDSplineTDIWaveform : public LISATDIonTheFly{
  public:
    CubicSpline *amp_spline;
    CubicSpline *phase_spline;
    int binary_index_storage;

    CUDA_CALLABLE_MEMBER
    TDSplineTDIWaveform(Orbits* orbits_, TDIConfig *tdi_config_, CubicSpline *amp_spline_, CubicSpline *phase_spline_): LISATDIonTheFly(orbits_, tdi_config_, 0, 1, 2, 3){
        amp_spline = amp_spline_;
        phase_spline = phase_spline_;
    };
    CUDA_CALLABLE_MEMBER
    ~TDSplineTDIWaveform(){};
    CUDA_CALLABLE_MEMBER
    int get_td_spline_buffer_size(int N){return get_tdi_buffer_size(N);};
    CUDA_DEVICE
    void check_x();
    CUDA_DEVICE
    double get_amp(double t, double *params, int spline_i);
    CUDA_DEVICE
    double get_phase(double t, double *params, int spline_i);
};

// ---------------------------------------------------------------------------
// TDDenseTDIWaveform: template-batched TD TDI-on-the-fly for multi-harmonic
// sources whose harmonics share ONE trajectory (EMRIs), with EXACT phases.
//
// Each template b carries the integrator's dense output: knots t_knots[b, :K_b]
// and the DOPR853 8th-order coefficients of its three fundamental phases
// (phase_coeffs[b, K-1, 3, 8], conventions -- massratio scaling, sign(xI0),
// backwards offset -- applied by the caller). Each harmonic (sub) s belongs to
// template sub_temp[s], carries integers (m, k, n) = sub_mkn[3s:3s+3] and a
// complex amplitude c_s(t) as a cubic spline over its template's knots
// (amp_re/amp_im[s, K-1, 4], monomials in dt from the left knot). The
// harmonic's strain term is  z_s(t) = amp_factor c_s(t) exp(-i Phi_s(t)),
// Phi_s = m Phi_phi + k Phi_theta + n Phi_r, zero outside the trajectory.
//
// Per (template, time) the link geometry (light-travel times, positions,
// projections, TDI weights) is computed ONCE and reused by every harmonic of
// the template: the per-harmonic cost is two complex evaluations of z_s per
// TDI unit. phi_ref[s, t] is the pure carrier Phi_s at spacecraft-1 time; the
// harmonic's amplitude phase lands in tdi_phase (unwrapped as usual).
// Kernel 1 (channels): one thread per (template, time). Kernel 2: one block per
// harmonic, the same amplitude/phase extraction + unwrap as get_tdi.
// ---------------------------------------------------------------------------
class TDDenseTDIWaveform : public LISATDIonTheFly{
  public:
    int n_temp;
    int K;               // knot stride per template (padded)
    int num_sub;
    double amp_factor;
    int *sub_temp;       // (num_sub,)
    int *sub_mkn;        // (num_sub, 3)
    int *n_knots;        // (n_temp,)
    double *t_knots;     // (n_temp, K)
    double *phase_coeffs;// (n_temp, K - 1, 3, 8)
    double *amp_re;      // (num_sub, K - 1, 4)
    double *amp_im;      // (num_sub, K - 1, 4)

    CUDA_CALLABLE_MEMBER
    TDDenseTDIWaveform(Orbits* orbits_, TDIConfig *tdi_config_, int n_temp_, int K_, int num_sub_,
        double amp_factor_, int *sub_temp_, int *sub_mkn_, int *n_knots_, double *t_knots_,
        double *phase_coeffs_, double *amp_re_, double *amp_im_): LISATDIonTheFly(orbits_, tdi_config_, 0, 1, 2, 3){
        n_temp = n_temp_; K = K_; num_sub = num_sub_; amp_factor = amp_factor_;
        sub_temp = sub_temp_; sub_mkn = sub_mkn_; n_knots = n_knots_; t_knots = t_knots_;
        phase_coeffs = phase_coeffs_; amp_re = amp_re_; amp_im = amp_im_;
    };
    CUDA_CALLABLE_MEMBER
    ~TDDenseTDIWaveform(){};
    CUDA_CALLABLE_MEMBER
    int get_td_dense_buffer_size(int N){return get_tdi_buffer_size(N);};
    // segment of template b containing t (-1 if t is outside the trajectory)
    CUDA_DEVICE
    int segment(int b, double t, bool clamp);
    CUDA_DEVICE
    void phases(int b, int seg, double t, double *Phi3);
    CUDA_DEVICE
    cmplx strain_term(int s, int b, int seg, double t, double *Phi3);
    CUDA_DEVICE
    void channels_point(int b, int i, double t, double *params_b, int sub_lo, int sub_hi,
        cmplx *tdi_channels_arr, double *phi_ref, int N, int *link_rec, int *link_em);
    CUDA_DEVICE
    void postprocess_sub(void *buffer, cmplx *chan, double *amp, double *phase, double *phi_ref, int N);
    CUDA_DEVICE
    double get_amp(double t, double *params, int spline_i){return 0.0;};
    CUDA_DEVICE
    double get_phase(double t, double *params, int spline_i){return 0.0;};
};

// Host launcher for TDDenseTDIWaveform. tdi_channels_arr/tdi_amp/tdi_phase are
// (num_sub, nchannels, N), phi_ref (num_sub, N), params (n_temp, n_params) with
// (inc, psi, lam, beta), t_arr (n_temp, N), sub_offsets (n_temp + 1): the subs
// of template b are sub_offsets[b] .. sub_offsets[b + 1] - 1. Outputs must be
// zero on entry (the channels accumulate).
void td_dense_run_wave_tdi_wrap(TDDenseTDIWaveform *tdi_on_fly, cmplx *tdi_channels_arr,
    double *tdi_amp, double *tdi_phase, double *phi_ref,
    double *params, double *t_arr, int *sub_offsets, int N, int n_params, int nchannels);

// Host launcher: pulls Orbits/TDIConfig/CubicSpline structs onto the
// device, configures the device-side TDSplineTDIWaveform, runs the
// per-bin kernel, frees the temporary device-side mirrors. CPU branch
// runs the same path on a heap buffer.
void td_spline_run_wave_tdi_wrap(TDSplineTDIWaveform *tdi_on_fly, cmplx *tdi_channels_arr,
    double *tdi_amp, double *tdi_phase, double *phi_ref,
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels);


class FDSplineTDIWaveform : public LISATDIonTheFly {
    public:
        CubicSpline *amp_spline;
        CubicSpline *freq_spline;

    CUDA_CALLABLE_MEMBER
    FDSplineTDIWaveform(Orbits* orbits_, TDIConfig *tdi_config_, CubicSpline *amp_spline_, CubicSpline *freq_spline_): LISATDIonTheFly(orbits_, tdi_config_, 0, 1, 2, 3)
    {
        amp_spline = amp_spline_;
        freq_spline = freq_spline_;
    };
    CUDA_CALLABLE_MEMBER
    ~FDSplineTDIWaveform(){};
    CUDA_CALLABLE_MEMBER
    int get_fd_spline_buffer_size(int N){return get_tdi_buffer_size(N);};
    CUDA_DEVICE
    double get_phase_ref(double t, double *params, int bin_i);
    CUDA_DEVICE
    double get_amp(double t, double *params, int spline_i);
    CUDA_DEVICE
    double get_phase(double t, double *params, int spline_i);
    CUDA_DEVICE
    void get_tdi(void *buffer, int buffer_length, cmplx *tdi_channels_arr, double *tdi_amp, double *tdi_phase, double* phi_ref, double *params, double *t_arr, int N, int bin_i, int nchannels);
    CUDA_DEVICE
    double get_amp_f(double t, double *params, int spline_i);
};

// Host launcher mirror of td_spline_run_wave_tdi_wrap for the FD path.
void fd_spline_run_wave_tdi_wrap(FDSplineTDIWaveform *tdi_on_fly, cmplx *tdi_channels_arr,
    double *tdi_amp, double *tdi_phase, double *phi_ref,
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels);

#endif // __LAT_SPLINE_TDI_WAVEFORM_HH__
