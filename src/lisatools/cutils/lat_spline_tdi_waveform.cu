// === TDSplineTDIWaveform + FDSplineTDIWaveform method bodies + host launchers ===
// Phase 3L.6 (2026-06-03): moved from
//   lisa-on-gpu/src/fastlisaresponse/cutils/TDIonTheFly.cu (lines 9659-9774, 9841-9976)
// to LISAanalysistools.

#include "lat_spline_tdi_waveform.hh"
#include "gbt_global.h"
#include "Interpolate.hh"
#include "LISAResponse.hh"
#include "Detector.hpp"
#include "lat_tdi_on_the_fly.hh"

#include <cstdlib>
#include <cstring>
#include <new>

#ifdef __CUDACC__
#define NUM_THREADS_HERE 128
#else
#define NUM_THREADS_HERE 1
#endif

#ifdef __CUDACC__
// ---------------------------------------------------------------
// Shared-memory-vs-global scratch launch helpers (GPU only)
// ---------------------------------------------------------------
// The td/fd spline kernels stage the phase-extract/unwrap scratch
// (get_tdi_buffer_size(N) = 21*N bytes) in dynamic shared memory. For large
// N (dense MBH chirp grids) that overruns the per-block shared budget, so we
// keep it in shared memory where it fits (opting in past the 48 KB default
// when the device allows) and fall back to per-block global memory above the
// device ceiling. See lat_tdi_on_the_fly.cu:get_tdi for the buffer carve.

// Per-block stride into the global-memory scratch fallback. The buffer is
// carved doubles-first (flip/pjump) in LISATDIonTheFly::get_tdi, so every
// block's slice must be 8-byte aligned; buffer_length = 21*N is aligned only
// when N % 8 == 0. Round up to 256 (>= double alignment, matches cudaMalloc
// granularity). MUST be identical host-side (allocation) and device-side
// (per-block offset), so it is __host__ __device__.
__host__ __device__ inline size_t spline_tdi_scratch_stride(int buffer_length)
{
    return ((size_t)buffer_length + 255) & ~(size_t)255;
}

// LISATOOLS_VERBOSE=1 (any non-empty, non-"0" value): print the shared-vs-
// global launch decision per call. Read once via a function-local static so
// there is no wrap/Python signature change and it works from every entry
// point. Matches the project's env-var debug-knob convention (GB_DEBUG=1).
static bool spline_tdi_verbose()
{
    static const bool v = [] {
        const char *e = std::getenv("LISATOOLS_VERBOSE");
        return e != nullptr && e[0] != '\0' && std::strcmp(e, "0") != 0;
    }();
    return v;
}

// Max dynamic shared bytes `kernel_func` may opt into on the CURRENT device:
// the opt-in ceiling minus the kernel's static shared footprint (params_here,
// link arrays, cub::BlockScan TempStorage). This is the shared-vs-global
// decision boundary that also guarantees the cudaFuncSetAttribute below is a
// legal value. Queried per call -- cheap next to the per-call cudaMallocs in
// these wraps, and correct per-device under multi-GPU runs.
static size_t spline_tdi_max_dynamic_shared(const void *kernel_func)
{
    int device = 0;
    gpuErrchk(cudaGetDevice(&device));
    int optin_max = 0;
    gpuErrchk(cudaDeviceGetAttribute(&optin_max,
        cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    cudaFuncAttributes attrs;
    gpuErrchk(cudaFuncGetAttributes(&attrs, kernel_func));
    return (size_t)optin_max - attrs.sharedSizeBytes;
}

// Shared scratch-placement decision for the td_spline/fd_spline kernels: keep
// dynamic shared memory when it fits (default 48 KB, or the device opt-in
// ceiling after cudaFuncSetAttribute), otherwise allocate a per-block global
// scratch. Mirrors + completes the GB fix (gb_tdi_on_the_fly.cu, commit
// 3cbdcf0), which opts in but has no fallback above the device ceiling.
//
// Returns the dynamic-shared byte count to launch with; 0 means "use the
// global fallback" and *d_scratch_out is set to the freshly cudaMalloc'd
// per-block scratch (caller frees after the launch), else *d_scratch_out is
// nullptr. `kernel_func` is passed only to cudaFunc{Get,Set}Attribute -- the
// actual <<<>>> launch stays on the concrete kernel name in each wrap (the
// codebase's established pattern; see gb_run_wave_tdi_kernel). `tag` names the
// kernel in verbose output ("td_spline" / "fd_spline").
static size_t spline_tdi_prepare_scratch(const char *tag, const void *kernel_func,
    int buffer_length, int N, int num_bin, char **d_scratch_out)
{
    const size_t DEFAULT_DYN_SMEM = 48 * 1024;  // default per-block cap, sm_70+
    size_t shared_bytes = (size_t)buffer_length;
    *d_scratch_out = nullptr;

    if (shared_bytes <= DEFAULT_DYN_SMEM)
    {
        // Legacy fast path: dynamic shared, no opt-in (bit-for-bit unchanged).
        if (spline_tdi_verbose())
            fprintf(stderr, "lisatools %s TDI-on-the-fly: N=%d, scratch=%d B "
                "<= 48 KB default; dynamic shared memory, no opt-in.\n",
                tag, N, buffer_length);
        return shared_bytes;
    }

    const size_t ceiling = spline_tdi_max_dynamic_shared(kernel_func);
    if (shared_bytes <= ceiling)
    {
        // Fast path: opt in to the larger per-block shared cap.
        if (spline_tdi_verbose())
            fprintf(stderr, "lisatools %s TDI-on-the-fly: N=%d, scratch=%d B "
                "> 48 KB default; cudaFuncSetAttribute opt-in "
                "(device ceiling %zu B), still dynamic shared memory.\n",
                tag, N, buffer_length, ceiling);
        gpuErrchk(cudaFuncSetAttribute(kernel_func,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared_bytes));
        return shared_bytes;
    }

    // Slow path: per-block global scratch. One-time notice even without
    // VERBOSE; every-call detail with it.
    const size_t stride = spline_tdi_scratch_stride(buffer_length);
    static bool warned = false;
    if (spline_tdi_verbose() || !warned)
    {
        fprintf(stderr, "lisatools %s TDI-on-the-fly: N=%d, scratch=%d B "
            "exceeds device dynamic-shared ceiling (%zu B); global-memory "
            "scratch fallback (%zu B = num_bin %d x %zu B/block; slower).\n",
            tag, N, buffer_length, ceiling,
            (size_t)num_bin * stride, num_bin, stride);
        warned = true;
    }
    // Footprint = num_bin x roundup(21*N, 256) -- one slice per block, smaller
    // than the 3*N*num_bin-double tdi_amp/tdi_phase arrays this call already
    // fills. If it ever bites, cap the grid at min(num_bin, max_blocks) and
    // allocate one slice per launched block: run_wave_tdi already grid-strides
    // bin_i by gridDim.x, so a grid smaller than num_bin is supported as-is.
    gpuErrchk(cudaMalloc(d_scratch_out, (size_t)num_bin * stride));
    return 0;
}
#endif // __CUDACC__

// ---------------------------------------------------------------
// FDSplineTDIWaveform::get_tdi (was TDIonTheFly.cu:9660-9689)
// ---------------------------------------------------------------
CUDA_DEVICE
void FDSplineTDIWaveform::get_tdi(void *buffer, int buffer_length, cmplx *tdi_channels_arr, double *tdi_amp, double *tdi_phase, double* phi_ref, double *params, double *t_arr, int N, int bin_i, int nchannels)
{
    LISATDIonTheFly::get_tdi(
        buffer, buffer_length,
        tdi_channels_arr, 
        tdi_amp, tdi_phase,
        phi_ref,
        params, t_arr, N, bin_i, nchannels
    );
    
    CUDA_SYNC_THREADS;
    double amp_f;
    
#ifdef __CUDACC__
    int start = threadIdx.x;
    int incr = blockDim.x;
#else // __CUDACC__
    int start = 0;
    int incr = 1;
#endif // __CUDACC__
    for (int i = start; i < N; i += incr)
    {
        amp_f = get_amp_f(t_arr[i], params, bin_i);
        for (int chan = 0; chan < tdi_config->num_channels; chan += 1)
        {
            tdi_amp[chan * N + i] *= amp_f;
        }
    }
    CUDA_SYNC_THREADS;
}

// ---------------------------------------------------------------
// TDSplineTDIWaveform accessors (was TDIonTheFly.cu:9692-9705)
// ---------------------------------------------------------------
CUDA_DEVICE
double TDSplineTDIWaveform::get_amp(double t, double *params, int spline_i)
{
    // printf("before amp: %d\n", amp_spline->ninterps);
    return amp_spline->eval_single(t, spline_i);
}

CUDA_DEVICE
double TDSplineTDIWaveform::get_phase(double t, double *params, int spline_i)
{
    // printf("before phase: %d\n", phase_spline->ninterps);
    
    return phase_spline->eval_single(t, spline_i);
}

// ---------------------------------------------------------------
// td_spline kernel + wrap (was TDIonTheFly.cu:9707-9774)
// ---------------------------------------------------------------
#ifdef __CUDACC__
CUDA_KERNEL
void td_spline_run_wave_tdi_kernel(TDSplineTDIWaveform *tdi_on_fly, int buffer_length, char *global_buffer, cmplx *tdi_channels_arr,
    double *tdi_amp, double *tdi_phase, double *phi_ref,
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels)
{
    extern CUDA_SHARED char shared_mem[];
    // Scratch selection: dynamic shared when the host launched with it
    // (global_buffer == nullptr), otherwise this block's slice of the global
    // scratch. run_wave_tdi grid-strides bin_i by gridDim.x, so the blockIdx.x
    // slice is private to this block across all its bins.
    void *buffer = (global_buffer != nullptr)
        ? (void*)(global_buffer + (size_t)blockIdx.x * spline_tdi_scratch_stride(buffer_length))
        : (void*)shared_mem;
    tdi_on_fly->run_wave_tdi(buffer, buffer_length, tdi_channels_arr, tdi_amp, tdi_phase, phi_ref,
        params, t_arr, N, num_bin, n_params, nchannels);
}

// Construct the polymorphic wave object IN device memory (placement new) so its
// vtable is the DEVICE vtable. A host-`new`'d object cudaMemcpy'd to the device
// carries a HOST vtable pointer; the first virtual get_amp/get_phase call in the
// kernel then dereferences host memory -> illegal access on GPU (silent on CPU,
// which never leaves host). The member pointers are the already-device-mirrored
// d_orbits / d_tdi_config / d_amp_spline / d_phase_spline.
CUDA_KERNEL
void td_spline_construct_kernel(TDSplineTDIWaveform *obj, Orbits *orbits, TDIConfig *tdi_config,
    CubicSpline *amp_spline, CubicSpline *phase_spline)
{
    new (obj) TDSplineTDIWaveform(orbits, tdi_config, amp_spline, phase_spline);
}
#endif

void td_spline_run_wave_tdi_wrap(TDSplineTDIWaveform *tdi_on_fly, cmplx *tdi_channels_arr,
    double *tdi_amp, double *tdi_phase, double *phi_ref,
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels)
{
#ifdef __CUDACC__
    Orbits *d_orbits;
    cudaMalloc(&d_orbits, sizeof(Orbits));
    gpuErrchk(cudaMemcpy(d_orbits, tdi_on_fly->orbits, sizeof(Orbits), cudaMemcpyHostToDevice));

    TDIConfig *d_tdi_config;
    cudaMalloc(&d_tdi_config, sizeof(TDIConfig));
    gpuErrchk(cudaMemcpy(d_tdi_config, tdi_on_fly->tdi_config, sizeof(TDIConfig), cudaMemcpyHostToDevice));

    CubicSpline *d_amp_spline;
    cudaMalloc(&d_amp_spline, sizeof(CubicSpline));
    gpuErrchk(cudaMemcpy(d_amp_spline, tdi_on_fly->amp_spline, sizeof(CubicSpline), cudaMemcpyHostToDevice));

    CubicSpline *d_phase_spline;
    cudaMalloc(&d_phase_spline, sizeof(CubicSpline));
    gpuErrchk(cudaMemcpy(d_phase_spline, tdi_on_fly->phase_spline, sizeof(CubicSpline), cudaMemcpyHostToDevice));

    // Build the wave object on the device (device vtable) rather than host-new +
    // cudaMemcpy (which would copy a host vtable pointer).
    TDSplineTDIWaveform *d_wave_here;
    cudaMalloc(&d_wave_here, sizeof(TDSplineTDIWaveform));
    td_spline_construct_kernel<<<1, 1>>>(d_wave_here, d_orbits, d_tdi_config, d_amp_spline, d_phase_spline);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    int buffer_length = tdi_on_fly->get_td_spline_buffer_size(N);

    char *d_scratch = nullptr;
    size_t shared_bytes = spline_tdi_prepare_scratch("td_spline",
        (const void *)td_spline_run_wave_tdi_kernel, buffer_length, N, num_bin, &d_scratch);

    td_spline_run_wave_tdi_kernel<<<num_bin, NUM_THREADS_HERE, shared_bytes>>>(
        d_wave_here, buffer_length, d_scratch, tdi_channels_arr, tdi_amp, tdi_phase, phi_ref,
        params, t_arr, N, num_bin, n_params, nchannels);

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    if (d_scratch != nullptr) gpuErrchk(cudaFree(d_scratch));

    gpuErrchk(cudaFree(d_orbits));
    gpuErrchk(cudaFree(d_tdi_config));
    gpuErrchk(cudaFree(d_amp_spline));
    gpuErrchk(cudaFree(d_phase_spline));
    gpuErrchk(cudaFree(d_wave_here));
#else

    // make buffer 
    int buffer_length = tdi_on_fly->get_td_spline_buffer_size(N);
    char *buffer = new char[buffer_length];
    tdi_on_fly->run_wave_tdi((void*)buffer, buffer_length, tdi_channels_arr, tdi_amp, tdi_phase, phi_ref,
        params, t_arr, N, num_bin, n_params, nchannels);
    delete[] buffer;
#endif
}

// ---------------------------------------------------------------
// FDSplineTDIWaveform accessors (was TDIonTheFly.cu:9841-9906)
// ---------------------------------------------------------------
CUDA_DEVICE
double FDSplineTDIWaveform::get_amp(double t, double *params, int spline_i)
{
    return 1.0;
}

CUDA_DEVICE
double FDSplineTDIWaveform::get_phase(double t, double *params, int spline_i)
{
    double f = freq_spline->eval_single(t, spline_i);
    return 2. * M_PI * f * t;
}

CUDA_DEVICE
double FDSplineTDIWaveform::get_amp_f(double t, double *params, int spline_i)
{
    // TODO: may want to do this in a fast way
    return amp_spline->eval_single(t, spline_i);
}


// CUDA_DEVICE
// void FDSplineTDIWaveform::run_wave_tdi(cmplx *tdi_channels_arr, 
//     double *Xamp, double *Xphase, double *Yamp, double *Yphase, double *Zamp, double *Zphase, double *phi_ref, 
//     double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels)
// {
//     for (int bin_i = 0; bin_i < num_sub; bin_i += 1)
//     {
//         // map to Tyson/Neil setup
//         double beta = params[bin_i * n_params + 3];
//         double costh = cos(M_PI / 2.0 - beta);
        
//         double lam = params[bin_i * n_params + 2];
//         double phi = lam;

//         double inc = params[bin_i * n_params + 0];
//         double cosi = cos(inc);

//         double psi = params[bin_i * n_params + 1];
//         double *params_here = &params[bin_i * n_params];
//         double *t_here = &t_arr[bin_i * N];
    
//         // TODO: CHECK THIS!!
//         get_tdi(
//             buffer, buffer_length, &X[bin_i * N], &Y[bin_i * N], &Z[bin_i * N], 
//             &Xamp[bin_i * N], &Xphase[bin_i * N],
//             &Yamp[bin_i * N], &Yphase[bin_i * N],
//             &Zamp[bin_i * N], &Zphase[bin_i * N], &phi_ref[bin_i * N],
//             params_here, t_here, N, costh, phi, cosi, psi, bin_i);
//     }
// }

CUDA_DEVICE
double FDSplineTDIWaveform::get_phase_ref(double t, double *params, int bin_i)
{
    // in FD, has to be fixed to 2 pi f_ssb t_ssb
    // t is t_ssb

    // t_i = t[i];
    // // TODO: should we make it so this is without the spline?
    double f = freq_spline->eval_single(t, bin_i);
    return 2. * M_PI * f * t;
    // phase[i] = 2. * M_PI * f * t_i;
    // phase[index] = phase_ref_store[spline_i * N + index];

}

// ---------------------------------------------------------------
// fd_spline kernel + wrap (was TDIonTheFly.cu:9909-9976)
// ---------------------------------------------------------------
#ifdef __CUDACC__
CUDA_KERNEL
void fd_spline_run_wave_tdi_kernel(FDSplineTDIWaveform *tdi_on_fly, int buffer_length, char *global_buffer, cmplx *tdi_channels_arr,
    double *tdi_amp, double *tdi_phase, double *phi_ref,
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels)
{
    extern CUDA_SHARED char shared_mem[];
    // See td_spline_run_wave_tdi_kernel: shared when launched with it,
    // otherwise this block's private slice of the global scratch.
    void *buffer = (global_buffer != nullptr)
        ? (void*)(global_buffer + (size_t)blockIdx.x * spline_tdi_scratch_stride(buffer_length))
        : (void*)shared_mem;
    tdi_on_fly->run_wave_tdi(buffer, buffer_length, tdi_channels_arr, tdi_amp, tdi_phase, phi_ref,
        params, t_arr, N, num_bin, n_params, nchannels);
}

// Device-side placement-new construction (device vtable) -- see
// td_spline_construct_kernel for why host-new + cudaMemcpy is wrong for a
// polymorphic object.
CUDA_KERNEL
void fd_spline_construct_kernel(FDSplineTDIWaveform *obj, Orbits *orbits, TDIConfig *tdi_config,
    CubicSpline *amp_spline, CubicSpline *freq_spline)
{
    new (obj) FDSplineTDIWaveform(orbits, tdi_config, amp_spline, freq_spline);
}
#endif

void fd_spline_run_wave_tdi_wrap(FDSplineTDIWaveform *tdi_on_fly, cmplx *tdi_channels_arr,
    double *tdi_amp, double *tdi_phase, double *phi_ref,
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels)
{
#ifdef __CUDACC__
    Orbits *d_orbits;
    cudaMalloc(&d_orbits, sizeof(Orbits));
    gpuErrchk(cudaMemcpy(d_orbits, tdi_on_fly->orbits, sizeof(Orbits), cudaMemcpyHostToDevice));

    TDIConfig *d_tdi_config;
    cudaMalloc(&d_tdi_config, sizeof(TDIConfig));
    gpuErrchk(cudaMemcpy(d_tdi_config, tdi_on_fly->tdi_config, sizeof(TDIConfig), cudaMemcpyHostToDevice));

    CubicSpline *d_amp_spline;
    cudaMalloc(&d_amp_spline, sizeof(CubicSpline));
    gpuErrchk(cudaMemcpy(d_amp_spline, tdi_on_fly->amp_spline, sizeof(CubicSpline), cudaMemcpyHostToDevice));

    CubicSpline *d_freq_spline;
    cudaMalloc(&d_freq_spline, sizeof(CubicSpline));
    gpuErrchk(cudaMemcpy(d_freq_spline, tdi_on_fly->freq_spline, sizeof(CubicSpline), cudaMemcpyHostToDevice));

    FDSplineTDIWaveform *d_wave_here;
    cudaMalloc(&d_wave_here, sizeof(FDSplineTDIWaveform));
    fd_spline_construct_kernel<<<1, 1>>>(d_wave_here, d_orbits, d_tdi_config, d_amp_spline, d_freq_spline);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    int buffer_length = tdi_on_fly->get_fd_spline_buffer_size(N);

    char *d_scratch = nullptr;
    size_t shared_bytes = spline_tdi_prepare_scratch("fd_spline",
        (const void *)fd_spline_run_wave_tdi_kernel, buffer_length, N, num_bin, &d_scratch);

    fd_spline_run_wave_tdi_kernel<<<num_bin, NUM_THREADS_HERE, shared_bytes>>>(
        d_wave_here, buffer_length, d_scratch, tdi_channels_arr, tdi_amp, tdi_phase, phi_ref,
        params, t_arr, N, num_bin, n_params, nchannels);

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    if (d_scratch != nullptr) gpuErrchk(cudaFree(d_scratch));

    gpuErrchk(cudaFree(d_orbits));
    gpuErrchk(cudaFree(d_tdi_config));
    gpuErrchk(cudaFree(d_amp_spline));
    gpuErrchk(cudaFree(d_freq_spline));
    gpuErrchk(cudaFree(d_wave_here));
#else

    // make buffer 
    int buffer_length = tdi_on_fly->get_fd_spline_buffer_size(N);
    char *buffer = new char[buffer_length];
    tdi_on_fly->run_wave_tdi((void*)buffer, buffer_length, tdi_channels_arr, tdi_amp, tdi_phase, phi_ref,
        params, t_arr, N, num_bin, n_params, nchannels);
    delete[] buffer;
#endif
}

// ===========================================================================
// TDDenseTDIWaveform: template-batched TD TDI-on-the-fly with exact (DOPR853
// dense-output) phases; see lat_spline_tdi_waveform.hh for the data layout.
// ===========================================================================
CUDA_DEVICE
int TDDenseTDIWaveform::segment(int b, double t, bool clamp)
{
    int nk = n_knots[b];
    double *tk = &t_knots[(size_t)b * K];
    if ((t < tk[0]) || (t > tk[nk - 1]))
    {
        if (!clamp) return -1;
        return (t < tk[0]) ? 0 : nk - 2;
    }
    int lo = 0;
    int hi = nk - 1;   // tk[lo] <= t; the segment is lo in [0, nk - 2]
    while (hi - lo > 1)
    {
        int mid = (lo + hi) / 2;
        if (tk[mid] <= t) lo = mid;
        else hi = mid;
    }
    return lo;
}

CUDA_DEVICE
void TDDenseTDIWaveform::phases(int b, int seg, double t, double *Phi3)
{
    // FEW DOPR853 dense output (few/trajectory/dopr853.py::eval):
    // r1 + s (r2 + s1 (r3 + s (r4 + s1 (r5 + s (r6 + s1 (r7 + s r8)))))),
    // s = (t - t_seg) / h_seg, s1 = 1 - s
    double *tk = &t_knots[(size_t)b * K];
    double s = (t - tk[seg]) / (tk[seg + 1] - tk[seg]);
    double s1 = 1.0 - s;
    for (int p = 0; p < 3; p += 1)
    {
        double *c = &phase_coeffs[(((size_t)b * (K - 1) + seg) * 3 + p) * 8];
        Phi3[p] = c[0] + s * (c[1] + s1 * (c[2] + s * (c[3] + s1 * (c[4] + s * (c[5] + s1 * (c[6] + s * c[7]))))));
    }
}

CUDA_DEVICE
cmplx TDDenseTDIWaveform::strain_term(int s, int b, int seg, double t, double *Phi3)
{
    double dx = t - t_knots[(size_t)b * K + seg];
    double *ar = &amp_re[((size_t)s * (K - 1) + seg) * 4];
    double *ai = &amp_im[((size_t)s * (K - 1) + seg) * 4];
    double cr = ar[0] + dx * (ar[1] + dx * (ar[2] + dx * ar[3]));
    double ci = ai[0] + dx * (ai[1] + dx * (ai[2] + dx * ai[3]));
    double ph = sub_mkn[3 * s] * Phi3[0] + sub_mkn[3 * s + 1] * Phi3[1] + sub_mkn[3 * s + 2] * Phi3[2];
    double cph = cos(ph);
    double sph = sin(ph);
    // amp_factor * c * exp(-i ph)
    return cmplx(amp_factor * (cr * cph + ci * sph), amp_factor * (ci * cph - cr * sph));
}

// A source that stops abruptly has no net T-like area (every delay term of a channel is the same
// truncated function), but SAMPLED, each term's cut falls at its own place between samples and the
// sum keeps an area that aliases into the lowest frequencies. A weight on the EMISSION time is the
// same for every term, so tapering there keeps the cancellation and makes each term smooth.
CUDA_DEVICE
double TDDenseTDIWaveform::stop_weight(int b, double s, double stop_taper)
{
    if (stop_taper <= 0.0) return 1.0;
    double x = (t_knots[(size_t)b * K + n_knots[b] - 1] - s) / stop_taper;
    if (x >= 1.0) return 1.0;
    if (x <= 0.0) return 0.0;
    double w = sin(0.5 * M_PI * x);
    return w * w;
}

CUDA_DEVICE
void TDDenseTDIWaveform::channels_point(int b, int i, double t, double *params_b, int sub_lo, int sub_hi,
    cmplx *tdi_channels_arr, double *phi_ref, int N, int *link_Space_craft_rec, int *link_Space_craft_em,
    int sum_subs, double stop_taper)
{
    Vec k(0.0, 0.0, 0.0);
    Vec u(0.0, 0.0, 0.0);
    Vec v(0.0, 0.0, 0.0);
    get_sky_vectors(&k, &u, &v, params_b);
    int nch = tdi_config->num_channels;

    // polarisation weights of the complex (analytic) strain: with
    // z = amp e^{-i phase}, the TDI unit term of get_tdi_Xf_single is
    // sign * pre * (xi_p A_p + xi_c A_c) (z_em - z_rec)  (see get_hp_hc)
    double inc = params_b[inc_index];
    double psi = params_b[psi_index];
    double cinc = cos(inc);
    double c2p = cos(2.0 * psi);
    double s2p = sin(2.0 * psi);
    cmplx I(0.0, 1.0);
    cmplx A_p = -(1.0 + cinc * cinc) * c2p + I * (2.0 * cinc * s2p);
    cmplx A_c = -(1.0 + cinc * cinc) * s2p - I * (2.0 * cinc * c2p);

    // carrier reference phase at spacecraft-1 time (get_phase_ref)
    {
        Vec x1 = orbits->get_pos(t, 1);
        double t_sc = t - k.dot(x1) * C_inv;
        double P3[3] = {0.0, 0.0, 0.0};
        int seg = segment(b, t_sc, false);
        if (seg >= 0)
        {
            phases(b, seg, t_sc, P3);
        }
        else
        {
            // outside the trajectory the phase is NOT evaluated: the reference phase is HELD at
            // its value at the nearest trajectory end (no jump for the output splines; the
            // channel's remaining phase -- its short TDI tail after the wave passes spacecraft 1
            // -- lands in tdi_phase)
            int nk = n_knots[b];
            double t_end = (t_sc > t_knots[(size_t)b * K + nk - 1]) ? t_knots[(size_t)b * K + nk - 1] : t_knots[(size_t)b * K];
            phases(b, segment(b, t_end, true), t_end, P3);
        }
        if (phi_ref != nullptr)
        {
            for (int s = sub_lo; s < sub_hi; s += 1)
            {
                phi_ref[(size_t)s * N + i] = sub_mkn[3 * s] * P3[0] + sub_mkn[3 * s + 1] * P3[1] + sub_mkn[3 * s + 2] * P3[2];
            }
        }
    }

    bool is_okay = true;
    for (int unit_i = 0; unit_i < tdi_config->num_units; unit_i += 1)
    {
        int unit_start = tdi_config->unit_starts[unit_i];
        int unit_length = tdi_config->unit_lengths[unit_i];
        int base_link = tdi_config->tdi_base_link[unit_i];
        int base_link_index = orbits->get_link_ind(base_link);
        int channel = tdi_config->channels[unit_i];
        double sign = tdi_config->tdi_signs_in[unit_i];

        double total_delay = 0.0;
        for (int sub_i = 0; sub_i < unit_length; sub_i += 1)
        {
            int combination_link = tdi_config->tdi_link_combinations[unit_start + sub_i];
            if (combination_link != -11)
            {
                total_delay += orbits->get_light_travel_time(t, combination_link);
            }
        }
        double time_rec = t - total_delay;
        if ((orbits->get_window(time_rec, orbits->ltt_t0, orbits->ltt_dt, orbits->ltt_N) == -1)
            || (orbits->get_window(time_rec, orbits->sc_t0, orbits->sc_dt, orbits->sc_N) == -1))
        {
            is_okay = false;
            break;
        }
        double L = orbits->get_light_travel_time(time_rec, base_link);
        double time_em = time_rec - L;
        int sc_r = link_Space_craft_rec[base_link_index];
        int sc_e = link_Space_craft_em[base_link_index];
        Vec x_rec = orbits->get_pos(time_rec, sc_r);
        Vec x_em = orbits->get_pos(time_em, sc_e);
        Vec n = x_rec - x_em;
        double norm = sqrt(n.dot(n));
        n = n / norm;
        double k_dot_n = k.dot(n);
        double denom = 1. - k_dot_n;
        if (fabs(denom) < 1.0e-12) continue;   // arm-singular line (see get_tdi_Xf_single)
        double pre_factor = 1. / denom;
        double delay_rec = time_rec - k.dot(x_rec) * C_inv;
        double delay_em = time_em - k.dot(x_em) * C_inv;
        double xi_p, xi_c;
        xi_projections(&xi_p, &xi_c, u, v, n);
        cmplx G = sign * pre_factor * (xi_p * A_p + xi_c * A_c);

        // geometry and the fundamental phases are shared by every harmonic
        int seg_r = segment(b, delay_rec, false);
        int seg_e = segment(b, delay_em, false);
        double P3r[3] = {0.0, 0.0, 0.0};
        double P3e[3] = {0.0, 0.0, 0.0};
        if (seg_r >= 0) phases(b, seg_r, delay_rec, P3r);
        if (seg_e >= 0) phases(b, seg_e, delay_em, P3e);
        double w_r = stop_weight(b, delay_rec, stop_taper);
        double w_e = stop_weight(b, delay_em, stop_taper);
        if (sum_subs)
        {
            cmplx acc(0.0, 0.0);
            for (int s = sub_lo; s < sub_hi; s += 1)
            {
                cmplx zr = (seg_r >= 0) ? strain_term(s, b, seg_r, delay_rec, P3r) : cmplx(0.0, 0.0);
                cmplx ze = (seg_e >= 0) ? strain_term(s, b, seg_e, delay_em, P3e) : cmplx(0.0, 0.0);
                acc += w_e * ze - w_r * zr;
            }
            tdi_channels_arr[((size_t)b * nch + channel) * N + i] += G * acc;
        }
        else
        {
            for (int s = sub_lo; s < sub_hi; s += 1)
            {
                cmplx zr = (seg_r >= 0) ? strain_term(s, b, seg_r, delay_rec, P3r) : cmplx(0.0, 0.0);
                cmplx ze = (seg_e >= 0) ? strain_term(s, b, seg_e, delay_em, P3e) : cmplx(0.0, 0.0);
                tdi_channels_arr[((size_t)s * nch + channel) * N + i] += G * (w_e * ze - w_r * zr);
            }
        }
    }
    if (!is_okay)
    {
        if (sum_subs)
        {
            for (int ch = 0; ch < nch; ch += 1)
            {
                tdi_channels_arr[((size_t)b * nch + ch) * N + i] = cmplx(0.0, 0.0);
            }
        }
        else
        {
            for (int s = sub_lo; s < sub_hi; s += 1)
            {
                for (int ch = 0; ch < nch; ch += 1)
                {
                    tdi_channels_arr[((size_t)s * nch + ch) * N + i] = cmplx(0.0, 0.0);
                }
            }
        }
    }
}

CUDA_DEVICE
void TDDenseTDIWaveform::postprocess_sub(void *buffer, cmplx *chan, double *amp, double *phase, double *phi_ref, int N)
{
    // same amplitude/phase extraction + unwrap as LISATDIonTheFly::get_tdi
    double *flip = (double*)buffer;
    double *pjump = &flip[N];
    int *count = (int *)&pjump[N];
    bool *fix_count = (bool *)&count[N];
    int nch = tdi_config->num_channels;
    for (int ch = 0; ch < nch; ch += 1)
    {
        new_extract_amplitude_and_phase(count, fix_count, flip, pjump, N, &amp[ch * N], &phase[ch * N], &chan[ch * N], phi_ref);
    }
    double *ph_correct_buffer = &flip[0];
    for (int ch = 0; ch < nch; ch += 1)
    {
        new_unwrap_phase(ph_correct_buffer, N, &phase[ch * N]);
    }
}

#ifdef __CUDACC__
CUDA_KERNEL
void td_dense_channels_kernel(TDDenseTDIWaveform *w, cmplx *tdi_channels_arr, double *phi_ref,
    double *params, double *t_arr, int *sub_offsets, int N, int n_params, int sum_subs, double stop_taper)
{
    CUDA_SHARED int link_rec[NLINKS];
    CUDA_SHARED int link_em[NLINKS];
    w->fill_link_arrays(link_rec, link_em);
    CUDA_SYNC_THREADS;
    for (int b = blockIdx.y; b < w->n_temp; b += gridDim.y)
    {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
        {
            w->channels_point(b, i, t_arr[(size_t)b * N + i], &params[(size_t)b * n_params],
                sub_offsets[b], sub_offsets[b + 1], tdi_channels_arr, phi_ref, N, link_rec, link_em, sum_subs,
                stop_taper);
        }
    }
}

CUDA_KERNEL
void td_dense_post_kernel(TDDenseTDIWaveform *w, int buffer_length, char *global_buffer, cmplx *tdi_channels_arr,
    double *tdi_amp, double *tdi_phase, double *phi_ref, int N, int num_sub, int nchannels)
{
    extern CUDA_SHARED char shared_mem[];
    void *buffer = (global_buffer != nullptr)
        ? (void*)(global_buffer + (size_t)blockIdx.x * spline_tdi_scratch_stride(buffer_length))
        : (void*)shared_mem;
    for (int s = blockIdx.x; s < num_sub; s += gridDim.x)
    {
        CUDA_SYNC_THREADS;
        w->postprocess_sub(buffer, &tdi_channels_arr[(size_t)s * nchannels * N], &tdi_amp[(size_t)s * nchannels * N],
            &tdi_phase[(size_t)s * nchannels * N], &phi_ref[(size_t)s * N], N);
        CUDA_SYNC_THREADS;
    }
}

// device-side construction (device vtable; see td_spline_construct_kernel)
CUDA_KERNEL
void td_dense_construct_kernel(TDDenseTDIWaveform *obj, Orbits *orbits, TDIConfig *tdi_config, int n_temp, int K,
    int num_sub, double amp_factor, int *sub_temp, int *sub_mkn, int *n_knots, double *t_knots,
    double *phase_coeffs, double *amp_re, double *amp_im)
{
    new (obj) TDDenseTDIWaveform(orbits, tdi_config, n_temp, K, num_sub, amp_factor, sub_temp, sub_mkn,
        n_knots, t_knots, phase_coeffs, amp_re, amp_im);
}
#endif

void td_dense_run_wave_tdi_wrap(TDDenseTDIWaveform *w, cmplx *tdi_channels_arr,
    double *tdi_amp, double *tdi_phase, double *phi_ref,
    double *params, double *t_arr, int *sub_offsets, int N, int n_params, int nchannels)
{
#ifdef __CUDACC__
    Orbits *d_orbits;
    gpuErrchk(cudaMalloc(&d_orbits, sizeof(Orbits)));
    gpuErrchk(cudaMemcpy(d_orbits, w->orbits, sizeof(Orbits), cudaMemcpyHostToDevice));
    TDIConfig *d_tdi_config;
    gpuErrchk(cudaMalloc(&d_tdi_config, sizeof(TDIConfig)));
    gpuErrchk(cudaMemcpy(d_tdi_config, w->tdi_config, sizeof(TDIConfig), cudaMemcpyHostToDevice));
    TDDenseTDIWaveform *d_wave;
    gpuErrchk(cudaMalloc(&d_wave, sizeof(TDDenseTDIWaveform)));
    td_dense_construct_kernel<<<1, 1>>>(d_wave, d_orbits, d_tdi_config, w->n_temp, w->K, w->num_sub, w->amp_factor,
        w->sub_temp, w->sub_mkn, w->n_knots, w->t_knots, w->phase_coeffs, w->amp_re, w->amp_im);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    int n_temp_grid = (w->n_temp < 65535) ? w->n_temp : 65535;
    dim3 grid1((N + 127) / 128, n_temp_grid);
    td_dense_channels_kernel<<<grid1, 128>>>(d_wave, tdi_channels_arr, phi_ref, params, t_arr, sub_offsets, N, n_params, 0,
        0.0);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    int buffer_length = w->get_td_dense_buffer_size(N);
    char *d_scratch = nullptr;
    size_t shared_bytes = spline_tdi_prepare_scratch("td_dense",
        (const void *)td_dense_post_kernel, buffer_length, N, w->num_sub, &d_scratch);
    td_dense_post_kernel<<<w->num_sub, NUM_THREADS_HERE, shared_bytes>>>(d_wave, buffer_length, d_scratch,
        tdi_channels_arr, tdi_amp, tdi_phase, phi_ref, N, w->num_sub, nchannels);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    if (d_scratch != nullptr) gpuErrchk(cudaFree(d_scratch));
    gpuErrchk(cudaFree(d_orbits));
    gpuErrchk(cudaFree(d_tdi_config));
    gpuErrchk(cudaFree(d_wave));
#else
    int link_rec[NLINKS];
    int link_em[NLINKS];
    w->fill_link_arrays(link_rec, link_em);
    for (int b = 0; b < w->n_temp; b += 1)
    {
        for (int i = 0; i < N; i += 1)
        {
            w->channels_point(b, i, t_arr[(size_t)b * N + i], &params[(size_t)b * n_params],
                sub_offsets[b], sub_offsets[b + 1], tdi_channels_arr, phi_ref, N, link_rec, link_em, 0, 0.0);
        }
    }
    int buffer_length = w->get_td_dense_buffer_size(N);
    char *buffer = new char[buffer_length];
    for (int s = 0; s < w->num_sub; s += 1)
    {
        w->postprocess_sub((void*)buffer, &tdi_channels_arr[(size_t)s * nchannels * N], &tdi_amp[(size_t)s * nchannels * N],
            &tdi_phase[(size_t)s * nchannels * N], &phi_ref[(size_t)s * N], N);
    }
    delete[] buffer;
#endif
}

void td_dense_run_channels_wrap(TDDenseTDIWaveform *w, cmplx *tdi_channels_arr,
    double *params, double *t_arr, int *sub_offsets, int N, int n_params, int sum_subs, double stop_taper)
{
#ifdef __CUDACC__
    Orbits *d_orbits;
    gpuErrchk(cudaMalloc(&d_orbits, sizeof(Orbits)));
    gpuErrchk(cudaMemcpy(d_orbits, w->orbits, sizeof(Orbits), cudaMemcpyHostToDevice));
    TDIConfig *d_tdi_config;
    gpuErrchk(cudaMalloc(&d_tdi_config, sizeof(TDIConfig)));
    gpuErrchk(cudaMemcpy(d_tdi_config, w->tdi_config, sizeof(TDIConfig), cudaMemcpyHostToDevice));
    TDDenseTDIWaveform *d_wave;
    gpuErrchk(cudaMalloc(&d_wave, sizeof(TDDenseTDIWaveform)));
    td_dense_construct_kernel<<<1, 1>>>(d_wave, d_orbits, d_tdi_config, w->n_temp, w->K, w->num_sub, w->amp_factor,
        w->sub_temp, w->sub_mkn, w->n_knots, w->t_knots, w->phase_coeffs, w->amp_re, w->amp_im);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    int n_temp_grid = (w->n_temp < 65535) ? w->n_temp : 65535;
    dim3 grid1((N + 127) / 128, n_temp_grid);
    td_dense_channels_kernel<<<grid1, 128>>>(d_wave, tdi_channels_arr, nullptr, params, t_arr, sub_offsets, N,
        n_params, sum_subs, stop_taper);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaFree(d_orbits));
    gpuErrchk(cudaFree(d_tdi_config));
    gpuErrchk(cudaFree(d_wave));
#else
    int link_rec[NLINKS];
    int link_em[NLINKS];
    w->fill_link_arrays(link_rec, link_em);
    for (int b = 0; b < w->n_temp; b += 1)
    {
        for (int i = 0; i < N; i += 1)
        {
            w->channels_point(b, i, t_arr[(size_t)b * N + i], &params[(size_t)b * n_params],
                sub_offsets[b], sub_offsets[b + 1], tdi_channels_arr, nullptr, N, link_rec, link_em, sum_subs,
                stop_taper);
        }
    }
#endif
}

// ===========================================================================
// wdm_lookup_sum: fused sparse-response -> WDM lookup (see lat_spline_tdi_waveform.hh)
// ===========================================================================
static CUDA_DEVICE inline void wdm_lookup_atomic_add(double *p, double v)
{
#ifdef __CUDA_ARCH__   // device pass only (atomicAdd is __device__)
    atomicAdd(p, v);
#else
    *p += v;
#endif
}

static CUDA_DEVICE void wdm_lookup_sum_point(const WDMLookupSumArgs &A, int s, int n,
    unsigned long long *n_look, unsigned long long *n_drop)
{
    if (n >= A.n_stop[s]) return;
    const double two_pi = 6.283185307179586476925286766559;
    double t = A.t0 + (double)n * A.layer_dt;

    // exact carrier (held -- derivatives zero -- outside the trajectory)
    double car = 0.0;
    double car1 = 0.0;
    double car2 = 0.0;
    if (A.carrier)
    {
        int b = A.sub_temp[s];
        const double *tk = &A.t_knots[(size_t)b * A.K];
        int nk = A.n_knots[b];
        bool held = (t < tk[0]) || (t > tk[nk - 1]);
        double tc = (t < tk[0]) ? tk[0] : ((t > tk[nk - 1]) ? tk[nk - 1] : t);
        int seg = wdm_spline_segment(tk, nk, tc);
        double h = tk[seg + 1] - tk[seg];
        double sv = (tc - tk[seg]) / h;
        const double *cb = &A.phase_coeffs[((size_t)b * (A.K - 1) + seg) * 24];
        const int *mk = &A.sub_mkn[3 * s];
        for (int p = 0; p < 3; p += 1)
        {
            double v, d1, d2;
            wdm_dense_phase_derivs(&cb[8 * p], sv, &v, &d1, &d2);
            car += mk[p] * v;
            if (!held)
            {
                car1 += mk[p] * (d1 / h);
                car2 += mk[p] * (d2 / (h * h));
            }
        }
    }

    const double *xs = &A.x[(size_t)s * A.nch * A.N];
    int seg = wdm_spline_segment(xs, A.N, t);
    double dx = t - xs[seg];
    int n_m = A.m_hi - A.m_lo;
    int row = A.sub_row[s];
    for (int ch = 0; ch < A.nch; ch += 1)
    {
        size_t off = ((size_t)s * A.nch + ch) * A.N + seg;
        double amp = A.amp_y[off] + dx * (A.amp_c1[off] + dx * (A.amp_c2[off] + dx * A.amp_c3[off]));
        double r, r1, r2;
        wdm_spline_derivs(&A.res_y[off - seg], &A.res_c1[off - seg], &A.res_c2[off - seg], &A.res_c3[off - seg],
            seg, dx, &r, &r1, &r2);
        double ph = r + car;
        double f = (r1 + car1) / two_pi;
        double fdot = (r2 + car2) / two_pi;
        if (f < 0.0)            // a -m partner: cos is even
        {
            ph = -ph;
            f = -f;
            fdot = -fdot;
        }
        if (!(fabs(fdot) <= A.fdot_max) || !(f > A.f_min) || !isfinite(f))
        {
            *n_drop += 1;
            continue;
        }
        *n_look += 1;
        int td[4];
        double wd[4];
        if (!wdm_table_fdot_axis(A.tab, fdot, td, wd)) continue;   // off the fdot axis: every layer 0
        double cph = cos(ph);
        double sph = sin(ph);
        int ms = (int)(f / A.layer_df);                  // truncation, as numpy .astype(int)
        double *out_rc = &A.out[((size_t)row * A.nch + ch) * n_m * (size_t)A.Nt];
        for (int dm = -A.num_m_layers; dm <= A.num_m_layers; dm += 1)
        {
            int m = ms + dm;
            if ((m < 0) || (m >= A.Nf) || (m < A.m_lo) || (m >= A.m_hi)) continue;
            double f_norm = f - (double)m * A.layer_df;
            if ((f_norm < A.tab.f_lo) || (f_norm > A.tab.f_hi)) continue;   // out_of_support="zero"
            double c, sn;
            if (!wdm_table_cs_at(A.tab, td, wd, f_norm, &c, &sn)) continue;
            double val = wdm_quarter_turn_value(c, sn, A.tab.ref_odd, (m + n) & 1, amp, cph, sph);
            wdm_lookup_atomic_add(&out_rc[(size_t)(m - A.m_lo) * A.Nt + n], val);
        }
    }
}

#ifdef __CUDACC__
CUDA_KERNEL
void wdm_lookup_sum_kernel(WDMLookupSumArgs A)
{
    CUDA_SHARED unsigned long long blk[2];
    if (threadIdx.x == 0)
    {
        blk[0] = 0;
        blk[1] = 0;
    }
    CUDA_SYNC_THREADS;
    unsigned long long n_look = 0;
    unsigned long long n_drop = 0;
    long long P = (long long)(A.n_hi - A.n_lo);
    long long total = (long long)A.num_sub * P;
    // adjacent threads = adjacent pixels of one harmonic: the same spline segment and
    // neighbouring table cells
    for (long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x; j < total;
         j += (long long)gridDim.x * blockDim.x)
    {
        int s = (int)(j / P);
        int n = A.n_lo + (int)(j - (long long)s * P);
        wdm_lookup_sum_point(A, s, n, &n_look, &n_drop);
    }
    if (A.counts != nullptr)
    {
        if (n_look) atomicAdd(&blk[0], n_look);
        if (n_drop) atomicAdd(&blk[1], n_drop);
        CUDA_SYNC_THREADS;
        if (threadIdx.x == 0)
        {
            atomicAdd(&A.counts[0], blk[0]);
            atomicAdd(&A.counts[1], blk[1]);
        }
    }
}
#endif

void wdm_lookup_sum_wrap(WDMLookupSumArgs args)
{
    long long P = (long long)(args.n_hi - args.n_lo);
    if ((P <= 0) || (args.num_sub <= 0)) return;
#ifdef __CUDACC__
    long long total = (long long)args.num_sub * P;
    int threads = 256;
    long long blocks = (total + threads - 1) / threads;
    if (blocks > (1LL << 20)) blocks = 1LL << 20;
    wdm_lookup_sum_kernel<<<(int)blocks, threads>>>(args);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
#else
    unsigned long long n_look = 0;
    unsigned long long n_drop = 0;
    for (int s = 0; s < args.num_sub; s += 1)
    {
        for (int n = args.n_lo; n < args.n_hi; n += 1)
        {
            wdm_lookup_sum_point(args, s, n, &n_look, &n_drop);
        }
    }
    if (args.counts != nullptr)
    {
        args.counts[0] += n_look;
        args.counts[1] += n_drop;
    }
#endif
}
