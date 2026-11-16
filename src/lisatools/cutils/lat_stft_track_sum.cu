// lat_stft_track_sum.cu: see lat_stft_track_sum.hh. One source for the CPU and GPU backends.

#include <cmath>
#include "lat_stft_track_sum.hh"

static CUDA_DEVICE inline void stft_track_atomic_add(double *target, double value) {
#ifdef __CUDA_ARCH__  // device pass only (atomicAdd is __device__)
  atomicAdd(target, value);
#else
  *target += value;
#endif
}

// One (component, channel, segment): normalise the track, then add its band to the output.
static CUDA_DEVICE void stft_track_sum_point(STFTFresnel &fresnel, const STFTTrackSumArgs &args,
                                             int component, int channel, int segment) {
  int half_width = args.half_width[(size_t)component * args.num_segments + segment];
  if (half_width < 0) return;

  size_t index = ((size_t)component * args.num_channels + channel) * args.num_segments + segment;
  double amp = args.amp[index];
  double phase = args.phase[index];
  double f = args.f[index];
  double fdot = args.fdot[index];
  double slope = (args.slope != nullptr) ? args.slope[index] : 0.0;
  if (!(amp != 0.0) || !std::isfinite(amp) || !std::isfinite(phase) || !std::isfinite(f) ||
      !std::isfinite(fdot))
    return;

  if (amp < 0.0) {
    amp = -amp;
    phase += M_PI;
  }
  // Frequency at mid-segment, the centre of the band: the anchor itself, or half a window later.
  double f_centre = fresnel.use_midpoint ? f : f + 0.5 * fdot * fresnel.dt;
  if (f_centre < 0.0) {  // amp cos(phase) = amp cos(-phase)
    f = -f;
    fdot = -fdot;
    phase = -phase;
    f_centre = -f_centre;
  }
  if (std::fabs(fdot) < STFT_TRACK_FDOT_FLOOR)
    fdot = (fdot < 0.0) ? -STFT_TRACK_FDOT_FLOOR : STFT_TRACK_FDOT_FLOOR;

  double centre = (f_centre - args.f_min) / fresnel.df;
  // ! Compared as doubles first: a centre far outside the output would overflow the int cast.
  if ((centre + half_width < 0.0) || (centre - half_width > (double)(args.num_freqs - 1))) return;
  int bin_centre = (int)std::floor(centre + 0.5);
  int bin_lo = bin_centre - half_width;
  int bin_hi = bin_centre + half_width + 1;
  if (bin_lo < 0) bin_lo = 0;
  if (bin_hi > args.num_freqs) bin_hi = args.num_freqs;

  int row = args.component_row[component];
  double *out = reinterpret_cast<double *>(args.out) +
                2 * ((((size_t)row * args.num_channels + channel) * args.num_segments + segment) *
                     args.num_freqs);
  for (int bin = bin_lo; bin < bin_hi; bin += 1) {
    // * Segment start at time zero: only times relative to the Fourier origin enter the value.
    cmplx value = fresnel.get_fourier_value(amp, phase, f, fdot, 0.0,
                                            args.f_min + bin * fresnel.df, 1.0, slope);
    stft_track_atomic_add(&out[2 * bin], args.scale * value.real());
    stft_track_atomic_add(&out[2 * bin + 1], args.scale * value.imag());
  }
}

#ifdef __CUDACC__
CUDA_KERNEL
void stft_track_sum_kernel(STFTFresnel fresnel, STFTTrackSumArgs args) {
  long long per_component = (long long)args.num_channels * args.num_segments;
  long long total = (long long)args.num_components * per_component;
  // Adjacent threads are adjacent segments of one (component, channel).
  for (long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x; j < total;
       j += (long long)gridDim.x * blockDim.x) {
    int component = (int)(j / per_component);
    long long rest = j - (long long)component * per_component;
    int channel = (int)(rest / args.num_segments);
    int segment = (int)(rest - (long long)channel * args.num_segments);
    stft_track_sum_point(fresnel, args, component, channel, segment);
  }
}
#endif

void stft_track_sum_wrap(STFTFresnel &fresnel, STFTTrackSumArgs args) {
  if ((args.num_components <= 0) || (args.num_segments <= 0) || (args.num_channels <= 0)) return;
#ifdef __CUDACC__
  long long total = (long long)args.num_components * args.num_channels * args.num_segments;
  int threads = 128;
  long long blocks = (total + threads - 1) / threads;
  if (blocks > (1LL << 20)) blocks = 1LL << 20;
  stft_track_sum_kernel<<<(int)blocks, threads>>>(fresnel, args);
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
#else
  for (int component = 0; component < args.num_components; component += 1)
    for (int channel = 0; channel < args.num_channels; channel += 1)
      for (int segment = 0; segment < args.num_segments; segment += 1)
        stft_track_sum_point(fresnel, args, component, channel, segment);
#endif
}
