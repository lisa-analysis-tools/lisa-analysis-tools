#ifndef __LAT_STFT_TRACK_SUM_HH__
#define __LAT_STFT_TRACK_SUM_HH__

// lat_stft_track_sum.{hh,cu}: STFT templates of signals given as sums of slowly chirping tracks.
//
// A template row is a sum of components. Per TDI channel, a component is the real signal
//     x(t) = amp(t) cos(phase(t)),
// and within one STFT segment it is modelled as a linear chirp about the segment anchor: the
// caller supplies (amp, phase, f, fdot) at the anchor, with f = phase' / 2 pi and
// fdot = phase'' / 2 pi, and optionally the fractional amplitude slope amp' / amp. The kernel
// evaluates the windowed transform of that chirp with STFTFresnel (exact for a linear chirp of
// any sweep width) and ADDS it into a dense template, so the dense inner product of the result
// carries every cross term between the components of a row.
//
// Nothing here knows what a component is: a harmonic of an EMRI, a mode of a massive black hole
// binary, or a whole quasi-monochromatic source all fill the same arrays. Whatever the linear
// chirp cannot represent in a segment (fast chirps, a track that ends inside the segment) is
// masked out with half_width < 0 and filled by the caller in another way.
//
// * Conventions, those of STFTFresnel and the STFT likelihood: the value of a pixel is
// * dt * rfft(w * x) at f = f_min + k df, the positive-frequency half, Fourier origin at the
// * segment start, w the Tukey window of the STFTFresnel: symmetric over N samples when it was
// * built with window_samples = N and sample_cadence = h, periodic otherwise. The anchor is the segment
// * midpoint when the STFTFresnel has use_midpoint, the segment start otherwise. Segment start
// * times do not enter: the caller evaluates its tracks at the anchors of whatever segment grid
// * it uses, so the hop between segments need not equal the window length.
// * Inputs are normalised per (component, channel, segment): a negative amplitude becomes
// * (|amp|, phase + pi); a negative frequency is mirrored (f, fdot, phase -> -f, -fdot, -phase);
// * |fdot| is floored at STFT_TRACK_FDOT_FLOOR with its sign kept.
// * Band of a component in a segment: the bins within half_width of the bin of the mid-segment
// * frequency, clipped to the output. The caller sets half_width, typically the side bins it
// * wants beyond half the sweep |fdot| dt / (2 df).
// ! The output is accumulated, never zeroed here.
// ! df of the STFTFresnel is the bin spacing of the output; dt is the window length.

#include "domains.hpp"

#define STFT_TRACK_FDOT_FLOOR 1.0e-22  // [Hz/s], STFT_FDOT_FLOOR of the existing response

struct STFTTrackSumArgs {
  cmplx *out;               // (num_rows, num_channels, num_segments, num_freqs), accumulated
  int num_rows;
  int num_channels;
  int num_segments;
  int num_freqs;
  double f_min;             // frequency of output bin 0 [Hz]
  double scale;             // multiplies every value; 0.5 gives the transform of amp cos(phase)
  int num_components;
  const int *component_row; // (num_components,) output row of each component
  const int *half_width;    // (num_components, num_segments) bins each side; < 0 skips
  const double *amp;        // (num_components, num_channels, num_segments), at the anchor
  const double *phase;      // same shape [rad]
  const double *f;          // same shape [Hz]
  const double *fdot;       // same shape [Hz/s]
  const double *slope;      // same shape, amp' / amp [1/s]; nullptr: constant envelope
};

void stft_track_sum_wrap(STFTFresnel &fresnel, STFTTrackSumArgs args);

#endif  // __LAT_STFT_TRACK_SUM_HH__
