# MBH batched, windowed WDM likelihood for the global fit

Date: 2026-09-29. Branch: `cd1l-merge` (PR #82 / #81 merged onto dev).
Status: design approved in conversation (Mike, 2026-09-29) with three rulings
folded in below; this document is the written spec that the implementation
plan is derived from.

## 1. Why

The 6mo CD1-L global fit scores every MBH proposal row one at a time through
the legacy `PhenomTHMTDIWaveform` path (phentax strain, `pyResponseTDI` at
order 30, a full 6-month TD-to-WDM transform, `inner_product`). Measured on
jobs 373/487/558: 1.16 to 1.43 s per row, 344 to 376 s per leaf visit (the
243-row information-matrix sweep is 80% of it), GPU utilisation 0.3 to 1.1%.
MBH plus EMRI are ~98% of the source-move wall; SOBBH, which already scores
through one vectorized call per batch, is 1.4%.

PR #81 (Aaron Johnson) made the legacy response batchable: rows evaluated on
one shared lattice go through one `pyResponseTDI` launch, `inner_product`
carries a leading source axis, and `FDSignal.wdmtransform` walks frequency
layers in bounded blocks. PR #82 (Maria Rosell) used it for single-container
CD1-L PE and measured 0.94 -> 0.117 s/row from overhead fixes and 2.09x more
from batching at B >= 16 on a 23-day window.

The global fit is not the single-container case. Each walker has its own
residual and its own PSD sample, rows arrive as `(coords, data_index)`
batches, and the move must expose/fold cold-chain sources per walker. So the
target is a batched MBH add/remove move shaped like `SOBBHChunkedLikeMove`,
scoring every row against its own container.

## 2. Rulings (user, 2026-09-29)

1. Go, carefully. Default OFF; the running campaign is untouched until the
   knob is flipped on a relaunch.
2. Do NOT generate 6-month MBH waveforms. Each MBH template is generated on
   a window from 90 days before the merger to 10 days after it, transformed
   on its own sub-grid ("sub-transform") with edge effects cut, and placed
   at the merger's position in the WDM grid. This holds for longer datasets
   too (1 yr, 2 yr): MBHs stay 90-day objects.
3. `MBH_RESPONSE_ORDER` defaults to 8 (PR evidence: mismatch flat from order
   30 down to 4; 1.59x cheaper).
4. `MBH_BATCH_MAX_SIZE` defaults to 16; the probe confirms the memory.

## 3. Requirements

Must:

- R1. Score a batch of rows `(coords_in, data_index)` where each row's
  likelihood uses container `data_index[i]`'s residual AND its PSD.
- R2. Generate all rows of a chunk in ONE `compute_tdi_channels` launch on a
  shared lattice (that is what buys the speed).
- R3. Templates live on a per-leaf window `[t_ref - W_before, t_ref +
  W_after]` (defaults 90 d / 10 d) snapped to WDM layer boundaries, and are
  transformed on a segment WDM grid of the same `Nf`, `dt` with `Nt_seg`
  layers, NOT on the full data grid. The kept coefficients are those of the
  full-grid transform to the accuracy pinned by a test (the edge layers are
  discarded).
- R4. Likelihood convention identical to the container path:
  `-1/2 <r-h|r-h>` plus the per-walker noise term, reproduced as
  `offset[walker] + (<r|h> - 1/2 <h|h>)` with `offset = acs.likelihood()` on
  the exposed residual, exactly as the SOBBH move does.
- R5. Expose/fold (`_apply_cold_chain_sources`) through the same batched
  generation, applying each row's sub-box template to its own container.
- R6. The information-matrix rows (243 per leaf) batch with no change to
  `eigen_refresh` (they already arrive through `compute_like` as one block).
- R7. Multi-GPU walker shards: rows partitioned per shard and run under the
  owning device context, reusing `_RoutedBandEngine._shard_views` and
  `_partition`; one-walker replica mode reaches the scorer through the
  existing `row_fanout` seam untouched.
- R8. Built-in fast-vs-slow check (`_verify_prev_logl`) against the stock
  generator at matching convention with an `MBH_CHECK_LL_TOL` knob.
- R9. A refusal (`BatchNotLaunchable`) or a domain error never kills the
  sampler: the chunk falls back to the base per-row container path, loudly,
  and domain errors keep the `-1e300` sentinel.
- R10. Every knob follows rule 0 (env name = capitalised field name), all
  defaults leave the current behaviour in place (`MBH_LIKELIHOOD=full`),
  and no new settings files.
- R11. Settings objects introduced or extended survive
  `pickle.loads(pickle.dumps(deepcopy(obj)))`.

Must not:

- Change the stock (`full`) MBH path's templates or defaults except the
  response order ruling (which applies to both paths).
- Batch the in-model repeat loop across sources (the sequential MH chain
  policy stands).
- Introduce a second likelihood implementation: inner products stay
  `lisatools.diagnostic.inner_product` on sliced `DomainBase` objects.

## 4. Architecture

### 4.1 Windowed grid-aligned generator

`WindowedGridAlignedMBHWaveform(GridAlignedPhenomTHMTDIWaveform)` in
`src/lisatools/sources/bbh/gridaligned.py`.

- Same constructor as the stock class. phentax generation window `T` (its
  `Tobs`) is `W_before`, so `time_bounded_start` places `t_min = -90 d`.
- `set_window(k0, n_grid)` stores the shared lattice as integer sample
  offsets from `waveform_t0`; `_common_grid_spec` returns them instead of
  the analysis-window lattice. `_aligned_polarizations` is unchanged: the
  batch shares every edge, `_apply_response` gets `merger_time = 0`, and a
  row's columns depend on its own parameters only.
- The lattice start `k0 * dt` is `n_lead` samples before the segment's
  first layer so the response's `_lead` crop lands where the serial path
  crops. The lattice end is the segment end; `_apply_response` right-pads
  `buffer_time` of zeros and `place_td_signal_on_grid` clips them.
- `waveform_t0` must sit on the data lattice (`_check_alignable`). The
  wiring snaps it (`waveform_t0 += snap`) and the move subtracts the same
  `snap` from every row's `t_plunge` column before generation, so absolute
  merger times are unchanged. Logged once at build.

### 4.2 Sub-transform adapter

`MBHWindowedWDMSignalGen` in `src/lisatools/sources/batching.py`, a
`BatchedDomainSignalGen` whose per-row domain step is the sub-transform:

- Geometry per leaf window: `n_start` (first kept layer), `Nt_keep`, and
  `n_pad` layers on each side. Segment: `Nt_seg = Nt_keep + 2 n_pad` (even),
  `t_seg = wdm.t0 + (n_start - n_pad) * layer_dt`, `N_seg = Nf * Nt_seg`
  samples on the data lattice (layer boundaries are multiples of `Nf * dt`
  from `t0`, so they are on the lattice by construction).
- Per row: `place_td_signal_on_grid(channels, TDSettings(N_seg, dt, t_seg),
  times=times)` -> `TDSignal.transform(WDMSettings(Nf, Nt_seg, dt,
  t0=t_seg, min_freq/max_freq as the data))` -> drop `n_pad` layers each
  side -> `WDMSignal(arr_kept, global_settings_with(min_time, max_time))`,
  i.e. a template whose `active_slice_t` is `[n_start, n_start + Nt_keep)`
  on the run's `WDMSettings`. Rows stack with a leading source axis
  (`_stack`), all on the same box.
- Why the edges are clean: the template is zero for `n_pad` layers before
  its onset (the onset ramp sits inside the kept box) and after the
  ringdown plus `buffer_time` (10 days of margin), so the periodic segment
  transform's wrap-around touches only discarded layers. The pad is set in
  TIME (`window_pad_days`, default 4 days) and converted to layers with
  the run's `layer_dt`, because the stock grids use half-day layers (the
  PR's 0.5 to 0.75 day wavelet duration) where 32 layers would be 16
  days. Measured on a toy grid (2026-09-29 spike, chirp exactly zero
  outside the kept box): kept-layer max relative error 1.2e-5 at 8 pad
  layers and 9e-7 at 16, falling ~10x per doubling, so 4 days at
  half-day layers (8 layers) is already at the 1e-5 level and the test
  pins it.
- The segment `WDMSettings` are cached per (device, geometry) so the window
  build is paid once per leaf window, not per row.

### 4.3 Container support for a WDM sub-box template

Three small, general additions, all covered by tests:

- `WDMSettings.get_slice((f_slice, t_slice))` returns settings with the same
  grid and narrowed `ind_min/max_{f,t}`. Today `get_slice` is implemented
  only for TD and STFT.
- `AnalysisContainer._slice_wdm_to_template(template)` mirrors
  `_slice_stft_to_template`: requires `eq_without_inds` and the template's
  box inside the data's box, returns `(data.get_array_slice(box),
  template, sens_mat.get_slice(box))`. Wired into `_slice_to_template`, so
  `template_likelihood`, `template_inner_product` and `template_snr` all
  accept a sub-box WDM template.
- `_apply_wdm_add` accepts a template whose active box is a sub-box of the
  target's (offset by the target's own `ind_min_{f,t}`), so
  `signal_operation` / `add_signal` fill a sub-box template into the full
  residual. Same-box behaviour is unchanged.

### 4.4 The move

`MBHBatchedLikeMove(ResidualAddOneRemoveOneMove)` in
`src/lisatools/globalfit/moves/mbhbatchedmove.py`, exported from
`globalfit.moves`.

Constructor: base positionals plus `batched_gen` (a `DeviceLocalWaveGen`
resolving to the windowed sub-transform adapter per device),
`batch_max_size=16`, `window_before`, `window_after`, `window_pad`,
`window_margin` (seconds), `t_plunge_snap`. No `dcga` (raises like the
SOBBH move). Generator kwargs: the adapter accepts only the phentax
reference keywords (`start_freq`, `ref_freq`, `T`); any other key in the
branch's `waveform_kwargs` (the legacy BBHx `modes`/`length` defaults, for
instance) is dropped and listed once at construction.

Per-leaf window (`setup_likelihood_here`):

- `_exposed_offset = acs.likelihood()` (per walker), as SOBBH.
- `t_ref` = median cold-chain `t_plunge` of the leaf (absolute, via
  `waveform_t0`). Window `[t_ref - W_before - margin, t_ref + W_after +
  margin]` with `margin = 1 day`, snapped outward to layer boundaries.
  Computed on the first visit of a leaf and kept for the process lifetime
  unless `t_ref` leaves the central `[-margin, +margin]` band (hysteresis),
  in which case it is rebuilt and logged. With the `t_plunge_pad = 3600 s`
  prior no walker can reach the window edge; a row whose merger does fall
  outside the kept box is still evaluated on the shared lattice and counted
  in telemetry.
- `batched_gen.set_window(...)` and the adapter geometry are refreshed
  together, per device.

`compute_like_local(coords_in, data_index)`:

1. Host-normalise rows; apply `t_plunge_snap`; mark non-finite rows invalid
   (`-1e300`).
2. Single shard: chunk valid rows by `batch_max_size`; per chunk one
   adapter call -> stacked sub-box templates; per row `i`: slice container
   `data_index[i]` to the box (4.3) and compute `d_h = <r|h>` and
   `h_h = <h|h>` with `inner_product` and the move's like kwargs;
   `ll = offset[idx] + d_h - 0.5 h_h`. Record `_last_d_h/_last_h_h`.
3. Multi-shard: partition rows per shard, run each split under its device
   context with the device-local generator, then step 2.
4. `BatchNotLaunchable` from a chunk: `n_batch_fallbacks += 1`, warn once
   per leaf, score that chunk through `compute_acs_like` (base path).
5. Telemetry `[MBH_BATCH]` per leaf: rows, chunks, s/row, generation vs
   scoring split, fallbacks, rows outside the kept box.

`_apply_cold_chain_sources(coords, sign)`: same generation in chunks, then
`acs.signal_operation(sign, templates, data_index=rows)`; rows outside the
domain are skipped deterministically. `MBH_BATCHED_FILL=0` restores the
base dense path (logged, as SOBBH).

`_record_leaf_inner_products`: base implementation (it reads
`_last_d_h/_last_h_h`).

`_verify_prev_logl`: the SOBBH implementation at matching convention with
`MBH_CHECK_LL_TOL` (default 0.5 nats, tighten after the probe). Cadence and
severity use the base `MBH_CHECK_LL` / `MBH_CHECK_LL_EVERY` knobs.

### 4.5 Wiring (stock erebor)

`SourceMBHSettings` gains, all env-backed by rule 0:

| field | env | default |
|---|---|---|
| `likelihood` | `MBH_LIKELIHOOD` | `"full"` (`"batched"` opts in) |
| `batch_max_size` | `MBH_BATCH_MAX_SIZE` | 16 |
| `window_before_days` | `MBH_WINDOW_BEFORE_DAYS` | 90.0 |
| `window_after_days` | `MBH_WINDOW_AFTER_DAYS` | 10.0 |
| `window_pad_days` | `MBH_WINDOW_PAD_DAYS` | 4.0 |
| `response_order` | `MBH_RESPONSE_ORDER` | 8 (was 30, not env-backed) |

Consistency rule for `batched`: both generators (the stock one the engine
installs for residual rebuilds and the check path, and the windowed one)
use `waveform_duration = window_before`. If `MBH_WAVEFORM_DURATION` is set
to a different value the build raises with the reason; the `full` path keeps
its existing default. `use_tdionfly=True` with `batched` raises (the
windowed generator is the legacy-response family).

`get_mbh_windowed_gen(general_info, cfg)` builds and caches the windowed
generator per device exactly as `get_mbh_phenom_gen` does (device-local
orbits and domain settings), and `build_mbh_move_runtime` selects
`MBHBatchedMoveBuilder(MBHMoveBuilder)` (`move_class =
MBHBatchedLikeMove`, `use_dcga = False`) when `cfg["mbh_likelihood"] ==
"batched"`, passing `batched_gen`, the batch size and the window knobs
through `move_kwargs`. `wave_gen` stays the slow exact generator, as in the
SOBBH builder. `find_source_cfg` / `make_source_cfg` carry the new keys.

### 4.6 Laptop validation against the mojito MBHB files (user ruling)

`scripts/mbh/mbh_batched_mojito_check.py`, CPU, one MBHB id at a time
(ids 16 to 19 are on this laptop under
`~/.mojito_cache/brickmarket/mojito_light_v1_0_0/data/MBHB/L1/`). It reads
the L1 stream through `lisatools.globalfit.preprocessing.L1ProcessingStep`
into a 120-day window with the catalogue merger 100 days in, builds the
WDM grid with `WDMSettings.adjust_to_even_bins(0.5 d, 0.75 d, dt, window)`,
and for the catalogue truth plus three jittered rows reports, stock
(full-grid, `T = 90 d`) versus batched-windowed: mismatch between the two
templates, `<d|h>`, `<h|h>`, logL of each against the mojito data, delta
logL, the stock template's power outside the kept box (what the sub-box
drops), the kept-layer relative error, and s/row for both. Runs under the
8 GB RSS watchdog the existing MBH scripts use. This is the "match/logL
will be okay" evidence; the acceptance line is |delta logL| within the
`MBH_CHECK_LL_TOL` default (0.5 nats) and mismatch below 1e-6.

### 4.7 Probe (cluster, GPU)

`scripts/mbh/mbh_batched_probe.py`: builds the stock 6mo fit's MBH branch
(no sampler), then for one injected MBH and a Fisher-scaled walker cloud
reports a table: stock s/row; batched s/row at B in {1, 4, 8, 16, 24};
generation/transform/scoring split; peak device memory (cupy mempool
`used_bytes` and JAX); max |delta lnL| batched vs stock; and the
sub-transform's kept-layer relative error vs the full-grid transform. JSON
next to the log. This is the go/no-go for flipping the knob.

## 5. Memory and timing expectations (to be measured)

6mo run, dt 2.5 s, 100-day window = 3.46M samples per row. Per batched row
about 0.4 GB during the response launch (strain, six links of `y_gw`, three
TDI channels) plus ~0.08 GB per stacked sub-box template; the per-row
transform transient is bounded by the 256 MiB layer budget. B = 16 is
roughly 7 GB transient plus 1.3 GB of templates. Expected s/row: 0.05 to
0.2 at B = 16 against 1.3 today, before the order 30 -> 8 gain. The probe
replaces these numbers.

## 6. Testing

Unit tests are `unittest`, CPU, small grids; each test names the line that
breaks it (mutation rule).

- `tests/test_wdm_subbox.py`: `WDMSettings.get_slice`; sub-box
  `add_signal` round trip; `_slice_wdm_to_template` inner product equals
  the full-grid inner product for a template supported inside the box;
  sub-transform vs full-grid transform on a synthetic chirp: kept layers
  agree to a pinned tolerance and the discarded pad layers do NOT (proves
  the pad is load-bearing); pickle/deepcopy of sliced settings.
- `tests/test_mbh_batched_move.py` (skips without jax/phentax): toy ACA of
  3 walkers with DIFFERENT residuals and DIFFERENT PSD scalings; batched
  `compute_like` equals the base container path within tolerance;
  expose/fold restores the residual; a remainder chunk of one row; 2-shard
  routing with a stub holder; `BatchNotLaunchable` fallback counts and
  scores; `_verify_prev_logl` passes at the default tolerance; builder
  selection on `MBH_LIKELIHOOD=batched` and the `full` default; the
  duration-consistency and tdionfly guards raise.
- Existing suites that must stay green: `test_batched_likelihood`,
  `test_batching_isolation`, `test_coarse_wdm`, `test_sobbh_chunked_move`,
  `test_psd_move_batched`, `test_aca_vectorized_dispatch`.

## 7. Rollout

1. Land the branch (worktree `cd1l-merge`) with the knob OFF.
2. Run the probe on the cluster on the 6mo data; record s/row, memory,
   max |delta lnL| and the kept-layer error in the run notes.
3. Relaunch the campaign segment with `MBH_LIKELIHOOD=batched`,
   `MBH_CHECK_LL_EVERY=10` for the first segment, and watch `[MBH_BATCH]`
   plus the check_ll lines; the fold-back invariant check
   (`_verify_entry_vs_acs`) stays on.

## 8. Out of scope

EMRI batching; the TDI-on-the-fly MBH path; changing `dt`; a per-row (rather
than per-leaf) window; the PR's CD1-L SLURM launchers.
