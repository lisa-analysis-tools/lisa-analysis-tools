# SOBBH direct-to-WDM lookup scorer for the add/remove move — design

Date: 2026-09-30. Branch `sobbh-wdm-lookup` (worktree `LISAanalysistools-sobbh-lookup`, from `dev`
@ `fd13429b`). Handoff it builds on: `~/.claude/plans/emri-wdm-lookup-handoff-for-sobbh.md`
(the EMRI direct-to-WDM method, LAT `dev`, `docs/emri-direct-wdm.md`).

## 1. Goal

Score SOBBH add/remove proposals in the global fit with a **vectorized direct-to-WDM lookup
template** instead of the chunked-heterodyne kernel: for a batch of parameter rows, run the
SOBBH TDI-on-the-fly response once for the whole batch, read per-channel amplitude, phase,
frequency and chirp rate at the WDM pixel centres, map them through the `n_ref` lookup table
(the EMRI evaluation rule, unchanged), and accumulate `<d|h>` / `<h|h>` against the live
residual buffers of the `AnalysisContainerArray`. The result slots into
`SOBBHChunkedLikeMove` as a drop-in replacement for the chunked comp, so the move's
choreography (expose/fold, in-model repeats, tempering, cross-check, telemetry, multi-shard
routing) is untouched.

Success:

1. Lookup template vs the same TDI-on-the-fly response transformed densely (TD -> WDM): per
   channel mismatch `1 - Re(O)` (no maximisation) at the table-resolution level (~1e-4..1e-3)
   AND norm ratio within 2e-3 of 1 on the production layer duration (3600 s).
2. Through the move seams, fast (lookup) vs slow (container) log-likelihood agree within the
   move's cross-check tolerance on the existing toy, and the chunked move's whole test suite
   passes with the lookup comp substituted.
3. A measured cost table (laptop CPU): lookup `get_ll` and fill per batch vs the chunked comp
   on the same grid, with the row-batch scaling. No speed claim without that table.

## 2. What was measured before designing (this session)

- **Table portability.** Two `n_ref_complex` tables with the same layer duration
  (`layer_dt = 3600 s`) but different sampling (`Nf=64, dt=56.25` and `Nf=128, dt=28.125`)
  agree entry by entry to 4e-9 of the peak, and the coarse-sample table evaluated on the
  finer grid reproduces the chirp truth with the same error as that grid's own table
  (2.2e-4 at fdot 0, 9e-4 at fdot 0.25 layer units; norm ratio within 3e-4). **The laptop
  EMRI table** `wdm_lookup_emri_cx_NF180_DT20_TL32_fd8x0p01_nld2.h5` (layer 3600 s, 30 MB,
  fdot axis +-8 layer units in steps of 0.01, offsets [-3, 3) layers) **is therefore usable on
  the production grid** (`Nf=1440, dt=2.5`, same layer duration). No cluster table build.
- **SOBBH chirp in layer units** (3.5PN, layer unit `layer_df/layer_dt = 3.86e-8 Hz/s`): every
  catalogue-like source (`f_low <= 1e-2`, masses <= 100) has `|fdot| <= 0.002` layer units
  over 6 months and a cubic-phase term `(pi/3)|fddot|(4 layer_dt)^3 <= 4e-6` rad. The 6-mo
  catalogue's worst chirper (id 0) sits at ~0.005. Sources merging inside the window at
  >= 2e-2 Hz run to hundreds of layer units in their last days. Consequences: 5 layers per
  pixel (`num_m_layers=2`) are plenty; no curvature handoff and no plunge chunk; pixels whose
  `|fdot|` leaves the table axis are dropped and counted, never extrapolated.

## 3. Architecture

Three new units, one wiring change, one validation ladder. All Python, `xp`-vectorized
(numpy on CPU, cupy on CUDA); no new C++/CUDA (see section 9).

### 3.1 `lisatools.wdm_lookup_eval.WDMLookupEvaluator` (source-agnostic)

Vectorized evaluation of an `n_ref_only` / `n_ref_complex` `WDMLookupTable` on arrays of any
shape, with the quarter-turn rule of `WDMLookupTable.get_wdm_coeffs` (BASIS_CYCLE
`"quarter_turn"`) reproduced exactly:

```python
ev = WDMLookupEvaluator(table, interp="cubic", force_backend="cpu")   # or the table's backend
m = ev.layers_for(f_ref, num_m_layers=2)                              # (..., L) int, L = 2k+1
w, ok = ev.coeffs(amp, phi, f, fdot, n, m)                            # (..., L) float, bool
```

- Inputs broadcast to a common shape `(..., L)`: `amp`, `phi` (the channel phase at `t_n`:
  the channel signal is `amp * cos(phi)` with `phi` increasing in time, which is what
  `tdi_phase + phase_ref` of a `TDTDIOutput` gives since `Re[amp exp(-i phase)] = amp cos(phase)`),
  `f`, `fdot` per element, `n` (absolute pixel index), `m` (explicit target layers). `ok` marks entries inside the table's
  offset and fdot support; outside entries are zero.
- Rule per element: `delta = f - m*layer_df`; read `c = cos table`, `s = sin table with the
  build's (-1)^block bake undone at the NODES` (precomputed once from `table_cx`); if
  `(m_ref + n_ref)` odd: `c -> -c`; if `(m + n)` odd: `(c, s) -> (s, -c)`;
  `w = amp * (c cos(phi) - s sin(phi))`.
- Interpolation on the uniform `(fdot, f_norm)` grid: `"linear"` (4 gathers) or `"cubic"`
  (Keys cubic convolution, a = -1/2, 16 gathers), both written with `xp` gathers so they run
  on numpy and cupy alike. Cubic is the default: linear left a 2.5e-4 amplitude deficit on
  the EMRI (handoff section 3). The table arrays live on the evaluator's backend.
- No array module stored on the instance (pickle rule): `xp` is a property off `backend`.

### 3.2 `lisatools.sources.sobbh.wdm_direct` (the SOBBH tracer + template)

- `sobbh_amp_phase_batch(params, times, reference_time)`: the 3.5PN core of
  `lisatools.sources.sobbh.waveform` vectorized over rows (chunked basis
  `(m1, m2, s1, s2, dist[pc], f_low, phi_c, inc, psi, lam, beta)`, the same columns the
  chunked comp consumes). Returns `amp (N, T)`, `gw_phase (N, T)`, `tc (N,)`; samples at or
  past `tc` get `amp = 0` and a frozen phase. The `tc` Newton refinement of
  `_pn_amp_phase_core` is kept, vectorized. Row-wise identical to
  `SOBBHWaveform.compute_amp_phase`.
- `SOBBHBatchedTOF(orbits, tdi_config, reference_time, n_grid, buffer_time, eval_dt,
  force_backend)`: builds ONE `TDTDIonTheFly(num_sub=N)` for a batch, exactly as
  `bbhx.sobbhtdionfly.SOBBHTDIonFly` builds it for one row (node grid `linspace(lo -
  buffer_time, hi + buffer_time, n_grid)`, `phase = gw_phase + pi`, intrinsic amplitude, real
  `inc`/`psi`, `(ra, dec)` in the orbits frame), but evaluated on a COARSE eval grid of step
  `eval_dt` (default 600 s) spanning the pixel centres plus two steps each side, and returned
  with splines (`return_spline=True`). The dense data grid is never built.
- `sobbh_tracer(out, t_pixels)`: `(amp, phase, f, fdot)` of shape `(N, nch, P)` from the
  output splines: `phase = tdi_phase + phase_ref` (the channel is `Re[amp exp(-i phase)]`,
  `TDTDIOutput.eval_tdi`), `f` and `fdot` from the splines' first and second derivatives
  (`CubicSplineInterpolant(..., derivative=1|2)`), so they carry the per-channel Doppler
  shift. Negative-frequency rows are mirrored as in the EMRI tracer.
- `SOBBHDirectWDM`: the template builder. `sparse(params) -> (w (N, nch, L, P), m (N, L, P),
  n (P,), ok)` and `dense(params) -> WDMSignal` (for gates and tests). Per pixel the layer
  window is common to all channels: `m = floor(f_ref/layer_df) + [-k..k]` with `f_ref` the
  channel-mean frequency; each channel is read at its own `delta = f_c - m*layer_df`.
  Pixel centres are `t_n = t_obs_start + n*layer_dt` for the active time pixels
  `n in [ind_min_t, ind_max_t]`. Dropped: `amp == 0` (past merger), `|fdot|` outside the table
  axis (counted in `last_stats["dropped_pixels"]`, one warning per call), layers outside the
  active band.
- `sparse_inner_products(w, m, n, holder_buffers, data_index, noise_index, wdm_settings,
  tdi_type) -> (d_h, h_h)`: the chunked kernel's accumulation, gathered at the sparse support:
  `d_h = sum_{pix} sum_{c,c'} d_c invC_{cc'} h_{c'}`, `h_h = sum h_c invC_{cc'} h_{c'}` with the
  full 3x3 `invC` for XYZ (6 unique reads) and diagonal for AET/AE, over the ACA's flat
  `linear_data_arr[0]` / `linear_psd_arr[0]` slabs `(nch, Nf_active, Nt_active)` and
  `(nch, nch, Nf_active, Nt_active)` per walker. No `4 * differential_component` factor:
  this is the kernel's convention (`-0.5*(h_h - 2 d_h)` equals the container's source term;
  the move adds the per-walker `-1/2<r|r>` offset).
- `scatter_add(w, m, n, buffer, data_index, factors)`: `xp.add.at` / `cupyx.scatter_add`
  of `factors[row] * w` into the active-band buffer (the fill).

### 3.3 `lisatools.sources.sobbh.wdm_direct.SOBBHLookupComputations` (the slot-in comp)

Duck-types the surface of `bbhx.sobbhcomps.SOBBHWDMComputations` that
`SOBBHChunkedLikeMove` and the engine signal generator use:

| used by the move / engine | lookup comp |
|---|---|
| `get_ll_wdm(params, holder, data_index=, noise_index=, m_band_half_width=)` | lookup scorer; returns `-0.5*(d_d + h_h - 2 d_h)`, stashes `d_h_out`, `h_h_out`, `d_h_im_out=None`, `last_call_spans` |
| `fill_global_wdm(params, templates, data_index=, factors=, m_band_half_width=, convert_to_ra_dec=)` | scatter-add into an ACA holder / `WDMSignal` / raw `(nch, Nf_active, Nt_active)` buffer |
| `d_d`, `wdm_settings`, `xp`, `backend`, `args`, `kwargs` (per-device replica rebuild) | same meaning |

`m_band_half_width` is accepted and ignored (the lookup's band is `num_m_layers`; logged once).
Constructor: `(wdm_settings, t_ref, table, *, orbits, tdi_config, tdi_type, t_obs_start,
n_grid, buffer_time, eval_dt, num_m_layers, interp, row_batch, force_backend, d_d=0.0)`.
Rows are processed in batches of `row_batch` (default 32) to bound the response-spline
memory. Multi-shard ACAs are served by the move's existing per-split routing (the comp is
single-shard, like the chunked one).

### 3.4 Stock wiring

- `SourceSOBBHSettings.likelihood` accepts `"lookup"` (next to `"chunked"`, `"full"`).
  New fields (env knob = capitalised name): `lookup_table_path` (`SOBBH_LOOKUP_TABLE_PATH`,
  required for `"lookup"`, build raises naming the builder command if unset),
  `lookup_num_m_layers` (`SOBBH_LOOKUP_NUM_M_LAYERS`, 2), `lookup_eval_dt`
  (`SOBBH_LOOKUP_EVAL_DT`, 600.0), `lookup_interp` (`SOBBH_LOOKUP_INTERP`, `"cubic"`),
  `lookup_row_batch` (`SOBBH_LOOKUP_ROW_BATCH`, 32).
- `source_signal_cfg` carries them; `get_sobbh_lookup_comp(general_info, cfg)` builds and
  caches the comp per device (same `t_ref`/`t_obs_start` resolution as the chunked comp, table
  loaded on `general_info.force_backend`); `build_sobbh_move_runtime` hands it to
  `SOBBHChunkedMoveBuilder(chunked_comp=...)`; `SourceSignalGen.__call__` and the engine
  signal generator pick the comp by `sobbh_likelihood in ("chunked", "lookup")` so residual
  bookkeeping and scoring share one template family, as today.

## 4. Data flow of one `get_ll_wdm` call

1. Validate rows (finite, `f_low` in band) — already done by the move; the comp re-checks
   nothing.
2. For each row batch: `sobbh_amp_phase_batch` on the node grid -> `TDTDIonTheFly` (one C++
   call for the batch) on the eval grid -> `TDTDIOutput` splines -> `sobbh_tracer` at the
   active pixel centres -> `WDMLookupEvaluator.coeffs` on the common 5-layer window ->
   `sparse_inner_products` against the walker slabs named by `data_index`/`noise_index`.
3. Concatenate `d_h`, `h_h`; return the likelihood vector; record spans
   (`stage`, `response`, `tracer`, `lookup`, `inner`, `total`) on `last_call_spans` so the
   move's `[SOBBH_LL_TIMING]` lines keep working.

The fill is the same chain with `scatter_add` in place of the inner products.

## 5. Error handling

- Table/grid mismatch (`layer_dt` of the table != `wdm_settings.layer_dt`): raise at
  construction with both values.
- Missing table path with `likelihood="lookup"`: raise at build with the
  `scripts/wdm/build_wdm_lookup_gpu.py` command that builds a table for the run's layer
  duration.
- Pixels past merger, outside the fdot axis, or outside the active band: dropped, counted,
  one `logger.warning` per call with the count when `dropped_pixels > 0` (never silent).
- Backend mismatch between table arrays and holder buffers: raise (no implicit copies).

## 6. Testing (every accuracy test has a negative control or a named mutation)

- `tests/test_wdm_lookup_eval.py`: parity with `WDMLookupTable.get_wdm_coeffs` (linear,
  rtol 1e-10); chirp truth with the quarter-turn rule (< 1e-3 rel L2) and the control that the
  parity swap disabled fails (> 0.1); Keys cubic interpolates the nodes exactly and beats linear by
  > 2x off-node (f and fdot); **table portability** pinned
  (two tiny tables at different `(Nf, dt)`, same `layer_dt`: entries equal to 1e-8).
- `tests/test_sobbh_wdm_direct.py`: batched PN amp/phase == `SOBBHWaveform.compute_amp_phase`
  row by row (1e-12); batched coarse-eval TOF == `SOBBHTDIonFly` dense TD on the toy window
  (per channel mismatch <= 1e-8, norm ratio within 1e-6; control: without the `+pi` phase
  convention the mismatch is ~2); tracer on a known quadratic phase; lookup `dense()` vs the
  TOF's own TD->WDM (mismatch <= 1e-3, norm ratio within 2e-3 per channel; control: the
  legacy 2-way rule fails); `sparse_inner_products` == `AnalysisContainer.template_inner_product`
  on the dense template (1e-10); fill round-trip restores the residual; dropped-pixel
  accounting on a merging source.
- `tests/test_sobbh_lookup_move.py`: the `test_sobbh_chunked_move` toy with the lookup comp:
  fast vs slow parity within `check_ll_tol`; expose invariant; out-of-band sentinel;
  `_record_leaf_inner_products`; stock settings pin (`likelihood="lookup"` builds the lookup
  comp; missing table raises). Mutation ritual before reporting green: delete the `h_h`
  accumulation and name the test that fails.
- `scripts/sobbh/sobbh_lookup_gate.py`: the laptop gate on the production layer duration
  (`Nf=180, dt=20`, up to 6 months) for catalogue-like sources: lookup vs TOF-dense mismatch,
  norm ratio, dlogL vs the production template; lookup vs chunked `get_ll` on one residual;
  wall time per batch for both at matching row counts. Writes JSON + a markdown table into
  `docs/sobbh-wdm-lookup.md`.

## 7. Out of scope (documented follow-ups in `docs/sobbh-wdm-lookup.md`)

- The CUDA/C++ lookup kernel in the chunked-het family (per-pixel `SOBBHTDIonTheFly::get_tdi`
  + device table) — the production-speed path once this Python path is validated and timed.
- The CD1L data gate (per-source mismatch/dlogL vs the mojito L1 SOBBH streams): the bricks
  are not on this laptop; the gate script runs where they are (runbook in the doc).
- A JAX mirror; the fused phase-max quadrature; F-stat / gradient methods on the comp.

## 8. Working rules applied

LAT edits only in this worktree; tests run through `.wtenv/run.sh` (editable-install shadow
shim, 2 threads, nice 10, RSS watchdog); one Python process at a time on the laptop; stage
only, no commit or push without Mike's OK; env knobs named for their fields; no backend
strings as method kwargs; no array modules on instances.

## 9. Known tension, stated

The sprint's backend hierarchy says GPU C++ leads. This design ships an `xp`-vectorized
Python evaluator first, like the EMRI direct-to-WDM path did, because (a) the laptop has no
CUDA, (b) the rule's purpose (one algorithm, validated at the inner-product level across
backends) is served by validating this reference implementation now, and (c) the kernel is
the follow-up above with this implementation as its reference. The scoring convention and the
`(d_h, h_h)` outputs are identical to the kernel's, so the swap later is a comp swap, not a
move change.
