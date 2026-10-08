# Eigen / information-matrix routes

Every proposal in the global fit that is shaped by a local curvature, which
information matrix feeds it, how to switch, and what is production. Status as
of **2026-10-06**.

Three proposal families need an information matrix:

- **GB in-model block proposal** (`gbspecialstretch.py`): per source, the
  Cholesky factor of the inverse information matrix (and, with
  `GB_INMODEL_OBSERVABLE_EIGEN`, the observable-basis `Gamma_z`), frozen across
  each in-model block.
- **Add/remove source branches** (SOBBH / EMRI / MBH,
  `addremovemove.py::ResidualAddOneRemoveOneMove`): one eigen table
  `(axes, sigmas)` per leaf for the eryn `EigenAxisMove` inner move, rebuilt
  every `{BRANCH}_EIGEN_REFRESH` leaf visits.
- **Noise branches** (psd / galfor / sgwb, `psdmove.py::PSDMove`): per-branch
  eigen tables for the same inner move, rebuilt every `{P}_EIGEN_REFRESH`
  proposes.

In all three the matrix only shapes a proposal that is **frozen and symmetric
between refreshes**, so every route is valid Metropolis-Hastings; the routes
differ in cost and in how well the proposal is scaled. The source / noise
opt-ins and the GB cache fall back to the older route on any error, with a
logged warning. `SIGHET_INFOMAT_ENGINE=lookup` by itself has no fallback:
without an attached lookup table GBGPU raises, so it always goes with
`SIGHET_REF_BUILD=lookup` (and, in production, the cache).

## At a glance

| Family | Default (code) | Other route | How to select | Fallback | Status 2026-10-06 |
|---|---|---|---|---|---|
| GB in-model factor | the chunked delegate's matrix; with `SIGHET_INFOMAT=1` + `GB_INFOMAT_PER_BLOCK=1` (set in the v9 launchers) sig-het second differences of lnL per block | **lookup Gram** `<dh_a\|dh_b>` (GBGPU) + **host factor cache** | `SIGHET_INFOMAT_ENGINE=lookup` (needs `SIGHET_REF_BUILD=lookup`), `GB_CHOL_CACHE=1` | cache error: cache disabled AND `SIGHET_INFOMAT_ENGINE` unset, direct per-block factors resume | **Production**: on in the 6mo / 1yr v9 launchers; healthy on the cluster since 6mo job 738 |
| SOBBH / EMRI / MBH eigen tables | `"ll"`: likelihood second differences | `"gram"`: Gram matrix of the move's own templates | `{BRANCH}_EIGEN_INFO=gram` | Gram error: the ll route | **Launcher default** (`gram`) in the 6mo / 1yr / 9mo v9 launchers since 2026-10-07 (V9-30, Mike's production line; `ll` on the line restores the second differences); CPU-validated, first GPU run = the first launch on those defaults |
| PSD / galfor (/ sgwb) eigen tables | `"ll"`: likelihood second differences | `"fisher"`: expected Fisher of the noise covariance | `{PSD,GALFOR,SGWB}_EIGEN_INFO=fisher` | Fisher error: the ll route | **Launcher default** (`fisher` for PSD and GALFOR) in the 6mo / 1yr / 9mo v9 launchers since 2026-10-07 (V9-30); CPU-validated, first GPU run = the first launch on those defaults |

## GB in-model factors: the lookup Gram and `GB_CHOL_CACHE`

The factor for source `x` is `chol(Gamma_y^-1)`, with
`Gamma_y = J^T Gamma_x J` the physical information matrix mapped to the
sampling basis by the transform Jacobian
(`GBSpecialBase._compute_proposal_cholesky`). `Gamma_x` comes from the comp's
`information_matrix`, routed per shard (`gbbands._RoutedBandEngine`). The
routes, in the order GBGPU's `GBSignalHetComputations.information_matrix`
takes them:

1. **`SIGHET_INFOMAT_ENGINE=lookup`** (GBGPU f8073ce, env read on every call):
   the Gram (expected Fisher) `<dh_a|dh_b>` from 2 lookup-table fills per
   parameter and one contraction against the walker's inverse-covariance rows
   (`GBLookupComputations.information_matrix`). f0 / fdot steps are
   `1e-4/Tobs` and `1e-4/Tobs^2`: the chunked defaults move the phase by only
   ~1e-7 / 1e-8 rad, where interpolation noise dominates (fdot-fdot came out
   6x the exact-template Gram at 30 d). Needs the table attached by
   `SIGHET_REF_BUILD=lookup` (else it raises). It needs no in-model block and
   no buffer slot, which is what lets the cache below build every alive
   source in one batch. The switch is process-wide: in the v9 launchers it
   also serves the VGB per-block factors (VGB builds its sig-het engine from
   the same `SIGHET_*` knobs).
2. **`SIGHET_INFOMAT=1`** inside an in-model block with buffer slots: second
   differences of the sig-het in-model scorer (`information_matrix_from_ll`),
   ~2.4 ms/source.
3. Otherwise the **chunked delegate**'s matrix (~46 ms/source).

`GB_INFOMAT_PER_BLOCK=1` (VGB: always) computes exact factors per block for
the block's own sources; with it off, a cold-chain table built once per
proposal is borrowed nearest-in-frequency.

**`GB_CHOL_CACHE=1`** (`src/lisatools/globalfit/moves/gb_chol_cache.py`; GB
only, off by default) keeps the factors and `Gamma_z` in HOST memory, one
cache per rank and branch shared by every GB move, each entry mapped to its
living source:

- **Map.** A source matches within its own **walker** (never its rung: a
  vertical swap only relabels the rung) against the best of the
  `GB_CHOL_CACHE_WINDOW` (32) f0-neighbours on each side, accepted when
  sampling columns 0-2 (amplitude, f0, Mc in the production basis) are all
  within `GB_CHOL_CACHE_TOL` (5) of the entry's own marginal widths. A hit
  moves the entry's key with the source, and every entry is re-keyed to its
  source's final coordinates at the end of each block.
- **Refresh.** Every `GB_CHOL_CACHE_EVERY` (40) proposes on a global ticker
  (the largest `move.time`), ALL alive sources are rebuilt together, in
  `GB_CHOL_CACHE_BATCH` (4096) chunks, at the first in-model block. Births and
  unmatched sources are computed per block in one batch before the repeats.
- **Fallback.** Any exception disables the cache for the rest of the run,
  unsets `SIGHET_INFOMAT_ENGINE` (process-wide: every move and cache on the
  rank leaves the lookup Gram), and logs `[GB_CHOL_CACHE] DISABLED`; the move
  continues on the direct per-block factors.

Cluster record: job 735 (GBGPU f8073ce alone) disabled itself on the first
refresh (`np.asarray` of a cupy step array; fixed in GBGPU 376c94f); job 738
(LAT a20072e7 + GBGPU 376c94f): misses ~1.5-2 %, `inmodel_cholesky` 53 -> ~5 s
per propose, iteration 6.6 -> 5.5 min. Larger Gram steps lowered in-model
acceptance (0.67 -> 0.49 in job 736) while moving sources further per
accepted step.

Tests: `tests/test_gb_chol_cache.py` (LAT, CPU); GBGPU
`tests/test_lookup_information_matrix.py` (lookup Gram vs the exact-template
Gram: elements 7.6e-3, marginal widths within 30 %; CPU);
`tests/test_sighet_engine_parity.py` (the `SIGHET_REF_BUILD` /
`SIGHET_ANCHOR_ENGINE` wiring). Open (1yr TODO list): a GPU unit test of the
lookup Gram; keying entries by a per-source ID instead of f0 matching.

## Add/remove source branches: `{BRANCH}_EIGEN_INFO=gram`

`ResidualAddOneRemoveOneMove._build_eigen_table` picks, in order:

1. an `eigen_table_builder` installed from the branch Settings'
   `info_matrix_gen` (custom; overrides everything);
2. **`"gram"`** when `{BRANCH}_EIGEN_INFO=gram` (move attribute `eigen_info`)
   and the move implements `_gram_templates`;
3. **`"ll"`** (default): second differences of the move's own `compute_like`
   at `{BRANCH}_EIGEN_EPS_REL` (1e-4) of the prior box, shaped by
   `{BRANCH}_EIGEN_SCOPE` (`walker_max`: one table at the max-lnL cold walker;
   `per_walker`: one per (temperature, walker) point). A failed ll build gives
   identity axes with 1 %-of-box steps.

**The Gram route** (`_build_eigen_table_gram`, `_gram_info`) builds
`<dh_a|dh_b>` at the max-lnL cold walker from central differences, in the
sampling basis, of the move's OWN batched templates, through that walker's
container (`_slice_to_template` + `inner_product`: the scorers' PSD and
normalization). One batch of `2 * ndim + 1` templates per pass. Steps start at
`{BRANCH}_EIGEN_GRAM_EPS_REL` (default: `{BRANCH}_EIGEN_EPS_REL`) of the box
and are rescaled once so each column moves the template by
`{BRANCH}_EIGEN_GRAM_TARGET` (1e-3) of `||h||` (factor clipped to
[1e-3, 1e3], step to [1e-12, 1e-2] of the box; `0` keeps the fixed steps).
No fixed relative step works for a chirping source: on the SOBBH toy, f_low
goes nonlinear above ~1e-6 of its box while the masses reach the template's
float noise floor below ~1e-7. The Gram route always builds ONE shared table
(`{BRANCH}_EIGEN_SCOPE` does not apply) and returns the same `(axes, sigmas)`
form as the ll route's `walker_max`.

Hooks (`_gram_templates(phys_rows, walker) -> (arr, box)`, plus an optional
`_gram_context(walker)` device context):

| Branch | Templates | Notes |
|---|---|---|
| SOBBH (`SOBBHChunkedLikeMove`) | `comp.fill_global_wdm`, one zeroed slab per row: lookup fills under the stock `SOBBH_LIKELIHOOD=lookup`, chunked fills under `chunked` | single-shard ACA only; holds 23 data-shaped slabs (~0.6 GB peak at 6 months, ~1.2 GB at 1 year) |
| EMRI (`EMRIDirectLikeMove`) | the direct-to-WDM adapter's templates on the active box, `batch_max_size` chunks | a refused row raises into the fallback |
| MBH (`MBHBatchedLikeMove`) | the batched windowed templates in the leaf's window, on the containers' box | no window / a refused batch raises into the fallback |

**Why** (SOBBH chunked toy, 2026-10-06): at its production step (1e-4 of the
box) the ll route gave an **f_low curvature ~250x too small** (that step is
deep in f_low's nonlinear range; the resulting f_low step ~16x too wide),
**spin entries dominated by lnL noise**, and **masses ~1.5x off** the Gram.
(The ba428237 commit message's "25-250x off in masses / spins / f_low"
overstates the masses.)

Accuracy (CPU): SOBBH chunked vs an independent dense stock-template Gram at
the same steps, diagonal within 1-3 %, marginal widths within 3.3 %; SOBBH
lookup comp, max correlation-normalized difference 1.5e-2; EMRI exact (1e-8)
vs the toy generator; MBH windowed within 2e-2. Step-target convergence:
3e-3 vs 1e-3 differs by 5e-3, 1e-4 vs 1e-3 by 4e-4.

Tests: `tests/test_sobbh_chunked_move.py::SOBBHGramInfoTest`,
`tests/test_sobbh_lookup_move.py::SOBBHLookupGramTest`,
`tests/test_source_gram_info.py` (EMRI + MBH: per-walker PSD with a negative
control, exactly flat columns, routing, fallback). The ll route:
`tests/test_eigen_refresh.py`, `tests/test_eigen_table_persist.py`.

The eigen-table sidecar does not record which route built a table: after a
restart the persisted table serves until the leaf's next refresh, whichever
route is selected (valid for the same frozen-proposal reason as above).

## Noise branches: `{PSD,GALFOR}_EIGEN_INFO=fisher`

`PSDMove._refresh_eigen_tables` expands at the points `{P}_EIGEN_SCOPE` picks
(`cold_per_walker`, the default: each walker's cold row, shared up its ladder
with tempered sigmas; `per_temp`: every rung at walker 0) and takes, per
branch:

- **`"ll"`** (default): second differences of the noise likelihood through
  `compute_psd_rows`, step `{P}_EIGEN_EPS_REL` (1e-4) of the box.
- **`"fisher"`** (`PSDMove._noise_fisher`, move attribute `eigen_info`, env per
  branch): the expected Fisher

      F_ij = kappa * sum_pix Tr(C^-1 dC/dx_i C^-1 dC/dx_j)

  with `kappa` the likelihood's `logdet_factor` and `dC` central differences
  (same `{P}_EIGEN_EPS_REL` step) of the SAME PSD_BATCH covariance the noise
  scorer builds (`_batched_covariance`); the other noise branches are held
  where the ll route holds them. It needs no residual, so it is positive
  semi-definite everywhere; the observed curvature is not (34 of 55 galfor
  builds were non-positive on the 3-month run). PSD_BATCH route only (raises
  into the ll route otherwise).

Accuracy (CPU, the synthetic noise_only fit, residual drawn from `C(x0)`, 60k
pixels): psd diagonal within 0.6 % of the observed curvature; galfor (loud)
alpha / f_1 within 4 %, while the observed curvature of fk / f_2 is negative
on the same draw. Tests: `tests/test_psd_move_batched.py::NoiseFisherTest`
(psd and galfor; `SGWB_EIGEN_INFO=fisher` runs the same code but has no test);
the ll route: `tests/test_psd_eigen_inner.py`.

Related (LAT 290c1902): the PSD_BATCH scorer now folds galfor / SGWB with the
backend's `WDM_PSD_METHOD` (production `layer_calibrated`), the same model the
per-walker sensitivity uses, so both noise routes differentiate the
likelihood the rest of the run scores.

## Knob reference

The new knobs follow the naming rule (env = branch prefix + capitalized
attribute). Two older ones do not and are kept because production launch
lines use them: `{BRANCH}_EIGEN_REFRESH` seeds `eigen_refresh_every` and
`{BRANCH}_EIGEN_SCOPE` seeds `eigen_table_scope`.

| Env knob | Attribute | Code default | 6mo / 1yr / 9mo v9 launchers |
|---|---|---|---|
| `SIGHET_INFOMAT` | (GBGPU env) | off | `1` |
| `GB_INFOMAT_PER_BLOCK` | move `infomat_per_block` | `0` (VGB: always on) | `1` |
| `SIGHET_REF_BUILD` | GB / VGB Settings `sighet_ref_build` | `fd` | `lookup` |
| `SIGHET_ANCHOR_ENGINE` | GB / VGB Settings `sighet_anchor_engine` | `chunked` | `lookup` |
| `SIGHET_INFOMAT_ENGINE` | (GBGPU env) | unset | `lookup` (set-empty on the launch line turns it off) |
| `GB_CHOL_CACHE` | (env, `gb_chol_cache.py`) | `0` | `1` |
| `GB_CHOL_CACHE_EVERY` / `_TOL` / `_WINDOW` / `_BATCH` | (env) | `40` / `5` / `32` / `4096` | `40` / `5` / default / default |
| `{SOBBH,EMRI,MBH}_EIGEN_INFO` | move `eigen_info` | `ll` | `gram` (default since 2026-10-07, V9-30; `${K:-gram}`) |
| `{SOBBH,EMRI,MBH}_EIGEN_GRAM_TARGET` | move `eigen_gram_target` | `1e-3` | not set |
| `{SOBBH,EMRI,MBH}_EIGEN_GRAM_EPS_REL` | move `eigen_gram_eps_rel` | `{BRANCH}_EIGEN_EPS_REL` | not set |
| `{SOBBH,EMRI,MBH}_EIGEN_EPS_REL` | move `eigen_eps_rel` | `1e-4` | not set |
| `{PSD,GALFOR,SGWB}_EIGEN_INFO` | PSDMove `eigen_info` | `ll` | `fisher` for PSD and GALFOR (default since 2026-10-07, V9-30; `${K:-fisher}`); SGWB not set |
| `{PSD,GALFOR,SGWB}_EIGEN_EPS_REL` | Settings / PSDMove `eigen_eps_rel` | `1e-4` | not set |

The `eigen_info` / `eigen_gram_*` move attributes are constructor kwargs too;
`None` (the default) defers to the env knob, read when the table is built, so
an attribute set on a live move wins. Unlike `eigen_eps_rel` /
`eigen_refresh_every` they are not Settings fields: set them on the launch
line (env) or on the built move.
