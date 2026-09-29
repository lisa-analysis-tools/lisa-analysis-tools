# Adaptive per-axis in-model step scales (GB observable eigen path)

**Branch `gb-inmodel-axis-adapt`, off `origin/dev` at `688328ce`. Not for
`dev` without a ruling.** This is "Part C" of the 2026-09-28 observable
work — the per-axis step scaling the launcher comment defers to.

Two of the findings below are about code that is running in production
today and are independent of whether the adaptation is ever armed. They
are §1 and §2.

---

## 1. `GB_INMODEL_OBSERVABLE_JUMP` does not reach the eigen step

Under `GB_INMODEL_OBSERVABLE_EIGEN=axis` (the production setting), the
jump knob is **exactly cancelled** on every eigen axis the information
matrix could measure.

`_observable_eigen_prepare` uses the step scales `w` as a *whitening
metric*, not as a step size:

```
gw      = gz * w (x) w
a, sig  = eigen_axis_set(gw, t_fiber, sigma_max=SMAX)   sig = 1/sqrt(aᵀ gw a)
table   = w * a * sig
```

Scale `w → c·w`: `gw → c² gw`, `sig → sig/c`, and
`table → (c·w) · a · (sig/c) = table`. The knob multiplies `w` and
nothing else.

Measured (`obs_axis_reorder`'s tests, 400 synthetic sources, nothing
railed) — sorted per-axis step magnitudes against `jump = 1`:

| jump | min ratio | median | max |
|---|---|---|---|
| 1.5 | 1.000000 | 1.000000 | 1.000000 |
| 2.0 | 1.000000 | 1.000000 | 1.000000 |
| 7.3 | 1.000000 | 1.000000 | 1.000000 |

It survives in exactly two places:

- **axes railed at `GB_INMODEL_OBSERVABLE_EIGEN_SMAX`** (default 10),
  where `sig` is a constant and the `w` in the product no longer has a
  `1/c` to cancel against. Measured ratio there is exactly the jump.
  These are the directions the matrix *could not* measure — i.e. the
  ones already most likely to be overshooting.
- **rows with no eigen table** (fresh births before their first infomat
  visit), which take the diagonal draw, where the knob is linear.

So the launcher's tuning history — `1.0 → 2.0` (2026-09-19) →
`1.5` (2026-09-28) — was moving the railed axes and the newborns, not
the well-measured axes the argument was about. The same applies to the
`axis_mult` seam that `_obs_axis_mult_for` was built as: it enters at
the same place.

**What to do about it is a judgement call, not a defect fix.** Making
`jump` multiply `sig` instead would change production sampling by a
factor of 1.5 on every eigen axis at once. This branch does not do that;
it adds a separate, off-by-default multiplier at `sig` and documents the
cancellation at both seams.

## 2. The eigen table's column index was not an identity

`eigen_axis_set` orders its columns by `|overlap with the fiber|` so the
fiber-aligned eigenvector lands last. That is the right key for *one*
column and a non-key for the other eight: the fiber is projected out
exactly (`P F P`), so the remaining eigenvectors are orthogonal to it to
machine precision. Their overlaps are rounding noise — measured median
`3e-18 … 2e-14` across the eight — and `argsort` over rounding noise is
a permutation, redrawn every time the table is rebuilt.

| column order | holds its direction after a 1% change to the matrix |
|---|---|
| by \|fiber overlap\| (shipped) | **7.2%** (median \|⟨a_k, a_k'⟩\| = 0.002) |
| by the axis's own curvature | **97.8%** (median 0.999) |

Consequences, in order of how much they matter:

1. **The per-axis census labels were wrong.** `[GB_OBS_BASIS] per-axis:`
   labelled bucket `k` with `GB_INTERNAL_BASIS[k]` — `lnA`, `f_mid`,
   `fdot`, … — which names a coordinate, not the axis in that bucket.
   The launcher's `2.0 → 1.5` note reads those numbers as "f_mid was
   accepting ~7% of its own draws and fdot 3–7%". The buckets were real
   and the spread was real; the *names* were not. The noise ordering is
   anti-correlated with eigenvalue rank (column 0 took the largest
   eigenvalue 59% of the time), so the line was closer to a per-width
   ranking than to anything per-coordinate.
2. A per-axis multiplier learned in one block would be applied to a
   different direction in the next.
3. A per-band mean table needs axis `k` to mean the same thing for every
   source in the band.

**Fix:** `obs_axis_reorder` sorts the non-fiber columns by the
**uncapped** whitened curvature `aᵀ Γ_w a` — not by `sig`, which is
`min(1/sqrt(quad), SMAX)` and therefore identical for every railed axis,
which would leave exactly the near-null directions arbitrary again. Axis
0 is the widest step, axis n-2 the tightest, fiber last where
`_observable_proposal` expects it.

This is **distribution-neutral**: `axis` mode picks uniformly over the
non-fiber columns and `full` mode contracts the table with iid normals,
and both are invariant under a column permutation. The realised RNG
stream differs.

The census now labels ranks `s0 … s7`, `fib`, and carries
`dom=<coord>(frac)` — the coordinate that most often dominates that
rank, over sources. `s0 dom=psi(0.41)` means "the widest axis was mostly
psi for 41% of sources".

**This section is cherry-pickable to `dev` on its own** (commit
`397caed1`): it changes no sampling distribution and it makes a
production log line honest. That is Mike's call.

---

## 3. The adaptation

`log g += gain · (accept_rate − target)`, clipped to `[1/bound, bound]`,
applied to `sig` after the `SMAX` cap.

### Granularity: `(temp, walker, band, axis)`, not `(source, axis)`

The original spec said per-source, carried on `branch_supplemental`
through the leaf repack. Three things argue against it:

1. **Statistics.** One axis is drawn per repeat and a source sees a
   handful of repeats per propose, so most `(source, axis)` pairs
   collect zero or one draw. A per-source rate is then 0 or 1 and the
   update is a coin flip. Pooling over a band's sources is the
   difference between an estimate and a random walk.
2. **The key.** `(temp, walker, leaf)` moves under you — leaves are
   repacked every propose, RJ renumbers them, temperature swaps exchange
   whole walkers. `band` and `temp` are slot properties (the band
   assignment is frozen per propose; a band is fixed for the run), so
   nothing has to be carried and no birth / death / swap case exists.
3. **What is actually wrong.** The step already divides by `rho` and is
   whitened by the source's own information matrix, so the per-source
   part of the answer is largely in there already. What the matrix gets
   *systematically* wrong belongs to the frequency regime and the axis.

Keying on `temp` is a bonus rather than a cost: the optimal width goes
as `1/sqrt(beta)` and nothing in this path knows that today (the
standing TODO in `project_todo_eigen_walker_max_beta_scaling_0918`).
Per-rung adaptation can discover it.

A newborn starts at its band's learned value for free — the spec's
"band-mean table for newborns" is not a second table here, it *is* the
table.

**Not captured:** the per-source residual — two sources in one band that
want different steps. That still needs the `branch_supplemental`
carrier, and remains the follow-up.

### Why a constant gain, not a decaying Robbins-Monro schedule

A vanishing gain buys convergence of the adapted parameter for a chain
whose target is fixed. Neither half holds here: this runs in the search
stage only, where the posterior a source explores moves under it as
neighbours are found and subtracted, and where detailed balance is
already not claimed. A constant gain is a tracking filter. Swapping in
`gain / (1 + visits)**kappa` is a one-line change — but the one stage
that would want a convergent adaptation is PE, which must not adapt at
all.

### Search-only arming

`build_gb_moves` stamps `gb_search_stage = True` on **search-exclusive**
moves. `gb_ridge_gibbs` is one object appended to both lists, so it is
left unstamped — a single object cannot be in two stages and "do not
adapt" is the safe resolution. The `"search" in name` test from
`_replace_fstat_max` stays as the fallback, because the in-model slots
(`in_model`, `in_model_fstat`, `in_model_replace`, `in_model_removal`)
and `rj_prior_removal` carry no stage in their names, and they are the
moves that run most of the in-model repeats.

### Knobs

| knob | default | meaning |
|---|---|---|
| `GB_INMODEL_OBSERVABLE_AXIS_ADAPT` | `0` | master arm. Off = table stays `None`, step byte-identical |
| `GB_INMODEL_OBSERVABLE_AXIS_TARGET` | `0.44` | target acceptance; 1-D optimum, since axis mode draws one axis per repeat |
| `GB_INMODEL_OBSERVABLE_AXIS_GAIN` | `0.2` | log-space gain per propose |
| `GB_INMODEL_OBSERVABLE_AXIS_BOUND` | `8.0` | multiplier clamp, `[1/8, 8]` |

### What to watch

```
[GB_OBS_AXIS <move>] adapted N of M (temp, walker, band, axis) cells on
D draws; log-multiplier mean +x.xxx min +x.xxx max +x.xxx; at the bound K
```

`at the bound K` climbing is the signal that the bound, the gain or the
target is wrong — or that the thing the multiplier is being asked to fix
is not a step-size problem.

Read it against the corrected per-axis line:

```
[GB_OBS_BASIS <move>] per-axis (s0=widest): s0 dom=psi(0.41) d=... a=... (0.63) ...
```

---

## Where the code is

| piece | location |
|---|---|
| column order + dominant coordinate | `gbspecialstretch.py::obs_axis_reorder` |
| the per-eigen-axis seam | `gbspecialstretch.py::obs_axis_scale_for` |
| the cancellation, documented at the old seam | `gbspecialstretch.py::_obs_axis_mult_for` |
| update rule | `gbspecialstretch.py::obs_axis_adapt_update` |
| arming | `gbspecialstretch.py::obs_axis_adapt_on` |
| cell key / accumulate / step | `GBSpecialBase._obs_axis_bind_keys`, `_obs_axis_adapt_accum`, `_obs_axis_adapt_step` |
| stage stamp | `recipe.py::build_gb_moves`, at the `_stamp_temper_seed_base` choke point |
| tests | `tests/test_gb_observable_eigen.py`, classes `AxisIdentityIsStable`, `CensusLabelsAreRanksNotCoordinates`, `PerAxisScaleReachesTheStep`, `AxisAdaptUpdate`, `AxisAdaptArming`, `AxisAdaptCellKey`, `AxisAdaptEndToEnd` |

Both §1 and §2 are encoded as **negative-control tests**
(`test_NEGATIVE_CONTROL_the_JUMP_knob_is_cancelled_too`,
`test_NEGATIVE_CONTROL_the_shipped_order_does_NOT_survive_it`), so
neither finding can rot silently.

## Not done

- **Per-source residual scaling** (`branch_supplemental` carrier).
- **Persistence across a restart.** The table lives on the move and is
  rebuilt from scratch on a restart; at gain 0.2 it re-learns in a few
  tens of proposes. A sidecar is straightforward if that turns out to
  matter.
- **A decision on §1.** Whether `GB_INMODEL_OBSERVABLE_JUMP` should be
  made to reach the eigen step is a production-sampling change and is
  Mike's call.
