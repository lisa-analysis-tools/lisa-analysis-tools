# Lessons

## 2026-07-08 — multi-GPU sync points
Proposed removing the unconditional `synchronize()` before the per-chunk D2H in
`AnalysisContainerArray._compute_group_likelihood`; user corrected: keep it.
**Pattern**: host-consumption barriers (sync before copying per-chunk results
to CPU) are load-bearing and shared across moves (PSD/MBH/EMRI). Optimize by
reducing barrier *frequency* (bigger batches, fewer chunks), never by removing
the barrier itself. Also: `.get()` syncing "should be enough" arguments ignore
off-stream work and shared-code blast radius — weigh risk vs measured gain.

## 2026-10-28 — check a library call before presenting its trade-off (galfor ladder)
Told the user that sizing the psd ladder on psd + galfor dimensions (as the reference does)
would change it; a surviving mutation showed `make_ladder(ndim, ntemps, Tmax)` ignores `ndim`
when both `ntemps` and a finite `Tmax` are given. **Pattern**: before putting a design question
or a downside to the user that rests on a library function's behaviour, call the function
with the two values in question; copying a reference's arguments is not evidence they matter.

## 2026-10-27 — prior bounds are out of scope (ereborpriors)
Reviewed the new `EMRIPrior` and, besides two real interface bugs, flagged its
ranges against the old settings file and FEW limits. User: bounds "are still
subject to change", don't question them. **Pattern**: when checking a prior the
user owns, check only the contract l2d consumes (factory signature, basis
names/order, transforms, periodicity, YAML round trip); never comment on ranges.

## 2026-10-27 — restrict vs fix (l2d recipe)
Proposed rejecting "generic move builder + subset temperature ladder" at config load
because `SingleSourcePEBuilder` hard-coded `coords_shape[0] = ntemps`; user pushed
back ("a move can use fewer temperatures"). The move itself supported it; one line
in the builder did too. **Pattern**: when library code blocks a valid configuration,
first check whether the limitation is in a thin setup layer (a builder, a shape
argument) and offer the small fix alongside the restriction; lead with the fix
when it is a line or two. State the direction of a rule with its reason, not just
the operator.

## 2026-07-08 — venv mutations
User intercepted my rebuild and ran it themselves. Don't run installs/rebuilds
into the shared erebor_org_setup/.venv unprompted — especially with a live
production run attached to it. Prepare the exact command, hand it to the user
(or ask), unless explicitly told to run it.

## 2026-10-28 — optimise the whole model, not just the user's current config
Proposed a `convolve_window` speedup (precomputed convolved bases) that was exact
only for the instrument-noise-only config the user happened to run; user rejected:
"the fix has to generalize to the galaxy and potentially splines, otherwise what's
the point". **Pattern**: a performance fix to a shared backend must cover every
model component the code supports (galaxy foreground, splines, asymmetric noise,
aliasing). Split explicitly into exact vs approximate paths and say which is which;
never ship a fast path that silently only works for one configuration.

## 2026-11-06 — a time anchor is a convention for every source, not a column shift
Planned `time_reference: data_start` only for blocks with a time column (MBH t_c),
reasoning that EMRI phases "can't be shifted". User: it must apply to every block
with a waveform_t0 — for EMRIs the time the reference phases refer to is set by the
search settings. **Pattern**: a reference-time option declares the frame of the
waveform AND of the block's search results; it is not a parameter transform. Only
externally-anchored inputs (the MOJITO catalogue) need conversion, and where that
is impossible (phases) raise for that input, don't restrict the option.
Follow-up: user then rejected converting catalogue values at all — data_start +
catalog search is a ValueError at load. Prefer refusing an inconsistent combination
over silently converting inputs between conventions.

## 2026-11-09 — scope multi-phase ports in rounds
Correction: on the SOBBH YAML port I put the whole 8-step port up for approval at once; the user approved only
steps 1–2 and changed a design point (GWTF_generator basis) for a later step. Rule: for multi-phase
work, propose the first round's scope as the actionable plan and list later phases only as a
sketch, expecting the user to steer them before they are planned in detail.

## 2026-11-09 — show the plan before acting on it
Correction: after a rejected ExitPlanMode I rewrote the plan and acted on an approval the user never saw
("you didnt show me the plan"). Rule: every plan revision goes through a visible ExitPlanMode on the
current plan file, and nothing (not even memory/todo bookkeeping) is written until that approval.

## 2026-11-11 — read the data's shape, don't impose one
Correction: for the catalogue-shaped .h5 search file I planned to always add a samples axis
(`values[:, None, :]`); the user pointed out the file may already carry the right shape. Rule: when a new
input format joins an existing reader, route it through the reader's existing dimensionality rules (here
the .npy 1-D/2-D/3-D normalization, where 2-D means sources in a file but samples in a per-source folder)
instead of a fixed reshape, and test each layout against the format it must match.

## 2026-11-12 — read the library call before building a design rule on it (permuting swap)
Planned to cut the inner-chain series at the walker-permuting temperature swap, assuming it reshuffles which
walker sits in which slot. User: "I thought that permuting the walker there was helpful to decorrelate them".
eryn's `temperature_swaps(permute_here=True)` only randomises the *pairing* across rungs; a cold slot changes only
on an accepted swap, which is a proper transition. **Pattern**: same as 2026-10-28 — a rule that rests on what a
library call does needs that call read (or run) first, also when the rule is "defensive".

## 2026-11-12 — reuse eryn's utilities before writing a new one (ACT)
Planned a new `sampling/act.py` estimator; user: "use the utilities already in eryn as much as possible to prevent
code duplication". **Pattern**: for sampler diagnostics (ACF, ACT, stopping rules) start from `eryn.utils`; add only
what it measurably cannot do, and show the measurement.
