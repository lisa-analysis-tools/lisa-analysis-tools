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
