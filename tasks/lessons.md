# Lessons

## 2026-07-08 — multi-GPU sync points
Proposed removing the unconditional `synchronize()` before the per-chunk D2H in
`AnalysisContainerArray._compute_group_likelihood`; user corrected: keep it.
**Pattern**: host-consumption barriers (sync before copying per-chunk results
to CPU) are load-bearing and shared across moves (PSD/MBH/EMRI). Optimize by
reducing barrier *frequency* (bigger batches, fewer chunks), never by removing
the barrier itself. Also: `.get()` syncing "should be enough" arguments ignore
off-stream work and shared-code blast radius — weigh risk vs measured gain.

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
