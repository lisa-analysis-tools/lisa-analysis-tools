# FlowGaussianMove: replace the add/remove Gibbs masks (2026-10-30)

Plan: `~/.claude/plans/graceful-skipping-dijkstra.md`. Branch `feat-flow-gaussian-move` in LAT and Eryn.

## Why
Flow-on-a-subset + stretch-on-the-rest was built as `CombineMove([flow(gibbs=cols), stretch(gibbs=~cols)])`,
which needed column masks, `gibbs_ndim`, `_inds_run` poking and per-mask ring-buffer slicing inside
`ResidualAddOneRemoveOneMove`. Replace it with one Eryn MH move that does the split in `get_proposal`,
so every inner move proposes on all parameters.

## Items
- [x] Eryn `gaussian.py`: `_make_proposal` factored out of `GaussianMove.__init__` (bit-identical vs main)
- [x] Eryn `flow.py`: `_propose_from_flow` factored out of `ConditionalFlowMove.get_proposal` (bit-identical vs main)
- [x] Eryn `flowgaussian.py`: `FlowGaussianMove(ConditionalFlowMove)`; fixed or per-leaf adapted (`adapt_cov`) cov
- [x] Eryn: fix pre-existing crash of `GaussianMove` with a diagonal (1-D) covariance (Cholesky of a vector)
- [x] Eryn tests `tests/test_flow_gaussian_move.py` (16), incl. sampler recovery of a correlated 3-D target
- [x] LAT `addremovemove.py`: Gibbs-mask machinery removed; inner moves with a `gibbs_sampling_setup` rejected
- [x] LAT l2d `FlowSpec.complement` -> `FlowSpec.gaussian` (FlowGaussianMove kwargs); `make_flow_move` builds it
- [x] LAT `mbh.yaml` example, `scripts/run_global.py` summary line
- [x] LAT tests: builder (subset -> FlowGaussianMove, all -> ConditionalFlowMove), spec validation, gibbs rejection

## Review
- Hastings: the flow ignores the Gaussian columns and the Gaussian step is symmetric, so the factor is the
  flow's `log q(A_old) - log q(A_new)` alone. Negative control: zeroing that factor makes the sampler test
  fail (cov 0.82 vs 1.0), with it the target mean/cov are recovered.
- Unfitted flow transform: flow columns held, Gaussian columns still move (base class was a full no-op).
- Periodic: only the Gaussian columns are wrapped; flow values stay as drawn (their log q is in the factor).
- `adapt_cov` uses the rows the add/remove move already submits per leaf (cold chain, thinned); the
  executor receives only the flow columns. Adaptive MH from history, same class as the online flow.
- LAT test baseline in this env (no data backend) already fails 41 l2d/addremove tests on the main checkout;
  per file, the worktree fails a subset of those and nothing new.
- Not done: a GPU run of the MBH block with `parameters` + `gaussian` to look at acceptance.

## Follow-up: random choice of half (`flow_prob`, default 0.5)
- [x] Each active row moves its flow columns with probability `flow_prob`, else its Gaussian columns (a
  state-independent random-scan mixture of two valid kernels); `flow_prob=None` keeps the joint proposal.
  A half that cannot propose yet hands all rows to the other.
- [x] Tests: one half per row, factor only on flow rows, fallbacks, sampler recovery in both modes; the
  negative control (no flow factor) fails in random-choice mode too.
- Toy 3-D target, 32 walkers x 600 steps, flow roughly on the A marginal, 1 Gaussian column:

  | coupling | mode | acceptance | A columns move | B column moves |
  |---|---|---|---|---|
  | weak | joint | 0.172 | 0.172 | 0.172 |
  | weak | flow_prob 0.5 | 0.386 | 0.166 | 0.220 |
  | strong | joint | 0.094 | 0.094 | 0.094 |
  | strong | flow_prob 0.5 | 0.197 | 0.087 | 0.110 |

  Acceptance doubles, but per likelihood evaluation the flow columns move about as often as with the
  joint move; the Gaussian columns gain ~25%. The gain should grow when the Gaussian half rejects
  more (more or multimodal Gaussian columns) -- to be confirmed on the MBH run.

## Follow-up: within-mode covariance for multimodal Gaussian columns (2026-11-02)
Why: the non-flow MBH parameters are multimodal; the marginal covariance spans the modes.
- [x] The add/remove ring buffer keeps the walker axis, `(n_snapshots, nwalkers, ndim)`;
  `ConditionalFlowMove.submit_by_leaf` flattens for the trainer, and `submit` (harvest) keeps the walker axis.
- [x] `FlowGaussianMove` adapts from the last `adapt_window` (5) submits: walker trajectories are cut
  at jumps, which gives a length scale, then walkers are grouped into modes per snapshot by single linkage
  (`adapt_trim` = 10 scales) and the spread about group means is pooled. Periodic columns are wrapped.
- [x] First design (pooled per-walker spread, trimming whole walkers) failed in the toy: with frequent
  jumps every walker hopped, and the estimate came out ~400x too wide. A segment-only version was ~7x too
  narrow (autocorrelation). The final version clusters walkers per snapshot.
- Bimodal toy (B ~ two modes 8 apart, within var 0.04 / 0.02; local move 0.5, reflection 0.15,
  FlowGaussianMove 0.35 starting without cov):

  | estimate | adapted B var | FlowGaussianMove acceptance |
  |---|---|---|
  | marginal (flat rows) | 16.3 / 15.8 | 0.19 |
  | within-mode (walker axis) | 0.041 / 0.019 | 0.36 |

  Both modes sample correctly (mode fraction 0.50, within-mode var right); only efficiency differs.
- [x] Tests: two modes, a walker that jumps, sticky AR(1) walkers jumping every ~5 snapshots (within 20%),
  periodic boundary, window accumulation and reset, flattening for the executor.
- Note: the LAT branch now needs the Eryn branch (3-D ring-buffer rows are flattened by Eryn's `submit_by_leaf`).
