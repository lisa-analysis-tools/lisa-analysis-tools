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
