# Task: merge the alternating PE moves with the recipe stages (l2d `build_recipe.py`)

The dffr merge (#92) left `StockRecipe.add_steps` with two bodies: the alternating one (gb `extra_moves` taking
turns with the `pe_move`, burn-in/pe only, reading the removed `self.burn_in_iterations`) and the stages loop (one
combined move, no turns). Goal: one setup that mimics `mojito_input/run_1_global_fit_settings.py:209-219`
(prior/refit combined moves at 0.95/0.05 with the mbh and psd moves) per stage, and generalizes.

## Plan
- [x] `Alternative(move, weight, stage_weights)` with `share(stage)`; `BranchMoves.alternatives` holds only the moves
      taking turns, the branch's `pe` moves take the rest of each stage.
- [x] `JointPEMoves.moves(stage)`: product over branches of their choices with a share in the stage, one cached
      `GFCombineMove` per combination (stages share instances), weights = product of shares. Lifts "one block only".
- [x] `StockRecipe.add_steps`: one stages loop, each step with the weights and the stage's `move_settings`.
- [x] gb `ExtraMoveSpec`: `burn_in_weight` -> `stage_weights`; pe_move share > 0 checked per stage.
- [x] io: `extra_moves.*.stage_weights` keys must be recipe stages.
- [x] Tests (`test_l2d_recipe`, `test_l2d_gb`, `test_l2d_stages` with real `GFCombineMove` + `setup_run`); docs.

## Review
- All 11 `tests/test_l2d_*.py`, baseline (HEAD) vs this branch: 10 recipe/stage tests fixed, no new failures; the
  68 other failures are identical at HEAD (io "global.yaml must not define 'blocks'", move-build stubs, gb build).
- Run against the venv's stale compiled backends with dffr's `KinematicsTableWrap{CPU,GPU}` stubbed in-process:
  the venv backends need a rebuild before a real run.
