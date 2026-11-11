# Recipe stages and moves that take turns (L2D stock recipe)

The L2D stock recipe (`StockRecipe` in `src/lisatools/globalfit/stock/l2d/build_recipe.py`) runs the psd
max-likelihood search, when there is a psd block, and then the joint parameter estimation (PE) in **stages**.
Within a stage, a block may let other moves **take turns** with its PE move: in each iteration it runs either its
PE move or one of those moves. This page describes how the two fit together.

The setup replaces the hand-written combinations of the settings files, e.g.
`mojito_input/run_1_global_fit_settings.py:209-219`:

```python
prior_combined = GFCombineMove(moves=[gb_pe_moves[0], mbh_pe_move, psd_pe_move], share_temperature_control=False)
refit_combined = GFCombineMove(moves=[gb_pe_moves[2], mbh_pe_move, psd_pe_move], share_temperature_control=False)
recipe.add_recipe_component(PERecipeStep(moves=[prior_combined, refit_combined], weights=[0.95, 0.05]), ...)
```

## Background: what the merge changed

The dffr merge (#92) brought in two setups of `StockRecipe.add_steps` that never met:

- **alternating moves**: the gb block's `extra_moves` taking turns with its `pe_move`, but only for two fixed phases
  (`burn-in`, `pe`, from `burn_in_iterations`) and only for one block;
- **recipe stages**: any list of named stages, each with its blocks' move settings, but with one combined move and no
  turns.

The merged code had both bodies, one after the other. The first one read an attribute that no longer existed
(`self.burn_in_iterations`), so every stock recipe failed to build. They are now one loop over the stages, and each
stage carries the turns.

## Configuration

### The stages (`global.yaml`)

```yaml
recipe:
  class: StockRecipe
  kwargs:
    stages:
      - {name: burn-in, iterations: 200}   # IterationCountRecipeStep
      - {name: pe}                         # the last stage: no iterations, runs to run.num_iterations (PERecipeStep)
```

`burn_in_iterations: N` is short for `[{name: burn-in, iterations: N}, {name: pe}]`. Without either, there is one
stage, `pe`. A stage is a recipe step, and its name is stored in the backend, so a restart resumes the stage it
stopped in.

### Per-stage settings of a block's PE move (`blocks/<block>.yaml`)

Unchanged: special PE moves take `num_prop_repeats`, `inner_move_weights` and `flow_weight` per stage.

```yaml
stages:
  burn-in: {num_prop_repeats: 100, inner_move_weights: [0.4, 0.2, 0.2, 0.1], flow_weight: 0.0}
  pe:      {num_prop_repeats: 50}
```

### Moves that take turns (`blocks/gb.yaml`, `extra_moves`)

```yaml
extra_moves:
  fstat:                    # the move "gb fstat move"
    class: GBFStatRJMove
    weight: 0.05            # its share of the iterations of every stage not in stage_weights
    stage_weights:          # stage name -> its share of that stage (0: not in that stage)
      burn-in: 0.3
    kwargs: {fstat_settings: {save_search_records: summary}}
```

The `pe_move` takes what the extra moves leave of each stage: here 0.7 of `burn-in` and 0.95 of `pe`.

**Changed key:** `burn_in_weight: x` is now `stage_weights: {burn-in: x}`. A YAML that sets only `weight` is
unaffected.

Checks when the config loads:

- every `stage_weights` key must be a stage of the recipe (as for a block's `stages`);
- in every stage, the extra moves' shares must sum to less than 1, so the `pe_move` keeps a share;
- an extra move must have a share in some stage (`weight > 0` or a positive `stage_weights` entry).

## How the steps are built

`build_block_moves` returns one `BranchMoves(search, pe, alternatives)` per block:

- `pe`: the block's PE moves, run together;
- `alternatives`: the moves that take turns with `pe`, each an `Alternative(move, weight, stage_weights)` with
  `share(stage)`. Empty for blocks without turns.

`JointPEMoves(moves).moves(stage)` gives each stage its step's moves and weights:

1. For each branch, list its choices in the stage: its `pe` moves, with share `1 - sum(alternative shares)`, and each
   alternative, with its share. Choices with a share of 0 are left out.
2. Take every combination of one choice per branch. Each combination is one `GFCombineMove`
   (`share_temperature_control=False`) holding the chosen moves in the branches' order.
3. A combination's weight is the product of its choices' shares. The weights sum to 1.
4. A combination is built once and cached, so the stages that use it share the same `GFCombineMove` instance (and
   its state).

`StockRecipe.add_steps` then adds one step per stage, with those moves and weights and the stage's `move_settings`
(the blocks' `stages`), which `BaseRecipeStep.apply_move_settings` hands to the moves inside every combined move.

### Example

Blocks gb (extra move `refit`, `weight: 0.05`, `stage_weights: {burn-in: 0.5}`), mbh and psd; stages `burn-in`, `pe`:

| stage   | combined moves                                         | weights      |
|---------|--------------------------------------------------------|--------------|
| burn-in | `[gb pe, mbh pe, psd pe]`, `[gb refit, mbh pe, psd pe]` | 0.5, 0.5     |
| pe      | the same two instances                                  | 0.95, 0.05   |

The `pe` row is the settings-file setup quoted at the top.

### Several blocks taking turns

Each branch chooses on its own, so several blocks may take turns. Before the merge this was an error ("only one
block may alternate"). Blocks gb (`b` at 0.25 against `a`) and mbh (`d` at 0.5 against `c`) give four combined
moves, `[a, c]`, `[a, d]`, `[b, c]`, `[b, d]`, with weights 0.375, 0.375, 0.125, 0.125. The number of combined moves
is the product of the branches' choice counts, so keep the alternatives few.

## In a recipe file

A recipe file (`recipe: {file: ..., function: ...}`) can build the same steps:

```python
from lisatools.globalfit.recipe import PERecipeStep
from lisatools.globalfit.stock.l2d.build_recipe import JointPEMoves


def setup_recipe(recipe, engine_info, curr, acs, priors, state, l2d):
    joint = JointPEMoves(l2d.block_moves(engine_info, curr, acs, priors, state))
    moves, weights = joint.moves("pe")
    recipe.add_recipe_component(
        PERecipeStep(moves=moves, weights=weights, move_settings=l2d.move_settings("pe")), name="pe"
    )
```

Two checks keep a recipe file from silently dropping what the blocks set:

- **at load:** when a block sets `extra_moves` or `stages`, the function must take `l2d`;
- **after the function has added its steps, before the first iteration** (`check_recipe_uses_the_blocks`):
  - every `stages` and `stage_weights` key must name one of its steps;
  - every extra move must sit in some step.

## Code

- `src/lisatools/globalfit/stock/l2d/build_recipe.py`: `Alternative`, `BranchMoves`, `JointPEMoves`,
  `StockRecipe.add_steps`, `gb_moves` (builds the gb `Alternative`s).
- `src/lisatools/globalfit/stock/l2d/blocks/gb.py`: `ExtraMoveSpec` (`weight`, `stage_weights`, `share`) and the
  per-stage share check.
- `src/lisatools/globalfit/stock/l2d/io.py`: `_block_stages_are_recipe_stages` (stage names of `stages` and
  `stage_weights`).

## Tests

- `tests/test_l2d_recipe.py::StockRecipeTest`: shares per stage, shared instances across stages, several blocks
  taking turns, move settings per stage.
- `tests/test_l2d_stages.py::RecipeStepTest::test_turns_and_settings_together`: real `GFCombineMove`s through
  `setup_run`, with the stage settings reaching the move inside both combined moves.
- `tests/test_l2d_gb.py::GBSchemaTest::test_extra_moves`: the `stage_weights` schema and its checks.
