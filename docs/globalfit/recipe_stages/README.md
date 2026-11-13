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

### Anything else of the move, per stage (`attributes`)

A stage can set any attribute of the block's PE move, or of something the move holds, by its path:

```yaml
stages:
  tune: {flow_weight: 0.5, attributes: {flow_buffer_thin: 10}}
  pe:
    flow_weight: 0.5
    attributes:
      "moves[-1].executor.tune": {max_buffer_samples: null, train_every: 10}
```

| the path ends on | what is done | example |
|---|---|---|
| an attribute | it is set; it must exist already, so a misspelt name is an error | `flow_buffer_thin: 10`, `moves[0].a: 1.5` |
| an index | that item is set | `betas[1]: 0.7` |
| a method | it is called with the value: a dict as keyword arguments, a list as arguments, anything else as its one argument | `"moves[-1].executor.tune": {train_every: 10}` |

- A path goes through attribute names and integer indices (`name`, `name[i]`, `name.sub`, negative indices too).
  Quote a path with brackets in YAML.
- It is set when the stage starts and **stays set in the later stages**. A restart into a later stage sets the
  earlier stages' attributes again, in the stages' order, so the move ends up as in a run through all of them.
- The syntax is checked when the config loads. Whether the path exists is checked on the built moves when the
  recipe's steps are added, before the first iteration, for every stage.
- It works for any source block's PE move, also one without `apply_settings` (gb), which cannot take
  `num_prop_repeats`, `inner_move_weights` or `flow_weight` per stage.
- The flow move is the last of the PE move's inner moves (`moves[-1]`), and its `executor` is the flow trainer.

**The flow trainer in the stage that keeps its samples.** The trainer refits the flow on a buffer of the chain's
recent samples (`flow.trainer.max_buffer_samples`, a few iterations' worth), so the proposal follows the chain
closely. That is what burn-in needs; in the stage whose samples are kept, a proposal that depends on the latest
samples is no longer a fixed proposal. `executor.tune` (eryn, `TrainerExecutor.tune`) changes the trainer while
it runs: `max_buffer_samples: null` lets the buffer grow with the run and `train_every: 10` refits on every tenth
batch only, so each refit changes the flow less than the one before. Tunable: `max_buffer_samples`, `train_every`,
`min_train_samples`, `epochs_per_round`, `refit_transform_every`, `fit_kwargs`, `save_every`, `diagnostics_every`,
`plot_corner`. A restarted run starts its trainer with an empty buffer, as before.

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

## Autocorrelation at runtime: adaptive repeats and a convergence stop

Two autocorrelation times (ACT) are measured while the run goes, one for each use:

| | Measured on | Unit | Used for |
|---|---|---|---|
| inner ACT | the cold chain of a block at every proposal repeat, over the last calls of its move put end to end | repeats | the block's `num_prop_repeats` |
| outer ACT | the stored cold chain of the last stage | iterations | stopping the run |

Both are the ACT of the same chain; the inner one sees it at every repeat instead of once per iteration, which
is what tells how many repeats an iteration needs.

### Adaptive repeats (`blocks/<mbh|emri|sobbh>.yaml`)

```yaml
num_prop_repeats: 50                             # the starting value
adaptive_repeats: {min: 10, max: 500}            # its presence switches the adapting on
```

| key | default | |
|---|---|---|
| `min`, `max` | 10, 500 | bounds of the repeats |
| `per_act` | 2 | inner ACTs per iteration |
| `every` | 10 | iterations of the block's move between updates |
| `window` | 20 | iterations of the block's move the ACT is measured over |
| `max_step` | 2 | largest factor the repeats change by in one update |

- **Stages before the last** tune: every `every` iterations of the move, the repeats go to `per_act` inner ACTs.
- **The last stage**, whose samples are kept, runs with the value reached, fixed.
- **Not resolved:** the ACT needs a window of about ten ACTs (`window` iterations x repeats), on a chain that has
  stopped drifting. Until then the log says "not resolved" and the repeats grow by `max_step`, which also
  lengthens the window. During burn-in from a tight start this is the normal reading; give the burn-in stage its
  own `num_prop_repeats` (below) to keep the repeats where they are until the ensemble has spread.
- A stage whose `stages` entry sets `num_prop_repeats` runs with that number, and nothing is tuned in it.
- The tuned values are stored in the backend (`GFHDFBackend.recipe_repeats`), so a restart resumes them.
- If the ACT is still not resolved at `max` repeats, a warning is logged and the last stage runs with `max`. An
  ACT above `max / per_act` gives `max` as well.

Only the single-source special moves measure an inner ACT (`ResidualAddOneRemoveOneMove.inner_act`); the gb and
psd moves keep fixed repeats. With several sources in the block, the slowest source sets the repeats. The inner
ACT is measured within each walker: walkers that sit in different modes without ever crossing do not show in it.

The inner ACT of an ensemble move can be long: a stretch move on a 12-dimensional Gaussian has an ACT of about
260 repeats with 14 walkers and 165 with 48. With `max` below `per_act` times that, the repeats end at `max`.

Checks when the config loads: the move must measure an inner ACT; `min <= num_prop_repeats <= max`;
`window >= every`; some stage before the last must leave `num_prop_repeats` to the tuning.

### Convergence stop (`global.yaml`, last stage)

```yaml
stages:
  - {name: burn-in, iterations: 200}
  - {name: tune, iterations: 300}
  - {name: pe, target_ess: 10000, check_every: 50}
```

Every `check_every` iterations (default 50) the last stage reads its cold chain from the backend and, per block,
takes the largest ACT over the block's series and `ESS = nwalkers * iterations / ACT`. The run stops when the
smallest ESS reaches `target_ess` and every ACT is reliable (`iterations > 50 * ACT`, eryn's
`AutoCorrelationStop` rule). `run.num_iterations` stays the hard limit. Without `target_ess` the last stage runs to
it, as before.

| block | gates the stop | logged only |
|---|---|---|
| mbh, emri, sobbh, psd, galfor | every sampled parameter of every active leaf (periodic ones as sin and cos) | |
| gb | the cold-chain number of sources | the per-band numbers of sources |

What does not hold the run: leaf slots that are not active, parameters with the same value in every walker and
iteration, and a gb block whose number of sources never changes (logged as "not gating"). What does: a walker that
never moves in some parameter (its ACT is not defined, and an undefined ACT is never read as converged).

Each check logs, per block, the ACT, the ESS, whether it is reliable, and the block's repeats.

### The estimates

- Outer: eryn's `get_acf` per walker, averaged over the walkers, with Sokal's automatic window
  (`lisatools.sampling.utility.cold_chain_act`). `eryn.utils.get_integrated_act` sums a fixed 50 lags, so it cannot
  return more than about 100.
- Inner: from the variogram, half the mean squared difference at each lag (`variogram`, `variogram_act`). Its
  plateau is the variance, and the ACT is twice the lag at which it reaches `1 - 1/e` of the plateau (exact for
  an exponential ACF). It needs no mean: `get_acf` subtracts each walker's own mean, which on a series of 10 ACTs
  halves the ACT, and a mean pooled over the walkers fails because each walker carries its own state of the other
  blocks. A variogram still rising in the second quarter of the window (a drifting chain) reads "not resolved".
  Calibrated on AR(1) chains (within 10% from ten ACTs) and on the stretch-move ensembles above (10 to 30% high).

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

A recipe file passes a stage's attributes as `move_attributes=l2d.move_attributes(stage)` to its step (in force
in the order the block lists its `stages`); a block that sets stage attributes must have them in some step.

A recipe file gets the same from `recipe.RepeatController` (`repeat_controllers=` of a step, with
`l2d.adaptive_repeats()` for the blocks' settings; `kept_repeats=` on the step that keeps its samples) and
`recipe.ConvergenceRecipeStep` (`monitors=` from `build_recipe.stage_monitors`). A block with `adaptive_repeats`
must have a controller in some step.

## Code

- `src/lisatools/globalfit/stock/l2d/build_recipe.py`: `Alternative`, `BranchMoves`, `JointPEMoves`,
  `StockRecipe.add_steps`, `gb_moves` (builds the gb `Alternative`s).
- `src/lisatools/globalfit/stock/l2d/blocks/gb.py`: `ExtraMoveSpec` (`weight`, `stage_weights`, `share`) and the
  per-stage share check.
- `src/lisatools/globalfit/stock/l2d/io.py`: `_block_stages_are_recipe_stages` (stage names of `stages` and
  `stage_weights`).

- `src/lisatools/globalfit/recipe.py`: `RepeatController`, `ConvergenceRecipeStep`, `ChainMonitor`,
  `GBCountMonitor`; `src/lisatools/sampling/utility.py`: `cold_chain_act`, `variogram`, `variogram_act`.

## Tests

- `tests/test_l2d_recipe.py::StockRecipeTest`: shares per stage, shared instances across stages, several blocks
  taking turns, move settings per stage.
- `tests/test_l2d_stages.py::RecipeStepTest::test_turns_and_settings_together`: real `GFCombineMove`s through
  `setup_run`, with the stage settings reaching the move inside both combined moves.
- `tests/test_l2d_gb.py::GBSchemaTest::test_extra_moves`: the `stage_weights` schema and its checks.
- `tests/test_l2d_stages.py`: `ActTest` (the estimates on AR(1) chains), `MoveInnerActTest`, `RepeatControllerTest`,
  `AdaptiveStagesTest` (tuning, freezing, restarts), `ConvergenceStepTest`, `AdaptiveRepeatsSchemaTest`.
- `tests/test_l2d_stages.py`: `SetAttributeTest` (paths, methods, errors, dry run), `StageAttributesTest` (set at
  stage start, restart, paths checked when the steps are built), `StageAttributesSchemaTest`.
