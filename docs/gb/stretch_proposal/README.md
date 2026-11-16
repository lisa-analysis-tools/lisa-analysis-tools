# Exact group stretch of the galactic-binary in-model move

The in-model move of the galactic-binary (GB) block of the global fit, in `GBSpecialBase`
(`src/lisatools/globalfit/moves/gbspecialmove.py`), mixes two proposals. In every repeat round one
coin, drawn with probability `stretch_probability` for the whole batch of sources, chooses between
the regularised Fisher jump (see [`../fisher_proposal/README.md`](../fisher_proposal/README.md)) and a
group stretch. In November 2026 the stretch's choice of friends was replaced by an exact rule. The
block now runs in two halves by walker parity. Each half draws its friends from tables of the other
parity's current sources at its own temperature, the window of candidate friends follows the source's
current frequency, and a reverse check rejects a proposal whose friend the reverse move could not draw.
At `stretch_probability = 0` the move runs as before.

| Document | Content |
| --- | --- |
| `README.md` | The stretch, the conditions on its friends, why the earlier rule broke them, the exact rule and why it is exact, the cost, where the code lives and how to switch it on. |
| [`results.md`](results.md) | The measurements: the bias of the earlier rule, the exactness of the new one, the burn-in from the production search's candidates, the cost, and the checks of the port. |

The checks behind `results.md` ran outside this repository, in a local test harness that runs the
production move classes on a DFFR likelihood against independent reference chains, and on a group of
11 production bands of the 714-day Mojito data. Author: Coen Rondeel.

## The stretch

A block is one source's turn in a propose: `num_repeat_proposals` in-model repeats (50 in the
global-fit settings). The sampling coordinates are
$x = (\ln A, f_0, \dot f, \phi_0, \cos\iota, \psi, \alpha, \sin\delta)$, with $f_0$ in mHz. A stretch
round takes, for each source of the batch, one friend $c$: a copy of a source in another walker. Eryn's
`GroupStretchMove.get_proposal` then proposes

$$x' = c + z\,(x - c), \qquad g(z) \propto 1/\sqrt{z} \ \text{ on } \ [1/a,\ a], \quad a = 2,$$

moves the periodic parameters on their circles, and returns the factor $(d - 1)\ln z$ with $d = 8$
(Goodman and Weare 2010). For a friend drawn independently of $x$ the repeat is accepted with
probability

$$\min\left(1,\ z^{d-1}\,\frac{\pi(x')\, \mathcal{L}(x')^{\beta}}{\pi(x)\, \mathcal{L}(x)^{\beta}}\right),$$

with $\pi$ the prior, $\mathcal{L}$ the likelihood and $\beta$ the inverse temperature of the source's
cell. The GB move hands the friends to `get_proposal` through `find_friends`.

## What the friends must satisfy

Let $w(c \mid x)$ be the probability of drawing the friend $c$ for a source at $x$. The stretch keeps
the target distribution only if:

1. the acceptance carries the ratio $w(c \mid x') / w(c \mid x)$ whenever the friend choice depends on
   $x$;
2. the friends hold none of the moving walker's own states;
3. the friends stay fixed while the walker moves, and the walkers that supply them do not move at the
   same time against it. This is the complementary-ensemble argument of the parallel stretch move
   (Foreman-Mackey et al. 2013): one half of the ensemble moves against the other, fixed half.

The rule used before November 2026 broke all three. It built one table per propose from the cold
sources of all walkers, sorted by $f_0$, and gave each source a window of `nfriends` rows around its
$f_0$ at the start of the propose. The window stayed there for the 50 repeats and the reverse window
was never checked (condition 1); the table held the moving walker's own cold copy (condition 2); and
every walker moved at once against a table that its own start position had helped to build
(condition 3). Hot chains drew their friends from the cold table. The bias is measurable (see
`results.md`), so the global-fit settings files kept the stretch at probability 0.

## The exact rule

1. **Walker-parity halves.** At `stretch_probability > 0` every block of repeats runs twice: first for
   the picked sources of the even walkers, then for those of the odd walkers.
2. **Tables of the other parity.** Before each half the move builds `FriendTables` from the band
   sorter's live rows of the other parity as they are at that moment: one table per temperature,
   sorted by $f_0$. The second half therefore sees the first half's accepted moves. A temperature
   with fewer than $\max(2, W)$ rows draws from the cold table; when any source of a block has no table
   at all, its rounds take the Fisher jump without drawing the coin.
3. **The window at the current frequency.** At every repeat the window of $W$ ranks is placed at the
   source's current $f_0$ in the table of its temperature,
   $s(f_0) = \mathrm{clip}\left(\mathrm{rank}(f_0) - \lfloor W/2 \rfloor,\ 0,\ n - W\right)$, with
   $\mathrm{rank}(f_0)$ the number of rows at or below $f_0$ and $n$ the table's length. The friend
   is the row of a uniform rank in $[s(f_0),\ s(f_0) + W)$.
4. **The reverse check.** Once $x'$ is formed, the window is placed at the proposed $f_0'$ in the same
   table. When the friend's rank lies outside it, the factor is $-\infty$: the proposal is rejected,
   and the row skips the likelihood call.

Why this is exact. The tables do not change while a half runs, so the window of a position depends on
nothing but its $f_0$: $w(c \mid x) = \mathbb{1}[c \in \mathcal{W}(x)] / W$, with $\mathcal{W}(x)$ the
window placed at the $f_0$ of $x$. The reverse move from $x'$ with the same friend and $1/z$ draws $c$
with $w(c \mid x') = \mathbb{1}[c \in \mathcal{W}(x')] / W$. Both windows hold $W$ rows, so the ratio
of condition 1 is the indicator $\mathbb{1}[c \in \mathcal{W}(x')]$, and the acceptance becomes

$$\min\left(1,\ \mathbb{1}[c \in \mathcal{W}(x')]\; z^{d-1}\,\frac{\pi(x')\, \mathcal{L}(x')^{\beta}}{\pi(x)\, \mathcal{L}(x)^{\beta}}\right).$$

The other parity's walkers do not move during the half, so condition 3 holds; the moving walker is not
in the other parity, so condition 2 holds without an extra test. Each half leaves the target of its
walkers, given the other half, invariant, and the two halves together leave the joint target of all
walkers invariant.

The window width is `nfriends // 2`. The recipe sets `nfriends` to the number of walkers, and one
parity holds `nwalkers / 2` copies of every resolved source, so a window spans about one source's
copies (16 at 32 walkers). A wider window takes friends from neighbouring sources: on a benchmark
state a window of 32 took about half its friends from other sources.

Hot chains draw their friends from their own temperature. A hot chain is wider than the cold
ensemble, so friends from colder copies would give it steps that are too short.

The rest of the move is unchanged: the coin is drawn once per repeat round for the whole batch, and the
Fisher factor of a block is built before its repeats at every probability.

## Cost

At probability above 0 every block runs as two halves at half the batch, so the number of repeat
rounds doubles. A repeat round has a fixed cost (the likelihood call's launch and the block's work
around it) that does not shrink with the batch, and at the batch sizes of the GB block this fixed part
dominates. On the benchmark group of 11 bands (32 walkers, 12 temperatures, one A100) a propose takes
1.27 times as long at probability 0.5 as at probability 0; the table builds take 0.1 percent of it.
The numbers are in `results.md`.

## Where the pieces live

| Code | Role |
| --- | --- |
| `gbbands.py`: `FriendTables` | The tables of one walker parity, one per temperature: the table of each source (`table_index`), the window start (`window_start`), the draw (`draw`), the reverse check (`in_window`) and a whole window (`window_rows`). |
| `GBSpecialBase._run_in_model_repeats` | At `stretch_probability > 0`, hands the block to `_run_in_model_halves`; each half runs the block's lines on its sources. |
| `GBSpecialBase._run_in_model_halves` | The split by walker parity and the table build before each half. |
| `GBSpecialBase.in_model_proposal` | The coin; inside a half, the stretch with the friend at the current $f_0$ and the reverse check; the Fisher jump otherwise. |
| `GBSpecialBase.find_friends` | Hands the friends to Eryn's `GroupStretchMove.get_proposal`. |
| `stock/erebor.py`: `GBMoveSettings.stretch_probability` | The probability of a stretch round, in the `moves` group of `GBSettings`; its default is 0. |
| `recipe.py`: `prepare_gb_moves`, `build_gb_moves` | `prepare_gb_moves` sets `nfriends` to the number of walkers; `build_gb_moves` passes the probability of the `moves` group to every GB move. |

The table of the earlier rule (`BandSorter.build_friend_index`, `draw_friends`) is still built at the
start of a propose at probability above 0, which costs one sort per propose; the exact rule does not
use it.

## How to switch it on

Set `stretch_probability` in the `GBMoveSettings` of a settings file (the `moves` group of
`GBSettings`), or among the `pe_move` knobs of the gb block in a YAML run. Its default is 0 in both,
and the global-fit settings files and YAML runs set 0.

The measurements favour the stretch in the burn-in phase only (see `results.md`). From the production
search's candidates, the stretch at 0.5 doubles the share of the missed sources that converge within
500 iterations. Near the posterior, the Fisher jump alone is the cheapest per independent sample. The
recommended use is therefore 0.5 in the burn-in phase of the parameter-estimation steps and 0 in the
kept steps. The recipe has no burn-in phase of its own, so a settings file splits parameter estimation
into two recipe steps on the same move instance, each setting `stretch_probability` (and, if wanted,
`num_stationary_fim_steps`) on its moves when it starts. The move reads both attributes at every block,
so the switch takes effect at once.

## Notes

- The frequency column (sampling column 1) must not be periodic: the reverse check reads the proposed
  $f_0$ before the periodic columns are wrapped. `_run_in_model_halves` raises a `ValueError` otherwise.
- The tables are built and read on the device of the band sorter, as the earlier table was. The move
  was not tested on several GPUs.
- The planned split/merge move of the GB block reads its covariance from the same tables
  (`window_rows`).

## References

- J. Goodman and J. Weare, Ensemble samplers with affine invariance, Communications in Applied
  Mathematics and Computational Science 5 (2010) 65-80.
- D. Foreman-Mackey, D. W. Hogg, D. Lang and J. Goodman, emcee: the MCMC hammer, Publications of the
  Astronomical Society of the Pacific 125 (2013) 306-312.
