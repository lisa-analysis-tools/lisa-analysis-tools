# F-statistic RJ move of the galactic-binary block

`GBFStatRJMove` (`src/lisatools/globalfit/moves/gbspecialmove.py`) is a reversible-jump (RJ) move of the GB block
whose births come from a search of the data. It runs beside the prior RJ move (`GBSpecialRJPriorMove`), which covers
the whole prior, and proposes sources where its search found them. It runs on the STFT domain with the DFFR response
only and replaces the FD serial search (`GBSpecialRJSerialSearchMCMC` stays as an alias). The search itself is a class
of its own in the same file, `FStatResidualSearch`, which the move builds for itself from its settings at its first
search; the move keeps the schedule, the choice of the bands and the proposal. The evidence behind its
settings and the checks of the port are in [`results.md`](results.md). Author: Coen Rondeel.

## How it works

A search (`FStatResidualSearch.run`) runs in the move's `setup()`, on the residual of the walker with the highest
likelihood, in every interior band that is not shut off.

1. **Stage 1.** The F-statistic $F = \tfrac12 N^\top M^{-1} N$ maximizes the likelihood over the four amplitude
   parameters ($A$, $\iota$, $\psi$, $\phi_0$) analytically. $N$ and $M$ come from two templates per row
   (`get_fstat_terms_stft`, GBGPU). One Eryn sampler per band (`nsamplers`) samples $(f_0, \dot f, \lambda, \sin\beta)$
   with $F$ as the likelihood, on the run's prior restricted to the band.
2. **The gate.** A band passes when its largest stored cold $\sqrt{2F}$ exceeds $\sqrt{\rho_t^2 + 8 - \delta}$. With one
   source in a band the peak of $2F$ lies at $\rho_\mathrm{eff}^2 + 8$, and $\delta$ is the median drop of the best
   stored sample below it. The target $\rho_t$ is `snr_eff_threshold`, divided by the square root of
   `galfor_inflation_factor` below `shutoff_high_frequency_threshold`, the upper edge of the galactic foreground. A band
   that passed none of its last `shutoff_band_iteration` searches is shut off when it lies wholly above that edge or
   wholly below `shutoff_low_frequency_threshold`; a threshold of None turns its shutoff off.
3. **Stage 2.** From the F-statistic maximizers at stage 1's last state, one sampler per passing band samples all eight
   columns with the full likelihood.
4. **The proposal.** One Gaussian mixture per passing band, fitted to the stage-2 cold chain, is the RJ birth proposal
   (`FStatPhaseProposal`). The template is the same at four copies of $(\phi_0, \psi)$ in $[0, 2\pi)^2$, so the mixture
   takes $\theta_\pm = 2\psi \pm \phi_0$, equal at all four copies, in place of the phases, with $\theta_\pm$ and
   $\lambda$ periodic. A draw takes each copy with probability 1/4, and the density of a row is the mixture's density
   at its $\theta_\pm$.

With `search_interval` M above 1 the move searches at every M-th of its proposes and keeps the latest proposal in
between; a move without a proposal searches at once.

## Settings

The move takes one dataclass, `fstat_settings` (a `GBFStatSettings`; None takes the defaults):

| field | default | meaning |
| --- | --- | --- |
| `nwalkers` | 32 | walkers of each search sampler |
| `betas_1`, `burn_1`, `nsteps_1` | [1, 0.5, 0.25, 0.12, 0.05, 0], 500, 300 | stage 1: start ladder (Eryn adapts it), burn-in, stored steps |
| `betas_2`, `burn_2`, `nsteps_2` | $1.2^{-k}$ for $k = 0, \ldots, 10$ and 0.1, 400, 500 | stage 2: 12 temperatures with $T = 10$ on top, burn-in, stored steps |
| `snr_eff_threshold` | 7.5 | the gate's effective-SNR target |
| `galfor_inflation_factor` | 1.3 | the foreground's inflation of the noise PSD |
| `shutoff_high_frequency_threshold` | 6e-3 Hz | the foreground's upper edge; bands wholly above it can be shut off |
| `shutoff_low_frequency_threshold` | 6e-4 Hz | bands wholly below it can be shut off |
| `shutoff_band_iteration` | 2 | searches without a detection before a band in a shutoff range is shut off |
| `search_interval` | 1 | proposes of the move from one search to the next |
| `progress` | True | Eryn's progress bar for each search stage |
| `save_search_records` | `"none"` | what each search saves: `"none"`, `"summary"` or `"full"` |
| `output_dir` | None | folder of the search log and the records; the recipe builders set `<run>_artifacts/fstat_search/` |

## How a run selects the move

The move is opt-in.

- **Settings files:** `GBMoveSettings.pe_move_names` names `"rj_fstat_mcmc"` next to `"rj_prior"` and
  `GBMoveSettings.fstat_settings` holds the settings. The settings file lets the GB moves take turns, one
  `GFCombineMove` per GB move in a weighted step.
- **L2D YAML:** the GB block's `extra_moves` lists the GB moves that take turns with its `pe_move`, each with its share
  of the iterations of each recipe stage (`recipe.kwargs.stages`); the `pe_move` takes the rest of each stage. Every
  stage's step holds one `GFCombineMove` per choice, each with the other blocks' PE moves.

```yaml
extra_moves:
  fstat:                    # the move "gb fstat move"
    class: GBFStatRJMove
    weight: 0.1             # share of the iterations of every stage not in stage_weights
    stage_weights:          # stage name -> share of that stage (0: not in it); names must be recipe stages
      burn-in: 0.3
    kwargs: {fstat_settings: {save_search_records: summary}}
```

## The search memory

The keyword `fstat_search_memory` (default None) hands the move an `FStatSearchMemory`, a host data holder in which the
F-statistic moves of a run share the Gaussian mixtures of their searches. After a search that fitted, this move sets one
fit per usable band as the memory's fits on the residual, replacing the earlier ones (the empty list after a search with
no band to search; a fixed memory refuses the list and the move goes on as before). The split and merge
(`GBFStatSplitMergeRJMove`, also an `extra_moves` class of the GB block) takes the same keyword in place of record files
and builds its search part from the memory at construction and at the start of each propose in which the memory's
version changed: per band one mixture, with two thirds of the weight on the fit on the residual and one third shared per
fit among the other fits (a band without a fit of one side gives its weight to the other). A band whose covariance
cannot be factored is left without a search part, with a warning. Its `apply_settings(fix_search_part=True)` brings the
part up to the memory's version and fixes the memory, which then takes no new fit. A recipe file must make that call on
every GB move that has the method, in its step of the kept iterations: the stock recipe does not, so the YAML builder
refuses a block that would get a memory on it. The builders make one memory per run, in
`<run>_artifacts/fstat_search_memory.npz`, and load it when the run resumes.

While the memory is not fixed (the burn-in), the split and merge also makes the searches that fill it, in its `setup()`
before the search part is rebuilt. It takes the walker with the highest likelihood, as this move does, and the sources
of that walker's cold chain in the interior bands. Around each source it opens a window of plus and minus the reach
window W of the source's band (half the own-source width at the band's centre), cut at the band's edges; windows of one
band that overlap become one. The windows go to `FStatResidualSearch` in calls of up to 40. In each window the first
stage starts from the best rows of a seed grid that is scored on the input of the search: f0 nodes one bin apart, five
fdot values two drift bins apart on the line of constant mid-span frequency through each source, and 192 sky points. The
two stages, the gate and the mixture fit then run as in this move. A window that passes the gate has the best row of its
second stage subtracted and is searched again on a new grid, up to four passes; every pass that passes gives one fit.

Two kinds of fit are made this way. The fit on the raw data is made once per run, at the split and merge's first
propose, on the walker's residual with all its sources added back. The fit on the residual plus the window's sources is
made around every source that lies in no range searched before, while fewer than five searches are counted within a
reach window, on the residual with every source inside those windows added back. At the first propose that is every
window, so both kinds run on the same input. Each kind works on the walker's residual in place and writes the saved copy
back bit for bit at its end. A call of windows that raises is skipped with a warning and its sources are searched again
at the next propose. A fit that the memory's check refuses (a value that is not finite, for instance) is left out with
a warning, in both moves. From the start of the kept iterations the memory is fixed and no search starts.

## Output

Each search writes one line to the run log (`[GB_FSTAT_SEARCH <move name>]`: the bands searched and passing, the outcome
and the times) and a block to `fstat_search.log` in `output_dir`, with one line per passing band. With
`save_search_records` it also saves a compressed `<move name>_search_<n>.npz` there: `"summary"` holds the per-band
values, the mixtures and the times (about 40 kB per search on 43 bands), `"full"` adds the chains (a few MB).

## Code

| piece | place |
| --- | --- |
| the move and its settings | `GBFStatRJMove`, `GBFStatSettings` in `src/lisatools/globalfit/moves/gbspecialmove.py`, with the objects handed to Eryn and the band sorter (`BandMaskTransform`, `FStatSearchLikelihood`, `FStatPhaseProposal`) |
| the search | `FStatResidualSearch` (the steps, the logs and the record) and `FStatSearchBase` (the columns, constants and static math, which the move inherits as well), in the same file |
| periodic mixture columns | `vec_fit_gmm_min_bic` in `src/lisatools/sampling/gmm.py`, `FullGaussianMixtureModel` in `src/lisatools/sampling/prior.py` |
| the F-statistic terms | `STFTGBComputations.get_fstat_terms_stft` (GBGPU) |
| the recipes | `build_gb_moves` in `src/lisatools/globalfit/recipe.py`; the GB block and the stock recipe in `src/lisatools/globalfit/stock/l2d/` |
