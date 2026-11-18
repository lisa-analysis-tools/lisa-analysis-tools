# F-statistic split and merge move of the galactic-binary block

`GBFStatSplitMergeRJMove` (`src/lisatools/globalfit/moves/gbspecialmove.py`) is a reversible-jump (RJ) move of the GB
block that splits one source of a band into two and merges two sources into one. It exists for a state that a search
can leave behind: one loud source held by several weaker templates. A birth, a death or an in-model step each lowers
the likelihood from such a state, so the prior RJ move (`GBSpecialRJPriorMove`) does not leave it; a merge that refits
the kept source in the same step can. The move runs beside the prior RJ move, on the STFT domain with the DFFR response
only. The evidence behind its constants and its tests are in [`results.md`](results.md). Author: Coen Rondeel.

## How it works

The move keeps the pass of the prior RJ move (the band sorter, the bands in two halves of alternating parity, the
in-model repeats, the tempering) and replaces the birth and the death by a split and a merge. A cell is one
temperature, walker and band. One alive source of the cell is picked as the primary: the parent of a split and the
kept source of a merge, so that its choice cancels in the acceptance.

1. **The direction.** A split is possible when the cell has a dead slot, a merge when the primary has a partner within
   the window $W$ in frequency. When both are possible each is drawn with probability 1/2.
2. **A split keeps the parent's position.** Child a keeps the parent's intrinsic point $\xi_1$ (frequency, frequency
   derivative, sky). The intrinsic point $\xi_2$ of child b is drawn from a mixture of three parts,
   $q_c = w_s q_s + w_l q_l + w_p q_p$: the components of the band's search fits whose centre lies within $W$ of the
   parent (the search part), a Gaussian at the parent (the local part), and the prior with the frequency uniform
   within $W$ (the prior part). A draw beyond $W$ or outside the band is a rejection.
3. **The amplitudes come from the F-statistic.** Let $r_B$ be the residual without the parent. At fixed intrinsic
   points the likelihood is quadratic in the four amplitudes of each source, so for the pair
   $\ln L(r_B - h_1 - h_2) - \ln L(r_B) = \mathcal{N}^\top U - \tfrac12 U^\top \mathcal{M} U$, with $U$ the eight
   amplitudes. $\mathcal{N}$ and $\mathcal{M}$ come from one F-statistic row per source and four swap rows for the
   cross block. The move draws $U \sim \mathcal{N}(\mu, \Lambda^{-1})$ with $\Lambda = \beta\mathcal{M} + \varepsilon I$
   and $\mu = \Lambda^{-1}\beta\mathcal{N}$ at the temperature $\beta$ of the cell, the tempered posterior of the
   amplitudes given the two positions. The ridge $\varepsilon$ keeps $\Lambda$ regular when the templates nearly
   coincide.
4. **A merge keeps the primary's position.** The partner is drawn among the sources within $W$ with a Gaussian weight
   of width $W/2$ in frequency. The merged source keeps the primary's intrinsic point and takes four amplitudes from
   the same tempered Gaussian, now of its own F-statistic row on the residual without the pair. A merge is therefore a
   death of the partner with a refit of the primary's amplitudes in one step.
5. **The acceptance.** With $k$ alive sources among the walker's $N$ slots, $m_c$ dead slots in the cell and $P_b$
   the share of the prior's frequency range in the band, a split is accepted with probability $\min(1, e^{A})$,

   $$A = \beta\,\Delta\ln L + \ln\frac{\pi(\lambda_1)\,\pi(\lambda_2)}{\pi(\lambda_*)} + \ln\frac{k+1}{N-k} - \ln P_b
   + \ln m_c + \ln q_\mathrm{partner} + \ln\frac{p_\mathrm{merge}}{p_\mathrm{split}} + \ln\frac{q_\mathrm{merge}}{q_\mathrm{split}}.$$

   $q_\mathrm{split}$ is the density of the companion's point and of the eight amplitudes, $q_\mathrm{merge}$ that of
   the parent's four amplitudes under the reverse merge, and $q_\mathrm{partner}$ the probability that the reverse
   merge picks child b. A merge takes the negative of its reverse split's $A$. $\Delta\ln L$ is the full likelihood
   change from the band buffer, not the quadratic form.

**Why it is exact.** Every density is a function of the state with fewer sources and of the residual without the
pair, which both directions hold: $\xi_1$, the best-fit $2F(\xi_1; r_B)$, $\beta$ and the fixed search part. The
positions are not transformed, so the only Jacobians are those of the amplitude maps. A split beyond $W$ is rejected
and a merge pairs only sources within $W$, so every kept split has a reverse and the reverse of every merge is a kept
split. The prior part keeps $q_c$ positive over the whole window.

**The local part** has the covariance $\Sigma_l = (\rho_\mathrm{eff}^2\, g + P)^{-1}$ with
$\rho_\mathrm{eff}^2 = \min(\beta\, 2F,\ \rho_\mathrm{ref}^2)$, $P$ the precision of the prior boxes and $g$ an analytic
metric per unit squared SNR: the chirp block of $(f_0, \dot f)$ at the catalogue epoch and the orbital Doppler block
of the sky, $\tfrac12 (2\pi f R)^2 (I - z z^\top)$ with $R$ the light time of one astronomical unit and $z$ the
direction of the ecliptic pole, plus a floor for the antenna pattern. No Fisher matrix of a template is computed.

**The window** $W$ is set per band: half the frequency width of one source at the band's centre (the orbital Doppler
swing, the drift of a system of 0.3 solar masses over the data span, and the spectral window's floor). In a run from
8 to 30 mHz it grows from 63 bins at 8 mHz to 183 bins near 18 mHz.

## The search part

The components of $q_s$ come from one of two places. With `record_files` the move reads the stored mixtures of earlier
searches, which stay fixed. With the keyword `fstat_search_memory` it reads an `FStatSearchMemory`, which it shares
with `GBFStatRJMove` and fills itself while the memory is not fixed: the move searches windows of $\pm W$ around the
sources of the best cold walker with `FStatResidualSearch`, from a seed grid. The weights of the fits, the windows,
the seed grid and the passes are described in ["The search memory"](../fstat_move/README.md#the-search-memory). The
search part must be fixed before the kept iterations start, since a proposal that follows the chain is not a valid
kernel for samples that are kept.

## Settings

The move takes one dataclass, `fstat_split_merge_settings` (a `GBFStatSplitMergeSettings`; None takes the defaults):

| field | default | meaning |
| --- | --- | --- |
| `record_files` | None | paths or glob patterns of search records; several files per band are merged with equal weight per file. None with no memory runs the move without a search part |

Every value of the proposal is a class constant, since only a test would change it:

| constant | value | meaning |
| --- | --- | --- |
| `COMPANION_WEIGHTS` | 0.50, 0.35, 0.15 | $w_s, w_l, w_p$; with no search component within $W$, $w_s$ goes to the other two in proportion |
| `RHO_REF` | 4.5 | the tempered SNR whose width the local part takes when the parent's own is larger |
| `ANTENNA_FLOOR` | 1 per rad$^2$ | the floor of the sky metric, which alone constrains the direction of the ecliptic pole near the ecliptic plane |
| `RIDGE_FRACTION` | 1e-3 | $\varepsilon$ as a fraction of the mean diagonal of $\beta\mathcal{M}$ |
| `FIDUCIAL_CHIRP_MASS_MSUN` | 0.3 | the chirp mass of the drift in the window rule |
| `RESIDUAL_FIT_WEIGHT`, `OTHER_FITS_WEIGHT` | 2/3, 1/3 | the weights of the memory's fits in a band's search part |

## How a run selects the move

The move is opt-in and needs the STFT domain with DFFR and the prior as `rj_proposal_distribution`.

- **Settings files:** `GBMoveSettings.pe_move_names` names `"rj_fstat_split_merge"` next to `"rj_prior"`, and
  `GBMoveSettings.fstat_split_merge_settings` holds the settings.
- **L2D YAML:** the GB block's `extra_moves` lists the move with its share of each recipe stage. Without record files
  the builders make one search memory per run, and the stock recipe asks that the search part is fixed in the last
  stage:

```yaml
extra_moves:
  split_merge:
    class: GBFStatSplitMergeRJMove
    weight: 0.3                           # share of the iterations of every stage not in stage_weights
    stage_weights: {burn-in: 0.4}
    stages:
      pe: {attributes: {apply_settings: {fix_search_part: true}}}
```

## Output

The move writes to the run log under `[GB_FSTAT_SPLIT_MERGE <move name>]`: at construction the data span and the
range of $W$; at each rebuild of the search part its fits, bands and components; per propose the accepted share of the
splits and merges, in the cold chain and over all temperatures, and the counts of every reason for no proposal and of
the companions by part; and per search the windows, calls, passes, fits, seconds and the walker's ln L before and
after. The memory is saved as `<run>_artifacts/fstat_search_memory.npz` and loaded when the run resumes.

## Code

| piece | place |
| --- | --- |
| the move, its settings and its static math | `GBFStatSplitMergeRJMove`, `GBFStatSplitMergeSettings` in `src/lisatools/globalfit/moves/gbspecialmove.py` |
| the cell helpers of a split and merge move (the pass, the partner kernel, the bookkeeping terms) | `GBSplitMergeCellsMixin`, same file |
| the search and the memory | `FStatResidualSearch`, `FStatSearchMemory`, same file |
| the F-statistic rows and the pair products | `get_fstat_terms` and `get_pair_products` of the STFT band engine in `gb_likelihood.py` (GBGPU), reached through `SubBandBuffer` in `src/lisatools/globalfit/moves/gbbands.py` |
| the recipes | `build_gb_moves` and `make_fstat_search_memory` in `src/lisatools/globalfit/recipe.py`; the GB block and the stock recipe in `src/lisatools/globalfit/stock/l2d/` |
