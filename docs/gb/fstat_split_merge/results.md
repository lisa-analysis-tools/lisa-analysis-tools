# F-statistic split and merge move: results

The tests of `GBFStatSplitMergeRJMove` and the numbers behind its constants ([`README.md`](README.md) describes the
move), from November 2026. All tests use the STFT grid with DFFR. The pair and the two bands ran on a local test
harness and with local unit tests, outside this repository; the runs from 8 mHz up are global-fit runs.

- **The pair** (exactness): two injected sources at 5 mHz, SNR 24 and 19, 1.5 bins and 2.9 degrees apart, with three
  slots, 20 walkers and 12 temperatures. The reference is a run at a fixed count of two; the evidence ratio of three
  sources to two is 0.0019.
- **Two bands** (recovery): two neighbouring bands of the 714-day Mojito data between 15 and 20 mHz with 5 injected
  sources, started from the 20 candidates of the production search, of which 13 pile up on one source; 32 walkers,
  12 temperatures up to $T = 10$. Arms: the prior move alone, both moves at equal share, the split and merge alone,
  and a run from the injected sources.
- **Runs from 8 mHz up** (production): 8.0 to 29.6 mHz of the same data, 40 bands, 148 resolved injected sources, 182
  start candidates, 240 slots, 32 walkers, 12 temperatures, fixed noise, 500 iterations of which 50 burn-in. `main`
  runs the prior move, `GBFStatRJMove` and the split and merge (shares 0.2, 0.4, 0.4 in the burn-in, then 0.5, 0.2,
  0.3); `without` leaves the split and merge out; `reference` runs the prior move from the injected sources.

## The constants

| constant | value | the number behind it |
| --- | --- | --- |
| window $W$ | half the width of one source | between 8 and 29 mHz 14 of 149 resolved sources have another one within $W$ in their band, none above 10 mHz: above 10 mHz a second source within $W$ is a duplicate |
| `COMPANION_WEIGHTS` | 0.50, 0.35, 0.15 | set by hand, not scanned. On the two bands the search part gave 116 of 140 accepted cold splits, the local part 16 and the prior part 8 |
| `ANTENNA_FLOOR` | 1 per rad$^2$ | the analytic width lies within a factor of 3 of the Fisher proposal's at every tested source for floors of 0.5 to 200, and at synthetic sources on the ecliptic plane for 0.5 to 5; 1 lies closest |
| `RHO_REF`, `RIDGE_FRACTION` | 4.5, 1e-3 | set by hand, not scanned |
| seed grid of the searches | 192 sky points, 5 values of $\dot f$ | it finds the source in 16 of 18 window searches at 62 s each; no grid finds 12 of 18 at 34 s, and 768 points with 21 values find 6 of 6 at 309 s |

## Checks

| check | result |
| --- | --- |
| the static math against a numpy prototype | 44 of 44 checks pass. The densities of a split and of its reverse merge cancel to 1.1e-13; an exact model with 1000 states keeps detailed balance to 2.5e-17 and detects each of three one-sided faults (3e-3 to 2e-2) |
| the F-statistic rows and pair products | equal to the direct inner products of real DFFR templates |
| the move's change in ln L against a full recomputation, on the pair | 5e-13 relative; limit 1e-4 |
| widths of the cold posterior on the pair, relative to the reference | 0.98 to 1.04; limit 0.95 to 1.05 |
| odds of three sources against two on the pair, relative to the prior move's; 2000 proposes, temperatures 1 to 2, the frequency prior narrowed to the window | 1.020 ± 0.047 for the move alone and 0.971 ± 0.053 with a search record; 1 is expected |
| `FStatResidualSearch` taken out of the search move | two searches in a row give the same bytes as before, 108 of 108 arrays |
| the move's own searches, in the main run | the searched walker's ln L is unchanged to the last digit after 16 of 16 searches |
| the search on two cards against one | 0 differing words in 227 of 227 comparisons of the F-statistic values; the seed grid takes 0.56 of the one-card time |

Not shown by the pair: the density of the search part in an accepted move. Its record held components on the two
sources only, so none of 178 891 companions from the search part was accepted there.

## Recovery

**Two bands.** With the move at equal share a cold walker ends with 6 sources (5 to 9 over the walkers) for 5 injected
ones, against 21 (17 to 25) with the prior move alone. All 32 walkers hold the three sources that the candidates
missed, by template overlap from iteration 17 on; with the prior move alone none does. After that the arm
samples the same posterior as the run from the injected sources: 5.76 sources per walker in both, the medians within
0.09 of the reference's half width and the widths within 4 percent. On the hotter temperatures the pile stays; this
was not examined.

**Runs from 8 mHz up**, over the 450 kept iterations (14 400 cold states per run). The cold samples are clustered
with the algorithm of `gather_gb_samples` (`src/lisatools/globalfit/gathergalaxy.py`), applied to the STFT templates;
a source counts as held when exactly one cluster that is present in at least half the states has a point estimate
with an overlap of at least 0.9 with it. Every source held this way has an SNR within 0.8 to 1.25 of the injected
one.

| measure | reference | main | without |
| --- | --- | --- | --- |
| mean alive sources per state | 161.8 | 164.1 | 218.3 |
| injected sources held, of 148 | 148 | 143 | 128 |
| also with two or more templates in at most 5 percent of the states | 148 | 139 | 125 |
| clusters in at least 10 percent of the states on no injected source | 0 | 4 | 86 |

- The piles of the candidates at 13.17, 15.65, 16.72, 19.67 and 20.38 mHz end as one source with an overlap of 1.000;
  without the move they stay (17.4 templates for the 3 sources of one band, 14.3 for the single source of another).
- One pile is kept in both runs: the source at 8.528 mHz (SNR 48.5) is held by 2 to 4 templates in 97 percent of the
  states, although the search part has a component on it. The cause is not established.
- The source at 10.886 mHz (SNR 268) is still two templates at one position in 56 percent of the states.
- Three sources of SNR 7.9 to 13.3 that the candidates lack are found by neither run; no source lies within $W$ of
  them, so only a birth can place them.
- For the 139 sources the largest shift of a median from the reference is 0.24 of its standard deviation, and the
  widths are 0.95 to 1.06 of the reference's (0.96 to 4.14 without the move). No error was computed for these shifts.

## Cost

The main run took 37 h 25 min, against 32 h 24 min without the move and 30 h 01 min for the reference. A propose of
the split and merge takes 241 s in the kept iterations, against 192 s for the prior move. Its first propose takes
3.5 h for the two searches (132 windows, 397 fits; the seed grids 48 percent, the two stages 25 and 27 percent); after
it the move searches a few windows per propose until the search part is fixed. On the two bands the move's own step
takes 26 s per iteration against 2.2 s for the prior move's, of which the kernel calls take 32 percent.
