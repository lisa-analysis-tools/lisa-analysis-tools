# F-statistic RJ move: results

The evidence behind the settings of `GBFStatRJMove` and the checks of its port ([`README.md`](README.md) describes the
move), from November 2026.

- **Benchmark group** (the settings): bands 1128 to 1138 of the 714-day band layout (2.83 to 2.87 mHz, 134 injected
  sources), the resolved population outside the group subtracted, the noise frozen at the production fit, DFFR on the
  STFT grid, searched from the production search's 92 candidates in the group.
- **Port check runs** (the checks): a copy of the L2D configuration `mojito_input/gb_psd_galfor_dffr` cut to 3.0 to
  3.2 mHz (43 searched bands, 379 start candidates, 32 walkers, 24 temperatures, one A100), with the F-statistic move
  and the prior move taking turns at 0.5; once through the YAML route and once through a settings file.

## The settings

| setting | value | why |
| --- | --- | --- |
| gate statistic | the largest stored cold $\sqrt{2F}$ | it lies within 0.26 of the one-peak model in the 9 bands an injected source explains, while the median lies 0.2 to 1.6 below: the cold walkers sit on several peaks of $F$ without mixing. The FD rule (the minimum above 8) passed 1 of 11 bands |
| `snr_eff_threshold` | 7.5 | the posterior odds of one more source (global $f_0$ prior) break even at a stage-1 maximum near 8.0, an effective SNR of 7.5; no noise-only band favours a birth, but the unresolved galaxy makes peaks up to 7.7 |
| `galfor_inflation_factor` | 1.3 | the median of $(\rho_\mathrm{pop}/\rho_\mathrm{search})^2$ over the group's sources is 1.28 (1.25 to 1.34) |
| thresholds | 7.16 below 6 mHz, 8.01 above | the gate's formula for both targets at 300 stored steps of 32 walkers |
| stage 1 | 6 temperatures, 500 + 300 steps | the search takes 783 s against 3656 s with 24 temperatures, and the maxima repeat within 0.06; a burn-in of 250 or 100 stored steps lost 10 and 6 of 72 passing bands (all near the threshold) and gained none |
| stage 2 | 12 temperatures with $T = 10$, 400 + 500 steps | shorter chains kept only 8 to 38 % of the proposal's effective sample size (ESS), while stage 2 takes 6 to 16 % of a search |
| proposal basis | $\theta_\pm$ with $\lambda$ periodic, the four copies drawn | 17.4 times the summed ESS of the FD move's six-column mixture with uniform phases, the gain coming from the phases; a proposal drawn with $\psi$ below $\pi$ and the same density at $\psi + \pi$ would be biased |
| number of components | production's BIC choice | it reaches 77 to 86 % of the best summed ESS of a scan; a held-out selection fitted the chains better but gave less ESS |

## Checks of the port

| check | result |
| --- | --- |
| `tests/test_gmm_periodic.py` (9 tests) | pass on the CPU and on an A100, where the density equals the CPU's to 1e-10; the mixture without periodic columns gives the same bytes as before |
| local unit tests of the move (14 tests) | pass |
| LAT's L2D and GB setup test modules | the same 71 failures before and after the port, all from the environment of the checks; every new test passes |
| the F-statistic on DFFR | the template at the maximizers reaches $F$ (to 5.4e-5 on noise-free injections) only once $\cos\iota$ takes the opposite sign of the FD map; on a production residual the kernel's $2F$ equals $N^\top M^{-1} N$ to 5.5e-16 |
| flat likelihood: production's RJ pass, 8 runs | 7 pass; the failing one is slow mixing, see below |
| the code after its restructuring, against the code before | the same search in one process, card and seed: 39 of 39 record arrays bit-identical |
| port check runs, both routes | both run as intended; in the YAML run the bands that pass fall from 11 to 1 as the found sources enter the chain (379 to 414 cold sources), and the move's cold RJ acceptance falls with them from 0.021 to 0.002 |

Two findings shaped the code. Eryn passes a likelihood a copy of the coordinates, so the FD search's maximizers, which
it wrote into its input, never reached its chain; the move computes them in a call of its own. And the PE variant of
the move had never been an RJ move, since recipes built it without a proposal; `GBFStatRJMove` is always built as one.

**Slow mixing of the move alone.** The flat-likelihood check samples the prior with production's RJ pass and the
move's proposal fitted to the prior. The failing run used the F-statistic proposal alone and failed on the counts of two
bands (largest z 4.68). An importance-sampling check of the three proposals used alone holds (150 statistics each,
largest z 2.49 to 2.82, mean weight $\pi/q$ of 0.9985 to 1.0009), so their density is right. Where a mixture covers the
prior thinly ($\pi/q$ up to about 200) sources are born and removed rarely, and batch means of 140 sweeps then
understate the errors. Every run with the prior move beside it, as in a production run, passes.

## Cost

On the port check runs a search takes 131 to 180 s: 116 s of stage 1, whatever passes, and 15 to 62 s of stage 2 and
the mixtures for 1 to 11 passing bands. The rest of the move's propose takes 278 to 383 s, against 237 to 276 s for the
prior move's whole propose. The band sorter draws every dead slot's birth in the passing bands, and the many dead slots
of the hot chains then need more pick rounds: up to 527 per unit with one passing band, against 25 to 29 for the prior
move.
