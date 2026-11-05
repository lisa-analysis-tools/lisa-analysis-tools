# Regularised Fisher proposal: results

The measurements behind the regularised Fisher proposal and the held Fisher matrices of the GB
in-model move (see [`README.md`](README.md) for the method). They were made in October 2026 with a
local test harness, outside this repository, which runs the production move classes, or copies of
their methods with the residual bookkeeping replaced by the full likelihood, against exact
reference chains.

## Method

- **Likelihood.** The DFFR likelihood of `STFTGBComputations` on the production grid of the
  high-frequency global fit: 713 one-day segments, 16 side bins, ESA orbits, the production noise
  model, noiseless data made of the injected sources.
- **Scenarios.** A pair of overlapping sources of SNR 24 and 19 at 5 mHz, the same pair at 18 mHz,
  and one source at 5 mHz, each in its production frequency band.
- **References.** Fixed-k chains of the per-leaf stretch move, 200,000 iterations with 32 walkers per
  scenario, independent of the Fisher proposal. A *reference sd* is the posterior standard deviation
  of the reference; a shift is the difference of posterior means in reference sd, with
  z = shift / standard error.
- **Runs.** One sweep is one propose: every source gets one block of 50 in-model repeats. Unless
  stated otherwise, 32 walkers, one temperature, 2000 sweeps from the injected values, statistics
  over the second half. Autocorrelation times are integrated over the walker-averaged
  autocorrelation function with a Sokal window of 5, after 200 sweeps.
- **Old and new.** "Old" is the factor of production before October 2026 (fdot rescaled by
  $10^{-16}$, eigenvalues floored at $10^{-10}\lambda_{\max}$); "new" is the regularised factor.

## Proposal widths against the Fisher widths

Proposal widths of the SNR 24 source at 5 mHz before the jump scale, against the one-source Fisher
width $\sqrt{(F^{-1})_{ii}}$:

| Parameter | Fisher width | Old factor | New factor, $\beta = 1$ | Old / new |
| --- | --- | --- | --- | --- |
| ln A | 0.199 | 0.0168 | 0.191 | 0.088 |
| f0 (mHz) | 1.92e-6 | 2.46e-7 | 1.91e-6 | 0.13 |
| fdot (Hz/s) | 5.31e-17 | 1.68e-18 | 5.29e-17 | 0.032 |
| phi0 | 0.673 | 0.0168 | 0.601 | 0.028 |
| cos iota | 0.167 | 0.0168 | 0.160 | 0.11 |
| psi | 0.326 | 0.0168 | 0.291 | 0.058 |
| alpha | 5.13e-3 | 5.03e-3 | 5.13e-3 | 0.98 |
| sin delta | 9.20e-3 | 8.95e-3 | 9.20e-3 | 0.97 |

The new widths are slightly below the Fisher widths where the prior adds information (phi0 and psi
by 11 percent). Condition numbers over the three sources and both frequencies:

| Matrix | Condition number | Directions lost |
| --- | --- | --- |
| Old: $F$ with fdot rescaled by $10^{-16}$ | 8.5e12 to 2.0e13 | 5 or 6 of 8 below the floor |
| Correlation form of $F$ | 5.7e2 to 2.0e3 | none |
| New: $C$ at $\beta = 1$ | 4.7e2 to 1.6e3 | none |
| New: $C$ at $\beta = 10^{-4}$ | 11 to 102 | none; 4 to 7 directions held mainly by the prior |

The f0 prior width is the cell's frequency range. At SNR 5 the f0 Fisher width is 0.50 frequency
bins, 1.1e-3 of the 5 mHz band (439 bins) and 4.3e-6 of the 18 mHz band (117,368 bins), below the
threshold of 1e-2 at both: the band never limits the f0 jumps of a detectable source.

## Mixing

Integrated autocorrelation time in sweeps, old / new; the two sources of a pair are separated by a
semicolon:

| Parameter | Pair, 18 mHz | Pair, 5 mHz | One source, 5 mHz |
| --- | --- | --- | --- |
| ln A | 59.5 / 1.7; 34.0 / 1.9 | 50.8 / 2.9; 59.6 / 9.4 | 42.4 / 1.6 |
| f0 | 139.6 / 2.8; 121.1 / 2.7 | 164.6 / 31.0; 146.3 / 29.4 | 92.7 / 1.4 |
| fdot | 144.1 / 3.6; 127.2 / 3.5 | 173.4 / 35.7; 159.7 / 34.6 | 95.7 / 1.4 |
| phi0 | 107.2 / 1.1; 91.1 / 1.1 | 106.0 / 1.2; 73.1 / 1.4 | 74.0 / 1.0 |
| cos iota | 60.0 / 1.4; 33.9 / 1.5 | 42.8 / 1.5; 37.4 / 1.6 | 43.0 / 1.6 |
| psi | 103.0 / 1.1; 81.8 / 1.0 | 103.2 / 1.1; 70.6 / 1.1 | 70.5 / 1.0 |
| alpha | 47.0 / 4.9; 52.7 / 5.0 | 12.4 / 11.7; 6.7 / 8.6 | 1.3 / 1.4 |
| sin delta | 2.2 / 2.1; 1.8 / 1.9 | 29.4 / 21.2; 57.0 / 23.7 | 1.5 / 1.6 |

The old chains hold only 11 to 25 autocorrelation times of f0, so their values are rough. With the
old factor, the f0 and fdot widths start at 0.19 to 0.52 of their final value and still grow after
200 sweeps; with the new factor they are at 0.99 to 1.02 of it from the first window. The overlapping
pair at 5 mHz keeps f0 and fdot at about 30 sweeps with the new factor: its chain marginals are about
twice one source's Fisher width, which the joint 16-dimensional Fisher matrix of the pair reproduces
within 8 percent for the SNR 24 source. That is a limit of moving one source at a time, not of the
factor.

In-model acceptance in the cold chain: old 0.47 to 0.51, new 0.28 to 0.29; per jump scale 0.168 /
0.84 / 2.1 the new factor gives 0.769 / 0.211 / 0.015 to 0.016.

## Posterior accuracy

Production's `GBSpecialRJPriorMove.propose()` run end to end on the DFFR target's container, 24
temperatures x 32 walkers, 500 proposes from the injected values, against the 5 mHz reference over the
second half:

| Quantity | New factor | Old factor |
| --- | --- | --- |
| Cold in-model acceptance | 0.283 | 0.477 |
| Acceptance per jump scale 0.168 / 0.84 / 2.1 | 0.769 / 0.211 / 0.016 | not counted |
| Chain sd over reference sd, 16 values | 0.966 to 1.014 | 0.724 to 1.029 (f0 and fdot 0.72 to 0.79) |
| Largest shift (z) | 0.111 sd, cos iota of source 2 (-6.2) | 0.431 sd, fdot of source 1 (-15.9) |

The new factor's largest shift is the block-start shift of the next section; with the old factor, f0
and fdot are still narrow and off by 0.29 to 0.43 sd after 500 proposes. The new factor's widths are
within 4 percent of the reference.

Coverage against the injections, three sources at 5 mHz (SNR 20, 15 and 10) in four noise
realisations, 12 sources x 8 parameters = 96 pairs per factor:

| Check | New factor | Old factor | Expected |
| --- | --- | --- | --- |
| Injection inside the 68 % interval | 0.698 | 0.677 | 0.68 +- 0.048 |
| Injection inside the 95 % interval | 0.979 | 0.979 | 0.95 +- 0.022 |
| abs z > 3 | none (largest 2.93) | none (largest 2.89) | about none |

Both factors pass; at this size the test does not separate them.

## The block-start shift

With the Fisher matrix computed at the start of every block and no Hastings term, the posterior means
move, against the references over the second half (shift in reference sd, z in brackets):

| Run | ln A | cos iota | sin delta | Other parameters |
| --- | --- | --- | --- | --- |
| Pair, 5 mHz, new factor, sources 1; 2 | +0.030 (+1.4); +0.009 | -0.070 (-3.8); -0.123 (-7.0) | -0.049; -0.018 | within 0.084 sd, abs z at most 2.8 |
| Pair, 18 mHz, new factor, sources 1; 2 | +0.052 (+3.5); +0.044 (+3.1) | -0.088 (-5.3); -0.090 (-6.8) | -0.015; -0.004 | within 0.044 sd, abs z at most 3.0 |
| One source, 5 mHz, new factor | +0.090 (+7.6) | -0.111 (-9.5) | +0.072 (+6.3) | within 0.010 sd |
| One source, 5 mHz, one fixed factor at the injection | +0.014 (+1.1) | -0.015 (-1.1) | +0.006 (+0.4) | within 0.021 sd |
| One source, 5 mHz, old factor | +0.085 (+2.1) | -0.082 (-2.0) | -0.000 | within 0.11 sd |

The fixed factor, which does not depend on the state, is exact; the shift is a property of computing
the factor from the current state, with either factor (the old one's slow chains resolve it only at
z of about 2). Drift tests from exact reference draws at both frequencies confirm it. Its direction
follows from the stationary density $\pi/s^2$ of a chain with state-dependent step size $s$: the
Fisher widths grow towards face-on and at low amplitude, so $\cos\iota$ moves down and $\ln A$ up.

## Held Fisher matrices

One temperature, 2000 sweeps, new factor, the matrix held for 50 or 200 blocks (shift in reference
sd, z in brackets, sources 1; 2):

| Run | ln A | cos iota | f0 | fdot | sin delta |
| --- | --- | --- | --- | --- | --- |
| Pair, 5 mHz, block start | +0.030 (+1.4); +0.009 (+0.3) | -0.070 (-3.8); -0.123 (-7.0) | -0.078; -0.022 | +0.084; +0.028 | -0.049; -0.018 |
| Pair, 5 mHz, held 50 | -0.009 (-0.4); +0.024 (+1.0) | -0.019 (-1.1); -0.028 (-1.9) | +0.028; +0.024 | -0.027; -0.030 | +0.001; +0.016 |
| Pair, 5 mHz, held 200 | -0.006 (-0.3); +0.006 (+0.2) | -0.006 (-0.4); -0.026 (-1.7) | -0.008; +0.005 | +0.010; -0.005 | -0.014; -0.006 |
| Pair, 18 mHz, block start | +0.052 (+3.5); +0.044 (+3.1) | -0.088 (-5.3); -0.090 (-6.8) | -0.010; -0.005 | +0.022; +0.005 | -0.015; -0.004 |
| Pair, 18 mHz, held 50 | +0.006 (+0.4); +0.001 (+0.1) | -0.009 (-0.6); -0.011 (-0.9) | -0.012; +0.006 | +0.014; -0.000 | +0.001; -0.012 |
| Pair, 18 mHz, held 200 | +0.008 (+0.6); +0.017 (+1.2) | -0.013 (-0.8); -0.017 (-1.3) | -0.019; -0.019 | +0.019; +0.023 | +0.022; -0.015 |
| One source, 5 mHz, block start | +0.090 (+7.6) | -0.111 (-9.5) | -0.001 | -0.003 | +0.072 (+6.3) |
| One source, 5 mHz, held 50 | +0.001 (+0.1) | -0.003 (-0.2) | | | -0.018 (-1.5) |

Held for 50 or 200 blocks, every value is within 0.030 sd with abs z at most 1.9, against -0.07 to
-0.12 sd at z -3.8 to -9.5 for the block start. The autocorrelation times equal those of the block
start within their scatter. A drift test on a pair only 0.75 frequency bins apart at 5 mHz, where 6
percent of the walker-sweeps swap the f0 order of the two leaves, gives within 0.022 sd (abs z at most
1.0) for a matrix held 50 blocks, against -0.052 sd (z -4.3) in cos iota for the block start.

With 24 temperatures, 32 walkers and tempering swaps (pair at 5 mHz, 2000 sweeps, the matrix stored
per source as in production, moving with the sources through swaps and the write-back):

| Quantity | Held 50 | Block start (held 1) | Criterion |
| --- | --- | --- | --- |
| Largest shift of the 16 values | 0.020 sd, abs z at most 0.97 | cos iota -0.065 (z -3.6) and -0.103 (z -6.1); ln A of source 2 +0.062 (z +3.4) | below 0.03 sd or abs z below 3 |
| Chain sd over reference sd | 0.980 to 1.006 | | |
| Cold in-model acceptance | 0.277 | 0.283 | within 0.02 |
| Acceptance per jump scale | 0.762 / 0.202 / 0.016 | 0.769 / 0.211 / 0.016 | within 0.02 |
| Swap acceptance | 0.297 to 0.852 | 0.299 to 0.858 | within 0.02 |
| Fisher matrices computed / blocks | 61,440 / 3,072,000 | one per block | |
| Time per sweep | 10.3 s | 13.5 s | |

Held for 50 blocks, the matrices pass every criterion; at 1 the run shows the block-start shift.

With reversible jump (prior births, 24 temperatures, 2000 sweeps from one leaf for the two sources,
held 50): 104,598 Fisher matrices for 1,582,581 blocks (births add new ones), no prior fallback and no
factor that is not finite. That run stays in a one-leaf blended state with either factor and either
interval, so it tests the store under births and deaths but not the bias of a two-source posterior.

A factor taken from a partner walker's source at the same leaf is also exact, but costs 2.3 times a
sweep (the walkers must run in two halves) and was not taken further.

## Cost

| Run | Fisher share of the sweep | Fisher per block |
| --- | --- | --- |
| Pair, one temperature, block start, both factors | 0.18 | 140 ms |
| One source, one temperature, block start | 0.31 | 140 ms |
| Reversible jump, 24 temperatures, block start | 0.20 to 0.27 | 315 ms |

A Hastings correction would need the Fisher matrix at the proposed points: about 3.5 times the
present sweep, up to 7 times. Holding the matrix for 50 blocks makes a sweep 18 percent cheaper at one
temperature (1.27 s against 1.55 s for the pairs) and 24 percent cheaper at 24 temperatures (10.3 s
against 13.5 s). The store costs 585 bytes per leaf slot on the host.

## Checks of the port into lisa-analysis-tools

The port was checked against the reviewed development code by identical chains: production's
`GBSpecialRJPriorMove.propose()` on the DFFR target's container (5 mHz pair, 24 temperatures x 32
walkers, 3 leaf slots, 20 proposes, one seed for numpy and CuPy), comparing every array of the cold
chain (coordinates, live flags, log-likelihood and log-prior at every propose) and of all
temperatures (log-likelihood at every propose, final coordinates, live flags and band temperatures).
Base: `feat-dffr` at a9c19015.

| Run | Against | Result |
| --- | --- | --- |
| Development code before the port, twice | each other | identical: the GPU run is reproducible |
| Regularised factor ported | development code before the port | identical |
| Held matrices ported, interval 1 | development code before the port | identical |

The held matrices at interval 50, with the store removed before propose 10 as a restart does,
computed one Fisher matrix per live source at propose 0 (1484 of 1484) and at propose 10 (955 of
955), and in between only for born sources (13 to 40 per propose, since no source reaches 50 blocks in
20 proposes); no prior fallback. Cold in-model acceptance 0.300, per jump scale 0.794 / 0.232 /
0.017; 32.9 s per propose against 35.8 s at interval 1.

Local unit tests, outside this repository: the factor against exact
rational inverses for SNR 0.3 to 300 and core condition numbers 10 to 1e6 at four temperatures, its
limits ($F = 0$, $\beta = 0$, a null direction), the column rescaling, the fallback, the prior box for
one-column and joint priors and its errors, the Jacobian at the edges, the production path through
`STFTGBComputations` (both responses, CPU and CUDA), the joint prior and `get_fdot_bounds`, and the store
(a toy branch through an accepted swap, a birth, a death and a change of leaf order; the refresh
blocks at intervals 1 and 3; the held matrix taking the block's $\beta$): 23 tests pass. The
repository test that builds the production move, `tests/test_stft_store_windows.py`, passes (15
tests) with both commits.
