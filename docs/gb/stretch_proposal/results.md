# Exact group stretch: results

The measurements behind the exact group stretch of the GB in-model move (see [`README.md`](README.md)
for the method). They were made in November 2026 with a local test harness, outside this repository,
which runs the production move classes. The rule was first measured as a layer in front of the
production class; the port into this repository was then checked to give the layer's chains bit for
bit.

## Method

- **DFFR pair.** The DFFR likelihood of `STFTGBComputations` on the production grid of the
  high-frequency global fit, ESA orbits, the production noise model, noiseless data made of a pair of
  overlapping sources of SNR 24 and 19 at 5 mHz in their production band (the pair of the Fisher
  proposal's measurements).
- **Reference.** An independent fixed-k reference chain of the pair. The GB waveform is unchanged under
  $(\phi_0, \psi) \to (\phi_0 + \pi, \psi + \pi/2)$, so the exact posterior holds two phase modes; the
  reference holds 0.484 and 0.506 of its samples in the partner mode of the two sources.
- **Benchmark group.** Bands 1128 to 1138 of the 714-day band layout (2.83 to 2.87 mHz): 134 injected
  sources inside, of which the production search found 64, plus 28 false candidates. The data are the
  714-day Mojito data with the resolved population outside the group subtracted, and the noise is
  frozen at the production fit. 32 walkers and 12 temperatures with the hottest at T = 10.
- **Statistics.** A shift is the difference of posterior means in reference sd; z = shift / standard
  error. Autocorrelation times are in proposes; one propose gives every source one block of 50
  repeats.

## The bias of the earlier rule

Stretch alone (`stretch_probability = 1`), one temperature: against the independent reference of the
DFFR pair; on a flat likelihood (inverse temperature $10^{-12}$), where every marginal must equal the
prior; and on the benchmark group at fixed k (the 134 injected sources), where the exact posterior puts
half of the samples in the $(\phi_0, \psi)$ partner mode:

| test | earlier rule | exact rule |
| --- | --- | --- |
| DFFR pair, largest abs z over the 16 parameter means of both sources | 5.6 ($f_0$) | 3.4 |
| flat likelihood, largest abs z over 16 marginal statistics | 122: a U-shape of the $f_0$ marginal, 14 percent excess at the edges | 4.4 (every share within 0.5 percent of the prior) |
| benchmark group at fixed k, share of samples in the partner phase mode, four sources | not run | 0.48 to 0.50 |

A toy with two Gaussian sources showed the same: the earlier rule's variance ratio to the exact
posterior was 1.011 to 1.033 (standard error 0.008), while a table of independent friends was exact.
The friend choice is the whole cause.

## Exactness in the parameter-estimation setting

DFFR pair from the injected values, 20 walkers x 12 temperatures, Fisher matrices held for 50
proposes, 2000 proposes; the second half (20 000 cold samples) against the independent reference,
over the 16 values (8 parameters of 2 sources). "Tables per half" rebuilds the tables before every
half, as the port does; "held 50" builds them once every 50 proposes.

| arm | s per propose | partner phase mode, source 1 / 2 | largest abs z | largest shift [sd] | sd ratios | KS D / noise, largest | tau [proposes] | s per independent sample |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fisher jump alone | 3.68 | 0.497 / 0.504 | 1.4 | 0.027 | 0.983 to 1.018 | 1.3 | 1.5 / 1.9 | 0.29 |
| earlier stretch, probability 0.5 | 3.78 | 0.498 / 0.497 | 1.7 | 0.030 | 0.996 to 1.019 | 1.2 | 1.5 / 1.8 | 0.29 |
| exact stretch 0.5, tables per half | 4.89 | 0.507 / 0.502 | 2.5 | 0.039 | 0.975 to 1.016 | 2.0 | 1.6 / 1.8 | 0.39 |
| exact stretch 0.5, tables held 50 | 4.95 | 0.496 / 0.499 | 2.3 | 0.048 | 0.985 to 1.014 | 1.4 | 1.6 / 1.7 | 0.39 |
| exact stretch 1, tables per half | 5.26 | 0.499 / 0.497 | 2.1 | 0.049 | 0.994 to 1.028 | 1.2 | 1.9 / 2.1 | 0.49 |
| exact stretch 1, tables held 50 | 5.16 | 0.499 / 0.500 | 1.5 | 0.034 | 0.973 to 1.018 | 0.9 | 1.9 / 2.1 | 0.50 |

Every arm agrees with the reference within the resolution of about 0.04 sd and holds both phase
modes: the 12-temperature ladder crosses between them. With 12 temperatures and the stretch mixed with
the Fisher jump, the earlier rule's bias stays below this resolution. The autocorrelation times are 1
to 2 proposes in every arm, so on this pair the arms differ in cost only, and the Fisher jump alone is
the cheapest per independent sample.

## Burn-in from the search candidates

The benchmark group, started from the production search's 92 candidates (64 of the injected sources and
28 false ones) or from an empty state, 500 proposes per arm, one run each, Fisher matrix refreshed every
propose. A run started at the 134 injected sources (20 walkers, 480 proposes) gives the reference
posterior. A walker's source counts as converged on an injected source when it is the walker's source
nearest in $f_0$ within one frequency bin and it lies inside the 99 percent region of the reference
posterior of that source (Mahalanobis distance against the reference's own 99th percentile); 126 of the
134 sources enter, the reference holding the other 8 in fewer than half its states. At propose 500:

| arm | missed sources converged | found sources converged | sources held by half the walkers (of 126) | cold ln L minus the reference | false candidates kept |
| --- | --- | --- | --- | --- | --- |
| reference (started at the injections) | 0.915 | 0.989 | 125 | 0 | 0.105 |
| Fisher jump alone | 0.117 | 0.915 | 64 | -704 | 0.635 |
| earlier stretch 0.5 | 0.215 | 0.907 | 67 | -605 | 0.525 |
| exact stretch 0.5 | 0.233 | 0.910 | 67 | -568 | 0.526 |
| earlier stretch 0.25 | 0.182 | 0.911 | 65 | -624 | 0.568 |
| exact stretch 0.25 | 0.179 | 0.911 | 64 | -623 | 0.550 |
| Fisher jump alone, empty start | 0.045 | 0.102 | 0 | -2057 | 0.247 |
| earlier stretch 0.5, empty start | 0.114 | 0.278 | 9 | -1298 | 0.265 |
| exact stretch 0.5, empty start | 0.093 | 0.272 | 6 | -1348 | 0.260 |

- From the search start, the stretch at 0.5 doubles the share of the missed sources that converge,
  mostly at SNR below 10, ends about 100 higher in cold ln L and keeps a tenth fewer false candidates.
  The sources the search found stay at 0.91 in every arm.
- The friend rule changes the burn-in little: the exact stretch is slightly ahead of the earlier one in
  every column, by amounts inside the unknown spread between runs. What the exact rule adds is
  exactness.
- At probability 0.25 the gain is about half.
- From an empty start every arm converges at most 0.28 of the sources the search found after 500
  proposes and holds 1.1 to 2.0 times the reference's number of sources per band, most of them faint;
  the stretch triples the share there.
- After 500 proposes every arm is far from the posterior: 570 to 700 below the reference in cold ln L.

In equal time, with the costs below, the exact stretch at 0.5 makes about 395 proposes where the Fisher
jump alone makes 500, and still holds 0.20 of the missed sources converged against 0.117.

## Cost

Benchmark group, proposes 2 to 4 from a state after 500 proposes, one A100, the Fisher matrix
refreshed every propose, seconds per propose (production's span timer, without device
synchronisation). The earlier rule can no longer run in the production class, so its cost comes from
the harness's copy of the repeat loop, which gives the same chains as the production class:

| code | stretch off | earlier stretch, probability 0.5 | exact stretch, probability 0.5 |
| --- | --- | --- | --- |
| production class, after the port | 85.4 | | 108.1 |
| harness copy of the loop | 83.3 | 85.1 (84.9 on the day of the port) | 109.2 (as a layer) |

The exact stretch at probability 0.5 costs 1.27 times the move without it; the earlier stretch cost 2
percent more. The table builds take 0.1 s per propose. Rows that fail the reverse check (21 percent
of the cold stretch proposals on the group, 9.5 percent on the DFFR pair) skip the likelihood call.

Where the extra time goes, from a profile with device synchronisation made before the device waits of
the repeat loop were removed: of the 52 s extra per propose, the walker-parity split took 40.6 s and
the stretch proposal 12.7 s. The split doubles the repeat rounds at half the batch. A block moves
about 330 live sources (every propose visits all 98 304 slots of the group in 78 pick rounds, and 26
percent of them are live); at that size a likelihood call cost about 3.5 ms plus 0.022 ms per source,
and the block's work around it about 4.7 ms per round, so the fixed part per round dominates.

## Checks of the port

The port was checked on the DFFR pair (20 walkers x 12 temperatures, Fisher matrices held for 50
proposes, a fixed seed) and with the LAT tests:

| check | result |
| --- | --- |
| default path: probability 0, 20 proposes, before and after the change | every chain, state and counter identical |
| the port against the measured layer: probability 0.5 and 1, 50 proposes, the class after the change against the layer in front of the class before it | identical bit for bit |
| rows that fail the reverse check skip the likelihood call: the same three runs | identical bit for bit |
| `FriendTables` against direct loops and the layer's tables, 20 random table sets, numpy and CuPy | equal |
| 13 LAT test modules of the GB block, before and after | the same outcome for every test |

Because the chains are identical to the layer's, the exactness measured above for the layer holds for
the port.
