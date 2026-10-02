# CD1-L MBHB posterior sampling (WDM likelihood)

Parameter estimation for the 20 CD1-L (Mojito) massive-black-hole binaries
with the batched WDM likelihood from PR #81. Noiseless injections; all runs
post-merger, dt = 10 s. The first campaign (`runs/`) used order 8 and 32 walkers
x 5 temperatures; the launcher now uses order 16 and 160 walkers x 1 temperature.

## Requirements

* this branch: PR #81 (`bd724d2d`) plus two library commits, the tdionfly
  Orbits cache (speed only, bit-identical) and data-relative grid alignment
  with cached WDM masks;
* Eryn with lisa-analysis-tools/Eryn#50 ("Avoid NaN log_like state from
  rejected infinite proposals"). Without it a rejected infinite proposal can leave NaN in the
  likelihood state;
* a GPU node. The venv SIGILLs on login nodes (AVX-512 build).

## Files

| file | role |
|---|---|
| `cd1l_pe.py` | driver: data + Fisher + eryn PT sampler; also the proposal probe (`CD1L_PROPOSAL_PROBE=1`) |
| `cd1l_sampling.py` | MC basis, priors, moves (Fisher/Roulet Gaussian, sky partners), t_det transform |
| `cd1l_roulet.py` | chi_eff / chi_diff spin reparametrisation (`CD1L_ROULET_SPINS=1`) |
| `cd1l_match_check.py`, `cd1l_windows.py` | Mojito data, orbits, waveform settings, observation windows |
| `cd1l_warm_inputs.py` | warm start / frozen proposal covariance from a finished chain |
| `cd1l_ridge_accounting.py` | chain loading, periodic handling, ridge regressions (`--runs runs`) |
| `cd1l_plots.py` | traces, corner and sky localisation |
| `slurm/` | launchers; `cd1l_env.sh` holds the module/venv setup |

## Main knobs

| env | default | meaning |
|---|---|---|
| `CD1L_TDET` | 0 (launcher: 1) | sample the arrival time at the constellation centre, t_det, instead of t_SSB. `t_SSB = t_det + n_hat . r_LISA/c`; exact reparametrisation, unit Jacobian |
| `CD1L_STRETCH_WEIGHT` | 0.15 (launcher: 0.5) | StretchMove weight; the Gaussian move takes the rest |
| `CD1L_PRIOR_DRAW` | 0 | PriorDrawMove (acceptance was exactly 0 on all 40 finished chains) |
| `CD1L_SKY_GIBBS` | auto | sky-partner Gibbs; auto keeps it only if a partner is within 1000 in log L |
| `CD1L_ORDER` | 8 (launcher: 16) | response Lagrange order |
| `CD1L_NWALKERS`, `CD1L_NTEMPS` | 32, 1 (launcher: 160, 1) | ensemble size and number of temperatures (fixed geometric ladder 1..1e-4) |

## Reproducing

```
sbatch scripts/mbh/slurm/submit_post1d_mc.sh          # campaign (t_det, order 16); resubmit to continue
sbatch scripts/mbh/slurm/submit_post1d_tdet_ab.sh     # t_det vs runs/ A/B on ids 2, 14
sbatch scripts/mbh/slurm/submit_proposal_probe.sh     # one-step probe, t_SSB vs t_det
R=$ROOT/runs P=$ROOT/plots IDS="2 14" sbatch scripts/mbh/slurm/submit_plots.sh
```

## Results so far (2026-09-24)

* Posteriors (t_SSB campaign, `runs/`): 19/20 sources have |pull| < 0.8 in every
  parameter. id19 is biased at 2.5-5 sigma because at SNR 3600 the template does
  not match the Mojito generator.
* Mixing: 12/20 chains have tau capped by chain length. The slow sources sit on
  a thin extrinsic ridge (d_L, psi, ra and, in t_SSB, the delay).
* t_det: the whole t_plunge excess over the Fisher is the LISA light delay (the
  regression on n_hat leaves a median of 7% of the t sd). Proposal probe, id14,
  N=2048, f=0.3: t_det ESJD 0.0805; t_SSB 0.019 (N=384). In the full chain
  the sky reaches ~85% of its final width after 1000 steps, against 51-62% for t_SSB.
  Over the full 5000 steps, though, tau falls only 206 -> 164 (id2) and 253 -> 226
  (id14), no more than the accompanying move-mix change explains. The slow mode is a
  direction mixing lnMc, t, phi_ref, spins and sky, and walkers take a long time to
  trade places along it (walker-mean offsets hold 18-44% of its variance).
* Tempering: the fixed x10 ladder swapped into the cold chain on 0.2% of
  walker-steps (the post-merger posterior is unimodal), and 160 walkers x 1
  temperature cost the same per step as 32 x 5, so the launcher uses 5x the cold
  walkers instead.
* Move mix: at equilibrium the Gaussian and Stretch moves are comparable (one-step
  probe in t_det: tie on id2, Gaussian 1.9x on id14; 160-walker chain in t_SSB:
  Stretch 1.5x). An earlier "Gaussian 4x Stretch" figure was an artifact:
  warm-starting every temperature at the cold posterior made cold<->hot swaps
  look like large moves. The launcher uses 50/50.
* Numerics: on id2 only, order 8 at dt = 10 leaves a log L ripple locked to
  t_SSB mod 10 s (first-harmonic amplitude 0.49; 0.04 at dt = 2.5), so it
  survives t_det. Order 16 at dt = 10 removes it (0.055), so the launcher uses
  `CD1L_ORDER=16` and dt = 2.5 is not needed. The ridge itself is physical.
* Tried and dropped: LISA-frame sky/psi. Near the frame pole psi_L - lam_L is a
  chart artifact (k = +1.00); after shearing it out the posterior is no thinner
  than the ICRS one.
