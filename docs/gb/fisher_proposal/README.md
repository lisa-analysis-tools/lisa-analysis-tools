# Regularised Fisher proposal of the galactic-binary in-model move

The in-model move of the galactic-binary (GB) block of the global fit, in `GBSpecialBase`
(`src/lisatools/globalfit/moves/gbspecialmove.py`), draws Gaussian jumps for each source from a
factor of a proposal covariance built from the source's Fisher matrix. In October 2026 two parts of
it were replaced. The covariance is now the inverse of the tempered Fisher matrix plus the precision
of the source's uniform prior box, inverted in correlation form, so no eigenvalue floor is needed.
And each source holds its Fisher matrix for a set number of blocks, instead of getting a new one at
its own position at the start of every block.

| Document | Content |
| --- | --- |
| `README.md` | The update, why its covariance has this form, how the inversion handles the ill-conditioning, the Jacobian at the prior edges, why the Fisher matrix is held, how to set the interval, where the code lives and what the logs show. |
| [`results.md`](results.md) | The measurements: proposal widths, mixing, posterior accuracy, the block-start shift and the held matrices, cost, and the checks of the port. |

The checks behind `results.md` ran outside this repository, in a local test harness that runs the
production move classes on a DFFR likelihood against exact reference chains, and in local unit
tests. Author: Coen Rondeel.

## The in-model update

A block is one source's turn in a propose: `num_repeat_proposals` in-model repeats (50 in the
global-fit settings). The source's sampling coordinates are
$x = (\ln A, f_0, \dot f, \phi_0, \cos\iota, \psi, \alpha, \sin\delta)$, with $f_0$ in mHz and
$\dot f$ in Hz/s. Each repeat proposes

$$x' = x + \gamma\, L z, \qquad z \sim \mathcal{N}(0, I_8), \qquad L L^\top = \Sigma,$$

with the jump scale $\gamma$ drawn for each source from $0.84 \times \{0.2,\ 1.0,\ 2.5\}$ with
probabilities $\{0.2,\ 0.6,\ 0.2\}$. While $L$ does not depend on $x$, the proposal is symmetric,
$q(x' \mid x) = q(x \mid x')$, and the repeat is accepted with probability

$$\min\left(1,\ \frac{\pi(x')\, \mathcal{L}(x')^{\beta}}{\pi(x)\, \mathcal{L}(x)^{\beta}}\right),$$

with $\pi$ the prior, $\mathcal{L}$ the likelihood and $\beta$ the inverse temperature of the
source's cell. The scale $\gamma$ is drawn independently of $x$, so the mixture over scales keeps the
proposal symmetric.

## Why the covariance is $(\beta F + P)^{-1}$

Near its mode, the tempered posterior of one source is close to a Gaussian (Laplace approximation),

$$\ln\left[\pi(x)\, \mathcal{L}(x)^{\beta}\right] \approx \mathrm{const} - \tfrac{1}{2}\, \beta\, (x - \hat x)^\top F\, (x - \hat x) + \ln \pi(x),$$

with $F$ the Fisher matrix in the sampling basis. The GB priors are uniform on a box of widths
$w_i$. A uniform distribution of width $w$ has variance $w^2/12$, so the Gaussian with the same
variance has precision $12/w^2$. Replacing the box by this Gaussian gives the precision of the
tempered posterior

$$\Sigma^{-1} = \beta F + P, \qquad P = \mathrm{diag}\left(12 / w_i^2\right).$$

A random-walk Metropolis step mixes fastest when its covariance has the shape of the target's,
scaled by $2.38/\sqrt{d}$ for $d$ parameters (Gelman, Roberts and Gilks 1996); for $d = 8$ that is
0.84, the middle jump scale.

The two limits are the ones a tempered sampler needs:

- where the data dominate, $\beta F_{ii} \gg 12/w_i^2$: $\Sigma \to (\beta F)^{-1}$, the Fisher
  widths widened by $1/\sqrt{\beta}$;
- where the data leave a direction open (the hottest chains with $\beta \to 0$, a source at zero
  amplitude, the $\phi_0$-$\psi$ combination of a face-on source): $\Sigma \to P^{-1} =
  \mathrm{diag}(w_i^2/12)$, steps the size of the prior box.

$P$ is positive definite, so $\beta F + P$ is positive definite for every positive semi-definite
$F$ and every $\beta \ge 0$, with its smallest eigenvalue at least $\min_i 12/w_i^2$. The inverse
exists without a floor, and the regularising term is not a tuning constant: it is the information
the prior adds to the posterior.

The widths come from the source's prior box in the sampling basis:

- $f_0$: the cell's frequency range (the in-model gate rejects any $f_0$ outside it), mapped into the
  sampling basis and cut at the prior's $f_0$ range;
- $\dot f$: the bounds of the joint $(f_0, \dot f)$ prior at the source's $f_0$
  (`MojitoF0mHzFdotPrior.get_fdot_bounds`), which widen with $f_0$;
- $\phi_0$, $\psi$ and $\alpha$: their periods;
- $\ln A$, $\cos\iota$ and $\sin\delta$: their prior ranges.

## Ill-conditioning, and how the inversion handles it

The difficulty is numerical. In the sampling units the diagonal entries of $F$ span about 31
orders of magnitude: at signal-to-noise ratio (SNR) 1, $F_{\ln A \ln A} \sim 1$,
$F_{f_0 f_0} \sim 10^{11}$, $F_{\dot f \dot f} \sim 10^{31}$ and $10^{4}$ for the sky angles. The
condition number of such a matrix is of order $10^{31}$, far beyond what double precision
($\epsilon = 2.2 \times 10^{-16}$) can resolve: an eigendecomposition returns the eigenvalues below
about $\epsilon\, \lambda_{\max}$ as rounding noise, and their directions are lost.

The earlier factor rescaled the $\dot f$ column by $10^{-16}$ and floored the eigenvalues at
$10^{-10}\, \lambda_{\max}$. After the rescale the matrix still had condition numbers of
$8.5 \times 10^{12}$ to $2.0 \times 10^{13}$ on the DFFR production grid (713 one-day segments), so 5
or 6 of its 8 eigenvalues fell below the floor. A floored direction gets the variance
$1/(10^{-10}\lambda_{\max})$ instead of $1/\lambda_i$, and $\lambda_{\max}$ belongs to the
$f_0$-$\dot f$ block, so the jumps were 6 to 40 times smaller than the Fisher widths in $\ln A$,
$f_0$, $\dot f$, $\phi_0$, $\cos\iota$ and $\psi$; only the sky angles kept their width.

The new factor removes the scales before the decomposition. With $M = \beta F + P$ and
$D = \mathrm{diag}\left(\sqrt{M_{ii}}\right)$, the correlation form

$$C = D^{-1} M D^{-1}, \qquad C_{ii} = 1, \qquad |C_{ij}| < 1,$$

holds only the correlations between the parameters, whatever their units. Its eigendecomposition
$C = V \Lambda V^\top$ is accurate, and

$$L = D^{-1} V \Lambda^{-1/2} \qquad \text{gives} \qquad L L^\top = D^{-1} C^{-1} D^{-1} = M^{-1}.$$

The scaling by $D$ is one division per entry, with no cancellation, so the 31 orders of magnitude
move into $D$ exactly and the eigen-solver sees only the correlation structure, of order one. This
diagonal scaling is close to the best possible: for a positive definite matrix, the condition number
of $D^{-1} M D^{-1}$ is within a factor of the dimension (8) of the smallest condition number any
diagonal scaling can reach (van der Sluis 1969). On the DFFR grid the condition number of $C$ is
$4.7 \times 10^{2}$ to $1.6 \times 10^{3}$ at $\beta = 1$ and 11 to 102 at $\beta = 10^{-4}$, so no
direction is lost. Against exact rational inverses of GB-like matrices with the same diagonal range,
every entry of $L L^\top$ agrees with $M^{-1}$ to $10^{-8}$ relative to
$\sqrt{\Sigma_{ii} \Sigma_{jj}}$ while the condition number of $C$ stays below $10^{7}$, and to
$100\, \epsilon$ times that condition number above it. Rescaling one column, as sampling
$\dot f\, T_{\rm obs}^2$ instead of $\dot f$ would, changes the factor by exactly that rescaling (to
$10^{-12}$), so the code carries no unit assumption.

Two safeguards remain. A source whose Fisher matrix is not finite (a NaN from the transform), or
whose correlation matrix has an eigenvalue that is not positive (a kernel matrix that is numerically
indefinite), gets the prior factor $\mathrm{diag}(w_i/\sqrt{12})$; its matrix is replaced by the
identity in the batched decomposition first, so one bad matrix does not spoil the others. And for
each eigenvector $v$ of $C$, the prior's share of its eigenvalue,
$v^\top D^{-1} P D^{-1} v / \lambda$, between 0 and 1, is logged at debug level: above 0.5 the
direction is held mainly by the prior.

## The Jacobian at the prior edges

The Fisher matrix comes from the domain's computation object (`STFTGBComputations`,
`GBFDComputations` or `GBWDMComputations`) in the physical parameters
$\theta = (A, f_0, \dot f, \phi_0, \iota, \psi, \alpha, \delta)$, $f_0$ in Hz. Each sampling
coordinate maps to one physical coordinate, so the Jacobian $J = \mathrm{diag}(\partial\theta_i /
\partial x_i)$ is diagonal and $F = J\, F_\theta\, J$. $J$ comes from central differences of the
transform container with steps $h_i = 10^{-6} \max(|x_i|, 10^{-3})$. At the prior edges these steps
would leave the domain of the transform: $\iota = \arccos(\cos\iota)$ has
$\partial\iota / \partial\cos\iota = -1/\sin\iota$, which diverges at $\cos\iota = \pm 1$, while the
waveform's derivative with respect to $\iota$ vanishes there, so the product is finite but evaluates
to $0 \times \infty$; $\sin\delta = \pm 1$ behaves the same. Both $J$ and $F_\theta$ are therefore
evaluated at the point clipped $2h$ inside the box, which gives a finite pair that belongs together
(within 5 percent of the analytic Jacobian at that point). The source itself stays where it is. The
earlier factor took $F_\theta$ at the edge and $J$ beside it, and its factor was NaN for such sources.

## Holding the Fisher matrix

The earlier scheme computed $F$ at the walker's position at the start of every block and used it for
the block without a Hastings term. A Metropolis step with a fixed $\Sigma$ is reversible for the
target, but choosing $\Sigma$ from the current state without the ratio
$q(x \mid x')/q(x' \mid x)$ is not: in the limit of small steps, a chain whose step size $s(x)$
depends on the state samples approximately $\pi(x)/s(x)^2$, since it leaves quickly where $s$ is
large and lingers where $s$ is small. For GB sources the Fisher widths grow towards face-on, along
the $\ln A$-$\cos\iota$ degeneracy, and at low amplitude ($F \propto A^2$), so the chain spent too
little time face-on and at low amplitude: for sources of SNR 19 to 24 the posterior mean of
$\cos\iota$ moved by $-0.06$ to $-0.12$ posterior standard deviations and that of $\ln A$ by up to
$+0.09$. A Hastings correction needs the Fisher matrix at every proposed point, 3.5 to 7 times the
cost of a sweep.

Each source now holds its Fisher matrix $F$ and prior widths $w$ for `num_stationary_fim_steps`
blocks. The factor is rebuilt at every block from the held pair, so between two refreshes it does not
depend on the source's current position: every repeat is a symmetric proposal, and every block is an
exact Metropolis update of the target for that $\Sigma$. What remains is the choice of the matrix at a
refresh, made at the position $x_r$ of that time. Every update after the refresh leaves the target
invariant whatever matrix it uses, so the distortion comes only from the correlation between $x_r$
and the following states, which dies out after a few autocorrelation times. Over an interval much
longer than the autocorrelation time, the shift therefore shrinks as the interval grows. With
autocorrelation times of 1 to 5 blocks for most parameters, an interval of 50 blocks puts it below the
resolution of the measurements, within 0.02 to 0.03 posterior standard deviations at one and at 24
temperatures, with the same mixing and a sweep 18 to 24 percent cheaper (`results.md`). An interval
of 1 is the earlier scheme exactly. The factor takes the cell's current $\beta$, so a source that
changed temperature in a swap keeps its matrix and gets the new $\beta$.

## How to set the refresh interval

`num_stationary_fim_steps` is the number of blocks a source's Fisher matrix serves before it is
recomputed at the source's position. A source without a matrix (a fresh start, a birth, a restart)
gets one at its next block.

- `GBMoveSettings.num_stationary_fim_steps` (default 1) sets it for the parameter-estimation moves
  (`rj_prior`, `rj_fstat_mcmc`, `rj_refit`) built by `build_gb_moves`. With 1, every block gets a
  new matrix, as before the change.
- The search moves (`rj_prior_search`, `rj_fstat_mcmc_search`, `rj_refit_search`) always use 1: a
  source climbing onto its signal needs a matrix at its current position.
- The value is a plain attribute of the move, so a recipe step can change it when it starts, for
  example a burn-in step at 1 followed by a parameter-estimation step at 50.

All GB moves of a run share the held matrices through the state, so the count is in blocks of any GB
move. With `rj_prior` as the only GB move, every live source gets one block per iteration, and the
interval counts iterations.

The matrices live on the host in the gb branch supplemental: per leaf slot the 8 x 8 matrix, the 8
prior widths, the blocks served and a flag, 585 bytes. That is 0.90 GB for 4000 leaf slots, 24
temperatures and 16 walkers, and 9 MB for the high-frequency settings (20 slots, 24 x 32). The HDF5
backends do not save branch supplementals, so after a restart every source gets a new matrix at its
next block.

## Where the pieces live

| Code | Role |
| --- | --- |
| `gbspecialmove.py`: `fisher_proposal_factor` | The factor of $(\beta F + P)^{-1}$ in correlation form, with the prior fallback and the diagnostics. |
| `GBSpecialBase._prior_box` | The prior box of every source from the prior container and the cell. |
| `GBSpecialBase._sampling_jacobian` | The diagonal Jacobian at the point clipped into the box. |
| `GBSpecialBase._information_matrix_phys` | The Fisher matrix of the domain's computation object, routed per GPU shard. |
| `GBSpecialBase._compute_fisher_sampling` | The Fisher matrix in the sampling basis and the prior widths. |
| `GBSpecialBase._compute_proposal_cholesky` | The held matrices: a refresh when due, the factor with the current $\beta$. |
| `GBSpecialBase._ensure_fim_store`, `_load_fim_rows` | The store in the branch supplemental and its copy per sorter row. |
| `GBSpecialBase._run_rj_step` | Clears the held matrix of every accepted birth and death. |
| `GBSpecialBase._write_back_state` | Moves each held matrix to its source's new slot. |
| `GBSpecialBase.in_model_proposal` | The Gaussian jump with a jump scale per source. |
| `priors/gbpriors.py`: `MojitoF0mHzFdotPrior.get_fdot_bounds` | The $\dot f$ bounds at a given $f_0$ (mHz). |
| `stock/erebor.py`: `GBMoveSettings.num_stationary_fim_steps` | The interval of the parameter-estimation moves. |
| `recipe.py`: `build_gb_moves` | Passes the interval to each move, 1 for the search moves. |

## What the logs show

Every propose logs one line `[GB_ACCEPT_SCALES <move>]` with the acceptance of the Fisher jumps per
jump scale, in the cold chain and over all temperatures; in the cold chain it was 0.77, 0.21 and
0.015 for the scales 0.168, 0.84 and 2.1. At debug level, every block logs the number of sources, how
many got a new matrix, how many of those were evaluated at a clipped point, how many have a direction
held mainly by the prior, the largest condition number of $C$ and the number of prior fallbacks.

## Open items

- Two overlapping sources (0.75 frequency bins apart at 5 mHz) keep $f_0$ and $\dot f$ at about 30
  sweeps of autocorrelation time. Their joint 16-dimensional Fisher matrix explains it; a joint
  proposal for such pairs would address it, not the one-source factor.
- The largest jump scale (2.1) is accepted 1.5 percent of the time; a cheaper mix of scales is a
  short harness test.
- From a start with one leaf for two blended sources, the reversible-jump runs stay blended for 2000
  sweeps with either factor: a search property of the reversible-jump and tempering scheme, not of
  the in-model factor.

## References

- A. Gelman, G. O. Roberts and W. R. Gilks, Efficient Metropolis jumping rules, Bayesian Statistics
  5 (1996).
- A. van der Sluis, Condition numbers and equilibration of matrices, Numerische Mathematik 14 (1969)
  14-23.
