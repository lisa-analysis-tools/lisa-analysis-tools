Global parameter estimation for Galactic double white dwarf binaries (GBs) within the Laser Interferometer Space Antenna (LISA) data stream requires partitioning the ultra-dense millihertz gravitational-wave spectrum into manageable, parallel frequency segments. A comprehensive investigation into the Short-Time Fourier Transform (STFT) band-structure framework, executed across multiple phases and code audits in August 2026, resolved fundamental questions regarding window leakage, time-frequency track excursions, numerical differentiability of the response kernels, residual staleness in blocked Markov Chain Monte Carlo (MCMC) samplers, and the operational criteria governing parallel band boundaries.

---

## 1. Mathematical Formulation and Theoretical Framework

### 1.1 Multi-Scale Domain Representation

The STFT domain discretizes the observation time $T_{\mathrm{obs}}$ into $N_T$ segments of length $\Delta T_{\mathrm{segment}} = \mathrm{stft\_DT}$ (nominally 1 day), yielding an STFT frequency pixel resolution $df = 1/\Delta T_{\mathrm{segment}} = 1.1574 \times 10^{-5}\text{ Hz}$. This contrasts with the coherent resolution element of the full observation record, $\Delta f_{\mathrm{coh}} = 1/T_{\mathrm{obs}}$ (e.g., $4.225 \times 10^{-8}\text{ Hz}$ at $T_{\mathrm{obs}} = 0.75\text{ yr}$, and $3.171 \times 10^{-8}\text{ Hz}$ at $1\text{ yr}$). Consequently, one STFT frequency pixel spans:


$$N_{\mathrm{coh}} = \frac{T_{\mathrm{obs}}}{\Delta T_{\mathrm{segment}}} \approx 274 \text{ to } 365 \text{ coherent resolution elements}$$


which establishes that physical source overlap is governed by coherent sub-bin dynamics rather than pixel-scale proximity.

### 1.2 Five Width Conventions in the Sub-Bin Framework

The framework abandons the initial premise that frequency band boundaries must snap to STFT pixel edges. Parameter estimation operates over five distinct widths:

* **$A$ — Stencil Half-Width ($n_{\mathrm{side\_bins}}$):** An integer number of STFT pixels on either side of the instantaneous frequency track, governing transverse window leakage within each 1-day time column.


* **$B$ — Frequency Band Width ($W_{\mathrm{band}}$):** A continuous interval in parameter space ($\text{Hz}$) defining prior support and leaf assignment; it is strictly sub-pixel across $>96\%$ of the band below $10\text{ mHz}$.


* **$C$ — Store Window ($W_{\mathrm{store}}$):** An integer pixel window rounded strictly outward, defining the physical data and inverse-covariance memory buffer assigned to a processing unit.


* **$D$ — Colour Stride ($\mathrm{band\_units}$):** The parallel update stride, bounded between an odd/even parity stride ($\mathrm{band\_units} = 2$) and the maximum store-window clique separation ($\mathrm{band\_units} = \lceil W_{\mathrm{store}}/W_{\mathrm{band}} \rceil$).


* **$E$ — Store Tile:** An integer-pixel contiguous memory block shared across multiple adjacent sub-bin bands to avoid redundant buffer replication.



### 1.3 Residual Staleness and Principal Derivative Coupling

When parallel sampler units evaluate proposals against a frozen (stale) sibling residual state, the perturbation in the log-likelihood difference $\Delta \ln \mathcal{L}_j$ of source $j$ due to un-subtracted displacement $\Delta h_i$ of sibling source $i$ is governed by the cross-term $\mathrm{Re}\langle \Delta h_i \mid \Delta h_j \rangle$. Linearizing perturbations relative to the Fisher Information Matrix ($\Gamma$), where $v^T \Gamma v = 1$ defines a $1\sigma$ parameter displacement, the maximum coupling is bounded by the principal singular value:


$$s_{\max}(\Delta f) = \sigma_{\max}\left( \Gamma_i^{-1/2} C \Gamma_j^{-1/2} \right), \quad C_{ab} = \mathrm{Re}\langle \partial_a h_i \mid \partial_b h_j \rangle$$


The coupling $s_{\max} \in [0, 1]$ represents the cosine of the smallest principal angle between the 8-dimensional tangent spaces of the two binary models. It is strictly scale-invariant with respect to source signal-to-noise ratio (SNR) and reparameterization, bounded below by the orientation-maximized template overlap $F(\Delta f)$, and spans an exact 4D linear subspace in polarization parameters $\{H_+, iH_+, H_\times, iH_\times\}$.

### 1.4 Detailed Balance and Metropolis Decision Reversals

For a proposal with log-acceptance ratio $R = \ln(\mathcal{L}'/\mathcal{L}) + \ln(\pi'/\pi) + \ln(q/q')$ perturbed by staleness error $\epsilon_R$, the probability of an erroneous Metropolis-Hastings decision is $P(\mathrm{flip} \mid R) = \vert{}\min(1, e^{R + \epsilon_R}) - \min(1, e^R)\vert{}$. Averaged across downhill proposals ($R < 0$), the expected decision reversal fraction satisfies:


$$\text{Flip Fraction} \le \vert{}\epsilon_R\vert{} \times \langle \text{Acceptance Rate} \rangle$$


Targeting a detailed balance error fraction $\le 1\%$, the maximum allowable coupling enforces:


$$s_{\max} \le \frac{0.01}{\langle \text{Acceptance Rate} \rangle \cdot \sigma_i \cdot \sigma_j}$$

---

## 2. Numerical Diagnostics, Kernel Differentiability, and Production Code Audits

### 2.1 STFT Evaluator Non-Differentiability and the August 2026 Repair

Prior to August 26, 2026, finite-difference step scans of the STFT likelihood generator on the production path (`freq_from_tdi_phase = True`) revealed a failure to converge:

* Numerical frequency derivatives exhibited a relative difference floor of $1.6 \times 10^{-3}$ to $1.2 \times 10^{-2}$, while sky coordinate derivatives stalled at $5.2 \times 10^{-3}$ to $3.0 \times 10^{-2}$.


* Polynomial residual evaluations within sub-posterior frequency displacements revealed an artificial log-likelihood jitter of $0.28\text{ to }0.85\text{ nats RMS}$. Through the flip-fraction relation, this generated an intrinsic baseline false-decision rate of $\sim 6.5\%$, swamping the entire $1\%$ parallel tolerance budget.



The root cause was isolated to a catastrophic numerical cancellation in the per-pixel chirp rate $\dot{f}_0$ inside `lat_stft_kernels.hh`, where second differences of the Time-Delay Interferometry (TDI) response phase were evaluated across a temporal baseline sized strictly for first differences. On August 25–26, 2026, an analytic correction was implemented:

1. The numerical differentiation for $\dot{f}_0$ was replaced by the closed-form orbital Doppler derivative:



$$\dot{f}_0 = \dot{f}_{\mathrm{astro}} - \frac{f_0}{c} (\mathbf{k} \cdot \mathbf{a}_{\mathrm{sc}})$$



where $\mathbf{a}_{\mathrm{sc}}$ is the spacecraft acceleration.


2. The frequency estimate was demodulated across a stabilized $2000\text{ s}$ baseline.



Following this repair, log-likelihood jitter collapsed by 4 to 5.5 decades: to $8.6 \times 10^{-5}\text{ nats}$ at $0.81\text{ mHz}$, $4.0 \times 10^{-6}\text{ nats}$ at $3.26\text{ mHz}$, and $1.25 \times 10^{-6}\text{ nats}$ at $13.17\text{ mHz}$. Derivatives along $f_0$ now exhibit linear convergence ($\delta^1$) across three decades of step size down to $10^{-4}\text{ rad}$ of template phase. The associated false-decision rate fell to $\sim 10^{-5}$, removing the evaluator noise floor from the tolerance budget. Disabling `freq_from_tdi_phase` was formally rejected because omission of the orbital Doppler rate induces template mismatches of $1.03 \times 10^{-3}$, resulting in severe parameter biases ranging from $2.8\sigma$ at $0.81\text{ mHz}$ to $32\sigma$ at $13.17\text{ mHz}$.

### 2.2 Re-Parameterization of the Numerical Fisher Information Matrix

Earlier studies identified a complete absence of a finite-difference step stability plateau in the numerical Fisher Information Matrix (FIM) for $\{f_0, \dot{f}, \alpha, \sin\delta\}$ below $3\text{ mHz}$. Investigation revealed two distinct mechanisms:

1. **Astrophysical Information Horizon:** The total coherent frequency drift over the observation, $\vert{}\dot{f}\vert{} T_{\mathrm{obs}}^2$, drops below 1 resolution element for frequencies below $3\text{ mHz}$ (e.g., $3.0 \times 10^{-4}$ elements at $0.854\text{ mHz}$). The parameter $\dot{f}$ carries negligible astrophysical information in this regime, rendering the raw numerical derivative along $\dot{f}$ unconstrained and causing near-degeneracy with $f_0$.


2. **Basis Disparity:** Historical differentiation used scaling constants of $10^{-3}\text{ Hz}$ for $f_0$ and $10^{-16}\text{ Hz/s}$ for $\dot{f}$, creating a disparity of $3 \times 10^5$ coherent elements between the perturbations.



Transforming to dimensionless record units ($f_0$ scaled by $1/T_{\mathrm{obs}}$ and $\dot{f}$ scaled by $1/T_{\mathrm{obs}}^2$) established a stable, universal plateau across eight parameters over the step range $\varepsilon \in [10^{-3}, 10^{-2}]$. A single differentiation step of $\mathrm{FIM\_EPS} = 3 \times 10^{-3}$ in record units was standardized across the band. Furthermore, inverting the scaled correlation matrix $R$ via Cholesky decomposition ($\Gamma = D R D$) eliminated false singular-value clamping and removed historical anomalies where marginal uncertainties appeared smaller than conditional uncertainties ($\sigma_{\mathrm{marg}} < \sigma_{\mathrm{cond}}$).

### 2.3 Cross-Domain Reference Benchmarks and Algorithmic Defects

Cross-comparisons among STFT, Frequency-Domain (FD, `GBFDComputations`), and legacy time-domain (`GBGPU.run_wave`) likelihood evaluators identified critical domain limitations:

* **The E1 Approximation Floor:** Above $1\text{ mHz}$, the converged STFT domain matches FD computations to an SNR fractional difference of $1 - \mathrm{SNR}_{\mathrm{STFT}}/\mathrm{SNR}_{\mathrm{FD}} \le 5.8 \times 10^{-4}$ ($\Delta \ln \mathcal{L} \approx 16.6$ for an SNR $146.6$ source at $20\text{ mHz}$). Below $0.5\text{ mHz}$, the STFT template norm exhibits an intrinsic $4\%$ physical excess over legacy generators due to low-frequency breakdown of the 1-day column Taylor expansion.


* **Legacy Armlength Modeling:** The legacy generator assumed an equal-arm length of $2.5 \times 10^9\text{ m}$, differing by $0.27\%$ from the true mean link length of the Mojito orbits ($2.49335 \times 10^9\text{ m}$). Setting `StudyConfig.legacy_armlength = 2493162305.42235` reduced peak template mismatch from $2.64 \times 10^{-3}$ to an irreducible $2.57 \times 10^{-3}$, eliminating an artificial $1.99\sigma$ bias in $\dot{f}$.


* **FD Channel-Y Response Defect:** At $11.301\text{ mHz}$, `GBFDComputations` overestimated optimal SNR by $20\%$ (an SNR of $114.98$ versus $95.57$ in STFT and $95.52$ in legacy) and yielded $\langle d \mid h \rangle / \langle h \mid h \rangle = 0.5009$. Decomposition confirmed the error was isolated entirely to TDI channel Y (where legacy and FD SNR differed by a factor of 2.008 with an overlap of $+0.4989$) within a specific orientation region $(\iota \approx 1.57, \psi \approx \pi/6, \beta \approx 0.0\text{ to }0.6)$. Legacy `run_wave` was established as the primary truth reference whenever FD calculations exhibited this failure mode.



### 2.4 Production Bookkeeping and Kernel Audits

Static audits of `lisa-analysis-tools` (`gbspecialstretch.py`, `gbbands.py`, and `utility.py`) uncovered multiple dormant and active bookkeeping defects:

* **Dead Guard Code:** The function `get_groups_from_band_structure`, containing logic to reject proposals crossing band edges (`diff = 0` for `num_groups_base = 2`), has zero call sites across the repository and is entirely inert.


* **Edge Band Proposal Freezing:** In `gbspecialstretch.py`, `extra_bool = (band_inds < num_bands - 1) & (band_inds > 0)` unconditionally excludes the lowest (index 0) and highest frequency bands from receiving any proposals. Because `run_proposal` is shared across classes, this exclusion affects both in-model parameter updates and Reversible-Jump (RJ) birth/death moves. Sources in band 0 are permanently frozen.


* **Asymmetric Likelihood Check:** In `run_proposal`, start-of-block likelihood drift checks compute $\mathrm{check} = \delta - \vert{}\delta\vert{}$ where $\delta = \mathcal{L}_{\mathrm{recomputed}} - \mathcal{L}_{\mathrm{stored}}$. This formulation evaluates to identically zero for all $\delta \ge 0$, silently permitting arbitrarily underestimated stored likelihoods to pass without triggering synchronization. Furthermore, `start_diffs` is stored as an absolute magnitude but subtracted from a signed drift, corrupting drift warnings.


* **Dormant Properties:** The validation helper `BandSorter.special_index_check` is an uncalled `@property`, providing no runtime verification against index corruption.


* **Omission of Configuration Defaults:** Production parameter files leave `band_units` unset, running the hardcoded default of 2 by omission. Simultaneously, while documentation specified `num_repeat_proposals` between 100 and 200, the production STFT recipe explicitly hardcodes 500 repeat proposals while setting `stretch_probability = 0.0`, forcing moves exclusively through information-matrix Cholesky draws.



---

## 3. Single-Source Stencil Leakage, Kinematic Drift, and Store-Window Geometry

### 3.1 Analytic Leakage Kernel and the Window Knee

In the STFT domain, parameter-dependent log-likelihood power outside a central frequency bin is governed by the Fourier transform of the segment Tukey window $w(u)$ with taper parameter $\alpha$. The discrete power leakage distribution across offset $x$ (in STFT pixels) from fractional bin offset $\delta \in [0, 0.5]$ is:


$$K(x) = \frac{\vert{}W(x)\vert{}^2}{\int_0^1 w^2(u) \, du}, \quad R(n, \delta) = \sum_{m=-n}^n K(m - \delta), \quad \varepsilon(n, \delta) = 1 - R(n, \delta)$$


Poisson summation guarantees exact normalization: $\sum_{k \in \mathbb{Z}} K(k - \delta) = 1$.

The Tukey kernel transitions sharply at the taper knee frequency $x_{\mathrm{knee}} = 1/\alpha$ STFT pixels. For offsets $\vert{}x\vert{} < 1/\alpha$, power decays as $x^{-2}$ (rectangular window behavior); for $\vert{}x\vert{} > 1/\alpha$, constructive ramp cancellation accelerates power decay to $x^{-6}$. In production ($\alpha = 0.11574$, corresponding to a $5000\text{ s}$ taper on 1-day segments), the knee is located at $8.64\text{ pixels}$. The production choice $n_{\mathrm{side\_bins}} = 10$ sits directly on this transition knee, having paid the computational cost of the mainlobe without accessing asymptotic sidelobe suppression. Beyond the knee, leakage follows the closed-form scaling:


$$n_{\mathrm{side\_bins}} \ge 1.15 \left[ 10 \pi^2 \alpha^4 (1 - 5\alpha/8) \varepsilon \right]^{-1/5} \propto \alpha^{-4/5} \varepsilon^{-1/5}$$

| Stencil Half-Width ($n_{\mathrm{side\_bins}}$) | Worst-case $\varepsilon$ ($\alpha = 0.11574$) | $\varepsilon$ ($\alpha = 0.5$, $6\text{ h}$ taper) | Truncated SNR Loss ($\rho = 146.6$) |
| --- | --- | --- | --- |
| **2** (Class default) | $3.42 \times 10^{-2}$ | $3.24 \times 10^{-3}$ | $\Delta \ln \mathcal{L} \approx 319$<br> |
| **10** (Production) | $1.42 \times 10^{-3}$ | $1.89 \times 10^{-5}$ | $\Delta \ln \mathcal{L} \approx 15.1$<br> |
| **12** (Leakage Floor) | $\le 1.00 \times 10^{-3}$ | $7.21 \times 10^{-6}$ | $\Delta \ln \mathcal{L} \approx 10.7$<br> |
| **20** | $1.99 \times 10^{-5}$ | $1.20 \times 10^{-7}$ | $\Delta \ln \mathcal{L} \approx 0.21$<br> |
| **30** | $5.79 \times 10^{-6}$ | $< 10^{-8}$ | $\Delta \ln \mathcal{L} \approx 0.06$<br> |
| **60** (Converged) | $1.08 \times 10^{-7}$ | $< 10^{-10}$ | $\Delta \ln \mathcal{L} \approx 0.00$<br> |

The intra-segment chirp parameter $\mu = \dot{f} \Delta T_{\mathrm{segment}}^2$ reaches a maximum of $0.362\text{ pixels}$ across the entire Mojito prior. Stencil leakage is invariant to $\mu$ up to $\mu \approx 10$. Therefore, single-source leakage is flat across the frequency band.

### 3.2 Kinematic Drift and Store-Window Containment ($E3$)

While stencil width $A$ ($n_{\mathrm{side\_bins}}$) is frequency-independent, store-window width $C$ ($W_{\mathrm{store}}$) must accommodate the total time-frequency trajectory excursion driven by secular spin-down and orbital Doppler shifts:


$$\mathrm{Span}(f_0) = \max_{t} j_{\mathrm{carrier}}(t) - \min_{t} j_{\mathrm{carrier}}(t) \approx \frac{\vert{}\dot{f}_{\max}\vert{} T_{\mathrm{obs}} + 2 f_0 (v/c)}{df}$$


Above $1\text{ mHz}$, empirical carrier tracks match analytic kinematic predictions to within 1 pixel across 7,919 evaluated sources. At $22\text{ mHz}$, maximum carrier excursion reaches 78 pixels over 9 months and 265 pixels over 2 years.

Masking evaluations demonstrate that store-window residual truncation error ($E3$) drops to machine precision ($\vert{}R_{\mathrm{masked}}/R_{\mathrm{explicit}} - 1\vert{} \le 2.2 \times 10^{-16}$) exactly when the margin $w$ outside the track reaches $n_{\mathrm{side\_bins}}$:


$$W_{\mathrm{store}}(f_0) = \left\lceil W_{\mathrm{band}}(f_0) + 2\left( \mathrm{Reach}(f_0) + [n_{\mathrm{side\_bins}} + g] \, df \right) \right\rceil$$


The empirical guard term $g$ does not represent unmodeled window leakage; it accounts for reach-model error. Empirical evaluation across the 3,747 resolved sources establishes:

* $g = 3\text{ pixels}$ for $f_0 \ge 1\text{ mHz}$ (maximum observed reach shortfall of 2.97 pixels across the catalogue).


* $g = 5\text{ pixels}$ for $f_0 < 1\text{ mHz}$ (compensating for TDI-phase carrier tracking spikes at antenna-pattern nulls where carrier placement was historically derived from channel X alone).



### 3.3 Likelihood Surface Degradation ($D$) versus Ignored Power ($\rho^2\varepsilon$)

Evaluating stencil truncation across realistic data profiles revealed that absolute likelihood loss ($\Delta \ln \mathcal{L} \approx \frac{1}{2} \rho^2 \varepsilon$) is not monotone in crowded regimes, as widening stencils incorporate unmodeled power from neighboring sources. The primary metric for stencil adequacy was established as likelihood surface distortion $D(n)$, measured as the 95th percentile residual across parameter slices relative to a converged ($n = 60$) stencil:


$$D(n) = P_{95,\theta} \left\vert{} \left[\ln \mathcal{L}_n(\theta) - \langle \ln \mathcal{L}_n \rangle\right] - \left[\ln \mathcal{L}_{60}(\theta) - \langle \ln \mathcal{L}_{60} \rangle\right] \right\vert{}$$

Evaluating $D(n)$ on catalogue GBs demonstrates that the production setting $n_{\mathrm{side\_bins}} = 10$ fails a target threshold of $D < 0.1$ for median sources in the confusion region ($D(10) = 0.218\text{ median}$, $0.742\text{ max}$). Satisfying $D < 0.1$ requires $n_{\mathrm{side\_bins}} = 16\text{ to }20$ below $5.5\text{ mHz}$, whereas $n_{\mathrm{side\_bins}} = 6$ suffices above $11\text{ mHz}$. For loud sources (e.g., $f_0 = 5.168\text{ mHz}$, $\rho = 320.8$), $n = 10$ leaves an un-subtracted residual power $\rho^2 \varepsilon = 128.6$ ($\mathrm{SNR} = 11.3$), generating a false ghost source above the detection threshold ($\rho \ge 7$). In noiseless simulations, isolated sources require only $n = 4\text{ to }6$ to achieve $D < 0.01$, proving that the operational requirement of $n = 16\text{ to }20$ is dictated by confusion from adjacent Galactic sources rather than single-source waveform modeling.

---

## 4. Population Dynamics, Multiplet Occupancy, and Coherent Separations

### 4.1 Covariate Independence and Occupancy Scaling

Evaluation across 3,747 resolved sources and 1.1 million confusion foreground binaries shows that fractional leakage $\varepsilon(10) \approx 1.32 \times 10^{-3}$ is completely flat across all astronomical parameters (SNR, frequency, chirp rate, latitude, inclination, local source density) with less than $3\%$ total variation. Conversely, surface degradation $D(10)$ scales with local confusion density:


$$D(10) \propto \rho^{-0.58} \, (\text{density})^{+0.31}, \quad R^2 = 0.622$$


Because Galactic binary spatial distribution locks local density to carrier frequency with a Spearman rank correlation of $-0.998$, keying stencil allocation to frequency identically captures Galactic confusion density.

### 4.2 Coherent Correlation Lengths

Phase-maximized template overlaps $\vert{}\langle h_i \mid h_j \rangle\vert{} / \sqrt{\langle h_i \mid h_i \rangle \langle h_j \mid h_j \rangle}$ evaluated over 40,000 source pairs demonstrate that the interaction distance in frequency space is extremely short:

* For the identical physical source displaced in frequency, overlap drops below $0.1$ at $3.02\text{ coherent elements}$ ($0.011\text{ STFT pixels}$).


* For distinct physical sources with independent sky locations and orientations, overlap drops below $0.1$ at $0.25\text{ coherent elements}$ ($0.0009\text{ STFT pixels}$).



Across the entire 9-month resolved catalogue, only 12 source pairs in the entire Galaxy exhibit an overlap greater than $0.1$. Binary sources sharing a sub-bin band do not interact destructively or form degeneracies, removing in-band parameter correlations as a design constraint.

### 4.3 In-Model MCMC Dimensionality

Ensemble MCMC tests (Eryn) sampling up to $k = 50$ real sources within a single $24\text{ pixel}$ sub-band at $3.5\text{ mHz}$ (a 400-dimensional in-model parameter space) confirmed:

* The integrated autocorrelation time $\tau$ remained completely flat between $k = 8$ ($\tau = 206.7$) and $k = 50$ ($\tau = 205.6$).


* Zero leaf misattributions occurred across all walkers; every source tracked its injected mode.


* Computational cost per effective sample scaled strictly as $\mathcal{O}(k)$.



---

## 5. Frequency Band Partitioning and Residual Staleness Tolerances

### 5.1 The Standardized Bandwidth Recipe ($W_{\mathrm{band}} = 4 B_q^{\mathrm{raw}}$)

To ensure that prior boundaries contain the true source parameters while maintaining adequate separation, the band partition is standardized to:


$$W_{\mathrm{band}}(f_0) = 4 \cdot B_q^{\mathrm{raw}}(f_0)$$


where $B_q^{\mathrm{raw}}$ is the raw, uncalibrated coherent containment width holding $q = 0.99$ of the power, evaluated at the maximum prior chirp rate:


$$B_q^{\mathrm{raw}}(f_0) = 2 A(f_0) \sin(q \Theta_{\mathrm{eff}}) + q \vert{}\dot{f}_{\max}\vert{} T_{\mathrm{obs}} + \sqrt{(2 n f_m)^2 + (W(q, \alpha)/T_{\mathrm{obs}})^2}$$


evaluated with $\sin\theta = 1$ and $\alpha = 0.11574$. Setting $k = 4$ absorbs physical safety margins, eliminates unphysical step discontinuities present in earlier formulations at $1.52\text{ mHz}$ and $7.07\text{ mHz}$, and produces smooth band counts:

* **$182\text{ d}$ ($0.5\text{ yr}$):** 1,078 bands.


* **$365\text{ d}$ ($1.0\text{ yr}$):** 1,555 bands, with a median width of $0.399\text{ pixels}$ ($140\text{ coherent elements}$) below $10\text{ mHz}$ and maximum occupancy of 17 sources per band.


* **$714\text{ d}$ ($1.95\text{ yr}$):** 1,898 bands.



### 5.2 Residual Staleness Limits

The empirical conversion factor from orientation-maximized overlap to derivative coupling $s_{\max}$ is $2.07$ (reduced from an un-repaired value of $3.36$). Balancing staleness accumulation against an empirical Metropolis decision flip target $\le 1\%$ sets strict limits on the number of repeat proposals permitted between residual updates:

| Parameter Region | Empirical Acceptance Rate | Tolerable Staleness | Max Proposals (Post-Repair) | Max Proposals (Pre-Repair) |
| --- | --- | --- | --- | --- |
| **Pooled Population** | 0.109 | $3.72\sigma$ | **127**<br> | 79

 |
| **$2\text{ mHz}$ (Confusion)** | 0.006 | $4.00\sigma$ (capped) | **2,670** (Unconstrained)

 | 2,670

 |
| **$5\text{ mHz}$** | 0.156 | $3.11\sigma$ | **62**<br> | 38

 |
| **$12\text{ mHz}$** | 0.259 | $2.42\sigma$ | **23**<br> | 14

 |

Production recipes execute between 100 and 500 repeat proposals per block. Above $5\text{ mHz}$, this places the sampler in a fully decorrelated staleness regime ($>4\sigma$), exceeding the allowable update budget by factors of $1.6\text{ to }8.7$ (and up to $21.7\times$ in the STFT recipe).

### 5.3 Low-Frequency Non-Monotone Coupling Revival Bottleneck

At $0.73\text{ mHz}$ and $0.81\text{ mHz}$, evaluating the repaired production path on a dense 289-separation axis revealed a non-monotone revival in $s_{\max}$. The coupling drops below the $1\%$ threshold near $100\text{ coherent elements}$, but revives to exceed tolerance between $170\text{ and }244\text{ elements}$ ($s_{\max} = 0.096\text{ to }0.124$) due to sky-dependent Doppler derivative terms $\dot{f}_0(\mathbf{k})$ sharing tangent space directions. Coupling permanently falls below tolerance only past $274.5\text{ coherent elements}$.

Against a $W_{\mathrm{band}} = 4 B_q^{\mathrm{raw}}$ bandwidth of $95\text{ to }97\text{ elements}$, the layout margin drops to $0.35$ at $0.73\text{ and }0.81\text{ mHz}$. Eliminating this bottleneck requires increasing band widths below $1\text{ mHz}$ by a factor of $2.9$ ($k \approx 11.5$), merging the 315 low-frequency bands (which hold only 34 resolved sources) into wider partitions.

### 5.4 The Periodic Taper Comb

Periodic segment windowing induces overlap revivals at integer-pixel separations with an analytic floor of $\vert{}c_n(w^2)\vert{}/c_0 = 0.0780$. However, spatial antenna patterns suppress empirical coupling on these comb harmonics by a factor of $5.0$ ($s_{\max} \le 0.0419\text{ median}$, $0.0768\text{ max}$). The resulting comb-induced decision flip rate is bounded between $0.46\%\text{ and }1.08\%$, confirming that comb revivals do not compromise detailed balance or require altering the segment taper.

---

## 6. Slicing, Buffer Memory, and Sibling Residual Models

### 6.1 Buffer Memory Slicing

In historical implementations, each STFT band allocated memory across the full active grid ($N_T = 273$, 3 channels, complex128 residual and covariance), requiring $103.8\text{ MiB}$ per band or $203\text{ GiB}$ across a 2,006-band partition. Slicing residual and covariance buffers to the local store window ($W_{\mathrm{store}} \approx 33\text{ pixels}$) reduces per-band memory to $1.65\text{ MiB}$:

* For $k = 4$ (1,555 bands at 1 yr), resident sliced memory is $2.6\text{ GiB}$.


* The store tile architecture, sharing overlapping store windows across adjacent sub-bands, further reduces resident storage to $\sim 0.1\text{ GiB}$.



Slicing eliminates GPU device memory as a constraint on parallel band counts.

### 6.2 Frozen versus Removed Sibling Residual States

When a color class opens, production bookkeeping removes cold-chain sources from the residual buffer, leaving sibling models absent ($h_i$) rather than frozen at their starting parameters ($\delta h_i$). Evaluating the staleness error term demonstrates that the removed-model error $\langle h_i \mid \delta h_j \rangle$ exceeds the frozen-model error $\langle \delta h_i \mid \delta h_j \rangle$ by a median factor of $3.9\text{ to }6.2$ (with extremes reaching 14). While this inflation is significant, it does not overturn the stability of the sampler, and the log-likelihood error remains within $0.08\sigma$ at the 90th percentile.

---

## 7. Operational Verification: The 365-Day `band_units` Sweep

### 7.1 Empirical Null Result Across the Stride Axis

On August 31, 2026, an end-to-end bookkeeping scan was executed on the 365-day leg (`layout_365d_k4_midpoint.npz`, 1,555 bands, 300 test blocks) using the corrected parameter epoch ($t_0 = \mathrm{DATA\_T0}$, incorporating the exact single $720850.5\text{ s}$ lag). The driver evaluated parallel strides across $\mathrm{band\_units} \in \{2, 3, 4, 8, 124\}$ against live refreshed references using shared random numbers at production's repeat block length ($N_{\mathrm{repeat}} = 500$):

| `band_units` | Parallel Bands Active | Nearest Cross-Band Sibling | Disp. $p_{90}$ ($\sigma$) | Disp. Max ($\sigma$) | Width Ratio ($p_{10}\text{--}p_{90}$) | Flip Fraction |
| --- | --- | --- | --- | --- | --- | --- |
| **2** | 778 | 2 bands | $0.082$ | $0.83$ | $0.87\text{ to }1.15$ | $0.0291 \pm 0.0034$<br> |
| **3** | 518 | 3 bands | $0.087$ | $0.90$ | $0.88\text{ to }1.14$ | $0.0309 \pm 0.0026$<br> |
| **4** | 389 | 4 bands | $0.069$ | $1.06$ | $0.90\text{ to }1.15$ | $0.0264 \pm 0.0025$<br> |
| **8** | 194 | 8 bands | $0.081$ | $1.00$ | $0.88\text{ to }1.14$ | $0.0293 \pm 0.0025$<br> |
| **124** (Disjoint Max) | 13 | 124 bands | $0.080$ | $0.90$ | $0.90\text{ to }1.11$ | $0.0267 \pm 0.0019$<br> |

Across a 62-fold expansion in spatial stride, posterior displacement $p_{90}$ remained locked between $0.069\sigma\text{ and }0.087\sigma$, while decision flip fractions spanned $0.0264\text{ to }0.0309$ (indistinguishable within the $1\sigma$ standard error of $\pm 0.0025$). Setting $\mathrm{band\_units} = 124$ isolates store windows completely, proving that **cross-band residual staleness has zero measurable impact on posterior quality, detailed balance, or MCMC convergence**.

### 7.2 Intra-Band Staleness as the Dominant Mechanism

The origin of the observed decision flips was isolated by comparing pools of cross-band and intra-band siblings:

* Replacing 6 cross-band siblings with 6 intra-band band-mates (12 intra-band leaves, 0 cross-band) increased the flip fraction from $0.0291 \pm 0.0034$ to $0.0507 \pm 0.0073$ ($1.74\times$) and increased displacement $p_{90}$ from $0.082\sigma$ to $0.135\sigma$ ($1.65\times$).


* Sources sharing a band share a colour class at every stride, remaining un-updated with respect to each other throughout the repeat block.



The residual staleness in the sampler is driven entirely by intra-band crowding, not cross-band leakage. Modifying $\mathrm{band\_units}$ cannot mitigate this effect. The physical lever governing sampler staleness is band width $W_{\mathrm{band}}$ (reducing source occupancy per band), not the parallel stride.

### 7.3 Diffusive Chain Regimes

At a proposal scale of $\mathrm{jump\_factor} = 0.005$ to $0.02$, a random-walk Metropolis chain requires $\sim 8,000\text{ proposals}$ per autocorrelation time. Within a 500-proposal block, the integrated autocorrelation time scales linearly with block length ($\tau / N_{\mathrm{repeat}} \approx 0.10$), indicating that individual blocks represent diffusive excursions exploring $0.1\text{ to }0.4\sigma$ around injection rather than converged equilibrium posteriors. Flip fractions over 500 proposals measure trajectory divergence on shared random numbers rather than single-step detailed balance violations. Analytic calculation of the shift $\Delta \theta = \Gamma^{-1} \nabla \Delta \ln \mathcal{L}$ provides the definitive equilibrium posterior bias, while acceptance inflation between stale and refreshed chains remains bounded at $\le 0.00306$, validating acceptance as an unbiased ranking statistic.

---

## 8. Synthesis of Framework Decisions and Operational Rules

```
+----------------------------------------------------------------------------------------------------+
|                                    BAND STRUCTURE SPECIFICATION                                    |
+------------------------------------+---------------------------------------------------------------+
| Quantity                           | Operational Value / Formulation                               |
+------------------------------------+---------------------------------------------------------------+
| Target Frequency Bandwidth         | W_band(f_0) = 4 * B_q^raw(f_0) [continuous interval, Hz]      |
| Band Count (365-day baseline)      | 1,555 bands (k = 4, q = 0.99, alpha = 0.11574)                |
| Store Window Sizing                | W_store(f_0) = ceil_outward[ W_band + 2(Reach + (n_side+g)df)]|
| Stencil Allocation                 | n_side_bins = 16-20 (f_0 < 5.5 mHz); n_side_bins = 6 (> 11)   |
| Store-Window Guard Margin          | g = 3 pixels (f_0 >= 1 mHz); g = 5 pixels (f_0 < 1 mHz)       |
| Parallel Color Stride              | band_units = 2 (odd/even parity, 778 bands parallel)          |
| In-Model Proposal Limit            | num_repeat_proposals <= 23 (at 12 mHz); <= 62 (at 5 mHz)      |
| Low-Frequency Band Merge           | Expand W_band by 2.9x (k ≈ 11.5) below 1.0 mHz               |
| Numerical Differentiation Basis    | FIM_EPS = 3e-3 in dimensionless record units (1/T_obs, 1/T_2) |
| Covariance Inversion Scheme        | Direct Cholesky factorization of correlation matrix R         |
+------------------------------------+---------------------------------------------------------------+

```

The investigation definitively establishes that the production default $\mathrm{band\_units} = 2$ is optimal. It yields maximal parallel concurrency (778 simultaneous band updates) without incurring any statistical or posterior-quality penalty relative to disjoint partitioning.

Efforts to safeguard detailed balance must focus on enforcing the proposal refresh budget ($\le 23\text{ to }62$ repeats at high frequencies), repairing the start-of-block likelihood checks in `gbspecialstretch.py`, eliminating the edge-band proposal exclusion on band 0, and merging low-frequency partitions below $1\text{ mHz}$ to clear the $0.81\text{ mHz}$ Doppler coupling revival. Slicing buffer allocations to local store windows resolves all GPU memory limits, providing a computationally efficient and statistically rigorous foundation for the global fit of Galactic binaries in the LISA STFT data stream.