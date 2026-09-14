The determination of the minimum initial gravitational wave (GW) frequency separation $\Delta f_{\min}$ required to resolve overlapping Galactic Binaries (GBs) in LISA data analysis depends on the geometric properties of the GB parameter manifold, the windowing structure of Short-Time Fourier Transform (STFT) evaluators, and the track-crossing kinematics of modulated signals.

## Parameter Manifold Maximization and Extrinsic Symmetries

The fundamental quantity governing source confusion is the fitting factor $\mathcal{F}(f_0; \Delta f) = \max_{\theta_2} \mathcal{O}(\theta_1, \theta_2)$ evaluated at a constrained frequency offset $f_{0,2} = f_{0,1} + \Delta f$. Matched extrinsic parameters ($\theta_2 = \theta_1$) do not maximize the constrained slice. Because metric correlations $g_{f_0 a}$ between frequency and extrinsic parameters (ecliptic coordinates $(\beta, \lambda)$ and chirp rate $\dot{f}$) are non-vanishing, matched parameters represent a saddle point on the constrained manifold. The marginal mismatch,

$$M^\star = \frac{1}{2} \Delta f^2 \left( g_{ff} - \frac{g_{fa}^2}{g_{aa}} \right) < \frac{1}{2} g_{ff} \Delta f^2,$$

allows a secondary template to trade frequency offsets against Doppler and chirp compensations, broadening the effective confusion region by factors up to $\sim 100$ relative to conditional slices.

Orientation parameters $(\iota, \psi)$, initial orbital phase $\varphi_0$, and amplitude $A$ can be eliminated analytically. The single-channel TDI response is strictly linear in the two polarization states $w_+ = A_+\cos 2\psi + iA_\times\sin 2\psi$ and $w_\times = A_+\sin 2\psi - iA_\times\cos 2\psi$. Overlap maximization over all physical orientations corresponds to evaluating the maximum generalized singular value of the $2 \times 2$ bilinear form $M_{ij} = \langle C_i^{(1)} \vert{} C_j^{(2)} \rangle$ with respect to the individual Gram matrices $G_1, G_2$. Because the physical Möbius projection $z = \tan 2\psi - i(A_\times/A_+)$ spans the entire complex ray space without restriction, the unconstrained singular value is exact and reduces a 1,600-point grid search per binary pair to six kernel evaluations.

## The Taper-Train Comb and Segmental Windowing

In STFT-based likelihood evaluations with segment duration $\mathrm{STFT\_DT}$ and total observation time $T_{\rm obs} = N_T\,\mathrm{STFT\_DT}$, the discrete Fourier sum over segments behaves via Parseval's identity as a time integral over the **squared** taper window $w(u)^2$:

$$\mathcal{O}(x) = \frac{\hat{W}^2(x)}{\hat{W}^2(0)} \times \left\vert{} D_{N_T}(x) \right\vert{}, \qquad x = \Delta f \cdot \mathrm{STFT\_DT},$$

where $D_{N_T}(x)$ is the Dirichlet kernel and $\hat{W}^2(x)$ is the Fourier transform of the squared Tukey window.

```
Overlap Envelope O(x)
 ^
1|=================\ (Smooth Dirichlet Main Lobe ~ 1/N)
 |                  \
 |                   \      Comb Plateau ~ (5α/8)/(1 - 5α/8)
 |--------------------\---------------------------------------- (Comb Revivals at x = n)
 |                     \    /\        /\        /\
 |                      \  /  \      /  \      /  \
0+-----------------------v/----v\----/----v\----/----v\--------> x = Δf · STFT_DT [pixels]
 0                       1      2         3         4

```

* **Periodic Comb Revivals:** The periodic repetition of window ramps generates discrete spectral revivals at integer STFT pixels ($n = \Delta f \cdot \mathrm{STFT\_DT}$). The squared Tukey deficit $1 - w^2$ contains a three-term cosine expansion yielding exact Fourier coefficients:



$$c_n(w^2) = -\frac{2\sin(n\pi\alpha)}{\pi n} \left[ \frac{5}{16} + \frac{(n\alpha/2)^2}{1 - (n\alpha)^2} + \frac{(n\alpha/2)^2}{16[1 - (n\alpha/2)^2]} \right], \qquad c_0(w^2) = 1 - \frac{5\alpha}{8}.$$


* **Plateau and Knee Scaling:** For $n \lesssim 1/\alpha$, revival amplitudes reside on a flat plateau $O_{\rm comb} \approx (5\alpha/8) / (1 - 5\alpha/8)$. Decay occurs only past the knee $n \approx 1/\alpha$. The comb structure is strictly scale-invariant in pixel units across differing segment durations.


* **Tolerance Axis Partitioning:** For production taper $\alpha = 0.11574$, the comb plateau enforces a floor at $\varepsilon \approx 0.078$. For overlap tolerances $\varepsilon > 0.078$, $\Delta f_{\min}$ is governed by the smooth intra-pixel Dirichlet envelope; for $\varepsilon < 0.078$, no separation within one pixel can satisfy the criterion, making $O_{\rm comb}(n, \alpha)$ the sole binding constraint.


* **Band-Layout Invariance:** In production configurations, the taper fraction is constrained by the lower boundary of the search band via $\alpha = 1 / (f_{\rm start} \mathrm{STFT\_DT})$. Consequently, comb suppression cannot be tuned arbitrarily without shifting $f_{\rm start}$ or segment length, imposing a hard structural constraint where bands cannot start below one STFT bin ($f_{\rm start} \ge 1/\mathrm{STFT\_DT}$).



## Track-Crossing Kinematics and Confusion Scaling

When maximizing over intrinsic and sky parameters, the dominant mechanism inducing high template correlation at wide separations is stationary relative phase: a **track crossing** in the time-frequency plane within the observation duration $T_{\rm obs}$. At a stationary phase point $t^\star = -\Delta f / \Delta \dot{f}$, the overlap integral decays according to the stationary-phase asymptotic scaling $\mathcal{F} \approx c(2N)^{-1/2}$ (where $N = \Delta f T_{\rm obs}$) rather than the standard Dirichlet $N^{-1}$.

```
Frequency f(t)
 ^
 |             Track 1: f_1(t) = f_0 + fdot_1 · t
 |                   \          /
 |                    \        /  Stationary Crossing Point (t*)
 |                     \      /   Relative phase dΦ/dt = 0
 |                      \    /    Overlap decays as N^(-1/2) instead of N^(-1)
 |                       \  /
 |                        \/
 |                        /\
 |                       /  \
 |                      /    \
 |                     /      \   Track 2: f_2(t) = (f_0 + Δf) + fdot_2 · t
0+--------------------+--------+---------------------------------------------> Time t
 0                   t*       T_obs

```

The maximum accessible parameter space sets two closed-form physical reaches in coherent elements ($1/T_{\rm obs}$):

$$\text{Differential Doppler Sky Reach: } R_{\rm sky} = 2\pi f_0 \frac{R_\oplus}{c},$$

$$\text{Chirp Prior Reach: } R_{\dot{f}} = \vert{}\Delta \dot{f}\vert{}_{\max} T_{\rm obs}^2.$$

* **Regime Transitions:** Below $f_0 \approx 6\text{ mHz}$, $R_{\rm sky}$ dominates and scales linearly with $f_0$, spanning $3.1$ to $18.8$ coherent elements. Above $6\text{ mHz}$, the mass-transfer astrophysical chirp prior $\vert{}\Delta \dot{f}\vert{}_{\max} \propto f_0^{16/3}$ rapidly expands $R_{\dot{f}}$ from $24.6$ to $11,945$ coherent elements at $20\text{ mHz}$.


* **Three-Branch Model:** The global separation bound is parameterized by:

$$\Delta f_{\min}(\varepsilon) = \max \left[ \Delta f_{\rm window}(\varepsilon),\, \min\left(\frac{c^2}{2\varepsilon^2},\, R_{\rm sky}\right),\, \min\left(\frac{c^2}{2\varepsilon^2},\, R_{\dot{f}}\right) \right],$$



where joint numerical maximization yields $c \approx 2.49$. At $\varepsilon = 0.1$, $\Delta f_{\min}$ increases by a factor of 40 across the LISA band ($7.6 \to 309.5$ coherent elements) before saturating at the tolerance ceiling $c^2/(2\varepsilon^2)$.


* **Pattern-Phase Perturbation:** Modulations in the transfer and antenna pattern phases $d/dt(\Phi_{\rm transfer} + \Phi_{\rm ant})$ introduce an annual sky-dependent frequency perturbation bounded strictly at $\le 2.98$ coherent elements. This acts as an $\mathcal{O}(10\%)$ perturbation between $2\text{--}7\text{ mHz}$ rather than an independent third reach.


* **Observation Time Scaling Failure:** While theoretical reaches scale as $T_{\rm obs}$ (sky) and $T_{\rm obs}^2$ (chirp), empirical measurements across $182\text{ d}$, $365\text{ d}$, and $714\text{ d}$ demonstrate that $\Delta f_{\min}$ scales as $T_{\rm obs}^{1.2\text{--}1.4}$. This occurs because $\Delta f_{\min}$ operates on the unsaturated $1/\varepsilon^2$ crossing branch rather than at the asymptotic prior reach boundary.



## Evaluator Numerical Floors and Likelihood Conversions

STFT Fresnel column approximations introduce discrete numerical artifacts that govern the evaluator noise floor:

| Evaluator Phenomenon | Physical Origin | Impact & Operational Mitigation |
| --- | --- | --- |
| **V-Shaped Evaluator Floor**<br> | Competition between truncation error at small stencil half-width $n$ and Fresnel expansion failure at large $n$.

 | Optimal stencil width scales as $n^\star \approx 0.5 f_0 \cdot \mathrm{STFT\_DT}$; floor reaches $\varepsilon \sim 10^{-11}$ at high $f_0$.

 |
| **Polarization Non-Linearity**<br> | Column expansion points keyed to orientation-dependent TDI phase $\phi_{\rm TDI}(t)$.

 | Induces an absolute floor ($1.5 \times 10^{-4}$ at $6\text{ mHz}$, $2.6 \times 10^{-2}$ at $1.01\text{ mHz}$); bypassed by re-evaluating at stationary maximum.

 |
| **Linear Envelope Instability**<br> | Singularities in $(f_0 - f)/\dot{f}_0$ and $1/\sqrt{\vert{}\dot{f}_0\vert{}}$ as column curvature vanishes near Doppler turnarounds.

 | Generates spurious factors up to $8.6\times$; locked to `linear_envelope = False`.

 |
| **Chirp-Rate Demodulation Defect**<br> | Catastrophic cancellation in pre-repair per-column $\dot{f}_0$ carrier computation.

 | Repaired kernel reduces relative template noise by $348\text{--}5778\times$, restoring theoretical segment-sum suppression at $1.01\text{ mHz}$.

 |

When converting bare overlap thresholds $\varepsilon$ to parameter bias limits $\delta\theta \le b\,\sigma$ or likelihood perturbations $\Delta \ln \mathcal{L} \le L$, the condition depends on the source signal-to-noise ratio: $\varepsilon_{\rm target} \approx \sqrt{2L}/\rho_2$. For the iteratively resolved LISA population ($\rho_{\max} \approx 365$), achieving $\Delta \ln \mathcal{L} \le 0.1$ requires $\varepsilon \lesssim 1.2 \times 10^{-3}$, placing the entire resolved population below the operational comb plateau.

Evaluations of the metric template difference $\langle \Delta h_i \vert{} \Delta h_j \rangle$ confirm that the bare overlap surface acts as a robust proxy for the derivative-space Fisher metric within a factor of $1.0\text{--}2.7$ across the entire observing band.

For global-fit frequency band layouts ($W_{\rm band} = k B_q$), empirical track-crossing widths remain bounded by $k \ge 2$, with the tightest margin (factor $\approx 4.4$) localized mid-band at $8.67\text{ mHz}$. Absorbing discrete regime-dependent safety factors into a uniform $k = 4$ scaling provides continuous, monotone containment across the full spectrum.