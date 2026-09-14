The spectral extent of quasi-monochromatic galactic binary (GB) white dwarf signals within LISA Time-Delay Interferometry (TDI) channels is governed by a combination of orbital kinematics, secular frequency drift, constellation antenna patterns, and data analysis windowing.

**Physical and Instrumental Broadening Mechanisms**

* **Doppler Frequency Modulation:** The heliocentric motion of the LISA guiding center at orbital modulation frequency $f_m = 1/\text{year} \approx 3.17 \times 10^{-8}\text{ Hz}$ introduces a Doppler shift of half-amplitude $A = 2\pi f_0 (R/c) f_m \sin\theta$, where $\theta$ is the ecliptic co-latitude and $R/c \approx 499.0\text{ s}$. Because worst-case observation windows straddle line-of-sight velocity zero-crossings, the Doppler excursion $2A\sin(q\Theta_{\rm eff})$ saturates at an observation duration of $T_{\rm obs} = 0.5\text{ yr}$ ($\Theta_{\rm eff} = \min(\pi f_m T_{\rm obs}, \pi/2)$), rather than $1.0\text{ yr}$.


* **Secular Frequency Drift:** Gravitational-wave radiation reaction drives inspiral ($\dot{f} > 0$) parameterized by chirp mass $\mathcal{M}_c$, bounded by the Chandrasekhar limit ($\mathcal{M}_c \le 1.254 M_\odot$). Conversely, dynamically stable Roche Lobe Overflow (RLOF) in interacting AM Canum Venaticorum (AM CVn) binaries produces orbital outspiral ($\dot{f} < 0$) bounded empirically by $\vert{}\dot{f}\vert{} \propto f_0^{16/3}$. Mass-transfer stability cutoffs occur at $14.1\text{ mHz}$ (weak tidal coupling, $q_{\rm crit} = 0.20$) or $82.8\text{ mHz}$ (strong tidal coupling, $q_{\rm crit} = 2/3$), above which the negative-chirp branch ceases.


* **Window Floor and Sideband Structure:** Annual amplitude modulation of the detector antenna pattern injects sideband power dominated by the second harmonic ($n = 2$, full width $2nf_m$). Simultaneously, finite-duration segment tapering (Tukey window with shape parameter $\alpha$) imposes spectral leakage $W(q,\alpha)/T_{\rm obs}$.



**Frequency Regimes and Dominant Contributions**

* **$f_0 \lesssim 3\text{ mHz}$ (Window/Floor Dominated):** Apparent signal bandwidth is dictated by analysis choices ($\alpha$, power containment level $q$) rather than source astrophysics. Untapered records ($\alpha = 0$) assign a monochromatic source an apparent $q = 0.999$ width exceeding $200$ coherent Fourier bins ($1/T_{\rm obs}$), whereas production tapers ($\alpha \approx 0.02\text{--}0.25$) suppress leakage by up to two orders of magnitude.


* **$3\text{ mHz} \lesssim f_0 \lesssim 10\text{ mHz}$ (Doppler Dominated):** Kinematic Doppler modulation dominates, converging with the classical Cornish & Larson (2003) annual formulation. At $q = 0.99$, power containment agrees with the unperturbed kinematic ridge to within $0.7\%$.


* **$f_0 \gtrsim 10\text{ mHz}$ (Chirp Dominated):** Frequency evolution $q\vert{}\dot{f}\vert{}T_{\rm obs}$ dominates over Doppler modulation. Because negative drift rates exceed radiation-driven positive rates in extreme systems, the containment interval $[f_{\min}, f_{\max}]$ becomes highly asymmetric and extends up to three times further below $f_0$ than above it.



**Closed-Form Bandwidth Formulation**

The total frequency band encompassing a power containment fraction $q$ is expressed analytically as:

$$B_q = \mathcal{F} \left[ 2A\sin(q\Theta_{\rm eff}) + q\vert{}\dot{f}\vert{}T_{\rm obs} + C(q,\alpha,T_{\rm obs})\sqrt{(2nf_m)^2 + \left(\frac{W(q,\alpha)}{T_{\rm obs}}\right)^2} \right]$$

where $C$ is a fitted correction factor accounting for the discrete sideband comb, and $\mathcal{F} \in [1.02, 5.78]$ is a strict per-regime safety factor.

Because the instantaneous frequency is a deterministic 1D function of time, range sub-additivity ($\mathrm{range}(g+h) \le \mathrm{range}(g) + \mathrm{range}(h)$) ensures that a linear sum of Doppler, drift, and floor terms constitutes a rigorous upper bound, whereas quadrature addition systematically under-predicts the measured band in crossover regimes.

**Detector Response and Parameter Maximization**

* **Containment vs. Kinematic Ridge:** Tracking kinematic frequency via $\frac{1}{2\pi}\frac{\mathrm{d}\phi}{\mathrm{d}t}$ is ill-conditioned at antenna pattern nulls due to rapid $\pi$-phase transitions. Frequency-domain power containment operating on baseband-demodulated TDI channels avoids amplitude-mask thresholds and phase unwrapping altogether.


* **Unequal Breathing Armlengths:** Realistic orbit models (e.g., L1, ESA trailing) do not broaden the intrinsic line width at a fixed sky location relative to equal-armlength approximations; rather, time-varying breathing shifts the sky positions of deep antenna nulls, altering the sky-maximized envelope by up to a factor of $2.8$ under narrow tapers.


* **Orientation and Sky Margins:** The TDI response is exactly linear in the two polarization states, enabling orientation ($\iota, \psi$) maximization from two waveform evaluations. Maximizing over sky position and orientation inflates the required bandwidth by a factor of $8.6$ for $T_{\rm obs} = 0.25\text{ yr}$, but decays to a $13\%$ penalty at $T_{\rm obs} = 4\text{ yr}$ due to annual antenna-pattern averaging.