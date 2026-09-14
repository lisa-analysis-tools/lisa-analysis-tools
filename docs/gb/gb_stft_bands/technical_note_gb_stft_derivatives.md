# Technical Note: Stabilization, Closed-Form Kinetics, and Optimization of the STFT/Fresnel Likelihood Pipeline

**Status:** Validated and locked in production (`GBGPU` / `lat_stft_kernels.hh`). This note documents the root causes of the STFT derivative breakdown, the exact closed-form mechanics implemented to resolve them, the empirical metrics across the Galactic Binary band ($0.35\text{--}20.4\text{ mHz}$), and the physical and mathematical reasons governing boundary guards and parameter settings.

---

### 1. Root Cause Analysis of the Derivative Breakdown

The failure of finite-difference derivatives (log-log Taylor slope $R(\delta) \propto \delta^{0.03}$ instead of $\delta^2$) and the $0.27\text{--}0.87\text{ nats}$ of likelihood surface roughness were traced to two distinct numerical defects:

#### A. Catastrophic Cancellation in TDI Phase Curvature ($\dot{f}_0$)

The original estimator extracted the instantaneous frequency rate $\dot{f}_0$ from a second difference of the TDI response phase across a baseline $D = 0.125 / f_{\rm astro}$:


$$\dot{f}_0 = \frac{\arg(\bar{z}_+ \bar{z}_- z_0^2)}{2\pi D^2}$$

* **The Seed & The Amplifier:** Double-precision carrier phase evaluation introduces a seed round-off of $\approx 2.2 \times 10^{-11}\text{ rad}$. Second-generation TDI combinations cancel common laser phase noise but amplify this seed as $\epsilon(f_0) \approx 2.2 \times 10^{-11} / x^{3.64}\text{ rad}$, where $x = 2\pi f L/c$. At $0.35\text{ mHz}$, the input phase error reaches $\epsilon \approx 3.6 \times 10^{-5}\text{ rad}$.


* **Noise-Dominated Curvature:** The physical phase curvature across the baseline is $2\pi \dot{f} D^2 \approx 10^{-10}\text{ to } 6 \times 10^{-9}\text{ rad}$. Because this physical advance is up to $1.6 \times 10^4$ times smaller than $\epsilon$, the second difference acts as a random-number generator, yielding rates with $\le 0.01\%$ signal at low frequencies.


* **Template Contamination:** Accumulating this error across a segment $T_{\rm seg} = 86400\text{ s}$ injects relative template noise $\eta \approx 0.071 \epsilon (T_{\rm seg}/D)^2$, completely destroying derivative convergence.



#### B. Antenna-Pattern Null Carrier Jitter

Carrier bin assignment was indexed from channel 0 alone (`s.carrier_j = stft->get_freq_index(s.f0[0])`). When channel 0 passes through an antenna null ($\vert{}z_0\vert{}^2 < 10^{-5}$ of median power), its phase derivative wanders, displacing the evaluation stencil by $\pm 2$ bins across all three channels.

---

### 2. Implemented Architecture & Production Pipeline

#### A. Closed-Form Analytic Doppler Rate ($\dot{f}_0$)

The instantaneous observed frequency at detector time $t$ for a spacecraft at $\mathbf{x}_{\rm sc}(t)$ is governed by the retarded emission time $t_{\rm ssb} = t - \mathbf{k}\cdot\mathbf{x}_{\rm sc}(t)/c$:


$$f_{\rm obs}(t) = f_{\rm astro}(t_{\rm ssb})\left(1 - \frac{\mathbf{k}\cdot\mathbf{v}_{\rm sc}(t)}{c}\right)$$

Differentiating with respect to $t$ yields the exact rate:


$$\dot{f}_0(t) = \dot{f}_{\rm astro}(t)\left(1 - \frac{\mathbf{k}\cdot\mathbf{v}_{\rm sc}}{c}\right)^2 - f_0(t)\left(\frac{\mathbf{k}\cdot\mathbf{a}_{\rm sc}(t)}{c}\right) \approx \dot{f}_{\rm astro}(t) - f_0(t)\left(\frac{\mathbf{k}\cdot\mathbf{a}_{\rm sc}(t)}{c}\right)$$

The relativistic cross-term $\sim 2(v_{\rm sc}/c)\dot{f}_{\rm astro} \lesssim 10^{-19}\text{ Hz/s}$ is negligible ($\Delta\Phi_{\rm seg} \sim 10^{-9}\text{ rad}$).

#### B. Macro-Step Acceleration Differencing & Aliasing Protection

Because `Orbits::interpolate` uses linear interpolation on a $500\text{ s}$ node grid, sub-grid differencing ($\Delta t = 10\text{ s}$) caused 1-in-5 node aliasing where $80\%$ of columns returned $\mathbf{a}_{\rm sc} = \mathbf{0}$.

* **Macro-Step:** Acceleration is evaluated using $\Delta t_{\rm orb} = 2000\text{ s}$ (spanning $4\times$ node intervals), which eliminates aliasing with only a $4 \times 10^{-5}$ chord sag.


* **Boundary Clamping:** The anchor time $t_{\rm anchor}$ is clamped to $[t_0 + \Delta t_{\rm orb},\, t_0 + (N_{\rm sc} - 2)\Delta t_{\rm sc} - \Delta t_{\rm orb}]$. The upper fence at $(N_{\rm sc} - 2)$ prevents `interpolate` from performing out-of-bounds array reads at `window + 1`.



#### C. Carrier Placement via `ch_loudest`

In `lat_stft_kernels.hh`, carrier bin assignment is anchored to the channel with maximum center power:

```cpp
int ch_loudest = 0;
for (int j = 1; j < 3; j += 1)
{
    if (s.amp[j] > s.amp[ch_loudest])
        ch_loudest = j;
}
s.carrier_j = stft->get_freq_index(s.f0[ch_loudest]);

```


This eliminated all displaced carrier columns across the 365-day survey ($1 \to 0$) without extra response calls.

#### D. Baseband Carrier Demodulation for $f_0$

To bypass the branch-cut constraint ($D < 0.25/f_{\rm astro} \approx 38\text{ s}$) on the first difference, the product $\bar{z}_+ z_-$ is counter-rotated by the analytic midpoint carrier phase $e^{-i 4\pi f_{\rm scale} D}$ prior to `arg()`:


$$\Delta \phi_{\rm res} = \arg\left(\bar{z}_+ z_- e^{-i 4\pi f_{\rm scale} D}\right) \approx 4\pi \delta f_{\rm Doppler} D$$

$$f_0 = f_{\rm scale} + \frac{\Delta \phi_{\rm res}}{4\pi D}$$

Because $\vert{}\Delta \phi_{\rm res}\vert{} < 10^{-3}\text{ rad}$, the baseline is safely widened to `STFT_DT_STENCIL_DEMOD = 2000.0 s`, suppressing the frequency estimation floor $\delta f_0 \propto D^{-1}$ by $42\times$ at $3.27\text{ mHz}$.

#### E. Dual-Table Retardation Buffer & Anchor Clamping Bound ($\le D$)

The `Orbits` struct contains independent grids for spacecraft positions (`sc_*`, $500\text{ s}$) and light-travel times (`ltt_*`, $2.5\text{ s}$).

* **Retardation Depth:** Second-generation TDI delay chains sum to $7L/c \approx 58.4\text{ s}$, followed by a $1L/c \approx 8.34\text{ s}$ link to the emitter position.
* **Buffer:** A margin $\Delta t_{\rm retard} = 70.0\text{ s}$ covers the full $8L/c$ depth, preventing out-of-bounds queries where `Orbits::get_pos` returns `Vec(0,0,0)` (silently placing the emitter at the SSB).


* **Clamping Bound:** The valid domain is defined as the intersection of both tables. If column time $t$ falls outside $[t_{\rm valid,min}, t_{\rm valid,max}]$, the required shift would exceed $D$; the column immediately degrades to `stft_freq_fdot_astro_fallback` rather than evaluating static ephemerides across unphysical mega-second offsets.



#### F. Sign-Preserving Singularity Guard

The Fresnel rate guard is set to $1.0 \times 10^{-22}\text{ Hz/s}$ using `std::signbit` to preserve negative zero (`-0.0`) signs:

```cpp
if (std::fabs(fdot0_out[ch]) < 1.0e-22)
{
    fdot0_out[ch] = std::signbit(fdot0_out[ch]) ? -1.0e-22 : 1.0e-22;
}

```

* Sits $10^6\times$ below the weakest physical chirp ($\dot{f}_{\rm astro} \sim 10^{-16}\text{ Hz/s}$).


* Guarantees side-bin Fresnel arguments $x^2 \approx (k \Delta f)^2 / \vert{}\dot{f}_0\vert{} < 2^{53}$, preventing double-precision trigonometric reduction bit-loss across stencils up to $n_{\rm side} = 100$.



---

### 3. Measured Empirical Metrics

| Frequency ($f_0$) | Baseline Noise Floor | Analytic $\dot{f}_0$ ($D=0.125/f_0$) | Demodulated $f_0$ ($D=2000\text{ s}$) | Taylor Slope ($p$) | Resolved Range |
| --- | --- | --- | --- | --- | --- |
| **$0.81\text{ mHz}$**<br> | $4.37 \times 10^{-2}$<br> | $1.26 \times 10^{-4}$ ($348\times$) | **$1.08 \times 10^{-5}$** ($4.8\epsilon$) | $1.99$<br> | $2.0$ decades |
| **$3.27\text{ mHz}$**<br> | $3.96 \times 10^{-3}$<br> | $2.81 \times 10^{-6}$ ($1406\times$) | **$6.75 \times 10^{-8}$** ($5.3\epsilon$) | $1.99$<br> | **$3.0$ decades**<br> |
| **$13.17\text{ mHz}$**<br> | $2.03 \times 10^{-3}$<br> | $3.50 \times 10^{-7}$ ($5778\times$) | **$2.27 \times 10^{-9}$** ($7.2\epsilon$) | $2.00$<br> | **$3.0$ decades**<br> |

* **Criterion 1 Passed:** Fully met above $3\text{ mHz}$ ($\ge 3$ decades of pure $\delta^2$ Taylor convergence with $p = 2.00 \pm 0.01$).


* **Low-Frequency Boundary:** At $0.81\text{ mHz}$, the residual floor reaches $4.8\epsilon$, bounded strictly by the irreducible $x^{-3.64}$ TDI phase round-off.


* **Execution Overhead:** Single-template evaluation increases by $+8.2\%$ ($43.9\text{ ms} \to 47.5\text{ ms}$ at $3.27\text{ mHz}$), which is amortized across MCMC runs via faster chain decorrelation.



---

### 4. Investigated & Rejected Extensions

#### A. Analytic Kinematic $f_0$ Below $1\text{ mHz}$ (`STFT_ANALYTIC_F0_THRESHOLD = 0.0`)

Evaluating $f_0 = f_{\rm astro}(1 - \mathbf{k}\cdot\mathbf{v}_{\rm sc}/c)$ below $1\text{ mHz}$ was tested to bypass low-frequency TDI phase round-off.

* **Finding:** The velocity projection missed a term $\delta f = 3.2 \times 10^{-8}\text{ Hz}$ that was **constant across all carriers**.


* **Physical Cause:** $\delta f = \Omega_{\rm orb} / 2\pi$ corresponds to 1 cycle per year. The TDI observable carries the time derivative of the antenna pattern and TDI transfer phase ($\frac{1}{2\pi}\frac{d}{dt}\Phi_{\rm ant}$), which enters the instantaneous frequency of the data stream but is absent from kinematic velocity.


* **Verdict:** At $0.81\text{ mHz}$, this introduced an unweighted systematic template error of $4.94 \times 10^{-3}$ ($0.111$ noise-weighted), losing by 4 orders of magnitude against the $1.08 \times 10^{-5}$ phase-differenced noise floor. The branch remains disabled in production (`STFT_ANALYTIC_F0_THRESHOLD = 0.0`).



#### B. `linear_envelope = True` in Derivative Pipelines & The Dynamic $a_j$ Clamp

The first-moment amplitude slope $a_j = (\vert{}z_+\vert{} - \vert{}z_-\vert{})/(2D\vert{}z_0\vert{})$ provides a $21\times$ forward model accuracy gain, but was evaluated across parameter derivatives:

* **Finding:** Enabling the envelope degraded the Taylor floor along orientation parameters $(\iota, \psi)$ by **$11\times\text{--}57\times$** at low frequencies.


* **Mathematical Cause:** Without the envelope ($a_j = 0$), extrinsic parameters $(A, \phi_0, \iota, \psi)$ project analytically onto the exact 4D basis $\{H_+, iH_+, H_\times, iH_\times\}$. Enabling $a_j$ couples the non-linear, numerically noisy amplitude derivative into the tangent space, destroying the bilinear basis property and failing Criterion 5 ($21\text{--}49\epsilon$ vs. $\le 10\epsilon$).


* **Dynamic Validity Clamp ($a_{j,\max} = 2 / \Delta t$):** Inside `STFTFresnel`, the linear envelope $\vert{}z(t)\vert{} \approx \vert{}z_0\vert{}(1 + a_j(t - t_{\rm ref}))$ must remain non-negative across $\vert{}t - t_{\rm ref}\vert{} \le \Delta t / 2$.


* At boundary $t - t_{\rm ref} = -\Delta t / 2$, maintaining $\vert{}z(t)\vert{} \ge 0$ strictly requires $\vert{}a_j\vert{} \le 2 / \Delta t$.
* Exceeding $2 / \Delta t$ inverts the amplitude envelope within the segment, causing the first-moment correction to overpower the leading Fresnel integral.


* Bounding $\vert{}a_j\vert{} \le 2.0 / \text{stft->dt}$ ensures mathematical non-negativity across arbitrary segment scans ($\Delta t \in [6\text{ h}, 96\text{ h}]$) and fast-chirping systems (SOBBHs) without imposing an unscaled heuristic constant.




* **Verdict:** `linear_envelope = False` remains locked for all Fisher matrix, tangent-space, and derivative calculations. The dynamic clamp $a_{j,\max} = 2 / \Delta t$ is implemented in `FresnelColumn::setup` for forward evaluations.



---

### 5. Final Kernel Logic (`lat_stft_kernels.hh`)

```cpp
template <class SourceT>
CUDA_DEVICE void stft_freq_fdot_from_tdi_phase(
    SourceT& src, double t,
    double* params, Vec k, Vec u, Vec v,
    int* link_space_craft_rec, int* link_space_craft_em, int bin_i,
    const cmplx* tdi_center,
    double* f0_out, double* fdot0_out,
    double* amp_p_out, double* amp_m_out, double* D_out)
{
    for (int ch = 0; ch < 3; ch += 1) {
        amp_p_out[ch] = 0.0;
        amp_m_out[ch] = 0.0;
    }
    *D_out = STFT_FREQ_FDOT_DT_MAX;

    double f_scale = src.get_f(t, params, bin_i);
    if (!(f_scale > 0.0)) {
        stft_freq_fdot_astro_fallback<SourceT>(src, t, params, bin_i, f0_out, fdot0_out);
        return;
    }

    double D = STFT_DT_STENCIL_DEMOD; // 2000.0 s
    if (D > STFT_FREQ_FDOT_DT_MAX) D = STFT_FREQ_FDOT_DT_MAX;
    *D_out = D;

    // Retardation buffer (70.0 s) covers 8 * L/c on the intersection of tables
    constexpr double DT_RETARD_BUFFER = 70.0;
    double sc_t_min = src.orbits->sc_t0;
    double sc_t_max = src.orbits->sc_t0 + (double)(src.orbits->sc_N - 2) * src.orbits->sc_dt;
    double ltt_t_min = src.orbits->ltt_t0;
    double ltt_t_max = src.orbits->ltt_t0 + (double)(src.orbits->ltt_N - 2) * src.orbits->ltt_dt;

    double t_valid_min = (sc_t_min > ltt_t_min ? sc_t_min : ltt_t_min) + DT_RETARD_BUFFER;
    double t_valid_max = (sc_t_max < ltt_t_max ? sc_t_max : ltt_t_max);

    // Degrade if true time t falls outside valid intersection
    if (t < t_valid_min || t > t_valid_max) {
        stft_freq_fdot_astro_fallback<SourceT>(src, t, params, bin_i, f0_out, fdot0_out);
        return;
    }

    double t_anchor = t;
    if (t_anchor < t_valid_min + D) t_anchor = t_valid_min + D;
    else if (t_anchor > t_valid_max - D) t_anchor = t_valid_max - D;

    cmplx tdi_p[3], tdi_m[3];
    src.get_tdi_Xf_single(&tdi_p[0], t_anchor + D, params, k, u, v,
                          link_space_craft_rec, link_space_craft_em, bin_i);
    src.get_tdi_Xf_single(&tdi_m[0], t_anchor - D, params, k, u, v,
                          link_space_craft_rec, link_space_craft_em, bin_i);

    double pow_ch[3];
    int ch_ref = 0;
    for (int ch = 0; ch < 3; ch += 1) {
        pow_ch[ch] = tdi_center[ch].real() * tdi_center[ch].real()
                   + tdi_center[ch].imag() * tdi_center[ch].imag();
        if (pow_ch[ch] > pow_ch[ch_ref]) ch_ref = ch;
    }

    double pow_p = tdi_p[ch_ref].real() * tdi_p[ch_ref].real() + tdi_p[ch_ref].imag() * tdi_p[ch_ref].imag();
    double pow_m = tdi_m[ch_ref].real() * tdi_m[ch_ref].real() + tdi_m[ch_ref].imag() * tdi_m[ch_ref].imag();
    if (pow_ch[ch_ref] == 0.0 || pow_p == 0.0 || pow_m == 0.0) {
        stft_freq_fdot_astro_fallback<SourceT>(src, t, params, bin_i, f0_out, fdot0_out);
        return;
    }

    double fdot_astro = src.get_fdot(t, params, bin_i);
    double phi_carrier = 4.0 * M_PI * f_scale * D;
    cmplx demod_phasor(cos(phi_carrier), -sin(phi_carrier));

    for (int ch = 0; ch < 3; ch += 1) {
        cmplx z_diff = gcmplx::conj(tdi_p[ch]) * tdi_m[ch] * demod_phasor;
        double dphi1_res = gcmplx::arg(z_diff);
        f0_out[ch] = f_scale + dphi1_res / (4.0 * M_PI * D);

        int sc_id = ch + 1;
        Vec a_sc = stft_get_spacecraft_acc(src, t, sc_id);
        double k_dot_a = k.dot(a_sc);
        fdot0_out[ch] = fdot_astro - f0_out[ch] * (k_dot_a * C_inv);

        // Sign-preserving rate guard
        if (std::fabs(fdot0_out[ch]) < 1.0e-22) {
            fdot0_out[ch] = std::signbit(fdot0_out[ch]) ? -1.0e-22 : 1.0e-22;
        }
    }

    if (isnan(f0_out[ch_ref]) || isnan(fdot0_out[ch_ref])) {
        stft_freq_fdot_astro_fallback<SourceT>(src, t, params, bin_i, f0_out, fdot0_out);
        return;
    }

    for (int ch = 0; ch < 3; ch += 1) {
        if (!(pow_ch[ch] >= 1.0e-24 * pow_ch[ch_ref])) {
            f0_out[ch] = f0_out[ch_ref];
            fdot0_out[ch] = fdot0_out[ch_ref];
        }
    }

    for (int ch = 0; ch < 3; ch += 1) {
        amp_p_out[ch] = gcmplx::abs(tdi_p[ch]);
        amp_m_out[ch] = gcmplx::abs(tdi_m[ch]);
    }
}

```

```cpp
template <class SourceT>
struct FresnelColumn
{
    // ... State struct definition ...

    CUDA_DEVICE static void setup(
        State& s, SourceT& src, STFTFresnel* fresnel, STFTDomain* stft,
        double* params, Vec k, Vec u, Vec v,
        int* link_space_craft_rec, int* link_space_craft_em, int bin_i,
        double t_seg, double t_anchor_shift,
        double window_factor, bool freq_from_tdi_phase)
    {
        s.fresnel = fresnel;
        s.t_seg = t_seg;
        s.window_factor = window_factor;
        double t_here = t_seg + t_anchor_shift;
        cmplx tdi_channel_val[3];
        src.get_tdi_Xf_single(&tdi_channel_val[0], t_here, params, k, u, v,
                              link_space_craft_rec, link_space_craft_em, bin_i);
        double amp_p[3], amp_m[3], D_stencil;
        stft_pixel_freq_fdot<SourceT>(
            src, t_here, params, k, u, v,
            link_space_craft_rec, link_space_craft_em, bin_i,
            tdi_channel_val, freq_from_tdi_phase, &s.f0[0], &s.fdot0[0],
            &amp_p[0], &amp_m[0], &D_stencil);
        for (int j = 0; j < 3; j += 1)
            fresnel->get_amp_phase(&s.amp[j], &s.phase[j],
                                   gcmplx::conj(tdi_channel_val[j]));

        int ch_loudest = 0;
        for (int j = 1; j < 3; j += 1) {
            if (s.amp[j] > s.amp[ch_loudest])
                ch_loudest = j;
        }

        // Dynamic clamp ensures envelope non-negativity: |a_j * dt / 2| <= 1.0
        double a_j_max = 2.0 / stft->dt;
        constexpr double A_J_AMP_GUARD = 1.0e-12;
        double amp_floor = A_J_AMP_GUARD * s.amp[ch_loudest];

        for (int j = 0; j < 3; j += 1) {
            if (s.amp[j] > 0.0 && s.amp[j] > amp_floor) {
                double slope = (amp_p[j] - amp_m[j]) / (2.0 * D_stencil * s.amp[j]);
                if (slope > a_j_max) slope = a_j_max;
                else if (slope < -a_j_max) slope = -a_j_max;
                s.a[j] = slope;
            } else {
                s.a[j] = 0.0;
            }
        }
        s.carrier_j = stft->get_freq_index(s.f0[ch_loudest]);
    }

    // ... value() method ...
};

```

*Build Dependency Note:* Header `lat_stft_kernels.hh` is compiled by `GBGPU`; modifying this file requires recompiling `GBGPU` (`uv pip install -e . --no-build-isolation --no-deps`).