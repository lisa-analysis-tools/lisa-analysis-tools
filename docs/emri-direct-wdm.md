# EMRI direct-to-WDM templates: status and measurements

Branch `emri-direct-wdm` (LAT). Plan: `~/.claude/plans/scalable-napping-toast.md`.
All numbers below are laptop CPU runs on CD1L (mojito light) data or synthetic sources;
mismatch is `1 - Re(O)` with no time or phase maximisation, per TDI channel X, Y, Z.

## What exists

| Piece | Where |
|---|---|
| TDI-on-the-fly X, Y, Z matching the production response | `sources/emri/emritdionfly.py` (`frame="icrs_special"`, `n_fine`, `t_fine_window`) |
| Restored n_ref lookup table + exact evaluation rule | `domains.py` `WDMLookupTable` (`BASIS_CYCLE="quarter_turn"`) |
| Harmonic tracks from the FEW holder | `sources/emri/wdm_direct.py` `harmonic_tracks_from_holder` |
| Plunge chunk (even-start 128-px chunks) | `wdm_het.py` `wdm_chunk_of_td`, `tail_chunk_plan`, `splice_chunk` |
| The fast template | `sources/emri/wdm_direct.py` `EMRIDirectWDM` |
| Laptop table (layer 3600 s, Nf 180, dt 20 s) | `wdm_lookup_emri_cx_NF180_DT20_TL32_fd8x0p01_nld2.h5` (sprint root) |

## Results

**TDI-on-the-fly vs production response** (EMRI 1, 15.2 d): mismatch ~1e-13 per channel at
mode thresholds 1e-3 and 1e-7; dlogL difference <= 4e-7. Signed off 2026-09-30.

**Direct-to-WDM vs production, CD1L EMRI 1** (16 d window, no plunge in the window):

| mode threshold | table interp | modes | direct vs production | norm ratio vs production | direct vs data | production vs data |
|---|---|---|---|---|---|---|
| 1e-3 (production EMRI_EPS) | linear | 38 | 1.05e-6 | 0.99973-0.99976 | 0.10 | 0.10 |
| 1e-3 (production EMRI_EPS) | **cubic (default)** | 38 | **3.5e-8** | **0.999995-0.999997** | 0.10 | 0.10 |
| 1e-5 | linear | 111 | 1.11e-6 | not measured | 2.6e-3 | 2.6e-3 |

The normalised mismatch alone hid a 2.5e-4 amplitude deficit with linear table interpolation
(the bias of linear interpolation across the table's peaked frequency response); cubic
removes it (EMRIDirectWDM default, `interp="cubic"`; CPU only).

The direct template is as close to the data as the production template; the data
mismatch is the production mode threshold's content loss, not the template method.

**Plunging synthetic source, single (2,2,0,0) harmonic** (42.7 d, plunge at 33.8 d):

| region (t / t_plunge) | direct vs TOF | TOF vs production |
|---|---|---|
| < 0.90 (lookup) | 0.7-1.0e-4 | ~3e-12 |
| 0.90-0.99 (chunk) | 0.9-1.4e-5 | ~5e-8 |
| >= 0.99 (chunk) | ~1e-14 | 1e-6 to 1.2e-5 (was 1.3-1.7e-3, fixed: plunge-end tail) |
| whole window | 1.3-1.5e-5 | 3e-7 to 3.8e-6 |

Direct vs production over the whole window: 1.4-1.9e-5 per channel; norm ratios within
1e-4 (whole window) and 1e-3 (< 0.90 region) of 1.

**Error budget of the lookup** (single harmonic, 170 d, p0 = 10.2): the exact local-chirp
model is at 1.9e-4 (median, before 0.9 t_plunge), the table's f/fdot interpolation at
1.1e-3 (fdot step 0.01 layer_df/layer_dt). The table dominates over the inspiral.

## Defects found and fixed on the way

1. EMRITDIonFly built the -m partner as the +m mode with a negated phase: off by pi for
   some higher-l modes (2% strain error at threshold 1e-7).
2. EMRITDIonFly passed Tobs in seconds where FEW's T is in years (integrated to plunge).
3. FEW merges call-time `inspiral_kwargs` into the generator permanently.
4. The historical lookup evaluation (2-way sign) was wrong; the exact rule is a
   parity-dependent quarter turn, with an odd-(m_ref + n_ref) variant and a block bake on
   the chirp term that must be undone at the table nodes (block seams).
5. `build_wdm_lookup_gpu.py` ignored `--time-layers` for complex builds (32x slower).
6. Lookup support must cover offsets [-2, 3] layers; 5 layers per pixel are needed.
7. An even-start WDM chunk is not exact in its interior: edge contamination decays
   algebraically (0.2 at the edge, 5e-5 at 16 px): discard Nt_sub/4 per side.
8. At a plunge the on-the-fly response grid ended 720 s early (delay trim at the end of the
   fine feed): the feed now continues past the stop with zero amplitude (two delay margins).
9. Retrograde input: FEW maps xI0 < 0 to (-a, +1) before its sign rule; the harmonic tracks
   now do the same (the old code flipped the phase for the user form).

## Open items

1. (resolved) TOF vs production at the abrupt plunge end.
2. Table resolution (fdot direction; cubic fixed the f-direction bias): finer fdot rows near 0 (where most pixels sit) to reach the model's
   ~1e-4 per-pixel level; production-grid table (Nf 1440, dt 2.5) on the cluster GPU.
3. Speed: EMRIDirectWDM is Python per harmonic and channel on CPU (88 s for 38 modes, 16 d).
   The GPU work (FEW Part B) and a vectorised lookup are the path.
4. Memory: every FEW EMRI generator reads the whole 5.1 GB amplitude file at construction
   (`few/amplitude/ampinterp2d.py:235`), a ~6 GB transient footprint. One per process.
5. EMRIs 0, 2-7: their L1 bricks are not on the laptop; run
   `scripts/emri/emri_direct_wdm_mismatch.py --src N` on the cluster.
6. Neighbour layers per pixel: 5 (num_m_layers=2) hold an EMRI inspiral (fdot mostly
   < 0.1 layer_df/layer_dt); a chirp at 0.25 units keeps ~1.3% of its power two layers out and
   loses ~2e-2 rel L2 in layers three out. Make num_m_layers grow with fdot if needed.
7. Mode selection: EMRIDirectWDM selects modes on a 256-point trajectory over ITS window;
   the production wrapper selects over its own span from the reference epoch. For a
   like-for-like comparison on another window, pass the production modes via
   `mode_selection=[(l, m, k, n), ...]`.
8. The kappa (intra-chunk sweep) guard was removed: the SOBBH limit is for heterodyned
   chunks; the plunge chunk is a raw TD->WDM transform, exact through the plunge.
