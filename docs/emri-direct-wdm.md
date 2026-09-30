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

| mode threshold | modes | direct vs production | direct vs data | production vs data |
|---|---|---|---|---|
| 1e-3 (production EMRI_EPS) | 38 | 1.05e-6 | 0.10 | 0.10 |
| 1e-5 | 111 | 1.11e-6 | 2.6e-3 | 2.6e-3 |

The direct template is as close to the data as the production template; the data
mismatch is the production mode threshold's content loss, not the template method.

**Plunging synthetic source, single (2,2,0,0) harmonic** (42.7 d, plunge at 33.8 d):

| region (t / t_plunge) | direct vs TOF | TOF vs production |
|---|---|---|
| < 0.90 (lookup) | 0.7-1.0e-4 | ~3e-12 |
| 0.90-0.99 (chunk) | 0.9-1.4e-5 | ~5e-8 |
| >= 0.99 (chunk) | ~1e-14 | 1.3-1.7e-3 (open item 1) |
| whole window | 1.3-1.5e-5 | 4.3-5.2e-4 |

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

## Open items

1. TOF vs production differs by ~1e-3 in the last 1% before plunge (abrupt end).
2. Table resolution: finer fdot rows near 0 (where most pixels sit) to reach the model's
   ~1e-4 per-pixel level; production-grid table (Nf 1440, dt 2.5) on the cluster GPU.
3. Speed: EMRIDirectWDM is Python per harmonic and channel on CPU (88 s for 38 modes, 16 d).
   The GPU work (FEW Part B) and a vectorised lookup are the path.
4. Memory: every FEW EMRI generator reads the whole 5.1 GB amplitude file at construction
   (`few/amplitude/ampinterp2d.py:235`), a ~6 GB transient footprint. One per process.
5. EMRIs 0, 2-7: their L1 bricks are not on the laptop; run
   `scripts/emri/emri_direct_wdm_mismatch.py --src N` on the cluster.
