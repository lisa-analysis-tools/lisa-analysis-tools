"""Move a global-fit HDF5 store's VGB branch onto the 6-column chirp basis.

The VGB distance basis ``[dist, phi0, cos_iota, psi, fdot_astro_ratio]``
(Mc a per-leaf fill) becomes the ``VGB_CHIRP_MASS_BASIS=1`` basis
``[dist, phi0, cos_iota, psi, Mc, fdot_astro_ratio]`` (only f0 / alpha /
sin_delta fixed). Resuming a 5-dim store with the flag on fails loudly
(``run.check_store_branch_ndims``, at backend construction) and points here.

TWO MODES
=========

``--restart-at-injection`` (**THE DEFAULT**, user ruling 2026-09-16: "keep
the hdf backend except for the VGBs (and just start them at injection
again)"). A clean vgb REBIRTH:

* everything that is not the vgb branch is left **byte-for-byte alone** --
  the GB chains and their band state, the noise / galfor chains, the
  mbh / emri / sobbh chains, log_like, log_prior, betas, the iteration
  counter. The run resumes exactly where it was;
* the vgb chain is DISCARDED and reseeded at catalogue INJECTION in the
  6-column basis, all six columns from the catalogue (see
  :func:`vgb_injection_rows`).

``--carry-history``: the original 2026-08-14 behaviour, kept for
completeness -- the five old columns are carried into slots [0,1,2,3,5],
Mc (slot 4) gets the catalogue value times a small multiplicative jitter,
and the ratio column is re-initialized with the additive fresh-init rule.

WHY REBIRTH IS THE DEFAULT. The old 5-column history is a chain of a
DIFFERENT model: Mc was pinned per leaf, so every stored ratio was fitted
against a fixed Mc, and the joint (Mc, r, dist) fiber the new basis exists
to explore was not sampled at all. Carrying those rows forward starts the
6-column sampler at a point that is only accidentally related to the new
posterior, whereas the catalogue truth is the exact point the campaign
starts VGBs from anyway (``VGB_START_FACTOR=0.0``).

NO JITTER ON THE REBIRTH SEED
-----------------------------
The seed is EXACT truth for every rung / walker / leaf --
``seed_injection_coords(inj, factor=0.0, ...)`` (``run.py:192``), the
campaign convention (``VGB_START_FACTOR=0.0`` in submit_gf_6mo_v8.sh).
Walkers therefore start IDENTICAL, which is fine for -- and only for --
the chirp-basis proposal stack: the observable composite, the generic
eigen draw and the ridge-Gibbs fiber move all build their steps from each
source's own information matrix / prior box, not from the ensemble spread.
A pure affine-invariant STRETCH move would degenerate on an identical
ensemble (it can never create spread it does not have), so a run that
falls back to ``VGB_INMODEL_PROPOSAL=stretch`` must not use a jitterless
seed. That is exactly why the jitterless default is safe here and is
stated rather than left to inference. ``--start-factor`` adds the
multiplicative scatter back if a stretch arm is ever wanted.

WHY THE WHOLE CHAIN IS FILLED, NOT JUST THE RESUME ROW
------------------------------------------------------
Only row ``iteration - 1`` is read on resume (``Backend.get_last_sample``
-> ``get_a_sample(self.iteration - 1)``), so the earlier rows are dead
history for a reborn branch and COULD be left NaN (the store's own
dead-leaf sentinel). They are filled with the same truth instead, because
NaN history is not actually free -- audited against the consumers:

* most of it IS tolerated: ``gf_monitor_gen.py`` front-trims unpopulated
  rows off its pooling window (``_vgb_pool_rows``, "Leading rows are
  legitimately NaN"), ``gf_compare_gen.py`` reads only the last row,
  ``build_truth.py`` never touches vgb, and ``postprocessing.py``
  explicitly repairs a NaN autocorrelation time;
* but NaN reaching the monitor's ~30-row pooling window is NOT tolerated:
  the per-leaf ``np.median`` / ``np.percentile(16, 84)`` credible
  intervals there are not nan-safe (silent NaN), and ``ax.hist`` raises
  ``ValueError: autodetected range of [nan, nan] is not finite`` outright
  when the whole window is NaN -- which is exactly the shape a
  "NaN history + one real row" migration would produce for the first ~30
  iterations after the relaunch.

Filling every row with truth cannot produce a NaN anywhere, so none of
that applies. The cosmetic cost is the opposite and much milder: the
monitor treats the filled rows as populated, so for the first ~30
iterations after the relaunch the vgb panels show a dead-flat trace at
truth with zero spread. That is an honest picture of a branch that was
just reseeded at truth, and it self-heals as real iterations accumulate.

Dead leaves still take the NaN sentinel -- VGB is fixed-dimensional with
every leaf alive, so in practice none are written.

STALE SIDECARS
--------------
Both sidecars hold 5-column vgb state and are QUARANTINED into
``pre_chirp_migration/`` next to the store, together with the store's own
backup:

* ``<base>_midit_checkpoint.pkl`` -- the mid-iteration checkpoint. It
  BEATS the HDF store on resume when it is at least as new, and although
  ``GlobalFit._midit_checkpoint_validate`` would reject it on the coords
  ndim check, "rejected" is a silent fallback: quarantining makes the
  migration the only possible resume path.
* ``<base>_running_backup_copy.h5`` -- the rolling backup. This is the
  dangerous one: ``promote_backup_if_store_unreadable`` promotes it over
  the primary if the primary ever looks unreadable, which would silently
  UNDO the migration mid-campaign.

The store's ``log_like`` is NOT patched: a resume recomputes it after
rebuilding the residuals from the loaded state
(``state.log_like[:] = acs.likelihood(complex=False)``, ``run.py:2150``),
so the reborn vgb templates are accounted for on the first iteration.

BAND BOOKKEEPING IS DELIBERATELY LEFT ALONE
-------------------------------------------
``sub_backend/vgb/band_*`` is NOT reset -- audited, not assumed. Every one
of those datasets is shaped ``(num_bands, ...)``; none carries a parameter
axis, so widening the chain 5 -> 6 columns cannot invalidate any of them.
Per dataset:

* ``band_temps`` -- the resume trusts its SHAPE (the stored rung count
  wins over ``VGB_NTEMPS``) but ``recipe.build_vgb_moves`` re-tiles the
  configured ladder over every band whenever the counts match. Zeroing it
  would buy nothing in that case and would be ACTIVELY DANGEROUS in the
  other: if the counts ever differed, the stored ladder is adopted and
  ``vgb_info.betas = band_temps[0]``, so an all-zero ladder would switch
  the VGB likelihood off -- the exact 2026-08-15 betas=[1e-4] failure.
* ``band_swaps_{proposed,accepted}``, ``band_num_{proposed,accepted}``
  (+ ``_rj``) -- accumulating diagnostics only. The ladder adaptation
  (``_adapt_band_temps``) allocates FRESH per-iteration counters and never
  reads the stored totals.
* ``band_num_binaries`` -- fully overwritten from the live band census on
  the first propose.
* the leaf-cap and band-shutoff families -- inert on vgb (no RJ surface,
  and ``build_vgb_moves`` passes ``leaf_caps=False`` / no
  ``leaf_cap_start``).

So stale values change no sampling decision. The single visible artifact:
stored counters are per-save-interval DELTAS (``save_step`` writes then
zeroes), so the first post-resume diagnostic row double-counts one
interval; it is clean from the second save on. That is a better trade than
discarding the GB-comparable swap/proposal statistics. Shapes and dtypes
are load-bearing either way (the resume refuses on any shape disagreement,
and ``nwalkers`` is inferred from ``band_num_binaries.shape[-2]``), which
is another reason not to rewrite them here. (To change the vgb RUNG COUNT
of a live store -- a different migration -- use ``fix_vgb_band_temps.py``.)

Usage::

    python scripts/fstat_proposal/migrate_vgb_chirp_basis.py \\
        <store.h5> <catalogue_dir>          # rebirth at injection (default)

    python scripts/fstat_proposal/migrate_vgb_chirp_basis.py \\
        <store.h5> <catalogue_dir> --carry-history   # legacy carry-forward

where ``<catalogue_dir>`` is the mojito data dir, e.g.
``.../mojito_cache/brickmarket/mojito_light_v1_0_0``. Resume with
``VGB_CHIRP_MASS_BASIS=1``.
"""

import argparse
import os
import shutil

import h5py
import numpy as np

from lisatools.globalfit.stock.erebor.vgb import (
    VGB_SAMPLED_BASIS_CHIRP,
    VGB_SAMPLED_BASIS_DIST,
    load_vgb_catalogue_file,
)

OLD_NDIM = len(VGB_SAMPLED_BASIS_DIST)  # 5
NEW_NDIM = len(VGB_SAMPLED_BASIS_CHIRP)  # 6
MC_COL = VGB_SAMPLED_BASIS_CHIRP.index("Mc")  # 4
RATIO_COL = VGB_SAMPLED_BASIS_CHIRP.index("fdot_astro_ratio")  # 5
# old -> new column map for the 5 carried-over columns (--carry-history)
OLD_TO_NEW = [0, 1, 2, 3, RATIO_COL]

#: Directory (created next to the store) holding the pre-migration backup
#: and every quarantined sidecar.
QUARANTINE_DIRNAME = "pre_chirp_migration"

# Fresh-init conventions for --carry-history (mirror VGBSettings defaults +
# run.py seeding): ratio_0 = START_FACTOR * RATIO_INIT_WIDTH * RATIO_MAX * randn.
START_FACTOR = float(os.environ.get("VGB_START_FACTOR", "1e-5"))
RATIO_INIT_WIDTH = float(os.environ.get("VGB_RATIO_INIT_WIDTH", "0.02"))
FDOT_ASTRO_RATIO_MAX = float(os.environ.get("GB_FDOT_ASTRO_RATIO_MAX", "5.0"))
MC_JITTER_REL = 1e-6  # multiplicative, START_FACTOR-style, per stored element


def leaf_chirp_masses(catalogue_dir: str) -> np.ndarray:
    """Per-leaf catalogue Mc, in prepare_vgb_branch's sorted-key leaf order."""
    # the loader stores (V)GB catalogue entries as whole arrays under one id;
    # prepare_vgb_branch concatenates catalogue[k][column] over sorted keys
    cat = load_vgb_catalogue_file(catalogue_dir)
    return np.concatenate([
        np.atleast_1d(np.asarray(cat[k]["ChirpMassSSBFrame"], dtype=float))
        for k in sorted(cat.keys())
    ])


def vgb_injection_rows(catalogue_dir: str) -> np.ndarray:
    """``(nleaves, 6)`` catalogue truth in the chirp sampling basis.

    A FAITHFUL MIRROR of the production seeding block --
    ``lisatools/globalfit/stock/erebor/vgb.py::prepare_vgb_branch`` lines
    1093-1151, the ``sample_distance`` + ``chirp_mass_basis`` path -- using
    the same helpers rather than re-deriving anything:

    * leaf order = ``sorted(catalogue.keys())`` then file order within a key
      (deterministic across restarts, so leaf i here is leaf i in the run);
    * rows = :func:`gb_catalogue_to_sampling_basis` per key, so ``phi0``,
      ``cos_iota`` and ``psi`` are the container's sampling-basis values
      (the phi0 sign convention stays routed through the container) and are
      copied VERBATIM;
    * ``dist`` = catalogue ``LuminosityDistance`` (Mpc) x 1e3 -> kpc, the
      unit the gbgpu amplitude convention takes;
    * ``Mc`` = catalogue ``ChirpMassSSBFrame``;
    * ``fdot_astro_ratio`` = ``fdot_cat / fdot_gr(d, f0, Mc) - 1``, COMPUTED
      through :class:`McDistFdotAstroQuad` -- never assumed zero. It is
      ~0 for a GW-driven catalogue, but a mass-transfer system is exactly
      the case this column exists for.

    The catalogue (f0, Mc, dist) -> A consistency check from the production
    path is kept too, so a catalogue unit / convention change fails loudly
    here instead of seeding a quietly wrong store.

    ``prepare_vgb_branch`` itself is deliberately NOT called: it also
    resolves band guards, prior boxes and the temperature ladder, and so
    needs a full ``GeneralSetup`` (domain_settings, Tobs, min_freq,
    max_freq, tdi_chan, force_backend, ntemps) that a chain rewrite has no
    business inventing.

    Returns:
        ``(nleaves, 6)`` rows ordered by ``VGB_SAMPLED_BASIS_CHIRP``.
    """
    from lisatools.globalfit.recipe import gb_catalogue_to_sampling_basis
    from lisatools.globalfit.stock.erebor.transforms import (
        McDistFdotAstroQuad,
        gb_amp_from_dist,
        make_gb_transform_container,
    )

    catalogue = load_vgb_catalogue_file(catalogue_dir)
    if not catalogue:
        raise SystemExit(f"no VGB catalogue found under {catalogue_dir!r}")
    keys = sorted(catalogue.keys())

    rows = np.array([gb_catalogue_to_sampling_basis(catalogue[k]) for k in keys])
    if rows.ndim == 3:
        rows = rows.reshape(-1, rows.shape[-1])

    full_basis = list(make_gb_transform_container(use_chirp_mass=False).input_basis)

    def _cat_col(name):
        return np.concatenate([
            np.atleast_1d(np.asarray(catalogue[k][name], dtype=float))
            for k in keys
        ])

    d_kpc = _cat_col("LuminosityDistance") * 1e3
    mc = _cat_col("ChirpMassSSBFrame")
    f0_hz_rows = rows[:, full_basis.index("f0")] * 1e-3
    a_phys = np.exp(rows[:, full_basis.index("A")])
    fdot_phys = rows[:, full_basis.index("fdot")]
    _, _, fdot_gr, _ = McDistFdotAstroQuad()(
        d_kpc, f0_hz_rows, mc, np.zeros_like(d_kpc))
    ratio = fdot_phys / fdot_gr - 1.0

    rel = np.abs(gb_amp_from_dist(f0_hz_rows, mc, d_kpc) / a_phys - 1.0)
    if rel.max() > 1e-3:
        raise SystemExit(
            "VGB catalogue (f0, Mc, dist) does not reproduce the catalogue "
            f"Amplitude (max rel {rel.max():.3e}); check the "
            "LuminosityDistance units / amplitude convention."
        )

    inj = np.empty((rows.shape[0], NEW_NDIM))
    inj[:, VGB_SAMPLED_BASIS_CHIRP.index("dist")] = d_kpc
    for name in ("phi0", "cos_iota", "psi"):
        inj[:, VGB_SAMPLED_BASIS_CHIRP.index(name)] = (
            rows[:, full_basis.index(name)])
    inj[:, MC_COL] = mc
    inj[:, RATIO_COL] = ratio
    return inj


def _seeded_block(inj, shape, start_factor, rng):
    """Broadcast ``inj`` over a stored-chain block shape.

    ``shape`` is the FULL dataset shape ``(..., nleaves, ndim)``. With
    ``start_factor = 0`` (the default) this is the exact injection in every
    slot -- identical to ``run.seed_injection_coords(inj, 0.0, ...)``, which
    reduces to ``inj[None, None] * 1.0``.
    """
    block = np.broadcast_to(inj, shape).astype(float, copy=True)
    if start_factor:
        # the production MULTIPLICATIVE convention (run.py:192
        # seed_injection_coords); the zero-truth ratio column takes the
        # documented ADDITIVE exception so it cannot collapse.
        block *= 1.0 + start_factor * rng.standard_normal(shape)
        block[..., RATIO_COL] = (
            inj[:, RATIO_COL]
            + start_factor * RATIO_INIT_WIDTH * FDOT_ASTRO_RATIO_MAX
            * rng.standard_normal(shape[:-1])
        )
    return block


def _replace_dataset(grp, name, new):
    """Swap ``grp[name]`` for ``new``, preserving compression + maxshape."""
    dset = grp[name]
    compression = dset.compression
    compression_opts = dset.compression_opts
    del grp[name]
    grp.create_dataset(
        name,
        data=new,
        maxshape=(None,) + new.shape[1:],
        compression=compression,
        compression_opts=compression_opts,
    )


def reseed_chain(grp, name, inj, *, inds=None, start_factor=0.0, rng=None):
    """Rewrite ``grp[name]`` (..., nleaves, 5) as the 6-col catalogue truth.

    EVERY stored row is filled (see the module docstring: NaN history would
    silently poison whole-run reductions in the monitor/extract consumers).
    """
    dset = grp[name]
    old_shape = dset.shape
    if old_shape[-1] != OLD_NDIM:
        raise SystemExit(
            f"{dset.name}: last axis is {old_shape[-1]}, expected {OLD_NDIM} "
            "(already migrated?)"
        )
    nleaves = old_shape[-2]
    if inj.shape[0] != nleaves:
        raise SystemExit(
            f"catalogue has {inj.shape[0]} sources but {dset.name} stores "
            f"{nleaves} leaves"
        )
    new_shape = old_shape[:-1] + (NEW_NDIM,)
    new = _seeded_block(inj, new_shape, start_factor, rng)
    if inds is not None:
        new[~np.asarray(inds, dtype=bool)] = np.nan  # dead-leaf sentinel
    _replace_dataset(grp, name, new)
    print(f"  reseeded {grp.name}/{name}: {old_shape} -> {new.shape} "
          f"(catalogue truth, start_factor={start_factor:g})")


def widen_chain(grp, name, mc_per_leaf, rng, inds=None):
    """``--carry-history``: carry the 5 old columns into the 6-col layout."""
    dset = grp[name]
    old = dset[:]
    assert old.shape[-1] == OLD_NDIM, (
        f"{dset.name}: last axis is {old.shape[-1]}, expected {OLD_NDIM} "
        "(already migrated?)"
    )
    nleaves = old.shape[-2]
    assert mc_per_leaf.shape[0] == nleaves, (
        f"catalogue has {mc_per_leaf.shape[0]} sources but {dset.name} "
        f"stores {nleaves} leaves"
    )
    new = np.empty(old.shape[:-1] + (NEW_NDIM,), dtype=old.dtype)
    for j_old, j_new in enumerate(OLD_TO_NEW):
        new[..., j_new] = old[..., j_old]
    # Mc from the catalogue per leaf, small multiplicative jitter so walkers
    # differ; ratio re-initialized with the additive fresh-init rule.
    new[..., MC_COL] = mc_per_leaf * (
        1.0 + MC_JITTER_REL * rng.standard_normal(old.shape[:-1])
    )
    new[..., RATIO_COL] = (
        START_FACTOR * RATIO_INIT_WIDTH * FDOT_ASTRO_RATIO_MAX
        * rng.standard_normal(old.shape[:-1])
    )
    if inds is not None:
        new[~np.asarray(inds, dtype=bool)] = np.nan  # dead-leaf sentinel
    _replace_dataset(grp, name, new)
    print(f"  rewrote {grp.name}/{name}: {old.shape} -> {new.shape}")


def quarantine_sidecars(h5_path: str) -> list:
    """Move the store backup + every stale 5-column sidecar out of the way.

    Returns the list of ``(src, dst)`` pairs actually moved. See the module
    docstring for why each one is dangerous.
    """
    store_dir = os.path.dirname(os.path.abspath(h5_path)) or "."
    quarantine = os.path.join(store_dir, QUARANTINE_DIRNAME)
    os.makedirs(quarantine, exist_ok=True)

    base, _ = os.path.splitext(h5_path)
    candidates = [
        base + "_midit_checkpoint.pkl",       # midit_checkpoint.checkpoint_path
        h5_path[:-3] + "_running_backup_copy.h5" if h5_path.endswith(".h5")
        else base + "_running_backup_copy.h5",
        h5_path[:-3] + "_running_backup_copy.h5.tmp" if h5_path.endswith(".h5")
        else base + "_running_backup_copy.h5.tmp",
    ]
    moved = []
    for src in candidates:
        if not os.path.exists(src):
            continue
        dst = os.path.join(quarantine, os.path.basename(src))
        if os.path.exists(dst):
            raise SystemExit(
                f"refusing to overwrite an existing quarantined file {dst!r} "
                "(a previous migration attempt left it there; move or delete "
                "it deliberately)"
            )
        shutil.move(src, dst)
        moved.append((src, dst))
        print(f"  quarantined {os.path.basename(src)} -> "
              f"{QUARANTINE_DIRNAME}/")
    return moved


def migrate_store(h5_path, inj, mc_per_leaf, *, carry_history=False,
                  start_factor=0.0, seed=101):
    """Rewrite the vgb branch of ``h5_path`` in place. Nothing else moves."""
    rng = np.random.default_rng(seed)
    with h5py.File(h5_path, "r+") as f:
        group = "global_fit" if "global_fit" in f else "mcmc"
        g = f[group]

        # main cold-chain history (dead-leaf mask from the stored inds)
        inds = g["inds"]["vgb"][:] if "vgb" in g.get("inds", {}) else None
        if carry_history:
            widen_chain(g["chain"], "vgb", mc_per_leaf, rng, inds=inds)
        else:
            reseed_chain(g["chain"], "vgb", inj, inds=inds,
                         start_factor=start_factor, rng=rng)
        g["ndims"].attrs["vgb"] = NEW_NDIM
        print(f"  {g.name}/ndims attrs['vgb'] = {NEW_NDIM}")
        if "key_order" in g and "vgb" in g["key_order"].attrs:
            old_ko = list(g["key_order"].attrs["vgb"])
            if len(old_ko) == OLD_NDIM:
                g["key_order"].attrs["vgb"] = np.arange(NEW_NDIM)
                print(f"  key_order['vgb']: {old_ko} -> {list(range(NEW_NDIM))}")

        # branch sub-backend (tempered history)
        sub = g["sub_backend"]["vgb"]
        sub_inds = sub["inds"][:] if "inds" in sub else None
        if carry_history:
            widen_chain(sub, "chain", mc_per_leaf, rng, inds=sub_inds)
        else:
            reseed_chain(sub, "chain", inj, inds=sub_inds,
                         start_factor=start_factor, rng=rng)
        sub.attrs["ndim"] = NEW_NDIM
        print(f"  {sub.name} attrs['ndim'] = {NEW_NDIM}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("h5_path", help="global-fit HDF5 file to migrate")
    parser.add_argument(
        "catalogue_dir",
        help="mojito data dir holding catalogues/vgb_cat_mojito_lite_"
             "processed.hdf5 (e.g. .../brickmarket/mojito_light_v1_0_0)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--restart-at-injection", dest="carry_history", action="store_false",
        help="DEFAULT: discard the vgb chain and reseed every stored row at "
             "catalogue truth in the 6-column basis; the rest of the store "
             "(GB / noise / source chains, iteration counter) is untouched.",
    )
    mode.add_argument(
        "--carry-history", dest="carry_history", action="store_true",
        help="legacy: carry the 5 old columns forward, jitter in Mc, "
             "re-initialize the ratio column.",
    )
    parser.set_defaults(carry_history=False)
    parser.add_argument(
        "--start-factor", type=float, default=0.0,
        help="multiplicative walker scatter for the reseed (default 0.0 = "
             "exact truth, the VGB_START_FACTOR=0.0 campaign convention). "
             "Only raise this for a stretch-move arm.",
    )
    parser.add_argument("--seed", type=int, default=101, help="jitter RNG seed")
    parser.add_argument(
        "--keep-sidecars", action="store_true",
        help="do NOT quarantine the midit checkpoint / running backup "
             "(unsafe: either can silently undo this migration on resume).",
    )
    args = parser.parse_args()

    inj = vgb_injection_rows(args.catalogue_dir)
    mc_per_leaf = inj[:, MC_COL]
    print(f"catalogue: {inj.shape[0]} VGB leaves, "
          f"Mc in [{mc_per_leaf.min():.3f}, {mc_per_leaf.max():.3f}] Msol, "
          f"|fdot_astro_ratio| max {np.abs(inj[:, RATIO_COL]).max():.3e}")

    store_dir = os.path.dirname(os.path.abspath(args.h5_path)) or "."
    quarantine = os.path.join(store_dir, QUARANTINE_DIRNAME)
    os.makedirs(quarantine, exist_ok=True)
    bak = os.path.join(quarantine, os.path.basename(args.h5_path) + ".bak")
    if os.path.exists(bak):
        raise SystemExit(f"refusing to overwrite existing backup {bak!r}")
    shutil.copy2(args.h5_path, bak)
    print(f"backup: {bak}")

    if not args.keep_sidecars:
        quarantine_sidecars(args.h5_path)

    migrate_store(
        args.h5_path, inj, mc_per_leaf,
        carry_history=args.carry_history,
        start_factor=args.start_factor,
        seed=args.seed,
    )

    if args.carry_history:
        print(
            "done (--carry-history). Old columns [dist, phi0, cos_iota, psi, "
            "ratio] -> new slots [0, 1, 2, 3, 5]; Mc (slot 4) = catalogue "
            f"value * (1 + {MC_JITTER_REL:g}*randn); ratio (slot 5) "
            f"re-initialized additively with width {START_FACTOR:g} * "
            f"{RATIO_INIT_WIDTH:g} * {FDOT_ASTRO_RATIO_MAX:g}."
        )
    else:
        print(
            "done (rebirth at injection). The vgb branch now holds the "
            f"catalogue truth {VGB_SAMPLED_BASIS_CHIRP} in every stored row; "
            "every other branch, and the iteration counter, are untouched. "
            "The resume recomputes log_like from the rebuilt residuals."
        )
    print("Resume with VGB_CHIRP_MASS_BASIS=1.")


if __name__ == "__main__":
    main()
