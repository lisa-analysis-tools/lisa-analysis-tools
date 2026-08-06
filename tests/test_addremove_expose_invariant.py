"""The expose invariant and the fancy-swap gate of ResidualAddOneRemoveOneMove.

Both guard the same class of bug: the expose/fold residual choreography going
out of step with the code that scores against it. ``_verify_entry_vs_acs``
detects it at runtime; ``_fancy_swap_now`` fixes the trigger that decides when
the ensemble is permuted inside that window.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from eryn.moves import StretchMove

from lisatools.globalfit.moves.addremovemove import ResidualAddOneRemoveOneMove


NTEMPS, NWALKERS, NLEAVES, NDIM = 4, 6, 2, 3


def _build_move(num_repeats=10, permute_every=20, branch_name="gb"):
    """Construct the move with inert collaborators.

    ``__init__`` only stores ``acs`` / ``transform_fn`` / ``priors`` and builds
    one TemperatureControl per leaf, so none of them need to be real here.
    """
    return ResidualAddOneRemoveOneMove(
        branch_name=branch_name,
        coords_shape=(NTEMPS, NWALKERS, NLEAVES, NDIM),
        waveform_gen=lambda *args, **kwargs: None,
        waveform_gen_kwargs={},
        waveform_like_kwargs={},
        acs=None,
        num_repeats=num_repeats,
        transform_fn=None,
        priors=None,
        inner_moves=[StretchMove()],
        permute_every=permute_every,
    )


# --------------------------------------------------------------------------
# the fancy-swap gate
# --------------------------------------------------------------------------


def test_fancy_swap_fires_exactly_once_per_leaf_visit_at_the_end():
    # the stock configuration: with the old `repeat % permute_every` trigger
    # num_repeats=10 < permute_every=20 meant it never fired at all
    move = _build_move(num_repeats=10, permute_every=20)
    fired = [r for r in range(move.num_repeats) if move._fancy_swap_now(r)]
    assert fired == [9]


def test_fancy_swap_value_is_a_gate_not_a_cadence():
    # any positive value fires on the same (final) repeat
    for permute_every in (1, 2, 3, 20, 1000):
        move = _build_move(num_repeats=5, permute_every=permute_every)
        fired = [r for r in range(move.num_repeats) if move._fancy_swap_now(r)]
        assert fired == [4], permute_every


def test_fancy_swap_single_repeat_still_fires():
    move = _build_move(num_repeats=1, permute_every=20)
    assert move._fancy_swap_now(0)


@pytest.mark.parametrize("permute_every", [0, -1])
def test_fancy_swap_disabled_never_fires_and_does_not_divide_by_zero(permute_every):
    # `repeat % 0` raised ZeroDivisionError under the old trigger
    move = _build_move(num_repeats=4, permute_every=permute_every)
    assert not any(move._fancy_swap_now(r) for r in range(move.num_repeats))


def test_permute_every_env_override(monkeypatch):
    monkeypatch.setenv("GB_PERMUTE_EVERY", "0")
    move = _build_move(num_repeats=4, permute_every=20)
    assert move.permute_every == 0
    assert not any(move._fancy_swap_now(r) for r in range(move.num_repeats))


# --------------------------------------------------------------------------
# the expose invariant
# --------------------------------------------------------------------------


def _prev_logl(cold_row):
    """Pack a cold-chain row into the (ntemps, nwalkers) block propose() builds."""
    out = np.zeros((NTEMPS, NWALKERS))
    out[0] = cold_row
    return out


def test_matching_likelihoods_are_silent(caplog):
    move = _build_move()
    ref = np.array([-10.0, -20.0, -30.0, -40.0, -50.0, -60.0])
    with caplog.at_level(logging.WARNING):
        move._verify_entry_vs_acs(_prev_logl(ref.copy()), ref, leaf=0)
    assert caplog.text == ""


def test_constant_small_offset_is_benign(caplog):
    # a pure normalization convention shifts every walker by the same amount
    move = _build_move()
    ref = np.array([-10.0, -20.0, -30.0, -40.0, -50.0, -60.0])
    with caplog.at_level(logging.WARNING):
        move._verify_entry_vs_acs(_prev_logl(ref + 3.0), ref, leaf=0)
    assert caplog.text == ""


def test_d_minus_2h_corruption_is_caught(caplog):
    # the real signature: scoring against `d - 2h` shifts each walker by its
    # own -2<h|h>, so the difference has a large per-walker SPREAD
    move = _build_move()
    ref = np.array([-10.0, -20.0, -30.0, -40.0, -50.0, -60.0])
    h_h = np.array([100.0, 220.0, 310.0, 90.0, 415.0, 260.0])
    with caplog.at_level(logging.WARNING):
        move._verify_entry_vs_acs(_prev_logl(ref - 2.0 * h_h), ref, leaf=3)
    assert "EXPOSE INVARIANT VIOLATED" in caplog.text
    assert "leaf 3" in caplog.text


def test_large_constant_offset_is_caught(caplog):
    move = _build_move()
    ref = np.array([-10.0, -20.0, -30.0, -40.0, -50.0, -60.0])
    with caplog.at_level(logging.WARNING):
        move._verify_entry_vs_acs(_prev_logl(ref - 500.0), ref, leaf=1)
    assert "EXPOSE INVARIANT VIOLATED" in caplog.text


def test_strict_mode_raises():
    move = _build_move()
    move.check_ll_mode = "strict"
    ref = np.array([-10.0, -20.0, -30.0, -40.0, -50.0, -60.0])
    with pytest.raises(ValueError, match="EXPOSE INVARIANT VIOLATED"):
        move._verify_entry_vs_acs(_prev_logl(ref - 500.0), ref, leaf=0)


def test_check_ll_env_selects_strict(monkeypatch):
    monkeypatch.setenv("GB_CHECK_LL", "strict")
    assert _build_move().check_ll_mode == "strict"


def test_check_ll_env_can_disable(monkeypatch):
    monkeypatch.setenv("ADDREMOVE_CHECK_LL", "0")
    assert _build_move().check_ll_mode == "0"


def test_missing_reference_is_a_no_op(caplog):
    # propose() passes None when the check is thinned out by CHECK_LL_EVERY
    move = _build_move()
    with caplog.at_level(logging.WARNING):
        move._verify_entry_vs_acs(_prev_logl(np.full(NWALKERS, -1.0)), None, leaf=0)
    assert caplog.text == ""


def test_sentinel_and_nonfinite_walkers_are_excluded(caplog):
    # -1e300 is compute_acs_like's "not evaluated" fill; comparing it would
    # manufacture an enormous spread out of nothing
    move = _build_move()
    ref = np.array([-10.0, -20.0, -30.0, -40.0, -50.0, -60.0])
    cold = ref.copy()
    cold[2] = -1e300
    cold[4] = -np.inf
    with caplog.at_level(logging.WARNING):
        move._verify_entry_vs_acs(_prev_logl(cold), ref, leaf=0)
    assert caplog.text == ""


def test_no_comparable_walkers_is_a_no_op(caplog):
    move = _build_move()
    ref = np.full(NWALKERS, -1e300)
    with caplog.at_level(logging.WARNING):
        move._verify_entry_vs_acs(_prev_logl(ref.copy()), ref, leaf=0)
    assert caplog.text == ""


def test_mismatched_reference_size_is_a_no_op(caplog):
    # the ACS array can hold more containers than this move has walkers
    move = _build_move()
    with caplog.at_level(logging.WARNING):
        move._verify_entry_vs_acs(
            _prev_logl(np.full(NWALKERS, -10.0)),
            np.full(NWALKERS + 3, -10.0),
            leaf=0,
        )
    assert caplog.text == ""
