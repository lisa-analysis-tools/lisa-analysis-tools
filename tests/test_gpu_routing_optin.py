"""``GF_GPU_ROUTING`` is OPT-IN: with it unset, nothing changes.

This is the contract the ``gpu-count-routing`` -> ``dev`` merge rests on.
The unified factorization (``n_compute = n_blocks * R`` at any walker count)
is a strict SUPERSET of the legacy rule, so the gate is not there to pick
between two behaviours -- it is there so that merging the superset cannot
change a run that did not ask for it.

Three things are asserted, in increasing order of what they would cost if
they were wrong:

1. the knob parses the way the other layout knobs do, and defaults OFF;
2. ON and OFF give the IDENTICAL factorization on every shape the legacy
   rule accepted -- the property that makes the merge a no-op;
3. the shapes the legacy rule refused are still refused, and the refusal
   names the knob.

TODO(gpu-routing): when the default flips (see ``ranks.GPU_ROUTING_ENV``),
this module is what says whether the flip is a behaviour change -- keep
``test_on_and_off_agree_wherever_legacy_resolves`` and delete the rest.
"""

import os
import unittest
from unittest import mock

from lisatools.globalfit.communication.fakecomm import FakeWorld
from lisatools.globalfit.communication.ranks import (
    GPU_ROUTING_ENV,
    build_layout,
    factorize_layout,
    gpu_routing_enabled,
)

#: every shape the LEGACY rule accepts: nwalkers % n_compute == 0, or one
#: walker (the replica carve-out). Derived, not listed, so the sweep below
#: cannot drift from the rule it is checking.
_MAX = 33


def _legacy_accepts(nwalkers, n_compute):
    return nwalkers == 1 or nwalkers % n_compute == 0


class KnobParsingTest(unittest.TestCase):
    def test_default_is_off(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(GPU_ROUTING_ENV, None)
            self.assertFalse(gpu_routing_enabled())

    def test_truthy_and_falsey_spellings(self):
        for raw, want in (
            ("1", True), ("true", True), ("True", True), ("TRUE", True),
            ("yes", True), ("on", True), (" 1 ", True),
            ("0", False), ("false", False), ("", False), ("   ", False),
            # an opt-IN treats anything it does not recognise as OFF, which is
            # the opposite of GF_ONE_WALKER_REPLICAS (an opt-OUT of a shipped
            # feature). A typo must not silently enable new routing.
            ("maybe", False), ("2", False), ("of", False),
        ):
            with mock.patch.dict(os.environ, {GPU_ROUTING_ENV: raw}):
                self.assertEqual(gpu_routing_enabled(), want, msg=repr(raw))


class FactorizationGateTest(unittest.TestCase):
    def test_on_and_off_agree_wherever_legacy_resolves(self):
        """★ The property that makes merging this branch a no-op.

        If this ever fails, some shape that a production runbook already
        launches would come out differently after the merge.
        """
        checked = 0
        for nwalkers in range(1, _MAX):
            for n_compute in range(1, _MAX):
                if not _legacy_accepts(nwalkers, n_compute):
                    continue
                checked += 1
                self.assertEqual(
                    factorize_layout(nwalkers, n_compute, gpu_routing=False),
                    factorize_layout(nwalkers, n_compute, gpu_routing=True),
                    msg=f"nwalkers={nwalkers} n_compute={n_compute}",
                )
        self.assertGreater(checked, 100)  # the sweep actually swept

    def test_off_reproduces_the_two_legacy_regimes(self):
        # walker blocks (R == 1) ...
        self.assertEqual(factorize_layout(4, 4, gpu_routing=False), (4, 1, 1))
        self.assertEqual(factorize_layout(24, 8, gpu_routing=False), (8, 1, 3))
        self.assertEqual(factorize_layout(10, 1, gpu_routing=False), (1, 1, 10))
        # ... and the one-walker replica carve-out (R == n_compute)
        self.assertEqual(factorize_layout(1, 4, gpu_routing=False), (1, 4, 1))
        self.assertEqual(factorize_layout(1, 1, gpu_routing=False), (1, 1, 1))

    def test_off_still_refuses_what_it_always_refused(self):
        for nwalkers, n_compute in ((4, 16), (24, 32), (6, 4), (2, 4), (3, 2)):
            with self.assertRaises(ValueError) as ctx:
                factorize_layout(nwalkers, n_compute, gpu_routing=False)
            # the refusal must name the way forward, or it is just an error
            self.assertIn(GPU_ROUTING_ENV, str(ctx.exception))

    def test_off_refuses_an_explicit_ranks_per_block(self):
        # RANKS_PER_BLOCK=1 is what the legacy layout always did, so it is
        # not a request for anything new and must keep working.
        self.assertEqual(factorize_layout(4, 4, 1, gpu_routing=False), (4, 1, 1))
        for R in (2, 4, 8):
            with self.assertRaises(ValueError) as ctx:
                factorize_layout(4, 4, R, gpu_routing=False)
            self.assertIn(GPU_ROUTING_ENV, str(ctx.exception))

    def test_the_public_rule_defaults_on(self):
        """``factorize_layout`` describes the rule; ``build_layout`` deploys it.

        A planner must be able to price a shape the current deployment has
        not opted into yet, so the PUBLIC rule answers for the superset.
        """
        self.assertEqual(factorize_layout(4, 16), (4, 4, 1))

    def test_the_parameter_beats_the_environment(self):
        # ``_factorize`` must stay pure: a planner polling it in a loop
        # cannot have the ambient environment decide the answer.
        with mock.patch.dict(os.environ, {GPU_ROUTING_ENV: "1"}):
            with self.assertRaises(ValueError):
                factorize_layout(4, 16, gpu_routing=False)
        with mock.patch.dict(os.environ, {GPU_ROUTING_ENV: "0"}):
            self.assertEqual(factorize_layout(4, 16, gpu_routing=True), (4, 4, 1))


class BuildLayoutGateTest(unittest.TestCase):
    """The launcher reads the environment; the rule takes a parameter."""

    def _build(self, size, nwalkers, **kw):
        world = FakeWorld(size, nodes=[0, 1, 0, 1, 0][:size])
        return world.run(
            lambda r, c: build_layout(c, nwalkers, [0, 1], legacy=False, **kw)
        )[0]

    def test_unset_environment_refuses_a_new_shape(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(GPU_ROUTING_ENV, None)
            with self.assertRaises(RuntimeError):  # FakeWorld wraps rank errors
                self._build(5, 2)

    def test_set_environment_resolves_it(self):
        with mock.patch.dict(os.environ, {GPU_ROUTING_ENV: "1"}):
            lay = self._build(5, 2)
        self.assertEqual((lay.n_blocks, lay.ranks_per_block, lay.block), (2, 2, 1))
        self.assertTrue(lay.gpu_routing)

    def test_the_legacy_shapes_build_either_way(self):
        for env in ("0", "1"):
            with mock.patch.dict(os.environ, {GPU_ROUTING_ENV: env}):
                blocks = self._build(5, 8)          # 4 compute ranks, 8 walkers
                one = self._build(5, 1)             # one-walker replicas
            self.assertEqual(
                (blocks.n_blocks, blocks.ranks_per_block, blocks.block), (4, 1, 2))
            self.assertEqual((one.n_blocks, one.ranks_per_block), (1, 4))
            self.assertTrue(one.replica_mode)

    def test_the_resolved_knob_reaches_describe_and_the_digest(self):
        """A rank that set the knob differently must FAIL, not route differently.

        ``build_layout`` allgathers ``describe()`` and refuses on
        disagreement, so carrying the flag there turns a half-configured
        launch into an error instead of two ranks quietly disagreeing about
        which band split to use.
        """
        with mock.patch.dict(os.environ, {GPU_ROUTING_ENV: "1"}):
            on = self._build(5, 8)
        with mock.patch.dict(os.environ, {GPU_ROUTING_ENV: "0"}):
            off = self._build(5, 8)
        self.assertIn("gpu_routing=on", on.describe())
        self.assertIn("gpu_routing=OFF(legacy)", off.describe())
        self.assertNotEqual(on.digest(), off.digest())

    def test_an_explicit_argument_overrides_the_environment(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(GPU_ROUTING_ENV, None)
            lay = self._build(5, 2, gpu_routing=True)
        self.assertEqual((lay.n_blocks, lay.ranks_per_block), (2, 2))


if __name__ == "__main__":
    unittest.main()
