"""With one compute rank the fan-out is a direct call: no comm, no copies."""

import unittest

import numpy as np

from lisatools.globalfit.communication.fakecomm import FakeWorld
from lisatools.globalfit.communication.fanout import WalkerFanout
from lisatools.globalfit.communication.ranks import build_layout


class _NeverComm:
    """A communicator whose every use is a test failure."""

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        raise AssertionError(f"comm.{name} was touched in single mode")


class PassthroughTest(unittest.TestCase):
    def setUp(self):
        self.layout = FakeWorld(1).run(lambda r, c: build_layout(c, 4, None, legacy=False))[0]
        self.assertTrue(self.layout.is_single())

    def test_single_is_a_direct_call_with_the_same_objects(self):
        fo = WalkerFanout(_NeverComm(), self.layout, 0, model="MODEL")
        fo.enter_stage("pe", "pe_kind")
        payload = {"x": np.arange(4)}
        seen = {}

        def body(p, model):
            seen["payload"] = p
            seen["model"] = model
            seen["clock"] = dict(fo.clock)
            return {"y": p["x"] * 2}

        out = fo.run(
            "op",
            move="m",
            per_rank_payload=lambda r, w0, w1: payload,
            local_body=body,
            merge=lambda results: results,
        )
        self.assertIs(seen["payload"], payload)
        self.assertEqual(seen["model"], "MODEL")
        self.assertEqual(seen["clock"]["stage"], "pe")
        self.assertEqual(list(out), [0])
        np.testing.assert_array_equal(out[0]["y"], [0, 2, 4, 6])

    def test_allgather_and_stop_and_ping_are_local(self):
        fo = WalkerFanout(_NeverComm(), self.layout, 0)
        np.testing.assert_array_equal(fo.allgather_walker_vector(np.arange(4)), np.arange(4))
        fo.stop()
        self.assertEqual(fo.ping(), {0: self.layout.digest()})


if __name__ == "__main__":
    unittest.main()
