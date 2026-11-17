"""JaxBase._to_jax must place host inputs on the caller's cupy device.

Regression test: host scalars (e.g. ``np.float64`` rows of a numpy ``coords`` array, as in
the add/remove cold-chain fold-back) used to go to ``jax.device_put(device=None)``, i.e.
gpu0, and that call also switched the calling thread's CUDA current device to 0, so a
per-split thread inside ``cp.cuda.Device(k)`` did all of its later cupy work on gpu0.
"""
import threading
import unittest

import numpy as np

try:
    import cupy as cp
    import jax
    import jax.numpy as jnp

    _N_GPU = len(jax.devices("gpu")) if jax.default_backend() == "gpu" else 0
except Exception:  # noqa: BLE001 -- no cupy / jax / GPU
    _N_GPU = 0

from lisatools.jax.jaxbase import JaxBase


class _J(JaxBase):
    pass


def _in_thread(fn):
    out = {}

    def run():
        try:
            out["value"] = fn()
        except BaseException as exc:  # noqa: BLE001
            out["error"] = exc

    t = threading.Thread(target=run)
    t.start()
    t.join()
    if "error" in out:
        raise out["error"]
    return out["value"]


@unittest.skipUnless(_N_GPU >= 2, "needs cupy + JAX with at least 2 visible GPUs")
class ToJaxDevicePlacementTest(unittest.TestCase):
    def setUp(self):
        jnp.zeros(1).block_until_ready()  # initialise JAX in the main thread, as in a run
        self.j = _J()

    def test_host_inputs_follow_cupy_device_in_thread(self):
        def work():
            with cp.cuda.Device(1):
                inputs = [np.float64(1.5), 2.0, 3, np.array([1.0, 2.0])]
                arrs = [self.j._to_jax(x) for x in inputs]
                res = jnp.sin(arrs[0]) + arrs[1]
                res.block_until_ready()
                h = self.j._from_jax(res)
                return (
                    [a.devices() for a in arrs],
                    res.devices(),
                    h.device.id,
                    cp.cuda.runtime.getDevice(),
                )

        arr_devs, res_dev, h_dev, cur = _in_thread(work)
        gpu1 = {jax.devices("gpu")[1]}
        for d in arr_devs:
            self.assertEqual(d, gpu1)
        self.assertEqual(res_dev, gpu1)
        self.assertEqual(h_dev, 1)
        self.assertEqual(cur, 1, "the thread's CUDA current device must not be switched")

    def test_explicit_device_id_wins(self):
        def work():
            with cp.cuda.Device(1):
                return self.j._to_jax(np.float64(1.0), device_id=0).devices()

        self.assertEqual(_in_thread(work), {jax.devices("gpu")[0]})

    def test_default_device_unchanged_on_device_0(self):
        def work():
            with cp.cuda.Device(0):
                return self.j._to_jax(np.float64(1.0)).devices()

        self.assertEqual(_in_thread(work), {jax.devices("gpu")[0]})

    def test_jax_default_device_context_wins(self):
        def work():
            with cp.cuda.Device(1), jax.default_device(jax.devices("cpu")[0]):
                return self.j._to_jax(np.float64(1.0)).devices()

        self.assertEqual(_in_thread(work), {jax.devices("cpu")[0]})

    def test_cupy_input_stays_on_its_device(self):
        def work():
            with cp.cuda.Device(1):
                return self.j._to_jax(cp.arange(3.0)).devices()

        self.assertEqual(_in_thread(work), {jax.devices("gpu")[1]})


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(_N_GPU >= 1, "needs cupy + JAX with a visible GPU")
class ToJaxAlignmentTest(unittest.TestCase):
    """cupy views at an offset that is not a multiple of 16 bytes must still reach JAX.

    Regression test: a single-source ``SignalGen`` call hands phentax 1-element columns of a ``(1, ndim)``
    coords array (``cupy.ascontiguousarray`` does not copy them), at offsets of 8 bytes; zero-copy DLPack
    then failed inside the first JAX op ("Address of buffer ... must be a multiple of 10").
    """

    def setUp(self):
        self.j = _J()

    def test_offset_views(self):
        coords = cp.arange(1.0, 12.0).reshape(1, 11)
        for col in range(coords.shape[1]):
            view = cp.ascontiguousarray(coords[:, col])
            out = jnp.abs(self.j._to_jax(view))
            self.assertEqual(float(out[0]), col + 1.0)

    def test_aligned_input_stays_zero_copy(self):
        x = cp.arange(4.0)
        y = self.j._to_jax(x)
        self.assertEqual(y.unsafe_buffer_pointer(), x.data.ptr)
