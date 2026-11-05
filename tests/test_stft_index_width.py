"""The STFT domain reads a data cell that lies beyond 2^31 values of its array.

A band buffer of 24636 cells of 64 bins, the full band on an 80 GB card, holds 3.4e9 complex values in one
split's data array, more than a signed 32-bit index addresses. This test writes the data of one container at
the start of a larger array and again in its last cell, past 2^31 values, and evaluates one source against
both cells with the same noise row. The two inner products must be equal. It needs a CUDA card with about
34 GiB free and skips otherwise.

    python -m unittest tests.test_stft_index_width -v
"""

from __future__ import annotations

import types
import unittest

from . import test_stft_store_windows as harness

# * Signed 32-bit offsets end here; the far cell must start past it.
INDEX_LIMIT = 2**31
BYTES_PER_VALUE = 16  # complex128
RELATIVE_TOLERANCE = 1e-12
# ? Room on the card besides the wide array: the case itself and the kernel's work arrays.
FREE_MARGIN_BYTES = 2 * 1024**3


class STFTIndexWidthCudaTest(unittest.TestCase):
    """One source, the same data at offset 0 and past 2^31 values of the array: equal inner products."""

    def setUp(self):
        self.backend_name = harness._cuda_backend_name()
        if self.backend_name is None or not harness._have_gbgpu_stft():
            self.skipTest("requires a usable CUDA backend and gbgpu STFT support")

    def test_cell_beyond_the_32_bit_range(self):
        import cupy as cp

        case = harness.build_case(self.backend_name)
        group = case["aca"].cpp_splits[0]
        domain_args = list(group.domain_args)
        num_times, num_freqs, num_channels = domain_args[0], domain_args[1], domain_args[2]
        cell_values = num_channels * num_times * num_freqs
        num_cells = INDEX_LIMIT // cell_values + 2
        far_cell = num_cells - 1
        self.assertGreater(far_cell * cell_values, INDEX_LIMIT)

        free_bytes, _total_bytes = cp.cuda.runtime.memGetInfo()
        needed_bytes = num_cells * cell_values * BYTES_PER_VALUE + FREE_MARGIN_BYTES
        if free_bytes < needed_bytes:
            self.skipTest(f"requires {needed_bytes / 2**30:.1f} GiB free, the card has {free_bytes / 2**30:.1f}")

        first_container = domain_args[8][:cell_values]
        wide_data = cp.zeros(num_cells * cell_values, dtype=cp.complex128)
        wide_data[:cell_values] = first_container
        wide_data[far_cell * cell_values:] = first_container
        domain_args[8] = wide_data
        domain_args[10] = num_cells
        wide_domain = group.backend.STFTDomainWrap(*domain_args)

        sorter = harness.make_sorter(case, case["windows"])
        live_ids = sorter.xp.arange(sorter.num_sources)[sorter.inds]
        physical = case["transform"].both_transforms(sorter.coords[live_ids[:1]], xp=sorter.xp)
        comp = case["comp"]
        # * The comp reads only the Fresnel factors, the domain and d_d of its group.
        comp.stft_comps = types.SimpleNamespace(cpp_fresnel=group.cpp_fresnel, cpp_domain=wide_domain, d_d=None)
        comp.get_ll_stft(cp.concatenate([physical, physical]), data_index=[0, far_cell], noise_index=[0, 0])
        d_h, h_h = comp.d_h_out.get(), comp.h_h_out.get()
        print(f"\n  cell {far_cell} starts at value {far_cell * cell_values:,} of {num_cells * cell_values:,}: "
              f"(d|h) {d_h[0]:.6e} and {d_h[1]:.6e}, (h|h) {h_h[0]:.6e} and {h_h[1]:.6e}")
        self.assertNotEqual(d_h[0], 0.0, msg="the source sees no data: the comparison is empty")
        # ? A relative tolerance, since the kernel may sum a source's pixels in another order per call.
        self.assertLess(abs(d_h[1] - d_h[0]), RELATIVE_TOLERANCE * abs(d_h[0]),
                        msg="the cell past 2^31 values reads other data")
        self.assertLess(abs(h_h[1] - h_h[0]), RELATIVE_TOLERANCE * abs(h_h[0]))
        del wide_domain, wide_data


if __name__ == "__main__":
    unittest.main()
