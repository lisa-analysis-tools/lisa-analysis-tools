"""CPU binding length checks with > INT_MAX elements (no large allocation).

    WCONV_NATIVE=1 .venv/bin/python test_length_checks.py   # worktree build: must pass
    .venv/bin/python test_length_checks.py                   # installed build: shows the overflow

Zero-stride writeable views give arrays of logical size > 2^31 backed by one element
(``array_type`` has no contiguity constraint). Every array has the correct length except
one deliberately wrong array, so the call must raise naming exactly that array, and only
that one: a check that overflowed int32 would reject a correct array first.
"""

from __future__ import annotations

import sys

import numpy as np
from numpy.lib.stride_tricks import as_strided

import _use_worktree  # noqa: F401
from fixtures import make_backend

FAIL = []


def big(n, dtype=np.float64):
    return as_strided(np.zeros(1, dtype=dtype), shape=(n,), strides=(0,), writeable=True)


def expect_only(name, should_be, fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        ok = msg.startswith(name + ":") and f"Should be {should_be}." in msg
        print(("  ok   " if ok else "  FAIL ") + f"raised for {msg.split(':')[0]!r} (expected {name!r}, length {should_be}): {msg[:140]}")
        if not ok:
            FAIL.append(name)
        return
    print("  FAIL no exception")
    FAIL.append(name)


def main():
    sm = make_backend("gal0_spl0_sym", "cpu")
    w = sm.pycpp_sensitivity_matrix
    nf, nt = 2, 1
    P = (2**31) // (nf * nt) + 7  # num_psds * nf * nt > INT_MAX, num_psds fits in int
    assert P * nf * nt > 2**31 - 1 and P < 2**31
    f = np.ascontiguousarray(np.asarray(sm.f_arr)[:nf])
    ti = np.zeros(nt, dtype=np.int32)
    npar = sm.n_noise_par
    r, c = (lambda n: big(n)), (lambda n: big(n, np.complex128))

    print(f"[get_noise_covariance_wrap] num_psds={P}, total={P * nf * nt:.3e}")
    expect_only("c22_arr", P * nf * nt, lambda: w.get_noise_covariance_wrap(
        f, ti, r(P * npar), r(P * npar), r(P), r(P), r(P), r(P), r(P),
        r(P * nf), r(P * nf),
        r(P * nf * nt), c(P * nf * nt), c(P * nf * nt), r(P * nf * nt), c(P * nf * nt), r(1),
        nf, nt, P, False))

    print(f"[psd_likelihood_wrap, external matrix] num_psds={P}")
    T = nf * nt
    expect_only("dips_mask", T, lambda: w.psd_likelihood_wrap(
        r(P), f, c(P * 3 * T), big(P, np.int32), ti, r(P * npar), r(P * npar),
        r(P), r(P), r(P), r(P), r(P), r(P * nf), r(P * nf),
        1.0, nf, nt, np.zeros(1, dtype=bool), P,
        r(P * T), r(P * T), r(P * T), c(P * T), c(P * T), c(P * T), True, False))

    print(f"\n{len(FAIL)} failures")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
