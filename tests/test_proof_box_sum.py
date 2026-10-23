"""Order-sensitive test of ``rvt.vis._box_sum`` (added after the release audit of 0.1.0).

``_box_sum`` (introduced by commit 8839afd, replacing ``mean_filter._box_sum`` and ``_mstp_box_sum``) must
evaluate ``((I[lo, lo] + I[hi, hi]) - I[hi, lo]) - I[lo, hi]`` in exactly this order, because that is the
order of the original ``roll + roll - roll - roll`` expression and the order decides the rounding.

The frozen tests use small DEMs whose float64 integral images are exact, so they cannot see the order (the
hand-written mutant m22 of tools/proof/mutants.py, which swaps the two subtractions, survived them). Here
the table holds large values with full 53-bit mantissas, where the two orders round differently; the test
first checks that the input is order-sensitive, then compares ``_box_sum`` bitwise with the reference.
"""

import numpy as np

import rvt.vis


def _reference(table, lo, hi, n_rows, n_cols):
    a = table[lo:lo + n_rows, lo:lo + n_cols]
    b = table[hi:hi + n_rows, hi:hi + n_cols]
    c = table[hi:hi + n_rows, lo:lo + n_cols]
    d = table[lo:lo + n_rows, hi:hi + n_cols]
    return ((a + b) - c) - d


def _swapped(table, lo, hi, n_rows, n_cols):
    a = table[lo:lo + n_rows, lo:lo + n_cols]
    b = table[hi:hi + n_rows, hi:hi + n_cols]
    c = table[hi:hi + n_rows, lo:lo + n_cols]
    d = table[lo:lo + n_rows, hi:hi + n_cols]
    return ((a + b) - d) - c


def _table(dtype):
    # cumulative sums of squares of elevations near 2,500 m, as in MSTP's dem_i2, with a random fraction
    rng = np.random.default_rng(20261023)
    z = (2500.0 + rng.random((70, 90)) * 37.0).astype(np.float64)
    return np.cumsum(np.cumsum(z * z * (1.0 + rng.random(z.shape) * 1e-9), axis=0), axis=1).astype(dtype)


def test_input_is_order_sensitive():
    t = _table(np.float64)
    assert not np.array_equal(_reference(t, 2, 9, 61, 81), _swapped(t, 2, 9, 61, 81))


def test_box_sum_matches_reference_order_float64():
    t = _table(np.float64)
    for lo, hi in ((0, 7), (2, 9), (5, 6)):
        n_rows, n_cols = t.shape[0] - hi, t.shape[1] - hi
        got = rvt.vis._box_sum(t, lo, hi, n_rows, n_cols)
        ref = _reference(t, lo, hi, n_rows, n_cols)
        assert got.dtype == ref.dtype and got.shape == ref.shape
        assert got.tobytes() == ref.tobytes()


def test_box_sum_matches_reference_order_float32():
    t = _table(np.float32)
    got = rvt.vis._box_sum(t, 1, 8, t.shape[0] - 8, t.shape[1] - 8)
    ref = _reference(t, 1, 8, t.shape[0] - 8, t.shape[1] - 8)
    assert got.dtype == np.float32 and got.tobytes() == ref.tobytes()


def test_box_sum_does_not_modify_the_table():
    t = _table(np.float64)
    before = t.copy()
    rvt.vis._box_sum(t, 0, 7, t.shape[0] - 7, t.shape[1] - 7)
    assert t.tobytes() == before.tobytes()
