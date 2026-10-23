"""
Block size shared by the row-blocked numpy paths of rvt (private module, not part of the public API).

Several loops process an image in row blocks instead of whole-image temporaries: the SVF horizon scan
(``rvt.vis._svf_compute_window``), the sliced path of ``rvt.vis.local_dominance``, the blocked sky
illumination path (``rvt.vis._sky_illum_block_rows``) and the blend kernels of ``rvt.blend_func``
(``_row_slices``). They all use the one constant below, so that it is defined and explained once.

The constant lives in its own module because both ``rvt.vis`` and ``rvt.blend_func`` need it, and
``rvt._accel`` is about the optional numba kernels, which ``rvt.blend_func`` does not use.
"""

# Number of elements processed per block. Why 32768: one float64 buffer of a block is 256 KiB (float32:
# 128 KiB), so the few per-block buffers of a loop (differences, quotients, running maximum, accumulators,
# blend temporaries) stay in the L2 / L3 cache of current CPUs instead of streaming image-sized temporaries
# through main memory. The value is not tuned per machine. It affects speed only: every blocked operation is
# elementwise or per pixel, so any block size gives the same results. (The sky illumination path also rounds
# its block height down to a multiple of 16 rows; see ``rvt.vis._sky_illum_block_rows`` for why.)
BLOCK_ELEMS = 32768
