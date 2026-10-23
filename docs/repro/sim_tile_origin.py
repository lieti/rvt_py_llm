"""Repro of KNOWN_ISSUES KI-03: a sky illumination tile equals the full window only for origins that are multiples of 8 px.

With the defaults of ``rvt.default.DefaultValues`` (pyramid scale 2, 3 coarse pyramid levels) the horizon pyramid of
``rvt.vis.sky_illumination`` is anchored at the top-left corner of the array passed in, so the coarse grids of a tile
match those of the full window only when the tile origin is a multiple of 2^3 = 8 px.

Set-up: synthetic float32 DEM of 1024 x 1024 px at 0.25 m/px (tilted plane, ridges, 0.05 m noise, +300 m offset,
seed 0). Tiles of 512 x 512 px start at (o, o) for each origin o; the central 256 x 256 px of each tile (a 128 px =
32 m buffer on every side) is compared with the same pixels of the full-window result. The script prints one line
per origin: the maximum absolute difference and the maximum difference after scaling 0..1 to 0..255 and rounding.

Usage, from the repository root or any other directory: ``python docs/repro/sim_tile_origin.py``. The package is
imported from the repository root of this file, not from an installed copy. Runtime: about 5 s with numba.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import rvt.default  # noqa: E402

N = 1024  # px, full window
TILE = 512  # px
BUFFER = 128  # px on every side of the compared interior
ORIGINS = (256, 248, 252, 254, 255)  # px; 256 and 248 are multiples of 8


def synthetic_dem(n, offset=300.0, seed=0):
    """Deterministic float32 surface of n x n px at 0.25 m/px with an elevation offset in m."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n].astype(np.float64) * 0.25
    z = 0.02 * x + 0.01 * y + 1.5 * np.sin(x / 7.0) * np.cos(y / 11.0) + 0.8 * np.sin((x + y) / 3.0)
    z += rng.normal(0.0, 0.05, (n, n))
    return (z + offset).astype(np.float32)


def main():
    print(f"rvt imported from {Path(rvt.default.__file__).resolve().parent}")
    dem = synthetic_dem(N)
    defaults = rvt.default.DefaultValues()
    full = defaults.get_sky_illumination(dem, 0.25)
    for o in ORIGINS:
        tile = defaults.get_sky_illumination(dem[o:o + TILE, o:o + TILE].copy(), 0.25)
        lo, hi = BUFFER, TILE - BUFFER
        t = tile[lo:hi, lo:hi].astype(np.float64)
        f = full[o + lo:o + hi, o + lo:o + hi].astype(np.float64)
        d = float(np.nanmax(np.abs(t - f)))
        dn = float(np.nanmax(np.abs(np.round(t * 255.0) - np.round(f * 255.0))))
        print(f"tile origin {o:3d} (mod 8 = {o % 8}): max |tile - full| = {d:.3g}, "
              f"max difference on a 0..1 -> 0..255 scale {dn:.0f} DN")


if __name__ == "__main__":
    main()
