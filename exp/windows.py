"""Window sums over square, disk and octagonal neighbourhoods, and DEV / MSTP built on them.

Pure numpy / scipy, float64. Every shape returns, for each radius r, the sums over the output window of z, of
z**2 and the valid-pixel count, so that mean = s1 / n, var = s2 / n - mean**2, as rvt.vis.max_elevation_deviation
does. The input is padded by R = max(radii) (plus one leading row/column for the tables) in `pad_mode`.

Shapes (half-width r, px):
  square  : |dx| <= r, |dy| <= r                      summed-area table, 4 lookups per pixel and radius
  disk    : dx^2 + dy^2 <= r^2                        FFT convolution; data spectra computed once, kernel per r
  octagon : square cut by |dx| + |dy| <= round(r*sqrt(2))   summed-area table + anti-diagonal tables, O(1) per pixel
  bf      : brute force with scipy.ndimage (small arrays only; the oracle of validate.py)

NaN: pixels with NaN are excluded from the sums and from the count (mask tables / mask spectra), as in rvt.
"""
import numpy as np
from scipy import fft as sfft
from scipy import ndimage

SHAPES = ("square", "disk", "octagon")


# ----------------------------------------------------------------------------------------------- masks (oracle)
def mask(shape, r, r_eff=None):
    """Boolean (2r+1)^2 window mask of a shape; r_eff overrides the disk radius (equal-area variant)."""
    if shape == "disk" and r_eff is not None:
        R = int(np.ceil(r_eff))
        y, x = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float64)
        return x * x + y * y <= r_eff * r_eff
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    if shape == "square":
        return np.ones_like(x, dtype=bool)
    if shape == "disk":
        return x * x + y * y <= r * r
    if shape == "octagon":
        return (np.abs(x) + np.abs(y)) <= octagon_m(r)
    raise ValueError(shape)


def octagon_m(r):
    """Taxicab bound of the octagon: |dx| + |dy| <= m, m = round(r * sqrt(2)) (apothem r on the diagonals)."""
    return int(round(r * np.sqrt(2.0)))


def equal_area_radius(r):
    """Radius of the disk with the area of the (2r+1)^2 square."""
    return (2 * r + 1) / np.sqrt(np.pi)


def bf_sums(z, radii, shape, pad_mode="symmetric", r_eff=None):
    """Brute force: correlate the padded z, z**2 and the valid mask with the window mask (oracle, small arrays)."""
    R = int(max(radii)) if r_eff is None else int(np.ceil(max(r_eff)))
    zp, valid = _pad(z, R, pad_mode)
    out = []
    for k, r in enumerate(radii):
        m = mask(shape, r, None if r_eff is None else r_eff[k]).astype(np.float64)
        s1 = ndimage.correlate(zp, m, mode="constant", cval=0.0)[R:-R, R:-R]
        s2 = ndimage.correlate(zp * zp, m, mode="constant", cval=0.0)[R:-R, R:-R]
        n = ndimage.correlate(valid, m, mode="constant", cval=0.0)[R:-R, R:-R]
        out.append((s1, s2, np.rint(n)))
    return out


# ------------------------------------------------------------------------------------------------- helpers
def _pad(z, R, pad_mode):
    """Pad by R on every side; NaN -> 0 in the data, valid mask as float64."""
    zp = np.pad(np.asarray(z, dtype=np.float64), R, mode=pad_mode)
    valid = np.isfinite(zp)
    zp = np.where(valid, zp, 0.0)
    return zp, valid.astype(np.float64)


def _sat(a):
    """Summed-area table with a leading zero row and column: S[i+1, j+1] = sum(a[:i+1, :j+1])."""
    S = np.zeros((a.shape[0] + 1, a.shape[1] + 1))
    np.cumsum(a, axis=0, out=S[1:, 1:])
    np.cumsum(S[1:, 1:], axis=1, out=S[1:, 1:])
    return S


def _box(S, r, R, H, W):
    """Box sums of half-width r for the output window (H, W) of an array padded by R, from its SAT."""
    lo, hi = R - r, R + r + 1
    return (S[hi:hi + H, hi:hi + W] + S[lo:lo + H, lo:lo + W]) - S[hi:hi + H, lo:lo + W] - S[lo:lo + H, hi:hi + W]


# -------------------------------------------------------------------------------------------------- square
def square_sums(z, radii, pad_mode="symmetric"):
    R = int(max(radii))
    zp, valid = _pad(z, R, pad_mode)
    H, W = z.shape
    S1, S2 = _sat(zp), _sat(zp * zp)
    has_nan = not np.all(valid)
    SN = _sat(valid) if has_nan else None
    out = []
    for r in radii:
        n = _box(SN, r, R, H, W) if has_nan else np.full((H, W), float((2 * r + 1) ** 2))
        out.append((_box(S1, r, R, H, W), _box(S2, r, R, H, W), n))
    return out


# ---------------------------------------------------------------------------------------------------- disk
def disk_sums(z, radii, pad_mode="symmetric", r_eff=None, workers=1):
    """Exact discrete disk by FFT; the spectra of z, z**2 (and the mask) are computed once for all radii."""
    R = int(max(radii)) if r_eff is None else int(np.ceil(max(r_eff)))
    zp, valid = _pad(z, R, pad_mode)
    H, W = z.shape
    shape = [sfft.next_fast_len(s + 2 * R + 1, real=True) for s in zp.shape]
    has_nan = not np.all(valid)
    fields = [zp, zp * zp] + ([valid] if has_nan else [])
    F = [sfft.rfft2(q, shape, workers=workers) for q in fields]
    out = []
    for k, r in enumerate(radii):
        m = mask("disk", r, None if r_eff is None else r_eff[k]).astype(np.float64)
        Rk = m.shape[0] // 2
        K = sfft.rfft2(m, shape, workers=workers)
        sums = []
        for f in F:
            full = sfft.irfft2(f * K, shape, workers=workers)
            # 'full' convolution: output pixel (i, j) of the unpadded image sits at (i + R + Rk, j + R + Rk)
            o = R + Rk
            sums.append(np.ascontiguousarray(full[o:o + H, o:o + W]))
        n = np.rint(sums[2]) if has_nan else np.full((H, W), float(m.sum()))
        out.append((sums[0], sums[1], n))
    return out


# ------------------------------------------------------------------------------------------------- octagon
def _diag_table(a):
    """Anti-diagonal cumulative table of a (Hp, Wp) array with a leading zero row and column.

    G[i+1, k+1] = sum of a[y, x] over y <= i, x >= 0, y + x <= k (k in 0 .. Hp+Wp-2); G[0, :] = 0, G[:, 0] = 0.
    Built row by row from the row prefix sums: O(Hp * (Hp + Wp)) time and memory.
    """
    Hp, Wp = a.shape
    nk = Hp + Wp - 1
    P = np.zeros((Hp, Wp + 1))
    np.cumsum(a, axis=1, out=P[:, 1:])            # P[y, x+1] = sum a[y, :x+1]
    G = np.zeros((Hp + 1, nk + 1))
    ks = np.arange(nk)
    acc = np.zeros(nk)
    for y in range(Hp):
        idx = np.clip(ks - y, -1, Wp - 1) + 1        # x <= k - y, clamped to the row
        acc += P[y, idx]
        G[y + 1, 1:] = acc
    return G


def _tl_triangle(S, G, y0, x0, c, Hp):
    """Sum over the staircase triangle {y >= y0, x >= x0, (y - y0) + (x - x0) <= c - 1} for arrays y0, x0.

    S: SAT with leading zeros (indices shifted by +1); G: anti-diagonal table with leading zeros.
    = sum{y >= y0, x >= 0, y + x <= K} - sum{y0 <= y <= y0 + c - 1, x <= x0 - 1}, K = y0 + x0 + c - 1.
    """
    if c <= 0:
        return 0.0
    K = y0 + x0 + c - 1
    # the triangle occupies rows y0 .. y0 + c - 1 only (x >= x0 forces y <= K - x0); on those rows every
    # x <= x0 - 1 satisfies y + x <= K, so the part left of the triangle is a plain box
    a = G[y0 + c, K + 1] - G[y0, K + 1]             # rows y0 .. y0+c-1, x >= 0, y + x <= K  (G[y0, ·] = rows <= y0-1)
    b = S[y0 + c, x0] - S[y0, x0]                   # rows y0 .. y0+c-1, cols <= x0-1
    return a - b


def _octagon_from_tables(zp, radii, R, H, W):
    """Octagon sums for one field: box minus the four corner triangles; corners by flipping the field."""
    S = _sat(zp)
    res = [_box(S, r, R, H, W) for r in radii]
    Hp, Wp = zp.shape
    Y, X = np.mgrid[0:H, 0:W]
    for flip_y, flip_x in ((False, False), (False, True), (True, False), (True, True)):
        f = zp
        if flip_y:
            f = f[::-1, :]
        if flip_x:
            f = f[:, ::-1]
        f = np.ascontiguousarray(f)
        Sf = _sat(f)
        Gf = _diag_table(f)
        for k, r in enumerate(radii):
            c = 2 * r - octagon_m(r)                # cells cut along each leg of the corner
            if c <= 0:
                continue
            # top-left corner of the window in the flipped frame; the window centre is (Y + R, X + R)
            cy = (Hp - 1 - (Y + R)) if flip_y else (Y + R)
            cx = (Wp - 1 - (X + R)) if flip_x else (X + R)
            y0, x0 = cy - r, cx - r
            res[k] = res[k] - _tl_triangle(Sf, Gf, y0, x0, c, Hp)
        del Sf, Gf
    return res


def octagon_sums(z, radii, pad_mode="symmetric"):
    R = int(max(radii))
    zp, valid = _pad(z, R, pad_mode)
    H, W = z.shape
    has_nan = not np.all(valid)
    s1 = _octagon_from_tables(zp, radii, R, H, W)
    s2 = _octagon_from_tables(zp * zp, radii, R, H, W)
    if has_nan:
        n = [np.rint(v) for v in _octagon_from_tables(valid, radii, R, H, W)]
    else:
        n = [np.full((H, W), float(mask("octagon", r).sum())) for r in radii]
    return [(s1[k], s2[k], n[k]) for k in range(len(radii))]


# ------------------------------------------------------------------------------------------------ dispatch
def window_sums(z, radii, shape, pad_mode="symmetric", r_eff=None):
    radii = [int(r) for r in radii]
    if shape == "square":
        return square_sums(z, radii, pad_mode)
    if shape == "disk":
        return disk_sums(z, radii, pad_mode, r_eff=r_eff)
    if shape == "octagon":
        return octagon_sums(z, radii, pad_mode)
    if shape == "bf":
        raise ValueError("use bf_sums(z, radii, <shape>) for the brute-force oracle")
    raise ValueError(shape)


# ------------------------------------------------------------------------------------------- DEV and MSTP
def dev_max(z, radii, shape, pad_mode="symmetric", remove_mean=True, r_eff=None, min_disk_r=None):
    """DEV_max as rvt: dev = (z0 - mean) / (sqrt(|var|) + 1e-6), keep the value with the largest |dev| over radii.

    remove_mean: subtract the nanmean of the tile first (exact in exact arithmetic; numerically safer).
    min_disk_r: radii below it use the square window (the threshold rule of the brief); None = all radii `shape`.
    """
    z = np.asarray(z, dtype=np.float64)
    zbar = np.nanmean(z) if remove_mean else 0.0
    zz = z - zbar
    radii = [int(r) for r in radii]
    if min_disk_r is None or shape == "square":
        sums = window_sums(zz, radii, shape, pad_mode, r_eff)
    else:
        small = [r for r in radii if r < min_disk_r]
        large = [r for r in radii if r >= min_disk_r]
        sums = []
        if small:
            sums += window_sums(zz, small, "square", pad_mode)
        if large:
            sums += window_sums(zz, large, shape, pad_mode,
                                None if r_eff is None else [e for r, e in zip(radii, r_eff) if r >= min_disk_r])
    out = None
    with np.errstate(divide="ignore", invalid="ignore"):
        for s1, s2, n in sums:
            mean = s1 / n
            std = np.sqrt(np.abs(s2 / n - mean * mean)) + 1e-6
            dev = (zz - mean) / std
            out = dev if out is None else np.where(np.abs(out) >= np.abs(dev), out, dev)
    out[~np.isfinite(z)] = np.nan
    return out


def mstp_rgb(z, shape, scales=((1, 5, 1), (5, 50, 5), (50, 500, 50)), lightness=0.9, **kw):
    """MSTP as rvt.vis.mstp: bands R = broad, G = meso, B = local; 1 - exp(-lightness * |DEV_max|), clipped."""
    bands = []
    for lo, hi, step in reversed(scales):
        radii = list(range(lo, hi + 1, step))
        d = dev_max(z, radii, shape, **kw)
        bands.append(np.clip(1.0 - np.exp(-lightness * np.abs(d)), 0.0, 1.0))
    return np.stack(bands).astype(np.float32)


def mean_filter(z, r, shape, pad_mode="edge", remove_mean=True):
    """Mean over the window (rvt.vis.mean_filter semantics: pad edge, NaN ignored), for SLRM and MSRM."""
    z = np.asarray(z, dtype=np.float64)
    zbar = np.nanmean(z) if remove_mean else 0.0
    s1, _, n = window_sums(z - zbar, [r], shape, pad_mode)[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        m = s1 / n + zbar
    m[~np.isfinite(z)] = np.nan
    return m.astype(np.float32)
