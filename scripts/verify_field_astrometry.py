#!/usr/bin/env python
"""Measure a HiPS layer's astrometry from its SERVED tiles against source FITS.

Generalises verify_coadd.py's approach (written for the gc_treasury MIRI
coadd) to any (hips_dir, {label: source_fits}) set, so it can check
per-field layers -- e.g. cloudef's Cloudef_MIRI_F770W_hips against its o004
and o008 source mosaics -- as well as coadds.

Two traps this encodes (both bit earlier work on this repo):

  * Cut on each source's DATA CENTROID, not the image centre.  These mosaics
    are mostly blank; a centre cutout can land entirely on blank sky and
    return a small, meaningless offset that reads as a pass.
  * Apply a CORRELATION FLOOR (r >= 0.35).  Below it there is no measurement
    -- report "not measurable", not a number.

A third trap shows up specifically on a COADD of layers with different native
resolutions (e.g. jwst_miri_hips, which merges MIRI fields whose pixel scales
range widely): HipsSampler.hips_order() reports the DEEPEST order present
ANYWHERE in the directory, but a field whose own layer tops out shallower
(cloudef's MIRI native order is 12; some other layer in the same coadd reaches
14) has no leaf tile at that deeper order over its own footprint.  Sampling
naively at the reported order then returns all-NaN there -- not because the
coadd lacks the field, but because this simple sampler (unlike a real HiPS
client) does not climb the pyramid to a shallower populated tile.
_sample_with_fallback does that climb by hand.

Usage
-----
  verify_field_astrometry.py HIPS_DIR LABEL1=path1.fits [LABEL2=path2.fits ...]
"""
import sys

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from skimage.registration import phase_cross_correlation

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from check_hips_astrometry import HipsSampler, TILE_BITS  # noqa: E402

R_FLOOR = 0.35            # below this it is not a measurement
OFFSET_TARGET_ARCSEC = 0.3
MIN_FALLBACK_ORDER = 3    # give up climbing the pyramid below this


def _sample_with_fallback(hips_dir, coords, min_order=MIN_FALLBACK_ORDER):
    """HipsSampler.sample(), climbing to a shallower order if the reported
    (globally deepest) order has no tile over this particular patch.

    Returns (values, order_used).
    """
    from astropy_healpix import HEALPix

    samp = HipsSampler(hips_dir)
    vals = samp.sample(coords)
    order = samp.order
    while not np.isfinite(vals).any() and order > min_order:
        order -= 1
        samp.order = order
        samp.hp_hi = HEALPix(nside=2 ** (order + TILE_BITS), order="nested",
                             frame=samp.frame_name)
        samp._cache.clear()
        vals = samp.sample(coords)
    return vals, order


def prep(a):
    a = np.array(a, float)
    m = np.isfinite(a) & (a > 0)
    if m.sum() < 200:
        return None
    a = np.log10(np.clip(a, np.nanpercentile(a[m], 5), None))
    a = np.where(np.isfinite(a), a, np.nanmedian(a[np.isfinite(a)]))
    a -= a.mean()
    return a / a.std() if a.std() > 0 else None


def measure(hips, src, n=500, scale=0.4):
    """Offset (arcsec) and correlation of a HiPS against one source mosaic.

    Samples an n x n TAN patch centred on the source's DATA centroid (median
    pixel of finite, nonzero data) -- not the array centre -- from both the
    served HiPS tiles and a fresh reprojection of the source FITS, then
    cross-correlates the two.
    """
    hdu = next(h for h in fits.open(src) if h.data is not None and h.data.ndim == 2)
    d = hdu.data.astype(float)
    fw = WCS(hdu.header).celestial
    good = np.isfinite(d) & (d != 0)
    if good.sum() < 10000:
        return None, None, "source nearly empty"
    ys, xs = np.nonzero(good)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    ctr = fw.pixel_to_world(cx, cy)

    gw = WCS(naxis=2)
    gw.wcs.crpix = [n / 2 + 0.5, n / 2 + 0.5]
    gw.wcs.cdelt = [-scale / 3600, scale / 3600]
    gw.wcs.crval = [ctr.ra.deg, ctr.dec.deg]
    gw.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    yy, xx = np.mgrid[0:n, 0:n]
    got, _order = _sample_with_fallback(hips, gw.pixel_to_world(xx.ravel(), yy.ravel()))
    got = got.astype(float).reshape(n, n)
    ref, _ = reproject_interp((d, fw), gw, shape_out=(n, n))
    A, B = prep(got), prep(ref)
    if A is None or B is None:
        return None, None, "insufficient overlap"
    sh, _, _ = phase_cross_correlation(B, A, upsample_factor=20)
    r = float((A * B).mean())
    return float(np.hypot(*sh) * scale), r, None


def check(hips, sources):
    """Run measure() over a {label: path} dict and print a report.

    Returns (results, all_ok) where results maps label -> (offset, r) or
    None, and all_ok is False if any measurable offset exceeds
    OFFSET_TARGET_ARCSEC.
    """
    print(f"hips: {hips}")
    print(f"{'label':20s} {'offset':>9s} {'r':>7s}  note")
    results = {}
    all_ok = True
    for label, src in sources.items():
        off, r, note = measure(hips, src)
        if off is None:
            print(f"{label:20s} {'--':>9s} {'--':>7s}  {note}")
            results[label] = None
            continue
        if r < R_FLOOR:
            print(f"{label:20s} {off:8.3f}\" {r:+7.2f}  NOT MEASURABLE (r<{R_FLOOR})")
            results[label] = None
            continue
        flag = "" if off <= OFFSET_TARGET_ARCSEC else "  OFF"
        if off > OFFSET_TARGET_ARCSEC:
            all_ok = False
        results[label] = (off, r)
        print(f"{label:20s} {off:8.3f}\" {r:+7.2f}{flag}")
    offs = [v[0] for v in results.values() if v]
    if offs:
        print(f"\nmedian {np.median(offs):.3f}\"  max {np.max(offs):.3f}\"")
    return results, all_ok


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(1)
    hips = sys.argv[1]
    sources = dict(a.split("=", 1) for a in sys.argv[2:])
    _, ok = check(hips, sources)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
