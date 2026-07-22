#!/usr/bin/env python
"""
Orientation check by CATALOG + FLUX (per user).

In a crowded field every pixel finds *a* star, so position alone is
ambiguous.  Flux disambiguates: project a few well-measured, bright, isolated,
unsaturated catalog stars onto the PNG for each candidate orientation, and
require them to land on BRIGHT pixels with brightness that tracks catalog flux.

The AVM is built faithfully as a CDMatrix from the reference FITS WCS (pyavm's
Scale+Rotation is degenerate near PA=90 for JWST GC fields).  We test the 4
dimension-preserving orientations of that WCS and score each by how well the
selected stars' catalog flux correlates with the PNG brightness at their
predicted positions.

Usage:
  check_catalog_flux.py --fits ref.fits --png img.png --catalog cat.fits
                        [--nstars 8]
"""
import argparse
import json

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def load_fits_wcs(path):
    h = fits.getheader(path)
    if h.get('NAXIS', 0) < 2:
        h = fits.getheader(path, ext=('SCI', 1))
    return WCS(h).celestial, int(h['NAXIS2']), int(h['NAXIS1'])


def oriented_wcs(fwcs, ny, nx, kind):
    """Faithful CDMatrix-equivalent WCS in one of 4 orientations."""
    w = WCS(naxis=2)
    w.wcs.crpix = fwcs.wcs.crpix[:2].copy()
    w.wcs.crval = fwcs.wcs.crval[:2]
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    cd = np.array(fwcs.pixel_scale_matrix, float)
    crpix = np.array(w.wcs.crpix, float)
    fr = kind in ('flipud', 'rot180')
    fc = kind in ('fliplr', 'rot180')
    if fc:
        crpix[0] = nx + 1 - crpix[0]
        cd[:, 0] = -cd[:, 0]
    if fr:
        crpix[1] = ny + 1 - crpix[1]
        cd[:, 1] = -cd[:, 1]
    w.wcs.crpix = crpix
    w.wcs.cd = cd
    return w


def select_stars(catalog, nstars, fwcs, ny, nx, edge=64):
    t = Table.read(catalog)
    sc = t['skycoord']
    flux = np.array(t['flux'], float)
    ok = np.isfinite(flux) & (flux > 0)
    if 'qfit' in t.colnames:
        ok &= np.array(t['qfit'], float) < 0.05      # well-measured (real)
    if 'is_saturated' in t.colnames:
        ok &= ~np.array(t['is_saturated'], bool)     # unsaturated
    if 'replaced_saturated' in t.colnames:
        ok &= ~np.array(t['replaced_saturated'], bool)  # not sat-core replaced
    if 'flags' in t.colnames:
        ok &= np.array(t['flags'], int) == 0
    # brightest REAL UNSATURATED star is the anchor; saturated blobs excluded
    idx = np.where(ok)[0]
    ra = sc.ra.deg[idx]
    dec = sc.dec.deg[idx]
    flux = flux[idx]
    # keep only stars that fall inside THIS reference image footprint
    fx, fy = fwcs.world_to_pixel_values(ra, dec)
    inside = ((fx > edge) & (fx < nx - edge) &
              (fy > edge) & (fy < ny - edge))
    ra, dec, flux = ra[inside], dec[inside], flux[inside]
    # brightest first, then greedily keep isolated (>5" from already picked)
    order = np.argsort(flux)[::-1]
    picked = []
    for i in order:
        if all(((ra[i] - ra[j]) ** 2 + (dec[i] - dec[j]) ** 2) > (5 / 3600.) ** 2
               for j in picked):
            picked.append(i)
        if len(picked) >= nstars:
            break
    return ra[picked], dec[picked], flux[picked]


def png_brightness(lum, x, y, win=6):
    ny, nx = lum.shape
    xi, yi = int(round(x)), int(round(y))
    if not (win <= xi < nx - win and win <= yi < ny - win):
        return None
    return float(np.nanmax(lum[yi - win:yi + win + 1, xi - win:xi + win + 1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fits', required=True)
    p.add_argument('--png', required=True)
    p.add_argument('--catalog', required=True)
    p.add_argument('--nstars', type=int, default=8)
    args = p.parse_args()

    fwcs, ny, nx = load_fits_wcs(args.fits)
    ra, dec, flux = select_stars(args.catalog, args.nstars, fwcs, ny, nx)
    if len(ra) < 3:
        print(json.dumps({'error': f'only {len(ra)} usable catalog stars'}))
        return

    im = np.array(Image.open(args.png)).astype(float)
    lum = im[:, :, :3].sum(axis=2) if im.ndim == 3 else im
    lum = lum[::-1]                       # reproject_to_hips flips the PNG
    lo, hi = np.nanpercentile(lum, [50, 99.9])
    logf = np.log10(flux)

    res = {}
    for kind in ('identity', 'flipud', 'fliplr', 'rot180'):
        w = oriented_wcs(fwcs, ny, nx, kind)
        xs, ys = w.world_to_pixel_values(ra, dec)
        bright, fl = [], []
        for x, y, lf in zip(xs, ys, logf):
            b = png_brightness(lum, x, y)
            if b is not None:
                bright.append(b)
                fl.append(lf)
        if len(bright) < 3:
            res[kind] = {'n': len(bright), 'corr': None, 'frac_bright': None}
            continue
        bright = np.array(bright)
        fl = np.array(fl)
        # fraction of stars landing on a genuinely bright pixel
        frac_bright = float(np.mean(bright > hi * 0.5))
        # rank correlation of catalog flux vs PNG brightness
        from scipy.stats import spearmanr
        corr = float(spearmanr(fl, bright).correlation)
        res[kind] = {'n': int(len(bright)),
                     'corr_flux_brightness': round(corr, 3),
                     'frac_on_bright': round(frac_bright, 3),
                     'median_brightness': round(float(np.median(bright)), 1)}

    def score(k):
        r = res[k]
        if r.get('frac_on_bright') is None:
            return -9
        return r['frac_on_bright'] + 0.5 * max(0, r.get('corr_flux_brightness') or 0)
    best = max(res, key=score)
    print(json.dumps({'n_stars': len(ra), 'orientations': res, 'best': best,
                      'png_bright_thresh': round(float(hi * 0.5), 1)}, indent=2))


if __name__ == '__main__':
    main()
