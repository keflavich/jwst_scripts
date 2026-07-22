#!/usr/bin/env python
"""
Determine the correct AVM for a PNG by STAR POSITIONS, and (optionally) embed it.

pyavm.AVM.from_header is lossy for rotated fields, so the embedded AVM can be
rotationally wrong (stars land at the wrong sky position, growing from the
image center outward).  This tool ignores the existing AVM entirely and derives
the correct one from the reference FITS WCS:

  1. detect stars in the reference FITS -> trusted sky positions (full WCS incl SIP)
  2. detect stars in the PNG (in the flipud orientation reproject_to_hips uses)
  3. for each dimension-preserving orientation of the FITS *linear* WCS
     (identity / flipud / fliplr / rot180), map the PNG stars to sky via that
     candidate WCS and cross-match to the FITS stars
  4. the orientation with the most matches is the correct AVM; store it as a
     flat Spatial.CDMatrix (faithful; pyavm to_wcs honors it verbatim)

Usage:
  find_correct_avm.py --fits ref.fits --png img.png [--embed] [--tol-arcsec 1.5]
"""
import argparse
import json
import os
import shutil

import numpy as np
import pyavm
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def load_fits_2d(path):
    with fits.open(path) as hdul:
        idx = 0
        for i, h in enumerate(hdul):
            if h.header.get('EXTNAME', '') == 'SCI':
                idx = i
                break
        hdr, data = hdul[idx].header, hdul[idx].data
        if data is None:
            for h in hdul:
                if h.data is not None:
                    hdr, data = h.header, h.data
                    break
        data = np.squeeze(data).astype(float)
        wcs = WCS(hdr).celestial
    return data, wcs


def detect_xy(data, maxn=300, isolated_px=0):
    finite = np.isfinite(data)
    _, med, std = sigma_clipped_stats(data[finite], sigma=3.0)
    src = DAOStarFinder(fwhm=3.0, threshold=15 * std,
                        exclude_border=True)(np.nan_to_num(data - med))
    if src is None or len(src) == 0:
        return None, None
    xc = 'x_centroid' if 'x_centroid' in src.colnames else 'xcentroid'
    yc = 'y_centroid' if 'y_centroid' in src.colnames else 'ycentroid'
    src.sort('flux')
    src.reverse()
    x = np.array(src[xc], float)
    y = np.array(src[yc], float)
    if isolated_px > 0 and len(x) > 1:
        # keep only stars with no detected neighbour within isolated_px
        keep = []
        for i in range(len(x)):
            d2 = (x - x[i]) ** 2 + (y - y[i]) ** 2
            d2[i] = np.inf
            if d2.min() > isolated_px ** 2:
                keep.append(i)
            if len(keep) >= maxn:
                break
        x, y = x[keep], y[keep]
    else:
        x, y = x[:maxn], y[:maxn]
    return x, y


def linear_wcs(wcs):
    """A pure-linear celestial WCS (drop SIP/distortion) from a WCS's CD/crval."""
    w = WCS(naxis=2)
    w.wcs.crpix = wcs.wcs.crpix[:2]
    w.wcs.crval = wcs.wcs.crval[:2]
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    cd = wcs.pixel_scale_matrix
    w.wcs.cd = cd
    return w


def oriented_wcs(linwcs, ny, nx, kind):
    """Return a linear WCS describing the PNG under the given orientation
    relative to the FITS array (identity/flipud/fliplr/rot180)."""
    w = linwcs.deepcopy()
    cd = np.array(w.wcs.cd, float)
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


def cdmatrix_avm(w, ny, nx):
    w = w.deepcopy()
    w.pixel_shape = (nx, ny)
    a = pyavm.AVM.from_wcs(w, shape=(ny, nx))
    cd = w.pixel_scale_matrix
    a.Spatial.CDMatrix = [cd[0, 0], cd[0, 1], cd[1, 0], cd[1, 1]]
    a.Spatial.Scale = None
    a.Spatial.Rotation = None
    return a


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fits', required=True)
    p.add_argument('--png', required=True)
    p.add_argument('--tol-arcsec', type=float, default=1.5, dest='tol')
    p.add_argument('--embed', action='store_true',
                   help='embed the best-matching CDMatrix AVM into the PNG')
    p.add_argument('--min-frac', type=float, default=0.3, dest='min_frac')
    args = p.parse_args()

    data, fwcs = load_fits_2d(args.fits)
    fx, fy = detect_xy(data, maxn=40, isolated_px=40)
    if fx is None or len(fx) < 4:
        print(json.dumps({'error': 'too few isolated FITS stars'}))
        return
    fits_sky = fwcs.pixel_to_world(fx, fy)   # full WCS incl SIP

    im = np.array(Image.open(args.png)).astype(float)
    lum = im[:, :, :3].sum(axis=2) if im.ndim == 3 else im
    lum = lum[::-1]  # reproject_to_hips flips the PNG vertically
    px, py = detect_xy(lum, maxn=300, isolated_px=20)
    if px is None or len(px) < 4:
        print(json.dumps({'error': 'too few PNG stars'}))
        return

    ny, nx = lum.shape
    linw = linear_wcs(fwcs)
    results = {}
    best = None
    for kind in ('identity', 'flipud', 'fliplr', 'rot180'):
        w = oriented_wcs(linw, ny, nx, kind)
        png_sky = w.pixel_to_world(px, py)
        _, sep2d, _ = fits_sky.match_to_catalog_sky(png_sky)
        frac = float(np.mean(sep2d.arcsec < args.tol))
        medsep = float(np.median(sep2d.arcsec))
        results[kind] = {'match_fraction': round(frac, 3),
                         'median_sep_arcsec': round(medsep, 2)}
        if best is None or frac > results[best]['match_fraction']:
            best = kind

    out = {'png': args.png, 'fits': args.fits, 'tol_arcsec': args.tol,
           'orientations': results, 'best': best,
           'best_fraction': results[best]['match_fraction'],
           'confident': results[best]['match_fraction'] >= args.min_frac}
    if args.embed:
        if not out['confident']:
            out['embedded'] = False
            out['embed_skipped'] = 'no orientation matched confidently'
        else:
            w = oriented_wcs(linw, ny, nx, best)
            avm = cdmatrix_avm(w, ny, nx)
            tmp = args.png + '.avm.png'
            avm.embed(args.png, tmp)
            shutil.move(tmp, args.png)
            out['embedded'] = True
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
