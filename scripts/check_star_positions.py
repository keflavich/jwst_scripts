#!/usr/bin/env python
"""
Robust orientation check by STAR POSITIONS (not morphology correlation).

Idea (per user): detect a few bright, unsaturated, well-separated stars in a
reference FITS (with a trusted WCS); get their sky coords.  Then verify that a
real star actually sits at each star's predicted location in
  (1) the PNG, via its embedded AVM WCS, and
  (2) the HiPS tiles.
If the AVM/PNG/HiPS place the SAME stars at the SAME sky positions, the
orientation is correct.  Using several stars avoids accidental matches.

PNG origin note: PIL loads row 0 = top, but the AVM's to_wcs() is FITS
convention (row 0 = bottom), and reproject_to_hips reconciles this by flipping
the image vertically (reproject/utils.py [:, ::-1]).  We flip the PNG the same
way before indexing it with the AVM WCS.

Usage:
  check_star_positions.py --fits ref.fits --png img.png [--nstars 4]
                          [--hips DIR] [--search 15]
"""
import argparse
import json
import os

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord
from photutils.detection import DAOStarFinder
from PIL import Image
import pyavm

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


def detect_stars(data, nstars, edge=64):
    finite = np.isfinite(data)
    mean, med, std = sigma_clipped_stats(data[finite], sigma=3.0)
    # bright but not saturated: threshold high, cap on peak
    sat = np.nanpercentile(data[finite], 99.9)
    dao = DAOStarFinder(fwhm=3.0, threshold=30 * std, exclude_border=True)
    src = dao(data - med)
    if src is None or len(src) == 0:
        dao = DAOStarFinder(fwhm=3.0, threshold=10 * std, exclude_border=True)
        src = dao(data - med)
    if src is None or len(src) == 0:
        return []
    ny, nx = data.shape
    xc = 'x_centroid' if 'x_centroid' in src.colnames else 'xcentroid'
    yc = 'y_centroid' if 'y_centroid' in src.colnames else 'ycentroid'
    src = src[(src['peak'] < sat) & (src['peak'] > 0)]
    src = src[(src[xc] > edge) & (src[xc] < nx - edge) &
              (src[yc] > edge) & (src[yc] < ny - edge)]
    src.sort('flux')
    src.reverse()
    # pick well-separated: greedily take brightest, skip any within min_sep
    min_sep = 0.15 * min(ny, nx)
    picked = []
    xcol = np.array(src[xc], dtype=float)
    ycol = np.array(src[yc], dtype=float)
    for x, y in zip(xcol, ycol):
        if all((x - px) ** 2 + (y - py) ** 2 > min_sep ** 2 for px, py in picked):
            picked.append((x, y))
        if len(picked) >= nstars:
            break
    return picked


def brightest_near(img, x, y, search):
    ny, nx = img.shape
    x0, y0 = int(round(x)), int(round(y))
    if not (0 <= x0 < nx and 0 <= y0 < ny):
        return None
    lo_y, hi_y = max(0, y0 - search), min(ny, y0 + search + 1)
    lo_x, hi_x = max(0, x0 - search), min(nx, x0 + search + 1)
    win = img[lo_y:hi_y, lo_x:hi_x]
    if win.size == 0 or not np.isfinite(win).any():
        return None
    dy, dx = np.unravel_index(np.nanargmax(win), win.shape)
    return (int(lo_x + dx), int(lo_y + dy))


def detect_all_sky(data, wcs, maxn=200, edge=32):
    """Detect stars in a 2D image, return SkyCoord list via wcs."""
    finite = np.isfinite(data)
    if not finite.any():
        return None
    mean, med, std = sigma_clipped_stats(data[finite], sigma=3.0)
    dao = DAOStarFinder(fwhm=3.0, threshold=15 * std, exclude_border=True)
    src = dao(np.nan_to_num(data - med))
    if src is None or len(src) == 0:
        return None
    xc = 'x_centroid' if 'x_centroid' in src.colnames else 'xcentroid'
    yc = 'y_centroid' if 'y_centroid' in src.colnames else 'ycentroid'
    ny, nx = data.shape
    src = src[(src[xc] > edge) & (src[xc] < nx - edge) &
              (src[yc] > edge) & (src[yc] < ny - edge)]
    src.sort('flux')
    src.reverse()
    src = src[:maxn]
    return wcs.pixel_to_world(np.array(src[xc], float), np.array(src[yc], float))


def check_png(fits_path, png_path, nstars, search):
    # 1. stars in reference FITS -> trusted sky positions
    data, fwcs = load_fits_2d(fits_path)
    fits_sky = detect_all_sky(data, fwcs)
    if fits_sky is None or len(fits_sky) < 5:
        return {'error': 'too few FITS stars'}

    # 2. stars in PNG -> sky via its AVM (flip like reproject_to_hips)
    im = np.array(Image.open(png_path)).astype(float)
    lum = im[:, :, :3].sum(axis=2) if im.ndim == 3 else im
    lum = lum[::-1]
    awcs = pyavm.AVM.from_image(png_path).to_wcs().celestial
    png_sky = detect_all_sky(lum, awcs)
    if png_sky is None or len(png_sky) < 5:
        return {'error': 'too few PNG stars'}

    # 3. cross-match FITS stars to nearest PNG star by sky position
    idx, sep2d, _ = fits_sky.match_to_catalog_sky(png_sky)
    sep = sep2d.arcsec
    # pixel scale (arcsec/px) from FITS WCS for a sensible tolerance
    scale = np.abs(fwcs.proj_plane_pixel_scales()[0].to('arcsec').value)
    tol = max(0.5, 3 * scale)  # a few pixels
    matched = int(np.sum(sep < tol))
    frac = matched / len(fits_sky)
    return {
        'n_fits_stars': int(len(fits_sky)),
        'n_png_stars': int(len(png_sky)),
        'pixscale_arcsec': round(float(scale), 4),
        'tol_arcsec': round(float(tol), 3),
        'n_matched': matched,
        'match_fraction': round(frac, 3),
        'median_sep_arcsec': round(float(np.median(sep)), 3),
        'match': frac > 0.3,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fits', required=True)
    p.add_argument('--png', required=True)
    p.add_argument('--nstars', type=int, default=4)
    p.add_argument('--search', type=int, default=15)
    p.add_argument('--json', default=None)
    args = p.parse_args()
    rep = check_png(args.fits, args.png, args.nstars, args.search)
    print(json.dumps(rep, indent=2))
    if args.json:
        with open(args.json, 'w') as fh:
            json.dump(rep, fh, indent=2)


if __name__ == '__main__':
    main()
