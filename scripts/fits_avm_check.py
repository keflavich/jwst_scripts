#!/usr/bin/env python
"""
Step 1 -- FITS <-> AVM orientation check (discrete, no stars).

The PNG's embedded AVM is derived from the FITS and shares its pixel grid, so
the only possible discrepancy is a discrete dihedral flip (identity / flipud /
fliplr / rot180) -- never a sub-pixel offset.  This compares the embedded AVM
WCS to the FITS WCS at the image corners+center and reports which flip relates
them.

A clean result is one dihedral with ~0 residual.  If NO dihedral gives ~0
residual, the AVM is not a clean flip of the FITS -- that is the pyavm
Scale+Rotation degeneracy near PA=90 (JWST GC roll), and the fix is to embed a
CDMatrix AVM instead (see fix step).

Usage:
  fits_avm_check.py --fits ref.fits --png img.png
"""
import argparse
import json

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image
import pyavm

Image.MAX_IMAGE_PIXELS = None


def fits_wcs_shape(path):
    h = fits.getheader(path)
    if h.get('NAXIS', 0) < 2:
        h = fits.getheader(path, ext=('SCI', 1))
    return WCS(h).celestial, int(h['NAXIS2']), int(h['NAXIS1'])


def flip_pixel(x, y, ny, nx, kind):
    """Map a pixel under a dihedral flip of an (ny,nx) array (0-based)."""
    if kind in ('fliplr', 'rot180'):
        x = (nx - 1) - x
    if kind in ('flipud', 'rot180'):
        y = (ny - 1) - y
    return x, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fits', required=True)
    p.add_argument('--png', required=True)
    args = p.parse_args()

    fwcs, ny, nx = fits_wcs_shape(args.fits)
    avm = pyavm.AVM.from_image(args.png)
    awcs = avm.to_wcs().celestial

    # sample points: corners + center + edge midpoints
    pts = [(0, 0), (nx - 1, 0), (0, ny - 1), (nx - 1, ny - 1),
           (nx // 2, ny // 2), (nx // 2, 0), (0, ny // 2)]
    px = np.array([q[0] for q in pts], float)
    py = np.array([q[1] for q in pts], float)
    avm_sky = awcs.pixel_to_world(px, py)          # where AVM puts each pixel

    results = {}
    for kind in ('identity', 'flipud', 'fliplr', 'rot180'):
        fx, fy = flip_pixel(px, py, ny, nx, kind)
        fits_sky = fwcs.pixel_to_world(fx, fy)     # FITS at the flipped pixel
        sep = avm_sky.separation(fits_sky).arcsec
        results[kind] = round(float(np.max(sep)), 3)

    best = min(results, key=results.get)
    clean = results[best] < 1.0
    out = {
        'fits': args.fits, 'png': args.png,
        'max_sep_arcsec_per_flip': results,
        'best_flip': best,
        'best_max_sep_arcsec': results[best],
        'clean_dihedral': clean,
        'avm_matches_fits': (best == 'identity' and clean),
        'note': ('AVM = FITS (identity)' if best == 'identity' and clean else
                 f'AVM = {best} of FITS' if clean else
                 'AVM is NOT a clean flip of FITS -> pyavm Scale+Rotation '
                 'degeneracy; re-embed as CDMatrix'),
    }
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
