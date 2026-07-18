#!/usr/bin/env python
"""
Check whether a PNG (with embedded AVM) is orientation-consistent with a
reference FITS WCS -- which is exactly what determines whether the HiPS built
from that PNG (via reproject_to_hips, which reads PNG + AVM) shows up flipped.

reproject.hips.reproject_from_hips is a no-arg stub in this env, so we cannot
round-trip through the HiPS tiles.  Instead we forward-reproject the PNG's
luminance through its AVM WCS onto the reference FITS WCS grid and correlate
against the FITS morphology in 4 orientations.  If the AVM correctly describes
the PNG pixels, reproject_to_hips produces a correctly-oriented HiPS and the
best correlation here is 'identity'.  Any other best match is the flip that
will appear in the HiPS.

Usage:
  check_hips_orientation.py --png a.png --fits ref.fits [--json out.json]
"""
import argparse
import json
import os

import numpy as np
from PIL import Image
from astropy.io import fits
from astropy.wcs import WCS
import pyavm
from pyavm.exceptions import NoXMPPacketFound
from reproject import reproject_interp


def load_fits_wcs_and_data(fits_path):
    with fits.open(fits_path) as hdul:
        sci_index = 0
        for idx, hdu in enumerate(hdul):
            if hdu.header.get('EXTNAME', '') == 'SCI':
                sci_index = idx
                break
        header = hdul[sci_index].header
        data = hdul[sci_index].data
        if data is None:
            # try first HDU with data
            for hdu in hdul:
                if hdu.data is not None:
                    header, data = hdu.header, hdu.data
                    break
        data = np.squeeze(data)
        if data.ndim != 2:
            raise ValueError(f"Expected 2D FITS, got {data.shape} in {fits_path}")
        wcs = WCS(header).celestial
    return wcs, data


def load_png_luminance_and_avm(png_path):
    image = np.array(Image.open(png_path)).astype(float)
    if image.ndim != 3:
        raise ValueError(f"Expected color image {png_path}, got {image.shape}")
    rgb = image[:, :, :3]
    lum = rgb.mean(axis=2)
    avm = pyavm.AVM.from_image(png_path)
    shape = lum.shape
    try:
        avm_wcs = avm.to_wcs(target_shape=shape).celestial
    except ValueError:
        try:
            avm_wcs = avm.to_wcs(target_shape=(shape[1], shape[0])).celestial
        except ValueError:
            avm_wcs = avm.to_wcs(use_full_header=True).celestial
    return lum, avm_wcs


def norm(a):
    finite = np.isfinite(a)
    if not finite.any():
        return a
    v = a[finite]
    lo, hi = np.percentile(v, 1), np.percentile(v, 99)
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((a - lo) / (hi - lo), 0, 1)


def corr(a, b, mask):
    a1, b1 = a[mask], b[mask]
    if a1.size < 50:
        return float('nan')
    a1 = a1 - np.nanmean(a1)
    b1 = b1 - np.nanmean(b1)
    d = np.sqrt(np.nansum(a1 * a1) * np.nansum(b1 * b1))
    if d == 0:
        return float('nan')
    return float(np.nansum(a1 * b1) / d)


def check(png_path, fits_path, maxdim=800):
    fits_wcs, fits_data = load_fits_wcs_and_data(fits_path)
    lum, avm_wcs = load_png_luminance_and_avm(png_path)

    # downsample the reference grid so orientation reprojection is fast;
    # orientation only needs coarse morphology.
    ny, nx = fits_data.shape
    step = max(1, int(np.ceil(max(ny, nx) / maxdim)))
    if step > 1:
        fits_wcs = fits_wcs[::step, ::step]
        fits_data = fits_data[::step, ::step]

    # reproject PNG luminance (via its AVM WCS) onto the FITS grid
    repr_lum, foot = reproject_interp((lum, avm_wcs), fits_wcs,
                                      shape_out=fits_data.shape)

    a = norm(fits_data)
    b = norm(repr_lum)
    mask = np.isfinite(a) & np.isfinite(b) & (foot > 0)

    corrs = {
        'identity': corr(a, b, mask),
        'flip_lr': corr(a, np.fliplr(b), mask),
        'flip_ud': corr(a, np.flipud(b), mask),
        'rot180': corr(a, np.rot90(b, 2), mask),
    }
    best = max(corrs, key=lambda k: (-2 if np.isnan(corrs[k]) else corrs[k]))
    return {
        'png': png_path,
        'fits': fits_path,
        'coverage': float(np.mean(foot > 0)),
        'correlations': corrs,
        'best': best,
        'ok': best == 'identity',
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--png', required=True)
    p.add_argument('--fits', required=True)
    p.add_argument('--json', default=None)
    args = p.parse_args()
    try:
        rep = check(args.png, args.fits)
    except NoXMPPacketFound:
        rep = {'png': args.png, 'error': 'no AVM in PNG'}
    print(json.dumps(rep, indent=2))
    if args.json:
        with open(args.json, 'w') as fh:
            json.dump(rep, fh, indent=2)


if __name__ == '__main__':
    main()
