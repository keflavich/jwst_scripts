#!/usr/bin/env python
"""
Sanity-check WCS/AVM/HiPS consistency for JWST RGB products.

Usage example:

python debug_wcs_avm_hips_consistency.py \
  --fits /orange/adamginsburg/jwst/w51/data_reprojected/jw06151-o001_t001_nircam_clear-f405n-merged_i2d_reprj_f140.fits \
  --png /orange/adamginsburg/jwst/w51/pngs_140/w51_RGB_405-360-335.png
"""

import argparse
import json
import os
import inspect

import numpy as np
from PIL import Image
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord
from reproject.hips import reproject_from_hips
import pyavm
from pyavm.exceptions import NoXMPPacketFound


def load_fits_wcs_and_shape(fits_path):
    with fits.open(fits_path) as hdul:
        sci_index = None
        for idx, hdu in enumerate(hdul):
            extname = hdu.header.get('EXTNAME', '')
            if extname == 'SCI':
                sci_index = idx
                break

        if sci_index is None:
            sci_index = 0

        header = hdul[sci_index].header
        data = hdul[sci_index].data
        if data is None:
            raise ValueError(f"No data found in FITS HDU index {sci_index}: {fits_path}")

        if data.ndim > 2:
            data = np.squeeze(data)

        if data.ndim != 2:
            raise ValueError(f"Expected 2D FITS image, got shape {data.shape} in {fits_path}")

        wcs = WCS(header).celestial
        shape = data.shape

    return wcs, shape


def load_image_rgb_and_avm_wcs(image_path, allow_missing_avm=False):
    image = Image.open(image_path)
    image_array = np.array(image)

    if image_array.ndim != 3:
        raise ValueError(f"Expected color image for {image_path}, got shape {image_array.shape}")

    if image_array.shape[2] == 4:
        rgb = image_array[:, :, :3]
    elif image_array.shape[2] == 3:
        rgb = image_array
    else:
        raise ValueError(f"Unexpected number of channels in {image_path}: {image_array.shape[2]}")

    try:
        avm = pyavm.AVM.from_image(image_path)
    except NoXMPPacketFound:
        if allow_missing_avm:
            return rgb.astype(float), None
        raise

    try:
        avm_wcs = avm.to_wcs(target_shape=rgb.shape[:2]).celestial
    except ValueError:
        try:
            avm_wcs = avm.to_wcs(target_shape=(rgb.shape[1], rgb.shape[0])).celestial
        except ValueError:
            avm_wcs = avm.to_wcs(use_full_header=True).celestial

    return rgb.astype(float), avm_wcs


def wcs_header_subset(wcs_obj):
    header = wcs_obj.to_header(relax=True)
    keys = []
    for key in header.keys():
        if key.startswith('CRPIX'):
            keys.append(key)
        elif key.startswith('CRVAL'):
            keys.append(key)
        elif key.startswith('CTYPE'):
            keys.append(key)
        elif key.startswith('CUNIT'):
            keys.append(key)
        elif key.startswith('CD'):
            keys.append(key)
        elif key.startswith('PC'):
            keys.append(key)
        elif key.startswith('CDELT'):
            keys.append(key)
        elif key.startswith('CROTA'):
            keys.append(key)
    return {key: header[key] for key in sorted(set(keys))}


def compare_wcs_headers(fits_wcs, avm_wcs):
    fits_header = wcs_header_subset(fits_wcs)
    avm_header = wcs_header_subset(avm_wcs)

    all_keys = sorted(set(fits_header) | set(avm_header))
    exact_matches = {}
    diffs = {}

    for key in all_keys:
        in_fits = key in fits_header
        in_avm = key in avm_header

        if in_fits and in_avm:
            exact = fits_header[key] == avm_header[key]
            exact_matches[key] = exact
            if not exact:
                fits_val = fits_header[key]
                avm_val = avm_header[key]
                if isinstance(fits_val, (int, float)) and isinstance(avm_val, (int, float)):
                    diffs[key] = {
                        'fits': float(fits_val),
                        'avm': float(avm_val),
                        'abs_diff': float(abs(fits_val - avm_val)),
                    }
                else:
                    diffs[key] = {
                        'fits': str(fits_val),
                        'avm': str(avm_val),
                    }
        else:
            exact_matches[key] = False
            diffs[key] = {
                'fits': fits_header.get(key, '<missing>'),
                'avm': avm_header.get(key, '<missing>'),
            }

    all_exact = all(exact_matches.values()) if exact_matches else False

    return {
        'all_exact': all_exact,
        'exact_matches': exact_matches,
        'differences': diffs,
    }


def grid_points(shape, ngrid):
    ny, nx = shape
    ys = np.linspace(0, ny - 1, ngrid)
    xs = np.linspace(0, nx - 1, ngrid)
    xv, yv = np.meshgrid(xs, ys)
    return xv.ravel(), yv.ravel()


def compare_pixel_to_sky(fits_wcs, avm_wcs, shape, ngrid):
    xs, ys = grid_points(shape, ngrid)

    c_fits = pixel_to_skycoord(xs, ys, fits_wcs, origin=0)
    c_avm = pixel_to_skycoord(xs, ys, avm_wcs, origin=0)

    sep_arcsec = c_fits.separation(c_avm).arcsec

    return {
        'n_samples': int(sep_arcsec.size),
        'max_sep_arcsec': float(np.nanmax(sep_arcsec)),
        'median_sep_arcsec': float(np.nanmedian(sep_arcsec)),
        'mean_sep_arcsec': float(np.nanmean(sep_arcsec)),
    }


def compare_png_jpg_rgb(png_rgb, jpg_rgb):
    if png_rgb.shape != jpg_rgb.shape:
        raise ValueError(f"PNG/JPG shape mismatch: {png_rgb.shape} vs {jpg_rgb.shape}")

    diff = np.abs(png_rgb - jpg_rgb)

    return {
        'shape': tuple(int(x) for x in png_rgb.shape),
        'mean_abs_diff': float(np.mean(diff)),
        'median_abs_diff': float(np.median(diff)),
        'max_abs_diff': float(np.max(diff)),
    }


def normalized_gray(rgb):
    gray = np.mean(rgb, axis=2)
    finite = np.isfinite(gray)
    if not np.any(finite):
        return gray

    vals = gray[finite]
    lo = np.percentile(vals, 1)
    hi = np.percentile(vals, 99)
    if hi <= lo:
        hi = lo + 1e-6

    out = (gray - lo) / (hi - lo)
    out = np.clip(out, 0, 1)
    return out


def correlation(a, b, mask):
    a1 = a[mask]
    b1 = b[mask]

    if a1.size < 10:
        return np.nan

    a1 = a1 - np.nanmean(a1)
    b1 = b1 - np.nanmean(b1)

    denom = np.sqrt(np.nansum(a1 * a1) * np.nansum(b1 * b1))
    if denom == 0:
        return np.nan

    return float(np.nansum(a1 * b1) / denom)


def hips_orientation_check(hips_dir, target_wcs, shape_out, png_rgb):
    hips_data, hips_foot = reproject_from_hips(
        input_data=hips_dir,
        output_projection=target_wcs,
        shape_out=shape_out,
    )

    if hips_data.ndim == 3:
        hips_gray = normalized_gray(hips_data)
    else:
        hips_gray = hips_data.astype(float)
        hips_gray = (hips_gray - np.nanpercentile(hips_gray, 1)) / (
            np.nanpercentile(hips_gray, 99) - np.nanpercentile(hips_gray, 1) + 1e-6
        )
        hips_gray = np.clip(hips_gray, 0, 1)

    png_gray = normalized_gray(png_rgb)

    valid = np.isfinite(hips_gray) & np.isfinite(png_gray) & (hips_foot > 0)

    corr_identity = correlation(hips_gray, png_gray, valid)
    corr_lr = correlation(hips_gray, np.fliplr(png_gray), valid)
    corr_ud = correlation(hips_gray, np.flipud(png_gray), valid)
    corr_rot180 = correlation(hips_gray, np.rot90(png_gray, 2), valid)

    corrs = {
        'identity': corr_identity,
        'flip_lr': corr_lr,
        'flip_ud': corr_ud,
        'rot180': corr_rot180,
    }

    best = max(corrs, key=lambda k: (-999 if np.isnan(corrs[k]) else corrs[k]))

    return {
        'coverage_fraction': float(np.mean(hips_foot > 0)),
        'correlations': corrs,
        'best_orientation_match': best,
    }


def hips_reproject_available():
    sig = inspect.signature(reproject_from_hips)
    return len(sig.parameters) > 0


def main():
    parser = argparse.ArgumentParser(description='Check FITS/AVM/PNG/JPG/HiPS WCS consistency.')
    parser.add_argument('--fits', required=True, help='Reference FITS file with expected WCS.')
    parser.add_argument('--png', required=True, help='PNG image with AVM metadata.')
    parser.add_argument('--jpg', default=None, help='JPG image with AVM metadata (default: PNG stem + .jpg).')
    parser.add_argument('--hips-dir', default=None, help='HiPS directory (default: PNG stem + _hips).')
    parser.add_argument('--grid', type=int, default=5, help='Grid size per axis for WCS sky-separation sampling.')
    parser.add_argument('--json', default=None, help='Optional output JSON report path.')

    args = parser.parse_args()

    jpg_path = args.jpg if args.jpg is not None else args.png.replace('.png', '.jpg')
    hips_dir = args.hips_dir if args.hips_dir is not None else args.png.replace('.png', '_hips')

    fits_wcs, fits_shape = load_fits_wcs_and_shape(args.fits)
    png_rgb, png_avm_wcs = load_image_rgb_and_avm_wcs(args.png, allow_missing_avm=True)
    jpg_rgb, jpg_avm_wcs = load_image_rgb_and_avm_wcs(jpg_path, allow_missing_avm=True)

    if png_avm_wcs is None:
        raise ValueError(f"PNG has no AVM/XMP metadata: {args.png}")

    if png_rgb.shape[:2] != fits_shape:
        print(f"WARNING: PNG shape {png_rgb.shape[:2]} != FITS shape {fits_shape}. WCS sample check uses PNG shape.")

    shape_for_wcs = png_rgb.shape[:2]

    report = {
        'inputs': {
            'fits': args.fits,
            'png': args.png,
            'jpg': jpg_path,
            'hips_dir': hips_dir,
        },
        'avm_presence': {
            'png_has_avm': png_avm_wcs is not None,
            'jpg_has_avm': jpg_avm_wcs is not None,
        },
        'fits_vs_png_avm_header': compare_wcs_headers(fits_wcs, png_avm_wcs),
        'fits_vs_png_avm_sky': compare_pixel_to_sky(fits_wcs, png_avm_wcs, shape_for_wcs, args.grid),
        'png_vs_jpg_rgb': compare_png_jpg_rgb(png_rgb, jpg_rgb),
    }

    if jpg_avm_wcs is not None:
        report['fits_vs_jpg_avm_header'] = compare_wcs_headers(fits_wcs, jpg_avm_wcs)
        report['png_avm_vs_jpg_avm_header'] = compare_wcs_headers(png_avm_wcs, jpg_avm_wcs)
        report['fits_vs_jpg_avm_sky'] = compare_pixel_to_sky(fits_wcs, jpg_avm_wcs, shape_for_wcs, args.grid)
    else:
        report['fits_vs_jpg_avm_header'] = {'skipped': True, 'reason': 'JPG has no AVM/XMP metadata'}
        report['png_avm_vs_jpg_avm_header'] = {'skipped': True, 'reason': 'JPG has no AVM/XMP metadata'}
        report['fits_vs_jpg_avm_sky'] = {'skipped': True, 'reason': 'JPG has no AVM/XMP metadata'}

    if os.path.isdir(hips_dir) and hips_reproject_available():
        report['hips_vs_png'] = hips_orientation_check(
            hips_dir=hips_dir,
            target_wcs=png_avm_wcs,
            shape_out=png_rgb.shape[:2],
            png_rgb=png_rgb,
        )
    elif os.path.isdir(hips_dir):
        report['hips_vs_png'] = {
            'skipped': True,
            'reason': 'Installed reproject.hips.reproject_from_hips is not implemented in this environment',
        }
    else:
        report['hips_vs_png'] = {'skipped': True, 'reason': f'HiPS directory not found: {hips_dir}'}

    print(json.dumps(report, indent=2, sort_keys=True))

    if args.json is not None:
        with open(args.json, 'w') as fh:
            json.dump(report, fh, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
