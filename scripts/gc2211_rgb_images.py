#!/usr/bin/env python
# coding: utf-8
"""
RGB / single-filter PNG + HiPS + AVM generation for JWST project 2211
(Galactic Center NIRCam survey).

Per-OBS fields (5 of them), each with a different filter mix.  No single field
has all three filters, so we make:
  - one single-filter asinh + log PNG per available filter per OBS
  - a 2-filter pseudo-RGB per OBS (R=longest, B=shortest, G=mean)
"""

from astropy.io import fits
import numpy as np
from astropy.visualization import simple_norm
from astropy.wcs import WCS
import os
from reproject import reproject_interp
import pyavm
from jwst_rgb.save_rgb import save_rgb as _save_rgb
from jwst_rgb.save_rgb import faithful_avm
from PIL import Image


CURRENT_TARGET_FILTER_IS_MIRI = False  # all NIRCam


def save_rgb(*args, **kwargs):
    kwargs.setdefault('transpose', Image.ROTATE_180 if CURRENT_TARGET_FILTER_IS_MIRI else None)
    kwargs.setdefault('alpha_only_edges', True)
    return _save_rgb(*args, **kwargs)


MERGED_DIR = "/orange/adamginsburg/jwst/gc2211/images-merged"
BASEPATH = "/orange/adamginsburg/jwst/gc2211"

# Per-OBS available filters (built from images-merged listing 2026-05-15)
obs_images = {
    "o023": {
        "f277w": f"{MERGED_DIR}/jw02211-o023_t001_nircam_clear-f277w-merged_i2d.fits",
    },
    "o028": {
        "f150w": f"{MERGED_DIR}/jw02211-o028_t001_nircam_clear-f150w-merged_i2d.fits",
        "f277w": f"{MERGED_DIR}/jw02211-o028_t001_nircam_clear-f277w-merged_i2d.fits",
    },
    "o046": {
        "f200w": f"{MERGED_DIR}/jw02211-o046_t001_nircam_clear-f200w-merged_i2d.fits",
        "f277w": f"{MERGED_DIR}/jw02211-o046_t001_nircam_clear-f277w-merged_i2d.fits",
    },
    "o049": {
        "f200w": f"{MERGED_DIR}/jw02211-o049_t001_nircam_clear-f200w-merged_i2d.fits",
        "f277w": f"{MERGED_DIR}/jw02211-o049_t001_nircam_clear-f277w-merged_i2d.fits",
    },
    "o050": {
        "f200w": f"{MERGED_DIR}/jw02211-o050_t001_nircam_clear-f200w-merged_i2d.fits",
        "f277w": f"{MERGED_DIR}/jw02211-o050_t001_nircam_clear-f277w-merged_i2d.fits",
    },
}


def _filter_wavelength(name):
    return int(''.join(filter(str.isdigit, name)))


def reproject_to_target(filenames, target_filter, new_basepath):
    """Reproject every filter in `filenames` to the WCS of filenames[target_filter].
    Returns dict {filter: reprojected_fits_path}."""
    os.makedirs(new_basepath, exist_ok=True)
    tgt_header = fits.getheader(filenames[target_filter], ext=('SCI', 1))
    out = {}
    for filt, src in filenames.items():
        outpath = os.path.join(
            new_basepath,
            os.path.basename(src).replace(
                ".fits", f"_reprj_{target_filter}.fits"
            ),
        )
        out[filt] = outpath
        if not os.path.exists(outpath):
            print(f"  Reprojecting {filt}: {src} -> {outpath}")
            if filt == target_filter:
                data = fits.getdata(src, ext=('SCI', 1))
            else:
                data, _ = reproject_interp(src, tgt_header, hdu_in='SCI')
            fits.PrimaryHDU(data=data, header=tgt_header).writeto(
                outpath, overwrite=True
            )
    return out, tgt_header


def make_single_filter_pngs(repr_files, filternames, avm, png_path, obs):
    for filt in filternames:
        data_raw = fits.getdata(repr_files[filt])
        # keep NaNs in `data_raw` for save_rgb alpha; use zero-filled copy for norm/stretch
        data = np.nan_to_num(data_raw, nan=0.0)
        fn = _filter_wavelength(filt)
        for stretch, lo in (('asinh', 1.0), ('log', 1.5)):
            norm = simple_norm(data, stretch=stretch, min_percent=lo, max_percent=99.5)
            img = norm(data)
            img3 = np.stack([img, img, img], axis=2)
            orig3 = np.stack([data_raw, data_raw, data_raw], axis=2)
            save_rgb(
                img3,
                f"{png_path}/GC2211_{obs}_F{fn}_{stretch}.png",
                avm=avm,
                original_data=orig3,
            )


def make_two_filter_rgb(repr_files, filternames, avm, png_path, obs):
    """Pseudo-RGB from 2 filters: R=longest, B=shortest, G=mean of asinh-normed R+B."""
    if len(filternames) != 2:
        return
    f_long, f_short = filternames  # filternames already sorted desc by wavelength
    data_r_raw = fits.getdata(repr_files[f_long])
    data_b_raw = fits.getdata(repr_files[f_short])
    data_r = np.nan_to_num(data_r_raw, nan=0.0)
    data_b = np.nan_to_num(data_b_raw, nan=0.0)

    fn_r, fn_b = _filter_wavelength(f_long), _filter_wavelength(f_short)

    for stretch, lo in (('asinh', 1.0), ('log', 1.5)):
        norm_r = simple_norm(data_r, stretch=stretch, min_percent=lo, max_percent=99.5)
        norm_b = simple_norm(data_b, stretch=stretch, min_percent=lo, max_percent=99.5)
        r = norm_r(data_r)
        b = norm_b(data_b)
        g = 0.5 * (np.asarray(r) + np.asarray(b))
        rgb_scaled = np.stack([r, g, b], axis=2)
        orig = np.stack([data_r_raw, 0.5 * (data_r_raw + data_b_raw), data_b_raw], axis=2)
        save_rgb(
            rgb_scaled,
            f"{png_path}/GC2211_{obs}_RGB_{fn_r}-mean-{fn_b}_{stretch}.png",
            avm=avm,
            original_data=orig,
        )


def make_pngs_for_obs(obs, filenames):
    png_path = f"{BASEPATH}/pngs/{obs}"
    new_basepath = f"{BASEPATH}/data_reprojected/{obs}"
    os.makedirs(png_path, exist_ok=True)

    # Target filter: longest wavelength available (F277W is in every OBS)
    target_filter = max(filenames.keys(), key=_filter_wavelength)
    print(f"\n=== {obs}: filters={list(filenames)} target={target_filter} ===")

    repr_files, tgt_header = reproject_to_target(filenames, target_filter, new_basepath)
    avm = faithful_avm(tgt_header)

    filternames = sorted(filenames.keys(), key=_filter_wavelength, reverse=True)

    make_single_filter_pngs(repr_files, filternames, avm, png_path, obs)
    make_two_filter_rgb(repr_files, filternames, avm, png_path, obs)


def main():
    for obs, filenames in obs_images.items():
        try:
            make_pngs_for_obs(obs, filenames)
        except Exception as ex:
            print(f"FAILED {obs}: {ex}")
            raise


if __name__ == '__main__':
    main()
