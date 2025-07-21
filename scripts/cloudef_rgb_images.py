#!/usr/bin/env python
# coding: utf-8

from astropy.io import fits
import numpy as np
from astropy.visualization import simple_norm
import pylab as plt
from astropy import wcs
import os
from reproject import reproject_interp
import reproject
import PIL
import shutil
from astropy.wcs import WCS
import pyavm
from astropy.coordinates import SkyCoord
import astropy.units as u
from PIL import Image
from jwst_rgb.save_rgb import save_rgb

# Updated image filenames for Cloudef data with project code 2092
image_filenames_pipe = {
    # "f210m": "/orange/adamginsburg/jwst/cloudef/mastDownload/JWST/jw02092-o002_t001_nircam_clear-f210m/jw02092-o002_t001_nircam_clear-f210m_i2d.fits",
    # "f360m": "/orange/adamginsburg/jwst/cloudef/mastDownload/JWST/jw02092-o002_t001_nircam_clear-f360m/jw02092-o002_t001_nircam_clear-f360m_i2d.fits",
    # "f480m": "/orange/adamginsburg/jwst/cloudef/mastDownload/JWST/jw02092-o002_t001_nircam_clear-f480m/jw02092-o002_t001_nircam_clear-f480m_i2d.fits",
    "f210mo2": "/orange/adamginsburg/jwst/cloudef/MAST_FITS/jw02092-o002_t001_nircam_clear-f210m_i2d.fits",
    "f360mo2": "/orange/adamginsburg/jwst/cloudef/MAST_FITS/jw02092-o002_t001_nircam_clear-f360m_i2d.fits",
    "f480mo2": "/orange/adamginsburg/jwst/cloudef/MAST_FITS/jw02092-o002_t001_nircam_clear-f480m_i2d.fits",
    "f162mo5": "/orange/adamginsburg/jwst/cloudef/MAST_FITS/jw02092-o002_t001_nircam_f150w2-f162m_i2d.fits",
    "f2100wo4": "/orange/adamginsburg/jwst/cloudef/MAST_FITS/jw02092-o004_t001_miri_f2100w_i2d.fits",
    "f770wo4": "/orange/adamginsburg/jwst/cloudef/MAST_FITS/jw02092-o004_t001_miri_f770w_i2d.fits",
    "f210mo5": "/orange/adamginsburg/jwst/cloudef/MAST_FITS/jw02092-o005_t002_nircam_clear-f210m_i2d.fits",
    "f360mo5": "/orange/adamginsburg/jwst/cloudef/MAST_FITS/jw02092-o005_t002_nircam_clear-f360m_i2d.fits",
    "f480mo5": "/orange/adamginsburg/jwst/cloudef/MAST_FITS/jw02092-o005_t002_nircam_clear-f480m_i2d.fits",
    "f162mo5": "/orange/adamginsburg/jwst/cloudef/MAST_FITS/jw02092-o005_t002_nircam_f150w2-f162m_i2d.fits",
    "f2100wo6": "/orange/adamginsburg/jwst/cloudef/MAST_FITS/jw02092-o006_t002_miri_f2100w_i2d.fits",
    "f770wo6": "/orange/adamginsburg/jwst/cloudef/MAST_FITS/jw02092-o006_t002_miri_f770w_i2d.fits",
    "f2100wo8": "/orange/adamginsburg/jwst/cloudef/MAST_FITS/jw02092-o008_t001_miri_f2100w_i2d.fits",
    "f770wo8": "/orange/adamginsburg/jwst/cloudef/MAST_FITS/jw02092-o008_t001_miri_f770w_i2d.fits",
    "f210mo2_sci": "/orange/adamginsburg/jwst/cloudef/MAST_FITS/jw02092-o002_t001_nircam_clear-f210m_i2d_sci.fits",
    "f162mo5_sci": "/orange/adamginsburg/jwst/cloudef/MAST_FITS/jw02092-o002_t001_nircam_f150w2-f162m_i2d_sci.fits",
}

# No subtracted images available for cloudef initially
image_sub_filenames_pipe = {}

def check_overlap(image1_path, image2_path, hdu_ext='SCI'):
    """
    Check if two images have spatial overlap using their WCS coordinates.

    Parameters:
    -----------
    image1_path : str
        Path to first image
    image2_path : str
        Path to second image
    hdu_ext : str or int
        HDU extension to use for WCS

    Returns:
    --------
    bool
        True if images overlap, False otherwise
    """
    try:
        # Get headers and WCS for both images
        if hdu_ext == 'SCI':
            hdr1 = fits.getheader(image1_path, ext=(hdu_ext, 1))
            hdr2 = fits.getheader(image2_path, ext=(hdu_ext, 1))
        else:
            hdr1 = fits.getheader(image1_path, ext=hdu_ext)
            hdr2 = fits.getheader(image2_path, ext=hdu_ext)

        wcs1 = WCS(hdr1)
        wcs2 = WCS(hdr2)

        # Get image dimensions
        naxis1_1, naxis2_1 = hdr1['NAXIS1'], hdr1['NAXIS2']
        naxis1_2, naxis2_2 = hdr2['NAXIS1'], hdr2['NAXIS2']

        # Get corners of each image in pixel coordinates
        corners1_pix = np.array([[0, 0], [naxis1_1-1, 0], [naxis1_1-1, naxis2_1-1], [0, naxis2_1-1]])
        corners2_pix = np.array([[0, 0], [naxis1_2-1, 0], [naxis1_2-1, naxis2_2-1], [0, naxis2_2-1]])

        # Convert to world coordinates
        corners1_world = wcs1.pixel_to_world(corners1_pix[:, 0], corners1_pix[:, 1])
        corners2_world = wcs2.pixel_to_world(corners2_pix[:, 0], corners2_pix[:, 1])

        # Get RA/Dec bounds for each image
        ra1_min, ra1_max = corners1_world.ra.deg.min(), corners1_world.ra.deg.max()
        dec1_min, dec1_max = corners1_world.dec.deg.min(), corners1_world.dec.deg.max()

        ra2_min, ra2_max = corners2_world.ra.deg.min(), corners2_world.ra.deg.max()
        dec2_min, dec2_max = corners2_world.dec.deg.min(), corners2_world.dec.deg.max()

        # Handle RA wrap-around at 0/360 degrees
        if ra1_max - ra1_min > 180:  # Image 1 crosses 0/360
            ra1_min, ra1_max = ra1_max, ra1_min + 360
        if ra2_max - ra2_min > 180:  # Image 2 crosses 0/360
            ra2_min, ra2_max = ra2_max, ra2_min + 360

        # Check for overlap
        ra_overlap = not (ra1_max < ra2_min or ra2_max < ra1_min)
        dec_overlap = not (dec1_max < dec2_min or dec2_max < dec1_min)

        return ra_overlap and dec_overlap

    except Exception as e:
        print(f"Error checking overlap between {image1_path} and {image2_path}: {e}")
        # If we can't determine overlap, assume they overlap to be safe
        return True

def make_pngs(target_filter='f480m', new_basepath='/orange/adamginsburg/jwst/cloudef/data_reprojected/'):
    print(f"Making PNGs for {target_filter}")

    png_path = f'/orange/adamginsburg/jwst/cloudef/pngs_{target_filter[1:-1]}'
    os.makedirs(png_path, exist_ok=True)
    os.makedirs(new_basepath, exist_ok=True)

    # Check if target filter exists
    if target_filter not in image_filenames_pipe:
        print(f"Warning: Target filter {target_filter} not found in image_filenames_pipe")
        return

    tgt_header = fits.getheader(image_filenames_pipe[target_filter], ext=('SCI', 1))
    AVM = pyavm.AVM.from_header(tgt_header)

    repr_image_filenames = {x: y.replace("i2d", f"i2d_pipeline_v0.1_reprj_{target_filter[:-1]}") for x,y in image_filenames_pipe.items()}
    repr_image_filenames = {x: (new_basepath+os.path.basename(y)) for x,y in repr_image_filenames.items()}
    repr_image_sub_filenames = {x: y.replace(".fits", f"reprj_{target_filter[:-1]}.fits") for x,y in image_sub_filenames_pipe.items()}
    repr_image_sub_filenames = {x: (new_basepath+os.path.basename(y)) for x,y in repr_image_sub_filenames.items()}

    # Get reference image path for overlap checking
    reference_image = image_filenames_pipe[target_filter]

    for filtername in image_filenames_pipe:
        if not os.path.exists(repr_image_filenames[filtername]):
            # Check for overlap before reprojecting
            if filtername != target_filter:  # Don't check overlap with itself
                has_overlap = check_overlap(reference_image, image_filenames_pipe[filtername])
                if not has_overlap:
                    print(f"Skipping {filtername} - no spatial overlap with reference {target_filter}")
                    continue

            print(f"Reprojecting {filtername} {image_filenames_pipe[filtername]} to {repr_image_filenames[filtername]}")
            try:
                result,_ = reproject.reproject_interp(image_filenames_pipe[filtername], tgt_header, hdu_in='SCI')
                hdu = fits.PrimaryHDU(data=result, header=tgt_header)
                hdu.writeto(repr_image_filenames[filtername], overwrite=True)
            except Exception as e:
                print(f"Error reprojecting {filtername}: {e}")
                continue

    for filtername in image_sub_filenames_pipe:
        if not os.path.exists(repr_image_sub_filenames[filtername]):
            # Check for overlap before reprojecting
            has_overlap = check_overlap(reference_image, image_sub_filenames_pipe[filtername])
            if not has_overlap:
                print(f"Skipping {filtername} - no spatial overlap with reference {target_filter}")
                continue

            print(f"Reprojecting {filtername} {image_sub_filenames_pipe[filtername]} to {repr_image_sub_filenames[filtername]}")
            try:
                result,_ = reproject.reproject_interp(image_sub_filenames_pipe[filtername], tgt_header, hdu_in='SCI')
                hdu = fits.PrimaryHDU(data=result, header=tgt_header)
                hdu.writeto(repr_image_sub_filenames[filtername], overwrite=True)
            except Exception as e:
                print(f"Error reprojecting {filtername}: {e}")
                continue

    # ALMA data not available for cloudef - skipping ALMA overlay functionality
    alma_cloudef_reprojected_jwst = None
    alma_level = None

    # Only use filters that were successfully reprojected
    available_filters = [f for f in image_filenames_pipe.keys() if os.path.exists(repr_image_filenames[f])]
    filternames = sorted(available_filters, key=lambda x: int(''.join(filter(str.isdigit, x))))[::-1]
    print(f"Sorted list of available filters: {filternames}")

    # Create RGB combinations with available filters
    if len(filternames) >= 3:
        for i in range(len(filternames) - 2):
            f1, f2, f3 = filternames[i], filternames[i+1], filternames[i+2]
            print(f1,f2,f3)
            try:
                rgb = np.array([
                    fits.getdata(repr_image_filenames[f1]),
                    fits.getdata(repr_image_filenames[f2]),
                    fits.getdata(repr_image_filenames[f3]),
                ]).swapaxes(0,2).swapaxes(0,1)
                rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,0]),
                                    simple_norm(rgb[:,:,1], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,1]),
                                    simple_norm(rgb[:,:,2], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)

                f1n, f2n, f3n = ''.join(filter(str.isdigit, f1)), ''.join(filter(str.isdigit, f2)), ''.join(filter(str.isdigit, f3))
                save_rgb(rgb_scaled, f'{png_path}/Cloudef_RGB_{f1n}-{f2n}-{f3n}.png', avm=AVM, original_data=rgb)

                rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='log', min_percent=1.5, max_percent=99.5)(rgb[:,:,0]),
                                    simple_norm(rgb[:,:,1], stretch='log', min_percent=1.5, max_percent=99.5)(rgb[:,:,1]),
                                    simple_norm(rgb[:,:,2], stretch='log', min_percent=1.5, max_percent=99.5)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)

                save_rgb(rgb_scaled, f'{png_path}/Cloudef_RGB_{f1n}-{f2n}-{f3n}_log.png', avm=AVM, original_data=rgb)
            except Exception as e:
                print(f"Error creating RGB image for {f1}-{f2}-{f3}: {e}")
                continue

    # Only use subtracted filters that were successfully reprojected
    available_sub_filters = [f for f in image_sub_filenames_pipe.keys() if os.path.exists(repr_image_sub_filenames[f])]
    filternames_sub = sorted(available_sub_filters, key=lambda x: int(''.join(filter(str.isdigit, x))))[::-1]
    print(f"Sorted list of available subtracted-filters: {filternames_sub}")

    if len(filternames_sub) >= 3:
        for i in range(len(filternames_sub) - 2):
            f1, f2, f3 = filternames_sub[i], filternames_sub[i+1], filternames_sub[i+2]
            print(f1,f2,f3)
            try:
                rgb = np.array([
                    fits.getdata(repr_image_sub_filenames[f1]),
                    fits.getdata(repr_image_sub_filenames[f2]),
                    fits.getdata(repr_image_sub_filenames[f3]),
                ]).swapaxes(0,2).swapaxes(0,1)
            except Exception as ex:
                print(ex)
                print(f"Shape of {f1} is {fits.getdata(repr_image_sub_filenames[f1]).shape}")
                print(f"Shape of {f2} is {fits.getdata(repr_image_sub_filenames[f2]).shape}")
                print(f"Shape of {f3} is {fits.getdata(repr_image_sub_filenames[f3]).shape}")
                continue

            try:
                rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,0]),
                                    simple_norm(rgb[:,:,1], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,1]),
                                    simple_norm(rgb[:,:,2], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)

                f1n, f2n, f3n = ''.join(filter(str.isdigit, f1)), ''.join(filter(str.isdigit, f2)), ''.join(filter(str.isdigit, f3))
                save_rgb(rgb_scaled, f'{png_path}/Cloudef_RGB_{f1n}-{f2n}-{f3n}_sub.png', avm=AVM, original_data=rgb)

                rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='log', min_percent=1.0, max_percent=99.5)(rgb[:,:,0]),
                                    simple_norm(rgb[:,:,1], stretch='log', min_percent=1.0, max_percent=99.5)(rgb[:,:,1]),
                                    simple_norm(rgb[:,:,2], stretch='log', min_percent=1.0, max_percent=99.5)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)

                save_rgb(rgb_scaled, f'{png_path}/Cloudef_RGB_{f1n}-{f2n}-{f3n}_sub_log.png', avm=AVM, original_data=rgb)
            except Exception as e:
                print(f"Error creating subtracted RGB image for {f1}-{f2}-{f3}: {e}")
                continue

    # Create single filter images
    print("Creating individual filter images:")
    for filtername in filternames:
        print(f"Creating image for {filtername}")
        try:
            data = fits.getdata(repr_image_filenames[filtername])

            # Asinh stretch
            norm_asinh = simple_norm(data, stretch='asinh', min_percent=1, max_percent=99.5)
            img_asinh = norm_asinh(data)
            img_asinh = np.stack([img_asinh, img_asinh, img_asinh], axis=2)

            fn = ''.join(filter(str.isdigit, filtername))
            save_rgb(img_asinh, f'{png_path}/Cloudef_{fn}_asinh.png', avm=AVM, original_data=np.stack([data, data, data], axis=2))

            # Log stretch
            norm_log = simple_norm(data, stretch='log', min_percent=1.5, max_percent=99.5)
            img_log = norm_log(data)
            img_log = np.stack([img_log, img_log, img_log], axis=2)

            save_rgb(img_log, f'{png_path}/Cloudef_{fn}_log.png', avm=AVM, original_data=np.stack([data, data, data], axis=2))
        except Exception as e:
            print(f"Error creating individual filter image for {filtername}: {e}")
            continue

def main():
    # Loop over all available filters as target filters
    available_targets = list(image_filenames_pipe.keys())
    print(f"Available target filters: {available_targets}")

    for target_filter in available_targets:
        print(f"\n=== Processing with {target_filter} as reference ===")
        try:
            make_pngs(target_filter)
        except Exception as e:
            print(f"Error processing {target_filter}: {e}")
            continue

if __name__ == '__main__':
    main()