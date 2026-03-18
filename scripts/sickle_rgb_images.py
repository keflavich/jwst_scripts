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
from PIL import Image
from jwst_rgb.save_rgb import save_rgb
from reproject.mosaicking import find_optimal_celestial_wcs, reproject_and_coadd

# Updated image filenames for Sickle data with project code 3958 - MIRI and NIRCAM observations
image_filenames_pipe = {
    # MIRI filters
    "f770w": "/orange/adamginsburg/jwst/sickle/mastDownload/JWST/jw03958-o001_t001_miri_f770w-brightsky/jw03958-o001_t001_miri_f770w-brightsky_i2d.fits",
    "f1130w": "/orange/adamginsburg/jwst/sickle/mastDownload/JWST/jw03958-o001_t001_miri_f1130w-brightsky/jw03958-o001_t001_miri_f1130w-brightsky_i2d.fits",
    "f1500w": "/orange/adamginsburg/jwst/sickle/mastDownload/JWST/jw03958-o001_t001_miri_f1500w-brightsky/jw03958-o001_t001_miri_f1500w-brightsky_i2d.fits",
    "f770wb": "/orange/adamginsburg/jwst/sickle/mastDownload/JWST/jw03958-o002_t002_miri_f770w-brightsky/jw03958-o002_t002_miri_f770w-brightsky_i2d.fits",
    "f1130wb": "/orange/adamginsburg/jwst/sickle/mastDownload/JWST/jw03958-o002_t002_miri_f1130w-brightsky/jw03958-o002_t002_miri_f1130w-brightsky_i2d.fits",
    "f1500wb": "/orange/adamginsburg/jwst/sickle/mastDownload/JWST/jw03958-o002_t002_miri_f1500w-brightsky/jw03958-o002_t002_miri_f1500w-brightsky_i2d.fits",
    "f770wc": "/orange/adamginsburg/jwst/sickle/mastDownload/JWST/jw03958-o003_t003_miri_f770w-brightsky/jw03958-o003_t003_miri_f770w-brightsky_i2d.fits",
    "f1130wc": "/orange/adamginsburg/jwst/sickle/mastDownload/JWST/jw03958-o003_t003_miri_f1130w-brightsky/jw03958-o003_t003_miri_f1130w-brightsky_i2d.fits",
    "f1500wc": "/orange/adamginsburg/jwst/sickle/mastDownload/JWST/jw03958-o003_t003_miri_f1500w-brightsky/jw03958-o003_t003_miri_f1500w-brightsky_i2d.fits",

    # NIRCAM filters
    # I##i#"f182m": "/orange/adamginsburg/jwst/sickle/mastDownload/JWST/jw03958-o007_t005_nircam_clear-f182m-sub640/jw03958-o007_t005_nircam_clear-f182m-sub640_i2d.fits",
    "f187n": "/orange/adamginsburg/jwst/sickle/mastDownload/JWST/jw03958-o007_t005_nircam_clear-f187n-sub640/jw03958-o007_t005_nircam_clear-f187n-sub640_i2d.fits",
    "f210m": "/orange/adamginsburg/jwst/sickle/mastDownload/JWST/jw03958-o007_t005_nircam_clear-f210m-sub640/jw03958-o007_t005_nircam_clear-f210m-sub640_i2d.fits",
    "f335m": "/orange/adamginsburg/jwst/sickle/mastDownload/JWST/jw03958-o007_t005_nircam_clear-f335m-sub640/jw03958-o007_t005_nircam_clear-f335m-sub640_i2d.fits",
    "f444w": "/orange/adamginsburg/jwst/sickle/mastDownload/JWST/jw03958-o007_t005_nircam_f444w-f470n-sub640/jw03958-o007_t005_nircam_f444w-f470n-sub640_i2d.fits",
    "f470n": "/orange/adamginsburg/jwst/sickle/mastDownload/JWST/jw03958-o007_t005_nircam_f444w-f470n-sub640/jw03958-o007_t005_nircam_f444w-f470n-sub640_i2d.fits",  # Note: same file as f444w
    "f480m": "/orange/adamginsburg/jwst/sickle/mastDownload/JWST/jw03958-o007_t005_nircam_clear-f480m-sub640/jw03958-o007_t005_nircam_clear-f480m-sub640_i2d.fits",
}

# No subtracted images available for sickle initially
image_sub_filenames_pipe = {}

def group_images_by_filter(image_dict):
    """
    Group images by their base filter name (e.g., f770w, f1130w, f1500w)
    ignoring suffixes like 'b', 'c', etc.

    Returns:
        dict: {filter_name: [list_of_filenames]}
    """
    filter_groups = {}

    for key, filename in image_dict.items():
        # Extract base filter name by removing trailing letters
        base_filter = key.rstrip('abcdefghijklmnopqrstuvwxyz')

        if base_filter not in filter_groups:
            filter_groups[base_filter] = []
        filter_groups[base_filter].append((key, filename))

    return filter_groups

def create_mosaic(image_files, output_path):
    """
    Create a mosaic from multiple FITS files of the same filter.

    Parameters:
        image_files: list of tuples (key, filename)
        output_path: path to save the mosaicked image

    Returns:
        str: path to the mosaicked image
    """
    if len(image_files) == 1:
        # If only one image, just copy it
        key, filename = image_files[0]
        shutil.copy2(filename, output_path)
        return output_path

    # Get all the input files for mosaicking
    input_files = [filename for key, filename in image_files]

    # Find optimal WCS for the mosaic
    wcs_out, shape_out = find_optimal_celestial_wcs(
        input_files,
        hdu_in='SCI'
    )

    # Create the mosaic
    array, footprint = reproject_and_coadd(
        input_files,
        wcs_out,
        shape_out=shape_out,
        hdu_in='SCI',
        reproject_function=reproject_interp
    )

    # Create HDU with the mosaicked data
    header = wcs_out.to_header()
    hdu = fits.PrimaryHDU(data=array, header=header)
    hdu.writeto(output_path, overwrite=True)

    return output_path

def make_pngs(target_filter='f1130w', new_basepath='/orange/adamginsburg/jwst/sickle/data_reprojected/'):
    print(f"Making PNGs for {target_filter}")

    png_path = f'/orange/adamginsburg/jwst/sickle/pngs_{target_filter[1:-1]}'
    os.makedirs(png_path, exist_ok=True)
    os.makedirs(new_basepath, exist_ok=True)

    # First, create mosaics for each filter
    print("Creating mosaics for each filter...")
    filter_groups = group_images_by_filter(image_filenames_pipe)

    # Create mosaicked images
    mosaicked_filenames = {}
    for base_filter, image_files in filter_groups.items():
        if len(image_files) > 1:
            print(f"Creating mosaic for {base_filter} from {len(image_files)} images")
            mosaic_path = os.path.join(new_basepath, f"{base_filter}_mosaic.fits")
            create_mosaic(image_files, mosaic_path)
            mosaicked_filenames[base_filter] = mosaic_path
        else:
            # Single image, just use it directly
            key, filename = image_files[0]
            mosaicked_filenames[base_filter] = filename
            print(f"Using single image for {base_filter}: {key}")

    # Use the first available mosaicked filter for target header
    target_base_filter = target_filter.rstrip('abcdefghijklmnopqrstuvwxyz')
    if target_base_filter in mosaicked_filenames:
        target_file = mosaicked_filenames[target_base_filter]
    else:
        # Fallback to original logic
        target_file = image_filenames_pipe[target_filter]

    # Determine the correct HDU extension based on file type
    if target_file.endswith('_mosaic.fits'):
        # Mosaic files use PrimaryHDU (extension 0)
        tgt_header = fits.getheader(target_file, ext=0)
    else:
        # Original pipeline files use SCI extension
        tgt_header = fits.getheader(target_file, ext=('SCI', 1))
    AVM = pyavm.AVM.from_header(tgt_header)

    # Create reprojected filenames based on mosaicked images
    repr_image_filenames = {}
    for base_filter in mosaicked_filenames:
        repr_filename = os.path.join(new_basepath, f"{base_filter}_reprj_{target_filter[:-1]}.fits")
        repr_image_filenames[base_filter] = repr_filename

    repr_image_sub_filenames = {x: y.replace(".fits", f"reprj_{target_filter[:-1]}.fits") for x,y in image_sub_filenames_pipe.items()}
    repr_image_sub_filenames = {x: (new_basepath+os.path.basename(y)) for x,y in repr_image_sub_filenames.items()}

    # Reproject mosaicked images to target coordinate system
    for base_filter, mosaic_file in mosaicked_filenames.items():
        if not os.path.exists(repr_image_filenames[base_filter]):
            print(f"Reprojecting {base_filter} mosaic {mosaic_file} to {repr_image_filenames[base_filter]}")
            # Determine the HDU extension based on file type
            hdu_ext = 0 if mosaic_file.endswith('_mosaic.fits') else ('SCI', 1)
            result,_ = reproject.reproject_interp(mosaic_file, tgt_header, hdu_in=hdu_ext)
            hdu = fits.PrimaryHDU(data=result, header=tgt_header)
            hdu.writeto(repr_image_filenames[base_filter], overwrite=True)

    for filtername in image_sub_filenames_pipe:
        if not os.path.exists(repr_image_sub_filenames[filtername]):
            print(f"Reprojecting {filtername} {image_sub_filenames_pipe[filtername]} to {repr_image_sub_filenames[filtername]}")
            result,_ = reproject.reproject_interp(image_sub_filenames_pipe[filtername], tgt_header, hdu_in='SCI')
            hdu = fits.PrimaryHDU(data=result, header=tgt_header)
            hdu.writeto(repr_image_sub_filenames[filtername], overwrite=True)

    # ALMA data not available for sickle - skipping ALMA overlay functionality
    alma_sickle_reprojected_jwst = None
    alma_level = None

    # Use mosaicked filter names instead of original individual image names
    filternames = sorted(list(mosaicked_filenames.keys()),
                        key=lambda x: int(''.join(filter(str.isdigit, x))))[::-1]
    print(f"Sorted list of mosaicked filters: {filternames}")

    # Create RGB combinations with available filters (we have exactly 3 MIRI filters)
    if len(filternames) >= 3:
        for i in range(len(filternames) - 2):
            f1, f2, f3 = filternames[i], filternames[i+1], filternames[i+2]
            print(f"Creating RGB with mosaicked filters: {f1}, {f2}, {f3}")
            rgb = np.array([
                fits.getdata(repr_image_filenames[f1]),
                fits.getdata(repr_image_filenames[f2]),
                fits.getdata(repr_image_filenames[f3]),
            ]).swapaxes(0,2).swapaxes(0,1)

            # Apply asinh stretch
            rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,0]),
                                simple_norm(rgb[:,:,1], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,1]),
                                simple_norm(rgb[:,:,2], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)

            f1n, f2n, f3n = ''.join(filter(str.isdigit, f1)), ''.join(filter(str.isdigit, f2)), ''.join(filter(str.isdigit, f3))
            save_rgb(rgb_scaled, f'{png_path}/Sickle_RGB_{f1n}-{f2n}-{f3n}.png', avm=AVM, original_data=rgb, transpose=None)

            # Apply log stretch
            rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='log', min_percent=1.5, max_percent=99.5)(rgb[:,:,0]),
                                simple_norm(rgb[:,:,1], stretch='log', min_percent=1.5, max_percent=99.5)(rgb[:,:,1]),
                                simple_norm(rgb[:,:,2], stretch='log', min_percent=1.5, max_percent=99.5)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)

            save_rgb(rgb_scaled, f'{png_path}/Sickle_RGB_{f1n}-{f2n}-{f3n}_log.png', avm=AVM, original_data=rgb, transpose=None)

    # Special RGB combination with optimal wavelength mapping for MIRI
    # f1500w (longest) -> Red, f1130w (middle) -> Green, f770w (shortest) -> Blue
    if 'f1500w' in filternames and 'f1130w' in filternames and 'f770w' in filternames:
        f_red, f_green, f_blue = 'f1500w', 'f1130w', 'f770w'
        print(f"Creating optimal MIRI RGB: R={f_red}, G={f_green}, B={f_blue}")

        rgb = np.array([
            fits.getdata(repr_image_filenames[f_red]),
            fits.getdata(repr_image_filenames[f_green]),
            fits.getdata(repr_image_filenames[f_blue]),
        ]).swapaxes(0,2).swapaxes(0,1)

        # Apply asinh stretch
        rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,0]),
                            simple_norm(rgb[:,:,1], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,1]),
                            simple_norm(rgb[:,:,2], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)

        save_rgb(rgb_scaled, f'{png_path}/Sickle_RGB_MIRI_optimal.png', avm=AVM, original_data=rgb, transpose=None)

        # Apply log stretch
        rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='log', min_percent=1.5, max_percent=99.5)(rgb[:,:,0]),
                            simple_norm(rgb[:,:,1], stretch='log', min_percent=1.5, max_percent=99.5)(rgb[:,:,1]),
                            simple_norm(rgb[:,:,2], stretch='log', min_percent=1.5, max_percent=99.5)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)

        save_rgb(rgb_scaled, f'{png_path}/Sickle_RGB_MIRI_optimal_log.png', avm=AVM, original_data=rgb, transpose=None)

    filternames_sub = sorted(list(image_sub_filenames_pipe.keys()),
                           key=lambda x: int(''.join(filter(str.isdigit, x))))[::-1]
    print(f"Sorted list of subtracted-filters: {filternames_sub}")

    if len(filternames_sub) >= 3:
        for i in range(len(filternames_sub) - 2):
            f1, f2, f3 = filternames_sub[i], filternames_sub[i+1], filternames_sub[i+2]
            print(f"Creating RGB with subtracted filters: {f1}, {f2}, {f3}")
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
                raise ex

            rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,0]),
                                simple_norm(rgb[:,:,1], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,1]),
                                simple_norm(rgb[:,:,2], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)

            f1n, f2n, f3n = ''.join(filter(str.isdigit, f1)), ''.join(filter(str.isdigit, f2)), ''.join(filter(str.isdigit, f3))
            save_rgb(rgb_scaled, f'{png_path}/Sickle_RGB_{f1n}-{f2n}-{f3n}_sub.png', avm=AVM, original_data=rgb, transpose=None)

            rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='log', min_percent=1.0, max_percent=99.5)(rgb[:,:,0]),
                                simple_norm(rgb[:,:,1], stretch='log', min_percent=1.0, max_percent=99.5)(rgb[:,:,1]),
                                simple_norm(rgb[:,:,2], stretch='log', min_percent=1.0, max_percent=99.5)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)

            save_rgb(rgb_scaled, f'{png_path}/Sickle_RGB_{f1n}-{f2n}-{f3n}_sub_log.png', avm=AVM, original_data=rgb, transpose=None)

    # Create single filter images from mosaicked data
    print("Creating individual mosaicked filter images:")
    for filtername in filternames:
        print(f"Creating image for mosaicked {filtername}")
        data = fits.getdata(repr_image_filenames[filtername])

        # Asinh stretch
        norm_asinh = simple_norm(data, stretch='asinh', min_percent=1, max_percent=99.5)
        img_asinh = norm_asinh(data)
        img_asinh = np.stack([img_asinh, img_asinh, img_asinh], axis=2)

        fn = ''.join(filter(str.isdigit, filtername))
        # Create original data stack for transparency detection
        original_data_stack = np.stack([data, data, data], axis=2)
        save_rgb(img_asinh, f'{png_path}/Sickle_{fn}_mosaic_asinh.png', avm=AVM, original_data=original_data_stack, transpose=None)

        # Log stretch
        norm_log = simple_norm(data, stretch='log', min_percent=1.5, max_percent=99.5)
        img_log = norm_log(data)
        img_log = np.stack([img_log, img_log, img_log], axis=2)

        save_rgb(img_log, f'{png_path}/Sickle_{fn}_mosaic_log.png', avm=AVM, original_data=original_data_stack, transpose=None)

def main():
    # Use f1130w as default target filter for sickle (middle wavelength)
    for target_filter in ('f1130w', 'f770w', 'f1500w'):
        make_pngs(target_filter)

if __name__ == '__main__':
    main()