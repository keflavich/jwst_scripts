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

def save_rgb(img, filename, avm=None, flip=-1, alma_data=None, alma_level=None, original_data=None):
    img = (img*256)
    img[img<0] = 0
    img[img>255] = 255
    img = img.astype('uint8')

    # Create alpha channel for transparency
    alpha = np.ones(img.shape[:2], dtype=np.uint8) * 255  # Start with fully opaque

    if original_data is not None:
        # Make pixels transparent where original data is NaN or very small
        # Check each channel for blank pixels
        for i in range(3):
            if i < original_data.shape[2]:
                blank_mask = (np.isnan(original_data[:,:,i]) |
                             (np.abs(original_data[:,:,i]) < 1e-10))
                alpha[blank_mask] = 0

    # Apply flip to alpha channel to match image
    alpha = alpha[::flip,:]

    if alma_data is not None and alma_level is not None:
        contour_mask = np.zeros_like(alma_data, dtype=bool)
        contour_mask[alma_data >= alma_level] = True
        from scipy.ndimage import binary_dilation
        contour_mask1 = binary_dilation(contour_mask)
        contour_mask = contour_mask1 ^ contour_mask

        # Apply flip to contour mask to match image
        contour_mask = contour_mask[::flip,:]

        for i in range(3):
            img[contour_mask, i] = 255 - img[contour_mask, i]

    # Create RGBA image for PNG with transparency
    img_rgba = np.dstack((img[::flip,:,:], alpha))
    img_pil = PIL.Image.fromarray(img_rgba, mode='RGBA')
    # empirical: 180 degree rotation required.
    flip_img = img_pil.transpose(Image.ROTATE_180)
    flip_img.save(filename)

    if avm is not None:
        base = os.path.basename(filename)
        dir = os.path.dirname(filename)
        avmname = os.path.join(dir, 'avm_'+base)
        avm.embed(filename, avmname)
        shutil.move(avmname, filename)

    # Save as JPEG without transparency (JPEG doesn't support alpha channel)
    filename_jpg = filename.replace('.png', '.jpg')
    img_rgb = PIL.Image.fromarray(img[::flip,:,:], mode='RGB')
    img_rgb = img_rgb.transpose(Image.ROTATE_180)
    img_rgb.save(filename_jpg, format='JPEG',
                 quality=95,
                 progressive=True)

    return img_pil

# Combined image filenames for Sgr A* data with multiple project codes
# Project 1939: NIRCam observations
# Project 3571: MIRI observations
# Project 5368: Additional NIRCam/NIRISS (if available)
image_filenames_pipe = {
    # NIRCam filters from project 1939
    "f115w": "/orange/adamginsburg/jwst/sgra/mastDownload/JWST/jw01939-o001_t001_nircam_clear-f115w/jw01939-o001_t001_nircam_clear-f115w_i2d.fits",
    "f212n": "/orange/adamginsburg/jwst/sgra/mastDownload/JWST/jw01939-o001_t001_nircam_clear-f212n/jw01939-o001_t001_nircam_clear-f212n_i2d.fits",
    "f323n": "/orange/adamginsburg/jwst/sgra/mastDownload/JWST/jw01939-o001_t001_nircam_f322w2-f323n/jw01939-o001_t001_nircam_f322w2-f323n_i2d.fits",
    "f444w": "/orange/adamginsburg/jwst/sgra/mastDownload/JWST/jw01939-o001_t001_nircam_f405n-f444w/jw01939-o001_t001_nircam_f405n-f444w_i2d.fits",

    # MIRI filters from project 3571
    "f560w": "/orange/adamginsburg/jwst/sgra/mastDownload/JWST/jw03571-o001_t001_miri_f560w/jw03571-o001_t001_miri_f560w_i2d.fits",
    "f770w": "/orange/adamginsburg/jwst/sgra/mastDownload/JWST/jw03571-o001_t001_miri_f770w/jw03571-o001_t001_miri_f770w_i2d.fits",
    "f1000w": "/orange/adamginsburg/jwst/sgra/mastDownload/JWST/jw03571-o001_t001_miri_f1000w/jw03571-o001_t001_miri_f1000w_i2d.fits",
    "f1280w": "/orange/adamginsburg/jwst/sgra/mastDownload/JWST/jw03571-o001_t001_miri_f1280w/jw03571-o001_t001_miri_f1280w_i2d.fits",
    "f1500w": "/orange/adamginsburg/jwst/sgra/mastDownload/JWST/jw03571-o001_t001_miri_f1500w/jw03571-o001_t001_miri_f1500w_i2d.fits",
}

# No subtracted images available for sgra initially
image_sub_filenames_pipe = {}

def make_pngs(target_filter='f770w', new_basepath='/orange/adamginsburg/jwst/sgra/data_reprojected/'):
    print(f"Making PNGs for {target_filter}")

    png_path = f'/orange/adamginsburg/jwst/sgra/pngs_{target_filter[1:-1]}'
    os.makedirs(png_path, exist_ok=True)
    os.makedirs(new_basepath, exist_ok=True)

    # Check if the target filter file exists
    if not os.path.exists(image_filenames_pipe[target_filter]):
        print(f"Warning: Target filter {target_filter} file does not exist: {image_filenames_pipe[target_filter]}")
        return

    # Create a combined header that encompasses all available images
    existing_filters = {k: v for k, v in image_filenames_pipe.items() if os.path.exists(v)}
    print(f"Creating combined header from {len(existing_filters)} images")

    # Get all headers and WCS objects
    headers = []
    wcs_objects = []
    for filtername, filename in existing_filters.items():
        header = fits.getheader(filename, ext=('SCI', 1))
        headers.append(header)
        wcs_objects.append(WCS(header))

            # Find the bounding box that encompasses all images
    from reproject.mosaicking import find_optimal_celestial_wcs
    from astropy import units as u

    # Get the finest resolution among all images
    resolutions = []
    for h in headers:
        if 'CDELT1' in h:
            resolutions.append(abs(h['CDELT1']))
        elif 'CD1_1' in h:
            resolutions.append(abs(h['CD1_1']))

    if resolutions:
        min_resolution = min(resolutions) * u.deg
    else:
        min_resolution = None

    combined_wcs, combined_shape = find_optimal_celestial_wcs(
        [((h['NAXIS2'], h['NAXIS1']), WCS(h)) for h in headers],
        resolution=min_resolution
    )

    # Create the target header from the combined WCS
    tgt_header = combined_wcs.to_header()
    tgt_header['NAXIS1'] = combined_shape[1]
    tgt_header['NAXIS2'] = combined_shape[0]
    tgt_header['NAXIS'] = 2

    # Copy over some useful keywords from the reference filter
    ref_header = fits.getheader(image_filenames_pipe[target_filter], ext=('SCI', 1))
    for keyword in ['TELESCOP', 'INSTRUME', 'FILTER', 'PHOTFNU', 'PHOTFLAM', 'BUNIT']:
        if keyword in ref_header:
            tgt_header[keyword] = ref_header[keyword]

    print(f"Combined header shape: {combined_shape}")
    AVM = pyavm.AVM.from_header(tgt_header)

    repr_image_filenames = {x: y.replace("i2d", f"i2d_pipeline_v0.1_reprj_{target_filter[:-1]}") for x,y in image_filenames_pipe.items()}
    repr_image_filenames = {x: (new_basepath+os.path.basename(y)) for x,y in repr_image_filenames.items()}
    repr_image_sub_filenames = {x: y.replace(".fits", f"reprj_{target_filter[:-1]}.fits") for x,y in image_sub_filenames_pipe.items()}
    repr_image_sub_filenames = {x: (new_basepath+os.path.basename(y)) for x,y in repr_image_sub_filenames.items()}

    print(f"Available filters: {list(existing_filters.keys())}")

    for filtername in existing_filters:
        if not os.path.exists(repr_image_filenames[filtername]):
            print(f"Reprojecting {filtername} {image_filenames_pipe[filtername]} to {repr_image_filenames[filtername]}")
            result,_ = reproject.reproject_interp(image_filenames_pipe[filtername], tgt_header, hdu_in='SCI')
            hdu = fits.PrimaryHDU(data=result, header=tgt_header)
            hdu.writeto(repr_image_filenames[filtername], overwrite=True)

    for filtername in image_sub_filenames_pipe:
        if not os.path.exists(repr_image_sub_filenames[filtername]):
            print(f"Reprojecting {filtername} {image_sub_filenames_pipe[filtername]} to {repr_image_sub_filenames[filtername]}")
            result,_ = reproject.reproject_interp(image_sub_filenames_pipe[filtername], tgt_header, hdu_in='SCI')
            hdu = fits.PrimaryHDU(data=result, header=tgt_header)
            hdu.writeto(repr_image_sub_filenames[filtername], overwrite=True)

    # ALMA data not available for sgra - skipping ALMA overlay functionality
    alma_sgra_reprojected_jwst = None
    alma_level = None

    # Separate NIRCam and MIRI filters
    nircam_filters = [f for f in existing_filters.keys() if f in ['f115w', 'f212n', 'f323n', 'f444w']]
    miri_filters = [f for f in existing_filters.keys() if f in ['f560w', 'f770w', 'f1000w', 'f1280w', 'f1500w']]

    print(f"NIRCam filters available: {nircam_filters}")
    print(f"MIRI filters available: {miri_filters}")

    # Create optimal RGB combinations

    # 1. MIRI-only RGB (5 filters available: f560w, f770w, f1000w, f1280w, f1500w)
    if len(miri_filters) >= 3:
        miri_sorted = sorted(miri_filters, key=lambda x: int(''.join(filter(str.isdigit, x))))[::-1]
        print(f"MIRI filters sorted by wavelength (descending): {miri_sorted}")

        # Best MIRI combination: longest->Red, middle->Green, shortest->Blue
        if len(miri_sorted) >= 3:
            f_red, f_green, f_blue = miri_sorted[0], miri_sorted[len(miri_sorted)//2], miri_sorted[-1]
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

            f_red_n, f_green_n, f_blue_n = ''.join(filter(str.isdigit, f_red)), ''.join(filter(str.isdigit, f_green)), ''.join(filter(str.isdigit, f_blue))
            save_rgb(rgb_scaled, f'{png_path}/SgrA_RGB_MIRI_{f_red_n}-{f_green_n}-{f_blue_n}.png', avm=AVM, original_data=rgb)

            # Apply log stretch
            rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='log', min_percent=1.5, max_percent=99.5)(rgb[:,:,0]),
                                simple_norm(rgb[:,:,1], stretch='log', min_percent=1.5, max_percent=99.5)(rgb[:,:,1]),
                                simple_norm(rgb[:,:,2], stretch='log', min_percent=1.5, max_percent=99.5)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)

            save_rgb(rgb_scaled, f'{png_path}/SgrA_RGB_MIRI_{f_red_n}-{f_green_n}-{f_blue_n}_log.png', avm=AVM, original_data=rgb)

    # 2. NIRCam-only RGB (if we have enough filters)
    if len(nircam_filters) >= 3:
        nircam_sorted = sorted(nircam_filters, key=lambda x: int(''.join(filter(str.isdigit, x))))[::-1]
        print(f"NIRCam filters sorted by wavelength (descending): {nircam_sorted}")

        f_red, f_green, f_blue = nircam_sorted[0], nircam_sorted[1], nircam_sorted[2]
        print(f"Creating NIRCam RGB: R={f_red}, G={f_green}, B={f_blue}")

        rgb = np.array([
            fits.getdata(repr_image_filenames[f_red]),
            fits.getdata(repr_image_filenames[f_green]),
            fits.getdata(repr_image_filenames[f_blue]),
        ]).swapaxes(0,2).swapaxes(0,1)

        # Apply asinh stretch
        rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,0]),
                            simple_norm(rgb[:,:,1], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,1]),
                            simple_norm(rgb[:,:,2], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)

        f_red_n, f_green_n, f_blue_n = ''.join(filter(str.isdigit, f_red)), ''.join(filter(str.isdigit, f_green)), ''.join(filter(str.isdigit, f_blue))
        save_rgb(rgb_scaled, f'{png_path}/SgrA_RGB_NIRCam_{f_red_n}-{f_green_n}-{f_blue_n}.png', avm=AVM, original_data=rgb)

        # Apply log stretch
        rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='log', min_percent=1.5, max_percent=99.5)(rgb[:,:,0]),
                            simple_norm(rgb[:,:,1], stretch='log', min_percent=1.5, max_percent=99.5)(rgb[:,:,1]),
                            simple_norm(rgb[:,:,2], stretch='log', min_percent=1.5, max_percent=99.5)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)

        save_rgb(rgb_scaled, f'{png_path}/SgrA_RGB_NIRCam_{f_red_n}-{f_green_n}-{f_blue_n}_log.png', avm=AVM, original_data=rgb)

    # 3. Multi-instrument RGB combinations (most interesting!)
    if len(nircam_filters) >= 1 and len(miri_filters) >= 2:
        # Use longest MIRI for red, mix for green/blue
        miri_sorted = sorted(miri_filters, key=lambda x: int(''.join(filter(str.isdigit, x))))[::-1]
        nircam_sorted = sorted(nircam_filters, key=lambda x: int(''.join(filter(str.isdigit, x))))[::-1]

        f_red = miri_sorted[0]  # Longest MIRI wavelength
        f_green = miri_sorted[1] if len(miri_sorted) > 1 else miri_sorted[0]  # Second longest MIRI
        f_blue = nircam_sorted[0]  # Longest NIRCam (shortest overall)

        print(f"Creating multi-instrument RGB: R={f_red} (MIRI), G={f_green} (MIRI), B={f_blue} (NIRCam)")

        rgb = np.array([
            fits.getdata(repr_image_filenames[f_red]),
            fits.getdata(repr_image_filenames[f_green]),
            fits.getdata(repr_image_filenames[f_blue]),
        ]).swapaxes(0,2).swapaxes(0,1)

        # Apply asinh stretch
        rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,0]),
                            simple_norm(rgb[:,:,1], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,1]),
                            simple_norm(rgb[:,:,2], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)

        f_red_n, f_green_n, f_blue_n = ''.join(filter(str.isdigit, f_red)), ''.join(filter(str.isdigit, f_green)), ''.join(filter(str.isdigit, f_blue))
        save_rgb(rgb_scaled, f'{png_path}/SgrA_RGB_MultiInstrument_{f_red_n}-{f_green_n}-{f_blue_n}.png', avm=AVM, original_data=rgb)

        # Apply log stretch
        rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='log', min_percent=1.5, max_percent=99.5)(rgb[:,:,0]),
                            simple_norm(rgb[:,:,1], stretch='log', min_percent=1.5, max_percent=99.5)(rgb[:,:,1]),
                            simple_norm(rgb[:,:,2], stretch='log', min_percent=1.5, max_percent=99.5)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)

        save_rgb(rgb_scaled, f'{png_path}/SgrA_RGB_MultiInstrument_{f_red_n}-{f_green_n}-{f_blue_n}_log.png', avm=AVM, original_data=rgb)

    # Handle subtracted images if available (placeholder for future)
    filternames_sub = sorted(list(image_sub_filenames_pipe.keys()),
                           key=lambda x: int(''.join(filter(str.isdigit, x))))[::-1]
    print(f"Sorted list of subtracted-filters: {filternames_sub}")

    # Create single filter images for all available filters
    print("Creating individual filter images:")
    for filtername in existing_filters:
        print(f"Creating image for {filtername}")
        data = fits.getdata(repr_image_filenames[filtername])

        # Asinh stretch
        norm_asinh = simple_norm(data, stretch='asinh', min_percent=1, max_percent=99.5)
        img_asinh = norm_asinh(data)
        img_asinh = np.stack([img_asinh, img_asinh, img_asinh], axis=2)

        fn = ''.join(filter(str.isdigit, filtername))
        # Create original data stack for transparency detection
        original_data_stack = np.stack([data, data, data], axis=2)
        save_rgb(img_asinh, f'{png_path}/SgrA_{fn}_asinh.png', avm=AVM, original_data=original_data_stack)

        # Log stretch
        norm_log = simple_norm(data, stretch='log', min_percent=1.5, max_percent=99.5)
        img_log = norm_log(data)
        img_log = np.stack([img_log, img_log, img_log], axis=2)

        save_rgb(img_log, f'{png_path}/SgrA_{fn}_log.png', avm=AVM, original_data=original_data_stack)

def main():
    # Use multiple target filters to create comprehensive coverage
    # Start with MIRI f770w as it's a good middle wavelength
    target_filters = ['f770w', 'f444w', 'f1280w']

    for target_filter in target_filters:
        if os.path.exists(image_filenames_pipe[target_filter]):
            make_pngs(target_filter)
        else:
            print(f"Skipping {target_filter} - file not found")

if __name__ == '__main__':
    main()