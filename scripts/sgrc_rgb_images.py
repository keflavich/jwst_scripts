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

# Updated image filenames for SGRC data with project code 4147
image_filenames_pipe = {
    "f115w": "/orange/adamginsburg/jwst/sgrc/mastDownload/JWST/jw04147-o012_t001_nircam_clear-f115w/jw04147-o012_t001_nircam_clear-f115w_i2d.fits",
    "f182m": "/orange/adamginsburg/jwst/sgrc/mastDownload/JWST/jw04147-o012_t001_nircam_clear-f182m/jw04147-o012_t001_nircam_clear-f182m_i2d.fits",
    "f212n": "/orange/adamginsburg/jwst/sgrc/mastDownload/JWST/jw04147-o012_t001_nircam_clear-f212n/jw04147-o012_t001_nircam_clear-f212n_i2d.fits",
    "f360m": "/orange/adamginsburg/jwst/sgrc/mastDownload/JWST/jw04147-o012_t001_nircam_clear-f360m/jw04147-o012_t001_nircam_clear-f360m_i2d.fits",
    "f480m": "/orange/adamginsburg/jwst/sgrc/mastDownload/JWST/jw04147-o012_t001_nircam_clear-f480m/jw04147-o012_t001_nircam_clear-f480m_i2d.fits",
    "f162m": "/orange/adamginsburg/jwst/sgrc/mastDownload/JWST/jw04147-o012_t001_nircam_f150w2-f162m/jw04147-o012_t001_nircam_f150w2-f162m_i2d.fits",
    "f405n": "/orange/adamginsburg/jwst/sgrc/mastDownload/JWST/jw04147-o012_t001_nircam_f405n-f444w/jw04147-o012_t001_nircam_f405n-f444w_i2d.fits",
}

# No subtracted images available for sgrc initially
image_sub_filenames_pipe = {}

def make_pngs(target_filter='f480m', new_basepath='/orange/adamginsburg/jwst/sgrc/data_reprojected/'):
    print(f"Making PNGs for {target_filter}")

    png_path = f'/orange/adamginsburg/jwst/sgrc/pngs_{target_filter[1:-1]}'
    os.makedirs(png_path, exist_ok=True)
    os.makedirs(new_basepath, exist_ok=True)

    tgt_header = fits.getheader(image_filenames_pipe[target_filter], ext=('SCI', 1))
    AVM = pyavm.AVM.from_header(tgt_header)

    repr_image_filenames = {x: y.replace("i2d", f"i2d_pipeline_v0.1_reprj_{target_filter[:-1]}") for x,y in image_filenames_pipe.items()}
    repr_image_filenames = {x: (new_basepath+os.path.basename(y)) for x,y in repr_image_filenames.items()}
    repr_image_sub_filenames = {x: y.replace(".fits", f"reprj_{target_filter[:-1]}.fits") for x,y in image_sub_filenames_pipe.items()}
    repr_image_sub_filenames = {x: (new_basepath+os.path.basename(y)) for x,y in repr_image_sub_filenames.items()}

    for filtername in image_filenames_pipe:
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

    # ALMA data not available for sgrc - skipping ALMA overlay functionality
    alma_sgrc_reprojected_jwst = None
    alma_level = None

    filternames = sorted(list(image_filenames_pipe.keys()),
                        key=lambda x: int(''.join(filter(str.isdigit, x))))[::-1]
    print(f"Sorted list of filters: {filternames}")

    # Create RGB combinations with available filters
    if len(filternames) >= 3:
        for i in range(len(filternames) - 2):
            f1, f2, f3 = filternames[i], filternames[i+1], filternames[i+2]
            print(f1,f2,f3)
            rgb = np.array([
                fits.getdata(repr_image_filenames[f1]),
                fits.getdata(repr_image_filenames[f2]),
                fits.getdata(repr_image_filenames[f3]),
            ]).swapaxes(0,2).swapaxes(0,1)
            rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,0]),
                                simple_norm(rgb[:,:,1], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,1]),
                                simple_norm(rgb[:,:,2], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)

            f1n, f2n, f3n = ''.join(filter(str.isdigit, f1)), ''.join(filter(str.isdigit, f2)), ''.join(filter(str.isdigit, f3))
            save_rgb(rgb_scaled, f'{png_path}/SGRC_RGB_{f1n}-{f2n}-{f3n}.png', avm=AVM, original_data=rgb)

            rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='log', min_percent=1.5, max_percent=99.5)(rgb[:,:,0]),
                                simple_norm(rgb[:,:,1], stretch='log', min_percent=1.5, max_percent=99.5)(rgb[:,:,1]),
                                simple_norm(rgb[:,:,2], stretch='log', min_percent=1.5, max_percent=99.5)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)

            save_rgb(rgb_scaled, f'{png_path}/SGRC_RGB_{f1n}-{f2n}-{f3n}_log.png', avm=AVM, original_data=rgb)

    filternames_sub = sorted(list(image_sub_filenames_pipe.keys()),
                           key=lambda x: int(''.join(filter(str.isdigit, x))))[::-1]
    print(f"Sorted list of subtracted-filters: {filternames_sub}")

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
                raise ex

            rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,0]),
                                simple_norm(rgb[:,:,1], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,1]),
                                simple_norm(rgb[:,:,2], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)

            f1n, f2n, f3n = ''.join(filter(str.isdigit, f1)), ''.join(filter(str.isdigit, f2)), ''.join(filter(str.isdigit, f3))
            save_rgb(rgb_scaled, f'{png_path}/SGRC_RGB_{f1n}-{f2n}-{f3n}_sub.png', avm=AVM, original_data=rgb)

            rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='log', min_percent=1.0, max_percent=99.5)(rgb[:,:,0]),
                                simple_norm(rgb[:,:,1], stretch='log', min_percent=1.0, max_percent=99.5)(rgb[:,:,1]),
                                simple_norm(rgb[:,:,2], stretch='log', min_percent=1.0, max_percent=99.5)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)

            save_rgb(rgb_scaled, f'{png_path}/SGRC_RGB_{f1n}-{f2n}-{f3n}_sub_log.png', avm=AVM, original_data=rgb)

    # Create single filter images
    print("Creating individual filter images:")
    for filtername in filternames:
        print(f"Creating image for {filtername}")
        data = fits.getdata(repr_image_filenames[filtername])

        # Asinh stretch
        norm_asinh = simple_norm(data, stretch='asinh', min_percent=1, max_percent=99.5)
        img_asinh = norm_asinh(data)
        img_asinh = np.stack([img_asinh, img_asinh, img_asinh], axis=2)

        fn = ''.join(filter(str.isdigit, filtername))
        save_rgb(img_asinh, f'{png_path}/SGRC_{fn}_asinh.png', avm=AVM, original_data=np.stack([data, data, data], axis=2))

        # Log stretch
        norm_log = simple_norm(data, stretch='log', min_percent=1.5, max_percent=99.5)
        img_log = norm_log(data)
        img_log = np.stack([img_log, img_log, img_log], axis=2)

        save_rgb(img_log, f'{png_path}/SGRC_{fn}_log.png', avm=AVM, original_data=np.stack([data, data, data], axis=2))

def main():
    # Use f480m as default target filter for sgrc
    for target_filter in ('f480m', 'f360m'):
        make_pngs(target_filter)

if __name__ == '__main__':
    main()
