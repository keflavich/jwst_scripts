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
from jwst_rgb import save_rgb

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
