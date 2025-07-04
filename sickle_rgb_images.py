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

def save_rgb(img, filename, avm=None, flip=-1, alma_data=None, alma_level=None):
    img = (img*256)
    img[img<0] = 0
    img[img>255] = 255
    img = img.astype('uint8')

    if alma_data is not None and alma_level is not None:
        contour_mask = np.zeros_like(alma_data, dtype=bool)
        contour_mask[alma_data >= alma_level] = True
        from scipy.ndimage import binary_dilation
        contour_mask1 = binary_dilation(contour_mask)
        contour_mask = contour_mask1 ^ contour_mask

        for i in range(3):
            img[contour_mask, i] = 255 - img[contour_mask, i]

    img = PIL.Image.fromarray(img[::flip,:,:])
    img.save(filename)

    if avm is not None:
        base = os.path.basename(filename)
        dir = os.path.dirname(filename)
        avmname = os.path.join(dir, 'avm_'+base)
        avm.embed(filename, avmname)
        shutil.move(avmname, filename)

    filename = filename.replace('.png', '.jpg')

    img.save(filename, format='JPEG',
             quality=95,
             progressive=True)

    return img

# Updated image filenames for Sickle data with project code 3958 - MIRI observations
image_filenames_pipe = {
    "f770w": "/orange/adamginsburg/jwst/sickle/mastDownload/JWST/jw03958-o001_t001_miri_f770w-brightsky/jw03958-o001_t001_miri_f770w-brightsky_i2d.fits",
    "f1130w": "/orange/adamginsburg/jwst/sickle/mastDownload/JWST/jw03958-o001_t001_miri_f1130w-brightsky/jw03958-o001_t001_miri_f1130w-brightsky_i2d.fits",
    "f1500w": "/orange/adamginsburg/jwst/sickle/mastDownload/JWST/jw03958-o001_t001_miri_f1500w-brightsky/jw03958-o001_t001_miri_f1500w-brightsky_i2d.fits",
}

# No subtracted images available for sickle initially
image_sub_filenames_pipe = {}

def make_pngs(target_filter='f1130w', new_basepath='/orange/adamginsburg/jwst/sickle/data_reprojected/'):
    print(f"Making PNGs for {target_filter}")

    png_path = f'/orange/adamginsburg/jwst/sickle/pngs_{target_filter[1:-1]}'
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

    # ALMA data not available for sickle - skipping ALMA overlay functionality
    alma_sickle_reprojected_jwst = None
    alma_level = None

    filternames = sorted(list(image_filenames_pipe.keys()),
                        key=lambda x: int(''.join(filter(str.isdigit, x))))[::-1]
    print(f"Sorted list of filters: {filternames}")

    # Create RGB combinations with available filters (we have exactly 3 MIRI filters)
    if len(filternames) >= 3:
        for i in range(len(filternames) - 2):
            f1, f2, f3 = filternames[i], filternames[i+1], filternames[i+2]
            print(f"Creating RGB with filters: {f1}, {f2}, {f3}")
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
            save_rgb(rgb_scaled, f'{png_path}/Sickle_RGB_{f1n}-{f2n}-{f3n}.png', avm=AVM)

            # Apply log stretch
            rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='log', min_percent=1.5, max_percent=99.5)(rgb[:,:,0]),
                                simple_norm(rgb[:,:,1], stretch='log', min_percent=1.5, max_percent=99.5)(rgb[:,:,1]),
                                simple_norm(rgb[:,:,2], stretch='log', min_percent=1.5, max_percent=99.5)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)

            save_rgb(rgb_scaled, f'{png_path}/Sickle_RGB_{f1n}-{f2n}-{f3n}_log.png', avm=AVM)

    # Special RGB combination with optimal wavelength mapping for MIRI
    # f1500w (longest) -> Red, f1130w (middle) -> Green, f770w (shortest) -> Blue
    if len(filternames) == 3:
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

        save_rgb(rgb_scaled, f'{png_path}/Sickle_RGB_MIRI_optimal.png', avm=AVM)

        # Apply log stretch
        rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='log', min_percent=1.5, max_percent=99.5)(rgb[:,:,0]),
                            simple_norm(rgb[:,:,1], stretch='log', min_percent=1.5, max_percent=99.5)(rgb[:,:,1]),
                            simple_norm(rgb[:,:,2], stretch='log', min_percent=1.5, max_percent=99.5)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)

        save_rgb(rgb_scaled, f'{png_path}/Sickle_RGB_MIRI_optimal_log.png', avm=AVM)

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
            save_rgb(rgb_scaled, f'{png_path}/Sickle_RGB_{f1n}-{f2n}-{f3n}_sub.png', avm=AVM)

            rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='log', min_percent=1.0, max_percent=99.5)(rgb[:,:,0]),
                                simple_norm(rgb[:,:,1], stretch='log', min_percent=1.0, max_percent=99.5)(rgb[:,:,1]),
                                simple_norm(rgb[:,:,2], stretch='log', min_percent=1.0, max_percent=99.5)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)

            save_rgb(rgb_scaled, f'{png_path}/Sickle_RGB_{f1n}-{f2n}-{f3n}_sub_log.png', avm=AVM)

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
        save_rgb(img_asinh, f'{png_path}/Sickle_{fn}_asinh.png', avm=AVM)

        # Log stretch
        norm_log = simple_norm(data, stretch='log', min_percent=1.5, max_percent=99.5)
        img_log = norm_log(data)
        img_log = np.stack([img_log, img_log, img_log], axis=2)

        save_rgb(img_log, f'{png_path}/Sickle_{fn}_log.png', avm=AVM)

def main():
    # Use f1130w as default target filter for sickle (middle wavelength)
    for target_filter in ('f1130w', 'f770w', 'f1500w'):
        make_pngs(target_filter)

if __name__ == '__main__':
    main()