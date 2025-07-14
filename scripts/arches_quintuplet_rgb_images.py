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
    img_pil.save(filename)

    if avm is not None:
        base = os.path.basename(filename)
        dir = os.path.dirname(filename)
        avmname = os.path.join(dir, 'avm_'+base)
        avm.embed(filename, avmname)
        shutil.move(avmname, filename)

    # Save as JPEG without transparency (JPEG doesn't support alpha channel)
    filename_jpg = filename.replace('.png', '.jpg')
    img_rgb = PIL.Image.fromarray(img[::flip,:,:], mode='RGB')
    img_rgb.save(filename_jpg, format='JPEG',
                 quality=95,
                 progressive=True)

    return img_pil

# Updated image filenames for Arches Quintuplet data with project code 2045
image_filenames_pipe = {
    "f212n": "/orange/adamginsburg/jwst/arches_quintuplet/mastDownload/JWST/jw02045-o001_t001_nircam_clear-f212n/jw02045-o001_t001_nircam_clear-f212n_i2d.fits",
    "f323n": "/orange/adamginsburg/jwst/arches_quintuplet/mastDownload/JWST/jw02045-o001_t001_nircam_f322w2-f323n/jw02045-o001_t001_nircam_f322w2-f323n_i2d.fits",
}

# No subtracted images available for arches_quintuplet initially
image_sub_filenames_pipe = {}

def create_composite_channel(data1, data2, method='average'):
    """
    Create a composite third channel from two filter images

    Parameters:
    -----------
    data1, data2 : numpy arrays
        The two filter images
    method : str
        Method for combining: 'average', 'difference', 'ratio'

    Returns:
    --------
    composite : numpy array
        The composite channel
    """
    if method == 'average':
        # Simple average of the two channels
        composite = (data1 + data2) / 2.0
    elif method == 'difference':
        # Absolute difference (enhanced contrast)
        composite = np.abs(data1 - data2)
    elif method == 'ratio':
        # Ratio image (with protection against division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            composite = np.where(data2 != 0, data1 / data2, 0)
        # Normalize ratio to reasonable range
        composite = np.clip(composite, 0, np.percentile(composite[composite > 0], 99))
    else:
        raise ValueError(f"Unknown method: {method}")

    return composite

def make_pngs(target_filter='f323n', new_basepath='/orange/adamginsburg/jwst/arches_quintuplet/data_reprojected/'):
    print(f"Making PNGs for {target_filter}")

    png_path = f'/orange/adamginsburg/jwst/arches_quintuplet/pngs_{target_filter[1:-1]}'
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

    # ALMA data not available for arches_quintuplet - skipping ALMA overlay functionality
    alma_arches_quintuplet_reprojected_jwst = None
    alma_level = None

    filternames = sorted(list(image_filenames_pipe.keys()),
                        key=lambda x: int(''.join(filter(str.isdigit, x))))[::-1]
    print(f"Available filters: {filternames}")

    # Create RGB combinations with the two available filters plus composite
    if len(filternames) >= 2:
        f1, f2 = filternames[0], filternames[1]  # f323n, f212n (sorted by wavelength, descending)
        print(f"Creating RGB with filters: {f1}, {f2}")

        # Load the two filter images
        data1 = fits.getdata(repr_image_filenames[f1])
        data2 = fits.getdata(repr_image_filenames[f2])

        # Create different composite methods for the third channel
        for composite_method in ['average', 'difference']:
            print(f"Creating RGB with composite method: {composite_method}")

            # Create composite third channel
            data_composite = create_composite_channel(data1, data2, method=composite_method)

            # Create RGB array: R=f323n, G=composite, B=f212n
            rgb = np.array([data1, data_composite, data2]).swapaxes(0,2).swapaxes(0,1)

            # Apply asinh stretch
            rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,0]),
                                simple_norm(rgb[:,:,1], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,1]),
                                simple_norm(rgb[:,:,2], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)

            f1n, f2n = ''.join(filter(str.isdigit, f1)), ''.join(filter(str.isdigit, f2))
            save_rgb(rgb_scaled, f'{png_path}/ArchesQuintuplet_RGB_{f1n}-{composite_method}-{f2n}.png', avm=AVM, original_data=rgb)

            # Apply log stretch
            rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='log', min_percent=1.5, max_percent=99.5)(rgb[:,:,0]),
                                simple_norm(rgb[:,:,1], stretch='log', min_percent=1.5, max_percent=99.5)(rgb[:,:,1]),
                                simple_norm(rgb[:,:,2], stretch='log', min_percent=1.5, max_percent=99.5)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)

            save_rgb(rgb_scaled, f'{png_path}/ArchesQuintuplet_RGB_{f1n}-{composite_method}-{f2n}_log.png', avm=AVM, original_data=rgb)

    # Handle subtracted images if available
    filternames_sub = sorted(list(image_sub_filenames_pipe.keys()),
                           key=lambda x: int(''.join(filter(str.isdigit, x))))[::-1]
    print(f"Available subtracted filters: {filternames_sub}")

    if len(filternames_sub) >= 2:
        f1, f2 = filternames_sub[0], filternames_sub[1]
        print(f"Creating RGB with subtracted filters: {f1}, {f2}")

        try:
            # Load the two subtracted filter images
            data1 = fits.getdata(repr_image_sub_filenames[f1])
            data2 = fits.getdata(repr_image_sub_filenames[f2])

            # Create composite third channel
            data_composite = create_composite_channel(data1, data2, method='average')

            rgb = np.array([data1, data_composite, data2]).swapaxes(0,2).swapaxes(0,1)

        except Exception as ex:
            print(ex)
            print(f"Shape of {f1} is {fits.getdata(repr_image_sub_filenames[f1]).shape}")
            print(f"Shape of {f2} is {fits.getdata(repr_image_sub_filenames[f2]).shape}")
            raise ex

        rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,0]),
                            simple_norm(rgb[:,:,1], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,1]),
                            simple_norm(rgb[:,:,2], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)

        f1n, f2n = ''.join(filter(str.isdigit, f1)), ''.join(filter(str.isdigit, f2))
        save_rgb(rgb_scaled, f'{png_path}/ArchesQuintuplet_RGB_{f1n}-composite-{f2n}_sub.png', avm=AVM, original_data=rgb)

        rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='log', min_percent=1.0, max_percent=99.5)(rgb[:,:,0]),
                            simple_norm(rgb[:,:,1], stretch='log', min_percent=1.0, max_percent=99.5)(rgb[:,:,1]),
                            simple_norm(rgb[:,:,2], stretch='log', min_percent=1.0, max_percent=99.5)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)

        save_rgb(rgb_scaled, f'{png_path}/ArchesQuintuplet_RGB_{f1n}-composite-{f2n}_sub_log.png', avm=AVM, original_data=rgb)

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
        save_rgb(img_asinh, f'{png_path}/ArchesQuintuplet_{fn}_asinh.png', avm=AVM, original_data=np.stack([data, data, data], axis=2))

        # Log stretch
        norm_log = simple_norm(data, stretch='log', min_percent=1.5, max_percent=99.5)
        img_log = norm_log(data)
        img_log = np.stack([img_log, img_log, img_log], axis=2)

        save_rgb(img_log, f'{png_path}/ArchesQuintuplet_{fn}_log.png', avm=AVM, original_data=np.stack([data, data, data], axis=2))

def main():
    # Use f323n as default target filter for arches_quintuplet
    for target_filter in ('f323n', 'f212n'):
        make_pngs(target_filter)

if __name__ == '__main__':
    main()