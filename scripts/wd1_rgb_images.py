from astropy.io import fits
import numpy as np
from astropy.visualization import simple_norm
import pylab as plt
from astropy import wcs
import os
from reproject import reproject_interp
import reproject
import PIL
import pyavm
import shutil
from typing import Dict, Tuple, Optional
from pathlib import Path
from jwst_rgb import save_rgb


def scale_image(img: np.ndarray, stretch: str = 'asinh', min_percent: int = 1, max_percent: int = 99) -> np.ndarray:
    """
    Scale an image using the specified stretch function.

    Args:
        img: Input image array
        stretch: Stretch function ('asinh' or 'log')
        min_percent: Minimum percentile for scaling
        max_percent: Maximum percentile for scaling

    Returns:
        Scaled image array
    """
    return simple_norm(img, stretch=stretch, min_percent=min_percent, max_percent=max_percent)(img)


def fix_nan(img: np.ndarray) -> np.ndarray:
    img[np.isnan(img)] = np.nanmax(img)
    return img


def create_rgb_image(filenames: Dict[str, str], red_key: str = 'f1130w', green_key: str = 'f1000w', blue_key: str = 'f770w', stretch: str = 'asinh', max_percent: int = 99, nan_to_max: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create an RGB image from three FITS files.

    Args:
        filenames: Dictionary mapping color channels to FITS filenames
        stretch: Stretch function to apply ('asinh' or 'log')

    Returns:
        Tuple of (scaled RGB image array, original RGB image array)
    """
    assert isinstance(filenames, dict)
    if nan_to_max:
        rgb_original = np.array([
            fits.getdata(filenames[red_key]),
            fits.getdata(filenames[green_key]),
            fits.getdata(filenames[blue_key])
        ]).swapaxes(0,2).swapaxes(0,1)

        rgb = np.array([
            fix_nan(fits.getdata(filenames[red_key])),
            fix_nan(fits.getdata(filenames[green_key])),
            fix_nan(fits.getdata(filenames[blue_key]))
        ]).swapaxes(0,2).swapaxes(0,1)
    else:
        rgb_original = np.array([
            fits.getdata(filenames[red_key]),
            fits.getdata(filenames[green_key]),
            fits.getdata(filenames[blue_key])
        ]).swapaxes(0,2).swapaxes(0,1)

        rgb = np.array([
            np.nan_to_num(fits.getdata(filenames[red_key])),
            np.nan_to_num(fits.getdata(filenames[green_key])),
            np.nan_to_num(fits.getdata(filenames[blue_key]))
        ]).swapaxes(0,2).swapaxes(0,1)

    rgb_scaled = np.array([
        scale_image(rgb[:,:,0], stretch=stretch, max_percent=max_percent),
        scale_image(rgb[:,:,1], stretch=stretch, max_percent=max_percent),
        scale_image(rgb[:,:,2], stretch=stretch, max_percent=max_percent)
    ]).swapaxes(0,2).swapaxes(0,1)

    return rgb_scaled, rgb_original


def reproject_images(image_filenames: Dict[str, str], target_header: fits.Header,
                    output_dir: str) -> Dict[str, str]:
    """
    Reproject images to match the target header.

    Args:
        image_filenames: Dictionary mapping filter names to input FITS files
        target_header: Target FITS header for reprojection
        output_dir: Directory for reprojected files

    Returns:
        Dictionary mapping filter names to reprojected filenames
    """
    output_filenames = {}

    for filtername, filename in image_filenames.items():
        output_filename = os.path.join(output_dir,
                                     os.path.basename(filename).replace("i2d", "i2d_reprj_f770"))

        if not os.path.exists(output_filename):
            print(f"Reprojecting {filtername} {filename} to {output_filename}")
            result, _ = reproject.reproject_interp(filename, target_header, hdu_in='SCI')
            hdu = fits.PrimaryHDU(data=result, header=target_header)
            hdu.writeto(output_filename, overwrite=True)

        output_filenames[filtername] = output_filename

    return output_filenames


def main():
    # Configuration
    base_path = '/orange/adamginsburg/jwst/wd1'
    data_path = os.path.join(base_path, 'data_reprojected')
    png_path = os.path.join(base_path, 'pngs')

    # Input filenames
    image_filenames = {
        "f1000w": 'miri_F1000W_pid1905_combined_SF_i2d.fits',
        "f1130w": 'miri_F1130W_pid1905_combined_SF_i2d.fits',
        "f770w": 'miri_F770W_pid1905_combined_SF_i2d.fits',
        # "f1000w": 'jw01905-o002_t001_miri_f1000w_i2d.fits',
        # "f1130w": 'jw01905-o002_t001_miri_f1130w_i2d.fits',
        # "f770w": 'jw01905-o002_t001_miri_f770w_i2d.fits',
    }

    nircam_image_filenames = {
        "f115w": 'jw01905-o001_t001_nircam_clear-f115w_i2d.fits',
        "f150w": 'jw01905-o001_t001_nircam_clear-f150w_i2d.fits',
        "f187n": 'jw01905-o001_t001_nircam_clear-f187n_i2d.fits',
        "f200w": 'jw01905-o001_t001_nircam_clear-f200w_i2d.fits',
        "f212n": 'jw01905-o001_t001_nircam_clear-f212n_i2d.fits',
        "f277w": 'jw01905-o001_t001_nircam_clear-f277w_i2d.fits',
        "f444w": 'jw01905-o001_t001_nircam_clear-f444w_i2d.fits',
        "f150w2": 'jw01905-o001_t001_nircam_f150w2-f164n_i2d.fits',
        "f322w2": 'jw01905-o001_t001_nircam_f322w2-f323n_i2d.fits',
        "f405n": 'jw01905-o001_t001_nircam_f405n-f444w_i2d.fits',
        "f444w": 'jw01905-o001_t001_nircam_f444w-f466n_i2d.fits',
    }

    # Get target header and AVM metadata
    target_header = fits.getheader(os.path.join(base_path, image_filenames['f770w']), ext=('SCI', 1))
    avm = pyavm.AVM.from_header(target_header)

    # Reproject images
    repr_filenames = reproject_images(image_filenames, target_header, data_path)

    # Create and save RGB images with different stretches
    for stretch in ['asinh', 'log']:
        for max_percent in [99, 95, 99.9]:
            rgb_scaled, rgb_original = create_rgb_image(repr_filenames, stretch=stretch, max_percent=max_percent)

            rgb_scaled[rgb_scaled.mean(axis=2) == 255, :] = 0

            plt.figure(figsize=(24,10))
            plt.imshow(rgb_scaled, origin='lower')
            plt.xticks([])
            plt.yticks([])

            output_filename = f'wd1_miri_RGB_1130-1000-770_{stretch}_max{max_percent}.png'
            save_rgb(rgb_scaled, os.path.join(png_path, output_filename), avm=avm, original_data=rgb_original)

    target_header = fits.getheader(os.path.join(base_path, nircam_image_filenames['f212n']), ext=('SCI', 1))
    avm = pyavm.AVM.from_header(target_header)

    # Reproject images
    repr_filenames_nircam = reproject_images(nircam_image_filenames, target_header, data_path)
    keylist = sorted(list(repr_filenames_nircam.keys()))

    # Create and save RGB images with different stretches
    for ii in range(len(repr_filenames_nircam)-3):
        for stretch in ['asinh', 'log']:
            for max_percent in [99, 95]:
                key1 = keylist[ii]
                key2 = keylist[ii+1]
                key3 = keylist[ii+2]
                rgb_scaled, rgb_original = create_rgb_image(repr_filenames_nircam, red_key=key1, green_key=key2, blue_key=key3, stretch=stretch, max_percent=max_percent, nan_to_max=False)

                rgb_scaled[rgb_scaled.mean(axis=2) == 255, :] = 0

                plt.figure(figsize=(24,10))
                plt.imshow(rgb_scaled, origin='lower')
                plt.xticks([])
                plt.yticks([])

                wl1 = key1[1:4]
                wl2 = key2[1:4]
                wl3 = key3[1:4]

                output_filename = f'wd1_nircam_RGB_{wl1}-{wl2}-{wl3}_{stretch}_max{max_percent}.png'
                save_rgb(rgb_scaled, os.path.join(png_path, output_filename), avm=avm, original_data=rgb_original)



if __name__ == '__main__':
    main()