from astropy.io import fits
import string
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
from PIL import Image


def save_rgb(img: np.ndarray, filename: str, avm: Optional[pyavm.AVM] = None, flip: int = -1, original_data: Optional[np.ndarray] = None) -> PIL.Image.Image:
    """
    Save an RGB image array to a file with optional AVM metadata.

    Args:
        img: RGB image array with values in range [0,1]
        filename: Output filename
        avm: Optional AVM metadata to embed
        flip: Flip direction for the image (-1 for vertical flip)
        original_data: Original unscaled data for transparency detection

    Returns:
        PIL Image object
    """
    img = (img * 256).clip(0, 255).astype('uint8')

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

    # Create RGBA image for PNG with transparency
    img_rgba = np.dstack((img[::flip,:,:], alpha))
    img_pil = PIL.Image.fromarray(img_rgba, mode='RGBA')
    # empirical: 180 degree rotation required.
    flip_img = img_pil.transpose(Image.ROTATE_180)
    flip_img.save(filename)
    print(f"Saved {filename}")

    if avm is not None:
        base = os.path.basename(filename)
        dir = os.path.dirname(filename)
        avmname = os.path.join(dir, 'avm_' + base)
        avm.embed(filename, avmname)
        shutil.move(avmname, filename)

    return img_pil


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

    print(f"Max percent being used in create_rgb_image={max_percent}")

    rgb_scaled = np.array([
        scale_image(rgb[:,:,0], stretch=stretch, max_percent=max_percent),
        scale_image(rgb[:,:,1], stretch=stretch, max_percent=max_percent),
        scale_image(rgb[:,:,2], stretch=stretch, max_percent=max_percent)
    ]).swapaxes(0,2).swapaxes(0,1)

    return rgb_scaled, rgb_original


def reproject_images(image_filenames: Dict[str, str], target_header: fits.Header,
                    output_dir: str, repr_suffix: str = '') -> Dict[str, str]:
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
                                     os.path.basename(filename).replace("i2d", f"i2d_reprj_{repr_suffix}"))

        if not os.path.exists(output_filename):
            print(f"Reprojecting {filtername} {filename} to {output_filename}")
            result, _ = reproject.reproject_interp(filename, target_header, hdu_in='SCI')
            hdu = fits.PrimaryHDU(data=result, header=target_header)
            hdu.writeto(output_filename, overwrite=True)

        output_filenames[filtername] = output_filename

    return output_filenames


def create_and_save_rgb_combination(repr_filenames: Dict[str, str], blue_key: str, green_key: str, red_key: str,
                                  stretch: str, max_percent: int, png_path: str, avm: pyavm.AVM) -> None:
    """
    Create and save an RGB image combination from three wavelength bands.

    Args:
        repr_filenames: Dictionary mapping filter names to reprojected FITS files
        key1: Red channel filter key
        key2: Green channel filter key
        key3: Blue channel filter key
        stretch: Stretch function to apply ('asinh' or 'log')
        max_percent: Maximum percentile for scaling
        png_path: Directory to save the output PNG
        avm: AVM metadata to embed in the image
    """
    rgb_scaled, rgb_original = create_rgb_image(repr_filenames, red_key=red_key, green_key=green_key, blue_key=blue_key,
                                             stretch=stretch, max_percent=max_percent, nan_to_max=False)

    rgb_scaled[rgb_scaled.mean(axis=2) == 255, :] = 0

    # Handle filter names that may contain 'sub'
    def extract_wavelength(filter_key):
        if 'sub' in filter_key:
            return filter_key[1:].replace('sub', '')  # Remove 'f' prefix and 'sub' suffix
        else:
            return filter_key[1:-1]  # Remove 'f' prefix and 'w'/'n'/'m' suffix

    wl1_str = extract_wavelength(blue_key)
    wl2_str = extract_wavelength(green_key)
    wl3_str = extract_wavelength(red_key)

    # Check if any filters contain 'sub' for filename modification
    has_sub = 'sub' in blue_key or 'sub' in green_key or 'sub' in red_key

    try:
        wl1 = int(wl1_str)
        wl2 = int(wl2_str)
        wl3 = int(wl3_str)
        maxwl = max(wl1, wl2, wl3)
        if maxwl < 30000: # ignore the 'sub' values
            assert wl1 < wl2 < wl3

        # Create filename with wavelength numbers
        if maxwl < 500:
            base_filename = f'wd2_nircam_RGB_{wl3}-{wl2}-{wl1}'
        else:
            base_filename = f'wd2_RGB_{wl3}-{wl2}-{wl1}'

    except ValueError:
        # If we can't convert to int (e.g., due to 'sub' filters), use filter names directly
        base_filename = f'wd2_RGB_{red_key}-{green_key}-{blue_key}'

    # Add 'sub' indicator if any filter contains 'sub'
    if has_sub:
        base_filename += '_sub'

    output_filename = f'{base_filename}_{stretch}_max{max_percent}.png'

    save_rgb(rgb_scaled, os.path.join(png_path, output_filename), avm=avm, original_data=rgb_original)


def main():
    # Configuration
    base_path = '/orange/adamginsburg/jwst/wd2'
    data_path = os.path.join(base_path, 'data_reprojected')
    png_path = os.path.join(base_path, 'pngs')

    # Input filenames
    miri_image_filenames = {
        "f1000w": 'miri_F1000W_pid3523_combined_SF_i2d.fits',
        "f1130w": 'miri_F1130W_pid3523_combined_SF_i2d.fits',
        "f770w": 'miri_F770W_pid3523_combined_SF_i2d.fits',
    }
    miri_image_filenames = {k: os.path.join(base_path, v) for k, v in miri_image_filenames.items()}

    nircam_image_filenames = {
        "f212n": 'jw03523-o005_t001_nircam_clear-f212n_i2d.fits',
        "f164n": 'jw03523-o005_t001_nircam_f150w2-f164n_i2d.fits',
        "f405n": 'jw03523-o005_t001_nircam_f405n-f444w_i2d.fits',
        "f115w": 'wd2_F115W_AB_i2d.fits',
        "f150w": 'wd2_F150W_AB_i2d.fits',
        "f162m": 'wd2_F162M_AB_i2d.fits',
        "f182m": 'wd2_F182M_AB_i2d.fits',
        "f187n": 'wd2_F187N_AB_i2d.fits',
        "f200w": 'wd2_F200W_AB_i2d.fits',
        "f250m": 'wd2_F250M_AB_i2d.fits',
        "f277w": 'wd2_F277W_AB_i2d.fits',
        "f300m": 'wd2_F300M_AB_i2d.fits',
        "f335m": 'wd2_F335M_AB_i2d.fits',
        "f410m": 'wd2_F410M_AB_i2d.fits',
        "f335300sub": "wd2_F335M_F300M_AB_i2d.fits",
        "f164162sub": "wd2_F164N_F162M_AB_i2d.fits",
    }
    nircam_image_filenames = {k: os.path.join(base_path, v) for k, v in nircam_image_filenames.items()}

    # Get target header and AVM metadata
    target_header = fits.getheader(os.path.join(base_path, miri_image_filenames['f770w']), ext=('SCI', 1))
    avm = pyavm.AVM.from_header(target_header)

    # Reproject images
    repr_filenames = reproject_images(miri_image_filenames, target_header, data_path, repr_suffix='f770w')

    # Create and save RGB images with different stretches
    for stretch in ['asinh', 'log']:
        for max_percent in [99, 95, 99.9]:
            rgb_scaled, rgb_original = create_rgb_image(repr_filenames, stretch=stretch, max_percent=max_percent)

            rgb_scaled[rgb_scaled.mean(axis=2) == 255, :] = 0

            output_filename = f'wd2_miri_RGB_1130-1000-770_{stretch}_max{max_percent}.png'
            save_rgb(rgb_scaled, os.path.join(png_path, output_filename), avm=avm, original_data=rgb_original)

    target_header = fits.getheader(os.path.join(base_path, nircam_image_filenames['f182m']), ext=('SCI', 1))
    avm = pyavm.AVM.from_header(target_header)

    # Reproject images
    repr_filenames_nircam = reproject_images(nircam_image_filenames, target_header, data_path, repr_suffix='f182m')
    keylist = sorted(list(repr_filenames_nircam.keys()), key=lambda x: int(x[1:-1].strip(string.ascii_letters)))
    repr_filenames_miri = reproject_images(miri_image_filenames, target_header, data_path, repr_suffix='f182m')
    keylist = keylist + sorted(list(repr_filenames_miri.keys()), key=lambda x: int(x[1:-1]))

    repr_filenames = {**repr_filenames_nircam, **repr_filenames_miri}

    # Create and save RGB images with different stretches
    for ii in range(len(keylist)-3):
        for stretch in ['asinh', 'log']:
            for max_percent in [99, 99.5]:
                key1 = keylist[ii]
                key2 = keylist[ii+1]
                key3 = keylist[ii+2]
                create_and_save_rgb_combination(repr_filenames, blue_key=key1, green_key=key2, red_key=key3, stretch=stretch, max_percent=max_percent, png_path=png_path, avm=avm)

                # key1 = keylist[ii]
                # key2 = keylist[ii+1]
                # key3 = keylist[ii+2]
                # create_and_save_rgb_combination(repr_filenames, red_key=key1, green_key=key2, blue_key=key3, stretch=stretch, max_percent=max_percent, png_path=png_path, avm=avm)


                if ii+4 < len(keylist):
                    key1 = keylist[ii]
                    key2 = keylist[ii+2]
                    key3 = keylist[ii+4]
                    create_and_save_rgb_combination(repr_filenames, blue_key=key1, green_key=key2, red_key=key3, stretch=stretch, max_percent=max_percent, png_path=png_path, avm=avm)



if __name__ == '__main__':
    main()