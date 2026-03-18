import numpy as np
import PIL
from scipy.ndimage import binary_dilation, label
from reproject.hips import reproject_to_hips
from reproject import reproject_interp
import os
import shutil
from PIL import Image
from tqdm import tqdm


def save_rgb(img, filename, avm=None, flip=-1, alma_data=None, alma_level=None,
             original_data=None, flip_alma=False,
             transpose=Image.ROTATE_180, verbose=True, hips=True, overwrite=True):
    """
    Save an RGB image to a PNG and a JPG file with embedded AVM metadata.

    Warning: the flipping conventions make almost no sense.  You'll probably
    have to try every combination of flipping and rotation imaginable to get it
    right, just like with a USB stick

    Parameters
    ----------
    img : array-like
        The image data to save.
    filename : str
        The filename to save the image to.  Will have both .png and .jpg extensions.
    avm : pyavm.AVM
        The AVM to embed in the image.
    flip : int, optional
        The flip direction to apply to the image.
    alma_data : array-like, optional
    """
    if verbose:
        print(f"Saving RGB image to {filename}")
    img = (img*256).clip(0, 255).astype('uint8')

    # Create alpha channel for transparency
    alpha = np.ones(img.shape[:2], dtype=np.uint8) * 255  # Start with fully opaque

    if original_data is not None:
        # Make pixels transparent where original data is NaN or very small
        # Check each channel for blank pixels
        for i in range(3):
            if i < original_data.shape[2]:
                blank_mask = (np.isnan(original_data[:,:,i]) |
                             (np.abs(original_data[:,:,i]) < 1e-5))
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
        if flip_alma:
            # but apparently this is backward, at least in the case of W51
            contour_mask = contour_mask[::flip,:]

        for i in range(3):
            img[contour_mask, i] = 255 - img[contour_mask, i]

    # Create RGBA image for PNG with transparency
    img_rgba = np.dstack((img[::flip,:,:], alpha))
    img_pil = PIL.Image.fromarray(img_rgba, mode='RGBA')
    # empirical: 180 degree rotation required.
    if transpose is not None:
        flip_img = img_pil.transpose(transpose)
    else:
        flip_img = img_pil
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
    if transpose is not None:
        img_rgb = img_rgb.transpose(transpose)
    img_rgb.save(filename_jpg, format='JPEG',
                 quality=95,
                 progressive=True)

    if hips:
        if overwrite and os.path.exists(filename.replace('.png', '_hips')):
            shutil.rmtree(filename.replace('.png', '_hips'))
        print("Reprojecting to HiPS...")
        reproject_to_hips(filename,
            level=None,
            reproject_function=reproject_interp,
            output_directory=filename.replace('.png', '_hips'),
            threads=8,
            coord_system_out='galactic',
            progress_bar=tqdm)

    if verbose:
        print(f"Saved {filename} and {filename_jpg}")

    return img_pil


def fill_nan(data):
    mask_nan = np.isnan(data)
    if not mask_nan.any():
        return data

    labeled, num_labels = label(mask_nan)
    if num_labels <= 1:
        return data

    counts = np.bincount(labeled.ravel())
    if counts.size == 0:
        return data
    counts[0] = 0  # ignore real signal
    largest_label = counts.argmax()

    faint_threshold = np.nanpercentile(data, 90)
    bright_threshold = np.nanpercentile(data, 99.)
    faint_value = np.nanpercentile(data, 1)
    datamax = np.nanmax(data)

    # check what the surroundings of our island looks like and decide how to replace them.  The expectation is that this is mostly infilling stars, but it _could_ infill more extended regions.
    for label_id in tqdm(range(1, num_labels + 1), desc='Filling NaN: '):
        if label_id == largest_label:
            continue
        mask = labeled == label_id

        dilated_mask = binary_dilation(mask, iterations=2)
        border_mask = dilated_mask & ~mask

        border_values = data[border_mask]

        median_value = np.nanmedian(border_values)

        if median_value < faint_threshold:
            data[mask] = faint_value
        elif median_value > bright_threshold:
            data[mask] = datamax
        else:
            data[mask] = median_value

    return data
