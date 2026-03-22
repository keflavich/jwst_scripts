import numpy as np
import PIL
from scipy.ndimage import binary_dilation, label, find_objects
from reproject.hips import reproject_to_hips
from reproject import reproject_interp
import os
import shutil
from PIL import Image
from tqdm import tqdm


def save_rgb(img, filename, avm=None, flip=-1, alma_data=None, alma_level=None,
             original_data=None, flip_alma=False,
             alpha_only_edges=True,
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
                if alpha_only_edges:
                    labeled, num_labels = label(blank_mask)
                    edge = np.zeros(blank_mask.shape, dtype=bool)
                    edge[:1, :] = True
                    edge[-1:, :] = True
                    edge[:, :1] = True
                    edge[:, -1:] = True
                    labels_to_keep = [label_id for label_id in range(1, num_labels + 1) if (labeled[edge] == label_id).any()]
                    blank_mask_ = np.isin(labeled, labels_to_keep)
                    # > 2 works if the edge nans are contiguous, but fails if there are two non-contiguous edge nan regions.
                    # this assertion is intentional and _must_ be included.  If there are any non-edge NaN regions, we want them to be _ignored_ in the alpha-setting.
                    # if num_labels > 3:
                    #     assert blank_mask_.sum() < blank_mask.sum()
                    # else:
                    assert blank_mask_.sum() <= blank_mask.sum()
                    blank_mask = blank_mask_
                        
                alpha[blank_mask] = 0

    # Apply flip to alpha channel to match image
    alpha = alpha[::flip,:]

    if alma_data is not None and alma_level is not None:
        contour_mask = np.zeros_like(alma_data, dtype=bool)
        contour_mask[alma_data >= alma_level] = True
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


def fill_nan(data, bad_data_min_threshold=1e-5, big_island_threshold=100):
    mask_nan = np.isnan(data)
    if bad_data_min_threshold is not None:
        mask_nan |= (data < bad_data_min_threshold)
    if not mask_nan.any():
        return data

    labeled, num_labels = label(mask_nan)
    if num_labels <= 1:
        return data

    counts = np.bincount(labeled.ravel())
    if counts.size == 0:
        return data

    edge = np.zeros(mask_nan.shape, dtype=bool)
    edge[:1, :] = True
    edge[-1:, :] = True
    edge[:, :1] = True
    edge[:, -1:] = True
    labels_to_keep = [label_id for label_id in range(1, num_labels + 1) if not (labeled[edge] == label_id).any()]
    to_mask = np.isin(labeled, labels_to_keep)

    slices = find_objects(labeled)

    faint_threshold = np.nanpercentile(data, 90)
    bright_threshold = np.nanpercentile(data, 99.)
    faint_value = np.nanpercentile(data, 10)
    very_faint_value = np.nanpercentile(data, 1)
    datamax = np.nanmax(data)

    nnan = (to_mask & mask_nan).sum()

    # check what the surroundings of our island looks like and decide how to replace them.  The expectation is that this is mostly infilling stars, but it _could_ infill more extended regions.
    for label_id in tqdm(labels_to_keep, desc=f'Filling {nnan} NaN: '):
        slc = slices[label_id - 1]

        slc_expanded = tuple(
            slice(max(0, s.start - 5), min(data.shape[i], s.stop + 5))
            for i, s in enumerate(slc))
        mask = labeled[slc_expanded] == label_id

        dilated_mask = binary_dilation(mask, iterations=4 if mask.sum() > big_island_threshold else 2)
        border_mask = dilated_mask & ~mask

        border_values = data[slc_expanded][border_mask]

        if mask.sum() > big_island_threshold:
            # handle MIRI giant region case?
            mask = binary_dilation(mask, iterations=2)
            replacement = np.nanpercentile(border_values, 99)
            data[slc_expanded][mask] = replacement
        else:
            median_value = np.nanmedian(border_values)

            if median_value > bright_threshold:
                data[slc_expanded][mask] = datamax
            else:
                data[slc_expanded][mask] = median_value
            # there was some faint_value logic, but there are too many edge cases: the faint values could be negatives (in MIRI images) or something else unknown

    print(f"Nans filled.  nnan={np.isnan(data).sum()} and should be {to_mask.sum()} (if the first number is lower, it could be because there are negatives in the mask that we haven't forced to be nan)")

    return data
