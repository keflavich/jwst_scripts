import numpy as np
import PIL
from scipy.ndimage import binary_dilation, label, find_objects
from reproject.hips import reproject_to_hips
from reproject import reproject_interp
import os
import shutil
from PIL import Image
from tqdm import tqdm


def _flip_wcs(wcs, ny, nx, flip_rows=False, flip_cols=False):
    """Return a copy of ``wcs`` with pixel axes reversed to match a numpy
    flipud (flip_rows) and/or fliplr (flip_cols) of a (ny, nx) array.

    array axis0 = rows = y = WCS pixel axis 2 (crpix2);
    array axis1 = cols = x = WCS pixel axis 1 (crpix1).
    astropy WCS slicing refuses negative steps, so transform CRPIX/CD(PC)
    by hand: reversing pixel axis j sets crpix_j -> N_j + 1 - crpix_j and
    negates column j of the linear transform.
    """
    w = wcs.deepcopy()
    has_cd = w.wcs.has_cd()
    mat = (w.wcs.cd if has_cd else w.wcs.get_pc()).copy()
    crpix = w.wcs.crpix.copy()
    if flip_cols:
        crpix[0] = nx + 1 - crpix[0]
        mat[:, 0] = -mat[:, 0]
    if flip_rows:
        crpix[1] = ny + 1 - crpix[1]
        mat[:, 1] = -mat[:, 1]
    w.wcs.crpix = crpix
    if has_cd:
        w.wcs.cd = mat
    else:
        w.wcs.pc = mat
    return w


def _net_flip(flip, transpose):
    """(flip, PIL transpose) -> (flip_rows, flip_cols) applied to the pixels.

    save_rgb does ``img[::flip]`` (flip==-1 reverses rows) then a PIL
    ``transpose`` (ROTATE_180 reverses both axes).  flipud and fliplr commute,
    so the net is the XOR of the row/col reversals.
    """
    flip_rows = False
    flip_cols = False
    if flip == -1:
        flip_rows ^= True
    elif flip == 1:
        pass
    else:
        raise ValueError(f"Unsupported flip={flip}; only +1/-1 handled")

    if transpose is None:
        pass
    elif transpose == Image.ROTATE_180:
        flip_rows ^= True
        flip_cols ^= True
    else:
        raise ValueError(
            f"Unsupported transpose={transpose}; only None/ROTATE_180 handled")
    return flip_rows, flip_cols


def _faithful_flipped_avm(wcs, img_shape, flip, transpose):
    """Build an AVM (as CDMatrix) describing the PNG *as saved* from a TRUE WCS.

    ``wcs`` must be a faithful celestial astropy WCS of the un-flipped image
    (e.g. ``WCS(fits_header)``).  Do NOT pass ``pyavm.AVM.from_header(...)
    .to_wcs()`` -- pyavm's Scale+Rotation model is lossy for rotated fields
    (it can be ~hundreds of arcsec off with a flipped diagonal sign), and the
    resulting flip is wrong on rotated GC fields.

    We apply the same net flipud/fliplr the pixels received to the WCS, then
    store the result as a flat ``Spatial.CDMatrix`` -- pyavm's to_wcs()
    honors CDMatrix verbatim, whereas its Scale+Rotation path hardcodes cdelt
    signs and silently drops the parity/handedness flip.
    """
    import pyavm

    ny, nx = img_shape[:2]
    flip_rows, flip_cols = _net_flip(flip, transpose)

    wcs_flipped = _flip_wcs(wcs, ny, nx, flip_rows=flip_rows, flip_cols=flip_cols)
    wcs_flipped.pixel_shape = (nx, ny)
    avm_out = pyavm.AVM.from_wcs(wcs_flipped, shape=(ny, nx))
    cd = wcs_flipped.pixel_scale_matrix
    avm_out.Spatial.CDMatrix = [cd[0, 0], cd[0, 1], cd[1, 0], cd[1, 1]]
    avm_out.Spatial.Scale = None
    avm_out.Spatial.Rotation = None
    return avm_out


def _avm_matching_pixels(avm, img_shape, flip, transpose):
    """Legacy path: derive the corrected AVM from an existing pyavm AVM.

    WARNING: ``avm.to_wcs()`` is lossy for rotated fields (pyavm from_header
    uses Scale+Rotation), so this is only reliable for near-axis-aligned
    fields.  Prefer passing the true FITS WCS via ``avm_wcs`` to save_rgb, or
    use ``_faithful_flipped_avm`` directly with ``WCS(header)``.
    """
    ny, nx = img_shape[:2]
    try:
        wcs = avm.to_wcs().celestial
    except ValueError:
        wcs = avm.to_wcs(target_shape=(nx, ny)).celestial
    return _faithful_flipped_avm(wcs, img_shape, flip, transpose)


def save_rgb(img, filename, avm=None, avm_wcs=None, flip=-1, alma_data=None,
             alma_level=None,
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

    if avm is not None or avm_wcs is not None:
        # The WCS describes the un-flipped FITS orientation, but the pixels
        # written above are flipped/transposed.  Transform the WCS to match
        # the saved pixels so the embedded AVM (and the HiPS built from it)
        # is correctly oriented.  Prefer a true FITS WCS (avm_wcs) -- deriving
        # it from a pyavm AVM is lossy for rotated fields (see
        # _avm_matching_pixels).
        if avm_wcs is not None:
            from astropy.wcs import WCS
            wcs = avm_wcs if isinstance(avm_wcs, WCS) else WCS(avm_wcs)
            avm_to_embed = _faithful_flipped_avm(wcs.celestial, img.shape,
                                                 flip, transpose)
        else:
            avm_to_embed = _avm_matching_pixels(avm, img.shape, flip, transpose)
        base = os.path.basename(filename)
        dir = os.path.dirname(filename)
        avmname = os.path.join(dir, 'avm_'+base)
        avm_to_embed.embed(filename, avmname)
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
