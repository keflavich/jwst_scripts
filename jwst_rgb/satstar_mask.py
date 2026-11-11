"""Display-only mask of the accepted-satstar cores in a star-subtracted mosaic.

The DAOPHOT residual mosaics of the 10678 chain carry a blue donut or a
yellow-red blob at every ACCEPTED saturated star: the model subtracts the
unsaturated wings correctly but over- or under-subtracts the core.  The
catalogue position is right (the strong residual sits ~0.08" from it); it is
the core model that is wrong, so the residual FITS has the feature and a
mis-centring fix would not remove it.

`mask_satstar_cores` returns a COPY in which every pixel inside a disk around
an accepted satstar is replaced by the local background.  Use it only on the
array that feeds a rendered product.  The ``*_mergedcat_residual_i2d.fits``
file seeds the next cataloguing phase's detection, so it must never be
masked or rewritten, and nothing here writes a file.

Rule, measured on o102 F480M (0.0629 arcsec/px) and re-checked on o109/o083:

    r_core = sqrt(sat_area / pi)             [px]
    r_mask = clip(r_core + PAD, RMIN, RMAX)  [px]

in pixels of the DETECTOR frames the catalogue was made on: sat_area counts the
saturated pixels of a per-exposure frame (NIRCam SW 0.031, LW 0.063 arcsec/px),
and DETECTOR_PIX_ARCSEC converts the radius to the pixel scale of the array it
is applied to.  The numbers were measured on F480M.  F212N takes them in its own
finer detector pixels, so its disks are half the angular size of F480M's; held
at F480M's angular size they would blank about 3.5 times as much of an F212N
mosaic (18-30 % of the finite pixels against 5-9 %), whose catalogue has about
five times the rows.

Fill: the 64-px block median of the image with every disk excluded,
bilinearly interpolated, evaluated only where it is used.  NaN is not an
option for the disks: save_rgb keeps interior NaN islands opaque (the
`alpha_only_edges` logic is load-bearing), so they would render as black
holes.  For the same reason a NaN core that lies wholly inside a disk
(saturated pixels the residual leaves as NaN) is filled too; NaN that
touches the outer footprint is never touched.
"""
import os
import warnings

import numpy as np
from astropy.table import Table
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy import ndimage
from scipy.spatial import cKDTree

CATALOG_DIR = "/orange/adamginsburg/jwst/gc-treasury/catalogs"
REF_PIX_ARCSEC = 0.0629091    # F480M i2d scale the radius rule was measured at
# Pixel scale of the detector frames each filter's sat_area was counted on; the
# i2d mosaics of the chain keep it (F212N 0.03122, F480M 0.06291 arcsec/px).
DETECTOR_PIX_ARCSEC = {"f480m": REF_PIX_ARCSEC, "f212n": 0.031223}
PAD_PX, RMIN_PX, RMAX_PX = 8.0, 12.0, 30.0
DEDUPE_ARCSEC = 0.15
BLOCK_PX = 64                 # 4.0 arcsec at the F480M scale
# a block with fewer finite pixels takes its nearest valid neighbour
MIN_BLOCK_FRAC = 0.25


def satstar_catalog_path(obs, filt="f480m", catalog_dir=CATALOG_DIR):
    """The per-observation consolidated satstar catalogue.

    One file per observation, not the per-field ``*_dao_basic`` catalogues:
    those once carried the whole survey's saturated-star list (overlays were
    duplicated up to 43x), the consolidated ones do not.  A module tag
    (o135_nrca) maps to the observation's file.
    """
    return f"{catalog_dir}/{filt}_{obs.split('_')[0]}_consolidated_satstar_catalog.fits"


def pixel_scale_arcsec(wcs):
    return float(proj_plane_pixel_scales(wcs)[0] * 3600.0)


def detector_pix_arcsec(filt):
    """Scale of the detector pixels `sat_area` counts for `filt`; no guess otherwise."""
    try:
        return DETECTOR_PIX_ARCSEC[filt]
    except KeyError:
        raise ValueError(
            f"no detector pixel scale for filter {filt!r}: the satstar display "
            f"mask is defined for {', '.join(sorted(DETECTOR_PIX_ARCSEC))}") from None


def load_satstars(obs, wcs, shape, filt="f480m", catalog_dir=CATALOG_DIR):
    """Accepted satstars of one observation, in the pixel frame of (wcs, shape).

    Guards, each counted in the returned info so an upstream change is visible:
      * rows with a non-finite sky position or sat_area <= 0 are dropped;
      * rows farther outside the image than the largest disk are dropped;
      * rows within DEDUPE_ARCSEC of a larger one are dropped (largest
        sat_area wins), so a repeated row cannot widen a disk's reach.
    """
    det_pix_as = detector_pix_arcsec(filt)
    path = satstar_catalog_path(obs, filt, catalog_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{obs}: no consolidated satstar catalogue {path}")
    ny, nx = shape
    pix_as = pixel_scale_arcsec(wcs)
    t = Table.read(path)
    sc = t["skycoord_fit"]
    x, y = wcs.world_to_pixel_values(sc.ra.deg, sc.dec.deg)
    x, y = np.asarray(x, float), np.asarray(y, float)
    area = np.asarray(t["sat_area"], float)
    margin = RMAX_PX * det_pix_as / pix_as
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(area) & (area > 0)
    inside = (valid & (x > -margin) & (x < nx - 1 + margin)
              & (y > -margin) & (y < ny - 1 + margin))
    idx = np.where(inside)[0]
    order = idx[np.argsort(-area[idx])]
    xo, yo = x[order], y[order]
    keep = np.ones(len(order), bool)
    if len(order):
        tree = cKDTree(np.c_[xo, yo])
        for k in range(len(order)):
            if not keep[k]:
                continue
            for j in tree.query_ball_point([xo[k], yo[k]], r=DEDUPE_ARCSEC / pix_as):
                if j != k:
                    keep[j] = False
    sel = order[keep]
    info = dict(catalog=path, n_rows=int(len(t)), n_valid=int(valid.sum()),
                n_in_image=int(inside.sum()), n_dupes=int((~keep).sum()),
                n_used=int(len(sel)))
    return dict(x=x[sel], y=y[sel], sat_area=area[sel], info=info)


def mask_radius_px(sat_area, pix_as, det_pix_as=REF_PIX_ARCSEC):
    """Disk radius, in pixels of an array with scale `pix_as`, for each star.

    `det_pix_as` is the scale of the detector pixels `sat_area` counts, which
    is also the unit PAD_PX, RMIN_PX and RMAX_PX are applied in.
    """
    r_core = np.sqrt(np.asarray(sat_area, float) / np.pi)
    return np.clip(r_core + PAD_PX, RMIN_PX, RMAX_PX) * (det_pix_as / pix_as)


def disk_mask(shape, x, y, r):
    """Union of disks (centre x, y; radius r; all in pixels) as a boolean image."""
    ny, nx = shape
    m = np.zeros(shape, bool)
    for xi, yi, ri in zip(x, y, r):
        xa, xb = max(0, int(np.floor(xi - ri)) - 1), min(nx, int(np.ceil(xi + ri)) + 2)
        ya, yb = max(0, int(np.floor(yi - ri)) - 1), min(ny, int(np.ceil(yi + ri)) + 2)
        if xb <= xa or yb <= ya:
            continue
        yy, xx = np.ogrid[ya:yb, xa:xb]
        m[ya:yb, xa:xb] |= (xx - xi) ** 2 + (yy - yi) ** 2 <= ri * ri
    return m


def block_median_blocks(img, blk=BLOCK_PX, min_frac=MIN_BLOCK_FRAC):
    """Per-block NaN-aware median; sparse blocks take their nearest valid block."""
    ny, nx = img.shape
    by, bx = -(-ny // blk), -(-nx // blk)
    pad = np.full((by * blk, bx * blk), np.nan, np.float32)
    pad[:ny, :nx] = img
    cube = (pad.reshape(by, blk, bx, blk).transpose(0, 2, 1, 3)
            .reshape(by, bx, blk * blk))
    del pad
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        blocks = np.nanmedian(cube, axis=-1)
    nfin = np.isfinite(cube).sum(axis=-1)
    bad = (nfin < min_frac * blk * blk) | ~np.isfinite(blocks)
    if bad.all():
        raise ValueError("no block of the image has enough finite pixels "
                         "for a background")
    if bad.any():
        idx = ndimage.distance_transform_edt(bad, return_distances=False,
                                             return_indices=True)
        blocks = blocks[tuple(idx)]
    return blocks.astype(np.float64)


def eval_blocks(blocks, yy, xx, blk=BLOCK_PX):
    """Bilinear value of the block field at integer pixel coordinates; edges clamp."""
    u = (np.asarray(yy, float) + 0.5) / blk - 0.5
    v = (np.asarray(xx, float) + 0.5) / blk - 0.5
    return ndimage.map_coordinates(blocks, [u, v], order=1, mode="nearest")


def interior_nan_islands(nan, disks):
    """NaN pixels of every NaN island that lies ENTIRELY inside `disks`."""
    lab, n = ndimage.label(nan, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros(nan.shape, bool)
    idx = np.arange(1, n + 1)
    inside = ndimage.minimum(disks, lab, idx) == 1
    return np.isin(lab, idx[inside])


def mask_satstar_cores(img, wcs, obs, filt="f480m", catalog_dir=CATALOG_DIR,
                       stars=None):
    """(masked copy of `img`, boolean mask of the replaced pixels, info).

    `img` itself is never modified.  Finite pixels inside a disk are replaced
    by the local background; a NaN island wholly inside the disks is filled
    as well, and the outer NaN footprint is left alone.  `info["n_at_cap"]` is
    the number of stars whose disk is held to RMAX_PX.
    """
    det_pix_as = detector_pix_arcsec(filt)
    pix_as = pixel_scale_arcsec(wcs)
    if stars is None:
        stars = load_satstars(obs, wcs, img.shape, filt, catalog_dir)
    r = mask_radius_px(stars["sat_area"], pix_as, det_pix_as)
    disks = disk_mask(img.shape, stars["x"], stars["y"], r)
    fin = np.isfinite(img)
    m_fin = disks & fin
    m_nan = interior_nan_islands(~fin, disks)
    m = m_fin | m_nan
    out = np.array(img, copy=True)
    if m.any():
        blocks = block_median_blocks(np.where(m, np.nan, img))
        yy, xx = np.nonzero(m)
        out[yy, xx] = eval_blocks(blocks, yy, xx).astype(out.dtype)
    info = dict(stars["info"])
    at_cap = r >= RMAX_PX * det_pix_as / pix_as * (1 - 1e-9)
    info.update(pix_as=pix_as, n_masked_px=int(m_fin.sum()),
                n_at_cap=int(at_cap.sum()),
                n_nan_core_px_filled=int(m_nan.sum()), n_finite_px=int(fin.sum()),
                frac_of_finite=float(m_fin.sum() / max(1, fin.sum())),
                r_mask_px_median=float(np.median(r)) if len(r) else float("nan"),
                r_mask_px_max=float(r.max()) if len(r) else float("nan"))
    return out, m, info
