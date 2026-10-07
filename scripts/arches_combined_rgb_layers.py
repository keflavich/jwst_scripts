#!/usr/bin/env python
"""
Arches cluster RGB + HiPS layers combining archival program 2045 with the
10678 GC Treasury.

Filters covering the Arches (RA 266.4604, Dec -28.8244):

    F212N  2045 o001  arches/F212N/pipeline/...-merged_i2d.fits   (0.031"/px)
    F323N  2045 o001  arches/F323N/pipeline/...-merged_i2d.fits
    F480M  10678 o078 + o079 (the cluster sits between the two fields)
    F770W  10678 o063 + o064 (MIRI)

Every combination of three of the four filters is built, reddest in R, on the
F212N pixel grid.  Treasury tiles of one filter are coadded onto that grid
(mean of the overlaps, no background matching).  Only sky covered by all
three filters is shown.  Stretch: fill_nan for the saturated cores, then a
per-channel asinh stretch over that common sky, pinned so each channel's
median sky maps to the same level (see stretch).

No astrometric correction is applied; the layers show each pipeline's WCS as
is, so any offset between 2045 and 10678 stays visible.  Measured 2026-10-07
from star matches 5-45" from the cluster centre (10678 minus 2045, mas):
F212N +2..+4 RA, -7..-15 Dec; F480M vs F323N +0 RA, -10..-12 Dec, i.e. at
most ~0.5 px on the 0.031"/px grid.

check_orientation validates the PNG -> AVM -> HiPS round trip against this
layer's own pixels; it does not test astrometry between the two programs.

--residual builds the same layers from the star-subtracted (DAOPHOT residual)
mosaics: the cataloguing chain's last stage, resbgsub_m7 for NIRCam and
resbgsub_m6 for MIRI.  A filter without a residual mosaic keeps its image
(F770W o063/o064 have none yet); a layer with every channel star-subtracted
is named `_residual`, a mixed one `_nirresidual`, and the inputs json records
which file fed each channel.  Over-subtracted star cores leave a long negative
tail, so residual channels take their black point at the 5th percentile
rather than the 1st.

Usage:
    arches_combined_rgb_layers.py --list [--residual]
    arches_combined_rgb_layers.py --index I [--residual]   # layer I (SLURM array)
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ARCHES = "/orange/adamginsburg/jwst/arches"
TREASURY = "/orange/adamginsburg/jwst/gc-treasury"
OUT = f"{ARCHES}/combined_rgb"

INPUTS = {
    212: [f"{ARCHES}/F212N/pipeline/"
          "jw02045-o001_t001_nircam_clear-f212n-merged_i2d.fits"],
    323: [f"{ARCHES}/F323N/pipeline/"
          "jw02045-o001_t001_nircam_clear-f323n-merged_i2d.fits"],
    480: [f"{TREASURY}/F480M/pipeline/"
          f"jw10678-o{o}_t001_nircam_clear-f480m-merged_i2d.fits"
          for o in ("078", "079")],
    770: [f"{TREASURY}/F770W/pipeline/jw10678-o{o}_t001_miri_f770w_i2d.fits"
          for o in ("063", "064")],
}
GRID = INPUTS[212][0]

RESIDUAL_STAGE = {"nircam": "resbgsub_m7", "miri": "resbgsub_m6"}


def residual_path(p):
    """The final-stage residual mosaic next to image mosaic `p`, or None."""
    base = os.path.basename(p)
    if "_miri_" in base:
        # jw10678-o063_t001_miri_f770w_i2d.fits ->
        # jw10678-o063_t001_miri_clear-f770w-mirimage_resbgsub_m6_...
        pre, filt = base.split("_miri_")[0], base.split("_miri_")[1][:-9]
        name = (f"{pre}_miri_clear-{filt}-mirimage_{RESIDUAL_STAGE['miri']}"
                "_daophot_basic_mergedcat_residual_i2d.fits")
    else:
        name = base.replace(
            "_i2d.fits", f"_{RESIDUAL_STAGE['nircam']}"
            "_daophot_basic_mergedcat_residual_i2d.fits")
    r = os.path.join(os.path.dirname(p), name)
    return r if os.path.exists(r) else None


def channel_inputs(wave, residual):
    """(paths, is_residual) for one filter."""
    if residual:
        res = [residual_path(p) for p in INPUTS[wave]]
        if all(res):
            return res, True
    return INPUTS[wave], False


# All 3-of-4 combinations, reddest first.
LAYERS = [tuple(sorted(c, reverse=True))
          for c in itertools.combinations(sorted(INPUTS), 3)]


def layer_name(trip, residual=False):
    name = "arches_2045_10678_RGB_{}-{}-{}".format(*trip)
    if residual:
        n = sum(channel_inputs(w, True)[1] for w in trip)
        name += "_residual" if n == 3 else "_nirresidual"
    return name


def load_on(paths, header, shape):
    from reproject import reproject_interp
    from reproject.mosaicking import reproject_and_coadd
    wcs = WCS(header)
    inputs = []
    for p in paths:
        with fits.open(p) as hl:
            inputs.append((hl["SCI"].data.astype(float), WCS(hl["SCI"].header)))
    # The grid file itself, or a residual written on the same pixel grid.
    if (len(inputs) == 1 and inputs[0][0].shape == shape
            and np.allclose(inputs[0][1].wcs.crval, wcs.wcs.crval)
            and np.allclose(inputs[0][1].wcs.crpix, wcs.wcs.crpix)
            and np.allclose(inputs[0][1].pixel_scale_matrix,
                            wcs.pixel_scale_matrix)):
        return inputs[0][0]
    data, foot = reproject_and_coadd(inputs, wcs, shape_out=shape,
                                     reproject_function=reproject_interp,
                                     combine_function="mean")
    data[foot == 0] = np.nan
    return data


def fill(c):
    from jwst_rgb.save_rgb import fill_nan
    # Saturated cores are NaN; fill_nan fills interior islands from their
    # border and leaves edge-touching NaN (outside the footprint) as NaN.
    return fill_nan(c.copy(), bad_data_min_threshold=None)


def stretch(c, common, target=0.15, top_pct=99.9, lo_pct=1):
    """asinh stretch pinned so the median sky lands at `target`.

    The four filters span very different dynamic ranges over the common sky:
    F212N's 99.9th percentile is ~700x its median above the floor, F770W's
    ~13x, because diffuse 7.7 um emission fills the field.  A plain
    percentile stretch saturates F770W into flat red; one shared asinh curve
    leaves it dim and the stars cyan.  Here each channel runs from its 1st to
    its 99.9th percentile and gets its own asinh softening, solved so that the
    median maps to `target`; a channel whose median already sits above
    `target` stays linear.  Percentiles use only the common sky.  `lo_pct`
    is the black point (5 for residuals, see the module docstring).
    """
    from scipy.optimize import brentq
    v = c[common & (c != 0)]
    p1, p50, ptop = np.percentile(v, [lo_pct, 50, top_pct])
    y = np.clip((c - p1) / (ptop - p1), 0, 1)
    m = (p50 - p1) / (ptop - p1)
    if m >= target:
        out = y
    else:
        a = brentq(lambda a: np.arcsinh(m / a) / np.arcsinh(1 / a) - target,
                   1e-8, 1e3)
        out = np.arcsinh(y / a) / np.arcsinh(1 / a)
    out[~common] = np.nan
    return out


def write_check_reference(name, scaled, header):
    """The layer's own channel sum, on its grid, for check_orientation.

    Correlating against a full input mosaic fails once the layer is masked to
    the common sky: for the F770W layers that keeps under half the F212N
    field, and the as-is and rot180 correlations both fell to ~0.09.  The
    channel sum has exactly the layer's footprint and content.
    """
    path = f"{OUT}/{name}/{name}_checkref.fits"
    hdr = WCS(header).to_header()
    fits.PrimaryHDU(data=np.nanmean(scaled, axis=2).astype("float32"),
                    header=hdr).writeto(path, overwrite=True)
    return path


def build(trip, residual=False):
    from jwst_rgb.save_rgb import save_rgb, avm_for_saved_png
    from gc_treasury_rgb_images import check_orientation

    name = layer_name(trip, residual)
    header = fits.getheader(GRID, ext=("SCI", 1))
    shape = (header["NAXIS2"], header["NAXIS1"])
    chans = {w: channel_inputs(w, residual) for w in trip}
    rgb = np.dstack([load_on(chans[w][0], header, shape) for w in trip])
    filled = [fill(rgb[:, :, k]) for k in range(3)]
    # Show only sky all three filters cover: elsewhere one or two channels
    # are empty and the layer paints false single-colour zones.  The NaN
    # outside touches the image edge, so save_rgb makes it transparent.
    common = np.logical_and.reduce([np.isfinite(c) for c in filled])
    scaled = np.dstack([stretch(c, common, lo_pct=5 if chans[w][1] else 1)
                        for c, w in zip(filled, trip)])

    os.makedirs(f"{OUT}/{name}", exist_ok=True)
    png = f"{OUT}/{name}/{name}.png"
    avm = avm_for_saved_png(WCS(header), *shape)
    rgb[~common] = np.nan                    # alpha follows the common sky
    save_rgb(np.nan_to_num(scaled), png, avm=avm, original_data=rgb,
             hips=True, overwrite=True)
    hips = f"{OUT}/{name}/{name}_hips"
    if not os.path.isdir(os.path.join(hips, "Norder3")):
        raise RuntimeError(f"{name}: build produced no Norder3")
    with open(f"{OUT}/{name}/{name}_inputs.json", "w") as fh:
        json.dump({"layer": name, "grid": GRID,
                   "inputs": {str(w): chans[w][0] for w in trip},
                   "star_subtracted": {str(w): chans[w][1] for w in trip}},
                  fh, indent=2)
    ok = check_orientation(hips, write_check_reference(name, scaled, header))
    print(f"done: {hips} orientation_ok={ok}", flush=True)
    if ok is not True:
        raise RuntimeError(f"{name}: orientation/astrometry check "
                           f"{'failed' if ok is False else 'was inconclusive'}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true")
    g.add_argument("--index", type=int, help="layer by position")
    ap.add_argument("--residual", action="store_true",
                    help="build from the star-subtracted mosaics")
    args = ap.parse_args()
    if args.list:
        for w in INPUTS:
            paths, res = channel_inputs(w, args.residual)
            for p in paths:
                print(f"F{w}: residual={res} exists={os.path.exists(p)} {p}")
        for i, trip in enumerate(LAYERS):
            print(f"layer {i}: {layer_name(trip, args.residual)}")
    else:
        build(LAYERS[args.index], args.residual)


if __name__ == "__main__":
    main()
