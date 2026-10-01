#!/usr/bin/env python
"""
Westerlund 2: RGB HiPS for every adjacent three-filter combination, plus
re-embedded copies of the published wd2 layers with a correct AVM.

Adjacent triplets
-----------------
The wavelength tour (ACES_Aladin_tour/wd2_wavelength_tour_linear.html) steps
through consecutive filter triplets, R/G/B = three neighbouring filters in
wavelength.  Inputs are the regular (not star-subtracted) mosaics:

    NIRCam: <FILTER>/pipeline/jw03523-o005_t001_nircam_clear-<filter>-merged_i2d.fits
    MIRI:   miri_<FILTER>_pid3523_combined_SF_i2d.fits   (top of the wd2 dir)

Each layer is put on the pixel grid of its bluest filter, the finest one in
the triplet.  NaN islands inside the field (saturated cores) are filled on the
native grid with jwst_rgb.fill_nan before reprojection; NaNs touching the
field edge are left alone and become transparent.  Each channel gets a
percentile asinh stretch.

Published-layer fix
-------------------
The six wd2 HiPS in avm_images (built 2025-07) embed a raw
``pyavm.AVM.from_header`` of the target grid.  save_rgb writes the pixels with
flip=-1 + ROTATE_180, so that AVM keeps the FITS CRPIX where the reflected one
is needed (see jwst_rgb.save_rgb.avm_for_saved_png).  ``--fix I`` copies the
published PNG pixels unchanged, embeds avm_for_saved_png(WCS(grid header)),
and rebuilds the HiPS.  It adds no shift: the AVM is derived from the FITS
WCS only.

Every build ends with check_orientation (gc_treasury_rgb_images), which
correlates served tiles against a source FITS for a 180 degree flip and
measures the translation (tolerance 0.3").

Usage:
    wd2_adjacent_rgb_layers.py --list            # resolve inputs, build nothing
    wd2_adjacent_rgb_layers.py --index I         # build triplet I (SLURM array)
    wd2_adjacent_rgb_layers.py --fix-list        # list the published layers
    wd2_adjacent_rgb_layers.py --fix I           # re-embed published layer I
    wd2_adjacent_rgb_layers.py --waypoints OUT   # write the tour's waypoints json
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
from astropy.io import fits
from astropy.visualization import simple_norm
from astropy.wcs import WCS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WD2 = "/orange/adamginsburg/jwst/wd2"
OUT = f"{WD2}/adjacent_rgb"
PUBLISHED = "/orange/adamginsburg/web/public/avm_images"

# Wavelength code (as in the layer names) -> filter name.  Every wd2 (o005)
# filter that has a mosaic.  F444W has no wd2 mosaic of its own (it was the
# F405N pupil partner), so it is absent.
NIRCAM = {115: "F115W", 150: "F150W", 162: "F162M", 164: "F164N",
          182: "F182M", 187: "F187N", 200: "F200W", 212: "F212N",
          250: "F250M", 277: "F277W", 300: "F300M", 323: "F323N",
          335: "F335M", 405: "F405N", 410: "F410M", 466: "F466N"}
MIRI = {770: "F770W", 1000: "F1000W", 1130: "F1130W"}
FILTER = {**NIRCAM, **MIRI}
FILTERS = sorted(FILTER)

# Consecutive triplets, R (reddest) first.
TRIPLETS = [tuple(reversed(FILTERS[i:i + 3])) for i in range(len(FILTERS) - 2)]


def mosaic(wave):
    f = FILTER[wave]
    if wave in MIRI:
        path = f"{WD2}/miri_{f}_pid3523_combined_SF_i2d.fits"
    else:
        path = (f"{WD2}/{f}/pipeline/jw03523-o005_t001_nircam_clear-"
                f"{f.lower()}-merged_i2d.fits")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def layer_name(trip):
    return "wd2_RGB_{}-{}-{}".format(*trip)


def label(wave):
    return f"{wave / 100:.2f}μm" if wave < 1000 else f"{wave / 100:.1f}μm"


def load_filled_on(path, header, shape):
    """SCI data with interior NaN islands filled, on the target grid."""
    from reproject import reproject_interp
    from jwst_rgb.save_rgb import fill_nan
    with fits.open(path) as hl:
        data = hl["SCI"].data.astype(np.float32)
        hdr = hl["SCI"].header
    # Only NaNs: the default threshold would also treat every negative
    # noise pixel as a hole.
    data = fill_nan(data, bad_data_min_threshold=None)
    if data.shape == shape and WCS(hdr).celestial.wcs.compare(
            WCS(header).celestial.wcs, tolerance=1e-9):
        return data
    out, _ = reproject_interp((data, WCS(hdr).celestial),
                              WCS(header).celestial, shape_out=shape)
    return out.astype(np.float32)


def build(trip):
    from jwst_rgb.save_rgb import save_rgb, avm_for_saved_png
    from gc_treasury_rgb_images import check_orientation

    name = layer_name(trip)
    paths = {w: mosaic(w) for w in trip}
    for w in trip:
        print(f"  {FILTER[w]}: {paths[w]}", flush=True)
    grid = paths[min(trip)]                  # bluest = finest pixels
    header = fits.getheader(grid, ext=("SCI", 1))
    shape = (header["NAXIS2"], header["NAXIS1"])
    rgb = np.dstack([load_filled_on(paths[w], header, shape) for w in trip])

    def stretch(c):
        good = np.isfinite(c) & (c != 0)
        norm = simple_norm(c[good], stretch="asinh", min_percent=1,
                           max_percent=99.5)
        return norm(c).filled(np.nan)

    scaled = np.dstack([stretch(rgb[:, :, k]) for k in range(3)])

    os.makedirs(f"{OUT}/{name}", exist_ok=True)
    png = f"{OUT}/{name}/{name}.png"
    avm = avm_for_saved_png(WCS(header), *shape)
    save_rgb(np.nan_to_num(scaled), png, avm=avm, original_data=rgb,
             hips=True, overwrite=True)
    hips = f"{OUT}/{name}/{name}_hips"
    if not os.path.isdir(os.path.join(hips, "Norder3")):
        raise RuntimeError(f"{name}: build produced no Norder3")
    ok = check_orientation(hips, grid)
    print(f"done: {hips} orientation_ok={ok}", flush=True)
    if ok is False:
        raise RuntimeError(f"{name}: orientation/astrometry check failed")


# Published layers: (png basename in avm_images, grid FITS the PNG was laid
# on, matching-band FITS for the tile check).  The NIRCam/mixed layers sit on
# the 2024 F250M mosaic's grid (4869x2108, identical CRPIX/CRVAL to the
# embedded AVM); the MIRI ones on the F770W mosaic's grid.
_F250M = f"{WD2}/wd2_F250M_AB_i2d.fits"
_F770W = f"{WD2}/miri_F770W_pid3523_combined_SF_i2d.fits"
FIXES = [
    ("wd2_miri_RGB_1130-1000-770_log_max99.9_transparent.png", _F770W, _F770W),
    ("wd2_miri_RGB_1130-1000-770_log_transparent.png", _F770W, _F770W),
    ("wd2_nircam_RGB_212-200-187_asinh_max99_transparent.png", _F250M,
     f"{WD2}/wd2_F200W_AB_i2d.fits"),
    ("wd2_nircam_RGB_410-405-335_asinh_max99.5_transparent.png", _F250M,
     f"{WD2}/wd2_F410M_AB_i2d.fits"),
    ("wd2_RGB_1000-770-410_asinh_max99.5_transparent.png", _F250M, _F770W),
    ("wd2_RGB_1130-770-164162_sub_asinh_max99.5_transparent.png", _F250M,
     f"{WD2}/miri_F1130W_pid3523_combined_SF_i2d.fits"),
]


def fix(index):
    """Copy a published PNG, embed the AVM for save_rgb's pixel path, rebuild
    its HiPS, and check it."""
    from PIL import Image
    from reproject import reproject_interp
    from reproject.hips import reproject_to_hips
    from tqdm import tqdm
    from jwst_rgb.save_rgb import avm_for_saved_png
    from jwst_rgb.hips_naming import properties_for
    from gc_treasury_rgb_images import check_orientation

    Image.MAX_IMAGE_PIXELS = None
    base, grid, ref = FIXES[index]
    header = fits.getheader(grid, ext=("SCI", 1))
    shape = (header["NAXIS2"], header["NAXIS1"])
    src = os.path.join(PUBLISHED, base)
    with Image.open(src) as im:
        if (im.size[1], im.size[0]) != shape:
            raise ValueError(f"{base}: PNG is {im.size}, grid {grid} is "
                             f"{shape[::-1]}")
    outdir = f"{OUT}/fixed_published"
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, base)
    tmp = os.path.join(outdir, "avm_" + base)
    # save_rgb's defaults (flip=-1, ROTATE_180), which wd2_rgb_images.py used.
    avm_for_saved_png(WCS(header), *shape).embed(src, tmp)
    shutil.move(tmp, png)
    hips = png.replace(".png", "_hips")
    if os.path.exists(hips):
        shutil.rmtree(hips)
    reproject_to_hips(png, level=None, reproject_function=reproject_interp,
                      output_directory=hips, threads=8,
                      coord_system_out="galactic",
                      properties=properties_for(hips), progress_bar=tqdm)
    ok = check_orientation(hips, ref)
    print(f"done: {hips} orientation_ok={ok}", flush=True)
    if ok is False:
        raise RuntimeError(f"{base}: orientation/astrometry check failed")


def waypoints(out):
    steps = []
    for trip in sorted(TRIPLETS, key=lambda t: t[0]):
        desc = " / ".join(label(w) for w in trip)
        steps.append({"wavelength": trip[0], "url": layer_name(trip) + "_hips",
                      "label": f"RGB: {desc}", "description": desc})
    first, last = steps[0]["description"], steps[-1]["description"]
    wp = {"waypoints": [{
        "ra": 155.992083, "dec": -57.7636111, "fov": 0.1,
        "transition_fov": 0.1, "transition_time": 3,
        "zoom_out_time": 2, "zoom_in_time": 3,
        "title": "Westerlund 2 Wavelength Explorer (Linear)",
        "description": " → ".join(s["description"] for s in steps),
        "pause_time": 300000,
        "wavelength_slider": {"enabled": True, "wavelengths": steps},
    }]}
    with open(out, "w") as fh:
        json.dump(wp, fh, indent=4, ensure_ascii=False)
        fh.write("\n")
    print(f"wrote {out}: {len(steps)} steps, {first} → {last}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true")
    g.add_argument("--index", type=int, help="triplet by position (SLURM array)")
    g.add_argument("--fix-list", action="store_true")
    g.add_argument("--fix", type=int, help="published layer by position")
    g.add_argument("--waypoints", metavar="OUT")
    args = ap.parse_args()
    if args.list:
        for i, trip in enumerate(TRIPLETS):
            print(i, layer_name(trip))
            for w in trip:
                print(f"    {FILTER[w]}: {mosaic(w)}")
    elif args.fix_list:
        for i, (base, grid, ref) in enumerate(FIXES):
            print(i, base, os.path.basename(grid), os.path.basename(ref))
    elif args.fix is not None:
        fix(args.fix)
    elif args.waypoints:
        waypoints(args.waypoints)
    else:
        trip = TRIPLETS[args.index]
        print(f"building {layer_name(trip)}", flush=True)
        build(trip)


if __name__ == "__main__":
    main()
