#!/usr/bin/env python
"""
RGB HiPS for every adjacent three-filter combination of a field, for the
linear wavelength tours (ACES_Aladin_tour/<field>_wavelength_tour_linear.html).

This is wd2_adjacent_rgb_layers.py generalized to other fields.  The tour
steps through consecutive filter triplets: R/G/B = three neighbouring
filters in wavelength.  Inputs are the regular (not star-subtracted)
mosaics, the astrometry-corrected pipeline mosaic wherever one exists.

Each layer is put on the pixel grid of its bluest filter, the finest one in
the triplet.  NaN islands inside the field (saturated cores) are filled on the
native grid with jwst_rgb.fill_nan before reprojection; NaNs touching the
field edge are left alone and become transparent.  Each channel gets a
percentile asinh stretch.  Every build ends with check_orientation
(gc_treasury_rgb_images), which correlates served tiles against the grid FITS
for a 180 degree flip and measures the translation (tolerance 0.3"); a layer
that fails or cannot be checked raises.

Usage:
    adjacent_rgb_layers.py FIELD --list             # resolve inputs only
    adjacent_rgb_layers.py FIELD --index I          # build triplet I (array)
    adjacent_rgb_layers.py FIELD --publish          # copy built HiPS to avm_images
    adjacent_rgb_layers.py FIELD --waypoints OUT    # write the tour's waypoints
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

JWST = "/orange/adamginsburg/jwst"
PUBLISHED = "/orange/adamginsburg/web/public/avm_images"


def _pipe(field, filt, stem):
    return f"{JWST}/{field}/{filt}/pipeline/{stem}"


# Per field: layer-name prefix (as in the published layers), tour title, and
# wavelength code -> mosaic path.  Codes are the filter number as it appears
# in the layer names (F212N -> 212, F1500W -> 1500).
FIELDS = {
    "sgra": {
        "prefix": "SgrA",
        "title": "Sgr A* Wavelength Explorer (Linear)",
        "mosaics": {
            115: _pipe("sgra", "F115W",
                       "jw01939-o001_t001_nircam_clear-f115w-merged_i2d.fits"),
            212: _pipe("sgra", "F212N",
                       "jw01939-o001_t001_nircam_clear-f212n-merged_i2d.fits"),
            # F323N has no pipeline mosaic.  The MAST i2d sits ~15" off
            # (the uncorrected 1939 frame), so it is left out.
            405: _pipe("sgra", "F405N",
                       "jw01939-o001_t001_nircam_clear-f405n-merged_i2d.fits"),
            # MIRI from GC program 3571, observation 1 (the Sgr A* pointing).
            560: _pipe("gc3571", "F560W", "jw03571-o001_t001_miri_f560w_i2d.fits"),
            770: _pipe("gc3571", "F770W", "jw03571-o001_t001_miri_f770w_i2d.fits"),
            1000: _pipe("gc3571", "F1000W",
                        "jw03571-o001_t001_miri_f1000w_i2d.fits"),
            1280: _pipe("gc3571", "F1280W",
                        "jw03571-o001_t001_miri_f1280w_i2d.fits"),
            1500: _pipe("gc3571", "F1500W",
                        "jw03571-o001_t001_miri_f1500w_i2d.fits"),
        },
    },
    "sgrc": {
        "prefix": "SGRC",
        "title": "Sgr C Wavelength Explorer (Linear)",
        "mosaics": {
            w: _pipe("sgrc", f"F{w}{s}",
                     f"jw04147-o012_t001_nircam_clear-f{w}{s.lower()}"
                     "-merged_i2d.fits")
            for w, s in [(115, "W"), (162, "M"), (182, "M"), (212, "N"),
                         (360, "M"), (405, "N"), (470, "N"), (480, "M")]
        },
    },
    "cloudef": {
        "prefix": "Cloudef",
        "title": "Clouds E/F Wavelength Explorer (Linear)",
        "mosaics": {
            **{w: _pipe("cloudef", f"F{w}M",
                        f"jw02092-o002_t001_nircam_clear-f{w}m-merged_i2d.fits")
               for w in (162, 210, 360, 480)},
            770: _pipe("cloudef", "F770W",
                       "jw02092-o004_t001_miri_f770w_i2d.fits"),
            2100: _pipe("cloudef", "F2100W",
                        "jw02092-o004_t001_miri_f2100w_i2d.fits"),
        },
    },
}


def triplets(field):
    """Consecutive triplets, R (reddest) first."""
    waves = sorted(FIELDS[field]["mosaics"])
    return [tuple(reversed(waves[i:i + 3])) for i in range(len(waves) - 2)]


def mosaic(field, wave):
    path = FIELDS[field]["mosaics"][wave]
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def layer_name(field, trip):
    return "{}_RGB_{}-{}-{}".format(FIELDS[field]["prefix"], *trip)


def outdir(field):
    return f"{JWST}/{field}/adjacent_rgb"


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


def build(field, trip):
    from jwst_rgb.save_rgb import save_rgb, avm_for_saved_png
    from gc_treasury_rgb_images import check_orientation

    name = layer_name(field, trip)
    paths = {w: mosaic(field, w) for w in trip}
    for w in trip:
        print(f"  {w}: {paths[w]}", flush=True)
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

    os.makedirs(f"{outdir(field)}/{name}", exist_ok=True)
    png = f"{outdir(field)}/{name}/{name}.png"
    avm = avm_for_saved_png(WCS(header), *shape)
    save_rgb(np.nan_to_num(scaled), png, avm=avm, original_data=rgb,
             hips=True, overwrite=True)
    hips = f"{outdir(field)}/{name}/{name}_hips"
    if not os.path.isdir(os.path.join(hips, "Norder3")):
        raise RuntimeError(f"{name}: build produced no Norder3")
    ok = check_orientation(hips, grid)
    print(f"done: {hips} orientation_ok={ok}", flush=True)
    # None means the check could not decide (too little overlap or signal);
    # a layer that was never verified is not published.
    if ok is not True:
        raise RuntimeError(f"{name}: orientation/astrometry check failed")
    with open(os.path.join(hips, "ORIENTATION_OK"), "w") as fh:
        fh.write(f"check_orientation against {grid}\n")


def publish(field):
    """Copy every verified layer into avm_images.  An existing layer of the
    same name is left alone."""
    for trip in triplets(field):
        name = layer_name(field, trip)
        src = f"{outdir(field)}/{name}/{name}_hips"
        dst = f"{PUBLISHED}/{name}_hips"
        if not os.path.exists(os.path.join(src, "ORIENTATION_OK")):
            print(f"skip {name}: not built or not verified")
        elif os.path.exists(dst):
            print(f"skip {name}: {dst} exists")
        else:
            shutil.copytree(src, dst + ".new")
            os.rename(dst + ".new", dst)
            print(f"published {dst}")


def waypoints(field, out):
    steps = []
    for trip in sorted(triplets(field), key=lambda t: t[0]):
        desc = " / ".join(label(w) for w in trip)
        steps.append({"wavelength": trip[0],
                      "url": layer_name(field, trip) + "_hips",
                      "label": f"RGB: {desc}", "description": desc})
    # Center and field of view from the first layer's HiPS properties.
    first = layer_name(field, sorted(triplets(field))[0])
    props = {}
    with open(f"{PUBLISHED}/{first}_hips/properties") as fh:
        for ln in fh:
            if "=" in ln:
                k, v = ln.split("=", 1)
                props[k.strip()] = v.strip()
    fov = round(float(props.get("hips_initial_fov", 0.1)), 3)
    wp = {"waypoints": [{
        "ra": float(props["hips_initial_ra"]),
        "dec": float(props["hips_initial_dec"]),
        "fov": fov, "transition_fov": fov, "transition_time": 3,
        "zoom_out_time": 2, "zoom_in_time": 3,
        "title": FIELDS[field]["title"],
        "description": " → ".join(s["description"] for s in steps),
        "pause_time": 300000,
        "wavelength_slider": {"enabled": True, "wavelengths": steps},
    }]}
    with open(out, "w") as fh:
        json.dump(wp, fh, indent=4, ensure_ascii=False)
        fh.write("\n")
    print(f"wrote {out}: {len(steps)} steps, {steps[0]['description']} → "
          f"{steps[-1]['description']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("field", choices=sorted(FIELDS))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true")
    g.add_argument("--index", type=int, help="triplet by position (SLURM array)")
    g.add_argument("--publish", action="store_true")
    g.add_argument("--waypoints", metavar="OUT")
    args = ap.parse_args()
    if args.list:
        for i, trip in enumerate(triplets(args.field)):
            print(i, layer_name(args.field, trip))
            for w in trip:
                print(f"    {w}: {mosaic(args.field, w)}")
    elif args.publish:
        publish(args.field)
    elif args.waypoints:
        waypoints(args.field, args.waypoints)
    else:
        trip = triplets(args.field)[args.index]
        print(f"building {layer_name(args.field, trip)}", flush=True)
        build(args.field, trip)


if __name__ == "__main__":
    main()
