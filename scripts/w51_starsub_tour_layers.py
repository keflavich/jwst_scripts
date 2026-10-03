#!/usr/bin/env python
"""
Star-subtracted RGB layers for the W51 linear wavelength tour.

The tour (ACES_Aladin_tour/w51_wavelength_tour_linear.html) steps through
consecutive filter triplets, R/G/B = three neighbouring filters in wavelength.
This builds the same sequence from the pipeline's FINAL DAOPHOT residual
mosaics:

    <FILTER>/pipeline/<prefix>-<filter>-<module>..._m<N>_daophot_basic_mergedcat_residual_i2d.fits

with the highest N per filter (resbgsub_m7 for NIRCam, resbgsub_m6 for MIRI as
of 2026-10).  F560W has per-exposure residuals only, no mosaic, so it is left
out and the triplets are formed from the remaining 14 filters (12 steps
instead of the image tour's 13).

Each layer is put on the pixel grid of its bluest filter, the finest one in
the triplet.  Stretch and hole filling follow starsub_rgb_layers.py: the
over-subtracted star cores are replaced by the local median, then each channel
gets a percentile asinh stretch (5th to 99.5th).  The image tour's per-layer
stretches were tuned on images with stars in them and do not carry over.

Usage:
    w51_starsub_tour_layers.py --list          # resolve inputs, build nothing
    w51_starsub_tour_layers.py --index I       # build triplet I (SLURM array)
    w51_starsub_tour_layers.py --waypoints OUT # write the tour's waypoints json
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
from astropy.io import fits
from astropy.visualization import simple_norm
from astropy.wcs import WCS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from starsub_rgb_layers import fill_holes, load_on  # noqa: E402

W51 = "/orange/adamginsburg/jwst/w51"
OUT = f"{W51}/starsub_rgb"

# Wavelength code (as in the layer names) -> filter directory.
FILTERS = [140, 162, 182, 187, 210, 335, 360, 405, 410, 480,
           770, 1000, 1280, 2100]
FILTER_DIR = {w: (f"F{w}W" if w >= 560 else
                  f"F{w}N" if w in (187, 405) else f"F{w}M")
              for w in FILTERS}

# Consecutive triplets, blue end first; R is the reddest.
TRIPLETS = [tuple(reversed(FILTERS[i:i + 3])) for i in range(len(FILTERS) - 2)]

RESID_RE = re.compile(r"-(?:merged|mirimage)(?:_resbgsub)?_"
                      r"(?:resbgsub_)?m(\d+)"
                      r"_daophot_basic_mergedcat_residual_i2d\.fits$")


def layer_name(trip):
    return "w51_starsub_RGB_{}-{}-{}".format(*trip)


def final_residual(wave):
    """Path of the highest-iteration residual mosaic for one filter."""
    d = f"{W51}/{FILTER_DIR[wave]}/pipeline"
    found = []
    for path in glob.glob(f"{d}/*residual_i2d.fits"):
        if "smoothed" in path or "badastrom" in path:
            continue
        m = RESID_RE.search(os.path.basename(path))
        if m:
            found.append((int(m.group(1)), path))
    if not found:
        raise FileNotFoundError(f"no residual mosaic for {FILTER_DIR[wave]} in {d}")
    # F770W/F1280W/F2100W also have `_group_` variants (`_resbgsub_group_m6`,
    # 2026-07) beside the per-exposure `_resbgsub_m6` (2026-09).  RESID_RE
    # excludes them, so the choice does not depend on file mtimes.
    if len({n for n, _ in found}) != len(found):
        raise RuntimeError(f"ambiguous residual mosaics in {d}: {found}")
    return max(found)[1]


def label(wave):
    return f"{wave / 100:.2f}μm" if wave < 1000 else f"{wave / 100:.1f}μm"


def write_inputs(trip, paths, grid):
    """Record the residual mosaics a layer was built from, next to its PNG."""
    name = layer_name(trip)
    out = f"{OUT}/{name}/{name}_inputs.json"
    with open(out, "w") as fh:
        json.dump({"layer": name, "grid": grid,
                   "inputs": {FILTER_DIR[w]: paths[w] for w in trip}},
                  fh, indent=2)
    return out


def build(trip):
    from jwst_rgb.save_rgb import save_rgb, avm_for_saved_png, fill_nan
    from gc_treasury_rgb_images import check_orientation

    name = layer_name(trip)
    paths = {w: final_residual(w) for w in trip}
    for w in trip:
        print(f"  {FILTER_DIR[w]}: {paths[w]}", flush=True)
    grid = paths[min(trip)]                  # bluest = finest pixels
    header = fits.getheader(grid, ext=("SCI", 1))
    shape = (header["NAXIS2"], header["NAXIS1"])
    rgb = np.dstack([load_on(paths[w], header, shape) for w in trip])

    def stretch(c):
        # Saturated cores are NaN in the residual mosaics, and the saturated
        # area grows with wavelength (F2100W > F1280W > F1000W), so unfilled
        # they render as nested cyan/blue/black rings.  fill_nan fills the
        # interior islands from their border and leaves edge-touching ones
        # NaN, which save_rgb makes transparent.
        c = fill_nan(c.copy(), bad_data_min_threshold=None)
        c = fill_holes(c)
        good = np.isfinite(c) & (c != 0)
        norm = simple_norm(c[good], stretch="asinh", min_percent=5,
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
    write_inputs(trip, paths, grid)
    ok = check_orientation(hips, grid)
    print(f"done: {hips} orientation_ok={ok}", flush=True)
    # None means the check could not decide (too little overlap or signal);
    # a layer that was never verified is not published.
    if ok is not True:
        raise RuntimeError(f"{name}: orientation/astrometry check "
                           f"{'failed' if ok is False else 'was inconclusive'}")


def waypoints(out):
    steps = []
    for trip in sorted(TRIPLETS, key=lambda t: t[0]):
        desc = " / ".join(label(w) for w in trip)
        steps.append({"wavelength": trip[0], "url": layer_name(trip) + "_hips",
                      "label": f"RGB: {desc}", "description": desc})
    first, last = steps[0]["description"], steps[-1]["description"]
    wp = {"waypoints": [{
        "title": "W51 Wavelength Explorer, stars subtracted (Linear)",
        "description": " → ".join(s["description"] for s in steps),
        "ra": 290.927082, "dec": 14.506353, "fov": 0.1,
        "transition_fov": 0.1, "transition_time": 3, "pause_time": 300000,
        "zoom_in_time": 3, "zoom_out_time": 2,
        "wavelength_slider": {"enabled": True, "wavelengths": steps},
    }]}
    with open(out, "w") as fh:
        json.dump(wp, fh, indent=2, ensure_ascii=False)
    print(f"wrote {out}: {len(steps)} steps, {first} → {last}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true")
    g.add_argument("--index", type=int, help="triplet by position (SLURM array)")
    g.add_argument("--waypoints", metavar="OUT")
    args = ap.parse_args()
    if args.list:
        for i, trip in enumerate(TRIPLETS):
            print(i, layer_name(trip))
            for w in trip:
                print(f"    {FILTER_DIR[w]}: {os.path.basename(final_residual(w))}")
    elif args.waypoints:
        waypoints(args.waypoints)
    else:
        trip = TRIPLETS[args.index]
        print(f"building {layer_name(trip)}", flush=True)
        build(trip)


if __name__ == "__main__":
    main()
