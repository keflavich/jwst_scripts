#!/usr/bin/env python
"""
MIRI RGB + HiPS layers for program 3571 (Galactic Center, five MIRI tiles).

Inputs are the pipeline's per-observation mosaics

    /orange/adamginsburg/jwst/gc3571/<FILTER>/pipeline/jw03571-o<obs>_t001_miri_<filter>_i2d.fits

for observations o001, o003, o004, o005, o016 (targets GAL-CENTER and
GAL-CENTER-Tile-2/3/4/6) in F560W, F770W, F1000W, F1280W and F1500W.

Step 1 (--mosaic I) coadds the five tiles of one filter onto a common
0.11"/px grid (mean of the overlaps, no background matching).
Step 2 (--index I) builds one RGB layer from three filter mosaics over the
sky all three cover: fill_nan fills the saturated cores, then an asinh
stretch with the channels' median sky matched (see matched_stretch),
save_rgb with avm_for_saved_png, HiPS, and a content-based orientation and
astrometry check that must pass.

No astrometric correction is applied here; the layers show the pipeline's
WCS as is.  check_orientation validates the PNG -> AVM -> HiPS round trip
against the green-channel mosaic; it does not test the mosaic's astrometry.

Usage:
    gc3571_rgb_layers.py --list
    gc3571_rgb_layers.py --mosaic I     # filter I (SLURM array 0-4)
    gc3571_rgb_layers.py --index I      # layer I (SLURM array)
"""
import argparse
import json
import os
import sys

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = "/orange/adamginsburg/jwst/gc3571"
MOSAICS = f"{BASE}/mosaics"
OUT = f"{BASE}/rgb"

OBS = ["001", "003", "004", "005", "016"]
FILTERS = [560, 770, 1000, 1280, 1500]

# R, G, B (reddest first): the three adjacent triplets plus two wide spans.
LAYERS = [(1000, 770, 560), (1280, 1000, 770), (1500, 1280, 1000),
          (1500, 1000, 560), (1280, 770, 560)]


def tile(wave, obs):
    return (f"{BASE}/F{wave}W/pipeline/"
            f"jw03571-o{obs}_t001_miri_f{wave}w_i2d.fits")


def mosaic_path(wave):
    return f"{MOSAICS}/gc3571_F{wave}W_mosaic_i2d.fits"


def layer_name(trip):
    return "gc3571_miri_RGB_{}-{}-{}".format(*trip)


def common_wcs():
    """One output grid for every filter, so all mosaics align pixel for pixel."""
    from reproject.mosaicking import find_optimal_celestial_wcs
    hdrs = [fits.getheader(tile(w, o), ext=("SCI", 1))
            for w in FILTERS for o in OBS]
    shapes = [(h["NAXIS2"], h["NAXIS1"]) for h in hdrs]
    return find_optimal_celestial_wcs(list(zip(shapes, [WCS(h) for h in hdrs])),
                                      resolution=0.11 * u.arcsec,
                                      frame="icrs")


def make_mosaic(wave):
    from reproject import reproject_interp
    from reproject.mosaicking import reproject_and_coadd
    wcs, shape = common_wcs()
    inputs = []
    for o in OBS:
        with fits.open(tile(wave, o)) as hl:
            inputs.append((hl["SCI"].data.astype(float), WCS(hl["SCI"].header)))
    data, foot = reproject_and_coadd(inputs, wcs, shape_out=shape,
                                     reproject_function=reproject_interp,
                                     combine_function="mean")
    data[foot == 0] = np.nan
    hdr = wcs.to_header()
    hdr["BUNIT"] = fits.getheader(tile(wave, OBS[0]), ext=("SCI", 1)).get("BUNIT", "")
    for i, o in enumerate(OBS):
        hdr[f"INPUT{i}"] = os.path.basename(tile(wave, o))
    os.makedirs(MOSAICS, exist_ok=True)
    out = mosaic_path(wave)
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(data=data.astype("float32"), header=hdr,
                                name="SCI")]).writeto(out, overwrite=True)
    print(f"wrote {out} {shape}", flush=True)


def fill(c):
    from jwst_rgb.save_rgb import fill_nan
    # Saturated cores are NaN; fill_nan fills interior islands from their
    # border and leaves edge-touching NaN (outside the footprint) as NaN.
    return fill_nan(c.copy(), bad_data_min_threshold=None)


def matched_stretch(channels, common):
    """asinh stretch with each channel's median sky mapped to the same level.

    A per-channel percentile stretch tints every layer: F1000W sits in the
    silicate absorption, its sky is close to its floor (median 52 vs 1st
    percentile 34 MJy/sr, 99th 3411), so it renders dark wherever it is.
    Here each channel is put in units of (median - 1st percentile) above its
    1st percentile, and all three share one asinh curve and one top end (the
    median over the channels of their 99.9th percentile in those units).
    """
    xs, tops = [], []
    for c in channels:
        v = c[common & (c != 0)]
        p1, p50, p999 = np.percentile(v, [1, 50, 99.9])
        xs.append(np.clip((c - p1) / (p50 - p1), 0, None))
        tops.append((p999 - p1) / (p50 - p1))
    top = np.median(tops)
    out = [np.arcsinh(x) / np.arcsinh(top) for x in xs]
    for o in out:
        o[~common] = np.nan
    return np.dstack(out)


def build(trip):
    from jwst_rgb.save_rgb import save_rgb, avm_for_saved_png
    from gc_treasury_rgb_images import check_orientation

    name = layer_name(trip)
    paths = {w: mosaic_path(w) for w in trip}
    header = fits.getheader(paths[trip[0]], ext=("SCI", 1))
    shape = (header["NAXIS2"], header["NAXIS1"])
    rgb = np.dstack([fits.getdata(paths[w], ext=("SCI", 1)).astype(float)
                     for w in trip])
    filled = [fill(rgb[:, :, k]) for k in range(3)]
    # Show only sky all three filters cover: the filters' footprints differ
    # slightly and the strips one channel lacks render as coloured rims.
    common = np.logical_and.reduce([np.isfinite(c) for c in filled])
    scaled = matched_stretch(filled, common)
    rgb[~common] = np.nan                    # alpha follows the common sky

    os.makedirs(f"{OUT}/{name}", exist_ok=True)
    png = f"{OUT}/{name}/{name}.png"
    avm = avm_for_saved_png(WCS(header), *shape)
    save_rgb(np.nan_to_num(scaled), png, avm=avm, original_data=rgb,
             hips=True, overwrite=True)
    hips = f"{OUT}/{name}/{name}_hips"
    if not os.path.isdir(os.path.join(hips, "Norder3")):
        raise RuntimeError(f"{name}: build produced no Norder3")
    with open(f"{OUT}/{name}/{name}_inputs.json", "w") as fh:
        json.dump({"layer": name,
                   "mosaics": {f"F{w}W": paths[w] for w in trip},
                   "tiles": {f"F{w}W": [tile(w, o) for o in OBS]
                             for w in trip}}, fh, indent=2)
    ok = check_orientation(hips, paths[trip[1]])
    print(f"done: {hips} orientation_ok={ok}", flush=True)
    if ok is not True:
        raise RuntimeError(f"{name}: orientation/astrometry check "
                           f"{'failed' if ok is False else 'was inconclusive'}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true")
    g.add_argument("--mosaic", type=int, help="filter by position")
    g.add_argument("--index", type=int, help="layer by position")
    args = ap.parse_args()
    if args.list:
        for i, w in enumerate(FILTERS):
            missing = [o for o in OBS if not os.path.exists(tile(w, o))]
            print(f"mosaic {i}: F{w}W missing={missing}")
        for i, trip in enumerate(LAYERS):
            print(f"layer {i}: {layer_name(trip)}")
    elif args.mosaic is not None:
        make_mosaic(FILTERS[args.mosaic])
    else:
        build(LAYERS[args.index])


if __name__ == "__main__":
    main()
