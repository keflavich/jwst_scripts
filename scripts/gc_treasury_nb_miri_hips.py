#!/usr/bin/env python
"""
Monochrome HiPS layers from the CMZ-wide F770W mosaics made by Nazar Budaiev:

    /orange/adamginsburg/jwst/sgrb2/NB/gc/claude_MIRI_mosaics/
        gc10678_f770w_combined_i2d.fits                  all 10678 fields plus
                                                         other CMZ F770W data
        gc10678_f770w_combined_starsub_i2d.fits          same, PSF-fitted stars
                                                         subtracted
        gc10678_f770w_combined_starsub_filled_i2d.fits   same, NaNs and poor
                                                         subtractions filled
                                                         and smoothed
        gc_f770w_fullframe_preview_i2d.fits              includes the 4QPM
                                                         coronagraph fields

They replace the per-tile treasury MIRI layers in the GC HiPS viewer.  The
cron pipeline's own MIRI coadds (jwst_gc_treasury_miri*_hips) are left alone;
these layers are named after the input files.

All four share ONE stretch, solved on the combined mosaic, so a given MJy/sr
renders as the same grey in every layer and they can be toggled against each
other: linear from its 1st to its 99.9th percentile, then asinh with the
softening chosen so the median sky lands at 0.15 of the range.  The
star-subtracted mosaics go negative where stars are over-subtracted; that
clips to black.  fill_nan fills the saturated cores of the two unsubtracted
mosaics; the star-subtracted ones are shown as delivered (the _filled variant
is the filled one).  NaN that touches the edge becomes transparent.

--islands builds a second flavour of a layer, named <layer>_islands, in which
every coverage island (jwst_rgb.island_stretch) gets its own stretch by the
same rule (1st to 99.9th percentile, median at 0.15) solved on that island
alone.  The shared stretch puts the high-latitude control field (median ~20
MJy/sr against a shared floor of ~20) and the fields on the Sgr C side of the
CMZ (medians ~70-77 against a shared median of ~128) at or near black; the
island flavour shows their structure, and in exchange a grey level no longer
means the same MJy/sr from one island to the next.  The shared-stretch layer
keeps its name and is unchanged.

No astrometric correction is applied.  check_orientation validates the
PNG -> AVM -> HiPS round trip against the input FITS; it does not test the
mosaic's astrometry.

Usage:
    gc_treasury_nb_miri_hips.py --list
    gc_treasury_nb_miri_hips.py --levels          # print the shared stretch
    gc_treasury_nb_miri_hips.py --index I         # layer I (SLURM array 0-3)
    gc_treasury_nb_miri_hips.py --index I --islands   # per-island stretch
"""
import argparse
import json
import os
import sys

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ~770 Mpx images; PIL refuses anything over ~180 Mpx by default.
Image.MAX_IMAGE_PIXELS = None

SRC = "/orange/adamginsburg/jwst/sgrb2/NB/gc/claude_MIRI_mosaics"
OUT = "/orange/adamginsburg/jwst/gc-treasury/nb_miri"

# (layer name, input file, fill saturated cores)
LAYERS = [
    ("gc10678_f770w_combined", "gc10678_f770w_combined_i2d.fits", True),
    ("gc10678_f770w_combined_starsub",
     "gc10678_f770w_combined_starsub_i2d.fits", False),
    ("gc10678_f770w_combined_starsub_filled",
     "gc10678_f770w_combined_starsub_filled_i2d.fits", False),
    ("gc_f770w_fullframe", "gc_f770w_fullframe_preview_i2d.fits", True),
]
REFERENCE = "gc10678_f770w_combined_i2d.fits"
TARGET = 0.15
LO_PCT, TOP_PCT = 1, 99.9
# Islands smaller than this many pixels (~0.6 arcmin2 at 0.111"/px) borrow
# the nearest large island's stretch; see jwst_rgb.island_stretch.
ISLAND_MIN_PIX = 50000


def image_hdu(path):
    """The 2D science image: SCI if present, otherwise the first 2D HDU."""
    hl = fits.open(path, memmap=True)
    if "SCI" in hl:
        return hl["SCI"]
    return next(h for h in hl if h.data is not None and h.data.ndim == 2)


def levels_from(v):
    """(lo, top, softening, median) of the sample v: linear from its 1st to
    its 99.9th percentile, then asinh with the softening that puts the
    median at TARGET.  asinh cannot lift the median ABOVE its linear
    position, so a sample whose median already sits at or above TARGET
    (a bright, flat island) gets softening 1e3, which is linear to <1e-6.
    A sample whose 1st and 99.9th percentiles are equal has no range to
    stretch and returns None."""
    from scipy.optimize import brentq
    lo, med, top = np.percentile(v, [LO_PCT, 50, TOP_PCT])
    if not top > lo:
        return None
    m = (med - lo) / (top - lo)
    if m >= TARGET:
        return float(lo), float(top), 1e3, float(med)
    a = brentq(lambda a: np.arcsinh(m / a) / np.arcsinh(1 / a) - TARGET,
               1e-8, 1e3)
    return float(lo), float(top), float(a), float(med)


def levels():
    """(lo, top, softening, median) from the combined mosaic, sampled every
    7th px."""
    data = image_hdu(os.path.join(SRC, REFERENCE)).data
    v = np.asarray(data[::7, ::7], dtype=np.float32).ravel()
    lv = levels_from(v[np.isfinite(v) & (v != 0)])
    if lv is None:
        raise ValueError(f"{REFERENCE}: flat sample, no stretch levels")
    return lv


def island_levels(v):
    """levels_from's (lo, top, softening) for one island, None if flat."""
    lv = levels_from(v)
    return None if lv is None else lv[:3]


def stretch(c, lo, top, a):
    y = np.clip((c - lo) / (top - lo), 0, 1, dtype=np.float32)
    return (np.arcsinh(y / a) / np.arcsinh(1 / a)).astype(np.float32)


def build(index, islands=False):
    from jwst_rgb.save_rgb import save_rgb, avm_for_saved_png, fill_nan
    from gc_treasury_rgb_images import check_orientation

    name, fn, do_fill = LAYERS[index]
    if islands:
        name = f"{name}_islands"
    src = os.path.join(SRC, fn)
    lo, top, a, med = levels()
    print(f"{name}: shared stretch lo={lo:.3f} median={med:.3f} "
          f"top={top:.3f} MJy/sr softening={a:.4g}", flush=True)

    hdu = image_hdu(src)
    header = hdu.header
    raw = np.asarray(hdu.data, dtype=np.float32)
    shape = raw.shape
    if do_fill:
        filled = fill_nan(raw.copy(), bad_data_min_threshold=None)
    else:
        filled = raw
    grey = np.nan_to_num(stretch(filled, lo, top, a))
    island_records = None
    if islands:
        from jwst_rgb.island_stretch import (island_labels, island_slices,
                                             apply_per_island,
                                             island_centres_galactic)
        labels, n = island_labels(np.isfinite(raw) & (raw != 0),
                                  min_area_pix=ISLAND_MIN_PIX)
        island_records = apply_per_island(
            grey, filled, labels, island_slices(labels, n),
            island_levels, stretch)
        del labels
        island_centres_galactic(island_records, WCS(header).celestial)
        for r in island_records:
            print(f"  island {r['label']:3d} l={r['glon']:+.3f} "
                  f"b={r['glat']:+.3f} n={r['n_sampled']} "
                  f"levels={r['levels']}", flush=True)
    del filled
    rgb = np.broadcast_to(grey[:, :, None], shape + (3,))
    original = np.broadcast_to(raw[:, :, None], shape + (3,))

    os.makedirs(f"{OUT}/{name}", exist_ok=True)
    png = f"{OUT}/{name}/{name}.png"
    avm = avm_for_saved_png(WCS(header).celestial, *shape)
    save_rgb(rgb, png, avm=avm, original_data=original, hips=True,
             overwrite=True)
    hips = f"{OUT}/{name}/{name}_hips"
    if not os.path.isdir(os.path.join(hips, "Norder3")):
        raise RuntimeError(f"{name}: build produced no Norder3")
    with open(f"{OUT}/{name}/{name}_inputs.json", "w") as fh:
        json.dump({"layer": name, "input": src, "filled": do_fill,
                   "stretch": {"lo": lo, "median": med, "top": top,
                               "softening": a, "target": TARGET,
                               "from": REFERENCE},
                   "islands": island_records}, fh, indent=2)
    ok = check_orientation(hips, src)
    print(f"done: {hips} orientation_ok={ok}", flush=True)
    if ok is not True:
        why = "failed" if ok is False else "was inconclusive"
        raise RuntimeError(f"{name}: orientation/astrometry check {why}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true")
    g.add_argument("--levels", action="store_true")
    g.add_argument("--index", type=int, help="layer by position")
    ap.add_argument("--islands", action="store_true",
                    help="give each coverage island its own stretch; "
                         "writes <layer>_islands")
    args = ap.parse_args(argv)
    if args.islands and args.index is None:
        ap.error("--islands needs --index")
    if args.list:
        for i, (name, fn, do_fill) in enumerate(LAYERS):
            p = os.path.join(SRC, fn)
            print(f"layer {i}: {name}  fill={do_fill}  "
                  f"exists={os.path.exists(p)}  {p}")
    elif args.levels:
        print(levels())
    else:
        build(args.index, islands=args.islands)


if __name__ == "__main__":
    main()
