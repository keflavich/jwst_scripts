#!/usr/bin/env python
"""
Star-subtracted RGB layers for the pre-treasury CMZ NIR HiPS (jwst_nir_hips).

Each layer mirrors one layer of rebuild_jwst_cmz_hips.py's NIR_LAYERS, but is
built from that field's FINAL crowdsource/daophot residual mosaic:

    <base>/<FILTER>/pipeline/<prefix>-<filter>-<module>[_resbgsub]_m<N>_daophot_basic_mergedcat_residual_i2d.fits

with the highest N per filter.  `_residual_smoothed_bg_`, `_im0_badastrom`,
per-module (nrca/nrcb) and per-group files are ignored.

The per-target stretches of the original layers were tuned on images with
stars in them and do not carry over, so every channel here gets a percentile
asinh stretch.  Over-subtracted star cores leave a long negative tail (1st
percentile ~ -90 against a median of ~5 on the Brick), which would set the
black point and wash out the background; pixels below the 2nd percentile are
replaced by the 9-pixel local median, and the black point is the 5th
percentile.

Usage:
    starsub_rgb_layers.py --list            # resolve inputs, build nothing
    starsub_rgb_layers.py --layer NAME      # build one layer (PNG + HiPS)
    starsub_rgb_layers.py --coadd           # coadd all built layers
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
from astropy.io import fits
from astropy.visualization import simple_norm
from astropy.wcs import WCS

JWST = "/orange/adamginsburg/jwst"
OUT = f"{JWST}/starsub_rgb"
COADD = f"{OUT}/jwst_nir_starsub_hips"

# name -> (base dir, file prefix, module token, (R, G, B)); G=None is mean(R, B).
# Bottom -> top, in the same order as NIR_LAYERS.
LAYERS = {
    "cloudcJWST_starsub_RGB_466-mean-405": (
        f"{JWST}/cloudc", "jw02221-o002_t001_nircam_clear", "merged",
        ("f466n", None, "f405n")),
    "SgrB2_starsub_RGB_480-405-187": (
        f"{JWST}/sgrb2", "jw05365-o001_t001_nircam_clear", "merged",
        ("f480m", "f405n", "f187n")),
    "Cloudef_starsub_RGB_480-360-210": (
        f"{JWST}/cloudef", "jw02092-o002_t001_nircam_clear", "merged",
        ("f480m", "f360m", "f210m")),
    "CloudefControl_starsub_RGB_480-360-210": (
        f"{JWST}/cloudef_controlfield", "jw02092-o005_t001_nircam_clear",
        "merged", ("f480m", "f360m", "f210m")),
    "SGRC_starsub_RGB_480-360-212": (
        f"{JWST}/sgrc", "jw04147-o012_t001_nircam_clear", "merged",
        ("f480m", "f360m", "f212n")),
    "SGRC_NIRISS_starsub_RGB_480-356-200": (
        f"{JWST}/sgrc/niriss", "jw04147-o012_t001_niriss_clear", "nis",
        ("f480m", "f356w", "f200w")),
    "Brick_starsub_RGB_444-356-200": (
        f"{JWST}/brick", "jw01182-o004_t001_nircam_clear", "merged",
        ("f444w", "f356w", "f200w")),
    "BrickJWST_starsub_RGB_466-410-405": (
        f"{JWST}/brick", "jw02221-o001_t001_nircam_clear", "merged",
        ("f466n", "f410m", "f405n")),
    "Arches_starsub_RGB_323-mean-212": (
        f"{JWST}/arches", "jw02045-o001_t001_nircam_clear", "merged",
        ("f323n", None, "f212n")),
    "Quintuplet_starsub_RGB_323-mean-212": (
        f"{JWST}/quintuplet", "jw02045-o003_t001_nircam_clear", "merged",
        ("f323n", None, "f212n")),
    # The SgrA NIR layer is F444W/F323N/F212N; only F405N, F212N and F115W
    # have residual mosaics.
    "SgrA_starsub_RGB_405-212-115": (
        f"{JWST}/sgra", "jw01939-o001_t001_nircam_clear", "merged",
        ("f405n", "f212n", "f115w")),
}


def final_residual(base, prefix, module, filt):
    """Path of the highest-iteration residual mosaic for one filter."""
    pat = re.compile(rf"^{re.escape(prefix)}-{filt}-{module}(_resbgsub)?_m(\d+)"
                     r"_daophot_basic_mergedcat_residual_i2d\.fits$")
    found = []
    for path in glob.glob(f"{base}/{filt.upper()}/pipeline/*residual_i2d.fits"):
        m = pat.match(os.path.basename(path))
        if m:
            found.append((int(m.group(2)), path))
    if not found:
        raise FileNotFoundError(f"no residual mosaic for {prefix} {filt} in {base}")
    return max(found)[1]


def inputs(name):
    base, prefix, module, filters = LAYERS[name]
    return {f: final_residual(base, prefix, module, f)
            for f in filters if f is not None}


def load_on(path, header, shape):
    from reproject import reproject_interp
    with fits.open(path) as hl:
        data, hdr = hl["SCI"].data, hl["SCI"].header
        if data.shape == shape and WCS(hdr).celestial.wcs.compare(
                WCS(header).celestial.wcs, tolerance=1e-9):
            return data.astype(np.float32)
        out, _ = reproject_interp((data, WCS(hdr).celestial),
                                  WCS(header).celestial, shape_out=shape)
    return out.astype(np.float32)


def fill_holes(d, percent=2, size=9):
    """Replace over-subtracted star cores with the local median."""
    from scipy.ndimage import median_filter
    good = np.isfinite(d) & (d != 0)
    lo = np.percentile(d[good], percent)
    holes = good & (d < lo)
    med = median_filter(np.where(good, d, np.median(d[good])), size=size,
                        mode="nearest")
    out = d.copy()
    out[holes] = med[holes]
    return out


def build(name):
    from jwst_rgb.save_rgb import save_rgb, avm_for_saved_png

    paths = inputs(name)
    fR, fG, fB = LAYERS[name][3]
    for f, p in paths.items():
        print(f"  {f}: {p}")
    header = fits.getheader(paths[fR], ext=("SCI", 1))
    shape = (header["NAXIS2"], header["NAXIS1"])
    R = load_on(paths[fR], header, shape)
    B = load_on(paths[fB], header, shape)
    G = (R + B) / 2 if fG is None else load_on(paths[fG], header, shape)
    rgb = np.dstack([R, G, B])

    def stretch(c):
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
    print(f"done: {png}")


def coadd():
    from reproject.hips import coadd_hips
    layers = [f"{OUT}/{n}/{n}_hips" for n in LAYERS]
    missing = [p for p in layers
               if not os.path.exists(os.path.join(p, "properties"))]
    for p in missing:
        print(f"skipping (not built): {p}")
    layers = [p for p in layers if p not in missing]
    if not layers:
        sys.exit("no layers built")
    stage = COADD + ".new"
    if os.path.exists(stage):
        import shutil
        shutil.rmtree(stage)
    print(f"coadding {len(layers)} layers -> {stage}")
    coadd_hips(layers, stage)
    if os.path.exists(COADD):
        import shutil
        shutil.rmtree(COADD)
    os.rename(stage, COADD)
    print(f"done: {COADD}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true")
    g.add_argument("--layer", choices=list(LAYERS))
    g.add_argument("--index", type=int, help="layer by position (SLURM array)")
    g.add_argument("--coadd", action="store_true")
    args = ap.parse_args()

    if args.list:
        for n in LAYERS:
            print(n)
            try:
                for f, p in inputs(n).items():
                    print(f"  {f}: {os.path.basename(p)}")
            except FileNotFoundError as ex:
                print(f"  MISSING: {ex}")
    elif args.coadd:
        coadd()
    else:
        name = args.layer or list(LAYERS)[args.index]
        print(f"building {name}")
        build(name)


if __name__ == "__main__":
    main()
