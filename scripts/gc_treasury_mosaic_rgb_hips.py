#!/usr/bin/env python
"""RGB + AVM + HiPS for the FULL-SURVEY combined mosaics of program 10678.

This is a different product from gc_treasury_rgb_images.py, which builds one
RGB+HiPS per OBSERVATION and coadds them incrementally.  This script instead
starts from the single already-reprojected, already-coadded FITS mosaics in
mosaics/ (built by make_mosaics.py / make_residual_mosaics.py from every
observation's stage-3 i2d at once) and makes one RGB and one HiPS from each.

Why this is a separate script rather than a mode of the per-observation one
--------------------------------------------------------------------------
The per-observation pipeline is per-obs for reasons that do not apply here
(see avm-hips's notes, 2026-09-16): incremental HiPS append against a full
rebuild, per-field background correction, and to bound memory at ~433 MB per
frame.  A single already-combined mosaic has none of a "per-field" axis left
to preserve, so building one big RGB directly is simpler and there is no
compositing question to get wrong.

Target grid: F480M, not F212N
------------------------------
The per-observation script reprojects onto F212N (the finer 0.031"/px grid)
because at single-pointing size (~11440x4736) that is cheap.  At full-survey
size the two mosaics are NOT comparably sized: F480M's optimal WCS came out to
22503x26635 (~599 Mpx) and F212N's to 45517x53756 (~2.45 Gpx) -- 4x more
pixels, because find_optimal_celestial_wcs was run independently per band and
each inherited its own input pixel scale over the same sky area. Reprojecting
the long-wavelength channel onto the short-wavelength grid would make a ~9.8
GB RGBA PNG (uint8) and cost 4x the memory and time for no resolution this
mosaic's cameras actually deliver in F480M. F480M's native grid is used as the
common target instead; F212N is reprojected down onto it. This is a deliberate
resolution choice for the survey-overview product -- flag it if a
full-native-resolution combined mosaic is wanted instead.

AVM / orientation
------------------
Use avm_for_saved_png(wcs, ny, nx, flip=-1, transpose=Image.ROTATE_180), NOT
faithful_avm and NOT a raw pyavm.AVM.from_header. save_rgb writes the PNG
flipped and rotated 180 degrees relative to the input array; only
avm_for_saved_png reflects CRPIX on both axes to describe the PNG as written.
Getting this wrong offsets every tile by |N+1-2*crpix| pixels -- silent,
invisible without a reference overlay, and it has bitten this exact codebase
before (1.26" on a GC Treasury field, 3.86" on an archival MIRI layer).

Namespacing
-----------
Products are named gctreasury_mosaic_* and live under mosaics/, deliberately
distinct from jwst_gc_treasury_* under gc-treasury/pngs/, which is the hourly
cron's tree (rebuilt automatically, gated by pngs/.auto.lock). Writing there
or reusing its name would let this script's output be silently overwritten,
or collide with a running rebuild.

Usage
-----
  gc_treasury_mosaic_rgb_hips.py --which main       # RGB + MIRI mono from the
                                                      # main (i2d) mosaics
  gc_treasury_mosaic_rgb_hips.py --which residual   # same, from the DAOPHOT
                                                      # residual mosaics
  gc_treasury_mosaic_rgb_hips.py --which both
  gc_treasury_mosaic_rgb_hips.py --which main --no-hips   # PNG/AVM only
"""
import argparse
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MOSAIC_DIR = "/orange/adamginsburg/jwst/gc-treasury/mosaics"
OUTDIR = MOSAIC_DIR  # our own tree; NOT gc-treasury/pngs (that is the cron's)

LONG_FILTER = "f480m"     # target grid; see module docstring
SHORT_FILTER = "f212n"
MIRI_FILTER = "f770w"

# simple_norm kwargs, applied to the raw FITS values in MJy/sr.  Mirrors the
# per-observation script's "pct" and "vminmax" flavours (see
# gc_treasury_rgb_images.STRETCHES) without importing that script's much
# larger, per-observation-specific module.
STRETCHES = {
    "pct": dict(stretch="asinh", min_percent=1, max_percent=99.5),
    "vminmax": dict(stretch="asinh", vmin=-0.5, vmax=100),
}
DEFAULT_STRETCH = "pct"


def mosaic_path(filt, which):
    suffix = "_residual" if which == "residual" else ""
    return f"{MOSAIC_DIR}/{filt}{suffix}_mosaic.fits"


def rgb_png_for(which, stretch):
    suffix = "_residual" if which == "residual" else ""
    tag = "" if stretch == DEFAULT_STRETCH else f"_{stretch}"
    return f"{OUTDIR}/gctreasury_mosaic_RGB_480-mean-212{suffix}{tag}.png"


def rgb_hips_for(which, stretch):
    return rgb_png_for(which, stretch).replace(".png", "_hips")


def miri_png_for(which):
    suffix = "_residual" if which == "residual" else ""
    return f"{OUTDIR}/gctreasury_mosaic_MIRI_F770W{suffix}.png"


def miri_hips_for(which):
    return miri_png_for(which).replace(".png", "_hips")


def _load_primary(path):
    """Read data + celestial WCS from a mosaics/*.fits file.

    make_mosaics.py / make_residual_mosaics.py write the mosaic as the
    PRIMARY HDU (not a 'SCI' extension), so this is simpler than the
    per-observation script's "first 2-D HDU" search over i2d files.
    """
    from astropy.io import fits
    from astropy.wcs import WCS
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(float)
        wcs = WCS(hdul[0].header).celestial
    return data, wcs


def _mask_mixed_nan(long_, short_):
    """A pixel real in one channel and NaN in the other renders as a flat
    false-coloured island; forbid the mixed state outright (same rule
    gc_treasury_rgb_images.build_obs applies per-observation)."""
    nan = [np.isnan(long_), np.isnan(short_)]
    bad = np.logical_or.reduce(nan) & ~np.logical_and.reduce(nan)
    if bad.any():
        print(f"  masking {100 * bad.mean():.2f}% NaN-in-one-not-both pixels",
              flush=True)
        long_ = np.where(bad, np.nan, long_)
        short_ = np.where(bad, np.nan, short_)
    return long_, short_


def _stretch_channels(chans, stretch):
    from astropy.visualization import simple_norm
    kw = STRETCHES[stretch]
    return np.stack([np.nan_to_num(simple_norm(c, **kw)(c)) for c in chans],
                    axis=2)


def build_rgb(which="main", stretch=DEFAULT_STRETCH, hips=True):
    from astropy.wcs import WCS
    from astropy.io import fits
    from PIL import Image
    from reproject import reproject_interp
    from jwst_rgb.save_rgb import save_rgb as _save_rgb
    from jwst_rgb.save_rgb import avm_for_saved_png
    Image.MAX_IMAGE_PIXELS = None  # full-survey mosaics legitimately exceed
                                    # PIL's decompression-bomb heuristic

    long_path = mosaic_path(LONG_FILTER, which)
    short_path = mosaic_path(SHORT_FILTER, which)
    for p in (long_path, short_path):
        if not os.path.exists(p):
            raise RuntimeError(f"missing mosaic: {p}")

    print(f"[{which}] loading target grid from {long_path}", flush=True)
    long_, twcs = _load_primary(long_path)
    ny, nx = long_.shape
    print(f"[{which}] target grid {nx}x{ny} ({LONG_FILTER.upper()} native)",
          flush=True)

    print(f"[{which}] reprojecting {SHORT_FILTER.upper()} onto it", flush=True)
    with fits.open(short_path) as hdul:
        short_data = hdul[0].data.astype(float)
        swcs = WCS(hdul[0].header).celestial
    short_, _ = reproject_interp((short_data, swcs), twcs, shape_out=(ny, nx))
    del short_data

    long_, short_ = _mask_mixed_nan(long_, short_)
    mid = np.nanmean(np.stack([long_, short_]), axis=0)
    chans = [long_, mid, short_]  # R = long, G = mean, B = short

    print(f"[{which}] stretch '{stretch}': {STRETCHES[stretch]}", flush=True)
    scaled = _stretch_channels(chans, stretch)

    png = rgb_png_for(which, stretch)
    avm = avm_for_saved_png(twcs, ny, nx, flip=-1, transpose=Image.ROTATE_180)
    _save_rgb(np.clip(scaled, 0, 1), png, avm=avm, transpose=Image.ROTATE_180,
              alpha_only_edges=True, original_data=np.stack(chans, axis=2),
              hips=False)
    print(f"[{which}] wrote {png}", flush=True)

    hips_dir = None
    if hips:
        hips_dir = _build_hips(png, rgb_hips_for(which, stretch))
    return png, hips_dir


def build_miri(which="main", hips=True):
    from astropy.wcs import WCS
    from PIL import Image
    from astropy.visualization import simple_norm
    from jwst_rgb.save_rgb import save_rgb as _save_rgb
    from jwst_rgb.save_rgb import avm_for_saved_png
    Image.MAX_IMAGE_PIXELS = None

    src = mosaic_path(MIRI_FILTER, which)
    if not os.path.exists(src):
        raise RuntimeError(f"missing mosaic: {src}")
    d, wcs = _load_primary(src)
    ny, nx = d.shape
    print(f"[{which}] MIRI {MIRI_FILTER.upper()} grid {nx}x{ny}", flush=True)

    g = simple_norm(d, stretch="asinh", min_percent=1, max_percent=99.5)(d)
    mono = np.stack([np.nan_to_num(g)] * 3, axis=2)

    png = miri_png_for(which)
    avm = avm_for_saved_png(wcs, ny, nx, flip=-1, transpose=Image.ROTATE_180)
    _save_rgb(np.clip(mono, 0, 1), png, avm=avm, transpose=Image.ROTATE_180,
              alpha_only_edges=True, original_data=np.stack([d] * 3, axis=2),
              hips=False)
    print(f"[{which}] wrote {png}", flush=True)

    hips_dir = None
    if hips:
        hips_dir = _build_hips(png, miri_hips_for(which))
    return png, hips_dir


def _build_hips(png, hips_dir):
    from tqdm import tqdm
    from reproject import reproject_interp
    from reproject.hips import reproject_to_hips
    from jwst_rgb.landing_page import patch_hips_dir

    if os.path.exists(hips_dir):
        shutil.rmtree(hips_dir)
    print(f"  building HiPS -> {hips_dir}", flush=True)
    reproject_to_hips(png, coord_system_out="galactic", level=None,
                      reproject_function=reproject_interp,
                      output_directory=hips_dir, threads=16,
                      progress_bar=tqdm)
    patch_hips_dir(hips_dir)
    if not os.path.isdir(os.path.join(hips_dir, "Norder3")):
        raise RuntimeError(f"{hips_dir}: build produced no Norder3")
    print(f"  wrote {hips_dir}", flush=True)
    return hips_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--which", choices=["main", "residual", "both"],
                    default="main")
    ap.add_argument("--stretch", choices=sorted(STRETCHES), default=DEFAULT_STRETCH)
    ap.add_argument("--no-hips", action="store_true")
    ap.add_argument("--rgb-only", action="store_true")
    ap.add_argument("--miri-only", action="store_true")
    args = ap.parse_args(argv)

    whichs = ["main", "residual"] if args.which == "both" else [args.which]
    hips = not args.no_hips
    for which in whichs:
        if not args.miri_only:
            build_rgb(which, stretch=args.stretch, hips=hips)
        if not args.rgb_only:
            build_miri(which, hips=hips)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
