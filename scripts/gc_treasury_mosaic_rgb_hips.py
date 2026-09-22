#!/usr/bin/env python
"""RGB + AVM + HiPS for the FULL-SURVEY combined mosaics of program 10678.

This is a different product from gc_treasury_rgb_images.py, which builds one
RGB+HiPS per OBSERVATION and coadds them incrementally.  This script instead
starts from the single already-reprojected, already-coadded FITS mosaics in
mosaics/ (built by make_mosaics.py / make_residual_mosaics.py from every
observation's stage-3 i2d at once) and builds two RGB combinations from them,
each with its own HiPS:

  480 grid (build_rgb):      R=F480M, G=mean(F480M,F212N), B=F212N,
                             on F480M's native grid.
  770 grid (build_rgb_trio): R=F770W, G=F480M, B=F212N,
                             on F770W's native grid.

Plus a plain monochrome F770W layer (build_miri) with no colour synthesis at
all, useful as a diagnostic independent of either RGB's stretch choices.

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
because at single-pointing size (~11440x4736) that is cheap.  F212N's pixel
scale is ~0.031"/px against F480M's ~0.063"/px -- 2x finer per axis, ~4x more
pixels per unit sky covered -- and find_optimal_celestial_wcs is run
independently per band, so the two full-survey mosaics are never comparably
sized. (Measured 2026-09-16: F480M 35915x53695 = 1.93 Gpx, F212N
45517x53756 = 2.45 Gpx; both grow release to release as more observations
land and are not re-measured by this docstring -- read them as illustrating
the ~4x-per-unit-area relationship, not a pinned ratio.) Reprojecting the
long-wavelength channel onto the short-wavelength grid buys no real
resolution -- F480M's own native pixels are already the coarser limit --
while costing ~4x the memory and time. F480M's native grid is used as the
common target instead; F212N is reprojected down onto it. This is a
deliberate resolution choice for the survey-overview product -- flag it if a
full-native-resolution combined mosaic is wanted instead.

Peak memory and wall time (measured 2026-09-16, full SLURM runs)
------------------------------------------------------------------
build_rgb_trio (F770W's 587 Mpx grid), --which both, one job: peak 41.3 GiB
RSS (sacct .batch MaxRSS 43297740 KiB; note that is KiB, not KB -- read the
unit off sacct before quoting a number from it into anything that sizes an
allocation), wall time 53 min for main+residual PNG+HiPS combined.
Comfortably inside a 120 GiB / 8 h request.

build_rgb (F480M's 1.93 Gpx grid): peak 152.2 GiB RSS (MaxRSS 159638068
KiB), roughly double the ~70-90 GiB the channel-array arithmetic above
predicts. The array work itself (load, reproject, stretch, save_rgb's PNG
write) finishes in well under an hour and is not the gap; the discrepancy
shows up somewhere in reproject_to_hips's own buffers while generating the
deepest HEALPix order, which is also the part that ran out of TIME: this
grid's finer pixel scale needs an extra order versus the trio's (Norder13
vs Norder12), the deepest order dominates total tile count, and a 250 GiB /
8 h job got through the array work and the PNG but hit the wall-time limit
partway into that HiPS build, having written 11,442 Norder13 tiles with no
coarser orders started yet (reproject_to_hips builds finest-to-coarsest, so
a run killed there is not resumable through the API used here -- delete the
partial output and rebuild). The mechanism inside reproject_to_hips that
holds the other ~60-80 GiB has not been identified; request at least 160
GiB and at least a day of wall time for build_rgb at this mosaic size until
it has been.

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
  gc_treasury_mosaic_rgb_hips.py --which main         # both RGB grids + MIRI
                                                        # mono, main mosaics
  gc_treasury_mosaic_rgb_hips.py --which residual     # same, DAOPHOT
                                                        # residual mosaics
  gc_treasury_mosaic_rgb_hips.py --which both
  gc_treasury_mosaic_rgb_hips.py --grids 480          # only the F480M-grid
                                                        # two-filter RGB
  gc_treasury_mosaic_rgb_hips.py --grids 770          # only the F770W-grid
                                                        # three-filter RGB
  gc_treasury_mosaic_rgb_hips.py --which main --no-hips   # PNG/AVM only
"""
import argparse
import os
import shutil
import sys

import numpy as np
from jwst_rgb.hips_naming import properties_for

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


def rgb_trio_png_for(which, stretch):
    suffix = "_residual" if which == "residual" else ""
    tag = "" if stretch == DEFAULT_STRETCH else f"_{stretch}"
    return f"{OUTDIR}/gctreasury_mosaic_RGB_770-480-212{suffix}{tag}.png"


def rgb_trio_hips_for(which, stretch):
    return rgb_trio_png_for(which, stretch).replace(".png", "_hips")


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
        # float32, not float64: at full-survey pixel counts (~2 Gpx) the
        # doubled footprint of float64 is the difference between fitting in
        # a generously-sized SLURM allocation and not (see "Peak memory"
        # above; a reviewer measured ~172 GiB at float64 for this call).
        # The i2d SCI data is BITPIX=-32 (float32) already, so this loses no
        # precision the mosaic ever had.
        data = hdul[0].data.astype(np.float32)
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
    # simple_norm returns float64 regardless of the input dtype; cast back
    # down immediately rather than let the stack inherit it (see "Peak
    # memory" in the module docstring).
    return np.stack([np.nan_to_num(simple_norm(c, **kw)(c)).astype(np.float32)
                     for c in chans], axis=2)


def _reproject_onto(filt, which, twcs, ny, nx):
    """Load one band's mosaic and reproject it onto (twcs, ny, nx).

    Always returns float32: reproject_interp itself always returns float64,
    so this casts straight back down rather than let it double the resident
    size of whichever input is being reprojected (see "Peak memory" in the
    module docstring).
    """
    from astropy.wcs import WCS
    from astropy.io import fits
    from reproject import reproject_interp

    path = mosaic_path(filt, which)
    if not os.path.exists(path):
        raise RuntimeError(f"missing mosaic: {path}")
    print(f"  reprojecting {filt.upper()} onto the target grid", flush=True)
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(np.float32)
        wcs = WCS(hdul[0].header).celestial
    out, _ = reproject_interp((data, wcs), twcs, shape_out=(ny, nx))
    del data
    return out.astype(np.float32)


def build_rgb(which="main", stretch=DEFAULT_STRETCH, hips=True):
    """R = F480M, G = mean(F480M, F212N), B = F212N, on F480M's native grid.

    The two channels are the SAME two filters this survey actually has in
    NIRCam (see gc_treasury_rgb_images's module docstring: "10678 observes
    F212N (SW) and F480M (LW) only, so there is no third colour"), so the
    green channel is synthesised as their mean rather than a real filter.
    Compare build_rgb_trio, which has three independent real filters and no
    synthesised channel.
    """
    from PIL import Image
    from jwst_rgb.save_rgb import save_rgb as _save_rgb
    from jwst_rgb.save_rgb import avm_for_saved_png
    Image.MAX_IMAGE_PIXELS = None  # full-survey mosaics legitimately exceed
                                    # PIL's decompression-bomb heuristic

    long_path = mosaic_path(LONG_FILTER, which)
    if not os.path.exists(long_path):
        raise RuntimeError(f"missing mosaic: {long_path}")

    print(f"[{which}] loading target grid from {long_path}", flush=True)
    long_, twcs = _load_primary(long_path)
    ny, nx = long_.shape
    print(f"[{which}] target grid {nx}x{ny} ({LONG_FILTER.upper()} native)",
          flush=True)

    short_ = _reproject_onto(SHORT_FILTER, which, twcs, ny, nx)

    long_, short_ = _mask_mixed_nan(long_, short_)
    # Plain mean, not np.nanmean(np.stack(...)): _mask_mixed_nan guarantees
    # a NaN in one channel is a NaN in both at that pixel, so (a+b)/2 already
    # gives NaN there and a correct average everywhere else, without the
    # extra (ny, nx, 2) stack nanmean would allocate.
    mid = (long_ + short_) / 2
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


def build_rgb_trio(which="main", stretch=DEFAULT_STRETCH, hips=True):
    """R = F770W, G = F480M, B = F212N, all on F770W's native grid.

    Unlike build_rgb, all three channels are real, independent filters -- no
    synthesised mean. R is NOT required to overlap G/B: 10678's MIRI
    parallel points several arcmin off the NIRCam prime for a given
    observation (gc_treasury_rgb_images's module docstring), so large parts
    of the full-survey F770W footprint have no NIRCam coverage at all, and a
    G/B-less pixel there is real footprint, not a bug -- it is deliberately
    left to render red-only rather than forced transparent.

    G and B, however, ARE the same co-designed NIRCam pair build_rgb combines
    (F480M/F212N observe the same field at the same time), so the SAME
    all-or-nothing rule build_rgb applies to them still holds: a pixel real
    in one and NaN in the other there is a mixed-coverage edge artifact, not
    real sky, and is masked with _mask_mixed_nan exactly as in build_rgb. R
    is never a party to that masking.

    Peak memory: F770W's native grid is much smaller than F480M's (measured
    2026-09-16: 20363x28821 = 587 Mpx at ~0.111"/px, roughly a third of
    F480M's 1.93 Gpx -- see the module docstring's "Target grid" section for
    why the two are unrelated grids), and it shows in the measurement: a
    full --which both run peaked at 41.3 GiB RSS in 53 min (see the module
    docstring's "Peak memory and wall time" section), against build_rgb's
    152 GiB and wall-time-limited HiPS build. The module docstring's memory
    and wall-time guidance is sized for build_rgb; running --grids 770 alone
    needs much less of both.
    """
    from PIL import Image
    from jwst_rgb.save_rgb import save_rgb as _save_rgb
    from jwst_rgb.save_rgb import avm_for_saved_png
    Image.MAX_IMAGE_PIXELS = None

    r_path = mosaic_path(MIRI_FILTER, which)
    if not os.path.exists(r_path):
        raise RuntimeError(f"missing mosaic: {r_path}")

    print(f"[{which}] loading target grid from {r_path}", flush=True)
    r, twcs = _load_primary(r_path)
    ny, nx = r.shape
    print(f"[{which}] target grid {nx}x{ny} ({MIRI_FILTER.upper()} native)",
          flush=True)

    g = _reproject_onto(LONG_FILTER, which, twcs, ny, nx)
    b = _reproject_onto(SHORT_FILTER, which, twcs, ny, nx)
    g, b = _mask_mixed_nan(g, b)   # NIRCam pair only -- R is never masked
    chans = [r, g, b]

    print(f"[{which}] stretch '{stretch}': {STRETCHES[stretch]}", flush=True)
    scaled = _stretch_channels(chans, stretch)

    png = rgb_trio_png_for(which, stretch)
    avm = avm_for_saved_png(twcs, ny, nx, flip=-1, transpose=Image.ROTATE_180)
    # save_rgb's alpha is the OR of each channel's OWN blank mask: a pixel is
    # made transparent if ANY of the three is blank there, not only if ALL
    # three are. That is correct for build_rgb, where _mask_mixed_nan already
    # equalizes F480M/F212N's NaN pattern before this point, so every channel
    # shares one blank mask. It is wrong here: G/B are blank almost
    # everywhere R (the target grid) has data, because NIRCam barely overlaps
    # the MIRI-parallel footprint (see this function's docstring) -- passing
    # [r, g, b] made a real run's main flavour 96.7% transparent (measured
    # 2026-09-16, before this fix) and its residual flavour 100% transparent.
    # R alone should drive alpha, so pass only R: save_rgb's per-channel loop
    # is `if i < original_data.shape[2]`, so shape (ny, nx, 1) makes it skip
    # G/B and compute every channel's blank mask from R -- one scipy.ndimage
    # label() pass over a zero-copy view, rather than three redundant passes
    # over a materialized (ny, nx, 3) copy of the same array.
    _save_rgb(np.clip(scaled, 0, 1), png, avm=avm, transpose=Image.ROTATE_180,
              alpha_only_edges=True, original_data=r[:, :, np.newaxis],
              hips=False)
    print(f"[{which}] wrote {png}", flush=True)

    hips_dir = None
    if hips:
        hips_dir = _build_hips(png, rgb_trio_hips_for(which, stretch))
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
                      properties=properties_for(hips_dir),
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
    ap.add_argument("--grids", choices=["480", "770", "both"], default="both",
                    help="which RGB grid(s) to build: 480 = R/G/B "
                         "F480M/mean/F212N on F480M's grid, 770 = "
                         "R/G/B F770W/F480M/F212N on F770W's grid")
    ap.add_argument("--no-hips", action="store_true")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--rgb-only", action="store_true",
                       help="skip the plain monochrome F770W layer")
    group.add_argument("--miri-only", action="store_true",
                       help="build only the plain monochrome F770W layer")
    args = ap.parse_args(argv)

    whichs = ["main", "residual"] if args.which == "both" else [args.which]
    hips = not args.no_hips
    for which in whichs:
        if not args.miri_only:
            if args.grids in ("480", "both"):
                build_rgb(which, stretch=args.stretch, hips=hips)
            if args.grids in ("770", "both"):
                build_rgb_trio(which, stretch=args.stretch, hips=hips)
        if not args.rgb_only:
            build_miri(which, hips=hips)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
