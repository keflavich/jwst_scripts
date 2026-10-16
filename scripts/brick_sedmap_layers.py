#!/usr/bin/env python
"""Rebuild the Brick HiPS layers shown on the Sickle/Brick MIRI SED map page.

Page: https://data.rc.ufl.edu/secure/adamginsburg/jwst/sickle/sed_portal/sickle_miri_sed_map.html

An audit on 2026-10-09 found these layers misaligned on the sky (measured on
the served tiles against the source i2d files and VIRAC2):

  * Brick_RGB_405-356-212, _466-410-356, _212-187-115 and
    BrickJWST_1182p2221_405_356_200: about (+0.05, +0.10)" (dRA*, dDec).
  * Brick_RGB_2550-1500-1130 and _2550-1130-466: the F2550W channel about
    (-4.5, -2.5)" and, in the latter, the F466N channel about +0.19" in RA.

The i2d files themselves agree with VIRAC2 and with each other to <~10 mas,
so the offsets came in at the PNG/HiPS step.  The exact cause per layer is
not fully reconstructed: the NIRCam layers (built 2026-08-23) predate the
current i2d files (08-25 / 09-01), and the F2550W channel was most likely
drawn from a reprojection cache written before the 09-04 F2550W
re-correction.  The two failure modes this script guards against:

  1. Reprojection caches reused by existence alone.  A cache written from an
     older i2d (or onto an older version of the target grid) carries the old
     astrometry, while the AVM is built from the current target header.  Here
     a cache is reused only if it is newer than both its source and the
     target-grid file AND its stored WCS matches the current target WCS.
  2. An AVM that does not describe the PNG as save_rgb writes it.  save_rgb
     (flip=-1, transpose=ROTATE_180) rotates the array 180 degrees, so the
     AVM comes from `avm_for_saved_png`, never `faithful_avm` or a raw
     `AVM.from_header`.

No offset is applied anywhere: the layers are rebuilt from the pipeline
products as they are.

Usage
-----
  brick_sedmap_layers.py build NAME [NAME ...]   # PNG + staged <web>/<NAME>_hips.new
  brick_sedmap_layers.py swap  NAME [NAME ...]   # park <NAME>_hips as _stale_<date>, move .new in
  brick_sedmap_layers.py list
"""
import argparse
import datetime
import os
import shutil
import sys

import numpy as np
from astropy.io import fits
from astropy.visualization import simple_norm
from astropy.wcs import WCS
from PIL import Image
import reproject
from reproject.hips import reproject_to_hips
from tqdm import tqdm

from jwst_rgb.save_rgb import save_rgb as _save_rgb, avm_for_saved_png
from jwst_rgb.hips_naming import properties_for

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brick_rgb_images import image_filenames_pipe as FN  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

WEB = "/orange/adamginsburg/web/public/avm_images"
PNGDIR = "/orange/adamginsburg/jwst/brick/pngs_sedmap"
CACHE = "/orange/adamginsburg/jwst/brick/data_reprojected"

# name -> ((R, G, B) filters, target-grid filter)
LAYERS = {
    "Brick_RGB_405-356-212": (("f405n", "f356w", "f212n"), "f187n"),
    "Brick_RGB_466-410-356": (("f466n", "f410m", "f356w"), "f187n"),
    "Brick_RGB_212-187-115": (("f212n", "f187n", "f115w"), "f187n"),
    "Brick_RGB_2550-1500-1130": (("f2550w", "f1500w", "f1130w"), "f187n"),
    "Brick_RGB_2550-1130-466": (("f2550w", "f1130w", "f466n"), "f187n"),
    "BrickJWST_1182p2221_405_356_200": (("f405n", "f356w", "f200w"), "f356w"),
}


def save_rgb(*args, **kwargs):
    kwargs.setdefault("transpose", Image.ROTATE_180)
    kwargs.setdefault("alpha_only_edges", True)
    return _save_rgb(*args, **kwargs)


def sci_header(path):
    with fits.open(path) as hl:
        return (hl["SCI"] if "SCI" in hl else hl[0]).header.copy()


def same_wcs(h1, h2, tol_pix=1e-3):
    """True if two headers put the same pixels on the same sky."""
    if (h1.get("NAXIS1"), h1.get("NAXIS2")) != (h2.get("NAXIS1"), h2.get("NAXIS2")):
        return False
    w1, w2 = WCS(h1).celestial, WCS(h2).celestial
    ny, nx = h1["NAXIS2"], h1["NAXIS1"]
    xs = np.array([0, nx - 1, 0, nx - 1, nx / 2])
    ys = np.array([0, 0, ny - 1, ny - 1, ny / 2])
    x2, y2 = w2.world_to_pixel(w1.pixel_to_world(xs, ys))
    return bool(np.all(np.hypot(x2 - xs, y2 - ys) < tol_pix))


def reprojected(filt, tgt_filt, tgt_header):
    """Data of `filt` on the `tgt_filt` grid, from a cache only if valid."""
    src = FN[filt]
    if filt == tgt_filt:
        with fits.open(src) as hl:
            return np.array((hl["SCI"] if "SCI" in hl else hl[0]).data, float)
    cache = os.path.join(CACHE, os.path.basename(src).replace(
        "i2d", f"i2d_pipeline_v0.1_reprj_{tgt_filt[:-1]}"))
    if os.path.exists(cache):
        newer = os.path.getmtime(cache) > max(os.path.getmtime(src),
                                              os.path.getmtime(FN[tgt_filt]))
        if newer and same_wcs(fits.getheader(cache), tgt_header):
            print(f"  cache ok   {filt}: {os.path.basename(cache)}", flush=True)
            return fits.getdata(cache).astype(float)
        print(f"  cache STALE {filt} (newer={newer}); reprojecting", flush=True)
    else:
        print(f"  no cache   {filt}; reprojecting", flush=True)
    arr, _ = reproject.reproject_interp(src, tgt_header, hdu_in="SCI")
    # write-then-rename: several layer jobs can share a cache file
    tmp = f"{cache}.tmp{os.getpid()}"
    fits.PrimaryHDU(data=arr, header=tgt_header).writeto(tmp, overwrite=True)
    os.replace(tmp, cache)
    return arr


def build(name):
    filts, tgt = LAYERS[name]
    tgt_header = sci_header(FN[tgt])
    ny, nx = tgt_header["NAXIS2"], tgt_header["NAXIS1"]
    print(f"== {name}: {filts} on the {tgt} grid ({nx}x{ny})", flush=True)
    for f in filts:
        h0 = fits.getheader(FN[f])
        h1 = sci_header(FN[f])
        print(f"  source {f}: {FN[f]}  MIRICOR={h1.get('MIRICOR', h0.get('MIRICOR', '-'))}",
              flush=True)
    rgb = np.stack([reprojected(f, tgt, tgt_header) for f in filts], axis=2)
    scaled = np.stack([simple_norm(rgb[:, :, k], stretch="asinh", min_percent=1,
                                   max_percent=99.5)(rgb[:, :, k]) for k in range(3)],
                      axis=2)
    avm = avm_for_saved_png(WCS(tgt_header).celestial, ny, nx)
    os.makedirs(PNGDIR, exist_ok=True)
    png = os.path.join(PNGDIR, f"{name}.png")
    save_rgb(np.nan_to_num(np.asarray(scaled, float)), png, avm=avm,
             original_data=rgb, hips=False)
    del rgb, scaled
    stage = os.path.join(WEB, f"{name}_hips.new")
    if os.path.isdir(stage):
        shutil.rmtree(stage)
    reproject_to_hips(png, coord_system_out="galactic", level=None,
                      reproject_function=reproject.reproject_interp,
                      output_directory=stage,
                      threads=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)),
                      properties=properties_for(stage, name=f"{name}_hips"),
                      progress_bar=tqdm)
    if not os.path.isdir(os.path.join(stage, "Norder3")):
        raise RuntimeError(f"{name}: staged build has no Norder3")
    print(f"STAGED {stage}", flush=True)


def swap(name, date=None):
    date = date or datetime.date.today().strftime("%Y%m%d")
    live = os.path.join(WEB, f"{name}_hips")
    stage = live + ".new"
    parked = f"{live}_stale_{date}"
    if not os.path.isdir(os.path.join(stage, "Norder3")):
        raise SystemExit(f"{name}: no complete staged build at {stage}")
    if os.path.exists(parked):
        raise SystemExit(f"{name}: {parked} already exists; refusing to overwrite")
    if os.path.isdir(live):
        os.rename(live, parked)
    os.rename(stage, live)
    print(f"SWAPPED {name}: live <- .new ; old parked at {parked}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=("build", "swap", "list"))
    p.add_argument("names", nargs="*")
    a = p.parse_args()
    if a.action == "list":
        for n, (f, t) in LAYERS.items():
            print(n, f, t)
        return
    for n in a.names:
        if n not in LAYERS:
            raise SystemExit(f"unknown layer {n}; known: {list(LAYERS)}")
        (build if a.action == "build" else swap)(n)


if __name__ == "__main__":
    main()
