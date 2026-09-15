#!/usr/bin/env python
"""MIRI F770W / F2100W monochrome images + HiPS for program 2092 (cloud E/F).

Program 2092 delivers a MIRI parallel alongside its NIRCam pointings, the same
way 10678's MIRI parallel is handled in gc_treasury_rgb_images.py
(find_i2d / build_miri_obs / cmd_coadd(miri=True)) -- this script follows that
model, generalised to two filters (F770W, F2100W) and two fields.

Which observations belong to which field
-----------------------------------------
The data directories name things by field (cloudef vs cloudef_controlfield),
but only the NIRCam side actually lives under the field it is named for:
cloudef_controlfield/ holds F162M/F210M/F360M/F480M only -- no MIRI directory
exists there at all.  All six MIRI mosaics (F770W and F2100W x o004/o006/o008)
sit under cloudef/{F770W,F2100W}/pipeline/, and reading the pointing back out
of their headers shows they do NOT all belong to the main field:

  field (NIRCam F210M centre)      l        b
  cloudef        (o002)        0.4857   0.0074
  cloudef_control(o005)        0.4137   0.1898

  MIRI obs   l        b       nearest field         offset from that centre
  o004     0.4901   0.0104   cloudef                 0.005 deg
  o008     0.5213   0.0243   cloudef                 0.04  deg (2nd tile)
  o006     0.4182   0.1927   cloudef_control         0.005 deg

o006 sits on the CONTROL field pointing, not the main field -- despite living
in the cloudef/ directory tree and despite an earlier script
(cloudef_rgb_images.py) treating o004/o006/o008 as if they were three tiles of
one field.  o004 and o008 are the two real tiles of the main field's MIRI
coverage.  Assignment here is done by measuring each mosaic's own header WCS
against both field centres (classify_field), not by trusting directory names,
and raises rather than guessing if a pointing doesn't land close to either.

Picking the current mosaic per (obs, filter)
---------------------------------------------
Each obs/filter has been reprocessed at least once, leaving two i2d.fits next
to each other with different names: the original `..._miri_f770w_i2d.fits`
and a later rerun `..._miri_clear-f770w-mirimage_data_i2d.fits` (o004 F770W is
the one exception -- it has not been rerun, so only the plain name exists).
Both are canonical stage-3 mosaics, not derived products, so pick between them
by DATE header (pipeline processing time) rather than filename -- confirmed
against mtime and against the CAL_VER/DATE pair for every pair on disk.

Usage
-----
  cloudef_miri_images.py --list                # inventory: obs -> field, filter, path
  cloudef_miri_images.py --build                # per-observation png + hips
  cloudef_miri_images.py --coadd                # per-field, per-filter hips
  cloudef_miri_images.py --all                  # both of the above
"""
import argparse
import glob
import os
import re
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gc_treasury_rgb_images import _is_exposure_level  # noqa: E402

BASE = {
    "cloudef": "/orange/adamginsburg/jwst/cloudef",
    "cloudef_control": "/orange/adamginsburg/jwst/cloudef_controlfield",
}
OUTDIR = {
    "cloudef": f"{BASE['cloudef']}/pngs_miri",
    "cloudef_control": f"{BASE['cloudef_control']}/pngs_miri",
}
LAYER_PREFIX = {"cloudef": "Cloudef", "cloudef_control": "CloudefControl"}

FILTERS = ["f770w", "f2100w"]

# Reference pointings, galactic (l, b) degrees, from each field's own NIRCam
# F210M mosaic header (program 2092 o002 and o005).  See module docstring.
FIELD_CENTERS = {
    "cloudef": (0.4857, 0.0074),
    "cloudef_control": (0.4137, 0.1898),
}
# The two field centres are ~0.20 deg apart; a MIRI pointing must land within
# this of the CLOSER one to be assigned to it, or classify_field raises.
FIELD_TOL_DEG = 0.1

# Stage-3 mosaic names actually seen for this program's MIRI data: the
# original per-filter name, and the later rerun's "clear-<filt>-mirimage_data"
# name (mirrors the NIRCam "clear-<filt>-merged" convention).  Both are
# genuine mosaics; anything else ending in _i2d.fits in these directories is a
# per-exposure product, a crf/skymatch intermediate, or a catalog/segm file
# and is excluded by requiring one of these two exact shapes.
_PLAIN = re.compile(r"^jw\d+-o\d+_t\d+_miri_[a-z0-9]+_i2d\.fits$")
_DATA = re.compile(r"^jw\d+-o\d+_t\d+_miri_clear-[a-z0-9]+-mirimage_data_i2d\.fits$")


def classify_field(l_deg, b_deg):
    """Nearest field centre to (l_deg, b_deg), or raise if neither is close.

    Raising rather than guessing matters here: this program's directory names
    (cloudef vs cloudef_controlfield) do NOT reliably indicate which field a
    MIRI pointing belongs to (see module docstring) -- o006 sits on the
    control field despite living in the cloudef/ directory tree.
    """
    best_field, best_sep = None, None
    for field, (fl, fb) in FIELD_CENTERS.items():
        sep = float(np.hypot(l_deg - fl, b_deg - fb))
        if best_sep is None or sep < best_sep:
            best_field, best_sep = field, sep
    if best_sep > FIELD_TOL_DEG:
        raise ValueError(
            f"pointing l={l_deg:.4f} b={b_deg:.4f} is {best_sep:.4f} deg from "
            f"the nearest known field centre ({best_field}); refusing to "
            f"guess -- update FIELD_CENTERS or FIELD_TOL_DEG if this is a new "
            f"field/pointing")
    return best_field


def _pointing(path):
    from astropy.io import fits
    from astropy.wcs import WCS
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h = fits.getheader(path, ext=("SCI", 1))
        w = WCS(h).celestial
        ny, nx = h["NAXIS2"], h["NAXIS1"]
        c = w.pixel_to_world(nx / 2, ny / 2).galactic
    return float(c.l.deg), float(c.b.deg)


def find_i2d(filt):
    """Canonical MIRI mosaics for one filter, keyed by (field, obs).

    Returns {(field, "o004"): path, ...}.  Mirrors gc_treasury_rgb_images'
    find_i2d: observation comes from the OBSERVTN header (filenames are not
    consistent about it), exposure-level products are excluded with the same
    regex used there, and -- specific to this program -- when both mosaic
    naming variants exist for one obs, the one with the later DATE header
    (the more recent pipeline run) wins.
    """
    from astropy.io import fits
    candidates = {}  # obs -> list of (date, path)
    for base in (BASE["cloudef"], BASE["cloudef_control"]):
        for p in glob.glob(f"{base}/{filt.upper()}/pipeline/*_i2d.fits"):
            b = os.path.basename(p)
            if _is_exposure_level(b):
                continue
            if not (_PLAIN.match(b) or _DATA.match(b)):
                continue
            try:
                h = fits.getheader(p)
            except OSError:  # still being written
                continue
            obs = h.get("OBSERVTN")
            if not obs:
                continue
            key = f"o{int(obs):03d}"
            candidates.setdefault(key, []).append((h.get("DATE", ""), p))

    out = {}
    for obs, entries in candidates.items():
        entries.sort(key=lambda e: e[0])  # DATE sorts chronologically as text
        path = entries[-1][1]
        l_deg, b_deg = _pointing(path)
        field = classify_field(l_deg, b_deg)
        out[(field, obs)] = path
    return out


def inventory():
    """{filt: {(field, obs): path}} for every filter."""
    return {f: find_i2d(f) for f in FILTERS}


def png_for(field, filt, obs):
    return f"{OUTDIR[field]}/{LAYER_PREFIX[field]}_MIRI_{filt.upper()}_{obs}.png"


def hips_for(field, filt, obs):
    return f"{OUTDIR[field]}/{LAYER_PREFIX[field]}_MIRI_{filt.upper()}_{obs}_hips"


def coadd_png_prefix(field, filt):
    return f"{OUTDIR[field]}/{LAYER_PREFIX[field]}_MIRI_{filt.upper()}"


def coadd_hips_for(field, filt):
    return f"{coadd_png_prefix(field, filt)}_hips"


def build_obs(field, filt, obs, src, hips=True):
    """One (field, filt, obs) mosaic -> monochrome png + per-obs HiPS.

    Mirrors gc_treasury_rgb_images.build_miri_obs: grayscale (a single MIRI
    band has nothing to make a colour from), asinh stretch on its own
    1-99.5 percentiles, and the avm_for_saved_png embedding -- NOT
    faithful_avm, which leaves CRPIX at the FITS value and puts tiles ~1.3"
    off (see jwst_rgb/save_rgb.py docstring).
    """
    from astropy.io import fits
    from astropy.visualization import simple_norm
    from astropy.wcs import WCS
    from PIL import Image
    from jwst_rgb.save_rgb import save_rgb as _save_rgb
    from jwst_rgb.save_rgb import avm_for_saved_png
    Image.MAX_IMAGE_PIXELS = None

    os.makedirs(OUTDIR[field], exist_ok=True)
    hdu = next(h for h in fits.open(src) if h.data is not None and h.data.ndim == 2)
    d = hdu.data.astype(float)
    ny, nx = d.shape
    print(f"{field} {obs}: MIRI {filt.upper()} grid {nx}x{ny}", flush=True)

    g = simple_norm(d, stretch="asinh", min_percent=1, max_percent=99.5)(d)
    mono = np.stack([np.nan_to_num(g)] * 3, axis=2)
    png = png_for(field, filt, obs)
    avm = avm_for_saved_png(WCS(hdu.header).celestial, ny, nx,
                            flip=-1, transpose=Image.ROTATE_180)
    _save_rgb(np.clip(mono, 0, 1), png, avm=avm, transpose=Image.ROTATE_180,
              alpha_only_edges=True, original_data=np.stack([d] * 3, axis=2),
              hips=False)
    print(f"  wrote {png}", flush=True)

    hips_dir = None
    if hips:
        from tqdm import tqdm
        from reproject import reproject_interp
        from reproject.hips import reproject_to_hips
        from jwst_rgb.landing_page import patch_hips_dir
        hips_dir = hips_for(field, filt, obs)
        if os.path.exists(hips_dir):
            shutil.rmtree(hips_dir)
        reproject_to_hips(png, coord_system_out="galactic", level=None,
                          reproject_function=reproject_interp,
                          output_directory=hips_dir, threads=16,
                          progress_bar=tqdm)
        patch_hips_dir(hips_dir)
        if not os.path.isdir(os.path.join(hips_dir, "Norder3")):
            raise RuntimeError(f"{field} {obs}: MIRI build produced no Norder3")
        print(f"  wrote {hips_dir}", flush=True)
    return png, hips_dir


def build_field_coadd(field, filt, obs_list):
    """Coadd every observation's per-obs HiPS for one (field, filt) into the
    single layer that gets fed to the CMZ MIRI overview (jwst_miri_hips)."""
    from reproject.hips import coadd_hips
    from jwst_rgb.landing_page import patch_hips_dir

    layers = [hips_for(field, filt, o) for o in sorted(obs_list)]
    missing = [L for L in layers if not os.path.isdir(L)]
    if missing:
        raise FileNotFoundError(f"missing per-obs HiPS: {missing}")
    out = coadd_hips_for(field, filt)
    if os.path.exists(out):
        shutil.rmtree(out)
    print(f"coadding {len(layers)} layer(s) -> {out}", flush=True)
    coadd_hips(layers, out)
    patch_hips_dir(out)
    print(f"  wrote {out}", flush=True)
    return out


def cmd_list():
    inv = inventory()
    for filt in FILTERS:
        print(f"\n{filt.upper()}:")
        for (field, obs), path in sorted(inv[filt].items()):
            print(f"  {field:16s} {obs}  {path}")


def cmd_build():
    inv = inventory()
    for filt in FILTERS:
        for (field, obs), path in sorted(inv[filt].items()):
            build_obs(field, filt, obs, path)


def cmd_coadd():
    inv = inventory()
    for filt in FILTERS:
        by_field = {}
        for (field, obs) in inv[filt]:
            by_field.setdefault(field, []).append(obs)
        for field, obs_list in by_field.items():
            build_field_coadd(field, filt, obs_list)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--coadd", action="store_true")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.list or not (a.build or a.coadd or a.all):
        cmd_list()
    if a.build or a.all:
        cmd_build()
    if a.coadd or a.all:
        cmd_coadd()


if __name__ == "__main__":
    main()
