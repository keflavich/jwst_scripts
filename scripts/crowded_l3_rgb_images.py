#!/usr/bin/env python
# coding: utf-8
"""
RGB images for the crowded_l3 field (program 9438, obs 006).

FIELDS ARE KEPT SEPARATE.  The pipeline provides `-merged`, `-nrca` and `-nrcb`
mosaics per filter.  The two NIRCam modules point at DIFFERENT, NON-OVERLAPPING
patches of sky -- module A sits at l=3.000 b=-0.001, module B at l=3.042
b=+0.022, ~2.5 arcmin apart -- so the `-merged` mosaic (10074x4571) is 15.7%
empty sky between the two footprints.  Compositing that would waste most of the
frame and stretch the two fields against each other's pixel statistics, so each
module is treated as its own field and the merged mosaic is deliberately unused.

FOURTEEN FILTERS are available per module:
  SW (0.031"/px, 4479x4450):  F070W F090W F115W F140M F150W F182M F210M
  LW (0.063"/px, 2173x2166):  F277W F300M F335M F360M F410M F430M F480M
Everything is reprojected onto one target grid per field (default F210M, the
longest SW filter -- keeps SW resolution and upsamples LW rather than throwing
SW detail away).

TWO FAMILIES OF COLOUR COMBINATIONS are produced:
  rolling  -- every consecutive filter triple (stride 1), 12 combos.  Adjacent
              filters look similar, so these show subtle colour structure.
  throw    -- deliberately wider wavelength baselines: stride 2, stride 3, and
              hand-picked full-span triples that pair the bluest SW against the
              reddest LW.  These separate hot stars, reddened stars and ice/PAH
              features that the rolling set compresses.

ORIENTATION: save_rgb lays pixels down with transpose=ROTATE_180, so the AVM
that matches those pixels is the rot180 dihedral of the target FITS WCS, NOT
the identity that faithful_avm produces.  Pairing faithful_avm's identity with
ROTATE_180 pixels renders the field rotated 180 deg on sky -- verified
repeatedly on brick/sgrb2/sickle/cloudc.  So the flip is determined ONCE per
(field, target grid) by correlating the rendered luminance against the target
FITS through each candidate AVM, and the winning CDMatrix AVM is then reused
for every combo on that grid.
"""
import argparse
import os
import sys

import numpy as np
from astropy.io import fits
from astropy.visualization import simple_norm
from astropy.wcs import WCS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apply_cdmatrix_flip import cdmatrix_avm  # noqa: E402

from jwst_rgb.save_rgb import save_rgb as _save_rgb  # noqa: E402

BASE = "/orange/adamginsburg/jwst/crowded_l3"
FIELDS = ("nrca", "nrcb")

# wavelength-ordered; SW first then LW
FILTERS = ["f070w", "f090w", "f115w", "f140m", "f150w", "f182m", "f210m",
           "f277w", "f300m", "f335m", "f360m", "f410m", "f430m", "f480m"]
SW = set(FILTERS[:7])


def wl(f):
    """filter -> wavelength in units of 0.01 um (f070w -> 70)."""
    return int("".join(c for c in f if c.isdigit()))


def image_filenames(field):
    return {f: f"{BASE}/{f.upper()}/pipeline/"
               f"jw09438-o006_t001_nircam_clear-{f}-{field}_i2d.fits"
            for f in FILTERS}


def save_rgb(*args, **kwargs):
    kwargs.setdefault("alpha_only_edges", True)
    return _save_rgb(*args, **kwargs)


def rolling_combos(filters):
    """Consecutive triples: subtle colour, dense wavelength sampling."""
    return [(filters[i + 2], filters[i + 1], filters[i])      # R=reddest
            for i in range(len(filters) - 2)]


def throw_combos(filters):
    """Wider wavelength baselines than the rolling set."""
    out = []
    for stride in (2, 3, 4):
        for i in range(len(filters) - 2 * stride):
            out.append((filters[i + 2 * stride], filters[i + stride], filters[i]))
    # hand-picked full-span triples (bluest SW vs reddest LW)
    for combo in (("f480m", "f277w", "f070w"),
                  ("f480m", "f210m", "f090w"),
                  ("f430m", "f300m", "f115w"),
                  ("f410m", "f210m", "f070w"),
                  ("f480m", "f335m", "f150w"),
                  ("f430m", "f182m", "f090w")):
        if all(c in filters for c in combo):
            out.append(combo)
    # dedupe, preserve order
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def resolve_avm(tgt_header, ref_data, png_probe_shape, png_path=None):
    """Measure which dihedral's CDMatrix AVM matches save_rgb's pixel layout.

    This actually renders a probe PNG through save_rgb (so the pixels are laid
    down exactly as the real products are, transpose=ROTATE_180 included),
    embeds each candidate CDMatrix AVM in turn, reprojects the PNG luminance
    through that AVM onto the target FITS grid, and keeps the dihedral whose
    projection correlates best with the reference data.

    An earlier version of this function documented that measurement but simply
    returned the rot180 answer, leaving `ref_data` unused -- the orientation was
    verified out-of-band and then asserted here.  Asserting it is fragile: the
    correct dihedral depends on save_rgb's transpose default, so a change there
    would silently rotate every product by 180 degrees.  Measuring costs one
    probe render per grid and makes the guarantee real.
    """
    import tempfile
    from reproject import reproject_interp
    from PIL import Image
    import pyavm

    fwcs = WCS(tgt_header).celestial
    ny, nx = png_probe_shape
    candidates = ("rot180", "identity", "flipud", "fliplr")

    if ref_data is None:
        return cdmatrix_avm(fwcs, ny, nx, "rot180")

    # Bin the probe (and its WCS) before rendering.  The dihedral question is
    # about orientation, not detail, so a binned analogue answers it identically
    # while keeping the four PNG round-trips cheap -- at full size this is four
    # renders of a 4479x4450 frame and takes minutes.
    k = max(1, int(max(ny, nx) // 700))
    bw = fwcs[::k, ::k]
    bref = np.asarray(ref_data)[::k, ::k]
    bny, bnx = bref.shape
    norm = simple_norm(bref, stretch="asinh", min_percent=1, max_percent=99.5)
    probe = np.nan_to_num(np.stack([norm(bref)] * 3, axis=2))

    best, best_corr = None, -np.inf
    with tempfile.TemporaryDirectory() as td:
        png = os.path.join(td, "probe.png")
        for flip in candidates:
            avm = cdmatrix_avm(bw, bny, bnx, flip)
            _save_rgb(np.clip(probe, 0, 1), png, avm=avm, hips=False,
                      transpose=Image.ROTATE_180, alpha_only_edges=True,
                      verbose=False, overwrite=True)
            lum = np.asarray(Image.open(png).convert("L"), float)[::-1]
            awcs = pyavm.AVM.from_image(png).to_wcs().celestial
            rep, _ = reproject_interp((lum, awcs), bw, shape_out=bref.shape)
            m = np.isfinite(rep) & np.isfinite(bref)
            if m.sum() < 100:
                continue
            x = rep[m] - rep[m].mean()
            y = bref[m] - bref[m].mean()
            den = np.sqrt((x * x).sum() * (y * y).sum())
            corr = float((x * y).sum() / den) if den > 0 else -np.inf
            print(f"    dihedral {flip:8s} corr={corr:+.3f}", flush=True)
            if corr > best_corr:
                best, best_corr = flip, corr
    if best is None:
        raise RuntimeError("could not measure the AVM dihedral")
    print(f"  AVM dihedral = {best} (corr {best_corr:+.3f})", flush=True)
    return cdmatrix_avm(fwcs, ny, nx, best)


def make_pngs(field, target_filter="f210m",
              new_basepath=f"{BASE}/data_reprojected",
              png_path=None, hips=False, combos="all"):
    import reproject
    fns = image_filenames(field)
    missing = [f for f, p in fns.items() if not os.path.exists(p)]
    if missing:
        print(f"[{field}] WARNING missing filters: {missing}")
    filters = [f for f in FILTERS if f not in missing]
    if len(filters) < 3:
        print(f"[{field}] fewer than 3 filters, skipping")
        return

    png_path = png_path or f"{BASE}/pngs_{field}_{target_filter[1:-1]}"
    os.makedirs(png_path, exist_ok=True)
    os.makedirs(new_basepath, exist_ok=True)
    print(f"[{field}] target={target_filter}  {len(filters)} filters -> {png_path}")

    tgt_header = fits.getheader(fns[target_filter], ext=("SCI", 1))

    data = {}
    for f in filters:
        out = f"{new_basepath}/{field}_{f}_reprj_{target_filter}.fits"
        if not os.path.exists(out):
            print(f"  reprojecting {f}")
            arr, _ = reproject.reproject_interp(fns[f], tgt_header, hdu_in="SCI")
            fits.PrimaryHDU(data=arr, header=tgt_header).writeto(out, overwrite=True)
        data[f] = fits.getdata(out)

    shape = data[target_filter].shape
    AVM = resolve_avm(tgt_header, data[target_filter], shape)

    sets = []
    if combos in ("all", "rolling"):
        sets += [("rolling", c) for c in rolling_combos(filters)]
    if combos in ("all", "throw"):
        sets += [("throw", c) for c in throw_combos(filters)]

    print(f"[{field}] {len(sets)} combos")
    for kind, (f1, f2, f3) in sets:
        rgb = np.stack([data[f1], data[f2], data[f3]], axis=2)
        tag = f"{wl(f1)}-{wl(f2)}-{wl(f3)}"
        for stretch, lo, suffix in (("asinh", 1, ""), ("log", 1.5, "_log")):
            scaled = np.stack([
                simple_norm(rgb[:, :, k], stretch=stretch, min_percent=lo,
                            max_percent=99.5)(rgb[:, :, k]) for k in range(3)
            ], axis=2)
            name = f"{png_path}/CrowdedL3_{field}_RGB_{tag}{suffix}.png"
            save_rgb(np.nan_to_num(scaled), name, avm=AVM, original_data=rgb,
                     hips=hips)
        print(f"  {kind:7s} {tag}")

    # single-filter greyscales
    for f in filters:
        d = data[f]
        for stretch, lo, suffix in (("asinh", 1, "_asinh"), ("log", 1.5, "_log")):
            img = simple_norm(d, stretch=stretch, min_percent=lo, max_percent=99.5)(d)
            save_rgb(np.nan_to_num(np.stack([img] * 3, axis=2)),
                     f"{png_path}/CrowdedL3_{field}_{wl(f)}{suffix}.png",
                     avm=AVM, original_data=np.stack([d] * 3, axis=2), hips=hips)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fields", nargs="+", default=list(FIELDS), choices=list(FIELDS))
    p.add_argument("--target-filter", default="f210m", choices=FILTERS)
    p.add_argument("--combos", default="all", choices=["all", "rolling", "throw"])
    p.add_argument("--hips", action="store_true")
    a = p.parse_args()
    for field in a.fields:
        make_pngs(field, target_filter=a.target_filter, hips=a.hips,
                  combos=a.combos)


if __name__ == "__main__":
    main()
