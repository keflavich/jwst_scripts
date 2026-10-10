#!/usr/bin/env python
"""
RGB HiPS for every adjacent three-filter combination of a field, for the
linear wavelength tours (ACES_Aladin_tour/<field>_wavelength_tour_linear.html).

This is wd2_adjacent_rgb_layers.py generalized to other fields.  The tour
steps through consecutive filter triplets: R/G/B = three neighbouring
filters in wavelength.  Inputs are the regular (not star-subtracted)
mosaics, the astrometry-corrected pipeline mosaic wherever one exists.

Each layer is put on the pixel grid of its bluest filter, the finest one in
the triplet.  Interior NaN islands are sorted on the native grid before
reprojection (fill_cores_mark_gaps): saturated cores are filled at the top of
the channel's range so they render white, and coverage gaps (e.g. the NIRCam
SW detector-gap strips) stay blank.  Edges and gaps in any channel become
transparent.  Each channel gets a
percentile asinh stretch.  Every build ends with check_orientation
(gc_treasury_rgb_images), which correlates served tiles against the grid FITS
for a 180 degree flip and measures the translation (tolerance 0.3"); a layer
that fails or cannot be checked raises.

Usage:
    adjacent_rgb_layers.py FIELD --list             # resolve inputs only
    adjacent_rgb_layers.py FIELD --index I          # build triplet I (array)
    adjacent_rgb_layers.py FIELD --publish          # copy built HiPS to avm_images
    adjacent_rgb_layers.py FIELD --publish --replace I [I ...]
                                     # also replace those published layers
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
    "wd1": {
        "prefix": "wd1",
        "title": "Westerlund 1 Wavelength Explorer (Linear)",
        "mosaics": {
            **{w: _pipe("wd1", f"F{w}{s}",
                        f"jw01905-o001_t001_nircam_clear-f{w}{s.lower()}"
                        "-merged_i2d.fits")
               for w, s in [(115, "W"), (150, "W"), (164, "N"), (187, "N"),
                            (200, "W"), (212, "N"), (277, "W"), (323, "N"),
                            (405, "N"), (444, "W"), (466, "N")]},
            # MIRI mosaics (program 1905, obs 002) sit at the top of the wd1
            # dir, not in a filter/pipeline subdir like the NIRCam ones.
            770: f"{JWST}/wd1/miri_F770W_pid1905_combined_SF_i2d.fits",
            1000: f"{JWST}/wd1/miri_F1000W_pid1905_combined_SF_i2d.fits",
            1130: f"{JWST}/wd1/miri_F1130W_pid1905_combined_SF_i2d.fits",
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


def fill_cores_mark_gaps(data, aspect_min=4.0, gap_min_pix=50,
                         core_border_pct=95.0, big_core_border_pct=99.5,
                         top_pct=99.5, grow=2):
    """Split the interior NaN islands of one mosaic into saturated cores and
    coverage gaps.

    Cores are islands whose surrounding ring is bright (median above the
    image's ``core_border_pct`` percentile; ``big_core_border_pct`` for
    islands of ``gap_min_pix`` pixels or more) and which are not long thin
    strips.  On wd1 the borders of real saturated cores rank above the 99.9th
    percentile; the F323N ghost patches rank 96.6-98.8, so a big island
    whose border lies between the two thresholds gets its border median
    instead of the core value.  They are grown by ``grow`` pixels (the pixels next to a
    saturated core are corrupted too) and set to a value at or above the
    channel's ``top_pct`` percentile, so they come out white in every
    channel after the stretch instead of a flat, mid-level colour.

    Gaps (islands of at least ``gap_min_pix`` pixels that are long thin
    strips or have a faint border: detector/dither gaps with no data, e.g.
    the NIRCam SW strips) stay NaN, so they become transparent instead of
    being painted with the border percentile.  Smaller faint-bordered holes
    (bad pixels) get their border median.  Islands touching the array edge
    stay NaN as before.  Exact zeros count as blank: the MIRI combined
    mosaics mark saturated cores and no-coverage with 0, not NaN.

    Returns (filled data, core mask).
    """
    from scipy import ndimage
    data = np.where(data == 0, np.nan, data)
    finite = np.isfinite(data)
    vals = data[finite]
    bright = np.percentile(vals, core_border_pct)
    very_bright = np.percentile(vals, big_core_border_pct)
    top = np.percentile(vals, top_pct)
    lab, nlab = ndimage.label(~finite)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    out = data
    core = np.zeros(data.shape, dtype=bool)
    ny, nx = data.shape
    pad = grow + 3
    ncore = ngap = nsmall = 0
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None or i in edge:
            continue
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        ex = (slice(max(sl[0].start - pad, 0), min(sl[0].stop + pad, ny)),
              slice(max(sl[1].start - pad, 0), min(sl[1].stop + pad, nx)))
        isl = lab[ex] == i
        ring = ndimage.binary_dilation(isl, iterations=2) & ~isl & finite[ex]
        npix = int(isl.sum())
        aspect = max(h, w) / max(min(h, w), 1)
        bmed = np.median(data[ex][ring]) if ring.any() else -np.inf
        faint = not bmed > bright
        big = npix >= gap_min_pix
        if big and (aspect >= aspect_min or faint):
            ngap += 1
            continue
        if faint or (big and not bmed > very_bright):
            nsmall += 1
            if ring.any():
                sub = out[ex]
                sub[isl] = bmed
            continue
        ncore += 1
        fillv = max(np.percentile(data[ex][ring], 99), top)
        grown = ndimage.binary_dilation(isl, iterations=grow)
        sub = out[ex]
        # NaN compares False, so NaN pixels in the grown core are filled too
        sub[grown & ~(sub >= fillv)] = fillv
        core[ex] |= grown
    print(f"    {ncore} saturated cores filled, {ngap} gaps left blank, "
          f"{nsmall} holes filled with border median, "
          f"{len(edge) - (0 in edge)} edge regions", flush=True)
    return out, core


def load_filled_on(path, header, shape):
    """SCI data on the target grid: saturated cores filled, coverage gaps
    and edges NaN.  Also returns the core mask on the target grid."""
    from reproject import reproject_interp
    with fits.open(path) as hl:
        data = hl["SCI"].data.astype(np.float32)
        hdr = hl["SCI"].header
    data, core = fill_cores_mark_gaps(data)
    if data.shape == shape and WCS(hdr).celestial.wcs.compare(
            WCS(header).celestial.wcs, tolerance=1e-9):
        return data, core
    out, _ = reproject_interp((data, WCS(hdr).celestial),
                              WCS(header).celestial, shape_out=shape)
    cmask, _ = reproject_interp((core.astype(np.float32), WCS(hdr).celestial),
                                WCS(header).celestial, shape_out=shape)
    return out.astype(np.float32), np.nan_to_num(cmask) > 0.25


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
    loaded = [load_filled_on(paths[w], header, shape) for w in trip]
    rgb = np.dstack([d for d, _ in loaded])

    def stretch(c, core):
        # limits from real pixels only: the filled cores would otherwise
        # push the top percentile up and dim the whole channel
        good = np.isfinite(c) & (c != 0) & ~core
        norm = simple_norm(c[good], stretch="asinh", min_percent=1,
                           max_percent=99.5)
        return np.clip(norm(c).filled(np.nan), 0, 1)

    scaled = np.dstack([stretch(rgb[:, :, k], loaded[k][1]) for k in range(3)])
    # Transparent wherever any channel has no data (array edges and
    # coverage gaps).  A 1/NaN stand-in keeps save_rgb's |x|<1e-5 test from
    # punching holes at real near-zero pixels; alpha_only_edges=False is
    # needed because the gaps are interior.
    blank = ~np.isfinite(rgb).all(axis=2)
    alpha_src = np.where(blank, np.nan, 1.0)[:, :, None].repeat(3, axis=2)

    os.makedirs(f"{outdir(field)}/{name}", exist_ok=True)
    png = f"{outdir(field)}/{name}/{name}.png"
    avm = avm_for_saved_png(WCS(header), *shape)
    save_rgb(np.nan_to_num(scaled), png, avm=avm, original_data=alpha_src,
             alpha_only_edges=False, hips=True, overwrite=True)
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


def publish(field, replace=(), stamp=None):
    """Copy every verified layer into avm_images.  An existing layer of the
    same name is left alone unless its triplet index is in ``replace``; then
    the old directory is parked as ``<dst>_stale_<stamp>`` (not deleted) and
    the new build is copied in its place."""
    import time
    stamp = stamp or time.strftime("%Y%m%d")
    for i, trip in enumerate(triplets(field)):
        name = layer_name(field, trip)
        src = f"{outdir(field)}/{name}/{name}_hips"
        dst = f"{PUBLISHED}/{name}_hips"
        if not os.path.exists(os.path.join(src, "ORIENTATION_OK")):
            print(f"skip {name}: not built or not verified")
        elif os.path.exists(dst) and i not in replace:
            print(f"skip {name}: {dst} exists")
        elif os.path.exists(dst):
            parked = f"{dst}_stale_{stamp}"
            if os.path.exists(parked):
                raise FileExistsError(parked)
            shutil.copytree(src, dst + ".new")
            os.rename(dst, parked)
            os.rename(dst + ".new", dst)
            print(f"replaced {dst} (old layer parked at {parked})")
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
    ap.add_argument("--replace", type=int, nargs="+", default=[],
                    metavar="I", help="with --publish: triplet indices whose "
                    "published layer is replaced (old one parked _stale_DATE)")
    args = ap.parse_args()
    if args.list:
        for i, trip in enumerate(triplets(args.field)):
            print(i, layer_name(args.field, trip))
            for w in trip:
                print(f"    {w}: {mosaic(args.field, w)}")
    elif args.publish:
        publish(args.field, replace=set(args.replace))
    elif args.waypoints:
        waypoints(args.field, args.waypoints)
    else:
        trip = triplets(args.field)[args.index]
        print(f"building {layer_name(args.field, trip)}", flush=True)
        build(args.field, trip)


if __name__ == "__main__":
    main()
