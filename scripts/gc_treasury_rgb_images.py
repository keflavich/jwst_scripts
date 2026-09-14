#!/usr/bin/env python
"""RGB images and a HiPS mosaic for JWST program 10678 (GC Treasury).

Program 10678, "The JWST/NIRCam Legacy Survey of the Galactic Center", tiles
139 NIRCam pointings over l = -0.57 .. +0.71, b = -0.14 .. +0.58.  Data land in
/orange/adamginsburg/jwst/gc-treasury/<FILTER>/pipeline/ and this script builds
one RGB per observation plus a HiPS, then coadds every observation's HiPS into a
single growing mosaic for the pipeline monitor's sky view.

Two filters, three channels
---------------------------
10678 observes F212N (SW) and F480M (LW) only, so there is no third colour.  The
repo convention for a two-filter field is R = long, G = mean, B = short (as in
GC2211_o028_RGB_277-mean-150), which keeps stars neutral and lets the F212N /
F480M colour difference carry the hue.

Orientation
-----------
save_rgb applies transpose=ROTATE_180 and embeds the AVM **as-is**;
reproject_to_hips flips the PNG internally to match the FITS-convention WCS, so
a raw AVM.from_header is what yields correctly-oriented tiles.  Pre-flipping
double-flips.  --avm rot180 exists only to re-test that empirically; the default
is the documented path, and every build is checked by correlating the served
tiles against the source FITS rather than trusted.

Usage
-----
  gc_treasury_rgb_images.py --list               # what data exists right now
  gc_treasury_rgb_images.py --obs o135           # build one observation
  gc_treasury_rgb_images.py --all                # every observation with i2d
  gc_treasury_rgb_images.py --coadd              # rebuild the combined mosaic
"""
import argparse
import glob
import os
import re
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = "/orange/adamginsburg/jwst/gc-treasury"
OUTDIR = f"{BASE}/pngs"
# Stage 3 products delivered by MAST, as opposed to the local image3 re-run.
# They sit outside {BASE}/{FILT}/pipeline and are named differently, so they are
# picked up separately -- see the MAST block at the end of find_i2d.
MAST_L3 = f"{BASE}/mastDownload/JWST"
WEB = "/orange/adamginsburg/repos/avm_images"
FILTERS = ["f480m", "f212n"]          # long first
COADD_NAME = "jwst_gc_treasury_hips"
TARGET_FILTER = "f212n"               # finer grid; F480M is reprojected onto it
SETTLE_SECONDS = 900                  # a mosaic must be untouched this long to be built from

# 10678 also runs a MIRI parallel.  It points several arcmin off the NIRCam
# prime, so it is a separate FIELD rather than a third colour channel, and it
# gets its own monochrome layer and its own coadd -- the same treatment the CMZ
# release gives MIRI (jwst_nir_hips vs jwst_miri_hips).
MIRI_FILTER = "f770w"
MIRI_COADD_NAME = "jwst_gc_treasury_miri_hips"
# Two MIRI mosaics are produced from the same tiles: one rendered tile-by-tile
# on its own percentiles, one with the overlap-derived background match and a
# single shared stretch applied.  They are kept side by side so the effect of
# the matching can be judged on the sky rather than argued about.
MIRI_BGMATCH_COADD_NAME = "jwst_gc_treasury_miri_bgmatch_hips"

# How far the served tiles may sit from the source FITS before check_orientation
# calls it a failure.  The HiPS grid is ~0.02"/px at order 14 and reprojection
# adds well under a tenth of an arcsec, so anything past this is a WCS error,
# not resampling noise.
ASTROMETRY_TOL_ARCSEC = 0.3


def _is_exposure_level(basename):
    """True for an image2 per-exposure i2d, false for a stage 3 mosaic.

    image2 writes one i2d per DETECTOR per exposure, e.g.
    jw10678135001_02101_00001_nrca1_i2d.fits (NIRCam) or
    jw10678135001_02201_00006_mirimage_i2d.fits (MIRI) -- a single chip, not a
    mosaic.  Stage 3 writes jw10678-o135_t001_nircam_..._i2d.fits.  Building an
    RGB from the exposure-level products would mean re-doing image3 (outlier
    rejection, skymatch, resample) by hand and badly, so they are counted and
    reported but never built from.
    """
    return re.search(r"_\d{5}_\d{5}_(nrc[ab](long|[1-4])|mirimage)_i2d\.fits$",
                     basename) is not None


# Only the canonical stage 3 products are real mosaics.  The pipeline also
# writes a pile of derived files ending in _i2d.fits into the same directory --
# daophot catalogue models, residuals, smoothed backgrounds, star-subtracted
# (resbgsub) variants -- and a name-agnostic glob picks whichever sorts last.
# That is not hypothetical: it silently fed a daophot MODEL image to the MIRI
# builder for o134 and o135, which is what produced a +64.94 background step
# between o133 and o134 where the real difference is +2.67.  A whitelist is used
# rather than a blacklist so a new suffix cannot reintroduce the bug.
CANONICAL_I2D = re.compile(
    r"^jw\d+-o\d+_t001_"
    r"(nircam_clear-[a-z0-9]+-(merged|nrca|nrcb)|miri_[a-z0-9]+)"
    r"_i2d\.fits$")

# MAST writes the same kind of product under a different name: the target token
# is the observation (t116, not t001) and NIRCam carries no module suffix.  Kept
# as its own whitelist so the local pattern stays exactly as strict as it was.
MAST_I2D = re.compile(
    r"^jw\d+-o\d+_t\d+_"
    r"(nircam_clear-[a-z0-9]+|miri_[a-z0-9]+)"
    r"_i2d\.fits$")


def find_i2d(filt, exposure_level=False):
    """i2d products for one filter, keyed by observation token (o135, ...).

    The observation comes from the OBSERVTN header keyword rather than from the
    filename, because the two stages name files differently (jw10678135001_...
    for image2, jw10678-o135_... for stage 3) and the header is the same in both.
    """
    from astropy.io import fits
    out = {}
    for p in glob.glob(f"{BASE}/{filt.upper()}/pipeline/**/*_i2d.fits",
                       recursive=True):
        b = os.path.basename(p)
        if _is_exposure_level(b) != exposure_level:
            continue
        if not exposure_level and not CANONICAL_I2D.match(b):
            continue
        try:
            obs = fits.getheader(p).get("OBSERVTN")
        except OSError:                       # still being written
            continue
        if not obs:
            continue
        key = f"o{int(obs):03d}"
        if exposure_level:
            out.setdefault(key, []).append(p)
            continue
        # Stage 3 emits one mosaic per MODULE: ..._clear-f480m-nrca_i2d.fits.
        # NIRCam modules A and B look at separate, non-overlapping sky, so each
        # is its own field (the same treatment crowded_l3 gets) rather than two
        # halves of one image.  A -merged product, if one appears, supersedes.
        mod = re.search(r"-(nrc[ab])_i2d\.fits$", b)
        if mod:
            key = f"{key}_{mod.group(1)}"
        if key in out and "merged" in os.path.basename(out[key]) \
                and "merged" not in b:
            continue
        out[key] = p
    if not exposure_level:
        # A -merged mosaic covers the same sky as its -nrca and -nrcb halves.
        # Keeping all three would build three RGBs of one pointing and paint
        # them over each other in the coadd, so merged supersedes the modules.
        merged = [k for k, v in out.items()
                  if "_nrc" not in k and "merged" in os.path.basename(v)]
        for key in merged:
            for mod in ("nrca", "nrcb"):
                out.pop(f"{key}_{mod}", None)
        # Fall back to the MAST stage 3 mosaic for any observation the local
        # image3 run has not produced.  Without this, an observation that has
        # been delivered but not locally re-reduced is invisible: no RGB is
        # built for it, and cmd_coadd retires any layer built for it by hand.
        # A local mosaic always wins, so this only ever fills gaps.
        prod = (f"miri_{filt.lower()}" if filt.lower() == MIRI_FILTER
                else f"nircam_clear-{filt.lower()}")
        for p in sorted(glob.glob(f"{MAST_L3}/jw*-o*_t*_{prod}/*_i2d.fits")):
            if not MAST_I2D.match(os.path.basename(p)):
                continue
            m = re.search(r"-(o\d+)_t\d+_", os.path.basename(p))
            if not m or m.group(1) in out:
                continue
            out[m.group(1)] = p
    return out


def inventory():
    inv = {f: find_i2d(f) for f in FILTERS}
    obs = sorted(set().union(*[set(v) for v in inv.values()]) if inv else [])
    return inv, obs


def cmd_list():
    inv, obs = inventory()
    print(f"{BASE}")
    for f in FILTERS:
        n_uncal = len(glob.glob(f"{BASE}/{f.upper()}/pipeline/**/*_uncal.fits",
                                recursive=True))
        n_rate = len(glob.glob(f"{BASE}/{f.upper()}/pipeline/**/*_rate.fits",
                               recursive=True))
        n_cal = len(glob.glob(f"{BASE}/{f.upper()}/pipeline/**/*_cal.fits",
                              recursive=True))
        n_exp = sum(len(v) for v in find_i2d(f, exposure_level=True).values())
        print(f"  {f.upper():7s} uncal={n_uncal:4d} rate={n_rate:4d} "
              f"cal={n_cal:4d} exp_i2d={n_exp:4d} mosaic_i2d={len(inv[f]):3d}")
    if not obs:
        print("\n  no stage 3 MOSAICS yet.  exp_i2d counts single-detector image2"
              "\n  products (2058x2058 chips); an RGB needs the stage 3 mosaic.")
        return 1
    print(f"\n  observations with i2d: {', '.join(obs)}")
    ready = [o for o in obs if all(o in inv[f] for f in FILTERS)]
    print(f"  complete in all {len(FILTERS)} filters: "
          f"{', '.join(ready) if ready else 'none'}")
    for o in ready:
        for f in FILTERS:
            print(f"    {o} {f.upper():7s} {inv[f][o]}")
    return 0


def build_obs(obs, avm_mode="raw", hips=True):
    """One observation -> RGB png + HiPS, returned as (png, hips_dir)."""
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.visualization import simple_norm
    from PIL import Image
    from reproject import reproject_interp
    import pyavm
    from jwst_rgb.save_rgb import save_rgb as _save_rgb
    from jwst_rgb.save_rgb import avm_for_saved_png
    Image.MAX_IMAGE_PIXELS = None

    inv, _ = inventory()
    missing = [f for f in FILTERS if obs not in inv[f]]
    if missing:
        raise RuntimeError(f"{obs}: no i2d for {missing}")

    os.makedirs(OUTDIR, exist_ok=True)
    tgt = fits.open(inv[TARGET_FILTER][obs])
    thdu = next(h for h in tgt if h.data is not None and h.data.ndim == 2)
    twcs = WCS(thdu.header).celestial
    ny, nx = thdu.data.shape
    print(f"{obs}: target grid {nx}x{ny} from {TARGET_FILTER.upper()}", flush=True)

    data = {}
    for f in FILTERS:
        if f == TARGET_FILTER:
            data[f] = thdu.data.astype(float)
            continue
        src = fits.open(inv[f][obs])
        shdu = next(h for h in src if h.data is not None and h.data.ndim == 2)
        print(f"  reprojecting {f.upper()} onto the {TARGET_FILTER.upper()} grid",
              flush=True)
        data[f], _ = reproject_interp(
            (shdu.data.astype(float), WCS(shdu.header).celestial),
            twcs, shape_out=(ny, nx))

    long_, short_ = data["f480m"], data["f212n"]
    # A pixel that is NaN in one channel but real in the others would otherwise
    # render as a flat false-coloured island; forbid the mixed state outright.
    nan = [np.isnan(long_), np.isnan(short_)]
    bad = np.logical_or.reduce(nan) & ~np.logical_and.reduce(nan)
    if bad.any():
        print(f"  masking {100 * bad.mean():.2f}% NaN-in-some-but-not-all pixels",
              flush=True)
        long_ = np.where(bad, np.nan, long_)
        short_ = np.where(bad, np.nan, short_)
    mid = np.nanmean(np.stack([long_, short_]), axis=0)

    chans = [long_, mid, short_]
    scaled = np.stack([np.nan_to_num(
        simple_norm(c, stretch="asinh", min_percent=1, max_percent=99.5)(c))
        for c in chans], axis=2)

    name = f"GCTreasury_{obs}_RGB_480-mean-212"  # obs may carry a _nrca/_nrcb module tag
    png = f"{OUTDIR}/{name}.png"
    # The AVM must describe the PNG as save_rgb writes it, not the input FITS:
    # flip=-1 plus ROTATE_180 leaves the array a FITS reader reconstructs
    # rotated by 180 degrees, so CRPIX has to be reflected on both axes.
    # AVM.from_header(thdu.header) left CRPIX at its FITS value and put every
    # tile |N+1-2*crpix| pixels off (1.26" for o112).
    avm = avm_for_saved_png(twcs, ny, nx, flip=-1, transpose=Image.ROTATE_180)
    if avm_mode == "rot180":
        from apply_cdmatrix_flip import cdmatrix_avm
        avm = cdmatrix_avm(twcs, ny, nx, "rot180")
    _save_rgb(np.clip(scaled, 0, 1), png, avm=avm, transpose=Image.ROTATE_180,
              alpha_only_edges=True, original_data=np.stack(chans, axis=2),
              hips=False)
    print(f"  wrote {png}", flush=True)

    hips_dir = None
    if hips:
        from tqdm import tqdm
        from reproject.hips import reproject_to_hips
        hips_dir = f"{OUTDIR}/{name}_hips"
        if os.path.exists(hips_dir):
            shutil.rmtree(hips_dir)
        reproject_to_hips(png, coord_system_out="galactic", level=None,
                          reproject_function=reproject_interp,
                          output_directory=hips_dir, threads=16,
                          progress_bar=tqdm)
        if not os.path.isdir(os.path.join(hips_dir, "Norder3")):
            raise RuntimeError(f"{obs}: build produced no Norder3")
        print(f"  wrote {hips_dir}", flush=True)
    return png, hips_dir


def check_orientation(hips_dir, src_fits):
    """Correlate served tiles against the source FITS, as-is vs rotated 180.

    A footprint check cannot catch a flip: rot180 about the field centre maps
    the footprint onto itself.  Only content can.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from astropy.io import fits
    from astropy.wcs import WCS
    from reproject import reproject_interp
    from check_hips_astrometry import HipsSampler

    p = {}
    for ln in open(os.path.join(hips_dir, "properties")):
        if "=" in ln:
            k, v = ln.split("=", 1)
            p[k.strip()] = v.strip()
    # 0.5"/px rather than 1": the tolerance below is 0.3", and on a 1" grid
    # even upsample_factor=20 puts the method's floor uncomfortably close to it
    # while resampling a 0.03"/px mosaic.  Same sky coverage, 4x the samples.
    n, scale = 800, 0.5
    gw = WCS(naxis=2)
    gw.wcs.crpix = [n / 2 + 0.5, n / 2 + 0.5]
    gw.wcs.cdelt = [-scale / 3600, scale / 3600]
    gw.wcs.crval = [float(p["hips_initial_ra"]), float(p["hips_initial_dec"])]
    gw.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    yy, xx = np.mgrid[0:n, 0:n]
    got = HipsSampler(hips_dir).sample(
        gw.pixel_to_world(xx.ravel(), yy.ravel())).astype(float).reshape(n, n)
    hdu = next(h for h in fits.open(src_fits)
               if h.data is not None and h.data.ndim == 2)
    ref, _ = reproject_interp((hdu.data.astype(float), WCS(hdu.header).celestial),
                              gw, shape_out=(n, n))

    def prep(a):
        a = np.array(a, float)
        m = np.isfinite(a) & (a > 0)
        if m.sum() < 50:
            return None
        a = np.log10(np.clip(a, np.nanpercentile(a[m], 5), None))
        a = np.where(np.isfinite(a), a, np.nanmedian(a[np.isfinite(a)]))
        a -= a.mean()
        return a / a.std() if a.std() > 0 else None

    A, B = prep(got), prep(ref)
    if A is None or B is None:
        print("  orientation: insufficient overlap to test")
        return None
    r_same, r_flip = float((A * B).mean()), float((A * B[::-1, ::-1]).mean())
    ok = r_same > r_flip
    print(f"  orientation: r_asis={r_same:+.4f} r_rot180={r_flip:+.4f}  "
          f"{'OK' if ok else 'FLIPPED'}", flush=True)

    # Orientation alone is not enough.  A wrong AVM reference pixel shifts the
    # tiles bodily without rotating them, and rot180 still loses that
    # comparison by a mile, so this check passed while every treasury layer sat
    # ~1.3" off.  Measure the translation too.
    try:
        from skimage.registration import phase_cross_correlation
    except ImportError:
        # Returning `ok` here would report the layer fine with the astrometry
        # gate silently skipped, which is the failure this check exists to
        # prevent.  None means "not checked".
        print("  WARNING: scikit-image unavailable; ASTROMETRY GATE SKIPPED -- "
              "the orientation result below does not cover translation")
        return None
    shift, _, _ = phase_cross_correlation(B, A, upsample_factor=20)
    off = float(np.hypot(*shift) * scale)          # gw is `scale` arcsec/px
    flag = "OK" if off <= ASTROMETRY_TOL_ARCSEC else "OFF"
    print(f"  astrometry: served tiles sit {off:.2f}\" from the source FITS "
          f"(tol {ASTROMETRY_TOL_ARCSEC}\")  {flag}", flush=True)
    if off > ASTROMETRY_TOL_ARCSEC:
        print(f"  WARNING: {os.path.basename(hips_dir)} is astrometrically "
              f"off by {off:.2f}\"; check the embedded AVM "
              f"(see avm_for_saved_png)", flush=True)
    return ok and off <= ASTROMETRY_TOL_ARCSEC


def png_for(obs):
    return f"{OUTDIR}/GCTreasury_{obs}_RGB_480-mean-212.png"


def miri_suffix(bgmatch):
    return "_bgmatch" if bgmatch else ""


def miri_png_for(obs, bgmatch=False):
    return f"{OUTDIR}/GCTreasury_{obs}_MIRI_F770W{miri_suffix(bgmatch)}.png"


def miri_hips_for(obs, bgmatch=False):
    return f"{OUTDIR}/GCTreasury_{obs}_MIRI_F770W{miri_suffix(bgmatch)}_hips"


MIRI_MATCH_JSON = f"{OUTDIR}/miri_background_match.json"


def miri_measure_pairs(paths):
    """Median (A - B) over the sky each overlapping pair shares.

    The 10678 MIRI parallel steps ~2.17' between pointings with a ~1.73' field
    radius, so consecutive tiles genuinely overlap and the offsets can be tied
    to each other rather than guessed from whole-image statistics.
    """
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.stats import sigma_clipped_stats
    from reproject import reproject_interp

    keys = sorted(paths)
    data, wcs_ = {}, {}
    for k in keys:
        hdu = next(h for h in fits.open(paths[k])
                   if h.data is not None and h.data.ndim == 2)
        data[k] = hdu.data.astype(float)
        wcs_[k] = WCS(hdu.header).celestial
    pairs = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            reB, _ = reproject_interp((data[b], wcs_[b]), wcs_[a],
                                      shape_out=data[a].shape)
            m = np.isfinite(data[a]) & np.isfinite(reB)
            if m.sum() < 5000:                      # too little shared sky
                continue
            _, med, _ = sigma_clipped_stats((data[a] - reB)[m], sigma=3.0,
                                            maxiters=5)
            pairs.append((a, b, float(med), int(m.sum())))
            print(f"  {a}-{b}: {m.sum():7d} shared px, median diff "
                  f"{med:+.4f}", flush=True)
    return keys, pairs


def miri_solve_offsets(keys, pairs):
    """Least-squares additive offsets from the pairwise differences.

    One equation per overlapping pair, off_a - off_b = median(A-B), plus a
    mean-zero constraint so the system is determined and the overall level of
    the mosaic is not dragged up or down.  Tiles with no overlap at all simply
    get zero, which is the honest answer -- nothing ties them to the rest.
    """
    idx = {k: i for i, k in enumerate(keys)}
    rows, rhs = [], []
    for a, b, d, n in pairs:
        r = np.zeros(len(keys))
        r[idx[a]], r[idx[b]] = 1.0, -1.0
        rows.append(r * np.sqrt(n))               # weight by shared area
        rhs.append(d * np.sqrt(n))
    rows.append(np.ones(len(keys)))               # mean-zero constraint
    rhs.append(0.0)
    off, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    return {k: float(off[i]) for k, i in idx.items()}


def cmd_miri_match():
    """Measure and store MIRI background offsets and one shared stretch.

    Two separate causes of seams get fixed here.  The tiles sit on different
    sky levels (49.4 to 81.6 MJy/sr across the first six, a 1.65x spread and
    ~4.9x the per-tile noise), and each was being stretched on its OWN
    percentiles, so identical sky rendered at different grey.  The offsets tie
    the levels together; the shared limits make one brightness mean one colour
    across the whole mosaic.
    """
    import json
    from astropy.io import fits

    import time
    paths = find_i2d(MIRI_FILTER)
    # Same settle rule the builds use.  The first run of this matcher read
    # mosaics while resample was still rewriting them and measured +64.94 for a
    # pair that is really +2.67 -- a 24x error that then propagated through the
    # least-squares solve into +-33 offsets across four tiles.  Erosion did not
    # change those numbers at all, so it was never an edge effect.
    fresh = {k: v for k, v in paths.items()
             if time.time() - os.path.getmtime(v) < SETTLE_SECONDS}
    for k in fresh:
        print(f"  skipping {k}: written in the last {SETTLE_SECONDS // 60} min")
        paths.pop(k)
    if len(paths) < 2:
        print("need at least two settled MIRI mosaics to match")
        return 1
    print(f"matching backgrounds across {len(paths)} MIRI mosaics")
    keys, pairs = miri_measure_pairs(paths)
    if not pairs:
        print("no overlapping pairs; nothing to tie together")
        return 1
    off = miri_solve_offsets(keys, pairs)
    print("\noffsets (subtracted from each tile):")
    for k in keys:
        print(f"  {k}: {off[k]:+.4f}")

    pooled = []
    for k in keys:
        hdu = next(h for h in fits.open(paths[k])
                   if h.data is not None and h.data.ndim == 2)
        d = hdu.data.astype(float) - off[k]
        v = d[np.isfinite(d)]
        pooled.append(v[:: max(1, v.size // 200000)])
    pooled = np.concatenate(pooled)
    lo, hi = np.percentile(pooled, [1.0, 99.5])
    print(f"\nshared stretch limits from the pooled, offset-corrected data: "
          f"{lo:.4f} .. {hi:.4f}")
    json.dump({"offsets": off, "vmin": float(lo), "vmax": float(hi),
               "pairs": [[a, b, d, n] for a, b, d, n in pairs]},
              open(MIRI_MATCH_JSON, "w"), indent=1)
    print(f"wrote {MIRI_MATCH_JSON}")
    return 0


def load_miri_match():
    import json
    if not os.path.exists(MIRI_MATCH_JSON):
        return None
    with open(MIRI_MATCH_JSON) as fh:
        return json.load(fh)


def build_miri_obs(obs, bgmatch=False, hips=True):
    """One observation's MIRI F770W mosaic -> monochrome png + HiPS.

    Grayscale rather than colour: 10678's MIRI parallel observes F770W alone, so
    there is nothing to make a colour from, and a single band rendered in three
    identical channels is the format a HiPS viewer expects.
    """
    from astropy.io import fits
    from astropy.visualization import simple_norm
    from PIL import Image
    from reproject import reproject_interp
    import pyavm
    from jwst_rgb.save_rgb import save_rgb as _save_rgb
    from jwst_rgb.save_rgb import avm_for_saved_png
    from astropy.wcs import WCS
    Image.MAX_IMAGE_PIXELS = None

    src = find_i2d(MIRI_FILTER).get(obs)
    if not src:
        raise RuntimeError(f"{obs}: no {MIRI_FILTER.upper()} mosaic")
    os.makedirs(OUTDIR, exist_ok=True)
    hdu = next(h for h in fits.open(src)
               if h.data is not None and h.data.ndim == 2)
    d = hdu.data.astype(float)
    ny, nx = d.shape
    print(f"{obs}: MIRI {MIRI_FILTER.upper()} grid {nx}x{ny}", flush=True)

    # Two renderings of the same data.  Without bgmatch, each tile is stretched
    # on its OWN percentiles, which is what a viewer sees today and what makes
    # the seams: equal sky brightness becomes unequal grey.  With bgmatch, the
    # tile's overlap-derived offset is removed and every tile is stretched on
    # one shared pair of cuts, so equal brightness becomes equal grey.
    if bgmatch:
        match = load_miri_match()
        if not match or obs not in match["offsets"]:
            raise RuntimeError(f"{obs}: no background match available; "
                               f"run --miri-match first")
        d = d - match["offsets"][obs]
        # this astropy's simple_norm takes vmin/vmax; min_cut/max_cut is the
        # older spelling and raises TypeError here
        g = simple_norm(d, stretch="asinh", vmin=match["vmin"],
                        vmax=match["vmax"], clip=True)(d)
        print(f"  background-matched: offset {match['offsets'][obs]:+.4f}, "
              f"shared cuts {match['vmin']:.3f}..{match['vmax']:.3f}", flush=True)
    else:
        g = simple_norm(d, stretch="asinh", min_percent=1, max_percent=99.5)(d)
    mono = np.stack([np.nan_to_num(g)] * 3, axis=2)
    png = miri_png_for(obs, bgmatch)
    avm = avm_for_saved_png(WCS(hdu.header).celestial, ny, nx,
                            flip=-1, transpose=Image.ROTATE_180)
    _save_rgb(np.clip(mono, 0, 1), png, avm=avm, transpose=Image.ROTATE_180,
              alpha_only_edges=True, original_data=np.stack([d] * 3, axis=2),
              hips=False)
    print(f"  wrote {png}", flush=True)

    hips_dir = None
    if hips:
        from tqdm import tqdm
        from reproject.hips import reproject_to_hips
        hips_dir = miri_hips_for(obs, bgmatch)
        if os.path.exists(hips_dir):
            shutil.rmtree(hips_dir)
        reproject_to_hips(png, coord_system_out="galactic", level=None,
                          reproject_function=reproject_interp,
                          output_directory=hips_dir, threads=16,
                          progress_bar=tqdm)
        if not os.path.isdir(os.path.join(hips_dir, "Norder3")):
            raise RuntimeError(f"{obs}: MIRI build produced no Norder3")
        print(f"  wrote {hips_dir}", flush=True)
    return png, hips_dir


def miri_needs_build(obs, src, bgmatch=False):
    import time
    if time.time() - os.path.getmtime(src) < SETTLE_SECONDS:
        return "SETTLING"
    if bgmatch:
        # Existence of the table is not enough: it carries offsets for the
        # fields --miri-match was last run over, and build_miri_obs raises on
        # any field missing from it.  Checking only the file meant every
        # uncovered field reported "no MIRI png yet", got built, and turned up
        # as a FAILED entry on every single tick -- noise that buries real
        # failures.  Report it as what it is.
        if not os.path.exists(MIRI_MATCH_JSON):
            return "NOMATCH"
        match = load_miri_match()
        if not match or obs not in match.get("offsets", {}):
            return "NOMATCH"
    png = miri_png_for(obs, bgmatch)
    if not os.path.exists(png):
        return "no MIRI png yet"
    hips = miri_hips_for(obs, bgmatch)
    if not os.path.isdir(os.path.join(hips, "Norder3")):
        return "png exists but its HiPS is missing or incomplete"
    if os.path.getmtime(src) > os.path.getmtime(png):
        return "F770W i2d is newer than the png (re-reduced?)"
    if not bgmatch:
        return None
    # A new background solution changes the offset and the shared stretch for
    # EVERY tile, not only the ones whose data moved, so the whole set has to
    # be re-rendered when the match is refreshed.  Source mtime alone cannot
    # catch this -- nor can it catch the source PATH changing, which is how the
    # daophot-model mix-up survived a rebuild.
    if os.path.exists(MIRI_MATCH_JSON) and \
            os.path.getmtime(MIRI_MATCH_JSON) > os.path.getmtime(png):
        return "background match is newer than the png"
    return None


def needs_build(obs, inv):
    """Is this observation missing an RGB, or is its RGB older than its data?

    The mtime comparison matters more than usual here: 10678 has no astrometric
    offsets table yet, so the current mosaics sit on the raw assign_wcs frame
    and are expected to be re-reduced once the tie is measured.  When that
    happens the i2d files are rewritten and every product built from them must
    be rebuilt -- silently serving the pre-tie version is the failure mode.
    """
    import time
    # A mosaic that changed in the last few minutes may still be being written;
    # building from it would produce a truncated RGB that then looks up to date.
    for f in FILTERS:
        src = inv[f].get(obs)
        if src and time.time() - os.path.getmtime(src) < SETTLE_SECONDS:
            return "SETTLING"
    png = png_for(obs)
    if not os.path.exists(png):
        return "no RGB yet"
    # The HiPS has to be checked separately.  A run killed between writing the
    # png and finishing the pyramid leaves a png NEWER than its sources, which
    # every later tick then reads as up to date -- so the observation silently
    # never reaches the coadd.  That happened to o135_nrcb.
    hips = f"{OUTDIR}/GCTreasury_{obs}_RGB_480-mean-212_hips"
    if not os.path.isdir(os.path.join(hips, "Norder3")):
        return "RGB exists but its HiPS is missing or incomplete"
    t_png = os.path.getmtime(png)
    for f in FILTERS:
        src = inv[f].get(obs)
        if src and os.path.getmtime(src) > t_png:
            return f"{f.upper()} i2d is newer than the RGB (re-reduced?)"
    return None


def _pending_summary():
    """What an unblocked --auto tick would build right now, as display lines.

    Read-only: used to report work stuck behind a held lock.
    """
    inv, obs_all = inventory()
    out = []
    for o in [o for o in obs_all if all(o in inv[f] for f in FILTERS)]:
        why = needs_build(o, inv)
        if why and why != "SETTLING":
            out.append(f"{o} NIRCam -- {why}")
    miri = find_i2d(MIRI_FILTER)
    for bgmatch in (False, True):
        tag = "MIRI+bg" if bgmatch else "MIRI"
        for o, src in sorted(miri.items()):
            why = miri_needs_build(o, src, bgmatch)
            if why and why not in ("SETTLING", "NOMATCH"):
                out.append(f"{o} {tag} -- {why}")
    return out


def cmd_auto(publish=False):
    """Build whatever is buildable and not yet built, then recoadd if anything
    changed.  Safe to run on a schedule: a lock file keeps a slow build from
    overlapping the next tick, and nothing is rebuilt unless its source moved.
    """
    import time
    lock = f"{OUTDIR}/.auto.lock"
    os.makedirs(OUTDIR, exist_ok=True)
    if os.path.exists(lock):
        age = time.time() - os.path.getmtime(lock)
        if age < 6 * 3600:
            # Say what is WAITING, not just that we are blocked.  A long
            # rebuild holding this lock starves the cron silently: every tick
            # printed one line and exited, so three observations whose L3
            # landed mid-rebuild sat unbuilt for hours and the only trace was
            # needs_build reporting "no RGB yet" to nobody.  A held lock with
            # nothing pending is routine; a held lock with work queued behind
            # it is worth seeing in the log.
            print(f"another run holds the lock ({age / 60:.0f} min old); exiting")
            try:
                print(f"  {lock} says: {open(lock).read().strip()}")
                pending = _pending_summary()
            except (OSError, KeyError, ValueError, TypeError) as exc:
                print(f"  (could not summarise pending work: "
                      f"{type(exc).__name__}: {exc})")
            else:
                if pending:
                    print(f"  WAITING ON THE LOCK: {len(pending)} item(s)")
                    for line in pending:
                        print(f"    {line}")
                else:
                    print("  nothing pending behind it")
            return 0
        print(f"stale lock ({age / 3600:.1f} h old); taking it")
    open(lock, "w").write(f"{os.getpid()} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    try:
        inv, obs_all = inventory()
        ready = [o for o in obs_all if all(o in inv[f] for f in FILTERS)]
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
              f"{len(ready)} observation(s) complete in all filters")
        built, failed = [], []
        for o in ready:
            why = needs_build(o, inv)
            if why == "SETTLING":
                print(f"  {o}: source written in the last "
                      f"{SETTLE_SECONDS // 60} min; leaving it to settle")
                continue
            if not why:
                print(f"  {o}: up to date")
                continue
            print(f"  {o}: building -- {why}", flush=True)
            try:
                _, hd = build_obs(o)
            except Exception as exc:                       # keep going on the rest
                print(f"  {o}: FAILED {type(exc).__name__}: {exc}", flush=True)
                failed.append(o)
                continue
            if hd:
                try:
                    check_orientation(hd, inv[TARGET_FILTER][o])
                except (RuntimeError, OSError, ValueError, TypeError,
                        KeyError) as exc:
                    print(f"  {o}: orientation check unavailable "
                          f"({type(exc).__name__}: {exc})", flush=True)
            built.append(o)
        # MIRI parallel: its own field, its own monochrome layers, its own
        # coadds.  Two flavours of every tile -- plain and background-matched --
        # so the two mosaics can be compared directly on the same sky.
        miri = find_i2d(MIRI_FILTER)
        print(f"  {len(miri)} MIRI {MIRI_FILTER.upper()} mosaic(s)")
        miri_built = {False: [], True: []}
        for bgmatch in (False, True):
            tag = "MIRI+bg" if bgmatch else "MIRI"
            for o, src in sorted(miri.items()):
                why = miri_needs_build(o, src, bgmatch)
                if why == "SETTLING":
                    print(f"  {o} {tag}: source written in the last "
                          f"{SETTLE_SECONDS // 60} min; leaving it to settle")
                    continue
                if why == "NOMATCH":
                    print(f"  {o} {tag}: no background match on disk; "
                          f"run --miri-match")
                    continue
                if not why:
                    print(f"  {o} {tag}: up to date")
                    continue
                print(f"  {o} {tag}: building -- {why}", flush=True)
                try:
                    _, hd = build_miri_obs(o, bgmatch=bgmatch)
                except (RuntimeError, OSError, ValueError,
                        TypeError, KeyError) as exc:
                    print(f"  {o} {tag}: FAILED {type(exc).__name__}: {exc}",
                          flush=True)
                    failed.append(f"{o}:{tag}")
                    continue
                if hd:
                    try:
                        check_orientation(hd, src)
                    except (RuntimeError, OSError, ValueError, TypeError,
                            KeyError) as exc:
                        print(f"  {o} {tag}: orientation check unavailable "
                              f"({type(exc).__name__}: {exc})", flush=True)
                miri_built[bgmatch].append(o)

        if built:
            print(f"built {len(built)}: {', '.join(built)} -- recoadding NIRCam")
            cmd_coadd()
            if publish:
                cmd_publish()
        else:
            print("nothing new to build; NIRCam coadd left alone")
        for bgmatch in (False, True):
            name = MIRI_BGMATCH_COADD_NAME if bgmatch else MIRI_COADD_NAME
            if miri_built[bgmatch]:
                print(f"built {len(miri_built[bgmatch])} for {name}: "
                      f"{', '.join(miri_built[bgmatch])} -- recoadding")
                cmd_coadd(miri=True, bgmatch=bgmatch)
            else:
                print(f"nothing new for {name}; left alone")
        if failed:
            print(f"FAILED: {', '.join(failed)}")
            return 1
        return 0
    finally:
        if os.path.exists(lock):
            os.remove(lock)


def cmd_publish():
    """Copy the coadd and every per-observation HiPS into the web tree.

    Staged: build into <name>.new, move the old aside, move the new in.  The
    destination is a live web root, so a direct copy would serve a partial
    pyramid for the duration.
    """
    import time
    # Seconds, not just the date: with a date-only stamp a second run on the
    # same day finds {n}_stale_{stamp} already there and shutil.move puts the
    # live tree INSIDE it rather than beside it.
    stamp = time.strftime("%Y%m%dT%H%M%S")
    src = [f"{OUTDIR}/{COADD_NAME}"] + sorted(
        glob.glob(f"{OUTDIR}/GCTreasury_*_RGB_480-mean-212_hips"))
    for s in src:
        if not os.path.isdir(os.path.join(s, "Norder3")):
            print(f"  skipping {os.path.basename(s)}: no Norder3")
            continue
        n = os.path.basename(s)
        dest, stage = f"{WEB}/{n}", f"{WEB}/{n}.new"
        shutil.rmtree(stage, ignore_errors=True)
        shutil.copytree(s, stage)
        if os.path.isdir(dest):
            aside = f"{WEB}/{n}_stale_{stamp}"
            if os.path.exists(aside):                  # same-second re-run
                shutil.rmtree(aside, ignore_errors=True)
            shutil.move(dest, aside)
        shutil.move(stage, dest)
        _prune_stale(n)
        print(f"  published {n}")
    return 0


# How many superseded copies of a layer to keep in the web root.  These are
# full tile pyramids and there are 50+ layers, so without a cap every publish
# leaves another complete copy behind in a live web root.
KEEP_STALE = 1


def _prune_stale(name):
    """Drop all but the newest KEEP_STALE `<name>_stale_*` trees."""
    old = sorted(glob.glob(f"{WEB}/{name}_stale_*"))
    for d in old[:-KEEP_STALE] if KEEP_STALE else old:
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
            print(f"    pruned {os.path.basename(d)}")


def _read_props(d):
    out = {}
    fn = os.path.join(d, "properties")
    if not os.path.exists(fn):
        return out
    for ln in open(fn):
        if "=" in ln and not ln.startswith("#"):
            k, v = ln.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def set_union_view(coadd_dir, layers):
    """Point the coadd's default view at ALL of its layers, not one of them.

    coadd_hips copies hips_initial_ra/dec/fov straight from a single input, so
    loading the mosaic drops the viewer onto whichever tile that happened to be
    -- with 139 pointings eventually, that means opening the survey zoomed into
    one field and seeing nothing else.  Replace it with the bounding circle of
    every layer: mean direction of the centres, radius large enough to contain
    each layer's own field of view.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    cen, fovs = [], []
    for L in layers:
        pr = _read_props(L)
        try:
            cen.append(SkyCoord(float(pr["hips_initial_ra"]),
                                float(pr["hips_initial_dec"]), unit="deg"))
            fovs.append(float(pr["hips_initial_fov"]))
        except (KeyError, ValueError):
            continue
    if not cen:
        print("  no layer view metadata; leaving the default view alone")
        return
    sc = SkyCoord(cen)
    v = np.stack([sc.cartesian.x.value, sc.cartesian.y.value,
                  sc.cartesian.z.value]).mean(axis=1)
    n = np.linalg.norm(v)
    if n == 0:
        print("  layers are antipodal; leaving the default view alone")
        return
    ctr = SkyCoord(*(v / n), representation_type="cartesian").spherical
    ctr = SkyCoord(ctr.lon, ctr.lat)
    radius = max(float(ctr.separation(c).deg) + f / 2.0
                 for c, f in zip(cen, fovs))
    fov = 2.0 * radius
    fn = os.path.join(coadd_dir, "properties")
    lines, seen = [], set()
    repl = {"hips_initial_ra": f"{ctr.ra.deg:.10f}",
            "hips_initial_dec": f"{ctr.dec.deg:.10f}",
            "hips_initial_fov": f"{fov:.10f}"}
    for ln in open(fn).read().rstrip("\n").split("\n"):
        k = ln.split("=", 1)[0].strip() if "=" in ln else None
        if k in repl:
            lines.append(f"{k:20s} = {repl[k]}")
            seen.add(k)
        else:
            lines.append(ln)
    for k, v_ in repl.items():
        if k not in seen:
            lines.append(f"{k:20s} = {v_}")
    open(fn, "w").write("\n".join(lines) + "\n")
    print(f"  default view set to cover {len(cen)} layer(s): "
          f"{ctr.ra.deg:.5f} {ctr.dec.deg:+.5f} fov={fov:.5f} deg")


def cmd_coadd(miri=False, bgmatch=False):
    """Coadd every per-observation HiPS into one growing mosaic.

    NIRCam and MIRI are coadded separately: the two point at different sky and
    coadd_hips paints last-wins, so mixing them would let whichever came last
    overwrite the other wherever they happen to touch.
    """
    from reproject.hips import coadd_hips
    if miri:
        pat = f"GCTreasury_*_MIRI_F770W{miri_suffix(bgmatch)}_hips"
    else:
        pat = "GCTreasury_*_RGB_480-mean-212_hips"
    layers = sorted(glob.glob(f"{OUTDIR}/{pat}"))
    if miri and not bgmatch:
        # the plain glob also matches the _bgmatch layers; keep them apart
        layers = [L for L in layers if "_bgmatch_hips" not in L]
    # Retire layers whose source is no longer current -- chiefly the per-module
    # halves once a -merged mosaic supersedes them.  Renamed, never deleted.
    active = set(find_i2d(MIRI_FILTER) if miri else inventory()[1])
    keep = []
    for L in layers:
        b = os.path.basename(L)
        tail = (f"_MIRI_F770W{miri_suffix(bgmatch)}_hips" if miri
                else "_RGB_480-mean-212_hips")
        key = b[len("GCTreasury_"):-len(tail)]
        if key in active:
            keep.append(L)
        else:
            dest = L + "_superseded"
            shutil.rmtree(dest, ignore_errors=True)
            shutil.move(L, dest)
            print(f"  retired {b} (superseded)")
    layers = keep
    if not layers:
        print(f"no per-observation {'MIRI ' if miri else ''}HiPS to coadd")
        return 1
    if miri:
        out = f"{OUTDIR}/{MIRI_BGMATCH_COADD_NAME if bgmatch else MIRI_COADD_NAME}"
    else:
        out = f"{OUTDIR}/{COADD_NAME}"
    if os.path.exists(out):
        shutil.rmtree(out)
    print(f"coadding {len(layers)} observation HiPS -> {out}")
    for L in layers:
        print(f"  {os.path.basename(L)}")
    coadd_hips(layers, out)
    set_union_view(out, layers)
    print(f"done: {out}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true",
                    help="report what data exists and exit")
    ap.add_argument("--obs", help="build one observation, e.g. o135")
    ap.add_argument("--all", action="store_true",
                    help="build every observation complete in all filters")
    ap.add_argument("--coadd", action="store_true",
                    help="rebuild the combined mosaic from per-observation HiPS")
    ap.add_argument("--miri", action="store_true",
                    help="with --coadd, rebuild the MIRI mosaic instead")
    ap.add_argument("--bgmatch", action="store_true",
                    help="with --coadd --miri, rebuild the background-matched "
                         "MIRI mosaic instead of the plain one")
    ap.add_argument("--miri-match", action="store_true",
                    help="measure MIRI background offsets + shared stretch")
    ap.add_argument("--auto", action="store_true",
                    help="build anything new or stale, then recoadd (for cron)")
    ap.add_argument("--publish", action="store_true",
                    help="with --auto, also install into the avm_images web tree")
    ap.add_argument("--avm", choices=("raw", "rot180"), default="raw",
                    help="AVM form; raw is the documented path (default)")
    ap.add_argument("--no-hips", action="store_true", help="png only")
    a = ap.parse_args()

    if a.list:
        return cmd_list()
    if a.miri_match:
        return cmd_miri_match()
    if a.auto:
        return cmd_auto(publish=a.publish)
    if a.coadd:
        return cmd_coadd(miri=a.miri, bgmatch=a.bgmatch)

    inv, obs = inventory()
    targets = ([a.obs] if a.obs else
               [o for o in obs if all(o in inv[f] for f in FILTERS)])
    if not targets:
        print("nothing to build: no observation has i2d in every filter yet")
        return 1
    for o in targets:
        png, hd = build_obs(o, avm_mode=a.avm, hips=not a.no_hips)
        if hd:
            check_orientation(hd, inv[TARGET_FILTER][o])
    return 0


if __name__ == "__main__":
    sys.exit(main())
