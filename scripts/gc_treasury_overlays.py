#!/usr/bin/env python
"""Catalogue-derived overlays for the GC Treasury Aladin view.

All built from the per-observation vetted daophot catalogues and all
regenerated together whenever those catalogues change:

  jwst-red-stars-hips          density of F212N-F480M > 0, F480M < 18 (AB)
  jwst-rc-blue-hips            red-clump stars bluer than the GC average
  jwst-rc-red-hips             red-clump stars redder than the GC average
  jwst_ultrared_stars          catalogue (ecsv/fits/json) of F212N-F480M > 4
  jwst-star-density-hips       all F212N sources, saturated included
  jwst-star-density-f480m-hips all F480M sources, saturated included
  jwst-star-density-f212n-cube-hips
                               cube: F212N sources in 1-mag bins, from the
                               saturation limit to the confusion limit
  jwst-star-density-colour-cube-hips
                               cube: F212N-F480M in 0.5-mag colour bins
                               (star counts by reddening: pseudo-extinction)
  jwst-median-colour-hips      median F212N-F480M per cell (pseudo-extinction)
  jwst-stars-colour-hips       F212N-F480M at every pixel: the inverse-distance
                               weighted mean colour of the nearest stars
  jwst-stars-catalog-hips      the same stars as a HiPS catalogue

Saturated stars
---------------
Each field's catalogue carries saturated stars as `is_saturated` rows.  As of
2026-09-24 those rows are the SURVEY-WIDE saturated-star list, not the field's
own: o118 and o124 each hold the same 278,998 of them, 99.9% at identical
positions, spanning 1.22 deg of RA against a field's 0.046 deg.  Taken as they
are, every saturated star in the survey is counted once per field -- up to 43
times in the 2026-09-23 build, and 72% of the red-star selection was such
copies.  `own_footprint` keeps a saturated row only near the field's own
unsaturated detections, and `dedupe_across_fields` then removes a source seen
by two overlapping fields.  Both stay necessary after an upstream fix: the
second handles genuine overlaps, which were always double-counted.

These began as one-off scripts operating on a cached match table.  They are
consolidated here so that new fields are picked up on their own: the expensive
step is the cross-filter positional match, which is cached under CACHE and
keyed on a fingerprint of the input catalogues, so a run with nothing new costs
one directory listing.

Photometry
----------
`flux` in these catalogues is a sum of pixels each in MJy/sr, so

    flux_Jy = flux * MJy/sr * pixel_solid_angle

using each catalogue's OWN PIXSCALE -- 0.0312" for the short-wave F212N
detectors and 0.0629" for the long-wave F480M ones.  That is a factor of 4 in
solid angle, so 1.5 mag if taken from the wrong table.  AB magnitudes use the
3631 Jy definition rather than the SVO Vega zeropoint merge_catalogs.py uses,
because every selection here is specified in AB.

Red-clump selection
-------------------
The reddening slope is measured from these data (0.890 in (F212N-F480M,
F480M), implying A_F480M/A_F212N = 0.471) rather than taken from
extinction_law/RESULTS.md (0.306 +- 0.103, slope 0.441).  The two disagree, and
a cut drawn at 0.441 slices across the observed ridge instead of along it.
This is therefore an EMPIRICAL selection, not the Bravo Ferres et al. (2025)
cut, and no extinction ratio should be quoted back from it -- a
reddening-aligned box returns its own slope.

    W    = F480M - SLOPE*(F212N-F480M)     constant along the reddening vector
    RC   : |W - WRC| < HW
    blue : colour <  SPLIT
    red  : colour >= SPLIT

The cut constants are held FIXED as new fields arrive so the layers stay
comparable between releases; --refit re-derives the ridge from whatever is on
disk and prints it without writing anything.  Every run reports the current
ridge so drift away from the fixed values is visible.
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys
import time

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter
from jwst_rgb.hips_naming import properties_for
from jwst_rgb.hips_formats import _read_properties

CAT = "/orange/adamginsburg/jwst/gc-treasury/catalogs"
OUT = "/orange/adamginsburg/jwst/gc-treasury/pngs"
WEB = "/orange/adamginsburg/web/public/avm_images"
CACHE = f"{OUT}/.overlay_match_cache.npz"
STAMP = f"{OUT}/.overlay_fingerprint.json"
LOCK = f"{OUT}/.overlays.lock"
# The viewer is served from starformation, which is a separate tree from the
# data.rc web root; publishing to WEB alone leaves the overlays invisible there.
REMOTE = ("starformation:/h/cnswww-starformation.astro/"
          "starformation.astro.ufl.edu/htdocs/avm_images/")
LAYERS = ["jwst-red-stars-hips", "jwst-rc-blue-hips", "jwst-rc-red-hips",
          "jwst-star-density-hips", "jwst-star-density-f480m-hips",
          "jwst-star-density-f212n-cube-hips",
          "jwst-star-density-colour-cube-hips", "jwst-median-colour-hips",
          "jwst-stars-colour-hips", "jwst-stars-catalog-hips"]
CATALOGUE_FILES = ["jwst_ultrared_stars.ecsv", "jwst_ultrared_stars.fits",
                   "jwst_ultrared_stars.json"]

AB_ZP_JY = 3631.0
MATCH_ARCSEC = 0.1
#: A field whose F212N and F480M catalogues sit further apart than this
#: (bulk offset, see `field_offset`) is left out of the matched products: at
#: MATCH_ARCSEC most of its matches would be chance pairs.  Its F212N and
#: F480M sources still count in the density maps.  o063 and o113 (F212N
#: 0.15-0.27" off F480M, jwst-gc-pipeline #921) trip it until reprocessed.
MAX_FIELD_OFFSET_ARCSEC = 0.1

# red + bright density layer
RED_COLOUR, RED_MAGLIMIT = 0.0, 18.0
# red-clump band and split (see module docstring -- empirical, held fixed)
SLOPE, WRC, HW, SPLIT = 0.890, 17.50, 0.9, -0.325
RC_M480_RANGE, RC_COLOUR_RANGE = (14.0, 19.0), (-2.5, 3.5)
#: em_min/em_max (m) of each cube: one filter's pivot +/- bandwidth/2 from
#: the JDox NIRCam filter table (approximate).  Aladin Lite cannot open a
#: HiPS cube without it, and its cube slider needs a narrow band to reach
#: every frame, so the colour cube carries the F480M band alone
#: (see jwst_rgb.hips_formats).
F212N_EM_RANGE = (2.108e-6, 2.135e-6)
F480M_EM_RANGE = (4.662e-6, 4.966e-6)
# ultra-red catalogue
ULTRARED_CUT = 4.0
# density grid
PIXEL_ARCSEC, SMOOTH_ARCSEC = 2.0, 6.0

# Saturated-row guard and cross-field dedupe (see module docstring).  The
# footprint radius is generous because a bright star's own core and wings
# suppress unsaturated detections for a few arcsec around it; a copy from
# another field that survives near the edge is a real star and is then caught
# by the dedupe, if its own field is in the set.
SAT_FOOTPRINT_ARCSEC = 10.0
DEDUPE_ARCSEC = 0.1
#: Bumped whenever the cached arrays change meaning, so an old cache (built
#: without the guard) is never read back as if it had one.  3: the cache also
#: holds every F480M source, for the F480M density map.  4: fields with an
#: F212N-F480M offset over MAX_FIELD_OFFSET_ARCSEC are left out of matched.
CACHE_VERSION = 4

# F212N magnitude cube.  Saturation: the faint end of the saturated rows and
# the bright end of the unsaturated ones meet near 17-18 AB.  Confusion: the
# turnover of the unsaturated luminosity function, ~22 AB in the o135 test
# field.  Held fixed so frames mean the same thing between releases; every
# build reports the measured values next to them (`report_limits`).
F212N_SAT_LIMIT, F212N_CONFUSION_LIMIT = 17.0, 22.0
# Colour cube: 0.5-mag bins over the 0.5-99.5 percentile range of the
# 2026-09-23 matched set (-1.5 .. 2.6).  Restricted to F480M brighter than
# COLOUR_MAGLIMIT so the counts are not dominated by completeness, which
# itself varies with extinction.
COLOUR_EDGES = np.arange(-1.5, 3.01, 0.5)
COLOUR_MAGLIMIT = 20.0
# median-colour map: cell size and minimum stars per cell
MEDIAN_PIXEL_ARCSEC, MEDIAN_MIN_STARS = 4.0, 5
# star rendering
#: colour field: 2x the F480M (long-wave) pixel, the neighbour count, and the
#: largest k-th neighbour distance that still counts as covered.
KNN_PIXEL_ARCSEC, KNN_NEIGHBORS, KNN_MAX_ARCSEC = 0.126, 15, 10.0
STAR_COLOUR_RANGE = (-1.5, 2.5)
STAR_CMAP = "RdYlBu_r"
#: F212N AB -> disc radius (render pixels) and alpha (0-255), interpolated.
STAR_RADIUS_MAG, STAR_RADIUS_PIX = (8.0, 13.0, 17.0, 23.0), (7.0, 4.0, 2.0, 1.0)
STAR_ALPHA_MAG, STAR_ALPHA = (10.0, 23.0), (255.0, 90.0)
CATALOG_ROWS_PER_TILE = 500

_GRID = (f"on a {PIXEL_ARCSEC:g}\" grid smoothed by a Gaussian of sigma "
         f"{SMOOTH_ARCSEC:g}\"")
_CATALOGUES = ("From the per-field vetted DAOPHOT catalogs of JWST GO "
               f"10678, F212N and F480M matched within {MATCH_ARCSEC:g}\"")
_SOURCE = f"{_CATALOGUES}; saturated stars included, each counted once."
_RC_BAND = (f"|F480M - {SLOPE:g}(F212N-F480M) - {WRC:g}| < {HW:g}, "
            f"{RC_M480_RANGE[0]:g} < F480M < {RC_M480_RANGE[1]:g}, "
            f"{RC_COLOUR_RANGE[0]:g} < F212N-F480M < {RC_COLOUR_RANGE[1]:g}")
#: `obs_description` for each layer.  HiPS 1.0 has no pixel-unit keyword, so
#: the unit is stated here, where viewers and hipslist readers show it.
LAYER_DESCRIPTIONS = {
    "jwst-red-stars-hips":
        "Pixel value: surface density in stars/arcmin^2 of stars with "
        f"F212N-F480M > {RED_COLOUR:g} and F480M < {RED_MAGLIMIT:g} (AB), "
        f"{_GRID}. {_SOURCE}",
    "jwst-rc-blue-hips":
        "Pixel value: surface density in stars/arcmin^2 of red-clump stars "
        f"({_RC_BAND}) bluer than "
        f"F212N-F480M = {SPLIT:g}, {_GRID}. {_SOURCE}",
    "jwst-rc-red-hips":
        "Pixel value: surface density in stars/arcmin^2 of red-clump stars "
        f"({_RC_BAND}) at or redder "
        f"than F212N-F480M = {SPLIT:g}, {_GRID}. {_SOURCE}",
    "jwst-star-density-hips":
        "Pixel value: surface density in stars/arcmin^2 of all F212N "
        f"sources, {_GRID}. {_SOURCE}",
    "jwst-star-density-f480m-hips":
        "Pixel value: surface density in stars/arcmin^2 of all F480M "
        f"sources, {_GRID}. {_SOURCE}",
    "jwst-star-density-f212n-cube-hips":
        "Pixel value: surface density in stars/arcmin^2 of F212N sources in "
        f"1-mag bins from F212N = {F212N_SAT_LIMIT:g} (saturation) to "
        f"{F212N_CONFUSION_LIMIT:g} (confusion) AB, {_GRID}; the cube axis "
        f"is the bin center in F212N AB mag. {_SOURCE}",
    "jwst-star-density-colour-cube-hips":
        "Pixel value: surface density in stars/arcmin^2 of stars with F480M "
        f"< {COLOUR_MAGLIMIT:g} AB in {COLOUR_EDGES[1] - COLOUR_EDGES[0]:g}-mag "
        f"bins of F212N-F480M from {COLOUR_EDGES[0]:g} to "
        f"{COLOUR_EDGES[-1]:g}, {_GRID}; the cube axis is the bin center in "
        "AB mag. Redder bins trace higher extinction (a pseudo-extinction "
        f"map, not calibrated to A_V). {_SOURCE}",
    "jwst-median-colour-hips":
        "Pixel value: median F212N-F480M in AB mag of unsaturated stars with "
        f"F480M < {COLOUR_MAGLIMIT:g} AB, in {MEDIAN_PIXEL_ARCSEC:g}\" cells "
        f"holding at least {MEDIAN_MIN_STARS} stars (blank otherwise). Higher "
        "values mean more reddening (a pseudo-extinction map, not calibrated "
        f"to A_V). {_CATALOGUES}.",
    "jwst-stars-colour-hips":
        "F212N-F480M at every pixel: the inverse-distance weighted mean color "
        f"of the {KNN_NEIGHBORS} nearest unsaturated matched stars on a "
        f"{KNN_PIXEL_ARCSEC:g}\" grid, shown in {STAR_CMAP} over "
        f"{STAR_COLOUR_RANGE[0]:g} to {STAR_COLOUR_RANGE[1]:g} AB mag; blank "
        f"where the {KNN_NEIGHBORS}th star is more than {KNN_MAX_ARCSEC:g}\" "
        "away. Pixel values are display colors, not flux. "
        f"{_CATALOGUES}.",
    "jwst-stars-catalog-hips":
        "Matched stars, brightest F212N first. Columns: ra, dec (deg, ICRS), "
        "f212n, f480m, color (F212N-F480M) in AB mag, saturated (1 if "
        "saturated in either filter), obs (10678 observation), rgb (the "
        f"star's color in the jwst-stars-colour-hips color map). {_SOURCE}",
}


def layer_properties(name):
    """`properties_for(name)` plus the layer's obs_description."""
    extra = {}
    if name in LAYER_DESCRIPTIONS:
        extra["obs_description"] = LAYER_DESCRIPTIONS[name]
    return properties_for(name, **extra)

# The qualifier group is the point: reductions gain tags over time
# (resbgsub, ...) and a pattern that does not allow for them does not fail --
# it silently selects an older iteration.
FILTERS_USED = ("f212n", "f480m")

CAT_RE = re.compile(
    r"(f\d+[a-z])_(merged|nrca|nrcb)_(o\d+)_indivexp_merged_"
    r"((?:[a-z0-9]+_)*)"                      # optional qualifiers, e.g. resbgsub_
    r"m(\d+)_dao_basic(_vetted)?\.fits$")

#: A whole-tile catalogue beats a single module's; `merged` covers both NIRCam
#: modules, `nrca`/`nrcb` half the tile each.  Seven Treasury observations have
#: been cataloged in module A only, so requiring `merged` excludes them
#: entirely rather than showing half of each.
MODULE_RANK = {"merged": 2, "nrca": 1, "nrcb": 1}


# --------------------------------------------------------------------------
# inputs


def latest_pairs(allow_unvetted=False, allow_module=False):
    """{obs: {filter: (iteration, path, qualifiers, vetted)}} keeping the
    highest m<N> per filter, over observations with BOTH filters.

    Vetting, not reduction stage, is what gates this list.  Measured
    2026-09-18: 19 observations have a catalogue in both F212N and F480M at
    some stage, and 12 have one VETTED in both -- the other seven are held out
    by a missing F480M vetting pass, at m2, while their F212N side is vetted.
    `allow_unvetted` takes the unvetted catalogue for a filter that has no
    vetted one, and every source it contributes is marked `vetted = False` so
    the distinction survives into the published file.

    `allow_module` additionally takes a SINGLE-MODULE catalogue where the
    whole-tile `merged` one does not exist.  Seven observations (o105, o109,
    o112, o116, o118, o133, o138) have been cataloged in module A only, so
    without this they contribute nothing at all rather than half a tile each;
    with it their sources carry `module = 'nrca'`.

    A vetted catalogue always wins over an unvetted one at the SAME iteration;
    a higher iteration wins over a lower one either way, because the iteration
    is the reduction and the vetting is a pass over it.
    """
    have, have_rank = {}, {}
    # Glob broadly and let CAT_RE decide.  The old pattern required "merged_m"
    # to be adjacent, so ..._indivexp_merged_resbgsub_m5_... was filtered out
    # before the regex ever saw it -- fixing the regex alone changed nothing.
    for f in glob.glob(f"{CAT}/f*_*_o*_indivexp_merged_*dao_basic*.fits"):
        m = CAT_RE.search(os.path.basename(f))
        if not m:
            continue
        filt, module, obs = m.group(1), m.group(2), m.group(3)
        qual, it, vetted = m.group(4), int(m.group(5)), bool(m.group(6))
        if not (vetted or allow_unvetted):
            continue
        if module != "merged" and not allow_module:
            continue
        d = have.setdefault(obs, {})
        # Rank: whole tile over a single module first, then the iteration
        # number, then vetting.  A qualifier is a variant of an iteration, not
        # a competitor to it.  Vetting breaks a TIE rather than outranking an
        # iteration -- a later reduction is a different measurement, a vetting
        # pass is a filter over one -- so o137's F480M nrca, vetted at m2 and
        # unvetted at m3, selects m3 when both fallbacks are on.  Whatever the
        # flags, neither can demote a catalogue a stricter run would have
        # picked: a merged file outranks every module file and a vetted one
        # wins its own iteration.
        key = (MODULE_RANK.get(module, 0), it, vetted)
        ranks = have_rank.setdefault(obs, {})
        if filt not in ranks or key > ranks[filt]:
            ranks[filt] = key
            # The published shape stays a 3-tuple: callers unpack it as
            # `it, path, qual`, and `provenance` reads module and vetting back
            # out of the filename rather than widening it.
            d[filt] = (it, f, qual.rstrip("_"))
    return {o: v for o, v in sorted(have.items())
            if "f212n" in v and "f480m" in v}


def provenance(path):
    """``{'module': ..., 'vetted': ...}`` for one catalogue, from its name.

    Module and vetting are spelled in the filename, so they need no extra
    channel out of `latest_pairs` and no extra column in the match cache; a
    consumer that has the path has the provenance.
    """
    m = CAT_RE.search(os.path.basename(path))
    if not m:
        return {"module": "unknown", "vetted": "unknown"}
    return {"module": m.group(2), "vetted": "yes" if m.group(6) else "no"}


def pair_provenance(pairs):
    """``{obs: {'module': ..., 'vetted': ...}}`` over a selected set.

    A source's colour is a difference of two catalogues, so the pair is only
    as good as its weaker half: `vetted` is "yes" only when both were, and
    `module` is "merged" only when both cover the whole tile.
    """
    out = {}
    for obs, v in pairs.items():
        halves = [provenance(v[f][1]) for f in FILTERS_USED if f in v]
        mods = {h["module"] for h in halves}
        out[obs] = {
            "module": mods.pop() if len(mods) == 1 else "+".join(sorted(mods)),
            "vetted": "yes" if all(h["vetted"] == "yes" for h in halves)
                      else "no",
        }
    return out


def lineage_of(pairs):
    """{lineage: [obs, ...]} over the selected set.

    A reduction variant is part of a catalogue's identity, not a detail of its
    filename: pooling two chains makes any systematic between them look like a
    property of the sky.
    """
    out = {}
    for obs, v in sorted(pairs.items()):
        tags = {(v[f][2] or "plain") for f in FILTERS_USED if f in v}
        key = "/".join(sorted(tags)) if tags else "unknown"
        out.setdefault(key, []).append(obs)
    return out


def report_lineage(pairs):
    """Print the lineage breakdown; return True when the set is uniform."""
    lin = lineage_of(pairs)
    if len(lin) <= 1:
        only = next(iter(lin), "none")
        print(f"reduction lineage: all {len(pairs)} field(s) {only}")
        return True
    print(f"MIXED REDUCTIONS across {len(pairs)} field(s):")
    for k in sorted(lin):
        print(f"  {k:12s} {len(lin[k]):2d}: {', '.join(lin[k])}")
    print("  a systematic between these groups would be PROCESSING, not sky;")
    print("  pick one chain if that matters for the science")
    return False


def fingerprint(pairs):
    """Identity of the input set: which files, and their size and mtime.

    The basename is included, so a switch between catalogue variants (m4 ->
    resbgsub_m5) invalidates the cache the same way a re-vet does.
    """
    items = []
    for obs, v in sorted(pairs.items()):
        for filt in sorted(v):
            p = v[filt][1]
            st = os.stat(p)
            items.append([obs, filt, os.path.basename(p), st.st_size, int(st.st_mtime)])
    return {"n_obs": len(pairs), "items": items, "version": CACHE_VERSION}


def abmag(t):
    if "PIXSCALE" not in t.meta:
        raise KeyError(
            "catalogue has no PIXSCALE in its metadata; refusing to guess -- "
            "the short- and long-wave scales differ by a factor 4 in solid "
            "angle, which is 1.5 mag")
    ps = t.meta["PIXSCALE"] * u.arcsec
    fjy = (np.asarray(t["flux"]) * u.MJy / u.sr * (ps ** 2).to(u.sr)).to(u.Jy).value
    with np.errstate(invalid="ignore", divide="ignore"):
        return -2.5 * np.log10(fjy / AB_ZP_JY)


def saturated(t):
    """`is_saturated` as a bool array; all False for a catalogue without it."""
    if "is_saturated" not in t.colnames:
        return np.zeros(len(t), bool)
    return np.asarray(t["is_saturated"], bool)


def own_footprint(t, radius=SAT_FOOTPRINT_ARCSEC):
    """Rows of `t` that belong to this field.

    Unsaturated rows always do.  A saturated row is kept only within `radius`
    of one of this catalogue's own unsaturated detections, which is what
    removes the survey-wide saturated list each catalogue carries (module
    docstring).  A catalogue with no unsaturated rows keeps nothing saturated
    -- there is no footprint to test against.
    """
    sat = saturated(t)
    keep = ~sat
    if sat.any() and (~sat).any():
        _, d2d, _ = t["skycoord"][sat].match_to_catalog_sky(t["skycoord"][~sat])
        keep[np.flatnonzero(sat)[d2d.arcsec < radius]] = True
    return keep


def dedupe_across_fields(ra, dec, who, tol=DEDUPE_ARCSEC):
    """Keep-mask removing a source already contributed by another field.

    Two rows closer than `tol` from DIFFERENT fields are one star seen twice
    (an overlap, or a saturated row copied into both); the one from the field
    that sorts later is dropped.  Pairs within one field are left alone --
    those are the crowding the photometry already resolved.
    """
    from scipy.spatial import cKDTree
    ra, dec = np.radians(ra), np.radians(dec)
    xyz = np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra),
                           np.sin(dec)])
    chord = 2 * np.sin(np.radians(tol / 3600) / 2)
    pairs = cKDTree(xyz).query_pairs(chord, output_type="ndarray")
    keep = np.ones(len(ra), bool)
    if len(pairs):
        who = np.asarray(who)
        i, j = pairs[:, 0], pairs[:, 1]
        cross = who[i] != who[j]
        i, j = i[cross], j[cross]
        later = np.where(who[i] > who[j], i, j)
        keep[later] = False
    return keep


def field_offset(sw, lw, idx, d2d, search=0.5, core=0.1):
    """Bulk offset (arcsec) of SkyCoord `lw` relative to `sw`, and its size.

    `idx`, `d2d` are ``lw.match_to_catalog_sky(sw)``'s first two outputs.

    Two passes, because in a crowded field the nearest neighbours out to
    `search` are mostly chance pairs that pull a one-pass median towards
    zero: the median (dRA cos dec, dDec) of pairs within `search`, then the
    median of the pairs within `core` of that first estimate.
    """
    if len(sw) == 0 or len(lw) == 0:
        return np.nan, np.nan, np.nan
    near = sw[idx]
    dra = ((lw.ra - near.ra).wrap_at("180d").deg
           * np.cos(np.radians(lw.dec.deg))) * 3600
    ddec = (lw.dec - near.dec).deg * 3600
    close = d2d.arcsec < search
    if not close.any():
        return np.nan, np.nan, np.nan
    x, y = np.median(dra[close]), np.median(ddec[close])
    core_ = close & (np.hypot(dra - x, ddec - y) < core)
    if core_.any():
        x, y = np.median(dra[core_]), np.median(ddec[core_])
    return x, y, float(np.hypot(x, y))


def cross_field_separations(ra, dec, who, rmax=0.5):
    """Separations (arcsec) of every cross-field pair closer than `rmax`.

    Reported once per build so DEDUPE_ARCSEC is a measured choice: the
    copied saturated rows sit at ~0, genuine overlap pairs spread out with
    the two reductions' astrometric scatter.
    """
    from scipy.spatial import cKDTree
    ra, dec = np.radians(ra), np.radians(dec)
    xyz = np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra),
                           np.sin(dec)])
    chord = 2 * np.sin(np.radians(rmax / 3600) / 2)
    pairs = cKDTree(xyz).query_pairs(chord, output_type="ndarray")
    if not len(pairs):
        return np.zeros(0)
    who = np.asarray(who)
    pairs = pairs[who[pairs[:, 0]] != who[pairs[:, 1]]]
    d = np.linalg.norm(xyz[pairs[:, 0]] - xyz[pairs[:, 1]], axis=1)
    return np.degrees(2 * np.arcsin(d / 2)) * 3600


def match_catalogs(pairs, tol=MATCH_ARCSEC):
    """Per-field guard, cross-filter match, then cross-field dedupe.

    Returns three dicts of aligned arrays:

    matched  -- sources in both filters: col (F212N-F480M), m480, m212, ra,
                dec (the F480M position), who (obs), sat (saturated in either)
    f212n    -- every F212N source, matched or not: m212, ra, dec, who, sat.
                The density products count these: F212N is the deeper, finer
                filter, and requiring an F480M match would make the counts
                depend on F480M completeness.
    f480m    -- every F480M source, matched or not: m480, ra, dec, who, sat,
                for the F480M density map.

    Returns (matched, f212n, f480m).

    Nearest-neighbour with no uniqueness pass, so in a field this crowded
    several F480M detections can claim the same F212N source within the
    tolerance.  That slightly inflates the colour products.  The tolerance is
    ~3 short-wave pixels, comfortably inside the astrometric scatter between
    the two filters, so tightening it would cost real matches instead."""
    M = {k: [] for k in ("col", "m480", "m212", "ra", "dec", "who", "sat")}
    F = {k: [] for k in ("m212", "ra", "dec", "who", "sat")}
    L = {k: [] for k in ("m480", "ra", "dec", "who", "sat")}
    for obs, v in pairs.items():
        a = Table.read(v["f212n"][1])
        b = Table.read(v["f480m"][1])
        na, nb = len(a), len(b)
        a = a[own_footprint(a)]
        b = b[own_footprint(b)]
        # A source is "vetted" only when BOTH of its catalogues were: the
        # colour is a difference of the two, so the weaker provenance governs.
        ma, mb = abmag(a), abmag(b)
        sa, sb = saturated(a), saturated(b)
        fa = np.isfinite(ma)
        F["m212"].append(ma[fa])
        F["ra"].append(a["skycoord"].ra.deg[fa])
        F["dec"].append(a["skycoord"].dec.deg[fa])
        F["who"].append(np.full(fa.sum(), obs))
        F["sat"].append(sa[fa])
        fb = np.isfinite(mb)
        L["m480"].append(mb[fb])
        L["ra"].append(b["skycoord"].ra.deg[fb])
        L["dec"].append(b["skycoord"].dec.deg[fb])
        L["who"].append(np.full(fb.sum(), obs))
        L["sat"].append(sb[fb])

        idx, d2d, _ = b["skycoord"].match_to_catalog_sky(a["skycoord"])
        dx, dy, off = field_offset(a["skycoord"], b["skycoord"], idx, d2d)
        ok = d2d.arcsec < tol
        if off > MAX_FIELD_OFFSET_ARCSEC:
            print(f"  {obs}: F480M is ({dx:+.3f}, {dy:+.3f})\" off F212N, over "
                  f"{MAX_FIELD_OFFSET_ARCSEC}\"; left out of the colour "
                  "products", flush=True)
            ok[:] = False
        c = ma[idx[ok]] - mb[ok]
        mm = mb[ok]
        g = np.isfinite(c) & np.isfinite(mm)
        M["col"].append(c[g])
        M["m480"].append(mm[g])
        M["m212"].append(ma[idx[ok]][g])
        M["ra"].append(b["skycoord"].ra.deg[ok][g])
        M["dec"].append(b["skycoord"].dec.deg[ok][g])
        M["who"].append(np.full(g.sum(), obs))
        M["sat"].append((sa[idx[ok]] | sb[ok])[g])
        prov = pair_provenance({obs: v})[obs]
        tag = ("" if prov["vetted"] == "yes" else "  [unvetted]") + \
              ("" if prov["module"] == "merged" else f"  [{prov['module']} only]")
        print(f"  {obs}: SW {na:,} -> {len(a):,} in own footprint "
              f"({sa.sum():,} saturated), LW {nb:,} -> {len(b):,} "
              f"({sb.sum():,} saturated); {g.sum():,} with colour{tag}",
              flush=True)
    M = {k: np.concatenate(v) for k, v in M.items()}
    F = {k: np.concatenate(v) for k, v in F.items()}
    L = {k: np.concatenate(v) for k, v in L.items()}
    for nm, D in (("matched", M), ("F212N", F), ("F480M", L)):
        sep = cross_field_separations(D["ra"], D["dec"], D["who"])
        edges = [0, 0.01, 0.05, DEDUPE_ARCSEC, 0.2, 0.3, 0.5]
        hist, _ = np.histogram(sep, bins=edges)
        print(f"  {nm}: cross-field pair separations (arcsec) " + ", ".join(
            f"{lo:g}-{hi:g}: {n:,}"
            for lo, hi, n in zip(edges[:-1], edges[1:], hist)), flush=True)
        keep = dedupe_across_fields(D["ra"], D["dec"], D["who"])
        print(f"  {nm}: {len(keep):,} rows, {(~keep).sum():,} seen by an "
              f"earlier field removed", flush=True)
        for k in D:
            D[k] = D[k][keep]
    return M, F, L


def load_matched(pairs, force=False):
    """(matched, f212n, f480m, fp): the `match_catalogs` dicts, from CACHE when it was
    built from this exact input set by this CACHE_VERSION."""
    fp = fingerprint(pairs)
    if not force and os.path.exists(CACHE) and os.path.exists(STAMP):
        with open(STAMP) as fh:
            old = json.load(fh)
        if old.get("match") == fp:
            d = np.load(CACHE, allow_pickle=True)
            M = {k[2:]: d[k] for k in d.files if k.startswith("M_")}
            F = {k[2:]: d[k] for k in d.files if k.startswith("F_")}
            L = {k[2:]: d[k] for k in d.files if k.startswith("L_")}
            print(f"match cache hit: {len(M['col']):,} matched, "
                  f"{len(F['m212']):,} F212N and {len(L['m480']):,} F480M "
                  f"sources over {fp['n_obs']} fields")
            return M, F, L, fp
    print(f"matching {fp['n_obs']} field(s) at {MATCH_ARCSEC}\"", flush=True)
    M, F, L = match_catalogs(pairs)
    np.savez(CACHE, **{f"M_{k}": v for k, v in M.items()},
             **{f"F_{k}": v for k, v in F.items()},
             **{f"L_{k}": v for k, v in L.items()})
    print(f"{len(M['col']):,} matched, {len(F['m212']):,} F212N and "
          f"{len(L['m480']):,} F480M sources over {fp['n_obs']} fields")
    return M, F, L, fp


# --------------------------------------------------------------------------
# density maps


def make_grid(allra, alldec, pixel=PIXEL_ARCSEC, galactic=False, pad=0.02):
    """(wcs, ny, nx): a TAN grid covering (allra, alldec) plus `pad` deg.

    `galactic` aligns the grid with l/b.  The survey is a strip along the
    plane, so an equatorial box around it is mostly empty; for the
    fine-pixel star rendering that is the difference between fitting in
    memory and not.
    """
    if galactic:
        c = SkyCoord(allra * u.deg, alldec * u.deg).galactic
        lon, lat = c.l.wrap_at(180 * u.deg).deg, c.b.deg
        ctype = ["GLON-TAN", "GLAT-TAN"]
    else:
        lon, lat = np.asarray(allra), np.asarray(alldec)
        ctype = ["RA---TAN", "DEC--TAN"]
    clon = 0.5 * (lon.min() + lon.max())
    clat = 0.5 * (lat.min() + lat.max())
    pix = pixel / 3600.0
    cosd = np.cos(np.radians(clat))
    nx = int(2 * (np.ptp(lon) * cosd / 2 + pad) / pix) + 1
    ny = int(2 * (np.ptp(lat) / 2 + pad) / pix) + 1
    w = WCS(naxis=2)
    w.wcs.ctype = ctype
    w.wcs.crval = [clon % 360, clat]
    w.wcs.crpix = [nx / 2, ny / 2]
    w.wcs.cdelt = [-pix, pix]
    w.pixel_shape = (nx, ny)
    return w, ny, nx


def sky_to_pix(w, ra, dec):
    """Pixel coordinates of ICRS (ra, dec) on `w`, whatever its frame."""
    return w.world_to_pixel(SkyCoord(np.asarray(ra) * u.deg,
                                     np.asarray(dec) * u.deg))


def density(ra, dec, allra, alldec, pixel=PIXEL_ARCSEC, smooth=SMOOTH_ARCSEC,
            grid=None):
    """Stars per square arcmin on a TAN grid, NaN outside the photometric
    footprint.  Zero density and no measurement are different statements, and a
    HiPS renders NaN as blank; coverage comes from ALL matched sources so an
    area with stars but none selected reads as a real zero.

    `grid` = make_grid(...) output, for products that must share pixels (the
    frames of a cube); by default the grid is fitted to (allra, alldec)."""
    w, ny, nx = grid if grid is not None else make_grid(allra, alldec, pixel)

    def hist(r, d):
        x, y = sky_to_pix(w, r, d)
        xi, yi = np.round(x).astype(int), np.round(y).astype(int)
        m = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
        h = np.zeros((ny, nx))
        np.add.at(h, (yi[m], xi[m]), 1.0)
        return h

    sig = smooth / pixel
    sel = gaussian_filter(hist(ra, dec), sig) / ((pixel / 60.0) ** 2)
    cov = gaussian_filter(hist(allra, alldec), sig * 3) > 0
    return np.where(cov, sel, np.nan).astype("float32"), w, cov


def build_hips(arr, w, name, level=None, threads=8, dest=None):
    from reproject import reproject_interp
    from reproject.hips import reproject_to_hips
    dest = dest or f"{OUT}/{name}"
    stage = dest + ".new"
    shutil.rmtree(stage, ignore_errors=True)
    # The identity is built from `name`, not from the staging directory: the
    # default obs_title is the output directory's basename, which here would
    # be "<name>.new" -- a directory that stops existing at the rename below.
    reproject_to_hips((arr, w), coord_system_out="galactic", level=level,
                      reproject_function=reproject_interp,
                      output_directory=stage, threads=threads,
                      properties=layer_properties(name), generate_moc=True)
    if not os.path.isdir(f"{stage}/Norder3"):
        raise RuntimeError(f"{name}: no Norder3; refusing to publish")
    swap_in(stage, dest)


def swap_in(stage, dest):
    old = dest + ".old"
    shutil.rmtree(old, ignore_errors=True)
    if os.path.isdir(dest):
        os.rename(dest, old)
    os.rename(stage, dest)
    shutil.rmtree(old, ignore_errors=True)
    print(f"  BUILT {dest}", flush=True)


def write_density(arr, w, name, bunit="stars/arcmin2"):
    hdr = w.to_header()
    hdr["BUNIT"] = bunit
    fits.PrimaryHDU(arr, hdr).writeto(f"{OUT}/{name}_density.fits", overwrite=True)


# --------------------------------------------------------------------------
# the four products


def build_red_stars(col, m480, ra, dec, level=None, threads=8):
    sel = np.isfinite(col) & np.isfinite(m480) & (col > RED_COLOUR) & (m480 < RED_MAGLIMIT)
    print(f"red+bright: {sel.sum():,} of {len(col):,} "
          f"({100 * sel.sum() / max(len(col), 1):.2f}%)")
    arr, w, cov = density(ra[sel], dec[sel], ra, dec)
    print(f"  grid {arr.shape}, coverage {100 * cov.mean():.1f}%, "
          f"median-in-coverage {np.nanmedian(arr):.2f}, max {np.nanmax(arr):.1f}/arcmin2")
    write_density(arr, w, "jwst-red-stars-hips")
    build_hips(arr, w, "jwst-red-stars-hips", level, threads)


def rc_masks(col, m480):
    W = m480 - SLOPE * col
    (mlo, mhi), (clo, chi) = RC_M480_RANGE, RC_COLOUR_RANGE
    rc = ((m480 > mlo) & (m480 < mhi) & (col > clo) & (col < chi)
          & (np.abs(W - WRC) < HW))
    return rc, rc & (col < SPLIT), rc & (col >= SPLIT)


def build_rc(col, m480, ra, dec, level=None, threads=8):
    rc, blue, red = rc_masks(col, m480)
    print(f"RC band {rc.sum():,}; blue {blue.sum():,}; red {red.sum():,}")
    for nm, m in (("blue", blue), ("red", red)):
        av = (np.median(col[m]) + 1.8) * 30.5
        print(f"  {nm}: median colour {np.median(col[m]):+.3f} -> A_V ~ {av:.0f}")
    for nm, m, layer in (("blue", blue, "jwst-rc-blue-hips"),
                         ("red", red, "jwst-rc-red-hips")):
        arr, w, cov = density(ra[m], dec[m], ra[rc], dec[rc])
        print(f"  {nm}: grid {arr.shape}, coverage {100 * cov.mean():.1f}%, "
              f"median-in-coverage {np.nanmedian(arr):.2f}, "
              f"max {np.nanmax(arr):.1f}/arcmin2")
        write_density(arr, w, layer)
        build_hips(arr, w, layer, level, threads)


def build_ultrared(col, m480, ra, dec, who, prov=None):
    """Catalogue rather than a density map: too few sources over too few fields
    for a meaningful surface density, and the positions are the useful product.
    NOT vetted by eye -- an extreme colour can be a genuinely embedded object or
    a mismatch between the two filters."""
    sel = col > ULTRARED_CUT
    fields = " ".join(sorted(set(np.asarray(who)[sel].tolist())))
    # Per OBSERVATION, looked up by `who`, rather than carried alongside every
    # source: provenance is a property of the catalogue pair, so an array of
    # it is the same value repeated 100,000 times.  Absent, it is "unknown" --
    # never "vetted", which is the assumption that makes an unvetted source
    # indistinguishable from a checked one.
    prov = prov or {}
    vet = np.array([prov.get(o, {}).get("vetted", "unknown")
                    for o in np.asarray(who)])
    mod = np.array([prov.get(o, {}).get("module", "unknown")
                    for o in np.asarray(who)])
    n_unvetted = int((vet[sel] == "no").sum())
    n_module = int(np.isin(mod[sel], ("nrca", "nrcb")).sum())
    print(f"ultra-red: {sel.sum()} sources with F212N-F480M > {ULTRARED_CUT} "
          f"over {len(fields.split())} field(s)"
          + (f"; {n_unvetted} from catalogues with no vetting pass"
             if n_unvetted else "")
          + (f"; {n_module} from a single NIRCam module"
             if n_module else ""))
    order = np.argsort(-col[sel])
    t = Table({"ra": ra[sel][order], "dec": dec[sel][order],
               "color_f212n_f480m": col[sel][order],
               "mag_ab_f480m": m480[sel][order],
               "mag_ab_f212n": (col[sel] + m480[sel])[order],
               "obs": np.asarray(who)[sel][order],
               "vetted": vet[sel][order],
               "module": mod[sel][order]})
    t.meta["SELECT"] = f"F212N-F480M > {ULTRARED_CUT} AB"
    t.meta["FIELDS"] = fields
    t.meta["MATCHTOL"] = f"{MATCH_ARCSEC} arcsec"
    t.write(f"{WEB}/jwst_ultrared_stars.ecsv", overwrite=True)
    t.write(f"{WEB}/jwst_ultrared_stars.fits", overwrite=True)
    rows = [{"ra": round(float(r), 6), "dec": round(float(d), 6),
             "color": round(float(c), 2), "f480m": round(float(m), 2),
             "obs": str(o), "vetted": str(w), "module": str(md)}
            for r, d, c, m, o, w, md in zip(
                t["ra"], t["dec"], t["color_f212n_f480m"], t["mag_ab_f480m"],
                t["obs"], t["vetted"], t["module"])]
    doc = {"name": f"JWST ultra-red (F212N-F480M > {ULTRARED_CUT})",
           "select": f"F212N-F480M > {ULTRARED_CUT} AB, {MATCH_ARCSEC}\" match",
           "fields": fields, "n": len(rows),
           "n_unvetted": n_unvetted, "n_single_module": n_module,
           "sources": rows}
    with open(f"{WEB}/jwst_ultrared_stars.json", "w") as fh:
        json.dump(doc, fh)
    print(f"  wrote ecsv/fits/json ({len(rows)} sources, "
          f"colour {t['color_f212n_f480m'].min():.2f}-"
          f"{t['color_f212n_f480m'].max():.2f})")


def report_ridge(col, m480):
    """Locate the RC ridge by the density mode perpendicular to the reddening
    vector, so drift away from the fixed cut constants is visible."""
    W = m480 - SLOPE * col
    bins = np.arange(np.percentile(col, 1), np.percentile(col, 99), 0.25)
    ridge = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = (col >= lo) & (col < hi) & (m480 > 14) & (m480 < 19)
        if s.sum() < 200:
            continue
        h, e = np.histogram(W[s], bins=np.arange(np.nanpercentile(W[s], 1),
                                                 np.nanpercentile(W[s], 99), 0.05))
        if h.sum() == 0:
            continue
        ridge.append(0.5 * (e[h.argmax()] + e[h.argmax() + 1]))
    if len(ridge) > 2:
        ridge = np.array(ridge)
        print(f"RC ridge now: W-mode median {np.median(ridge):.3f} "
              f"(cut holds WRC={WRC}), spread {ridge.std():.3f} mag")
        if abs(np.median(ridge) - WRC) > HW / 2:
            print(f"  NOTE: ridge has moved more than half the band half-width "
                  f"from WRC={WRC}; consider --refit")


# --------------------------------------------------------------------------
# star-count products: total density, F212N magnitude cube, colour cube,
# median colour


def report_limits(F):
    """Measured F212N saturation and confusion limits beside the fixed ones.

    Saturation: the brightest 0.1% of unsaturated detections.  Confusion: the
    turnover (mode) of the unsaturated luminosity function in 0.25-mag bins.
    Both vary field to field with crowding; the survey-wide values are what
    the cube's fixed edges should be compared against.
    """
    m, sat = F["m212"], F["sat"]
    un = m[~sat & np.isfinite(m)]
    if len(un) == 0:
        return None, None
    bright = float(np.percentile(un, 0.1))
    h, e = np.histogram(un, bins=np.arange(10, 27.01, 0.25))
    peak = float(0.5 * (e[h.argmax()] + e[h.argmax() + 1]))
    print(f"F212N limits now: saturation ~{bright:.2f} (brightest 0.1% "
          f"unsaturated; {sat.sum():,} saturated rows), confusion ~{peak:.2f} "
          f"(LF turnover); cube holds {F212N_SAT_LIMIT}-{F212N_CONFUSION_LIMIT}")
    return bright, peak


def density_cube(ra, dec, value, edges, allra, alldec, name, level=None,
                 threads=8, bunit3="", em_range=None):
    """One density frame per [edges[i], edges[i+1]) of `value`, as a HiPS
    cube plus a FITS cube of the same frames.  All frames share one grid,
    which `assemble_hips_cube` needs and checks."""
    from jwst_rgb.hips_formats import assemble_hips_cube
    grid = make_grid(allra, alldec, PIXEL_ARCSEC)
    frames, planes = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (value >= lo) & (value < hi)
        arr, w, cov = density(ra[sel], dec[sel], allra, alldec, grid=grid)
        print(f"  [{lo:+.2f}, {hi:+.2f}): {sel.sum():,} stars, median-in-"
              f"coverage {np.nanmedian(arr):.2f}, max {np.nanmax(arr):.1f}"
              f"/arcmin2", flush=True)
        fdir = f"{OUT}/_frames/{name}/{len(frames):02d}"
        os.makedirs(os.path.dirname(fdir), exist_ok=True)
        build_hips(arr, w, name, level, threads, dest=fdir)
        frames.append(fdir)
        planes.append(arr)
    cube = np.array(planes, dtype="float32")
    step = float(edges[1] - edges[0])
    hdr = w.to_header()
    hdr["NAXIS"] = 3
    hdr["CTYPE3"], hdr["CRPIX3"] = "BINCENTR", 1.0
    hdr["CRVAL3"] = float(edges[0]) + step / 2
    hdr["CDELT3"] = step
    hdr["BUNIT"] = "stars/arcmin2"
    hdr["BINAXIS"] = bunit3
    fits.PrimaryHDU(cube, hdr).writeto(f"{OUT}/{name}.fits", overwrite=True)
    vmax = float(np.nanpercentile(cube, 99.5))
    assemble_hips_cube(frames, f"{OUT}/{name}", crval3=hdr["CRVAL3"],
                       cdelt3=step, bunit3=bunit3, pixel_cut=(0.0, vmax),
                       extra=layer_properties(name), em_range=em_range)
    shutil.rmtree(f"{OUT}/_frames/{name}", ignore_errors=True)
    print(f"  BUILT {OUT}/{name} ({len(frames)} frames, cut 0-{vmax:.1f})",
          flush=True)


def build_star_density(F, L, level=None, threads=8):
    """Total F212N and F480M star densities (saturated included) and the
    F212N 1-mag cube."""
    ra, dec, m = F["ra"], F["dec"], F["m212"]
    report_limits(F)
    arr, w, cov = density(ra, dec, ra, dec)
    print(f"all stars: {len(ra):,}; grid {arr.shape}, coverage "
          f"{100 * cov.mean():.1f}%, median-in-coverage {np.nanmedian(arr):.0f}, "
          f"max {np.nanmax(arr):.0f}/arcmin2")
    write_density(arr, w, "jwst-star-density-hips")
    build_hips(arr, w, "jwst-star-density-hips", level, threads)
    # F480M on the F212N grid, with the F212N coverage: a pixel inside the
    # F212N footprint with no F480M source is a zero, not a blank.
    arr480, _, _ = density(L["ra"], L["dec"], ra, dec, grid=make_grid(
        ra, dec, PIXEL_ARCSEC))
    print(f"all F480M sources: {len(L['ra']):,}; median-in-coverage "
          f"{np.nanmedian(arr480):.0f}, max {np.nanmax(arr480):.0f}/arcmin2")
    write_density(arr480, w, "jwst-star-density-f480m-hips")
    build_hips(arr480, w, "jwst-star-density-f480m-hips", level, threads)
    edges = np.arange(F212N_SAT_LIMIT, F212N_CONFUSION_LIMIT + 0.01, 1.0)
    print(f"F212N magnitude cube, {len(edges) - 1} frames:")
    density_cube(ra, dec, m, edges, ra, dec,
                 "jwst-star-density-f212n-cube-hips", level, threads,
                 bunit3="mag (F212N, AB)", em_range=F212N_EM_RANGE)


def build_colour(M, level=None, threads=8):
    """Star counts in colour bins, and the median colour: the two
    pseudo-extinction products.

    Stars behind more dust are redder, so counts move from blue bins to red
    ones as extinction rises; the median colour of a cell tracks the typical
    reddening of the stars in it.  Neither is calibrated to A_V here -- the
    intrinsic-colour spread of the population is folded in.  Saturated rows
    count toward the cube (they are stars) but not toward the median, whose
    colours are the least reliable ones.
    """
    from scipy.stats import binned_statistic_2d
    col, m480, ra, dec, sat = M["col"], M["m480"], M["ra"], M["dec"], M["sat"]
    bright = np.isfinite(col) & (m480 < COLOUR_MAGLIMIT)
    print(f"colour products: {bright.sum():,} of {len(col):,} matched stars "
          f"with F480M < {COLOUR_MAGLIMIT}")
    density_cube(ra[bright], dec[bright], col[bright], COLOUR_EDGES, ra, dec,
                 "jwst-star-density-colour-cube-hips", level, threads,
                 bunit3="mag (F212N-F480M, AB)",
                 em_range=F480M_EM_RANGE)

    w, ny, nx = make_grid(ra, dec, MEDIAN_PIXEL_ARCSEC)
    use = bright & ~sat
    x, y = sky_to_pix(w, ra[use], dec[use])
    rng = [[-0.5, ny - 0.5], [-0.5, nx - 0.5]]
    med = binned_statistic_2d(y, x, col[use], "median", bins=[ny, nx],
                              range=rng).statistic
    n = binned_statistic_2d(y, x, None, "count", bins=[ny, nx],
                            range=rng).statistic
    med = np.where(n >= MEDIAN_MIN_STARS, med, np.nan).astype("float32")
    good = np.isfinite(med)
    print(f"median colour: {good.sum():,} cells of {MEDIAN_PIXEL_ARCSEC}\" with "
          f">= {MEDIAN_MIN_STARS} stars; 5/50/95%: "
          f"{np.round(np.nanpercentile(med, [5, 50, 95]), 2)}")
    write_density(med, w, "jwst-median-colour-hips", bunit="mag")
    build_hips(med, w, "jwst-median-colour-hips", level, threads)


# --------------------------------------------------------------------------
# the star overlay, as an image and as a catalogue


def star_style(m212, col):
    """(rgb uint8 [N,3], alpha uint8 [N], radius int [N]) for each star."""
    import matplotlib
    lo, hi = STAR_COLOUR_RANGE
    cmap = matplotlib.colormaps[STAR_CMAP]
    rgb = (cmap(np.clip((col - lo) / (hi - lo), 0, 1))[:, :3] * 255).round()
    alpha = np.interp(m212, STAR_ALPHA_MAG, STAR_ALPHA).round()
    radius = np.maximum(np.interp(m212, STAR_RADIUS_MAG,
                                  STAR_RADIUS_PIX).round(), 1)
    return rgb.astype(np.uint8), alpha.astype(np.uint8), radius.astype(int)


def colour_field(M, pixel=None, k=None, max_arcsec=None, workers=8,
                 chunk_rows=64):
    """(colour [ny, nx] float32, wcs): F212N-F480M at every pixel of a
    galactic grid, the inverse-distance weighted mean colour of the `k`
    nearest unsaturated matched stars.  NaN where the k-th of them is
    farther than `max_arcsec`, which blanks the gaps between fields and the
    sky beyond the survey edge.

    Saturated stars are left out: their colours are the least reliable ones,
    and one bright star would otherwise paint its whole neighbourhood.  The
    weight is 1/d with d floored at half a pixel, so a pixel centred on a star
    is not that star's colour alone.
    """
    from scipy.spatial import cKDTree
    pixel = KNN_PIXEL_ARCSEC if pixel is None else pixel
    k = KNN_NEIGHBORS if k is None else k
    max_arcsec = KNN_MAX_ARCSEC if max_arcsec is None else max_arcsec
    ok = np.isfinite(M["col"]) & ~M["sat"]
    ra, dec, col = M["ra"][ok], M["dec"][ok], M["col"][ok]
    w, ny, nx = make_grid(ra, dec, pixel, galactic=True, pad=0.005)
    print(f"colour field: {ok.sum():,} unsaturated stars, {k} nearest, on "
          f"{nx} x {ny} at {pixel}\"/pix", flush=True)
    x, y = sky_to_pix(w, ra, dec)
    tree = cKDTree(np.c_[x, y])
    out = np.full((ny, nx), np.nan, np.float32)
    xs = np.arange(nx)
    for r0 in range(0, ny, chunk_rows):
        rows = np.arange(r0, min(r0 + chunk_rows, ny))
        X, Y = np.meshgrid(xs, rows)
        d, i = tree.query(np.c_[X.ravel(), Y.ravel()], k=k,
                          distance_upper_bound=max_arcsec / pixel,
                          workers=workers)
        d, i = d.reshape(-1, k), i.reshape(-1, k)
        good = np.isfinite(d[:, -1])
        wt = 1.0 / np.maximum(d[good], 0.5)
        block = np.full(len(d), np.nan, np.float32)
        block[good] = (wt * col[i[good]]).sum(1) / wt.sum(1)
        out[rows] = block.reshape(len(rows), nx)
    print(f"  {100 * np.isfinite(out).mean():.1f}% of the grid filled",
          flush=True)
    return out, w


def render_colour_field(M, **kw):
    """RGBA image (FITS row order: row 0 is the bottom) of `colour_field`,
    in the star colour map, transparent where the field is blank."""
    import matplotlib
    colour, w = colour_field(M, **kw)
    lo, hi = STAR_COLOUR_RANGE
    cmap = matplotlib.colormaps[STAR_CMAP]
    good = np.isfinite(colour)
    img = np.zeros(colour.shape + (4,), np.uint8)
    lut = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).round().astype(np.uint8)
    idx = np.clip((colour[good] - lo) / (hi - lo), 0, 1)
    img[good, :3] = lut[np.round(idx * 255).astype(int)]
    img[good, 3] = 255
    return img, w


def build_star_image(M, level=None, threads=8):
    """The star-colour field (`render_colour_field`) as an image HiPS with
    transparent sky.

    Written through the same pixel path as `save_rgb` (flip the rows, then
    ROTATE_180) with `avm_for_saved_png` describing the result, which is the
    combination the astrometry tests pin.  `save_rgb` itself is not used
    because its alpha comes from NaN islands in the data, and this layer's
    transparency is decided by the neighbour distance instead.
    """
    import PIL
    from PIL import Image
    from reproject import reproject_interp
    from reproject.hips import reproject_to_hips
    from jwst_rgb.save_rgb import avm_for_saved_png

    name = "jwst-stars-colour-hips"
    img, w = render_colour_field(M, workers=threads)
    ny, nx = img.shape[:2]
    PIL.Image.MAX_IMAGE_PIXELS = None
    png = f"{OUT}/{name.replace('-hips', '')}.png"
    Image.fromarray(img[::-1], mode="RGBA").transpose(
        Image.ROTATE_180).save(png)
    del img
    avm = avm_for_saved_png(w, ny, nx, flip=-1, transpose=Image.ROTATE_180)
    tmp = png + ".avm.png"
    avm.embed(png, tmp)
    shutil.move(tmp, png)

    dest = f"{OUT}/{name}"
    stage = dest + ".new"
    shutil.rmtree(stage, ignore_errors=True)
    reproject_to_hips(png, coord_system_out="galactic", level=level,
                      reproject_function=reproject_interp,
                      output_directory=stage, threads=threads,
                      properties=layer_properties(name))
    if not os.path.isdir(f"{stage}/Norder3"):
        raise RuntimeError(f"{name}: no Norder3; refusing to publish")
    swap_in(stage, dest)


def build_star_catalog(M):
    """Every matched star as a HiPS catalogue, brightest (F212N) first.

    `rgb` repeats the image layer's colour for each star, so a client can
    draw the catalogue the same way without reimplementing the colour map.
    """
    from jwst_rgb.hips_formats import write_hips_catalogue
    ok = np.isfinite(M["m212"]) & np.isfinite(M["col"])
    m, col = M["m212"][ok], M["col"][ok]
    rgb, _, _ = star_style(m, col)
    uniq, inv = np.unique(rgb, axis=0, return_inverse=True)
    hexes = np.array([f"#{r:02x}{g:02x}{b:02x}" for r, g, b in uniq])
    columns = {
        "ra": M["ra"][ok], "dec": M["dec"][ok],
        "f212n": m, "f480m": M["m480"][ok], "color": col,
        "saturated": M["sat"][ok].astype(np.int16),  # VOTable has no int8
        "obs": M["who"][ok].astype(str),
        "rgb": hexes[inv.ravel()],
    }
    info = write_hips_catalogue(
        f"{OUT}/jwst-stars-catalog-hips", columns, priority=m,
        n_per_tile=CATALOG_ROWS_PER_TILE,
        formats={"f212n": "%.3f", "f480m": "%.3f", "color": "%.3f"},
        units={"ra": "deg", "dec": "deg", "f212n": "mag", "f480m": "mag",
               "color": "mag"},
        ucds={"ra": "pos.eq.ra;meta.main", "dec": "pos.eq.dec;meta.main",
              "f212n": "phot.mag;em.IR.K", "f480m": "phot.mag;em.IR.4-8um",
              "color": "phot.color", "saturated": "meta.code.qual",
              "obs": "meta.id;obs", "rgb": "meta.code"},
        descriptions={
            "f212n": "F212N AB magnitude", "f480m": "F480M AB magnitude",
            "color": "F212N - F480M (AB)",
            "saturated": "1 if saturated in either filter (flux from the "
                         "saturated-star fit)",
            "obs": "10678 observation the source was taken from",
            "rgb": f"display color: {STAR_CMAP} over {STAR_COLOUR_RANGE}"},
        properties=layer_properties("jwst-stars-catalog-hips"))
    print(f"  BUILT {OUT}/jwst-stars-catalog-hips: {info['nrows']:,} rows, "
          f"orders 1-{info['order_max']}, tiles per order "
          f"{info['tiles_per_order']}", flush=True)


# --------------------------------------------------------------------------
# publish


def publish(dry=False):
    """Stage the HiPS layers into the web root; the catalogues are written
    there directly.  Symlinked layers are dereferenced on the starformation
    copy (-L) so the remote gets real files."""
    import subprocess
    for name in LAYERS:
        src = f"{OUT}/{name}"
        if not os.path.isdir(src):
            print(f"  {name}: not built; skipping")
            continue
        dest = f"{WEB}/{name}"
        stage = dest + ".new"
        cmd = ["rsync", "-a", "--delete", f"{src}/", f"{stage}/"]
        print("  " + " ".join(cmd), flush=True)
        if dry:
            continue
        subprocess.run(cmd, check=True)
        # A catalogue HiPS starts at order 1; the image layers at 3.
        order_min = 3
        if os.path.exists(f"{stage}/properties"):
            order_min = int(_read_properties(f"{stage}/properties")
                            .get("hips_order_min", 3))
        if not os.path.isdir(f"{stage}/Norder{order_min}"):
            raise RuntimeError(f"{stage}: no Norder{order_min}; "
                               "refusing to swap in")
        old = dest + ".old"
        shutil.rmtree(old, ignore_errors=True)
        if os.path.isdir(dest):
            os.rename(dest, old)
        os.replace(stage, dest)
        shutil.rmtree(old, ignore_errors=True)
        print(f"  published {dest}", flush=True)


def push_remote(dry=False):
    """Mirror the published overlays to the starformation viewer host.

    Runs from wherever it is invoked, so the caller has to have working
    non-interactive ssh -- cron does, on the login node.  -L dereferences
    symlinked layers so the remote gets real files.  No --delete: this tree is
    shared with products this script does not own.

    The CALLER must hold the lock.  publish() swaps each layer in with
    os.rename + os.replace, and an rsync walking the tree across those two
    calls sees it vanish and be replaced underneath, mirroring half a pyramid
    to a live viewer host.  Nothing orders the cron's push against the build
    job it just submitted, so the push takes the same lock the build does.
    """
    import subprocess
    srcs = [f"{WEB}/{n}/" for n in LAYERS if os.path.isdir(f"{WEB}/{n}")]
    for src in srcs:
        name = os.path.basename(src.rstrip("/"))
        # --partial-dir, never bare --partial: bare --partial keeps a
        # half-transferred file at the DESTINATION name, so an interrupted push
        # serves truncated tiles until the next tick repairs them.  A partial
        # dir keeps the resume benefit without ever exposing a fragment.
        cmd = ["rsync", "-a", "-L", "--partial-dir=.rsync-partial",
               src, f"{REMOTE}{name}/"]
        print("  " + " ".join(cmd), flush=True)
        if not dry:
            subprocess.run(cmd, check=True)
    files = [f"{WEB}/{f}" for f in CATALOGUE_FILES if os.path.exists(f"{WEB}/{f}")]
    if files:
        cmd = ["rsync", "-a", "-L", "--partial-dir=.rsync-partial"] + files + [REMOTE]
        print("  " + " ".join(cmd), flush=True)
        if not dry:
            subprocess.run(cmd, check=True)
    print(f"  pushed {len(srcs)} layer(s) and {len(files)} catalogue file(s)",
          flush=True)


# --------------------------------------------------------------------------


#: Everything a full build makes; `--only` picks from these.
PRODUCTS = ("red", "rc", "ultrared", "density", "colour", "stars", "catalog")


#: `--check`'s exit status for "a rebuild is due" -- distinct from 1, which
#: main() already returns for "no catalogue pairs", so the cron can tell a due
#: rebuild from a broken check.
REBUILD_DUE = 3


def take_lock(max_age=6 * 3600):
    if os.path.exists(LOCK):
        age = time.time() - os.path.getmtime(LOCK)
        if age < max_age:
            print(f"another run holds {LOCK} ({age / 60:.0f} min old); exiting")
            return False
        print(f"stale lock ({age / 3600:.1f} h old); taking it")
    os.makedirs(OUT, exist_ok=True)
    open(LOCK, "w").write(f"{os.getpid()} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    return True


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--auto", action="store_true",
                    help="rebuild only when the input catalogues have changed")
    ap.add_argument("--force", action="store_true", help="rebuild regardless")
    ap.add_argument("--allow-module", action="store_true",
                    help="use a single-module catalogue where the whole-tile "
                         "merged one does not exist; those sources are marked "
                         "with the module they came from")
    ap.add_argument("--allow-unvetted", action="store_true",
                    help="use a filter's unvetted catalogue where it has no "
                         "vetted one; every source from such a pair is marked "
                         "vetted=False in the outputs")
    ap.add_argument("--publish", action="store_true",
                    help="copy the HiPS layers into the web root when done")
    ap.add_argument("--push-remote", action="store_true",
                    help="also mirror the published overlays to starformation "
                         "(needs working non-interactive ssh)")
    ap.add_argument("--push-only", action="store_true",
                    help="mirror whatever is already published to starformation "
                         "and exit; takes no lock and builds nothing")
    ap.add_argument("--check", action="store_true",
                    help="exit 0 when the published build matches the input "
                         "catalogues, %d when a rebuild is due; builds nothing "
                         "and takes no lock.  Lets the cron size (or skip) the "
                         "build job before submitting it" % REBUILD_DUE)
    ap.add_argument("--refit", action="store_true",
                    help="report the RC ridge and exit without building")
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--only", choices=PRODUCTS, action="append",
                    help="build a subset (repeatable)")
    a = ap.parse_args()

    # Stands on its own so a scheduler can push from a host with working ssh
    # while the build runs elsewhere.  Deliberately ahead of the --auto
    # up-to-date check: a push can be outstanding even when nothing needs
    # rebuilding, which is the normal case on the tick after a build.
    if a.push_only:
        # Same lock as the build: publish() may be swapping trees in right now.
        # Skipping is the right answer rather than waiting -- the next tick
        # pushes, and the products are unchanged in the meantime.
        if not take_lock():
            print("a build holds the lock; leaving the push to the next tick")
            return 0
        try:
            print("pushing published overlays to starformation")
            push_remote()
        finally:
            if os.path.exists(LOCK):
                os.remove(LOCK)
        return 0

    pairs = latest_pairs(allow_unvetted=a.allow_unvetted,
                         allow_module=a.allow_module)
    if not pairs:
        print(f"no catalogue pairs under {CAT}")
        return 1
    prov = pair_provenance(pairs)
    unvetted = sorted(o for o, p in prov.items() if p["vetted"] != "yes")
    partial = sorted(o for o, p in prov.items() if p["module"] != "merged")
    print(f"{len(pairs)} observation(s) with both filters: {', '.join(pairs)}")
    if unvetted:
        print(f"  {len(unvetted)} of them use a catalogue with no vetting pass "
              f"in at least one filter: {', '.join(unvetted)}")
    if partial:
        print(f"  {len(partial)} of them are a single NIRCam module, so they "
              f"cover half the tile: {', '.join(partial)}")
    report_lineage(pairs)

    fp = fingerprint(pairs)
    if a.check:
        built = None
        if os.path.exists(STAMP):
            with open(STAMP) as fh:
                built = json.load(fh).get("built")
        if built == fp:
            print("overlays up to date")
            return 0
        print("overlays rebuild due")
        return REBUILD_DUE
    if a.auto and not a.force and os.path.exists(STAMP):
        with open(STAMP) as fh:
            old = json.load(fh)
        if old.get("built") == fp:
            print("overlays up to date; nothing to do")
            return 0
        prev = {i[0] for i in old.get("built", {}).get("items", [])}
        new = {i[0] for i in fp["items"]} - prev
        print("rebuilding -- " + (f"new field(s): {', '.join(sorted(new))}" if new
                                  else "catalogues changed"))

    if not take_lock():
        return 0
    try:
        M, F, L, fp = load_matched(pairs, force=a.force)
        col, m480, ra, dec, who = (M[k] for k in ("col", "m480", "ra", "dec",
                                                   "who"))
        report_ridge(col, m480)
        if a.refit:
            report_limits(F)
            return 0
        want = set(a.only) if a.only else set(PRODUCTS)
        if "red" in want:
            build_red_stars(col, m480, ra, dec, a.level, a.threads)
        if "rc" in want:
            build_rc(col, m480, ra, dec, a.level, a.threads)
        if "ultrared" in want:
            build_ultrared(col, m480, ra, dec, who, pair_provenance(pairs))
        if "density" in want:
            build_star_density(F, L, a.level, a.threads)
        if "colour" in want:
            build_colour(M, a.level, a.threads)
        if "stars" in want:
            build_star_image(M, None, a.threads)
        if "catalog" in want:
            build_star_catalog(M)
        if a.publish:
            print("publishing")
            publish()
        if a.push_remote:
            print("pushing to starformation")
            push_remote()
        # The match cache is valid for this input set whatever subset was
        # built, but "built" must only claim a FULL build: recording it after
        # --only would make the next --auto tick report everything up to date
        # and leave the layers it skipped frozen until some other catalogue
        # changes.
        full = want == set(PRODUCTS)
        stamp = {"match": fp, "when": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if full:
            stamp["built"] = fp
        elif os.path.exists(STAMP):
            with open(STAMP) as fh:
                stamp["built"] = json.load(fh).get("built")
        stamp["lineage"] = lineage_of(pairs)
        with open(STAMP, "w") as fh:
            json.dump(stamp, fh, indent=1)
        if not full:
            print(f"partial build ({', '.join(sorted(want))}); "
                  f"not marking the input set as built")
        print("done")
        return 0
    finally:
        if os.path.exists(LOCK):
            os.remove(LOCK)


if __name__ == "__main__":
    sys.exit(main())
