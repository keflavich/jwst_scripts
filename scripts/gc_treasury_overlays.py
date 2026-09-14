#!/usr/bin/env python
"""Catalogue-derived overlays for the GC Treasury Aladin view.

Four products, all built from the per-observation vetted daophot catalogues and
all regenerated together whenever those catalogues change:

  jwst-red-stars-hips   density of F212N-F480M > 0, F480M < 18 (AB)
  jwst-rc-blue-hips     red-clump stars bluer than the GC average
  jwst-rc-red-hips      red-clump stars redder than the GC average
  jwst_ultrared_stars   catalogue (ecsv/fits/json) of F212N-F480M > 4

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
LAYERS = ["jwst-red-stars-hips", "jwst-rc-blue-hips", "jwst-rc-red-hips"]
CATALOGUE_FILES = ["jwst_ultrared_stars.ecsv", "jwst_ultrared_stars.fits",
                   "jwst_ultrared_stars.json"]

AB_ZP_JY = 3631.0
MATCH_ARCSEC = 0.1

# red + bright density layer
RED_COLOUR, RED_MAGLIMIT = 0.0, 18.0
# red-clump band and split (see module docstring -- empirical, held fixed)
SLOPE, WRC, HW, SPLIT = 0.890, 17.50, 0.9, -0.325
# ultra-red catalogue
ULTRARED_CUT = 4.0
# density grid
PIXEL_ARCSEC, SMOOTH_ARCSEC = 2.0, 6.0

CAT_RE = re.compile(
    r"(f\d+[a-z])_merged_(o\d+)_indivexp_merged_m(\d+)_dao_basic_vetted\.fits$")


# --------------------------------------------------------------------------
# inputs


def latest_pairs():
    """{obs: {filter: (iteration, path)}} keeping the highest m<N> per filter,
    restricted to observations vetted in BOTH filters."""
    have = {}
    for f in glob.glob(f"{CAT}/f*_merged_o*_indivexp_merged_m*_dao_basic_vetted.fits"):
        m = CAT_RE.search(os.path.basename(f))
        if not m:
            continue
        filt, obs, it = m.group(1), m.group(2), int(m.group(3))
        d = have.setdefault(obs, {})
        if filt not in d or it > d[filt][0]:
            d[filt] = (it, f)
    return {o: v for o, v in sorted(have.items())
            if "f212n" in v and "f480m" in v}


def fingerprint(pairs):
    """Identity of the input set: which files, and their size and mtime."""
    items = []
    for obs, v in sorted(pairs.items()):
        for filt in sorted(v):
            p = v[filt][1]
            st = os.stat(p)
            items.append([obs, filt, os.path.basename(p), st.st_size, int(st.st_mtime)])
    return {"n_obs": len(pairs), "items": items}


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


def match_catalogs(pairs, tol=MATCH_ARCSEC):
    """Cross-filter positional match; returns colour, F480M, ra, dec, obs.

    Nearest-neighbour with no uniqueness pass, so in a field this crowded
    several F480M detections can claim the same F212N source within the
    tolerance.  That slightly inflates the density maps.  The tolerance is
    ~3 short-wave pixels, comfortably inside the astrometric scatter between
    the two filters, so tightening it would cost real matches instead."""
    col, m480, ra, dec, who = [], [], [], [], []
    for obs, v in pairs.items():
        a = Table.read(v["f212n"][1])
        b = Table.read(v["f480m"][1])
        ma, mb = abmag(a), abmag(b)
        idx, d2d, _ = b["skycoord"].match_to_catalog_sky(a["skycoord"])
        ok = d2d.arcsec < tol
        c = ma[idx[ok]] - mb[ok]
        mm = mb[ok]
        g = np.isfinite(c) & np.isfinite(mm)
        col.append(c[g])
        m480.append(mm[g])
        ra.append(b["skycoord"].ra.deg[ok][g])
        dec.append(b["skycoord"].dec.deg[ok][g])
        who.append(np.full(g.sum(), obs))
        print(f"  {obs}: {len(b):,} LW, {ok.sum():,} matched, {g.sum():,} with colour",
              flush=True)
    return (np.concatenate(col), np.concatenate(m480), np.concatenate(ra),
            np.concatenate(dec), np.concatenate(who))


def load_matched(pairs, force=False):
    """Matched arrays, from CACHE when it was built from this exact input set."""
    fp = fingerprint(pairs)
    if not force and os.path.exists(CACHE) and os.path.exists(STAMP):
        with open(STAMP) as fh:
            old = json.load(fh)
        if old.get("match") == fp:
            d = np.load(CACHE, allow_pickle=True)
            print(f"match cache hit: {len(d['col']):,} sources over {fp['n_obs']} fields")
            return d["col"], d["m480"], d["ra"], d["dec"], d["who"], fp
    print(f"matching {fp['n_obs']} field(s) at {MATCH_ARCSEC}\"", flush=True)
    col, m480, ra, dec, who = match_catalogs(pairs)
    np.savez(CACHE, col=col, m480=m480, ra=ra, dec=dec, who=who)
    print(f"{len(col):,} matched sources over {fp['n_obs']} fields")
    return col, m480, ra, dec, who, fp


# --------------------------------------------------------------------------
# density maps


def density(ra, dec, allra, alldec, pixel=PIXEL_ARCSEC, smooth=SMOOTH_ARCSEC):
    """Stars per square arcmin on a TAN grid, NaN outside the photometric
    footprint.  Zero density and no measurement are different statements, and a
    HiPS renders NaN as blank; coverage comes from ALL matched sources so an
    area with stars but none selected reads as a real zero."""
    ctr = SkyCoord(allra.mean() * u.deg, alldec.mean() * u.deg)
    pix = pixel / 3600.0
    cosd = np.cos(np.radians(ctr.dec.deg))
    nx = int(2 * (np.ptp(allra) * cosd / 2 + 0.02) / pix) + 1
    ny = int(2 * (np.ptp(alldec) / 2 + 0.02) / pix) + 1
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ctr.ra.deg, ctr.dec.deg]
    w.wcs.crpix = [nx / 2, ny / 2]
    w.wcs.cdelt = [-pix, pix]

    def hist(r, d):
        x, y = w.world_to_pixel_values(r, d)
        xi, yi = np.round(x).astype(int), np.round(y).astype(int)
        m = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
        h = np.zeros((ny, nx))
        np.add.at(h, (yi[m], xi[m]), 1.0)
        return h

    sig = smooth / pixel
    sel = gaussian_filter(hist(ra, dec), sig) / ((pixel / 60.0) ** 2)
    cov = gaussian_filter(hist(allra, alldec), sig * 3) > 0
    return np.where(cov, sel, np.nan).astype("float32"), w, cov


def build_hips(arr, w, name, level=None, threads=8):
    from reproject import reproject_interp
    from reproject.hips import reproject_to_hips
    dest = f"{OUT}/{name}"
    stage = dest + ".new"
    shutil.rmtree(stage, ignore_errors=True)
    reproject_to_hips((arr, w), coord_system_out="galactic", level=level,
                      reproject_function=reproject_interp,
                      output_directory=stage, threads=threads, generate_moc=True)
    if not os.path.isdir(f"{stage}/Norder3"):
        raise RuntimeError(f"{name}: no Norder3; refusing to publish")
    p = f"{stage}/properties"
    txt = open(p).read().replace(f"obs_title            = {name}.new",
                                 f"obs_title            = {name}")
    open(p, "w").write(txt)
    old = dest + ".old"
    shutil.rmtree(old, ignore_errors=True)
    if os.path.isdir(dest):
        os.rename(dest, old)
    os.rename(stage, dest)
    shutil.rmtree(old, ignore_errors=True)
    print(f"  BUILT {dest}", flush=True)


def write_density(arr, w, name):
    hdr = w.to_header()
    hdr["BUNIT"] = "stars/arcmin2"
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
    rc = ((m480 > 14) & (m480 < 19) & (col > -2.5) & (col < 3.5)
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


def build_ultrared(col, m480, ra, dec, who):
    """Catalogue rather than a density map: too few sources over too few fields
    for a meaningful surface density, and the positions are the useful product.
    NOT vetted by eye -- an extreme colour can be a genuinely embedded object or
    a mismatch between the two filters."""
    sel = col > ULTRARED_CUT
    fields = " ".join(sorted(set(np.asarray(who)[sel].tolist())))
    print(f"ultra-red: {sel.sum()} sources with F212N-F480M > {ULTRARED_CUT} "
          f"over {len(fields.split())} field(s)")
    order = np.argsort(-col[sel])
    t = Table({"ra": ra[sel][order], "dec": dec[sel][order],
               "color_f212n_f480m": col[sel][order],
               "mag_ab_f480m": m480[sel][order],
               "mag_ab_f212n": (col[sel] + m480[sel])[order],
               "obs": np.asarray(who)[sel][order]})
    t.meta["SELECT"] = f"F212N-F480M > {ULTRARED_CUT} AB"
    t.meta["FIELDS"] = fields
    t.meta["MATCHTOL"] = f"{MATCH_ARCSEC} arcsec"
    t.write(f"{WEB}/jwst_ultrared_stars.ecsv", overwrite=True)
    t.write(f"{WEB}/jwst_ultrared_stars.fits", overwrite=True)
    rows = [{"ra": round(float(r), 6), "dec": round(float(d), 6),
             "color": round(float(c), 2), "f480m": round(float(m), 2), "obs": str(o)}
            for r, d, c, m, o in zip(t["ra"], t["dec"], t["color_f212n_f480m"],
                                     t["mag_ab_f480m"], t["obs"])]
    doc = {"name": f"JWST ultra-red (F212N-F480M > {ULTRARED_CUT})",
           "select": f"F212N-F480M > {ULTRARED_CUT} AB, {MATCH_ARCSEC}\" match",
           "fields": fields, "n": len(rows), "sources": rows}
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
        if not os.path.isdir(f"{stage}/Norder3"):
            raise RuntimeError(f"{stage}: no Norder3; refusing to swap in")
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
    ap.add_argument("--publish", action="store_true",
                    help="copy the HiPS layers into the web root when done")
    ap.add_argument("--push-remote", action="store_true",
                    help="also mirror the published overlays to starformation "
                         "(needs working non-interactive ssh)")
    ap.add_argument("--push-only", action="store_true",
                    help="mirror whatever is already published to starformation "
                         "and exit; takes no lock and builds nothing")
    ap.add_argument("--refit", action="store_true",
                    help="report the RC ridge and exit without building")
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--only", choices=("red", "rc", "ultrared"), action="append",
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

    pairs = latest_pairs()
    if not pairs:
        print(f"no vetted catalogue pairs under {CAT}")
        return 1
    print(f"{len(pairs)} observation(s) vetted in both filters: {', '.join(pairs)}")

    fp = fingerprint(pairs)
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
        col, m480, ra, dec, who, fp = load_matched(pairs, force=a.force)
        report_ridge(col, m480)
        if a.refit:
            return 0
        want = set(a.only) if a.only else {"red", "rc", "ultrared"}
        if "red" in want:
            build_red_stars(col, m480, ra, dec, a.level, a.threads)
        if "rc" in want:
            build_rc(col, m480, ra, dec, a.level, a.threads)
        if "ultrared" in want:
            build_ultrared(col, m480, ra, dec, who)
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
        full = want == {"red", "rc", "ultrared"}
        stamp = {"match": fp, "when": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if full:
            stamp["built"] = fp
        elif os.path.exists(STAMP):
            with open(STAMP) as fh:
                stamp["built"] = json.load(fh).get("built")
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
