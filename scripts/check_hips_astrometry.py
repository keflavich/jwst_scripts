#!/usr/bin/env python
"""
Measure the astrometric offset of a HiPS directly from its TILES.

Existing tooling only ever checked the source PNG (check_star_positions.py) or
correlated morphology (check_hips_orientation.py).  Neither tells you the
offset as served, and `reproject.hips.reproject_from_hips` is a no-arg stub in
this env, so we sample the HiPS tiles ourselves.

Method
------
1. Take N bright, well-separated reference stars from an astrometric catalog
   (or detected in a reference FITS).
2. Around each star, build a small gnomonic (TAN) cutout by sampling the HiPS
   tiles: sky -> HEALPix nested index at order (hips_order + 9) -> tile index +
   in-tile (x, y) via bit de-interleaving.
3. Locate the actual brightness peak in the cutout and report its offset from
   the catalog position, in arcsec, as dRA*cos(dec) and dDec.

A correct HiPS gives small (sub-arcsec / few-pixel) offsets that are consistent
star to star.  A rot180 error gives offsets that grow linearly with distance
from the field centre and reverse sign across it -- which is why a single
central star can look fine while the field is flipped.

LIMITS -- when this test says nothing
-------------------------------------
It needs STARS in the image.  It is meaningless, and will happily report a
confident-looking but spurious offset, for:
  * star-subtracted images (e.g. *_merged_longwave_narrowband without
    "_withstars") -- the sources it would match on have been removed;
  * MIRI-wavelength composites, where the NIRCam reference catalogue's stars
    are largely absent.
Treat SNR < ~8 as "untestable", not "bad", and check whether the product is
star-subtracted before believing any offset at all.

Usage
-----
  check_hips_astrometry.py --hips DIR --catalog cat.fits [--nstars 8]
  check_hips_astrometry.py --hips DIR --fits ref.fits   [--nstars 8]
"""
import argparse
import glob
import json
import os
import re

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
import astropy.units as u
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

TILE = 512
TILE_BITS = 9          # 512 = 2**9


# ---------------------------------------------------------------- HiPS access
def hips_order(hips_dir):
    props = os.path.join(hips_dir, "properties")
    order = None
    frame = "equatorial"
    with open(props) as fh:
        for line in fh:
            if line.startswith("hips_order"):
                order = int(line.split("=")[1].strip())
            elif line.startswith("hips_frame"):
                frame = line.split("=")[1].strip()
    # use the deepest order actually present on disk
    present = sorted(int(m.group(1)) for m in
                     (re.match(r"Norder(\d+)$", os.path.basename(d))
                      for d in glob.glob(os.path.join(hips_dir, "Norder*")))
                     if m)
    if present:
        order = max(present) if order is None else min(order, max(present))
    return order, frame


def deinterleave(k):
    """nested sub-index -> (x, y) within a HiPS tile.

    Calibrated 2026-08-23 against a control HiPS built from a FITS with a known
    WCS (CONTROL_F187N): the ODD bits give the column and the EVEN bits give the
    row, with NO vertical flip.  That combination reproduces the control's
    astrometry to (0.00, 0.00) arcsec at SNR 27.5; the three other
    bit-order/flip combinations land 0.5-7 arcsec off at much lower SNR.
    """
    a = np.zeros_like(k)
    b_ = np.zeros_like(k)
    for b in range(TILE_BITS):
        a |= ((k >> (2 * b)) & 1) << b
        b_ |= ((k >> (2 * b + 1)) & 1) << b
    # a = even bits -> row, b_ = odd bits -> column
    return b_, a


class HipsSampler:
    """Sample a local HiPS by sky coordinate."""

    def __init__(self, hips_dir, flip_y=False):
        from astropy_healpix import HEALPix
        self.dir = hips_dir
        self.order, self.frame = hips_order(hips_dir)
        self.flip_y = flip_y
        fr = "galactic" if self.frame.startswith("gal") else "icrs"
        self.frame_name = fr
        self.hp_hi = HEALPix(nside=2 ** (self.order + TILE_BITS), order="nested",
                             frame=fr)
        self._cache = {}

    def _tile(self, ipix):
        if ipix in self._cache:
            return self._cache[ipix]
        d = (ipix // 10000) * 10000
        path = os.path.join(self.dir, f"Norder{self.order}", f"Dir{d}",
                            f"Npix{ipix}.png")
        img = None
        if os.path.exists(path):
            a = np.asarray(Image.open(path).convert("L"), float)
            if a.shape == (TILE, TILE):
                img = a
        if len(self._cache) > 256:
            self._cache.clear()
        self._cache[ipix] = img
        return img

    def sample(self, coords):
        """coords: SkyCoord array -> float array of tile values (nan if absent)."""
        c = coords.transform_to("galactic" if self.frame_name == "galactic"
                                else "icrs")
        ipix_hi = self.hp_hi.lonlat_to_healpix(c.spherical.lon, c.spherical.lat)
        ipix_hi = np.asarray(ipix_hi, dtype=np.int64)
        tile_ix = ipix_hi >> (2 * TILE_BITS)
        sub = ipix_hi & ((1 << (2 * TILE_BITS)) - 1)
        x, y = deinterleave(sub)
        row = (TILE - 1 - y) if self.flip_y else y
        out = np.full(len(ipix_hi), np.nan)
        for t in np.unique(tile_ix):
            img = self._tile(int(t))
            if img is None:
                continue
            m = tile_ix == t
            out[m] = img[row[m], x[m]]
        return out

    def cutout(self, center, half_px, pixscale_arcsec):
        """Gnomonic cutout around `center`; returns (image, wcs)."""
        n = 2 * half_px + 1
        w = WCS(naxis=2)
        w.wcs.crpix = [half_px + 1, half_px + 1]
        w.wcs.cdelt = [-pixscale_arcsec / 3600.0, pixscale_arcsec / 3600.0]
        w.wcs.crval = [center.icrs.ra.deg, center.icrs.dec.deg]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        yy, xx = np.mgrid[0:n, 0:n]
        sky = w.pixel_to_world(xx.ravel(), yy.ravel())
        vals = self.sample(sky).reshape(n, n)
        return vals, w


# ---------------------------------------------------------------- star lists
def stars_from_catalog(path, nstars, center=None, radius_arcmin=None):
    t = Table.read(path)
    cols = {c.lower(): c for c in t.colnames}
    ra = next((cols[c] for c in ("ra", "raj2000", "ra_deg", "skycoord.ra") if c in cols), None)
    dec = next((cols[c] for c in ("dec", "dej2000", "dec_deg", "skycoord.dec") if c in cols), None)
    if ra is None or dec is None:
        raise SystemExit(f"no RA/Dec columns in {path}: {t.colnames[:20]}")
    sc = SkyCoord(np.asarray(t[ra], float) * u.deg, np.asarray(t[dec], float) * u.deg)
    # brightness column, if any (prefer a long-wavelength mag)
    magcol = next((cols[c] for c in ("mag_ab_f405n", "mag_ab_f410m", "mag_ab_f444w",
                                     "mag_ab_f356w", "mag", "flux") if c in cols), None)
    keep = np.isfinite(sc.ra.deg) & np.isfinite(sc.dec.deg)
    if center is not None and radius_arcmin is not None:
        keep &= sc.separation(center).arcmin < radius_arcmin
    sc = sc[keep]
    if magcol is not None:
        m = np.asarray(t[magcol], float)[keep]
        good = np.isfinite(m)
        sc, m = sc[good], m[good]
        order = np.argsort(m)          # brightest = smallest mag
        sc = sc[order]
    return sc


def spread_out(sc, nstars, min_sep_arcsec=8.0):
    picked = []
    for c in sc:
        if all(c.separation(p).arcsec > min_sep_arcsec for p in picked):
            picked.append(c)
        if len(picked) >= nstars:
            break
    return SkyCoord([p.ra for p in picked], [p.dec for p in picked])


# ---------------------------------------------------------------- main check
def measure(hips_dir, stars, pixscale, half_px, flip_y):
    s = HipsSampler(hips_dir, flip_y=flip_y)
    rows = []
    for st in stars:
        img, w = s.cutout(st, half_px, pixscale)
        if not np.isfinite(img).any():
            rows.append(dict(ok=False, reason="no tile coverage"))
            continue
        finite = np.isfinite(img)
        if finite.sum() < 0.3 * img.size:
            rows.append(dict(ok=False, reason="sparse coverage"))
            continue
        a = np.where(finite, img, np.nan)
        med = np.nanmedian(a)
        peak = np.nanmax(a)
        if not np.isfinite(peak) or peak <= med:
            rows.append(dict(ok=False, reason="flat cutout"))
            continue
        iy, ix = np.unravel_index(np.nanargmax(a), a.shape)
        found = w.pixel_to_world(ix, iy)
        d_ra = (found.ra - st.ra).to(u.arcsec).value * np.cos(st.dec.rad)
        d_dec = (found.dec - st.dec).to(u.arcsec).value
        rows.append(dict(ok=True,
                         ra=float(st.ra.deg), dec=float(st.dec.deg),
                         d_ra_arcsec=round(float(d_ra), 3),
                         d_dec_arcsec=round(float(d_dec), 3),
                         sep_arcsec=round(float(np.hypot(d_ra, d_dec)), 3),
                         contrast=round(float((peak - med) / (np.nanstd(a) + 1e-9)), 2)))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hips", required=True, nargs="+")
    p.add_argument("--catalog")
    p.add_argument("--fits", help="reference FITS (for field centre / star detection)")
    p.add_argument("--nstars", type=int, default=8)
    p.add_argument("--pixscale", type=float, default=0.05, help="arcsec/px of cutout")
    p.add_argument("--half", type=int, default=40, help="cutout half-size in px")
    p.add_argument("--flip-y", dest="flip_y", action="store_true", default=False)
    p.add_argument("--no-flip-y", dest="flip_y", action="store_false")
    p.add_argument("--json")
    args = p.parse_args()

    center, radius = None, None
    if args.fits:
        with fits.open(args.fits) as hdul:
            idx = next((i for i, h in enumerate(hdul)
                        if h.header.get("EXTNAME") == "SCI"), 0)
            hdr = hdul[idx].header
        w = WCS(hdr).celestial
        ny, nx = hdr["NAXIS2"], hdr["NAXIS1"]
        center = w.pixel_to_world(nx / 2, ny / 2)
        radius = 0.5 * max(ny, nx) * np.abs(
            w.proj_plane_pixel_scales()[0].to("arcsec").value) / 60.0

    if not args.catalog:
        raise SystemExit("--catalog required")
    sc = stars_from_catalog(args.catalog, args.nstars, center, radius)
    stars = spread_out(sc, args.nstars)
    print(f"# using {len(stars)} catalog stars"
          f"{' within %.1f arcmin of field centre' % radius if radius else ''}")

    report = {}
    for hd in args.hips:
        name = os.path.basename(hd.rstrip("/"))
        rows = measure(hd, stars, args.pixscale, args.half, args.flip_y)
        good = [r for r in rows if r.get("ok")]
        seps = [r["sep_arcsec"] for r in good]
        summary = {
            "n_stars": len(rows),
            "n_measured": len(good),
            "median_sep_arcsec": round(float(np.median(seps)), 3) if seps else None,
            "max_sep_arcsec": round(float(np.max(seps)), 3) if seps else None,
            "median_d_ra": round(float(np.median([r["d_ra_arcsec"] for r in good])), 3) if good else None,
            "median_d_dec": round(float(np.median([r["d_dec_arcsec"] for r in good])), 3) if good else None,
            "stars": rows,
        }
        report[name] = summary
        print(f"\n=== {name} ===")
        print(f"  measured {len(good)}/{len(rows)} stars")
        if seps:
            print(f"  median sep = {summary['median_sep_arcsec']}\"  "
                  f"max = {summary['max_sep_arcsec']}\"  "
                  f"median offset = ({summary['median_d_ra']}, {summary['median_d_dec']})\"")
            for r in good:
                print(f"    {r['ra']:.6f} {r['dec']:+.6f}  "
                      f"d=({r['d_ra_arcsec']:+7.2f},{r['d_dec_arcsec']:+7.2f})\"  "
                      f"sep={r['sep_arcsec']:6.2f}\"  contrast={r['contrast']}")
        else:
            print("  no stars measurable: " +
                  ", ".join(sorted({r.get('reason', '?') for r in rows})))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
