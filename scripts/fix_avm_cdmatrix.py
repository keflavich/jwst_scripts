#!/usr/bin/env python
"""
Definitive per-frame AVM fixer (two-step, discrete).

Step 1 (FITS<->AVM): the AVM must be an exact dihedral of the FITS WCS.  We
build it as a faithful CDMatrix from the FITS WCS (no pyavm Scale+Rotation
degeneracy near PA=90).

Step 2 (AVM<->HiPS orientation): the only freedom left is which of the 4
dimension-preserving flips matches the PNG pixels.  Decide it with the
BRIGHTEST REAL UNSATURATED well-measured catalog stars (saturated blobs
excluded): project each onto the PNG (in reproject_to_hips' flipud frame) and
require it to sit centered on a local point-source peak, with peak brightness
tracking catalog flux.  Bright real stars are rare, so this is unambiguous even
in crowded fields.

--embed writes the CDMatrix AVM in the winning flip into the PNG.

Usage:
  fix_avm_cdmatrix.py --fits ref.fits --png img.png --catalog cat.fits
                      [--nstars 8] [--embed]
"""
import argparse
import json
import shutil

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from PIL import Image
import pyavm

Image.MAX_IMAGE_PIXELS = None


def fits_wcs_shape(path):
    h = fits.getheader(path)
    if h.get('NAXIS', 0) < 2:
        h = fits.getheader(path, ext=('SCI', 1))
    return WCS(h).celestial, int(h['NAXIS2']), int(h['NAXIS1'])


def oriented(fwcs, ny, nx, kind):
    w = WCS(naxis=2)
    w.wcs.crval = fwcs.wcs.crval[:2]
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    cd = np.array(fwcs.pixel_scale_matrix, float)
    crpix = np.array(fwcs.wcs.crpix[:2], float)
    fr = kind in ('flipud', 'rot180')
    fc = kind in ('fliplr', 'rot180')
    if fc:
        crpix[0] = nx + 1 - crpix[0]
        cd[:, 0] = -cd[:, 0]
    if fr:
        crpix[1] = ny + 1 - crpix[1]
        cd[:, 1] = -cd[:, 1]
    w.wcs.crpix = crpix
    w.wcs.cd = cd
    w.pixel_shape = (nx, ny)
    return w


def cdmatrix_avm(w, ny, nx):
    a = pyavm.AVM.from_wcs(w, shape=(ny, nx))
    cd = w.pixel_scale_matrix
    a.Spatial.CDMatrix = [cd[0, 0], cd[0, 1], cd[1, 0], cd[1, 1]]
    a.Spatial.Scale = None
    a.Spatial.Rotation = None
    return a


def select_real_stars(catalog, fwcs, ny, nx, nstars, edge=80):
    t = Table.read(catalog)
    sc = t['skycoord']
    flux = np.array(t['flux'], float)
    ok = np.isfinite(flux) & (flux > 0)
    for col, want in [('qfit', 'lt'), ('is_saturated', 'false'),
                      ('replaced_saturated', 'false'), ('flags', 'zero')]:
        if col in t.colnames:
            v = np.array(t[col])
            if want == 'lt':
                ok &= v.astype(float) < 0.05
            elif want == 'false':
                ok &= ~v.astype(bool)
            elif want == 'zero':
                ok &= v.astype(int) == 0
    ra = sc.ra.deg[ok]
    dec = sc.dec.deg[ok]
    flux = flux[ok]
    fx, fy = fwcs.world_to_pixel_values(ra, dec)
    inside = (fx > edge) & (fx < nx - edge) & (fy > edge) & (fy < ny - edge)
    ra, dec, flux = ra[inside], dec[inside], flux[inside]
    order = np.argsort(flux)[::-1]
    picked = []
    for i in order:
        if all(((ra[i] - ra[j]) ** 2 + (dec[i] - dec[j]) ** 2) > (6 / 3600.) ** 2
               for j in picked):
            picked.append(i)
        if len(picked) >= nstars:
            break
    return ra[picked], dec[picked], flux[picked]


def star_hit(lum, x, y, win=8, center=2):
    """Return peak brightness if (x,y) is centered on a local max, else None."""
    ny, nx = lum.shape
    xi, yi = int(round(x)), int(round(y))
    if not (win <= xi < nx - win and win <= yi < ny - win):
        return None
    sub = lum[yi - win:yi + win + 1, xi - win:xi + win + 1]
    dy, dx = np.unravel_index(np.nanargmax(sub), sub.shape)
    # peak must be near the predicted center (a real centered point source)
    if abs(dx - win) <= center and abs(dy - win) <= center:
        return float(sub[dy, dx])
    return None


def score_flip(lum, w, ra, dec, flux):
    xs, ys = w.world_to_pixel_values(ra, dec)
    hits, bright, fl = 0, [], []
    for x, y, f in zip(xs, ys, flux):
        b = star_hit(lum, x, y)
        if b is not None:
            hits += 1
            bright.append(b)
            fl.append(np.log10(f))
    corr = None
    if len(bright) >= 3:
        from scipy.stats import spearmanr
        corr = float(spearmanr(fl, bright).correlation)
    return {'n_hits': hits, 'n_stars': len(ra),
            'hit_frac': round(hits / max(1, len(ra)), 3),
            'flux_corr': None if corr is None else round(corr, 3)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fits', required=True)
    p.add_argument('--png', required=True)
    p.add_argument('--catalog', required=True)
    p.add_argument('--nstars', type=int, default=8)
    p.add_argument('--embed', action='store_true')
    p.add_argument('--min-hitfrac', type=float, default=0.6, dest='minhf')
    args = p.parse_args()

    fwcs, ny, nx = fits_wcs_shape(args.fits)
    ra, dec, flux = select_real_stars(args.catalog, fwcs, ny, nx, args.nstars)
    if len(ra) < 3:
        print(json.dumps({'error': f'only {len(ra)} real unsaturated stars'}))
        return

    im = np.array(Image.open(args.png)).astype(float)
    lum = im[:, :, :3].sum(axis=2) if im.ndim == 3 else im
    lum = lum[::-1]  # reproject_to_hips flips the PNG vertically

    res = {}
    for kind in ('identity', 'flipud', 'fliplr', 'rot180'):
        res[kind] = score_flip(lum, oriented(fwcs, ny, nx, kind), ra, dec, flux)

    def key(k):
        return (res[k]['hit_frac'], res[k]['flux_corr'] or -1)
    best = max(res, key=key)
    confident = (res[best]['hit_frac'] >= args.minhf and
                 sorted((res[k]['hit_frac'] for k in res))[-1] >
                 sorted((res[k]['hit_frac'] for k in res))[-2] + 0.15)
    out = {'png': args.png, 'n_real_stars': int(len(ra)),
           'flips': res, 'best': best, 'confident': bool(confident)}
    if args.embed and confident:
        w = oriented(fwcs, ny, nx, best)
        avm = cdmatrix_avm(w, ny, nx)
        tmp = args.png + '.avm.png'
        avm.embed(args.png, tmp)
        shutil.move(tmp, args.png)
        out['embedded'] = best
    elif args.embed:
        out['embedded'] = False
        out['reason'] = 'not confident; not embedding'
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
