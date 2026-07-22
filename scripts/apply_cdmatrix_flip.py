#!/usr/bin/env python
"""
Apply a faithful CDMatrix AVM in a given flip to every PNG in a dir + regen HiPS.

This is the fix step: embed an AVM that is an EXACT dihedral of the FITS WCS
(CDMatrix -- no pyavm Scale+Rotation degeneracy near PA=90) in the flip that
matches the pixels (determined once, by fix_avm_cdmatrix / by-eye).

Usage:
  apply_cdmatrix_flip.py --png-dir DIR --tgt-header FITS --flip rot180
                         [--no-hips] [--glob '*.png']
"""
import argparse
import glob
import os
import shutil

import numpy as np
import pyavm
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image
from tqdm import tqdm
from reproject import reproject_interp
from reproject.hips import reproject_to_hips

Image.MAX_IMAGE_PIXELS = None


def load_wcs_shape(path):
    h = fits.getheader(path)
    if h.get('NAXIS', 0) < 2:
        try:
            h = fits.getheader(path, ext=('SCI', 1))
        except (KeyError, IndexError):
            pass
    return WCS(h).celestial, int(h['NAXIS2']), int(h['NAXIS1'])


def cdmatrix_avm(fwcs, ny, nx, flip):
    w = WCS(naxis=2)
    w.wcs.crval = fwcs.wcs.crval[:2]
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    cd = np.array(fwcs.pixel_scale_matrix, float)
    crpix = np.array(fwcs.wcs.crpix[:2], float)
    fr = flip in ('flipud', 'rot180')
    fc = flip in ('fliplr', 'rot180')
    if fc:
        crpix[0] = nx + 1 - crpix[0]
        cd[:, 0] = -cd[:, 0]
    if fr:
        crpix[1] = ny + 1 - crpix[1]
        cd[:, 1] = -cd[:, 1]
    w.wcs.crpix = crpix
    w.wcs.cd = cd
    w.pixel_shape = (nx, ny)
    a = pyavm.AVM.from_wcs(w, shape=(ny, nx))
    a.Spatial.CDMatrix = [cd[0, 0], cd[0, 1], cd[1, 0], cd[1, 1]]
    a.Spatial.Scale = None
    a.Spatial.Rotation = None
    return a


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--png-dir', required=True)
    p.add_argument('--tgt-header', required=True)
    p.add_argument('--flip', required=True,
                   choices=['identity', 'flipud', 'fliplr', 'rot180'])
    p.add_argument('--no-hips', action='store_true')
    p.add_argument('--glob', default='*.png')
    args = p.parse_args()

    fwcs, ny, nx = load_wcs_shape(args.tgt_header)
    pngs = sorted(glob.glob(os.path.join(args.png_dir, args.glob)))
    print(f"Applying CDMatrix flip={args.flip} to {len(pngs)} PNGs in "
          f"{args.png_dir} (hips={not args.no_hips})")
    n = 0
    for png in pngs:
        try:
            im = Image.open(png)
            pny, pnx = im.size[1], im.size[0]
            if (pny, pnx) != (ny, nx):
                # PNG grid differs from header; skip (needs its own header)
                print(f"  SKIP {os.path.basename(png)}: shape {(pny, pnx)} != "
                      f"{(ny, nx)}")
                continue
            avm = cdmatrix_avm(fwcs, ny, nx, args.flip)
            tmp = os.path.join(os.path.dirname(png), 'avm_' + os.path.basename(png))
            avm.embed(png, tmp)
            shutil.move(tmp, png)
            if not args.no_hips:
                hd = png.replace('.png', '_hips')
                if os.path.exists(hd):
                    shutil.rmtree(hd)
                reproject_to_hips(png, coord_system_out='galactic', level=None,
                                  reproject_function=reproject_interp,
                                  output_directory=hd, threads=8,
                                  progress_bar=tqdm)
            n += 1
        except (ValueError, OSError) as e:
            print(f"  SKIP {os.path.basename(png)}: {e!r}")
    print(f"  applied {n}/{len(pngs)} in {args.png_dir}")


if __name__ == '__main__':
    main()
