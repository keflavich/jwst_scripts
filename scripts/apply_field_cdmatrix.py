#!/usr/bin/env python
"""
Apply a CDMatrix AVM in a given flip to every png dir of a target + regen HiPS.
Reuses the png-dir / grid map from retrofix_orientation.TARGETS.

The correct HiPS is `original pixels + CDMatrix(true WCS, identity)` for every
field (proven equivalent to the confirmed gc2211 rot180+rot180 case).  Fields
whose pixels are currently at their original save_rgb orientation therefore
take --flip identity; the two fields left at rot180 pixels (gc2211, sgrc) take
--flip rot180.

Usage:
  apply_field_cdmatrix.py <target> --flip identity [--no-hips]
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from retrofix_orientation import TARGETS, find_grid_fits  # noqa: E402
from apply_cdmatrix_flip import load_wcs_shape, cdmatrix_avm  # noqa: E402

import shutil
from PIL import Image
from tqdm import tqdm
from reproject import reproject_interp
from reproject.hips import reproject_to_hips

Image.MAX_IMAGE_PIXELS = None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('target', choices=sorted(TARGETS))
    p.add_argument('--flip', default='identity',
                   choices=['identity', 'flipud', 'fliplr', 'rot180'])
    p.add_argument('--no-hips', action='store_true')
    p.add_argument('--glob', default='*.png')
    args = p.parse_args()

    reproj_dir, dirs = TARGETS[args.target]
    for png_dir, grid, _t in dirs:
        if not os.path.isdir(png_dir):
            print(f"SKIP {png_dir}: missing")
            continue
        ref = grid if (os.path.sep in grid and os.path.exists(grid)) \
            else find_grid_fits(reproj_dir, grid)
        if ref is None:
            print(f"SKIP {png_dir}: no grid FITS for {grid}")
            continue
        fwcs, ny, nx = load_wcs_shape(ref)
        pngs = sorted(glob.glob(os.path.join(png_dir, args.glob)))
        print(f"[{args.target}] {png_dir}: {len(pngs)} pngs, flip={args.flip}, "
              f"ref={os.path.basename(ref)}")
        n = 0
        for png in pngs:
            try:
                im = Image.open(png)
                if (im.size[1], im.size[0]) != (ny, nx):
                    print(f"  SKIP {os.path.basename(png)}: shape mismatch")
                    continue
                avm = cdmatrix_avm(fwcs, ny, nx, args.flip)
                tmp = os.path.join(png_dir, 'avm_' + os.path.basename(png))
                avm.embed(png, tmp)
                shutil.move(tmp, png)
                if not args.no_hips:
                    hd = png.replace('.png', '_hips')
                    if os.path.exists(hd):
                        shutil.rmtree(hd)
                    reproject_to_hips(png, coord_system_out='galactic',
                                      level=None,
                                      reproject_function=reproject_interp,
                                      output_directory=hd, threads=8,
                                      progress_bar=tqdm)
                n += 1
            except (ValueError, OSError) as e:
                print(f"  SKIP {os.path.basename(png)}: {e!r}")
        print(f"  applied {n}/{len(pngs)}")


if __name__ == '__main__':
    main()
