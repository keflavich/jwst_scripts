#!/usr/bin/env python
"""
Restore correct HiPS by re-embedding the RAW AVM (AVM.from_header of the target
grid) and regenerating HiPS.

Background: a buggy orientation checker (which omitted the vertical flip that
reproject_to_hips applies internally, reproject/utils.py [:, ::-1]) led to an
AVM "flip fix" in save_rgb that actually double-flipped every regenerated /
retro-fixed HiPS.  The raw AVM.from_header(target_header) is correct because
reproject_to_hips flips the PNG to match it.  This tool undoes the damage:
pixels are untouched, only the embedded AVM + the HiPS tiles are rewritten.

Usage:
  restore_raw_avm_hips.py --png-dir DIR --tgt-header FITS [--no-hips]
                          [--glob '*.png'] [--limit N]
"""
import argparse
import glob
import os
import shutil
import sys

import pyavm
from astropy.io import fits
from tqdm import tqdm
from reproject import reproject_interp
from reproject.hips import reproject_to_hips


def load_tgt_avm(tgt_header_path):
    """Raw AVM.from_header of the target grid (the correct, original AVM)."""
    try:
        hdr = fits.getheader(tgt_header_path, ext=('SCI', 1))
    except (KeyError, IndexError):
        hdr = fits.getheader(tgt_header_path)
    return pyavm.AVM.from_header(hdr)


def restore_one(png, raw_avm, make_hips):
    tmp = os.path.join(os.path.dirname(png), 'avm_' + os.path.basename(png))
    raw_avm.embed(png, tmp)
    shutil.move(tmp, png)
    if make_hips:
        hips_dir = png.replace('.png', '_hips')
        if os.path.exists(hips_dir):
            shutil.rmtree(hips_dir)
        reproject_to_hips(png, level=None, reproject_function=reproject_interp,
                          output_directory=hips_dir, threads=8,
                          coord_system_out='galactic', progress_bar=tqdm)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--png-dir', required=True)
    p.add_argument('--tgt-header', required=True)
    p.add_argument('--no-hips', action='store_true')
    p.add_argument('--glob', default='*.png')
    p.add_argument('--limit', type=int, default=None)
    args = p.parse_args()

    raw_avm = load_tgt_avm(args.tgt_header)
    pngs = sorted(glob.glob(os.path.join(args.png_dir, args.glob)))
    if args.limit:
        pngs = pngs[:args.limit]
    print(f"Restoring raw AVM in {len(pngs)} PNGs in {args.png_dir} "
          f"(hips={not args.no_hips})")
    n = 0
    for png in pngs:
        try:
            restore_one(png, raw_avm, make_hips=not args.no_hips)
            n += 1
        except (ValueError, OSError) as e:
            print(f"  SKIP {os.path.basename(png)}: {e!r}")
    print(f"  restored {n}/{len(pngs)} in {args.png_dir}")


if __name__ == '__main__':
    main()
