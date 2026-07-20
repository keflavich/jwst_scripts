#!/usr/bin/env python
"""
Retro-fix flipped HiPS in-place, cheaply.

The orientation bug was AVM-metadata-only: save_rgb flips the pixels but used
to embed the raw (un-flipped) FITS WCS, so reproject_to_hips produced flipped
HiPS.  The PNG pixels themselves are already correct, so any previously-made
PNG can be corrected without rerunning the reproject/RGB pipeline -- just
re-embed the corrected AVM (via jwst_rgb.save_rgb._avm_matching_pixels) and
regenerate the HiPS.

Per png dir you supply the target-grid FITS header (for the WCS) and whether
that grid is a MIRI target (MIRI target grids use transpose=ROTATE_180; NIRCam
use None).  flip is always -1, matching the scripts.

Usage:
  reembed_avm_hips.py --png-dir DIR --tgt-header FITS [--miri] [--no-hips]
                      [--glob '*.png'] [--limit N]
"""
import argparse
import glob
import os
import shutil
import sys

from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image
from tqdm import tqdm
from reproject import reproject_interp
from reproject.hips import reproject_to_hips

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from jwst_rgb.save_rgb import _faithful_flipped_avm  # noqa: E402


def load_tgt_wcs(tgt_header_path):
    """True celestial WCS of the target grid (not via a lossy pyavm AVM)."""
    try:
        hdr = fits.getheader(tgt_header_path, ext=('SCI', 1))
    except (KeyError, IndexError):
        hdr = fits.getheader(tgt_header_path)
    return WCS(hdr).celestial


def reembed_one(png, tgt_wcs, transpose, make_hips):
    im = Image.open(png)
    shape = (im.size[1], im.size[0])  # ny, nx
    corrected = _faithful_flipped_avm(tgt_wcs, shape, flip=-1, transpose=transpose)
    tmp = os.path.join(os.path.dirname(png), 'avm_' + os.path.basename(png))
    corrected.embed(png, tmp)
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
    p.add_argument('--tgt-header', required=True,
                   help='FITS whose SCI header holds the target-grid WCS.')
    p.add_argument('--miri', action='store_true',
                   help='Target grid is MIRI -> transpose=ROTATE_180 (else None).')
    p.add_argument('--no-hips', action='store_true',
                   help='Only re-embed AVM; skip HiPS regeneration.')
    p.add_argument('--glob', default='*.png')
    p.add_argument('--limit', type=int, default=None)
    args = p.parse_args()

    transpose = Image.ROTATE_180 if args.miri else None
    tgt_wcs = load_tgt_wcs(args.tgt_header)

    pngs = sorted(glob.glob(os.path.join(args.png_dir, args.glob)))
    if args.limit:
        pngs = pngs[:args.limit]
    print(f"Re-embedding {len(pngs)} PNGs in {args.png_dir} "
          f"(transpose={'ROTATE_180' if args.miri else 'None'}, "
          f"hips={not args.no_hips})")

    n_ok = 0
    for png in pngs:
        try:
            reembed_one(png, tgt_wcs, transpose, make_hips=not args.no_hips)
            n_ok += 1
            print(f"  [{n_ok}/{len(pngs)}] {os.path.basename(png)}")
        except (ValueError, OSError) as e:
            print(f"  SKIP {os.path.basename(png)}: {e!r}")

    print(f"Done: {n_ok}/{len(pngs)} corrected in {args.png_dir}")


if __name__ == '__main__':
    main()
