#!/usr/bin/env python
"""
SUPERSEDED -- this tool re-embeds the RAW AVM, which is the bug.  Use
`jwst_rgb.save_rgb.avm_for_saved_png()` instead.  It refuses to run without
--i-know-this-reintroduces-the-offset.

Background: a buggy orientation checker (which omitted the vertical flip that
reproject_to_hips applies internally, reproject/utils.py [:, ::-1]) led to an
AVM "flip fix" in save_rgb that actually double-flipped every regenerated /
retro-fixed HiPS.  This tool was written to undo that damage, on the belief
that "the raw AVM.from_header(target_header) is correct because
reproject_to_hips flips the PNG to match it".

That belief is wrong, and the error it leaves is small enough to look right.
save_rgb reverses BOTH pixel axes (flip, then ROTATE_180) and reproject undoes
only one, so a WCS-aware reader reconstructs a 180 degree rotation of the
input.  The raw AVM keeps the FITS CRPIX; pyavm's lossy Scale+Rotation
round-trip happens to return the negated CD on its own, so orientation and
scale come out right and the reference pixel is the only thing wrong.  The
result is a pure translation of |N+1-2*crpix| pixels per axis -- zero when
CRPIX sits at the image centre, and 1.26" on a JWST stage 3 mosaic, which is
what every GC Treasury layer was serving.

Running this against such a layer re-introduces exactly that offset.

Usage:
  restore_raw_avm_hips.py --png-dir DIR --tgt-header FITS [--no-hips]
                          [--glob '*.png'] [--limit N]
"""
import argparse
import glob
import os
import shutil
import sys

import numpy as np
import pyavm
from astropy.io import fits
from PIL import Image
from tqdm import tqdm
from reproject import reproject_interp
from reproject.hips import reproject_to_hips


def load_tgt_avm(tgt_header_path):
    """Raw AVM.from_header of the target grid.

    NOT correct for a save_rgb PNG -- see the module docstring.  Kept so the
    tool still does what it says when explicitly overridden."""
    try:
        hdr = fits.getheader(tgt_header_path, ext=('SCI', 1))
    except (KeyError, IndexError):
        hdr = fits.getheader(tgt_header_path)
    return pyavm.AVM.from_header(hdr)


def restore_one(png, raw_avm, make_hips, rot180=False):
    # transpose=None targets (gc2211, w51 NIRCam grids, sickle non-470) come
    # out flipped with the raw AVM and need a 180-deg rotation of the pixels
    # (confirmed by-eye against VVV).  ROTATE_180 targets are already correct
    # with the raw AVM (rot180=False).
    if rot180:
        arr = np.array(Image.open(png))
        Image.fromarray(arr[::-1, ::-1]).save(png)
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
    p.add_argument('--rot180', action='store_true',
                   help='Rotate pixels 180deg before embedding (transpose=None '
                        'targets). NOT idempotent -- run exactly once per dir.')
    p.add_argument('--glob', default='*.png')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--i-know-this-reintroduces-the-offset', action='store_true',
                   dest='override',
                   help='Required.  This tool embeds the raw AVM, which leaves '
                        'the layer |N+1-2*crpix| pixels off per axis (1.26" on '
                        'a stage 3 mosaic).  Use avm_for_saved_png() instead.')
    args = p.parse_args()

    if not args.override:
        p.error(
            "refusing to run: re-embedding the raw AVM is the astrometric "
            "offset, not the fix for it -- it leaves the layer "
            '|N+1-2*crpix| pixels off per axis (1.26" measured on a GC '
            "Treasury stage 3 mosaic).  Use "
            "jwst_rgb.save_rgb.avm_for_saved_png(), or pass "
            "--i-know-this-reintroduces-the-offset if you really want the old "
            "behaviour.")

    raw_avm = load_tgt_avm(args.tgt_header)
    pngs = sorted(glob.glob(os.path.join(args.png_dir, args.glob)))
    if args.limit:
        pngs = pngs[:args.limit]
    print(f"Restoring raw AVM in {len(pngs)} PNGs in {args.png_dir} "
          f"(hips={not args.no_hips}, rot180={args.rot180})")
    n = 0
    for png in pngs:
        try:
            restore_one(png, raw_avm, make_hips=not args.no_hips,
                        rot180=args.rot180)
            n += 1
        except (ValueError, OSError) as e:
            print(f"  SKIP {os.path.basename(png)}: {e!r}")
    print(f"  restored {n}/{len(pngs)} in {args.png_dir}")


if __name__ == '__main__':
    main()
