#!/usr/bin/env python
"""
Restore a target's HiPS to the correct (raw AVM) orientation, undoing the
double-flip damage.  Reuses the png-dir / grid-FITS map from
retrofix_orientation.  Transpose is irrelevant here -- the raw AVM.from_header
is embedded verbatim and reproject_to_hips handles the pixel flip.

Usage:
  restore_target.py <target> [--no-hips]
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from retrofix_orientation import TARGETS, find_grid_fits  # noqa: E402
from restore_raw_avm_hips import load_tgt_avm, restore_one  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('target', choices=sorted(TARGETS))
    p.add_argument('--no-hips', action='store_true')
    p.add_argument('--glob', default='*.png')
    args = p.parse_args()

    reproj_dir, dirs = TARGETS[args.target]
    for png_dir, grid, _transpose in dirs:
        if not os.path.isdir(png_dir):
            print(f"SKIP {png_dir}: missing")
            continue
        if os.path.sep in grid and os.path.exists(grid):
            ref = grid
        else:
            ref = find_grid_fits(reproj_dir, grid)
        if ref is None:
            print(f"SKIP {png_dir}: no grid FITS for {grid}")
            continue
        raw_avm = load_tgt_avm(ref)
        pngs = sorted(glob.glob(os.path.join(png_dir, args.glob)))
        print(f"[{args.target}] {png_dir}: {len(pngs)} pngs, "
              f"ref={os.path.basename(ref)}")
        n = 0
        for png in pngs:
            try:
                restore_one(png, raw_avm, make_hips=not args.no_hips)
                n += 1
            except (ValueError, OSError) as e:
                print(f"  SKIP {os.path.basename(png)}: {e!r}")
        print(f"  restored {n}/{len(pngs)}")


if __name__ == '__main__':
    main()
