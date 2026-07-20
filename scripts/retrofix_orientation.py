#!/usr/bin/env python
"""
Retro-fix HiPS orientation for one target, in place, using the faithful
CDMatrix-from-true-WCS recipe (reembed_avm_hips.reembed_one).

The rotated GC fields need this because pyavm.AVM.from_header is lossy for
them; the fix re-embeds a faithful AVM built from the true grid WCS and
regenerates the HiPS.  Pixels are untouched.

Per target we know the transpose convention:
  * direct-import scripts (sgra/sgrb2/sgrc/arches/quintuplet/cloudef/brick/
    cloudc) call save_rgb with the default transpose=ROTATE_180 for EVERY
    png, regardless of the target grid.
  * gc2211 uses the wrapper -> None (all NIRCam).
  * sickle: ROTATE_180 for pngs_470, None otherwise.

For each png dir we need the true WCS of that grid: any single-filter
reprojected FITS on the grid works (they share the grid WCS).

Usage:
  retrofix_orientation.py <target>            # all dirs for the target
  retrofix_orientation.py <target> --no-hips  # AVM only (fast dry check)
"""
import argparse
import glob
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
# reembed_avm_hips is imported lazily inside main() so that importing this
# module only for TARGETS / find_grid_fits (e.g. from restore_target) does not
# require it.

ROT = Image.ROTATE_180

# target -> (data_reprojected dir, [(png_dir, grid_tag, transpose), ...])
# grid_tag is used to glob a reprojected FITS: *reprj*<grid_tag>*.fits
TARGETS = {
    'sgrb2': ('/orange/adamginsburg/jwst/sgrb2/NB/data_reprojected', [
        ('/orange/adamginsburg/jwst/sgrb2/pngs_150', 'f150', ROT),
        ('/orange/adamginsburg/jwst/sgrb2/pngs_466', 'f466', ROT),
    ]),
    'sgra': ('/orange/adamginsburg/jwst/sgra/data_reprojected', [
        ('/orange/adamginsburg/jwst/sgra/pngs_770', 'f770', ROT),
        ('/orange/adamginsburg/jwst/sgra/pngs_1280', 'f1280', ROT),
        ('/orange/adamginsburg/jwst/sgra/pngs_444', 'f444', ROT),
    ]),
    'sgrc': ('/orange/adamginsburg/jwst/sgrc/data_reprojected', [
        ('/orange/adamginsburg/jwst/sgrc/pngs_360', 'f360', ROT),
        ('/orange/adamginsburg/jwst/sgrc/pngs_480', 'f480', ROT),
    ]),
    'arches': ('/orange/adamginsburg/jwst/arches/data_reprojected', [
        ('/orange/adamginsburg/jwst/arches/pngs_212', 'f212', ROT),
        ('/orange/adamginsburg/jwst/arches/pngs_323', 'f323', ROT),
    ]),
    'quintuplet': ('/orange/adamginsburg/jwst/quintuplet/data_reprojected', [
        ('/orange/adamginsburg/jwst/quintuplet/pngs_212', 'f212', ROT),
        ('/orange/adamginsburg/jwst/quintuplet/pngs_323', 'f323', ROT),
    ]),
    'cloudef': ('/orange/adamginsburg/jwst/cloudef/data_reprojected', [
        ('/orange/adamginsburg/jwst/cloudef/pngs_210mo', 'f210mo', ROT),
        ('/orange/adamginsburg/jwst/cloudef/pngs_2100wo', 'f2100wo', ROT),
        ('/orange/adamginsburg/jwst/cloudef/pngs_360', 'f360', ROT),
        ('/orange/adamginsburg/jwst/cloudef/pngs_480', 'f480', ROT),
        ('/orange/adamginsburg/jwst/cloudef/pngs_770wo', 'f770wo', ROT),
        ('/orange/adamginsburg/jwst/cloudef/pngs_480mo', 'f480mo', ROT),
        ('/orange/adamginsburg/jwst/cloudef/pngs_360mo', 'f360mo', ROT),
        ('/orange/adamginsburg/jwst/cloudef/pngs_162mo', 'f162mo', ROT),
        ('/orange/adamginsburg/jwst/cloudef/pngs_210mo2_sc', 'f210mo2_sc', ROT),
        ('/orange/adamginsburg/jwst/cloudef/pngs_162mo5_sc', 'f162mo5_sc', ROT),
    ]),
    'sickle': ('/orange/adamginsburg/jwst/sickle/data_reprojected', [
        ('/orange/adamginsburg/jwst/sickle/pngs_470', 'f470', ROT),
        ('/orange/adamginsburg/jwst/sickle/pngs_1130', 'f1130', None),
        ('/orange/adamginsburg/jwst/sickle/pngs_770', 'f770', None),
        ('/orange/adamginsburg/jwst/sickle/pngs_1500', 'f1500', None),
    ]),
    # gc2211: wrapper -> transpose=None (all NIRCam). Per-OBS grids; each png
    # dir sits on that OBS's f277w merged i2d WCS (explicit path, not a tag).
    'gc2211': ('/orange/adamginsburg/jwst/gc2211/images-merged', [
        (f'/orange/adamginsburg/jwst/gc2211/pngs/{obs}',
         f'/orange/adamginsburg/jwst/gc2211/images-merged/'
         f'jw02211-{obs}_t001_nircam_clear-f277w-merged_i2d.fits', None)
        for obs in ('o023', 'o028', 'o046', 'o049', 'o050')
    ]),
    # Below entries are for the raw-AVM RESTORE (transpose is irrelevant there).
    'w51': ('/orange/adamginsburg/jwst/w51/data_reprojected', [
        ('/orange/adamginsburg/jwst/w51/pngs_140', 'f140', None),
        ('/orange/adamginsburg/jwst/w51/pngs_480', 'f480', None),
        ('/orange/adamginsburg/jwst/w51/pngs_2100', 'f2100', None),
    ]),
    'wd2': ('/orange/adamginsburg/jwst/wd2/data_reprojected', [
        ('/orange/adamginsburg/jwst/wd2/pngs',
         '/orange/adamginsburg/jwst/wd2/data_reprojected/'
         'jw02640-o001_t003_nircam_clear-f140m_i2d_reprj_f140m.fits', None),
        ('/orange/adamginsburg/jwst/wd2/pngs_2640',
         '/orange/adamginsburg/jwst/wd2/data_reprojected_2640/'
         'jw02640-o001_t003_nircam_clear-f115w_i2d_reprj_f140m.fits', None),
    ]),
    'brick': ('/orange/adamginsburg/jwst/brick/data_reprojected', [
        ('/orange/adamginsburg/jwst/brick/pngs_187', 'f187', None),
        ('/orange/adamginsburg/jwst/brick/pngs_200', 'f200', None),
        ('/orange/adamginsburg/jwst/brick/pngs_444', 'f444', None),
        ('/orange/adamginsburg/jwst/brick/pngs_466', 'f466', None),
    ]),
    'cloudc': ('/orange/adamginsburg/jwst/cloudc/data_reprojected', [
        ('/orange/adamginsburg/jwst/cloudc/pngs_182m', 'f182m', None),
        ('/orange/adamginsburg/jwst/cloudc/pngs_405n', 'f405n', None),
    ]),
}


def find_grid_fits(reproj_dir, grid_tag):
    """A single-filter reprojected FITS on the grid (for its true WCS).

    Patterns are ordered most-specific first so an exact grid tag (f162mo)
    does not accidentally match a longer one (f162mo5_sc).
    """
    def ok(f):
        low = os.path.basename(f).lower()
        return not any(s in low for s in ('minus', 'ratio', 'over', '_sub',
                                          'nanfilled'))

    pats = [
        f'*reprj_{grid_tag}.fits',        # exact grid, image
        f'*reprj_{grid_tag}_sci.fits',    # exact grid, _sci variant
        f'*reprj_{grid_tag}_for_images.fits',
        f'*reprj{grid_tag}.fits',
        f'*reprj_{grid_tag}[._]*.fits',   # exact grid + suffix, boundary-guarded
    ]
    for pat in pats:
        cands = [f for f in sorted(glob.glob(os.path.join(reproj_dir, pat)))
                 if ok(f)]
        if cands:
            return cands[0]
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('target', choices=sorted(TARGETS))
    p.add_argument('--no-hips', action='store_true')
    p.add_argument('--glob', default='*.png')
    args = p.parse_args()

    from reembed_avm_hips import reembed_one, load_tgt_wcs
    reproj_dir, dirs = TARGETS[args.target]
    for png_dir, grid_tag, transpose in dirs:
        if not os.path.isdir(png_dir):
            print(f"SKIP {png_dir}: missing")
            continue
        # grid_tag may be an explicit FITS path (e.g. gc2211 per-OBS grids)
        # or a tag globbed inside reproj_dir.
        if os.path.sep in grid_tag and os.path.exists(grid_tag):
            ref = grid_tag
        else:
            ref = find_grid_fits(reproj_dir, grid_tag)
        if ref is None:
            print(f"SKIP {png_dir}: no grid FITS for {grid_tag} in {reproj_dir}")
            continue
        tgt_wcs = load_tgt_wcs(ref)
        pngs = [x for x in sorted(glob.glob(os.path.join(png_dir, args.glob)))]
        tname = 'ROTATE_180' if transpose is ROT else 'None'
        print(f"[{args.target}] {png_dir}: {len(pngs)} pngs, transpose={tname}, "
              f"ref={os.path.basename(ref)}")
        n = 0
        for png in pngs:
            try:
                reembed_one(png, tgt_wcs, transpose, make_hips=not args.no_hips)
                n += 1
            except (ValueError, OSError) as e:
                print(f"  SKIP {os.path.basename(png)}: {e!r}")
        print(f"  corrected {n}/{len(pngs)}")


if __name__ == '__main__':
    main()
