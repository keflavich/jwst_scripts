#!/usr/bin/env python
"""
Build CDMatrix candidate test HiPS (identity + rot180) for one representative
PNG in each brick dir, so the correct pixel<->AVM pairing can be confirmed
by-eye vs VVV before the full brick fleet is committed.

brick PA = 88.58 deg (near-90 -> raw AVM degenerate), so every dir needs a
CDMatrix AVM; only the flip is in question.  Derivation says pngs_187 (restored
to original pixels) -> identity and pngs_444/466 (still rot180 pixels) -> rot180,
but brick was never eyeball-confirmed, so emit both candidates for each.
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(__file__))
from apply_cdmatrix_flip import load_wcs_shape, cdmatrix_avm  # noqa: E402

from PIL import Image
from tqdm import tqdm
from reproject import reproject_interp
from reproject.hips import reproject_to_hips

Image.MAX_IMAGE_PIXELS = None

WEB = "/orange/adamginsburg/web/public/avm_images"

# (label, source png, grid ref FITS)
CASES = [
    ("BRICK_187",
     "/orange/adamginsburg/jwst/brick/pngs_187/Brick_RGB_1130-466-410.png",
     "/orange/adamginsburg/jwst/brick/data_reprojected/"
     "jw01182-o004_t001_nircam_clear-f200w-merged_i2d_pipeline_v0.1_reprj_f187.fits"),
    ("BRICK_444",
     "/orange/adamginsburg/jwst/brick/pngs_444/Brick_RGB_444-356-200.png",
     "/orange/adamginsburg/jwst/brick/data_reprojected/"
     "jw01182-o004_t001_nircam_clear-f200w-merged_i2d_pipeline_v0.1_reprj_f444.fits"),
]


def build(label, src_png, ref, flip):
    fwcs, ny, nx = load_wcs_shape(ref)
    im = Image.open(src_png)
    if (im.size[1], im.size[0]) != (ny, nx):
        raise ValueError(f"{src_png} {im.size} != grid {(nx, ny)}")
    work = os.path.join(WEB, f"{label}_{flip}_test.png")
    shutil.copy2(src_png, work)
    avm = cdmatrix_avm(fwcs, ny, nx, flip)
    tmp = work + ".tmp.png"
    avm.embed(work, tmp)
    shutil.move(tmp, work)
    hips = os.path.join(WEB, f"{label}_{flip}_test_hips")
    if os.path.exists(hips):
        shutil.rmtree(hips)
    reproject_to_hips(work, coord_system_out='galactic', level=None,
                      reproject_function=reproject_interp,
                      output_directory=hips, threads=8, progress_bar=tqdm)
    print(f"  built {os.path.basename(hips)}")


def main():
    for label, src, ref in CASES:
        if not os.path.exists(src):
            print(f"SKIP {label}: missing {src}")
            continue
        for flip in ('identity', 'rot180'):
            print(f"[{label}] flip={flip}")
            build(label, src, ref, flip)


if __name__ == "__main__":
    main()
