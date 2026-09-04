#!/usr/bin/env python
"""
Build a HiPS from a PNG and publish it into the avm_images web root.

Two things this does deliberately:

1. NO transparency pass.  `convert_black_to_transparent` flood-fills
   edge-connected black, which is safe only when the sky is bright relative to
   blank.  For dark-sky renders (background-subtracted images, or crowded
   stellar fields stretched from 1%) it eats real sky -- it took one brick image
   from 14.5% to 36% transparent inside the field.  save_rgb already writes
   correct edge alpha via alpha_only_edges=True, so the PNG is used verbatim.

2. STAGED build.  The publish target is the live web root, so a pyramid built
   in place would serve a half-written layer for several minutes and would
   leave the layer destroyed if the job died partway.  The pyramid is built to
   <name>_hips.new, the old directory is moved aside, the new one is renamed in,
   and only then is the old one deleted -- an outage of two renames.
"""
import argparse
import os
import shutil

from PIL import Image
from tqdm import tqdm
from reproject import reproject_interp
from reproject.hips import reproject_to_hips

Image.MAX_IMAGE_PIXELS = None
WEB = "/orange/adamginsburg/web/public/avm_images"


def publish(png, web=WEB, name=None, threads=8):
    name = name or os.path.basename(png)[:-4]
    dest = os.path.join(web, f"{name}_hips")
    stage = dest + ".new"
    if os.path.isdir(stage):
        shutil.rmtree(stage)
    print(f"building {name} -> {stage}", flush=True)
    reproject_to_hips(png, coord_system_out="galactic", level=None,
                      reproject_function=reproject_interp,
                      output_directory=stage, threads=threads,
                      progress_bar=tqdm)
    if not os.path.isdir(os.path.join(stage, "Norder3")):
        raise RuntimeError(f"{name}: build produced no Norder3, refusing to publish")
    old = dest + ".old"
    if os.path.isdir(old):
        shutil.rmtree(old)
    if os.path.isdir(dest):
        os.rename(dest, old)
    os.rename(stage, dest)
    if os.path.isdir(old):
        shutil.rmtree(old)
    print(f"PUBLISHED {dest}", flush=True)
    return dest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pngs", nargs="+")
    p.add_argument("--web", default=WEB)
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    for png in a.pngs:
        if not os.path.exists(png):
            print(f"SKIP missing {png}", flush=True)
            continue
        publish(png, web=a.web, threads=a.threads)
    print("ALL_PUBLISHED", flush=True)


if __name__ == "__main__":
    main()
