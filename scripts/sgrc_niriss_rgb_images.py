#!/usr/bin/env python
# coding: utf-8
"""
RGB images for the NIRISS Sgr C parallel field (project 4147).

Four NIRISS filters (F158M, F200W, F356W, F480M) are drizzled to a single
common output frame (2782x2233, PA=92.09deg), so no reprojection is needed --
they share one WCS.  PA~92deg is in the pyavm Scale+Rotation degeneracy zone,
so the embedded AVM is built with faithful_avm (flat CDMatrix), not
pyavm.AVM.from_header.

Primary combo for the CMZ mosaic: SGRC_NIRISS_RGB_480-356-200.
"""
import os

import numpy as np
from astropy.io import fits
from astropy.visualization import simple_norm

from jwst_rgb.save_rgb import save_rgb
from jwst_rgb.save_rgb import faithful_avm

NIRISS_BASE = "/orange/adamginsburg/jwst/sgrc/niriss"

image_filenames = {
    f: f"{NIRISS_BASE}/{f.upper()}/pipeline/"
       f"jw04147-o012_t001_niriss_clear-{f}-nis_data_i2d.fits"
    for f in ("f158m", "f200w", "f356w", "f480m")
}


def make_pngs(target_filter="f480m",
              png_path=f"{NIRISS_BASE}/pngs_480"):
    print(f"Making NIRISS Sgr C PNGs (target grid {target_filter})")
    os.makedirs(png_path, exist_ok=True)

    tgt_header = fits.getheader(image_filenames[target_filter], ext=("SCI", 1))
    AVM = faithful_avm(tgt_header)

    # all filters share the target WCS -> read SCI directly, no reprojection
    data = {f: fits.getdata(p, ext=("SCI", 1)) for f, p in image_filenames.items()}
    tgt_shape = data[target_filter].shape
    for f, d in data.items():
        if d.shape != tgt_shape:
            raise ValueError(f"{f} shape {d.shape} != target {tgt_shape}; "
                             "NIRISS frames expected to share the grid")

    # wavelength-descending: R=longest, B=shortest
    filternames = sorted(image_filenames,
                         key=lambda x: int(''.join(filter(str.isdigit, x))))[::-1]
    print(f"Filters (R->B): {filternames}")

    def _digits(x):
        return ''.join(filter(str.isdigit, x))

    # 3-color combos over consecutive filters
    for i in range(len(filternames) - 2):
        f1, f2, f3 = filternames[i], filternames[i + 1], filternames[i + 2]
        rgb = np.array([data[f1], data[f2], data[f3]]).swapaxes(0, 2).swapaxes(0, 1)
        tag = f"{_digits(f1)}-{_digits(f2)}-{_digits(f3)}"
        print(f"  RGB {tag}")
        for stretch, lo, suffix in (('asinh', 1, ''), ('log', 1.5, '_log')):
            rgb_scaled = np.array([
                simple_norm(rgb[:, :, k], stretch=stretch, min_percent=lo,
                            max_percent=99.5)(rgb[:, :, k]) for k in range(3)
            ]).swapaxes(0, 2).swapaxes(0, 1)
            save_rgb(rgb_scaled, f"{png_path}/SGRC_NIRISS_RGB_{tag}{suffix}.png",
                     avm=AVM, original_data=rgb)

    # single-filter images
    for f in filternames:
        d = data[f]
        stack = np.stack([d, d, d], axis=2)
        for stretch, lo, suffix in (('asinh', 1, '_asinh'), ('log', 1.5, '_log')):
            img = simple_norm(d, stretch=stretch, min_percent=lo,
                              max_percent=99.5)(d)
            img = np.stack([img, img, img], axis=2)
            save_rgb(img, f"{png_path}/SGRC_NIRISS_{_digits(f)}{suffix}.png",
                     avm=AVM, original_data=stack)


def main():
    make_pngs()


if __name__ == "__main__":
    main()
