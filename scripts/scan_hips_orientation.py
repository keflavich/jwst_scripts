#!/usr/bin/env python
"""
Batch orientation scan across all targets.

For each png dir that contains at least one *_hips sibling, pick one
representative PNG and correlate its AVM-reprojected luminance against a
single-filter reference FITS for that target (see check_hips_orientation).
best != 'identity' means the HiPS built from that PNG is flipped.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from check_hips_orientation import check  # noqa: E402

# target -> (glob for reference single-filter FITS, list of png dirs)
TARGETS = {
    'w51': (
        '/orange/adamginsburg/jwst/w51/data_reprojected/jw06151-o001_t001_nircam_clear-f210m-merged_i2d_reprj_f140.fits',
        ['/orange/adamginsburg/jwst/w51/pngs_140', '/orange/adamginsburg/jwst/w51/pngs_2100'],
    ),
    'arches': (
        '/orange/adamginsburg/jwst/arches/data_reprojected/jw02045-o001_t001_nircam_clear-f212n_i2d_pipeline_v0.1_reprj_f212.fits',
        ['/orange/adamginsburg/jwst/arches/pngs_212', '/orange/adamginsburg/jwst/arches/pngs_323'],
    ),
    'quintuplet': (
        '/orange/adamginsburg/jwst/quintuplet/data_reprojected/jw02045-o003_t002_nircam_clear-f212n_i2d_pipeline_v0.1_reprj_f323.fits',
        ['/orange/adamginsburg/jwst/quintuplet/pngs_212', '/orange/adamginsburg/jwst/quintuplet/pngs_323'],
    ),
    'wd2': (
        '/orange/adamginsburg/jwst/wd2/data_reprojected_2640/jw02640-o001_t003_nircam_clear-f187n_i2d_reprj_f140m.fits',
        ['/orange/adamginsburg/jwst/wd2/pngs', '/orange/adamginsburg/jwst/wd2/pngs_2640'],
    ),
    'sgra': (
        '/orange/adamginsburg/jwst/sgra/data_reprojected/jw01939-o001_t001_nircam_clear-f212n_i2d_pipeline_v0.1_reprj_f770.fits',
        ['/orange/adamginsburg/jwst/sgra/pngs_770', '/orange/adamginsburg/jwst/sgra/pngs_1280'],
    ),
    'sgrb2': (
        '/orange/adamginsburg/jwst/sgrb2/data_reprojected/jw05365-o002_t002_miri_f2550w_i2d_pipeline_v0.1_reprj_f150.fits',
        ['/orange/adamginsburg/jwst/sgrb2/pngs_150', '/orange/adamginsburg/jwst/sgrb2/pngs_466'],
    ),
    'sgrc': (
        '/orange/adamginsburg/jwst/sgrc/data_reprojected/jw04147-o012_t001_nircam_clear-f212n_i2d_pipeline_v0.1_reprj_f360.fits',
        ['/orange/adamginsburg/jwst/sgrc/pngs_360', '/orange/adamginsburg/jwst/sgrc/pngs_480'],
    ),
    'sickle': (
        '/orange/adamginsburg/jwst/sickle/data_reprojected/f1130_reprj_f470.fits',
        ['/orange/adamginsburg/jwst/sickle/pngs_470', '/orange/adamginsburg/jwst/sickle/pngs_1130'],
    ),
    'cloudef': (
        '/orange/adamginsburg/jwst/cloudef/data_reprojected/jw02092-o006_t002_miri_f2100w_i2d_pipeline_v0.1_reprj_f210mo.fits',
        ['/orange/adamginsburg/jwst/cloudef/pngs_210mo', '/orange/adamginsburg/jwst/cloudef/pngs_2100wo'],
    ),
}


def pick_png(png_dir):
    """First PNG in dir that has a sibling _hips directory."""
    for png in sorted(glob.glob(os.path.join(png_dir, '*.png'))):
        if png.endswith('_alma.png'):
            continue
        hips = png.replace('.png', '_hips')
        if os.path.isdir(hips):
            return png
    # fall back: any png with hips even alma
    for png in sorted(glob.glob(os.path.join(png_dir, '*.png'))):
        if os.path.isdir(png.replace('.png', '_hips')):
            return png
    return None


def main():
    results = []
    for target, (ref_fits, png_dirs) in TARGETS.items():
        if not os.path.exists(ref_fits):
            results.append({'target': target, 'error': f'ref fits missing: {ref_fits}'})
            continue
        for png_dir in png_dirs:
            png = pick_png(png_dir)
            if png is None:
                results.append({'target': target, 'png_dir': png_dir,
                                'error': 'no PNG with _hips sibling'})
                continue
            try:
                rep = check(png, ref_fits)
                rep['target'] = target
                rep['png_dir'] = png_dir
                results.append(rep)
                flag = 'OK  ' if rep['ok'] else 'FLIP'
                c = rep['correlations']
                print(f"[{flag}] {target:11s} {os.path.basename(png_dir):16s} "
                      f"best={rep['best']:9s} "
                      f"id={c['identity']:+.2f} lr={c['flip_lr']:+.2f} "
                      f"ud={c['flip_ud']:+.2f} r180={c['rot180']:+.2f}  "
                      f"{os.path.basename(png)}")
            except Exception as e:  # noqa: BLE001 - scan tool, want to continue
                results.append({'target': target, 'png_dir': png_dir,
                                'png': png, 'error': repr(e)})
                print(f"[ERR ] {target:11s} {os.path.basename(png_dir):16s} {e!r}")

    out = os.path.join(os.path.dirname(__file__), 'hips_orientation_scan.json')
    with open(out, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
