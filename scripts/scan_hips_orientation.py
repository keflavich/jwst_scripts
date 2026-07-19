#!/usr/bin/env python
"""
Batch orientation scan across all targets.

For each png dir, sample several PNGs and correlate each one's AVM-reprojected
luminance against a single-filter reference FITS for that target (see
check_hips_orientation).  Per PNG the verdict is:

  identity     -> HiPS correctly oriented
  flip_*/rot180-> HiPS flipped (that transform)
  no-overlap   -> reference FITS does not overlap this PNG (all-nan / tiny
                  coverage); verdict undetermined, NOT a pass

Per dir we report the majority verdict over the sampled PNGs.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from check_hips_orientation import check  # noqa: E402

SAMPLES_PER_DIR = 5

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
    'w51': (
        '/orange/adamginsburg/jwst/w51/data_reprojected/jw06151-o001_t001_nircam_clear-f210m-merged_i2d_reprj_f140.fits',
        ['/orange/adamginsburg/jwst/w51/pngs_140',
         '/orange/adamginsburg/jwst/w51/pngs_480',
         '/orange/adamginsburg/jwst/w51/pngs_2100'],
    ),
}


def pick_pngs(png_dir, n):
    """Up to n PNGs (spread across the dir) that have a sibling _hips dir."""
    cands = [p for p in sorted(glob.glob(os.path.join(png_dir, '*.png')))
             if not p.endswith('_alma.png')
             and os.path.isdir(p.replace('.png', '_hips'))]
    if not cands:
        cands = [p for p in sorted(glob.glob(os.path.join(png_dir, '*.png')))
                 if os.path.isdir(p.replace('.png', '_hips'))]
    if len(cands) <= n:
        return cands
    idx = np.linspace(0, len(cands) - 1, n).round().astype(int)
    return [cands[i] for i in sorted(set(idx))]


def classify(rep):
    """identity / flip_* / rot180 / no-overlap from a single check result."""
    corrs = rep['correlations']
    if all(np.isnan(v) for v in corrs.values()) or rep['coverage'] < 0.02:
        return 'no-overlap'
    return rep['best']


def main():
    results = []
    for target, (ref_fits, png_dirs) in TARGETS.items():
        if not os.path.exists(ref_fits):
            print(f"[ERR ] {target:11s} ref fits missing: {ref_fits}")
            results.append({'target': target, 'error': f'ref fits missing: {ref_fits}'})
            continue
        for png_dir in png_dirs:
            pngs = pick_pngs(png_dir, SAMPLES_PER_DIR)
            if not pngs:
                print(f"[----] {target:11s} {os.path.basename(png_dir):16s} no PNG with _hips")
                results.append({'target': target, 'png_dir': png_dir,
                                'error': 'no PNG with _hips sibling'})
                continue
            verdicts = {}
            per_png = []
            for png in pngs:
                try:
                    rep = check(png, ref_fits)
                    v = classify(rep)
                except (ValueError, OSError) as e:
                    v = 'error'
                    rep = {'error': repr(e)}
                verdicts[v] = verdicts.get(v, 0) + 1
                per_png.append({'png': os.path.basename(png), 'verdict': v,
                                'correlations': rep.get('correlations'),
                                'coverage': rep.get('coverage')})
            # majority verdict, ignoring no-overlap/error unless they dominate
            judged = {k: c for k, c in verdicts.items()
                      if k not in ('no-overlap', 'error')}
            if judged:
                majority = max(judged, key=judged.get)
            else:
                majority = max(verdicts, key=verdicts.get)
            ok = majority == 'identity'
            flag = 'OK  ' if ok else ('----' if majority in ('no-overlap', 'error')
                                      else 'FLIP')
            counts = ' '.join(f"{k}={c}" for k, c in sorted(verdicts.items()))
            print(f"[{flag}] {target:11s} {os.path.basename(png_dir):16s} "
                  f"majority={majority:10s} n={len(pngs)}  ({counts})")
            results.append({'target': target, 'png_dir': png_dir,
                            'majority': majority, 'ok': ok,
                            'verdict_counts': verdicts, 'per_png': per_png})

    out = os.path.join(os.path.dirname(__file__), 'hips_orientation_scan.json')
    with open(out, 'w') as fh:
        json.dump(results, fh, indent=2)
    n_flip = sum(1 for r in results if r.get('majority')
                 not in (None, 'identity', 'no-overlap', 'error'))
    n_ok = sum(1 for r in results if r.get('ok'))
    print(f"\n{n_ok} dirs OK, {n_flip} dirs FLIPPED. wrote {out}")


if __name__ == '__main__':
    main()
