from astropy.io import fits
import numpy as np
from astropy.visualization import simple_norm
import pylab as plt
from astropy import wcs
import os
import PIL
import shutil
from astropy.wcs import WCS
import pyavm
from PIL import Image
from jwst_rgb.save_rgb import save_rgb as _save_rgb, fill_nan
import subprocess
import pickle
import tempfile
import sys
import hashlib


CURRENT_TARGET_FILTER_IS_MIRI = False


def save_rgb(*args, **kwargs):
    kwargs.setdefault('transpose', Image.ROTATE_180 if CURRENT_TARGET_FILTER_IS_MIRI else None)
    kwargs.setdefault('alpha_only_edges', True)
    return _save_rgb(*args, **kwargs)



image_filenames ={
    "f140m": "/orange/adamginsburg/jwst/w51/F140M/pipeline/jw06151-o001_t001_nircam_clear-f140m-merged_i2d.fits",
    "f182m": "/orange/adamginsburg/jwst/w51/F182M/pipeline/jw06151-o001_t001_nircam_clear-f182m-merged_i2d.fits",
    "f187n": "/orange/adamginsburg/jwst/w51/F187N/pipeline/jw06151-o001_t001_nircam_clear-f187n-merged_i2d.fits",
    "f210m": "/orange/adamginsburg/jwst/w51/F210M/pipeline/jw06151-o001_t001_nircam_clear-f210m-merged_i2d.fits",
    "f162m": "/orange/adamginsburg/jwst/w51//mastDownload/JWST/jw06151-o001_t003_nircam_f150w2-f162m/jw06151-o001_t003_nircam_f150w2-f162m_i2d.fits",
    "f335m": "/orange/adamginsburg/jwst/w51/F335M/pipeline/jw06151-o001_t001_nircam_clear-f335m-merged_i2d.fits",
    "f360m": "/orange/adamginsburg/jwst/w51/F360M/pipeline/jw06151-o001_t001_nircam_clear-f360m-merged_i2d.fits",
    "f405n": "/orange/adamginsburg/jwst/w51/F405N/pipeline/jw06151-o001_t001_nircam_clear-f405n-merged_i2d.fits",
    "f410m": "/orange/adamginsburg/jwst/w51/F410M/pipeline/jw06151-o001_t001_nircam_clear-f410m-merged_i2d.fits",
    "f480m": "/orange/adamginsburg/jwst/w51/F480M/pipeline/jw06151-o001_t001_nircam_clear-f480m-merged_i2d.fits",
    # 2026-03-18 note: MIRI images in subdirectory have saturation recovery disabled.
    "f1000w": "/orange/adamginsburg/jwst/w51/F1000W//jw06151-o002_t001_miri_f1000w_i2d.fits",
    "f1280w": "/orange/adamginsburg/jwst/w51/F1280W//jw06151-o002_t001_miri_f1280w_i2d.fits",
    "f2100w": "/orange/adamginsburg/jwst/w51/F2100W//jw06151-o002_t001_miri_f2100w_i2d.fits",
    "f560w": "/orange/adamginsburg/jwst/w51/F560W//jw06151-o002_t001_miri_f560w_i2d.fits",
    "f770w": "/orange/adamginsburg/jwst/w51/F770W//jw06151-o002_t001_miri_f770w_i2d.fits",
}
image_sub_filenames = {
    "f182m-f187n": "/orange/adamginsburg/jwst/w51/filter_subtractions/f182m_minus_f187n.fits",
    "f187n-f182m": "/orange/adamginsburg/jwst/w51/filter_subtractions/f187n_minus_f182m.fits",
    "f210m-f182m": "/orange/adamginsburg/jwst/w51/filter_subtractions/f210m_minus_f182m_unscaled.fits",
    "f405n-f410m": "/orange/adamginsburg/jwst/w51/filter_subtractions/f405n_minus_f410m.fits",
    "f410m-f405n": "/orange/adamginsburg/jwst/w51/filter_subtractions/f410m_minus_f405n.fits",
    "f360m-f182m": "/orange/adamginsburg/jwst/w51/filter_subtractions/f360m_minus_f182m_unscaled.fits",
    "f360m-f405n": "/orange/adamginsburg/jwst/w51/filter_subtractions/f360m_minus_f405n.fits",
    "f405n-f360m": "/orange/adamginsburg/jwst/w51/filter_subtractions/f405n_minus_f360m.fits",
    "f480m-f360m": "/orange/adamginsburg/jwst/w51/filter_subtractions/f480m_minus_f360m_unscaled.fits",
    "f335m-f2100w": "/orange/adamginsburg/jwst/w51/filter_subtractions/f335_f2100_ratio.fits",
    "f770w-f2100w": "/orange/adamginsburg/jwst/w51/filter_subtractions/f770_f2100_ratio.fits",
    "f187n-f405n": "/orange/adamginsburg/jwst/w51/filter_subtractions/paa_bra_ratio_non_neg.fits",
    "f360m-f335m": "/orange/adamginsburg/jwst/w51/filter_subtractions/f360m_minus_f335m_scaled_BB.fits",
    "f335m-f480m": "/orange/adamginsburg/jwst/w51/filter_subtractions/f335_f480_ratio.fits",
    "f405n-f480m": "/orange/adamginsburg/jwst/w51/filter_subtractions/f405_f480_ratio.fits",
    "f335m-f405n": "/orange/adamginsburg/jwst/w51/filter_subtractions/f335_f405_ratio.fits",
    "f560w-f480m": "/orange/adamginsburg/jwst/w51/filter_subtractions/f560_f480_ratio.fits",
    "f140m_kuband": "/orange/adamginsburg/jwst/w51/filter_subtractions/f140m_kuband_ratio.fits",
    "f1280w-f2100w": "/orange/adamginsburg/jwst/w51/filter_subtractions/f1280_f2100_ratio.fits",
    "f335m-f2100w": "/orange/adamginsburg/jwst/w51/filter_subtractions/f335_f2100_ratio.fits",
    "f770w-f2100w": "/orange/adamginsburg/jwst/w51/filter_subtractions/f770_f2100_ratio.fits",
    "f335m-f560w": "/orange/adamginsburg/jwst/w51/filter_subtractions/f335_f560_ratio.fits",
    "f770w-f560w": "/orange/adamginsburg/jwst/w51/filter_subtractions/f770_f560_ratio.fits",
    "f182m-f162m": "/orange/adamginsburg/jwst/w51/filter_subtractions/f182m_minus_f162m_scaled_BB.fits",
    "f162m-f140m": "/orange/adamginsburg/jwst/w51/filter_subtractions/f162m_minus_f140m_scaled_BB.fits",
    "f480m-f360m": "/orange/adamginsburg/jwst/w51/filter_subtractions/f480m_minus_f360m_scaled_BB.fits",
}

custom_negative_thresholds = {
    'f2100w': -300,
    'f1280w': -50,
    'f1000w': -10,
    'f770w': -30,
    'f560w': -10,
}

submitted_rgb_filenames = set()
MIRI_FILTERNAMES = {'f560w', 'f770w', 'f1000w', 'f1280w', 'f2100w'}


def make_pngs(target_filter='f140m', new_basepath='/orange/adamginsburg/jwst/w51/data_reprojected/'):
    """
    Generate job specifications for RGB image creation.
    All actual data loading happens in the worker jobs.
    """
    print(f"Submitting PNG creation jobs for {target_filter}")
    
    png_path = f'/orange/adamginsburg/jwst/w51/pngs_{target_filter[1:-1]}'
    os.makedirs(png_path, exist_ok=True)
    
    # Submit job for 210-162-140 combination with various stretches
    submit_rgb_job({
        'target_filter': target_filter,
        'filters': ['f210m', 'f162m', 'f140m'],
        'stretches': [
            {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99}},
            {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99}},
            {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99}},
        ],
        'filename': f'{png_path}/w51_RGB_210-162-140.png',
        'alma_overlay': True,
        'new_basepath': new_basepath,
    })
    
    submit_rgb_job({
        'target_filter': target_filter,
        'filters': ['f210m', 'f162m', 'f140m'],
        'stretches': [
            {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5}},
            {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5}},
            {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5}},
        ],
        'filename': f'{png_path}/w51_RGB_210-162-140_99.5.png',
        'alma_overlay': True,
        'new_basepath': new_basepath,
    })
    
    submit_rgb_job({
        'target_filter': target_filter,
        'filters': ['f210m', 'f162m', 'f140m'],
        'stretches': [
            {'func': simple_norm, 'kwargs': {'stretch': 'log', 'min_percent': 1, 'max_percent': 99.95}},
            {'func': simple_norm, 'kwargs': {'stretch': 'log', 'min_percent': 1, 'max_percent': 99.95}},
            {'func': simple_norm, 'kwargs': {'stretch': 'log', 'min_percent': 1, 'max_percent': 99.95}},
        ],
        'filename': f'{png_path}/w51_RGB_210-162-140_99.95.png',
        'alma_overlay': True,
        'new_basepath': new_basepath,
    })
    
    submit_rgb_job({
        'target_filter': target_filter,
        'filters': ['f210m', 'f162m', 'f140m'],
        'stretches': [
            {'func': simple_norm, 'kwargs': {'stretch': 'log', 'min_percent': 1, 'max_percent': 99.995}},
            {'func': simple_norm, 'kwargs': {'stretch': 'log', 'min_percent': 1, 'max_percent': 99.995}},
            {'func': simple_norm, 'kwargs': {'stretch': 'log', 'min_percent': 1, 'max_percent': 99.995}},
        ],
        'filename': f'{png_path}/w51_RGB_210-162-140_99.995.png',
        'alma_overlay': True,
        'new_basepath': new_basepath,
    })
    
    submit_rgb_job({
        'target_filter': target_filter,
        'filters': ['f210m', 'f162m', 'f140m'],
        'stretches': [
            {'func': simple_norm, 'kwargs': {'stretch': 'log', 'vmin': 0, 'vmax': 223}},
            {'func': simple_norm, 'kwargs': {'stretch': 'log', 'vmin': 0, 'vmax': 121}},
            {'func': simple_norm, 'kwargs': {'stretch': 'log', 'vmin': 0, 'vmax': 85}},
        ],
        'filename': f'{png_path}/w51_RGB_210-162-140_carta.png',
        'alma_overlay': True,
        'new_basepath': new_basepath,
    })
    
    # f360m, f335m, f210m
    submit_rgb_job({
        'target_filter': target_filter,
        'filters': ['f360m', 'f335m', 'f210m'],
        'stretches': [
            {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99}},
            {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99}},
            {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99}},
        ],
        'filename': f'{png_path}/w51_RGB_360-335-210.png',
        'alma_overlay': True,
        'new_basepath': new_basepath,
    })
    
    # f480m, f405n, f187n
    submit_rgb_job({
        'target_filter': target_filter,
        'filters': ['f480m', 'f405n', 'f187n'],
        'stretches': [
            {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5}},
            {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5}},
            {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5}},
        ],
        'filename': f'{png_path}/w51_RGB_480-405-187.png',
        'alma_overlay': True,
        'new_basepath': new_basepath,
    })

    submit_rgb_job({
        'target_filter': target_filter,
        'filters': ['f480m', 'f410m', 'f405n'],
        'stretches': [
            {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5}},
            {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5}},
            {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5}},
        ],
        'filename': f'{png_path}/w51_RGB_480-410-405.png',
        'alma_overlay': True,
        'new_basepath': new_basepath,
    })
    
    submit_rgb_job({
        'target_filter': target_filter,
        'filters': ['f480m', 'f405n', 'f187n'],
        'stretches': [
            {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 97}},
            {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99}},
            {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99}},
        ],
        'filename': f'{png_path}/w51_RGB_480-405-187_scaled.png',
        'alma_overlay': True,
        'new_basepath': new_basepath,
    })
    
    # f480m, f405n, f210m
    submit_rgb_job({
        'target_filter': target_filter,
        'filters': ['f480m', 'f405n', 'f210m'],
        'stretches': [
            {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 97}},
            {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99}},
            {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99}},
        ],
        'filename': f'{png_path}/w51_RGB_480-405-210_scaled.png',
        'alma_overlay': True,
        'new_basepath': new_basepath,
    })

    # f480m, f360m, f335m
    submit_rgb_job({
        'target_filter': target_filter,
        'filters': ['f480m', 'f360m', 'f335m'],
        'stretches': [
            {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5}},
            {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5}},
            {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5}},
        ],
        'filename': f'{png_path}/w51_RGB_480-360-335.png',
        'alma_overlay': True,
        'new_basepath': new_basepath,
    })

    submit_rgb_job({
        'target_filter': target_filter,
        'filters': ['f480m', 'f405n', 'f335m'],
        'stretches': [
            {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5}},
            {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5}},
            {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5}},
        ],
        'filename': f'{png_path}/w51_RGB_480-405-335.png',
        'alma_overlay': True,
        'new_basepath': new_basepath,
    })

    # f480m, f335m, f187n
    submit_rgb_job({
        'target_filter': target_filter,
        'filters': ['f480m', 'f335m', 'f187n'],
        'stretches': [
            {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.95}},
            {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.95}},
            {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.95}},
        ],
        'filename': f'{png_path}/w51_RGB_480-335-187_scaled.png',
        'alma_overlay': True,
        'new_basepath': new_basepath,
    })


    
    # f210m, f182m, f187n
    submit_rgb_job({
        'target_filter': target_filter,
        'filters': ['f210m', 'f182m', 'f187n'],
        'stretches': [
            {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5}},
            {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5}},
            {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.0}},
        ],
        'filename': f'{png_path}/w51_RGB_210-182-187.png',
        'alma_overlay': True,
        'new_basepath': new_basepath,
    })
    
    # MIRI combinations
    submit_rgb_job({
        'target_filter': target_filter,
        'filters': ['f2100w', 'f1280w', 'f770w'],
        'stretches': [
            {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5}},
            {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5}},
            {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5}},
        ],
        'filename': f'{png_path}/w51_RGB_2100-1280-770.png',
        'alma_overlay': True,
        'new_basepath': new_basepath,
    })
    
    submit_rgb_job({
        'target_filter': target_filter,
        'filters': ['f2100w', 'f770w', 'f335m'],
        'stretches': [
            {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5}},
            {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5}},
            {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5}},
        ],
        'filename': f'{png_path}/w51_RGB_2100-770-335_pah.png',
        'alma_overlay': True,
        'new_basepath': new_basepath,
    })
    
    if False: # default to skipping special combos
        # Special combinations
        submit_rgb_job({
            'target_filter': target_filter,
            'filters': ['f405n-f410m', 'f335m-f2100w', 'f770w-f2100w'],
            'stretches': [
                {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 0.1, 'max_percent': 99.9}},
                {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 0.1, 'max_percent': 99.9}},
                {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 0.1, 'max_percent': 99.9}},
            ],
            'filename': f'{png_path}/w51_RGB_405m410-335r2100-770r2100.png',
            'alma_overlay': True,
            'new_basepath': new_basepath,
        })
        
        submit_rgb_job({
            'target_filter': target_filter,
            'filters': ['f480m', 'f405n-f410m', 'f187n-f182m'],
            'stretches': [
                {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99}},
                {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99}},
                {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99}},
            ],
            'filename': f'{png_path}/w51_RGB_480-405m410-187m182_scaled.png',
            'alma_overlay': False,
            'new_basepath': new_basepath,
        })
        
        submit_rgb_job({
            'target_filter': target_filter,
            'filters': ['f480m', 'f405n-f410m', 'f187n-f182m'],
            'stretches': [
                {'func': simple_norm, 'kwargs': {'stretch': 'log', 'min_percent': 1, 'max_percent': 99.5}},
                {'func': simple_norm, 'kwargs': {'stretch': 'log', 'min_percent': 1, 'max_percent': 99.5}},
                {'func': simple_norm, 'kwargs': {'stretch': 'log', 'min_percent': 1, 'max_percent': 99.5}},
            ],
            'filename': f'{png_path}/w51_RGB_480-405m410-187m182_log.png',
            'alma_overlay': False,
            'new_basepath': new_basepath,
        })
        
        submit_rgb_job({
            'target_filter': target_filter,
            'filters': ['f335m-f2100w', 'f480m-f360m', 'f405n-f410m'],
            'stretches': [
                {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99}},
                {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99}},
                {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99}},
            ],
            'filename': f'{png_path}/w51_RGB_335r2100-480m360-405m410_scaled.png',
            'alma_overlay': False,
            'new_basepath': new_basepath,
        })
        
        submit_rgb_job({
            'target_filter': target_filter,
            'filters': ['f335m-f2100w', 'f480m-f360m', 'f405n-f410m'],
            'stretches': [
                {'func': simple_norm, 'kwargs': {'stretch': 'log', 'min_percent': 1, 'max_percent': 99.5}},
                {'func': simple_norm, 'kwargs': {'stretch': 'log', 'min_percent': 1, 'max_percent': 99.5}},
                {'func': simple_norm, 'kwargs': {'stretch': 'log', 'min_percent': 1, 'max_percent': 99.5}},
            ],
            'filename': f'{png_path}/w51_RGB_335r2100-480m360-405m410_log.png',
            'alma_overlay': False,
            'new_basepath': new_basepath,
        })
        
        submit_rgb_job({
            'target_filter': target_filter,
            'filters': ['f480m-f360m', 'f410m-f405n', 'f405n-f410m'],
            'stretches': [
                {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99}},
                {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99}},
                {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99}},
            ],
            'filename': f'{png_path}/w51_RGB_480m360-410m405-405m410_scaled.png',
            'alma_overlay': False,
            'new_basepath': new_basepath,
        })
        
        submit_rgb_job({
            'target_filter': target_filter,
            'filters': ['f480m-f360m', 'f410m-f405n', 'f405n-f410m'],
            'stretches': [
                {'func': simple_norm, 'kwargs': {'stretch': 'log', 'min_percent': 1, 'max_percent': 99.5}},
                {'func': simple_norm, 'kwargs': {'stretch': 'log', 'min_percent': 1, 'max_percent': 99.5}},
                {'func': simple_norm, 'kwargs': {'stretch': 'log', 'min_percent': 1, 'max_percent': 99.5}},
            ],
            'filename': f'{png_path}/w51_RGB_480m360-410m405-405m410_log.png',
            'alma_overlay': False,
            'new_basepath': new_basepath,
        })

    filternames = sorted(list(image_filenames.keys()),
                         key=lambda x: int(''.join(filter(str.isdigit, x))))[::-1]
    print(f"Sorted list of filters: {filternames}")

    for f1, f2, f3 in zip(filternames, filternames[1:], filternames[2:]):
        f1n = ''.join(filter(str.isdigit, f1))
        f2n = ''.join(filter(str.isdigit, f2))
        f3n = ''.join(filter(str.isdigit, f3))

        submit_rgb_job({
            'target_filter': target_filter,
            'filters': [f1, f2, f3],
            'stretches': [
                {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5}},
                {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5}},
                {'func': simple_norm, 'kwargs': {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5}},
            ],
            'filename': f'{png_path}/w51_RGB_{f1n}-{f2n}-{f3n}.png',
            'alma_overlay': True,
            'new_basepath': new_basepath,
        })

        submit_rgb_job({
            'target_filter': target_filter,
            'filters': [f1, f2, f3],
            'stretches': [
                {'func': simple_norm, 'kwargs': {'stretch': 'log', 'min_percent': 1.5, 'max_percent': 99.5}},
                {'func': simple_norm, 'kwargs': {'stretch': 'log', 'min_percent': 1.5, 'max_percent': 99.5}},
                {'func': simple_norm, 'kwargs': {'stretch': 'log', 'min_percent': 1.5, 'max_percent': 99.5}},
            ],
            'filename': f'{png_path}/w51_RGB_{f1n}-{f2n}-{f3n}_log.png',
            'alma_overlay': True,
            'new_basepath': new_basepath,
        })


def submit_rgb_job(job_spec):
    """
    Submit a SLURM job to create and save an RGB image.
    
    Parameters
    ----------
    job_spec : dict
        Dictionary containing all parameters needed to create the RGB image:
        - filters: list of 3 filter names to combine
        - stretches: list of stretch parameters for each filter
        - filename: output PNG filename
        - alma_overlay: bool, whether to add ALMA overlay
        - target_filter: target filter for reprojection
    """
    filename = job_spec['filename']
    if filename in submitted_rgb_filenames:
        print(f"Skipping duplicate RGB job for {filename}")
        return None
    submitted_rgb_filenames.add(filename)

    # Create a persistent shared directory for job specs
    job_spec_dir = '/orange/adamginsburg/jwst/w51/job_specs/'
    os.makedirs(job_spec_dir, exist_ok=True)
    
    # Create a deterministic filename based on job spec content
    spec_hash = hashlib.md5(str(sorted(job_spec.items())).encode()).hexdigest()[:8]
    job_name_base = os.path.basename(job_spec['filename']).replace('.png', '')[:15]
    job_spec_file = os.path.join(job_spec_dir, f'{job_name_base}_{spec_hash}.pkl')
    
    # Write job spec to shared directory
    with open(job_spec_file, 'wb') as f:
        pickle.dump(job_spec, f)
    
    # Create Python command to execute
    script_path = os.path.abspath(__file__)
    python_cmd = f"python {script_path} --worker-job {job_spec_file}"
    
    mem = {'f140m': 64,
           'f480m': 32,
           'f2100w': 16}[job_spec['target_filter']]
    
    # Submit SLURM job
    job_name = os.path.basename(job_spec['filename']).replace('.png', '')[:30] + "_" + job_spec['target_filter']
    log_file = f'/blue/adamginsburg/adamginsburg/logs/w51_quickimages-{job_name}_%j.log'
    sbatch_cmd = [
        'sbatch',
        f'--job-name={job_name}',
        '--account=astronomy-dept',
        '--qos=astronomy-dept',
        '--nodes=1',
        '--ntasks=1',
        f'--mem={mem}gb',
        '--time=96:00:00',
        f'--output={log_file}',
        '--wrap', python_cmd
    ]
    
    result = subprocess.run(sbatch_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error submitting job for {job_spec['filename']}:")
        print(result.stderr)
        raise ValueError(f"Failed to submit job for {job_spec['filename']}")
    else:
        print(f"Submitted {job_name} job for {job_spec['filename']}: {result.stdout.strip()}")
    
    return job_spec_file


def worker_create_rgb(job_spec_file):
    """
    Worker function to create and save an RGB image.
    This runs in a SLURM job and does all data loading and processing.
    
    Parameters
    ----------
    job_spec_file : str
        Path to pickle file containing the job specification
    """
    with open(job_spec_file, 'rb') as f:
        job_spec = pickle.load(f)
    
    # Import reproject here to avoid issues in main process
    from reproject import reproject_interp
    import reproject
    
    global CURRENT_TARGET_FILTER_IS_MIRI

    target_filter = job_spec['target_filter']
    CURRENT_TARGET_FILTER_IS_MIRI = target_filter in MIRI_FILTERNAMES
    transpose_mode = 'Image.ROTATE_180' if CURRENT_TARGET_FILTER_IS_MIRI else 'None'
    print(f"Worker transpose mode for target_filter={target_filter}: {transpose_mode}")
    filters = job_spec['filters']
    stretches = job_spec['stretches']
    filename = job_spec['filename']
    alma_overlay = job_spec.get('alma_overlay', False)
    new_basepath = job_spec.get('new_basepath', '/orange/adamginsburg/jwst/w51/data_reprojected/')
    os.makedirs(new_basepath, exist_ok=True)

    repr_image_filenames = {x: y.replace("i2d", f"i2d_reprj_{target_filter[:-1]}") for x, y in image_filenames.items()}
    repr_image_filenames = {x: (new_basepath + os.path.basename(y)) for x, y in repr_image_filenames.items()}
    repr_image_sub_filenames = {x: y.replace(".fits", f"reprj_{target_filter[:-1]}.fits") for x, y in image_sub_filenames.items()}
    repr_image_sub_filenames = {x: (new_basepath + os.path.basename(y)) for x, y in repr_image_sub_filenames.items()}
    
    png_path = os.path.dirname(filename)
    os.makedirs(png_path, exist_ok=True)
    
    # Load and prepare target header for reprojection
    tgt_header = fits.getheader(image_filenames[target_filter], ext=('SCI', 1))
    AVM = pyavm.AVM.from_header(tgt_header)
    
    # Load/reproject filter images
    filter_data = {}
    for filtername in filters:
        if filtername in image_filenames:
            image_file = image_filenames[filtername]
            repr_file = repr_image_filenames[filtername]
        elif filtername in image_sub_filenames:
            image_file = image_sub_filenames[filtername]
            repr_file = repr_image_sub_filenames[filtername]
        else:
            raise KeyError(f"Filter {filtername} not found in image_filenames or image_sub_filenames")

        nanfilled_file = repr_file.replace('.fits', '_nanfilled.fits')

        if os.path.exists(nanfilled_file):
            print(f"Loading cached nanfilled {filtername} from {nanfilled_file}")
            filter_data[filtername] = fits.getdata(nanfilled_file)
            continue

        if os.path.exists(repr_file):
            print(f"Loading cached reprojected {filtername} from {repr_file}")
            reprojected_data = fits.getdata(repr_file)
        else:
            print(f"Reprojecting {filtername} {image_file} to {repr_file}")
            reprojected_data, _ = reproject.reproject_interp(image_file, tgt_header, hdu_in='SCI')
            fits.PrimaryHDU(data=reprojected_data, header=tgt_header).writeto(repr_file, overwrite=True)

        if filtername in MIRI_FILTERNAMES:
            filter_data[filtername] = fill_nan(reprojected_data, big_island_threshold=10, bad_data_min_threshold=custom_negative_thresholds[filtername])
        else:
            filter_data[filtername] = fill_nan(reprojected_data, bad_data_min_threshold=custom_negative_thresholds.get(filtername, 1e-5))

        fits.PrimaryHDU(data=filter_data[filtername], header=tgt_header).writeto(nanfilled_file, overwrite=True)
        print(f"Wrote nanfilled cache for {filtername} to {nanfilled_file}")
    
    # Load ALMA data if needed
    alma_data = None
    alma_level = None
    if alma_overlay:
        alma_reproj_fn = f'{new_basepath}/alma_w51_reprojected_jwst_{target_filter[:-1]}.fits'
        if os.path.exists(alma_reproj_fn):
            alma_data = fits.getdata(alma_reproj_fn)
        else:
            print(f"Reprojecting ALMA data...")
            alma_w51e2_3mm = '/orange/adamginsburg/w51/2017.1.00293.S/may2021_successful_imaging/w51e2.spw0thru19.14500.robust0.thr0.075mJy.mfs.I.startmod.selfcal7.image.tt0.pbcor.fits'
            alma_w51irs2_3mm = '/orange/adamginsburg/w51/2017.1.00293.S/may2021_successful_imaging/w51n.spw0thru19.14500.robust0.thr0.075mJy.mfs.I.startmod.selfcal7.image.tt0.pbcor.fits'
            alma_level = 8.4e-5
            
            fh = fits.open(alma_w51e2_3mm)
            data = fh[0].data.squeeze()
            hdr = WCS(fh[0].header).celestial
            alma_w51e2_3mm_reprojected, footprint_e2 = reproject.reproject_interp((data, hdr), tgt_header)
            
            fh = fits.open(alma_w51irs2_3mm)
            data = fh[0].data.squeeze()
            hdr = WCS(fh[0].header).celestial
            alma_w51irs2_3mm_reprojected, footprint_irs2 = reproject.reproject_interp((data, hdr), tgt_header)
            
            alma_data = ((np.nan_to_num(alma_w51e2_3mm_reprojected) +
                         np.nan_to_num(alma_w51irs2_3mm_reprojected)) /
                        (footprint_e2 + footprint_irs2))
            
            os.makedirs(new_basepath, exist_ok=True)
            fits.writeto(alma_reproj_fn, alma_data, tgt_header, overwrite=True)
    
    # Build RGB array from loaded filters
    rgb = np.array([filter_data[f] for f in filters]).swapaxes(0, 2).swapaxes(0, 1)
    original_data = rgb.copy()
    
    # Apply stretches and normalization
    rgb_scaled = np.array([
        stretches[0]['func'](rgb[:, :, 0], **stretches[0]['kwargs'])(rgb[:, :, 0]),
        stretches[1]['func'](rgb[:, :, 1], **stretches[1]['kwargs'])(rgb[:, :, 1]),
        stretches[2]['func'](rgb[:, :, 2], **stretches[2]['kwargs'])(rgb[:, :, 2])
    ]).swapaxes(0, 2).swapaxes(0, 1)
    
    # Save RGB image
    save_rgb(
        rgb_scaled,
        filename,
        avm=AVM,
        alma_data=alma_data,
        alma_level=alma_level,
        original_data=original_data,
    )
    
    print(f"Saved {filename}")
    
    # Clean up job spec file
    if os.path.exists(job_spec_file):
        os.remove(job_spec_file)


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker-job', type=str, default=None,
                       help='Pickle file for RGB job worker')
    parser.add_argument('--target-filters', type=str, default='f140m,f480m,f2100w',
                       help='Comma-separated list of target filters')
    
    args = parser.parse_args()
    
    if args.worker_job:
        # This is a worker process
        worker_create_rgb(args.worker_job)
    else:
        # This is the main process
        target_filters = args.target_filters.split(',')
        for target_filter in target_filters:
            make_pngs(target_filter)


if __name__ == '__main__':
    main()
