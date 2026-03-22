from astropy.io import fits
import numpy as np
from astropy.visualization import simple_norm
import os
import pyavm
from PIL import Image
from jwst_rgb.save_rgb import save_rgb as _save_rgb, fill_nan
import subprocess
import pickle
import argparse
import hashlib
from astropy.wcs import WCS


CURRENT_TARGET_FILTER_IS_MIRI = False
MIRI_FILTERNAMES = {'f2550w', 'f1130w', 'f1500w', 'f770w'}


def save_rgb(*args, **kwargs):
    kwargs.setdefault('transpose', Image.ROTATE_180)
    kwargs.setdefault('alpha_only_edges', True)
    return _save_rgb(*args, **kwargs)


image_filenames_pipe = {
    'f115w': '/orange/adamginsburg/jwst/brick/F115W/pipeline/jw01182-o004_t001_nircam_clear-f115w-merged_i2d.fits',
    'f182m': '/orange/adamginsburg/jwst/brick/F182M/pipeline/jw02221-o001_t001_nircam_clear-f182m-merged_i2d.fits',
    'f187n': '/orange/adamginsburg/jwst/brick/F187N/pipeline/jw02221-o001_t001_nircam_clear-f187n-merged_i2d.fits',
    'f200w': '/orange/adamginsburg/jwst/brick/F200W/pipeline/jw01182-o004_t001_nircam_clear-f200w-merged_i2d.fits',
    'f212n': '/orange/adamginsburg/jwst/brick/F212N/pipeline/jw02221-o001_t001_nircam_clear-f212n-merged_i2d.fits',
    'f356w': '/orange/adamginsburg/jwst/brick/F356W/pipeline/jw01182-o004_t001_nircam_clear-f356w-merged_i2d.fits',
    'f405n': '/orange/adamginsburg/jwst/brick/F405N/pipeline/jw02221-o001_t001_nircam_clear-f405n-merged_i2d.fits',
    'f410m': '/orange/adamginsburg/jwst/brick/F410M/pipeline/jw02221-o001_t001_nircam_clear-f410m-merged_i2d.fits',
    'f444w': '/orange/adamginsburg/jwst/brick/F444W/pipeline/jw01182-o004_t001_nircam_clear-f444w-merged_i2d.fits',
    'f466n': '/orange/adamginsburg/jwst/brick/F466N/pipeline/jw02221-o001_t001_nircam_clear-f466n-merged_i2d.fits',
    'f2550w': '/orange/adamginsburg/jwst/brick/F2550W/pipeline/jw02221-o002_t001_miri_f2550w_i2d.fits',
    'f1130w': '/orange/adamginsburg/jwst//sickle/mastDownload/JWST/jw03958-o003_t003_miri_f1130w-brightsky/jw03958-o003_t003_miri_f1130w-brightsky_i2d.fits',
    'f1500w': '/orange/adamginsburg/jwst//sickle/mastDownload/JWST/jw03958-o003_t003_miri_f1500w-brightsky/jw03958-o003_t003_miri_f1500w-brightsky_i2d.fits',
    'f770w': '/orange/adamginsburg/jwst//sickle/mastDownload/JWST/jw03958-o003_t003_miri_f770w-brightsky/jw03958-o003_t003_miri_f770w-brightsky_i2d.fits',
}


image_sub_filenames_pipe = {
    'f405n-f410m': '/orange/adamginsburg/jwst/brick/images/F405_minus_F410cont_refitted405wcsto410_merged_destarred6.fits',
    'f410m-f405n': '/orange/adamginsburg/jwst/brick/images/F410_minus_F405_refitted405wcsto410_merged.fits',
    'f187n-f182m': '/orange/adamginsburg/jwst/brick/images/F187_minus_F182cont_refitted187wcsto182_merged_destarred6.fits',
    'f466n-f410m': '/orange/adamginsburg/jwst/brick/images/F466_minus_F410cont_refitted466wcsto410_merged_destarred6.fits',
}


submitted_rgb_filenames = set()


custom_negative_thresholds = {
    'f2550w': -50,
    'f1500w': -30,
    'f1130w': -30,
    'f770w': -20,
}


def build_reprojected_filenames(target_filter, new_basepath):
    repr_image_filenames = {x: y.replace('i2d', f'i2d_pipeline_v0.1_reprj_{target_filter[:-1]}') for x, y in image_filenames_pipe.items()}
    repr_image_filenames = {x: os.path.join(new_basepath, os.path.basename(y)) for x, y in repr_image_filenames.items()}
    repr_image_sub_filenames = {x: y.replace('.fits', f'reprj_{target_filter[:-1]}.fits') for x, y in image_sub_filenames_pipe.items()}
    repr_image_sub_filenames = {x: os.path.join(new_basepath, os.path.basename(y)) for x, y in repr_image_sub_filenames.items()}
    return repr_image_filenames, repr_image_sub_filenames


def make_pngs(
    target_filter='f466n',
    new_basepath='/orange/adamginsburg/jwst/brick/data_reprojected/',
    image_filenames_local=image_filenames_pipe,
    image_sub_filenames_local=image_sub_filenames_pipe,
    do_special_bgr=True,
):
    print(f'Submitting PNG creation jobs for {target_filter}')
    png_path = f'/orange/adamginsburg/jwst/brick/pngs_{target_filter[1:-1]}'
    os.makedirs(png_path, exist_ok=True)
    os.makedirs(new_basepath, exist_ok=True)

    filternames = sorted(list(image_filenames_local.keys()), key=lambda x: int(''.join(filter(str.isdigit, x))))[::-1]
    to_iterate_over = list(zip(filternames, filternames[1:], filternames[2:])) + list(zip(filternames, filternames[2:], filternames[4:]))

    for f1, f2, f3 in to_iterate_over:
        f1n, f2n, f3n = ''.join(filter(str.isdigit, f1)), ''.join(filter(str.isdigit, f2)), ''.join(filter(str.isdigit, f3))
        submit_rgb_job({
            'target_filter': target_filter,
            'channels': [f1, f2, f3],
            'stretches': [
                {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5},
                {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5},
                {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5},
            ],
            'filename': f'{png_path}/Brick_RGB_{f1n}-{f2n}-{f3n}.png',
            'alma_overlay': True,
            'new_basepath': new_basepath,
        })
        submit_rgb_job({
            'target_filter': target_filter,
            'channels': [f1, f2, f3],
            'stretches': [
                {'stretch': 'log', 'min_percent': 1.5, 'max_percent': 99.5},
                {'stretch': 'log', 'min_percent': 1.5, 'max_percent': 99.5},
                {'stretch': 'log', 'min_percent': 1.5, 'max_percent': 99.5},
            ],
            'filename': f'{png_path}/Brick_RGB_{f1n}-{f2n}-{f3n}_log.png',
            'alma_overlay': True,
            'new_basepath': new_basepath,
        })

    filternames_sub = sorted(list(image_sub_filenames_local.keys()), key=lambda x: int(''.join(filter(str.isdigit, x))))[::-1]
    to_iterate_over_sub = list(zip(filternames_sub, filternames_sub[1:], filternames_sub[2:])) + list(zip(filternames_sub, filternames_sub[2:], filternames_sub[4:]))

    for f1, f2, f3 in to_iterate_over_sub:
        f1n, f2n, f3n = ''.join(filter(str.isdigit, f1)), ''.join(filter(str.isdigit, f2)), ''.join(filter(str.isdigit, f3))
        submit_rgb_job({
            'target_filter': target_filter,
            'channels': [f1, f2, f3],
            'stretches': [
                {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5},
                {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5},
                {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5},
            ],
            'filename': f'{png_path}/Brick_RGB_{f1n}-{f2n}-{f3n}_sub.png',
            'alma_overlay': True,
            'new_basepath': new_basepath,
        })
        submit_rgb_job({
            'target_filter': target_filter,
            'channels': [f1, f2, f3],
            'stretches': [
                {'stretch': 'log', 'min_percent': 1.0, 'max_percent': 99.5},
                {'stretch': 'log', 'min_percent': 1.0, 'max_percent': 99.5},
                {'stretch': 'log', 'min_percent': 1.0, 'max_percent': 99.5},
            ],
            'filename': f'{png_path}/Brick_RGB_{f1n}-{f2n}-{f3n}_sub_log.png',
            'alma_overlay': True,
            'new_basepath': new_basepath,
        })

    if do_special_bgr:
        submit_rgb_job({
            'target_filter': target_filter,
            'channels': ['f405n', 'sum:f405n+f466n', 'f466n'],
            'stretches': [
                {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5},
                {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5},
                {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5},
            ],
            'filename': f'{png_path}/Brick_BGR_405-405466-466.png',
            'alma_overlay': True,
            'new_basepath': new_basepath,
        })
        submit_rgb_job({
            'target_filter': target_filter,
            'channels': ['f405n', 'sum:f405n+f466n', 'f466n'],
            'stretches': [
                {'stretch': 'log', 'min_percent': 1.5, 'max_percent': 99.5},
                {'stretch': 'log', 'min_percent': 1.5, 'max_percent': 99.5},
                {'stretch': 'log', 'min_percent': 1.5, 'max_percent': 99.5},
            ],
            'filename': f'{png_path}/Brick_BGR_405-405466-466_log.png',
            'alma_overlay': True,
            'new_basepath': new_basepath,
        })

        submit_rgb_job({
            'target_filter': target_filter,
            'channels': ['f410m', 'sum:f410m+f466n', 'f466n'],
            'stretches': [
                {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5},
                {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5},
                {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5},
            ],
            'filename': f'{png_path}/Brick_BGR_410-410466-466.png',
            'alma_overlay': True,
            'new_basepath': new_basepath,
        })
        submit_rgb_job({
            'target_filter': target_filter,
            'channels': ['f410m', 'sum:f410m+f466n', 'f466n'],
            'stretches': [
                {'stretch': 'log', 'min_percent': 1.5, 'max_percent': 99.5},
                {'stretch': 'log', 'min_percent': 1.5, 'max_percent': 99.5},
                {'stretch': 'log', 'min_percent': 1.5, 'max_percent': 99.5},
            ],
            'filename': f'{png_path}/Brick_BGR_410-410466-466_log.png',
            'alma_overlay': True,
            'new_basepath': new_basepath,
        })

        submit_rgb_job({
            'target_filter': target_filter,
            'channels': ['f212n', 'f405n', 'f466n'],
            'stretches': [
                {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5},
                {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5},
                {'stretch': 'asinh', 'min_percent': 1, 'max_percent': 99.5},
            ],
            'filename': f'{png_path}/Brick_BGR_212-405-466.png',
            'alma_overlay': True,
            'new_basepath': new_basepath,
        })
        submit_rgb_job({
            'target_filter': target_filter,
            'channels': ['f212n', 'f405n', 'f466n'],
            'stretches': [
                {'stretch': 'log', 'min_percent': 1.5, 'max_percent': 99.5},
                {'stretch': 'log', 'min_percent': 1.5, 'max_percent': 99.5},
                {'stretch': 'log', 'min_percent': 1.5, 'max_percent': 99.5},
            ],
            'filename': f'{png_path}/Brick_BGR_212-405-466_log.png',
            'alma_overlay': True,
            'new_basepath': new_basepath,
        })


def submit_rgb_job(job_spec):
    filename = job_spec['filename']
    alma_overlay = job_spec.get('alma_overlay', False)

    if alma_overlay and not filename.endswith('_alma.png'):
        non_alma_spec = dict(job_spec)
        non_alma_spec['alma_overlay'] = False

        alma_spec = dict(job_spec)
        alma_spec['filename'] = filename.replace('.png', '_alma.png')
        alma_spec['alma_overlay'] = True

        submit_rgb_job(non_alma_spec)
        return submit_rgb_job(alma_spec)

    submit_key = (filename, alma_overlay)
    if submit_key in submitted_rgb_filenames:
        print(f'Skipping duplicate RGB job for {filename}')
        return None
    submitted_rgb_filenames.add(submit_key)

    job_spec_dir = '/orange/adamginsburg/jwst/brick/job_specs/'
    os.makedirs(job_spec_dir, exist_ok=True)

    spec_hash = hashlib.md5(str(sorted(job_spec.items())).encode()).hexdigest()[:8]
    job_name_base = os.path.basename(filename).replace('.png', '')[:20]
    job_spec_file = os.path.join(job_spec_dir, f'{job_name_base}_{spec_hash}.pkl')
    with open(job_spec_file, 'wb') as f:
        pickle.dump(job_spec, f)

    script_path = os.path.abspath(__file__)
    python_cmd = f'python {script_path} --worker-job {job_spec_file}'

    mem = {'f466n': 64, 'f200w': 64, 'f187n': 64, 'f444w': 64}.get(job_spec['target_filter'], 64)
    alma_tag = '_alma' if alma_overlay else ''
    job_name = os.path.basename(filename).replace('.png', '')[:30] + alma_tag + '_' + job_spec['target_filter']
    log_file = f'/blue/adamginsburg/adamginsburg/logs/brick_quickimages-{job_name}_%j.log'
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
        '--wrap',
        python_cmd,
    ]

    result = subprocess.run(sbatch_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'Error submitting job for {filename}:')
        print(result.stderr)
        raise ValueError(f'Failed to submit job for {filename}')
    print(f'Submitted {job_name} job for {filename}: {result.stdout.strip()}')
    return job_spec_file


def _reproject_if_needed(input_file, output_file, tgt_header):
    import reproject

    if os.path.exists(output_file):
        return fits.getdata(output_file)

    print(f'Reprojecting {input_file} to {output_file}')
    try:
        result, _ = reproject.reproject_interp(input_file, tgt_header, hdu_in='SCI')
    except KeyError:
        result, _ = reproject.reproject_interp(input_file, tgt_header, hdu_in=0)
    fits.PrimaryHDU(data=result, header=tgt_header).writeto(output_file, overwrite=True)
    return result


def _load_alma_data(new_basepath, target_filter, tgt_header):
    import reproject

    alma_brick_3mm = '/orange/adamginsburg/brick/alma/rathborne/brick.cont.alma.image.fits'
    alma_level = 3e-4
    alma_reproj_fn = f'{new_basepath}/alma_brick_reprojected_jwst_{target_filter[:-1]}.fits'

    if os.path.exists(alma_reproj_fn):
        return fits.getdata(alma_reproj_fn), alma_level

    print(f'Reprojecting ALMA data to {alma_reproj_fn}')
    fh = fits.open(alma_brick_3mm)
    data = fh[0].data.squeeze()
    hdr = WCS(fh[0].header).celestial
    alma_brick_reprojected_jwst, footprint = reproject.reproject_interp((data, hdr), tgt_header)
    alma_brick_reprojected_jwst = np.nan_to_num(alma_brick_reprojected_jwst) / footprint
    fits.writeto(alma_reproj_fn, alma_brick_reprojected_jwst, tgt_header, overwrite=True)
    return alma_brick_reprojected_jwst, alma_level


def worker_create_rgb(job_spec_file):
    with open(job_spec_file, 'rb') as f:
        job_spec = pickle.load(f)

    target_filter = job_spec['target_filter']
    channels = job_spec['channels']
    stretches = job_spec['stretches']
    filename = job_spec['filename']
    alma_overlay = job_spec.get('alma_overlay', False)
    new_basepath = job_spec.get('new_basepath', '/orange/adamginsburg/jwst/brick/data_reprojected/')
    os.makedirs(new_basepath, exist_ok=True)

    global CURRENT_TARGET_FILTER_IS_MIRI
    CURRENT_TARGET_FILTER_IS_MIRI = target_filter in MIRI_FILTERNAMES
    transpose_mode = 'Image.ROTATE_180' if CURRENT_TARGET_FILTER_IS_MIRI else 'None'
    print(f'Worker transpose mode for target_filter={target_filter}: {transpose_mode}')

    repr_image_filenames, repr_image_sub_filenames = build_reprojected_filenames(target_filter, new_basepath)

    try:
        tgt_header = fits.getheader(image_filenames_pipe[target_filter], ext=('SCI', 1))
    except KeyError:
        tgt_header = fits.getheader(image_filenames_pipe[target_filter], ext=0)

    avm = pyavm.AVM.from_header(tgt_header)
    cache = {}

    def ensure_reprojected(key):
        if key in image_filenames_pipe:
            if key in cache:
                return cache[key]['repr'], repr_image_filenames[key]
            data = _reproject_if_needed(image_filenames_pipe[key], repr_image_filenames[key], tgt_header)
            cache[key] = {'repr': data}
            return data, repr_image_filenames[key]
        if key in image_sub_filenames_pipe:
            if key in cache:
                return cache[key]['repr'], repr_image_sub_filenames[key]
            data = _reproject_if_needed(image_sub_filenames_pipe[key], repr_image_sub_filenames[key], tgt_header)
            cache[key] = {'repr': data}
            return data, repr_image_sub_filenames[key]
        raise KeyError(f'Unknown data key: {key}')

    def get_data_for_key(key, nanfill=True):
        repr_data, repr_file = ensure_reprojected(key)
        if not nanfill:
            return repr_data

        nan_key = f'{key}__nanfill'
        if nan_key in cache:
            return cache[nan_key]

        nanfilled_file = repr_file.replace('.fits', '_nanfilled.fits')
        if os.path.exists(nanfilled_file):
            data = fits.getdata(nanfilled_file)
        else:
            if key in MIRI_FILTERNAMES:
                data = fill_nan(
                    repr_data,
                    big_island_threshold=10,
                    bad_data_min_threshold=custom_negative_thresholds.get(key, None),
                )
            else:
                data = fill_nan(repr_data, bad_data_min_threshold=None)
            fits.PrimaryHDU(data=data, header=tgt_header).writeto(nanfilled_file, overwrite=True)
            print(f'Wrote nanfilled cache for {key} to {nanfilled_file}')

        cache[nan_key] = data
        return data

    def resolve_channel(channel_token):
        if channel_token.startswith('sum:'):
            expr = channel_token.split(':', 1)[1]
            left, right = expr.split('+', 1)
            return get_data_for_key(left, nanfill=True) + get_data_for_key(right, nanfill=True)
        return get_data_for_key(channel_token, nanfill=True)

    rgb = np.array([resolve_channel(ch) for ch in channels]).swapaxes(0, 2).swapaxes(0, 1)
    original_data = rgb.copy()

    rgb_scaled = np.array([
        simple_norm(rgb[:, :, 0], **stretches[0])(rgb[:, :, 0]),
        simple_norm(rgb[:, :, 1], **stretches[1])(rgb[:, :, 1]),
        simple_norm(rgb[:, :, 2], **stretches[2])(rgb[:, :, 2]),
    ]).swapaxes(0, 2).swapaxes(0, 1)

    alma_data = None
    alma_level = None
    if alma_overlay:
        alma_data, alma_level = _load_alma_data(new_basepath, target_filter, tgt_header)

    output_filename = filename
    if alma_overlay and not output_filename.endswith('_alma.png'):
        output_filename = output_filename.replace('.png', '_alma.png')

    save_rgb(
        rgb_scaled,
        output_filename,
        avm=avm,
        alma_data=alma_data,
        alma_level=alma_level,
        original_data=original_data,
    )
    print(f'Saved {output_filename}')

    if os.path.exists(job_spec_file):
        os.remove(job_spec_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker-job', type=str, default=None, help='Pickle file for RGB job worker')
    parser.add_argument('--target-filters', type=str, default='f466n,f200w,f187n,f444w', help='Comma-separated list of target filters')
    args = parser.parse_args()

    if args.worker_job:
        worker_create_rgb(args.worker_job)
        return

    wide_only = {x: y for x, y in image_filenames_pipe.items() if x.endswith('w')}
    for target_filter in ('f200w', 'f444w'):
        if target_filter in args.target_filters.split(','):
            make_pngs(
                target_filter,
                image_filenames_local=wide_only,
                do_special_bgr=False,
            )

    for target_filter in args.target_filters.split(','):
        make_pngs(target_filter)


if __name__ == '__main__':
    main()
