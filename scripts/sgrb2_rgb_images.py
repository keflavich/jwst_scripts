#!/usr/bin/env python
# coding: utf-8

from astropy.io import fits
import numpy as np
from astropy.visualization import simple_norm
import pylab as plt
from astropy import wcs
import os
from reproject import reproject_interp
import reproject
import PIL
import shutil
from astropy.wcs import WCS
import pyavm
from PIL import Image
from jwst_rgb.save_rgb import save_rgb #NB: brought the function back with some modifications to keep nans white.
from jwst_rgb.save_rgb import fill_nan


# def save_rgb(img, filename, avm=None, flip=-1):

#     nan_mask = ~np.isfinite(img).all(axis=2)
#     img[nan_mask] = [1, 1, 1]
#     img = (img*256)
#     img[img<0] = 0
#     img[img>255] = 255
#     img = img.astype('uint8')
#     img = PIL.Image.fromarray(img[::flip,:,:])
#     img.save(filename)
#     if avm is not None:
#         base = os.path.basename(filename)
#         dir = os.path.dirname(filename)
#         avmname = os.path.join(dir, 'avm_'+base)
#         avm.embed(filename, avmname)
#         shutil.move(avmname, filename)
#     return img




image_filenames_pipe = {
    "f150w": "/orange/adamginsburg/jwst/sgrb2/NB/F150W/pipeline/jw05365-o001_t001_nircam_clear-f150w-merged_i2d.fits",
    "f182m": "/orange/adamginsburg/jwst/sgrb2/NB/F182M/pipeline/jw05365-o001_t001_nircam_clear-f182m-merged_i2d.fits",
    "f187n": "/orange/adamginsburg/jwst/sgrb2/NB/F187N/pipeline/jw05365-o001_t001_nircam_clear-f187n-merged_i2d.fits",
    "f210m": "/orange/adamginsburg/jwst/sgrb2/NB/F210M/pipeline/jw05365-o001_t001_nircam_clear-f210m-merged_i2d.fits",
    "f212n": "/orange/adamginsburg/jwst/sgrb2/NB/F212N/pipeline/jw05365-o001_t001_nircam_clear-f212n-merged_i2d.fits",
    "f300m": "/orange/adamginsburg/jwst/sgrb2/NB/F300M/pipeline/jw05365-o001_t001_nircam_clear-f300m-merged_i2d.fits",
    "f360m": "/orange/adamginsburg/jwst/sgrb2/NB/F360M/pipeline/jw05365-o001_t001_nircam_clear-f360m-merged_i2d.fits",
    "f405n": "/orange/adamginsburg/jwst/sgrb2/NB/F405N/pipeline/jw05365-o001_t001_nircam_clear-f405n-merged_i2d.fits",
    "f410m": "/orange/adamginsburg/jwst/sgrb2/NB/F410M/pipeline/jw05365-o001_t001_nircam_clear-f410m-merged_i2d.fits",
    "f466n": "/orange/adamginsburg/jwst/sgrb2/NB/F466N/pipeline/jw05365-o001_t001_nircam_clear-f466n-merged_i2d.fits",
    "f480m": "/orange/adamginsburg/jwst/sgrb2/NB/F480M/pipeline/jw05365-o001_t001_nircam_clear-f480m-merged_i2d.fits",
    #"f770w": "/orange/adamginsburg/jwst/sgrb2/mastDownload/JWST/jw05365-o002_t002_miri_f770w/jw05365-o002_t002_miri_f770w_i2d.fits",
    #"f1280w": "/orange/adamginsburg/jwst/sgrb2/mastDownload/JWST/jw05365-o002_t002_miri_f1280w/jw05365-o002_t002_miri_f1280w_i2d.fits",
    #"f2550w": "/orange/adamginsburg/jwst/sgrb2/mastDownload/JWST/jw05365-o002_t002_miri_f2550w/jw05365-o002_t002_miri_f2550w_i2d.fits",
    # upd Jul 21, 2025, Nazar B.
    # These files have no saturation checks, which make them look nicer for visualizations.
    "f770w": "/orange/adamginsburg/jwst/sgrb2/NB/pipeline_reruns/MIRI_no_saturation_checks/jw05365-o002_t002_miri_f770w_i2d.fits",
    "f1280w": "/orange/adamginsburg/jwst/sgrb2/NB/pipeline_reruns/MIRI_no_saturation_checks/jw05365-o002_t002_miri_f1280w_i2d.fits",
    "f2550w": "/orange/adamginsburg/jwst/sgrb2/NB/pipeline_reruns/MIRI_no_saturation_checks/jw05365-o002_t002_miri_f2550w_i2d.fits",
}

image_sub_filenames_pipe = {
    "f405n-f410m": "/orange/adamginsburg/jwst/sgrb2/NB/F405_minus_F410cont_pipeline_v0.1.fits",
    "f410m-f405n": "/orange/adamginsburg/jwst/sgrb2/NB/F410_minus_F405_fractional_bandwidth_pipeline_v0.1.fits",
    "f212n-f210m": "/orange/adamginsburg/jwst/sgrb2/NB/F212_minus_F210cont_pipeline_v0.1.fits",
    "f187n-f182m": "/orange/adamginsburg/jwst/sgrb2/NB/F187_minus_F182cont_pipeline_v0.1.fits", # NB: added the new pipeline version
    "f480m-f360m": "/orange/adamginsburg/jwst/sgrb2/NB/filter_subtractions/f480m_minus_f360m_scaled_BB.fits", # NB: added another filter pair
}

def make_pngs(target_filter='f466n', new_basepath='/orange/adamginsburg/jwst/sgrb2/NB/data_reprojected/'):
    print(f"Making PNGs for {target_filter}")

    png_path = f'/orange/adamginsburg/jwst/sgrb2/pngs_{target_filter[1:-1]}'
    os.makedirs(png_path, exist_ok=True)

    tgt_header = fits.getheader(image_filenames_pipe[target_filter], ext=('SCI', 1))
    AVM = pyavm.AVM.from_header(tgt_header)

    repr_image_filenames = {x: y.replace("i2d", f"i2d_pipeline_v0.1_reprj_{target_filter[:-1]}") for x,y in image_filenames_pipe.items()}
    repr_image_filenames = {x: (new_basepath+os.path.basename(y)) for x,y in repr_image_filenames.items()}
    repr_image_sub_filenames = {x: y.replace(".fits", f"reprj_{target_filter[:-1]}.fits") for x,y in image_sub_filenames_pipe.items()}
    repr_image_sub_filenames = {x: (new_basepath+os.path.basename(y)) for x,y in repr_image_sub_filenames.items()}

    for filtername in image_filenames_pipe:
        if not os.path.exists(repr_image_filenames[filtername]):
            print(f"Reprojecting {filtername} {image_filenames_pipe[filtername]} to {repr_image_filenames[filtername]}")
            result,_ = reproject.reproject_interp(image_filenames_pipe[filtername], tgt_header, hdu_in='SCI')
            hdu = fits.PrimaryHDU(data=result, header=tgt_header)
            hdu.writeto(repr_image_filenames[filtername], overwrite=True)

    for filtername in image_sub_filenames_pipe:
        if not os.path.exists(repr_image_sub_filenames[filtername]):
            print(f"Reprojecting {filtername} {image_sub_filenames_pipe[filtername]} to {repr_image_sub_filenames[filtername]}")
            result,_ = reproject.reproject_interp(image_sub_filenames_pipe[filtername], tgt_header, hdu_in='SCI')
            hdu = fits.PrimaryHDU(data=result, header=tgt_header)
            hdu.writeto(repr_image_sub_filenames[filtername], overwrite=True)

    alma_sgrb2n_3mm = "/orange/adamginsburg/sgrb2/NB/the_end/sgr_b2.N.B3.cont.r0.5.1m0.075mJy.cal2.image.tt0.pbcor.fits"
    alma_sgrb2m_3mm = "/orange/adamginsburg/sgrb2/NB/the_end/sgr_b2.M.B3.cont.r0.5.1m0.125mJy.cal3.image.tt0.pbcor.fits"
    alma_level = 3e-4

    alma_reproj_fn = f'/orange/adamginsburg/jwst/sgrb2/data_reprojected/alma_sgrb2_reprojected_jwst_{target_filter[:-1]}.fits'
    if os.path.exists(alma_reproj_fn):
        alma_sgrb2_reprojected_jwst = fits.getdata(alma_reproj_fn)
    else:
        print(f"Reprojecting ALMA data to {alma_reproj_fn}")
        fh = fits.open(alma_sgrb2n_3mm)
        data = fh[0].data.squeeze()
        hdr = WCS(fh[0].header).celestial
        alma_sgrb2n_3mm_reprojected, footprint_n = reproject.reproject_interp((data, hdr), tgt_header)

        fh = fits.open(alma_sgrb2m_3mm)
        data = fh[0].data.squeeze()
        hdr = WCS(fh[0].header).celestial
        alma_sgrb2m_3mm_reprojected, footprint_m = reproject.reproject_interp((data, hdr), tgt_header)

        alma_sgrb2_reprojected_jwst = ((np.nan_to_num(alma_sgrb2n_3mm_reprojected) +
                                        np.nan_to_num(alma_sgrb2m_3mm_reprojected)) /
                                    (footprint_n + footprint_m))
        del alma_sgrb2n_3mm_reprojected, alma_sgrb2m_3mm_reprojected, footprint_n, footprint_m
        fits.writeto(alma_reproj_fn, alma_sgrb2_reprojected_jwst, tgt_header, overwrite=True)




    filternames = sorted(list(image_filenames_pipe.keys()),
                        key=lambda x: int(''.join(filter(str.isdigit, x))))[::-1]
    print(f"Sorted list of filters: {filternames}")

    for f1, f2, f3 in zip(filternames, filternames[1:], filternames[2:]):
        print(f1,f2,f3)
        rgb = np.array([
            fits.getdata(repr_image_filenames[f1]),
            fits.getdata(repr_image_filenames[f2]),
            fits.getdata(repr_image_filenames[f3]),
        ]).swapaxes(0,2).swapaxes(0,1)
        rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,0]),
                            simple_norm(rgb[:,:,1], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,1]),
                            simple_norm(rgb[:,:,2], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)

        f1n, f2n, f3n = ''.join(filter(str.isdigit, f1)), ''.join(filter(str.isdigit, f2)), ''.join(filter(str.isdigit, f3))
        save_rgb(rgb_scaled, f'{png_path}/SgrB2_RGB_{f1n}-{f2n}-{f3n}.png', avm=AVM, original_data=rgb)
        save_rgb(rgb_scaled, f'{png_path}/SgrB2_RGB_{f1n}-{f2n}-{f3n}_alma.png', avm=AVM, alma_data=alma_sgrb2_reprojected_jwst, alma_level=alma_level, original_data=rgb)

        rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='log', min_percent=1.5, max_percent=99.5)(rgb[:,:,0]),
                            simple_norm(rgb[:,:,1], stretch='log', min_percent=1.5, max_percent=99.5)(rgb[:,:,1]),
                            simple_norm(rgb[:,:,2], stretch='log', min_percent=1.5, max_percent=99.5)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)

        save_rgb(rgb_scaled, f'{png_path}/SgrB2_RGB_{f1n}-{f2n}-{f3n}_log.png', avm=AVM, original_data=rgb)
        save_rgb(rgb_scaled, f'{png_path}/SgrB2_RGB_{f1n}-{f2n}-{f3n}_log_alma.png', avm=AVM, alma_data=alma_sgrb2_reprojected_jwst, alma_level=alma_level, original_data=rgb)

    filternames_sub = sorted(list(image_sub_filenames_pipe.keys()),
                           key=lambda x: int(''.join(filter(str.isdigit, x))))[::-1]
    print(f"Sorted list of subtracted-filters: {filternames_sub}")

    for f1, f2, f3 in zip(filternames_sub, filternames_sub[1:], filternames_sub[2:]):
        print(f1,f2,f3)
        try:
            rgb = np.array([
                fits.getdata(repr_image_sub_filenames[f1]),
                fits.getdata(repr_image_sub_filenames[f2]),
                fits.getdata(repr_image_sub_filenames[f3]),
            ]).swapaxes(0,2).swapaxes(0,1)
        except Exception as ex:
            print(ex)
            print(f"Shape of {f1} is {fits.getdata(repr_image_sub_filenames[f1]).shape}")
            print(f"Shape of {f2} is {fits.getdata(repr_image_sub_filenames[f2]).shape}")
            print(f"Shape of {f3} is {fits.getdata(repr_image_sub_filenames[f3]).shape}")
            raise ex

        rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,0]),
                            simple_norm(rgb[:,:,1], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,1]),
                            simple_norm(rgb[:,:,2], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)

        f1n, f2n, f3n = ''.join(filter(str.isdigit, f1)), ''.join(filter(str.isdigit, f2)), ''.join(filter(str.isdigit, f3))
        save_rgb(rgb_scaled, f'{png_path}/SgrB2_RGB_{f1n}-{f2n}-{f3n}_sub.png', avm=AVM, original_data=rgb)
        try:
            save_rgb(rgb_scaled, f'{png_path}/SgrB2_RGB_{f1n}-{f2n}-{f3n}_sub_alma.png', avm=AVM, alma_data=alma_sgrb2_reprojected_jwst, alma_level=alma_level, original_data=rgb)
        except Exception as ex:
            print(ex)
            print(f"ALMA data shape = {fits.getdata(alma_reproj_fn).shape}")
            print(f"RGB data shape = {rgb.shape}")
            raise ex

        rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='log', min_percent=1.0, max_percent=99.5)(rgb[:,:,0]),
                            simple_norm(rgb[:,:,1], stretch='log', min_percent=1.0, max_percent=99.5)(rgb[:,:,1]),
                            simple_norm(rgb[:,:,2], stretch='log', min_percent=1.0, max_percent=99.5)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)

        save_rgb(rgb_scaled, f'{png_path}/SgrB2_RGB_{f1n}-{f2n}-{f3n}_sub_log.png', avm=AVM, original_data=rgb)
        try:
            save_rgb(rgb_scaled, f'{png_path}/SgrB2_RGB_{f1n}-{f2n}-{f3n}_sub_log_alma.png', avm=AVM, alma_data=alma_sgrb2_reprojected_jwst, alma_level=alma_level, original_data=rgb)
        except Exception as ex:
            print(ex)
            print(f"ALMA data shape = {fits.getdata(alma_reproj_fn).shape}")
            print(f"RGB data shape = {rgb.shape}")
            raise ex

    # Special BGR combinations as requested
    print("Creating special BGR combinations:")

    # BGR = 405, 405+466, 466
    print("Creating BGR: 405, 405+466, 466")
    f405_data = fits.getdata(repr_image_filenames['f405n'])
    f466_data = fits.getdata(repr_image_filenames['f466n'])

    # Create composite 405+466 channel
    f405_466_data = f405_data + f466_data # NB: shouldn't this be divided by 2 to normalize relative to other colors in the RGB?

    # BGR arrangement: Blue=405, Green=405+466, Red=466
    bgr_405_405466_466 = np.array([
        f405_data,      # Blue
        f405_466_data,  # Green
        f466_data       # Red
    ]).swapaxes(0,2).swapaxes(0,1)

    bgr_scaled = np.array([
        simple_norm(bgr_405_405466_466[:,:,0], stretch='asinh', min_percent=1, max_percent=99.5)(bgr_405_405466_466[:,:,0]),
        simple_norm(bgr_405_405466_466[:,:,1], stretch='asinh', min_percent=1, max_percent=99.5)(bgr_405_405466_466[:,:,1]),
        simple_norm(bgr_405_405466_466[:,:,2], stretch='asinh', min_percent=1, max_percent=99.5)(bgr_405_405466_466[:,:,2])
    ]).swapaxes(0,2).swapaxes(0,1)

    save_rgb(bgr_scaled, f'{png_path}/SgrB2_BGR_405-405466-466.png', avm=AVM, original_data=bgr_405_405466_466)
    save_rgb(bgr_scaled, f'{png_path}/SgrB2_BGR_405-405466-466_alma.png', avm=AVM, alma_data=alma_sgrb2_reprojected_jwst, alma_level=alma_level, original_data=bgr_405_405466_466)

    # Log version
    bgr_scaled_log = np.array([
        simple_norm(bgr_405_405466_466[:,:,0], stretch='log', min_percent=1.5, max_percent=99.5)(bgr_405_405466_466[:,:,0]),
        simple_norm(bgr_405_405466_466[:,:,1], stretch='log', min_percent=1.5, max_percent=99.5)(bgr_405_405466_466[:,:,1]),
        simple_norm(bgr_405_405466_466[:,:,2], stretch='log', min_percent=1.5, max_percent=99.5)(bgr_405_405466_466[:,:,2])
    ]).swapaxes(0,2).swapaxes(0,1)

    save_rgb(bgr_scaled_log, f'{png_path}/SgrB2_BGR_405-405466-466_log.png', avm=AVM, original_data=bgr_405_405466_466)
    save_rgb(bgr_scaled_log, f'{png_path}/SgrB2_BGR_405-405466-466_log_alma.png', avm=AVM, alma_data=alma_sgrb2_reprojected_jwst, alma_level=alma_level, original_data=bgr_405_405466_466)

    # BGR = 410, 410+466, 466
    print("Creating BGR: 410, 410+466, 466")
    f410_data = fits.getdata(repr_image_filenames['f410m'])

    # Create composite 410+466 channel
    f410_466_data = f410_data + f466_data

    # BGR arrangement: Blue=410, Green=410+466, Red=466
    bgr_410_410466_466 = np.array([
        f410_data,      # Blue
        f410_466_data,  # Green
        f466_data       # Red
    ]).swapaxes(0,2).swapaxes(0,1)

    bgr_scaled = np.array([
        simple_norm(bgr_410_410466_466[:,:,0], stretch='asinh', min_percent=1, max_percent=99.5)(bgr_410_410466_466[:,:,0]),
        simple_norm(bgr_410_410466_466[:,:,1], stretch='asinh', min_percent=1, max_percent=99.5)(bgr_410_410466_466[:,:,1]),
        simple_norm(bgr_410_410466_466[:,:,2], stretch='asinh', min_percent=1, max_percent=99.5)(bgr_410_410466_466[:,:,2])
    ]).swapaxes(0,2).swapaxes(0,1)

    save_rgb(bgr_scaled, f'{png_path}/SgrB2_BGR_410-410466-466.png', avm=AVM, original_data=bgr_410_410466_466)
    save_rgb(bgr_scaled, f'{png_path}/SgrB2_BGR_410-410466-466_alma.png', avm=AVM, alma_data=alma_sgrb2_reprojected_jwst, alma_level=alma_level, original_data=bgr_410_410466_466)

    # Log version
    bgr_scaled_log = np.array([
        simple_norm(bgr_410_410466_466[:,:,0], stretch='log', min_percent=1.5, max_percent=99.5)(bgr_410_410466_466[:,:,0]),
        simple_norm(bgr_410_410466_466[:,:,1], stretch='log', min_percent=1.5, max_percent=99.5)(bgr_410_410466_466[:,:,1]),
        simple_norm(bgr_410_410466_466[:,:,2], stretch='log', min_percent=1.5, max_percent=99.5)(bgr_410_410466_466[:,:,2])
    ]).swapaxes(0,2).swapaxes(0,1)

    save_rgb(bgr_scaled_log, f'{png_path}/SgrB2_BGR_410-410466-466_log.png', avm=AVM, original_data=bgr_410_410466_466)
    save_rgb(bgr_scaled_log, f'{png_path}/SgrB2_BGR_410-410466-466_log_alma.png', avm=AVM, alma_data=alma_sgrb2_reprojected_jwst, alma_level=alma_level, original_data=bgr_410_410466_466)

    # BGR = 212, 405, 466
    print("Creating BGR: 212, 405, 466")
    f212_data = fits.getdata(repr_image_filenames['f212n'])


    # BGR arrangement: Blue=212, Green=405, Red=466
    bgr_212_405_466 = np.array([
        f212_data,  # Blue
        f405_data,  # Green
        f466_data   # Red
    ]).swapaxes(0,2).swapaxes(0,1)

    bgr_scaled = np.array([
        simple_norm(bgr_212_405_466[:,:,0], stretch='asinh', min_percent=1, max_percent=99.5)(bgr_212_405_466[:,:,0]),
        simple_norm(bgr_212_405_466[:,:,1], stretch='asinh', min_percent=1, max_percent=99.5)(bgr_212_405_466[:,:,1]),
        simple_norm(bgr_212_405_466[:,:,2], stretch='asinh', min_percent=1, max_percent=99.5)(bgr_212_405_466[:,:,2])
    ]).swapaxes(0,2).swapaxes(0,1)

    save_rgb(bgr_scaled, f'{png_path}/SgrB2_BGR_212-405-466.png', avm=AVM, original_data=bgr_212_405_466)
    save_rgb(bgr_scaled, f'{png_path}/SgrB2_BGR_212-405-466_alma.png', avm=AVM, alma_data=alma_sgrb2_reprojected_jwst, alma_level=alma_level, original_data=bgr_212_405_466)

    # Log version
    bgr_scaled_log = np.array([
        simple_norm(bgr_212_405_466[:,:,0], stretch='log', min_percent=1.5, max_percent=99.5)(bgr_212_405_466[:,:,0]),
        simple_norm(bgr_212_405_466[:,:,1], stretch='log', min_percent=1.5, max_percent=99.5)(bgr_212_405_466[:,:,1]),
        simple_norm(bgr_212_405_466[:,:,2], stretch='log', min_percent=1.5, max_percent=99.5)(bgr_212_405_466[:,:,2])
    ]).swapaxes(0,2).swapaxes(0,1)

    save_rgb(bgr_scaled_log, f'{png_path}/SgrB2_BGR_212-405-466_log.png', avm=AVM, original_data=bgr_212_405_466)
    save_rgb(bgr_scaled_log, f'{png_path}/SgrB2_BGR_212-405-466_log_alma.png', avm=AVM, alma_data=alma_sgrb2_reprojected_jwst, alma_level=alma_level, original_data=bgr_212_405_466)

    rgb = np.array([
        fill_nan(fits.getdata(repr_image_filenames['f2550w'])),
        fill_nan(fits.getdata(repr_image_filenames['f770w'])),
        fits.getdata(repr_image_filenames['f480m'])
    ]).swapaxes(0,2).swapaxes(0,1)

    rgb_scaled = np.array([
        simple_norm(rgb[:,:,0], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,0]),
        simple_norm(rgb[:,:,1], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,1]),
        simple_norm(rgb[:,:,2], stretch='log', min_percent=1, max_percent=99.5)(rgb[:,:,2])
    ]).swapaxes(0,2).swapaxes(0,1)

    save_rgb(rgb_scaled, f'{png_path}/SgrB2_RGB_2550-770-480.png', avm=AVM, original_data=rgb)
    save_rgb(rgb_scaled, f'{png_path}/SgrB2_RGB_2550-770-480_alma.png', avm=AVM, alma_data=alma_sgrb2_reprojected_jwst, alma_level=alma_level, original_data=rgb)


    ratio_images = {
        '770d2550': fill_nan(fits.getdata(repr_image_filenames['f770w'])) / fill_nan(fits.getdata(repr_image_filenames['f2550w'])),
        '1280d2550': fill_nan(fits.getdata(repr_image_filenames['f1280w'])) / fill_nan(fits.getdata(repr_image_filenames['f2550w'])),
        '405410d480': fill_nan(fits.getdata(repr_image_sub_filenames['f405n-f410m'])) / fill_nan(fits.getdata(repr_image_filenames['f480m'])),
        '480d2550': fill_nan(fits.getdata(repr_image_filenames['f480m'])) / fill_nan(fits.getdata(repr_image_filenames['f2550w'])),
        '360d480': fill_nan(fits.getdata(repr_image_filenames['f360m'])) / fill_nan(fits.getdata(repr_image_filenames['f480m'])),
    }
    for key, value in ratio_images.items():
        # write out the ratio images so we can look at them in CARTA
        writekey = key.replace('d', '_over_')
        fits.PrimaryHDU(data=value, header=tgt_header).writeto(f'{new_basepath}/{writekey}_reprj_{target_filter}.fits', overwrite=True)


    rgb = np.array([
        fits.getdata(repr_image_filenames['f2550w']),
        fits.getdata(repr_image_filenames['f770w']),
        fits.getdata(repr_image_sub_filenames['f405n-f410m'])
    ]).swapaxes(0,2).swapaxes(0,1)

    rgb_scaled = np.array([
        simple_norm(rgb[:,:,0], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,0]),
        simple_norm(rgb[:,:,1], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,1]),
        simple_norm(rgb[:,:,2], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,2])
    ]).swapaxes(0,2).swapaxes(0,1)

    save_rgb(rgb_scaled, f'{png_path}/SgrB2_RGB_2550-770-405410.png', avm=AVM, original_data=rgb)
    save_rgb(rgb_scaled, f'{png_path}/SgrB2_RGB_2550-770-405410_alma.png', avm=AVM, alma_data=alma_sgrb2_reprojected_jwst, alma_level=alma_level, original_data=rgb)

    rgb = np.array([
        ratio_images['770d2550'],
        ratio_images['1280d2550'],
        ratio_images['405410d480'],
    ]).swapaxes(0,2).swapaxes(0,1)
    rgb_scaled = np.array([
        simple_norm(rgb[:,:,0], stretch='asinh', min_percent=0.5, max_percent=99.95)(rgb[:,:,0]),
        simple_norm(rgb[:,:,1], stretch='asinh', min_percent=2, max_percent=99.995)(rgb[:,:,1]),
        simple_norm(rgb[:,:,2], stretch='asinh', min_percent=0.5, max_percent=99.5)(rgb[:,:,2])
    ]).swapaxes(0,2).swapaxes(0,1)

    save_rgb(rgb_scaled, f'{png_path}/SgrB2_RGB_770d2550-1280d2550-405410d480.png', avm=AVM, original_data=rgb)
    save_rgb(rgb_scaled, f'{png_path}/SgrB2_RGB_770d2550-1280d2550-405410d480_alma.png', avm=AVM, alma_data=alma_sgrb2_reprojected_jwst, alma_level=alma_level, original_data=rgb)

    rgb = np.array([
        ratio_images['770d2550'],
        ratio_images['480d2550'],
        ratio_images['405410d480'],
    ]).swapaxes(0,2).swapaxes(0,1)
    rgb_scaled = np.array([
        simple_norm(rgb[:,:,0], stretch='asinh', min_percent=0.5, max_percent=99.95)(rgb[:,:,0]),
        simple_norm(rgb[:,:,1], stretch='asinh', min_percent=0.5, max_percent=95)(rgb[:,:,1]),
        simple_norm(rgb[:,:,2], stretch='asinh', min_percent=0.5, max_percent=99.5)(rgb[:,:,2])
    ]).swapaxes(0,2).swapaxes(0,1)

    save_rgb(rgb_scaled, f'{png_path}/SgrB2_RGB_770d2550-480d25550-405410d480.png', avm=AVM, original_data=rgb)
    save_rgb(rgb_scaled, f'{png_path}/SgrB2_RGB_770d2550-480d25550-405410d480_alma.png', avm=AVM, alma_data=alma_sgrb2_reprojected_jwst, alma_level=alma_level, original_data=rgb)

    rgb = np.array([
        ratio_images['770d2550'],
        ratio_images['480d2550'],
        ratio_images['360d480'],
    ]).swapaxes(0,2).swapaxes(0,1)
    rgb_scaled = np.array([
        simple_norm(rgb[:,:,0], stretch='asinh', min_percent=0.5, max_percent=99.5)(rgb[:,:,0]),
        simple_norm(rgb[:,:,1], stretch='asinh', min_percent=0.5, max_percent=99.5)(rgb[:,:,1]),
        simple_norm(rgb[:,:,2], stretch='asinh', min_percent=0.5, max_percent=99.5)(rgb[:,:,2])
    ]).swapaxes(0,2).swapaxes(0,1)

    save_rgb(rgb_scaled, f'{png_path}/SgrB2_RGB_770d2550-480d25550-360d480.png', avm=AVM, original_data=rgb)
    save_rgb(rgb_scaled, f'{png_path}/SgrB2_RGB_770d2550-480d25550-360d480_alma.png', avm=AVM, alma_data=alma_sgrb2_reprojected_jwst, alma_level=alma_level, original_data=rgb)

    rgb = np.array([
        fits.getdata((repr_image_sub_filenames['f480m-f360m'])),
        fits.getdata((repr_image_sub_filenames['f410m-f405n'])),
        fits.getdata((repr_image_sub_filenames['f405n-f410m'])),
    ]).swapaxes(0,2).swapaxes(0,1)

    rgb_scaled = np.array([
        simple_norm(rgb[:,:,0], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,0]),
        simple_norm(rgb[:,:,1], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,1]),
        simple_norm(rgb[:,:,2], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,2])
    ]).swapaxes(0,2).swapaxes(0,1)

    save_rgb(rgb_scaled, f'{png_path}/SgrB2_RGB_480m360-410m405-405m410.png', avm=AVM, original_data=rgb)
    save_rgb(rgb_scaled, f'{png_path}/SgrB2_RGB_480m360-410m405-405m410_alma.png', avm=AVM, alma_data=alma_sgrb2_reprojected_jwst, alma_level=alma_level, original_data=rgb)






def main():
    for target_filter in ('f466n', 'f150w', ):
        make_pngs(target_filter)

if __name__ == '__main__':
    main()
