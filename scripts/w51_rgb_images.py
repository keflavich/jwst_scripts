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




def save_rgb(img, filename, avm=None, flip=-1, alma_data=None, alma_level=None, original_data=None):
    # Continue with the original save_rgb logic
    img = (img*256)
    img[img<0] = 0
    img[img>255] = 255
    img = img.astype('uint8')

    # Create alpha channel for transparency
    alpha = np.ones(img.shape[:2], dtype=np.uint8) * 255  # Start with fully opaque

    if original_data is not None:
        # Make pixels transparent where original data is NaN or very small
        # Check each channel for blank pixels
        for i in range(3):
            if i < original_data.shape[2]:
                blank_mask = (np.isnan(original_data[:,:,i]) |
                             (np.abs(original_data[:,:,i]) < 1e-10))
                alpha[blank_mask] = 0

    # Apply flip to alpha channel to match image
    alpha = alpha[::flip,:]

    # If ALMA data is provided, add contours
    if alma_data is not None and alma_level is not None:
        # Create a binary mask for the contour
        contour_mask = np.zeros_like(alma_data, dtype=bool)
        # Find pixels above the contour level
        contour_mask[alma_data >= alma_level] = True
        # Dilate the mask by 1 pixel to make the contour visible
        from scipy.ndimage import binary_dilation
        contour_mask1 = binary_dilation(contour_mask)
        # bitwise xor: we only want the thin rind around the selected region
        contour_mask = contour_mask1 ^ contour_mask

        # Apply flip to contour mask to match image
        contour_mask = contour_mask[::flip,:]

        # invert the color of pixels in the contour
        for i in range(3):  # For each color channel
            img[contour_mask, i] = 255 - img[contour_mask, i]

    # Create RGBA image for PNG with transparency
    img_rgba = np.dstack((img[::flip,:,:], alpha))
    img_pil = PIL.Image.fromarray(img_rgba, mode='RGBA')
    # empirical: 180 degree rotation required.
    flip_img = img_pil.transpose(Image.ROTATE_180)
    flip_img.save(filename)

    if avm is not None:
        base = os.path.basename(filename)
        dir = os.path.dirname(filename)
        avmname = os.path.join(dir, 'avm_'+base)
        avm.embed(filename, avmname)
        shutil.move(avmname, filename)

    # Save as JPEG without transparency (JPEG doesn't support alpha channel)
    filename_jpg = filename.replace('.png', '.jpg')
    img_rgb = PIL.Image.fromarray(img[::flip,:,:], mode='RGB')
    img_rgb = img_rgb.transpose(Image.ROTATE_180)
    img_rgb.save(filename_jpg, format='JPEG',
                 quality=95,  # High quality setting
                 progressive=True)  # Enable progressive loading

    return img_pil



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
    "f1000w": "/orange/adamginsburg/jwst/w51/F1000W/pipeline/jw06151-o002_t001_miri_f1000w_i2d.fits",
    "f1280w": "/orange/adamginsburg/jwst/w51/F1280W/pipeline/jw06151-o002_t001_miri_f1280w_i2d.fits",
    "f2100w": "/orange/adamginsburg/jwst/w51/F2100W/pipeline/jw06151-o002_t001_miri_f2100w_i2d.fits",
    "f560w": "/orange/adamginsburg/jwst/w51/F560W/pipeline/jw06151-o002_t001_miri_f560w_i2d.fits",
    "f770w": "/orange/adamginsburg/jwst/w51/F770W/pipeline/jw06151-o002_t001_miri_f770w_i2d.fits",
}
image_sub_filenames = {
    "f182m-f187n": "/orange/adamginsburg/jwst/w51/filter_subtractions/f182m_minus_f187n.fits",
    "f187n-f182m": "/orange/adamginsburg/jwst/w51/filter_subtractions/f187n_minus_f182m.fits",
    "f210m-f182m": "/orange/adamginsburg/jwst/w51/filter_subtractions/f210m_minus_f182m_unscaled.fits",
    "f210m-f212n": "/orange/adamginsburg/jwst/w51/filter_subtractions/f210m_minus_f212n.fits",
    "f212n-f210m": "/orange/adamginsburg/jwst/w51/filter_subtractions/f212n_minus_f210m.fits",
    "f405n-f410m": "/orange/adamginsburg/jwst/w51/filter_subtractions/f405n_minus_f410m.fits",
    "f410m-f405n": "/orange/adamginsburg/jwst/w51/filter_subtractions/f410m_minus_f405n.fits",
    "f360m-f182m": "/orange/adamginsburg/jwst/w51/filter_subtractions/f360m_minus_f182m_unscaled.fits",
    "f360m-f405n": "/orange/adamginsburg/jwst/w51/filter_subtractions/f360m_minus_f405n.fits",
    "f405n-f360m": "/orange/adamginsburg/jwst/w51/filter_subtractions/f405n_minus_f360m.fits",
    "f480m-f360m": "/orange/adamginsburg/jwst/w51/filter_subtractions/f480m_minus_f360m_unscaled.fits",
}


def make_pngs(target_filter='f140m', new_basepath = '/orange/adamginsburg/jwst/w51/data_reprojected/'):
    print(f"Making PNGs for {target_filter}")

    png_path = f'/orange/adamginsburg/jwst/w51/pngs_{target_filter[1:-1]}'
    os.makedirs(png_path, exist_ok=True)
    tgt_header = fits.getheader(image_filenames[target_filter], ext=('SCI', 1))
    AVM = pyavm.AVM.from_header(tgt_header)

    repr_image_filenames = {x: y.replace("i2d", f"i2d_reprj_{target_filter[:-1]}") for x,y in image_filenames.items()}
    repr_image_filenames = {x: (new_basepath+os.path.basename(y)) for x,y in repr_image_filenames.items()}
    repr_image_sub_filenames = {x: y.replace(".fits", f"reprj_{target_filter[:-1]}.fits") for x,y in image_sub_filenames.items()}
    repr_image_sub_filenames = {x: (new_basepath+os.path.basename(y)) for x,y in repr_image_sub_filenames.items()}


    alma_w51e2_3mm = '/orange/adamginsburg/w51/2017.1.00293.S/may2021_successful_imaging/w51e2.spw0thru19.14500.robust0.thr0.075mJy.mfs.I.startmod.selfcal7.image.tt0.pbcor.fits'
    alma_w51irs2_3mm = '/orange/adamginsburg/w51/2017.1.00293.S/may2021_successful_imaging/w51n.spw0thru19.14500.robust0.thr0.075mJy.mfs.I.startmod.selfcal7.image.tt0.pbcor.fits'
    alma_level = 8.4e-5

    alma_reproj_fn = f'/orange/adamginsburg/jwst/w51/data_reprojected/alma_w51_reprojected_jwst_{target_filter[:-1]}.fits'
    if os.path.exists(alma_reproj_fn):
        alma_w51_reprojected_jwst = fits.getdata(alma_reproj_fn)
    else:
        print(f"Reprojecting ALMA data to {alma_reproj_fn}")
        fh = fits.open(alma_w51e2_3mm)
        data = fh[0].data.squeeze()
        hdr = WCS(fh[0].header).celestial
        alma_w51e2_3mm_reprojected, footprint_e2 = reproject.reproject_interp((data, hdr), tgt_header)

        fh = fits.open(alma_w51irs2_3mm)
        data = fh[0].data.squeeze()
        hdr = WCS(fh[0].header).celestial
        alma_w51irs2_3mm_reprojected, footprint_irs2 = reproject.reproject_interp((data, hdr), tgt_header)

        alma_w51_reprojected_jwst = ((np.nan_to_num(alma_w51e2_3mm_reprojected) +
                                    np.nan_to_num(alma_w51irs2_3mm_reprojected)) /
                                    (footprint_e2 + footprint_irs2))
        del alma_w51e2_3mm_reprojected, alma_w51irs2_3mm_reprojected, footprint_e2, footprint_irs2
        fits.writeto(alma_reproj_fn, alma_w51_reprojected_jwst, tgt_header, overwrite=True)




    for filtername in image_filenames:
        if not os.path.exists(repr_image_filenames[filtername]):
            print(f"Reprojecting {filtername} {image_filenames[filtername]} to {repr_image_filenames[filtername]}")
            result,_ = reproject.reproject_interp(image_filenames[filtername], tgt_header, hdu_in='SCI')
            hdu = fits.PrimaryHDU(data=result, header=tgt_header)
            hdu.writeto(repr_image_filenames[filtername], overwrite=True)




    rgb = np.array([
        fits.getdata(repr_image_filenames['f210m']),
        fits.getdata(repr_image_filenames['f335m']),
        fits.getdata(repr_image_filenames['f360m'])
    ]).swapaxes(0,2).swapaxes(0,1)


    rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='asinh', min_percent=1, max_percent=99)(rgb[:,:,0]),
                        simple_norm(rgb[:,:,1], stretch='asinh', min_percent=1, max_percent=99)(rgb[:,:,1]),
                        simple_norm(rgb[:,:,2], stretch='asinh', min_percent=1, max_percent=99)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)


    save_rgb(rgb_scaled, f'{png_path}/w51_RGB_210-300-360.png', avm=AVM, original_data=rgb)
    save_rgb(rgb_scaled, f'{png_path}/w51_RGB_210-300-360_alma.png', avm=AVM, alma_data=alma_w51_reprojected_jwst, alma_level=alma_level, original_data=rgb)


    rgb2 = np.array([
        fits.getdata(repr_image_filenames['f210m']),
        fits.getdata(repr_image_filenames['f335m']),
        fits.getdata(repr_image_filenames['f480m'])
    ]).swapaxes(0,2).swapaxes(0,1)


    rgb_scaled2 = np.array([simple_norm(rgb2[:,:,0], stretch='asinh', min_percent=1, max_percent=99)(rgb2[:,:,0]),
                        simple_norm(rgb2[:,:,1], stretch='asinh', min_percent=1, max_percent=99)(rgb2[:,:,1]),
                        simple_norm(rgb2[:,:,2], stretch='asinh', min_percent=1, max_percent=99)(rgb2[:,:,2])]).swapaxes(0,2).swapaxes(0,1)


    rgb = np.array([fits.getdata(repr_image_filenames['f480m']),
                    fits.getdata(repr_image_filenames['f405n']),
                    fits.getdata(repr_image_filenames['f187n'])]).swapaxes(0,2).swapaxes(0,1)
    save_rgb(rgb/np.nanmedian(rgb), f'{png_path}/w51_RGB_480-405-187.png', avm=AVM, original_data=rgb)
    save_rgb(rgb/np.nanmedian(rgb), f'{png_path}/w51_RGB_480-405-187_alma.png', avm=AVM, alma_data=alma_w51_reprojected_jwst, alma_level=alma_level, original_data=rgb)




    rgb = np.array([fits.getdata(repr_image_filenames['f480m']),
                    fits.getdata(repr_image_filenames['f405n']),
                    fits.getdata(repr_image_filenames['f210m'])]).swapaxes(0,2).swapaxes(0,1)
    rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='asinh', min_percent=1, max_percent=97)(rgb[:,:,0]),
                        simple_norm(rgb[:,:,1], stretch='asinh', min_percent=1, max_percent=99)(rgb[:,:,1]),
                        simple_norm(rgb[:,:,2], stretch='asinh', min_percent=1, max_percent=99)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)
    save_rgb(rgb_scaled, f'{png_path}/w51_RGB_480-405-210_scaled.png', avm=AVM, original_data=rgb)
    save_rgb(rgb_scaled, f'{png_path}/w51_RGB_480-405-210_alma.png', avm=AVM, alma_data=alma_w51_reprojected_jwst, alma_level=alma_level, original_data=rgb)



    rgb = np.array([fits.getdata(repr_image_filenames['f480m']),
                    fits.getdata(repr_image_filenames['f405n']),
                    fits.getdata(repr_image_filenames['f187n'])]).swapaxes(0,2).swapaxes(0,1)
    rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='asinh', min_percent=1, max_percent=97)(rgb[:,:,0]),
                        simple_norm(rgb[:,:,1], stretch='asinh', min_percent=1, max_percent=99)(rgb[:,:,1]),
                        simple_norm(rgb[:,:,2], stretch='asinh', min_percent=1, max_percent=99)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)
    save_rgb(rgb_scaled, f'{png_path}/w51_RGB_480-405-187_scaled.png', avm=AVM, original_data=rgb)
    save_rgb(rgb_scaled, f'{png_path}/w51_RGB_480-405-187_alma.png', avm=AVM, alma_data=alma_w51_reprojected_jwst, alma_level=alma_level, original_data=rgb)




    # list of filters from long to short so it's in RGB order
    filternames = sorted(list(image_filenames.keys()),
                        key=lambda x: int(''.join(filter(str.isdigit, x))))[::-1]
    print(f"Sorted list of filters: {filternames}")


    for f1, f2, f3 in zip (filternames, filternames[1:], filternames[2:]):
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
        save_rgb(rgb_scaled, f'{png_path}/w51_RGB_{f1n}-{f2n}-{f3n}.png', avm=AVM, original_data=rgb)
        save_rgb(rgb_scaled, f'{png_path}/w51_RGB_{f1n}-{f2n}-{f3n}_alma.png', avm=AVM, alma_data=alma_w51_reprojected_jwst, alma_level=alma_level, original_data=rgb)

        rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='log', min_percent=1.5, max_percent=99.5)(rgb[:,:,0]),
                            simple_norm(rgb[:,:,1], stretch='log', min_percent=1.5, max_percent=99.5)(rgb[:,:,1]),
                            simple_norm(rgb[:,:,2], stretch='log', min_percent=1.5, max_percent=99.5)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)

        save_rgb(rgb_scaled, f'{png_path}/w51_RGB_{f1n}-{f2n}-{f3n}_log.png', avm=AVM, original_data=rgb)
        save_rgb(rgb_scaled, f'{png_path}/w51_RGB_{f1n}-{f2n}-{f3n}_log_alma.png', avm=AVM, alma_data=alma_w51_reprojected_jwst, alma_level=alma_level, original_data=rgb)




    for filtername in image_sub_filenames:
        if not os.path.exists(repr_image_sub_filenames[filtername]):
            print(f"Reprojecting {filtername} {image_sub_filenames[filtername]} to {repr_image_sub_filenames[filtername]}")
            result,_ = reproject.reproject_interp(image_sub_filenames[filtername], tgt_header, hdu_in='SCI')
            hdu = fits.PrimaryHDU(data=result, header=tgt_header)
            hdu.writeto(repr_image_sub_filenames[filtername], overwrite=True)


    filternames = sorted(list(image_sub_filenames.keys()),
                        key=lambda x: int(''.join(filter(str.isdigit, x))))[::-1]
    print(f"Sorted list of subtracted-filters: {filternames}")

    for f1, f2, f3 in zip (filternames, filternames[1:], filternames[2:]):
        print(f1,f2,f3, repr_image_sub_filenames[f1], repr_image_sub_filenames[f2], repr_image_sub_filenames[f3])
        rgb = np.array([
            fits.getdata(repr_image_sub_filenames[f1]),
            fits.getdata(repr_image_sub_filenames[f2]),
            fits.getdata(repr_image_sub_filenames[f3]),
        ]).swapaxes(0,2).swapaxes(0,1)
        rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,0]),
                            simple_norm(rgb[:,:,1], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,1]),
                            simple_norm(rgb[:,:,2], stretch='asinh', min_percent=1, max_percent=99.5)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)

        f1n, f2n, f3n = ''.join(filter(str.isdigit, f1)), ''.join(filter(str.isdigit, f2)), ''.join(filter(str.isdigit, f3))
        save_rgb(rgb_scaled, f'{png_path}/w51_RGB_{f1n}-{f2n}-{f3n}_sub.png', avm=AVM, original_data=rgb)
        save_rgb(rgb_scaled, f'{png_path}/w51_RGB_{f1n}-{f2n}-{f3n}_sub_alma.png', avm=AVM, alma_data=alma_w51_reprojected_jwst, alma_level=alma_level, original_data=rgb)

        rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='log', min_percent=1.0, max_percent=99.5)(rgb[:,:,0]),
                            simple_norm(rgb[:,:,1], stretch='log', min_percent=1.0, max_percent=99.5)(rgb[:,:,1]),
                            simple_norm(rgb[:,:,2], stretch='log', min_percent=1.0, max_percent=99.5)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)

        save_rgb(rgb_scaled, f'{png_path}/w51_RGB_{f1n}-{f2n}-{f3n}_sub_log.png', avm=AVM, original_data=rgb)
        save_rgb(rgb_scaled, f'{png_path}/w51_RGB_{f1n}-{f2n}-{f3n}_sub_log_alma.png', avm=AVM, alma_data=alma_w51_reprojected_jwst, alma_level=alma_level, original_data=rgb)





    # filternames_sub = list(image_sub_filenames.keys())[::-1]
    # for f1, f2, f3 in zip (filternames_sub, filternames_sub[1:], filternames_sub[2:]):
    #     print(f1,f2,f3)
    #     rgb = np.array([
    #         fits.getdata(repr480_image_sub_filenames[f1]),
    #         fits.getdata(repr480_image_sub_filenames[f2]),
    #         fits.getdata(repr480_image_sub_filenames[f3]),
    #     ]).swapaxes(0,2).swapaxes(0,1)
    #     rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='asinh', min_percent=1, max_percent=99)(rgb[:,:,0]),
    #                            simple_norm(rgb[:,:,1], stretch='asinh', min_percent=1, max_percent=99)(rgb[:,:,1]),
    #                            simple_norm(rgb[:,:,2], stretch='asinh', min_percent=1, max_percent=99)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)
    #
    #     save_rgb(rgb_scaled, f'{png_path}/w51_RGB_{f1}_{f2}_{f3}.png', avm=AVM)








    # rgb = np.array([fits.getdata(repr480_image_filenames['f480m']),
    #                 fits.getdata(repr480_image_sub_filenames['f405n-f410m']),
    #                 fits.getdata(repr480_image_sub_filenames['f187n-f182m'])]).swapaxes(0,2).swapaxes(0,1)
    # save_rgb(rgb/np.nanmedian(rgb), f'{png_path}/w51_RGB_480-405m410-187m182.png', avm=AVM)
    #



    # rgb = np.array([fits.getdata(repr480_image_filenames['f480m']),
    #                 fits.getdata(repr480_image_sub_filenames['f405n-f410m']),
    #                 fits.getdata(repr480_image_sub_filenames['f187n-f182m'])]).swapaxes(0,2).swapaxes(0,1)
    # rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='asinh', min_percent=1, max_percent=97)(rgb[:,:,0]),
    #                        simple_norm(rgb[:,:,1], stretch='asinh', min_percent=1, max_percent=99)(rgb[:,:,1]),
    #                        simple_norm(rgb[:,:,2], stretch='asinh', min_percent=1, max_percent=99)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)
    # save_rgb(rgb_scaled, f'{png_path}/w51_RGB_480-405m410-187m182_scaled.png', avm=AVM)

    print(repr_image_filenames['f480m'], repr_image_sub_filenames['f405n-f410m'], repr_image_sub_filenames['f187n-f182m'])
    rgb = np.array([fits.getdata(repr_image_filenames['f480m']),
                    fits.getdata(repr_image_sub_filenames['f405n-f410m']),
                    fits.getdata(repr_image_sub_filenames['f187n-f182m'])]).swapaxes(0,2).swapaxes(0,1)
    rgb_scaled = np.array([simple_norm(rgb[:,:,0], stretch='asinh', min_percent=1, max_percent=97)(rgb[:,:,0]),
                        simple_norm(rgb[:,:,1], stretch='asinh', min_percent=1, max_percent=99)(rgb[:,:,1]),
                        simple_norm(rgb[:,:,2], stretch='asinh', min_percent=1, max_percent=99)(rgb[:,:,2])]).swapaxes(0,2).swapaxes(0,1)
    save_rgb(rgb_scaled, f'{png_path}/w51_RGB_480-405m410-187m182_scaled.png', avm=AVM, original_data=rgb)


def main():
    for target_filter in ('f140m', 'f480m'):
        make_pngs(target_filter)

if __name__ == '__main__':
    main()