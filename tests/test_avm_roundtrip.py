"""Round-trip the AVM a PNG carries, through the same path reproject uses.

This pins the fix for a ~1.3" astrometric offset in every HiPS built from a
`save_rgb` PNG.  The failure mode is a served product that looks completely
right and sits an arcsecond off the sky, so nothing about the image itself
flags it -- only the reconstructed WCS does.

The check is deliberately self-contained: it synthesises a header with an
off-centre CRPIX, writes a PNG exactly the way `save_rgb` writes one, embeds an
AVM, reads it back through `reproject.utils.parse_input_data`, and compares the
sky position each pixel index actually holds against the truth.  No data files.
"""
import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image

pyavm = pytest.importorskip("pyavm")
parse_input_data = pytest.importorskip("reproject.utils").parse_input_data

from jwst_rgb.save_rgb import (  # noqa: E402
    avm_for_saved_png, faithful_avm, _net_flip, _reader_flip,
)

NX, NY = 320, 240
PA_DEG = 73.0
PIXSCALE = 0.031 / 3600.0
# deliberately nowhere near the centre: the error is |N+1-2*crpix| per axis,
# so a centred CRPIX hides it entirely (which is why this went unnoticed)
CRPIX = (199.4, 128.4)


def truth_wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [266.5, -28.7]
    w.wcs.crpix = list(CRPIX)
    t = np.radians(PA_DEG)
    w.wcs.cd = PIXSCALE * np.array([[-np.cos(t), np.sin(t)],
                                    [np.sin(t), np.cos(t)]])
    w.pixel_shape = (NX, NY)
    return w


def write_png_like_save_rgb(path, arr, avm, flip=-1, transpose=Image.ROTATE_180):
    """The pixel path of save_rgb: img[::flip], then the PIL transpose."""
    rgba = np.dstack([arr[::flip, :]] * 3 + [np.full(arr.shape, 255, np.uint8)])
    im = Image.fromarray(rgba.astype(np.uint8), mode="RGBA")
    if transpose is not None:
        im = im.transpose(transpose)
    im.save(path)
    tmp = str(path) + ".avm"
    avm.embed(str(path), tmp)
    import os
    import shutil
    shutil.move(tmp, str(path))
    assert os.path.exists(path)


def worst_offset(png, wcs_true, flip=-1, transpose=Image.ROTATE_180):
    """Max separation, in arcsec, between where the reconstructed WCS puts a
    pixel and where that pixel's value actually came from."""
    data, wcs_read = parse_input_data(str(png))
    ny, nx = data.shape[-2:]
    # which input index each reconstructed index holds, for this flip/transpose
    fr, fc = _reader_flip(flip, transpose)
    j, i = np.array([0, 0, ny - 1, ny - 1, ny // 2]), np.array([0, nx - 1, 0, nx - 1, nx // 2])
    src_j = (ny - 1 - j) if fr else j
    src_i = (nx - 1 - i) if fc else i
    got = wcs_read.pixel_to_world(i, j)
    want = wcs_true.pixel_to_world(src_i, src_j)
    return got.separation(want).arcsec


@pytest.fixture
def frame(tmp_path):
    rng = np.random.default_rng(1234)
    arr = rng.integers(0, 255, size=(NY, NX), dtype=np.uint8)
    return tmp_path, arr, truth_wcs()


def test_avm_for_saved_png_round_trips(frame):
    tmp_path, arr, w = frame
    png = tmp_path / "good.png"
    write_png_like_save_rgb(png, arr, avm_for_saved_png(w, NY, NX))
    assert np.max(worst_offset(png, w)) < 1e-3


def test_raw_header_avm_is_a_pure_translation_of_the_predicted_size(frame):
    """The bug: CRPIX is never reflected, so the layer sits
    |N+1-2*crpix| pixels off per axis -- the same everywhere, which is why it
    reads as a plausible image rather than a broken one."""
    tmp_path, arr, w = frame
    png = tmp_path / "raw.png"
    hdr = w.to_header()
    hdr["NAXIS"], hdr["NAXIS1"], hdr["NAXIS2"] = 2, NX, NY
    write_png_like_save_rgb(png, arr, pyavm.AVM.from_header(fits.Header(hdr)))
    seps = worst_offset(png, w)
    dx = NX + 1 - 2 * CRPIX[0]
    dy = NY + 1 - 2 * CRPIX[1]
    predicted = np.hypot(dx, dy) * PIXSCALE * 3600
    assert predicted > 2.0, "test frame no longer exercises the bug"
    assert np.allclose(seps, predicted, atol=0.01), (seps, predicted)


def test_centred_crpix_hides_the_bug(frame):
    """Why it went unnoticed: with CRPIX at the image centre the reflection is
    a no-op, so the hand-built mosaics were unaffected."""
    tmp_path, arr, _ = frame
    w = truth_wcs()
    w.wcs.crpix = [(NX + 1) / 2, (NY + 1) / 2]
    png = tmp_path / "centred.png"
    hdr = w.to_header()
    hdr["NAXIS"], hdr["NAXIS1"], hdr["NAXIS2"] = 2, NX, NY
    write_png_like_save_rgb(png, arr, pyavm.AVM.from_header(fits.Header(hdr)))
    assert np.max(worst_offset(png, w)) < 1e-3


def test_faithful_avm_is_a_rotation_not_a_translation(frame):
    """faithful_avm stores the true CD *and* the FITS CRPIX, defeating the
    accidental CD negation that leaves the raw AVM merely translated: the
    result is a 180 degree rotation about the centre, worst at the corners and
    near zero in the middle."""
    tmp_path, arr, w = frame
    png = tmp_path / "faithful.png"
    hdr = w.to_header()
    hdr["NAXIS"], hdr["NAXIS1"], hdr["NAXIS2"] = 2, NX, NY
    write_png_like_save_rgb(png, arr, faithful_avm(fits.Header(hdr), shape=(NY, NX)))
    seps = worst_offset(png, w)
    diag = np.hypot(NX, NY) * PIXSCALE * 3600
    assert seps[:4].min() > 0.5 * diag, seps      # corners: order the diagonal
    assert seps[4] < 0.2                          # centre: nearly right


@pytest.mark.parametrize("flip", [1, -1])
@pytest.mark.parametrize("transpose", [None, Image.ROTATE_180])
def test_round_trips_for_every_flip_transpose(frame, flip, transpose):
    """Nothing in the pipeline uses the three non-default combinations today,
    so nothing else would catch them going wrong."""
    tmp_path, arr, w = frame
    png = tmp_path / f"c_{flip}_{transpose}.png"
    avm = avm_for_saved_png(w, NY, NX, flip=flip, transpose=transpose)
    write_png_like_save_rgb(png, arr, avm, flip=flip, transpose=transpose)
    assert np.max(worst_offset(png, w, flip=flip, transpose=transpose)) < 1e-3


def test_reader_flip_is_net_flip_plus_the_pil_row_convention():
    for flip in (1, -1):
        for transpose in (None, Image.ROTATE_180):
            nr, nc = _net_flip(flip, transpose)
            rr, rc = _reader_flip(flip, transpose)
            assert rc == nc
            assert rr == (not nr)


def test_unsupported_flip_and_transpose_are_rejected():
    w = truth_wcs()
    with pytest.raises(ValueError):
        avm_for_saved_png(w, NY, NX, flip=0)
    with pytest.raises(ValueError):
        avm_for_saved_png(w, NY, NX, transpose=Image.ROTATE_90)
