"""Unit coverage for verify_field_astrometry's pure helpers.

The end-to-end measure()/check() path needs real HiPS tiles and FITS mosaics
and is exercised by hand against the built layers (see the cloudef MIRI PR
description); this covers the parts that don't.
"""
import numpy as np
import pytest

V = pytest.importorskip("verify_field_astrometry")


def test_prep_returns_none_for_mostly_empty_array():
    a = np.zeros((50, 50))
    a[0, 0] = 1.0  # far fewer than 200 finite positive pixels
    assert V.prep(a) is None


def test_prep_normalizes_a_real_array():
    rng = np.random.default_rng(0)
    a = rng.uniform(1, 10, size=(64, 64))
    out = V.prep(a)
    assert out is not None
    assert np.isclose(out.mean(), 0, atol=1e-6)
    assert np.isclose(out.std(), 1, atol=1e-6)


def test_check_reports_not_measurable_below_the_correlation_floor(monkeypatch):
    monkeypatch.setattr(V, "measure", lambda hips, src, **k: (0.05, 0.1, None))
    results, ok = V.check("dummy_hips", {"o004": "dummy.fits"})
    assert results["o004"] is None
    assert ok  # a floored measurement can't fail the offset check


def test_check_flags_offsets_beyond_target(monkeypatch):
    monkeypatch.setattr(V, "measure", lambda hips, src, **k: (0.9, 0.8, None))
    results, ok = V.check("dummy_hips", {"o004": "dummy.fits"})
    assert results["o004"] == (0.9, 0.8)
    assert not ok


def test_check_passes_for_a_good_measurement(monkeypatch):
    monkeypatch.setattr(V, "measure", lambda hips, src, **k: (0.05, 0.9, None))
    results, ok = V.check("dummy_hips", {"o004": "dummy.fits"})
    assert results["o004"] == (0.05, 0.9)
    assert ok


def _write_tile(hips_dir, order, ipix, value):
    """A minimal single-tile HiPS directory entry, grayscale PNG."""
    import os
    from PIL import Image
    d = (ipix // 10000) * 10000
    tile_dir = os.path.join(hips_dir, f"Norder{order}", f"Dir{d}")
    os.makedirs(tile_dir, exist_ok=True)
    arr = np.full((512, 512), value, dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(os.path.join(tile_dir, f"Npix{ipix}.png"))


def _tile_index(order, coord):
    from astropy_healpix import HEALPix
    hp = HEALPix(nside=2 ** (order + 9), order="nested", frame="galactic")
    ipix_hi = int(hp.skycoord_to_healpix(coord))
    return ipix_hi >> 18


def test_sample_with_fallback_climbs_to_a_shallower_populated_order(tmp_path):
    """A coadd can report a deep global hips_order (from some OTHER field)
    while this field's own layer only ever reached a shallower one -- the
    real bug found while verifying jwst_miri_hips against cloudef, whose
    MIRI layer tops out at order 12 inside a coadd that reaches order 14
    elsewhere. Sampling must climb down to the order that actually has a
    tile here, not return all-NaN."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    hips = tmp_path / "fake_hips"
    hips.mkdir()
    (hips / "properties").write_text(
        "hips_order = 5\nhips_frame = galactic\n")
    # a directory exists for the deep order (as a real coadd would have,
    # from some other field) but with no tile covering our test coordinate
    (hips / "Norder5").mkdir()
    (hips / "Norder4").mkdir()

    coord = SkyCoord(l=1.0 * u.deg, b=1.0 * u.deg, frame="galactic")
    ipix3 = _tile_index(3, coord)
    _write_tile(str(hips), 3, ipix3, 200)

    coords = SkyCoord(l=[1.0] * u.deg, b=[1.0] * u.deg, frame="galactic")
    vals, order_used = V._sample_with_fallback(str(hips), coords, min_order=1)
    assert order_used == 3
    assert np.isfinite(vals).all()
    assert vals[0] == 200


def test_sample_with_fallback_gives_up_below_min_order(tmp_path):
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    hips = tmp_path / "empty_hips"
    hips.mkdir()
    (hips / "properties").write_text(
        "hips_order = 4\nhips_frame = galactic\n")
    (hips / "Norder4").mkdir()

    coords = SkyCoord(l=[1.0] * u.deg, b=[1.0] * u.deg, frame="galactic")
    vals, order_used = V._sample_with_fallback(str(hips), coords, min_order=2)
    assert order_used == 2
    assert not np.isfinite(vals).any()
