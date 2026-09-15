"""Field classification for cloudef's MIRI parallel (program 2092).

The cloudef/ and cloudef_controlfield/ directory names do NOT reliably say
which field a MIRI mosaic belongs to: all MIRI data lives under cloudef/, but
one of its three observations (o006) actually points at the control field.
classify_field is the safety net for that -- it must go by the mosaic's own
pointing, and it must refuse to guess for a pointing that isn't close to
either known field centre.
"""
import pytest

C = pytest.importorskip("cloudef_miri_images")


def test_main_field_pointing_classifies_as_cloudef():
    l, b = C.FIELD_CENTERS["cloudef"]
    assert C.classify_field(l + 0.005, b + 0.003) == "cloudef"


def test_control_field_pointing_classifies_as_control():
    l, b = C.FIELD_CENTERS["cloudef_control"]
    assert C.classify_field(l - 0.004, b + 0.002) == "cloudef_control"


def test_second_main_field_tile_still_classifies_as_cloudef():
    """o008 sits ~0.04 deg from the main field centre -- well inside
    FIELD_TOL_DEG but far enough that a tight tolerance would wrongly reject
    it."""
    l, b = C.FIELD_CENTERS["cloudef"]
    assert C.classify_field(l + 0.036, b + 0.017) == "cloudef"


def test_pointing_far_from_both_centres_raises():
    with pytest.raises(ValueError):
        C.classify_field(10.0, 10.0)


def test_field_centres_are_far_enough_apart_for_the_tolerance():
    """classify_field assigns to the NEAREST centre and only rejects a
    pointing farther than FIELD_TOL_DEG from it, so what matters is that the
    two centres are farther apart than the tolerance -- not that their
    tolerance radii avoid overlapping. If this ever fails the two fields are
    too close together for FIELD_TOL_DEG to mean anything."""
    import numpy as np
    (l1, b1), (l2, b2) = C.FIELD_CENTERS.values()
    sep = np.hypot(l1 - l2, b1 - b2)
    assert sep > C.FIELD_TOL_DEG


def test_exposure_level_products_are_excluded_from_i2d_matching():
    assert C._PLAIN.match("jw02092-o004_t001_miri_f770w_i2d.fits")
    assert C._DATA.match(
        "jw02092-o006_t001_miri_clear-f770w-mirimage_data_i2d.fits")
    assert not C._PLAIN.match(
        "jw02092004004_02101_00001_mirimage_i2d.fits")
    assert not C._DATA.match(
        "jw02092-o008_t001_miri_f2100w_cat.ecsv")


# --- the two gaps a reviewer found by mutating the module -------------------

REAL_MOSAICS = {
    # (obs, field it actually sits on).  o006 lives in the cloudef/ directory
    # tree and points at the CONTROL field; that mismatch is the reason
    # classify_field exists, so it is the thing worth pinning.
    "o004": "cloudef",
    "o006": "cloudef_control",
    "o008": "cloudef",
}


def _mosaic(obs):
    import glob
    hits = glob.glob(
        f"/orange/adamginsburg/jwst/cloudef*/F770W/pipeline/"
        f"jw02092-{obs}_t*_miri_f770w_i2d.fits")
    return hits[0] if hits else None


@pytest.mark.parametrize("obs,field", sorted(REAL_MOSAICS.items()))
def test_classify_field_on_a_real_header(obs, field):
    """Pin which name goes with which sky position.

    Swapping the two FIELD_CENTERS entries left the whole suite passing: the
    centres are ~0.20 deg apart and FIELD_TOL_DEG is 0.1, so each sits inside
    the other's tolerance band and nothing held the mapping.  Reading a real
    header ties the names to the sky rather than to each other.
    """
    from astropy.io import fits
    from astropy.wcs import WCS

    path = _mosaic(obs)
    if path is None:
        pytest.skip(f"no F770W mosaic on disk for {obs}")
    with fits.open(path) as hl:
        hdu = next(h for h in hl if h.data is not None and h.data.ndim == 2)
        w = WCS(hdu.header).celestial
        ny, nx = hdu.data.shape
    c = w.pixel_to_world(nx / 2, ny / 2).galactic
    assert C.classify_field(c.l.deg, c.b.deg) == field


def test_measured_pointings_classify_without_the_cluster_tree():
    """The same pin as the real-header tests, with nothing on disk.

    Those tests skip on a machine without /orange, and this repo has no CI, so
    a checkout without the cluster tree is the normal case for anyone else on
    the branch -- the FIELD_CENTERS swap survives there unless something holds
    it data-free.

    The literals are the measured pointings recorded in the module docstring,
    written out rather than read back from FIELD_CENTERS: a test that takes a
    field's coordinates out of the table and asks which key they belong to is
    self-consistent under any permutation of that table, so it cannot see the
    swap.  o006 is the one carrying the point -- it sits in the cloudef/
    directory tree and points at the control field.
    """
    assert C.classify_field(0.4901, 0.0104) == "cloudef"          # o004
    assert C.classify_field(0.5213, 0.0243) == "cloudef"          # o008
    assert C.classify_field(0.4182, 0.1927) == "cloudef_control"  # o006


def test_the_saved_png_avm_builder_is_pinned():
    """save_rgb writes the PNG rotated 180 relative to the FITS array.

    avm_for_saved_png reflects CRPIX on both axes to describe the PNG as
    written; faithful_avm does not, and substituting it puts every tile
    |N + 1 - 2*crpix| pixels off -- 1.26" for one measured field, and 3.8" for
    two archival MIRI layers found the same day.  Swapping the import left the
    suite passing, so the distinction is pinned here on the numbers rather
    than on the name.
    """
    import numpy as np
    from astropy.wcs import WCS
    from jwst_rgb.save_rgb import avm_for_saved_png, faithful_avm

    ny, nx = 400, 600
    w = WCS(naxis=2)
    w.wcs.crpix = [137.0, 251.0]          # deliberately off centre
    w.wcs.cdelt = [-1 / 3600, 1 / 3600]
    w.wcs.crval = [266.5, -28.6]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    saved = avm_for_saved_png(w, ny, nx)
    naive = faithful_avm(w, (ny, nx))     # its signature is (wcs, shape)
    sx, sy = saved.Spatial.ReferencePixel
    fx, fy = naive.Spatial.ReferencePixel
    # the reflection is the whole point: x -> nx + 1 - x, y -> ny + 1 - y
    assert sx == pytest.approx(nx + 1 - 137.0, abs=1e-6)
    assert sy == pytest.approx(ny + 1 - 251.0, abs=1e-6)
    assert not (np.isclose(sx, fx) and np.isclose(sy, fy)), \
        "avm_for_saved_png and faithful_avm agree; the builders are not distinct"


def test_the_module_imports_the_saved_png_builder_by_name():
    """The numeric test above shows the two builders differ; this shows which
    one the module actually calls.

    Substituting `from jwst_rgb.save_rgb import faithful_avm as
    avm_for_saved_png` left the suite passing, because nothing exercised the
    import.  Aliasing under the same local name defeats a check on the call
    site, so the check is on what is imported.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(C))
    imported = [
        (n.module, a.name, a.asname)
        for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
        for a in n.names
        if n.module == "jwst_rgb.save_rgb"
    ]
    names = {orig for _, orig, _ in imported}
    assert "avm_for_saved_png" in names, (
        f"module imports {names or 'nothing'} from jwst_rgb.save_rgb; the "
        f"saved-PNG AVM builder must be avm_for_saved_png")
    assert "faithful_avm" not in names, (
        "faithful_avm does not reflect CRPIX for a rot180 PNG; using it here "
        "puts every tile |N + 1 - 2*crpix| pixels off")
