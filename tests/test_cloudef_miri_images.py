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
