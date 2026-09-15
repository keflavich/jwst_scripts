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
