"""Star selection in check_hips_astrometry, driven through its caller.

A field the reference catalogue does not cover selects no stars, and the empty
list then reached SkyCoord([p.ra ...], [p.dec ...]) and failed there with
"Longitude instances require units equivalent to 'rad'" -- astropy has no
element to infer a unit from.  A non-empty selection was fine, so the defect
was the missing coverage check rather than the rebuild.  The main() tests
reproduce the o040 condition, where gaia_virac2_refcat covers none of the
footprint, instead of hand-making an empty list.
"""
import os

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
import astropy.units as u

C = pytest.importorskip("check_hips_astrometry")


def _grid(n=6, sep_arcsec=30.0):
    """n stars in a row, far enough apart to survive the min-separation cut."""
    step = sep_arcsec / 3600.0
    return SkyCoord([266.4 + i * step for i in range(n)] * u.deg,
                    [-29.0] * n * u.deg)


def test_spread_out_returns_a_usable_skycoord():
    out = C.spread_out(_grid(), 4)
    assert isinstance(out, SkyCoord)
    assert len(out) == 4
    # the bug produced no object at all, so the unit is the assertion that
    # matters: a Longitude that kept its degrees round-trips to itself
    assert np.allclose(out.ra.deg[:4], _grid().ra.deg[:4])
    assert out.ra.unit.is_equivalent(u.deg)


def test_spread_out_drops_stars_closer_than_the_minimum_separation():
    # six stars 1" apart span 5", so every one of them sits inside the 8"
    # circle around the first and only that first star survives
    out = C.spread_out(_grid(n=6, sep_arcsec=1.0), 6, min_sep_arcsec=8.0)
    assert len(out) == 1


def test_spread_out_rejects_an_empty_selection():
    # ValueError, not SystemExit: the helper stays usable from other code
    with pytest.raises(ValueError, match="no catalogue stars"):
        C.spread_out(SkyCoord([] * u.deg, [] * u.deg), 4)


def _catalog(path, coords):
    from astropy.table import Table
    Table({"RA": coords.ra.deg, "DEC": coords.dec.deg,
           "mag": np.arange(len(coords), dtype=float)}).write(path)


def test_main_selects_stars_without_a_unit_error(tmp_path, monkeypatch, capsys):
    """The call site: --catalog + --hips, no --fits, must reach measurement.

    measure() is stubbed because real tiles are not the subject here; what is
    being asserted is that selection hands it a SkyCoord rather than raising.
    """
    cat = tmp_path / "cat.fits"
    _catalog(str(cat), _grid())
    seen = {}

    def fake_measure(hips_dir, stars, pixscale, half_px, flip_y):
        seen["n"] = len(stars)
        seen["unit"] = stars.ra.unit
        return [{"ok": False, "reason": "stubbed"}]

    monkeypatch.setattr(C, "measure", fake_measure)
    monkeypatch.setattr("sys.argv",
                        ["check_hips_astrometry.py",
                         "--hips", str(tmp_path / "some_hips"),
                         "--catalog", str(cat), "--nstars", "3"])
    C.main()
    assert seen["n"] == 3
    assert seen["unit"].is_equivalent(u.deg)
    assert "no stars measurable" in capsys.readouterr().out


def test_main_says_so_when_the_catalog_misses_the_field(tmp_path, monkeypatch):
    """A catalogue covering none of the footprint is the o040 case.

    gaia_virac2_refcat covers the Brick and no Treasury field; before this the
    empty selection surfaced as a unit error and the coverage gap was never
    named.  main() converts the helper's ValueError into an exit.
    """
    from astropy.io import fits
    from astropy.wcs import WCS

    cat = tmp_path / "cat.fits"
    _catalog(str(cat), _grid())            # stars at RA 266.4, Dec -29

    w = WCS(naxis=2)                       # a field 40 degrees away
    w.wcs.crpix = [50, 50]
    w.wcs.cdelt = [-1 / 3600, 1 / 3600]
    w.wcs.crval = [226.4, -29.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    hdr = w.to_header()
    hdr["EXTNAME"] = "SCI"
    ref = tmp_path / "ref.fits"
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(np.zeros((100, 100)), hdr)]).writeto(ref)

    monkeypatch.setattr("sys.argv",
                        ["check_hips_astrometry.py",
                         "--hips", str(tmp_path / "some_hips"),
                         "--catalog", str(cat), "--fits", str(ref)])
    with pytest.raises(SystemExit, match="no catalogue stars"):
        C.main()


def test_repo_has_no_other_unitless_skycoord_rebuild():
    """The same construction elsewhere would fail the same way, silently."""
    import re
    # a bare attribute -- .ra rather than .ra.deg -- is the unitless form;
    # the fixed call reads SkyCoord([p.ra.deg ...] * u.deg, ...) and passes
    pat = re.compile(r"SkyCoord\(\s*\[\s*\w+\.(?:ra|dec)(?!\.)")
    root = os.path.dirname(os.path.dirname(os.path.abspath(C.__file__)))
    bad = []
    for dirpath, _, names in os.walk(os.path.join(root, "scripts")):
        for n in names:
            if not n.endswith(".py"):
                continue
            p = os.path.join(dirpath, n)
            for i, line in enumerate(open(p), 1):
                # the fix's own comment quotes the broken call; scan code only
                code = line.split("#", 1)[0]
                if pat.search(code):
                    bad.append(f"{p}:{i}")
    assert not bad, f"unitless SkyCoord rebuild(s): {bad}"
