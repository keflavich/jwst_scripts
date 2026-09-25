"""A mosaic that disappears mid-tick is skipped, not a crash.

Every mosaic path the RGB builder stats came out of a glob, so it existed when
the inventory was taken.  The astrometry checkpoint renames a merged i2d to
``*_im0_badastrom.fits`` when it corrects a tile, and a rename landing between
the glob and the stat left a live path pointing at nothing:

    File "gc_treasury_rgb_images.py", line 992, in needs_build
      if src and time.time() - os.path.getmtime(src) < SETTLE_SECONDS:
    FileNotFoundError: [Errno 2] No such file or directory:
      '.../jw10678-o111_t001_nircam_clear-f212n-merged_i2d.fits'

That killed the whole hourly run on one regenerating tile (2026-09-18
02:59:43, o111).  These pin the two halves of the fix: the tick survives, and
it SAYS which tile it skipped -- a monitor that silently stops rebuilding a
field is the failure this script exists to catch.
"""
import os
import time

import pytest

G = pytest.importorskip("gc_treasury_rgb_images")

OBS = "o111"


@pytest.fixture
def settled(tmp_path):
    """One settled source file per NIRCam filter, and the inventory for them."""
    inv = {}
    old = time.time() - (G.SETTLE_SECONDS + 600)
    for f in G.FILTERS:
        src = tmp_path / f"{f}_{OBS}_i2d.fits"
        src.write_bytes(b"0")
        os.utime(src, (old, old))
        inv[f] = {OBS: str(src)}
    return inv


def test_source_mtime_reports_a_vanished_file_as_none(tmp_path):
    path = tmp_path / "gone_i2d.fits"
    path.write_bytes(b"0")
    assert G.source_mtime(str(path)) == pytest.approx(os.path.getmtime(path))
    path.unlink()
    assert G.source_mtime(str(path)) is None


def test_source_mtime_lets_other_os_errors_through(tmp_path):
    """Only absence is handled; other OS errors still raise.

    Catching every OSError would report a genuinely broken filesystem as "this
    tile is being regenerated", which is the misreport the VANISHED verdict
    exists to avoid.  A path whose parent is a regular file gives ENOTDIR --
    an OSError that is NOT FileNotFoundError.
    """
    parent = tmp_path / "not_a_directory"
    parent.write_bytes(b"0")
    with pytest.raises(NotADirectoryError):
        G.source_mtime(str(parent / "child_i2d.fits"))


def test_needs_build_says_vanished_instead_of_raising(settled, tmp_path):
    os.unlink(settled[G.FILTERS[0]][OBS])
    assert G.needs_build(OBS, settled) == "VANISHED"


def test_needs_build_still_settles_and_builds_when_the_file_is_there(settled,
                                                                    tmp_path,
                                                                    monkeypatch):
    """The guard must not swallow the verdicts that were already correct."""
    monkeypatch.setattr(G, "png_for",
                        lambda obs, stretch=None, residual=False:
                        str(tmp_path / "none.png"))
    monkeypatch.setattr(G, "stretch_match", lambda stretch: False)
    assert G.needs_build(OBS, settled) == "no RGB yet"
    now = time.time()
    os.utime(settled[G.FILTERS[0]][OBS], (now, now))
    assert G.needs_build(OBS, settled) == "SETTLING"


def test_a_source_newer_than_the_rgb_still_rebuilds(settled, tmp_path,
                                                    monkeypatch):
    """The second stat site keeps working on files that are present."""
    png = tmp_path / "rgb.png"
    png.write_bytes(b"0")
    old = time.time() - (G.SETTLE_SECONDS + 1200)
    os.utime(png, (old, old))
    hips = tmp_path / "hips" / "Norder3"
    hips.mkdir(parents=True)
    monkeypatch.setattr(G, "png_for", lambda obs, stretch=None, residual=False: str(png))
    monkeypatch.setattr(G, "hips_for",
                        lambda obs, stretch=None, residual=False:
                        str(tmp_path / "hips"))
    monkeypatch.setattr(G, "stretch_match", lambda stretch: False)
    why = G.needs_build(OBS, settled)
    assert why and "newer than the RGB" in why


def test_the_second_stat_site_also_reports_vanished(settled, tmp_path,
                                                    monkeypatch):
    """A rename can land after the settle check and before the mtime compare."""
    png = tmp_path / "rgb.png"
    png.write_bytes(b"0")
    hips = tmp_path / "hips" / "Norder3"
    hips.mkdir(parents=True)
    monkeypatch.setattr(G, "png_for", lambda obs, stretch=None, residual=False: str(png))
    monkeypatch.setattr(G, "hips_for",
                        lambda obs, stretch=None, residual=False:
                        str(tmp_path / "hips"))
    monkeypatch.setattr(G, "stretch_match", lambda stretch: False)

    real = G.source_mtime
    calls = []

    def vanish_on_the_second_call(path):
        calls.append(path)
        return None if len(calls) > len(G.FILTERS) else real(path)

    monkeypatch.setattr(G, "source_mtime", vanish_on_the_second_call)
    assert G.needs_build(OBS, settled) == "VANISHED"


def test_miri_needs_build_says_vanished(tmp_path):
    src = tmp_path / "miri_i2d.fits"
    src.write_bytes(b"0")
    old = time.time() - (G.SETTLE_SECONDS + 600)
    os.utime(src, (old, old))
    assert G.miri_needs_build(OBS, str(src)) != "VANISHED"
    src.unlink()
    assert G.miri_needs_build(OBS, str(src)) == "VANISHED"


def test_pending_summary_does_not_list_a_vanished_tile_as_work(monkeypatch):
    """VANISHED is not pending work -- nothing is waiting on it.

    Reporting it in the held-lock summary would tell an operator a build is
    queued when the tile is mid-regeneration and will return on a later tick.
    """
    monkeypatch.setattr(G, "inventory",
                        lambda **k: ({f: {OBS: "/nonexistent_i2d.fits"}
                                      for f in G.FILTERS}, [OBS]))
    monkeypatch.setattr(G, "find_i2d", lambda filt, **kw: {})
    monkeypatch.setattr(G, "find_residual_i2d", lambda filt, **kw: {})
    assert G._pending_summary() == []
