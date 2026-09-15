"""NIRCam background matching: the per-filter solve and the staleness rules.

The offsets are solved SEPARATELY per filter because this is a colour image.
One offset applied to both would move brightness without fixing colour, and a
wrong relative offset between F480M and F212N tints a whole tile -- more
visible than a brightness seam, not less.  These tests pin that the two
filters stay independent and that the staleness rules learned from the MIRI
bugs carry over.
"""
import json
import os
import time

import numpy as np
import pytest

G = pytest.importorskip("gc_treasury_rgb_images")


def test_solve_offsets_recovers_injected_levels():
    """Least squares on pairwise differences, with the mean-zero constraint."""
    keys = ["o1", "o2", "o3"]
    truth = {"o1": +0.5, "o2": -0.2, "o3": -0.3}      # already mean-zero
    pairs = [(a, b, truth[a] - truth[b], 10000)
             for i, a in enumerate(keys) for b in keys[i + 1:]]
    off = G.miri_solve_offsets(keys, pairs)
    for k in keys:
        assert off[k] == pytest.approx(truth[k], abs=1e-9)


def test_solve_offsets_is_mean_zero():
    """The constraint exists so matching does not drag the mosaic's overall
    level up or down."""
    keys = ["o1", "o2", "o3"]
    pairs = [("o1", "o2", 3.0, 1000), ("o2", "o3", 3.0, 1000)]
    off = G.miri_solve_offsets(keys, pairs)
    assert np.mean([off[k] for k in keys]) == pytest.approx(0.0, abs=1e-9)
    assert off["o1"] - off["o2"] == pytest.approx(3.0, abs=1e-9)
    assert off["o2"] - off["o3"] == pytest.approx(3.0, abs=1e-9)


def test_solve_offsets_leaves_unconnected_fields_at_zero():
    """A field with no overlap has nothing tying it to the rest; zero is the
    honest answer rather than an invented shift."""
    keys = ["o1", "o2", "lonely"]
    pairs = [("o1", "o2", 1.0, 5000)]
    off = G.miri_solve_offsets(keys, pairs)
    assert off["lonely"] == pytest.approx(0.0, abs=1e-6)


def test_solve_offsets_weights_by_shared_area():
    """A pair sharing more sky should dominate a contradictory small one."""
    keys = ["a", "b"]
    big = G.miri_solve_offsets(keys, [("a", "b", 2.0, 1_000_000),
                                      ("a", "b", -2.0, 1)])
    assert big["a"] - big["b"] > 1.9


@pytest.fixture
def match_file(tmp_path, monkeypatch):
    path = tmp_path / "nircam_background_match.json"
    doc = {"filters": {"f480m": {"offsets": {"o127": 1.0, "o128": -1.0},
                                 "vmin": 1.0, "vmax": 10.0},
                       "f212n": {"offsets": {"o127": 0.5, "o128": -0.5},
                                 "vmin": 2.0, "vmax": 20.0}},
           "green": {"vmin": 1.5, "vmax": 15.0}}
    path.write_text(json.dumps(doc))
    monkeypatch.setattr(G, "NIRCAM_MATCH_JSON", str(path))
    return path, doc


def test_green_limits_are_the_mean_of_the_two_filters(match_file):
    """Green is the pixelwise mean of the two filters, so a pixel sitting at
    both filters' vmin must land exactly at the green vmin."""
    _, doc = match_file
    lo = np.mean([doc["filters"][f]["vmin"] for f in ("f480m", "f212n")])
    hi = np.mean([doc["filters"][f]["vmax"] for f in ("f480m", "f212n")])
    assert doc["green"]["vmin"] == pytest.approx(lo)
    assert doc["green"]["vmax"] == pytest.approx(hi)


def test_load_nircam_match_round_trips(match_file):
    path, doc = match_file
    assert G.load_nircam_match() == doc


def test_load_nircam_match_is_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "NIRCAM_MATCH_JSON", str(tmp_path / "absent.json"))
    assert G.load_nircam_match() is None


def _inv(tmp_path, obs, age=10000):
    """Two settled source files for one observation."""
    inv = {}
    for f in G.FILTERS:
        p = tmp_path / f"{obs}_{f}.fits"
        p.write_bytes(b"0")
        old = time.time() - age
        os.utime(p, (old, old))
        inv[f] = {obs: str(p)}
    return inv


def test_needs_build_nomatch_when_field_absent_from_table(
        tmp_path, monkeypatch, match_file):
    """Same rule as MIRI: the table existing is not enough, because build_obs
    raises on a field missing from it."""
    inv = _inv(tmp_path, "o999")
    assert G.needs_build("o999", inv, bgmatch=True) == "NOMATCH"


def test_needs_build_plain_ignores_the_match_table(tmp_path, monkeypatch, match_file):
    """Only the background-matched flavour consults it."""
    inv = _inv(tmp_path, "o999")
    monkeypatch.setattr(G, "png_for", lambda o, bg=False: str(tmp_path / "absent.png"))
    assert G.needs_build("o999", inv, bgmatch=False) == "no RGB yet"


def test_needs_build_flags_a_refreshed_match_as_stale(
        tmp_path, monkeypatch, match_file):
    """A new solution changes the offset and the shared cuts for EVERY tile,
    not only the ones whose data moved, so source mtime alone cannot catch it.
    """
    path, _ = match_file
    inv = _inv(tmp_path, "o127")
    png = tmp_path / "o127.png"
    png.write_bytes(b"0")
    hips = tmp_path / "o127_hips" / "Norder3"
    hips.mkdir(parents=True)
    monkeypatch.setattr(G, "png_for", lambda o, bg=False: str(png))
    monkeypatch.setattr(G, "hips_for", lambda o, bg=False: str(tmp_path / "o127_hips"))

    old = time.time() - 500
    os.utime(png, (old, old))
    os.utime(path, (old - 100, old - 100))
    assert G.needs_build("o127", inv, bgmatch=True) is None

    now = time.time()
    os.utime(path, (now, now))
    assert G.needs_build("o127", inv, bgmatch=True) == \
        "background match is newer than the RGB"


def test_plain_and_bgmatch_have_distinct_names():
    """The two flavours must not collide on disk, and the plain coadd glob
    must not swallow the bgmatch layers."""
    assert G.png_for("o127", False) != G.png_for("o127", True)
    assert G.hips_for("o127", False) != G.hips_for("o127", True)
    assert G.png_for("o127", True).endswith("_bgmatch.png")
    assert "_bgmatch_hips" in G.hips_for("o127", True)
    assert G.COADD_NAME != G.BGMATCH_COADD_NAME
