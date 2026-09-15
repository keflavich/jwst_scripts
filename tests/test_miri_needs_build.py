"""Background-match availability is per field, not per file.

`build_miri_obs` raises on any field missing from the offsets table, so
`miri_needs_build` has to report those as NOMATCH.  Checking only that the
table exists made every uncovered field report "no MIRI png yet", get built,
and land as a FAILED entry on every hourly tick -- noise that buries real
failures, and the reason the bgmatch layer's coverage gap was invisible.
"""
import json
import os
import time

import pytest

G = pytest.importorskip("gc_treasury_rgb_images")

COVERED, UNCOVERED = "o127", "o106"


@pytest.fixture
def env(tmp_path, monkeypatch):
    match = tmp_path / "miri_background_match.json"
    match.write_text(json.dumps(
        {"offsets": {COVERED: 1.25}, "vmin": 40.0, "vmax": 216.0}))
    monkeypatch.setattr(G, "MIRI_MATCH_JSON", str(match))
    monkeypatch.setattr(G, "load_miri_match",
                        lambda: json.loads(match.read_text()))
    # a source older than the settle window, so SETTLING never masks the answer
    src = tmp_path / "src_i2d.fits"
    src.write_bytes(b"0")
    old = time.time() - (G.SETTLE_SECONDS + 600)
    os.utime(src, (old, old))
    # no pngs anywhere, so the only thing that can differ is the match check
    monkeypatch.setattr(G, "miri_png_for",
                        lambda obs, bg=False: str(tmp_path / f"{obs}_{bg}.png"))
    monkeypatch.setattr(G, "miri_hips_for",
                        lambda obs, bg=False: str(tmp_path / f"{obs}_{bg}_hips"))
    return str(src), match


def test_field_absent_from_the_offsets_table_is_nomatch(env):
    src, _ = env
    assert G.miri_needs_build(UNCOVERED, src, bgmatch=True) == "NOMATCH"


def test_field_present_in_the_offsets_table_is_buildable(env):
    src, _ = env
    assert G.miri_needs_build(COVERED, src, bgmatch=True) == "no MIRI png yet"


def test_missing_table_is_still_nomatch(env, monkeypatch, tmp_path):
    src, match = env
    monkeypatch.setattr(G, "MIRI_MATCH_JSON", str(tmp_path / "absent.json"))
    assert G.miri_needs_build(COVERED, src, bgmatch=True) == "NOMATCH"


def test_plain_miri_does_not_consult_the_table(env):
    """Only the background-matched flavour needs offsets."""
    src, _ = env
    assert G.miri_needs_build(UNCOVERED, src, bgmatch=False) == "no MIRI png yet"


def test_pending_summary_excludes_unbuildable_bgmatch(env, monkeypatch):
    """The held-lock report must not list work that would fail if it ran."""
    src, _ = env
    monkeypatch.setattr(G, "inventory", lambda: ({f: {} for f in G.FILTERS}, []))
    monkeypatch.setattr(G, "find_i2d", lambda filt: {UNCOVERED: src, COVERED: src})
    lines = G._pending_summary()
    assert not any(f"{UNCOVERED} MIRI+bg" in ln for ln in lines), lines
    assert any(f"{COVERED} MIRI+bg" in ln for ln in lines), lines


# --- the background match has to keep up with the survey -------------------

def _write_match(tmp_path, monkeypatch, **payload):
    p = tmp_path / "miri_background_match.json"
    p.write_text(json.dumps(payload))
    monkeypatch.setattr(G, "MIRI_MATCH_JSON", str(p))
    return str(p)


def test_a_new_field_makes_the_match_stale(tmp_path, monkeypatch):
    """The failure this guards: 12 fields solved, 22 landed, NOMATCH forever.

    miri_background_match.json covered the 12 fields that existed when
    --miri-match was last run by hand.  Every later field reported NOMATCH and
    --auto skipped it on every tick, so the background-matched mosaic stopped
    tracking the survey two days before anyone noticed.
    """
    _write_match(tmp_path, monkeypatch,
                 offsets={"o127": 0.0, "o128": 1.0}, fields=["o127", "o128"])
    why = G.miri_match_is_stale({"o127": "a", "o128": "b", "o129": "c"})
    assert why and "o129" in why


def test_a_covered_set_is_not_stale(tmp_path, monkeypatch):
    _write_match(tmp_path, monkeypatch,
                 offsets={"o127": 0.0, "o128": 1.0}, fields=["o127", "o128"])
    assert G.miri_match_is_stale({"o127": "a", "o128": "b"}) is None


def test_an_unmatchable_field_does_not_force_a_resolve(tmp_path, monkeypatch):
    """A field with no overlapping neighbour is covered but has no offset.

    Gating on the offsets instead of on the field set would re-run the whole
    pairwise measurement every tick for as long as such a field exists.
    """
    _write_match(tmp_path, monkeypatch,
                 offsets={"o127": 0.0},                 # o128 got none
                 fields=["o127", "o128"])
    assert G.miri_match_is_stale({"o127": "a", "o128": "b"}) is None


def test_a_match_without_a_field_list_is_refreshed_once(tmp_path, monkeypatch):
    """Files written before the field set was recorded."""
    _write_match(tmp_path, monkeypatch, offsets={"o127": 0.0})
    why = G.miri_match_is_stale({"o127": "a"})
    assert why and "predates" in why


def test_a_missing_match_is_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "MIRI_MATCH_JSON", str(tmp_path / "absent.json"))
    assert G.miri_match_is_stale({"o127": "a"})


def test_the_publisher_ships_both_miri_coadds():
    """Neither was in cmd_publish's source list, so rebuilds never shipped."""
    import inspect
    src = inspect.getsource(G.cmd_publish)
    assert "MIRI_COADD_NAME" in src
    assert "MIRI_BGMATCH_COADD_NAME" in src
