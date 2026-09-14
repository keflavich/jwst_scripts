"""The decisions the hourly overlay cron depends on.

No data files and no network: every test drives the discovery, fingerprint and
selection logic against a temp directory, because those four decisions are what
stand between "a new field landed" and "the published overlay is stale for a
week without anyone noticing".
"""
import json
import os
import sys

import numpy as np
import pytest

overlays = pytest.importorskip("gc_treasury_overlays")


def touch(d, name, size=16):
    p = os.path.join(d, name)
    with open(p, "wb") as fh:
        fh.write(b"x" * size)
    return p


def cat(obs, filt, it):
    return f"{filt}_merged_{obs}_indivexp_merged_m{it}_dao_basic_vetted.fits"


@pytest.fixture
def catdir(tmp_path, monkeypatch):
    d = tmp_path / "catalogs"
    d.mkdir()
    monkeypatch.setattr(overlays, "CAT", str(d))
    return d


def test_keeps_the_highest_iteration_per_filter(catdir):
    for it in (1, 2, 3):
        touch(catdir, cat("o127", "f212n", it))
    for it in (1, 4):
        touch(catdir, cat("o127", "f480m", it))
    pairs = overlays.latest_pairs()
    assert set(pairs) == {"o127"}
    assert pairs["o127"]["f212n"][0] == 3
    assert pairs["o127"]["f480m"][0] == 4


def test_drops_observations_vetted_in_only_one_filter(catdir):
    touch(catdir, cat("o127", "f212n", 1))
    touch(catdir, cat("o127", "f480m", 1))
    touch(catdir, cat("o128", "f212n", 1))          # no f480m
    assert set(overlays.latest_pairs()) == {"o127"}


def test_ignores_filenames_that_do_not_match_the_pattern(catdir):
    touch(catdir, cat("o127", "f212n", 1))
    touch(catdir, cat("o127", "f480m", 1))
    touch(catdir, "f212n_merged_o128_dao_basic_vetted.fits")
    touch(catdir, "notes.txt")
    assert set(overlays.latest_pairs()) == {"o127"}


def test_fingerprint_tracks_size_and_mtime(catdir):
    p = touch(catdir, cat("o127", "f212n", 1))
    touch(catdir, cat("o127", "f480m", 1))
    fp0 = overlays.fingerprint(overlays.latest_pairs())

    assert overlays.fingerprint(overlays.latest_pairs()) == fp0   # stable

    with open(p, "ab") as fh:                                     # size
        fh.write(b"more")
    fp1 = overlays.fingerprint(overlays.latest_pairs())
    assert fp1 != fp0

    st = os.stat(p)                                               # mtime (re-vet)
    os.utime(p, (st.st_atime, st.st_mtime + 120))
    assert overlays.fingerprint(overlays.latest_pairs()) != fp1


def test_fingerprint_sees_a_new_field(catdir):
    touch(catdir, cat("o127", "f212n", 1))
    touch(catdir, cat("o127", "f480m", 1))
    fp0 = overlays.fingerprint(overlays.latest_pairs())
    touch(catdir, cat("o128", "f212n", 1))
    touch(catdir, cat("o128", "f480m", 1))
    fp1 = overlays.fingerprint(overlays.latest_pairs())
    assert fp1["n_obs"] == fp0["n_obs"] + 1
    assert fp1 != fp0


def test_auto_is_up_to_date_only_for_the_exact_input_set(catdir, tmp_path, monkeypatch):
    """The --auto decision, as the cron makes it."""
    stamp = tmp_path / "stamp.json"
    monkeypatch.setattr(overlays, "STAMP", str(stamp))
    touch(catdir, cat("o127", "f212n", 1))
    touch(catdir, cat("o127", "f480m", 1))
    fp = overlays.fingerprint(overlays.latest_pairs())
    stamp.write_text(json.dumps({"built": fp, "match": fp}))

    def up_to_date():
        cur = overlays.fingerprint(overlays.latest_pairs())
        return json.loads(stamp.read_text()).get("built") == cur

    assert up_to_date()
    touch(catdir, cat("o128", "f212n", 1))
    touch(catdir, cat("o128", "f480m", 1))
    assert not up_to_date()


def test_rc_masks_split_the_band_at_the_colour_cut():
    S, W, H, SP = (overlays.SLOPE, overlays.WRC, overlays.HW, overlays.SPLIT)
    # place sources on the ridge either side of the split, plus two outside it
    colour = np.array([SP - 0.5, SP + 0.5, SP - 0.5, SP + 0.5])
    m480 = W + S * colour            # exactly on the ridge -> inside the band
    m480 = np.concatenate([m480, [W + S * SP + H + 0.5, 25.0]])
    colour = np.concatenate([colour, [SP, SP]])
    rc, blue, red = overlays.rc_masks(colour, m480)
    assert rc.tolist() == [True, True, True, True, False, False]
    assert blue.tolist() == [True, False, True, False, False, False]
    assert red.tolist() == [False, True, False, True, False, False]
    assert not (blue & red).any()
    assert ((blue | red) == rc).all()


def test_rc_band_excludes_sources_off_the_ridge():
    S, W, H = overlays.SLOPE, overlays.WRC, overlays.HW
    colour = np.array([0.0, 0.0])
    m480 = np.array([W + S * 0.0, W + S * 0.0 + 2 * H])   # on ridge, and off it
    rc, _, _ = overlays.rc_masks(colour, m480)
    assert rc.tolist() == [True, False]


def test_abmag_without_pixscale_fails_by_name():
    from astropy.table import Table
    t = Table({"flux": [1.0, 2.0]})
    with pytest.raises(KeyError, match="PIXSCALE"):
        overlays.abmag(t)


# --------------------------------------------------------------------------
# what gets WRITTEN to the stamp.  The tests above cover the decision made
# *given* a stamp; without these, reverting the partial-build guard to
# `full = True` passes the whole suite, which is how the bug shipped.


def _run_main(monkeypatch, tmp_path, catdir, argv, builders=None):
    """Drive main() with the builders stubbed out, so only the bookkeeping runs."""
    import numpy as np
    stamp = tmp_path / "stamp.json"
    monkeypatch.setattr(overlays, "STAMP", str(stamp))
    monkeypatch.setattr(overlays, "LOCK", str(tmp_path / "lock"))
    monkeypatch.setattr(overlays, "OUT", str(tmp_path))
    called = []
    for name in ("build_red_stars", "build_rc", "build_ultrared"):
        monkeypatch.setattr(overlays, name,
                            lambda *a, _n=name, **k: called.append(_n))
    monkeypatch.setattr(overlays, "publish", lambda *a, **k: None)
    monkeypatch.setattr(overlays, "push_remote", lambda *a, **k: None)
    monkeypatch.setattr(overlays, "report_ridge", lambda *a, **k: None)
    fake = np.zeros(3)
    monkeypatch.setattr(overlays, "load_matched",
                        lambda pairs, force=False: (
                            fake, fake, fake, fake,
                            np.array(["o127"] * 3),
                            overlays.fingerprint(pairs)))
    monkeypatch.setattr(sys, "argv", ["gc_treasury_overlays.py"] + argv)
    rc = overlays.main()
    return rc, stamp, called


def test_partial_build_does_not_claim_the_input_set_as_built(
        catdir, tmp_path, monkeypatch):
    touch(catdir, cat("o127", "f212n", 1))
    touch(catdir, cat("o127", "f480m", 1))
    rc, stamp, called = _run_main(monkeypatch, tmp_path, catdir, ["--only", "red"])
    assert rc == 0
    assert called == ["build_red_stars"]
    written = json.loads(stamp.read_text())
    assert written.get("built") is None, (
        "a partial build recorded the whole input set; the next --auto tick "
        "would report everything up to date and leave rc/ultrared frozen")
    assert written["match"]["n_obs"] == 1      # cache key is still valid


def test_full_build_does_claim_the_input_set_as_built(catdir, tmp_path, monkeypatch):
    touch(catdir, cat("o127", "f212n", 1))
    touch(catdir, cat("o127", "f480m", 1))
    rc, stamp, called = _run_main(monkeypatch, tmp_path, catdir, [])
    assert rc == 0
    assert sorted(called) == ["build_rc", "build_red_stars", "build_ultrared"]
    written = json.loads(stamp.read_text())
    assert written["built"] == overlays.fingerprint(overlays.latest_pairs())


def test_partial_build_preserves_an_existing_built_stamp(
        catdir, tmp_path, monkeypatch):
    """A partial build must not erase the record of the last full one either."""
    touch(catdir, cat("o127", "f212n", 1))
    touch(catdir, cat("o127", "f480m", 1))
    previous = overlays.fingerprint(overlays.latest_pairs())
    stamp = tmp_path / "stamp.json"
    stamp.write_text(json.dumps({"built": previous, "match": previous}))
    rc, stamp, _ = _run_main(monkeypatch, tmp_path, catdir, ["--only", "ultrared"])
    assert rc == 0
    assert json.loads(stamp.read_text())["built"] == previous


def test_auto_rebuilds_after_a_partial_build(catdir, tmp_path, monkeypatch):
    """End to end on the bookkeeping: --only red then --auto must still build."""
    touch(catdir, cat("o127", "f212n", 1))
    touch(catdir, cat("o127", "f480m", 1))
    _run_main(monkeypatch, tmp_path, catdir, ["--only", "red"])
    rc, stamp, called = _run_main(monkeypatch, tmp_path, catdir, ["--auto"])
    assert rc == 0
    assert sorted(called) == ["build_rc", "build_red_stars", "build_ultrared"], (
        "--auto treated a partial build as complete")
