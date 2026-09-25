"""The star-subtracted ("residual") layers built from the DAOPHOT residuals.

They are extra flavours of the same per-tile layers, so they share every name
pattern with the image layers except one token.  cmd_coadd RETIRES a globbed
layer whose key is not in the active inventory, so a residual layer picked up
by an image glob (or the reverse) would be renamed out of the way on the next
tick rather than merely drawn in the wrong mosaic.  These tests pin the names
apart, the stage choice, and the build budget that lets the backlog drain.
"""
import fnmatch
import os
import sys

import pytest

import gc_treasury_rgb_images as G

NIRCAM = "jw10678-{obs}_t001_nircam_clear-{filt}-{mod}_{stage}_daophot_basic_mergedcat_residual_i2d.fits"
MIRI = "jw10678-{obs}_t001_miri_clear-f770w-mirimage_{stage}_daophot_basic_mergedcat_residual_i2d.fits"


def _touch(base, filt, name):
    d = base / filt.upper() / "pipeline"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_bytes(b"0")
    return str(d / name)


@pytest.fixture
def residual_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "BASE", str(tmp_path))
    filt = G.FILTERS[0]
    made = {}
    for obs, mod, stage in [
            ("o101", "merged", "m4"), ("o101", "merged", "resbgsub_m7"),
            ("o102", "merged", "resbgsub_m6"),
            ("o103", "nrca", "resbgsub_m7"), ("o103", "nrcb", "resbgsub_m7"),
            ("o104", "merged", "resbgsub_m7"), ("o104", "nrca", "resbgsub_m7")]:
        made[(obs, mod, stage)] = _touch(tmp_path, filt, NIRCAM.format(
            obs=obs, filt=filt, mod=mod, stage=stage))
    # the image and the model beside it are not residuals
    _touch(tmp_path, filt, f"jw10678-o105_t001_nircam_clear-{filt}-merged_i2d.fits")
    _touch(tmp_path, filt, f"jw10678-o105_t001_nircam_clear-{filt}-merged_"
                           f"resbgsub_m7_daophot_basic_mergedcat_model_i2d.fits")
    for obs, stage in [("o132", "resbgsub_m6"), ("o133", "m4")]:
        made[(obs, "miri", stage)] = _touch(tmp_path, G.MIRI_FILTER,
                                            MIRI.format(obs=obs, stage=stage))
    return filt, made


def test_only_the_final_stage_is_used(residual_tree):
    filt, made = residual_tree
    got = G.find_residual_i2d(filt)
    assert got == {
        "o101": made[("o101", "merged", "resbgsub_m7")],
        "o103_nrca": made[("o103", "nrca", "resbgsub_m7")],
        "o103_nrcb": made[("o103", "nrcb", "resbgsub_m7")],
        # a -merged residual supersedes the per-module ones, as for images
        "o104": made[("o104", "merged", "resbgsub_m7")],
    }


def test_a_tile_mid_chain_is_reported_but_not_built(residual_tree):
    filt, made = residual_tree
    got = G.find_residual_i2d(filt, final_only=False)
    assert got["o102"] == made[("o102", "merged", "resbgsub_m6")]
    assert got["o101"] == made[("o101", "merged", "resbgsub_m7")]


def test_miri_final_stage_is_m6(residual_tree):
    """The MIRI chain has no cross-band m7."""
    _, made = residual_tree
    assert G.find_residual_i2d(G.MIRI_FILTER) == {
        "o132": made[("o132", "miri", "resbgsub_m6")]}


def _all_flavours():
    out = [dict(miri=False, bgmatch=False, stretch=s, residual=r)
           for r, s in G.nircam_flavours()]
    out += [dict(miri=True, bgmatch=b, stretch=G.DEFAULT_STRETCH, residual=r)
            for r, b in G.MIRI_FLAVOURS]
    return out


def _layer(fl, obs="o112"):
    if fl["miri"]:
        return os.path.basename(G.miri_hips_for(obs, fl["bgmatch"], fl["residual"]))
    return os.path.basename(G.hips_for(obs, fl["stretch"], fl["residual"]))


def _coadd(fl):
    if fl["miri"]:
        return G.miri_coadd_name_for(fl["bgmatch"], fl["residual"])
    return G.coadd_name_for(fl["stretch"], residual=fl["residual"])


def test_every_flavour_has_its_own_layer_png_and_coadd():
    fls = _all_flavours()
    layers = [_layer(f) for f in fls]
    coadds = [_coadd(f) for f in fls]
    pngs = [G.miri_png_for("o112", f["bgmatch"], f["residual"]) if f["miri"]
            else G.png_for("o112", f["stretch"], f["residual"]) for f in fls]
    assert len(set(layers)) == len(fls)
    assert len(set(coadds)) == len(fls)
    assert len(set(pngs)) == len(fls)


@pytest.mark.parametrize("fl", _all_flavours(),
                         ids=lambda f: f"{'miri' if f['miri'] else 'nircam'}-"
                                       f"{f['stretch']}-bg{f['bgmatch']}-"
                                       f"res{f['residual']}")
def test_a_coadd_glob_matches_only_its_own_flavour(fl):
    tail = G.layer_tail(fl["miri"], fl["bgmatch"], fl["stretch"], fl["residual"])
    for other in _all_flavours():
        hit = fnmatch.fnmatchcase(_layer(other), f"GCTreasury_*{tail}")
        assert hit == (other == fl), (tail, _layer(other))
    assert _layer(fl).endswith(tail)


def test_residual_has_no_pct_or_bgmatch_flavour():
    for bad in ("pct", "bgmatch"):
        with pytest.raises(ValueError):
            G.check_residual_stretch(bad, True)
        G.check_residual_stretch(bad, False)       # images keep them
    with pytest.raises(ValueError):
        G.build_miri_obs("o132", bgmatch=True, residual=True)


def test_cli_refuses_a_residual_stretch_that_does_not_exist(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["gc_treasury_rgb_images.py", "--residual",
                                      "--stretch", "pct", "--obs", "o105"])
    with pytest.raises(SystemExit) as exc:
        G.main()
    assert exc.value.code == 2


def test_image_flavours_are_built_before_residual_ones():
    fls = G.nircam_flavours()
    first_residual = [r for r, _ in fls].index(True)
    assert not any(r for r, _ in fls[:first_residual])
    assert all(r for r, _ in fls[first_residual:])
    assert {s for r, s in fls if r} == set(G.RESIDUAL_STRETCHES)


def test_an_exhausted_budget_defers_every_build(tmp_path, monkeypatch, capsys):
    """Builds past the budget wait for the next tick instead of running into
    the job's wall time before the coadd."""
    monkeypatch.setattr(G, "OUTDIR", str(tmp_path))
    built = []
    monkeypatch.setattr(G, "build_obs",
                        lambda o, **k: built.append(o) or (None, None))
    monkeypatch.setattr(G, "build_miri_obs",
                        lambda o, **k: built.append(o) or (None, None))
    monkeypatch.setattr(G, "needs_build", lambda o, inv, **k: "no RGB yet")
    monkeypatch.setattr(G, "miri_needs_build",
                        lambda o, s, bg=False, residual=False: "no MIRI png yet")
    monkeypatch.setattr(G, "miri_match_is_stale", lambda m: None)
    monkeypatch.setattr(G, "find_i2d", lambda *a, **k: {"o132": "x"})
    monkeypatch.setattr(G, "find_residual_i2d", lambda *a, **k: {"o132": "x"})
    monkeypatch.setattr(G, "inventory",
                        lambda **k: ({f: {"o001": "x"} for f in G.FILTERS},
                                     ["o001"]))
    coadds = []
    monkeypatch.setattr(G, "cmd_coadd", lambda **k: coadds.append(k) or 0)
    assert G.cmd_auto(budget_hours=0) == 0
    out = capsys.readouterr().out
    assert built == [] and coadds == []
    n = len(G.nircam_flavours()) + len(G.MIRI_FLAVOURS)
    assert f"{n} build(s) left for the next tick" in out
    assert "o001 residual/vminmax" in out and "o132 MIRI+res" in out
