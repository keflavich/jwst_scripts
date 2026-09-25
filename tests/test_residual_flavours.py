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
import time

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
                        lambda o, s, bg=False, residual=False, **k:
                        "no MIRI png yet")
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


# ---- cmd_auto wiring: which source each flavour reads, in which order ----
#
# The two inventories differ on purpose: the image one holds o001 and o002,
# the residual one o001 only, and likewise o132/o133 against o132 for MIRI.
# A residual pass that read the image inventory would build o002 or o133.

def _auto_fakes(monkeypatch, tmp_path, clock=None):
    """cmd_auto with every build, verdict and coadd recorded instead of run.

    With `clock`, each build advances it by an hour.
    """
    monkeypatch.setattr(G, "OUTDIR", str(tmp_path))
    rec = dict(calls=[], coadds=[], nircam=[], miri=[])

    def took_an_hour():
        if clock is not None:
            clock[0] += 3600

    def build_obs(o, **k):
        rec["calls"].append(("nircam", o, k.get("stretch"),
                             k.get("residual", False)))
        took_an_hour()
        return None, None

    def build_miri_obs(o, **k):
        rec["calls"].append(("miri", o, k.get("bgmatch", False),
                             k.get("residual", False)))
        took_an_hour()
        return None, None

    def needs_build(o, inv, stretch=G.DEFAULT_STRETCH, residual=False,
                    image_inv=None):
        rec["nircam"].append((o, residual, inv, image_inv))
        return "no RGB yet"

    def miri_needs_build(o, src, bgmatch=False, residual=False,
                         image_src=None):
        rec["miri"].append((o, residual, src, image_src))
        return "no MIRI png yet"

    image = {f: {o: f"img-{f}-{o}" for o in ("o001", "o002")}
             for f in G.FILTERS}
    resid = {f: {"o001": f"res-{f}-o001"} for f in G.FILTERS}
    monkeypatch.setattr(G, "inventory", lambda residual=False: (
        (resid, ["o001"]) if residual else (image, ["o001", "o002"])))
    monkeypatch.setattr(G, "find_i2d", lambda filt, **k: {
        "o132": "img-o132", "o133": "img-o133"})
    monkeypatch.setattr(G, "find_residual_i2d",
                        lambda filt, **k: {"o132": "res-o132"})
    monkeypatch.setattr(G, "build_obs", build_obs)
    monkeypatch.setattr(G, "build_miri_obs", build_miri_obs)
    monkeypatch.setattr(G, "needs_build", needs_build)
    monkeypatch.setattr(G, "miri_needs_build", miri_needs_build)
    monkeypatch.setattr(G, "miri_match_is_stale", lambda m: None)
    monkeypatch.setattr(G, "cmd_coadd", lambda **k: rec["coadds"].append(k) or 0)
    return rec


def _expected_calls():
    """Every build an unlimited tick makes over _auto_fakes, in order."""
    out = [("nircam", o, s, False)
           for s in sorted(G.STRETCHES) for o in ("o001", "o002")]
    out += [("miri", o, bg, False)
            for r, bg in G.MIRI_FLAVOURS if not r for o in ("o132", "o133")]
    out += [("nircam", "o001", s, True) for s in G.RESIDUAL_STRETCHES]
    out += [("miri", "o132", False, True)]
    return out


def _tag(call):
    kind, o, x, residual = call
    if kind == "nircam":
        return f"{o} residual/{x}" if residual else f"{o} {x}"
    return f"{o} " + ("MIRI+res" if residual else "MIRI+bg" if x else "MIRI")


def test_a_full_budget_builds_every_flavour_from_its_own_inventory(
        tmp_path, monkeypatch):
    rec = _auto_fakes(monkeypatch, tmp_path)
    assert G.cmd_auto(budget_hours=100) == 0
    # kwargs, tiles and order in one go: residual builds carry residual=True,
    # cover the residual inventory's tiles only, and all come after the last
    # image build
    assert rec["calls"] == _expected_calls()
    for o, residual, inv, image_inv in rec["nircam"]:
        want = "res-" if residual else "img-"
        assert inv[G.TARGET_FILTER][o].startswith(want), (o, residual)
        # the residual verdict gets the image inventory for the age check
        assert (image_inv is not None and "o002" in image_inv[G.TARGET_FILTER]
                ) if residual else image_inv is None
    for o, residual, src, image_src in rec["miri"]:
        assert src.startswith("res-" if residual else "img-"), (o, residual)
        if residual:
            assert image_src == f"img-{o}"
    got = sorted((k.get("miri", False), k.get("bgmatch", False),
                  "" if k.get("miri") else k["stretch"], k.get("residual", False))
                 for k in rec["coadds"])
    want = sorted([(False, False, s, r) for r, s in G.nircam_flavours()] +
                  [(True, bg, "", r) for r, bg in G.MIRI_FLAVOURS])
    assert got == want


def test_a_budget_that_runs_out_defers_exactly_the_tail(tmp_path, monkeypatch,
                                                         capsys):
    clock = [1_000_000.0]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    rec = _auto_fakes(monkeypatch, tmp_path, clock)
    # each fake build takes an hour, so a 2.5 h budget lets three start
    assert G.cmd_auto(budget_hours=2.5) == 0
    full = _expected_calls()
    assert rec["calls"] == full[:3]
    out = capsys.readouterr().out
    assert (f"{len(full) - 3} build(s) left for the next tick: "
            f"{', '.join(_tag(c) for c in full[3:])}") in out
    # only what was built is recoadded
    assert {(k["stretch"], k["residual"]) for k in rec["coadds"]} == \
        {(s, r) for _, _, s, r in full[:3]}


def test_a_chain_verdict_is_skipped_not_built_or_deferred(tmp_path, monkeypatch,
                                                         capsys):
    rec = _auto_fakes(monkeypatch, tmp_path)
    monkeypatch.setattr(G, "needs_build", lambda o, inv, residual=False, **k:
                        "CHAIN" if residual else None)
    monkeypatch.setattr(G, "miri_needs_build",
                        lambda o, s, bg=False, residual=False, **k:
                        "CHAIN" if residual else None)
    assert G.cmd_auto(budget_hours=100) == 0
    assert rec["calls"] == [] and rec["coadds"] == []
    out = capsys.readouterr().out
    assert "o001 residual/vminmax: residual predates the image mosaic" in out
    assert "o132 MIRI+res: residual predates the image mosaic" in out
    assert "left for the next tick" not in out
    # the one place these waits are counted: the lock report leaves them out
    held = [f"o001 residual/{s}" for s in G.RESIDUAL_STRETCHES] + [
        "o132 MIRI+res"]
    assert (f"{len(held)} residual build(s) waiting for their chain to "
            f"rewrite a residual older than its image: {', '.join(held)}"
            ) in out


def test_a_tick_with_no_chain_wait_prints_no_count(tmp_path, monkeypatch,
                                                   capsys):
    _auto_fakes(monkeypatch, tmp_path)
    assert G.cmd_auto(budget_hours=100) == 0
    assert "waiting for their chain" not in capsys.readouterr().out


def test_the_lock_report_reads_each_flavour_from_its_own_inventory(monkeypatch):
    image = {f: {"o001": "i", "o002": "i"} for f in G.FILTERS}
    resid = {f: {"o001": "r"} for f in G.FILTERS}
    monkeypatch.setattr(G, "inventory", lambda residual=False: (
        (resid, ["o001"]) if residual else (image, ["o001", "o002"])))
    monkeypatch.setattr(G, "needs_build", lambda o, inv, **k: "no RGB yet")
    monkeypatch.setattr(G, "find_i2d",
                        lambda filt, **k: {"o132": "i", "o133": "i"})
    monkeypatch.setattr(G, "find_residual_i2d", lambda filt, **k: {"o132": "r"})
    monkeypatch.setattr(G, "miri_needs_build",
                        lambda o, s, bg=False, residual=False, **k:
                        "no MIRI png yet")
    lines = G._pending_summary()
    assert "o002 NIRCam/vminmax -- no RGB yet" in lines
    assert "o001 NIRCam/residual/vminmax -- no RGB yet" in lines
    assert not [ln for ln in lines if ln.startswith("o002 NIRCam/residual")]
    assert "o133 MIRI -- no MIRI png yet" in lines
    assert "o132 MIRI+res -- no MIRI png yet" in lines
    assert not [ln for ln in lines if ln.startswith("o133 MIRI+res")]


def test_the_lock_report_gives_residual_verdicts_their_image(monkeypatch):
    # Without the image inventory the CHAIN guard is off, so a residual older
    # than its image would be reported as work waiting on the lock.
    image = {f: {"o001": "i"} for f in G.FILTERS}
    resid = {f: {"o001": "r"} for f in G.FILTERS}
    monkeypatch.setattr(G, "inventory", lambda residual=False: (
        (resid, ["o001"]) if residual else (image, ["o001"])))
    monkeypatch.setattr(G, "find_i2d", lambda filt, **k: {"o132": "i"})
    monkeypatch.setattr(G, "find_residual_i2d", lambda filt, **k: {"o132": "r"})
    seen = []

    def needs_build(o, inv, residual=False, image_inv=None, **k):
        seen.append(("nircam", residual, image_inv))
        return "CHAIN" if residual and image_inv is not None else "stale"

    def miri_needs_build(o, s, bg=False, residual=False, image_src=None):
        seen.append(("miri", residual, image_src))
        return "CHAIN" if residual and image_src is not None else "stale"

    monkeypatch.setattr(G, "needs_build", needs_build)
    monkeypatch.setattr(G, "miri_needs_build", miri_needs_build)
    lines = G._pending_summary()
    assert not [ln for ln in lines if "residual" in ln or "MIRI+res" in ln]
    held = [(kind, img) for kind, residual, img in seen if residual]
    assert held and all(img is image if kind == "nircam" else img == "i"
                        for kind, img in held)


def test_budget_hours_reaches_cmd_auto(monkeypatch):
    got = {}
    monkeypatch.setattr(G, "cmd_auto", lambda **k: got.update(k) or 0)
    monkeypatch.setattr(sys, "argv", ["gc_treasury_rgb_images.py", "--auto",
                                      "--budget-hours", "1.5"])
    assert G.main() == 0
    assert got["budget_hours"] == 1.5


# ---- the residual file is what a residual build reads ----

class _Opened(Exception):
    pass


def test_a_residual_build_reads_the_residual_mosaic(tmp_path, monkeypatch):
    from astropy.io import fits
    base = tmp_path / "base"
    monkeypatch.setattr(G, "BASE", str(base))
    monkeypatch.setattr(G, "OUTDIR", str(tmp_path / "out"))
    res = {f: _touch(base, f, NIRCAM.format(obs="o201", filt=f, mod="merged",
                                            stage="resbgsub_m7"))
           for f in G.FILTERS}
    for f in G.FILTERS:       # the image beside it, which must not be read
        _touch(base, f, f"jw10678-o201_t001_nircam_clear-{f}-merged_i2d.fits")
    mres = _touch(base, G.MIRI_FILTER, MIRI.format(obs="o132",
                                                   stage="resbgsub_m6"))
    _touch(base, G.MIRI_FILTER,
           "jw10678-o132_t001_miri_clear-f770w-mirimage_i2d.fits")
    opened = []

    def fake_open(path, *a, **k):
        opened.append(str(path))
        raise _Opened(path)
    monkeypatch.setattr(fits, "open", fake_open)
    with pytest.raises(_Opened):
        G.build_obs("o201", stretch="vminmax", residual=True)
    assert opened == [res[G.TARGET_FILTER]]
    with pytest.raises(_Opened):
        G.build_miri_obs("o132", residual=True)
    assert opened[-1] == mres


def _aged(tmp_path, name, t):
    p = tmp_path / name
    p.write_bytes(b"0")
    os.utime(p, (t, t))
    return str(p)


def test_needs_build_reads_the_png_of_its_own_flavour(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "OUTDIR", str(tmp_path))
    old = time.time() - G.SETTLE_SECONDS - 3600
    inv = {f: {"o201": _aged(tmp_path, f"{f}.fits", old)} for f in G.FILTERS}

    def make(residual):
        open(G.png_for("o201", "vminmax", residual), "wb").close()
        os.makedirs(os.path.join(G.hips_for("o201", "vminmax", residual),
                                 "Norder3"))
    make(residual=False)
    assert G.needs_build("o201", inv, stretch="vminmax") is None
    assert G.needs_build("o201", inv, stretch="vminmax",
                         residual=True) == "no RGB yet"
    make(residual=True)
    os.remove(G.png_for("o201", "vminmax", False))
    assert G.needs_build("o201", inv, stretch="vminmax", residual=True) is None
    assert G.needs_build("o201", inv, stretch="vminmax") == "no RGB yet"


def test_a_residual_older_than_its_image_waits_for_the_chain(tmp_path,
                                                            monkeypatch):
    """After a re-reduction the previous run's residual sits beside the new
    image until the chain rewrites it; building it then would pair a residual
    of the old reduction with the new image layer."""
    monkeypatch.setattr(G, "OUTDIR", str(tmp_path))
    now = time.time()
    img = {f: {"o201": _aged(tmp_path, f"img_{f}", now - 2 * 86400)}
           for f in G.FILTERS}
    stale = {f: {"o201": _aged(tmp_path, f"old_{f}", now - 5 * 86400)}
             for f in G.FILTERS}
    fresh = {f: {"o201": _aged(tmp_path, f"new_{f}", now - 86400)}
             for f in G.FILTERS}

    def nb(inv, **k):
        return G.needs_build("o201", inv, stretch="vminmax", **k)
    assert nb(stale, residual=True, image_inv=img) == "CHAIN"
    assert nb(fresh, residual=True, image_inv=img) == "no RGB yet"
    # one filter behind is enough
    mixed = {f: dict(v) for f, v in fresh.items()}
    mixed[G.FILTERS[0]] = stale[G.FILTERS[0]]
    assert nb(mixed, residual=True, image_inv=img) == "CHAIN"
    # nothing to compare against (image stale-tagged for regeneration)
    assert nb(stale, residual=True, image_inv={}) == "no RGB yet"
    # the image flavours never look
    assert nb(img, image_inv=stale) == "no RGB yet"

    mimg = _aged(tmp_path, "mimg", now - 2 * 86400)
    mold = _aged(tmp_path, "mold", now - 5 * 86400)
    mnew = _aged(tmp_path, "mnew", now - 86400)
    assert G.miri_needs_build("o132", mold, residual=True,
                              image_src=mimg) == "CHAIN"
    assert G.miri_needs_build("o132", mnew, residual=True,
                              image_src=mimg) != "CHAIN"
    assert G.miri_needs_build("o132", mold, residual=True,
                              image_src=None) != "CHAIN"


# ---- the residual coadds retire and publish on their own terms ----

class _Painted(Exception):
    pass


def test_a_residual_coadd_retires_on_the_residual_inventory(tmp_path,
                                                           monkeypatch):
    """cmd_coadd renames away every globbed layer outside the active set, so
    the active set has to be the residual inventory: taken from the images,
    it would retire residual layers the images lack and keep ones they have."""
    import jwst_rgb.incremental_coadd as ic
    monkeypatch.setattr(G, "OUTDIR", str(tmp_path))

    def stop(out, layers):
        raise _Painted(layers)
    monkeypatch.setattr(ic, "order_layers", stop)
    monkeypatch.setattr(G, "inventory", lambda residual=False: (
        {}, ["o001"] if residual else ["o002"]))
    monkeypatch.setattr(G, "find_residual_i2d", lambda filt, **k: {"o132": "r"})
    monkeypatch.setattr(G, "find_i2d", lambda filt, **k: {"o133": "i"})
    for kw, layer, keep, drop in [
            (dict(stretch="vminmax", residual=True),
             lambda o: G.hips_for(o, "vminmax", True), "o001", "o002"),
            (dict(miri=True, residual=True),
             lambda o: G.miri_hips_for(o, False, True), "o132", "o133")]:
        for o in (keep, drop):
            os.makedirs(layer(o))
        with pytest.raises(_Painted) as exc:
            G.cmd_coadd(**kw)
        assert exc.value.args[0] == [layer(keep)]
        assert os.path.isdir(layer(drop) + "_superseded")
        assert not os.path.exists(layer(drop))


def test_publish_copies_every_residual_coadd(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "OUTDIR", str(tmp_path / "out"))
    monkeypatch.setattr(G, "WEB", str(tmp_path / "web"))
    os.makedirs(tmp_path / "web")
    names = ([G.coadd_name_for(k, residual=True) for k in G.RESIDUAL_STRETCHES]
             + [G.MIRI_RESIDUAL_COADD_NAME])
    for n in names:
        os.makedirs(tmp_path / "out" / n / "Norder3")
    assert G.cmd_publish() == 0
    for n in names:
        assert (tmp_path / "web" / n / "Norder3").is_dir(), n
