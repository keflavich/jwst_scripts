"""A coadd left half-built must be rebuilt by the next --auto tick.

keflavich/jwst-gc-pipeline#1252: cmd_coadd's full rebuild removed the live
coadd before building the new one, a kill left Norder* tiles with no
`properties`, and --auto only called cmd_coadd for a flavour that had built a
tile in that same tick -- so nothing ever repaired it and the publisher skipped
the directory every hour.
"""
import json
import os
import shutil

import pytest

pytest.importorskip("reproject.hips")

import gc_treasury_rgb_images as G                            # noqa: E402
import jwst_rgb.incremental_coadd as IC                      # noqa: E402
from jwst_rgb.incremental_coadd import (                     # noqa: E402
    coadd_needs_repair, coadd_repair_reason, load_manifest,
    recover_interrupted_swap,
)
from test_incremental_coadd import make_layer                # noqa: E402


def _coadd(path, properties="hips_order = 3\n", manifest="{}"):
    os.makedirs(os.path.join(path, "Norder3", "Dir0"), exist_ok=True)
    if properties is not None:
        with open(os.path.join(path, "properties"), "w") as fh:
            fh.write(properties)
    if manifest is not None:
        with open(os.path.join(path, "coadd_manifest.json"), "w") as fh:
            fh.write(manifest)
    return str(path)


LAYERS = ["/somewhere/layer_hips"]


# ---- coadd_needs_repair ----------------------------------------------------

def test_a_missing_directory_needs_repair(tmp_path):
    """A kill between the old rmtree and the first makedirs leaves nothing."""
    out = str(tmp_path / "gone_hips")
    assert coadd_needs_repair(out, LAYERS)
    assert coadd_repair_reason(out, LAYERS) == "no coadd directory"


def test_norder_only_needs_repair(tmp_path):
    out = _coadd(tmp_path / "half_hips", properties=None, manifest=None)
    assert coadd_needs_repair(out, LAYERS)
    assert coadd_repair_reason(out, LAYERS) == "no readable properties"


def test_empty_properties_needs_repair(tmp_path):
    out = _coadd(tmp_path / "empty_hips", properties="")
    assert coadd_needs_repair(out, LAYERS)


def test_properties_without_manifest_needs_repair(tmp_path):
    """A kill after coadd_hips but before save_manifest."""
    out = _coadd(tmp_path / "nomanifest_hips", manifest=None)
    assert coadd_repair_reason(out, LAYERS) == "no manifest"


def test_a_truncated_manifest_needs_repair(tmp_path):
    out = _coadd(tmp_path / "trunc_hips", manifest='{"order": ["a", ')
    assert coadd_repair_reason(out, LAYERS) == "unreadable manifest"


def test_a_finished_coadd_needs_no_repair(tmp_path):
    out = _coadd(tmp_path / "ok_hips")
    assert not coadd_needs_repair(out, LAYERS)
    assert coadd_repair_reason(out, LAYERS) is None


def test_a_flavour_with_no_layers_is_never_repaired(tmp_path):
    """cmd_coadd returns 1 for a flavour with no layers; repairing it would
    log a failure every tick for something there is nothing to build."""
    assert not coadd_needs_repair(str(tmp_path / "gone_hips"), [])
    assert not coadd_needs_repair(_coadd(tmp_path / "x_hips", properties=None),
                                  [])


# ---- cmd_auto ----------------------------------------------------------------

@pytest.fixture
def auto(tmp_path, monkeypatch):
    """cmd_auto with nothing buildable and cmd_coadd recorded."""
    monkeypatch.setattr(G, "OUTDIR", str(tmp_path))
    monkeypatch.setattr(G, "inventory", lambda residual=False: ({}, []))
    monkeypatch.setattr(G, "find_i2d", lambda *a, **k: {})
    monkeypatch.setattr(G, "find_residual_i2d", lambda *a, **k: {})
    monkeypatch.setattr(G, "needs_build", lambda *a, **k: None)
    monkeypatch.setattr(G, "miri_needs_build", lambda *a, **k: None)
    monkeypatch.setattr(G, "mono_needs_build", lambda *a, **k: None)
    monkeypatch.setattr(G, "miri_match_is_stale", lambda m: None)
    calls = []
    monkeypatch.setattr(G, "cmd_coadd", lambda **k: calls.append(k) or 0)
    return calls


def _flavours():
    for r, s in G.nircam_flavours():
        yield (dict(stretch=s, residual=r), G.coadd_name_for(s, residual=r))
    for r, bg in G.MIRI_FLAVOURS:
        yield (dict(miri=True, bgmatch=bg, residual=r),
               G.miri_coadd_name_for(bg, r))
    for f in G.MONO_FILTERS:
        yield dict(mono=f), G.mono_coadd_name_for(f)


def _layer_for(kw, tmp_path):
    """Create one per-observation layer of the flavour `kw`."""
    if kw.get("mono"):
        d = G.mono_hips_for("o001", kw["mono"])
    elif kw.get("miri"):
        d = G.miri_hips_for("o001", kw["bgmatch"], kw["residual"])
    else:
        d = G.hips_for("o001", kw["stretch"], kw["residual"])
    os.makedirs(d, exist_ok=True)
    return d


def test_a_half_built_coadd_is_recoadded_with_nothing_newly_built(
        auto, tmp_path, capsys):
    """The #1252 case: no build this tick, Norder* and no properties."""
    for kw, name in _flavours():
        _layer_for(kw, tmp_path)
        _coadd(tmp_path / name, properties=None, manifest=None)
    assert G.cmd_auto() == 0
    assert sorted(map(repr, (sorted(k.items()) for k in auto))) == \
        sorted(repr(sorted(kw.items())) for kw, _ in _flavours())
    out = capsys.readouterr().out
    assert "repairing jwst_gc_treasury_f212n_residual_hips: " \
           "no readable properties" in out


def test_a_missing_coadd_with_layers_is_recoadded(auto, tmp_path):
    kw, name = next(f for f in _flavours() if f[0].get("mono") == "f212n")
    _layer_for(kw, tmp_path)
    G.cmd_auto()
    assert auto == [kw]


def test_a_healthy_coadd_is_left_alone(auto, tmp_path, capsys):
    for kw, name in _flavours():
        _layer_for(kw, tmp_path)
        _coadd(tmp_path / name)
    assert G.cmd_auto() == 0
    assert auto == []
    assert "repairing" not in capsys.readouterr().out


def test_a_flavour_with_no_layers_is_not_recoadded(auto, tmp_path):
    for kw, name in _flavours():
        _coadd(tmp_path / name, properties=None, manifest=None)
    assert G.cmd_auto() == 0
    assert auto == []


def test_a_failed_repair_is_reported(auto, tmp_path, monkeypatch, capsys):
    kw, name = next(f for f in _flavours() if f[0].get("mono") == "f212n")
    _layer_for(kw, tmp_path)
    monkeypatch.setattr(G, "cmd_coadd", lambda **k: 1)
    assert G.cmd_auto() == 1
    assert f"FAILED: coadd:{name}" in capsys.readouterr().out


# ---- the rebuild itself ------------------------------------------------------

@pytest.fixture
def two_layers(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "OUTDIR", str(tmp_path))
    for o in ("o001", "o002"):
        make_layer(str(tmp_path), os.path.basename(G.mono_hips_for(o, "f212n")),
                   [(3, 1)], colour=100)
    monkeypatch.setattr(G, "find_residual_i2d",
                        lambda filt, **k: {"o001": "x", "o002": "x"})
    return str(tmp_path / G.mono_coadd_name_for("f212n")), None


def test_a_killed_rebuild_leaves_the_live_coadd_intact(two_layers, monkeypatch):
    import reproject.hips
    out, _ = two_layers
    assert G.cmd_coadd(mono="f212n") == 0
    before = open(os.path.join(out, "coadd_manifest.json")).read()
    props = open(os.path.join(out, "properties")).read()

    def killed(layers, output_directory, **k):
        os.makedirs(os.path.join(output_directory, "Norder3"))
        raise MemoryError("OOM")
    monkeypatch.setattr(reproject.hips, "coadd_hips", killed)
    with pytest.raises(MemoryError):
        G.cmd_coadd(mono="f212n", full=True)

    assert open(os.path.join(out, "coadd_manifest.json")).read() == before
    assert open(os.path.join(out, "properties")).read() == props
    assert not coadd_needs_repair(out, LAYERS)
    # the only debris is the stage directory
    assert os.path.isdir(out + ".new") and not os.path.exists(out + ".old")


def test_a_stale_stage_directory_does_not_block_the_next_rebuild(two_layers):
    out, _ = two_layers
    os.makedirs(out + ".new")
    os.makedirs(out + ".old")
    assert G.cmd_coadd(mono="f212n") == 0
    assert not os.path.exists(out + ".new")
    assert not os.path.exists(out + ".old")
    manifest = json.load(open(os.path.join(out, "coadd_manifest.json")))
    assert len(manifest["order"]) == 2


def test_a_rebuild_replaces_a_half_built_coadd(two_layers):
    out, _ = two_layers
    _coadd(out, properties=None, manifest=None)
    assert coadd_needs_repair(out, LAYERS)
    assert G.cmd_coadd(mono="f212n") == 0
    assert not coadd_needs_repair(out, LAYERS)
    # named for the mosaic, not copied from its first layer
    text = open(os.path.join(out, "properties")).read()
    assert "nircam-f212n-residual" in text and "GCTreasury_o001" not in text


# ---- a layer rebuilt after the coadd -----------------------------------------

def test_a_layer_newer_than_the_manifest_is_recoadded(auto, tmp_path, capsys):
    kw, name = next(f for f in _flavours() if f[0].get("mono") == "f212n")
    layer = _layer_for(kw, tmp_path)
    out = _coadd(tmp_path / name)
    with open(os.path.join(layer, "properties"), "w") as fh:
        fh.write("hips_order = 3\n")
    old = os.path.getmtime(os.path.join(out, "coadd_manifest.json")) - 100
    os.utime(os.path.join(out, "coadd_manifest.json"), (old, old))
    G.cmd_auto()
    assert auto == [kw]
    assert "is newer than the coadd" in capsys.readouterr().out


def test_a_layer_older_than_the_manifest_is_left_alone(auto, tmp_path):
    kw, name = next(f for f in _flavours() if f[0].get("mono") == "f212n")
    layer = _layer_for(kw, tmp_path)
    with open(os.path.join(layer, "properties"), "w") as fh:
        fh.write("hips_order = 3\n")
    old = os.path.getmtime(os.path.join(layer, "properties")) - 100
    os.utime(os.path.join(layer, "properties"), (old, old))
    _coadd(tmp_path / name)
    G.cmd_auto()
    assert auto == []


def test_an_up_to_date_verdict_stops_the_newer_layer_trigger(two_layers):
    from jwst_rgb.incremental_coadd import newer_layer_reason
    out, _ = two_layers
    assert G.cmd_coadd(mono="f212n") == 0
    layers = G.coadd_layer_glob(mono="f212n")
    old = os.path.getmtime(os.path.join(out, "coadd_manifest.json")) - 100
    os.utime(os.path.join(out, "coadd_manifest.json"), (old, old))
    assert newer_layer_reason(out, layers)
    assert G.cmd_coadd(mono="f212n") == 0        # plan says "none"
    assert newer_layer_reason(out, layers) is None


# ---- an unusable manifest must not break the repair that diagnosed it -------

@pytest.mark.parametrize("text", ["", '{"order": ["a", ', "[]", "3", '"x"',
                                  '{"order": 3}', '{"layers": []}'])
def test_an_unusable_manifest_is_rebuilt_not_raised(two_layers, text):
    """json.load on a truncated manifest raised out of order_layers, so the
    repair died on the file coadd_repair_reason had just flagged; a valid but
    non-dict manifest raised AttributeError in plan_coadd."""
    out, _ = two_layers
    assert G.cmd_coadd(mono="f212n") == 0
    with open(os.path.join(out, "coadd_manifest.json"), "w") as fh:
        fh.write(text)
    assert coadd_repair_reason(out, LAYERS) == "unreadable manifest"
    assert G.cmd_coadd(mono="f212n") == 0
    manifest = json.load(open(os.path.join(out, "coadd_manifest.json")))
    assert manifest["order"] == sorted(manifest["order"]) and \
        len(manifest["order"]) == 2
    assert coadd_repair_reason(out, LAYERS) is None


def test_cmd_auto_repairs_a_truncated_manifest(two_layers, monkeypatch):
    """End to end: the tick returns 0, later flavours are not abandoned, and
    the coadd is whole afterwards."""
    out, _ = two_layers
    assert G.cmd_coadd(mono="f212n") == 0
    with open(os.path.join(out, "coadd_manifest.json"), "w") as fh:
        fh.write('{"order": ["GCTre')
    monkeypatch.setattr(G, "inventory", lambda residual=False: ({}, []))
    monkeypatch.setattr(G, "find_i2d", lambda *a, **k: {})
    for name in ("needs_build", "miri_needs_build", "mono_needs_build",
                 "miri_match_is_stale"):
        monkeypatch.setattr(G, name, lambda *a, **k: None)
    assert G.cmd_auto() == 0
    assert coadd_repair_reason(out, LAYERS) is None


# ---- a kill between the two renames of the swap ------------------------------

def test_a_kill_between_the_renames_restores_the_old_coadd(two_layers):
    out, _ = two_layers
    assert G.cmd_coadd(mono="f212n") == 0
    os.rename(out, out + ".old")                  # killed right after this
    assert G.cmd_coadd(mono="f212n") == 0
    assert not os.path.exists(out + ".old")
    assert coadd_repair_reason(out, LAYERS) is None


def test_a_kill_between_the_renames_finishes_a_complete_new(two_layers):
    out, _ = two_layers
    assert G.cmd_coadd(mono="f212n") == 0
    shutil.copytree(out, out + ".new")                # finished stage
    os.rename(out, out + ".old")
    marker = os.path.join(out + ".new", "marker")
    open(marker, "w").write("new")
    G.cmd_coadd(mono="f212n")
    assert os.path.exists(os.path.join(out, "marker"))
    assert not os.path.exists(out + ".old")


def test_live_coadd_plus_nonempty_old_debris(two_layers):
    out, _ = two_layers
    assert G.cmd_coadd(mono="f212n") == 0
    os.makedirs(out + ".old/Norder3")
    open(out + ".old/properties", "w").write("x\n")   # debris, kill after swap
    assert G.cmd_coadd(mono="f212n", full=True) == 0
    assert not os.path.exists(out + ".old")


def test_old_and_new_are_gone_after_a_full_rebuild_over_a_live_coadd(two_layers):
    out, _ = two_layers
    assert G.cmd_coadd(mono="f212n") == 0
    assert G.cmd_coadd(mono="f212n", full=True) == 0
    assert not os.path.exists(out + ".old") and not os.path.exists(out + ".new")


# ---- what recover_interrupted_swap may promote ------------------------------

def _copy(src, dst, mark, drop=None, empty=None):
    shutil.copytree(src, dst)
    open(os.path.join(dst, "MARK"), "w").write(mark)
    if drop:
        os.remove(os.path.join(dst, drop))
    if empty:
        open(os.path.join(dst, empty), "w").close()


@pytest.mark.parametrize("kind", [dict(drop="properties"),
                                  dict(empty="properties"),
                                  dict(drop="coadd_manifest.json"),
                                  dict(empty="coadd_manifest.json")])
def test_an_unfinished_new_is_not_promoted_over_a_complete_old(two_layers,
                                                               kind):
    out, _ = two_layers
    assert G.cmd_coadd(mono="f212n") == 0
    gold = out + ".gold"
    shutil.copytree(out, gold)
    shutil.rmtree(out)
    _copy(gold, out + ".new", "NEW", **kind)
    _copy(gold, out + ".old", "OLD")
    assert recover_interrupted_swap(out) == "restored .old"
    assert open(os.path.join(out, "MARK")).read() == "OLD"


def test_an_old_without_properties_is_not_restored(two_layers):
    out, _ = two_layers
    assert G.cmd_coadd(mono="f212n") == 0
    gold = out + ".gold"
    shutil.copytree(out, gold)
    shutil.rmtree(out)
    _copy(gold, out + ".old", "OLD", drop="properties")
    assert recover_interrupted_swap(out) is None
    assert not os.path.exists(out)


def test_a_restore_is_not_a_rebuild(two_layers, monkeypatch):
    """The old coadd comes back as it was: cmd_coadd must not rebuild it."""
    import reproject.hips
    out, _ = two_layers
    assert G.cmd_coadd(mono="f212n") == 0
    os.rename(out, out + ".old")
    open(os.path.join(out + ".old", "MARK"), "w").write("OLD")
    monkeypatch.setattr(reproject.hips, "coadd_hips", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("rebuilt instead of restoring")))
    assert G.cmd_coadd(mono="f212n") == 0
    assert open(os.path.join(out, "MARK")).read() == "OLD"


def test_a_manifest_with_invalid_utf8_is_rebuilt_not_raised(two_layers):
    out, _ = two_layers
    assert G.cmd_coadd(mono="f212n") == 0
    with open(os.path.join(out, "coadd_manifest.json"), "wb") as fh:
        fh.write(b"\xff\xfe\x00{")
    assert load_manifest(out) is None
    assert G.cmd_coadd(mono="f212n") == 0
    assert coadd_repair_reason(out, G.coadd_layer_glob(mono="f212n")) is None


class _Killed(BaseException):
    pass


def _boom(*a, **k):
    raise _Killed()


def test_a_first_build_killed_before_the_stamps_is_not_promoted(two_layers,
                                                                monkeypatch):
    """No previous coadd: a kill after the manifest but before stamp_identity
    left `.new` looking finished, and the next tick promoted it unstamped."""
    out, _ = two_layers
    with monkeypatch.context() as m:
        m.setattr(IC, "stamp_release_date", _boom)
        with pytest.raises(_Killed):
            G.cmd_coadd(mono="f212n")
    assert not os.path.exists(os.path.join(out + ".new", "coadd_manifest.json"))
    assert G.cmd_coadd(mono="f212n") == 0
    props = open(os.path.join(out, "properties")).read()
    assert f"creator_did          = {IC.coadd_did(os.path.basename(out))}" in props


def test_a_killed_append_leaves_the_live_coadd_alone(tmp_path, monkeypatch):
    """hardlink_tree shared the manifest and properties inodes, so the append
    rewrote the LIVE files and a kill left a manifest naming a layer the live
    tiles lack; the next tick then said 'none'."""
    monkeypatch.setattr(G, "OUTDIR", str(tmp_path))
    inv = {"o001": "x"}
    monkeypatch.setattr(G, "find_residual_i2d", lambda filt, **k: dict(inv))
    make_layer(str(tmp_path), os.path.basename(G.mono_hips_for("o001", "f212n")),
               [(3, 1)], colour=100)
    out = str(tmp_path / G.mono_coadd_name_for("f212n"))
    assert G.cmd_coadd(mono="f212n") == 0
    files = ("coadd_manifest.json", "properties")
    before = {f: open(os.path.join(out, f)).read() for f in files}
    inv["o002"] = "x"
    make_layer(str(tmp_path), os.path.basename(G.mono_hips_for("o002", "f212n")),
               [(3, 5)], colour=100)
    with monkeypatch.context() as m:
        m.setattr(IC, "stamp_identity", _boom)
        with pytest.raises(_Killed):
            G.cmd_coadd(mono="f212n")
    for f in files:
        assert open(os.path.join(out, f)).read() == before[f]
    assert G.cmd_coadd(mono="f212n") == 0           # and the next tick appends
    assert len(json.load(open(os.path.join(out, "coadd_manifest.json")))["order"]) == 2
