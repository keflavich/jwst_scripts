"""A re-rendered or dropped layer must retile, not rebuild, the coadd.

keflavich/jwst-gc-pipeline#941: any changed layer used to force a 1h40m rebuild
of every tile of the mosaic, although one layer touches ~540 of ~22,600 tiles.
The justification for retiling is the same as for appending -- it produces
exactly what a full rebuild produces -- so every scenario below is compared
tile-for-tile against `coadd_hips` of the same ordered list.
"""
import json
import os
import shutil

import numpy as np
import pytest

pytest.importorskip("reproject.hips")
from reproject.hips import coadd_hips                        # noqa: E402

import gc_treasury_rgb_images as G                            # noqa: E402
from jwst_rgb.incremental_coadd import (                     # noqa: E402
    backfill_manifest_files, hardlink_tree, load_manifest, plan_coadd,
    retile_coadd, save_manifest, set_hips_order,
)
from test_incremental_coadd import (                         # noqa: E402
    assert_same_tiles, make_layer, tiles_of,
)


def tile(n, order=3):
    return os.path.join(f"Norder{order}", "Dir0", f"Npix{n}.png")


def rerender(src, name, tiles, colour, alpha=255):
    """Replace a layer wholesale, as a re-render does: old tiles are gone."""
    shutil.rmtree(os.path.join(src, name))
    d = make_layer(src, name, tiles, colour, alpha)
    # newer than the manifest whatever the clock resolution
    for dirpath, _, filenames in os.walk(d):
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            os.utime(p, (os.path.getmtime(p) + 100,) * 2)
    return d


@pytest.fixture
def built(tmp_path):
    """A, B, C painted in that order into a coadd with a manifest."""
    src = tmp_path / "src"
    src.mkdir()
    a = make_layer(str(src), "layerA", [(3, 10), (3, 11)], (255, 0, 0), alpha=128)
    b = make_layer(str(src), "layerB", [(3, 10), (3, 12)], (0, 255, 0), alpha=128)
    c = make_layer(str(src), "layerC", [(3, 11), (3, 99)], (0, 0, 255))
    out = str(tmp_path / "out")
    coadd_hips([a, b, c], out)
    save_manifest(out, [a, b, c])
    return tmp_path, str(src), [a, b, c], out


def retiled(tmp_path, out, layers, name="stage", **kw):
    action, ls, reason, affected = plan_coadd(out, layers,
                                              max_retile_fraction=1.0, **kw)
    assert action == "retile", reason
    stage = str(tmp_path / name)
    hardlink_tree(out, stage)
    retile_coadd(stage, ls, affected)
    return stage, affected


def rebuilt(tmp_path, layers, name="full"):
    full = str(tmp_path / name)
    coadd_hips(layers, full)
    return full


# ---- equivalence with a full rebuild ---------------------------------------

def test_retile_matches_full_rebuild(built):
    """B (the middle layer) changes colour, loses (3,12) and gains (3,13)."""
    tmp_path, src, (a, b, c), out = built
    b = rerender(src, "layerB", [(3, 10), (3, 13)], (200, 200, 0), alpha=128)
    stage, affected = retiled(tmp_path, out, [a, b, c])
    assert [os.path.basename(t) for t in affected] == \
        [f"Npix{n}.png" for n in (10, 12, 13)]
    assert_same_tiles(rebuilt(tmp_path, [a, b, c]), stage)


def test_a_tile_the_layer_dropped_is_removed_or_falls_back(built):
    tmp_path, src, (a, b, c), out = built
    # B no longer covers (3,10), which A also covers, nor (3,12), which only B did
    b = rerender(src, "layerB", [(3, 13)], (0, 255, 0))
    stage, _ = retiled(tmp_path, out, [a, b, c])
    t = tiles_of(stage)
    assert tile(12) not in t                    # nobody covers it any more
    assert tile(13) in t                        # gained beyond its old footprint
    assert_same_tiles(rebuilt(tmp_path, [a, b, c]), stage)
    # (3,10) is A alone now: a copy, not a composite
    assert np.array_equal(t[tile(10)], tiles_of(a)[tile(10)])


def test_a_layer_that_gains_tiles_beyond_its_footprint(built):
    tmp_path, src, (a, b, c), out = built
    a = rerender(src, "layerA", [(3, 10), (3, 11), (3, 40), (3, 41)],
                 (255, 0, 0), alpha=128)
    stage, _ = retiled(tmp_path, out, [a, b, c])
    assert tile(40) in tiles_of(stage) and tile(41) in tiles_of(stage)
    assert_same_tiles(rebuilt(tmp_path, [a, b, c]), stage)


def test_a_dropped_layer_retiles_without_a_rebuild(built):
    tmp_path, _, (a, b, c), out = built
    stage, affected = retiled(tmp_path, out, [a, c])
    assert [os.path.basename(t) for t in affected] == ["Npix10.png", "Npix12.png"]
    assert tile(12) not in tiles_of(stage)
    assert_same_tiles(rebuilt(tmp_path, [a, c]), stage)


def test_retile_and_a_new_layer_in_the_same_tick(built):
    tmp_path, src, (a, b, c), out = built
    b = rerender(src, "layerB", [(3, 10), (3, 12)], (5, 5, 5), alpha=128)
    d = make_layer(src, "layerD", [(3, 12), (3, 70)], (9, 90, 9), alpha=200)
    stage, affected = retiled(tmp_path, out, [a, b, c, d])
    assert tile(70) in affected
    assert_same_tiles(rebuilt(tmp_path, [a, b, c, d]), stage)


def test_the_first_layer_is_on_top_for_three_layers(built):
    """Three layers on one tile: coadd_hips re-saves after each composite."""
    tmp_path, src, (a, b, c), out = built
    # all three cover (3,10) after this
    c = rerender(src, "layerC", [(3, 10), (3, 11), (3, 99)], (0, 0, 255), alpha=100)
    stage, _ = retiled(tmp_path, out, [a, b, c])
    assert_same_tiles(rebuilt(tmp_path, [a, b, c]), stage)


def test_retile_with_a_worker_pool_matches(built):
    tmp_path, src, (a, b, c), out = built
    b = rerender(src, "layerB", [(3, 10), (3, 13)], (200, 200, 0), alpha=128)
    action, ls, _, affected = plan_coadd(out, [a, b, c], max_retile_fraction=1.0)
    stage = str(tmp_path / "pool")
    hardlink_tree(out, stage)
    retile_coadd(stage, ls, affected, workers=2)
    assert_same_tiles(rebuilt(tmp_path, [a, b, c]), stage)


def test_an_emptied_directory_is_pruned(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    a = make_layer(str(src), "layerA", [(3, 10)], (255, 0, 0))
    b = make_layer(str(src), "layerB", [(3, 10), (4, 7)], (0, 255, 0))
    c = make_layer(str(src), "layerC", [(3, k) for k in range(200, 210)],
                   (0, 0, 255))                      # unrelated bulk
    out = str(tmp_path / "out")
    coadd_hips([a, b, c], out)
    save_manifest(out, [a, b, c])
    assert os.path.isdir(os.path.join(out, "Norder4"))
    b = rerender(str(src), "layerB", [(3, 10)], (0, 255, 0))
    stage, _ = retiled(tmp_path, out, [a, b, c])
    assert not os.path.exists(os.path.join(stage, "Norder4"))
    assert_same_tiles(rebuilt(tmp_path, [a, b, c]), stage)


def test_hardlink_clone_is_not_mutated_by_retile(built):
    """The clone shares inodes with the live coadd until the swap."""
    tmp_path, src, (a, b, c), out = built
    before = tiles_of(out)
    b = rerender(src, "layerB", [(3, 10), (3, 13)], (200, 200, 0), alpha=128)
    stage, _ = retiled(tmp_path, out, [a, b, c])
    changed = [r for r in set(before) & set(tiles_of(stage))
               if not np.array_equal(before[r], tiles_of(stage)[r])]
    assert changed, "fixture no longer changes a shared tile"
    after = tiles_of(out)
    assert set(after) == set(before)
    for rel in before:
        assert np.array_equal(before[rel], after[rel]), rel


# ---- the plan --------------------------------------------------------------

def test_the_manifest_records_each_layers_tiles(built):
    _, _, (a, b, c), out = built
    m = load_manifest(out)
    assert m["layers"]["layerB"]["files"] == [tile(10), tile(12)]
    assert m["layers"]["layerB"]["tiles"] == 2      # old readers still work


def test_a_legacy_manifest_with_a_changed_layer_rebuilds(built):
    tmp_path, src, (a, b, c), out = built
    m = load_manifest(out)
    for rec in m["layers"].values():
        del rec["files"]
    with open(os.path.join(out, "coadd_manifest.json"), "w") as fh:
        json.dump(m, fh)
    b = rerender(src, "layerB", [(3, 10)], (1, 1, 1))
    action, _, reason, affected = plan_coadd(out, [a, b, c],
                                             max_retile_fraction=1.0)
    assert action == "rebuild" and affected is None
    assert "no tile list" in reason and "layerB" in reason


def test_a_legacy_manifest_is_backfilled_while_layers_are_unchanged(built):
    tmp_path, src, (a, b, c), out = built
    m = load_manifest(out)
    for rec in m["layers"].values():
        del rec["files"]
    with open(os.path.join(out, "coadd_manifest.json"), "w") as fh:
        json.dump(m, fh)
    assert plan_coadd(out, [a, b, c])[0] == "none"
    assert backfill_manifest_files(out, [a, b, c]) is True
    assert backfill_manifest_files(out, [a, b, c]) is False
    # now the first re-render retiles instead of rebuilding
    b = rerender(src, "layerB", [(3, 10), (3, 13)], (1, 1, 1))
    assert plan_coadd(out, [a, b, c], max_retile_fraction=1.0)[0] == "retile"


def test_a_reorder_still_rebuilds(built):
    _, _, (a, b, c), out = built
    action, _, reason, _ = plan_coadd(out, [b, a, c], max_retile_fraction=1.0)
    assert action == "rebuild" and "reordered" in reason


def test_retile_falls_back_to_rebuild_when_it_would_cost_too_much(built):
    """B re-rendered touches 3 of 4 coadd tiles here, over half of the work."""
    tmp_path, src, (a, b, c), out = built
    b = rerender(src, "layerB", [(3, 10), (3, 13), (3, 11)], (1, 1, 1))
    action, _, reason, affected = plan_coadd(out, [a, b, c])
    assert action == "rebuild" and affected is None
    assert "rebuild operations" in reason
    assert plan_coadd(out, [a, b, c], max_retile_fraction=1.0)[0] == "retile"


def test_a_pure_addition_is_still_an_append(built):
    tmp_path, src, (a, b, c), out = built
    d = make_layer(src, "layerD", [(3, 70)], (1, 2, 3))
    action, new, _, affected = plan_coadd(out, [a, b, c, d])
    assert action == "append" and affected is None
    assert [os.path.basename(x) for x in new] == ["layerD"]


# ---- hips_order -------------------------------------------------------------

def test_hips_order_is_the_deepest_layer(built):
    tmp_path, src, (a, b, c), out = built
    text = open(os.path.join(b, "properties")).read()
    open(os.path.join(b, "properties"), "w").write(
        text.replace("hips_order           = 3", "hips_order           = 5"))
    stage = str(tmp_path / "orderstage")
    hardlink_tree(out, stage)
    assert set_hips_order(stage, [a, b, c]) == 5
    assert "hips_order           = 5" in open(os.path.join(stage, "properties")).read()
    # the live coadd's properties is untouched (hardlinked inode)
    assert "hips_order           = 3" in open(os.path.join(out, "properties")).read()


# ---- cmd_coadd end to end ---------------------------------------------------

OBS = ("o001", "o002", "o003", "o004")


@pytest.fixture
def mono(tmp_path, monkeypatch):
    """Four obs of disjoint tiles in the f212n residual mono flavour."""
    monkeypatch.setattr(G, "OUTDIR", str(tmp_path))
    for i, o in enumerate(OBS):
        make_layer(str(tmp_path), os.path.basename(G.mono_hips_for(o, "f212n")),
                   [(3, 100 * i + k) for k in range(8)], (40 * i, 0, 0))
    monkeypatch.setattr(G, "find_residual_i2d",
                        lambda filt, **k: {o: "x" for o in OBS})
    return tmp_path, str(tmp_path / G.mono_coadd_name_for("f212n"))


def test_cmd_coadd_retiles_a_rerendered_layer(mono, capsys):
    tmp_path, out = mono
    assert G.cmd_coadd(mono="f212n") == 0
    capsys.readouterr()
    # layers are in the dict order cmd_coadd sees; re-render o002 (middle)
    d = os.path.join(str(tmp_path), os.path.basename(G.mono_hips_for("o002", "f212n")))
    rerender_dir = rerender(str(tmp_path), os.path.basename(d),
                            [(3, 100 + k) for k in (0, 1, 2, 3, 4, 5, 6)],
                            (255, 255, 0))
    assert os.path.isdir(rerender_dir)
    assert G.cmd_coadd(mono="f212n") == 0
    text = capsys.readouterr().out
    assert "coadd plan: retile" in text
    # compared with a forced rebuild of the same layers
    live = tiles_of(out)
    shutil.copytree(out, str(tmp_path / "kept"))
    assert G.cmd_coadd(mono="f212n", full=True) == 0
    assert_same_tiles(str(tmp_path / "kept"), out)
    assert tile(107) not in live                    # the tile it dropped
    m = load_manifest(out)
    assert all("files" in rec for rec in m["layers"].values())
    assert not os.path.exists(out + ".new") and not os.path.exists(out + ".old")


def test_cmd_coadd_will_not_retile_around_an_unreadable_layer(mono, capsys):
    tmp_path, out = mono
    assert G.cmd_coadd(mono="f212n") == 0
    before = tiles_of(out)
    d = os.path.join(str(tmp_path), os.path.basename(G.mono_hips_for("o002", "f212n")))
    rerender_dir = rerender(str(tmp_path), os.path.basename(d),
                            [(3, 100 + k) for k in (0, 1, 2)], (1, 2, 3))
    os.remove(os.path.join(rerender_dir, "properties"))   # killed mid-write
    capsys.readouterr()
    assert G.cmd_coadd(mono="f212n") == 1
    assert "not readable" in capsys.readouterr().out
    after = tiles_of(out)
    assert set(after) == set(before)
    for rel in before:
        assert np.array_equal(before[rel], after[rel])
