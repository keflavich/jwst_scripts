"""Appending a layer must produce exactly what a full rebuild produces.

That equivalence is the whole justification for the incremental path, so it is
asserted on real tile bytes rather than argued from the precedence rules.
"""
import os

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("reproject.hips")
from reproject.hips import coadd_hips                        # noqa: E402

from jwst_rgb.incremental_coadd import (                     # noqa: E402
    hardlink_tree, layer_fingerprint, load_manifest, merge_layer,
    plan_coadd, save_manifest,
)

PROPS = """\
creator_did          = ivo://test/P/{name}
obs_title            = {name}
hips_version         = 1.4
hips_release_date    = 2026-01-01T00:00Z
hips_status          = public master clonableOnce
hips_tile_format     = png
hips_tile_width      = 8
hips_order           = 3
hips_frame           = galactic
dataproduct_type     = image
"""


def make_layer(root, name, tiles, colour, alpha=255):
    """tiles: iterable of (order, npix)."""
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "properties"), "w") as fh:
        fh.write(PROPS.format(name=name))
    for order, npix in tiles:
        sub = os.path.join(d, f"Norder{order}", "Dir0")
        os.makedirs(sub, exist_ok=True)
        arr = np.zeros((8, 8, 4), np.uint8)
        arr[..., :3] = colour
        arr[..., 3] = alpha
        Image.fromarray(arr, "RGBA").save(os.path.join(sub, f"Npix{npix}.png"))
    return d


def tiles_of(d):
    out = {}
    for dirpath, _, filenames in os.walk(d):
        for fn in filenames:
            if fn.endswith(".png"):
                rel = os.path.relpath(os.path.join(dirpath, fn), d)
                out[rel] = np.array(Image.open(os.path.join(dirpath, fn))
                                    .convert("RGBA"))
    return out


def assert_same_tiles(a, b):
    ta, tb = tiles_of(a), tiles_of(b)
    assert set(ta) == set(tb), (sorted(set(ta) ^ set(tb)))
    for rel in ta:
        assert np.array_equal(ta[rel], tb[rel]), f"tile differs: {rel}"


@pytest.fixture
def layers(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    # A and B overlap on (3, 10); C overlaps A on (3, 11) and adds (3, 99)
    a = make_layer(str(src), "layerA", [(3, 10), (3, 11)], (255, 0, 0))
    b = make_layer(str(src), "layerB", [(3, 10), (3, 12)], (0, 255, 0))
    c = make_layer(str(src), "layerC", [(3, 11), (3, 99)], (0, 0, 255))
    return str(src), [a, b, c]


def test_append_matches_full_rebuild(tmp_path, layers):
    _, (a, b, c) = layers

    full = str(tmp_path / "full")
    coadd_hips([a, b, c], full)

    part = str(tmp_path / "part")
    coadd_hips([a, b], part)
    save_manifest(part, [a, b])
    inc = str(tmp_path / "inc")
    hardlink_tree(part, inc)
    merge_layer(c, inc)

    assert_same_tiles(full, inc)


def test_append_of_two_layers_matches_full_rebuild(tmp_path, layers):
    _, (a, b, c) = layers
    full = str(tmp_path / "full2")
    coadd_hips([a, b, c], full)

    part = str(tmp_path / "part2")
    coadd_hips([a], part)
    inc = str(tmp_path / "inc2")
    hardlink_tree(part, inc)
    merge_layer(b, inc)
    merge_layer(c, inc)

    assert_same_tiles(full, inc)


def test_hardlink_clone_is_not_mutated_by_merge(tmp_path, layers):
    """The clone shares inodes with the previous coadd; compositing must not
    write through the link."""
    _, (a, b, c) = layers
    part = str(tmp_path / "part3")
    coadd_hips([a, b], part)
    before = tiles_of(part)
    inc = str(tmp_path / "inc3")
    hardlink_tree(part, inc)
    merge_layer(c, inc)
    after = tiles_of(part)
    assert set(after) == set(before)
    for rel in before:
        assert np.array_equal(before[rel], after[rel]), (
            f"previous coadd was mutated through the hardlink: {rel}")


def test_plan_detects_a_changed_layer(tmp_path, layers):
    src, (a, b, c) = layers
    out = str(tmp_path / "out")
    coadd_hips([a, b], out)
    save_manifest(out, [a, b])

    assert plan_coadd(out, [a, b])[0] == "none"
    assert plan_coadd(out, [a, b, c])[0] == "append"

    # rewrite a tile of an existing layer: must force a rebuild
    make_layer(src, "layerA", [(3, 10)], (1, 2, 3))
    action, _, reason = plan_coadd(out, [a, b, c])
    assert action == "rebuild", reason
    assert "changed" in reason


def test_plan_detects_a_dropped_layer(tmp_path, layers):
    _, (a, b, c) = layers
    out = str(tmp_path / "out2")
    coadd_hips([a, b], out)
    save_manifest(out, [a, b])
    action, _, reason = plan_coadd(out, [a])
    assert action == "rebuild"
    assert "no longer present" in reason


def test_plan_detects_reordering(tmp_path, layers):
    _, (a, b, c) = layers
    out = str(tmp_path / "out3")
    coadd_hips([a, b], out)
    save_manifest(out, [a, b])
    action, _, reason = plan_coadd(out, [b, a])
    assert action == "rebuild"
    assert "reordered" in reason


def test_plan_rebuilds_without_a_manifest(tmp_path, layers):
    """Coadds built before this existed have no manifest and must not be
    appended to blindly."""
    _, (a, b, _) = layers
    out = str(tmp_path / "out4")
    coadd_hips([a, b], out)
    action, _, reason = plan_coadd(out, [a, b])
    assert action == "rebuild"
    assert "manifest" in reason


def test_fingerprint_moves_when_a_tile_changes(tmp_path, layers):
    src, (a, _, _) = layers
    fp = layer_fingerprint(a)
    make_layer(src, "layerA", [(3, 10), (3, 11), (3, 55)], (9, 9, 9))
    assert layer_fingerprint(a) != fp


def test_manifest_round_trips(tmp_path, layers):
    _, (a, b, _) = layers
    out = str(tmp_path / "out5")
    coadd_hips([a, b], out)
    written = save_manifest(out, [a, b])
    assert load_manifest(out) == written
    assert written["order"] == ["layerA", "layerB"]
