"""The identity keywords a published HiPS carries.

These are the two keywords the HiPS network indexes on, so the properties
file a build leaves behind is what a user sees in a layer list.  reproject
mints a fresh `ivo://reproject/P/<uuid4>` on every run and echoes back
whatever title it was handed, so every build site has to pass its own.
"""
import os

import pytest

from jwst_rgb.hips_naming import (AUTHORITY, CREATOR, describe, properties_for,
                                  slugify, stamp_properties)


@pytest.mark.parametrize("name, filters", [
    ("BrickJWST_1182p2221_405_356_200_hips", ["F405N", "F356W", "F200W"]),
    ("Brick_RGB_444-356-200_hips", ["F444W", "F356W", "F200W"]),
    ("SgrA_RGB_MIRI_1500-1000-560_hips", ["F1500W", "F1000W", "F560W"]),
    ("Cloudef_MIRI_RGB_2100-1500-770_hips", ["F2100W", "F1500W", "F770W"]),
])
def test_channels_are_named_by_filter(name, filters):
    """The numbers are wavelengths in units of 10 nm; users read filters."""
    title = describe(name)[1]
    for f in filters:
        assert f in title


def test_the_did_is_ours_and_derived_from_the_directory():
    did, _ = describe("Brick_RGB_444-356-200_hips")
    assert did == f"{AUTHORITY}/brick-rgb-444-356-200"
    assert did.startswith("ivo://UFL/P/")


def test_the_target_is_spelled_out():
    assert "G0.253+0.016" in describe("Brick_RGB_444-356-200_hips")[1]


@pytest.mark.parametrize("token, expect", [("140", "F140M"), ("150", "F150W"),
                                           ("182", "F182M"), ("187", "F187N")])
def test_the_width_letter_comes_from_the_lookup(token, expect):
    """The suffix cannot be derived from the number: F150W but F140M."""
    assert expect in describe(f"Brick_RGB_444-{token}-090_hips")[1]


def test_323_is_the_narrowband():
    """Every published 323 channel is F323N -- arches/, quintuplet/ and the
    sgra NIRCam list (scripts/sgra_rgb_images.py) all observe it, and no
    published directory carries F322W2 data."""
    assert "F323N" in describe("ArchesQuintuplet_RGB_323-average-212_log_hips")[1]
    assert "F323N" in describe("SgrA_RGB_NIRCam_444-323-212_hips")[1]


@pytest.mark.parametrize("name", [
    "jwst_gc_treasury_hips_stale_20260912",   # superseded copy
    "GINSBURG_P_something_hips",              # hipsgen duplicate
    "heic1509a_transparent_hips",             # third-party press image
])
def test_names_we_do_not_publish_are_declined(name):
    assert describe(name) is None


def test_properties_for_carries_all_three_keys():
    p = properties_for("/some/build/tree/Brick_RGB_444-356-200_hips")
    assert p["creator_did"] == f"{AUTHORITY}/brick-rgb-444-356-200"
    assert p["hips_creator"] == CREATOR
    assert "F444W" in p["obs_title"]


def test_properties_for_passes_extras_through_and_lets_them_win():
    p = properties_for("Brick_RGB_444-356-200_hips", obs_title="mine",
                       hips_status="private")
    assert p["obs_title"] == "mine"
    assert p["hips_status"] == "private"


def test_an_unparsed_name_still_gets_an_id_under_our_authority():
    """A name no rule matches falls back to the directory name as the title:
    uninformative, and still ours and stable, which the uuid4 was not."""
    p = properties_for("whatever_hips")
    assert p["creator_did"] == f"{AUTHORITY}/whatever"
    assert p["obs_title"] == "whatever"


def test_a_declined_name_gets_no_identity():
    """Better an unidentified HiPS than one labelled as something else."""
    assert properties_for("heic1509a_transparent_hips") == {}


def test_slugify_is_url_safe():
    assert slugify("SgrA_RGB_MIRI_1500-1000-560_hips") == "sgra-rgb-miri-1500-1000-560"


def test_stamp_properties_rewrites_in_place(tmp_path):
    d = tmp_path / "Brick_RGB_444-356-200_hips"
    d.mkdir()
    (d / "properties").write_text(
        "creator_did          = ivo://reproject/P/9d3c0e2a-1111-4000-8000-000000000000\n"
        "obs_title            = /orange/adamginsburg/jwst/brick/pngs_444\n"
        "hips_order           = 11\n")
    assert stamp_properties(str(d)) is True
    got = dict(ln.split("=", 1) for ln in
               (d / "properties").read_text().strip().split("\n"))
    got = {k.strip(): v.strip() for k, v in got.items()}
    assert got["creator_did"] == f"{AUTHORITY}/brick-rgb-444-356-200"
    assert "F444W" in got["obs_title"]
    assert got["hips_creator"] == CREATOR
    assert got["hips_order"] == "11"          # untouched
    assert stamp_properties(str(d)) is False  # idempotent


def test_stamp_properties_leaves_a_declined_tree_alone(tmp_path):
    d = tmp_path / "heic1509a_transparent_hips"
    d.mkdir()
    before = "creator_did          = ivo://reproject/P/x\n"
    (d / "properties").write_text(before)
    assert stamp_properties(str(d)) is False
    assert (d / "properties").read_text() == before


def test_a_staged_build_is_identified_by_its_published_name(tmp_path):
    """`<name>_hips.new` is a declined name; the published one is what counts."""
    assert properties_for("jwst_miri_hips.new") == {}
    p = properties_for("jwst_miri_hips.new", name="jwst_miri_hips")
    assert p["creator_did"].startswith(f"{AUTHORITY}/")
    assert ".new" not in p["creator_did"] + p["obs_title"]

    d = tmp_path / "jwst_miri_hips.new"
    d.mkdir()
    (d / "properties").write_text("creator_did = ivo://reproject/P/x\n")
    assert stamp_properties(str(d), name="jwst_miri_hips") is True
    assert ".new" not in (d / "properties").read_text()
