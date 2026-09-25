"""A coadd must not carry its first input layer's dataset identity.

coadd_hips copies ``all_properties[0]`` verbatim, so before stamp_identity the
34-field mosaic and the single o098 field shared one creator_did -- the IVOA
dataset identifier a HiPS client keys a layer on -- and every mosaic was
labelled "GCTreasury_o098_..." in the layer list.

The identity is also what the HiPS network publishes: `creator_did` names the
dataset under our own authority and `obs_title` is what a user reads in the
layer list, so both come from `hips_naming.describe` rather than from the
directory name.
"""
import os

import pytest

from jwst_rgb.hips_naming import CREATOR, describe
from jwst_rgb.incremental_coadd import coadd_did, stamp_identity

INHERITED = """\
creator_did          = ivo://reproject/P/f2f7993f-43e9-4cee-a814-e9159ce8af5f
obs_title            = GCTreasury_o098_RGB_480-mean-212_hips
hips_version         = 1.4
hips_tile_format     = png
"""


def props(d):
    return dict(
        (ln.split("=", 1)[0].strip(), ln.split("=", 1)[1].strip())
        for ln in open(os.path.join(d, "properties")).read().splitlines()
        if "=" in ln)


@pytest.fixture
def coadd(tmp_path):
    d = tmp_path / "jwst_gc_treasury_hips"
    d.mkdir()
    (d / "properties").write_text(INHERITED)
    return str(d)


def test_title_and_did_become_the_coadds_own(coadd):
    title, did = stamp_identity(coadd)
    assert did == coadd_did("jwst_gc_treasury_hips")
    p = props(coadd)
    assert p["obs_title"] == title
    assert p["creator_did"] == did
    assert "o098" not in p["creator_did"] + p["obs_title"]


def test_the_identity_is_the_described_one(coadd):
    """Not the directory name: that is what the HiPS network would show."""
    title, did = stamp_identity(coadd)
    assert (did, title) == describe("jwst_gc_treasury_hips")
    assert did.startswith("ivo://UFL/P/")
    assert title != "jwst_gc_treasury_hips"
    assert "Galactic Center" in title
    assert props(coadd)["hips_creator"] == CREATOR


def test_an_undescribed_name_still_gets_a_stable_id(tmp_path):
    """Names describe() declines (a superseded copy) keep the uuid5 id."""
    d = tmp_path / "jwst_gc_treasury_hips_stale_20260912b"
    d.mkdir()
    (d / "properties").write_text(INHERITED)
    assert describe(d.name) is None
    title, did = stamp_identity(str(d))
    assert title == d.name
    assert did == coadd_did(d.name)
    assert did.startswith("ivo://reproject/P/")


def test_other_keys_are_left_alone(coadd):
    stamp_identity(coadd)
    p = props(coadd)
    assert p["hips_version"] == "1.4"
    assert p["hips_tile_format"] == "png"


def test_the_id_is_stable_across_rebuilds(coadd):
    first = stamp_identity(coadd)
    # a rebuild starts from the inherited properties again
    with open(os.path.join(coadd, "properties"), "w") as fh:
        fh.write(INHERITED)
    assert stamp_identity(coadd) == first


def test_different_flavours_get_different_ids():
    names = ["jwst_gc_treasury_hips", "jwst_gc_treasury_vminmax_hips",
             "jwst_gc_treasury_log_hips", "jwst_gc_treasury_miri_hips"]
    assert len({coadd_did(n) for n in names}) == len(names)


def test_staged_append_is_stamped_with_its_final_name(tmp_path):
    """The append path stages into <name>.new and renames afterwards."""
    stage = tmp_path / "jwst_gc_treasury_hips.new"
    stage.mkdir()
    (stage / "properties").write_text(INHERITED)
    title, did = stamp_identity(str(stage), "jwst_gc_treasury_hips")
    assert (did, title) == describe("jwst_gc_treasury_hips")
    assert ".new" not in did + title


def test_missing_properties_is_not_an_error(tmp_path):
    assert stamp_identity(str(tmp_path)) is None


def test_keys_are_added_when_absent(tmp_path):
    d = tmp_path / "jwst_gc_treasury_log_hips"
    d.mkdir()
    (d / "properties").write_text("hips_version         = 1.4\n")
    stamp_identity(str(d))
    p = props(str(d))
    assert p["obs_title"] == describe("jwst_gc_treasury_log_hips")[1]
    assert p["creator_did"] == coadd_did("jwst_gc_treasury_log_hips")
