"""A layer that fails the orientation gate never reaches the path coadd() reads.

starsub_rgb_layers.build() stages each layer in <name>.new/ and swaps it over
<name>/ only when check_orientation returns True (jwst_scripts#41, PR #53
review).  coadd() takes every <name>/<name>_hips with a properties file, so a
failed build left there would be folded into jwst_nir_starsub_hips.
"""
import json
import os

import pytest

import starsub_rgb_layers as S

NAME = "Test_starsub_RGB_1-2-3"


def _writer(tag):
    def write(stage):
        hips = os.path.join(stage, f"{NAME}_hips")
        os.makedirs(os.path.join(hips, "Norder3"))
        with open(os.path.join(hips, "properties"), "w") as fh:
            fh.write(f"build = {tag}\n")
        open(os.path.join(stage, f"{NAME}.png"), "w").close()
    return write


def _props(out):
    with open(os.path.join(out, NAME, f"{NAME}_hips", "properties")) as fh:
        return fh.read()


@pytest.mark.parametrize("verdict", [False, None])
def test_failed_rebuild_keeps_last_verified_layer(tmp_path, verdict):
    out = str(tmp_path)
    S.stage_verify_promote(NAME, _writer("good"), lambda h, g: True,
                           {"f1": "a.fits"}, "a.fits", out=out)
    assert "build = good" in _props(out)
    with pytest.raises(RuntimeError, match="unchanged"):
        S.stage_verify_promote(NAME, _writer("bad"), lambda h, g: verdict,
                               {"f1": "a.fits"}, "a.fits", out=out)
    # coadd() reads <OUT>/<name>/<name>_hips: still the verified build.
    assert "build = good" in _props(out)
    with open(os.path.join(out, NAME, f"{NAME}_inputs.json")) as fh:
        assert json.load(fh)["orientation_ok"] is True
    # The unverified build is kept for inspection, with its verdict recorded.
    with open(os.path.join(out, f"{NAME}.new", f"{NAME}_inputs.json")) as fh:
        assert json.load(fh)["orientation_ok"] is verdict


def test_failed_first_build_leaves_nothing_for_coadd(tmp_path):
    out = str(tmp_path)
    with pytest.raises(RuntimeError):
        S.stage_verify_promote(NAME, _writer("bad"), lambda h, g: False,
                               {}, "a.fits", out=out)
    assert not os.path.exists(os.path.join(out, NAME))


def test_passing_rebuild_replaces_layer(tmp_path):
    out = str(tmp_path)
    for tag in ("one", "two"):
        S.stage_verify_promote(NAME, _writer(tag), lambda h, g: True,
                               {}, "a.fits", out=out)
    assert "build = two" in _props(out)
    assert sorted(os.listdir(out)) == [NAME]
