"""Every published HiPS landing page must expose the reticle toggle.

Aladin Lite has the toggle already -- a checkbox with colour and size beside
it -- inside the settings control, which Aladin Lite defaults to off and the
CDS landing page template never enables.  Without the injected option the
viewer renders a reticle the user cannot turn off.
"""
import os

import pytest

from jwst_rgb.landing_page import patch_hips_dir, patch_landing_page, patch_tree

AS_WRITTEN = """\
<!DOCTYPE html>
<html>
<head>
    <script src="https://aladin.cds.unistra.fr/hips-templates/hips-landing-page.js"></script>
</head>
<body></body>
<script type="text/javascript">
    buildLandingPage({alScriptURL: 'https://aladin.cds.unistra.fr/AladinLite/api/v3/3.7.2-beta/aladin.js'});
</script>
</html>
"""


@pytest.fixture
def page(tmp_path):
    p = tmp_path / "index.html"
    p.write_text(AS_WRITTEN)
    return str(p)


def test_the_option_is_added(page):
    assert patch_landing_page(page) is True
    s = open(page).read()
    assert "aladinParams: {showSettingsControl: true}" in s


def test_the_original_options_survive(page):
    patch_landing_page(page)
    s = open(page).read()
    assert "alScriptURL: 'https://aladin.cds.unistra.fr" in s
    assert s.count("buildLandingPage(") == 1


def test_it_is_idempotent(page):
    assert patch_landing_page(page) is True
    first = open(page).read()
    assert patch_landing_page(page) is False
    assert open(page).read() == first


def test_a_page_without_the_template_is_left_alone(tmp_path):
    p = tmp_path / "index.html"
    p.write_text("<html><body>a hand-written viewer</body></html>")
    before = p.read_text()
    assert patch_landing_page(str(p)) is False
    assert p.read_text() == before


def test_patch_hips_dir_handles_a_hips_without_a_landing_page(tmp_path):
    d = tmp_path / "some_hips"
    (d / "Norder3").mkdir(parents=True)
    assert patch_hips_dir(str(d)) is False


def test_patch_hips_dir_patches_one_directory(tmp_path):
    d = tmp_path / "some_hips"
    d.mkdir()
    (d / "index.html").write_text(AS_WRITTEN)
    assert patch_hips_dir(str(d)) is True
    assert "showSettingsControl" in (d / "index.html").read_text()


def test_patch_tree_walks_every_hips(tmp_path):
    for name in ("a_hips", "b_hips", "c_hips"):
        d = tmp_path / name
        d.mkdir()
        (d / "index.html").write_text(AS_WRITTEN)
    (tmp_path / "notes.html").write_text(AS_WRITTEN)   # not an index.html
    changed, seen = patch_tree(str(tmp_path))
    assert (changed, seen) == (3, 3)
    changed, seen = patch_tree(str(tmp_path))
    assert (changed, seen) == (0, 3)


def test_a_custom_option_set_is_honoured(tmp_path):
    p = tmp_path / "index.html"
    p.write_text(AS_WRITTEN)
    patch_landing_page(str(p), {"showReticle": "false"})
    assert "aladinParams: {showReticle: false}" in p.read_text()
