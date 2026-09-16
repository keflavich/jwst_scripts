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
    changed, seen, unhandled = patch_tree(str(tmp_path))
    assert (changed, seen, unhandled) == (3, 3, [])
    changed, seen, unhandled = patch_tree(str(tmp_path))
    assert (changed, seen, unhandled) == (0, 3, [])


def test_a_custom_option_set_is_honoured(tmp_path):
    p = tmp_path / "index.html"
    p.write_text(AS_WRITTEN)
    patch_landing_page(str(p), {"showReticle": "false"})
    assert "aladinParams: {showReticle: false}" in p.read_text()


from jwst_rgb.landing_page import CSS_MARK, PANEL_CSS   # noqa: E402


def test_the_panel_css_is_injected(page):
    patch_landing_page(page)
    s = open(page).read()
    assert CSS_MARK in s
    assert ".box_v .menu-container { width: 0; }" in s


def test_the_css_lands_inside_head(page):
    patch_landing_page(page)
    s = open(page).read()
    assert s.index(CSS_MARK) < s.index("</head>")


def test_a_page_that_already_has_the_params_still_gets_the_css(tmp_path):
    """The 399 pages patched before the panel work were in exactly this state."""
    p = tmp_path / "index.html"
    p.write_text(AS_WRITTEN.replace(
        "buildLandingPage({",
        "buildLandingPage({aladinParams: {showSettingsControl: true}, "))
    assert patch_landing_page(str(p)) is True
    s = p.read_text()
    assert CSS_MARK in s
    assert s.count("aladinParams") == 1


def test_both_edits_are_idempotent_together(page):
    assert patch_landing_page(page) is True
    first = open(page).read()
    assert patch_landing_page(page) is False
    assert open(page).read() == first
    assert first.count(CSS_MARK) == 1
    assert first.count("aladinParams") == 1


def test_the_collapsed_rule_loses_to_the_templates_expanded_rule():
    """The drawer must still open.

    Template: `#hamburger:checked~.box_v .menu-container` is (1,2,0).
    Ours:     `.box_v .menu-container`                    is (0,2,0).
    An id beats two classes, so checking the box still widens the panel -- and
    that holds regardless of which <style> the browser sees first, which is why
    there is no !important here.
    """
    assert "!important" not in PANEL_CSS
    assert "#hamburger" not in PANEL_CSS      # we never restate the open state


def test_the_toggle_itself_is_never_hidden():
    """Zero-width plus overflow:hidden would take the disc with the strip.

    The label is a child of body, so it survives; the background-color moves
    the disc onto it.  Hiding the label instead would make the panel
    unreachable.
    """
    assert "label.hamburger" in PANEL_CSS
    assert "display: none" not in PANEL_CSS


# --- the call shape the docroot actually uses ------------------------------

BARE = """\
<!DOCTYPE html>
<html>
<head>
    <script src="https://aladin.cds.unistra.fr/hips-templates/hips-landing-page.js"></script>
</head>
<body></body>
<script type="text/javascript">
    buildLandingPage();
</script>
</html>
"""


def test_the_bare_call_form_is_patched(tmp_path):
    """53 published pages call buildLandingPage() with no options object.

    The regex matched only buildLandingPage({, so those pages were skipped --
    and skipped via the same False that means "already patched", so a run over
    the docroot reported a tidy number and said nothing about them.
    """
    p = tmp_path / "index.html"
    p.write_text(BARE)
    assert patch_landing_page(str(p)) is True
    s = p.read_text()
    assert "buildLandingPage({aladinParams: {showSettingsControl: true}})" in s
    assert CSS_MARK in s


def test_the_bare_form_is_idempotent(tmp_path):
    p = tmp_path / "index.html"
    p.write_text(BARE)
    assert patch_landing_page(str(p)) is True
    first = p.read_text()
    assert patch_landing_page(str(p)) is False
    assert p.read_text() == first


def test_an_unknown_call_shape_raises_rather_than_skipping(tmp_path):
    """Silence is what made the 53 invisible.

    A page that names buildLandingPage in a form this cannot edit has to say
    so; returning False would file it alongside "nothing to do".
    """
    p = tmp_path / "index.html"
    p.write_text(BARE.replace("buildLandingPage();",
                              "buildLandingPage(options);"))
    with pytest.raises(ValueError, match="call shape"):
        patch_landing_page(str(p))


def test_patch_tree_reports_what_it_could_not_do(tmp_path):
    for name, body in (("a_hips", BARE),
                       ("b_hips", BARE.replace("buildLandingPage();",
                                               "buildLandingPage(options);"))):
        d = tmp_path / name
        d.mkdir()
        (d / "index.html").write_text(body)
    changed, seen, unhandled = patch_tree(str(tmp_path))
    assert (changed, seen) == (1, 2)
    assert len(unhandled) == 1 and "b_hips" in unhandled[0]
