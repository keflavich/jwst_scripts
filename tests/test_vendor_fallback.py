"""Self-hosted-first fallback chains (aladin.cds.unistra.fr outage, 2026-10-28).

`jwst_rgb.vendor_fallback` holds both chains (Aladin Lite itself, and the
HiPS landing-page template that also loads Aladin Lite internally).  The
landing-page chain is exercised end-to-end via `patch_landing_page` in
`test_landing_page.py`; these tests cover the Aladin Lite chain -- used
directly by `hips_formats._cube_index_html` -- and the shared chain
constant's own shape.
"""
from jwst_rgb import hips_formats
from jwst_rgb.vendor_fallback import (
    ALADIN_LITE_CDS,
    ALADIN_LITE_CHAIN,
    ALADIN_LITE_MARKER,
    ALADIN_LITE_MIRROR,
    ALADIN_LITE_SELFHOSTED,
    ALADIN_LITE_VERSION,
    LANDING_CHAIN,
)


def test_aladin_lite_chain_is_self_hosted_first_cds_last():
    assert ALADIN_LITE_CHAIN.count(ALADIN_LITE_MARKER) == 1
    selfhosted = ALADIN_LITE_CHAIN.index(ALADIN_LITE_SELFHOSTED)
    mirror = ALADIN_LITE_CHAIN.index(ALADIN_LITE_MIRROR)
    cds = ALADIN_LITE_CHAIN.index(ALADIN_LITE_CDS)
    assert selfhosted < mirror < cds


def test_aladin_lite_chain_is_pinned_to_3_8_2_not_3_7_2_beta():
    assert ALADIN_LITE_VERSION == "3.8.2"
    assert "3.7.2-beta" not in ALADIN_LITE_CHAIN


def _cube_html(**props):
    base = {"obs_title": "test cube", "obs_description": "a test"}
    base.update(props)
    return hips_formats._cube_index_html(base, depth=3, crval3=0.0,
                                         cdelt3=1.0, bunit3="mag")


def test_cube_page_loads_the_self_hosted_copy_before_cds():
    page = _cube_html()
    assert page.count(ALADIN_LITE_MARKER) == 1
    selfhosted = page.index(ALADIN_LITE_SELFHOSTED)
    cds = page.index(ALADIN_LITE_CDS)
    assert selfhosted < cds


def test_cube_page_has_no_lingering_3_7_2_beta_reference():
    assert "3.7.2-beta" not in _cube_html()


def test_cube_page_has_exactly_one_bare_aladin_js_script_tag_removed():
    """The old single `<script src=".../3.7.2-beta/aladin.js">` tag must be
    gone outright, not left alongside the new chain."""
    page = _cube_html()
    assert 'src="https://aladin.cds.unistra.fr/AladinLite/api/v3/3.7.2-beta/aladin.js"' not in page
    # the chain's own CDS entry (3.8.2) is still present, as the last source
    assert 'src="https://aladin.cds.unistra.fr/AladinLite/api/v3/3.8.2/aladin.js"' in page


def test_chains_end_with_a_guarded_stub():
    """When every source fails, each chain installs a stub behind a guard
    on the symbol the sources define, placed after the last source."""
    a_guard = ALADIN_LITE_CHAIN.index("if (!window.A) {")
    assert a_guard > ALADIN_LITE_CHAIN.rindex("aladin.cds.unistra.fr")
    assert "init: new Promise(function () {})" in ALADIN_LITE_CHAIN[a_guard:]
    l_guard = LANDING_CHAIN.index("if (typeof buildLandingPage === 'undefined') {")
    assert l_guard > LANDING_CHAIN.rindex("aladin.cds.unistra.fr")
    assert "window.buildLandingPage = function" in LANDING_CHAIN[l_guard:]
