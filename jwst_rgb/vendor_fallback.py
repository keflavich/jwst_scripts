"""Self-hosted-first fallback chains for CDS script tags.

``aladin.cds.unistra.fr`` went down on 2026-10-28: a TCP connect to it hung
for roughly 25 s and then was refused.  Every page that loaded Aladin Lite
or the HiPS landing-page template from a single CDS ``<script src="...">``
tag hung for that long on load and then rendered broken (``window.A`` /
``buildLandingPage`` left undefined) -- the sky view never showed, and
anything built on it broke too.

Verified same-build mirrors are kept at two UF-controlled hosts:

* primary: ``https://starformation.astro.ufl.edu/vendor/...``
* mirror:  ``https://data.rc.ufl.edu/pub/adamginsburg/vendor/...``

CDS is kept as the LAST source in both chains below -- it is the upstream
and usually works, and at least one existing test
(``tests/test_landing_page.py``) asserts the CDS URL still appears in a
generated landing page.

Every generator that loads Aladin Lite or the HiPS landing template from a
static ``<script>`` tag should use :data:`ALADIN_LITE_CHAIN` /
:data:`LANDING_CHAIN` instead of writing its own single-source tag, so the
chain lives in one place rather than being pasted per file.

See ``/vendor/README.txt`` on the starformation host for the mirrors'
provenance.
"""
import re

# --- Aladin Lite (classic viewer pages, e.g. HiPS cube landing pages) ------

#: The Aladin Lite build every mirror and CDS serve.  Pinned, not "latest":
#: the self-hosted copies are snapshots of this one version, so "latest"
#: cannot be offered consistently across the chain.
ALADIN_LITE_VERSION = "3.8.2"

ALADIN_LITE_SELFHOSTED = (
    f"https://starformation.astro.ufl.edu/vendor/aladin-lite/{ALADIN_LITE_VERSION}/aladin.js")
ALADIN_LITE_MIRROR = (
    f"https://data.rc.ufl.edu/pub/adamginsburg/vendor/aladin-lite/{ALADIN_LITE_VERSION}/aladin.js")
ALADIN_LITE_CDS = (
    f"https://aladin.cds.unistra.fr/AladinLite/api/v3/{ALADIN_LITE_VERSION}/aladin.js")

#: Left in the generated HTML -- both so a second run is a no-op and so a
#: grep can find every page carrying the chain.
ALADIN_LITE_MARKER = "aladin-fallback-chain"

#: Replaces a single ``<script src=".../aladin.js">`` tag.  Loads
#: :data:`ALADIN_LITE_SELFHOSTED`; if ``window.A`` is not set after that
#: (blocked, 404, offline host), falls through to :data:`ALADIN_LITE_MIRROR`,
#: then :data:`ALADIN_LITE_CDS`.  If none of the three set ``window.A``,
#: installs a stub ``A`` whose ``init`` promise never resolves and shows a
#: one-line notice, so the rest of the page keeps working instead of
#: throwing on the first ``A.whatever`` call.
ALADIN_LITE_CHAIN = (
    f'<!-- {ALADIN_LITE_MARKER}: Aladin Lite {ALADIN_LITE_VERSION} from self-hosted copies first, so the viewer keeps working when\n'
    '     aladin.cds.unistra.fr is down; CDS is the last source.  See /vendor/README.txt. -->\n'
    f'<script src="{ALADIN_LITE_SELFHOSTED}" charset="utf-8"></script>\n'
    f'<script>window.A || document.write(\'<script src="{ALADIN_LITE_MIRROR}" charset="utf-8"><\\/script>\');</script>\n'
    f'<script>window.A || document.write(\'<script src="{ALADIN_LITE_CDS}" charset="utf-8"><\\/script>\');</script>\n'
    '<script>\n'
    'if (!window.A) {\n'
    '  window.A = {init: new Promise(function () {}), unavailable: true};\n'
    '  document.addEventListener(\'DOMContentLoaded\', function () {\n'
    '    var d = document.createElement(\'div\');\n'
    '    d.setAttribute(\'role\', \'status\');\n'
    '    d.style.cssText = \'margin:8px;padding:8px 12px;border:1px solid #c90;background:#fff6e0;color:#000;font:14px sans-serif\';\n'
    '    d.textContent = \'The interactive sky viewer (Aladin Lite) could not be loaded: its servers are unreachable right now. The rest of this page still works; reload later to get the sky map back.\';\n'
    '    document.body.insertBefore(d, document.body.firstChild);\n'
    '  });\n'
    '}\n'
    '</script>'
)


# --- HiPS landing pages -----------------------------------------------------
#
# The HiPS landing page every builder writes (`jwst_rgb.landing_page`) loads
# the CDS landing-page template from a single
# ``<script src="https://aladin.cds.unistra.fr/hips-templates/hips-landing-page.js">``
# tag, which fails the same way: the page hangs while CDS is unreachable,
# then ``buildLandingPage`` is undefined and the landing page renders blank.
# The self-hosted template (``/vendor/hips-templates/``) also loads Aladin
# Lite from the self-hosted copies first and opens the viewer on the HiPS
# itself.

LANDING_TAG = re.compile(
    r"""[ \t]*<script[^>]*\ssrc=(["'])https?://aladin\.(?:cds\.unistra\.fr|u-strasbg\.fr)"""
    r"""/hips-templates/hips-landing-page\.js\1[^>]*>\s*</script>""")

#: Left in the rewritten page -- both so a second run is a no-op and so a
#: grep can find every landing page carrying the chain.
LANDING_MARKER = "landing-fallback-chain"

LANDING_JS_SELFHOSTED = "https://starformation.astro.ufl.edu/vendor/hips-templates/hips-landing-page.js"
LANDING_JS_MIRROR = "https://data.rc.ufl.edu/pub/adamginsburg/vendor/hips-templates/hips-landing-page.js"
LANDING_JS_CDS = "https://aladin.cds.unistra.fr/hips-templates/hips-landing-page.js"

LANDING_CHAIN = f"""    <!-- {LANDING_MARKER}: HiPS landing template from self-hosted copies first (it loads Aladin Lite from them too);
         CDS is the last source.  See /vendor/README.txt. -->
    <script src="{LANDING_JS_SELFHOSTED}"></script>
    <script>typeof buildLandingPage !== 'undefined' || document.write('<script src="{LANDING_JS_MIRROR}"><\\/script>');</script>
    <script>typeof buildLandingPage !== 'undefined' || document.write('<script src="{LANDING_JS_CDS}"><\\/script>');</script>
    <script>
    if (typeof buildLandingPage === 'undefined') {{
      window.buildLandingPage = function () {{
        document.addEventListener('DOMContentLoaded', function () {{
          document.body.textContent = 'The interactive HiPS viewer could not be loaded: its servers are unreachable right now. The HiPS itself (properties, Norder*/ tiles in this directory) is unaffected; reload later.';
        }});
      }};
    }}
    </script>"""
