"""Fallback chains for CDS script tags: CDS first for Aladin Lite, with a
per-source timeout; self-hosted first for the patched HiPS landing template.

``aladin.cds.unistra.fr`` went down on 2026-10-28: a TCP connect to it hung
for roughly 25 s and then was refused.  Every page that loaded Aladin Lite
or the HiPS landing-page template from a single CDS ``<script src="...">``
tag hung for that long on load and then rendered broken (``window.A`` /
``buildLandingPage`` left undefined) -- the sky view never showed, and
anything built on it broke too.

Verified same-build mirrors are kept at two UF-controlled hosts:

* primary: ``https://starformation.astro.ufl.edu/vendor/...``
* mirror:  ``https://data.rc.ufl.edu/pub/adamginsburg/vendor/...``

Aladin Lite: CDS stays the DEFAULT, first source; the self-hosted copies are
tried only when CDS fails to load or does not answer within
:data:`ALADIN_LITE_TIMEOUT_MS`.  The timeout is what makes a CDS-first order
safe: a source whose connect HANGS never fires ``onerror`` until the
browser's own ~30 s connect timeout, and a parser-inserted ``<script src>``
blocks the page for all of it.  So :data:`ALADIN_LITE_CHAIN` loads every
source asynchronously and keeps the first library that loads (a source that
answers after its timeout still executes; its ``window.A`` is undone).

HiPS landing template: our copy is a PATCHED version of the CDS template (it
loads Aladin Lite CDS first with the same per-source timeout, and opens the
viewer on the HiPS itself), so :data:`LANDING_CHAIN` loads it from the
self-hosted copies first (data.rc, then starformation) and the stock CDS
template last.

Every generator that loads Aladin Lite or the HiPS landing template from a
static ``<script>`` tag should use :data:`ALADIN_LITE_CHAIN` /
:data:`LANDING_CHAIN` instead of writing its own single-source tag, so the
chain lives in one place rather than being pasted per file.  Both are kept
byte-identical to ``jwst_gc_pipeline.aladin_loader`` in jwst-gc-pipeline.

At least one existing test (``tests/test_landing_page.py``) asserts the CDS
URL appears in a generated landing page.

See ``/vendor/README.txt`` on the starformation host for the mirrors'
provenance.
"""
import json
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

#: Ordered CDS first (the default), then the self-hosted copies.
ALADIN_LITE_URLS = (ALADIN_LITE_CDS, ALADIN_LITE_SELFHOSTED, ALADIN_LITE_MIRROR)

#: How long a source may take to load before the next one is tried.  Matches
#: the per-source timeout in the self-hosted landing template.
ALADIN_LITE_TIMEOUT_MS = 8000

#: Left in the generated HTML -- both so a second run is a no-op and so a
#: grep can find every page carrying the chain.
ALADIN_LITE_MARKER = "aladin-fallback-chain"

_NOTICE = ("The interactive sky viewer (Aladin Lite) could not be loaded: its "
           "servers are unreachable right now. The rest of this page still "
           "works; reload later to get the sky map back.")

_LOADER_JS = """(function () {
  var sources = %(sources)s;
  var timeoutMs = %(timeout)d;
  var stub = {unavailable: false};
  var settle;
  stub.init = new Promise(function (resolve) { settle = resolve; });
  window.A = stub;
  var winner = null;
  var notice = null;
  function giveUp() {
    stub.unavailable = true;
    var show = function () {
      notice = document.createElement('div');
      notice.setAttribute('role', 'status');
      notice.style.cssText = 'margin:8px;padding:8px 12px;border:1px solid #c90;background:#fff6e0;color:#000;font:14px sans-serif';
      notice.textContent = %(notice)s;
      document.body.insertBefore(notice, document.body.firstChild);
    };
    if (document.body) { show(); } else { document.addEventListener('DOMContentLoaded', show); }
  }
  function tryFrom(i) {
    if (i >= sources.length) { giveUp(); return; }
    var moved = false;
    function next(why) {
      if (moved || winner) { return; }
      moved = true;
      console.warn('Aladin Lite: ' + why + ' ' + sources[i] + '; trying the next source');
      tryFrom(i + 1);
    }
    var timer = setTimeout(function () {
      next('no answer within ' + timeoutMs + ' ms from');
    }, timeoutMs);
    var s = document.createElement('script');
    s.charset = 'utf-8';
    s.onerror = function () { clearTimeout(timer); next('failed to load'); };
    s.onload = function () {
      clearTimeout(timer);
      if (winner) {
        // a source that answered after its timeout: keep the first library
        window.A = winner;
        return;
      }
      if (!window.A || window.A === stub) { next('no Aladin Lite in'); return; }
      winner = window.A;
      if (notice && notice.parentNode) { notice.parentNode.removeChild(notice); }
      Promise.resolve(winner.init).then(function () { settle(); });
    };
    s.src = sources[i];
    document.head.appendChild(s);
  }
  tryFrom(0);
})();"""

#: Replaces a single ``<script src=".../aladin.js">`` tag.  Loads
#: :data:`ALADIN_LITE_CDS` asynchronously; if it fails, or has not loaded
#: within :data:`ALADIN_LITE_TIMEOUT_MS`, moves on to
#: :data:`ALADIN_LITE_SELFHOSTED`, then :data:`ALADIN_LITE_MIRROR`.  Until a
#: source has loaded, ``window.A`` is a stub whose ``init`` promise resolves
#: once the loaded library's own ``A.init`` has, so ``A.init.then(...)``
#: written after this block runs against the real library.  If no source
#: loads, ``A.init`` never resolves and a one-line notice is shown.
ALADIN_LITE_CHAIN = (
    f'<!-- {ALADIN_LITE_MARKER}: Aladin Lite {ALADIN_LITE_VERSION} from CDS first; when CDS fails or does not answer within\n'
    f'     {ALADIN_LITE_TIMEOUT_MS // 1000} s, from the self-hosted copies.  See /vendor/README.txt. -->\n'
    '<script>\n'
    + _LOADER_JS % {"sources": json.dumps(list(ALADIN_LITE_URLS)),
                    "timeout": ALADIN_LITE_TIMEOUT_MS,
                    "notice": json.dumps(_NOTICE)}
    + '\n</script>'
)


# --- HiPS landing pages -----------------------------------------------------
#
# The HiPS landing page every builder writes (`jwst_rgb.landing_page`) loads
# the CDS landing-page template from a single
# ``<script src="https://aladin.cds.unistra.fr/hips-templates/hips-landing-page.js">``
# tag, which fails the same way: the page hangs while CDS is unreachable,
# then ``buildLandingPage`` is undefined and the landing page renders blank.
# The template is a patched copy of the CDS one, so the chain loads it from
# the self-hosted copies first and CDS last.  The data.rc copy is first: it
# is served straight from the /orange tree where the template is maintained,
# while the starformation copy is synced from it by hand and can lag (on
# 2026-10-28 it still loaded Aladin Lite self-hosted first).  The template
# (``/vendor/hips-templates/``) then loads Aladin Lite in the same order as
# :data:`ALADIN_LITE_URLS` -- CDS first, with the same per-source timeout --
# and opens the viewer on the HiPS itself.

LANDING_TAG = re.compile(
    r"""[ \t]*<script[^>]*\ssrc=(["'])https?://aladin\.(?:cds\.unistra\.fr|u-strasbg\.fr)"""
    r"""/hips-templates/hips-landing-page\.js\1[^>]*>\s*</script>""",
    re.IGNORECASE)

#: Left in the rewritten page -- both so a second run is a no-op and so a
#: grep can find every landing page carrying the chain.
LANDING_MARKER = "landing-fallback-chain"

LANDING_JS_SELFHOSTED = "https://starformation.astro.ufl.edu/vendor/hips-templates/hips-landing-page.js"
LANDING_JS_MIRROR = "https://data.rc.ufl.edu/pub/adamginsburg/vendor/hips-templates/hips-landing-page.js"
LANDING_JS_CDS = "https://aladin.cds.unistra.fr/hips-templates/hips-landing-page.js"

LANDING_CHAIN = f"""    <!-- {LANDING_MARKER}: patched HiPS landing template from self-hosted copies first (data.rc, then
         starformation), CDS template last; the template loads Aladin Lite from CDS first, self-hosted copies
         on failure.  See /vendor/README.txt. -->
    <script src="{LANDING_JS_MIRROR}"></script>
    <script>typeof buildLandingPage !== 'undefined' || document.write('<script src="{LANDING_JS_SELFHOSTED}"><\\/script>');</script>
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
