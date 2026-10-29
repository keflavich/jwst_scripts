"""Fallback chains (aladin.cds.unistra.fr outage, 2026-10-28).

`jwst_rgb.vendor_fallback` holds both chains: Aladin Lite itself (CDS first,
self-hosted copies on failure or after a per-source timeout) and the patched
HiPS landing-page template (self-hosted first).  The landing-page chain is
exercised end-to-end via `patch_landing_page` in `test_landing_page.py`;
these tests cover the Aladin Lite chain -- used directly by
`hips_formats._cube_index_html` -- including a replay of the generated
loader under node against a hanging CDS.
"""
import json
import re
import shutil
import subprocess

import pytest

from jwst_rgb import hips_formats
from jwst_rgb.vendor_fallback import (
    ALADIN_LITE_CDS,
    ALADIN_LITE_CHAIN,
    ALADIN_LITE_MARKER,
    ALADIN_LITE_MIRROR,
    ALADIN_LITE_SELFHOSTED,
    ALADIN_LITE_TIMEOUT_MS,
    ALADIN_LITE_URLS,
    ALADIN_LITE_VERSION,
    LANDING_CHAIN,
)


def test_aladin_lite_chain_is_cds_first_then_self_hosted():
    assert ALADIN_LITE_CHAIN.count(ALADIN_LITE_MARKER) == 1
    cds = ALADIN_LITE_CHAIN.index(ALADIN_LITE_CDS)
    selfhosted = ALADIN_LITE_CHAIN.index(ALADIN_LITE_SELFHOSTED)
    mirror = ALADIN_LITE_CHAIN.index(ALADIN_LITE_MIRROR)
    assert cds < selfhosted < mirror
    assert ALADIN_LITE_URLS == (ALADIN_LITE_CDS, ALADIN_LITE_SELFHOSTED, ALADIN_LITE_MIRROR)


def test_aladin_lite_chain_has_no_parser_blocking_tag():
    """A `<script src>` tag (or a document.write of one) blocks the parser
    for as long as its host takes to answer -- ~30 s when CDS hangs."""
    assert "<script src=" not in ALADIN_LITE_CHAIN
    assert "document.write" not in ALADIN_LITE_CHAIN


def test_aladin_lite_chain_is_pinned_to_3_8_2_not_3_7_2_beta():
    assert ALADIN_LITE_VERSION == "3.8.2"
    assert "3.7.2-beta" not in ALADIN_LITE_CHAIN


def _cube_html(**props):
    base = {"obs_title": "test cube", "obs_description": "a test"}
    base.update(props)
    return hips_formats._cube_index_html(base, depth=3, crval3=0.0,
                                         cdelt3=1.0, bunit3="mag")


def test_cube_page_tries_cds_before_the_self_hosted_copy():
    page = _cube_html()
    assert page.count(ALADIN_LITE_MARKER) == 1
    assert page.index(ALADIN_LITE_CDS) < page.index(ALADIN_LITE_SELFHOSTED)


def test_cube_page_has_no_lingering_3_7_2_beta_reference():
    assert "3.7.2-beta" not in _cube_html()


def test_cube_page_has_no_bare_aladin_js_script_tag():
    """The old single `<script src=".../3.7.2-beta/aladin.js">` tag must be
    gone outright, not left alongside the new chain; the chain's own CDS
    entry (3.8.2) is still present, as the first source."""
    page = _cube_html()
    assert 'src="https://aladin.cds.unistra.fr/AladinLite/api/v3/3.7.2-beta/aladin.js"' not in page
    assert '"https://aladin.cds.unistra.fr/AladinLite/api/v3/3.8.2/aladin.js"' in page


def test_landing_chain_ends_with_a_guarded_stub():
    """When every template source fails, the landing chain installs a stub
    behind a guard on the symbol the sources define, after the last source."""
    l_guard = LANDING_CHAIN.index("if (typeof buildLandingPage === 'undefined') {")
    assert l_guard > LANDING_CHAIN.rindex("aladin.cds.unistra.fr")
    assert "window.buildLandingPage = function" in LANDING_CHAIN[l_guard:]


# ------------------------------------------ the loader, replayed under node
#
# A fake DOM and a fake clock: each `<script>` the loader appends is answered
# from a per-URL script -- load (sets window.A) or error after N ms, or hang
# (never answers).  Microtasks are flushed between timer events, so the
# promise chain behaves as it would in a browser.

_HARNESS = r"""
const vm = require('vm');
let input = '';
process.stdin.on('data', d => { input += d; });
process.stdin.on('end', main);
async function main() {
  const {js, net} = JSON.parse(input);
  let now = 0, seq = 0;
  const timers = [], requested = [], warnings = [];
  let notice = false;
  const setT = (fn, ms) => { seq += 1; timers.push({at: now + ms, fn, id: seq}); return seq; };
  const clearT = id => { const i = timers.findIndex(t => t.id === id); if (i >= 0) { timers.splice(i, 1); } };
  const ctx = {console: {warn: m => warnings.push(String(m)), log() {}},
               setTimeout: setT, clearTimeout: clearT};
  const body = {
    firstChild: null,
    insertBefore(el) { el.parentNode = body; notice = true; },
    removeChild(el) { el.parentNode = null; notice = false; },
  };
  ctx.document = {
    body,
    createElement: tag => ({tag, style: {}, setAttribute() {}, parentNode: null}),
    addEventListener(ev, fn) { if (ev === 'DOMContentLoaded') { setT(fn, 0); } },
    head: {appendChild(s) {
      requested.push({url: s.src, t: now});
      const r = net[s.src] || {kind: 'hang'};
      if (r.kind === 'load') {
        setT(() => {
          ctx.A = {name: r.name, init: new Promise(res => { setT(res, r.init_ms || 0); })};
          s.onload();
        }, r.ms);
      } else if (r.kind === 'error') {
        setT(() => s.onerror(), r.ms);
      }
    }},
  };
  vm.createContext(ctx);
  ctx.window = ctx;
  ctx.clock = () => now;
  vm.runInContext(js, ctx);
  vm.runInContext('var ready = null, readyAt = null;'
                  + ' A.init.then(function () { ready = A.name; readyAt = clock(); });', ctx);
  for (let step = 0; step < 1000; step++) {
    await new Promise(r => setImmediate(r));
    if (!timers.length) { break; }
    timers.sort((a, b) => a.at - b.at || a.id - b.id);
    const t = timers.shift();
    now = t.at;
    t.fn();
  }
  await new Promise(r => setImmediate(r));
  console.log(JSON.stringify({ready: ctx.ready, readyAt: ctx.readyAt, final: (ctx.A && ctx.A.name) || null,
                              unavailable: Boolean(ctx.A && ctx.A.unavailable),
                              requested, notice, warnings}));
}
"""


def _replay(net):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    js = re.search(r"<script>\n(.*)\n</script>", ALADIN_LITE_CHAIN, re.S).group(1)
    out = subprocess.run([node, "-e", _HARNESS], input=json.dumps({"js": js, "net": net}),
                         capture_output=True, text=True, timeout=60, check=True)
    return json.loads(out.stdout)


def test_page_code_waits_for_the_library_init():
    """`A.init.then(...)` written against the stub runs only once the loaded
    library's own `A.init` has resolved (its WebAssembly start-up), not as
    soon as the bundle has loaded."""
    res = _replay({ALADIN_LITE_CDS: {'kind': 'load', 'ms': 300, 'init_ms': 500, 'name': 'cds'}})
    assert res['ready'] == 'cds'
    assert res['readyAt'] == 800


def test_loader_moves_on_when_cds_hangs():
    """The 2026-10-28 failure: CDS never answers.  The self-hosted copy is
    requested at the timeout, not after the browser's own connect timeout."""
    res = _replay({ALADIN_LITE_SELFHOSTED: {'kind': 'load', 'ms': 200, 'name': 'self'}})
    assert res['ready'] == 'self'
    assert [r['url'] for r in res['requested']] == [ALADIN_LITE_CDS, ALADIN_LITE_SELFHOSTED]
    assert res['requested'][1]['t'] == ALADIN_LITE_TIMEOUT_MS
    assert not res['notice']


def test_loader_moves_on_at_once_when_cds_errors():
    res = _replay({ALADIN_LITE_CDS: {'kind': 'error', 'ms': 50},
                   ALADIN_LITE_SELFHOSTED: {'kind': 'load', 'ms': 200, 'name': 'self'}})
    assert res['ready'] == 'self'
    assert res['requested'][1]['t'] == 50


def test_late_cds_does_not_replace_the_loaded_library():
    """CDS answers after its timeout, once the self-hosted copy is running:
    the late bundle still executes and reassigns window.A, and the loader
    must put the first library back."""
    res = _replay({ALADIN_LITE_CDS: {'kind': 'load', 'ms': ALADIN_LITE_TIMEOUT_MS + 900, 'name': 'cds'},
                   ALADIN_LITE_SELFHOSTED: {'kind': 'load', 'ms': 200, 'name': 'self'}})
    assert res['ready'] == 'self'
    assert res['final'] == 'self'


def test_late_cds_that_beats_the_fallback_is_used():
    """If CDS answers late but before the self-hosted copy, CDS is the first
    library loaded and is kept; the mirror is never requested."""
    res = _replay({ALADIN_LITE_CDS: {'kind': 'load', 'ms': ALADIN_LITE_TIMEOUT_MS + 100, 'name': 'cds'},
                   ALADIN_LITE_SELFHOSTED: {'kind': 'load', 'ms': 900, 'name': 'self'}})
    assert res['ready'] == 'cds'
    assert res['final'] == 'cds'
    assert ALADIN_LITE_MIRROR not in [r['url'] for r in res['requested']]


def test_late_cds_stops_the_chain():
    """CDS answers late while the self-hosted copy hangs: the self-hosted
    timeout firing afterwards must not go on to fetch the mirror."""
    res = _replay({ALADIN_LITE_CDS: {'kind': 'load', 'ms': ALADIN_LITE_TIMEOUT_MS + 100, 'name': 'cds'}})
    assert res['ready'] == 'cds'
    assert [r['url'] for r in res['requested']] == [ALADIN_LITE_CDS, ALADIN_LITE_SELFHOSTED]


def test_loader_reaches_the_mirror():
    res = _replay({ALADIN_LITE_MIRROR: {'kind': 'load', 'ms': 100, 'name': 'mirror'}})
    assert res['ready'] == 'mirror'
    assert [r['url'] for r in res['requested']] == list(ALADIN_LITE_URLS)


def test_a_source_answering_after_the_notice_clears_it():
    """Every source timed out, so the notice is up; CDS then answers.  The
    page uses it and the notice goes away."""
    res = _replay({ALADIN_LITE_CDS: {'kind': 'load', 'ms': 3 * ALADIN_LITE_TIMEOUT_MS + 500, 'name': 'cds'}})
    assert res['ready'] == 'cds'
    assert res['final'] == 'cds'
    assert not res['notice']


def test_loader_gives_up_with_a_notice_and_a_pending_init():
    res = _replay({url: {'kind': 'error', 'ms': 10} for url in ALADIN_LITE_URLS})
    assert res['ready'] is None
    assert res['unavailable']
    assert res['notice']


def test_landing_tag_match_ignores_case():
    from jwst_rgb.vendor_fallback import LANDING_TAG
    tag = ('<SCRIPT SRC="https://aladin.cds.unistra.fr/hips-templates/'
           'hips-landing-page.js"></SCRIPT>')
    assert LANDING_TAG.search(tag)
