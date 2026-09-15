"""Turn on Aladin Lite's settings control in a HiPS landing page.

The reticle toggle is already in Aladin Lite -- it is a checkbox, with colour
and size beside it, inside the settings control.  Aladin Lite defaults that
control to off (``showSettingsControl: !1`` in the 3.7.2 bundle), and the CDS
landing page template that reproject writes builds its viewer with

    {showZoomControl, realFullscreen, showContextMenu, showCooGridControl,
     showSimbadPointerControl}

and never mentions it.  So every HiPS we publish renders a viewer whose reticle
cannot be turned off.

The template accepts ``aladinParams``, which it copies over its own defaults
before calling ``A.aladin``, so one key restores the control.  Injecting it into
the generated page keeps us on the CDS template -- the alternative, writing our
own viewer page, means owning a copy of their layout for one checkbox.
"""
import glob
import os
import re

#: Aladin Lite options added to every landing page we publish.
ALADIN_PARAMS = {"showSettingsControl": "true"}

_CALL = re.compile(r"buildLandingPage\(\s*\{")

#: Marker so the injected block is recognised on a later run and not repeated.
CSS_MARK = "gc-treasury landing page overrides"

#: Collapse the left panel by default; see the module docstring for why these
#: selectors are safe against the stylesheet the template injects at runtime.
PANEL_CSS = f"""<style>
/* {CSS_MARK} */
/* The collapsed drawer is 100px of chrome plus a blue disc over the image.
   Zero-width hides both; the hamburger stays, because it is a child of body
   rather than of the container. */
.box_v .menu-container {{ width: 0; }}
/* The disc the container used to draw, moved onto the label that opens it. */
label.hamburger {{ background-color: #1377d6; }}
</style>
"""


def params_snippet(params=None):
    return ", ".join(f"{k}: {v}" for k, v in (params or ALADIN_PARAMS).items())


def patch_landing_page(path, params=None, css=True):
    """Apply our landing page overrides to one index.html.

    Two independent edits: the aladinParams that expose Aladin Lite's settings
    control, and a stylesheet that collapses the left panel.  Either may
    already be present -- a page built before one of them was added has the
    other -- so each is applied on its own and the return says whether
    anything changed.

    Idempotent, so this can run on every build without accumulating duplicates.
    """
    with open(path) as fh:
        s = fh.read()
    if "buildLandingPage" not in s:
        return False
    before = s
    if not all(k in s for k in (params or ALADIN_PARAMS)):
        s, n = _CALL.subn(
            "buildLandingPage({aladinParams: {%s}, " % params_snippet(params),
            s, count=1)
        if not n:
            return False
    if css and CSS_MARK not in s:
        if "</head>" in s:
            s = s.replace("</head>", PANEL_CSS + "</head>", 1)
        else:
            # no head to land in; the template builds the body itself, so
            # anywhere before the call works
            s = PANEL_CSS + s
    if s == before:
        return False
    with open(path, "w") as fh:
        fh.write(s)
    return True


def patch_hips_dir(hips_dir, params=None):
    """Patch the landing page of one HiPS directory, if it has one."""
    fn = os.path.join(hips_dir, "index.html")
    return os.path.exists(fn) and patch_landing_page(fn, params)


def patch_tree(root, params=None):
    """Patch every landing page under `root`.  Returns (changed, seen)."""
    changed = seen = 0
    for fn in sorted(glob.glob(os.path.join(root, "**", "index.html"),
                               recursive=True)):
        seen += 1
        changed += bool(patch_landing_page(fn, params))
    return changed, seen
