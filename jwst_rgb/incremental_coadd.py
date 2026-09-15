"""Append new layers to an existing HiPS coadd instead of rebuilding it.

Why this is possible at all: `reproject.hips.coadd_hips` does NO pyramid
recomputation.  Every per-observation HiPS already carries its own full
Norder0..N pyramid, so combining them is a pure per-tile file merge -- copy the
tile if the output has no tile at that (order, npix), alpha-composite if it
does.  Nothing downstream depends on tiles the new layer does not touch, so
there is no reason to walk the ones it does not.

What that costs today, on the GC Treasury NIRCam coadd:

    coadd                                  10,558 tiles
    one observation                           ~537 tiles
    observations                                 29
    tile operations per FULL rebuild         15,476

A full rebuild does ~15.5k tile operations to add ~537 tiles of new sky, and
that figure grows by ~537 with every observation that lands, permanently.
Appending is ~29x less work now and the gap widens.

Precedence
----------
`coadd_hips` composites as ``alpha_composite(new_input, accumulated)``, and PIL
composites its SECOND argument over its first, so the ACCUMULATED output wins:
the first directory to cover a tile is the one on top.  (The coadd_hips
docstring says the opposite -- "the last image in the order of
input_directories is used" -- but the code and a direct test both say first.)

That is what makes appending exact rather than approximate: putting a layer in
last, under everything already there, is precisely what a full rebuild with
that layer at the end of the list produces.  Same output, less work.

What this CANNOT do
-------------------
Update or remove a layer.  Once tiles are composited the individual
contributions are gone, so a layer whose tiles changed cannot be subtracted
out.  `plan_coadd` detects that and asks for a full rebuild instead.  An AVM
correction applied across every observation, for instance, is a full rebuild
no matter what.
"""
import json
import os
import shutil
import time
import uuid

from PIL import Image

MANIFEST = "coadd_manifest.json"


def layer_fingerprint(directory, tile_format="png"):
    """Cheap identity for a layer's contribution: how many tiles and when they
    last changed.  Content hashing every tile would be exact but costs the walk
    this module exists to avoid; a changed tile always moves its mtime."""
    count = 0
    newest = 0.0
    for dirpath, _, filenames in os.walk(directory):
        for fn in filenames:
            if fn.endswith("." + tile_format):
                count += 1
                m = os.path.getmtime(os.path.join(dirpath, fn))
                if m > newest:
                    newest = m
    return {"tiles": count, "newest": round(newest, 3)}


def load_manifest(coadd_dir):
    path = os.path.join(coadd_dir, MANIFEST)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def save_manifest(coadd_dir, layers, tile_format="png"):
    data = {"order": [os.path.basename(d) for d in layers],
            "layers": {os.path.basename(d): layer_fingerprint(d, tile_format)
                       for d in layers}}
    with open(os.path.join(coadd_dir, MANIFEST), "w") as fh:
        json.dump(data, fh, indent=1)
    return data


def plan_coadd(coadd_dir, layers, tile_format="png"):
    """Decide between appending and rebuilding.

    Returns (action, new_layers, reason) where action is "none", "append" or
    "rebuild".  Anything other than a pure addition to the end of the existing
    set is a rebuild, because a flattened coadd cannot give a layer back.

    Deciding is O(all tiles): fingerprinting every known layer stats each of
    its tiles.  That is far cheaper than the open/composite/save a full rebuild
    does on the same tiles -- which is where the saving comes from -- but the
    decision itself does not scale better than the rebuild it avoids.
    """
    if not os.path.isdir(coadd_dir):
        return "rebuild", layers, "no existing coadd"
    manifest = load_manifest(coadd_dir)
    if manifest is None:
        return "rebuild", layers, "no manifest (coadd predates incremental support)"

    names = [os.path.basename(d) for d in layers]
    known = manifest.get("order", [])
    recorded = manifest.get("layers", {})

    dropped = [n for n in known if n not in names]
    if dropped:
        return "rebuild", layers, f"layer(s) no longer present: {', '.join(dropped)}"

    if names[:len(known)] != known:
        return "rebuild", layers, "existing layers reordered or inserted before the end"

    changed = []
    for n in known:
        d = next(x for x in layers if os.path.basename(x) == n)
        if layer_fingerprint(d, tile_format) != recorded.get(n):
            changed.append(n)
    if changed:
        return ("rebuild", layers,
                f"{len(changed)} existing layer(s) changed: "
                f"{', '.join(changed[:6])}{' ...' if len(changed) > 6 else ''}")

    new = [d for d in layers if os.path.basename(d) not in known]
    if not new:
        return "none", [], "coadd is up to date"
    return "append", new, f"{len(new)} new layer(s)"


def merge_layer(layer_dir, out_dir, tile_format="png"):
    """Merge one layer into an existing coadd tree, matching coadd_hips.

    Writes composites through a temp file and os.replace, so the output tree
    can safely be a hardlink copy of the previous coadd -- the original tiles
    are never mutated in place.
    """
    copied = composited = 0
    for dirpath, _, filenames in os.walk(layer_dir):
        for fn in filenames:
            if not fn.endswith("." + tile_format):
                continue
            src = os.path.join(dirpath, fn)
            tdir = os.path.join(out_dir, os.path.relpath(dirpath, layer_dir))
            os.makedirs(tdir, exist_ok=True)
            tgt = os.path.join(tdir, fn)
            if os.path.exists(tgt):
                if tile_format != "png":
                    raise NotImplementedError(
                        f"cannot composite {tile_format} tiles")
                # accumulated over new, exactly as coadd_hips does
                result = Image.alpha_composite(
                    Image.open(src).convert("RGBA"),
                    Image.open(tgt).convert("RGBA"))
                # format must be explicit: the temp name ends .tmp and PIL
                # infers the writer from the extension
                tmp = tgt + ".tmp"
                result.save(tmp, format=tile_format.upper())
                os.replace(tmp, tgt)
                composited += 1
            else:
                shutil.copyfile(src, tgt)
                copied += 1
    return copied, composited


def stamp_release_date(coadd_dir, when=None):
    """Set hips_release_date on a coadd to now.

    Neither coadd path does this on its own: coadd_hips writes
    ``reference_properties = all_properties[0]`` verbatim, so the coadd
    inherits the FIRST layer's date.  New observations sort last, so adding one
    left the date untouched.  publish_hips_layers.py decides with
    ``release_date(src) > release_date(dst)``, so a coadd carrying new sky
    reported "up to date" and was never published -- which would make this
    module's cheap append invisible to the viewers.
    """
    fn = os.path.join(coadd_dir, "properties")
    if not os.path.exists(fn):
        return None
    stamp = when or time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())
    lines, seen = [], False
    for ln in open(fn).read().rstrip("\n").split("\n"):
        if "=" in ln and ln.split("=", 1)[0].strip() == "hips_release_date":
            lines.append(f"{'hips_release_date':20s} = {stamp}")
            seen = True
        else:
            lines.append(ln)
    if not seen:
        lines.append(f"{'hips_release_date':20s} = {stamp}")
    open(fn, "w").write("\n".join(lines) + "\n")
    return stamp


#: Namespace for coadd dataset ids.  Fixed, so uuid5 of the same coadd name
#: gives the same id on every machine and every rebuild.
COADD_DID_NAMESPACE = uuid.UUID("6f0b6f0e-2f6a-5c8a-9c1e-0b2a7d4e5f31")


def coadd_did(name):
    """Stable ivo:// identifier for a coadd, derived from its directory name."""
    return f"ivo://reproject/P/{uuid.uuid5(COADD_DID_NAMESPACE, name)}"


def stamp_identity(coadd_dir, name=None):
    """Replace the inherited obs_title/creator_did with the coadd's own.

    `name` is the name the coadd will be published under, which is not always
    the directory's current name: the append path stages into
    ``<name>_hips.new`` and renames afterwards, and an id derived from the
    staging name would belong to a directory that never exists.  Both fields
    come from this one name, so the title and the id cannot drift apart.

    Returns the (obs_title, creator_did) written.  Call after coadd_hips and
    before publishing: a coadd sharing its first layer's creator_did is, to a
    HiPS client, the same dataset as that single field.
    """
    fn = os.path.join(coadd_dir, "properties")
    if not os.path.exists(fn):
        return None
    name = name or os.path.basename(os.path.normpath(coadd_dir))
    did = coadd_did(name)
    want = {"obs_title": name, "creator_did": did}
    lines, seen = [], set()
    for ln in open(fn).read().rstrip("\n").split("\n"):
        key = ln.split("=", 1)[0].strip() if "=" in ln else None
        if key in want:
            lines.append(f"{key:20s} = {want[key]}")
            seen.add(key)
        else:
            lines.append(ln)
    for key in want:
        if key not in seen:
            lines.append(f"{key:20s} = {want[key]}")
    open(fn, "w").write("\n".join(lines) + "\n")
    return name, did


def hardlink_tree(src, dst):
    """Clone a coadd cheaply.  Safe only because merge_layer never writes in
    place -- it replaces via a temp file, which breaks the link."""
    shutil.copytree(src, dst, copy_function=os.link)
