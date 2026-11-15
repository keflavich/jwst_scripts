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

Updating or removing a layer
----------------------------
Once tiles are composited the individual contributions are gone, so a layer
whose tiles changed cannot be subtracted out of the flattened tiles.  It does
not need to be: tiles are independent, and a layer touches only ~500 of the
~22,600 tiles of a coadd.  The manifest records each layer's tile list, so
`plan_coadd` can name exactly the tiles a changed or dropped layer ever
contributed to ("retile"), and `retile_coadd` recomputes those tiles from all
the layers in paint order and leaves the rest alone.  Same output as a full
rebuild, a small fraction of the work.

A full rebuild is still what happens when the manifest records no tile list
for a changed layer (coadds made before this existed), when the layers were
reordered, or when retiling would cost more than half what a rebuild does.
An AVM correction applied across every observation, for instance, is a full
rebuild no matter what.
"""
import json
import os
import shutil
import time
import uuid

from PIL import Image

from jwst_rgb.hips_naming import CREATOR, describe

MANIFEST = "coadd_manifest.json"


def layer_tiles(directory, tile_format="png"):
    """Walk one layer once: (sorted tile relpaths, newest mtime).

    The relpath is the tile's identity in the coadd tree: every layer carries
    its own full Norder*/Dir*/Npix* pyramid, and the coadd tile at a relpath is
    the composite of whichever layers have a file there.
    """
    files = []
    newest = 0.0
    for dirpath, _, filenames in os.walk(directory):
        for fn in filenames:
            if fn.endswith("." + tile_format):
                full = os.path.join(dirpath, fn)
                files.append(os.path.relpath(full, directory))
                m = os.path.getmtime(full)
                if m > newest:
                    newest = m
    files.sort()
    return files, newest


def layer_fingerprint(directory, tile_format="png"):
    """Cheap identity for a layer's contribution: how many tiles and when they
    last changed.  Content hashing every tile would be exact but costs the walk
    this module exists to avoid; a changed tile always moves its mtime."""
    files, newest = layer_tiles(directory, tile_format)
    return {"tiles": len(files), "newest": round(newest, 3)}


def load_manifest(coadd_dir):
    """The coadd's manifest, or None if it is missing or unusable.

    A kill mid-write leaves a truncated or empty file, and json.load on that
    raised out of order_layers and plan_coadd -- so the repair that
    coadd_repair_reason asked for died on the very file it had diagnosed.  A
    manifest that parses but is not a dict (``[]``, ``3``) or whose `order` /
    `layers` have the wrong type is as useless, and plan_coadd would raise
    AttributeError on it.  All of those read as "no manifest": a full rebuild.
    """
    path = os.path.join(coadd_dir, MANIFEST)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if not isinstance(data.get("order", []), list):
        return None
    if not isinstance(data.get("layers", {}), dict):
        return None
    return data


def recover_interrupted_swap(coadd_dir):
    """Undo a kill between `rename(out, out.old)` and `replace(out.new, out)`.

    The swap is two steps, and between them `coadd_dir` does not exist while
    `.new` (finished) and `.old` (the previous coadd) both do.  The next tick
    called that "no coadd directory", and the rebuild started by removing both
    trees.  Finish the swap when `.new` is complete (properties and manifest
    are written last), else put `.old` back.  Returns what it did, or None.
    """
    if os.path.exists(coadd_dir):
        return None
    stage, old = coadd_dir + ".new", coadd_dir + ".old"

    def has(d, name):
        f = os.path.join(d, name)
        return os.path.isfile(f) and os.path.getsize(f) > 0

    if has(stage, "properties") and has(stage, MANIFEST):
        os.replace(stage, coadd_dir)
        shutil.rmtree(old, ignore_errors=True)
        return "finished the swap from .new"
    if has(old, "properties"):
        os.rename(old, coadd_dir)
        return "restored .old"
    return None


def coadd_repair_reason(coadd_dir, layers):
    """Why `coadd_dir` is not a finished coadd of `layers`, or None if it is.

    A rebuild killed part-way used to leave a directory of Norder* tiles with
    no `properties`; plan_coadd would call that a rebuild, but --auto only
    asked plan_coadd when a tile had been built that same tick, so the broken
    coadd sat there and publish_hips_layers.py skipped it every hour.

    Cheap on purpose: three stats and one small JSON read, no tile walk.
    `layers` empty means there is nothing to coadd, so nothing to repair.
    Also true for a missing directory (a kill between the old rmtree and the
    first makedirs), and for a coadd that has properties but no readable
    manifest (a kill between coadd_hips and save_manifest).
    """
    if not layers:
        return None
    if not os.path.isdir(coadd_dir):
        return "no coadd directory"
    props = os.path.join(coadd_dir, "properties")
    if not os.path.isfile(props) or os.path.getsize(props) == 0:
        return "no readable properties"
    if load_manifest(coadd_dir) is None:
        if os.path.exists(os.path.join(coadd_dir, MANIFEST)):
            return "unreadable manifest"
        return "no manifest"
    return None


def newer_layer_reason(coadd_dir, layers):
    """Name a layer whose `properties` is newer than the coadd's manifest.

    A tick killed after build_obs but before the recoadd leaves a rebuilt layer
    the coadd has never seen.  One stat per layer, no tile walk.  Only
    meaningful for a coadd that passed coadd_repair_reason.  cmd_coadd's "none"
    verdict calls touch_manifest so a layer that changed without changing its
    tiles does not trigger this every tick.
    """
    path = os.path.join(coadd_dir, MANIFEST)
    if not os.path.exists(path):
        return None
    built = os.path.getmtime(path)
    for d in layers:
        props = os.path.join(d, "properties")
        if os.path.exists(props) and os.path.getmtime(props) > built:
            return f"{os.path.basename(d)} is newer than the coadd"
    return None


def touch_manifest(coadd_dir):
    """Record that the coadd was checked against its layers just now."""
    os.utime(os.path.join(coadd_dir, MANIFEST))


def coadd_needs_repair(coadd_dir, layers):
    return coadd_repair_reason(coadd_dir, layers) is not None


def save_manifest(coadd_dir, layers, tile_format="png"):
    """Record paint order and, per layer, its fingerprint and tile list.

    `files` is what lets a later tick retile instead of rebuild: once a layer
    has been re-rendered its old tile set is gone from disk, and the tiles it
    used to cover are exactly the ones that must be recomputed.  About 540
    strings per layer; `tiles` and `newest` stay, so older readers still work.
    """
    per_layer = {}
    for d in layers:
        files, newest = layer_tiles(d, tile_format)
        per_layer[os.path.basename(d)] = {
            "tiles": len(files), "newest": round(newest, 3), "files": files}
    data = {"order": [os.path.basename(d) for d in layers],
            "layers": per_layer}
    with open(os.path.join(coadd_dir, MANIFEST), "w") as fh:
        json.dump(data, fh, indent=1)
    return data


def backfill_manifest_files(coadd_dir, layers, tile_format="png"):
    """Add tile lists to a manifest written before `files` existed.

    Only valid when every layer still matches its recorded fingerprint (plan
    verdict "none"): an unchanged layer's tile list on disk is its list at the
    time of the coadd.  Without this a legacy coadd would stay legacy until
    something wrote a new manifest, and the first layer re-rendered in the
    meantime would force a full rebuild.  Returns True if it wrote.
    """
    manifest = load_manifest(coadd_dir)
    if manifest is None:
        return False
    recorded = manifest.get("layers", {})
    missing = [d for d in layers
               if "files" not in recorded.get(os.path.basename(d), {"files": 0})]
    if not missing:
        return False
    for d in missing:
        recorded[os.path.basename(d)]["files"] = layer_tiles(d, tile_format)[0]
    path = os.path.join(coadd_dir, MANIFEST)
    with open(path + ".tmp", "w") as fh:
        json.dump(manifest, fh, indent=1)
    os.replace(path + ".tmp", path)
    return True


def order_layers(coadd_dir, layers):
    """Put `layers` in the order this coadd already paints them, new ones last.

    Paint order used to be whatever `sorted()` gave, which for the treasury is
    obsid order.  A mosaic fills in by tile number, so a newly built obsid
    usually sorts into the MIDDLE of the existing set -- and `plan_coadd`
    requires the known layers to be a prefix of the list, because `merge_layer`
    can only add a layer at the bottom of the stack.  Every tick therefore read
    "existing layers reordered or inserted before the end" and did a full
    rebuild: four 53-layer NIRCam coadds, ~4 h, once an hour.

    Taking the order from the manifest instead makes arrival order the paint
    order, so a new layer is always an append.  It also keeps the two paths
    agreeing: a later `--full` rebuild now paints in the same order the
    appended coadd did, which `sorted()` could not promise once anything had
    been appended.

    A coadd with no manifest (or a layer the manifest has never seen) falls
    back to the name sort, so the first run after this lands is unchanged.
    """
    known = (load_manifest(coadd_dir) or {}).get("order", [])
    rank = {name: i for i, name in enumerate(known)}
    return sorted(layers, key=lambda d: (rank.get(os.path.basename(d),
                                                  len(known)),
                                         os.path.basename(d)))


#: `retile` must cost at most this fraction of a full rebuild's tile
#: operations, or the plan falls back to a rebuild.  Not measured on real data;
#: a batch of re-rendered layers touches many tiles, and past this point the
#: bookkeeping is not worth the risk of a subtle difference from coadd_hips.
RETILE_MAX_FRACTION = 0.5


def plan_coadd(coadd_dir, layers, tile_format="png",
               max_retile_fraction=RETILE_MAX_FRACTION):
    """Decide between doing nothing, appending, retiling and rebuilding.

    Returns (action, layers, reason, affected).  `action` is one of:

    none     coadd matches its layers.
    append   pure addition at the end; `layers` is the new ones.
    retile   known layers changed and/or were dropped (and/or new ones were
             added); `layers` is the full ordered list and `affected` is the
             sorted list of tile relpaths to recompute.
    rebuild  anything else; `layers` is the full list.

    `affected` is None except for retile.  A retile needs the manifest to
    record the old tile list of every changed or dropped layer (`files`);
    without it the tiles that layer used to cover are unknowable.

    Deciding is O(all tiles): every layer is walked once.  That is far cheaper
    than the open/composite/save a full rebuild does on the same tiles, but it
    does not scale better than the rebuild it avoids.
    """
    if not os.path.isdir(coadd_dir):
        return "rebuild", layers, "no existing coadd", None
    manifest = load_manifest(coadd_dir)
    if manifest is None:
        return ("rebuild", layers,
                "no manifest (coadd predates incremental support)", None)

    names = [os.path.basename(d) for d in layers]
    known = manifest.get("order", [])
    recorded = manifest.get("layers", {})

    dropped = [n for n in known if n not in names]
    kept = [n for n in known if n in names]
    if names[:len(kept)] != kept:
        return ("rebuild", layers,
                "existing layers reordered or inserted before the end", None)

    walked = {os.path.basename(d): layer_tiles(d, tile_format) for d in layers}
    changed = []
    for n in kept:
        files, newest = walked[n]
        rec = recorded.get(n, {})
        if len(files) != rec.get("tiles") or round(newest, 3) != rec.get("newest"):
            changed.append(n)

    new = [d for d in layers if os.path.basename(d) not in known]
    if not changed and not dropped:
        if not new:
            return "none", [], "coadd is up to date", None
        return "append", new, f"{len(new)} new layer(s)", None

    parts = []
    if changed:
        parts.append(f"{len(changed)} existing layer(s) changed: "
                     f"{', '.join(changed[:6])}{' ...' if len(changed) > 6 else ''}")
    if dropped:
        parts.append(f"layer(s) no longer present: {', '.join(dropped)}")
    what = "; ".join(parts)

    no_list = [n for n in changed + dropped if "files" not in recorded.get(n, {})]
    if no_list:
        return ("rebuild", layers,
                f"{what}; manifest has no tile list for {', '.join(no_list[:6])}",
                None)

    affected = set()
    for n in changed:
        affected.update(recorded[n]["files"])       # what it used to cover
        affected.update(walked[n][0])               # what it covers now
    for n in dropped:
        affected.update(recorded[n]["files"])
    for d in new:
        affected.update(walked[os.path.basename(d)][0])

    cover = {}
    for files, _ in walked.values():
        for rel in files:
            cover[rel] = cover.get(rel, 0) + 1
    rebuild_ops = sum(cover.values())
    # a tile no layer covers any more still costs one operation (the delete)
    retile_ops = sum(max(cover.get(rel, 0), 1) for rel in affected)
    if retile_ops > max_retile_fraction * rebuild_ops:
        return ("rebuild", layers,
                f"{what}; retiling {len(affected)} tiles is {retile_ops} of "
                f"{rebuild_ops} rebuild operations, over the "
                f"{max_retile_fraction:.0%} limit", None)
    return ("retile", layers,
            f"{what}; {len(affected)} tiles, {retile_ops} of {rebuild_ops} "
            f"rebuild operations", sorted(affected))


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


def _retile_one(rel, layers, out_dir, tile_format):
    """Recompute the coadd tile at `rel`: 'copied', 'composited' or 'removed'
    (or None if nothing covered it and nothing was there).

    Matches coadd_hips: the first layer covering the tile is copied as is, and
    each later one goes UNDER the accumulated tile, so the first layer in paint
    order ends up on top.  A tile covered by one layer is copied, not decoded
    and re-encoded, to stay byte-identical to a rebuild.
    """
    tgt = os.path.join(out_dir, rel)
    covering = [os.path.join(L, rel) for L in layers
                if os.path.isfile(os.path.join(L, rel))]
    if not covering:
        if os.path.lexists(tgt):
            os.remove(tgt)
            return "removed"
        return None
    os.makedirs(os.path.dirname(tgt), exist_ok=True)
    # always via a temp name: the tile may be a hardlink shared with the live
    # coadd, and writing through it would corrupt what is being served
    tmp = tgt + ".tmp"
    if len(covering) == 1:
        shutil.copyfile(covering[0], tmp)
        os.replace(tmp, tgt)
        return "copied"
    if tile_format != "png":
        raise NotImplementedError(f"cannot composite {tile_format} tiles")
    acc = Image.open(covering[0]).convert("RGBA")
    for src in covering[1:]:
        acc = Image.alpha_composite(Image.open(src).convert("RGBA"), acc)
    # format explicit: the temp name ends .tmp
    acc.save(tmp, format=tile_format.upper())
    os.replace(tmp, tgt)
    return "composited"


def _retile_chunk(args):
    rels, layers, out_dir, tile_format = args
    counts = {"copied": 0, "composited": 0, "removed": 0}
    for rel in rels:
        res = _retile_one(rel, layers, out_dir, tile_format)
        if res:
            counts[res] += 1
    return counts


def _prune_empty_dirs(out_dir, rels):
    """Remove directories a deleted tile left empty, up to (not including)
    out_dir.  coadd_hips never makes an empty Dir*, so neither should we."""
    root = os.path.normpath(out_dir)
    for rel in rels:
        d = os.path.dirname(os.path.normpath(os.path.join(out_dir, rel)))
        while d != root and os.path.isdir(d) and not os.listdir(d):
            os.rmdir(d)
            d = os.path.dirname(d)


def retile_coadd(stage_dir, layers_in_order, affected, tile_format="png",
                 workers=1):
    """Recompute only the `affected` tiles of a coadd from all its layers.

    `stage_dir` may be a hardlink clone of the live coadd: every write goes
    through a temp file and os.replace.  Lower-order tiles need no special
    case, because each layer carries its own full pyramid, so recomputing the
    same relpath at every order is the whole job.  Tiles covered by no layer any
    more are deleted and their empty directories pruned.

    `workers` > 1 spreads the tiles over a process pool (tiles are
    independent).  Returns (copied, composited, removed).
    """
    layers = list(layers_in_order)
    affected = list(affected)
    if workers > 1 and len(affected) > 1:
        from concurrent.futures import ProcessPoolExecutor
        n = min(workers, len(affected))
        chunks = [(affected[i::n], layers, stage_dir, tile_format)
                  for i in range(n)]
        with ProcessPoolExecutor(max_workers=n) as pool:
            results = list(pool.map(_retile_chunk, chunks))
    else:
        results = [_retile_chunk((affected, layers, stage_dir, tile_format))]
    _prune_empty_dirs(stage_dir, affected)
    return tuple(sum(r[k] for r in results)
                 for k in ("copied", "composited", "removed"))


def _read_hips_order(properties_path):
    if not os.path.isfile(properties_path):
        return None
    with open(properties_path) as fh:
        for ln in fh:
            if "=" in ln and ln.split("=", 1)[0].strip() == "hips_order":
                try:
                    return int(ln.split("=", 1)[1].strip())
                except ValueError:
                    return None
    return None


def set_hips_order(coadd_dir, layers):
    """Set the coadd's hips_order to the deepest of its layers'.

    coadd_hips does this on a full rebuild only; the append and retile paths
    hardlink the old `properties`, so a layer re-rendered deeper than the first
    layer would be truncated in the viewer.  Latent while every layer is at the
    same order.  Returns the order written, or None if no layer states one.
    """
    orders = [o for o in (_read_hips_order(os.path.join(d, "properties"))
                          for d in layers) if o is not None]
    fn = os.path.join(coadd_dir, "properties")
    if not orders or not os.path.exists(fn):
        return None
    top = max(orders)
    lines, seen = [], False
    for ln in open(fn).read().rstrip("\n").split("\n"):
        if "=" in ln and ln.split("=", 1)[0].strip() == "hips_order":
            lines.append(f"{'hips_order':20s} = {top}")
            seen = True
        else:
            lines.append(ln)
    if not seen:
        lines.append(f"{'hips_order':20s} = {top}")
    # properties may be a hardlink to the live coadd's: replace, don't write
    with open(fn + ".tmp", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(fn + ".tmp", fn)
    return top


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
    """Stable ivo:// identifier for a coadd, derived from its directory name.

    The uuid5 fallback for names `describe` does not recognise: unique and
    reproducible, but carrying no provenance, so it is the second choice.
    """
    described = describe(name)
    if described is not None:
        return described[0]
    return f"ivo://reproject/P/{uuid.uuid5(COADD_DID_NAMESPACE, name)}"


def stamp_identity(coadd_dir, name=None):
    """Replace the inherited obs_title/creator_did with the coadd's own.

    `name` is the name the coadd will be published under, which is not always
    the directory's current name: the append path stages into
    ``<name>_hips.new`` and renames afterwards, and an id derived from the
    staging name would belong to a directory that never exists.  Both fields
    come from this one name, so the title and the id cannot drift apart.

    The title is a description rather than the directory name: the HiPS
    network lists datasets by obs_title, and "jwst_gc_treasury_vminmax_hips"
    tells a user nothing about what they would be looking at.  CDS asked for
    both of these when we submitted the HiPS list (T. Boch, Sept 2026).

    Returns the (obs_title, creator_did) written.  Call after coadd_hips and
    before publishing: a coadd sharing its first layer's creator_did is, to a
    HiPS client, the same dataset as that single field.
    """
    fn = os.path.join(coadd_dir, "properties")
    if not os.path.exists(fn):
        return None
    name = name or os.path.basename(os.path.normpath(coadd_dir))
    described = describe(name)
    did = described[0] if described else coadd_did(name)
    title = described[1] if described else name
    want = {"obs_title": title, "creator_did": did,
            "hips_creator": CREATOR}
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
    return title, did


def hardlink_tree(src, dst):
    """Clone a coadd cheaply.  Safe only because merge_layer never writes in
    place -- it replaces via a temp file, which breaks the link."""
    shutil.copytree(src, dst, copy_function=os.link)
