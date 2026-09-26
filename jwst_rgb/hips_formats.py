"""HiPS layouts that `reproject.hips` does not write: 1.0 cubes and catalogues.

Both follow IVOA HiPS 1.0 (REC-HIPS-1.0-20170519):

cube (sec. 4.2.3, 4.3)
    Each frame is an ordinary image tile carrying a ``_<f>`` suffix; the
    unsuffixed name is frame 0, so a client that knows nothing about cubes
    still shows the first frame.  ``Allsky`` follows the same rule.  This is
    NOT reproject's HiPS3D, which tiles the third axis too, needs a spectral
    WCS, and is not what Aladin's frame slider reads.

    Aladin Lite (checked 3.7.2-beta through 3.9.0-beta) departs from this,
    so a cube with ``em_min``/``em_max`` is laid out for Aladin Lite:

    - it aborts on a cube without ``em_min``/``em_max``, which it uses to map
      a frequency onto a frame;
    - its cube slider (the "Player" box) turns slice s into the wavelength
      ``em_min + s/depth * (em_max - em_min)`` and that frequency back into
      a tile suffix ``trunc((f - f_lo) / (f_hi - f_lo) * depth)``.  Over a
      narrow band this reverses the order and sends slice 0 to the edge,
      suffix ``_<depth>`` by that arithmetic and ``_<depth-1>`` in the
      browser (3.7.2-beta, headless Chromium), so slice 0 gets both.  `aladin_slice_suffixes` repeats that arithmetic and each
      frame is written under the suffix its slider slice asks for, so slider
      slice s shows frame s.  The unsuffixed name stays frame 0, for clients
      that know nothing about cubes.  A standard HiPS 1.0 cube reader will
      see the frames out of order; without ``em_range`` the standard layout
      is kept.

catalog (sec. 4.2.2, 4.4.3, 6.3.2)
    UTF-8 TSV tiles; the first non-comment line is the column names; ``#``
    lines are comments.  Sources are distributed down the hierarchy: a tile at
    order K keeps the first N sources of its cell in priority order and the
    rest go to its four children at K+1.  Column descriptions live in a
    VOTable, ``metadata.xml``.
"""
import glob
import html
import json
import os
import shutil
import time

import numpy as np


def _dir_of(npix):
    return (npix // 10000) * 10000


def _read_properties(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def _write_properties(path, props):
    with open(path, "w") as fh:
        for k, v in props.items():
            fh.write(f"{k:<20} = {v}\n")


# --------------------------------------------------------------------------
# cube


C_LIGHT = 299792458.0  # m/s


def aladin_slice_suffixes(em_range, depth):
    """Tile suffix Aladin Lite requests for each slice of its cube slider.

    Repeats ``setSliceNumber`` (aladin.js) and ``channel_idx`` (d3/mod.rs):
    slice s -> wavelength ``em_min + s / depth * (em_max - em_min)`` ->
    frequency f -> suffix ``trunc((f - f_lo) / (f_hi - f_lo) * depth)``.
    Slice 0 sits on the upper edge, x = depth exactly; the browser was seen
    to ask for ``depth - 1`` there, so `assemble_hips_cube` writes frame 0
    under both.  Raises if two slices would share a suffix or a later slice
    lands within 1e-3 of a suffix boundary, where rounding in the browser
    could decide it; a narrow `em_range` (a few per cent) avoids both.
    """
    em_min, em_max = min(em_range), max(em_range)
    f_lo, f_hi = C_LIGHT / em_max, C_LIGHT / em_min
    suffixes = []
    for s in range(depth):
        freq = C_LIGHT / (em_min + s / depth * (em_max - em_min))
        x = (freq - f_lo) / (f_hi - f_lo) * depth
        k = max(int(x), 0)
        if s > 0 and min(x - k, k + 1 - x) < 1e-3:
            raise ValueError(f"slice {s} lands at {x:.4f}, on a frame "
                             "boundary; use a narrower em_range")
        suffixes.append(k)
    if len(set(suffixes)) != depth or depth - 1 in suffixes[1:]:
        raise ValueError(f"em_range {em_range} maps slices {list(range(depth))}"
                         f" onto frames {suffixes}; use a narrower em_range")
    return suffixes


def assemble_hips_cube(frame_dirs, out_dir, crval3=0.0, cdelt3=1.0,
                       bunit3="", pixel_cut=None, extra=None, em_range=None):
    """Merge per-frame image HiPS into one HiPS 1.0 cube at `out_dir`.

    Every frame must have been built on the same grid at the same order: the
    frames of a cube share tile indices, and a frame built one order shallower
    would leave the slider showing blank cells at the deepest order.  That is
    checked, not assumed.

    `pixel_cut` should span ALL frames: the properties copied from frame 0
    would otherwise stretch every frame to frame 0's range, and a cube read
    frame-by-frame at one stretch is the point of making it a cube.

    `em_range` is a (min, max) wavelength in m, written as
    ``em_min``/``em_max``.  Aladin Lite cannot open a cube without it; with
    it, the tiles are laid out so Aladin Lite's cube slider shows frame s at
    slice s, and the cube gets an ``index.html`` that opens that slider (see
    the module docstring).  Keep it narrow: `aladin_slice_suffixes` refuses a
    range over which the slider would skip or repeat frames.
    """
    if not frame_dirs:
        raise ValueError("no frames")
    props0 = _read_properties(os.path.join(frame_dirs[0], "properties"))
    for d in frame_dirs[1:]:
        p = _read_properties(os.path.join(d, "properties"))
        for key in ("hips_order", "hips_tile_format", "hips_frame"):
            if p.get(key) != props0.get(key):
                raise ValueError(f"{d}: {key}={p.get(key)!r} differs from "
                                 f"frame 0's {props0.get(key)!r}")

    depth = len(frame_dirs)
    if em_range is None:
        suffixes = list(range(depth))
    else:
        suffixes = aladin_slice_suffixes(em_range, depth)

    stage = out_dir.rstrip("/") + ".new"
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage)
    for f, src in enumerate(frame_dirs):
        names = [f"_{suffixes[f]}"]
        if f == 0:
            names.append("")
            if em_range is not None and depth > 1:
                names.append(f"_{depth - 1}")
        for path in glob.glob(os.path.join(src, "Norder*", "**", "*"),
                              recursive=True):
            if os.path.isdir(path):
                continue
            rel = os.path.relpath(path, src)
            stem, ext = os.path.splitext(rel)
            base = os.path.basename(stem)
            if not (base.startswith("Npix") or base == "Allsky"):
                continue
            os.makedirs(os.path.join(stage, os.path.dirname(rel)),
                        exist_ok=True)
            for suffix in names:
                shutil.copy2(path, os.path.join(stage, stem + suffix + ext))
    for aux in ("Moc.fits", "index.html", "metadata.fits"):
        if os.path.exists(os.path.join(frame_dirs[0], aux)):
            shutil.copy2(os.path.join(frame_dirs[0], aux),
                         os.path.join(stage, aux))

    props = dict(props0)
    props.update({
        "dataproduct_type": "cube",
        "hips_cube_depth": str(depth),
        "hips_cube_firstframe": "0",
        "data_cube_crpix3": "1",
        "data_cube_crval3": repr(float(crval3)),
        "data_cube_cdelt3": repr(float(cdelt3)),
        "data_cube_bunit3": bunit3,
        "hips_release_date": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()),
    })
    if pixel_cut is not None:
        props["hips_pixel_cut"] = f"{pixel_cut[0]} {pixel_cut[1]}"
    if em_range is not None:
        props["em_min"] = repr(float(min(em_range)))
        props["em_max"] = repr(float(max(em_range)))
    props.update(extra or {})
    _write_properties(os.path.join(stage, "properties"), props)
    if em_range is not None:
        with open(os.path.join(stage, "index.html"), "w") as fh:
            fh.write(_cube_index_html(props, depth, crval3, cdelt3, bunit3))

    if not os.path.isdir(os.path.join(stage, "Norder3")):
        raise RuntimeError(f"{stage}: no Norder3; refusing to swap in")
    _swap_in(stage, out_dir)
    return out_dir


ALADIN_LITE = "https://aladin.cds.unistra.fr/AladinLite/api/v3/3.7.2-beta/aladin.js"


def _cube_index_html(props, depth, crval3, cdelt3, bunit3):
    """A landing page that opens the cube with a slider over its frames.

    The slider calls the layer's ``setSliceNumber``, the call behind Aladin
    Lite's own cube Player, so it shows what the Player shows for this cube.
    The box copies the Player's layout and adds the frame's range as a label.
    """
    labels = []
    for k in range(depth):
        center = crval3 + k * cdelt3
        lo, hi = center - cdelt3 / 2, center + cdelt3 / 2
        labels.append(f"{lo:g} to {hi:g}" + (f" {bunit3}" if bunit3 else ""))
    title = html.escape(props.get("obs_title", "HiPS cube"))
    desc = html.escape(props.get("obs_description", ""))
    start = depth // 2
    target = f'{props.get("hips_initial_ra", "0")} {props.get("hips_initial_dec", "0")}'
    fov = props.get("hips_initial_fov", "1")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script src="{ALADIN_LITE}"></script>
<style>
  :root {{ --bg: #ffffff; --fg: #1a1a1a; --muted: #555555; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg: #16181c; --fg: #e8e8e8; --muted: #a0a0a0; }}
  }}
  body {{ margin: 0; padding: 0 16px 16px; background: var(--bg);
         color: var(--fg); font: 15px/1.4 system-ui, sans-serif; }}
  h1 {{ font-size: 1.2em; margin: 12px 0 4px; }}
  p {{ margin: 4px 0; color: var(--muted); max-width: 70em; }}
  #aladin {{ width: 100%; height: 75vh; min-height: 320px; }}
  .player {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: center;
            color: #fff; font-size: 13px; }}
  .player input {{ width: min(300px, 50vw); }}
  .player output {{ font-variant-numeric: tabular-nums; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p>{desc}</p>
<p>Step through the frames with the Player box on the map.  The same cube
opened in any Aladin Lite (HiPS browser, paste this page's folder URL) gets
Aladin's own Player, which shows the same frames.</p>
<div id="aladin"></div>
<script>
  const LABELS = {json.dumps(labels)};
  const DEPTH = {depth};
  A.init.then(() => {{
    const aladin = A.aladin("#aladin", {{
      target: "{target}", fov: {fov},
      showCooGridControl: true, showSimbadPointerControl: false }});
    window.aladin = aladin;
    const layer = A.imageHiPS(new URL(".", window.location.href).href, {{
      name: document.title,
      successCallback: (hips) => {{
        let slice = {start};
        const row = document.createElement("div");
        row.className = "player";
        const prev = document.createElement("button");
        const next = document.createElement("button");
        const range = document.createElement("input");
        const out = document.createElement("output");
        prev.textContent = "<"; next.textContent = ">";
        range.type = "range"; range.min = 0; range.max = DEPTH - 1;
        range.setAttribute("aria-label", "Frame");
        const show = () => {{
          range.value = slice;
          out.textContent = (slice + 1) + "/" + DEPTH + ": " + LABELS[slice];
          hips.setSliceNumber(slice);
        }};
        prev.onclick = () => {{ slice = Math.max(slice - 1, 0); show(); }};
        next.onclick = () => {{ slice = Math.min(slice + 1, DEPTH - 1); show(); }};
        range.oninput = () => {{ slice = +range.value; show(); }};
        row.append(prev, next, range, out);
        aladin.addUI(A.box({{
          close: false, header: {{ title: "Player for: " + document.title,
                                   draggable: true }},
          content: row, position: {{ anchor: "center top" }} }}));
        show();
      }} }});
    aladin.setBaseImageLayer(layer);
  }});
</script>
</body>
</html>
"""


def _swap_in(stage, dest):
    old = dest.rstrip("/") + ".old"
    shutil.rmtree(old, ignore_errors=True)
    if os.path.isdir(dest):
        os.rename(dest, old)
    os.rename(stage, dest)
    shutil.rmtree(old, ignore_errors=True)


# --------------------------------------------------------------------------
# catalogue


def distribute(ra, dec, priority, n_per_tile=500, order_min=1, order_max=14):
    """Assign each source an (order, npix) in the catalogue hierarchy.

    Sources are taken in ascending `priority` (brightest first when priority
    is a magnitude).  At each order, every cell keeps its first `n_per_tile`
    unassigned sources; the rest fall through to the next order.  Everything
    still unassigned at `order_max` lands there, so no source is dropped --
    the deepest tiles are simply fuller than `n_per_tile`.

    Returns (order, npix) arrays aligned with the inputs.  NESTED indexing,
    ICRS, which is what hips_frame = equatorial means.
    """
    from astropy_healpix import HEALPix
    import astropy.units as u
    from astropy.coordinates import ICRS

    ra = np.asarray(ra, float)
    dec = np.asarray(dec, float)
    n = len(ra)
    order = np.full(n, -1, dtype=np.int16)
    npix = np.full(n, -1, dtype=np.int64)
    # Stable sort: ties keep input order, so a rebuild from the same input
    # writes byte-identical tiles.
    rank_order = np.argsort(np.asarray(priority, float), kind="stable")
    todo = rank_order
    for k in range(order_min, order_max + 1):
        if len(todo) == 0:
            break
        hp = HEALPix(nside=2 ** k, order="nested", frame=ICRS())
        cell = hp.lonlat_to_healpix(ra[todo] * u.deg, dec[todo] * u.deg)
        if k == order_max:
            order[todo], npix[todo] = k, cell
            break
        # rank within cell, preserving priority order
        by_cell = np.argsort(cell, kind="stable")
        c_sorted = cell[by_cell]
        start = np.r_[0, np.flatnonzero(np.diff(c_sorted)) + 1]
        first = np.repeat(start, np.diff(np.r_[start, len(c_sorted)]))
        rank = np.empty(len(todo), dtype=np.int64)
        rank[by_cell] = np.arange(len(todo)) - first
        keep = rank < n_per_tile
        order[todo[keep]], npix[todo[keep]] = k, cell[keep]
        todo = todo[~keep]
    return order, npix


def write_hips_catalogue(out_dir, columns, ra_col="ra", dec_col="dec",
                         priority=None, n_per_tile=500, order_min=1,
                         order_max=14, units=None, ucds=None,
                         descriptions=None, formats=None, properties=None):
    """Write a HiPS catalogue from `columns` ({name: array}, all one length).

    `formats` maps a column to a %-format for the TSV (floats default to %.6g
    except the position columns, %.7f -- 0.1" at the equator is 2.8e-5 deg,
    so 6 significant figures of an RA near 266 would be ~4" wide).

    Staged to ``<out_dir>.new`` and swapped in only once Norder`order_min`
    exists, like every other layer here.
    """
    import astropy.units as u
    from astropy.table import Table

    names = list(columns)
    nrow = len(columns[names[0]])
    for k in names:
        if len(columns[k]) != nrow:
            raise ValueError(f"column {k} has {len(columns[k])} rows, "
                             f"expected {nrow}")
    ra = np.asarray(columns[ra_col], float)
    dec = np.asarray(columns[dec_col], float)
    if priority is None:
        priority = np.arange(nrow)
    order, npix = distribute(ra, dec, priority, n_per_tile=n_per_tile,
                             order_min=order_min, order_max=order_max)

    formats = dict(formats or {})
    formats.setdefault(ra_col, "%.7f")
    formats.setdefault(dec_col, "%.7f")
    text_cols = []
    for k in names:
        a = np.asarray(columns[k])
        if a.dtype.kind == "f":
            fmt = formats.get(k, "%.6g")
            s = np.char.mod(fmt, a)
            s[~np.isfinite(a)] = ""            # empty field = null (4.2.2.1)
        elif a.dtype.kind in "iub":
            s = np.char.mod(formats.get(k, "%d"), a.astype(np.int64))
        else:
            s = a.astype(str)
        text_cols.append(s.astype(object))
    lines = text_cols[0]
    for s in text_cols[1:]:
        lines = lines + "\t" + s
    header = "\t".join(names) + "\n"

    stage = out_dir.rstrip("/") + ".new"
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage)
    # Group rows by (order, npix) once, then write each tile in priority
    # order so the brightest source of a tile is its first row.
    rank = np.empty(nrow, dtype=np.int64)
    rank[np.argsort(np.asarray(priority, float), kind="stable")] = np.arange(nrow)
    key = np.lexsort((rank, npix, order))
    o_s, p_s = order[key], npix[key]
    bounds = np.r_[0, np.flatnonzero((np.diff(o_s) != 0) |
                                     (np.diff(p_s) != 0)) + 1, nrow]
    per_order = {}
    for a, b in zip(bounds[:-1], bounds[1:]):
        k, p = int(o_s[a]), int(p_s[a])
        d = os.path.join(stage, f"Norder{k}", f"Dir{_dir_of(p)}")
        os.makedirs(d, exist_ok=True)
        body = "\n".join(lines[key[a:b]]) + "\n"
        with open(os.path.join(d, f"Npix{p}.tsv"), "w") as fh:
            fh.write(f"# Completeness = {b - a} / {nrow}\n")
            fh.write(header)
            fh.write(body)
        per_order.setdefault(k, []).append(body)
    # Clients load the shallow orders whole (6.3.2, "Allsky usage").
    for k in range(order_min, min(order_min + 2, order_max + 1)):
        if k in per_order:
            with open(os.path.join(stage, f"Norder{k}", "Allsky.tsv"), "w") as fh:
                fh.write(header)
                fh.write("".join(per_order[k]))

    t = Table()
    for k in names:
        a = np.asarray(columns[k])
        t[k] = a[:0]
        if units and k in units:
            t[k].unit = u.Unit(units[k])
        if descriptions and k in descriptions:
            t[k].description = descriptions[k]
    from astropy.io.votable import from_table
    vot = from_table(t)
    for fld in vot.get_first_table().fields:
        if ucds and fld.name in ucds:
            fld.ucd = ucds[fld.name]
    # The spec names it metadata.xml; Aladin has historically asked for
    # Metadata.xml.  Case matters on this server, so write both.
    vot.to_xml(os.path.join(stage, "metadata.xml"))
    shutil.copy2(os.path.join(stage, "metadata.xml"),
                 os.path.join(stage, "Metadata.xml"))

    kmax = int(order.max())
    try:
        from mocpy import MOC
    except ImportError:
        MOC = None
    if MOC is not None:
        moc = MOC.from_lonlat(ra * u.deg, dec * u.deg, max_norder=min(kmax, 12))
        moc.save(os.path.join(stage, "Moc.fits"), format="fits", overwrite=True)

    ctr_ra, ctr_dec, fov = _view(ra, dec)
    props = {
        "dataproduct_type": "catalog",
        "hips_version": "1.4",
        "hips_release_date": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()),
        "hips_status": "public master clonableOnce",
        "hips_frame": "equatorial",
        "hips_order": str(kmax),
        "hips_order_min": str(order_min),
        "hips_tile_format": "tsv",
        "hips_cat_nrows": str(nrow),
        "hips_initial_ra": repr(ctr_ra),
        "hips_initial_dec": repr(ctr_dec),
        "hips_initial_fov": repr(fov),
        "hips_builder": "jwst_rgb.hips_formats",
    }
    props.update(properties or {})
    _write_properties(os.path.join(stage, "properties"), props)

    if not os.path.isdir(os.path.join(stage, f"Norder{order_min}")):
        raise RuntimeError(f"{stage}: no Norder{order_min}; refusing to swap in")
    _swap_in(stage, out_dir)
    counts = {k: len(v) for k, v in sorted(per_order.items())}
    return {"nrows": nrow, "order_max": kmax, "tiles_per_order": counts}


def _view(ra, dec):
    """Centre (mean unit vector, safe across RA=0) and a fov covering all."""
    r, d = np.radians(ra), np.radians(dec)
    v = np.array([np.mean(np.cos(d) * np.cos(r)), np.mean(np.cos(d) * np.sin(r)),
                  np.mean(np.sin(d))])
    v /= np.linalg.norm(v)
    cra = float(np.degrees(np.arctan2(v[1], v[0])) % 360)
    cdec = float(np.degrees(np.arcsin(v[2])))
    cosang = (np.cos(d) * np.cos(r) * v[0] + np.cos(d) * np.sin(r) * v[1]
              + np.sin(d) * v[2])
    sep = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    return cra, cdec, float(2 * np.percentile(sep, 99.9) * 1.05)
