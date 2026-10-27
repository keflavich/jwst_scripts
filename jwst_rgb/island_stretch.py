"""Per-island display levels for mosaics made of separate coverage islands.

A CMZ-wide mosaic rendered with ONE stretch shows a faint island (the
high-latitude control field, or a field whose background-matched level sits
near the global 1st percentile) as black on a black sky.  The helpers here
label the islands of coverage and give each its own levels, so every island
uses its full dynamic range.  The price is that brightness is no longer
comparable between islands; keep a globally stretched layer next to any
layer built this way.

An island is a connected region of coverage found on a coarse grid of
``step`` x ``step`` blocks (a block counts as covered if any of its pixels
is), after a small binary closing that joins tiles separated by a seam a few
pixels wide and a hole fill that keeps saturated cores inside their island.
Islands smaller than ``min_area_pix`` pixels are slivers whose percentiles
mean little; they borrow the levels of the nearest large island.

Only display levels change.  No pixel is moved, smoothed or filled here.
"""
import numpy as np


def island_labels(valid, step=8, min_area_pix=50000, close_iter=2):
    """Label every covered pixel with the island it belongs to.

    Parameters
    ----------
    valid : 2D bool array
        True where the mosaic has data.
    step : int
        Block size of the coarse grid the islands are found on.
    min_area_pix : int
        Islands with fewer valid pixels than this get no levels of their
        own; their pixels are labelled with the nearest large island.
    close_iter : int
        Iterations of the binary closing on the coarse grid (each iteration
        bridges a gap of about ``step`` pixels).

    Returns
    -------
    labels : uint16 array, shape of ``valid``
        1..n on pixels inside a coarse island, 0 elsewhere.
    n : int
        Number of islands with their own levels.
    """
    from scipy import ndimage

    ny, nx = valid.shape
    v = np.pad(valid, ((0, -ny % step), (0, -nx % step)))
    blocks = v.reshape(v.shape[0] // step, step, v.shape[1] // step, step)
    counts = blocks.sum(axis=(1, 3), dtype=np.int64)
    del v, blocks
    coarse = counts > 0
    if close_iter:
        coarse |= ndimage.binary_closing(coarse, iterations=close_iter)
    coarse = ndimage.binary_fill_holes(coarse)
    lab, n = ndimage.label(coarse, structure=np.ones((3, 3), bool))
    if n == 0:
        return np.zeros(valid.shape, np.uint16), 0
    area = ndimage.sum(counts, lab, index=np.arange(1, n + 1))
    big = np.flatnonzero(area >= min_area_pix) + 1
    if big.size == 0:
        big = np.array([int(np.argmax(area)) + 1])
    if big.size >= np.iinfo(np.uint16).max:
        raise ValueError(f"{big.size} islands do not fit in uint16")
    keep = np.zeros(n + 1, np.uint16)
    keep[big] = np.arange(1, big.size + 1, dtype=np.uint16)
    klab = keep[lab]
    if (klab[coarse] == 0).any():
        # Small islands take the label of the nearest large one.
        _, (iy, ix) = ndimage.distance_transform_edt(klab == 0,
                                                     return_indices=True)
        klab = np.where(coarse, klab[iy, ix], 0).astype(np.uint16)
    full = np.repeat(np.repeat(klab, step, axis=0), step, axis=1)[:ny, :nx]
    return np.ascontiguousarray(full), int(big.size)


def island_slices(labels, n):
    """Bounding-box slices of islands 1..n (None for an empty label)."""
    from scipy import ndimage
    slices = ndimage.find_objects(labels, max_label=n)
    return slices + [None] * (n - len(slices))


def apply_per_island(out, data, labels, slices, levels_fn, stretch_fn,
                     sample=7, min_count=1000):
    """Overwrite ``out`` island by island with ``stretch_fn(data, *levels)``.

    ``levels_fn`` receives a 1D sample (every ``sample``-th finite, nonzero
    pixel of the island) and returns the level tuple that ``stretch_fn``
    takes after the data.  Pixels outside every island, and islands with
    fewer than ``min_count`` sampled pixels, keep what ``out`` already holds
    (normally the global stretch).  ``out`` and ``data`` are 2D and may be
    strided views.  Returns one dict per island: label, bounding box,
    sampled pixel count and levels (None if skipped).
    """
    records = []
    for k, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        m = labels[sl] == k
        sub = data[sl]
        vals = sub[m]
        vals = vals[np.isfinite(vals) & (vals != 0)][::sample]
        rec = {"label": k,
               "bbox": [[sl[0].start, sl[0].stop], [sl[1].start, sl[1].stop]],
               "n_sampled": int(vals.size), "levels": None}
        if vals.size >= min_count:
            lv = levels_fn(vals)
            o = out[sl]
            o[m] = np.nan_to_num(stretch_fn(sub[m], *lv))
            rec["levels"] = [float(x) for x in lv]
        records.append(rec)
    return records


def island_centres_galactic(records, wcs):
    """Add the (l, b) of each record's bounding-box centre, in degrees."""
    from astropy.coordinates import SkyCoord
    if not records:
        return records
    y = np.array([(r["bbox"][0][0] + r["bbox"][0][1] - 1) / 2 for r in records])
    x = np.array([(r["bbox"][1][0] + r["bbox"][1][1] - 1) / 2 for r in records])
    sky = SkyCoord.from_pixel(x, y, wcs).galactic
    for r, l, b in zip(records, sky.l.wrap_at("180d").deg, sky.b.deg):
        r["glon"], r["glat"] = round(float(l), 4), round(float(b), 4)
    return records
