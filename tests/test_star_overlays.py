"""The catalogue-derived star layers: saturated-row guard, cross-field dedupe,
star styling and rendering, and the HiPS catalogue and cube writers."""
import os

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

import gc_treasury_overlays as overlays
from jwst_rgb import hips_formats


def _field(ra, dec, sat):
    t = Table({"skycoord": SkyCoord(np.asarray(ra) * u.deg,
                                    np.asarray(dec) * u.deg),
               "is_saturated": np.asarray(sat, bool)})
    return t


# -- saturated-row guard ----------------------------------------------------

def test_own_footprint_drops_saturated_rows_outside_the_field():
    # Three unsaturated detections near (266.4, -29.0); one saturated star
    # among them and one 0.5 deg away (a row copied in from another field).
    ra = [266.4, 266.4005, 266.4010, 266.4003, 266.9]
    dec = [-29.0, -29.0, -29.0, -29.0003, -29.0]
    sat = [False, False, False, True, True]
    keep = overlays.own_footprint(_field(ra, dec, sat), radius=10.0)
    assert keep.tolist() == [True, True, True, True, False]


def test_own_footprint_without_unsaturated_rows_keeps_none_saturated():
    keep = overlays.own_footprint(_field([266.4, 266.5], [-29.0, -29.0],
                                         [True, True]))
    assert not keep.any()


def test_own_footprint_without_saturation_column_keeps_everything():
    t = Table({"skycoord": SkyCoord([266.4, 266.5] * u.deg,
                                    [-29.0, -29.0] * u.deg)})
    assert overlays.own_footprint(t).all()


def test_dedupe_drops_the_later_field_only_across_fields():
    d = 0.02 / 3600                       # 0.02", inside the 0.1" tolerance
    ra = np.array([266.4, 266.4 + d, 266.5, 266.5 + d, 266.6])
    dec = np.full(5, -29.0)
    who = np.array(["o040", "o073", "o040", "o040", "o073"])
    keep = overlays.dedupe_across_fields(ra, dec, who, tol=0.1)
    # pair 0/1 is cross-field: o073 (later) goes.  Pair 2/3 is within one
    # field: crowding, both stay.  Row 4 has no partner.
    assert keep.tolist() == [True, False, True, True, True]


def test_dedupe_collapses_a_row_repeated_in_many_fields():
    who = np.array([f"o{n:03d}" for n in range(40, 83)])
    ra = np.full(len(who), 266.4)
    dec = np.full(len(who), -29.0)
    keep = overlays.dedupe_across_fields(ra, dec, who)
    assert keep.sum() == 1 and keep[0]


# -- star styling and rendering ---------------------------------------------

def test_star_style_colour_size_and_alpha():
    m212 = np.array([8.0, 17.0, 23.0, 30.0])
    col = np.array([-3.0, 0.5, 2.5, 9.0])
    rgb, alpha, radius = overlays.star_style(m212, col)
    assert rgb.dtype == np.uint8 and rgb.shape == (4, 3)
    # blue end of RdYlBu_r for blue stars, red end for red ones
    assert rgb[0, 2] > rgb[0, 0] and rgb[3, 0] > rgb[3, 2]
    assert radius.tolist() == [7, 2, 1, 1]
    assert alpha[0] == 255 and alpha[-1] == 90
    assert np.all(np.diff(alpha.astype(int)) <= 0)


def test_render_stars_puts_each_star_at_its_position():
    # Two stars 20" apart in galactic latitude; the bright one is red.
    c = SkyCoord(l=[0.1, 0.1] * u.deg, b=[0.0, 20 / 3600] * u.deg,
                 frame="galactic").icrs
    M = {"ra": c.ra.deg, "dec": c.dec.deg,
         "m212": np.array([12.0, 21.0]), "col": np.array([2.5, -1.5])}
    img, w = overlays.render_stars(M, pixel=0.2)
    assert img.shape[2] == 4
    x, y = overlays.sky_to_pix(w, M["ra"], M["dec"])
    xi, yi = np.round(x).astype(int), np.round(y).astype(int)
    bright, faint = img[yi[0], xi[0]], img[yi[1], xi[1]]
    assert bright[3] > faint[3] > 0
    assert bright[0] > bright[2] and faint[2] > faint[0]
    # the grid is galactic and b increases with the row index (FITS order)
    assert yi[1] > yi[0] and abs(xi[1] - xi[0]) <= 1
    # nothing painted far from either star
    assert img[..., 3].sum() == img[max(yi.min() - 8, 0):yi.max() + 9,
                                    max(xi.min() - 8, 0):xi.max() + 9,
                                    3].sum()


# -- HiPS catalogue ---------------------------------------------------------

def test_distribute_caps_tiles_and_keeps_every_row():
    rng = np.random.default_rng(1)
    n = 5000
    ra = 266.4 + rng.uniform(-0.05, 0.05, n)
    dec = -29.0 + rng.uniform(-0.05, 0.05, n)
    mag = rng.uniform(10, 25, n)
    order, npix = hips_formats.distribute(
        ra, dec, mag, n_per_tile=100, order_min=1, order_max=12)
    assert (order >= 1).all() and (npix >= 0).all()
    for k in range(1, 12):
        sel = order == k
        if sel.any():
            _, counts = np.unique(npix[sel], return_counts=True)
            assert counts.max() <= 100
    # the brightest source goes to the shallowest order
    assert order[np.argmin(mag)] == 1


def test_write_hips_catalogue_layout(tmp_path):
    rng = np.random.default_rng(2)
    n = 800
    cols = {"ra": 266.4 + rng.uniform(-0.01, 0.01, n),
            "dec": -29.0 + rng.uniform(-0.01, 0.01, n),
            "f212n": rng.uniform(12, 24, n),
            "saturated": rng.random(n) < 0.1,
            "rgb": np.array(["#ff0000"] * n)}
    cols["f212n"][3] = np.nan
    out = str(tmp_path / "cat_hips")
    info = hips_formats.write_hips_catalogue(
        out, cols, priority=cols["f212n"], n_per_tile=100, order_min=1,
        order_max=10, units={"f212n": "mag"})
    assert info["nrows"] == n
    props = hips_formats._read_properties(os.path.join(out, "properties"))
    assert props["dataproduct_type"] == "catalog"
    assert props["hips_tile_format"] == "tsv"
    assert props["hips_cat_nrows"] == str(n)
    assert os.path.exists(os.path.join(out, "metadata.xml"))
    assert os.path.exists(os.path.join(out, "Norder1", "Allsky.tsv"))
    assert not os.path.exists(out + ".new")

    rows = 0
    for root, _, files in os.walk(out):
        for f in files:
            if f.startswith("Npix") and f.endswith(".tsv"):
                lines = open(os.path.join(root, f)).read().splitlines()
                assert lines[0].startswith("# Completeness = ")
                assert lines[1].split("\t") == list(cols)
                rows += len(lines) - 2
                for line in lines[2:]:
                    assert len(line.split("\t")) == len(cols)
    assert rows == n
    # a NaN magnitude is written as an empty field
    allrows = []
    for root, _, files in os.walk(out):
        for f in files:
            if f.startswith("Npix"):
                allrows += open(os.path.join(root, f)).read().splitlines()[2:]
    assert sum(r.split("\t")[2] == "" for r in allrows) == 1


# -- HiPS cube --------------------------------------------------------------

def _fake_frame(root, name, order="3"):
    d = root / name
    (d / "Norder3" / "Dir0").mkdir(parents=True)
    (d / "Norder3" / "Dir0" / "Npix12.fits").write_bytes(name.encode())
    (d / "Norder3" / "Allsky.fits").write_bytes(name.encode())
    (d / "Moc.fits").write_bytes(b"moc")
    (d / "properties").write_text(
        f"hips_order = {order}\nhips_tile_format = fits\n"
        "hips_frame = galactic\ndataproduct_type = image\n")
    return str(d)


def test_assemble_hips_cube_names_frames(tmp_path):
    frames = [_fake_frame(tmp_path, f"f{i}") for i in range(3)]
    out = str(tmp_path / "cube")
    hips_formats.assemble_hips_cube(frames, out, crval3=17.5, cdelt3=1.0,
                                    bunit3="mag", pixel_cut=(0, 5))
    tiles = os.path.join(out, "Norder3", "Dir0")
    assert open(os.path.join(tiles, "Npix12.fits"), "rb").read() == b"f0"
    assert open(os.path.join(tiles, "Npix12_1.fits"), "rb").read() == b"f1"
    assert open(os.path.join(tiles, "Npix12_2.fits"), "rb").read() == b"f2"
    assert os.path.exists(os.path.join(out, "Norder3", "Allsky_2.fits"))
    props = hips_formats._read_properties(os.path.join(out, "properties"))
    assert props["dataproduct_type"] == "cube"
    assert props["hips_cube_depth"] == "3"
    assert props["hips_cube_firstframe"] == "0"
    assert float(props["data_cube_crval3"]) == 17.5
    assert props["hips_pixel_cut"] == "0 5"


def test_assemble_hips_cube_refuses_mismatched_orders(tmp_path):
    frames = [_fake_frame(tmp_path, "a"), _fake_frame(tmp_path, "b", "4")]
    with pytest.raises(ValueError, match="hips_order"):
        hips_formats.assemble_hips_cube(frames, str(tmp_path / "cube"))
