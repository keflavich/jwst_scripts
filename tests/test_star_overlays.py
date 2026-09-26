"""The catalogue-derived star layers: saturated-row guard, cross-field dedupe,
star styling and rendering, and the HiPS catalogue and cube writers."""
import json
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



def test_cross_field_separations_measures_only_cross_field_pairs():
    ra = np.array([266.4, 266.4 + 0.03 / 3600, 266.5, 266.5 + 0.2 / 3600,
                   266.6, 266.6 + 0.01 / 3600])
    dec = np.full(6, -29.0)
    who = np.array(["o040", "o073", "o040", "o073", "o040", "o040"])
    sep = np.sort(overlays.cross_field_separations(ra, dec, who))
    # the within-field pair (rows 4/5) is not reported
    np.testing.assert_allclose(sep, [0.03 * np.cos(np.radians(29)),
                                     0.2 * np.cos(np.radians(29))], rtol=1e-3)


def _staged_layer(root, name, order_min, orders):
    d = root / name
    for k in orders:
        (d / f"Norder{k}").mkdir(parents=True)
    (d / "properties").write_text(f"hips_order_min = {order_min}\n")


def test_publish_checks_each_layer_at_its_own_order_min(tmp_path, monkeypatch):
    out, web = tmp_path / "out", tmp_path / "web"
    web.mkdir()
    _staged_layer(out, "cat", 1, [1, 2])      # catalogue: no Norder3 needed
    _staged_layer(out, "img", 3, [3, 4])
    monkeypatch.setattr(overlays, "OUT", str(out))
    monkeypatch.setattr(overlays, "WEB", str(web))
    monkeypatch.setattr(overlays, "LAYERS", ["cat", "img"])
    overlays.publish()
    assert (web / "cat" / "Norder1").is_dir()
    assert (web / "img" / "Norder3").is_dir()


def test_publish_refuses_an_image_layer_without_its_order_min(tmp_path,
                                                              monkeypatch):
    out, web = tmp_path / "out", tmp_path / "web"
    web.mkdir()
    _staged_layer(out, "img", 3, [1, 2])
    monkeypatch.setattr(overlays, "OUT", str(out))
    monkeypatch.setattr(overlays, "WEB", str(web))
    monkeypatch.setattr(overlays, "LAYERS", ["img"])
    with pytest.raises(RuntimeError, match="no Norder3"):
        overlays.publish()
    assert not (web / "img").exists()

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


def _galactic_stars(l_arcsec, b_arcsec, col, sat=None):
    c = SkyCoord(l=(0.1 + np.asarray(l_arcsec) / 3600) * u.deg,
                 b=np.asarray(b_arcsec) / 3600 * u.deg, frame="galactic").icrs
    n = len(col)
    return {"ra": c.ra.deg, "dec": c.dec.deg, "col": np.asarray(col, float),
            "m212": np.full(n, 15.0), "m480": np.full(n, 14.0),
            "sat": np.zeros(n, bool) if sat is None else np.asarray(sat)}


def _at(field, w, M, j):
    x, y = overlays.sky_to_pix(w, M["ra"][j], M["dec"][j])
    return field[int(round(float(y))), int(round(float(x)))]


def test_colour_field_is_the_inverse_distance_mean_of_the_neighbours():
    # Two stars 4" apart in l, colours 0 and 2: the midpoint is 1 with k=2;
    # the pixel on a star leans to that star's colour; k=1 is the nearest.
    M = _galactic_stars([0, 4, 2], [0, 0, 0], [0.0, 2.0, 9.0],
                        sat=[False, False, True])
    field, w = overlays.colour_field(M, pixel=0.1, k=2, max_arcsec=10,
                                     workers=1)
    mid = overlays.sky_to_pix(w, M["ra"][2], M["dec"][2])
    assert abs(field[int(round(float(mid[1]))),
                     int(round(float(mid[0])))] - 1.0) < 0.05
    assert 0.0 < _at(field, w, M, 0) < 0.2
    assert 1.8 < _at(field, w, M, 1) < 2.0
    one, w1 = overlays.colour_field(M, pixel=0.1, k=1, max_arcsec=10,
                                    workers=1)
    assert _at(one, w1, M, 0) == 0.0 and _at(one, w1, M, 1) == 2.0
    # the saturated star (colour 9) is never used
    assert np.nanmax(field) <= 2.0 and np.nanmax(one) <= 2.0


def test_colour_field_is_blank_beyond_the_neighbour_distance():
    M = _galactic_stars([0, 1], [0, 0], [0.5, 0.5])
    field, w = overlays.colour_field(M, pixel=0.2, k=2, max_arcsec=3,
                                     workers=1, chunk_rows=7)
    x, y = overlays.sky_to_pix(w, M["ra"], M["dec"])
    yy, xx = np.indices(field.shape)
    dist = 0.2 * np.hypot(xx - x.max(), yy - y.mean())
    assert np.isfinite(field[dist < 1.5]).all()
    assert np.isnan(field[dist > 5]).all()
    assert np.allclose(field[np.isfinite(field)], 0.5)


def test_render_colour_field_is_opaque_only_where_covered():
    M = _galactic_stars([0, 1, 30], [0, 0, 0], [-1.5, 2.5, 0.5])
    img, w = overlays.render_colour_field(M, pixel=0.2, k=1, max_arcsec=3,
                                          workers=1)
    rgb, _, _ = overlays.star_style(M["m212"], M["col"])
    for j in range(3):
        px = _at(img, w, M, j)
        assert px[3] == 255
        assert np.abs(px[:3].astype(int) - rgb[j]).max() <= 2
    assert set(np.unique(img[..., 3])) == {0, 255}
    # 15" from every star: transparent
    gap = _galactic_stars([15], [0], [0.0])
    assert _at(img, w, gap, 0)[3] == 0


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


def test_build_star_catalog_writes_every_column_type(tmp_path, monkeypatch):
    """The builder's own columns (bool -> int flag, obs strings, hex colours)
    go through to metadata.xml; the writer test alone used floats and
    strings, and an int8 flag once failed only on the full-data run."""
    from astropy.io.votable import parse_single_table
    rng = np.random.default_rng(3)
    n = 300
    M = {"ra": 266.4 + rng.uniform(-0.01, 0.01, n),
         "dec": -29.0 + rng.uniform(-0.01, 0.01, n),
         "m212": rng.uniform(14, 23, n), "m480": rng.uniform(12, 20, n),
         "col": rng.uniform(-1, 4, n), "sat": rng.random(n) < 0.1,
         "who": np.array(["o040", "o127"])[rng.integers(0, 2, n)]}
    monkeypatch.setattr(overlays, "OUT", str(tmp_path))
    overlays.build_star_catalog(M)
    out = tmp_path / "jwst-stars-catalog-hips"
    fields = {f.name: f for f in
              parse_single_table(str(out / "metadata.xml")).fields}
    assert set(fields) == {"ra", "dec", "f212n", "f480m", "color",
                           "saturated", "obs", "rgb"}
    assert fields["f212n"].unit == "mag"
    props = hips_formats._read_properties(str(out / "properties"))
    assert props["hips_cat_nrows"] == str(n)


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
    # without em_range: HiPS 1.0 layout, frame 0 also as _0 (Aladin asks so)
    frames = [_fake_frame(tmp_path, f"f{i}") for i in range(3)]
    out = str(tmp_path / "cube")
    hips_formats.assemble_hips_cube(frames, out, crval3=17.5, cdelt3=1.0,
                                    bunit3="mag", pixel_cut=(0, 5))
    tiles = os.path.join(out, "Norder3", "Dir0")
    assert open(os.path.join(tiles, "Npix12.fits"), "rb").read() == b"f0"
    assert open(os.path.join(tiles, "Npix12_0.fits"), "rb").read() == b"f0"
    assert open(os.path.join(tiles, "Npix12_1.fits"), "rb").read() == b"f1"
    assert open(os.path.join(tiles, "Npix12_2.fits"), "rb").read() == b"f2"
    assert os.path.exists(os.path.join(out, "Norder3", "Allsky_2.fits"))
    props = hips_formats._read_properties(os.path.join(out, "properties"))
    assert props["dataproduct_type"] == "cube"
    assert props["hips_cube_depth"] == "3"
    assert props["hips_cube_firstframe"] == "0"
    assert float(props["data_cube_crval3"]) == 17.5
    assert props["hips_pixel_cut"] == "0 5"


def _aladin_lite_slice(s, props):
    """Tile suffix Aladin Lite 3.7-3.9 requests at cube slider slice `s`:
    setSliceNumber (aladin.js), then channel_idx (d3/mod.rs)."""
    c = 299792458.0
    em_min, em_max = float(props["em_min"]), float(props["em_max"])
    depth = int(props["hips_cube_depth"])
    freq = c / (em_min + s / depth * (em_max - em_min))
    f_lo, f_hi = c / em_max, c / em_min
    return max(int((freq - f_lo) / (f_hi - f_lo) * depth), 0)


@pytest.mark.parametrize("depth", [5, 9])
@pytest.mark.parametrize("em_range", [(2.135e-6, 2.108e-6),
                                      (4.662e-6, 4.966e-6)])
def test_aladin_cube_slider_shows_frame_s_at_slice_s(tmp_path, depth,
                                                     em_range):
    frames = [_fake_frame(tmp_path, f"f{i}") for i in range(depth)]
    out = str(tmp_path / "cube")
    hips_formats.assemble_hips_cube(frames, out, crval3=17.5, cdelt3=1.0,
                                    bunit3="mag", em_range=em_range)
    props = hips_formats._read_properties(os.path.join(out, "properties"))
    assert float(props["em_min"]) < float(props["em_max"])
    tiles = os.path.join(out, "Norder3")
    for s in range(depth):
        k = _aladin_lite_slice(s, props)
        assert open(os.path.join(tiles, "Dir0", f"Npix12_{k}.fits"),
                    "rb").read() == f"f{s}".encode()
        assert open(os.path.join(tiles, f"Allsky_{k}.fits"),
                    "rb").read() == f"f{s}".encode()
    # slice 0 is on the edge; Aladin Lite 3.7.2-beta asks for _<depth-1>
    assert open(os.path.join(tiles, "Dir0", f"Npix12_{depth - 1}.fits"),
                "rb").read() == b"f0"
    # clients that ignore cubes see frame 0
    assert open(os.path.join(tiles, "Dir0", "Npix12.fits"), "rb").read() == b"f0"
    page = open(os.path.join(out, "index.html")).read()
    assert "setSliceNumber" in page and "17 to 18 mag" in page


def test_aladin_slice_suffixes_refuses_a_range_the_slider_cannot_cover():
    # F212N to F480M: the slider would show one frame twice and skip another
    with pytest.raises(ValueError, match="narrower"):
        hips_formats.aladin_slice_suffixes((2.108e-6, 4.966e-6), 9)
    assert hips_formats.aladin_slice_suffixes((2.108e-6, 2.135e-6), 5) == [
        5, 3, 2, 1, 0]


def test_cube_without_em_range_keeps_the_frame_0_landing_page(tmp_path):
    frames = [_fake_frame(tmp_path, f"f{i}") for i in range(2)]
    open(os.path.join(frames[0], "index.html"), "w").write("frame 0 page")
    out = str(tmp_path / "cube")
    hips_formats.assemble_hips_cube(frames, out)
    props = hips_formats._read_properties(os.path.join(out, "properties"))
    assert "em_min" not in props
    assert open(os.path.join(out, "index.html")).read() == "frame 0 page"


def test_both_density_cubes_carry_a_wavelength_range(monkeypatch):
    got = {}

    def fake_cube(*args, em_range=None, **kw):
        got[args[6]] = em_range
        raise StopIteration  # the median-colour half of build_colour is not under test

    for fn in ("report_limits", "write_density", "build_hips"):
        monkeypatch.setattr(overlays, fn, lambda *a, **k: None)
    monkeypatch.setattr(overlays, "density",
                        lambda *a, **k: (np.ones((2, 2)), None, np.ones((2, 2), bool)))
    monkeypatch.setattr(overlays, "density_cube", fake_cube)
    x = np.array([0.0, 1.0])
    with pytest.raises(StopIteration):
        overlays.build_star_density({"ra": x, "dec": x, "m212": x + 18},
                                    {"ra": x, "dec": x, "m480": x + 15})
    with pytest.raises(StopIteration):
        overlays.build_colour({"ra": x, "dec": x, "col": x, "m480": x + 15,
                               "sat": np.zeros(2, bool)})
    assert got["jwst-star-density-f212n-cube-hips"] == overlays.F212N_EM_RANGE
    assert (got["jwst-star-density-colour-cube-hips"]
            == overlays.F480M_EM_RANGE)
    for lo, hi in got.values():
        assert 1e-6 < lo < hi < 6e-6


def test_f480m_density_counts_f480m_sources_within_the_f212n_coverage(
        monkeypatch):
    calls, built = [], []

    def fake_density(ra, dec, allra, alldec, **kw):
        calls.append((ra.copy(), allra.copy()))
        return np.ones((2, 2)), "wcs", np.ones((2, 2), bool)

    monkeypatch.setattr(overlays, "report_limits", lambda *a, **k: None)
    monkeypatch.setattr(overlays, "write_density", lambda *a, **k: None)
    monkeypatch.setattr(overlays, "density", fake_density)
    monkeypatch.setattr(overlays, "density_cube", lambda *a, **k: None)
    monkeypatch.setattr(overlays, "build_hips",
                        lambda arr, w, name, *a, **k: built.append(name))
    f212 = {"ra": np.array([266.40, 266.41]), "dec": np.array([-29.0, -29.0]),
            "m212": np.array([18.0, 19.0])}
    f480 = {"ra": np.array([266.405]), "dec": np.array([-29.0]),
            "m480": np.array([15.0])}
    overlays.build_star_density(f212, f480)
    assert built == ["jwst-star-density-hips", "jwst-star-density-f480m-hips"]
    ra480, cover = calls[1]
    assert ra480.tolist() == f480["ra"].tolist()
    assert cover.tolist() == f212["ra"].tolist()
    assert "jwst-star-density-f480m-hips" in overlays.LAYERS
    assert "F480M" in overlays.LAYER_DESCRIPTIONS["jwst-star-density-f480m-hips"]


def test_assemble_hips_cube_refuses_mismatched_orders(tmp_path):
    frames = [_fake_frame(tmp_path, "a"), _fake_frame(tmp_path, "b", "4")]
    with pytest.raises(ValueError, match="hips_order"):
        hips_formats.assemble_hips_cube(frames, str(tmp_path / "cube"))


# -- the guard and dedupe as wired into match_catalogs -----------------------

def _write_cat(d, obs, filt, ra, dec, sat, pixscale):
    t = Table({"skycoord": SkyCoord(np.asarray(ra) * u.deg,
                                    np.asarray(dec) * u.deg),
               "flux": np.full(len(ra), 1e3),
               "is_saturated": np.asarray(sat, bool)})
    t.meta["PIXSCALE"] = pixscale
    p = os.path.join(d, f"{filt}_merged_{obs}_indivexp_merged_m4_dao_basic_"
                        "vetted.fits")
    t.write(p)
    return p


def test_match_catalogs_keeps_each_saturated_star_in_its_own_field(
        tmp_path, monkeypatch):
    """Two fields on disk, each carrying the SAME survey-wide saturated list
    (the upstream bug), plus one unsaturated star both fields detect."""
    monkeypatch.setattr(overlays, "CAT", str(tmp_path))
    step = 2.0 / 3600
    own_a = [(266.40 + i * step, -29.0) for i in range(5)]
    own_b = [(266.60 + i * step, -29.0) for i in range(5)]
    shared = (266.50, -29.0)                  # overlap: both fields see it
    sat_a = (266.40 + 1.5 * step, -29.0003)   # inside field a only
    sat_b = (266.60 + 1.5 * step, -29.0003)   # inside field b only
    near_shared_a = (266.50 - step, -29.0)    # gives each field a detection
    near_shared_b = (266.50 + step, -29.0)    # next to the shared star
    rows = {
        "o040": own_a + [near_shared_a, shared, sat_a, sat_b],
        "o073": own_b + [near_shared_b, shared, sat_a, sat_b],
    }
    for obs, pos in rows.items():
        ra, dec = np.array(pos).T
        sat = [False] * (len(pos) - 2) + [True, True]
        for filt, ps in (("f212n", 0.031), ("f480m", 0.063)):
            _write_cat(tmp_path, obs, filt, ra, dec, sat, ps)

    pairs = overlays.latest_pairs()
    assert sorted(pairs) == ["o040", "o073"]
    M, F, L = overlays.match_catalogs(pairs)

    for D in (M, F, L):
        pos = SkyCoord(D["ra"] * u.deg, D["dec"] * u.deg)

        def owners(p):
            sep = pos.separation(SkyCoord(p[0] * u.deg, p[1] * u.deg)).arcsec
            return sorted(D["who"][sep < 0.05])

        # each saturated star once, credited to the field it lies in
        assert owners(sat_a) == ["o040"]
        assert owners(sat_b) == ["o073"]
        # the overlap star once, from the field that sorts first
        assert owners(shared) == ["o040"]
        assert D["sat"].sum() == 2
        assert len(D["ra"]) == 5 + 5 + 2 + 1 + 2


def _crowded(n, shift_arcsec, seed=1):
    """A crowded SW list, and an LW list of 30% of it shifted in RA by
    `shift_arcsec` with 20 mas scatter."""
    rng = np.random.default_rng(seed)
    ra = 266.4 + rng.uniform(0, 20 / 3600, n)
    dec = -29.0 + rng.uniform(0, 20 / 3600, n)
    pick = rng.random(n) < 0.3
    cosd = np.cos(np.radians(-29.0))
    lra = ra[pick] + (shift_arcsec + rng.normal(0, 0.02, pick.sum())) / 3600 / cosd
    ldec = dec[pick] + rng.normal(0, 0.02, pick.sum()) / 3600
    return ra, dec, lra, ldec


@pytest.mark.parametrize("shift", [0.0, 0.2])
def test_field_offset_recovers_a_bulk_offset_among_chance_pairs(shift):
    # ~5 stars/arcsec^2: a nearest neighbour within 0.5" is often a chance pair
    ra, dec, lra, ldec = _crowded(2000, shift)
    sw = SkyCoord(ra * u.deg, dec * u.deg)
    lw = SkyCoord(lra * u.deg, ldec * u.deg)
    idx, d2d, _ = lw.match_to_catalog_sky(sw)
    dx, dy, off = overlays.field_offset(sw, lw, idx, d2d)
    assert abs(dx - shift) < 0.01 and abs(dy) < 0.01
    assert abs(off - shift) < 0.01


def test_match_catalogs_leaves_an_offset_field_out_of_the_colours(
        tmp_path, monkeypatch):
    monkeypatch.setattr(overlays, "CAT", str(tmp_path))
    for obs, shift, ra0 in (("o040", 0.0, 266.40), ("o113", 0.2, 266.60)):
        ra, dec, lra, ldec = _crowded(400, shift)
        ra, lra = ra - 266.4 + ra0, lra - 266.4 + ra0
        _write_cat(tmp_path, obs, "f212n", ra, dec, np.zeros(len(ra)), 0.031)
        _write_cat(tmp_path, obs, "f480m", lra, ldec, np.zeros(len(lra)),
                   0.063)
    M, F, L = overlays.match_catalogs(overlays.latest_pairs())
    assert set(M["who"]) == {"o040"}
    assert set(F["who"]) == set(L["who"]) == {"o040", "o113"}
    assert overlays.CACHE_VERSION >= 4


# -- the published star PNG reads back at the right sky positions -----------

def test_star_png_round_trips_through_its_avm(tmp_path, monkeypatch):
    """PNG -> embedded AVM -> WCS -> pixel: each star's colour must be found
    at that star's own position.  Three stars in an L, each a different
    colour, so a flip about either axis (or a 180 degree rotation) moves a
    colour onto the wrong star."""
    from PIL import Image
    import pyavm
    import reproject.hips

    def fake_hips(png, output_directory, **kw):
        os.makedirs(os.path.join(output_directory, "Norder3"))
    monkeypatch.setattr(reproject.hips, "reproject_to_hips", fake_hips)
    monkeypatch.setattr(overlays, "OUT", str(tmp_path))

    g = SkyCoord(l=[0.1, 0.1, 0.1 + 30 / 3600] * u.deg,
                 b=[0.0, 20 / 3600, 0.0] * u.deg, frame="galactic").icrs
    M = {"ra": g.ra.deg, "dec": g.dec.deg,
         "m212": np.array([12.0, 12.0, 12.0]),
         "col": np.array([-1.5, 2.5, 0.5]),     # blue, red, yellow-ish
         "sat": np.zeros(3, bool)}
    # one neighbour: each star's own pixel carries exactly its colour
    monkeypatch.setattr(overlays, "KNN_NEIGHBORS", 1)
    monkeypatch.setattr(overlays, "KNN_PIXEL_ARCSEC", 0.2)
    overlays.build_star_image(M)

    png = tmp_path / "jwst-stars-colour.png"
    w = pyavm.AVM.from_image(str(png)).to_wcs()
    arr = np.asarray(Image.open(png))[::-1]      # FITS row order
    rgb, _, _ = overlays.star_style(M["m212"], M["col"])
    x, y = w.world_to_pixel(g)
    for k in range(3):
        px = arr[int(round(float(y[k]))), int(round(float(x[k])))]
        assert px[3] == 255, f"star {k}: transparent at its own position"
        assert np.abs(px[:3].astype(int) - rgb[k]).max() <= 2, (
            f"star {k}: found {px[:3]}, expected {rgb[k]}")


def test_match_catalogs_guards_the_f480m_side_too(tmp_path, monkeypatch):
    """A foreign saturated row in the F480M catalogue must not reach the
    matched set, even where the F212N catalogue has a source for it to match.
    X is an F212N detection far from the rest of the field; F480M carries X
    only as a (survey-wide, foreign) saturated row."""
    monkeypatch.setattr(overlays, "CAT", str(tmp_path))
    step = 2.0 / 3600
    own = [(266.40 + i * step, -29.0) for i in range(5)]
    X = (266.60, -29.0)
    ra, dec = np.array(own + [X]).T
    _write_cat(tmp_path, "o040", "f212n", ra, dec, [False] * 6, 0.031)
    _write_cat(tmp_path, "o040", "f480m", ra, dec, [False] * 5 + [True], 0.063)
    M, F, L = overlays.match_catalogs(overlays.latest_pairs())
    sep = SkyCoord(M["ra"] * u.deg, M["dec"] * u.deg).separation(
        SkyCoord(X[0] * u.deg, X[1] * u.deg)).arcsec
    assert not (sep < 0.05).any()
    assert len(M["ra"]) == 5 and not M["sat"].any()
    # nor the F480M source list behind the F480M density map
    sepL = SkyCoord(L["ra"] * u.deg, L["dec"] * u.deg).separation(
        SkyCoord(X[0] * u.deg, X[1] * u.deg)).arcsec
    assert not (sepL < 0.05).any()
    assert len(L["ra"]) == 5 and not L["sat"].any()


def test_every_layer_carries_a_one_line_description():
    for name in overlays.LAYERS:
        desc = overlays.LAYER_DESCRIPTIONS[name]
        assert "\n" not in desc and desc.strip() == desc
        assert overlays.layer_properties(name)["obs_description"] == desc
    assert set(overlays.LAYER_DESCRIPTIONS) == set(overlays.LAYERS)
    for name in overlays.LAYERS:
        if "density" in name or name.startswith(("jwst-red-", "jwst-rc-")):
            assert "stars/arcmin^2" in overlays.LAYER_DESCRIPTIONS[name]


def test_density_layer_titles_match_their_selection():
    from jwst_rgb.hips_naming import properties_for
    title = {n: properties_for(n)["obs_title"] for n in
             ("jwst-red-stars-hips", "jwst-rc-blue-hips", "jwst-rc-red-hips")}
    assert f"> {overlays.RED_COLOUR:g}" in title["jwst-red-stars-hips"]
    assert f"< {overlays.RED_MAGLIMIT:g}" in title["jwst-red-stars-hips"]
    for n in ("jwst-rc-blue-hips", "jwst-rc-red-hips"):
        assert "red-clump" in title[n] and f"{overlays.SPLIT:g}" in title[n]
        assert "cluster" not in title[n]


def test_red_clump_descriptions_state_every_rc_masks_cut():
    for n in ("jwst-rc-blue-hips", "jwst-rc-red-hips"):
        d = overlays.LAYER_DESCRIPTIONS[n]
        for v in (*overlays.RC_M480_RANGE, *overlays.RC_COLOUR_RANGE,
                  overlays.SLOPE, overlays.WRC, overlays.HW, overlays.SPLIT):
            assert f"{v:g}" in d, (n, v)


# -- the match cache carries all three source lists --------------------------

def _fake_lists(n=4):
    r = np.arange(n, dtype=float)
    M = {k: r + i for i, k in enumerate(("col", "m480", "m212", "ra", "dec"))}
    M["who"], M["sat"] = np.array(["o001"] * n), np.zeros(n, bool)
    F = {"m212": r + 10, "ra": r, "dec": r, "who": M["who"], "sat": M["sat"]}
    L = {"m480": r + 20, "ra": r + 0.5, "dec": r, "who": M["who"],
         "sat": np.array([True] + [False] * (n - 1))}
    return M, F, L


def test_load_matched_round_trips_every_list_through_the_cache(
        tmp_path, monkeypatch):
    monkeypatch.setattr(overlays, "CACHE", str(tmp_path / "cache.npz"))
    monkeypatch.setattr(overlays, "STAMP", str(tmp_path / "stamp.json"))
    monkeypatch.setattr(overlays, "fingerprint", lambda pairs: {
        "n_obs": 1, "items": [], "version": overlays.CACHE_VERSION})
    calls = []
    lists = _fake_lists()
    monkeypatch.setattr(overlays, "match_catalogs",
                        lambda pairs: calls.append(1) or lists)
    first = overlays.load_matched({})
    with open(overlays.STAMP, "w") as fh:     # main() records the match
        json.dump({"match": first[3]}, fh)
    M, F, L, fp = overlays.load_matched({})
    assert len(calls) == 1, "the second call must be a cache hit"
    for got, want in zip((M, F, L), lists):
        assert sorted(got) == sorted(want)
        for k in want:
            assert np.array_equal(got[k], want[k]), k
    # a cache from another CACHE_VERSION is never read back
    monkeypatch.setattr(overlays, "CACHE_VERSION", overlays.CACHE_VERSION + 1)
    overlays.load_matched({})
    assert len(calls) == 2


def test_cache_version_postdates_the_f480m_list():
    # version 2 caches hold no L_ arrays; reading one back would KeyError
    assert overlays.CACHE_VERSION >= 3


def test_match_catalogs_drops_f480m_rows_without_a_magnitude(
        tmp_path, monkeypatch):
    monkeypatch.setattr(overlays, "CAT", str(tmp_path))
    step = 2.0 / 3600
    ra = 266.40 + np.arange(5) * step
    dec = np.full(5, -29.0)
    _write_cat(tmp_path, "o040", "f212n", ra, dec, [False] * 5, 0.031)
    p = _write_cat(tmp_path, "o040", "f480m", ra, dec, [False] * 5, 0.063)
    t = Table.read(p)
    t["flux"][-1] = np.nan
    t.write(p, overwrite=True)
    M, F, L = overlays.match_catalogs(overlays.latest_pairs())
    assert len(L["ra"]) == 4 and np.isfinite(L["m480"]).all()
    assert len(F["ra"]) == 5


# -- coverage: no border of zeros, NaN beyond the data -----------------------

def _uniform_field(n=20000, side_arcsec=120.0, seed=2):
    rng = np.random.default_rng(seed)
    ra = 266.4 + rng.uniform(0, side_arcsec, n) / 3600 / np.cos(np.radians(-29))
    dec = -29.0 + rng.uniform(0, side_arcsec, n) / 3600
    return ra, dec


def test_density_is_blank_beyond_the_stars_and_flat_up_to_the_edge():
    ra, dec = _uniform_field()
    arr, w, cov = overlays.density(ra, dec, ra, dec)
    x, y = overlays.sky_to_pix(w, ra, dec)
    ys, xs = np.nonzero(cov)
    # coverage ends within a few pixels of the outermost stars
    assert xs.min() >= x.min() - 4 and xs.max() <= x.max() + 4
    assert ys.min() >= y.min() - 4 and ys.max() <= y.max() + 4
    assert np.isnan(arr[~cov]).all() and np.isfinite(arr[cov]).all()
    # 20000 stars / 4 arcmin^2; the edge is not diluted by the empty sky
    expect = 20000 / 4.0
    inner = arr[int(np.median(ys)), int(np.median(xs))]
    edge = arr[int(np.median(ys)), xs.min() + 1]
    assert abs(inner / expect - 1) < 0.2
    assert abs(edge / expect - 1) < 0.3


def test_footprint_fills_an_empty_hole_inside_the_field():
    ra, dec = _uniform_field()
    grid = overlays.make_grid(ra, dec, 2.0)
    x, y = overlays.sky_to_pix(grid[0], ra, dec)
    cx, cy = np.median(x), np.median(y)
    hole = np.hypot(x - cx, y - cy) < 8          # a 16" dark cloud
    cov = overlays.footprint(ra[~hole], dec[~hole], grid, 3)
    assert cov[int(round(cy)), int(round(cx))]


def test_knn_median_covers_the_footprint_and_takes_the_local_median():
    ra, dec = _uniform_field()
    x0 = np.median(ra)
    col = np.where(ra < x0, 0.0, 2.0)          # two halves, sharp boundary
    grid = overlays.make_grid(ra, dec, 2.0)
    cov = overlays.footprint(ra, dec, grid, 3)
    # only 1 star in 20 passes the cuts: sparse, as behind a dark cloud
    use = np.arange(len(ra)) % 20 == 0
    med, reach = overlays.knn_median(ra[use], dec[use], col[use], grid, cov, 15)
    assert np.isfinite(med[cov]).all() and np.isnan(med[~cov]).all()
    ys, xs = np.nonzero(cov)
    row = int(np.median(ys))
    left, right = med[row, xs.min() + 2], med[row, xs.max() - 2]
    assert {left, right} == {0.0, 2.0}
    assert np.nanmax(reach) < 30
