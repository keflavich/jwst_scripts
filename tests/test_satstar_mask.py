"""The opt-in display mask of the accepted-satstar cores in the residual layers.

What these pin: the radius rule, read in the detector pixels each filter's sat_area is
counted in; that the input array is never touched; that only finite pixels inside disks
(and NaN cores wholly inside them) change; that a repeated or far-away catalogue row
cannot widen the mask; that each filter reads its own catalogue, a missing one is an
error and a filter without a detector scale is refused; that the builder masks exactly
G.SATMASK_FILTERS, each in its own channel; that the masked layer has a name no other
flavour's glob can match; that the flag is refused wherever it does not apply, with
nothing behind main() that could touch the survey; and -- through build_obs -- that the
residual i2d on disk is only read.
"""

import fnmatch
import hashlib
import os
import sys

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

import gc_treasury_rgb_images as G
from jwst_rgb import satstar_mask as SM

PIX = 0.0629091  # arcsec per pixel of the F480M i2d
PIX212 = 0.031223  # ... and of the F212N i2d
FILTS = [("f480m", PIX), ("f212n", PIX212)]
RA0, DEC0 = 266.5, -28.9


def _wcs(n, pix_as, crval=(RA0, DEC0)):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = list(crval)
    w.wcs.crpix = [(n + 1) / 2, (n + 1) / 2]
    w.wcs.cdelt = [-pix_as / 3600.0, pix_as / 3600.0]
    return w


def _catalogue(tmp_path, wcs, rows, obs="o999", filt="f480m"):
    """rows: (x, y, sat_area) in pixels of `wcs`."""
    x, y, area = (np.array(c, float) for c in zip(*rows))
    ra, dec = wcs.wcs_pix2world(x, y, 0)
    t = Table({"skycoord_fit": SkyCoord(ra, dec, unit="deg"), "sat_area": area})
    path = tmp_path / f"{filt}_{obs}_consolidated_satstar_catalog.fits"
    t.write(path, overwrite=True)
    return str(tmp_path)


def _sky(n=240, level=10.0, seed=1):
    rng = np.random.default_rng(seed)
    return level + rng.normal(0, 0.5, (n, n))


def test_the_radius_rule():
    r = SM.mask_radius_px([1, 100, 400, 5000], PIX)
    assert r[0] == 12.0  # floor
    assert r[1] == pytest.approx(np.sqrt(100 / np.pi) + 8)
    assert r[2] == pytest.approx(np.sqrt(400 / np.pi) + 8)
    assert r[3] == 30.0  # ceiling
    # the same sky radius on the finer F212N grid
    assert SM.mask_radius_px([100], PIX / 2)[0] == pytest.approx(2 * r[1])


def test_the_f480m_radius_is_exactly_the_rule_it_was_measured_as():
    area = np.array([1.0, 80.0, 400.0, 1521.0, 5000.0])
    r = SM.mask_radius_px(area, SM.REF_PIX_ARCSEC)
    assert np.array_equal(r, np.clip(np.sqrt(area / np.pi) + 8.0, 12.0, 30.0))


def test_the_radius_is_in_the_detector_pixels_sat_area_was_counted_in():
    # F212N sat_area counts 0.031" pixels of a per-exposure frame, so the rule gives
    # the same radius in those pixels: half the angular size of F480M's
    area = [1, 100, 400, 5000]
    r480 = SM.mask_radius_px(area, PIX)
    r212 = SM.mask_radius_px(area, PIX212, SM.DETECTOR_PIX_ARCSEC["f212n"])
    assert r212 == pytest.approx(r480)
    assert r212 * PIX212 / (r480 * PIX) == pytest.approx(PIX212 / PIX)
    # the same sky radius on an array twice as fine as the detector frames
    assert SM.mask_radius_px([100], PIX212 / 2, PIX212)[0] == pytest.approx(2 * r212[1])


def test_the_detector_scales_the_rule_is_read_in():
    assert SM.DETECTOR_PIX_ARCSEC["f480m"] == SM.REF_PIX_ARCSEC == PIX
    assert SM.DETECTOR_PIX_ARCSEC["f212n"] == pytest.approx(PIX212)
    # the builder's gate and the module's table cannot drift apart
    assert G.SATMASK_FILTERS == ("f212n", "f480m")
    assert set(G.SATMASK_FILTERS) == set(SM.DETECTOR_PIX_ARCSEC)


@pytest.mark.parametrize("filt,pix", FILTS)
def test_a_donut_is_replaced_by_background_and_the_input_is_untouched(
    tmp_path, filt, pix
):
    n = 240
    w = _wcs(n, pix)
    img = _sky(n)
    yy, xx = np.mgrid[:n, :n]
    rr = np.hypot(xx - 120, yy - 120)
    img[rr < 5] -= 150.0  # over-subtracted core
    img[(rr >= 5) & (rr < 9)] += 60.0  # ring
    before = img.copy()
    cat = _catalogue(tmp_path, w, [(120, 120, 80)], filt=filt)
    out, m, info = SM.mask_satstar_cores(img, w, "o999", filt=filt, catalog_dir=cat)
    assert np.array_equal(img, before)  # a copy, never in place
    r = float(SM.mask_radius_px([80], pix, SM.DETECTOR_PIX_ARCSEC[filt])[0])
    assert info["n_used"] == 1 and info["n_dupes"] == 0 and info["n_at_cap"] == 0
    assert np.array_equal(m, rr <= r) or abs(int(m.sum()) - int((rr <= r).sum())) <= 4
    assert np.array_equal(out[~m], img[~m])  # nothing outside the disk moves
    assert np.abs(out[rr < r - 1] - 10.0).max() < 2.0  # filled with the local sky
    assert info["frac_of_finite"] == pytest.approx(m.sum() / n / n)


@pytest.mark.parametrize("filt,pix", FILTS)
def test_a_nan_core_inside_a_disk_is_filled_but_the_outer_footprint_is_not(
    tmp_path, filt, pix
):
    n = 240
    w = _wcs(n, pix)
    img = _sky(n)
    img[117:124, 117:124] = np.nan  # saturated core, wholly inside the disk
    img[:, :6] = np.nan  # outer footprint, touches the edge
    cat = _catalogue(
        tmp_path, w, [(120, 120, 80), (14, 120, 80)], filt=filt
    )  # second disk overlaps the footprint edge
    out, m, info = SM.mask_satstar_cores(img, w, "o999", filt=filt, catalog_dir=cat)
    assert np.isfinite(out[117:124, 117:124]).all()
    assert info["n_nan_core_px_filled"] == 49
    assert np.isnan(out[:, :6]).all()  # alpha edge logic sees the same footprint
    assert np.array_equal(np.isnan(out[:, 6:]), np.zeros((n, n - 6), bool))


@pytest.mark.parametrize("filt,pix", FILTS)
def test_a_repeated_or_far_row_does_not_widen_the_mask(tmp_path, filt, pix):
    n = 240
    w = _wcs(n, pix)
    img = _sky(n)
    one = _catalogue(tmp_path, w, [(120, 120, 80)], filt=filt)
    _, m1, _ = SM.mask_satstar_cores(img, w, "o999", filt=filt, catalog_dir=one)
    # the same star listed twice (the larger sat_area wins), plus a row the
    # survey-wide injection would have added: a whole degree away
    many = _catalogue(
        tmp_path,
        w,
        [(120, 120, 80), (120.5, 120, 400), (120, 120, 80), (5000, 5000, 900)],
        filt=filt,
    )
    _, m2, info = SM.mask_satstar_cores(img, w, "o999", filt=filt, catalog_dir=many)
    assert (
        info["n_rows"] == 4
        and info["n_in_image"] == 3
        and info["n_dupes"] == 2
        and info["n_used"] == 1
    )
    assert m2.sum() > m1.sum()  # the largest duplicate wins: 400 px, not 80
    r400 = float(SM.mask_radius_px([400], pix, SM.DETECTOR_PIX_ARCSEC[filt])[0])
    yy, xx = np.mgrid[:n, :n]
    assert int(m2.sum()) == pytest.approx(
        int((np.hypot(xx - 120, yy - 120) <= r400).sum()), abs=60
    )


@pytest.mark.parametrize("filt,pix", FILTS)
def test_stars_on_the_radius_cap_are_counted_and_not_masked_beyond_it(
    tmp_path, filt, pix
):
    n = 400
    w = _wcs(n, pix)
    # r_core + 8 reaches the cap of 30 px at sat_area = pi * 22**2 = 1521
    rows = [(100, 100, 80), (300, 100, 1400), (100, 300, 1700), (300, 300, 9000)]
    cat = _catalogue(tmp_path, w, rows, filt=filt)
    _, m, info = SM.mask_satstar_cores(
        _sky(n), w, "o999", filt=filt, catalog_dir=cat
    )
    assert info["n_used"] == 4 and info["n_at_cap"] == 2
    assert info["r_mask_px_max"] == pytest.approx(30.0)
    assert m[300, 329] and not m[300, 331]  # the 9000 px star: r_core 53, mask 30


@pytest.mark.parametrize("filt,pix", FILTS)
def test_each_filter_reads_its_own_catalogue(tmp_path, filt, pix):
    n = 240
    w = _wcs(n, pix)
    # one directory, both filters' catalogues, a star in a different place in each
    _catalogue(tmp_path, w, [(60, 60, 80)], filt="f480m")
    cat = _catalogue(tmp_path, w, [(180, 180, 80)], filt="f212n")
    _, m, info = SM.mask_satstar_cores(_sky(n), w, "o999", filt=filt, catalog_dir=cat)
    here, there = (60, 180) if filt == "f480m" else (180, 60)
    assert m[here, here] and not m[there, there]
    assert info["catalog"].endswith(f"{filt}_o999_consolidated_satstar_catalog.fits")


@pytest.mark.parametrize("filt,other", [("f212n", "f480m"), ("f480m", "f212n")])
def test_a_missing_catalogue_is_an_error_not_a_silent_no_op(tmp_path, filt, other):
    w = _wcs(64, PIX)
    with pytest.raises(FileNotFoundError, match=rf"o998.*{filt}_o998"):
        SM.mask_satstar_cores(
            np.ones((64, 64)), w, "o998", filt=filt, catalog_dir=str(tmp_path)
        )
    # the other filter's catalogue does not stand in for it
    cat = _catalogue(tmp_path, w, [(32, 32, 80)], obs="o998", filt=other)
    with pytest.raises(FileNotFoundError, match=rf"o998.*{filt}_o998"):
        SM.mask_satstar_cores(
            np.ones((64, 64)), w, "o998", filt=filt, catalog_dir=cat
        )


@pytest.mark.parametrize("filt", ["f187n", "f770w"])
def test_a_filter_without_a_detector_scale_is_refused(tmp_path, filt):
    w = _wcs(64, PIX)
    # even with a catalogue of that name on disk: no scale, no mask
    cat = _catalogue(tmp_path, w, [(32, 32, 80)], filt=filt)
    with pytest.raises(ValueError, match="no detector pixel scale"):
        SM.mask_satstar_cores(np.ones((64, 64)), w, "o999", filt=filt, catalog_dir=cat)
    with pytest.raises(ValueError, match="no detector pixel scale"):
        SM.load_satstars("o999", w, (64, 64), filt, cat)


def test_a_module_tag_reads_the_observations_catalogue():
    assert SM.satstar_catalog_path("o135_nrca").endswith(
        "f480m_o135_consolidated_satstar_catalog.fits"
    )
    assert SM.satstar_catalog_path("o135_nrca", "f212n").endswith(
        "f212n_o135_consolidated_satstar_catalog.fits"
    )


# --- names: the masked layer is a flavour of its own -------------------------


def test_the_unmasked_names_are_unchanged():
    assert (
        os.path.basename(G.png_for("o112", "vminmax", True))
        == "GCTreasury_o112_RGB_480-mean-212_residual_vminmax.png"
    )
    assert (
        G.coadd_name_for("vminmax", residual=True)
        == "jwst_gc_treasury_residual_vminmax_hips"
    )
    assert (
        G.layer_tail(False, False, "bgmatch", True)
        == "_RGB_480-mean-212_residual_bgmatch_hips"
    )
    assert G.residual_suffix(False) == "" and G.residual_suffix(True) == "_residual"


@pytest.mark.parametrize("stretch", G.RESIDUAL_STRETCHES)
def test_a_satmask_layer_is_matched_by_its_own_glob_only(stretch):
    mine = os.path.basename(G.hips_for("o112", stretch, True, satmask=True))
    assert "_residual_satmask" in mine
    tail = G.layer_tail(False, False, stretch, True, satmask=True)
    assert fnmatch.fnmatchcase(mine, f"GCTreasury_*{tail}")
    for residual, s in G.nircam_flavours():
        other = os.path.basename(G.hips_for("o112", s, residual))
        other_tail = G.layer_tail(False, False, s, residual)
        assert not fnmatch.fnmatchcase(other, f"GCTreasury_*{tail}"), other
        assert not fnmatch.fnmatchcase(mine, f"GCTreasury_*{other_tail}"), (
            mine,
            other_tail,
        )
    assert G.coadd_name_for(stretch, residual=True, satmask=True) not in {
        G.coadd_name_for(s, residual=r) for r, s in G.nircam_flavours()
    }


# --- the command line: nothing behind main() that could touch the survey -------

# every directory G writes under, and every entry point main() or the builders reach
_SURVEY_DIRS = ("BASE", "OUTDIR", "SATSTAR_CATALOG_DIR", "MAST_L3", "WEB")
_SURVEY_WORK = (
    "cmd_list",
    "cmd_nircam_residual_match",
    "cmd_nircam_match",
    "cmd_miri_match",
    "cmd_auto",
    "cmd_coadd",
    "coadd_lock",
    "find_i2d",
    "find_residual_i2d",
    "inventory",
    "build_obs",
    "build_miri_obs",
    "build_mono_obs",
    "check_orientation",
)


def _sandboxed(monkeypatch, tmp_path, keep=()):
    """Cut G off from the survey for one test.

    Every directory it writes under points into `tmp_path`, and every entry point
    but those in `keep` records the call and raises.  A guard that stops refusing
    then fails the test with the name of what it reached, instead of running --auto,
    a match solve or a build on the real tree.  Returns the list of names reached.
    """
    ran = []

    def tripwire(name):
        def reached(*args, **kwargs):
            ran.append(name)
            raise AssertionError(f"{name} was reached; it must stay out of this test")

        return reached

    for name in _SURVEY_DIRS:
        monkeypatch.setattr(G, name, str(tmp_path / name.lower()))
    for name in _SURVEY_WORK:
        if name not in keep:
            monkeypatch.setattr(G, name, tripwire(name))
    return ran


def test_satmask_needs_residual(monkeypatch, tmp_path):
    ran = _sandboxed(monkeypatch, tmp_path, keep=("build_obs", "cmd_coadd"))
    with pytest.raises(ValueError):
        G.residual_suffix(False, satmask=True)
    with pytest.raises(ValueError):
        G.build_obs("o201", stretch="pct", residual=False, satmask=True)
    with pytest.raises(ValueError):
        G.cmd_coadd(miri=True, residual=True, satmask=True)
    with pytest.raises(ValueError):
        G.layer_tail(mono="f212n", satmask=True)
    assert ran == [] and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--miri", "--residual"],
        ["--mono", "f212n"],
        ["--residual", "--auto"],
        ["--residual", "--nircam-match"],
        ["--residual", "--list"],
    ],
)
def test_cli_refuses_the_flag_where_it_does_not_apply(monkeypatch, tmp_path, extra):
    ran = _sandboxed(monkeypatch, tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["gc_treasury_rgb_images.py", "--mask-satstar-cores"] + extra
    )
    with pytest.raises(SystemExit) as exc:
        G.main()
    assert exc.value.code == 2
    assert ran == [] and list(tmp_path.iterdir()) == []


def test_cli_passes_the_flag_to_build_obs_and_defaults_off(tmp_path, monkeypatch):
    ran = _sandboxed(monkeypatch, tmp_path)
    seen = []
    monkeypatch.setattr(
        G,
        "inventory",
        lambda residual=False: ({f: {"o201": "x"} for f in G.FILTERS}, ["o201"]),
    )
    monkeypatch.setattr(G, "build_obs", lambda o, **k: seen.append(k) or (None, None))
    for argv, want in (
        (["--residual", "--obs", "o201"], False),
        (["--residual", "--mask-satstar-cores", "--obs", "o201"], True),
    ):
        monkeypatch.setattr(sys, "argv", ["gc_treasury_rgb_images.py"] + argv)
        assert G.main() == 0
        assert seen[-1]["satmask"] is want and seen[-1]["residual"] is True
    assert ran == [] and list(tmp_path.iterdir()) == []


# --- build_obs, end to end on a tiny synthetic tile ------------------------------


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _tile(tmp_path, donut=("f480m", "f212n")):
    """F480M 200 px and F212N 400 px on the same sky, one saturated star in the
    middle with a donut around it in the filters of `donut` (radii in the pixels of
    each filter, the units the mask rule is in), and a catalogue for each filter."""
    n = 200
    out = {}
    cat = tmp_path / "cat"
    cat.mkdir()
    for f, size, pix in (("f480m", n, PIX), ("f212n", 2 * n, PIX / 2)):
        w = _wcs(size, pix)
        rng = np.random.default_rng(2 if f == "f480m" else 3)
        img = 10.0 + rng.normal(0, 0.3, (size, size))
        yy, xx = np.mgrid[:size, :size]
        rr = np.hypot(xx - (size - 1) / 2, yy - (size - 1) / 2)
        if f in donut:
            img[rr < 5] -= 150.0
            img[(rr >= 5) & (rr < 9)] += 60.0
            img[rr < 1.5] = np.nan  # saturated core
        h = fits.PrimaryHDU(img.astype(np.float32), header=w.to_header())
        path = tmp_path / f"{f}_residual.fits"
        h.writeto(path)
        out[f] = str(path)
        _catalogue(cat, w, [((size - 1) / 2, (size - 1) / 2, 80)], filt=f)
    return out, str(cat)


def _point_at(monkeypatch, tmp_path, paths, cat):
    monkeypatch.setattr(G, "OUTDIR", str(tmp_path / "out"))
    monkeypatch.setattr(G, "SATSTAR_CATALOG_DIR", cat)
    monkeypatch.setattr(
        G,
        "inventory",
        lambda residual=False: ({f: {"o999": p} for f, p in paths.items()}, ["o999"]),
    )


def _render(**kw):
    from PIL import Image

    png, hd = G.build_obs("o999", stretch="vminmax", residual=True, hips=False, **kw)
    assert hd is None
    return png, np.asarray(Image.open(png).convert("RGBA")).astype(int)


def test_build_obs_masks_the_rendered_copy_and_only_reads_the_fits(
    tmp_path, monkeypatch
):
    pytest.importorskip("pyavm")
    pytest.importorskip("reproject")
    paths, cat = _tile(tmp_path)
    _point_at(monkeypatch, tmp_path, paths, cat)
    shas = {f: _sha(p) for f, p in paths.items()}
    plain, a = _render()
    masked, b = _render(satmask=True)
    assert plain.endswith("_residual_vminmax.png") and masked.endswith(
        "_residual_satmask_vminmax.png"
    )
    assert {
        f: _sha(p) for f, p in paths.items()
    } == shas  # the residual i2d is only read
    changed = (a != b).any(axis=2)
    assert changed.any()
    ys, xs = np.nonzero(changed)
    # the change is the star's disk (centred, radius ~ 13 F480M px = 26 F212N px)
    # and nothing else
    assert (
        abs(ys.mean() - (a.shape[0] - 1) / 2) < 3
        and abs(xs.mean() - (a.shape[1] - 1) / 2) < 3
    )
    assert changed.mean() < 0.02
    assert max(ys.max() - ys.min(), xs.max() - xs.min()) < 70


def test_each_filter_is_masked_in_its_own_channel(tmp_path, monkeypatch):
    """R is F480M, B is F212N.  Both carry a donut at the star; the mask must clear
    the channel of exactly the filters it is asked to take."""
    pytest.importorskip("pyavm")
    pytest.importorskip("reproject")
    paths, cat = _tile(tmp_path)
    _point_at(monkeypatch, tmp_path, paths, cat)
    yy, xx = np.mgrid[:400, :400]
    rr = np.hypot(xx - 199.5, yy - 199.5)  # PNG pixels = F212N pixels
    b_ring = (rr >= 6) & (rr < 8)  # F212N ring: 5-9 F212N px
    r_ring = (rr >= 14) & (rr < 17)  # F480M ring: 5-9 F480M px = 10-18 F212N px
    far = (rr >= 60) & (rr < 100)

    def ring_excess(rgba):
        return (
            rgba[..., 2][b_ring].mean() - rgba[..., 2][far].mean(),
            rgba[..., 0][r_ring].mean() - rgba[..., 0][far].mean(),
        )

    b_plain, r_plain = ring_excess(_render()[1])
    assert b_plain > 30 and r_plain > 30  # the donut is in both channels
    for take, b_left, r_left in (
        (("f212n",), False, True),
        (("f480m",), True, False),
        (("f212n", "f480m"), False, False),
    ):
        monkeypatch.setattr(G, "SATMASK_FILTERS", take)
        b_ex, r_ex = ring_excess(_render(satmask=True)[1])
        assert (b_ex > 30) == b_left and (r_ex > 30) == r_left, (take, b_ex, r_ex)
        if not b_left:
            assert abs(b_ex) < 10
        if not r_left:
            assert abs(r_ex) < 10


def test_no_filter_outside_the_gate_is_masked(tmp_path, monkeypatch):
    pytest.importorskip("pyavm")
    pytest.importorskip("reproject")
    paths, cat = _tile(tmp_path)
    paths["f187n"] = paths["f480m"]  # a third filter in the build, with no catalogue
    _point_at(monkeypatch, tmp_path, paths, cat)
    monkeypatch.setattr(G, "FILTERS", ["f480m", "f212n", "f187n"])
    seen = []
    real = SM.mask_satstar_cores

    def spy(img, wcs, obs, filt="f480m", **kwargs):
        seen.append(filt)
        return real(img, wcs, obs, filt=filt, **kwargs)

    monkeypatch.setattr(SM, "mask_satstar_cores", spy)
    _render(satmask=True)
    assert sorted(seen) == ["f212n", "f480m"]
    seen.clear()
    _render()  # without the flag nothing is masked, whatever the gate says
    assert seen == []


@pytest.mark.parametrize("missing", ["f212n", "f480m"])
def test_a_missing_catalogue_stops_build_obs_before_it_writes_anything(
    tmp_path, monkeypatch, missing
):
    pytest.importorskip("pyavm")
    pytest.importorskip("reproject")
    paths, cat = _tile(tmp_path)
    os.remove(f"{cat}/{missing}_o999_consolidated_satstar_catalog.fits")
    _point_at(monkeypatch, tmp_path, paths, cat)
    with pytest.raises(FileNotFoundError, match=rf"o999.*{missing}_o999"):
        _render(satmask=True)
    assert os.listdir(G.OUTDIR) == []  # no png, no jpg, no HiPS
    # the catalogue is read by the flag and by nothing else
    plain, _ = _render()
    assert plain.endswith("_residual_vminmax.png") and os.path.exists(plain)
