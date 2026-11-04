"""The opt-in display mask of the accepted-satstar cores in the F480M residual.

What these pin: the radius rule, that the input array is never touched, that
only finite pixels inside disks (and NaN cores wholly inside them) change, that
a repeated or far-away catalogue row cannot widen the mask, that the masked
layer has a name no other flavour's glob can match, that the flag is refused
wherever it does not apply, and -- through build_obs -- that the residual i2d
on disk is only read.
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


def test_a_donut_is_replaced_by_background_and_the_input_is_untouched(tmp_path):
    n = 240
    w = _wcs(n, PIX)
    img = _sky(n)
    yy, xx = np.mgrid[:n, :n]
    rr = np.hypot(xx - 120, yy - 120)
    img[rr < 5] -= 150.0  # over-subtracted core
    img[(rr >= 5) & (rr < 9)] += 60.0  # ring
    before = img.copy()
    cat = _catalogue(tmp_path, w, [(120, 120, 80)])
    out, m, info = SM.mask_satstar_cores(img, w, "o999", catalog_dir=cat)
    assert np.array_equal(img, before)  # a copy, never in place
    r = float(SM.mask_radius_px([80], PIX)[0])
    assert info["n_used"] == 1 and info["n_dupes"] == 0
    assert np.array_equal(m, rr <= r) or abs(int(m.sum()) - int((rr <= r).sum())) <= 4
    assert np.array_equal(out[~m], img[~m])  # nothing outside the disk moves
    assert np.abs(out[rr < r - 1] - 10.0).max() < 2.0  # filled with the local sky
    assert info["frac_of_finite"] == pytest.approx(m.sum() / n / n)


def test_a_nan_core_inside_a_disk_is_filled_but_the_outer_footprint_is_not(tmp_path):
    n = 240
    w = _wcs(n, PIX)
    img = _sky(n)
    img[117:124, 117:124] = np.nan  # saturated core, wholly inside the disk
    img[:, :6] = np.nan  # outer footprint, touches the edge
    cat = _catalogue(
        tmp_path, w, [(120, 120, 80), (14, 120, 80)]
    )  # second disk overlaps the footprint edge
    out, m, info = SM.mask_satstar_cores(img, w, "o999", catalog_dir=cat)
    assert np.isfinite(out[117:124, 117:124]).all()
    assert info["n_nan_core_px_filled"] == 49
    assert np.isnan(out[:, :6]).all()  # alpha edge logic sees the same footprint
    assert np.array_equal(np.isnan(out[:, 6:]), np.zeros((n, n - 6), bool))


def test_a_repeated_or_far_row_does_not_widen_the_mask(tmp_path):
    n = 240
    w = _wcs(n, PIX)
    img = _sky(n)
    one = _catalogue(tmp_path, w, [(120, 120, 80)])
    _, m1, _ = SM.mask_satstar_cores(img, w, "o999", catalog_dir=one)
    # the same star listed twice (the larger sat_area wins), plus a row the
    # survey-wide injection would have added: a whole degree away
    many = _catalogue(
        tmp_path,
        w,
        [(120, 120, 80), (120.5, 120, 400), (120, 120, 80), (5000, 5000, 900)],
    )
    _, m2, info = SM.mask_satstar_cores(img, w, "o999", catalog_dir=many)
    assert (
        info["n_rows"] == 4
        and info["n_in_image"] == 3
        and info["n_dupes"] == 2
        and info["n_used"] == 1
    )
    assert m2.sum() > m1.sum()  # the largest duplicate wins: 400 px, not 80
    r400 = float(SM.mask_radius_px([400], PIX)[0])
    yy, xx = np.mgrid[:n, :n]
    assert int(m2.sum()) == pytest.approx(
        int((np.hypot(xx - 120, yy - 120) <= r400).sum()), abs=60
    )


def test_a_missing_catalogue_is_an_error_not_a_silent_no_op(tmp_path):
    w = _wcs(64, PIX)
    with pytest.raises(FileNotFoundError, match="o998"):
        SM.mask_satstar_cores(np.ones((64, 64)), w, "o998", catalog_dir=str(tmp_path))


def test_a_module_tag_reads_the_observations_catalogue():
    assert SM.satstar_catalog_path("o135_nrca").endswith(
        "f480m_o135_consolidated_satstar_catalog.fits"
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


def test_satmask_needs_residual():
    with pytest.raises(ValueError):
        G.residual_suffix(False, satmask=True)
    with pytest.raises(ValueError):
        G.build_obs("o201", stretch="pct", residual=False, satmask=True)
    with pytest.raises(ValueError):
        G.cmd_coadd(miri=True, residual=True, satmask=True)
    with pytest.raises(ValueError):
        G.layer_tail(mono="f212n", satmask=True)


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
def test_cli_refuses_the_flag_where_it_does_not_apply(monkeypatch, extra):
    monkeypatch.setattr(
        sys, "argv", ["gc_treasury_rgb_images.py", "--mask-satstar-cores"] + extra
    )
    with pytest.raises(SystemExit) as exc:
        G.main()
    assert exc.value.code == 2


def test_cli_passes_the_flag_to_build_obs_and_defaults_off(tmp_path, monkeypatch):
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


# --- build_obs, end to end on a tiny synthetic tile ------------------------------


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _tile(tmp_path):
    """F480M 200 px and F212N 400 px on the same sky, one saturated star in the
    middle."""
    n = 200
    out = {}
    for f, size, pix in (("f480m", n, PIX), ("f212n", 2 * n, PIX / 2)):
        w = _wcs(size, pix)
        rng = np.random.default_rng(2 if f == "f480m" else 3)
        img = 10.0 + rng.normal(0, 0.3, (size, size))
        yy, xx = np.mgrid[:size, :size]
        rr = (
            np.hypot(xx - (size - 1) / 2, yy - (size - 1) / 2) * pix / PIX
        )  # radius in F480M pixels
        if f == "f480m":
            img[rr < 5] -= 150.0
            img[(rr >= 5) & (rr < 9)] += 60.0
            img[rr < 1.5] = np.nan  # saturated core
        h = fits.PrimaryHDU(img.astype(np.float32), header=w.to_header())
        path = tmp_path / f"{f}_residual.fits"
        h.writeto(path)
        out[f] = str(path)
    w480 = _wcs(n, PIX)
    cat = tmp_path / "cat"
    cat.mkdir()
    _catalogue(cat, w480, [((n - 1) / 2, (n - 1) / 2, 80)])
    return out, str(cat)


def test_build_obs_masks_the_rendered_copy_and_only_reads_the_fits(
    tmp_path, monkeypatch
):
    pytest.importorskip("pyavm")
    pytest.importorskip("reproject")
    paths, cat = _tile(tmp_path)
    monkeypatch.setattr(G, "OUTDIR", str(tmp_path / "out"))
    monkeypatch.setattr(G, "SATSTAR_CATALOG_DIR", cat)
    monkeypatch.setattr(
        G,
        "inventory",
        lambda residual=False: ({f: {"o999": paths[f]} for f in G.FILTERS}, ["o999"]),
    )
    shas = {f: _sha(p) for f, p in paths.items()}
    plain, _ = G.build_obs("o999", stretch="vminmax", residual=True, hips=False)
    masked, hd = G.build_obs(
        "o999", stretch="vminmax", residual=True, hips=False, satmask=True
    )
    assert hd is None
    assert plain.endswith("_residual_vminmax.png") and masked.endswith(
        "_residual_satmask_vminmax.png"
    )
    assert {
        f: _sha(p) for f, p in paths.items()
    } == shas  # the residual i2d is only read
    from PIL import Image

    a = np.asarray(Image.open(plain).convert("RGBA")).astype(int)
    b = np.asarray(Image.open(masked).convert("RGBA")).astype(int)
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
