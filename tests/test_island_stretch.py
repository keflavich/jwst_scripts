"""jwst_rgb.island_stretch and the --islands / --red nbfull callers."""
import numpy as np
import pytest
from astropy.visualization import simple_norm
from astropy.wcs import WCS

from jwst_rgb.island_stretch import (apply_per_island, island_centres_galactic,
                                     island_labels, island_slices)


def _two_islands(ny=120, nx=200):
    """A faint island on the left, a bright one on the right, blank between;
    the faint one has a 3-px seam down its middle and a saturated-core hole."""
    rng = np.random.default_rng(0)
    data = np.full((ny, nx), np.nan, np.float32)
    data[10:110, 10:80] = rng.normal(20, 2, (100, 70))
    data[10:110, 120:190] = rng.normal(200, 20, (100, 70))
    data[10:110, 44:47] = np.nan          # seam inside the faint island
    data[50:60, 20:30] = np.nan           # saturated core
    return data


def _two_big_islands(ny=160, nx=260):
    """Two islands ~50 px apart, large enough to hold a hole wider than the
    closing can bridge."""
    rng = np.random.default_rng(0)
    data = np.full((ny, nx), np.nan, np.float32)
    data[10:150, 10:110] = rng.normal(20, 2, (140, 100))
    data[10:150, 160:250] = rng.normal(200, 20, (140, 90))
    return data


def test_islands_join_across_a_seam():
    data = _two_islands()
    labels, n = island_labels(np.isfinite(data), step=4, min_area_pix=100)
    assert n == 2
    left, right = labels[60, 15], labels[60, 150]
    assert left != right and 0 not in (left, right)
    assert labels[60, 45] == left         # the seam
    assert labels[60, 100] == 0           # blank sky between the islands


def test_a_seam_narrower_than_the_closing_is_bridged():
    """A 6-px blank seam (under two coarse blocks at step 4) splits one
    tile; the closing must rejoin it, and close_iter=0 must not."""
    data = np.full((160, 260), np.nan, np.float32)
    data[10:150, 10:200] = 1.0
    data[:, 100:106] = np.nan
    labels, n = island_labels(np.isfinite(data), step=4, min_area_pix=100)
    assert n == 1
    assert labels[80, 50] == labels[80, 150] != 0
    _, n0 = island_labels(np.isfinite(data), step=4, min_area_pix=100,
                          close_iter=0)
    assert n0 == 2


def test_a_hole_wider_than_the_closing_stays_in_its_island():
    """A 60x60 px core at step 4 is 15 blocks across, beyond what
    binary_closing(iterations=2) bridges, so only the hole fill keeps it."""
    data = _two_big_islands()
    data[40:100, 30:90] = np.nan
    labels, n = island_labels(np.isfinite(data), step=4, min_area_pix=100)
    assert n == 2
    assert labels[70, 60] == labels[20, 20] != 0


def test_a_small_island_borrows_the_nearest_large_islands_label():
    """A 16-px sliver ~25 px from either island, with no closing to merge it,
    is a separate small island and must take a large island's label."""
    data = _two_big_islands()
    data[60:64, 135:139] = 5
    labels, n = island_labels(np.isfinite(data), step=2, min_area_pix=100,
                              close_iter=0)
    assert n == 2
    assert labels[61, 136] in (labels[60, 50], labels[60, 200])
    assert labels[61, 136] != 0


def test_each_island_reaches_the_full_range():
    """The point of the flavour: a faint island stretched alone fills 0..1
    instead of sitting at the bottom of the global range."""
    data = _two_islands()
    lo, hi = np.nanpercentile(data, [1, 99.5])
    out = np.nan_to_num(np.clip((data - lo) / (hi - lo), 0, 1))
    labels, n = island_labels(np.isfinite(data), step=4, min_area_pix=100)
    faint = np.isfinite(data) & (labels == labels[60, 15])
    assert np.median(out[faint]) < 0.05
    recs = apply_per_island(
        out, data, labels, island_slices(labels, n),
        lambda v: tuple(np.percentile(v, [1, 99.5])),
        lambda c, a, b: np.clip((c - a) / (b - a), 0, 1), sample=1,
        min_count=10)
    assert len(recs) == 2 and all(r["levels"] for r in recs)
    assert 0.3 < np.median(out[faint]) < 0.7
    assert np.all(out[~np.isfinite(data) & (labels == 0)] == 0)


def test_an_island_with_too_little_data_keeps_the_global_value():
    """5 samples is below min_count (1000) but not zero, so the count check
    itself decides."""
    data = _two_big_islands()
    labels, n = island_labels(np.isfinite(data), step=4, min_area_pix=100)
    out = np.full(data.shape, 0.25, np.float32)
    chan = np.where(labels == labels[60, 200], data, np.nan)
    chan[20, 20:25] = 1.0
    recs = apply_per_island(out, chan, labels, island_slices(labels, n),
                            lambda v: (0.0, 1.0), lambda c, a, b: c * 0 + 1,
                            sample=1)
    left = [r for r in recs if r["label"] == labels[20, 20]][0]
    assert left["n_sampled"] == 5 and left["levels"] is None
    assert np.all(out[labels == labels[20, 20]] == 0.25)
    assert np.all(out[(labels == labels[60, 200]) & np.isfinite(chan)] == 1)


def test_zeros_are_left_out_of_the_sample():
    data = _two_big_islands()
    data[10:80, 10:110] = 0               # half the left island is zero fill
    labels, n = island_labels(np.isfinite(data), step=4, min_area_pix=100)
    out = np.zeros(data.shape, np.float32)
    recs = apply_per_island(out, data, labels, island_slices(labels, n),
                            lambda v: (float(v.min()),),
                            lambda c, a: c * 0, sample=1, min_count=10)
    assert all(r["levels"][0] > 0 for r in recs)


def test_a_flat_island_keeps_the_global_value():
    """A constant island has no range: the levels functions return None and
    the island keeps the global stretch instead of raising or dividing by
    zero."""
    import gc_treasury_mosaic_rgb_hips as G
    import gc_treasury_nb_miri_hips as M
    data = _two_big_islands()
    data[10:150, 10:110] = 30.0
    labels, n = island_labels(np.isfinite(data), step=4, min_area_pix=100)
    flat = labels == labels[20, 20]
    for levels_fn, stretch_fn in ((G._pct_levels, G._pct_apply),
                                  (M.island_levels, M.stretch)):
        out = np.full(data.shape, 0.25, np.float32)
        recs = apply_per_island(out, data, labels, island_slices(labels, n),
                                levels_fn, stretch_fn, sample=1)
        assert [r["levels"] is None for r in recs].count(True) == 1
        assert np.all(out[flat] == 0.25)
        assert np.all(np.isfinite(out))
    assert M.levels_from(np.full(100, 3.0)) is None
    assert G._pct_levels(np.full(100, 3.0)) is None


def test_centres_are_galactic():
    w = WCS(naxis=2)
    w.wcs.ctype = ["GLON-CAR", "GLAT-CAR"]
    w.wcs.crval = [0.5, -0.1]
    w.wcs.crpix = [1, 1]
    w.wcs.cdelt = [-1e-3, 1e-3]
    recs = island_centres_galactic([{"bbox": [[0, 1], [0, 1]]}], w)
    assert recs[0]["glon"] == pytest.approx(0.5, abs=1e-6)
    assert recs[0]["glat"] == pytest.approx(-0.1, abs=1e-6)


def test_rgb_island_stretch_matches_simple_norm():
    """--islands must reproduce the "pct" look within an island: the same
    percentiles and asinh as simple_norm, just solved per island."""
    import gc_treasury_mosaic_rgb_hips as G
    rng = np.random.default_rng(3)
    v = rng.lognormal(3, 1, 5000).astype(np.float32)
    ref = simple_norm(v, **G.STRETCHES["pct"])(v)
    mine = G._pct_apply(v, *G._pct_levels(v))
    np.testing.assert_allclose(mine, np.clip(ref, 0, 1), atol=2e-6)


def test_nbfull_and_islands_names():
    import gc_treasury_mosaic_rgb_hips as G
    names = {G.rgb_trio_hips_for("main", "pct", red, isl)
             for red in ("mosaic", "nb", "nbfull") for isl in (False, True)}
    assert len(names) == 6
    full = G.rgb_trio_hips_for("main", "pct", "nbfull")
    assert full.endswith("/gctreasury_nbfullmosaic_RGB_770-480-212_hips")
    assert G.rgb_trio_hips_for("main", "pct", "nbfull", True).endswith(
        "/gctreasury_nbfullmosaic_RGB_770-480-212_islands_hips")
    # the published nb layer keeps its name
    assert G.rgb_trio_hips_for("main", "pct", "nb").endswith(
        "/gctreasury_nbmosaic_RGB_770-480-212_hips")


@pytest.mark.parametrize("argv", [
    ["--grids", "770", "--rgb-only", "--islands"],                 # red mosaic
    ["--grids", "770", "--rgb-only", "--red", "nbfull", "--islands",
     "--stretch", "vminmax"],
    ["--grids", "770", "--rgb-only", "--red", "nbfull", "--which", "both"],
])
def test_islands_and_nbfull_usage_errors(argv):
    import gc_treasury_mosaic_rgb_hips as G
    with pytest.raises(SystemExit):
        G.main(argv)


def test_miri_levels_fall_back_to_linear_when_the_median_is_high():
    import gc_treasury_nb_miri_hips as M
    v = np.linspace(0, 1, 1001)           # median at 0.5 of the range
    lo, top, a, med = M.levels_from(v)
    assert a == 1e3
    y = M.stretch(np.array([med]), lo, top, a)
    assert y[0] == pytest.approx((med - lo) / (top - lo), abs=1e-5)
    v = np.random.default_rng(1).lognormal(3, 1, 20000)
    lo, top, a, med = M.levels_from(v)
    assert a < 1e3
    assert M.stretch(np.array([med]), lo, top, a)[0] == pytest.approx(
        M.TARGET, abs=1e-3)


def test_global_miri_levels_refuse_a_flat_reference(monkeypatch):
    import gc_treasury_nb_miri_hips as M
    from types import SimpleNamespace
    monkeypatch.setattr(M, "image_hdu",
                        lambda path: SimpleNamespace(data=np.full((70, 70), 3.0)))
    with pytest.raises(ValueError):
        M.levels()


def test_nb_miri_islands_needs_an_index():
    import gc_treasury_nb_miri_hips as M
    with pytest.raises(SystemExit):
        M.main(["--list", "--islands"])


@pytest.mark.parametrize("islands", [False, True])
def test_build_rgb_trio_runs_the_island_pass_only_with_islands(monkeypatch,
                                                               islands):
    """The --islands flag must reach _restretch_islands, with the raw R
    (before fill_nan) and the target WCS; without it the pass is skipped."""
    import gc_treasury_mosaic_rgb_hips as G

    class Stop(Exception):
        pass

    rng = np.random.default_rng(5)
    r = rng.lognormal(3, 1, (40, 50)).astype(np.float32)
    r[5:8, 5:8] = np.nan
    w = WCS(naxis=2)
    w.wcs.ctype = ["GLON-CAR", "GLAT-CAR"]
    calls = []
    monkeypatch.setattr(G, "_load_nb_miri", lambda path=None: (r.copy(), w))
    monkeypatch.setattr(G, "_reproject_onto",
                        lambda f, which, twcs, ny, nx:
                        rng.lognormal(2, 1, (ny, nx)).astype(np.float32))

    def spy(scaled, chans, r_raw, twcs):
        calls.append((r_raw, twcs))
        return scaled

    def stop(*a, **k):
        raise Stop

    monkeypatch.setattr(G, "_restretch_islands", spy)
    monkeypatch.setattr(G, "rgb_trio_png_for", stop)
    with pytest.raises(Stop):
        G.build_rgb_trio("main", "pct", hips=False, red="nbfull",
                         islands=islands)
    assert len(calls) == int(islands)
    if islands:
        r_raw, twcs = calls[0]
        assert np.isnan(r_raw[6, 6]) and twcs is w
