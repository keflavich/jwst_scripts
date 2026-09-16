"""Caller-level tests for gc_treasury_mosaic_rgb_hips.

The helpers this script calls (avm_for_saved_png, save_rgb, reproject_to_hips)
are already tested elsewhere (test_avm_roundtrip.py). What is NOT tested
anywhere else is that THIS caller uses them correctly: the right AVM builder,
with the right flip/transpose, save_rgb told not to build its own HiPS (we
build it ourselves via reproject_to_hips + patch_hips_dir so patch_hips_dir
actually runs), and product names that stay out of the cron's namespace. Every
one of these has broken a real layer before (see module docstring), so the
call site is what needs pinning, not just the helper.
"""
import os
import sys

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image

import gc_treasury_mosaic_rgb_hips as G

NY, NX = 24, 32


def _fake_wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [266.5, -28.7]
    w.wcs.crpix = [15.4, 11.6]              # off-centre, see test_avm_roundtrip
    w.wcs.cd = np.array([[-1e-5, 0], [0, 1e-5]])
    w.pixel_shape = (NX, NY)
    return w


def _write_mosaic(path, seed):
    rng = np.random.default_rng(seed)
    data = rng.uniform(0, 10, size=(NY, NX)).astype("float32")
    hdr = _fake_wcs().to_header()
    fits.PrimaryHDU(data=data, header=hdr).writeto(path, overwrite=True)


@pytest.fixture
def mosaics(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "MOSAIC_DIR", str(tmp_path))
    monkeypatch.setattr(G, "OUTDIR", str(tmp_path))
    for filt, seed in [(G.LONG_FILTER, 1), (G.SHORT_FILTER, 2), (G.MIRI_FILTER, 3)]:
        _write_mosaic(G.mosaic_path(filt, "main"), seed)
    return tmp_path


def test_names_stay_out_of_the_cron_namespace():
    """gc-treasury/pngs is the hourly cron's tree, rebuilt automatically and
    gated by pngs/.auto.lock; writing there or reusing its jwst_gc_treasury_*
    names risks exactly the collision avm-hips hit earlier today (a manual
    write racing a cron recoadd produced a coadd with 2,007 of 11,426 tiles)."""
    for which in ("main", "residual"):
        for stretch in G.STRETCHES:
            assert "/pngs/" not in G.rgb_png_for(which, stretch)
            assert "jwst_gc_treasury" not in G.rgb_hips_for(which, stretch)
            assert "/pngs/" not in G.rgb_trio_png_for(which, stretch)
            assert "jwst_gc_treasury" not in G.rgb_trio_hips_for(which, stretch)
        assert "/pngs/" not in G.miri_png_for(which)
        assert "jwst_gc_treasury" not in G.miri_hips_for(which)


def test_main_and_residual_names_do_not_collide():
    assert G.rgb_png_for("main", "pct") != G.rgb_png_for("residual", "pct")
    assert G.rgb_trio_png_for("main", "pct") != G.rgb_trio_png_for("residual", "pct")
    assert G.miri_png_for("main") != G.miri_png_for("residual")


def test_the_two_rgb_grids_do_not_collide():
    """480-grid and 770-grid RGBs are different products (different target
    grid, different channel semantics -- one has a synthesised mean green,
    the other three real filters) and must never share a filename."""
    for which in ("main", "residual"):
        for stretch in G.STRETCHES:
            assert G.rgb_png_for(which, stretch) != G.rgb_trio_png_for(which, stretch)
            assert G.rgb_hips_for(which, stretch) != G.rgb_trio_hips_for(which, stretch)


def _patch_save_rgb(monkeypatch):
    """Patch jwst_rgb.save_rgb's avm_for_saved_png/faithful_avm/save_rgb and
    return the dict build_rgb/build_rgb_trio's calls get recorded into.

    Shared by every AVM-correctness test below because getting the module
    object right is fiddly: jwst_rgb/__init__.py does
    `from .save_rgb import save_rgb`, which rebinds the ATTRIBUTE
    jwst_rgb.save_rgb to the function on package import -- so
    `import jwst_rgb.save_rgb as x` resolves `x` via that shadowed attribute,
    not the actual submodule. sys.modules is unambiguous.
    """
    import sys
    import jwst_rgb.save_rgb  # noqa: F401  (ensure it is in sys.modules)
    calls = {}

    def fake_avm_for_saved_png(wcs, ny, nx, flip=-1, transpose=None):
        calls["avm"] = dict(ny=ny, nx=nx, flip=flip, transpose=transpose)
        return "AVM-SENTINEL"

    def fake_faithful_avm(*a, **k):
        raise AssertionError("faithful_avm must not be used for a save_rgb PNG")

    def fake_save_rgb(img, filename, avm=None, transpose=None, hips=None, **k):
        calls["save_rgb"] = dict(avm=avm, transpose=transpose, hips=hips,
                                 filename=filename)
        open(filename, "w").close()

    save_rgb_mod = sys.modules["jwst_rgb.save_rgb"]
    monkeypatch.setattr(save_rgb_mod, "avm_for_saved_png", fake_avm_for_saved_png)
    monkeypatch.setattr(save_rgb_mod, "faithful_avm", fake_faithful_avm)
    # build_rgb/build_rgb_trio do `from jwst_rgb.save_rgb import save_rgb as
    # _save_rgb` and `import avm_for_saved_png` at call time, so patching the
    # module's attributes (above) is what that fresh import picks up.
    monkeypatch.setattr(save_rgb_mod, "save_rgb", fake_save_rgb)
    return calls


def test_build_rgb_uses_avm_for_saved_png_not_faithful_avm(mosaics, monkeypatch):
    """The bug this pins: faithful_avm (or a raw AVM.from_header) describes the
    FITS array, not the PNG save_rgb actually writes, and is off by
    |N+1-2*crpix| pixels per axis. avm_for_saved_png is the only builder that
    reflects CRPIX to match save_rgb's flip+ROTATE_180."""
    calls = _patch_save_rgb(monkeypatch)
    monkeypatch.setattr(G, "_build_hips", lambda png, hips_dir: hips_dir)

    G.build_rgb("main", stretch="pct", hips=True)

    assert calls["avm"]["flip"] == -1
    assert calls["avm"]["transpose"] == Image.ROTATE_180
    assert calls["save_rgb"]["avm"] == "AVM-SENTINEL"
    # save_rgb must NOT build its own HiPS: G._build_hips is the one call site
    # that runs reproject_to_hips AND patch_hips_dir together (see next test).
    assert calls["save_rgb"]["hips"] is False


def test_build_rgb_trio_uses_avm_for_saved_png_not_faithful_avm(mosaics, monkeypatch):
    """Same bug, same fix, second call site: the F770W-grid trio needs its
    own AVM built from ITS OWN (F770W) grid, not reused from build_rgb."""
    calls = _patch_save_rgb(monkeypatch)
    monkeypatch.setattr(G, "_build_hips", lambda png, hips_dir: hips_dir)

    G.build_rgb_trio("main", stretch="pct", hips=True)

    assert calls["avm"]["flip"] == -1
    assert calls["avm"]["transpose"] == Image.ROTATE_180
    assert calls["avm"]["ny"] == NY and calls["avm"]["nx"] == NX
    assert calls["save_rgb"]["avm"] == "AVM-SENTINEL"
    assert calls["save_rgb"]["hips"] is False


def test_build_rgb_trio_targets_f770w_grid_not_f480m(mosaics, monkeypatch):
    """build_rgb targets F480M's grid; build_rgb_trio must target F770W's --
    mixing them up would silently apply the wrong reprojection direction."""
    seen_targets = []
    real = G._reproject_onto

    def spy(filt, which, twcs, ny, nx):
        seen_targets.append(filt)
        return real(filt, which, twcs, ny, nx)

    monkeypatch.setattr(G, "_reproject_onto", spy)
    _patch_save_rgb(monkeypatch)
    monkeypatch.setattr(G, "_build_hips", lambda png, hips_dir: hips_dir)

    G.build_rgb_trio("main", stretch="pct", hips=True)

    # F770W itself is the TARGET (loaded via _load_primary, not reprojected);
    # only the other two filters get reprojected onto its grid.
    assert G.MIRI_FILTER not in seen_targets
    assert set(seen_targets) == {G.LONG_FILTER, G.SHORT_FILTER}


def test_build_rgb_trio_does_not_apply_the_mixed_nan_mask(mosaics, monkeypatch):
    """build_rgb's _mask_mixed_nan rule -- a pixel real in one NIRCam channel
    and NaN in the other is a bug -- does not hold here: 10678's MIRI
    parallel points several arcmin off the NIRCam prime, so most of the
    full-survey F770W footprint legitimately has no NIRCam coverage at all.
    Applying that mask would blank those real F770W pixels."""
    def boom(*a, **k):
        raise AssertionError(
            "build_rgb_trio must not call _mask_mixed_nan: a NIRCam-less "
            "pixel here is real footprint, not a mixed-coverage bug")

    monkeypatch.setattr(G, "_mask_mixed_nan", boom)
    _patch_save_rgb(monkeypatch)
    monkeypatch.setattr(G, "_build_hips", lambda png, hips_dir: hips_dir)

    G.build_rgb_trio("main", stretch="pct", hips=True)  # must not raise


def test_build_hips_patches_after_reprojecting(tmp_path, monkeypatch):
    """patch_hips_dir has to run on every build, not just sometimes: it is
    what turns on Aladin Lite's settings control (where the reticle toggle
    lives). A call site that reprojects without patching silently ships a
    HiPS with that control missing."""
    calls = []
    hips_dir = tmp_path / "some_hips"

    def fake_reproject_to_hips(png, **kw):
        calls.append("reproject")
        os.makedirs(hips_dir / "Norder3")

    def fake_patch(d):
        calls.append("patch")
        assert d == str(hips_dir)
        assert os.path.isdir(os.path.join(d, "Norder3")), \
            "patch_hips_dir must run AFTER the HiPS tree exists"

    import reproject.hips as hips_mod
    import jwst_rgb.landing_page as lp_mod
    monkeypatch.setattr(hips_mod, "reproject_to_hips", fake_reproject_to_hips)
    monkeypatch.setattr(lp_mod, "patch_hips_dir", fake_patch)

    png = tmp_path / "x.png"
    open(png, "w").close()
    G._build_hips(str(png), str(hips_dir))

    assert calls == ["reproject", "patch"]


def test_build_rgb_raises_if_a_mosaic_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "MOSAIC_DIR", str(tmp_path))
    monkeypatch.setattr(G, "OUTDIR", str(tmp_path))
    with pytest.raises(RuntimeError):
        G.build_rgb("main")


def test_mask_mixed_nan_never_leaves_a_pixel_real_in_only_one_channel():
    long_ = np.array([[1.0, np.nan, 3.0]])
    short_ = np.array([[1.0, 2.0, np.nan]])
    long_out, short_out = G._mask_mixed_nan(long_.copy(), short_.copy())
    mixed_before = np.isnan(long_) ^ np.isnan(short_)
    assert mixed_before.any(), "test input no longer exercises the mixed case"
    assert np.array_equal(np.isnan(long_out), np.isnan(short_out))


def test_rgb_only_and_miri_only_together_is_a_usage_error():
    """They read as filters, not switches -- taking both used to silently
    build nothing and exit 0. A mutually exclusive group turns that into an
    argparse error instead of a quiet no-op."""
    with pytest.raises(SystemExit):
        G.main(["--rgb-only", "--miri-only"])


def test_build_rgb_holds_only_float32(mosaics, monkeypatch):
    """Peak memory for this call is documented in the module docstring on the
    assumption everything is float32; np.nanmean(np.stack(...)) or a missed
    .astype would silently double it back to float64."""
    seen = {}

    def fake_save_rgb(img, filename, avm=None, original_data=None, **k):
        seen["img_dtype"] = img.dtype
        seen["original_data_dtype"] = original_data.dtype
        open(filename, "w").close()

    import sys
    save_rgb_mod = sys.modules["jwst_rgb.save_rgb"]
    monkeypatch.setattr(save_rgb_mod, "save_rgb", fake_save_rgb)
    monkeypatch.setattr(save_rgb_mod, "avm_for_saved_png",
                        lambda *a, **k: "AVM-SENTINEL")
    monkeypatch.setattr(G, "_build_hips", lambda png, hips_dir: hips_dir)

    G.build_rgb("main", stretch="pct", hips=True)

    assert seen["img_dtype"] == np.float32
    assert seen["original_data_dtype"] == np.float32
