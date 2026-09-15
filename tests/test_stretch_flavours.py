"""The STRETCHES table is the whole description of a rendering flavour.

Adding a flavour used to mean editing the table AND the hard-coded
stretch="asinh" at the call site.  These tests pin the properties the rest of
the pipeline assumes about the table: every entry is a complete simple_norm
call, every flavour gets its own filenames, and the default flavour's name
stays the historical un-suffixed one.
"""
import numpy as np
import pytest
from astropy.visualization import simple_norm

import gc_treasury_rgb_images as G


def test_every_flavour_names_its_stretch_function():
    for name, kw in G.STRETCHES.items():
        assert "stretch" in kw, f"{name} does not name a stretch function"


@pytest.mark.parametrize("name", sorted(G.STRETCHES))
def test_every_flavour_is_a_valid_simple_norm_call(name):
    d = np.linspace(-1, 1000, 64).reshape(8, 8)
    out = simple_norm(d, **G.STRETCHES[name])(d)
    assert np.isfinite(np.clip(out, 0, 1)).all()


def test_default_flavour_keeps_the_unsuffixed_names():
    assert G.stretch_suffix(G.DEFAULT_STRETCH) == ""
    assert G.coadd_name_for(G.DEFAULT_STRETCH) == G.COADD_NAME
    assert G.png_for("o112").endswith("_RGB_480-mean-212.png")


def test_flavour_names_do_not_collide():
    pngs = {G.png_for("o112", stretch=k) for k in G.STRETCHES}
    coadds = {G.coadd_name_for(k) for k in G.STRETCHES}
    assert len(pngs) == len(G.STRETCHES)
    assert len(coadds) == len(G.STRETCHES)


def test_nondefault_layer_names_are_excludable_from_the_default_glob():
    """cmd_coadd filters the default glob with f"_{other}_hips".

    A flavour whose layer name did not contain that exact token would be
    silently coadded into the default mosaic.
    """
    for k in G.STRETCHES:
        if k == G.DEFAULT_STRETCH:
            continue
        assert f"_{k}_hips" in G.hips_for("o112", stretch=k)
        assert f"_{k}_hips" in G.coadd_name_for(k)


def test_log_flavour_covers_its_stated_range():
    """-0.5 to 500 in MJy/sr, with the call site's clip doing the bounding."""
    kw = G.STRETCHES["log"]
    assert (kw["vmin"], kw["vmax"]) == (-0.5, 500)
    d = np.array([[-5.0, -0.5, 0.0, 500.0, 5000.0]])
    out = np.clip(simple_norm(d, **kw)(d), 0, 1)
    assert out[0, 0] == 0.0          # below vmin bottoms out
    assert out[0, 3] == pytest.approx(1.0, abs=1e-6)   # vmax reaches white
    assert out[0, 4] == 1.0          # above vmax stays white
    assert 0 < out[0, 2] < 1         # zero signal is not black under log
