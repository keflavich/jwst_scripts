"""NIRCam background matching: the per-filter solve and the staleness rules.

The offsets are solved SEPARATELY per filter because this is a colour image.
One offset applied to both would move brightness without fixing colour, and a
wrong relative offset between F480M and F212N tints a whole tile -- more
visible than a brightness seam, not less.  These tests pin that the two
filters stay independent and that the staleness rules learned from the MIRI
bugs carry over.
"""
import json
import os
import time

import numpy as np
import pytest

G = pytest.importorskip("gc_treasury_rgb_images")


def test_solve_offsets_recovers_injected_levels():
    """Least squares on pairwise differences, with the mean-zero constraint."""
    keys = ["o1", "o2", "o3"]
    truth = {"o1": +0.5, "o2": -0.2, "o3": -0.3}      # already mean-zero
    pairs = [(a, b, truth[a] - truth[b], 10000)
             for i, a in enumerate(keys) for b in keys[i + 1:]]
    off = G.miri_solve_offsets(keys, pairs)
    for k in keys:
        assert off[k] == pytest.approx(truth[k], abs=1e-9)


def test_solve_offsets_is_mean_zero():
    """The constraint exists so matching does not drag the mosaic's overall
    level up or down."""
    keys = ["o1", "o2", "o3"]
    pairs = [("o1", "o2", 3.0, 1000), ("o2", "o3", 3.0, 1000)]
    off = G.miri_solve_offsets(keys, pairs)
    assert np.mean([off[k] for k in keys]) == pytest.approx(0.0, abs=1e-9)
    assert off["o1"] - off["o2"] == pytest.approx(3.0, abs=1e-9)
    assert off["o2"] - off["o3"] == pytest.approx(3.0, abs=1e-9)


def test_solve_offsets_leaves_unconnected_fields_at_zero():
    """A field with no overlap has nothing tying it to the rest; zero is the
    honest answer rather than an invented shift."""
    keys = ["o1", "o2", "lonely"]
    pairs = [("o1", "o2", 1.0, 5000)]
    off = G.miri_solve_offsets(keys, pairs)
    assert off["lonely"] == pytest.approx(0.0, abs=1e-6)


def test_solve_offsets_weights_by_shared_area():
    """A pair sharing more sky should dominate a contradictory small one."""
    keys = ["a", "b"]
    big = G.miri_solve_offsets(keys, [("a", "b", 2.0, 1_000_000),
                                      ("a", "b", -2.0, 1)])
    assert big["a"] - big["b"] > 1.9


@pytest.fixture
def match_file(tmp_path, monkeypatch):
    path = tmp_path / "nircam_background_match.json"
    doc = {"filters": {"f480m": {"offsets": {"o127": 1.0, "o128": -1.0},
                                 "vmin": 1.0, "vmax": 10.0},
                       "f212n": {"offsets": {"o127": 0.5, "o128": -0.5},
                                 "vmin": 2.0, "vmax": 20.0}},
           "green": {"vmin": 1.5, "vmax": 15.0}}
    path.write_text(json.dumps(doc))
    monkeypatch.setattr(G, "NIRCAM_MATCH_JSON", str(path))
    return path, doc


def test_green_limits_are_the_mean_of_the_two_filters(match_file):
    """Green is the pixelwise mean of the two filters, so a pixel sitting at
    both filters' vmin must land exactly at the green vmin."""
    _, doc = match_file
    lo = np.mean([doc["filters"][f]["vmin"] for f in ("f480m", "f212n")])
    hi = np.mean([doc["filters"][f]["vmax"] for f in ("f480m", "f212n")])
    assert doc["green"]["vmin"] == pytest.approx(lo)
    assert doc["green"]["vmax"] == pytest.approx(hi)


def test_load_nircam_match_round_trips(match_file):
    path, doc = match_file
    assert G.load_nircam_match() == doc


def test_load_nircam_match_is_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "NIRCAM_MATCH_JSON", str(tmp_path / "absent.json"))
    assert G.load_nircam_match() is None


def _inv(tmp_path, obs, age=10000):
    """Two settled source files for one observation."""
    inv = {}
    for f in G.FILTERS:
        p = tmp_path / f"{obs}_{f}.fits"
        p.write_bytes(b"0")
        old = time.time() - age
        os.utime(p, (old, old))
        inv[f] = {obs: str(p)}
    return inv


def test_needs_build_nomatch_when_field_absent_from_table(
        tmp_path, monkeypatch, match_file):
    """Same rule as MIRI: the table existing is not enough, because build_obs
    raises on a field missing from it."""
    inv = _inv(tmp_path, "o999")
    assert G.needs_build("o999", inv, stretch="bgmatch") == "NOMATCH"


def test_needs_build_plain_ignores_the_match_table(tmp_path, monkeypatch, match_file):
    """Only the background-matched flavour consults it."""
    inv = _inv(tmp_path, "o999")
    monkeypatch.setattr(G, "png_for", lambda o, stretch=None, residual=False: str(tmp_path / "absent.png"))
    assert G.needs_build("o999", inv, stretch="pct") == "no RGB yet"


def test_needs_build_flags_a_refreshed_match_as_stale(
        tmp_path, monkeypatch, match_file):
    """A new solution changes the offset and the shared cuts for EVERY tile,
    not only the ones whose data moved, so source mtime alone cannot catch it.
    """
    path, _ = match_file
    inv = _inv(tmp_path, "o127")
    png = tmp_path / "o127.png"
    png.write_bytes(b"0")
    hips = tmp_path / "o127_hips" / "Norder3"
    hips.mkdir(parents=True)
    monkeypatch.setattr(G, "png_for", lambda o, stretch=None, residual=False: str(png))
    monkeypatch.setattr(G, "hips_for", lambda o, stretch=None, residual=False: str(tmp_path / "o127_hips"))

    old = time.time() - 500
    os.utime(png, (old, old))
    os.utime(path, (old - 100, old - 100))
    assert G.needs_build("o127", inv, stretch="bgmatch") is None

    now = time.time()
    os.utime(path, (now, now))
    assert G.needs_build("o127", inv, stretch="bgmatch") == \
        "background match is newer than the png"


def test_plain_and_bgmatch_have_distinct_names():
    """The two flavours must not collide on disk, and the plain coadd glob
    must not swallow the bgmatch layers."""
    assert G.png_for("o127", "pct") != G.png_for("o127", "bgmatch")
    assert G.hips_for("o127", "pct") != G.hips_for("o127", "bgmatch")
    assert G.png_for("o127", "bgmatch").endswith("_bgmatch.png")
    assert "_bgmatch_hips" in G.hips_for("o127", "bgmatch")
    assert G.COADD_NAME != G.coadd_name_for("bgmatch")


def test_background_match_binning_stays_off():
    """The comment says do not re-enable; this makes that a check.

    Block-averaging by 8 cuts the reprojected area by 64 in a solve that takes
    over two hours unbinned (2 h 22 min image, 2 h 19 min residual, 68 tiles
    per filter, 2026-10-23), which is why someone will try it.  Measured
    against values the unbinned path had already produced:

        pair          unbinned     bin=8      error
        o098-o105       0.1317   -0.6077    -0.7394
        o098-o107       1.2270    0.5990    -0.6280
        o102-o109       0.0447   -0.5624    -0.6071

    A roughly constant -0.6, the same size as the offsets being solved for.
    Strict NaN propagation did not help, ruling out the footprint-edge
    explanation, and factor 1 reproduced the unbinned values to -0.0000, so the
    bias belongs to the binning rather than to the comparison.  The mechanism
    is unconfirmed.

    Raising BG_MATCH_BIN should fail here with that reason rather than quietly
    corrupting every field's correction.  Re-run the comparison first.
    """
    assert G.BG_MATCH_BIN == 1


# ---- the residual bgmatch flavour: its own table, residual vminmax's cuts ----
#
# The residuals keep the diffuse background, so the image layers' seams show
# in them too, and they get a bgmatch flavour of their own.  Its offsets are
# solved on the residual mosaics into a separate table; residual vminmax and
# log stay uncorrected.  These pin that neither table stands in for the other.

def _tables(tmp_path, monkeypatch, image=None, residual=None):
    """Point both match tables at tmp files; write the ones given."""
    paths = {}
    for attr, doc in (("NIRCAM_MATCH_JSON", image),
                      ("NIRCAM_RESIDUAL_MATCH_JSON", residual)):
        p = tmp_path / f"{attr.lower()}.json"
        if doc is not None:
            p.write_text(json.dumps(doc))
        monkeypatch.setattr(G, attr, str(p))
        paths[attr] = p
    return paths


def _offsets(obs):
    return {"filters": {f: {"offsets": {obs: 0.25}} for f in G.FILTERS}}


def test_residual_bgmatch_takes_residual_vminmax_cuts():
    assert G.stretch_match("bgmatch", residual=True) == "nircam_residual"
    assert G.stretch_kwargs("bgmatch", residual=True) == \
        G.stretch_kwargs("vminmax", residual=True)
    assert not G.stretch_cuts_from_match("bgmatch", residual=True)
    # the image flavour keeps taking its cuts from its own table
    assert G.stretch_match("bgmatch") == "nircam"
    assert G.stretch_cuts_from_match("bgmatch")


def test_residual_bgmatch_names():
    assert G.png_for("o105", "bgmatch", True).endswith(
        "GCTreasury_o105_RGB_480-mean-212_residual_bgmatch.png")
    assert G.hips_for("o105", "bgmatch", True).endswith(
        "GCTreasury_o105_RGB_480-mean-212_residual_bgmatch_hips")
    assert G.coadd_name_for("bgmatch", residual=True) == \
        "jwst_gc_treasury_residual_bgmatch_hips"
    assert G.nircam_match_json("nircam_residual") != G.nircam_match_json()


@pytest.mark.parametrize("covered_by", ["image", "residual"])
def test_each_bgmatch_flavour_reads_only_its_own_table(
        tmp_path, monkeypatch, covered_by):
    """A field covered by the other flavour's table is still NOMATCH."""
    monkeypatch.setattr(G, "OUTDIR", str(tmp_path))
    _tables(tmp_path, monkeypatch,
            image=_offsets("o127") if covered_by == "image" else None,
            residual=_offsets("o127") if covered_by == "residual" else None)
    inv = _inv(tmp_path, "o127")
    image = G.needs_build("o127", inv, stretch="bgmatch")
    resid = G.needs_build("o127", inv, stretch="bgmatch", residual=True)
    if covered_by == "image":
        assert (image, resid) == ("no RGB yet", "NOMATCH")
    else:
        assert (image, resid) == ("NOMATCH", "no RGB yet")


def test_a_refreshed_residual_match_rebuilds_only_the_residual_flavour(
        tmp_path, monkeypatch):
    monkeypatch.setattr(G, "OUTDIR", str(tmp_path))
    paths = _tables(tmp_path, monkeypatch, image=_offsets("o127"),
                    residual=_offsets("o127"))
    inv = _inv(tmp_path, "o127")
    old = time.time() - 500
    for residual, stretch in ((False, "bgmatch"), (True, "bgmatch"),
                              (True, "vminmax")):
        png = G.png_for("o127", stretch, residual)
        open(png, "wb").close()
        os.utime(png, (old, old))
        os.makedirs(os.path.join(G.hips_for("o127", stretch, residual),
                                 "Norder3"))
    for p in paths.values():
        os.utime(p, (old - 100, old - 100))
    assert G.needs_build("o127", inv, stretch="bgmatch") is None
    assert G.needs_build("o127", inv, stretch="bgmatch", residual=True) is None

    now = time.time()
    os.utime(paths["NIRCAM_RESIDUAL_MATCH_JSON"], (now, now))
    assert G.needs_build("o127", inv, stretch="bgmatch", residual=True) == \
        "background match is newer than the png"
    assert G.needs_build("o127", inv, stretch="bgmatch") is None
    # the uncorrected residual layer reads neither table
    assert G.needs_build("o127", inv, stretch="vminmax", residual=True) is None

    os.utime(paths["NIRCAM_RESIDUAL_MATCH_JSON"], (old - 100, old - 100))
    os.utime(paths["NIRCAM_MATCH_JSON"], (now, now))
    assert G.needs_build("o127", inv, stretch="bgmatch") == \
        "background match is newer than the png"
    assert G.needs_build("o127", inv, stretch="bgmatch", residual=True) is None


def test_residual_match_writes_offsets_to_its_own_table(tmp_path, monkeypatch):
    """Offsets and pairs only, never the image table, and a residual older
    than its image left out of the solve."""
    paths = _tables(tmp_path, monkeypatch)
    old = time.time() - 86400

    def aged(name, t):
        p = tmp_path / name
        p.write_bytes(b"0")
        os.utime(p, (t, t))
        return str(p)
    obs = ("o1", "o2", "o3")
    res = {f: {o: aged(f"res_{f}_{o}", old) for o in obs} for f in G.FILTERS}
    img = {f: {o: aged(f"img_{f}_{o}", old - 3600) for o in obs}
           for f in G.FILTERS}
    # o3's F480M image was re-reduced after its residual was written
    os.utime(img["f480m"]["o3"], (old + 3600, old + 3600))
    monkeypatch.setattr(G, "inventory", lambda residual=False: (
        (res if residual else img), list(obs)))
    calls = []

    def measure(p):
        keys = sorted(p)
        calls.append(keys)
        return keys, [(a, b, 0.5, 1000) for a, b in zip(keys, keys[1:])]
    monkeypatch.setattr(G, "miri_measure_pairs", measure)
    assert G.cmd_nircam_residual_match() == 0
    assert calls == [["o1", "o2"], ["o1", "o2", "o3"]]       # f480m, f212n
    doc = json.loads(paths["NIRCAM_RESIDUAL_MATCH_JSON"].read_text())
    assert set(doc) == {"filters"}
    for f in G.FILTERS:
        assert set(doc["filters"][f]) == {"offsets", "pairs"}
    assert set(doc["filters"]["f480m"]["offsets"]) == {"o1", "o2"}
    assert not paths["NIRCAM_MATCH_JSON"].exists()
    assert G.load_nircam_match("nircam_residual") == doc
    assert G.load_nircam_match() is None


@pytest.mark.parametrize("residual", [False, True])
def test_cli_nircam_match_picks_the_table(monkeypatch, residual):
    import sys
    called = []
    monkeypatch.setattr(G, "cmd_nircam_match",
                        lambda: called.append("image") or 0)
    monkeypatch.setattr(G, "cmd_nircam_residual_match",
                        lambda: called.append("residual") or 0)
    monkeypatch.setattr(sys, "argv", ["gc_treasury_rgb_images.py",
                                      "--nircam-match"] +
                        (["--residual"] if residual else []))
    assert G.main() == 0
    assert called == (["residual"] if residual else ["image"])


def test_cli_refuses_a_residual_miri_match(monkeypatch):
    import sys
    monkeypatch.setattr(sys, "argv", ["gc_treasury_rgb_images.py",
                                      "--miri-match", "--residual"])
    with pytest.raises(SystemExit) as exc:
        G.main()
    assert exc.value.code == 2


# ---- partial coverage, the lock report, and the exit-1 retry ----------------
#
# Both solves drop a SETTLING or VANISHED mosaic per filter, and the residual
# solve a CHAIN one, so either table can hold a tile's offset in one filter
# only.  needs_build calls that NOMATCH, the same bare string callers compare;
# --list says which filter is missing, build_obs refuses the field, and the
# lock report counts NOMATCH fields per flavour without calling them pending
# work.

def _fields(tmp_path, obs):
    """One settled inventory holding every field in `obs`."""
    inv = {f: {} for f in G.FILTERS}
    for o in obs:
        for f, paths in _inv(tmp_path, o).items():
            inv[f].update(paths)
    return inv


PARTIAL = {"filters": {"f480m": {"offsets": {"o127": 0.1}},
                       "f212n": {"offsets": {"o127": 0.1, "o128": 0.2}}}}


def test_partial_coverage_is_still_the_bare_nomatch(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "OUTDIR", str(tmp_path))
    _tables(tmp_path, monkeypatch, image=PARTIAL, residual=PARTIAL)
    inv = _fields(tmp_path, ("o127", "o128", "o129"))
    for residual in (False, True):
        kw = dict(stretch="bgmatch", residual=residual,
                  image_inv=inv if residual else None)
        assert G.needs_build("o127", inv, **kw) == "no RGB yet"
        assert G.needs_build("o128", inv, **kw) == "NOMATCH"
        assert G.needs_build("o129", inv, **kw) == "NOMATCH"
    assert G.missing_offsets("o128", PARTIAL) == ["f480m"]
    assert G.missing_offsets("o129", PARTIAL) == list(G.FILTERS)
    assert G.missing_offsets("o127", PARTIAL) == []
    assert G.missing_offsets("o127", None) == list(G.FILTERS)


def test_list_names_the_filter_a_partial_match_is_missing(tmp_path,
                                                          monkeypatch):
    monkeypatch.setattr(G, "OUTDIR", str(tmp_path))
    _tables(tmp_path, monkeypatch, image=PARTIAL, residual=PARTIAL)
    inv = _fields(tmp_path, ("o127", "o128", "o129"))
    monkeypatch.setattr(G, "inventory", lambda residual=False: (
        inv, ["o127", "o128", "o129"]))
    lines = G._match_status_lines(inv)
    # once for the image flavour, once for the residual one
    assert lines.count("    NOMATCH (no F480M offset) (1): o128") == 2
    assert lines.count("    NOMATCH (not in the table) (1): o129") == 2
    assert lines.count("    no RGB yet (1): o127") == 2


def test_list_keeps_the_bare_nomatch_without_a_table(tmp_path, monkeypatch):
    """The header line already says the table is not on disk."""
    monkeypatch.setattr(G, "OUTDIR", str(tmp_path))
    _tables(tmp_path, monkeypatch)
    inv = _fields(tmp_path, ("o127",))
    monkeypatch.setattr(G, "inventory", lambda residual=False: (inv, ["o127"]))
    lines = G._match_status_lines(inv)
    assert lines.count("    NOMATCH (1): o127") == 2
    assert sum("not on disk" in ln for ln in lines) == 2


def test_list_output_keeps_the_line_the_cron_greps_for(tmp_path, monkeypatch,
                                                       capsys):
    """gc_treasury_cron.sh submits only when --list prints "complete in all 2
    filters: o..."; the match block must neither drop nor imitate it."""
    monkeypatch.setattr(G, "OUTDIR", str(tmp_path))
    monkeypatch.setattr(G, "BASE", str(tmp_path))
    _tables(tmp_path, monkeypatch, image=PARTIAL, residual=PARTIAL)
    inv = _fields(tmp_path, ("o127", "o128"))
    monkeypatch.setattr(G, "inventory", lambda residual=False: (
        inv, ["o127", "o128"]))
    monkeypatch.setattr(G, "find_i2d", lambda f, exposure_level=False: {})
    assert G.cmd_list() == 0
    out = capsys.readouterr().out.splitlines()
    grepped = [ln for ln in out if "complete in all 2 filters: o" in ln]
    assert grepped == ["  complete in all 2 filters: o127, o128"]
    assert "    NOMATCH (no F480M offset) (1): o128" in out


def test_lock_report_counts_nomatch_per_flavour_outside_pending(
        tmp_path, monkeypatch):
    """#55 dropped NOMATCH from the pending list, which also hid the image
    bgmatch fields; they come back as one waiting line per flavour."""
    monkeypatch.setattr(G, "OUTDIR", str(tmp_path))
    image = {"filters": {f: {"offsets": {"o127": 0.1}} for f in G.FILTERS}}
    _tables(tmp_path, monkeypatch, image=image, residual=PARTIAL)
    inv = _fields(tmp_path, ("o127", "o128", "o129"))
    monkeypatch.setattr(G, "inventory", lambda residual=False: (
        inv, ["o127", "o128", "o129"]))
    monkeypatch.setattr(G, "find_i2d", lambda filt, **k: {})
    monkeypatch.setattr(G, "find_residual_i2d", lambda filt, **k: {})
    pending, waiting = G._pending_summary()
    assert waiting == [
        "2 field(s) waiting on --nircam-match (NIRCam/bgmatch)",
        "2 field(s) waiting on --nircam-match --residual "
        "(NIRCam/residual/bgmatch)"]
    assert "o127 NIRCam/bgmatch -- no RGB yet" in pending
    assert "o127 NIRCam/residual/bgmatch -- no RGB yet" in pending
    assert not [ln for ln in pending
                if "bgmatch" in ln and not ln.startswith("o127 ")]
    # the uncorrected flavours of the same fields are ordinary pending work
    assert "o129 NIRCam/vminmax -- no RGB yet" in pending


def test_lock_report_has_no_waiting_line_when_every_field_is_matched(
        tmp_path, monkeypatch):
    monkeypatch.setattr(G, "OUTDIR", str(tmp_path))
    _tables(tmp_path, monkeypatch, image=_offsets("o127"),
            residual=_offsets("o127"))
    inv = _fields(tmp_path, ("o127",))
    monkeypatch.setattr(G, "inventory", lambda residual=False: (inv, ["o127"]))
    monkeypatch.setattr(G, "find_i2d", lambda filt, **k: {})
    monkeypatch.setattr(G, "find_residual_i2d", lambda filt, **k: {})
    pending, waiting = G._pending_summary()
    assert waiting == []
    assert "o127 NIRCam/bgmatch -- no RGB yet" in pending


@pytest.mark.parametrize("residual", [False, True])
def test_an_unsolvable_match_exits_1_and_writes_nothing(
        tmp_path, monkeypatch, capsys, residual):
    """Exit 1 with the "nothing written" line is the retry signal for a SLURM
    solve job: the table on disk unchanged, and the log says to rerun later.
    (An uncaught exception also exits 1, with a traceback in place of that
    line.)"""
    before = {"filters": {f: {"offsets": {"o1": 0.5}} for f in G.FILTERS}}
    paths = _tables(tmp_path, monkeypatch, image=before, residual=before)
    inv = _fields(tmp_path, ("o1",))            # one mosaic: nothing to tie
    monkeypatch.setattr(G, "inventory", lambda residual=False: (inv, ["o1"]))

    def measure(p):
        raise AssertionError("measured with fewer than two mosaics")
    monkeypatch.setattr(G, "miri_measure_pairs", measure)
    texts = {k: p.read_text() for k, p in paths.items()}
    run = G.cmd_nircam_residual_match if residual else G.cmd_nircam_match
    assert run() == 1
    assert {k: p.read_text() for k, p in paths.items()} == texts
    out = capsys.readouterr().out
    cli = "--nircam-match --residual" if residual else "--nircam-match"
    fn = paths["NIRCAM_RESIDUAL_MATCH_JSON" if residual
               else "NIRCAM_MATCH_JSON"]
    assert f"nothing written; {fn.name} is unchanged" in out
    assert f"This exit 1 means rerun {cli} later" in out


def _mosaics(directory, layout, age=10000):
    """Settled FITS mosaics, {filter: [obs, ...]} -> an inventory.

    Real files, since the image solve reads each one back for its pooled cuts
    and build_obs reprojects one filter onto the other.  Every mosaic sits on
    the same small TAN grid.
    """
    from astropy.io import fits
    from astropy.wcs import WCS
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [266.4, -28.9]
    w.wcs.crpix = [4.5, 4.5]
    w.wcs.cdelt = [-1e-5, 1e-5]
    os.makedirs(directory, exist_ok=True)
    inv = {f: {} for f in G.FILTERS}
    old = time.time() - age
    for f, obs in layout.items():
        for o in obs:
            p = os.path.join(directory, f"{o}_{f}_i2d.fits")
            fits.PrimaryHDU(np.full((8, 8), 10.0), w.to_header()).writeto(p)
            os.utime(p, (old, old))
            inv[f][o] = p
    return inv


def _solve_setup(tmp_path, monkeypatch, layout, on_disk=True):
    """Both tables (with `before` in them when on_disk), the mosaics in a
    subdirectory, and inventory() serving them for both kinds.  Returns the
    table paths and their text, plus a listing of tmp_path to compare after."""
    before = {"filters": {f: {"offsets": {"o1": 0.5}} for f in G.FILTERS}}
    doc = before if on_disk else None
    paths = _tables(tmp_path, monkeypatch, image=doc, residual=doc)
    inv = _mosaics(tmp_path / "mosaics", layout)
    obs = sorted(set().union(*inv.values()))
    monkeypatch.setattr(G, "inventory", lambda residual=False: (inv, obs))
    texts = {k: p.read_text() if p.exists() else None
             for k, p in paths.items()}
    return paths, texts, sorted(os.listdir(tmp_path))


def _unchanged(tmp_path, paths, texts, listing):
    assert {k: p.read_text() if p.exists() else None
            for k, p in paths.items()} == texts
    assert sorted(os.listdir(tmp_path)) == listing      # no stray .tmp either


@pytest.mark.parametrize("on_disk", [True, False])
@pytest.mark.parametrize("residual", [False, True])
def test_a_later_filter_failing_writes_neither_table(
        tmp_path, monkeypatch, capsys, residual, on_disk):
    """F480M solves, then F212N has one mosaic: exit 1, nothing written.

    The test above fails in the first filter.  This one gets a whole filter
    solved first, so a table written per filter, or one holding the solved
    F480M alone, shows up as a changed or new file.
    """
    assert G.FILTERS == ["f480m", "f212n"]
    paths, texts, listing = _solve_setup(
        tmp_path, monkeypatch, {"f480m": ["o1", "o2"], "f212n": ["o1"]},
        on_disk)
    measured = []

    def measure(p):
        measured.append(sorted(p))
        return sorted(p), [("o1", "o2", 0.5, 1000)]
    monkeypatch.setattr(G, "miri_measure_pairs", measure)
    run = G.cmd_nircam_residual_match if residual else G.cmd_nircam_match
    assert run() == 1
    assert measured == [["o1", "o2"]]               # F480M only
    _unchanged(tmp_path, paths, texts, listing)
    out = capsys.readouterr().out
    assert "F480M offsets (subtracted from each tile)" in out
    assert "f212n: need at least two settled mosaics to match" in out
    assert "nothing written; " in out


@pytest.mark.parametrize("residual", [False, True])
def test_no_overlapping_pair_exits_1_and_writes_nothing(
        tmp_path, monkeypatch, capsys, residual):
    """Two settled mosaics per filter that share no sky leave nothing to tie
    them together, which is the other exit-1 retry."""
    paths, texts, listing = _solve_setup(
        tmp_path, monkeypatch, {f: ["o1", "o2"] for f in G.FILTERS})
    monkeypatch.setattr(G, "miri_measure_pairs", lambda p: (sorted(p), []))
    run = G.cmd_nircam_residual_match if residual else G.cmd_nircam_match
    assert run() == 1
    _unchanged(tmp_path, paths, texts, listing)
    out = capsys.readouterr().out
    assert "f480m: no overlapping pairs; nothing to tie together" in out
    assert "nothing written; " in out
    inv, _ = G.inventory()
    assert G._nircam_match_filter("f480m", inv["f480m"]) is None


@pytest.mark.parametrize("residual", [False, True])
def test_a_write_that_dies_midway_leaves_the_table_whole(
        tmp_path, monkeypatch, residual):
    """Both tables go through a .tmp and os.replace.  Written straight into
    place, a dump that raised partway would leave a truncated table, which
    needs_build then raises on in every --list and --auto tick."""
    paths, texts, _ = _solve_setup(
        tmp_path, monkeypatch, {f: ["o1", "o2"] for f in G.FILTERS})
    # json cannot encode a complex pair difference, so json.dump raises after
    # it has written the start of the document
    monkeypatch.setattr(G, "miri_measure_pairs", lambda p: (
        sorted(p), [("o1", "o2", complex(0.5, 0), 1000)]))
    monkeypatch.setattr(G, "miri_solve_offsets",
                        lambda keys, pairs: {k: 0.0 for k in keys})
    run = G.cmd_nircam_residual_match if residual else G.cmd_nircam_match
    with pytest.raises(TypeError):
        run()
    assert {k: p.read_text() for k, p in paths.items()} == texts
    assert G.load_nircam_match(
        "nircam_residual" if residual else "nircam") is not None


@pytest.mark.parametrize("residual", [False, True])
def test_build_obs_refuses_a_field_with_an_offset_in_one_filter(
        tmp_path, monkeypatch, residual):
    """PARTIAL covers o128 in F212N only.  build_obs stops with its own
    "no background offset" error before rendering anything, naming the
    missing filter and the solve to rerun."""
    monkeypatch.setattr(G, "OUTDIR", str(tmp_path / "out"))
    _tables(tmp_path, monkeypatch, image=PARTIAL, residual=PARTIAL)
    inv = _mosaics(tmp_path / "mosaics",
                   {f: ["o127", "o128"] for f in G.FILTERS})
    monkeypatch.setattr(G, "inventory", lambda residual=False: (
        inv, ["o127", "o128"]))
    res = " residual" if residual else ""
    cli = "--nircam-match --residual" if residual else "--nircam-match"
    with pytest.raises(RuntimeError) as exc:
        G.build_obs("o128", stretch="bgmatch", residual=residual)
    assert str(exc.value) == \
        f"o128: no{res} background offset for f480m; re-run {cli}"
    assert not os.path.exists(G.png_for("o128", "bgmatch", residual))
