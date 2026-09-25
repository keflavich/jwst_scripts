"""The units are tested; these test that the pipeline still calls them.

Both failures this file guards against were silent: deleting the
patch_hips_dir call from a coadd writer, or narrowing the overlay glob back to
its old pattern, leaves every other test passing while changing what gets
published.  A unit test of landing_page or CAT_RE cannot see either.
"""
import ast
import os

import pytest

import gc_treasury_overlays as O

SCRIPTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")


def calls_in(func_name, module_path):
    """Names called anywhere inside a top-level function."""
    tree = ast.parse(open(module_path).read())
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == func_name)
    return {n.func.id for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}


@pytest.mark.parametrize("func", ["build_obs", "build_miri_obs", "cmd_coadd"])
def test_every_hips_writer_patches_its_landing_page(func):
    """"Called from all four writers" has to stay true to mean anything.

    The full coadd and the staged append both live in cmd_coadd, so that name
    covers two of the four.
    """
    path = os.path.join(SCRIPTS, "gc_treasury_rgb_images.py")
    assert "patch_hips_dir" in calls_in(func, path)


def test_the_coadd_patches_both_of_its_output_paths():
    src = open(os.path.join(SCRIPTS, "gc_treasury_rgb_images.py")).read()
    body = src[src.index("def cmd_coadd"):src.index("def _tile_count")]
    # the staged append and the full rebuild write to different directories
    assert "patch_hips_dir(stage)" in body
    assert "patch_hips_dir(out)" in body


def write_cat(d, name):
    open(os.path.join(d, name), "w").close()


def test_the_selector_sees_a_qualified_catalogue(tmp_path, monkeypatch):
    """o132/o135 carry ..._indivexp_merged_resbgsub_m5_...

    The old glob required "merged_m" to be adjacent, so those files were
    filtered out before CAT_RE ever saw them and the field silently fell back
    to m4 -- 113,774 sources instead of 125,786 for o132 F480M.
    """
    d = str(tmp_path)
    for filt in ("f212n", "f480m"):
        write_cat(d, f"{filt}_merged_o132_indivexp_merged_m4_dao_basic_vetted.fits")
        write_cat(d, f"{filt}_merged_o132_indivexp_merged_resbgsub_m5_"
                     "dao_basic_vetted.fits")
    monkeypatch.setattr(O, "CAT", d)
    pairs = O.latest_pairs()
    assert set(pairs) == {"o132"}
    for filt in ("f212n", "f480m"):
        it, path, qual = pairs["o132"][filt]
        assert (it, qual) == (5, "resbgsub")
        assert "resbgsub_m5" in path


def test_a_plain_catalogue_still_wins_on_a_higher_iteration(tmp_path,
                                                            monkeypatch):
    """A qualifier is a variant of an iteration, not a rank above it."""
    d = str(tmp_path)
    for filt in ("f212n", "f480m"):
        write_cat(d, f"{filt}_merged_o120_indivexp_merged_resbgsub_m5_"
                     "dao_basic_vetted.fits")
        write_cat(d, f"{filt}_merged_o120_indivexp_merged_m6_dao_basic_vetted.fits")
    monkeypatch.setattr(O, "CAT", d)
    pairs = O.latest_pairs()
    assert pairs["o120"]["f480m"][0] == 6
    assert pairs["o120"]["f480m"][2] == ""


def test_a_mixed_set_is_reported_as_mixed(tmp_path, monkeypatch):
    d = str(tmp_path)
    for filt in ("f212n", "f480m"):
        write_cat(d, f"{filt}_merged_o131_indivexp_merged_m4_dao_basic_vetted.fits")
        write_cat(d, f"{filt}_merged_o132_indivexp_merged_resbgsub_m5_"
                     "dao_basic_vetted.fits")
    monkeypatch.setattr(O, "CAT", d)
    pairs = O.latest_pairs()
    lineage = O.lineage_of(pairs)
    assert lineage == {"plain": ["o131"], "resbgsub": ["o132"]}
    assert O.report_lineage(pairs) is False


CRON = os.path.join(SCRIPTS, "gc_treasury_cron.sh")


def test_the_cron_script_can_find_sbatch():
    """cron's PATH is minimal and this script is not a login shell.

    Every tick since installation reported "sbatch: command not found" and
    submitted nothing -- 63 failures, 0 submissions -- while the overlay rsync
    that runs before them kept succeeding, so the log read as healthy.  Fields
    that landed in that window never reached a coadd.
    """
    s = open(CRON).read()
    assert "/opt/slurm/bin" in s, "the cron script does not name SLURM's bin"
    assert "command -v sbatch" in s, "no preflight check for sbatch"


def test_the_cron_script_stops_when_sbatch_is_missing():
    """A missing binary must stop the tick rather than log and continue.

    set -euo pipefail does not cover it: the sbatch calls are followed by ||
    or end a pipeline, so exit 127 becomes a message rather than a stop.
    """
    s = open(CRON).read()
    i = s.index("command -v sbatch")
    assert "exit 127" in s[i:i + 400]
    assert s.index("export PATH=/opt/slurm/bin") < s.index("sbatch --job-name")


def test_the_cron_script_publishes_before_it_submits_a_build():
    """The publisher shares .auto.lock with --auto, which holds it for hours.

    On its own hourly schedule the publisher loses that race: one log held 80
    "the build lock is held, skipping" against 5 publishes, and the treasury
    MIRI coadd sat two days stale on starformation while a fresh one waited in
    the build tree.  The window between one build releasing the lock and the
    next claiming it was 14 minutes, and no publish fire landed in it.

    Running the publish here, before the sbatch, makes the order a fact rather
    than a race.  Moving it after the submit restores the race silently.
    """
    s = open(CRON).read()
    assert "publish_hips_layers.py" in s, "the cron script does not publish"
    assert s.index('"$PY313" "$PUBLISH"') < s.index("sbatch --job-name=gctreasury_auto")


def test_the_cron_publish_cannot_overlap_the_standalone_one():
    """The :50 crontab entry stays as a backstop, so both can fire at once.

    Two publishers running together would walk each other's staged trees.
    They take the same flock, so whichever is second exits immediately.
    """
    s = open(CRON).read()
    i = s.index('/usr/bin/flock -n "$PUBLISH_LOCK"')
    assert '"$PY313" "$PUBLISH"' in s[i:i + 300], \
        "the publish does not run under the flock"
    assert "PUBLISH_LOCK=$LOGDIR/.hips_publish.lock" in s, \
        "not the same lock file the :50 crontab entry takes"


# --- the call sites, not just the helpers ----------------------------------
#
# Each of the three fixes in this branch could be deleted with the suite green,
# because every test exercised the helper directly and none went through the
# code that calls it.  These drive the call sites.

import sys                                                      # noqa: E402

import pytest                                                   # noqa: E402

G = pytest.importorskip("gc_treasury_rgb_images")


def _layer(d, name, properties=True):
    p = d / name
    (p / "Norder3").mkdir(parents=True)
    if properties:
        (p / "properties").write_text("hips_order = 14\n")
    return str(p)


def test_cmd_coadd_checks_inputs_before_removing_the_output(tmp_path,
                                                            monkeypatch):
    """The ordering is the fix, and nothing pinned it.

    Replacing unreadable_layers(layers) with [] leaves every other test green
    while reproducing the o114 failure: rmtree the output, then die reading an
    input.
    """
    out = tmp_path / "jwst_gc_treasury_hips"
    (out / "Norder3").mkdir(parents=True)
    (out / "properties").write_text("hips_order = 14\n")
    (out / "Norder3" / "Npix1.png").write_bytes(b"x")

    layers = [_layer(tmp_path, "GCTreasury_o001_RGB_480-mean-212_hips"),
              _layer(tmp_path, "GCTreasury_o002_RGB_480-mean-212_hips",
                     properties=False)]           # mid-build: tiles, no properties

    monkeypatch.setattr(G, "OUTDIR", str(tmp_path))
    monkeypatch.setattr(G, "find_i2d", lambda *a, **k: {})
    monkeypatch.setattr(G, "inventory", lambda **k: ({}, ["o001", "o002"]))
    monkeypatch.setattr(G.glob, "glob", lambda pat: sorted(layers))

    def explode(*a, **k):                         # coadd_hips must never run
        raise AssertionError("coadd_hips called with an unreadable input")
    monkeypatch.setattr("reproject.hips.coadd_hips", explode)

    rc = G.cmd_coadd(full=True)
    assert rc == 1
    assert (out / "Norder3" / "Npix1.png").exists(), \
        "the existing coadd was destroyed before its inputs were checked"


def test_main_takes_the_lock_around_a_manual_coadd(tmp_path, monkeypatch):
    """The lock is taken in main(), and no test went through main().

    Deleting the `with coadd_lock(...)` left all four lock tests passing,
    because they call the context manager directly.
    """
    monkeypatch.setattr(G, "OUTDIR", str(tmp_path))
    seen = {}

    def fake_coadd(**kw):
        seen["held"] = os.path.exists(os.path.join(str(tmp_path), ".auto.lock"))
        return 0

    monkeypatch.setattr(G, "cmd_coadd", fake_coadd)
    monkeypatch.setattr(sys, "argv", ["gc_treasury_rgb_images.py", "--coadd"])
    assert G.main() == 0
    assert seen.get("held") is True, "cmd_coadd ran without the lock held"
    assert not os.path.exists(os.path.join(str(tmp_path), ".auto.lock")), \
        "the lock outlived the run"


def test_cmd_publish_acts_on_both_miri_coadds(tmp_path, monkeypatch):
    """Asserting on inspect.getsource passes for a comment mentioning the name.

    This drives cmd_publish with the copy stubbed and asks what it acted on,
    which also covers the `src = keep` rebind that filters the flavour globs.
    """
    monkeypatch.setattr(G, "OUTDIR", str(tmp_path))
    monkeypatch.setattr(G, "WEB", str(tmp_path / "web"))
    for name in (G.MIRI_COADD_NAME, G.MIRI_BGMATCH_COADD_NAME,
                 G.coadd_name_for(G.DEFAULT_STRETCH)):
        (tmp_path / name / "Norder3").mkdir(parents=True)
        (tmp_path / name / "properties").write_text("hips_order = 14\n")

    acted = []
    monkeypatch.setattr(G.shutil, "copytree",
                        lambda s, d, **k: acted.append(os.path.basename(s)))
    monkeypatch.setattr(G.shutil, "move", lambda s, d: None)
    monkeypatch.setattr(G.shutil, "rmtree", lambda *a, **k: None)
    G.cmd_publish()
    for name in (G.MIRI_COADD_NAME, G.MIRI_BGMATCH_COADD_NAME):
        assert name in acted, f"cmd_publish did not act on {name}"


def test_a_refused_coadd_makes_the_tick_fail(tmp_path, monkeypatch):
    """cmd_coadd's refusal has to reach the tick's exit status.

    The input guard returns non-zero, and cmd_auto discarded it: the tick
    printed NOT rebuilding, `failed` stayed empty, and cmd_auto returned 0.
    A mid-build input layer arrives on the cron far more often than by hand,
    so the one path that reports the guard firing was the one that dropped it.
    """
    monkeypatch.setattr(G, "OUTDIR", str(tmp_path))
    monkeypatch.setattr(G, "cmd_coadd", lambda **kw: 1)
    monkeypatch.setattr(G, "cmd_publish", lambda *a, **k: 0)
    monkeypatch.setattr(G, "build_obs", lambda o, **k: (None, None))
    monkeypatch.setattr(G, "build_miri_obs", lambda o, **k: (None, None))
    monkeypatch.setattr(G, "needs_build", lambda o, inv, **k: "no RGB yet")
    monkeypatch.setattr(G, "miri_needs_build",
                        lambda o, s, bg=False, residual=False, **k: None)
    monkeypatch.setattr(G, "miri_match_is_stale", lambda m: None)
    monkeypatch.setattr(G, "find_i2d", lambda *a, **k: {})
    monkeypatch.setattr(G, "find_residual_i2d", lambda *a, **k: {})
    monkeypatch.setattr(G, "inventory",
                        lambda **k: ({f: {"o001": "x"} for f in G.FILTERS},
                                     ["o001"]))
    assert G.cmd_auto() == 1


def test_the_lock_claim_is_exclusive(tmp_path):
    """open(lock, "w") reads as a simplification of the O_EXCL claim.

    It is not: the plain open overwrites whatever is there, so two processes
    arriving together both proceed, which is the failure the lock exists to
    prevent.  Without this test the comment is the only thing stopping that
    edit.
    """
    lock = str(tmp_path / ".auto.lock")
    with open(lock, "w") as fh:
        fh.write("someone else\n")
    with pytest.raises(FileExistsError):
        G._claim_lock(lock, "--coadd pct")
    # and the holder's file is untouched
    assert open(lock).read() == "someone else\n"
