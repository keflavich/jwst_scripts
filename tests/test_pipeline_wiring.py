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
