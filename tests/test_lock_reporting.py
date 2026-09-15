"""cmd_auto must SAY what is stuck behind a held lock.

_pending_summary is tested on its own, but nothing tested that cmd_auto calls
it when the lock is held -- which is the behaviour that was missing, and the
reason three observations sat unbuilt for four hours while every tick printed
one line and exited.  A silent-reporting fix whose absence is also silent is
exactly the shape that regresses unnoticed.
"""
import os
import time

import pytest

G = pytest.importorskip("gc_treasury_rgb_images")


@pytest.fixture
def held_lock(tmp_path, monkeypatch):
    """A fresh lock file in a redirected OUTDIR, as another run would leave."""
    monkeypatch.setattr(G, "OUTDIR", str(tmp_path))
    lock = tmp_path / ".auto.lock"
    lock.write_text("12345 avm-astrometry-rebuild 2026-09-14T13:35:57\n")
    return lock


def test_reports_the_work_waiting_behind_the_lock(held_lock, monkeypatch, capsys):
    monkeypatch.setattr(G, "_pending_summary",
                        lambda: ["o105 NIRCam -- no RGB yet",
                                 "o106 NIRCam -- no RGB yet",
                                 "o107 NIRCam -- no RGB yet"])
    assert G.cmd_auto() == 0
    out = capsys.readouterr().out
    assert "another run holds the lock" in out
    assert "WAITING ON THE LOCK: 3 item(s)" in out
    for obs in ("o105", "o106", "o107"):
        assert obs in out


def test_names_the_lock_holder(held_lock, monkeypatch, capsys):
    """"263 min old" says how long; the holder line says who, which is what
    decides whether to wait or clear it."""
    monkeypatch.setattr(G, "_pending_summary", lambda: [])
    G.cmd_auto()
    out = capsys.readouterr().out
    assert "avm-astrometry-rebuild" in out
    assert "12345" in out


def test_stays_a_one_liner_when_nothing_is_pending(held_lock, monkeypatch, capsys):
    """A held lock with no work behind it is routine; it must not add noise to
    every tick."""
    monkeypatch.setattr(G, "_pending_summary", lambda: [])
    G.cmd_auto()
    out = capsys.readouterr().out
    assert "nothing pending behind it" in out
    assert "WAITING ON THE LOCK" not in out


def test_does_not_build_while_the_lock_is_held(held_lock, monkeypatch, capsys):
    """Reporting must not become doing."""
    called = []
    monkeypatch.setattr(G, "_pending_summary", lambda: ["o105 NIRCam -- no RGB yet"])
    monkeypatch.setattr(G, "inventory",
                        lambda: called.append("inventory") or ({}, []))
    monkeypatch.setattr(G, "build_obs",
                        lambda *a, **k: called.append("build") or (None, None))
    assert G.cmd_auto() == 0
    assert "build" not in called


def test_an_unreadable_lock_still_reports_pending_work(held_lock, monkeypatch, capsys):
    """The two failure sources are independent: losing the holder line must not
    cost the pending list, which is the half that says what is stuck."""
    monkeypatch.setattr(G, "_pending_summary", lambda: ["o105 NIRCam -- no RGB yet"])
    held_lock.chmod(0o000)
    try:
        G.cmd_auto()
        out = capsys.readouterr().out
    finally:
        held_lock.chmod(0o644)
    assert "WAITING ON THE LOCK: 1 item(s)" in out
    assert "o105" in out


def test_a_failing_summary_still_reports_the_holder(held_lock, monkeypatch, capsys):
    """And the other direction."""
    def boom():
        raise KeyError("inventory blew up")
    monkeypatch.setattr(G, "_pending_summary", boom)
    G.cmd_auto()
    out = capsys.readouterr().out
    assert "avm-astrometry-rebuild" in out
    assert "could not summarise pending work" in out
    assert "KeyError" in out


def test_a_stale_lock_is_taken_rather_than_reported(tmp_path, monkeypatch, capsys):
    """Past the age limit the lock is claimed, so the report does not apply."""
    monkeypatch.setattr(G, "OUTDIR", str(tmp_path))
    lock = tmp_path / ".auto.lock"
    lock.write_text("1 old\n")
    old = time.time() - 7 * 3600
    os.utime(lock, (old, old))
    monkeypatch.setattr(G, "inventory", lambda: ({f: {} for f in G.FILTERS}, []))
    monkeypatch.setattr(G, "find_i2d", lambda filt: {})
    monkeypatch.setattr(G, "cmd_coadd", lambda **k: 0)
    G.cmd_auto()
    out = capsys.readouterr().out
    assert "stale lock" in out
    assert "WAITING ON THE LOCK" not in out


# --- every writer of a coadd has to take the lock ---------------------------

def test_coadd_lock_holds_and_releases(tmp_path, monkeypatch):
    import gc_treasury_rgb_images as G
    monkeypatch.setattr(G, "OUTDIR", str(tmp_path))
    lock = tmp_path / ".auto.lock"
    with G.coadd_lock("--coadd pct"):
        assert lock.exists()
        assert "--coadd pct" in lock.read_text()
    assert not lock.exists()


def test_coadd_lock_releases_when_the_body_raises(tmp_path, monkeypatch):
    """An orphaned lock starves the cron for hours; that has happened here."""
    import gc_treasury_rgb_images as G
    monkeypatch.setattr(G, "OUTDIR", str(tmp_path))
    lock = tmp_path / ".auto.lock"
    with pytest.raises(ValueError):
        with G.coadd_lock("--coadd pct"):
            raise ValueError("boom")
    assert not lock.exists()


def test_coadd_lock_refuses_rather_than_racing(tmp_path, monkeypatch):
    """The failure this prevents: a manual --coadd and the cron's recoadd
    writing one output directory at once produced a coadd with 2007 of its
    11426 tiles and a zero exit status."""
    import gc_treasury_rgb_images as G
    monkeypatch.setattr(G, "OUTDIR", str(tmp_path))
    (tmp_path / ".auto.lock").write_text("999 held by someone else\n")
    with pytest.raises(RuntimeError, match="held"):
        with G.coadd_lock("--coadd pct", wait=False):
            pass


def test_a_stale_lock_is_taken(tmp_path, monkeypatch):
    import os
    import time
    import gc_treasury_rgb_images as G
    monkeypatch.setattr(G, "OUTDIR", str(tmp_path))
    lock = tmp_path / ".auto.lock"
    lock.write_text("1 ancient\n")
    old = time.time() - 7 * 3600
    os.utime(lock, (old, old))
    with G.coadd_lock("--coadd pct", wait=False):
        assert "--coadd pct" in lock.read_text()
