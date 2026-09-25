#!/bin/bash
# Hourly tick for the 10678 (GC Treasury) RGB/HiPS pipeline.
#
# Submits the work to SLURM rather than running it on the login node: a single
# observation is a ~5900x4600 reproject plus a full HiPS pyramid, and the login
# node is cgroup-limited to one core, where that takes the better part of an
# hour.  The script itself is cheap, so cron only pays for an sbatch.
#
# --auto is idempotent and takes its own lock, so a tick that fires while the
# previous build is still running exits immediately rather than piling up.
# Nothing is rebuilt unless its source i2d is newer than the RGB, which is what
# picks up the re-reduction that 10678 is expected to get once its astrometric
# offsets table exists.
#
# Two jobs are submitted per tick: the imaging pipeline (--auto) and the
# catalogue-derived overlays (gc_treasury_overlays.py --auto), which key off
# different inputs and so go stale independently.
#
# Installed as:
#   0 * * * * /orange/adamginsburg/jwst/jwst_scripts/scripts/gc_treasury_cron.sh
set -euo pipefail

# cron's PATH is minimal and this is not a login shell, so SLURM's bin has to
# be named.  Without it every tick since this was installed reported
# "sbatch: command not found" and submitted nothing, while the overlay rsync
# above kept succeeding and made the log look healthy.
export PATH=/opt/slurm/bin:$PATH
if ! command -v sbatch >/dev/null; then
    echo "[$(date +%Y%m%dT%H%M%S)] sbatch not on PATH ($PATH); nothing submitted" >&2
    exit 127
fi

PY=/blue/adamginsburg/adamginsburg/miniconda3/envs/python312/bin/python
SCRIPT=/orange/adamginsburg/jwst/jwst_scripts/scripts/gc_treasury_rgb_images.py
OVERLAYS=/orange/adamginsburg/jwst/jwst_scripts/scripts/gc_treasury_overlays.py
LOGDIR=/blue/adamginsburg/adamginsburg/logs
STAMP=$(date +%Y%m%dT%H%M%S)

# The HiPS publisher lives in jwst-gc-pipeline and runs under that repo's
# environment, which is not the one the builder uses.
PY313=/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python
PUBLISH=/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-schedule/scripts/monitoring/publish_hips_layers.py
PUBLISH_LOCK=$LOGDIR/.hips_publish.lock
PUBLISH_LOG=$LOGDIR/hips_publish_cron.log

# One PENDING copy of each job is enough.  Both jobs are idempotent and lock
# themselves, so a second queued copy can only start, find the lock held or
# nothing new, and exit.  When the queue stalls, though, a submission every
# hour piles them up: 22 gctreasury_auto and 20 gctreasury_overlays were
# pending at once on 2026-09-24, while the burst QOS ran nothing for ~12 h.
# A RUNNING copy does not count -- the next one still has to be queued behind
# it to pick up whatever that run did not see.
#
# If squeue itself fails, submit anyway: a spare queued job is harmless, and a
# tick that silently submits nothing is the failure the --list check below
# exists to prevent.
already_pending() {
    local name=$1 ids
    if ! ids=$(squeue -h -u "$(id -un)" -t PENDING -n "$name" -o %i 2>&1); then
        echo "[$STAMP] squeue failed ($ids); submitting $name anyway" >&2
        return 1
    fi
    if [ -n "$ids" ]; then
        echo "[$STAMP] $name already pending ($(echo $ids | tr ' ' ',')); not submitting another"
        return 0
    fi
    return 1
}

# Mirror the published overlays to the starformation viewer host.  This runs
# HERE, on the login node, rather than inside the sbatch: the compute nodes are
# not guaranteed working non-interactive ssh, and cron is.  It pushes whatever
# the PREVIOUS tick built -- an unconditional incremental rsync, cheap when
# nothing changed and self-healing when a push was missed.
#
# Deliberately BEFORE the sbatch below.  Both take the same lock (the push
# must not walk a tree publish() is swapping), so whichever starts first makes
# the other skip.  Running the push here, synchronously and in seconds, means
# it has released the lock long before the queued build ever starts -- with
# the order reversed a slow first mirror could make the build skip a tick.
"$PY" "$OVERLAYS" --push-only \
  || echo "[$STAMP] overlay push to starformation failed"

# Push the coadds the PREVIOUS tick built to the docroot and to
# starformation, BEFORE the build below is submitted.
#
# The publisher shares .auto.lock with the build, and --auto holds that lock
# for its whole run -- 4 h 45 m on 2026-09-21.  Run on its own hourly schedule
# the publisher simply loses: 80 "skipping publishing, the build lock is held"
# against 5 publishes in one log, with the treasury MIRI coadd sitting two days
# stale on starformation while a fresh one waited in the build tree.  The gap
# between one build releasing the lock and the next claiming it was 14 minutes,
# and no publish fire landed in it.
#
# Doing it here makes the order explicit rather than a race: the build queued
# below cannot start until this has finished and released the lock.  The cost
# is that a tick whose publish is still running when the job starts loses that
# build, because --auto exits rather than waits when the lock is held.  That is
# the trade this is here to make -- an already-good mosaic reaching the viewers
# is worth more than starting the next one a tick earlier.
#
# flock -n against the same file the :50 cron uses, so the two cannot overlap.
/usr/bin/flock -n "$PUBLISH_LOCK" \
  env HIPS_PUBLISH_LOCK_WAIT_S=300 "$PY313" "$PUBLISH" >> "$PUBLISH_LOG" 2>&1 \
  || echo "[$STAMP] HiPS publish returned non-zero (or was already running)"

# Catalogue-derived overlays (red-star and red-clump density HiPS, the
# ultra-red source catalogue, the star-colour image and catalogue, the density
# cubes and the colour maps).  Handled first, whenever no copy is already
# pending or running, and without the imaging gate below: these track the
# vetted daophot catalogues, not the i2d mosaics, so that gate says nothing
# about whether they are stale.
#
# The job is sized for a FULL rebuild of all seven products (the star-image
# render and its order-11 HiPS dominate): publish and the stamp run only at
# the end, so a timeout publishes nothing.  A job that size waits longer on
# astronomy-dept-b, so it is submitted only when --check (run here, on the
# login node: a glob and a stat per catalogue) says a rebuild is due, rather
# than every tick as a no-op.  Exit 3 = due; anything but 0/3 is a broken
# check and must not look like "up to date".
# RUNNING counts too: a running build has not written its stamp yet, so
# --check would say "due" and queue a second 128 GB job behind it.
overlays_running() {
    local ids
    ids=$(squeue -h -u "$(id -un)" -t RUNNING -n gctreasury_overlays -o %i 2>/dev/null) || return 1
    if [ -n "$ids" ]; then
        echo "[$STAMP] gctreasury_overlays running ($(echo $ids | tr ' ' ',')); not submitting another"
        return 0
    fi
    return 1
}
# A build that fails every time writes no stamp, so --check keeps saying
# "due" and each tick would queue another 128 GB job behind the failure.
# Two failed runs (FAILED/OOM/TIMEOUT/NODE_FAIL) in the last 24 h stop
# submission until someone looks; the log line says why.
overlays_failing() {
    local n
    n=$(sacct -n -X -u "$(id -un)" --name=gctreasury_overlays \
          -S "$(date -d '24 hours ago' +%Y-%m-%dT%H:%M)" \
          -s FAILED,OUT_OF_MEMORY,TIMEOUT,NODE_FAIL -o JobID 2>/dev/null \
          | grep -c . || true)
    if [ "${n:-0}" -ge 2 ]; then
        echo "[$STAMP] gctreasury_overlays failed $n times in 24 h; not submitting (see $LOGDIR/gctreasury_overlays_*.log)"
        return 0
    fi
    return 1
}
if ! already_pending gctreasury_overlays && ! overlays_running \
   && ! overlays_failing; then
  OVERLAY_RC=0
  OVERLAY_CHECK=$("$PY" "$OVERLAYS" --check 2>&1) || OVERLAY_RC=$?
  case $OVERLAY_RC in
    0) echo "[$STAMP] overlays up to date; not submitting" ;;
    3) sbatch --job-name=gctreasury_overlays \
         --account=astronomy-dept --qos=astronomy-dept-b \
         --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=128gb --time=24:00:00 \
         --output="$LOGDIR/gctreasury_overlays_%j.log" \
         --wrap "$PY $OVERLAYS --auto --publish --threads 16" ;;
    *) echo "[$STAMP] overlays --check FAILED (exit $OVERLAY_RC); not submitting.  Output:"
       printf '%s\n' "$OVERLAY_CHECK" | tail -20 ;;
  esac
fi


# Checked before --list, which is the slow part of a tick.
if already_pending gctreasury_auto; then
  exit 0
fi

# Skip the sbatch entirely when there is nothing to do, so the queue does not
# collect no-op jobs once the survey is fully built.
#
# "the check said no" and "the check failed" have to stay distinguishable.
# Piping --list straight into grep conflated them: any failure -- an import
# error, a moved path, a reworded message -- made the negated pipeline true, so
# every tick from then on reported "nothing to submit" and exited 0, with
# 2>/dev/null hiding the reason.  A silently disabled pipeline looks exactly
# like a finished one.
if ! LIST_OUT=$("$PY" "$SCRIPT" --list 2>&1); then
  echo "[$STAMP] --list FAILED; not submitting.  Output:"
  printf '%s\n' "$LIST_OUT" | tail -20
  exit 1
fi
if ! printf '%s\n' "$LIST_OUT" | grep -q "complete in all 2 filters: o"; then
  echo "[$STAMP] no observation complete in both filters; nothing to submit"
  exit 0
fi

sbatch --job-name=gctreasury_auto \
  --account=astronomy-dept --qos=astronomy-dept-b \
  --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96gb --time=8:00:00 \
  --output="$LOGDIR/gctreasury_auto_%j.log" \
  --wrap "$PY $SCRIPT --auto"
