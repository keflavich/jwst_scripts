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

PY=/blue/adamginsburg/adamginsburg/miniconda3/envs/python312/bin/python
SCRIPT=/orange/adamginsburg/jwst/jwst_scripts/scripts/gc_treasury_rgb_images.py
OVERLAYS=/orange/adamginsburg/jwst/jwst_scripts/scripts/gc_treasury_overlays.py
LOGDIR=/blue/adamginsburg/adamginsburg/logs
STAMP=$(date +%Y%m%dT%H%M%S)

# Catalogue-derived overlays (red-star and red-clump density HiPS, the
# ultra-red source catalogue).  Submitted first and unconditionally: these
# track the vetted daophot catalogues, not the i2d mosaics, so the imaging
# gate below says nothing about whether they are stale.  --auto fingerprints
# the input catalogues and exits without a rebuild when nothing changed, and
# it takes its own lock, so an extra tick is cheap.
sbatch --job-name=gctreasury_overlays \
  --account=astronomy-dept --qos=astronomy-dept-b \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64gb --time=4:00:00 \
  --output="$LOGDIR/gctreasury_overlays_%j.log" \
  --wrap "$PY $OVERLAYS --auto --publish --threads 8"

# Mirror the published overlays to the starformation viewer host.  This runs
# HERE, on the login node, rather than inside the sbatch: the compute nodes are
# not guaranteed working non-interactive ssh, and cron is.  It therefore pushes
# whatever the PREVIOUS tick built, which is why it is an unconditional
# incremental rsync rather than something gated on this tick's job -- it is
# cheap when nothing changed and self-healing when a push was missed.
"$PY" "$OVERLAYS" --push-only \
  || echo "[$STAMP] overlay push to starformation failed"

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
