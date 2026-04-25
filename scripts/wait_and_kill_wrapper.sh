#!/bin/bash
# Watchdog: once base eval (PID arg) exits, kill the wrapper (PID arg) so it
# does not redundantly redo the checkpoint eval on GPU 1.
BASE_PID=$1
WRAPPER_PID=$2
LOG=/data/lsg/work/OPSD/logs/watchdog.log

echo "[$(date +%T)] watchdog: waiting for base eval PID=$BASE_PID to finish, then killing wrapper PID=$WRAPPER_PID" >> "$LOG"

while kill -0 "$BASE_PID" 2>/dev/null; do
    sleep 30
done

echo "[$(date +%T)] watchdog: base eval PID=$BASE_PID exited; killing wrapper PID=$WRAPPER_PID" >> "$LOG"
# Kill the wrapper and any other child procs
kill "$WRAPPER_PID" 2>/dev/null
sleep 2
kill -9 "$WRAPPER_PID" 2>/dev/null
# In case wrapper already moved to next eval and spawned a new python
pkill -f "evaluate_math.py.*--checkpoint_dir.*lsg" 2>/dev/null
echo "[$(date +%T)] watchdog: done" >> "$LOG"
