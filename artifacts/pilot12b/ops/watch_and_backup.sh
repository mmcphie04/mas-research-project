#!/usr/bin/env bash
# Long-lived watcher (runs in the background; does not expire). Polls the pilot every ~2.5 min,
# appends milestones to watch.log, backs up after each newly completed chunk, and when the pilot
# process exits: final backup (3 attempts), then stops the Colab session. If the final backup
# cannot be completed, the session is LEFT RUNNING so nothing is lost, and that is logged.
export PATH="$HOME/.local/bin:$HOME/google-cloud-sdk/bin:$PATH"
S=<scratch>
LOG=$S/colab/watch.log
seen=$S/colab/watch_seen.txt; : > $seen
synced=""
fails=0
log() { echo "[$(date +%H:%M:%S)] $*" >> $LOG; }
log "watcher started"
while true; do
  out=$(timeout 150 colab --auth=adc exec -s mas -f $S/colab/poll.py 2>&1)
  if ! echo "$out" | grep -q "^\[pilot\] process"; then
    fails=$((fails+1)); log "poll failed ($fails): $(echo "$out" | grep -E 'Error|not found' | tail -1 | cut -c1-150)"
    # a traceback can contain "not found"; only `colab status` saying so means the session is gone
    if timeout 90 colab --auth=adc status -s mas 2>&1 | grep -q "Session 'mas' not found"; then
      log "SESSION GONE (confirmed by colab status)"; exit 2
    fi
    [ $fails -ge 20 ] && { log "giving up after 20 failed polls"; exit 3; }
    sleep 60; continue
  fi
  fails=0
  echo "$out" > $S/colab/last_poll.txt
  echo "$out" | grep -E "model loaded|stress|c[0-9]{3}:|FAILED|stopping|out of memory|Traceback|Error|time cap|finished|incomplete|VIEW" | while read -r line; do
    grep -qxF "$line" $seen || { log "$(echo "$line" | cut -c1-260)"; echo "$line" >> $seen; }
  done
  done_tags=$(echo "$out" | grep -oE "  c[0-9]{3}: " | sort -u | tr -d ' :' | tr '\n' ' ')
  if [ "$done_tags" != "$synced" ]; then
    if r=$($S/colab/sync.sh pilot12b 2>&1); then synced="$done_tags"; log "BACKUP ok: $(echo "$r" | tail -2 | tr '\n' ' ')"
    else log "BACKUP failed: $(echo "$r" | tail -2 | tr '\n' ' ')"; fi
  fi
  if echo "$out" | grep -q "process EXITED"; then
    log "PILOT PROCESS EXITED"
    for a in 1 2 3; do
      if r=$($S/colab/sync.sh pilot12b 2>&1); then
        log "FINAL BACKUP ok: $(echo "$r" | tail -2 | tr '\n' ' ')"
        timeout 120 colab --auth=adc download -s mas /content/mas_runs/pilot.log $S/colab/pilot_resume.log > /dev/null 2>&1
        timeout 120 colab --auth=adc stop -s mas >> $LOG 2>&1 && log "SESSION STOPPED"
        exit 0
      fi
      log "final backup attempt $a failed: $(echo "$r" | tail -2 | tr '\n' ' ')"; sleep 60
    done
    log "FINAL BACKUP FAILED 3 times: session left running so nothing is lost"
    exit 4
  fi
  sleep 150
done
