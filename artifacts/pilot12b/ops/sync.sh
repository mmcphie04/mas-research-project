#!/usr/bin/env bash
# Laptop side: pack on the runtime, download, unpack into the local archive root.
set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/google-cloud-sdk/bin:$PATH"
S=<scratch>
DEST=/home/mmcph/projects/mas_research_project/runs/colab
mkdir -p "$DEST"
echo -n "$1" > $S/colab/sync_target
timeout 120 colab --auth=adc upload -s mas $S/colab/sync_target /content/sync_target > /dev/null
timeout 600 colab --auth=adc exec -s mas -f $S/colab/pack.py 2>&1 | tail -1
t0=$(date +%s)
timeout 3000 colab --auth=adc download -s mas /content/sync/sync.tar $S/colab/sync.tar > /dev/null
tar -xf $S/colab/sync.tar -C "$DEST"
echo "downloaded $(du -h $S/colab/sync.tar | cut -f1) in $(( $(date +%s) - t0 ))s -> $DEST"
