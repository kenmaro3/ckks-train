#!/bin/bash
# Sync the hekit working tree (tracked + untracked-not-ignored files, plus .git) to quadruped-gpu:~/hekit/.
# Server-only dirs (target/, data/mnist/, results/logs/) are protected from deletion.
set -euo pipefail
SRC=/Users/marconi/Documents/research/hekit
cd "$SRC"
LIST=$(mktemp)
{ git ls-files -co --exclude-standard; } > "$LIST"
rsync -a --delete --delete-excluded \
  --filter='P /target/' --filter='P /results/quadruped/' --filter='P /data/mnist/' --filter='P /results/logs/' --filter='P /.synced-commit' \
  --filter='P /spikes/*/target/' --filter='P /crates/*/target/' \
  --include='/.git/***' \
  --include-from=<(sed 's|^|/|' "$LIST"; sed 's|^|/|; s|/[^/]*$|/|' "$LIST" | awk -F/ '{p=""; for(i=2;i<NF;i++){p=p"/"$i; print p"/"}}' | sort -u) \
  --exclude='*' \
  "$SRC/" quadruped-gpu:~/hekit/
rm -f "$LIST"
C=$(git rev-parse --short HEAD); F=$(git rev-parse HEAD); D=$(git status --porcelain | wc -l | tr -d ' ')
DIFFHASH=$(git diff HEAD | shasum -a 256 | cut -c1-16)
ssh -o BatchMode=yes quadruped-gpu "printf 'commit %s\nfull %s\ndirty_files %s\nworking_tree_diff_sha256_16 %s\nsynced_at %s\n' $C $F $D $DIFFHASH \"\$(date -Is)\" > ~/hekit/.synced-commit; cat ~/hekit/.synced-commit; du -sh ~/hekit"
