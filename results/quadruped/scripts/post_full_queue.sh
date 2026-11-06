#!/bin/bash
# Launch under nohup; does not SSH or alter any pre-existing clone.
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$HERE/post_full_queue.py" "$@"
