# Mac baseline virtual-environment cleanup — 2026-11-08

User-authorized local disk cleanup. Removed only the unused, gitignored `spikes/reboot/_ext/venv` after recording67 installed package names/versions and the Python3.11.9 environment configuration. The prior Mac baseline ran out of memory; authoritative ReBoot comparisons use quadruped-gpu. This cleanup changes no benchmark/scientific result and no server state.

The removed directory occupied1,275,260,928 allocated bytes before deletion (about1.19GiB). Own-user Python command inspection found no matching baseline/venv process; own-user lsof found no open file under it. No process was terminated. Root directory shape and absence of Git-tracked files were checked. An initial guard rejected the normal .DS_Store entry before any write/deletion; it was inspected and explicitly allowed before the successful check. The one deletion controller session70150 exited0; the exact path is now absent.

27 protected log/data/paper/run-document files were byte-verified before and immediately after deletion. External baseline source clones, build trees, installed OpenFHE libraries, original logs, patches and datasets remain in place; no baseline code or patch is copied into this archive. RUN.md was subsequently updated to record cleanup, so its pre-cleanup hash is historical. df still reported32GiB available afterward amid other work; removed-directory allocation is not a claim of an exact increase in filesystem free space.

To restore local exploratory use, recreate the Python3.11.9 venv at the recorded path, reinstall the recorded package versions, rebuild/reinstall the OpenFHE-Python module from the retained pinned source/build using the environment flags in spikes/reboot/RUN.md, and restore the documented libomp symlinks. Package versions are a restoration inventory, not a lock with artifact hashes. Reproduction of the old environment is not tested, and local ReBoot run.sh now requires that restoration. Full-size baseline measurements remain on the server.

Six exact cleanup records are preserved here; SHA256SUMS covers those records and this README. No scientific JSON was generated or edited.
