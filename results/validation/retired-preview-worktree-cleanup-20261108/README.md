# Retired preview worktree cleanup, 2026-11-08

The user authorized removal of unused project files to free Mac storage. The
retired preview worktree at tools/release/_ext/research-preview-candidate was
removed with ordinary git worktree remove, without force. Before removal its HEAD
was6fba46c38f1cdb5be88b9bf1eeecab0f4566b534, tracked status was clean, the complete
untracked/ignored cleanup preview was empty, and an own-user lsof check found no
open files in it. All5,316 files are reproducible Git copies. The source remains
in Git and the immutable public preview; active main and archives/logs are retained.

Observed allocated size was156,964KiB (about153.3MiB). Free space immediately after
removal was5,333,479,424bytes (about4.97GiB). Concurrent host activity can affect
free-space changes. cleanup.json records times, exact path, command and outcome.
This administrative record is not a scientific result or new validation run.
