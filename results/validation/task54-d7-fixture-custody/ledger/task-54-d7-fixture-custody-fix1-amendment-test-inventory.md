# FIX1 amendment finite inventory before edits

Exactly8 tracked symlinks (LICENSE/THIRD_PARTY_NOTICES in ckks/core/cuda/nn) will become regular0644 copies of current root bytes, without content-hash or selected-path changes. JSON inventory records every old lstat/Git mode/target and planned root length/SHA256. All originals/copies currently compare equal.

Reason: satisfy unchanged no-symlink source policy. Alternatives: policy relaxation changes trust; exclusion weakens coverage; postponement leaves a known build prerequisite. Cost: eight copies need synchronization if root notices change. No speculative script change.

Attempt01 stays immutable: driver exit1 on added regular-file precondition, ZERO pure-preflight calls, absent result. After changes verify lstat regular0644 and root byte equality for8, all410 content pins and43 original/snapshot files unchanged. Then freeze distinct attempt02 driver/controller and execute exactly one bounded pure preflight call. No Cargo/tests/Clippy/fmt/native/helper/network/cache actions. Failed Mac physical equality remains separate.
