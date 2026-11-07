# Resource design: primary OS reference checked by root

2026-11-08 JST. Design reference only; no server observation or runtime validation.

The Linux 6.7 [cgroup v2 documentation](https://www.kernel.org/doc/html/v6.7/admin-guide/cgroup-v2.html), sections Memory Interface Files, The Root and Views, and Deprecated v1 Core Features, specifies memory.current for non-root groups. A namespace-visible root can be a subtree of the real hierarchy. For v2, root cgroup.controllers supplies controller information; /proc/cgroups is not the authoritative interface.

These facts support distinguishing a proved global structural root from a missing or hidden accounting interface. They do not establish this server's actual hierarchy, justify fabricated zero usage, or require a Unix-root-owned execution lock. Native applicability and trusted origin still need actual evidence. The new administrator installation prerequisite in proposal80d6bdb9 is under independent review against the user's explicit no-sudo constraint; nothing is ratified by this note.
