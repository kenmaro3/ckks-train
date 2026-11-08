# Mac incremental-cache cleanup, 2026-11-08

Administrative disk cleanup only; not a benchmark or scientific result.
The user authorized removing unused Mac artifacts from this project.

The sole source writer confirmed no active build and held new Cargo/rustc
invocations while source editing continued. A local process-name check also
found no active Cargo/rustc process for this user before deletion.
Only `/Users/marconi/Documents/research/hekit/target/debug/incremental`
was removed, after validating its exact nonsymlink path. It is reproducible
compiler cache, not source, binary output, test logs or measured results.

The cache occupied4,281,636KiB by `du -sk` (about4.08GiB).
Available filesystem space was2,392,825,856bytes before and5,856,251,904bytes
after. These are observed filesystem values; shared storage/background activity
means the difference need not equal the cache's allocated-size estimate.
The deletion completed2026-11-08T03:42:20.716961Z (12:42JST).

Rust dependency/build outputs, Linux cross-compilation cache, executable files,
source, original result files and all retained validation evidence were preserved.
The writer's build hold was then released with `CARGO_INCREMENTAL=0` for subsequent
checks. The [administrative record](cleanup.json) preserves exact times and sizes.
