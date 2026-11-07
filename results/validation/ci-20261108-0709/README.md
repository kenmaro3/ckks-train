# CI observation: 2026-11-08 07:09 JST

Read-only GitHub API capture begun at 2026-11-07 22:09:35 UTC. These are operational records, not benchmark JSON or research results.

| Source commit | GitHub run | Observed result |
| --- | --- | --- |
| a29dcf6613ed05d5e1ec7b5300458e74214b5ee9 | [42416425934](https://github.com/kenmaro3/ckks-train/actions/runs/42416425934) | Six jobs completed/failure, zero steps each |
| 767fb525c41fd39b1e305419eec884dfbd36d0ec | [42418803177](https://github.com/kenmaro3/ckks-train/actions/runs/42418803177) | Six jobs completed/failure, zero steps each |

All twelve check-run annotations say the job did not start because recent account payments failed or the spending limit needs increasing. The API does not distinguish which of these billing conditions applies. Thus these runs provide no code-test outcome. Jobs are fmt, clippy, test, deny, cuda-check and research-tools. The separate Ubuntu image migration notice is not the recorded failure reason.

`*-command.json` records the exact read-only API argv; `*.stdout.json` and `*.stderr` are unmodified API output; `*-exit.json` records command returncode (all zero). `summary.json` is a derived operational index. Jobs include the exact head SHA, check-run URL and job ID; annotations are named by that job ID. Root checked six jobs per head, empty step arrays, and the failure annotation for each job. `SHA256SUMS` covers all retained files other than itself.

No billing setting, payment, workflow, or runner was changed; no rerun was requested. Earlier successful CI applies only to its historical SHA. Local tests, current source review, native Linux execution, research gates and final release CI remain separate requirements. This capture does not establish the status of any subsequent commit.
