# 2205 observation and 2210 bounded progress observer review

2026-11-08. **SPEC PASS / QUALITY PASS for the2205 archive and corrected2210 v2 observer.** The initial v1 open-blocking finding below is resolved in v2. v1 remains unchanged and uninvoked; it is not the reviewed invocation candidate. This is an operational source/archive review, not scientific acceptance or terminal-collector qualification. No SSH, network, Git, build/native/scientific execution or observer invocation was performed. Only this report was written.

## 2205 record consistency

The archive results/quadruped/raw/R-41-FULL2-OBS-20261108-2205 contains10 files: eight original records, README and SHA256SUMS. All eight originals match their ledger counterparts byte-for-byte. All nine manifest entries match, are unique and cover every archive file except the manifest itself.

The captured stdout is2401B, SHA256 b637005761cd11d36e875bbac04b4435dfe1d389f9d0074d41da1ebe505057c7. Its timestamp1794143193.1112843 converts to2026-11-08T22:06:33.111284+09:00, matching README/root outcome/requests. The command's earliest22:05 time precedes invocation. Metadata records returncode0, zero stderr, the same stdout size/hash, held source bytes and automatic_retry=false; launch and final metadata refer to the same controller/SSH operation. This checks one retained invocation record, not a system-wide proof that no unrecorded invocation exists.

The old chain1700979 is absent. Queue1739838 has start467041669, UID1000 and matching_own_live_process=true; its direct child2393682 has start488014488 and UID1000. These identities match the2105 snapshot. The child identity is a comparison of observed pid/start/uid, not a newly registered expected_start_ticks value. Its descendant list alone does not establish a particular job stage or numeric progress.

queue_entries is byte-for-byte equivalent as a parsed list to2105: R-48-1/2/3 and optimizer filenames are present; R-48-4, R-52-2 and queue terminal are absent. The retained FULL2 log tail is historical completed evidence, not a new measurement. README and results/quadruped/requests.md:1321–1331 preserve this distinction and correctly leave collectors unqualified/uninvoked. The archive contains no new scientific JSON adoption.

Archive pins:

- README.md:953B;6b8de7f0d9e5d584f0e52fc21c668a38ab07fe491e091293b0579fb3d2035f1a.
- SHA256SUMS:983B;47296f689b1ca67ba8bfc60ea5b29ff6d3c86fef89e69912c8f69c38dd8f8235. This pins the eight exact source/command/transport/outcome originals plus README.
- post-full-observation-20261108-2205.meta.json:1204B;d22840383a6335c39d4019c6e5591d97122d277c19f59e99f7a831cd2d4d6e79.
- post-full-observation-20261108-2205-root-outcome.json:412B;de621c367cd4e6eb8a994ce6e31d5e08dc304b3351663e4a0cdebcbea4eab192.

## Initial finding and v2 correction

The original1474B remote script opened each path at line11 with O_RDONLY|O_NOFOLLOW, then checked fstat for a regular own-UID file. A FIFO could block in open before the rejection check. SSH keepalives do not impose an overall remote-operation deadline. I reported the finite remedy before invocation: add O_NONBLOCK, retaining the same fstat/UID/type check before any content read.

Root preserved v1 and supplied a separate1488B v2. The complete remote diff is only:

```python
fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
```

This makes the FIFO read-open nonblocking so the existing nonregular refusal can run; regular-file semantics and the content bound remain unchanged. No path expansion, data adoption, polling loop or job mutation was added. v2 command changes only source path/hash; controller changes only command filename, asserted source length/hash. The old controller-to2205 comparison likewise contains only prefix/manifest/source pin differences.

## Corrected observer and controller scope

The remote loops exactly once over R-48-4, R-52-1 and R-52-2 under the fixed /home/kmihara/hekit-runs/logs directory. It opens read-only with leaf O_NOFOLLOW and O_NONBLOCK, fstats the opened descriptor, requires regular-file mode and current UID, then preads at most8192 bytes from max(0,before_size-8192). There is at most one content read per accepted file, hence at most24576 source bytes across three logs. Returned text uses replacement decoding and at most12 split lines per log. JSON escaping can make transport output larger than24576 bytes; this is not a transport-byte cap.

The same descriptor supplies before/after size/mtime and inode/device/UID. Files can grow or truncate concurrently: the record explicitly marks concurrent_append_possible=true and scientific_acceptance=false. A sliced first line or replacement character is possible; the result is a bounded tail observation, not a complete/atomic log or validated progress counter. Absence is explicit exists=false; symlink/open/type/ownership failures are explicit refused/error_type/error. No content is read before regular/own-UID validation. Finally closes every opened descriptor. Leaf O_NOFOLLOW does not independently validate every parent-directory component; the path root is the fixed registered own-log location, not a general untrusted-path reader.

The script neither reads scientific result JSON nor signals/restarts jobs, creates remote files, changes collector triggers or recursively scans other logs/users. os.nice(0) only observes current niceness. Its outer kind remains read_only_progress_not_scientific_result; absence/refusal rows cannot be mistaken for terminal success.

The v2 controller checks earliest2026-11-08T22:10:00+09:00, holds the exact1488 source bytes, validates the pinned SHA, and supplies those bytes on stdin to one subprocess. The command remains key-only SSH to quadruped-gpu with ConnectionAttempts=1,90-second connect timeout, keepalives, and nice -n10 /usr/bin/python3.14 -I -B -. No automatic retry loop exists. Exclusive creation of launch/stdout/stderr/meta under the shared2210 capture prefix prevents accidental reuse after a prior launch. It hashes retained outputs and propagates the process returncode. An interrupted/failing capture may leave partial local files; there is no silent retry/success promotion.

This is a finite file/count/byte observer, not a hard overall wall-clock timeout: regular filesystem I/O and an otherwise live remote transport can still wait. No stronger deadline guarantee is claimed. At review, all four capture outputs were absent; requests1334 onward names command-v2 and says prepared/uninvoked, v1 preserved, independent re-review pending. This review does not itself invoke or schedule anything.

## Checks and pins

Read-only Python pathlib/json/hashlib/ast/difflib inspections exited0. One initial rg lookup for docs/requests.md exited2 because that path does not exist; discovery located results/quadruped/requests.md, whose two relevant sections were then read. No verification conclusion relies on that failed lookup.

```text
MANIFEST 9 PASS; exact file coverage PASS
ORIGINAL_COPIES 8/8 PASS
QUEUE_FILES_EQUAL True
ACTUAL_JST 2026-11-08T22:06:33.111284+09:00
v2 remote/controller AST parse: 2/2 PASS
command source hash / earliest time / unchanged SSH argv binding: PASS
OUTPUTS_ABSENT True
v1 source/command/controller bytes and hashes preserved
```

Selected v2 invocation artifacts:

- task-48-52-progress-observer-20261108-2210.remote-v2.py:1488B;4c4f366046f85a0fd6ce065296a59863662320875840adedbd44cbb31ae2bc62.
- task-48-52-progress-observation-20261108-2210-command-v2.json:822B;b0ad86792bd76180c786f8f61fb8f53361eb6a9ea3cec308cb9adfaad0d4474f.
- task-48-52-progress-observation-20261108-2210-controller-v2.py:1570B;5fb8d7099817e9c854f10955ecb0543ad70a6ac57009f194fd642d41b5430748.

Preserved uninvoked v1:

- remote.py:1474B;c5c6c6c2ed62448f5e6a6732bcdccac61f35b5e0ecffe42be7f09b32d644925d.
- command.json:819B;5dfcb1831068670a221775617b7b924ac270e83dc24069cfefabbc72ac48f1b8.
- controller.py:1567B;8b3cda5708a7ed96ecdf21d9b6d8286a45ba510dad7f3510ff85cbe14e8b9982.

The PASS applies to the exact v2 bytes for the proposed single operational observation after root reads this review. It does not certify remote current state before that observation, real R48/R52 results, queue completion or collector eligibility.
