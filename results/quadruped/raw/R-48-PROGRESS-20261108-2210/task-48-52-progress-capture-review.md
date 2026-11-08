# 2210 progress capture and 2245 preparation review

2026-11-08. **SPEC PASS / QUALITY PASS** for the retained2210 operational capture/archive and exact2245 preparation delta. No blocking finding. No SSH, Git, network, build/native execution or observer invocation by this reviewer. The previous source review is unchanged; only this report was written.

The11-file archive results/quadruped/raw/R-48-PROGRESS-20261108-2210 contains nine selected originals plus README/SHA256SUMS. All nine records, including the prior review, match ledger originals byte-for-byte. All ten unique manifest entries match and exactly cover the archive except SHA256SUMS. The archived source/command/controller are the previously reviewed v2 pins, not the rejected v1.

The retained launch and final metadata describe the same invocation. Metadata reports returncode0, empty stderr, held1488-byte source, automatic_retry=false and the exact v2 command/source hashes. Its2365-byte stdout hash is2bc7aa204b97934b33ee05fc10b4fa885e50be1b6d043ad7f72b5aae67e2965c. Observed_unix converts to2026-11-08T22:12:23.066980+09:00; it lies between controller start/finish and after the22:10 earliest time. Root outcome and requests agree. This verifies the retained single invocation record; it does not establish a system-wide invocation census.

The three returned jobs are exactly R-48-4/R-52-1/R-52-2. R-48-4 records own UID1000,2554 before/after bytes, offset0,2554 bytes read and12 retained lines, within the reviewed8192-byte bound. Its final line is the hoist=0,bsgs=1,lazy_relin=0 iter1 ablation output. R-52-1/2 explicitly report exists=false. Equal before/after size/mtime does not make the log atomic or prove there was no intervening change; concurrent_append_possible=true and scientific_acceptance=false remain explicit.

README and requests accurately describe this as nonterminal iteration progress. No timing statistic is adopted, no paper-table or collector qualification follows, and absence of R52 log paths alone does not establish an attempted/successful/failed R52 job. The observation does not inspect queue terminal state and cannot replace the required terminal observation. Prior source review covered preparation and the2205 archive; it is not retroactively presented as a review of these newly captured measurements. The no-job-change interpretation agrees with the read-only v2 source, which was not modified for this capture.

For2245, the controller is exactly the2205 controller with the occurrence string20261108-2205 replaced by20261108-2245. Its1547B hash is e1886babe7c3ba558d376efbfddf00f5f9b08dd03259485d6606a95ca03e06f0. The813B command hash is c954cbe352622a446d3a273c4876745ac4f5614a423e3f8a1116e31935a6af0e. Earliest time is2026-11-08T22:45:00+09:00. SSH argv and fixed1560-byte remote observer SHA17550e5d285a03329782d8ca9f3af7ad5b4e39468596d41781607a942572c243 are unchanged. AST parsing passed. All four new capture outputs were absent at review. requests registers prepared/not invoked and does not infer collector eligibility from2210. This approves the bounded preparation delta, not a current remote observation or guaranteed completion time; the controller still has no automatic retry or overall wall-clock deadline.

Read-only Python checks (pathlib/json/hashlib/datetime/ast) exited0:

```text
MANIFEST 10/10; ORIGINAL_COPIES 9/9 PASS
CAPTURE_BINDINGS_TIME_BOUNDS PASS 2026-11-08T22:12:23.066980+09:00
2245_PREFIX_ONLY / AST / SOURCE_ARGV_TIME / OUTPUTS_ABSENT PASS
```

Additional exact reviewed pins:

- Archive README.md:1175B;153434c74b6a80617d57f61ea0516c36eaf766410f914888f646ce42b66fb580.
- Archive SHA256SUMS:1174B;cd4a292b07779e1aab6ff0e05e0f5c24a8a8a3965b4952641c7bc9e394415ce9; pins all nine selected originals and README.
- task-48-52-progress-observation-20261108-2210.meta.json:1202B;0e2fd200f5dbfdf3aa2ae768f4dcc7759656348bf1120bc754045f4fbfbd058a.
- Unchanged task-48-52-progress-observer-review.md:8223B;f1e4b057ed0f6ac4fc040c99b23d5a41dba6c6722c2c9c84c4c08acf2545cd00.

The separately retained465b32f CI observation is outside this review; its results are not used to qualify R48/R52 progress, source execution, or scientific outcomes.

## Additional documentation check

Read the final latest-evidence-20261108-r48-progress section in docs/handover.md and docs/m5-report.md. The sections are byte-identical, each anchor occurs once, and both relative raw-archive README links resolve. The22:06 queue-file absence is distinguished from22:12 log presence; the3 fixed-log observation,8192-byte/12-line cap, R52 log absence, nonterminal/no-scientific-adoption boundary and next22:45 registration agree with the reviewed raw records and requests. Existing M3/R47 adoption is not changed by these operational observations. SPEC / QUALITY PASS for this append. No Git comparison was performed, so earlier body-byte immutability is not independently asserted.

Exact appended section: 1256B; SHA256 166c6c268ab94b3acf12a9ce0d5c36fd0d6cd526ecd84bef66061ec72506418d.
