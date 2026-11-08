# R48/R52 controller preparation

IMPLEMENTED LOCAL PREPARATION; independent review and request registration pending. No actual SSH capture is authorized or performed by this report. Pre-edit reason/alternative/gates in task-48-52-terminal-controller-plan.md. Derived unchanged transport/custody logic from reviewed R47 controller SHA4de62d14af065aee39ae9c74e9e82b6bd71e39f921a0936901961beb7cc8a6d6; finite diffs retained beside this report.

Each controller is one fixed family, with a separately fresh task-48/task-52-terminal-fetch-v1 prefix. Both pin current candidate collector19552B354b7787..., whose independent review is pending. It takes literal r48/r52 in the remote command, never user-provided commands. They require authenticated archived observer output proving FULL2 final-path presence and original chain terminal/gone, gate-released, and either the family's last-job state or released-queue terminal. A queue failure before that family is meaningful capture input; the collector still reports missing terminal records, never not-run or science acceptance.

Preserved key-only/nice10/110s/noautomaticretry/localSSHchild-only-timeout, six exclusive files, original observer bytes, source pin and per-transport raw bytes/hashes. The64MiB stdout envelope check is POST-CAPTURE, not streaming or stderr hard cap; remote reader separately limits raw originals to32MiB. No research process termination. Positive SSH, timeout, launch-error and excess-envelope branches are unexecuted in this preparation.

Test first:20 subcases originally failed solely because controller files did not exist (source-absence RED, not independently exercised behavior). GREEN: one unittest with20 synthetic pure precondition subcases passed, checking each family, completed/failed queue, wrong-family/pending/no-gate/no-result/live-or-unknown chain, bad transport/hash, literal command/prefix and no output creation. Popen patched to reject any launch. Full raw RED/GREEN command transcripts are in their JSON files. Then each actual --check-only invocation using post-full-observation-20261108-1523 exited2 with FULL2-result-not-observed before any SSH or output reservation; both fetch prefixes remain absent. These negative actual checks do not prove a successful production invocation.

## Exact candidate identities

[
  {
    "file": "task-48-terminal-controller.py",
    "bytes": 7611,
    "sha256": "a2ada1a88e64d77372e728e30700e5798c7b02038fe79d81b459ca10efbbd01e"
  },
  {
    "file": "task-52-terminal-controller.py",
    "bytes": 7611,
    "sha256": "24b142678ad57812625093994bfa5159ef57214d0d3b9846d2631104798388f1"
  },
  {
    "file": "task-48-52-terminal-controller-tests.py",
    "bytes": 3469,
    "sha256": "cb5dfcd7bc185c526fc531786ded5348629cd26723cd0fa191925a0f2d6f2124"
  },
  {
    "file": "task-48-52-terminal-controller-plan.md",
    "bytes": 1598,
    "sha256": "be23308fbdc1a1fa7f5b093c8fd9d9b7f2250a9c070ee1fe9fec51d0f57629d0"
  },
  {
    "file": "task-48-52-terminal-controller-red.json",
    "bytes": 24001,
    "sha256": "8e72429b76075087c937d7db687cf17ea5cbc6b88b247807b5b2869ec1a4bbc7"
  },
  {
    "file": "task-48-52-terminal-controller-green.json",
    "bytes": 1295,
    "sha256": "e0759920e330fe67c1dd2ce662dfe4d24df83898c023459d81f9f33acb97bace"
  }
]
