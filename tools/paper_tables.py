#!/usr/bin/env python3
"""Paper tables from result JSON (Task 54 Step 2). Python 3 standard library only.

    python3 tools/paper_tables.py                     # write paper/tables/*.tex, results/tables/*.md
    python3 tools/paper_tables.py --sources [TABLE]   # print file and field behind every cell
    python3 tools/paper_tables.py --check             # exit 1 if the written tables are stale
    python3 tools/paper_tables.py --from-run-all full # rows from results/run_all/full/ instead,
                                                      # written to results/run_all/full/tables/

Rules:
- Every number in a cell is read from a JSON field (or, where no JSON exists, from a named
  line of a committed log or doc table; --sources says which). Nothing is typed in here
  except labels, and the model constants are read from the JSON that declares them.
- A row whose input file does not exist yet prints "pending (<request id>)". A present file
  that fails a consistency check stops the generation (it is never silently skipped).
- Several seeds: mean ± sample standard deviation and the seed count n. One seed: the value
  and n = 1 (no ± is shown; that is not zero uncertainty).
- results/run_all/quick/ outputs are pipeline checks and never feed paper/tables/.

The RQ1 four-way ratio rule (docs/rq1-finalization.md, docs/m3-report.md §2) is computed from
the measured counters: K total key switches, H bootstraps, S samples; ReBoot is always at its
own B = 48 with 4 bootstraps per update; R is ReBoot's non-bootstrap KS/sample (196 inputs:
the M0 model 37; 784 subset rows: the in-library Lazy measurement 35.06, the M3 ruling).
  (a) (K/S) / (R + 4*502/48)      own bootstrap costs: hekit 111 measured, ReBoot 502 model
  (b) ((K - 111H + 502H)/S) / (R + 4*502/48)   both 502
  (c) (K/S) / (R + 4*111/48)      both 111
  (d) ((K - 111H)/S) / R          bootstraps excluded on both sides
"""
import argparse
import json
import math
from pathlib import Path
import re
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq1_summary as rq1  # noqa: E402  (FULL-1/2 validation, shared with Task 42)

REPO = Path(__file__).resolve().parents[1]
GATE_KS = 0.70


class TableError(ValueError):
    """An input cannot support the table it feeds."""


def require(ok, message):
    if not ok:
        raise TableError(message)


# ---------------------------------------------------------------------------------------------
# Sources and cells
# ---------------------------------------------------------------------------------------------

class Cell:
    """A rendered value and the (file, field) pairs it was computed from."""

    def __init__(self, text, sources=(), formula=''):
        self.text = text
        self.sources = list(sources)
        self.formula = formula


def plain(text):
    return Cell(text)


class Store:
    """Reads JSON relative to the repository root, remembering what was read."""

    def __init__(self, root):
        self.root = Path(root)
        self.cache = {}

    def exists(self, rel):
        return (self.root / rel).is_file()

    def json(self, rel):
        if rel not in self.cache:
            try:
                raw = (self.root / rel).read_bytes()
                data = json.loads(raw, object_pairs_hook=rq1.unique_object)
                rq1.finite_tree(data)
            except (OSError, ValueError, rq1.SummaryError) as exc:
                raise TableError(f'{rel}: {exc}') from exc
            self.cache[rel] = data
        return self.cache[rel]

    def get(self, rel, field):
        """Value at a dotted field path ('a.b[3].c', '[-1]' allowed) and its source pair."""
        node = self.json(rel)
        for part in re.findall(r'[^.\[\]]+|\[-?\d+\]', field):
            try:
                node = node[int(part[1:-1])] if part.startswith('[') else node[part]
            except (KeyError, IndexError, TypeError) as exc:
                raise TableError(f'{rel}: missing field {field}') from exc
        return node, (rel, field)

    def num(self, rel, field):
        value, src = self.get(rel, field)
        require(type(value) in (int, float) and math.isfinite(value), f'{rel}: {field} is not a finite number')
        return value, src

    def text_line(self, rel, pattern):
        """First line of a text file matching pattern: (match, (rel, 'line N'))."""
        try:
            lines = (self.root / rel).read_text().splitlines()
        except OSError as exc:
            raise TableError(f'{rel}: {exc}') from exc
        for number, line in enumerate(lines, 1):
            m = re.search(pattern, line)
            if m:
                return m, (rel, f'line {number}')
        raise TableError(f'{rel}: no line matches {pattern!r}')


def first_existing(store, paths):
    for p in paths:
        if store.exists(p):
            return p
    return None


def f(x, digits):
    return f'{x:.{digits}f}'


def pct(x, digits=2):
    return f(100 * x, digits)


def mean_std(values, sources, digits, scale=1.0, unit=''):
    """'m ± s (n = k)' over seeds; one value prints 'v (n = 1)'."""
    xs = [v * scale for v in values]
    if len(xs) == 1:
        return Cell(f'{f(xs[0], digits)}{unit} (n = 1)', sources)
    return Cell(f'{f(statistics.mean(xs), digits)} ± {f(statistics.stdev(xs), digits)}{unit} '
                f'(n = {len(xs)})', sources, 'mean ± sample standard deviation over seeds')


class Table:
    def __init__(self, name, title, columns, wide=False):
        self.name, self.title, self.columns, self.wide = name, title, columns, wide
        self.rows = []      # list of ('section', text) or ('row', [Cell])
        self.notes = []

    def section(self, text):
        self.rows.append(('section', text))

    def row(self, cells):
        cells = [c if isinstance(c, Cell) else plain(str(c)) for c in cells]
        require(len(cells) == len(self.columns), f'{self.name}: row width {len(cells)} != {len(self.columns)}')
        self.rows.append(('row', cells))

    def note(self, text, sources=()):
        self.notes.append(Cell(text, sources))


# ---------------------------------------------------------------------------------------------
# Inputs: canonical result files, or run_all outputs
# ---------------------------------------------------------------------------------------------

RUN = {
    # key: (canonical candidates, run_all id, request id for "pending")
    'FULL-1': (['results/quadruped/raw/R-41-FULL-1/mnist196_ni_when_needed_b64_full_seed0.json',
                'results/quadruped/mnist196_ni_when_needed_b64_full_seed0.json'],
               'ni196_wn_b64_full_s0', 'R-41-FULL-1'),
    'FULL-2': (['results/quadruped/raw/R-41-FULL-2/mnist196_ni_b48_full_seed0.json',
                'results/quadruped/mnist196_ni_b48_full_seed0.json'],
               'ni196_wn_b48_full_s0', 'R-41-FULL-2'),
    'F1': (['results/quadruped/raw/R-41-F1/mnist_emlp1_ni_when_needed_b32_faithful.json'], 'ni784_wn_b32_s0', 'R-41-F1'),
    'F2': (['results/quadruped/raw/R-41-F2/mnist_emlp1_ni_when_needed_b64_faithful.json'], 'ni784_wn_b64_s0', 'R-41-F2'),
    'R-41-1': (['results/quadruped/raw/R-41-1/mnist_emlp1_ni_after_update_b32.json'], 'ni784_au_b32_s0', 'R-41-1'),
    'R-41-2': (['results/quadruped/raw/R-41-2/mnist_emlp1_ni_after_update_b64.json'], 'ni784_au_b64_s0', 'R-41-2'),
    'R-41-3': (['results/quadruped/raw/R-41-3/mnist_emlp1_ni_when_needed_b32.json'], None, 'R-41-3'),
    'R-41-4': (['results/quadruped/raw/R-41-4/mnist_emlp1_ni_when_needed_b64.json'], None, 'R-41-4'),
    'R-41-5': (['results/quadruped/raw/R-41-5/mnist_emlp1_ni_after_activation_b32.json'], 'ni784_aa_b32_s0', 'R-41-5'),
    'INT-B32': (['results/quadruped/raw/mnist_interactive_b32/mnist.json'], 'int784_b32_cpu', 'mnist_interactive_b32'),
    'R-52-1': (['results/quadruped/mnist196_emlp2_ni_b48_subset1008_seed0.json'], 'ni196_emlp2_b48_sub1008_s0', 'R-52-1'),
    'R-52-2': (['results/quadruped/mnist196_emlp3_ni_b48_subset1008_seed0.json'], 'ni196_emlp3_b48_sub1008_s0', 'R-52-2'),
    'ABL-CPU-L31': (['results/quadruped/ablation_cpu.json'], 'ablation_cpu_l31', 'R-48-4'),
    'ABL-CUDA-L7': (['results/quadruped/ablation_cuda.json', 'results/quadruped/ablation_cuda_sgd.json'], 'ablation_cuda_l7', 'R-48-2'),
    'ABL-CPU-L7': (['results/quadruped/ablation_cpu_l7.json', 'results/quadruped/ablation_cpu_l7_sgd.json'], 'ablation_cpu_l7', 'R-48-3'),
    'ABL-SMOKE': (['results/ablation_cpu_smoke_small.json'], None, 'Task 48 smoke'),
}
PLAIN196 = [(f'results/mnist196_plain_faithful_seed{k}.json', f'plain196_emlp1_s{k}') for k in range(10)]
TASK51 = {
    'mnist': ('results/plain_mnist_emlp1_paper.json', 'plain_mnist_emlp1_paper'),
    'fmnist': ('results/plain_fmnist_emlp1.json', 'plain_fmnist_emlp1'),
    'bc': ('results/plain_bc_mlp.json', 'plain_bc_mlp'),
    'bc_split': ('results/plain_bc_mlp_split_per_run.json', 'plain_bc_mlp_split_per_run'),
    'bc_z': ('results/plain_bc_mlp_standardize.json', 'plain_bc_mlp_standardize'),
}
REBOOT_SUBSET_STATS = 'results/quadruped/raw/reboot_emlp1_official_subset/stats.json'
REBOOT_SUBSET = 'results/quadruped/reboot_emlp1_official_subset.json'
REBOOT_FULL_CFG = 'results/quadruped/raw/reboot_emlp1_full/config_values.json'
REBOOT_23_CFG = 'results/quadruped/raw/reboot_emlp23_full/config_values.json'
REBOOT_PAPER = 'results/quadruped/raw/reboot_emlp1_paper_setting/summary.json'
LAYOUT = 'results/mnist_reboot_layout.json'
CRITERION = 'results/quadruped/raw/R-47-2/criterion/cuda_ops_n16_train/{}/new/estimates.json'
R47_LOG = 'results/quadruped/cuda_ops_r47.log'
LITERATURE = 'results/reference/literature.json'
SECURITY_DOC = 'docs/security.md'


class Inputs:
    def __init__(self, store, run_all=None):
        self.store, self.run_all = store, run_all

    def path(self, key):
        """The input file of a registered run, or None (pending)."""
        canonical, run_id, _ = RUN[key]
        # Quick IDs are deliberately not aliases for full or archived measurements.
        if self.run_all == 'quick':
            return None
        if self.run_all:
            if run_id is None:
                return None
            p = f'results/run_all/{self.run_all}/{run_id}.json'
            return self.new_result(p)
        return first_existing(self.store, canonical)

    def simple(self, canonical, run_id):
        p = f'results/run_all/{self.run_all}/{run_id}.json' if self.run_all else canonical
        if self.run_all:
            return self.new_result(p)
        return p if self.store.exists(p) else None

    def new_result(self, rel):
        if not self.store.exists(rel):
            return None
        provenance = self.store.json(rel).get('provenance', {})
        require(re.fullmatch(r'the [a-z_]+ binary \(raw output, not assembled\)',
                             str(provenance.get('written_by', ''))) is not None,
                f'{rel}: new output requires binary-written provenance')
        require(re.fullmatch(r'[0-9a-f]{40}', str(provenance.get('git_commit', ''))) is not None,
                f'{rel}: missing binary git commit')
        require(not provenance.get('note'), f'{rel}: annotated provenance is historical only')
        return rel

    @staticmethod
    def pending(key):
        return f'pending ({RUN[key][2]})'


# ---------------------------------------------------------------------------------------------
# Checks on measured NI / interactive JSON
# ---------------------------------------------------------------------------------------------

def ni_counts(s, rel):
    """Validated counters of a non-interactive run: K, H, S, updates, boot KS."""
    k, ks = s.num(rel, 'measured.total.key_switches')
    rot, _ = s.num(rel, 'measured.total.rotations')
    rel_, _ = s.num(rel, 'measured.total.relins')
    sse, _ = s.num(rel, 'measured.total.sse_switches')
    h, hs = s.num(rel, 'measured.total.bootstraps')
    n, ns = s.num(rel, 'training.samples')
    boot, bs = s.num(rel, 'setup.boot_ks')
    require(n > 0 and boot > 0, f'{rel}: samples and bootstrap cost must be positive')
    require(k == rot + rel_ + sse, f'{rel}: key_switches != rotations + relins + sse_switches')
    require(sse == 2 * h, f'{rel}: sse_switches != 2 x bootstraps')
    every, _ = s.get(rel, 'measured.every_update_equals_prediction')
    require(every is True, f'{rel}: measured.every_update_equals_prediction is not true')
    written, _ = s.get(rel, 'provenance.written_by')
    require(written == 'the mnist binary (raw output, not assembled)', f'{rel}: not raw mnist binary output')
    per, _ = s.num(rel, 'measured.ks_per_sample_incl_bootstraps')
    require(math.isclose(per, k / n, rel_tol=1e-9), f'{rel}: ks_per_sample_incl_bootstraps != K / S')
    return {'K': (k, ks), 'H': (h, hs), 'S': (n, ns), 'boot': (boot, bs)}


def validate_full(s, rel, batch):
    try:
        rq1.validate_binary(s.json(rel))
        ref = s.json(REBOOT_FULL_CFG)
        rq1.validate_reference(ref)
        rq1.validate_full(s.json(rel), batch, ref)
    except (KeyError, TypeError, AttributeError, rq1.SummaryError) as exc:
        raise TableError(f'{rel}: {exc}') from exc


def command_flag(s, rel, flag):
    cmd, _ = s.get(rel, 'provenance.command_line')
    return flag in cmd.split()


def config_label(s, rel):
    return 'faithful' if command_flag(s, rel, '--faithful') else 'legacy (R-M3-12)'


# ---------------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------------

def reboot_constants(s):
    """ReBoot's B, bootstraps/update (measured on the official subset run) and 502 (model)."""
    calls, src_calls = s.num(REBOOT_SUBSET_STATS, 'bootstrap.bootstrap_array_calls')
    cts, src_cts = s.num(REBOOT_SUBSET_STATS, 'bootstrap.ciphertexts_bootstrapped')
    batch, src_b = s.num(REBOOT_SUBSET, 'reboot_hyperparameters.batch_size')
    f2 = RUN['F2'][0][0]
    boot502, src502 = s.num(f2, 'rq1.by_reboot_side.m0_model.a_own_bootstrap_costs.reboot_boot_ks')
    m0, src_m0 = s.num(f2, 'rq1.by_reboot_side.m0_model.ks_per_sample_without_bootstraps')
    lazy, src_lazy = s.num(LAYOUT, 'modes.lazy.run.measured.key_switches_per_sample')
    require(min(calls, batch, boot502, m0, lazy) > 0, 'ReBoot comparison denominators must be positive')
    require(cts % calls == 0, 'ReBoot bootstraps per update is not an integer')
    return {'per_update': (cts // calls, [src_cts, src_calls]), 'batch': (batch, [src_b]),
            'boot': (boot502, [src502]), 'm0': (m0, [src_m0]), 'lazy': (lazy, [src_lazy])}


def table_rq1(s, inp):
    t = Table('rq1_ks', 'RQ1: key switches per sample, four ways (ratio hekit / ReBoot at B = 48)',
              ['Run', 'Input', 'Placement', 'B', 'Config', 'Samples', 'hekit KS/sample', 'Boots',
               'ReBoot non-boot', '(a)', '(b)', '(c)', '(d)', 'Ways ≤ 0.70'], wide=True)
    c = reboot_constants(s)
    per_update, rb_batch, b502 = c['per_update'][0], c['batch'][0], c['boot'][0]
    const_src = c['per_update'][1] + c['batch'][1] + c['boot'][1]

    def ratio_row(label, key, input_label, placement, basis_key, basis_label, batch_label=None, full=None):
        rel = inp.path(key)
        if rel is None:
            r, rsrc = c[basis_key]
            t.row([label, input_label, placement, batch_label or '', '', '', inp.pending(key), '',
                   Cell(f'{r:g} ({basis_label})', rsrc)] + ['pending'] * 4 + [''])
            return
        if full:
            validate_full(s, rel, full)
        cn = ni_counts(s, rel)
        (k, ks), (h, hs), (n, ns), (boot, bs) = cn['K'], cn['H'], cn['S'], cn['boot']
        batch, bsrc = s.num(rel, 'setup.batch')
        r, rsrc = c[basis_key]
        den_a = r + per_update * b502 / rb_batch
        den_c = r + per_update * boot / rb_batch
        own = k / n
        ways = [own / den_a, (k - boot * h + b502 * h) / n / den_a, own / den_c, (k - boot * h) / n / r]
        src = [ks, hs, ns, bs] + rsrc + const_src
        forms = ['(K/S)/(R+4*502/48)', '((K-111H+502H)/S)/(R+4*502/48)', '(K/S)/(R+4*111/48)', '((K-111H)/S)/R']
        passing = ''.join(w for w, v in zip('abcd', ways) if v <= GATE_KS) or 'none'
        t.row([label, input_label, placement, Cell(batch_label or str(batch), [bsrc]), Cell(config_label(s, rel), [(rel, 'provenance.command_line')]),
               Cell(str(n), [ns]), Cell(f(own, 3), [ks, ns], 'K/S'), Cell(str(h), [hs]),
               Cell(f'{r:g} ({basis_label})', rsrc)]
              + [Cell(f(v, 3), src, fm) for v, fm in zip(ways, forms)]
              + [Cell(passing, src, 'ways with ratio <= 0.70')])

    t.section('196 inputs, ReBoot full setting, 4800 samples x 10 epochs, seed 0 (headline)')
    ratio_row('FULL-1', 'FULL-1', '196', 'WN', 'm0', 'model', '64', full=64)
    ratio_row('FULL-2', 'FULL-2', '196', 'WN', 'm0', 'model', '48 in 64', full=48)
    t.section('784 inputs, ReBoot encrypted-script subset, 1 epoch (supplementary)')
    for label, key, placement in (('F1', 'F1', 'WN'), ('F2', 'F2', 'WN'), ('R-41-3', 'R-41-3', 'WN'),
                                  ('R-41-4', 'R-41-4', 'WN'), ('R-41-1', 'R-41-1', 'AU'),
                                  ('R-41-2', 'R-41-2', 'AU'), ('R-41-5', 'R-41-5', 'AA')):
        if inp.run_all and RUN[key][1] is None:
            continue
        ratio_row(label, key, '784', placement, 'lazy', 'Lazy, measured')
    t.section('Interactive hekit, client refresh, no bootstraps (separate row, not way (d))')
    rel = inp.path('INT-B32')
    if rel is None:
        t.row(['interactive', '784', '—', '32', '', '', inp.pending('INT-B32'), '0', ''] + ['—'] * 5)
    else:
        v, vs = s.num(rel, 'measured.per_sample.key_switches_per_sample')
        n, ns = s.num(rel, 'training.samples')
        b, bsrc = s.num(rel, 'training.batches')
        require(b > 0 and n > 0, f'{rel}: interactive samples/batches must be positive')
        lazy, lsrc = c['lazy']
        ratio = v / lazy
        t.row(['interactive', '784', '—', Cell(f'{n // b}', [ns, bsrc]), 'legacy (M2)', Cell(str(n), [ns]), Cell(f(v, 3), [vs]), '0',
               Cell(f'{lazy:g} (Lazy, measured)', lsrc)] + ['—'] * 5)
        t.note(f'Independent interactive reference: ratio {f(ratio, 3)} against ReBoot Lazy; '
               'client refresh, not NI-derived way (d) and not a four-way pass result.', [vs] + lsrc)
    t.note('KS = rotations + relinearizations + SSE key switches (OpCounter). hekit bootstrap = '
           '111 KS (measured, setup.boot_ks). ReBoot bootstrap = 502 KS and 196-input non-boot = 37 '
           'KS/sample are M0 cost-model values (OpenFHE does not count key switches); ReBoot '
           'bootstraps 4 ciphertexts per update at B = 48 (measured on the official subset run: '
           '84 bootstraps / 21 steps).', c['per_update'][1] + c['batch'][1] + c['boot'][1] + c['m0'][1])
    t.note('(a) own bootstrap costs; (b) both 502; (c) both 111; (d) bootstraps excluded on both '
           'sides. (b)-(d) are derived from the measured counters, not separate executions. '
           'Criterion: ratio ≤ 0.70 (research plan §4.3.2).')
    t.note('784 rows are the supplementary subset rows (R-M3-13): ReBoot non-boot = Lazy 35.06, '
           "the in-library measurement of ReBoot's layout (Task 33, baseline c). Legacy rows use the "
           'pre-R-M3-12 config; their counts equal the faithful config but their accuracy is not '
           'comparable.', c['lazy'][1])
    return t


def table_wallclock(s, inp):
    t = Table('wallclock_same_cpu', 'Wall-clock on the same CPU (quadruped-gpu, i9-13900KF, 12 threads)',
              ['Run', 'Input', 'B', 'Samples', 'Train s', 's/sample', 'vs ReBoot', 'Boot s/sample',
               'Non-boot s/sample', 'Threads'], wide=True)
    steps, ssrc = s.get(REBOOT_SUBSET_STATS, 'steps')
    require(isinstance(steps, list) and steps, 'ReBoot stats: no steps')
    rb_train = sum(st['reboot_step_time_s'] for st in steps)
    calls, csrc = s.get(REBOOT_SUBSET_STATS, 'bootstrap.seconds_per_call')
    rb_boot = sum(calls)
    rb_n, nsrc = s.num(REBOOT_SUBSET, 'reboot_hyperparameters.trained_samples')
    rb_b, bsrc = s.num(REBOOT_SUBSET, 'reboot_hyperparameters.batch_size')
    require(rb_n == len(steps) * rb_b, 'ReBoot trained samples != steps x batch')
    host, hsrc = s.get(REBOOT_SUBSET, 'provenance.host')
    rb_thr, tsrc = s.num(REBOOT_SUBSET, 'setting.omp_num_threads')
    require(rb_n > 0 and rb_train > 0 and rb_thr > 0, 'ReBoot timing denominators must be positive')
    step_src = (REBOOT_SUBSET_STATS, 'steps[*].reboot_step_time_s')
    boot_src = (REBOOT_SUBSET_STATS, 'bootstrap.seconds_per_call[*]')
    rb_per = rb_train / rb_n
    t.section('196 inputs, full setting (headline)')
    for key, b in (('FULL-1', '64'), ('FULL-2', '48 in 64')):
        rel = inp.path(key)
        if rel is None:
            t.row([key, '196', b, '48000', inp.pending(key), '', 'ReBoot 196 full: not measured', '', '', ''])
        else:
            validate_full(s, rel, 64 if key == 'FULL-1' else 48)
            hekit_time_row(t, s, rel, key, '196', host, hsrc, None, rb_thr, b)
    t.section('784 inputs, ReBoot encrypted-script subset')
    t.row(['ReBoot official (OpenFHE)', '784', Cell(f'{rb_b:g}', [bsrc]), Cell(f'{rb_n:g}', [nsrc]),
           Cell(f(rb_train, 1), [step_src], 'sum of step_time'), Cell(f(rb_per, 3), [step_src, nsrc]), '1',
           Cell(f(rb_boot / rb_n, 3), [boot_src, nsrc]), Cell(f((rb_train - rb_boot) / rb_n, 3), [step_src, boot_src, nsrc]),
           Cell(f'OpenMP {rb_thr:g}', [tsrc])])
    for key in ('F1', 'F2', 'R-41-3', 'R-41-4', 'R-41-1', 'R-41-2', 'R-41-5'):
        if inp.run_all and RUN[key][1] is None:
            continue
        rel = inp.path(key)
        if rel is None:
            t.row([key, '784', '', '', inp.pending(key)] + [''] * 5)
            continue
        hekit_time_row(t, s, rel, key, '784', host, hsrc, (rb_per, [step_src, nsrc]), rb_thr)
    t.note('ReBoot time = sum of its own step_time (encrypt batch, forward/backward, decrypt prediction, '
           'weight bootstrap; setup excluded). hekit time = time_s.train_total (all updates including '
           'bootstraps; key generation and bootstrapper setup excluded). Present rows fail closed unless '
           "the binary's provenance names the same host and thread count as the ReBoot run.", [hsrc, tsrc])
    t.note('Shared machine: other users\' jobs (SLAM, ~4 cores) ran during some rows; see '
           'docs/m3-report.md §3.2. ReBoot\'s full 196-input setting was not run (estimated 47.6 h), so '
           'the headline has no same-CPU wall-clock ratio.')
    return t


def hekit_time_row(t, s, rel, key, input_label, host, hsrc, reboot, rb_threads, batch_label=None):
    h, hs = s.get(rel, 'provenance.machine.host')
    require(h == host, f'{rel}: host {h} is not the ReBoot host {host} (same-CPU table)')
    train, trs = s.num(rel, 'time_s.train_total')
    boot, bts = s.num(rel, 'time_s.bootstrap_time_total')
    n, ns = s.num(rel, 'training.samples')
    b, bs = s.num(rel, 'setup.batch')
    thr, ths = s.get(rel, 'provenance.threads')
    m = re.search(r'RAYON_NUM_THREADS = (\d+)', thr)
    require(m is not None and int(m.group(1)) == rb_threads,
            f'{rel}: thread count does not match ReBoot {rb_threads:g} threads')
    require(n > 0 and train > 0 and 0 <= boot <= train, f'{rel}: invalid timing/sample denominator')
    per = train / n
    vs = Cell(f(per / reboot[0], 3), [trs, ns] + reboot[1], 'hekit s/sample / ReBoot s/sample') if reboot else 'not measured'
    t.row([key, input_label, Cell(batch_label or f'{b:g}', [bs]), Cell(f'{n:g}', [ns]), Cell(f(train, 1), [trs]),
           Cell(f(per, 3), [trs, ns]), vs, Cell(f(boot / n, 3), [bts, ns]), Cell(f((train - boot) / n, 3), [trs, bts, ns]),
           Cell(f'rayon {m.group(1)}' if m else thr, [ths, hs, hsrc])])


def table_accuracy(s, inp):
    t = Table('accuracy_vs_reboot', 'Test accuracy against ReBoot (R-M3-15)',
              ['Row', 'Setting', 'Seeds', 'Test acc (%)', 'Twin (%)', 'Reference (%)', 'Δ (pt)', 'Gate'], wide=True)
    per_seed, _ = s.get(REBOOT_FULL_CFG, 'per_seed')
    keys = sorted(per_seed, key=int)
    rb_vals = [s.num(REBOOT_FULL_CFG, f'per_seed.{k}.twin.test_acc_argmax10') for k in keys]
    rb0, rb0s = s.num(REBOOT_FULL_CFG, 'per_seed.0.twin.test_acc_argmax10')
    t.section('Full setting: 196 inputs, 4800 samples x 10 epochs, B = 48 (the ±0.5 pt gate)')
    t.row(["ReBoot released code, plaintext (recomputed twin)", 'full', len(rb_vals),
           mean_std([v for v, _ in rb_vals], [x for _, x in rb_vals], 2, 100), '—', '—', '—', 'reference'])
    tw, matched_rb = [], []
    for seed, (canonical, run_id) in enumerate(PLAIN196):
        rel = inp.simple(canonical, run_id)
        if rel is not None:
            tw.append(s.num(rel, 'epochs[-1].test_acc'))
            require(str(seed) in per_seed, f'ReBoot reference missing twin seed {seed}')
            matched_rb.append(s.num(REBOOT_FULL_CFG, f'per_seed.{seed}.twin.test_acc_argmax10'))
    if tw:
        d = statistics.mean(v for v, _ in tw) - statistics.mean(v for v, _ in matched_rb)
        t.row(['hekit plaintext twin (ReBoot weights, order, lr)', 'full', len(tw),
               mean_std([v for v, _ in tw], [x for _, x in tw], 2, 100), '—',
               mean_std([v for v, _ in matched_rb], [x for _, x in matched_rb], 2, 100),
               Cell(f'{100 * d:+.2f}', [x for _, x in tw] + [x for _, x in matched_rb], 'difference of means over matching seeds'), 'twin check'])
    else:
        t.row(['hekit plaintext twin', 'full', 0, 'pending', '', '', '', ''])
    for key, label, gate in (('FULL-2', 'hekit encrypted, B = 48 in 64 lanes, WN', True),
                             ('FULL-1', 'hekit encrypted, B = 64, WN', False)):
        rel = inp.path(key)
        if rel is None:
            t.row([label, 'full', '1 (seed 0)', inp.pending(key), inp.pending(key), Cell(pct(rb0), [rb0s]), '', 'pending' if gate else 'reported separately'])
            continue
        validate_full(s, rel, 48 if key == 'FULL-2' else 64)
        acc, acs = s.num(rel, 'evaluation.test_acc_encrypted_trained')
        twin, tws = s.num(rel, 'evaluation.test_acc_twin')
        d = acc - rb0
        verdict = ('PASS' if abs(d) <= 0.005 + 1e-12 else 'FAIL') if gate else 'not the gate (B ≠ 48)'
        t.row([label, 'full', '1 (seed 0)', Cell(pct(acc), [acs]), Cell(pct(twin), [tws]), Cell(pct(rb0) + ' (seed 0)', [rb0s]),
               Cell(f'{100 * d:+.2f}', [acs, rb0s]), Cell(verdict, [acs, rb0s], '|Δ| ≤ 0.5 pt')])
    t.section('eMLP-2 / eMLP-3 block MLPs, full setting, seed 0')
    for model, blocks in (('emlp2', '64'), ('emlp3', '128,64')):
        ref, refs = s.num(REBOOT_23_CFG, f'models.{model}.per_seed.0.twin.test_acc_argmax10')
        rel = inp.simple(f'results/mnist196_plain_{model}_seed0.json', f'plain196_{model}_s0')
        if rel is None:
            t.row([f'hekit plaintext twin {model}', 'full', 1, 'pending', '', Cell(pct(ref), [refs]), '', ''])
        else:
            acc, acs = s.num(rel, 'test_acc')
            t.row([f'hekit plaintext twin {model} (blocks {blocks})', 'full', '1 (seed 0)', Cell(pct(acc), [acs]), '—',
                   Cell(pct(ref) + ' (ReBoot twin)', [refs]), Cell(f'{100 * (acc - ref):+.2f}', [acs, refs]), 'twin check'])
    t.section('784 inputs, ReBoot encrypted-script subset, 1008 samples, 1 epoch (not a gate, R-M3-15)')
    rb, rbs = s.num(REBOOT_SUBSET_STATS, 'evaluation.test_acc_encrypted_trained_argmax10')
    t.row(['ReBoot official encrypted (OpenFHE)', 'subset, B = 48', '1 (seed 0)', Cell(pct(rb), [rbs]), '—', '—', '—', 'reference'])
    for key, b in (('F1', 32), ('F2', 64)):
        rel = inp.path(key)
        if rel is None:
            t.row([f'hekit encrypted WN, B = {b}', 'subset', '1', inp.pending(key), '', Cell(pct(rb), [rbs]), '', 'not a gate'])
            continue
        acc, acs = s.num(rel, 'evaluation.test_acc_encrypted_trained')
        twin, tws = s.num(rel, 'evaluation.test_acc_twin')
        t.row([f'hekit encrypted WN, B = {b}', 'subset', '1 (seed 0)', Cell(pct(acc), [acs]), Cell(pct(twin), [tws]),
               Cell(pct(rb), [rbs]), Cell(f'{100 * (acc - rb):+.2f}', [acs, rbs]), 'not a gate'])
    for key, label in (('R-52-1', 'hekit encrypted eMLP-2, B = 48 (196)'), ('R-52-2', 'hekit encrypted eMLP-3, B = 48 (196)')):
        rel = inp.path(key)
        if rel is None:
            t.row([label, 'subset 1008', '1', inp.pending(key), inp.pending(key), '', '', 'twin check'])
        else:
            acc, acs = s.num(rel, 'evaluation.test_acc_encrypted_trained')
            twin, tws = s.num(rel, 'evaluation.test_acc_twin')
            t.row([label, 'subset 1008', '1 (seed 0)', Cell(pct(acc), [acs]), Cell(pct(twin), [tws]), '—',
                   Cell(f'{100 * (acc - twin):+.2f} vs twin', [acs, tws]), 'twin check'])
    t.section('Reference only: the paper value and the released code at the paper setting (R-REB-3)')
    paper, ps = s.num(TASK51['mnist'][0], 'target.test_acc_mean')
    paper_sd, pss = s.num(TASK51['mnist'][0], 'target.test_acc_std')
    t.row(['ReBoot paper, Table 4 (plaintext)', '60k samples', Cell('10 (paper)', [(TASK51['mnist'][0], 'target.source')]),
           Cell(f'{pct(paper)} ± {pct(paper_sd)}', [ps, pss]), '—', '—', '—', 'not reproducible from the released defaults'])
    seeds, ss = s.get(REBOOT_PAPER, 'seeds')
    for sched, label in (('const', 'constant lr'), ('sched', "ReBoot's scheduler")):
        for at, fld in (('epoch 50', 'at_epoch.50'), ('best epoch (test-selected)', 'best_epoch_test_acc')):
            m, ms = s.num(REBOOT_PAPER, f'per_schedule.{sched}.{fld}.mean')
            sd, sds = s.num(REBOOT_PAPER, f'per_schedule.{sched}.{fld}.std_sample')
            t.row([f'ReBoot released code, paper setting, {label}, {at}', '60k samples, 50 epochs',
                   Cell(f'{len(seeds)}', [ss]), Cell(f'{pct(m)} ± {pct(sd)} (n = {len(seeds)})', [ms, sds]), '—',
                   Cell(f'{pct(paper)}', [ps]), Cell(f'{100 * (m - paper):+.2f}', [ms, ps]), 'reference'])
    t.note('R-M3-15: the ±0.5 pt gate compares hekit encrypted B = 48 (FULL-2) with ReBoot\'s released '
           'code at its full setting, seed 0. Subset rows are 1 epoch and are not a gate. The paper value '
           'and R-REB-3 are reported with this note only.')
    return t


def table_ablation(s, inp):
    t = Table('rq3_ablation', 'RQ3 ablation: one interactive eMLP-1 iteration under each EvalConfig',
              ['Source', 'Backend / level', 'hoist bsgs lazy', 'Rotations', 'Relins', 'KS/iter',
               'Closed form', 'Step median (s)', 'Step / default', 'Max |ΔW| vs default'], wide=True)
    caveats = None
    pair = [inp.path(key) for key in ('ABL-CUDA-L7', 'ABL-CPU-L7')]
    if all(pair):
        conditions = ('params', 'start_level', 'optimizer', 'layout_d', 'lanes_b', 'seed')
        gpu_setup, cpu_setup = [s.get(rel, 'setup')[0] for rel in pair]
        require(all(gpu_setup.get(k) == cpu_setup.get(k) for k in conditions),
                'RQ3 CPU/GPU pair must use the same params/level/optimizer/layout/batch/seed')
    for key, label in (('ABL-CPU-L31', 'CPU, n16_train, level 31'), ('ABL-CUDA-L7', 'GPU, n16_train, level 7'),
                       ('ABL-CPU-L7', 'CPU, n16_train, level 7'), ('ABL-SMOKE', 'CPU smoke')):
        rel = inp.path(key)
        if rel is None:
            if not (inp.run_all and RUN[key][1] is None):
                t.row([label, '', '', '', '', '', '', inp.pending(key), '', ''])
            continue
        params, psrc = s.get(rel, 'setup.params')
        backend, bsrc = s.get(rel, 'backend.device')
        top, tsrc = s.get(rel, 'backend.options.top_level')
        smoke = params == 'small'
        ok, oks = s.get(rel, 'all_checks_pass')
        require(ok is True, f'{rel}: all_checks_pass is not true')
        configs, _ = s.get(rel, 'configs')
        require(isinstance(configs, list) and len(configs) == 8,
                f'{rel}: the RQ3 research table requires all eight configurations')
        setup, _ = s.get(rel, 'setup')
        t.note(f'{label}: optimizer {setup["optimizer"]}, B = {setup["lanes_b"]}, '
               f'layout d = {setup["layout_d"]}, start level {setup["start_level"]}. '
               'Rows with unlike conditions are not a CPU/GPU speed comparison.',
               [(rel, 'setup.optimizer'), (rel, 'setup.lanes_b'),
                (rel, 'setup.layout_d'), (rel, 'setup.start_level')])
        for i, cfg in enumerate(configs):
            src = lambda fld: (rel, f'configs[{i}].{fld}')  # noqa: E731
            counts = cfg['counter_per_iteration']
            dw = cfg['max_abs_weight_diff_vs_default']
            t.row([Cell(label + (' (insecure N = 2^12: counts only)' if smoke else ''), [psrc]),
                   Cell(f'{backend}, level {top if top is not None else "max"}', [bsrc, tsrc]),
                   Cell(cfg['config'].replace('hoist=', '').replace('bsgs=', '').replace('lazy_relin=', ''), [src('config')]),
                   Cell(str(counts['rotations']), [src('counter_per_iteration.rotations')]),
                   Cell(str(counts['relins']), [src('counter_per_iteration.relins')]),
                   Cell(str(counts['key_switches']), [src('counter_per_iteration.key_switches')]),
                   Cell('match' if cfg['closed_form_matches'] else 'MISMATCH', [src('closed_form_matches')]),
                   Cell('n/a (smoke)' if smoke else f(cfg['step_s_median'], 3), [src('step_s_median')]),
                   Cell('n/a (smoke)' if smoke else f(cfg['step_time_over_default'], 3), [src('step_time_over_default')]),
                   Cell('0' if dw == 0 else f'{dw:.2e}', [src('max_abs_weight_diff_vs_default')])])
        if caveats is None:
            caveats = (rel, s.get(rel, 'caveats')[0])
    if caveats:
        rel, items = caveats
        for i, text in enumerate(items):
            t.note(text, [(rel, f'caveats[{i}]')])
    t.note('The smoke rows (insecure N = 2^12, high Mac load) check counts and closed forms only; '
           'their times are not comparative.')
    return t


def table_gpu(s, inp):
    t = Table('gpu_vs_cpu_ops', 'GPU vs CPU vs HEonGPU, n16_train (R-47-2, criterion medians, ms)',
              ['Operation', 'Level', 'hekit GPU', 'CPU 1 thr', 'CPU 12 thr', 'GPU speed-up (1 / 12 thr)',
               'HEonGPU (RTX 4090)', 'GPU / HEonGPU'], wide=True)
    lit = s.json(LITERATURE)
    heon = {e['operation']: (i, e) for i, e in enumerate(lit['heongpu_rtx4090'])}
    rows = (('add', 31, 'add'), ('relinearize', 31, 'relinearize'), ('mul + relin + rescale', 31, 'mul_relin_rescale'),
            ('rotate(1)', 31, 'rotate1'), ('hoisted rotate x16', 31, 'hoisted_rotate_16'),
            ('hoisted rotate x16 (capped keys)', 9, 'hoisted_rotate_16_L9'), ('matvec_enc d = 64', 31, 'matvec_enc_d64'),
            ('matvec_enc d = 64 (capped keys)', 9, 'matvec_enc_d64_L9'), ('matvec_enc d = 64 (capped keys)', 5, 'matvec_enc_d64_L5'))

    def med(bench):
        rel = CRITERION.format(bench)
        if not s.exists(rel):
            return None, None
        v, src = s.num(rel, 'median.point_estimate')
        return v / 1e6, src
    for label, level, bench in rows:
        g, gs = med('gpu_' + bench)
        c1, c1s = med('cpu_1t_' + bench)
        c12, c12s = med('cpu_12t_' + bench)
        if g is None:
            m, msrc = s.text_line(R47_LOG, rf'^gpu/{re.escape(bench)}: (skipped: .*)$')
            gcell = Cell(m.group(1), [msrc])
            speed = '—'
        else:
            gcell = Cell(f(g, 3), [gs])
            speed = Cell(f'{c1 / g:.0f}x / {c12 / g:.0f}x', [gs, c1s, c12s])
        hcell, ratio = 'none published', '—'
        if bench in heon:
            i, e = heon[bench]
            hms = e['value_us'] / 1000
            hsrc = (LITERATURE, f'heongpu_rtx4090[{i}].value_us')
            hcell = Cell(f'{f(hms, 3)} ({e["parameters"]})', [hsrc, (LITERATURE, f'heongpu_rtx4090[{i}].parameters')])
            if g is not None:
                ratio = Cell(f'{g / hms:.2f}x (nominal)', [gs, hsrc])
        t.row([label, level, gcell, Cell(f(c1, 3), [c1s]), Cell(f(c12, 3), [c12s]), speed, hcell, ratio])
    for label, bench in (('matvec_enc x4, 1 stream (sequential)', 'matvec_enc_x4_L5'),
                         ('matvec_enc x4, 4 streams (untuned R-47-2)', 'matvec_enc_batch4_L5')):
        g, gs = med('gpu_' + bench)
        t.row([label, 5, Cell(f(g, 3), [gs]), '—', '—', '—', 'none published', '—'])
    t.row(['matvec_enc x4, re-measured after the fix round', 5, 'pending (R-47-3)', '—', '—', '—', '—', '—'])
    for i, e in heon.values():
        t.note(f'HEonGPU {e["operation"]}: {e["source"]}. Caveat: {e["caveat"]}',
               [(LITERATURE, f'heongpu_rtx4090[{i}].source'), (LITERATURE, f'heongpu_rtx4090[{i}].caveat')])
    t.note('Capped keys: the input is dropped to level 9 or 5 and the keys are cut to that level '
           '(faithful for that level, not the top-level cost); the CPU uses full keys at the same level. '
           'The shared RTX 4090 had 2-3 GiB free, so full-key GPU rows that need more were skipped.')
    return t


def table_security(s, inp):
    t = Table('security', 'Security estimates (lattice-estimator; log2 rop, minimum over attacks)',
              ['Parameters', 'N', 'log PQ', 'Secret', 'BDGL16', 'MATZOV', '>= 128 (both)'], wide=True)

    def doc_row(label, pattern):
        try:
            m, src = s.text_line(SECURITY_DOC, r'^\| ' + pattern + r' \|(.*)\|\s*$')
        except TableError:
            return False
        cells = [x.strip() for x in m.group(1).split('|')]
        require(len(cells) == 6, f'{SECURITY_DOC}: unexpected columns in row {label!r}')
        bd, mz = float(cells[3].strip('*')), float(cells[4].strip('*'))
        t.row([label, Cell(cells[0], [src]), Cell(cells[1], [src]), Cell(cells[2], [src]),
               Cell(f'{bd:.1f}', [src]), Cell(f'{mz:.1f}', [src]), Cell('yes' if min(bd, mz) >= 128 else 'no', [src])])
        return True
    doc_row('hekit (a): n16_boot main key', re.escape('hekit (a): `n16_boot` main key'))
    doc_row("hekit (b): SSE key s -> s' (h' = 32)", re.escape("hekit (b): SSE key s → s'"))
    if doc_row("ReBoot main key (R-REB-4: OpenFHE's own chain)", re.escape('ReBoot main key')):
        m, src = s.text_line(SECURITY_DOC, r'ReBoot is at ([0-9.]+) / ([0-9.]+)')
        t.note(f'ReBoot row: the MITM tier (operation counts; about 2^104-2^110 memory). Without '
               f'MITM attacks ReBoot is at {m.group(1)} / {m.group(2)} (BDGL16 / MATZOV), also below 128.', [src])
    else:
        t.row(["ReBoot main key (R-REB-4)", '2^16', '', 'sparse ternary, h = 192', 'pending (Task 39)',
               'pending (Task 39)', 'pending'])
    t.note('Source: the summary table of docs/security.md (the estimator outputs are recorded in that '
           'doc; no JSON exists). Same lattice-estimator models for every row; m = N; Core-SVP is in the '
           'doc for information only. Equal N does not mean equal security: no equal-security claim.', [])
    return t


def table_datasets(s, inp):
    t = Table('plain_datasets', 'Plaintext dataset reproduction against ReBoot Table 4 (Task 51)',
              ['Dataset', 'Config', 'Epochs', 'Seeds', 'Fixed final epoch (%)', 'Best epoch, test-selected (%)',
               'ReBoot Table 4 (%)', 'Δ fixed (pt)', 'Within 1 pt'], wide=True)
    for key, name, cfg in (('mnist', 'MNIST 14x14', 'paper setting'), ('fmnist', 'Fashion-MNIST 14x14', 'paper setting'),
                           ('bc', 'Breast Cancer', 'fixed split, min-max (primary)'),
                           ('bc_split', 'Breast Cancer', 'per-run split (sensitivity)'),
                           ('bc_z', 'Breast Cancer', 'z-score (diverges)')):
        rel = inp.simple(*TASK51[key])
        if rel is None:
            t.row([name, cfg, '', '', 'pending', '', '', '', ''])
            continue
        runs, _ = s.get(rel, 'runs')
        epochs, es = s.num(rel, 'effective.epochs')
        tgt, tgs = s.num(rel, 'target.test_acc_mean')
        tsd, tss = s.num(rel, 'target.test_acc_std')
        final, best, diverged, srcs = [], [], 0, []
        for i, r in enumerate(runs):
            accs = r['test_acc']
            require(len(accs) == epochs, f'{rel}: runs[{i}] has {len(accs)} epochs, expected {epochs}')
            if r['nonfinite_from_epoch'] is not None:
                diverged += 1
            final.append(accs[-1])
            best.append(max(accs))
            srcs.append((rel, f'runs[{i}].test_acc'))
        ref = Cell(f'{pct(tgt)} ± {pct(tsd)}', [tgs, tss])
        if diverged:
            dsrc = [(rel, f'runs[{i}].nonfinite_from_epoch') for i in range(len(runs))]
            t.row([name, cfg, Cell(f'{epochs:g}', [es]), len(runs), Cell(f'diverged ({diverged}/{len(runs)} seeds)', dsrc),
                   '—', ref, '—', 'no'])
            continue
        d = statistics.mean(final) - tgt
        t.row([name, cfg, Cell(f'{epochs:g}', [es]), len(runs), mean_std(final, srcs, 2, 100),
               mean_std(best, srcs, 2, 100), ref, Cell(f'{100 * d:+.2f}', srcs + [tgs]),
               Cell('yes' if abs(d) <= 0.01 + 1e-12 else 'no', srcs + [tgs])])
    t.note('Fixed final epoch is the primary figure. The best epoch is selected on the test set and is '
           'optimistic; it is shown only because ReBoot does not state its epoch count. Seeds: the '
           "plain binary's own seeds (not ReBoot's initial weights).")
    t.note('Fashion-MNIST and Breast Cancer are plaintext twins only (no encrypted runs).')
    return t


TABLES = [table_rq1, table_wallclock, table_accuracy, table_ablation, table_gpu, table_security, table_datasets]


# ---------------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------------

def md_escape(text):
    return str(text).replace('|', r'\|').replace('\n', ' ')


def render_md(t, origin):
    out = [f'# {t.title}', '', f'<!-- generated by tools/paper_tables.py from {origin}; do not edit -->', '',
           '| ' + ' | '.join(md_escape(c) for c in t.columns) + ' |',
           '|' + '|'.join('---' for _ in t.columns) + '|']
    for kind, item in t.rows:
        if kind == 'section':
            out.append(f'| **{md_escape(item)}** |' + ' |' * (len(t.columns) - 1))
        else:
            out.append('| ' + ' | '.join(md_escape(c.text) for c in item) + ' |')
    if t.notes:
        out.append('')
        out += [f'- {md_escape(n.text)}' for n in t.notes]
    return '\n'.join(out) + '\n'


TEX_MAP = {'§': r'\S{}', '≈': r'$\approx$', '·': r'$\cdot$', 'ℓ': r'$\ell$', "'": "'",'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
           '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
           '±': r'$\pm$', '≠': r'$\neq$', '≤': r'$\leq$', '≥': r'$\geq$', 'Δ': r'$\Delta$', '→': r'$\to$', '—': '---',
           '–': '--', '×': r'$\times$', '`': '', '\n': ' '}


def tex_escape(text):
    return ''.join(TEX_MAP.get(ch, ch) for ch in str(text))


def tex_prose(text):
    """A note for LaTeX: Markdown emphasis, code and link brackets dropped; an inline Markdown
    table becomes 'cell / cell / ...' clauses. The words are unchanged."""
    # The binary's verbatim caveat serializes its five-column Markdown table inline.
    # Turn each row into a clause, including the header; separator rows carry no words.
    def inline_row(match):
        cells = [c.strip() for c in match.group(1).split('|')]
        if all(re.fullmatch(r'[-: ]+', c) for c in cells):
            return ' '
        return '(' + ' / '.join(cells) + '); '

    text = re.sub(r'\|([^|\n]+(?:\|[^|\n]+){4})\|', inline_row, str(text))
    out = []
    for line in str(text).split('\n'):
        s = line.strip()
        if s.startswith('|'):
            if set(s) <= set('|-: '):
                continue
            s = '(' + ' / '.join(c.strip() for c in s.strip('|').split('|')) + ')'
        out.append(s)
    s = ' '.join(x for x in out if x)
    s = s.replace('**', '').replace('`', '')
    s = re.sub(r'\[([^\]]+)\]', r'\1', s)
    return tex_escape(s)


def table_quick(s, inp):
    """Quick outputs expose coverage only, never FULL validation or research gates."""
    t = Table('quick_coverage', 'Quick pipeline coverage (not paper measurements)',
              ['Run', 'Scope', 'Evidence'])
    for path in sorted((s.root / 'results/run_all/quick').glob('q_*.json')):
        rel = str(path.relative_to(s.root))
        inp.new_result(rel)
        t.row([path.stem, 'pipeline smoke only', Cell('binary output present; research gates not evaluated',
                                                      [(rel, 'provenance.written_by')])])
    t.note('Quick subsets are not full-length results. No FULL validator, acceptance verdict, '
           'paper accuracy gate, or CPU/GPU performance comparison is applied here.')
    return t


def render_tex(t, origin):
    env = 'table*' if t.wide else 'table'
    out = [f'% generated by tools/paper_tables.py from {origin}; do not edit',
           f'\\begin{{{env}}}[t]', r'\centering', r'\scriptsize',
           f'\\caption{{{tex_escape(t.title)}}}', f'\\label{{tab:{t.name}}}',
           r'\resizebox{\linewidth}{!}{%',
           f'\\begin{{tabular}}{{l{"r" * (len(t.columns) - 1)}}}', r'\toprule',
           ' & '.join(tex_escape(c) for c in t.columns) + r' \\', r'\midrule']
    first = True
    for kind, item in t.rows:
        if kind == 'section':
            if not first:
                out.append(r'\midrule')
            out.append(f'\\multicolumn{{{len(t.columns)}}}{{l}}{{\\textit{{{tex_escape(item)}}}}} \\\\')
        else:
            out.append(' & '.join(tex_escape(c.text) for c in item) + r' \\')
        first = False
    out += [r'\bottomrule', r'\end{tabular}}']
    if t.notes:
        out.append(r'\par\smallskip\begin{minipage}{\linewidth}\scriptsize')
        out += [tex_prose(n.text) + r'\par' for n in t.notes]
        out.append(r'\end{minipage}')
    out.append(f'\\end{{{env}}}')
    return '\n'.join(out) + '\n'


def source_lines(t):
    lines = []
    for kind, item in t.rows:
        if kind != 'row':
            continue
        for col, cell in zip(t.columns, item):
            for rel, field in cell.sources:
                lines.append('\t'.join([t.name, item[0].text, col, cell.text, rel, field, cell.formula]))
    for i, n in enumerate(t.notes):
        for rel, field in n.sources:
            lines.append('\t'.join([t.name, f'note {i + 1}', '', '', rel, field, '']))
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sources', nargs='?', const='all', metavar='TABLE',
                    help='print table/row/column/value/file/field for every cell (TSV) and exit')
    ap.add_argument('--check', action='store_true', help='compare with the files on disk; exit 1 if stale')
    ap.add_argument('--from-run-all', choices=('full', 'quick'),
                    help='read rows from results/run_all/<mode>/ and write to results/run_all/<mode>/tables/')
    ap.add_argument('--tex-dir', help='default paper/tables')
    ap.add_argument('--md-dir', help='default results/tables')
    ap.add_argument('--root', default=str(REPO), help='repository root (default: this checkout)')
    args = ap.parse_args(argv)
    store = Store(args.root)
    inp = Inputs(store, args.from_run_all)
    origin = f'results/run_all/{args.from_run_all}/' if args.from_run_all else 'results/'
    if args.from_run_all:
        base = Path(args.root) / 'results' / 'run_all' / args.from_run_all / 'tables'
        tex_dir, md_dir = Path(args.tex_dir or base / 'tex'), Path(args.md_dir or base / 'md')
    else:
        tex_dir = Path(args.tex_dir or Path(args.root) / 'paper' / 'tables')
        md_dir = Path(args.md_dir or Path(args.root) / 'results' / 'tables')
    try:
        tables = [table_quick(store, inp)] if args.from_run_all == 'quick' else [build(store, inp) for build in TABLES]
        # Render into memory before touching any output, even if a later renderer fails.
        files = {}
        for t in tables:
            files[md_dir / f'{t.name}.md'] = render_md(t, origin)
            files[tex_dir / f'{t.name}.tex'] = render_tex(t, origin)
    except (TableError, KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        print(f'paper_tables.py: {exc}', file=sys.stderr)
        return 2
    header = 'table\trow\tcolumn\tvalue\tfile\tfield\tformula'
    if args.sources:
        names = {t.name for t in tables}
        if args.sources != 'all' and args.sources not in names:
            print(f'paper_tables.py: unknown table {args.sources} ({", ".join(sorted(names))})', file=sys.stderr)
            return 2
        print(header)
        for t in tables:
            if args.sources in ('all', t.name):
                print('\n'.join(source_lines(t)))
        return 0
    files[md_dir / 'sources.tsv'] = header + '\n' + '\n'.join(line for t in tables for line in source_lines(t)) + '\n'
    if args.check:
        stale = [p for p, text in files.items() if not p.is_file() or p.read_text() != text]
        for p in stale:
            print(f'stale: {p}', file=sys.stderr)
        return 1 if stale else 0
    for p, text in files.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    print(f'wrote {len(tables)} tables to {md_dir} (.md, sources.tsv) and {tex_dir} (.tex)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
