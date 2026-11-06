"""Task54 tables: FULL fixtures are SYNTHETIC, never measurements."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import paper_tables as tables
from test_rq1_summary import synthetic_full

ROOT = Path(__file__).resolve().parents[1]


def archived_overlay(root):
    """Link read-only archived inputs; new fixture/output paths stay in the temp tree."""
    (root / 'docs').symlink_to(ROOT / 'docs', target_is_directory=True)
    for base in ('results', 'results/quadruped', 'results/quadruped/raw'):
        dest = root / base
        dest.mkdir(parents=True, exist_ok=True)
        for p in (ROOT / base).iterdir():
            q = dest / p.name
            if (q.exists() or p.name in ('run_all', 'tables', 'R-41-FULL-1', 'R-41-FULL-2',
                                         'ablation_cuda.json', 'ablation_cpu_l7.json')):
                continue
            if p.is_file() or (p.is_dir() and base.endswith('/raw')) or p.name == 'reference':
                q.symlink_to(p, target_is_directory=p.is_dir())


def put_full(root, batch=64, mutate=None):
    doc = synthetic_full(batch)
    # Match only the fixture's machine/thread metadata to the historical timing reference.
    doc['provenance']['machine']['host'] = 'kmihara-Z790-S01'
    if mutate:
        mutate(doc)
    key = 'FULL-1' if batch == 64 else 'FULL-2'
    path = root / tables.RUN[key][0][0]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return path


def rows(table):
    return [[c.text for c in item] for kind, item in table.rows if kind == 'row']


class PaperTablesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        archived_overlay(self.root)

    def build(self, fn):
        store = tables.Store(self.root)
        return fn(store, tables.Inputs(store))

    def test_interactive_reference_never_populates_ni_d_or_ways(self):
        # Break caught: labeling the independent interactive run as NI-derived way d.
        t = self.build(tables.table_rq1)
        r = next(r for r in rows(t) if r[0] == 'interactive')
        self.assertEqual(r[9:14], ['—'] * 5)
        self.assertTrue(any('interactive' in n.text.lower() and '0.564' in n.text for n in t.notes))

    def test_synthetic_full_rows_and_accuracy_gate(self):
        # Break caught: untested populated FULL branch, shifted columns or wrong B48 label.
        put_full(self.root)
        put_full(self.root, 48)
        t = self.build(tables.table_rq1)
        one, two = [r for r in rows(t) if r[0] in ('FULL-1', 'FULL-2')]
        self.assertEqual(one[9:13], ['0.234', '0.389', '0.399', '0.405'])
        self.assertEqual(two[3], '48 in 64')
        acc = rows(self.build(tables.table_accuracy))
        self.assertEqual(next(r for r in acc if r[0].startswith('hekit encrypted, B = 48'))[-1], 'PASS')
        wall = rows(self.build(tables.table_wallclock))
        self.assertEqual(next(r for r in wall if r[0] == 'FULL-2')[2], '48 in 64')
        self.assertEqual(next(r for r in wall if r[0] == 'FULL-1')[6], 'not measured')

    def test_each_consumer_rejects_malformed_full(self):
        # Break caught: wall-clock accepting truncated/invalid FULL despite RQ1 validation.
        put_full(self.root, mutate=lambda d: d['training'].update(samples=1))
        for fn in (tables.table_rq1, tables.table_accuracy, tables.table_wallclock):
            with self.subTest(table=fn.__name__), self.assertRaises(tables.TableError):
                self.build(fn)

    def test_failed_full_writes_no_partial_files(self):
        # Break caught: writing early tables before discovering malformed later input.
        p = put_full(self.root)
        p.write_text('{"provenance":')
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            rc = tables.main(['--root', str(self.root)])
        self.assertEqual(rc, 2)
        self.assertFalse((self.root / 'paper').exists())
        self.assertFalse((self.root / 'results/tables').exists())

    def test_same_cpu_timing_rejects_thread_mismatch(self):
        # Break caught: creating a 12-thread ratio from a 1-thread hekit result.
        put_full(self.root, mutate=lambda d: d['provenance'].update(
            threads='rayon (feature parallel), RAYON_NUM_THREADS = 1, 32 logical CPUs'))
        with self.assertRaises(tables.TableError):
            self.build(tables.table_wallclock)

    def test_zero_samples_fail_as_table_error(self):
        # Break caught: divide-by-zero escaping validation in ordinary NI rows.
        s = tables.Store(self.root)
        rel = tables.RUN['F1'][0][0]
        d = copy.deepcopy(s.json(rel))
        d['training']['samples'] = 0
        s.cache[rel] = d
        with self.assertRaises(tables.TableError):
            tables.table_rq1(s, tables.Inputs(s))

    def test_twin_comparison_uses_the_same_seed_subset(self):
        # Break caught: comparing one available twin against all ten ReBoot reference seeds.
        store = tables.Store(self.root)
        inp = tables.Inputs(store)
        original = inp.simple
        inp.simple = lambda canonical, run_id: (
            None if run_id.startswith('plain196_emlp1_s') and not run_id.endswith('_s0')
            else original(canonical, run_id))
        t = tables.table_accuracy(store, inp)
        r = next(r for r in rows(t) if r[0].startswith('hekit plaintext twin (ReBoot weights'))
        self.assertEqual(r[5], '90.41 (n = 1)')

    def test_inline_caveat_table_is_readable_prose_in_tex(self):
        # Break caught: inline Markdown pipes leaking unchanged into a PDF note.
        t = self.build(tables.table_ablation)
        tex = tables.render_tex(t, 'SYNTHETIC TEST rendering of archived input')
        notes = tex.split(r'\begin{minipage}', 1)[1]
        self.assertNotIn('|---|', notes)
        self.assertNotIn('| form |', notes)
        for label in ('Definition of', 'hoist changes no', 'gradient'):
            self.assertIn(label, notes)
        self.assertIn('default BSGS', notes)
        self.assertIn('13', notes)

    def test_quick_tables_do_not_validate_or_label_smokes_as_full(self):
        # Break caught: treating q_ subset IDs as absent or full training results.
        p = self.root / 'results/run_all/quick/q_ni196_wn_b64_sub128_s0.json'
        p.parent.mkdir(parents=True)
        d = synthetic_full()
        d['training'].update(samples=128, updates=2)
        p.write_text(json.dumps(d))
        s = tables.Store(self.root)
        inp = tables.Inputs(s, 'quick')
        self.assertIsNone(inp.path('FULL-1'))
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()) as output:
            rc = tables.main(['--root', str(self.root), '--from-run-all', 'quick'])
        self.assertEqual(rc, 0, output.getvalue())
        self.assertFalse((self.root / 'paper').exists())
        quick = (self.root / 'results/run_all/quick/tables/md/quick_coverage.md').read_text()
        self.assertIn('q_ni196_wn_b64_sub128_s0', quick)
        self.assertIn('pipeline', quick.lower())
        self.assertNotIn('PASS', quick)

    def test_compatibility_entrypoint_regenerates_identical_canonical_tables(self):
        # Break caught: legacy tools/tables.py path missing or diverging from the authority.
        p = subprocess.run([sys.executable, str(ROOT / 'tools/tables.py'), '--root', str(self.root)],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertTrue((self.root / 'paper/tables/rq1_ks.tex').is_file())

    def test_new_full_directory_rejects_runner_only_and_annotated_provenance(self):
        # Break caught: using wrapper metadata or historical annotations as a new writer.
        p = self.root / 'results/run_all/full/ni196_wn_b64_full_s0.json'
        p.parent.mkdir(parents=True)
        for provenance in ({}, {'written_by': 'scripts/run_all.sh', 'git_commit': '1'*40},
                           {'written_by': 'the mnist binary (raw output, not assembled)',
                            'git_commit': '1'*40, 'note': 'hand-added'}):
            with self.subTest(provenance=provenance):
                p.write_text(json.dumps({'provenance': provenance}))
                s = tables.Store(self.root)
                with self.assertRaises(tables.TableError):
                    tables.Inputs(s, 'full').path('FULL-1')

    def test_ablation_gpu_cpu_pair_requires_the_same_optimizer_level_layout_and_batch(self):
        # Break caught: presenting NAG CPU against SGD GPU, or unlike layouts/levels.
        base = json.loads((ROOT / 'results/ablation_cpu_smoke_small.json').read_text())
        for field, value in (('optimizer', 'sgd'), ('start_level', 6), ('layout_d', 512), ('lanes_b', 16)):
            with self.subTest(field=field):
                gpu, cpu = copy.deepcopy(base), copy.deepcopy(base)
                gpu['backend']['device'] = 'cuda'
                for doc in (gpu, cpu):
                    doc['setup'].update(params='n16_train', start_level=7)
                    doc['backend']['options']['top_level'] = 7
                gpu['setup'][field] = value
                for name, doc in (('ablation_cuda.json', gpu), ('ablation_cpu_l7.json', cpu)):
                    (self.root / 'results/quadruped' / name).write_text(json.dumps(doc))
                with self.assertRaises(tables.TableError):
                    self.build(tables.table_ablation)


if __name__ == '__main__':
    unittest.main()
