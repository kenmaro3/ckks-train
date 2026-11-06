"""Focused checks for result-derived paper figures; no research runs."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('paper_figs', Path(__file__).with_name('figs.py'))
figs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(figs)


class FiguresTest(unittest.TestCase):
    def test_current_ratios_have_traceable_sources(self):
        data = figs.figure_data(Path(__file__).resolve().parents[1])
        self.assertEqual([r['run'] for r in data['rows']], ['F1', 'F2'])
        self.assertEqual(data['scope'], 'supplementary 784-input subset; one seed per row')
        self.assertEqual([r['batch'] for r in data['rows']], [32, 64])
        self.assertEqual(data['rows'][0]['ratios'], [0.435, 0.779, 0.756, 0.742])
        self.assertEqual(data['rows'][1]['ratios'], [0.233, 0.414, 0.404, 0.397])
        self.assertTrue(data['sources'])
        self.assertTrue(all(len(src['sha256']) == 64 and src['fields'] for src in data['sources']))

    def test_missing_measured_row_is_an_error(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(figs.FigureError):
                figs.figure_data(Path(root))

    def test_render_preserves_values_and_scope(self):
        data = figs.figure_data(Path(__file__).resolve().parents[1])
        tex, svg = figs.render(data)
        self.assertIn('one seed', tex)
        self.assertIn('0.779', tex)
        self.assertIn('<svg', svg)
        self.assertIn('supplementary', svg)
        self.assertNotIn('FULL-1', svg)
        changed = copy.deepcopy(data)
        changed['rows'][0]['ratios'][0] = 0.123
        self.assertNotEqual(figs.render(changed), (tex, svg))


if __name__ == '__main__':
    unittest.main()
