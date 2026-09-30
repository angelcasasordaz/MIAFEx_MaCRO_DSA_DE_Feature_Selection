"""Validate the reporting PNG sink using tiny actual renders."""
import ast
from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt
from PIL import Image

from reporting import core, figures


class ReportingPNGTests(unittest.TestCase):
    def test_explicit_png_sink_preserves_historical_pdf(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pdf = root / 'historical.pdf'
            pdf.write_bytes(b'historical PDF')
            stage = root / 'generated'; stage.mkdir()
            fig, ax = plt.subplots(figsize=(.5, .5), dpi=72)
            ax.plot([0, 1])
            figures.save_png(fig, stage / 'new.pdf')
            self.assertFalse(plt.fignum_exists(fig.number))
            self.assertEqual(pdf.read_bytes(), b'historical PDF')
            self.assertFalse((stage / 'new.pdf').exists())
            with Image.open(stage / 'new.png') as png:
                self.assertEqual(png.format, 'PNG')
                for dpi in png.info['dpi']:
                    self.assertAlmostEqual(dpi, 600, delta=.1)
            core._validate_artifacts(stage, stage, ['new.png'])

    def test_reporting_has_one_600_dpi_png_sink(self):
        writes = []
        for path in Path(core.__file__).parent.glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Call):
                    name = getattr(node.func, 'attr', getattr(node.func, 'id', ''))
                    self.assertNotIn(name, {'PdfPages', 'print_pdf'})
                    if name == 'savefig':
                        writes.append(node)
        self.assertEqual(len(writes), 1)
        keywords = {key.arg: ast.literal_eval(key.value) for key in writes[0].keywords}
        self.assertEqual((keywords['format'], keywords['dpi']), ('png', 600))

    def test_invalid_dpi_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.new('RGB', (10, 10)).save(root / 'bad.png', dpi=(150, 150))
            with self.assertRaisesRegex(ValueError, '600 dpi'):
                core._validate_artifacts(root, root, ['bad.png'])
