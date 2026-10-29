"""
The Colab notebook copies its classes from src/paging (scripts/build_notebook.py). If src/paging changes and the
notebook is not rebuilt, the notebook would run different code from the tested library: this test catches that.
Fix: python scripts/build_notebook.py
"""
import importlib.util
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def test_notebook_matches_library():
    spec = importlib.util.spec_from_file_location('build_notebook', os.path.join(ROOT, 'scripts', 'build_notebook.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)            # builds the cell list in memory; writes nothing
    with open(mod.OUT, encoding='utf-8') as f:
        on_disk = [(c['cell_type'], ''.join(c['source'])) for c in json.load(f)['cells']]
    assert on_disk == mod.CELLS, 'notebook is out of date: run  python scripts/build_notebook.py'
