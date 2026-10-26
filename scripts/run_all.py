"""Run the whole pipeline end to end (synthetic traces always; real traces if present),
then open results/report.html in the browser.  Use --no-open to skip opening it."""
import os, subprocess, sys, webbrowser
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
from paging.config import RESULTS_DIR
steps = ['01_generate_synthetic.py', '03_evaluate_predictors.py', '04_baselines.py',
         '05_ablation.py', '06_head_to_head.py', '07_ghost_estimate.py', '08_evaluation_harness.py',
         '09_phase8_baselines.py', '10_improvements.py', '11_robustness.py', '12_time_model.py',
         '14_time_model.py', '13_report.py']   # 15/16 chose the frozen settings: run them only to re-derive those
for s in steps:
    print(f'\n{"="*70}\n>>> {s}\n{"="*70}')
    r = subprocess.run([sys.executable, os.path.join(HERE, s)])
    if r.returncode != 0:
        sys.exit(f'{s} failed')
print('\nAll stages complete.')
report = os.path.abspath(os.path.join(RESULTS_DIR, 'report.html'))
print(f'Report: {report}')
if '--no-open' not in sys.argv and os.path.exists(report):
    webbrowser.open('file:///' + report.replace(os.sep, '/'))
