"""Run the whole pipeline end to end (synthetic traces always; real traces if present)."""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
steps = ['01_generate_synthetic.py', '03_evaluate_predictors.py', '04_baselines.py', '05_ablation.py']
for s in steps:
    print(f'\n{"="*70}\n>>> {s}\n{"="*70}')
    r = subprocess.run([sys.executable, os.path.join(HERE, s)])
    if r.returncode != 0:
        sys.exit(f'{s} failed')
print('\nAll stages complete.')
