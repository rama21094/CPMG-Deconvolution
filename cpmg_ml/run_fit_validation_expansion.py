"""Bounded exploratory check: 3 exchange rates x 2 noise draws, fixed pB=.05."""
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path('docs/ml_results/fit_expansion_20260910')
root.mkdir(exist_ok=True)
for kex in (300,600,1800):
    for seed in (99101,99102):
        out=root/f'k{kex}_seed{seed}.json'
        if out.exists():
            continue
        cmd=[sys.executable,'-m','cpmg_ml.fit_validation_pilot','--kex',str(kex),'--noise-seed',str(seed),
             '--work-dir',f'/private/tmp/cpmg-expand-20260910-k{kex}-s{seed}','--out',str(out)]
        result=subprocess.run(cmd,env=dict(os.environ,OPENBLAS_NUM_THREADS='1'),capture_output=True,text=True)
        (root/f'k{kex}_seed{seed}.log').write_text(result.stdout+result.stderr)
        if result.returncode:
            raise RuntimeError(f'Pilot failed; inspect {out.with_suffix(".log")}')
        data=json.loads(out.read_text())
        print(kex,seed,[(r['run'],r.get('returncode')) for r in data['results']],flush=True)
