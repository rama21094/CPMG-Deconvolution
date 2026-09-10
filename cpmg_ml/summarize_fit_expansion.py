"""Summarize the exploratory six-case fitting expansion without selecting runs."""
import json
from pathlib import Path
import numpy as np


def main():
    files=sorted(Path('docs/ml_results/fit_expansion_20260910').glob('k*_seed*.json'))
    arms=['clean_j0','noisy_j0','noisy_coupled','cnn_corrected']
    cases=[]
    for path in files:
        d=json.loads(path.read_text())
        case={'kex':d['truth']['KEX_AB'],'seed':d['noise_seed'],'arms':{}}
        for arm in arms:
            runs=[r for r in d['results'] if r['run'].startswith(arm+'_')]
            if len(runs)!=2 or any(r.get('returncode')!=0 or not r.get('fitted') for r in runs):
                case['arms'][arm]={'failed':True}
                continue
            f=runs[0]['fitted']
            errors={'KEX_AB':100*(f['GLOBAL']['KEX_AB']/case['kex']-1),
                    'PB':100*(f['GLOBAL']['PB']/.05-1),
                    'DW_AB':float(np.mean([100*abs(abs(f['DW_AB'][f'{i+1}N'])/v-1) for i,v in enumerate([1,3,5])]))}
            agreement=max(abs(runs[0]['fitted']['GLOBAL'][k]-runs[1]['fitted']['GLOBAL'][k])/max(abs(runs[0]['fitted']['GLOBAL'][k]),1e-12) for k in ['KEX_AB','PB'])
            case['arms'][arm]={'failed':False,'errors_percent':errors,'start_max_relative_difference':agreement,
                               'fitted':f}
        cases.append(case)
    aggregate={}
    for arm in arms:
        valid=[c['arms'][arm] for c in cases if not c['arms'][arm]['failed']]
        aggregate[arm]={k:{'mean_absolute_percent_error':float(np.mean([abs(v['errors_percent'][k]) for v in valid])),
                          'max_absolute_percent_error':float(np.max([abs(v['errors_percent'][k]) for v in valid]))} for k in ['KEX_AB','PB','DW_AB']} if valid else {}
    paired={}
    for k in ['KEX_AB','PB','DW_AB']:
        ok=[c for c in cases if not c['arms']['noisy_coupled']['failed'] and not c['arms']['cnn_corrected']['failed']]
        paired[k]={'improved':sum(abs(c['arms']['cnn_corrected']['errors_percent'][k])<abs(c['arms']['noisy_coupled']['errors_percent'][k]) for c in ok),'total':len(ok)}
    out={'scope':'Exploratory fixed geometry, pB=.05, kex=300/600/1800, two noise draws each. Six cases are not six independent physical systems. DW metric compares shift magnitudes, averaging three residue absolute relative errors per case; signed fits are retained. No population inference.',
         'complete':len(cases)==6,'aggregate':aggregate,'paired_improvement':paired,'cases':cases}
    Path('docs/ml_results/fit_expansion_20260910_summary.json').write_text(json.dumps(out,indent=2))
    print(json.dumps({k:v for k,v in out.items() if k!='cases'},indent=2))


if __name__=='__main__':
    main()
