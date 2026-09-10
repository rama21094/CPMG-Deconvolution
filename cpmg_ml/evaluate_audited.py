"""Re-evaluate existing CNNs with duplicate exclusions and a smoothing control.

Existing test data are retrospective, not a new locked confirmatory test set.
Noise generation retains the historical clipping policy so results can be
compared; censoring is quantified rather than silently interpreted as valid R2.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scipy.signal import savgol_filter
from .baselines import MODELS, load_split, pack


def infer(checkpoint, raw, sigma, seed, device):
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = MODELS[ck['arch']](raw['with_j'].shape[1], 4).to(device)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    data, _ = pack(raw, ck['stats'], sigma_rel=sigma, seed=seed)
    pred = []
    with torch.inference_mode():
        for i in range(0, len(data['wj']), 256):
            pred.append(model(data['pts'][i:i+256].to(device), data['glb'][i:i+256].to(device)).cpu().numpy())
    return data['wj'] - data['scale_np']*np.concatenate(pred), data['wj']


def metrics(pred, target, mask):
    error = pred-target
    counts = mask.sum(1)
    ok = counts>0
    per = np.sqrt(np.sum(np.where(mask,error**2,0),axis=1)[ok]/counts[ok])
    return {'profiles':int(ok.sum()),'points':int(mask.sum()),
            'rmse':float(np.sqrt((error[mask]**2).mean())),
            'median_profile_rmse':float(np.median(per)),
            'p95_profile_rmse':float(np.quantile(per,.95))}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,default=Path('data/cpmg_dej_chemex_500k'))
    p.add_argument('--audit',type=Path,default=Path('docs/ml_results/benchmark_audit_500k.json'))
    p.add_argument('--out',type=Path,default=Path('docs/ml_results/audited_cnn_evaluation.json'))
    a=p.parse_args()
    audit=json.loads(a.audit.read_text())
    raw={s:load_split(a.data_dir,s) for s in ('val','test')}
    for s in raw:
        keep=np.ones(len(raw[s]['with_j']),dtype=bool)
        keep[audit['excluded_duplicate_row_indices'][s]]=False
        raw[s]={k:v[keep] if isinstance(v,np.ndarray) else v for k,v in raw[s].items()}
    dev='mps' if torch.backends.mps.is_available() else 'cpu'
    # Choose smoothing window against clean J=0 validation targets, never test.
    val,_=pack(raw['val'],sigma_rel=.01,seed=23456)
    scores={w:float(np.mean((savgol_filter(val['wj'],w,2,axis=1)-raw['val']['no_j'])**2)) for w in (5,9,15,21,31,39)}
    window=min(scores,key=scores.get)
    checkpoints={0.:Path('runs/ladder500k_clean/cnn.pt'), .01:Path('runs/b500k_noise01/cnn.pt')}
    results=[]
    for sigma,path in checkpoints.items():
        for seed in ([34567] if sigma==0 else [34567,45678,56789]):
            pred,observed=infer(path,raw['test'],sigma,seed,dev)
            target=raw['test']['no_j']
            candidates={'unchanged':observed,'cnn':pred,'savgol':savgol_filter(observed,window,2,axis=1)}
            masks={'all':np.ones_like(target,dtype=bool),
                   'expected_intensity_above_0.05':np.exp(-raw['test']['with_j']*raw['test']['t_relax'])>=.05}
            record={'sigma_rel':sigma,'noise_seed':seed,'metrics':{}}
            for label,mask in masks.items():
                record['metrics'][label]={name:metrics(v,target,mask) for name,v in candidates.items()}
            if sigma:
                rng=np.random.default_rng(seed)
                ref=1+sigma*rng.standard_normal((len(target),1))
                intensity=np.exp(-raw['test']['with_j']*raw['test']['t_relax'])+sigma*rng.standard_normal(target.shape)
                record['nonpositive_intensity_fraction']=float((intensity<=0).mean())
                record['nonpositive_reference_fraction']=float((ref<=0).mean())
            results.append(record)
            print(json.dumps(record),flush=True)
    output={'status':'Retrospective evaluation; not a fresh locked test. Repeats vary noise only, not training seed.',
            'savgol_window_selected_on_validation':window,'validation_smoothing_mse':scores,
            'test_profiles_after_exclusions':len(raw['test']['with_j']),
            'checkpoint_sha256':{str(v):hashlib.sha256(v.read_bytes()).hexdigest() for v in checkpoints.values()},
            'results':results}
    a.out.write_text(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
