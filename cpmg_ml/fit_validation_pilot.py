"""Small downstream ChemEx pilot: one exchange group, three shifts, two fields.

Known R1 is fixed and R2 is fitted: an optimistic controlled benchmark, not
an experimental validation. Clean J=0 is the fitting-model control. Each arm
uses two starting points. Corrected intensities are pseudo-data; their supplied
errors are fitting weights, not calibrated DNN uncertainties.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tomllib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .simulator import CPMGSimulator, b0_mhz_to_tesla
from .baselines import add_intensity_noise
from .evaluate_audited import infer


def fit(job):
    folder, exe = job
    command=[exe,'fit','-e','600.toml','800.toml','-p','parameters.toml','-m','method.toml','-o','Output','--plot','nothing']
    env=dict(os.environ, MPLCONFIGDIR=str(folder.parent/'mplcache'), OPENBLAS_NUM_THREADS='1')
    try:
        result=subprocess.run(command,cwd=folder,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
        (folder/'fit.log').write_text(result.stdout)
        paths=list((folder/'Output').rglob('fitted.toml'))
        return {'run':folder.name,'returncode':result.returncode,
                'fitted':tomllib.loads(paths[0].read_text()) if paths else None}
    except subprocess.TimeoutExpired as exc:
        (folder/'fit.log').write_text(str(exc.stdout))
        return {'run':folder.name,'status':'timeout','fitted':None}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--work-dir',type=Path,required=True)
    p.add_argument('--chemex',default='/opt/anaconda3/envs/NMR/bin/chemex')
    p.add_argument('--out',type=Path,default=Path('docs/ml_results/downstream_fit_pilot.json'))
    p.add_argument('--kex',type=float,default=600.)
    p.add_argument('--pb',type=float,default=.05)
    p.add_argument('--noise-seed',type=int,default=87654)
    a=p.parse_args()
    work=a.work_dir.resolve()
    work.mkdir(parents=True,exist_ok=False)
    sim=CPMGSimulator()
    wj,nj,nu,glob,r1=[],[],[],[],{}
    for field in (600.,800.):
        for dw in (1.,3.,5.):
            params=dict(B0=b0_mhz_to_tesla(field),B1_N=5555.,tau_m=5e-9,tau_e=50e-12,S2=.85,r_IS=1.02e-10,r_eff=1.86e-10,csa_N=-160e-6,theta_N=np.radians(22.),J_IS=92.,k_ex=a.kex,p_B=a.pb,dw_N=dw)
            x,y=sim.simulate_dej_pair(params,ncyc_range=np.arange(1,41),t_relax=.04,sequence='chemex')
            wj.append(x.r2_eff);nj.append(y.r2_eff);nu.append(x.nu_cp);glob.append([field,5555.,92.])
            matrix=sim.build_16x16(params['B0'],5e-9,50e-12,.85,1.02e-10,1.86e-10,-160e-6,np.radians(22.),0.,0.)
            r1[field]=float(-matrix[6,6].real)
    raw={'with_j':np.array(wj),'no_j':np.array(nj),'nu':np.array(nu),'glob':np.array(glob),'t_relax':.04}
    corrected,coupled=infer(Path('runs/b500k_noise01/cnn.pt'),raw,.01,a.noise_seed,'cpu')
    noisy_j0=add_intensity_noise(raw['no_j'],.04,.01,np.random.default_rng(a.noise_seed))
    arms={'clean_j0':raw['no_j'],'noisy_j0':noisy_j0,'noisy_coupled':coupled,'cnn_corrected':corrected}
    jobs=[]
    for arm,curves in arms.items():
        for start,(pb,kex) in enumerate(((.02,300.),(.10,1200.))):
            folder=work/f'{arm}_start{start}'
            folder.mkdir()
            for f,field in enumerate((600,800)):
                profiles=[]
                for i in range(3):
                    filename=f'{field}_{i+1}.out'
                    rows=np.column_stack((np.arange(41),np.r_[1.,np.exp(-curves[3*f+i]*.04)],np.full(41,.01)))
                    np.savetxt(folder/filename,rows,fmt=['%d','%.12g','%.12g'])
                    profiles.append(f'{i+1}N = "{filename}"')
                (folder/f'{field}.toml').write_text(f'[experiment]\nname="cpmg_15n_ip"\ntime_t2=0.04\ncarrier=0.0\npw90={1/(4*5555)}\ntime_equil=0.002\n[conditions]\nh_larmor_frq={float(field)}\n[data]\npath="."\nerror="file"\n[data.profiles]\n'+'\n'.join(profiles)+'\n')
            rates='\n'.join(f'["R1_A, B0->{field:.1f}MHZ"]\n1N={r1[field]}\n2N={r1[field]}\n3N={r1[field]}' for field in (600.,800.))
            (folder/'parameters.toml').write_text(f'[GLOBAL]\nPB={pb}\nKEX_AB={kex}\nDW_AB=2.0\nR2_A=7.0\nR1_A=2.0\n[CS_A]\n1N=0.0\n2N=0.0\n3N=0.0\n'+rates+'\n')
            (folder/'method.toml').write_text('[STEP1]\nINCLUDE="ALL"\nFIT=["PB","KEX_AB","DW_AB","R2_A"]\nFIX=["CS_A","R1_A"]\nCONSTRAINTS=["[R2_B]=[R2_A]","[R1_B]=[R1_A]"]\n')
            jobs.append((folder,a.chemex))
    with ThreadPoolExecutor(2) as pool:
        results=list(pool.map(fit,jobs))
    output={'scope':__doc__,'truth':{'KEX_AB':a.kex,'PB':a.pb,'DW_AB':[1.,3.,5.]},'noise_seed':a.noise_seed,
            'work_dir':str(work),'results':results}
    a.out.write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
