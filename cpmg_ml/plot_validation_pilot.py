"""Plot pilot fitting bias, explicitly one exchange group and one noise draw."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    source=Path('docs/ml_results/downstream_fit_pilot.json')
    result=json.loads(source.read_text())
    rows=[r for r in result['results'] if r['run'].endswith('start0')]
    labels=['Clean J=0','Noisy J=0','Noisy coupled','CNN corrected']
    plt.rcParams.update({'font.family':'Arial','font.size':16})
    fig,axes=plt.subplots(1,2,figsize=(12,5.5))
    for ax,key,title in zip(axes,['KEX_AB','PB'],[r'$k_{ex}$',r'$p_B$']):
        values=[100*(r['fitted']['GLOBAL'][key]/result['truth'][key]-1) for r in rows]
        ax.bar(np.arange(4),values,color=['#777777','#7196b3','#d28b32','#2e8b72'])
        ax.axhline(0,color='black',lw=1)
        ax.set_xticks(np.arange(4),labels,rotation=25,ha='right')
        ax.set_ylabel('Signed error (%)')
        ax.set_title(title)
        ax.set_ylim(-22,27)
        for i,v in enumerate(values):
            ax.text(i,v+(1 if v>=0 else -1),f'{v:+.1f}%',ha='center',va='bottom' if v>=0 else 'top',fontsize=16)
    fig.suptitle('Grouped ChemEx fitting pilot: 600 + 800 MHz',fontsize=19)
    fig.text(.5,.015,'One exchange group; 3 shifts; one SNR 100 noise draw. Two starts agree.\nKnown R1 fixed; R2 fitted. Pilot evidence, not population statistics.',ha='center',fontsize=16)
    fig.tight_layout(rect=(0,.16,1,.94))
    out=Path('docs/group_meeting_figs/downstream_fit_pilot_20260909.png')
    fig.savefig(out,dpi=180)
    print(out)


if __name__=='__main__':
    main()
