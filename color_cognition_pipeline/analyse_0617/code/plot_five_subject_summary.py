"""Static, publication-style figures from the saved five-subject 0617 outputs."""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path('/home/lirui/liulab_project/ieeg/Project_colorieeg_2026')
ANALYSE = ROOT / 'color_cognition_pipeline' / 'analyse_0617'
OLD = ANALYSE / 'doc'
NEW = ANALYSE / 'run_5subjects_original' / 'doc'
OUT = ANALYSE / 'result' / 'final_report' / 'figures'
SUBJECTS = ['test001', 'test002', 'test003', 'test005', 'test006']
COLORS = {'test001':'#E69F00', 'test002':'#009E73', 'test003':'#0072B2',
          'test005':'#CC79A7', 'test006':'#D55E00'}
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False,
                     'axes.spines.right':False, 'axes.linewidth':.8, 'savefig.bbox':'tight'})


def save(fig, stem):
    fig.savefig(OUT / f'{stem}.png', dpi=600)
    fig.savefig(OUT / f'{stem}.pdf')
    plt.close(fig)


def read_combined(name):
    a, b = pd.read_csv(OLD/name), pd.read_csv(NEW/name)
    ac = [c for c in a if re.fullmatch(r'test\d+_Acc', c)]
    bc = [c for c in b if re.fullmatch(r'test\d+_Acc', c)]
    d = a[['Time_ms', *ac]].merge(b[['Time_ms', *bc]], on='Time_ms', validate='one_to_one')
    d['Group_Mean_Acc'] = d[ac+bc].mean(axis=1)
    return d


def plot_counts(summary):
    erp = summary.groupby('Subject').ERP_Selected.sum().reindex(SUBJECTS)
    hg = summary.groupby('Subject').HG_Selected.sum().reindex(SUBJECTS)
    x = np.arange(len(SUBJECTS)); w=.35
    fig, ax = plt.subplots(figsize=(7.2,4.2))
    ax.bar(x-w/2, erp, w, label='ERP', color='#0072B2')
    ax.bar(x+w/2, hg, w, label='HG', color='#D55E00')
    ax.set_xticks(x, SUBJECTS); ax.set_ylabel('Number of selected electrodes'); ax.set_title('Color-selective electrodes across subjects', weight='bold')
    ax.legend(frameon=False, ncol=2); ax.grid(axis='y', alpha=.2); save(fig, 'fig1_color_select_electrode_counts')


def plot_mni(summary):
    d=summary.copy(); d['ERP_Selected']=d.ERP_Selected.astype(bool); d['HG_Selected']=d.HG_Selected.astype(bool)
    d['Evidence']=np.select([d.ERP_Selected&d.HG_Selected,d.ERP_Selected,d.HG_Selected],['ERP + HG','ERP only','HG only'],default='Not selected')
    fig=plt.figure(figsize=(8.2,6.5)); ax=fig.add_subplot(111,projection='3d')
    markers={'Not selected':'o','ERP only':'^','HG only':'s','ERP + HG':'*'}
    for s in SUBJECTS:
        q=d[d.Subject==s]
        for evidence,m in markers.items():
            z=q[q.Evidence==evidence]
            if len(z): ax.scatter(z.MNI_X,z.MNI_Y,z.MNI_Z,s=22 if m!='*' else 65, marker=m, color=COLORS[s], alpha=.72, edgecolor='white', linewidth=.25)
    ax.set_xlabel('MNI X (mm)'); ax.set_ylabel('MNI Y (mm)'); ax.set_zlabel('MNI Z (mm)'); ax.set_title('Unified MNI distribution of all localized electrodes', weight='bold', pad=15)
    handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor=COLORS[s],label=s,markersize=7) for s in SUBJECTS]
    handles += [Line2D([0],[0],marker=m,color='k',linestyle='None',label=e,markersize=7) for e,m in markers.items()]
    ax.legend(handles=handles, fontsize=7, loc='upper left', bbox_to_anchor=(-.08,1.02), frameon=False)
    save(fig,'fig2_unified_mni_electrode_distribution')


def plot_decoding():
    fig, axes=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True)
    for ax,(modality,scheme) in zip(axes.flat,[(m,s) for m in ('ERP','HG') for s in ('strategy4','union','memorysig')]):
        d=read_combined(f'decoding_data_{modality.lower()}_{scheme}.csv')
        for s in SUBJECTS:
            c=f'{s}_Acc'
            if c in d: ax.plot(d.Time_ms,d[c],color=COLORS[s],lw=.8,alpha=.42)
        ax.plot(d.Time_ms,d.Group_Mean_Acc,color='#111111',lw=2.2,label='5-subject mean')
        ax.axhline(.5,color='#777777',lw=.8,ls=':'); ax.axvline(0,color='#777777',lw=.6)
        ax.set_xlim(-200,800); ax.set_ylim(.35,.8); ax.set_title(f'{modality} · {scheme}',weight='bold'); ax.grid(alpha=.15)
    axes[1,1].set_xlabel('Time relative to stimulus (ms)'); axes[0,0].set_ylabel('Accuracy'); axes[1,0].set_ylabel('Accuracy')
    subject_handles=[Line2D([0],[0],color=COLORS[s],lw=2,label=s) for s in SUBJECTS]
    subject_handles.append(Line2D([0],[0],color='#111111',lw=2.5,label='5-subject mean'))
    axes[0,0].legend(handles=subject_handles,frameon=False,fontsize=7.5,loc='upper right',ncol=2)
    fig.suptitle('Memory color decoding: unified five-subject results',weight='bold',y=1.01)
    save(fig,'fig3_five_subject_decoding_curves')


def plot_estp():
    frames=[]
    for root in (OLD,NEW):
        for mod in ('erp','hg'):
            p=root/f'select_channel_memory_decoding_estp_{mod}.csv'
            if p.exists(): frames.append(pd.read_csv(p).assign(Modality=mod.upper()))
    d=pd.concat(frames,ignore_index=True); d=d[d.Subject.isin(SUBJECTS)]
    fig,ax=plt.subplots(figsize=(7.2,4.3)); positions=np.arange(2)
    vals=[d.loc[d.Modality==m,'ESTP'].dropna() for m in ('ERP','HG')]
    bp=ax.boxplot(vals,positions=positions,widths=.45,patch_artist=True,showfliers=False)
    for patch,color in zip(bp['boxes'],['#0072B2','#D55E00']): patch.set_facecolor(color); patch.set_alpha(.65)
    for i,m in enumerate(('ERP','HG')):
        q=d[d.Modality==m]
        for s in SUBJECTS:
            z=q[q.Subject==s].ESTP.dropna(); ax.scatter(np.full(len(z),i)+np.linspace(-.12,.12,len(z)) if len(z) else [],z,color=COLORS[s],s=18,alpha=.8)
    ax.set_xticks(positions,['ERP','HG']); ax.set_ylabel('ESTP (ms)'); ax.set_title('Memory decoding onset latency',weight='bold'); ax.grid(axis='y',alpha=.2)
    save(fig,'fig4_memory_decoding_estp')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    summary=pd.read_csv(NEW/'select_channel_summary.csv'); summary=summary[summary.Subject.isin(SUBJECTS)].copy()
    summary['ERP_Selected']=summary.ERP_Selected.astype(bool); summary['HG_Selected']=summary.HG_Selected.astype(bool)
    plot_counts(summary); plot_mni(summary); plot_decoding(); plot_estp()
    print(f'Wrote figures to {OUT}')


if __name__=='__main__': main()
