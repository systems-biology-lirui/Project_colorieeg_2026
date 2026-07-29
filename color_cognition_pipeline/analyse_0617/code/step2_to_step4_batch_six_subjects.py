"""Six-subject continuation of the analyse_0617 select-channel pipeline.

Runs the downstream analyses that are independent of the legacy MATLAB file
layout: Task3 color selectivity, Task2 memory red/green significance, and
time-resolved decoding. Results are stored separately from the original
three-subject outputs and are summarized with subjects as statistical units.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal, ranksums

ROOT=Path('/home/lirui/liulab_project/ieeg/Project_colorieeg_2026')
PIPE=ROOT/'color_cognition_pipeline/analyse_0720'; sys.path.insert(0,str(PIPE))
import config
from utils.epochs import load_epochs
from utils.decoding import make_time_windows, decode_accuracy, decode_permutation, fixed_cv
from utils.stats import cluster_permutation_1d

OUT=ROOT/'color_cognition_pipeline/analyse_0617/result/six_subject_downstream'
SUBJECTS=('test001','test002','test003','test004','test005','test006')
FAST_PERM=100


def unified_selection():
    """Combine legacy 001--003 tables with functional 004--006 tables."""
    old=pd.read_csv(ROOT/'color_cognition_pipeline/analyse_0617/doc/select_channel_summary.csv')
    rows=[]
    for _,r in old.iterrows():
        evidence=[]
        if bool(r.get('ERP_Selected',False)): evidence.append('ERP')
        if bool(r.get('HG_Selected',False)): evidence.append('HG')
        rows.append({'subject':r.Subject,'channel':r.Electrode,'mni_x':r.get('MNI_X',np.nan),'mni_y':r.get('MNI_Y',np.nan),'mni_z':r.get('MNI_Z',np.nan),'atlas_region':r.get('AAL3',np.nan),'selection_source':'analyse_0617_legacy','evidence':';'.join(evidence)})
    for s in ('test004','test005','test006'):
        p=ROOT/'color_cognition_pipeline/analyse_0617/result/select_channel_batch_004_006/subjects'/s/'select_channel_summary.csv'
        d=pd.read_csv(p)
        for _,r in d.iterrows(): rows.append({'subject':s,'channel':r.channel,'mni_x':r.get('mni_x',np.nan),'mni_y':r.get('mni_y',np.nan),'mni_z':r.get('mni_z',np.nan),'atlas_region':r.get('atlas_region',np.nan),'selection_source':'analyse_0617_reimplemented','evidence':r.get('evidence','')})
    d=pd.DataFrame(rows).drop_duplicates(['subject','channel'])
    d.to_csv(OUT/'tables/select_channel_summary_six_subjects.csv',index=False)
    return d


def _ep(subject,task,modality):
    return load_epochs(config.ALL_INTERMEDIATE_ROOT/subject/'preprocessing'/f'task{task}_{modality}_clean.npz',False)


def _pick(ep, channels):
    names=list(ep['channel_names']); use=[c for c in channels if c in names]
    return [names.index(c) for c in use],use


def _clean(x): return x[~np.isnan(x).any(axis=tuple(range(1,x.ndim)))]


def color_selectivity(selection):
    rows=[]
    for s in SUBJECTS:
        channels=selection.loc[selection.subject==s,'channel'].astype(str).tolist()
        for modality in ('erp','hg'):
            ep=_ep(s,3,modality); trig=np.char.replace(ep['triggers'].astype(str),'Trigger-In:',''); picks,use=_pick(ep,channels); win=(ep['times_ms']>=50)&(ep['times_ms']<=400)
            for ch,pi in zip(use,picks):
                vals=[_clean(ep['data'][trig==str(code),pi,:])[:,win].mean(1) for code in (51,52,53,54)]
                if min(map(len,vals))==0: continue
                h,p=kruskal(*vals); rg,zp=ranksums(vals[0],vals[3]); yb,yp=ranksums(vals[1],vals[2])
                rows.append({'subject':s,'channel':ch,'modality':modality,'overall_csi':h,'overall_p':p,'red_green_z':abs(rg),'red_green_p':zp,'yellow_blue_z':abs(yb),'yellow_blue_p':yp})
    d=pd.DataFrame(rows); d.to_csv(OUT/'tables/task3_color_selectivity_six_subjects.csv',index=False); return d


def memory_significance(selection):
    rows=[]
    red=('121','122','123','131','132','133'); green=('101','102','103','111','112','113')
    for s in SUBJECTS:
        channels=selection.loc[selection.subject==s,'channel'].astype(str).tolist()
        for modality in ('erp','hg'):
            ep=_ep(s,2,modality); trig=np.char.replace(ep['triggers'].astype(str),'Trigger-In:',''); picks,use=_pick(ep,channels); win=(ep['times_ms']>=100)&(ep['times_ms']<=400); dt=float(np.median(np.diff(ep['times_ms'])))
            for ch,pi in zip(use,picks):
                r=_clean(ep['data'][np.isin(trig,red),pi,:]); g=_clean(ep['data'][np.isin(trig,green),pi,:])
                if not len(r) or not len(g): continue
                pmean=ranksums(r[:,win].mean(1),g[:,win].mean(1)).pvalue; pt=np.array([ranksums(r[:,i],g[:,i]).pvalue for i in np.flatnonzero(win)]); run=best=0
                for v in pt<.05: run=run+1 if v else 0; best=max(best,run)
                dur=best*dt; rows.append({'subject':s,'channel':ch,'modality':modality,'mean_p':pmean,'max_cont_duration_ms':dur,'sig_category':'Both_Sig' if pmean<.05 and dur>=50 else ('Mean_Sig_Only' if pmean<.05 else ('Cont_Sig_Only' if dur>=50 else 'Non_Sig'))})
    d=pd.DataFrame(rows); d.to_csv(OUT/'tables/task2_memory_significance_six_subjects.csv',index=False); return d


def decode(selection, memory):
    rows=[]; outfig=OUT/'figures'; outfig.mkdir(parents=True,exist_ok=True)
    for s in SUBJECTS:
        base=selection.loc[selection.subject==s,'channel'].astype(str).tolist()
        for modality in ('erp','hg'):
            ep=_ep(s,3,modality); trig=np.char.replace(ep['triggers'].astype(str),'Trigger-In:',''); picks,use=_pick(ep,base); m=np.isin(trig,('51','54')); y=(trig[m]=='54').astype(int)
            if not use or y.sum()==0 or y.sum()==len(y): continue
            data=ep['data'][m][:,picks,:]; windows,centers=make_time_windows(ep['times_ms'],config.WINDOW_MS,config.STEP_MS); cv=fixed_cv(y,min(5,np.bincount(y).min()),config.RANDOM_SEED); acc=decode_accuracy(data,y,windows,cv); null=decode_permutation(data,y,windows,cv,FAST_PERM,config.RANDOM_SEED+7,config.N_JOBS); p=(1+(null>=acc).sum(0))/(1+len(null)); pc=cluster_permutation_1d(acc,null)
            for t,a,pu,pp in zip(centers,acc,p,pc): rows.append({'subject':s,'modality':modality,'analysis':'task3_red_green','n_channels':len(use),'time_ms':t,'accuracy':a,'p_uncorrected':pu,'p_cluster':pp,'permutations':FAST_PERM})
            fig,ax=plt.subplots(figsize=(6,3)); ax.plot(centers,acc,label=f'{s} {modality.upper()}'); ax.axhline(.5,color='gray',ls='--',lw=.8); ax.set(xlabel='Time (ms)',ylabel='Accuracy',ylim=(.35,1)); ax.spines[['top','right']].set_visible(False); ax.legend(); fig.savefig(outfig/f'{s}_{modality}_task3_red_green.png',dpi=300,bbox_inches='tight'); fig.savefig(outfig/f'{s}_{modality}_task3_red_green.pdf',bbox_inches='tight'); plt.close(fig)
    d=pd.DataFrame(rows); d.to_csv(OUT/'tables/task3_red_green_decoding_six_subjects.csv',index=False); return d


def summarize(d):
    if d.empty:return d
    g=d.groupby(['analysis','modality','time_ms'],as_index=False).agg(n_subjects=('subject','nunique'),mean_accuracy=('accuracy','mean'),sd_accuracy=('accuracy','std'))
    g['sem_accuracy']=g.sd_accuracy/np.sqrt(g.n_subjects); g['ci_low']=g.mean_accuracy-1.96*g.sem_accuracy; g['ci_high']=g.mean_accuracy+1.96*g.sem_accuracy; g.to_csv(OUT/'tables/group_accuracy_mean_six_subjects.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,3),sharey=True)
    for ax,mod in zip(axes,('erp','hg')):
        x=g[g.modality==mod]; ax.plot(x.time_ms,x.mean_accuracy); ax.fill_between(x.time_ms,x.ci_low,x.ci_high,alpha=.2); ax.axhline(.5,color='gray',ls='--'); ax.set(title=mod.upper(),xlabel='Time (ms)',ylabel='Mean accuracy'); ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/'figures/group_accuracy_mean_six_subjects.png',dpi=600,bbox_inches='tight'); fig.savefig(OUT/'figures/group_accuracy_mean_six_subjects.pdf',bbox_inches='tight'); plt.close(fig)


def main():
    for p in (OUT/'tables',OUT/'figures'): p.mkdir(parents=True,exist_ok=True)
    selection=unified_selection(); csi=color_selectivity(selection); mem=memory_significance(selection); dec=decode(selection,mem); summarize(dec)
    pd.DataFrame([{'subjects':';'.join(SUBJECTS),'n_select_channel':len(selection),'n_csi_rows':len(csi),'n_memory_rows':len(mem),'n_decoding_rows':len(dec),'permutations':FAST_PERM}]).to_csv(OUT/'tables/run_summary.csv',index=False)

if __name__=='__main__': main()
