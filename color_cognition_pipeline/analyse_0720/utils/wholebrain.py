"""Whole-channel preprocessing, functional selection, and decoding."""
from __future__ import annotations

import ast, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from scipy.stats import false_discovery_control, ttest_ind

import config
from .epochs import baseline_subtract, baseline_zscore, load_epochs, save_epochs
from .preprocessing import contact_laplacian
from .qc import detect_bad_epochs
from .decoding import make_time_windows, fixed_cv, decode_accuracy, decode_permutation, temporal_generalization
from .stats import cluster_permutation_1d
from .plotting_science import decoding_curve, set_science_style, save_figure, COLORS


def location_table(subject):
    path=config.LOCATION_FILES.get(subject)
    if path is None or not Path(path).exists(): return pd.DataFrame(columns=["channel","mni_x","mni_y","mni_z","hemisphere","atlas_region"])
    d=pd.read_excel(path) if Path(path).suffix.lower()==".xlsx" else pd.read_csv(path,sep="\t")
    ch=next(c for c in d if c.lower() in ("channel","channelname","name")); atlas=next((c for c in ("DKT","AAL3 (MNI-linear)","AAL3 (MNI-segment)") if c in d),None)
    def xyz(v):
        try:
            x=np.asarray(ast.literal_eval(str(v)),float); return x if x.shape==(3,) else np.full(3,np.nan)
        except Exception:return np.full(3,np.nan)
    coords=d["MNI"].map(xyz); out=pd.DataFrame({"channel":d[ch].astype(str),"mni_x":[x[0] for x in coords],"mni_y":[x[1] for x in coords],"mni_z":[x[2] for x in coords]})
    out["hemisphere"]=np.where(out.mni_x<0,"left",np.where(out.mni_x>0,"right","midline")); out["atlas_region"]=d[atlas].astype(str) if atlas else "unknown"
    return out.drop_duplicates("channel")


def _events_epoch(ref, values, baseline_ms, zscore=False):
    events,event_id=mne.events_from_annotations(ref,verbose=False); selected={n:c for n,c in event_id.items() if n.startswith("Trigger-In:")}
    signal=mne.io.RawArray(values,mne.create_info(ref.ch_names,ref.info["sfreq"],"seeg"),verbose=False); signal.set_annotations(ref.annotations.copy())
    ep=mne.Epochs(signal,events,event_id=selected,tmin=config.EPOCH_TMIN_S,tmax=config.EPOCH_TMAX_S,baseline=None,preload=True,reject_by_annotation=False,verbose=False)
    data=ep.get_data(); times=ep.times*1000; data=(baseline_zscore if zscore else baseline_subtract)(data,times,baseline_ms)
    inv={c:n for n,c in selected.items()}; triggers=np.asarray([inv[int(c)] for c in ep.events[:,2]])
    return data,times,triggers,ep.ch_names


def _prepare_raw(subject,set_path,erp):
    raw=mne.io.read_raw_eeglab(set_path,preload=True,verbose=False); raw.notch_filter(config.LINE_NOISE_HZ,verbose=False)
    if erp: raw.resample(config.SFREQ_ERP,verbose=False); raw.filter(1.,30.,verbose=False)
    else: raw.resample(config.SFREQ_ERP,verbose=False)
    return raw,contact_laplacian(raw,raw.ch_names,config.BAD_CHANNELS.get(subject,()))


def preprocess_subject(subject,include_hg=True):
    idir=config.ALL_INTERMEDIATE_ROOT/subject/"preprocessing"; qdir=config.all_subject_dir(subject)/"preprocessing"; rows=[]
    for task,run_name in config.RUNS.items():
        cached_erp=idir/f"task{task}_erp_clean.npz"; cached_hg=idir/f"task{task}_hg_clean.npz"; cached_qc=qdir/f"task{task}_epoch_qc.csv"
        if cached_erp.exists() and (not include_hg or cached_hg.exists()) and cached_qc.exists():
            ep_cached=load_epochs(cached_erp,prefer_clean=False); qc_cached=pd.read_csv(cached_qc)
            rows.append({"subject":subject,"task":task,"epochs_before":len(qc_cached),"epochs_rejected":int((~qc_cached.keep.astype(bool)).sum()),"epochs_kept":int(qc_cached.keep.astype(bool).sum()),"n_channels":len(ep_cached["channel_names"])}); continue
        source=config.subject_raw_dir(subject)/f"{run_name}.set"; raw,ref=_prepare_raw(subject,source,True)
        erp,times,triggers,names=_events_epoch(ref,ref.get_data(),tuple(v*1000 for v in config.ERP_BASELINE_S)); erp*=1e6
        keep,metrics=detect_bad_epochs(erp,config.BAD_EPOCH_ROBUST_Z,config.BAD_EPOCH_CHANNEL_FRACTION)
        save_epochs(idir/f"task{task}_erp_clean.npz",erp[keep],times,triggers[keep],names,{"subject":subject,"scope":"all_valid_laplacian","bad_channels":config.BAD_CHANNELS.get(subject,()),"rejected":int((~keep).sum())})
        pd.DataFrame({"epoch":np.arange(len(keep)),"trigger":triggers,"keep":keep,**metrics}).to_csv(qdir/f"task{task}_epoch_qc.csv",index=False)
        set_science_style(); fig,ax=plt.subplots(figsize=(3.5,2.1),constrained_layout=True); ax.plot(metrics["bad_channel_fraction"],color=COLORS["gray"],lw=.7); ax.scatter(np.flatnonzero(~keep),metrics["bad_channel_fraction"][~keep],s=8,color=COLORS["red"]); ax.axhline(config.BAD_EPOCH_CHANNEL_FRACTION,color="black",ls="--",lw=.7); ax.set(xlabel="Epoch",ylabel="Flagged-channel fraction",title=f"{subject} · Task {task} QC"); ax.spines[["top","right"]].set_visible(False); save_figure(fig,qdir/f"task{task}_epoch_qc"); plt.close(fig)
        if include_hg:
            del raw,ref; raw,ref=_prepare_raw(subject,source,False); accum=np.zeros(ref.get_data().shape,dtype=np.float32)
            for lo,hi in config.HG_BANDS_HZ:
                band=ref.copy().filter(lo,hi,verbose=False).apply_hilbert(envelope=True,verbose=False); accum+=(np.log10(np.maximum(np.square(band.get_data()),np.finfo(float).eps))/len(config.HG_BANDS_HZ)).astype(np.float32)
            hg,htimes,htriggers,hnames=_events_epoch(ref,accum,tuple(v*1000 for v in config.HG_BASELINE_S),True)
            if not np.array_equal(triggers,htriggers): raise RuntimeError(f"{subject} task{task}: ERP/HG trigger mismatch")
            save_epochs(idir/f"task{task}_hg_clean.npz",hg[keep],htimes,htriggers[keep],hnames,{"subject":subject,"scope":"all_valid_laplacian","bands_hz":config.HG_BANDS_HZ,"rejected":int((~keep).sum())})
        rows.append({"subject":subject,"task":task,"epochs_before":len(keep),"epochs_rejected":int((~keep).sum()),"epochs_kept":int(keep.sum()),"n_channels":len(names)})
    pd.DataFrame(rows).to_csv(qdir/"epoch_qc_summary.csv",index=False); return rows


def _permutation_effect(values,y,n_perm,seed):
    y=np.asarray(y,bool); observed=values[y].mean(0)-values[~y].mean(0); rng=np.random.default_rng(seed); null=np.empty((n_perm,values.shape[1]),np.float32)
    for i in range(n_perm):
        yp=rng.permutation(y); null[i]=values[yp].mean(0)-values[~yp].mean(0)
    p1=(1+(null>=observed).sum(0))/(1+n_perm); p2=(1+(np.abs(null)>=np.abs(observed)).sum(0))/(1+n_perm)
    return observed,p1,p2


def screen_subject(subject,n_perm=None):
    n_perm=n_perm or config.SCREEN_N_PERMUTATIONS; idir=config.ALL_INTERMEDIATE_ROOT/subject/"preprocessing"; out=config.all_subject_dir(subject)/"color_select"; loc=location_table(subject); tables=[]
    for mi,modality in enumerate(("erp","hg")):
        ep=load_epochs(idir/f"task1_{modality}_clean.npz",prefer_clean=False); trig=np.char.replace(ep["triggers"].astype(str),"Trigger-In:",""); color=np.isin(trig,["11","21","31","41"]); gray=np.isin(trig,["12","22","32","42"]); use=color|gray; y=color[use]; win=(ep["times_ms"]>=100)&(ep["times_ms"]<=400); values=ep["data"][use][:,:,win].mean(2)
        effect,p1,p2=_permutation_effect(values,y,n_perm,config.RANDOM_SEED+mi); sem=np.sqrt(values[y].var(0,ddof=1)/y.sum()+values[~y].var(0,ddof=1)/(~y).sum()); q1=false_discovery_control(p1,method="bh"); q2=false_discovery_control(p2,method="bh")
        p_param=ttest_ind(values[y],values[~y],axis=0,equal_var=False,alternative="greater").pvalue; p_param_two=ttest_ind(values[y],values[~y],axis=0,equal_var=False,alternative="two-sided").pvalue; q_param=false_discovery_control(p_param,method="bh"); q_param_two=false_discovery_control(p_param_two,method="bh")
        d=pd.DataFrame({"subject":subject,"channel":ep["channel_names"].astype(str),"modality":modality,"color_mean":values[y].mean(0),"gray_mean":values[~y].mean(0),"effect":effect,"ci_low":effect-1.96*sem,"ci_high":effect+1.96*sem,"p_color_gt_gray":p1,"p_two_sided":p2,"q_color_gt_gray":q1,"q_two_sided":q2,"p_parametric_color_gt_gray":p_param,"p_parametric_two_sided":p_param_two,"q_parametric_color_gt_gray":q_param,"q_parametric_two_sided":q_param_two,"screening_permutations":n_perm}); d["color_select"]=(d.effect>0)&(d.p_color_gt_gray<.05); d["gray_preferring"]=(d.effect<0)&(d.p_two_sided<.05); d=d.merge(loc,on="channel",how="left"); d.to_csv(out/f"{modality}_screening.csv",index=False); tables.append(d)
    erp=tables[0].set_index("channel"); hg=tables[1].set_index("channel"); channels=sorted(set(erp.index)|set(hg.index)); selected=[]
    for ch in channels:
        es=bool(erp.loc[ch,"color_select"]) if ch in erp.index else False; hs=bool(hg.loc[ch,"color_select"]) if ch in hg.index else False
        if not (es or hs): continue
        evidence="ERP_and_HG" if es and hs else ("ERP_only" if es else "HG_only"); base=(erp.loc[ch] if ch in erp.index else hg.loc[ch]).to_dict(); selected.append({"subject":subject,"channel":ch,"color_select_evidence":evidence,**{k:base.get(k) for k in ("mni_x","mni_y","mni_z","hemisphere","atlas_region")}})
    sel=pd.DataFrame(selected,columns=["subject","channel","color_select_evidence","mni_x","mni_y","mni_z","hemisphere","atlas_region"]); sel.to_csv(out/"color_select_electrodes.csv",index=False); pd.concat(tables).to_csv(out/"channel_statistics.csv",index=False); return sel


def plot_distribution(subject):
    out=config.all_subject_dir(subject)/"color_select"; loc=location_table(subject); sel=pd.read_csv(out/"color_select_electrodes.csv"); merged=loc.merge(sel[["channel","color_select_evidence"]],on="channel",how="left"); set_science_style()
    colors={"ERP_only":COLORS["blue"],"HG_only":COLORS["orange"],"ERP_and_HG":COLORS["red"]}; fig=plt.figure(figsize=(6.8,2.5),constrained_layout=True)
    axes=[fig.add_subplot(1,3,1),fig.add_subplot(1,3,2),fig.add_subplot(1,3,3,projection="3d")]; pairs=[("mni_x","mni_y"),("mni_y","mni_z")]
    for ax,(x,y) in zip(axes[:2],pairs):
        ax.scatter(merged[x],merged[y],s=8,c="#D4D7DA",alpha=.6)
        for ev,c in colors.items():
            d=merged[merged.color_select_evidence==ev]; ax.scatter(d[x],d[y],s=25,c=c,label=ev)
            for _,r in d.iterrows(): ax.text(r[x],r[y],r.channel,fontsize=5)
        ax.set(xlabel=x.replace("mni_","").upper()+" (mm)",ylabel=y.replace("mni_","").upper()+" (mm)"); ax.spines[["top","right"]].set_visible(False)
    ax=axes[2]; ax.scatter(merged.mni_x,merged.mni_y,merged.mni_z,s=7,c="#D4D7DA",alpha=.4)
    for ev,c in colors.items():
        d=merged[merged.color_select_evidence==ev]; ax.scatter(d.mni_x,d.mni_y,d.mni_z,s=24,c=c,label=ev)
    ax.set(xlabel="X",ylabel="Y",zlabel="Z",title=subject); axes[0].legend(fontsize=6); return save_figure(fig,out/"electrode_distribution")


def _selected_picks(subject,ep,electrode_set="union"):
    sel=pd.read_csv(config.all_subject_dir(subject)/"color_select"/"color_select_electrodes.csv")
    if electrode_set=="intersection": sel=sel[sel.color_select_evidence=="ERP_and_HG"]
    names=list(ep["channel_names"]); channels=[c for c in sel.channel.astype(str) if c in names]; return [names.index(c) for c in channels],channels


def _decode_save(subject,modality,analysis,electrode_set,ep,data,y,cv,windows,centers,channels,seed):
    observed=decode_accuracy(data,y,windows,cv); null=decode_permutation(data,y,windows,cv,config.N_PERMUTATIONS,seed,config.N_JOBS); pc=cluster_permutation_1d(observed,null); p=(1+(null>=observed).sum(0))/(1+len(null)); out=config.all_subject_dir(subject)/"decoding"; stem=f"{analysis}_{modality}_{electrode_set}"; table=pd.DataFrame({"subject":subject,"modality":modality,"analysis":analysis,"electrode_set":electrode_set,"electrode_count":len(channels),"electrode_names":";".join(channels),"permutation_count":config.N_PERMUTATIONS,"time_ms":centers,"accuracy":observed,"p_uncorrected":p,"p_cluster":pc}); table.to_csv(out/f"{stem}.csv",index=False); np.savez_compressed(out/f"{stem}_null.npz",null=null.astype("float32"),time_ms=centers); decoding_curve(centers,observed,f"{subject} · {analysis} · {modality.upper()}",out/stem,null,pc); plt.close("all"); return table


def decode_subject(subject,electrode_sets=("union","intersection")):
    idir=config.ALL_INTERMEDIATE_ROOT/subject/"preprocessing"; status=[]
    for modality in ("erp","hg"):
        t2=load_epochs(idir/f"task2_{modality}_clean.npz",False); t3=load_epochs(idir/f"task3_{modality}_clean.npz",False); g2=np.char.replace(t2["triggers"].astype(str),"Trigger-In:",""); g3=np.char.replace(t3["triggers"].astype(str),"Trigger-In:",""); windows,centers=make_time_windows(t2["times_ms"],config.WINDOW_MS,config.STEP_MS)
        for electrode_set in electrode_sets:
            out_dec=config.all_subject_dir(subject)/"decoding"; stemset=f"_{modality}_{electrode_set}"
            cached=[out_dec/f"task2_memory{stemset}.csv",out_dec/f"task3_red_green{stemset}.csv",out_dec/f"true_false{stemset}.csv",out_dec/f"cross_task{stemset}.npz"]
            if all(p.exists() for p in cached):
                status.append({"subject":subject,"modality":modality,"electrode_set":electrode_set,"status":"cached"}); continue
            p2,chs=_selected_picks(subject,t2,electrode_set); p3,chs3=_selected_picks(subject,t3,electrode_set); common=[c for c in chs if c in chs3]
            if not common: status.append({"subject":subject,"modality":modality,"electrode_set":electrode_set,"status":"no_color_select_electrode"}); continue
            p2=[list(t2["channel_names"]).index(c) for c in common]; p3=[list(t3["channel_names"]).index(c) for c in common]
            m=np.isin(g2,["123","133","103","113"]); codes=g2[m]; y=np.isin(codes,["103","113"]).astype(int); pairs=[(("123","103"),("133","113")),(("133","113"),("123","103")),(("123","113"),("133","103")),(("133","103"),("123","113"))]; cv=[(np.flatnonzero(np.isin(codes,a)),np.flatnonzero(np.isin(codes,b))) for a,b in pairs]; _decode_save(subject,modality,"task2_memory",electrode_set,t2,t2["data"][m][:,p2],y,cv,windows,centers,common,config.RANDOM_SEED)
            m3=np.isin(g3,["51","54"]); y3=(g3[m3]=="54").astype(int); cv3=fixed_cv(y3,config.N_SPLITS,config.RANDOM_SEED); _decode_save(subject,modality,"task3_red_green",electrode_set,t3,t3["data"][m3][:,p3],y3,cv3,windows,centers,common,config.RANDOM_SEED+3)
            mapping={"101":("cabbage",1),"102":("cabbage",0),"111":("kiwi",1),"112":("kiwi",0),"121":("strawberry",1),"122":("strawberry",0),"131":("watermelon",1),"132":("watermelon",0)}; mt=np.isin(g2,list(mapping)); cc=g2[mt]; fruits=np.asarray([mapping[x][0] for x in cc]); yt=np.asarray([mapping[x][1] for x in cc]); cvt=[(np.flatnonzero(fruits!=f),np.flatnonzero(fruits==f)) for f in np.unique(fruits)]; _decode_save(subject,modality,"true_false",electrode_set,t2,t2["data"][mt][:,p2],yt,cvt,windows,centers,common,config.RANDOM_SEED+7)
            x2=t2["data"][m][:,p2]; x3=t3["data"][m3][:,p3]; tg32=temporal_generalization(x3,x2,y3,y,windows,config.RANDOM_SEED); tg23=temporal_generalization(x2,x3,y,y3,windows,config.RANDOM_SEED); np.savez_compressed(config.all_subject_dir(subject)/"decoding"/f"cross_task_{modality}_{electrode_set}.npz",task3_to_task2=tg32.astype("float32"),task2_to_task3=tg23.astype("float32"),time_ms=centers,channels=np.asarray(common)); status.append({"subject":subject,"modality":modality,"electrode_set":electrode_set,"status":"complete","n_channels":len(common)})
    pd.DataFrame(status).to_csv(config.all_subject_dir(subject)/"decoding"/"status.csv",index=False); return status
