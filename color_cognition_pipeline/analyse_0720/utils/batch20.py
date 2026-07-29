"""End-to-end subject functions for the independent 20-mm batch."""
from __future__ import annotations

import ast
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

import config
from .epochs import baseline_subtract, baseline_zscore, load_epochs, save_epochs
from .preprocessing import add_laplacian_neighbors, contact_laplacian
from .qc import detect_bad_epochs
from .decoding import make_time_windows, fixed_cv, decode_accuracy, decode_permutation, temporal_generalization
from .stats import cluster_permutation_1d
from .plotting_science import decoding_curve, electrode_distance_plot, set_science_style, save_figure


def _xyz(value):
    try:
        xyz=np.asarray(ast.literal_eval(str(value)),float)
        return xyz if xyz.shape==(3,) else np.full(3,np.nan)
    except Exception:
        return np.full(3,np.nan)


def load_coverage(subject):
    path=config.LOCATION_FILES.get(subject)
    if path is None or not Path(path).exists(): return None, pd.DataFrame()
    table=pd.read_excel(path) if Path(path).suffix.lower()==".xlsx" else pd.read_csv(path,sep="\t")
    channel=next(c for c in table if c.lower() in ("channel","channelname","name"))
    table=table.copy(); table["channel"]=table[channel].astype(str); table["coord"]=table["MNI"].map(_xyz)
    for side,target in config.FMRI_TARGETS.items():
        table[f"d_{side}_mm"]=table.coord.map(lambda x:np.linalg.norm(x-np.asarray(target)) if np.isfinite(x).all() else np.nan)
    table["target_side"]=np.where(table.d_left_mm<table.d_right_mm,"left","right")
    table["target_distance_mm"]=table[["d_left_mm","d_right_mm"]].min(axis=1)
    candidates=table.loc[table.target_distance_mm<=config.PRIMARY_TARGET_RADIUS_MM].drop_duplicates("channel").copy()
    return table,candidates


def write_coverage(subject):
    out=config.batch_subject_dir(subject)/"coverage"; out.mkdir(parents=True,exist_ok=True)
    table,candidates=load_coverage(subject)
    if table is None:
        pd.DataFrame([{"subject":subject,"status":"missing_localization","n_candidates":0}]).to_csv(out/"status.csv",index=False)
        return candidates
    table.drop(columns="coord").to_csv(out/"electrode_distance_all.csv",index=False)
    candidates.drop(columns="coord").to_csv(out/"candidate_electrodes_20mm.csv",index=False)
    status="eligible" if len(candidates) else "no_20mm_coverage"
    pd.DataFrame([{"subject":subject,"status":status,"n_candidates":len(candidates)}]).to_csv(out/"status.csv",index=False)
    if len(candidates): electrode_distance_plot(candidates,subject,out/"candidate_distance")
    return candidates


def _raw_and_reference(subject, set_path, target_labels, erp=False):
    raw=mne.io.read_raw_eeglab(set_path,preload=True,verbose=False)
    labels=[x for x in target_labels if x in raw.ch_names]
    raw.pick(add_laplacian_neighbors(raw.ch_names,labels))
    raw.notch_filter(config.LINE_NOISE_HZ,verbose=False)
    if erp:
        raw.resample(config.SFREQ_ERP,verbose=False); raw.filter(1.,30.,verbose=False)
    return raw,contact_laplacian(raw,labels,config.BAD_CHANNELS.get(subject,()))


def _epoch(ref, data_raw, baseline, zscore=False):
    events,event_id=mne.events_from_annotations(ref,verbose=False)
    selected={name:code for name,code in event_id.items() if name.startswith("Trigger-In:")}
    info=mne.create_info(ref.ch_names,ref.info["sfreq"],"seeg")
    signal=mne.io.RawArray(data_raw,info,verbose=False); signal.set_annotations(ref.annotations.copy())
    epochs=mne.Epochs(signal,events,event_id=selected,tmin=config.EPOCH_TMIN_S,tmax=config.EPOCH_TMAX_S,baseline=None,preload=True,reject_by_annotation=False,verbose=False)
    values=epochs.get_data(); times=epochs.times*1000
    values=(baseline_zscore if zscore else baseline_subtract)(values,times,baseline)
    inverse={code:name for name,code in selected.items()}; triggers=[inverse[int(code)] for code in epochs.events[:,2]]
    return values,times,triggers,epochs.ch_names


def preprocess_subject(subject, include_hg=True):
    _,cand=load_coverage(subject)
    if cand.empty: return {"subject":subject,"status":"not_eligible"}
    labels=cand.channel.tolist(); idir=config.BATCH_INTERMEDIATE_ROOT/subject/"preprocessing"; qdir=config.batch_subject_dir(subject)/"preprocessing"
    summary=[]
    for task,run_name in config.RUNS.items():
        set_path=config.subject_raw_dir(subject)/f"{run_name}.set"
        raw,ref=_raw_and_reference(subject,set_path,labels,erp=True)
        erp,times,triggers,names=_epoch(ref,ref.get_data(),tuple(v*1000 for v in config.ERP_BASELINE_S)); erp*=1e6
        keep,metrics=detect_bad_epochs(erp,config.BAD_EPOCH_ROBUST_Z,config.BAD_EPOCH_CHANNEL_FRACTION)
        save_epochs(idir/f"task{task}_erp_clean.npz",erp[keep],times,np.asarray(triggers)[keep],names,{"subject":subject,"radius_mm":20,"bad_channels":config.BAD_CHANNELS.get(subject,()),"rejected":int((~keep).sum())})
        pd.DataFrame({"epoch":np.arange(len(keep)),"trigger":triggers,"keep":keep,**metrics}).to_csv(qdir/f"task{task}_epoch_qc.csv",index=False)
        set_science_style(); fig,ax=plt.subplots(figsize=(3.5,2.1),constrained_layout=True); ax.plot(metrics["bad_channel_fraction"],color="#687078",lw=.7); ax.scatter(np.flatnonzero(~keep),metrics["bad_channel_fraction"][~keep],s=8,color="#C94C4C"); ax.axhline(config.BAD_EPOCH_CHANNEL_FRACTION,color="black",ls=(0,(3,2)),lw=.7); ax.set(xlabel="Epoch",ylabel="Flagged-channel fraction",title=f"{subject} · Task {task} QC"); ax.spines[["top","right"]].set_visible(False); save_figure(fig,qdir/f"task{task}_epoch_qc")
        if include_hg:
            del raw,ref
            raw,ref=_raw_and_reference(subject,set_path,labels,erp=False)
            accum=np.zeros(ref.get_data().shape,dtype=np.float32)
            for lo,hi in config.HG_BANDS_HZ:
                band=ref.copy().filter(lo,hi,verbose=False).apply_hilbert(envelope=True,verbose=False)
                accum+=(np.log10(np.maximum(np.square(band.get_data()),np.finfo(float).eps))/len(config.HG_BANDS_HZ)).astype(np.float32)
            hg,htimes,htriggers,hnames=_epoch(ref,accum,tuple(v*1000 for v in config.HG_BASELINE_S),zscore=True)
            if list(htriggers)!=list(triggers): raise RuntimeError(f"{subject} task{task}: ERP/HG trigger mismatch")
            save_epochs(idir/f"task{task}_hg_clean.npz",hg[keep],htimes,np.asarray(htriggers)[keep],hnames,{"subject":subject,"radius_mm":20,"bands_hz":config.HG_BANDS_HZ,"rejected":int((~keep).sum())})
        summary.append({"subject":subject,"task":task,"epochs_before":len(keep),"epochs_rejected":int((~keep).sum()),"epochs_kept":int(keep.sum()),"n_channels":len(names)})
    pd.DataFrame(summary).to_csv(qdir/"epoch_qc_summary.csv",index=False)
    return {"subject":subject,"status":"complete","n_channels":summary[0]["n_channels"]}


def _candidate_picks(ep):
    return np.arange(len(ep["channel_names"]),dtype=int)


def _save_decode(subject,modality,analysis,centers,observed,null,p_cluster):
    out=config.batch_subject_dir(subject)/analysis
    table=pd.DataFrame({"time_ms":centers,"accuracy":observed,"p_uncorrected":(1+(null>=observed).sum(0))/(1+len(null)),"p_cluster":p_cluster})
    table.to_csv(out/f"{modality}_decoding.csv",index=False); np.savez_compressed(out/f"{modality}_null.npz",null=null.astype("float32"),time_ms=centers.astype("float32"))
    decoding_curve(centers,observed,f"{subject} · {analysis} · {modality.upper()}",out/f"{modality}_decoding",null,p_cluster)
    return table


def analyse_subject(subject, modalities=("erp","hg")):
    idir=config.BATCH_INTERMEDIATE_ROOT/subject/"preprocessing"; outputs=[]
    for modality in modalities:
        t1=load_epochs(idir/f"task1_{modality}_clean.npz",prefer_clean=False); trig=np.char.replace(t1["triggers"].astype(str),"Trigger-In:","")
        color=np.isin(trig,["11","21","31","41"]); gray=np.isin(trig,["12","22","32","42"]); win=(t1["times_ms"]>=100)&(t1["times_ms"]<=400)
        rows=[]
        for ci,ch in enumerate(t1["channel_names"]):
            a=t1["data"][color,ci][:,win].mean(1); b=t1["data"][gray,ci][:,win].mean(1); effect=a.mean()-b.mean(); se=np.sqrt(a.var(ddof=1)/len(a)+b.var(ddof=1)/len(b))
            rows.append({"subject":subject,"modality":modality,"channel":ch,"effect":effect,"se":se,"n_color":len(a),"n_gray":len(b)})
        pd.DataFrame(rows).to_csv(config.batch_subject_dir(subject)/"task1"/f"{modality}_channel_effects.csv",index=False)

        t2=load_epochs(idir/f"task2_{modality}_clean.npz",prefer_clean=False); g2=np.char.replace(t2["triggers"].astype(str),"Trigger-In:",""); m2=np.isin(g2,["123","133","103","113"]); selected=g2[m2]; y2=np.isin(selected,["103","113"]).astype(int)
        windows,centers=make_time_windows(t2["times_ms"],config.WINDOW_MS,config.STEP_MS); pair=[(("123","103"),("133","113")),(("133","113"),("123","103")),(("123","113"),("133","103")),(("133","103"),("123","113"))]; cv=[(np.flatnonzero(np.isin(selected,a)),np.flatnonzero(np.isin(selected,b))) for a,b in pair]
        observed=decode_accuracy(t2["data"][m2],y2,windows,cv); null=decode_permutation(t2["data"][m2],y2,windows,cv,config.N_PERMUTATIONS,config.RANDOM_SEED,config.N_JOBS); pc=cluster_permutation_1d(observed,null); outputs.append(_save_decode(subject,modality,"task2",centers,observed,null,pc))

        t3=load_epochs(idir/f"task3_{modality}_clean.npz",prefer_clean=False); g3=np.char.replace(t3["triggers"].astype(str),"Trigger-In:",""); m3=np.isin(g3,["51","54"]); y3=(g3[m3]=="54").astype(int); cv3=fixed_cv(y3,config.N_SPLITS,config.RANDOM_SEED)
        observed=decode_accuracy(t3["data"][m3],y3,windows,cv3); null=decode_permutation(t3["data"][m3],y3,windows,cv3,config.N_PERMUTATIONS,config.RANDOM_SEED+3,config.N_JOBS); pc=cluster_permutation_1d(observed,null); outputs.append(_save_decode(subject,modality,"task3",centers,observed,null,pc))

        mapping={"101":("cabbage",1),"102":("cabbage",0),"111":("kiwi",1),"112":("kiwi",0),"121":("strawberry",1),"122":("strawberry",0),"131":("watermelon",1),"132":("watermelon",0)}; mt=np.isin(g2,list(mapping)); codes=g2[mt]; fruits=np.asarray([mapping[x][0] for x in codes]); yt=np.asarray([mapping[x][1] for x in codes]); cvt=[(np.flatnonzero(fruits!=f),np.flatnonzero(fruits==f)) for f in np.unique(fruits)]
        observed=decode_accuracy(t2["data"][mt],yt,windows,cvt); null=decode_permutation(t2["data"][mt],yt,windows,cvt,config.N_PERMUTATIONS,config.RANDOM_SEED+7,config.N_JOBS); pc=cluster_permutation_1d(observed,null); outputs.append(_save_decode(subject,modality,"cross_task",centers,observed,null,pc))
    return outputs
