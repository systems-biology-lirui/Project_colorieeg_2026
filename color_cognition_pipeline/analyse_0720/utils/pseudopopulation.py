"""Condition-matched pseudo-population construction and decoding."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, parallel_config

import config
from .epochs import load_epochs
from .decoding import make_time_windows, decode_accuracy, decode_permutation, fixed_cv
from .stats import cluster_permutation_1d
from .plotting_science import decoding_curve
from .wholebrain import location_table


TASK_SPECS={
    "task2_memory":{"task":2,"codes":["123","133","103","113"],"positive":["103","113"]},
    "task3_red_green":{"task":3,"codes":["51","54"],"positive":["54"]},
    "true_false":{"task":2,"codes":["101","102","111","112","121","122","131","132"],"positive":["101","111","121","131"]},
}


def _subjects_with_features(modality,electrode_set="union"):
    result=[]
    for subject in config.WHOLE_SUBJECTS:
        path=config.all_subject_dir(subject)/"color_select"/"color_select_electrodes.csv"
        if not path.exists(): continue
        d=pd.read_csv(path)
        if electrode_set=="intersection": d=d[d.color_select_evidence=="ERP_and_HG"]
        if len(d): result.append((subject,d))
    return result


def construct(analysis,modality,seed,electrode_set="union"):
    spec=TASK_SPECS[analysis]; subject_entries=_subjects_with_features(modality,electrode_set); loaded=[]; feature_rows=[]; offset=0
    for subject,sel in subject_entries:
        ep=load_epochs(config.ALL_INTERMEDIATE_ROOT/subject/"preprocessing"/f"task{spec['task']}_{modality}_clean.npz",False); names=list(ep["channel_names"]); channels=[c for c in sel.channel.astype(str) if c in names]
        if not channels: continue
        picks=[names.index(c) for c in channels]; trig=np.char.replace(ep["triggers"].astype(str),"Trigger-In:",""); loaded.append((subject,ep,trig,picks,channels))
        loc=location_table(subject).set_index("channel")
        for ch in channels:
            row=sel.set_index("channel").loc[ch]; feature_rows.append({"feature_index":offset,"subject":subject,"channel":ch,"modality":modality,"mni_x":loc.loc[ch,"mni_x"] if ch in loc.index else np.nan,"mni_y":loc.loc[ch,"mni_y"] if ch in loc.index else np.nan,"mni_z":loc.loc[ch,"mni_z"] if ch in loc.index else np.nan,"hemisphere":loc.loc[ch,"hemisphere"] if ch in loc.index else np.nan,"atlas_region":loc.loc[ch,"atlas_region"] if ch in loc.index else np.nan,"color_select_evidence":row.color_select_evidence}); offset+=1
    if not loaded: raise RuntimeError(f"No selected features for {analysis}/{modality}")
    groups=min(sum(trig==code)//config.PSEUDO_TRIAL_SIZE for _,_,trig,_,_ in loaded for code in spec["codes"]); rng=np.random.default_rng(seed); blocks=[]; labels=[]; codes_out=[]; manifest=[]
    for code in spec["codes"]:
        subject_blocks=[]
        for subject,ep,trig,picks,channels in loaded:
            idx=rng.permutation(np.flatnonzero(trig==code))[:groups*config.PSEUDO_TRIAL_SIZE].reshape(groups,config.PSEUDO_TRIAL_SIZE)
            subject_blocks.append(np.stack([ep["data"][g][:,picks,:].mean(0) for g in idx]))
            for gi,g in enumerate(idx): manifest.append({"subject":subject,"condition":code,"pseudo_trial":gi,"raw_trial_indices":json.dumps(g.tolist())})
        blocks.append(np.concatenate(subject_blocks,axis=1)); labels.extend([int(code in spec["positive"])]*groups); codes_out.extend([code]*groups)
    return np.concatenate(blocks),np.asarray(labels),np.asarray(codes_out),loaded[0][1]["times_ms"],pd.DataFrame(feature_rows),pd.DataFrame(manifest)


def _cv_for(analysis,codes,y,seed):
    if analysis=="task2_memory":
        pairs=[(("123","103"),("133","113")),(("133","113"),("123","103")),(("123","113"),("133","103")),(("133","103"),("123","113"))]
        return [(np.flatnonzero(np.isin(codes,a)),np.flatnonzero(np.isin(codes,b))) for a,b in pairs]
    if analysis=="true_false":
        fruit={"101":"cabbage","102":"cabbage","111":"kiwi","112":"kiwi","121":"strawberry","122":"strawberry","131":"watermelon","132":"watermelon"}; f=np.asarray([fruit[x] for x in codes]); return [(np.flatnonzero(f!=z),np.flatnonzero(f==z)) for z in np.unique(f)]
    return fixed_cv(y,min(config.N_SPLITS,np.bincount(y).min()),seed)


def decode_virtual(analysis,modality,electrode_set="union"):
    def one(rep):
        data,y,codes,times,features,manifest=construct(analysis,modality,config.RANDOM_SEED+1000*rep,electrode_set)
        windows,centers=make_time_windows(times,config.WINDOW_MS,config.STEP_MS)
        cv=_cv_for(analysis,codes,y,config.RANDOM_SEED)
        return decode_accuracy(data,y,windows,cv),(data,y,codes,windows,centers,cv,features,manifest)
    with parallel_config(backend="loky", n_jobs=config.N_JOBS, inner_max_num_threads=1):
        results=Parallel(n_jobs=config.N_JOBS, prefer="processes")(
            delayed(one)(rep) for rep in range(config.PSEUDO_REPETITIONS)
        )
    curves=np.asarray([r[0] for r in results]); first=results[0][1]
    data,y,codes,windows,centers,cv,features,manifest=first; null=decode_permutation(data,y,windows,cv,config.N_PERMUTATIONS,config.RANDOM_SEED+99,config.N_JOBS); curves=np.asarray(curves); observed=curves.mean(0); pc=cluster_permutation_1d(observed,null); p=(1+(null>=observed).sum(0))/(1+len(null)); out=config.ALL_RESULT_ROOT/"virtual_subject"; stem=f"{analysis}_{modality}_{electrode_set}"; pd.DataFrame({"time_ms":centers,"mean_accuracy":observed,"ci_low":np.quantile(curves,.025,axis=0),"ci_high":np.quantile(curves,.975,axis=0),"p_uncorrected":p,"p_cluster":pc,"pseudo_repetitions":config.PSEUDO_REPETITIONS,"permutation_count":config.N_PERMUTATIONS}).to_csv(out/"decoding"/f"{stem}.csv",index=False); features.to_csv(out/"feature_manifest.csv",index=False); manifest.to_csv(out/"pseudo_trial_manifest.csv",index=False); np.savez_compressed(out/"decoding"/f"{stem}.npz",curves=curves.astype("float32"),null=null.astype("float32"),time_ms=centers); decoding_curve(centers,observed,f"Virtual subject · {analysis} · {modality.upper()}",out/"figures"/stem,null,pc); plt.close("all"); return {"analysis":analysis,"modality":modality,"n_features":len(features),"n_subjects":features.subject.nunique(),"n_pseudo_trials":len(y)}


def spatial_groups():
    path=config.ALL_RESULT_ROOT/"virtual_subject"/"feature_manifest.csv"; d=pd.read_csv(path); d=d.dropna(subset=["mni_x","mni_y","mni_z","atlas_region","hemisphere"]); d["spatial_group"]=d.hemisphere.astype(str)+" | "+d.atlas_region.astype(str); summary=d.groupby("spatial_group").agg(n_electrodes=("feature_index","size"),n_subjects=("subject","nunique"),subjects=("subject",lambda x:";".join(sorted(set(x))))).reset_index(); summary["eligible"]=(summary.n_electrodes>=3)&(summary.n_subjects>=2); summary.to_csv(config.ALL_RESULT_ROOT/"group"/"spatial_groups"/"spatial_group_summary.csv",index=False); d.to_csv(config.ALL_RESULT_ROOT/"group"/"spatial_groups"/"feature_spatial_groups.csv",index=False); return summary
