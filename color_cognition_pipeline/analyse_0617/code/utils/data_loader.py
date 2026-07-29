import os
import numpy as np
from pymatreader import read_mat

def get_epoch_data(mat_path, trigs_to_extract, elecs, is_hg=False, baseline_correct=True):
    """
    统一的 SEEG ERP/HG 数据读取与 Trial-wise 基线减法
    """
    if not os.path.exists(mat_path):
        return None, None
    try:
        mat = read_mat(mat_path)
        epoch = mat['epoch']
        ch_names = [str(x).strip() for x in epoch['ch']['labels']]
        
        # 确定时间轴
        if 'time_ms' in epoch:
            time_ms = epoch['time_ms']
        else:
            time_len = epoch['data'].shape[-1] if 'data' in epoch else epoch['data_cell'][0].shape[-1]
            time_ms = np.linspace(-500, 998, time_len)
            
        all_trigs = list(epoch['trigger'])
        
        # 寻找电极索引
        ch_indices = [ch_names.index(e) for e in elecs if e in ch_names]
        if not ch_indices:
            return None, None
            
        # 寻找 Trigger 索引
        idx_list = [all_trigs.index(t) for t in trigs_to_extract if t in all_trigs]
        if not idx_list:
            return None, None
            
        data_list = []
        for idx in idx_list:
            if is_hg:
                # HG: (Rep, Ch, Time)
                trial_data = epoch['data_cell'][idx][:, ch_indices, :]
            else:
                # ERP: (Cond, Rep, Ch, Time)
                # 使用整数切片防止维度变化
                trial_data = epoch['data'][idx, :, :, :]
                trial_data = trial_data[:, ch_indices, :]
            data_list.append(trial_data)
            
        merged_data = np.concatenate(data_list, axis=0)
        
        # 是否进行 Trial-wise 基线校正
        if baseline_correct:
            baseline_mask = time_ms < 0
            baseline_indices = np.where(baseline_mask)[0]
            if len(baseline_indices) > 0:
                mean_bl = np.mean(merged_data[:, :, baseline_indices], axis=2, keepdims=True)
                merged_data = merged_data - mean_bl
                
        return merged_data, time_ms
    except Exception as e:
        print(f"  [ERROR] get_epoch_data failed for {mat_path}: {e}")
        return None, None

def resample_10ms_bins(data, time_ms, start_time=-100.0, end_time=700.0):
    """
    对 [-100, 700]ms 的时间轴按照每 10ms 宽度进行分箱均值重采样
    """
    t_indices = np.where((time_ms >= start_time) & (time_ms <= end_time))[0]
    bin_size = 5  # 500Hz下，5个采样点 = 10ms
    n_bins = len(t_indices) // bin_size
    
    # data shape: (Rep, n_ch, Time)
    resampled_data = np.zeros((data.shape[0], data.shape[1], n_bins))
    resampled_time = np.zeros(n_bins)
    
    for b in range(n_bins):
        bin_idx = t_indices[b*bin_size : (b+1)*bin_size]
        resampled_time[b] = np.mean(time_ms[bin_idx])
        resampled_data[:, :, b] = np.mean(data[:, :, bin_idx], axis=2)
        
    return resampled_data, resampled_time
