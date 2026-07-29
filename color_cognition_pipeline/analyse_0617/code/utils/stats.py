import numpy as np

def get_mean_sem(epochs_data):
    """
    计算 Mean & SEM
    epochs_data: [trials, time] 或 [trials, channels, time]
    """
    mean_val = np.mean(epochs_data, axis=0)
    sem_val = np.std(epochs_data, axis=0) / np.sqrt(epochs_data.shape[0])
    return mean_val, sem_val

def has_continuous_sig(sig_bool, consec_pts):
    """
    检查布尔数组中是否存在至少连续 consec_pts 个 True
    """
    count = 0
    for val in sig_bool:
        if val:
            count += 1
            if count >= consec_pts:
                return True
        else:
            count = 0
    return False

def binomial_test_p(k, n, p_chance=0.5):
    """
    单尾二项检验，测试准确率是否显著大于随机机会
    """
    try:
        from scipy.stats import binomtest
        res = binomtest(k, n, p=p_chance, alternative='greater')
        return res.pvalue
    except ImportError:
        from scipy.stats import binom_test
        return binom_test(k, n, p=p_chance, alternative='greater')

def find_significant_windows(p_vals, time_ms, p_thresh=0.05, min_duration=20):
    """
    查找连续显著的时间窗口 (p < p_thresh, 且持续时长不小于 min_duration ms)
    """
    dt = time_ms[1] - time_ms[0]
    # 需要连续多少个点
    consec_pts = int(np.ceil(min_duration / dt))
    
    sig_mask = p_vals < p_thresh
    windows = []
    in_window = False
    start_idx = None
    
    for idx, is_sig in enumerate(sig_mask):
        if is_sig and not in_window:
            in_window = True
            start_idx = idx
        elif not is_sig and in_window:
            in_window = False
            end_idx = idx - 1
            duration = (end_idx - start_idx + 1) * dt
            if duration >= min_duration:
                windows.append((time_ms[start_idx], time_ms[end_idx]))
    if in_window:
        end_idx = len(sig_mask) - 1
        duration = (end_idx - start_idx + 1) * dt
        if duration >= min_duration:
            windows.append((time_ms[start_idx], time_ms[end_idx]))
            
    return windows
