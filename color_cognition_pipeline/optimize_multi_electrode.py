import os
import sys
import numpy as np
import scipy.io as sio
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

base_dir = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026'
pipeline_dir = os.path.join(base_dir, 'color_cognition_pipeline')

# test002 High Gamma 有 15 个电极，最适合用于多电极过拟合分析
subject = 'test002'
elecs = ['A2', 'A3', 'A4', 'A8', 'A9', 'B6', 'C9', 'C10', 'F4', 'F6', 'G5', 'G6', 'G7', 'H5', 'H8']

# 数据路径
t2_path = os.path.join(pipeline_dir, 'feature', 'subband_60_150', subject, 'task2_hg_subband.mat')
mat_data = sio.loadmat(t2_path, squeeze_me=True, struct_as_record=False)
epoch = mat_data['epoch']
time_ms = epoch.time_ms if hasattr(epoch, 'time_ms') else np.linspace(-500, 998, epoch.data_cell[0].shape[-1])

# 获取 Triggers 对应的数据
trigs_red = ['Trigger-In:123', 'Trigger-In:133'] # 隐含红
trigs_green = ['Trigger-In:103', 'Trigger-In:113'] # 隐含绿

def extract_trials(trigs):
    all_trigs = [str(t) for t in epoch.trigger] if hasattr(epoch, 'trigger') else [str(t) for t in epoch.eventtype]
    ch_names = [str(ch.labels) for ch in epoch.ch]
    ch_idx = [ch_names.index(e) for e in elecs if e in ch_names]
    
    idx_list = [all_trigs.index(t) for t in trigs if t in all_trigs]
    c_data = np.concatenate([epoch.data_cell[idx][:, ch_idx, :] for idx in idx_list], axis=0)
    # clean NaNs
    c_data = c_data[~np.isnan(c_data).any(axis=(1,2))]
    return c_data

data_red = extract_trials(trigs_red) # [N_trials, N_ch, N_time]
data_green = extract_trials(trigs_green)

print(f"Loaded {subject} HG data: Red trials = {data_red.shape}, Green trials = {data_green.shape}")

# 基线期 (-500ms 到 0ms) 对应的 indices
baseline_mask = time_ms < 0
baseline_indices = np.where(baseline_mask)[0]

# ==================== 1. 数据预处理方法对比 ====================
# 方法 A: 原始方法 (无 trial-wise 基线校正)
# 方法 B: Trial-wise 基线校正 (每个 trial 每个通道减去其基线期均值)
# 方法 C: Trial-wise 基线 Z-score 校正 (减去基线期均值并除以基线期标准差)

# 实施方法 B 和 C
def apply_baseline_correction(data, mode='subtraction'):
    data_corr = data.copy()
    for i in range(data.shape[0]): # 每一个 trial
        for j in range(data.shape[1]): # 每一个通道
            bl_vals = data[i, j, baseline_indices]
            mean_bl = np.mean(bl_vals)
            std_bl = np.std(bl_vals) if np.std(bl_vals) > 0 else 1.0
            if mode == 'subtraction':
                data_corr[i, j, :] = data[i, j, :] - mean_bl
            elif mode == 'zscore':
                data_corr[i, j, :] = (data[i, j, :] - mean_bl) / std_bl
    return data_corr

data_red_sub = apply_baseline_correction(data_red, 'subtraction')
data_green_sub = apply_baseline_correction(data_green, 'subtraction')

data_red_z = apply_baseline_correction(data_red, 'zscore')
data_green_z = apply_baseline_correction(data_green, 'zscore')

# 划分 Train-Test 配对，同之前 4-fold 配对逻辑 (2个水果对2个水果)
# 4种配对划分:
# pair 1: train (r1, g1), test (r2, g2)
# r1, r2, g1, g2 的划分：数据提取时是合并了的。
# 为保证跟之前 decode_strategy4_electrodes.py 的 4-fold CV 配对逻辑一致：
# 我们把数据重新分为 r1, r2, g1, g2.
r1_trig, r2_trig = ['Trigger-In:123'], ['Trigger-In:133']
g1_trig, g2_trig = ['Trigger-In:103'], ['Trigger-In:113']

d_r1 = extract_trials(r1_trig)
d_r2 = extract_trials(r2_trig)
d_g1 = extract_trials(g1_trig)
d_g2 = extract_trials(g2_trig)

# ==================== 解码函数 ====================
def evaluate_pipeline(prep_func, clf_maker, name):
    # 应用预处理
    r1, r2, g1, g2 = map(prep_func, [d_r1, d_r2, d_g1, d_g2])
    
    pairs = [
        (r1, g1, r2, g2),
        (r1, g2, r2, g1),
        (r2, g1, r1, g2),
        (r2, g2, r1, g1)
    ]
    
    n_time = time_ms.shape[0]
    all_acc = np.zeros((4, n_time))
    
    for fold, (train_r, train_g, test_r, test_g) in enumerate(pairs):
        for t in range(n_time):
            X_tr = np.vstack([train_r[:, :, t], train_g[:, :, t]])
            y_tr = np.hstack([np.zeros(train_r.shape[0]), np.ones(train_g.shape[0])])
            X_te = np.vstack([test_r[:, :, t], test_g[:, :, t]])
            y_te = np.hstack([np.zeros(test_r.shape[0]), np.ones(test_g.shape[0])])
            
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            
            clf = clf_maker()
            clf.fit(X_tr, y_tr)
            all_acc[fold, t] = accuracy_score(y_te, clf.predict(X_te))
            
    mean_acc = np.mean(all_acc, axis=0)
    
    # 计算基线期性能指标
    bl_accs = mean_acc[baseline_indices]
    bl_mean = np.mean(bl_accs)
    bl_std = np.std(bl_accs)
    
    # 刺激后最大准确率
    post_accs = mean_acc[~baseline_mask]
    post_peak = np.max(post_accs)
    post_peak_t = time_ms[~baseline_mask][np.argmax(post_accs)]
    
    print(f"[{name}] BaseMean: {bl_mean:.4f}, BaseStd: {bl_std:.4f}, Peak: {post_peak:.4f} at {post_peak_t:.1f}ms")
    return mean_acc

# 预处理函数包装
def identity(x): return x.copy()
def subtract_bl(x): return apply_baseline_correction(x, 'subtraction')
def zscore_bl(x): return apply_baseline_correction(x, 'zscore')

# 分类器构建函数
def make_svc_linear_c1(): return SVC(kernel='linear', C=1.0)
def make_svc_linear_c01(): return SVC(kernel='linear', C=0.1)
def make_logreg_l2(): return LogisticRegression(penalty='l2', C=0.5, solver='liblinear')
def make_logreg_l1(): return LogisticRegression(penalty='l1', C=0.3, solver='liblinear')

# 测试不同管线
pipelines = {
    "1. Baseline (Raw + Linear SVM C=1.0)": (identity, make_svc_linear_c1),
    "2. Raw + Logistic L2 (C=0.5)": (identity, make_logreg_l2),
    "3. Raw + Logistic L1 (C=0.3)": (identity, make_logreg_l1),
    "4. Baseline Subtraction + Linear SVM C=1.0": (subtract_bl, make_svc_linear_c1),
    "5. Baseline Subtraction + Linear SVM C=0.1": (subtract_bl, make_svc_linear_c01),
    "6. Baseline Subtraction + Logistic L2 (C=0.5)": (subtract_bl, make_logreg_l2),
    "7. Baseline Subtraction + Logistic L1 (C=0.3)": (subtract_bl, make_logreg_l1),
    "8. Baseline Z-Score + Linear SVM C=1.0": (zscore_bl, make_svc_linear_c1),
    "9. Baseline Z-Score + Logistic L1 (C=0.3)": (zscore_bl, make_logreg_l1),
}

# 绘图设置
plt.figure(figsize=(12, 8))
results = {}

for name, (prep, clf_m) in pipelines.items():
    acc = evaluate_pipeline(prep, clf_m, name)
    results[name] = acc
    # 平滑后画图显示
    smoothed = np.convolve(acc, np.ones(5)/5, mode='same')
    plt.plot(time_ms, smoothed, label=name, alpha=0.8)

plt.axhline(0.5, color='gray', linestyle='--')
plt.axvline(0, color='black', linestyle='-')
plt.ylim([0.38, 0.72])
plt.xlabel("Time (ms)")
plt.ylabel("Accuracy")
plt.title(f"Comparison of Decoding Optimization Pipelines ({subject} HG, {len(elecs)} Chs)")
plt.legend(loc='lower left', fontsize=8)
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(pipeline_dir, 'images', f'optimization_comparison_{subject}.png'), dpi=300)
plt.close()
print(f"Comparison plot saved at {os.path.join(pipeline_dir, 'images', f'optimization_comparison_{subject}.png')}")
