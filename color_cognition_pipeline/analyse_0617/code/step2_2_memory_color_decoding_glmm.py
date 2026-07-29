import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.stats as stats
from pymatreader import read_mat
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import statsmodels.genmod.bayes_mixed_glm as bmg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import multiprocessing
import os
import time
import warnings

# 忽略绘图与模型计算的一些不重要警告
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置
base_dir = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026'
pipeline_dir = os.path.join(base_dir, 'color_cognition_pipeline')
analyse_dir = os.path.join(pipeline_dir, 'analyse_0617')
feature_dir = os.path.join(analyse_dir, 'feature')
doc_dir = os.path.join(analyse_dir, 'doc')
result_dir = os.path.join(analyse_dir, 'result')
out_dir = os.path.join(result_dir, 'select_channel', 'decoding')
os.makedirs(out_dir, exist_ok=True)

subjects = ['test001', 'test002', 'test003']
# 交叉配对训练测试的水果 Triggers (Task 2)
# r1/r2: 红色记忆；g1/g2: 绿色记忆
r1_trigs = ['Trigger-In:123'] # 灰色草莓
r2_trigs = ['Trigger-In:133'] # 灰色西瓜
g1_trigs = ['Trigger-In:103'] # 灰色卷心菜
g2_trigs = ['Trigger-In:113'] # 灰色猕猴桃

# ----------------- 1. 多核并行有效性检测 -----------------
def check_multiprocessing_and_benchmark():
    num_cores = multiprocessing.cpu_count()
    print("="*70)
    print(f"[MP AUDIT] CPU Core Detection: {num_cores} logical cores available.")
    print("[MP AUDIT] Testing Joblib Parallel with n_jobs=-1...")
    
    # 模拟并行任务
    start_t = time.time()
    def dummy_task(x):
        # 复杂算术消耗 CPU 资源
        return sum(np.sin(np.linspace(0, 100, 20000)))
    Parallel(n_jobs=-1)(delayed(dummy_task)(i) for i in range(300))
    elapsed = time.time() - start_t
    print(f"[MP AUDIT] Benchmark completed. 300 jobs processed in {elapsed:.4f} seconds.")
    print("="*70)

# ----------------- 2. 数据读取与清理 -----------------
def get_data(mat_path, is_erp, trigs_to_extract, elecs):
    if not os.path.exists(mat_path):
        return None, None
    try:
        mat = read_mat(mat_path)
        epoch = mat['epoch']
        ch_names = list(epoch['ch']['labels'])
        time_ms = epoch['time_ms'] if 'time_ms' in epoch else np.linspace(-500, 998, epoch['data'].shape[-1] if 'data' in epoch else epoch['data_cell'][0].shape[-1])
        all_trigs = list(epoch['trigger'])
        
        # 找到目标电极索引
        ch_indices = [ch_names.index(e) for e in elecs if e in ch_names]
        if not ch_indices:
            return None, None
            
        idx_list = [all_trigs.index(t) for t in trigs_to_extract if t in all_trigs]
        if not idx_list:
            return None, None
            
        data_list = []
        for idx in idx_list:
            if not is_erp:
                # HG: data_cell shape (Rep, Ch, Time) -> (Rep, n_ch, Time)
                trial_data = epoch['data_cell'][idx][:, ch_indices, :]
            else:
                # ERP: data shape (Cond, Rep, Ch, Time) -> (Rep, n_ch, Time)
                trial_data = epoch['data'][idx, :, :, :]
                trial_data = trial_data[:, ch_indices, :]
            data_list.append(trial_data)
            
        merged_data = np.concatenate(data_list, axis=0)
        # 向量化 Trial-wise 基线减法校正 (Baseline Subtraction)
        baseline_mask = time_ms < 0
        baseline_indices = np.where(baseline_mask)[0]
        if len(baseline_indices) > 0:
            mean_bl = np.mean(merged_data[:, :, baseline_indices], axis=2, keepdims=True)
            merged_data = merged_data - mean_bl
        return merged_data, time_ms
    except Exception as e:
        print(f"  [ERROR] Loading {mat_path} failed: {e}")
        return None, None

def clean_data(x):
    if x is None:
        return None
    # 剔除在任何通道、任何时间点上包含 NaN 的 trial (在 Ch, Time 维度查找)
    return x[~np.isnan(x).any(axis=(1,2))]

# ----------------- 3. 单时间步并行拟合支持向量机 -----------------
def fit_eval_single_t(t, train_r, train_g, test_r, test_g):
    """
    单个时间点上的 SVM 训练与测试
    """
    # 提取第 t 个时间点上的特征 (Rep, n_ch)
    X_tr_r = train_r[:, :, t]
    X_tr_g = train_g[:, :, t]
    X_te_r = test_r[:, :, t]
    X_te_g = test_g[:, :, t]
    
    # 训练集: 红色记忆 = 0, 绿色记忆 = 1
    X_tr = np.vstack([X_tr_r, X_tr_g])
    y_tr = np.hstack([np.zeros(X_tr_r.shape[0]), np.ones(X_tr_g.shape[0])])
    
    # 测试集: 红色记忆 = 0, 绿色记忆 = 1
    X_te = np.vstack([X_te_r, X_te_g])
    y_te = np.hstack([np.zeros(X_te_r.shape[0]), np.ones(X_te_g.shape[0])])
    
    # 标准化
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    
    # 支持向量机线性分类器 (升级为更强正则化的 C=0.1 以抵御过拟合)
    clf = SVC(kernel='linear', C=0.1)
    clf.fit(X_tr, y_tr)
    
    # 预测并评估每个试次
    y_pred = clf.predict(X_te)
    correct = (y_pred == y_te).astype(int)
    acc = np.mean(correct)
    return acc, correct

# ----------------- 4. 组水平 GLMM 显著时间窗检测 -----------------
def find_significant_windows(p_vals, time_ms, p_thresh=0.05, min_duration=20):
    dt = time_ms[1] - time_ms[0]
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
        else:
            pass
            
    if in_window:
        end_idx = len(sig_mask) - 1
        duration = (end_idx - start_idx + 1) * dt
        if duration >= min_duration:
            windows.append((time_ms[start_idx], time_ms[end_idx]))
            
    return windows

# ----------------- 5. 解码核心管线 -----------------
def run_decoding_pipeline(is_hg=False):
    feature_label = "HG" if is_hg else "ERP"
    print(f"\n[INFO] Starting Memory Color SVM Decoding Pipeline for {feature_label}...")
    
    # A. 加载元数据表
    summary_path = os.path.join(doc_dir, 'select_channel_summary.xlsx')
    if not os.path.exists(summary_path):
        print(f"[ERROR] select_channel_summary.xlsx 不存在，无法进行解码")
        return
    df_sum = pd.read_excel(summary_path)
    
    # 加载 Step 2_1 的显著电极数据
    sig_table_path = os.path.join(doc_dir, f'select_channel_memory_significance_{feature_label.lower()}.csv')
    df_sig = pd.read_csv(sig_table_path) if os.path.exists(sig_table_path) else None
    
    # 定义三套电极方案
    schemes = {
        'strategy4': 'Scheme 1: Strategy 4',
        'union': 'Scheme 2: Union of Selected',
        'memorysig': 'Scheme 3: Significant Memory Color'
    }
    
    for scheme_key, scheme_name in schemes.items():
        print(f"\n>>> Running Decoding on: {scheme_name} ({feature_label})")
        
        # 1. 为每个被试确定电极列表
        subj_elecs = {}
        for subj in subjects:
            if scheme_key == 'strategy4':
                # 策略 4 的通道
                df_subj = df_sum[(df_sum['Subject'] == subj) & (df_sum[f'{feature_label}_Strategies_Matched'].astype(str).str.contains('4'))]
                subj_elecs[subj] = df_subj['Electrode'].astype(str).tolist()
            elif scheme_key == 'union':
                # 选中的并集通道
                df_subj = df_sum[(df_sum['Subject'] == subj) & (df_sum[f'{feature_label}_Selected'] == True)]
                subj_elecs[subj] = df_subj['Electrode'].astype(str).tolist()
            elif scheme_key == 'memorysig':
                # 在记忆颜色上显著的通道
                if df_sig is not None:
                    df_subj = df_sig[(df_sig['Subject'] == subj) & (df_sig['Sig_Category'] != 'Non_Sig')]
                    subj_elecs[subj] = df_subj['Electrode'].astype(str).tolist()
                else:
                    subj_elecs[subj] = []
                    
        # 过滤掉电极数为空的被试
        active_subjs = [s for s, e in subj_elecs.items() if len(e) > 0]
        if not active_subjs:
            print(f"  [WARNING] 方案 {scheme_name} 无任何被试拥有足够电极，跳过该方案的解码！")
            continue
            
        print(f"  Active subjects for this scheme: {active_subjs}")
        for s in active_subjs:
            print(f"    - {s}: {len(subj_elecs[s])} electrodes: {subj_elecs[s]}")
            
        # 2. 读取数据并存储
        correct_data = {s: [None]*4 for s in active_subjs} # 保存 4 种配对在每个时间点上的 0/1 预测结果
        subj_accs = {s: [] for s in active_subjs}
        time_ms = None
        
        for subj in active_subjs:
            # 数据路径
            if is_hg:
                mat_path = os.path.join(feature_dir, subj, 'task2_hg_subband.mat')
            else:
                mat_path = os.path.join(feature_dir, subj, 'task2_ERP_epoched.mat')
                
            elecs = subj_elecs[subj]
            
            # 提取 4 种灰色水果的 trial 数据
            d_r1, t_arr = get_data(mat_path, not is_hg, r1_trigs, elecs)
            d_r2, _ = get_data(mat_path, not is_hg, r2_trigs, elecs)
            d_g1, _ = get_data(mat_path, not is_hg, g1_trigs, elecs)
            d_g2, _ = get_data(mat_path, not is_hg, g2_trigs, elecs)
            
            if any(d is None for d in [d_r1, d_r2, d_g1, d_g2]):
                print(f"  [ERROR] {subj} 缺少某种灰色水果的数据，跳过该被试")
                active_subjs.remove(subj)
                continue
                
            if time_ms is None:
                time_ms = t_arr
                
            # 清理数据 (剔除含 NaN 的 trial)
            d_r1, d_r2, d_g1, d_g2 = map(clean_data, [d_r1, d_r2, d_g1, d_g2])
            
            # 4 种配对组合: (train_r, train_g, test_r, test_g)
            pairs = [
                (d_r1, d_g1, d_r2, d_g2),
                (d_r1, d_g2, d_r2, d_g1),
                (d_r2, d_g1, d_r1, d_g2),
                (d_r2, d_g2, d_r1, d_g1)
            ]
            
            n_time = time_ms.shape[0]
            pair_accs = np.zeros((4, n_time))
            
            # 解码循环
            for pair_idx, (train_r, train_g, test_r, test_g) in enumerate(pairs):
                # 利用 Joblib 并行加速时间轴上的循环
                results = Parallel(n_jobs=-1)(
                    delayed(fit_eval_single_t)(t, train_r, train_g, test_r, test_g)
                    for t in range(n_time)
                )
                accs_t = np.array([r[0] for r in results])
                corrects_t = [r[1] for r in results] # 长度为 n_time 的 list of 0/1 array
                
                pair_accs[pair_idx, :] = accs_t
                correct_data[subj][pair_idx] = corrects_t
                
            # 被试的平均准确率
            subj_accs[subj] = np.mean(pair_accs, axis=0)
            
        if not active_subjs:
            print(f"  [WARNING] 所有被试均无法完成此方案解码，跳过")
            continue
            
        # 3. 逐时间步运行 GLMM 组水平显著性分析
        n_time = time_ms.shape[0]
        glmm_p_vals = np.zeros(n_time)
        glmm_est_accs = np.zeros(n_time)
        glmm_z_stats = np.zeros(n_time)
        
        print(f"  Fitting GLMM (Binomial family, random intercept by Subject) across {n_time} timepoints...")
        
        for t in range(n_time):
            y_list = []
            subj_list = []
            for subj in active_subjs:
                for pair_idx in range(4):
                    y_vector = correct_data[subj][pair_idx][t]
                    y_list.append(y_vector)
                    subj_list.extend([subj] * len(y_vector))
                    
            # 整合为 DataFrame
            df_t = pd.DataFrame({
                'Y': np.concatenate(y_list),
                'Subject': subj_list
            })
            
            try:
                # 定义二项分布的 Bayes GLMM (每个被试作为一个随机截距)
                model = bmg.BinomialBayesMixedGLM.from_formula(
                    'Y ~ 1',
                    {'Subject': '0 + C(Subject)'},
                    data=df_t
                )
                res = model.fit_vb()
                fe_mean = res.fe_mean[0]
                fe_sd = res.fe_sd[0]
                z_stat = fe_mean / fe_sd
                
                # 单尾 wald z 检验，判断截距是否显著大于 0 (即正确率显著高于 0.5)
                p_val = 1.0 - stats.norm.cdf(z_stat)
                est_acc = 1.0 / (1.0 + np.exp(-fe_mean))
                
                glmm_p_vals[t] = p_val
                glmm_est_accs[t] = est_acc
                glmm_z_stats[t] = z_stat
            except Exception as e:
                # 异常降级处理
                glmm_p_vals[t] = 1.0
                glmm_est_accs[t] = df_t['Y'].mean()
                glmm_z_stats[t] = 0.0
                
        # 4. 统计并寻找显著的连续时间窗 (>20ms)
        sig_windows = find_significant_windows(glmm_p_vals, time_ms, p_thresh=0.05, min_duration=20)
        print(f"  GLMM Significant Windows (>20ms): {sig_windows}")
        
        # 5. 提前保存解码与统计数据表格
        # 计算组平均曲线
        group_acc_list = [subj_accs[s] for s in active_subjs]
        group_mean_acc = np.mean(group_acc_list, axis=0)
        
        export_dict = {
            'Time_ms': time_ms,
            'Group_Mean_Acc': group_mean_acc,
            'GLMM_Est_Acc': glmm_est_accs,
            'GLMM_Z': glmm_z_stats,
            'GLMM_P': glmm_p_vals
        }
        for s in subjects:
            if s in active_subjs:
                export_dict[f'{s}_Acc'] = subj_accs[s]
            else:
                export_dict[f'{s}_Acc'] = np.nan
                
        df_export = pd.DataFrame(export_dict)
        export_xlsx_path = os.path.join(doc_dir, f'decoding_data_{feature_label.lower()}_{scheme_key}.xlsx')
        export_csv_path = os.path.join(doc_dir, f'decoding_data_{feature_label.lower()}_{scheme_key}.csv')
        df_export.to_excel(export_xlsx_path, index=False)
        df_export.to_csv(export_csv_path, index=False)
        print(f"  [DATA SAVED] Saved decoding details to:\n    - {export_xlsx_path}\n    - {export_csv_path}")
        
        # 6. 绘图展示 (包含三被试虚线，均值实线，显著时间阴影，顶部粗红线)
        plot_decoding_and_glmm_shading(
            time_ms, subj_accs, group_mean_acc, glmm_p_vals, sig_windows,
            feature_label, scheme_key, scheme_name, active_subjs
        )

# ----------------- 6. 图像绘制函数 -----------------
def plot_decoding_and_glmm_shading(
    time_ms, subj_accs, group_mean_acc, glmm_p_vals, sig_windows,
    feature_label, scheme_key, scheme_name, active_subjs
):
    fig, ax = plt.subplots(figsize=(12, 7.5), dpi=300)
    
    # 裁剪到 [-200, 800] ms 绘图区间
    t_idx_plot = np.where((time_ms >= -200) & (time_ms <= 800))[0]
    time_plot = time_ms[t_idx_plot]
    
    # 1. 绘制各个被试的细虚线
    subj_colors = {
        'test001': '#ff7f0e',  # 橙色
        'test002': '#2ca02c',  # 绿色
        'test003': '#1f77b4',  # 蓝色
        'test005': '#9467bd',  # 紫色
        'test006': '#8c564b'   # 棕色
    }
    
    for subj in active_subjs:
        acc_plot = subj_accs[subj][t_idx_plot]
        ax.plot(time_plot, acc_plot, color=subj_colors[subj], lw=1.3, linestyle='--', alpha=0.55, label=f"Subj: {subj}")
        
    # 2. 绘制组平均粗实线
    mean_plot = group_mean_acc[t_idx_plot]
    ax.plot(time_plot, mean_plot, color='#6f2da8', lw=3.5, label='Group Average')
    
    # 3. 绘制机会水平线 (50% 虚线)
    ax.axhline(0.5, color='#9e9e9e', linestyle=':', lw=1.5, label='Chance Level (50%)')
    ax.axvline(0, color='#757575', linestyle='-', lw=1.2)
    
    # 4. 绘制 GLMM 显著时间阴影及顶部粗横线 (y = 0.73)
    y_line_val = 0.73
    has_shaded = False
    
    for start, end in sig_windows:
        # 只在 [-200, 800] 范围内进行阴影和线段绘制
        if end < -200 or start > 800:
            continue
        s_plot = max(start, -200)
        e_plot = min(end, 800)
        
        # 阴影
        ax.axvspan(s_plot, e_plot, color='#d62728', alpha=0.12, zorder=1)
        
        # 顶部加粗红线
        label_line = 'GLMM Significant (p < 0.05, >20ms)' if not has_shaded else ""
        ax.plot([s_plot, e_plot], [y_line_val, y_line_val], color='#d62728', lw=4.5, solid_capstyle='butt', label=label_line, zorder=4)
        has_shaded = True
        
    # 美化修饰
    ax.set_title(f"{feature_label} Memory Color Decoding Performance\n{scheme_name} (Active Subjects N = {len(active_subjs)})", 
                 fontsize=13.5, fontweight='bold', pad=12)
    ax.set_xlabel("Time relative to stimulus onset (ms)", fontsize=11)
    ax.set_ylabel("Decoding Accuracy", fontsize=11)
    ax.set_xlim([-200, 800])
    ax.set_ylim([0.35, 0.76])
    ax.grid(True, linestyle=':', alpha=0.45)
    ax.set_facecolor('#fafafa')
    
    # 组合图例
    ax.legend(loc='lower left', framealpha=0.9, fontsize=9.5)
    
    plt.tight_layout()
    out_fig = os.path.join(out_dir, f"{feature_label.lower()}_{scheme_key}_decoding.png")
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"  [FIGURE SAVED] Saved decoding plot to: {out_fig}")

# ----------------- 主流程入口 -----------------
def main():
    print("="*80)
    print("Step 2_2: Running Multi-Electrode Memory Color Decoding and GLMM Analysis")
    print("="*80)
    
    # 1. 运行多核检测与并行基准审计
    check_multiprocessing_and_benchmark()
    
    # 2. 对 ERP 特征开展分析
    run_decoding_pipeline(is_hg=False)
    
    # 3. 对 High Gamma 特征开展分析
    run_decoding_pipeline(is_hg=True)
    
    print("\n" + "="*80)
    print("Step 2_2 Memory Color Decoding and GLMM Analysis Successfully Completed!")
    print("="*80)

if __name__ == '__main__':
    main()
