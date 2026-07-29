import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.stats as stats
from pymatreader import read_mat
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import warnings

# 忽略绘图与模型拟合中的警告
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
out_fig_dir = os.path.join(result_dir, 'select_channel', 'decoding', 'single_electrode')
os.makedirs(out_fig_dir, exist_ok=True)

subjects = ['test001', 'test002', 'test003']
r1_trigs = ['Trigger-In:123'] # 灰色草莓
r2_trigs = ['Trigger-In:133'] # 灰色西瓜
g1_trigs = ['Trigger-In:103'] # 灰色卷心菜
g2_trigs = ['Trigger-In:113'] # 灰色猕猴桃

# ----------------- 1. 兼容性二项检验 -----------------
def binomial_test_p(k, n, p_chance=0.5):
    """
    单尾二项检验，测试准确率是否显著大于随机机会 (0.5)
    """
    try:
        from scipy.stats import binomtest
        res = binomtest(k, n, p=p_chance, alternative='greater')
        return res.pvalue
    except ImportError:
        from scipy.stats import binom_test
        return binom_test(k, n, p=p_chance, alternative='greater')

# ----------------- 2. 数据读取与向量化基线减除 -----------------
def get_single_channel_data(mat_path, is_erp, trig_list, elec):
    if not os.path.exists(mat_path):
        return None, None
    try:
        mat = read_mat(mat_path)
        epoch = mat['epoch']
        ch_names = list(epoch['ch']['labels'])
        time_ms = epoch['time_ms'] if 'time_ms' in epoch else np.linspace(-500, 998, epoch['data'].shape[-1] if 'data' in epoch else epoch['data_cell'][0].shape[-1])
        all_trigs = list(epoch['trigger'])
        
        if elec not in ch_names:
            return None, None
        ch_idx = ch_names.index(elec)
        
        idx_list = [all_trigs.index(t) for t in trig_list if t in all_trigs]
        if not idx_list:
            return None, None
            
        data_list = []
        for idx in idx_list:
            if not is_erp:
                # HG shape (Rep, Ch, Time) -> 取单通道 -> (Rep, 1, Time)
                trial_data = epoch['data_cell'][idx][:, [ch_idx], :]
            else:
                # ERP shape (Cond, Rep, Ch, Time) -> (Rep, 1, Time)
                # 使用整数索引配合 np.expand_dims 避免 numpy 高级索引混合切片导致维度换位 (1, Rep, Time) 的问题
                trial_data = epoch['data'][idx, :, ch_idx, :]
                trial_data = np.expand_dims(trial_data, axis=1)
            data_list.append(trial_data)
            
        merged_data = np.concatenate(data_list, axis=0)
        
        # 向量化 Trial-wise 基线减法校正 (Baseline Subtraction)
        baseline_mask = time_ms < 0
        baseline_indices = np.where(baseline_mask)[0]
        if len(baseline_indices) > 0:
            # merged_data shape: (Total_Rep, 1, n_time)
            # 在时间轴 (axis=2) 上求基线均值，保持维度 (Total_Rep, 1, 1) 用于广播
            mean_bl = np.mean(merged_data[:, :, baseline_indices], axis=2, keepdims=True)
            merged_data = merged_data - mean_bl
            
        return merged_data, time_ms
    except Exception as e:
        print(f"  [ERROR] get_single_channel_data failed for {elec} in {mat_path}: {e}")
        return None, None

def clean_data(x):
    if x is None:
        return None
    return x[~np.isnan(x).any(axis=(1,2))]

# ----------------- 3. 单时间点 SVM 拟合 (D=1) -----------------
def fit_eval_single_t(t, train_r, train_g, test_r, test_g):
    X_tr_r = train_r[:, :, t]
    X_tr_g = train_g[:, :, t]
    X_te_r = test_r[:, :, t]
    X_te_g = test_g[:, :, t]
    
    X_tr = np.vstack([X_tr_r, X_tr_g])
    y_tr = np.hstack([np.zeros(X_tr_r.shape[0]), np.ones(X_tr_g.shape[0])])
    
    X_te = np.vstack([X_te_r, X_te_g])
    y_te = np.hstack([np.zeros(X_te_r.shape[0]), np.ones(X_te_g.shape[0])])
    
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    
    clf = SVC(kernel='linear', C=0.1) # 沿用强正则化 C=0.1
    clf.fit(X_tr, y_tr)
    
    y_pred = clf.predict(X_te)
    correct = (y_pred == y_te).astype(int)
    acc = np.mean(correct)
    return acc, correct

# ----------------- 4. 显著性检验与潜伏期 (ESTP) 计算 -----------------
def calculate_estp_latency(correct_trials_all_pairs, time_ms, start_time_threshold=80.0):
    """
    计算刺激后 (>=80ms) 最早显著解码的时间点
    """
    n_time = time_ms.shape[0]
    p_vals = np.ones(n_time)
    
    for t in range(n_time):
        # 合并 4 种配对的所有测试 trial 的正确性对错向量 (0/1)
        y_list = [correct_trials_all_pairs[pair_idx][t] for pair_idx in range(4)]
        y_all = np.concatenate(y_list)
        
        k = np.sum(y_all)
        n = len(y_all)
        if n > 0:
            p_vals[t] = binomial_test_p(k, n, p_chance=0.5)
            
    # 提取所有 t >= 80ms 且单尾 p < 0.05 的显著点
    search_indices = np.where(time_ms >= start_time_threshold)[0]
    sig_indices = [idx for idx in search_indices if p_vals[idx] < 0.05]
    
    if sig_indices:
        estp = time_ms[sig_indices[0]]
        return estp, p_vals
    else:
        return np.nan, p_vals

# ----------------- 5. 解码与相关性分析核心流程 -----------------
def run_single_electrode_decoding_pipeline(is_hg=False):
    feature_label = "HG" if is_hg else "ERP"
    print(f"\n[INFO] Starting Single Electrode Decoding & Correlation Pipeline for {feature_label}...")
    
    # 1. 读取 Step 2_1 的显著电极明细表
    sig_path = os.path.join(doc_dir, f'select_channel_memory_significance_{feature_label.lower()}.csv')
    if not os.path.exists(sig_path):
        print(f"[ERROR] 显著电极数据文件不存在，请运行 step2_1！")
        return
        
    df_sig = pd.read_csv(sig_path)
    # 筛选记忆显著的电极
    df_memory_color = df_sig[
        (df_sig['Sig_Category'] != 'Non_Sig') &
        (df_sig['Subject'].astype(str).isin(subjects))
    ]
    
    if df_memory_color.empty:
        print(f"[WARNING] 无记忆颜色显著的 {feature_label} 通道！跳过解码分析")
        return
        
    print(f"Total memory-selective {feature_label} electrodes: {len(df_memory_color)}")
    
    decoding_results = []
    electrode_curves = {}
    time_ms = None
    
    # 2. 逐电极计算 SVM 4折交叉解码
    for idx, row in df_memory_color.iterrows():
        subj = str(row['Subject']).strip()
        elec = str(row['Electrode']).strip()
        
        # 数据路径
        if is_hg:
            mat_path = os.path.join(feature_dir, subj, 'task2_hg_subband.mat')
        else:
            mat_path = os.path.join(feature_dir, subj, 'task2_ERP_epoched.mat')
            
        # 提取 4 种灰色水果数据 (只对该电极)
        d_r1, t_arr = get_single_channel_data(mat_path, not is_hg, r1_trigs, elec)
        d_r2, _ = get_single_channel_data(mat_path, not is_hg, r2_trigs, elec)
        d_g1, _ = get_single_channel_data(mat_path, not is_hg, g1_trigs, elec)
        d_g2, _ = get_single_channel_data(mat_path, not is_hg, g2_trigs, elec)
        
        if any(d is None for d in [d_r1, d_r2, d_g1, d_g2]):
            print(f"  [WARNING] 电极 {subj}-{elec} 的灰色刺激数据为空，跳过该电极解码")
            continue
            
        if time_ms is None:
            time_ms = t_arr
            
        # 清理 NaN
        d_r1, d_r2, d_g1, d_g2 = map(clean_data, [d_r1, d_r2, d_g1, d_g2])
        
        # 4 种配对组合
        pairs = [
            (d_r1, d_g1, d_r2, d_g2),
            (d_r1, d_g2, d_r2, d_g1),
            (d_r2, d_g1, d_r1, d_g2),
            (d_r2, d_g2, d_r1, d_g1)
        ]
        
        n_time = time_ms.shape[0]
        pair_accs = np.zeros((4, n_time))
        correct_trials_all_pairs = [None]*4
        
        # 并行解码
        for p_idx, (train_r, train_g, test_r, test_g) in enumerate(pairs):
            results = Parallel(n_jobs=-1)(
                delayed(fit_eval_single_t)(t, train_r, train_g, test_r, test_g)
                for t in range(n_time)
            )
            pair_accs[p_idx, :] = np.array([r[0] for r in results])
            correct_trials_all_pairs[p_idx] = [r[1] for r in results]
            
        mean_acc = np.mean(pair_accs, axis=0)
        electrode_curves[f"{subj}_{elec}"] = mean_acc
        
        # 提取 80ms 以后的最早显著时间点 (ESTP)
        estp, p_vals = calculate_estp_latency(correct_trials_all_pairs, time_ms, start_time_threshold=80.0)
        
        print(f"  Electrode {subj} - {elec} (Y = {row['MNI_Y']:.2f}): ESTP = {estp} ms")
        
        decoding_results.append({
            'Subject': subj,
            'Electrode': elec,
            'MNI_X': float(row['MNI_X']),
            'MNI_Y': float(row['MNI_Y']),
            'MNI_Z': float(row['MNI_Z']),
            'AAL3_ROI': str(row['AAL3_ROI']),
            'Strategies_Matched': str(row['Strategies_Matched']),
            'ESTP': estp
        })
        
    # 整理结果为 DataFrame 并提前保存
    df_estp = pd.DataFrame(decoding_results)
    excel_path = os.path.join(doc_dir, f'select_channel_memory_decoding_estp_{feature_label.lower()}.xlsx')
    csv_path = os.path.join(doc_dir, f'select_channel_memory_decoding_estp_{feature_label.lower()}.csv')
    df_estp.to_excel(excel_path, index=False)
    df_estp.to_csv(csv_path, index=False)
    print(f"[SUCCESS] Saved ESTP latency results to:\n  - {excel_path}\n  - {csv_path}")
    
    # 3. 绘制组水平及被试个体水平的折线与相关大图 (共 4 张大图)
    # A. 组水平大图 (Group Level)
    plot_line_and_correlation_multi_panel(
        df_estp, electrode_curves, time_ms, "Group", feature_label, 
        os.path.join(out_fig_dir, f"{feature_label.lower()}_group_decoding_estp.png")
    )
    
    # B. 个体水平大图 (Subject Level)
    for subj in subjects:
        df_subj = df_estp[df_estp['Subject'] == subj]
        # 提取对应被试的曲线子集
        curves_subj = {k: v for k, v in electrode_curves.items() if k.startswith(subj)}
        plot_line_and_correlation_multi_panel(
            df_subj, curves_subj, time_ms, subj, feature_label,
            os.path.join(out_fig_dir, f"{feature_label.lower()}_{subj}_decoding_estp.png")
        )

# ----------------- 6. 1行2列 渐变折线与散点回归图绘制 -----------------
def plot_line_and_correlation_multi_panel(df, curves, time_ms, subj_name, feature_label, out_path):
    """
    绘制 1行2列 大图：
    - 左子图：时程折线图。根据每个电极的 MNI Y 坐标由后脑（蓝）到前脑（红）渐变着色，黑色粗实线为平均值。
    - 右子图：ESTP (ms) 与 MNI Y 坐标的相关散点图 + 回归拟合线。
    """
    if df.empty:
        # 画张空白警告图以防崩溃
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, f"No memory-selective electrodes\nfor {subj_name} ({feature_label})", 
                ha='center', va='center', fontsize=14, color='red')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        return
        
    fig, (ax_line, ax_scatter) = plt.subplots(1, 2, figsize=(19, 8), gridspec_kw={'width_ratios': [1.3, 1]}, dpi=300)
    fig.suptitle(f"{feature_label} Single-Electrode Latency & Location Analysis: {subj_name}", 
                 fontsize=15, fontweight='bold', y=0.97)
                 
    # MNI Y 的极限值，用于归一化上色
    y_vals_all = df['MNI_Y'].values
    y_min, y_max = y_vals_all.min(), y_vals_all.max()
    y_range = (y_max - y_min) if (y_max - y_min) > 0 else 1.0
    
    # X 轴波形过滤
    t_idx_plot = np.where((time_ms >= -200) & (time_ms <= 800))[0]
    time_plot = time_ms[t_idx_plot]
    
    # ------------------ (1) 左图: 渐变时程折线 ------------------
    all_curves_list = []
    
    # 逐个电极画折线
    for idx, row in df.iterrows():
        key = f"{row['Subject']}_{row['Electrode']}"
        if key in curves:
            curve_acc = curves[key][t_idx_plot]
            all_curves_list.append(curve_acc)
            
            # 颜色映射 (MNI Y 映射 coolwarm)
            norm_y = (row['MNI_Y'] - y_min) / y_range
            color = plt.cm.coolwarm(norm_y)
            
            ax_line.plot(time_plot, curve_acc, color=color, lw=1.2, alpha=0.55, zorder=2)
            
    # 画群体平均线
    if all_curves_list:
        group_mean = np.mean(all_curves_list, axis=0)
        ax_line.plot(time_plot, group_mean, color='black', lw=3.6, label='Average Across Channels', zorder=3)
        
    # 机会线与时间线
    ax_line.axhline(0.5, color='#757575', linestyle=':', lw=1.5, label='Chance Level (50%)')
    ax_line.axvline(0, color='gray', linestyle='-', lw=1.2)
    ax_line.axvline(80, color='#d62728', linestyle='-.', lw=1.2, alpha=0.7, label='Search Boundary (80ms)')
    
    ax_line.set_title("Single-Electrode Decoding Curves", fontsize=12.5, fontweight='bold')
    ax_line.set_xlabel("Time relative to stimulus onset (ms)", fontsize=10.5)
    ax_line.set_ylabel("Decoding Accuracy", fontsize=10.5)
    ax_line.set_xlim([-200, 800])
    ax_line.set_ylim([0.35, 0.78])
    ax_line.grid(True, linestyle=':', alpha=0.45)
    ax_line.legend(loc='lower left', fontsize=9)
    ax_line.set_facecolor('#fafafa')
    
    # ------------------ (2) 右图: ESTP vs MNI Y 相关性 ------------------
    # 提取有效 ESTP（即非 NaN）的数据
    df_valid = df[~df['ESTP'].isna()]
    
    if len(df_valid) < 2:
        ax_scatter.text(0.5, 0.5, "Insufficient channels with\nsignificant latency >= 80ms\n(N < 2)", 
                        ha='center', va='center', fontsize=12.5, color='gray', fontweight='semibold')
        ax_scatter.set_title("Latency vs. Anterior-Posterior Location", fontsize=12.5, fontweight='bold')
        ax_scatter.grid(True, linestyle=':', alpha=0.3)
        ax_scatter.set_facecolor('#fafafa')
    else:
        mni_y_valid = df_valid['MNI_Y'].values
        estp_valid = df_valid['ESTP'].values
        
        # 散点颜色映射与左侧曲线一致
        colors_scatter = []
        for y_coord in mni_y_valid:
            norm_y = (y_coord - y_min) / y_range
            colors_scatter.append(plt.cm.coolwarm(norm_y))
            
        # 画散点
        ax_scatter.scatter(mni_y_valid, estp_valid, color=colors_scatter, s=70, edgecolor='#555555', linewidth=1.0, zorder=3)
        
        # 标注电极名称
        for _, r_valid in df_valid.iterrows():
            ax_scatter.annotate(f"{r_valid['Electrode']}", (r_valid['MNI_Y'], r_valid['ESTP']), xytext=(0, 6), 
                                textcoords='offset points', ha='center', fontsize=8.5, alpha=0.85)
            
        # 计算相关性
        s_r, s_p = stats.spearmanr(mni_y_valid, estp_valid)
        p_r, p_p = stats.pearsonr(mni_y_valid, estp_valid)
        
        # 线性拟合线绘制
        slope, intercept, r_val, p_val, std_err = stats.linregress(mni_y_valid, estp_valid)
        x_fit = np.linspace(mni_y_valid.min() - 3, mni_y_valid.max() + 3, 100)
        y_fit = slope * x_fit + intercept
        ax_scatter.plot(x_fit, y_fit, color='#d62728', lw=2.2, label='Linear Trend Line', zorder=2)
        
        # 自带置信区间 (以回归的标准误简单估计拟合范围并绘制阴影)
        y_fit_err = std_err * x_fit
        ax_scatter.fill_between(x_fit, y_fit - y_fit_err * 2.0, y_fit + y_fit_err * 2.0, color='#d62728', alpha=0.1)
        
        # 显著性星号标注
        def star_p(p):
            if p < 0.001: return "***"
            elif p < 0.01: return "**"
            elif p < 0.05: return "*"
            else: return "(n.s.)"
            
        # 标注结果文本框
        corr_text = (
            f"Active Channels N = {len(df_valid)}\n\n"
            f"Spearman $r_s$ = {s_r:.3f}{star_p(s_p)}\n(p = {s_p:.2e})\n"
            f"Pearson $r_p$ = {p_r:.3f}{star_p(p_p)}\n(p = {p_p:.2e})"
        )
        ax_scatter.text(0.05, 0.95, corr_text, transform=ax_scatter.transAxes, fontsize=10, fontweight='semibold',
                        verticalalignment='top', bbox=dict(boxstyle='round,pad=0.45', facecolor='#fbfbfb', alpha=0.9, edgecolor='#cccccc'))
                        
        ax_scatter.set_title("Latency vs. Anterior-Posterior Location", fontsize=12.5, fontweight='bold')
        ax_scatter.set_xlabel("Electrode MNI Y Coordinate\n(Posterior/后脑 <-- 0 --> Anterior/前脑)", fontsize=10.5)
        ax_scatter.set_ylabel("Earliest Significant Time Point (ESTP, ms)", fontsize=10.5)
        ax_scatter.grid(True, linestyle=':', alpha=0.45)
        ax_scatter.set_facecolor('#fafafa')
        ax_scatter.legend(loc='lower right', fontsize=9.5)
        
    # 底部添加渐变色标 (从后脑到前脑 Y 轴位置的色标)
    cax = fig.add_axes([0.38, 0.03, 0.28, 0.025])
    norm = plt.Normalize(vmin=y_min, vmax=y_max)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.coolwarm), cax=cax, orientation='horizontal')
    cb.set_label('Electrode Position (MNI Y)', fontsize=9)
    cb.ax.tick_params(labelsize=8)
    
    plt.tight_layout(rect=[0, 0.07, 1, 0.94])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [FIGURE SAVED] Saved multi-panel figure to: {out_path}")

# ----------------- 主程序入口 -----------------
def main():
    print("="*85)
    print("Step 2_3: Single Electrode Memory Color Decoding and MNI Y Spatial Latency Correlation")
    print("="*85)
    
    # 1. 运行 ERP 信号管线
    run_single_electrode_decoding_pipeline(is_hg=False)
    
    # 2. 运行 HG 信号管线
    run_single_electrode_decoding_pipeline(is_hg=True)
    
    print("\n" + "="*85)
    print("Step 2_3 Single Electrode Latency Analysis Successfully Completed!")
    print("="*85)

if __name__ == '__main__':
    main()
