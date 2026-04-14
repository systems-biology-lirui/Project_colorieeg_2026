import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # 依然使用非交互式后端
import matplotlib.pyplot as plt
from scipy.ndimage import label

# ===================== 配置区域 =====================
# 指向你刚刚保存 .npy 文件的具体文件夹路径
DATA_DIR = r'/home/lirui/liulab_project/ieeg/Project_colorieeg_2026/analyse_code/newanalyse/fft_decoding_task/decoding_highgamma_auc_binary_2vs3_lda10fold/test002'
ROI_NAME = 'Fusiform_R'

# 核心统计阈值 (随心所欲地调整这里！)
CLUSTER_FORMING_P = 0.05  # 候选聚类的时间点阈值 (0.05 对应 95% 线，0.01 对应 99% 线)
CLUSTER_SIG_P = 0.1      # 最终 Cluster Mass 的显著性阈值 (通常保持 0.05)

# ====================================================

def main():
    # 1. 极速加载事先算好的数据
    print(f"Loading pre-computed data for {ROI_NAME}...")
    try:
        mean_auc = np.load(os.path.join(DATA_DIR, f'{ROI_NAME}_mean_auc.npy'))
        sem_auc = np.load(os.path.join(DATA_DIR, f'{ROI_NAME}_sem_auc.npy'))
        perm_dist = np.load(os.path.join(DATA_DIR, f'{ROI_NAME}_perm_dist.npy'))
        plot_times = np.load(os.path.join(DATA_DIR, f'{ROI_NAME}_plot_times.npy'))
    except FileNotFoundError as e:
        print(f"Error: 找不到 .npy 文件，请确认路径是否正确。\n{e}")
        return

    n_perms = perm_dist.shape[0]
    print(f"Loaded {n_perms} permutations. Re-evaluating statistics...")

    # 2. 重新计算统计阈值
    # 将 p-value 转化为百分位数 (例如 p=0.05 -> 95th percentile)
    forming_percentile = 100 * (1 - CLUSTER_FORMING_P)
    sig_percentile = 100 * (1 - CLUSTER_SIG_P)

    # 重新计算随时间变化的红线
    threshold_curve = np.percentile(perm_dist, forming_percentile, axis=0)

    # 3. 重新寻找候选聚类
    binary_map = mean_auc > threshold_curve
    clusters, n_clusters = label(binary_map)
    
    cluster_masses = []
    for i in range(1, n_clusters + 1):
        idx = (clusters == i)
        mass = np.sum(mean_auc[idx] - threshold_curve[idx])
        cluster_masses.append(mass)

    # 4. 重新构建 Null Cluster Mass 分布
    null_cluster_masses = []
    for p in range(n_perms):
        perm_curve = perm_dist[p, :]
        p_binary = perm_curve > threshold_curve
        p_clusters, p_n = label(p_binary)
        
        max_mass = 0
        for i in range(1, p_n + 1):
            idx = (p_clusters == i)
            mass = np.sum(perm_curve[idx] - threshold_curve[idx])
            if mass > max_mass:
                max_mass = mass
        null_cluster_masses.append(max_mass)
        
    null_cluster_masses = np.array(null_cluster_masses)
    if len(null_cluster_masses) > 0:
        critical_mass = np.percentile(null_cluster_masses, sig_percentile)
    else:
        critical_mass = np.inf

    # 5. 判定新的显著聚类
    sig_indices = np.zeros_like(mean_auc, dtype=bool)
    sig_cluster_count = 0
    for i in range(1, n_clusters + 1):
        if cluster_masses[i-1] > critical_mass:
            sig_indices[clusters == i] = True
            sig_cluster_count += 1
            
    print(f"Found {n_clusters} candidate clusters, {sig_cluster_count} survived at p<{CLUSTER_SIG_P}.")

    # 6. 重新画图出炉
    plot_results(ROI_NAME, mean_auc, sem_auc, threshold_curve, sig_indices, plot_times, DATA_DIR, CLUSTER_FORMING_P, CLUSTER_SIG_P)

def plot_results(roi_name, mean_auc, sem_auc, threshold_curve, sig_indices, plot_times, output_dir, forming_p, sig_p):
    plt.figure(figsize=(10, 6))
    
    # 绘制基础曲线
    plt.fill_between(plot_times, mean_auc - sem_auc, mean_auc + sem_auc, color='b', alpha=0.2, label='SEM')
    plt.plot(plot_times, mean_auc, 'b-', linewidth=2, label='Mean AUC')
    
    # 绘制新的阈值线
    plt.plot(plot_times, threshold_curve, 'r--', linewidth=1.5, label=f'Perm Threshold (p<{forming_p})')
    
    # 绘制新的阴影
    if np.any(sig_indices):
        ymin, ymax = plt.ylim()
        plt.fill_between(plot_times, 0, 1, where=sig_indices, color='gray', alpha=0.3, 
                         transform=plt.gca().get_xaxis_transform(),
                         label=f'Cluster Sig (p<{sig_p})')

    plt.axhline(0.5, color='k', linestyle=':', label='Chance (0.5)')
    
    # 标题加上阈值信息，方便区分不同版本
    plt.title(f'Temporal Decoding AUC (LDA) - {roi_name}\nForming p<{forming_p}, Cluster p<{sig_p}')
    plt.xlabel('Time (ms)')
    plt.ylabel('ROC AUC')
    plt.xlim(plot_times[0], plot_times[-1])
    plt.ylim(0.4, 1.0)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 文件名也加上阈值标签，防止覆盖原始图像
    save_path = os.path.join(output_dir, f'{roi_name}_AUC_p{forming_p}_sig{sig_p}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved adjusted plot: {save_path}")

if __name__ == "__main__":
    main()