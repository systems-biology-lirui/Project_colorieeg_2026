import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from newanalyse_paths import get_feature_dir, project_root


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mat-file",
        type=Path,
        default=Path(
            get_feature_dir(project_root(), 'erp', 'test001') / 'Color_patch.mat'
        ),
    )
    parser.add_argument("--field", type=str, default="erp_task1")
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--fs", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--condition-prefix", type=str, default="cond")
    return parser.parse_args()


def build_time_axis(n_time, tmin, fs):
    return tmin + np.arange(n_time) / fs


def prepare_styles(n_conditions):
    base_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    
    # 将同一对物体（彩色和灰色）分配相同的颜色
    # Python索引: 0,1 -> 颜色0; 2,3 -> 颜色1; 4,5 -> 颜色2; 6,7 -> 颜色3
    colors = [base_colors[(i // 2) % len(base_colors)] for i in range(n_conditions)]
    
    # Python 索引为偶数（0,2,4,6，即实验的奇数条件/彩色）使用实线 "-"
    # Python 索引为奇数（1,3,5,7，即实验的偶数条件/灰色）使用虚线 "--"
    linestyles = ["-" if i % 2 == 0 else "--" for i in range(n_conditions)]
    
    return colors, linestyles


def main():
    args = parse_args()
    data = loadmat(args.mat_file)
    if args.field not in data:
        raise KeyError(f"{args.field} 不在 mat 文件中，可用键: {[k for k in data.keys() if not k.startswith('__')]}")

    erp = np.asarray(data[args.field], dtype=float)
    if erp.ndim != 4:
        raise ValueError(f"{args.field} 维度应为 4，当前为 {erp.shape}")

    n_conditions, n_repeat, n_channels, n_time = erp.shape
    mean_resp = erp.mean(axis=1)
    sem_resp = erp.std(axis=1, ddof=1) / np.sqrt(n_repeat)
    t = build_time_axis(n_time, args.tmin, args.fs)
    colors, linestyles = prepare_styles(n_conditions)

    n_cols = int(np.ceil(np.sqrt(n_channels)))
    n_rows = int(np.ceil(n_channels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.8 * n_rows), squeeze=False)
    
    # 更新主标题以反映新的图例逻辑
    fig.suptitle(
        f"{args.field} | shape={erp.shape} | mean±SEM across repeat\nOdd conditions (Color): solid, Even conditions (Gray): dashed",
        fontsize=12,
    )

    for ch in range(n_channels):
        ax = axes[ch // n_cols, ch % n_cols]
        for cond in range(n_conditions):
            y = mean_resp[cond, ch, :]
            e = sem_resp[cond, ch, :]
            label = f"{args.condition_prefix}{cond + 1}"
            
            # 判断逻辑：Python索引为奇数时，代表实验的偶数条件（灰色）
            if cond % 2 == 1:
                label += " (gray)"
            else:
                label += " (color)"
                
            ax.plot(t, y, color=colors[cond], linestyle=linestyles[cond], linewidth=1.8, label=label)
            ax.fill_between(t, y - e, y + e, color=colors[cond], alpha=0.18)

        ax.set_title(f"Channel {ch + 1}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.25)

    for empty_idx in range(n_channels, n_rows * n_cols):
        fig.delaxes(axes[empty_idx // n_cols, empty_idx % n_cols])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=True, ncol=2)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=300, bbox_inches="tight")
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()