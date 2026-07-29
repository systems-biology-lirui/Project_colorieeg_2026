"""CCEP 独立批处理入口。

这个版本只调用当前打包目录 code/ 下的脚本，
输入数据和输出结果都位于同一份独立目录中。
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


BASE_PATH = Path(__file__).resolve().parent.parent
CODE_DIR = Path(__file__).resolve().parent

DEFAULT_BATCH_CONFIG = {
    "subjects": ["test001"],
    "run_preprocess": True,
    "run_feature_extraction": True,
    "run_erp_stats": True,
    "run_tfa_stats": True,
    "matlab_bin": "matlab",
    "python_bin": sys.executable,
    "stop_on_error": True,
    "keep_runtime_config_files": False,
    "ccep_defaults": {
        "alpha": 0.05,
        "apply_fdr": True,
        "artifact_start_ms": 10.0,
        "min_consecutive_sig_points": 3,
    },
}


def merge_dicts(base, extra):
    """递归合并批处理配置。"""
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def parse_args():
    """读取命令行参数。"""
    parser = argparse.ArgumentParser(description="Run the CCEP preprocessing and response-stat pipeline.")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config file.")
    return parser.parse_args()


def load_batch_config(config_path):
    """加载用户配置，并覆盖默认批处理参数。"""
    config = dict(DEFAULT_BATCH_CONFIG)
    if config_path is None:
        return config
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return merge_dicts(config, payload)


def write_runtime_config(runtime_dir, subject, overrides, suffix):
    """为当前被试和模态生成临时 runtime config。"""
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = runtime_dir / f"{subject}_{suffix}.json"
    payload = {"ccep_defaults": merge_dicts({"subject": subject}, overrides)}
    runtime_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return runtime_path


def run_command(command, env=None):
    """执行单个外部命令并打印耗时。"""
    start = time.time()
    print(f"[RUN] {' '.join(command)}")
    completed = subprocess.run(command, cwd=BASE_PATH, env=env, check=False)
    duration = time.time() - start
    print(f"[DONE] exit={completed.returncode} | {duration:.1f}s")
    return completed.returncode


def maybe_raise(exit_code, stop_on_error, step_name):
    """在启用 stop_on_error 时把失败步骤转成异常。"""
    if exit_code != 0 and stop_on_error:
        raise RuntimeError(f"Step failed: {step_name} (exit={exit_code})")


def build_matlab_run_expr(script_path):
    """生成 MATLAB -batch 所需的 run(...) 表达式。"""
    script_text = str(script_path).replace("'", "''")
    return f"run('{script_text}')"


def main():
    """按配置顺序执行 CCEP 预处理、特征提取和统计。"""
    args = parse_args()
    config = load_batch_config(args.config)
    runtime_dir = CODE_DIR / ".runtime_configs"

    for subject in config["subjects"]:
        print(f"\n=== Running CCEP pipeline for {subject} ===")

        if config["run_preprocess"]:
            exit_code = run_command([
                config["matlab_bin"],
                "-batch",
                build_matlab_run_expr(CODE_DIR / "Sec1_ccep_preanalyse.m"),
            ])
            maybe_raise(exit_code, config["stop_on_error"], "Sec1_ccep_preanalyse")

        if config["run_feature_extraction"]:
            exit_code = run_command([
                config["matlab_bin"],
                "-batch",
                build_matlab_run_expr(CODE_DIR / "Sec2_ccep_preprocess_roi_features.m"),
            ])
            maybe_raise(exit_code, config["stop_on_error"], "Sec2_ccep_preprocess_roi_features")

        for modality, enabled in (("erp", config["run_erp_stats"]), ("tfa", config["run_tfa_stats"])):
            if not enabled:
                continue

            runtime_path = write_runtime_config(
                runtime_dir=runtime_dir,
                subject=subject,
                overrides=merge_dicts(config.get("ccep_defaults", {}), {"modality": modality}),
                suffix=f"{modality}_stats",
            )
            env = os.environ.copy()
            env["NEWANALYSE_USE_CONFIG"] = "1"
            env["NEWANALYSE_CONFIG_PATH"] = str(runtime_path)

            exit_code = run_command([
                config["python_bin"],
                str(CODE_DIR / "Sec3_ccep_electrode_response_stats.py"),
            ], env=env)
            maybe_raise(exit_code, config["stop_on_error"], f"Sec3_ccep_electrode_response_stats ({modality})")

            if not config["keep_runtime_config_files"] and runtime_path.exists():
                runtime_path.unlink()


if __name__ == "__main__":
    script_start = time.time()
    try:
        main()
    finally:
        print(f"Total runtime: {time.time() - script_start:.2f} s")