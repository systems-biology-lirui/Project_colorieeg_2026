import importlib.util
import time
from pathlib import Path

from newanalyse_paths import project_root


BASE_PATH = project_root()
SCRIPT_PATH = BASE_PATH / "newanalyse" / "Sec3_s2_roi_condition_tfa.py"

# =========================
# User Config
# =========================
SUBJECT = "test001"
RUNS = [
    {
        "id": "task1_color_vs_gray",
        "task": "task1",
        "group_a": [1, 3, 5, 7],
        "group_b": [2, 4, 6, 8],
        "label_a": "Color",
        "label_b": "Gray",
        "roi": None,
    },
]

FREQS = None
N_CYCLES = None
BASELINE_MS = (-100.0, -50.0)
BASELINE_MODE = "db"
SAVE_TRIAL_LEVEL_POWER = True

OUTPUT_ROOT = BASE_PATH / "result" / "roi_condition_tfa_batch" / SUBJECT


def load_single_run_module():
    spec = importlib.util.spec_from_file_location("roi_condition_tfa_single", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_one(run_cfg):
    module = load_single_run_module()
    module.SUBJECT = SUBJECT
    module.TASK = run_cfg["task"]
    module.GROUP_A = run_cfg["group_a"]
    module.GROUP_B = run_cfg["group_b"]
    module.GROUP_A_LABEL = run_cfg["label_a"]
    module.GROUP_B_LABEL = run_cfg["label_b"]
    module.ROI = run_cfg.get("roi")
    module.BASELINE_MS = BASELINE_MS
    module.BASELINE_MODE = BASELINE_MODE
    module.SAVE_TRIAL_LEVEL_POWER = SAVE_TRIAL_LEVEL_POWER
    module.OUTPUT_ROOT = OUTPUT_ROOT / run_cfg["id"]

    if FREQS is not None:
        module.FREQS = FREQS
    if N_CYCLES is not None:
        module.N_CYCLES = N_CYCLES

    print(f"Running {run_cfg['id']} | task={run_cfg['task']} | roi={run_cfg.get('roi')}")
    module.main()


def main():
    for run_cfg in RUNS:
        run_one(run_cfg)


if __name__ == "__main__":
    _script_start_time = time.time()
    try:
        main()
    finally:
        print(f"Total runtime: {time.time() - _script_start_time:.2f} s")