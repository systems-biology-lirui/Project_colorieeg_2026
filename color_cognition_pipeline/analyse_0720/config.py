"""Central configuration for the restarted 0720 analysis pipeline."""
from pathlib import Path

PROJECT_ROOT = Path("/home/lirui/liulab_project/ieeg/Project_colorieeg_2026")
ANALYSE_ROOT = PROJECT_ROOT / "color_cognition_pipeline" / "analyse_0720"
RAW_ROOT = PROJECT_ROOT / "seegdata"
INTERMEDIATE_ROOT = ANALYSE_ROOT / "intermediate"
RESULT_ROOT = ANALYSE_ROOT / "result"
REPORT_ROOT = ANALYSE_ROOT / "reports"

SUBJECTS = [f"test{i:03d}" for i in range(1, 7)]
RAW_SUBJECT_DIR = {subject: subject.replace("test00", "test") for subject in SUBJECTS}
RUNS = {1: "erp1", 2: "erp2", 3: "erp3"}

FMRI_TARGETS = {
    "left": (-35.0, -67.0, -17.0),
    "right": (45.0, -43.0, -15.0),
}
PRIMARY_TARGET_RADIUS_MM = 20.0
SENSITIVITY_RADII_MM = (5.0, 10.0, 15.0, 20.0)

SFREQ_RAW = 1000.0
SFREQ_ERP = 500.0
EPOCH_TMIN_S = -0.5
EPOCH_TMAX_S = 1.0
ERP_BASELINE_S = (-0.2, 0.0)
HG_BASELINE_S = (-0.25, -0.05)
HG_BANDS_HZ = tuple((float(lo), float(lo + 10)) for lo in range(70, 150, 10))
LINE_NOISE_HZ = (50.0, 100.0, 150.0)

TASK1_TRIGGERS = (11, 12, 21, 22, 31, 32, 41, 42)
TASK2_MEMORY_RED = (123, 133)
TASK2_MEMORY_GREEN = (103, 113)
TASK3_RED = 51
TASK3_GREEN = 54
TASK3_CHROMATIC = (51, 52, 53, 54)
TASK3_ACHROMATIC = (55, 56)
TASK2_TRUE_FALSE = (101, 102, 111, 112, 121, 122, 131, 132)

RANDOM_SEED = 20260720
N_SPLITS = 5
FAST_N_PERMUTATIONS = 100
FULL_N_PERMUTATIONS = 1000
PERMUTATION_MODE = "fast"
N_PERMUTATIONS = FAST_N_PERMUTATIONS if PERMUTATION_MODE == "fast" else FULL_N_PERMUTATIONS
# Four workers keeps permutation decoding below the workstation's CPU/memory budget.
N_JOBS = 4
WINDOW_MS = 50.0
STEP_MS = 10.0

# Confirmed exclusions are applied before rereferencing. F15 was repeatedly
# high-variance in sampled QC and is also an endpoint; it must not reference F14.
BAD_CHANNELS = {
    "test001": ("F15",),
    # Repeated high-variance channels in at least two independent ERP runs.
    "test003": ("C13", "D5", "H7", "I1", "I2", "I3"),
    "test002": ("A8", "G7"),
    "test005": ("C13", "C14", "F7", "I10"),
}
BAD_EPOCH_ROBUST_Z = 6.0
BAD_EPOCH_CHANNEL_FRACTION = 0.15

# The 20-mm batch is kept separate from the earlier 10-mm exploratory run.
BATCH_INTERMEDIATE_ROOT = ANALYSE_ROOT / "intermediate_20mm"
BATCH_RESULT_ROOT = ANALYSE_ROOT / "result_20mm"
LOCATION_FILES = {
    "test001": PROJECT_ROOT / "processed_data/test001/test001_ieegloc.xlsx",
    "test002": PROJECT_ROOT / "processed_data/test002/test002_ieegloc.xlsx",
    "test003": PROJECT_ROOT / "processed_data/test003/test003_ieegloc.xlsx",
    "test004": None,
    "test005": PROJECT_ROOT / "processed_data/test005/test005.tsv",
    "test006": PROJECT_ROOT / "processed_data/test006/test006_ieegloc.xlsx",
}
WHOLE_SUBJECTS = ["test001", "test002", "test003", "test005", "test006"]
WHOLE_EXCLUDED = {"test004": "excluded_localization_failure"}
ALL_INTERMEDIATE_ROOT = ANALYSE_ROOT / "intermediate_all_channels"
ALL_RESULT_ROOT = ANALYSE_ROOT / "result_all_channels"
SCREEN_N_PERMUTATIONS = 100
PSEUDO_TRIAL_SIZE = 5
PSEUDO_REPETITIONS = 50


def subject_raw_dir(subject: str) -> Path:
    return RAW_ROOT / RAW_SUBJECT_DIR[subject]


def subject_result_dir(subject: str) -> Path:
    return RESULT_ROOT / subject


def batch_subject_dir(subject: str) -> Path:
    return BATCH_RESULT_ROOT / "subjects" / subject


def all_subject_dir(subject: str) -> Path:
    return ALL_RESULT_ROOT / "subjects" / subject


def ensure_output_dirs() -> None:
    for path in (INTERMEDIATE_ROOT, RESULT_ROOT, REPORT_ROOT):
        path.mkdir(parents=True, exist_ok=True)
    for subject in SUBJECTS:
        for child in ("preprocessing", "localizer", "task3", "task2", "cross_task", "figures"):
            (subject_result_dir(subject) / child).mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "group" / "tables").mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "group" / "figures").mkdir(parents=True, exist_ok=True)
    for subject in SUBJECTS:
        for child in ("coverage", "preprocessing", "task1", "task2", "task3", "cross_task", "figures"):
            (batch_subject_dir(subject) / child).mkdir(parents=True, exist_ok=True)
        (BATCH_INTERMEDIATE_ROOT / subject / "preprocessing").mkdir(parents=True, exist_ok=True)
    for child in ("tables", "figures"):
        (BATCH_RESULT_ROOT / "group" / child).mkdir(parents=True, exist_ok=True)
    for subject in WHOLE_SUBJECTS + list(WHOLE_EXCLUDED):
        for child in ("preprocessing", "color_select", "decoding", "spatial_groups", "figures"):
            (all_subject_dir(subject) / child).mkdir(parents=True, exist_ok=True)
        (ALL_INTERMEDIATE_ROOT / subject / "preprocessing").mkdir(parents=True, exist_ok=True)
    for child in ("decoding", "figures", "tables", "spatial_groups"):
        (ALL_RESULT_ROOT / "virtual_subject" / child).mkdir(parents=True, exist_ok=True)
        (ALL_RESULT_ROOT / "group" / child).mkdir(parents=True, exist_ok=True)
