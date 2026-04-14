import importlib.util
import json
import sys
import uuid
from pathlib import Path

import numpy as np

from newanalyse_paths import get_feature_dir


PYTHON_OVERRIDE_MAP = {
    'subject': 'SUBJECT',
    'base_path': 'BASE_PATH',
    'feature_kind': 'FEATURE_KIND',
    'n_splits': 'N_SPLITS',
    'n_repeats_real': 'N_REPEATS_REAL',
    'n_repeats_perm': 'N_REPEATS_PERM',
    'n_perms': 'N_PERMS',
    'time_smooth_win': 'TIME_SMOOTH_WIN',
    'decoding_step': 'DECODING_STEP',
    'random_state': 'RANDOM_STATE',
    'n_jobs': 'N_JOBS',
    'batch_name': 'BATCH_NAME',
    'roi_pattern': 'ROI_FILE_PATTERN',
    'max_electrodes': 'MAX_ELECTRODES',
    'run_permutation_test': 'RUN_PERMUTATION_TEST',
    'tasks': 'TASKS',
}


def load_python_module(script_path, module_name_prefix='newanalyse_batch'):
    script_path = Path(script_path)
    module_name = f"{module_name_prefix}_{script_path.stem}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    script_dir = str(script_path.parent)
    inserted = False
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
        inserted = True
    try:
        spec.loader.exec_module(module)
    finally:
        if inserted:
            sys.path.pop(0)
    return module


def build_feature_dir(base_path, feature_subdir, subject):
    return str(get_feature_dir(base_path, feature_subdir, subject))


def refresh_python_module_state(module, feature_subdir=None):
    if feature_subdir and hasattr(module, 'FEATURE_DIR') and hasattr(module, 'BASE_PATH') and hasattr(module, 'SUBJECT'):
        module.FEATURE_DIR = build_feature_dir(module.BASE_PATH, feature_subdir, module.SUBJECT)

    if all(hasattr(module, key) for key in ('T_START', 'T_END', 'N_POINTS')):
        module.TIMES = np.linspace(module.T_START, module.T_END, module.N_POINTS)
        if hasattr(module, 'DECODING_STEP'):
            module.PLOT_TIMES = module.TIMES[::module.DECODING_STEP]


def apply_python_overrides(module, overrides=None, feature_subdir=None, subject=None):
    merged = dict(overrides or {})
    if subject is not None:
        merged['subject'] = subject

    for key, value in merged.items():
        target = PYTHON_OVERRIDE_MAP.get(key, key)
        setattr(module, target, value)
        if key == 'roi_pattern' and hasattr(module, 'ROI_PATTERN'):
            setattr(module, 'ROI_PATTERN', value)

    refresh_python_module_state(module, feature_subdir=feature_subdir)
    return module


def write_runtime_config_file(config_dir, payload, stem='runtime_config'):
    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f'{stem}_{uuid.uuid4().hex[:8]}.json'
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return config_path
