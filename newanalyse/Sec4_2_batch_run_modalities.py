import json
import os
import subprocess
import time
from pathlib import Path

from batch_runner_utils import apply_python_overrides, load_python_module, write_runtime_config_file
from newanalyse_paths import project_root


BASE_PATH = project_root()
NEWANALYSE_DIR = BASE_PATH / 'newanalyse'

MATLAB_BIN = 'matlab'
STOP_ON_ERROR = True
DRY_RUN = False

RUN_PREPROCESS = False
RUN_DECODING = True

SUBJECTS = ['test003']
CONFIG_PATH = None
KEEP_RUNTIME_CONFIG_FILES = False

GLOBAL_OVERRIDES = {
"run_permutation_test": True,
"n_perms": 200,
"time_smooth_win": 10
}

STEP_OVERRIDES = {}

RUNTIME_CONFIG_DIR = NEWANALYSE_DIR / '.runtime_configs'

PREPROCESS_STEPS = [
    {
        'enabled': True,
        'kind': 'matlab',
        'name': 'ERP ROI Features',
        'script': NEWANALYSE_DIR / 'Sec2_1_preprocess_erp.m',
        'entry': 'Sec2_1_preprocess_erp',
        'script_key': 'Sec2_1_preprocess_erp',
    },
    {
        'enabled': True,
        'kind': 'matlab',
        'name': 'High-Gamma ROI Features',
        'script': NEWANALYSE_DIR / 'Sec2_2_preprocess_highgamma.m',
        'entry': 'Sec2_2_preprocess_highgamma',
        'script_key': 'Sec2_2_preprocess_highgamma',
    },
    {
        'enabled': True,
        'kind': 'matlab',
        'name': 'Low-Gamma ROI Features',
        'script': NEWANALYSE_DIR / 'Sec2_3_preprocess_lowgamma.m',
        'entry': 'Sec2_3_preprocess_lowgamma',
        'script_key': 'Sec2_3_preprocess_lowgamma',
    },
    {
        'enabled': True,
        'kind': 'matlab',
        'name': 'TFA ROI Features',
        'script': NEWANALYSE_DIR / 'Sec2_4_preprocess_tfa.m',
        'entry': 'Sec2_4_preprocess_tfa',
        'script_key': 'Sec2_4_preprocess_tfa',
    },
    {
        'enabled': True,
        'kind': 'matlab',
        'name': 'Gamma Multiband ROI Features',
        'script': NEWANALYSE_DIR / 'Sec2_6_preprocess_gamma_multiband.m',
        'entry': 'Sec2_6_preprocess_gamma_multiband',
        'script_key': 'Sec2_6_preprocess_gamma_multiband',
    },
]

DECODING_STEPS = [
    {
        'enabled': True,
        'kind': 'python',
        'name': 'ERP Within Decoding',
        'script': NEWANALYSE_DIR / 'Sec3_1_all_roi_result_erp.py',
        'script_key': 'Sec3_1_all_roi_result_erp',
        'feature_subdir': 'decoding_erp_features',
    },
    {
        'enabled': True,
        'kind': 'python',
        'name': 'High-Gamma Within Decoding',
        'script': NEWANALYSE_DIR / 'Sec3_2_all_roi_result_high.py',
        'script_key': 'Sec3_2_all_roi_result_high',
        'feature_subdir': 'decoding_highgamma_features',
    },
    {
        'enabled': True,
        'kind': 'python',
        'name': 'Low-Gamma Within Decoding',
        'script': NEWANALYSE_DIR / 'Sec3_3_all_roi_result_low.py',
        'script_key': 'Sec3_3_all_roi_result_low',
        'feature_subdir': 'decoding_lowgamma_features',
    },
    {
        'enabled': False,
        'kind': 'python',
        'name': 'Cross Decoding',
        'script': NEWANALYSE_DIR / 'Sec3_7_all_roi_result_cross.py',
        'script_key': 'Sec3_7_all_roi_result_cross',
        'feature_subdir': 'decoding_lowgamma_features',
    },
    {
        'enabled': True,
        'kind': 'python',
        'name': 'TFA Within Decoding',
        'script': NEWANALYSE_DIR / 'Sec3_4_all_roi_result_tfa.py',
        'script_key': 'Sec3_4_all_roi_result_tfa',
        'feature_subdir': 'decoding_tfa_features',
    },
    {
        'enabled': True,
        'kind': 'python',
        'name': 'Gamma Multiband Within Decoding',
        'script': NEWANALYSE_DIR / 'Sec3_6_all_roi_result_gamma_multiband.py',
        'script_key': 'Sec3_6_all_roi_result_gamma_multiband',
        'feature_subdir': 'decoding_gamma_multiband_features',
    },
]


def format_seconds(seconds):
    return f'{seconds:.1f}s'


def read_external_config():
    if not CONFIG_PATH:
        return {}
    config_file = Path(CONFIG_PATH)
    if not config_file.is_file():
        raise FileNotFoundError(f'Batch config not found: {config_file}')
    return json.loads(config_file.read_text(encoding='utf-8'))


def merge_named_dicts(left, right):
    merged = dict(left)
    for key, value in right.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            updated = dict(merged[key])
            updated.update(value)
            merged[key] = updated
        else:
            merged[key] = value
    return merged


def normalize_subjects(value):
    if isinstance(value, (str, bytes)):
        return [str(value)]
    return [str(item) for item in value]


def step_key(step):
    return step.get('script_key', step['script'].stem)


def configure_steps(default_steps, overrides_by_step):
    steps = []
    for source in default_steps:
        step = dict(source)
        override = dict(overrides_by_step.get(step_key(step), {}))
        if 'enabled' in override:
            step['enabled'] = bool(override.pop('enabled'))
        step['runtime_overrides'] = override
        steps.append(step)
    return steps


def collect_settings():
    external = read_external_config()
    subjects = normalize_subjects(external.get('subjects', SUBJECTS))
    run_preprocess = bool(external.get('run_preprocess', RUN_PREPROCESS))
    run_decoding = bool(external.get('run_decoding', RUN_DECODING))
    dry_run = bool(external.get('dry_run', DRY_RUN))
    stop_on_error = bool(external.get('stop_on_error', STOP_ON_ERROR))
    keep_runtime = bool(external.get('keep_runtime_config_files', KEEP_RUNTIME_CONFIG_FILES))

    global_overrides = merge_named_dicts(GLOBAL_OVERRIDES, external.get('global_overrides', {}))
    step_overrides = merge_named_dicts(STEP_OVERRIDES, external.get('steps', {}))

    preprocess_steps = configure_steps(PREPROCESS_STEPS, step_overrides)
    decoding_steps = configure_steps(DECODING_STEPS, step_overrides)

    return {
        'subjects': subjects,
        'run_preprocess': run_preprocess,
        'run_decoding': run_decoding,
        'dry_run': dry_run,
        'stop_on_error': stop_on_error,
        'keep_runtime': keep_runtime,
        'global_overrides': global_overrides,
        'preprocess_steps': preprocess_steps,
        'decoding_steps': decoding_steps,
    }


def build_matlab_command(step):
    script_dir = step['script'].parent.as_posix().replace("'", "''")
    newanalyse_dir = NEWANALYSE_DIR.as_posix().replace("'", "''")
    entry = step['entry']
    batch_code = f"addpath('{script_dir}'); addpath('{newanalyse_dir}'); {entry};"
    return [MATLAB_BIN, '-batch', batch_code]


def build_subject_runtime_payload(subject, global_overrides, step_groups):
    payload = {
        'global': {'subject': subject},
        'matlab_defaults': dict(global_overrides),
        'steps': {},
    }
    for group in step_groups:
        for step in group:
            if step['kind'] != 'matlab':
                continue
            overrides = dict(global_overrides)
            overrides.update(step.get('runtime_overrides', {}))
            overrides['subject'] = subject
            payload['steps'][step_key(step)] = overrides
    return payload


def build_step_overrides(subject, global_overrides, step):
    overrides = dict(global_overrides)
    overrides.update(step.get('runtime_overrides', {}))
    overrides['subject'] = subject
    return overrides


def run_python_step(step, subject, overrides, dry_run):
    print(f"\n===== {step['name']} | subject={subject} =====")
    print(f"Script: {step['script']}")
    print(f"Overrides: {overrides}")
    if dry_run:
        print('DRY_RUN=True, skip execution.')
        return 0.0

    start_time = time.time()
    module = load_python_module(step['script'], module_name_prefix='sec4_2')
    apply_python_overrides(
        module,
        overrides=overrides,
        feature_subdir=step.get('feature_subdir'),
        subject=subject,
    )
    module.main()
    duration = time.time() - start_time
    print(f"Finished in {format_seconds(duration)}")
    return duration


def run_matlab_step(step, subject, overrides, runtime_config_path, dry_run):
    cmd = build_matlab_command(step)
    print(f"\n===== {step['name']} | subject={subject} =====")
    print(f"Script: {step['script']}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Overrides: {overrides}")
    if dry_run:
        print('DRY_RUN=True, skip execution.')
        return 0.0

    env = os.environ.copy()
    env['NEWANALYSE_USE_CONFIG'] = '1'
    env['NEWANALYSE_CONFIG_PATH'] = str(runtime_config_path)

    start_time = time.time()
    subprocess.run(cmd, cwd=BASE_PATH, env=env, check=True)
    duration = time.time() - start_time
    print(f"Finished in {format_seconds(duration)}")
    return duration


def run_group(group_name, steps, subject, settings, runtime_config_path):
    enabled_steps = [step for step in steps if step['enabled']]
    if not enabled_steps:
        print(f'{group_name} | subject={subject}: no enabled steps, skip.')
        return []

    print(f"\n######## {group_name} | subject={subject} ########")
    group_start = time.time()
    timings = []
    for step in enabled_steps:
        overrides = build_step_overrides(subject, settings['global_overrides'], step)
        try:
            if step['kind'] == 'python':
                duration = run_python_step(step, subject, overrides, settings['dry_run'])
            elif step['kind'] == 'matlab':
                duration = run_matlab_step(step, subject, overrides, runtime_config_path, settings['dry_run'])
            else:
                raise ValueError(f"Unsupported step kind: {step['kind']}")

            timings.append(
                {
                    'subject': subject,
                    'group': group_name,
                    'name': step['name'],
                    'duration': duration,
                }
            )
            print(f"[TIME] {step['name']}: {format_seconds(duration)}")
        except Exception as exc:
            print(f"[ERROR] {step['name']} failed for {subject}: {exc}")
            if settings['stop_on_error']:
                raise

    group_duration = time.time() - group_start
    print(f"[TIME] {group_name} total for {subject}: {format_seconds(group_duration)}")
    return timings


def print_timing_summary(all_timings, total_duration):
    print('\n######## TIMING SUMMARY ########')
    if not all_timings:
        print('No enabled steps were run.')
        print(f'Batch total: {format_seconds(total_duration)}')
        return

    for item in all_timings:
        print(f"{item['subject']} | {item['group']} | {item['name']}: {format_seconds(item['duration'])}")
    print(f'Batch total: {format_seconds(total_duration)}')


def main():
    settings = collect_settings()
    total_start = time.time()
    print(f'Batch run root: {BASE_PATH}')
    print(f"Subjects: {', '.join(settings['subjects'])}")
    all_timings = []

    for subject in settings['subjects']:
        runtime_payload = build_subject_runtime_payload(
            subject,
            settings['global_overrides'],
            [settings['preprocess_steps'], settings['decoding_steps']],
        )
        runtime_config_path = write_runtime_config_file(
            RUNTIME_CONFIG_DIR,
            runtime_payload,
            stem=f'sec4_2_{subject}',
        )

        try:
            if settings['run_preprocess']:
                all_timings.extend(
                    run_group('PREPROCESS', settings['preprocess_steps'], subject, settings, runtime_config_path)
                )
            else:
                print(f'PREPROCESS disabled for {subject}.')

            if settings['run_decoding']:
                all_timings.extend(
                    run_group('DECODING', settings['decoding_steps'], subject, settings, runtime_config_path)
                )
            else:
                print(f'DECODING disabled for {subject}.')
        finally:
            if not settings['keep_runtime'] and runtime_config_path.exists():
                runtime_config_path.unlink()

    total_duration = time.time() - total_start
    print_timing_summary(all_timings, total_duration)
    print(f"\nAll selected steps finished in {format_seconds(total_duration)}")


if __name__ == '__main__':
    main()
