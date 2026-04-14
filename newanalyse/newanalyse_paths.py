from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_KIND_BY_KEY = {
    'decoding_erp_features': 'erp',
    'decoding_highgamma_features': 'highgamma',
    'decoding_lowgamma_features': 'lowgamma',
    'decoding_tfa_features': 'tfa',
    'decoding_gamma_features': 'gamma',
    'decoding_gamma_multiband_features': 'gamma_multiband',
    'erp': 'erp',
    'highgamma': 'highgamma',
    'lowgamma': 'lowgamma',
    'tfa': 'tfa',
    'gamma': 'gamma',
    'gamma_multiband': 'gamma_multiband',
}


def project_root(base_path=None):
    if base_path is None:
        return PROJECT_ROOT
    return Path(base_path).resolve()


def analysis_code_root(base_path=None):
    return project_root(base_path) / 'newanalyse'


def feature_root(base_path=None):
    return project_root(base_path) / 'feature'


def result_root(base_path=None):
    return project_root(base_path) / 'result'


def sanitize_token(text):
    token = ''.join(ch if ch.isalnum() or ch in {'_', '-'} else '_' for ch in str(text))
    return token.strip('_') or 'Unknown'


def append_path_tokens(path, suffix):
    if suffix is None:
        return path
    parts = suffix if isinstance(suffix, (list, tuple)) else Path(str(suffix)).parts
    for part in parts:
        if not part or part == '/':
            continue
        path /= sanitize_token(part)
    return path


def resolve_feature_kind(feature_key):
    key = str(feature_key)
    return FEATURE_KIND_BY_KEY.get(key, key)


def get_feature_dir(base_path, feature_key, subject):
    return feature_root(base_path) / resolve_feature_kind(feature_key) / str(subject)


def get_within_decoding_batch_dir(base_path, data_type, subject, batch_name=None):
    path = result_root(base_path) / 'decoding' / '_batch' / sanitize_token(data_type) / str(subject)
    return append_path_tokens(path, batch_name)


def get_within_decoding_task_dir(base_path, task_id, data_type, subject, perm_tag, variant=None, batch_name=None):
    path = result_root(base_path) / 'decoding' / sanitize_token(task_id) / sanitize_token(data_type) / str(subject) / str(perm_tag)
    path = append_path_tokens(path.parent, batch_name) / str(perm_tag)
    if variant:
        path /= sanitize_token(variant)
    return path


def get_cross_decoding_batch_dir(base_path, data_type, subject, batch_name=None):
    path = result_root(base_path) / 'cross_decoding' / '_batch' / sanitize_token(data_type) / str(subject)
    return append_path_tokens(path, batch_name)


def get_cross_decoding_task_dir(base_path, task_id, data_type, subject, perm_tag, batch_name=None):
    path = result_root(base_path) / 'cross_decoding' / sanitize_token(task_id) / sanitize_token(data_type) / str(subject)
    path = append_path_tokens(path, batch_name)
    return path / str(perm_tag)


def get_roi_condition_tfa_dir(base_path, task, subject, comparison_id):
    return result_root(base_path) / 'roi_condition_tfa' / sanitize_token(task) / 'tfa' / str(subject) / sanitize_token(comparison_id)


def get_roi_electrode_condition_dir(base_path, task, data_type, subject, comparison_id):
    return result_root(base_path) / 'roi_electrode_condition' / sanitize_token(task) / sanitize_token(data_type) / str(subject) / sanitize_token(comparison_id)


def get_all_electrode_task_dir(base_path, task_id, data_type, subject, perm_tag, batch_name=None):
    path = result_root(base_path) / 'all_electrode_decoding' / sanitize_token(task_id) / sanitize_token(data_type) / str(subject)
    path = append_path_tokens(path, batch_name)
    return path / str(perm_tag)


def get_all_electrode_summary_path(base_path, data_type, subject, perm_tag, batch_name=None):
    path = result_root(base_path) / 'all_electrode_decoding' / '_summary' / sanitize_token(data_type) / str(subject)
    path = append_path_tokens(path, batch_name)
    return path / str(perm_tag) / 'summary.csv'


def get_smoothing_compare_task_dir(base_path, task_id, data_type, subject, smooth_tag, perm_tag='real_only', variant='with_sti'):
    path = result_root(base_path) / 'smoothing_compare' / sanitize_token(task_id) / sanitize_token(data_type) / str(subject) / sanitize_token(smooth_tag)
    if perm_tag:
        path /= str(perm_tag)
    if variant:
        path /= sanitize_token(variant)
    return path


def get_smoothing_compare_summary_dir(base_path):
    return result_root(base_path) / 'smoothing_compare' / '_summary'


def get_report_dir(base_path, report_name=None):
    root = result_root(base_path) / 'reports'
    if report_name:
        root /= sanitize_token(report_name)
    return root