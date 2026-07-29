import json
import os
from pathlib import Path


TRUE_VALUES = {"1", "true", "yes", "y", "on"}


def env_truthy(name, default=False):
    """把环境变量解析成布尔值。"""
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in TRUE_VALUES


def merge_dicts(base, extra):
    """浅层合并两个字典。"""
    merged = dict(base)
    if isinstance(extra, dict):
        merged.update(extra)
    return merged


def _get_step_override(steps, script_stem):
    """从运行时配置里提取当前脚本对应的 step 级覆盖项。"""
    if not isinstance(steps, dict):
        return {}
    if script_stem in steps and isinstance(steps[script_stem], dict):
        return steps[script_stem]
    script_name = f"{script_stem}.py"
    if script_name in steps and isinstance(steps[script_name], dict):
        return steps[script_name]
    return {}


def load_runtime_config(script_path=None, sections=()):
    """按环境变量指定的 JSON 文件加载运行时配置。

    这个实现与原项目保持接口兼容，
    但副本已经放在本地 code/ 目录中，不再依赖 newanalyse。
    """
    if not env_truthy("NEWANALYSE_USE_CONFIG", default=False):
        return {}

    config_path = os.environ.get("NEWANALYSE_CONFIG_PATH")
    if not config_path:
        return {}

    config_file = Path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"Runtime config not found: {config_file}")

    with config_file.open("r", encoding="utf-8") as handle:
        root = json.load(handle)

    merged = {}
    merged = merge_dicts(merged, root.get("global", {}))
    for section in sections:
        merged = merge_dicts(merged, root.get(section, {}))

    script_stem = Path(script_path).stem if script_path else ""
    if script_stem:
        merged = merge_dicts(merged, _get_step_override(root.get("steps", {}), script_stem))

    return merged
