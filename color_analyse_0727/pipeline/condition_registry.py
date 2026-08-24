"""Single source of truth for tasks, trigger codes, and condition names.

All downstream code must use exact trigger-code matching through this module.
In particular, Task 2 is stored with all 12 experimental conditions. Analysis
code can select the Gray, True, or False subset without losing information.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Condition:
    name: str
    trigger: str
    group: str
    fruit: str | None = None


def _conditions(*items: tuple[str, str, str, str | None]) -> tuple[Condition, ...]:
    return tuple(Condition(*item) for item in items)


TASK_CONDITIONS: dict[int, tuple[Condition, ...]] = {
    1: _conditions(
        ("face_color", "11", "color", None),
        ("face_gray", "12", "gray", None),
        ("object_color", "21", "color", None),
        ("object_gray", "22", "gray", None),
        ("body_color", "31", "color", None),
        ("body_gray", "32", "gray", None),
        ("place_color", "41", "color", None),
        ("place_gray", "42", "gray", None),
    ),
    2: _conditions(
        ("cabbage_true", "101", "true", "cabbage"),
        ("cabbage_false", "102", "false", "cabbage"),
        ("cabbage_gray", "103", "gray", "cabbage"),
        ("kiwi_true", "111", "true", "kiwi"),
        ("kiwi_false", "112", "false", "kiwi"),
        ("kiwi_gray", "113", "gray", "kiwi"),
        ("strawberry_true", "121", "true", "strawberry"),
        ("strawberry_false", "122", "false", "strawberry"),
        ("strawberry_gray", "123", "gray", "strawberry"),
        ("watermelon_true", "131", "true", "watermelon"),
        ("watermelon_false", "132", "false", "watermelon"),
        ("watermelon_gray", "133", "gray", "watermelon"),
    ),
    3: _conditions(
        ("red", "51", "color", None),
        ("yellow", "52", "color", None),
        ("blue", "53", "color", None),
        ("green", "54", "color", None),
        ("black", "55", "color", None),
        ("white", "56", "color", None),
    ),
}


TASK_NAMES = {
    1: "passive_real_gray",
    2: "fruit_true_false_gray",
    3: "pure_color_patches",
}


TRIGGER_TO_CONDITION = {
    task: {item.trigger: item for item in conditions}
    for task, conditions in TASK_CONDITIONS.items()
}


def normalize_trigger(value: object) -> str:
    """Normalize EEGLAB trigger values to exact digit strings."""

    if value is None:
        return ""
    # EEGLAB v7.2 often stores an event type as a one-element MATLAB string
    # array. Convert nested arrays/lists before calling ``str``; otherwise
    # ``array(['Trigger-In:112'])`` becomes the literal text "['112']".
    while isinstance(value, (np.ndarray, list, tuple)):
        if len(value) == 0:
            return ""
        value = value.reshape(-1)[0] if isinstance(value, np.ndarray) else value[0]
    text = str(value).strip().upper()
    text = text.replace("TRIGGER-IN:", "").replace("TRIGGER-IN", "").strip()
    text = text.replace("EVENT:", "").strip()
    if re.fullmatch(r"[+-]?\d+\.0+", text):
        text = text.split(".", 1)[0]
    if re.fullmatch(r"[+-]?\d+", text):
        try:
            return str(int(text))
        except ValueError:
            return text
    return text


def condition_for_trigger(task_num: int, trigger: object) -> Condition | None:
    """Return the exact condition for a trigger, never using prefix matching."""

    return TRIGGER_TO_CONDITION.get(int(task_num), {}).get(normalize_trigger(trigger))


def conditions_for_group(task_num: int, group: str = "all") -> tuple[str, ...]:
    """Return canonical condition names for an analysis subset."""

    group = group.lower().strip()
    conditions = TASK_CONDITIONS[int(task_num)]
    if group in {"all", "any"}:
        return tuple(item.name for item in conditions)
    if group == "gray" and task_num == 2:
        return tuple(item.name for item in conditions if item.group == "gray")
    if group in {"true", "false"} and task_num == 2:
        return tuple(item.name for item in conditions if item.group == group)
    if group in {"color", "gray"} and task_num == 1:
        return tuple(item.name for item in conditions if item.group == group)
    raise ValueError(f"Unsupported task/group combination: task={task_num}, group={group}")


def event_inventory(task_num: int, triggers: Iterable[object]) -> dict[str, object]:
    """Summarize recognized and unexpected triggers for one recording."""

    counts = {name: 0 for name in conditions_for_group(task_num, "all")}
    unknown: dict[str, int] = {}
    for raw in triggers:
        trigger = normalize_trigger(raw)
        condition = condition_for_trigger(task_num, trigger)
        if condition is None:
            unknown[trigger] = unknown.get(trigger, 0) + 1
        else:
            counts[condition.name] += 1
    return {
        "task_num": int(task_num),
        "recognized_counts": counts,
        "unknown_counts": unknown,
        "recognized_total": sum(counts.values()),
        "unknown_total": sum(unknown.values()),
    }


def registry_rows() -> list[dict[str, str | int | None]]:
    """Return a flat, serialization-friendly registry table."""

    rows: list[dict[str, str | int | None]] = []
    for task_num, conditions in TASK_CONDITIONS.items():
        for order, condition in enumerate(conditions, start=1):
            rows.append(
                {
                    "task_num": task_num,
                    "task_name": TASK_NAMES[task_num],
                    "order": order,
                    "condition": condition.name,
                    "trigger": condition.trigger,
                    "group": condition.group,
                    "fruit": condition.fruit,
                }
            )
    return rows
