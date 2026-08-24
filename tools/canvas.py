#!/usr/bin/env python3
"""Small, dependency-free, conflict-aware Obsidian Canvas patch utility."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or not isinstance(value.get("nodes"), list):
        raise ValueError(f"Not a valid Canvas JSON file: {path}")
    value.setdefault("edges", [])
    return value


def find_node(canvas: dict[str, Any], node_id: str) -> dict[str, Any]:
    for node in canvas["nodes"]:
        if node.get("id") == node_id:
            return node
    raise KeyError(f"Canvas node not found: {node_id}")


def atomic_write(path: Path, canvas: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(canvas, ensure_ascii=False, indent=2) + "\n"
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def guarded_update(path: Path, expected_sha: str | None, mutate) -> None:
    before = sha256(path)
    if expected_sha and before != expected_sha:
        raise RuntimeError("Canvas changed before the update; refusing to overwrite it")
    canvas = load(path)
    mutate(canvas)
    if sha256(path) != before:
        raise RuntimeError("Canvas changed while preparing the update; refusing to overwrite it")
    atomic_write(path, canvas)


def add_node(canvas: dict[str, Any], node: dict[str, Any]) -> None:
    if any(existing.get("id") == node.get("id") for existing in canvas["nodes"]):
        raise ValueError(f"Node already exists: {node.get('id')}")
    canvas["nodes"].append(node)


def add_edge(canvas: dict[str, Any], edge: dict[str, Any]) -> None:
    if any(existing.get("id") == edge.get("id") for existing in canvas["edges"]):
        raise ValueError(f"Edge already exists: {edge.get('id')}")
    canvas["edges"].append(edge)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect")
    inspect.add_argument("canvas", type=Path)

    update = sub.add_parser("update-text")
    update.add_argument("canvas", type=Path)
    update.add_argument("node_id")
    update.add_argument("text")
    update.add_argument("--expected-sha")

    attach = sub.add_parser("attach-image")
    attach.add_argument("canvas", type=Path)
    attach.add_argument("node_id")
    attach.add_argument("file")
    attach.add_argument("--expected-sha")

    args = parser.parse_args()
    if args.command == "inspect":
        canvas = load(args.canvas)
        print(json.dumps({"sha256": sha256(args.canvas), "nodes": len(canvas["nodes"]), "edges": len(canvas["edges"])}, ensure_ascii=False))
        return

    if args.command == "update-text":
        def mutate(canvas: dict[str, Any]) -> None:
            node = find_node(canvas, args.node_id)
            node["type"] = "text"
            node["text"] = args.text

        guarded_update(args.canvas, args.expected_sha, mutate)
        return

    if args.command == "attach-image":
        def mutate(canvas: dict[str, Any]) -> None:
            node = find_node(canvas, args.node_id)
            node["type"] = "file"
            node.pop("text", None)
            node["file"] = args.file.replace("\\", "/")

        guarded_update(args.canvas, args.expected_sha, mutate)
        return


if __name__ == "__main__":
    main()
