#!/usr/bin/env python3
"""Validate and project the one cross-repository dependency lock."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Mapping


SCHEMA = "flyto.stack-lock.v1"
DEPENDENCIES = ("flyto-blueprint", "flyto-core", "flyto-indexer")
_SHA = re.compile(r"^[a-f0-9]{40}$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def load_stack_lock(path: Path) -> Dict[str, Dict[str, str]]:
    """Read the closed manifest or raise ``ValueError``."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("stack lock is unreadable") from exc
    if not isinstance(value, Mapping) or set(value) != {"schema", "dependencies"}:
        raise ValueError("stack lock has unsupported fields")
    if value.get("schema") != SCHEMA:
        raise ValueError("stack lock schema is unsupported")
    dependencies = value.get("dependencies")
    if not isinstance(dependencies, Mapping) or set(dependencies) != set(DEPENDENCIES):
        raise ValueError("stack lock dependency set is incomplete")
    result: Dict[str, Dict[str, str]] = {}
    for name in DEPENDENCIES:
        item = dependencies.get(name)
        if not isinstance(item, Mapping) or set(item) != {"repository", "path", "revision"}:
            raise ValueError("stack lock dependency entry is malformed")
        repository = item.get("repository")
        relative = item.get("path")
        revision = item.get("revision")
        if not isinstance(repository, str) or not _REPOSITORY.fullmatch(repository):
            raise ValueError("stack lock repository is invalid")
        if relative != name:
            raise ValueError("stack lock path must match its dependency name")
        if not isinstance(revision, str) or not _SHA.fullmatch(revision):
            raise ValueError("stack lock revision must be a full Git SHA-1")
        result[name] = {
            "repository": repository,
            "path": relative,
            "revision": revision,
        }
    return result


def verify_workspace(dependencies: Mapping[str, Mapping[str, str]], parent: Path) -> None:
    """Require every checked-out sibling to equal the manifest revision."""

    for name in DEPENDENCIES:
        checkout = parent / dependencies[name]["path"]
        if not checkout.is_dir():
            raise ValueError("locked dependency checkout is missing: {}".format(name))
        completed = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
        if completed.returncode != 0 or completed.stdout.strip() != dependencies[name]["revision"]:
            raise ValueError("locked dependency revision mismatch: {}".format(name))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lock", default=str(Path(__file__).resolve().parents[1] / "stack-lock.json"),
    )
    parser.add_argument("--workspace-parent")
    parser.add_argument("--github-output")
    args = parser.parse_args(argv)
    try:
        dependencies = load_stack_lock(Path(args.lock))
        if args.workspace_parent:
            verify_workspace(dependencies, Path(args.workspace_parent).resolve())
        if args.github_output:
            output = Path(args.github_output)
            with output.open("a", encoding="utf-8") as stream:
                for name in DEPENDENCIES:
                    key = name.replace("-", "_")
                    item = dependencies[name]
                    stream.write("{}_repository={}\n".format(key, item["repository"]))
                    stream.write("{}_path={}\n".format(key, item["path"]))
                    stream.write("{}_revision={}\n".format(key, item["revision"]))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print("stack-lock: {}".format(exc), file=sys.stderr)
        return 1
    print("stack-lock: {} pinned dependencies verified".format(len(dependencies)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
