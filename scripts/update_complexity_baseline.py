#!/usr/bin/env python3
"""Rewrite the complexity baseline, downward only.

The baseline in `tests/complexity_baseline.json` is a debt register, and a debt
register that can be rewritten to whatever today happens to be is not a register
at all — it is a rubber stamp. So this refuses by default to raise any recorded
number or to add any entry that is not already there, and prints exactly what it
would have had to allow.

Paying debt down needs no flag:

    python scripts/update_complexity_baseline.py

Deliberately admitting new debt needs one, and shows up in review as a flag in
the diff rather than as a quietly larger number:

    python scripts/update_complexity_baseline.py --accept-new
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from test_complexity_budget import (  # noqa: E402
    BASELINE,
    PACKAGE,
    measure,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accept-new",
        action="store_true",
        help="Permit raising a recorded number or recording a new entry",
    )
    args = parser.parse_args()

    files, functions = measure(PACKAGE)
    current = {"files": files, "functions": functions}
    recorded = json.loads(BASELINE.read_text(encoding="utf-8"))

    raised = []
    for section in ("files", "functions"):
        for name, value in current[section].items():
            was = recorded[section].get(name)
            if was is None:
                raised.append(f"  new: {name} = {value}")
            elif value > was:
                raised.append(f"  worse: {name} = {value} (recorded {was})")

    if raised and not args.accept_new:
        print("refusing to write a baseline that admits new or worse debt:")
        print("\n".join(sorted(raised)))
        print("\nFix the code, or pass --accept-new to record this deliberately.")
        return 1

    cleared = [
        name
        for section in ("files", "functions")
        for name in recorded[section]
        if name not in current[section]
    ]

    BASELINE.write_text(
        json.dumps(
            {
                "files": dict(sorted(files.items())),
                "functions": dict(sorted(functions.items())),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"baseline written: {len(files)} files, {len(functions)} functions")
    for name in sorted(cleared):
        print(f"  cleared: {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
