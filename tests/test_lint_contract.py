"""Project lint stays reproducible and retains the existing CI error checks."""

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_project_lint_selection_is_exactly_the_existing_ci_contract():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    selectors = re.findall(r"python -m ruff check --select ([A-Z0-9,]+) flyto_ai tests", workflow)
    assert len(selectors) == 1
    assert config["tool"]["ruff"]["lint"]["select"] == selectors[0].split(",")


def test_default_project_check_rejects_undefined_names_like_ci(tmp_path):
    # An unrelated working directory cannot make an SDK file inherit its rules.
    (tmp_path / "ruff.toml").write_text('[lint]\nselect = []\n')
    command = [sys.executable, "-m", "ruff", "check", "--output-format=json",
               "--stdin-filename", str(ROOT / "lint_contract_probe.py"), "-"]
    result = subprocess.run(command, cwd=tmp_path, input="print(missing_runtime_value)\n",
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 1
    assert {row["code"] for row in json.loads(result.stdout)} == {"F821"}
