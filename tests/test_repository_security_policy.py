"""Regression tests for GitHub repository security policy."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(relative_path: str) -> dict:
    return yaml.safe_load((ROOT / relative_path).read_text(encoding="utf-8"))


def test_ci_uses_least_privilege_default_permissions():
    workflow = _load_yaml(".github/workflows/ci.yml")

    assert workflow["permissions"] == {"contents": "read"}


def test_dependabot_suppresses_only_routine_version_prs():
    config = _load_yaml(".github/dependabot.yml")

    assert config["updates"]
    assert all(
        update["open-pull-requests-limit"] == 0 for update in config["updates"]
    )


def test_pypi_publish_action_uses_patched_pin():
    workflow = (
        ROOT / ".github/workflows/publish-pypi.yml"
    ).read_text(encoding="utf-8")

    assert (
        "pypa/gh-action-pypi-publish@"
        "ba38be9e461d3875417946c167d0b5f3d385a247"
    ) in workflow


def test_grype_ignore_is_exactly_scoped_to_patched_action_sha():
    config = _load_yaml(".grype.yaml")

    assert config == {
        "ignore": [
            {
                "vulnerability": "GHSA-vxmw-7h4f-hqxh",
                "package": {
                    "name": "pypa/gh-action-pypi-publish",
                    "version": "ba38be9e461d3875417946c167d0b5f3d385a247",
                    "type": "github-action",
                },
            }
        ]
    }
