# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Tests for system prompt rendering."""
from flyto_ai.prompt.system_prompt import (
    build_system_prompt,
    DEFAULT_SYSTEM_PROMPT,
    EXECUTE_SYSTEM_PROMPT,
    LANGUAGE_POLICY,
    FAILURE_POLICY,
    _VALID_MODES,
)


class TestBuildSystemPrompt:

    def test_default_prompt(self):
        prompt = build_system_prompt(module_count=350)
        assert "350+" in prompt
        assert "Flyto AI Assistant" in prompt

    def test_custom_module_count(self):
        prompt = build_system_prompt(module_count=100)
        assert "100+" in prompt

    def test_custom_template(self):
        prompt = build_system_prompt(
            module_count=50,
            template="You have {module_count} modules.",
        )
        # Custom template still gets language/failure policies prepended
        assert "You have 50 modules." in prompt
        assert "Language Policy" in prompt

    def test_context_appended(self):
        prompt = build_system_prompt(
            module_count=300,
            context={"name": "My Workflow", "steps": [
                {"id": "s1", "module": "browser.click"},
            ]},
        )
        assert "My Workflow" in prompt
        # Context now only exposes category, not full module id
        assert "browser.*" in prompt

    def test_admin_addition(self):
        prompt = build_system_prompt(
            module_count=300,
            admin_addition="Always use formal language.",
        )
        assert "Admin Instructions" in prompt
        assert "formal language" in prompt

    def test_default_has_key_sections(self):
        assert "Blueprint Learning" in DEFAULT_SYSTEM_PROMPT
        assert "inspect_page" in DEFAULT_SYSTEM_PROMPT

    def test_language_policy_always_present(self):
        """Language policy is prepended to every prompt."""
        for mode in _VALID_MODES:
            prompt = build_system_prompt(module_count=300, mode=mode)
            assert "Language Policy" in prompt

        # Also for toolless
        prompt = build_system_prompt(module_count=300, has_tools=False)
        assert "Language Policy" in prompt

    def test_failure_policy_always_present(self):
        """Failure policy is prepended to every prompt."""
        prompt = build_system_prompt(module_count=300, mode="execute")
        assert "Failure Response Style" in prompt

    def test_execute_has_session_protocol(self):
        assert "Browser Session Protocol" in EXECUTE_SYSTEM_PROMPT

    def test_unknown_mode_falls_back_to_yaml(self):
        """Unknown mode falls back to yaml (DEFAULT_SYSTEM_PROMPT)."""
        prompt = build_system_prompt(module_count=300, mode="unknown")
        assert "ONLY generate" in prompt

    def test_context_masks_module_category(self):
        """Context suffix shows only category, not full module id."""
        prompt = build_system_prompt(
            module_count=300,
            context={"name": "Test", "steps": [
                {"id": "s1", "module": "secrets.get_key"},
            ]},
        )
        assert "secrets.*" in prompt
        assert "secrets.get_key" not in prompt

    def test_mode_execute_uses_execute_prompt(self):
        prompt = build_system_prompt(module_count=300, mode="execute")
        assert "EXECUTE" in prompt
        assert "ONLY generate" not in prompt

    def test_mode_yaml_uses_default_prompt(self):
        prompt = build_system_prompt(module_count=300, mode="yaml")
        assert "ONLY generate" in prompt
