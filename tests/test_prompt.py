# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Tests for system prompt rendering."""
from flyto_ai.prompt.system_prompt import build_system_prompt, DEFAULT_SYSTEM_PROMPT, EXECUTE_SYSTEM_PROMPT


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
        assert prompt == "You have 50 modules."

    def test_context_appended(self):
        prompt = build_system_prompt(
            module_count=300,
            context={"name": "My Workflow", "steps": [
                {"id": "s1", "module": "browser.click"},
            ]},
        )
        assert "My Workflow" in prompt
        assert "browser.click" in prompt

    def test_admin_addition(self):
        prompt = build_system_prompt(
            module_count=300,
            admin_addition="Always use formal language.",
        )
        assert "Admin Instructions" in prompt
        assert "formal language" in prompt

    def test_default_has_key_sections(self):
        assert "Blueprint Learning Rules" in DEFAULT_SYSTEM_PROMPT
        assert "report_blueprint_outcome" in DEFAULT_SYSTEM_PROMPT
        assert "inspect_page" in DEFAULT_SYSTEM_PROMPT
