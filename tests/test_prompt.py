# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Tests for three-layer system prompt architecture."""
from flyto_ai.prompt.system_prompt import (
    build_system_prompt,
    LAYER_A_POLICY,
    LAYER_B_EXECUTE,
    LAYER_B_YAML,
    LAYER_B_TOOLLESS,
    LAYER_C_GATES,
    # Backward-compatible aliases
    DEFAULT_SYSTEM_PROMPT,
    EXECUTE_SYSTEM_PROMPT,
    LANGUAGE_POLICY,
    FAILURE_POLICY,
    _VALID_MODES,
)


class TestLayerA:
    """Layer A (POLICY) must be present in every mode."""

    def test_layer_a_always_present(self):
        for mode in _VALID_MODES:
            prompt = build_system_prompt(module_count=300, mode=mode)
            assert "Output Contract" in prompt
            assert "Language" in prompt
            assert "On Failure" in prompt
            assert "Safety" in prompt

        # Toolless too
        prompt = build_system_prompt(module_count=300, has_tools=False)
        assert "Output Contract" in prompt
        assert "Safety" in prompt

    def test_never_invent_modules_rule(self):
        prompt = build_system_prompt(module_count=300)
        assert "NEVER invent module names" in prompt

    def test_never_guess_selectors_rule(self):
        prompt = build_system_prompt(module_count=300)
        assert "NEVER guess CSS selectors" in prompt

    def test_env_var_safety(self):
        prompt = build_system_prompt(module_count=300)
        assert "env.VAR_NAME" in prompt


class TestLayerB:
    """Layer B varies by mode."""

    def test_execute_has_execution_loop(self):
        prompt = build_system_prompt(module_count=300, mode="execute")
        assert "EXECUTION LOOP" in prompt
        assert "Browser Protocol" in prompt

    def test_execute_has_schema_gate(self):
        prompt = build_system_prompt(module_count=300, mode="execute")
        assert "get_module_info" in prompt
        assert "NEVER call execute_module" in prompt

    def test_yaml_has_yaml_loop(self):
        prompt = build_system_prompt(module_count=300, mode="yaml")
        assert "YAML GENERATION LOOP" in prompt
        assert "validate_params" in prompt

    def test_yaml_has_schema_gate(self):
        prompt = build_system_prompt(module_count=300, mode="yaml")
        assert "get_module_info" in prompt
        assert "NEVER put a module in YAML" in prompt

    def test_toolless_has_todo(self):
        prompt = build_system_prompt(module_count=300, has_tools=False)
        assert "TODO" in prompt

    def test_mode_execute_not_yaml(self):
        prompt = build_system_prompt(module_count=300, mode="execute")
        assert "EXECUTE" in prompt
        assert "YAML GENERATION LOOP" not in prompt

    def test_mode_yaml_not_execute(self):
        prompt = build_system_prompt(module_count=300, mode="yaml")
        assert "YAML GENERATION LOOP" in prompt
        assert "EXECUTION LOOP" not in prompt

    def test_module_count_placeholder(self):
        prompt = build_system_prompt(module_count=350)
        assert "350+" in prompt

    def test_unknown_mode_falls_back_to_yaml(self):
        prompt = build_system_prompt(module_count=300, mode="unknown")
        assert "YAML GENERATION LOOP" in prompt


class TestLayerC:
    """Layer C (GATES) must be present in every mode."""

    def test_layer_c_always_present(self):
        for mode in _VALID_MODES:
            prompt = build_system_prompt(module_count=300, mode=mode)
            assert "QUALITY GATES" in prompt
            assert "Evidence Rule" in prompt

        prompt = build_system_prompt(module_count=300, has_tools=False)
        assert "QUALITY GATES" in prompt

    def test_yaml_structure_rules(self):
        prompt = build_system_prompt(module_count=300)
        assert "name, steps[]" in prompt
        assert "snake_case" in prompt

    def test_evidence_rule_content(self):
        prompt = build_system_prompt(module_count=300)
        assert "params_schema" in prompt


class TestBlueprintRemoved:
    """Blueprint instructions must NOT appear in any prompt."""

    def test_no_blueprint_in_prompt(self):
        for mode in _VALID_MODES:
            prompt = build_system_prompt(module_count=300, mode=mode)
            assert "save_as_blueprint" not in prompt
            assert "list_blueprints() FIRST" not in prompt
            assert "use_blueprint" not in prompt
            assert "Blueprint Learning" not in prompt

        prompt = build_system_prompt(module_count=300, has_tools=False)
        assert "save_as_blueprint" not in prompt

    def test_no_tool_list_in_prompt(self):
        """Available tools section removed — tools are in function calling schema."""
        for mode in _VALID_MODES:
            prompt = build_system_prompt(module_count=300, mode=mode)
            assert "## Available tools:" not in prompt


class TestSchemaRule:
    """Schema-before-use is the core enforcement rule."""

    def test_schema_rule_in_execute(self):
        prompt = build_system_prompt(module_count=300, mode="execute")
        assert "get_module_info" in prompt

    def test_schema_rule_in_yaml(self):
        prompt = build_system_prompt(module_count=300, mode="yaml")
        assert "get_module_info" in prompt


class TestContextAndAdmin:
    """Context suffix and admin addition behavior unchanged."""

    def test_context_appended(self):
        prompt = build_system_prompt(
            module_count=300,
            context={"name": "My Workflow", "steps": [
                {"id": "s1", "module": "browser.click"},
            ]},
        )
        assert "My Workflow" in prompt
        assert "browser.*" in prompt

    def test_context_masks_module_category(self):
        prompt = build_system_prompt(
            module_count=300,
            context={"name": "Test", "steps": [
                {"id": "s1", "module": "secrets.get_key"},
            ]},
        )
        assert "secrets.*" in prompt
        assert "secrets.get_key" not in prompt

    def test_admin_addition(self):
        prompt = build_system_prompt(
            module_count=300,
            admin_addition="Always use formal language.",
        )
        assert "Admin Instructions" in prompt
        assert "formal language" in prompt

    def test_custom_template(self):
        prompt = build_system_prompt(
            module_count=50,
            template="You have {module_count} modules.",
        )
        assert "You have 50 modules." in prompt
        # Policy layer still present
        assert "Output Contract" in prompt


class TestBackwardCompat:
    """Old constant names still exported."""

    def test_aliases_exist(self):
        assert DEFAULT_SYSTEM_PROMPT is LAYER_B_YAML
        assert EXECUTE_SYSTEM_PROMPT is LAYER_B_EXECUTE

    def test_language_policy_exported(self):
        assert "Language" in LANGUAGE_POLICY

    def test_failure_policy_exported(self):
        assert "Failure" in FAILURE_POLICY

    def test_valid_modes_unchanged(self):
        assert _VALID_MODES == {"execute", "yaml"}
