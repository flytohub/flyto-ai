# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Tests for three-layer system prompt architecture."""
from flyto_ai.prompt.system_prompt import (
    build_system_prompt,
    detect_language,
    LAYER_B_EXECUTE,
    LAYER_B_YAML,
    DEFAULT_SYSTEM_PROMPT,
    EXECUTE_SYSTEM_PROMPT,
    LANGUAGE_POLICY,
    FAILURE_POLICY,
    _VALID_MODES,
)


class TestLayerA:
    """Layer A (POLICY) must be present in every mode."""

    def test_layer_a_always_present(self):
        for mode in ("execute", "yaml"):
            prompt = build_system_prompt(module_count=300, mode=mode)
            assert "POLICY" in prompt
            assert "Language" in prompt
            assert "Safety" in prompt

    def test_never_invent_modules_rule(self):
        prompt = build_system_prompt(module_count=300)
        assert "NEVER" in prompt
        assert "module" in prompt.lower()

    def test_never_guess_selectors_rule(self):
        prompt = build_system_prompt(module_count=300)
        assert "NEVER guess CSS selectors" in prompt

    def test_env_var_safety(self):
        prompt = build_system_prompt(module_count=300)
        assert "env.VAR_NAME" in prompt


class TestLayerB:
    """Layer B varies by mode."""

    def test_execute_has_discovery(self):
        prompt = build_system_prompt(module_count=300, mode="execute")
        assert "DISCOVERY" in prompt
        assert "search_modules" in prompt

    def test_execute_has_schema_gate(self):
        prompt = build_system_prompt(module_count=300, mode="execute")
        assert "get_module_info" in prompt

    def test_yaml_has_discovery(self):
        prompt = build_system_prompt(module_count=300, mode="yaml")
        assert "DISCOVERY" in prompt
        assert "search_modules" in prompt

    def test_yaml_has_generation(self):
        prompt = build_system_prompt(module_count=300, mode="yaml")
        assert "GENERATION" in prompt
        assert "validate_params" in prompt

    def test_toolless_has_todo(self):
        prompt = build_system_prompt(module_count=300, has_tools=False)
        assert "TODO" in prompt

    def test_mode_execute_not_yaml_generation(self):
        prompt = build_system_prompt(module_count=300, mode="execute")
        assert "EXECUTE" in prompt
        # yaml mode has GENERATION section, execute mode does not
        assert "GENERATION" not in prompt

    def test_mode_yaml_not_execute(self):
        prompt = build_system_prompt(module_count=300, mode="yaml")
        assert "GENERATION" in prompt

    def test_module_count_placeholder(self):
        prompt = build_system_prompt(module_count=350)
        assert "350+" in prompt

    def test_unknown_mode_falls_back_to_yaml(self):
        prompt = build_system_prompt(module_count=300, mode="unknown")
        assert "GENERATION" in prompt

    def test_no_hardcoded_module_names(self):
        """System prompt must NOT contain specific module names."""
        for mode in ("execute", "yaml"):
            prompt = build_system_prompt(module_count=300, mode=mode)
            assert "browser.launch" not in prompt
            assert "browser.goto" not in prompt
            assert "browser.snapshot" not in prompt
            assert "browser.close" not in prompt
            assert "core.api.google_search" not in prompt
            assert "core.api.serpapi_search" not in prompt
            assert "core.api.http_get" not in prompt
            assert "string.uppercase" not in prompt
            assert "image.resize" not in prompt

    def test_no_hardcoded_sites(self):
        """System prompt must NOT contain specific website names."""
        for mode in ("execute", "yaml"):
            prompt = build_system_prompt(module_count=300, mode=mode)
            assert "tixcraft" not in prompt
            assert "Jay Chou" not in prompt
            assert "google.com" not in prompt
            assert "GOOGLE_API_KEY" not in prompt
            assert "SERPAPI_KEY" not in prompt


class TestLayerC:
    """Layer C (GATES) must be present in every mode."""

    def test_layer_c_always_present(self):
        for mode in ("execute", "yaml"):
            prompt = build_system_prompt(module_count=300, mode=mode)
            assert "QUALITY GATES" in prompt
            assert "Evidence Rule" in prompt

    def test_yaml_structure_rules(self):
        prompt = build_system_prompt(module_count=300)
        assert "name, steps[]" in prompt
        assert "snake_case" in prompt

    def test_evidence_rule_content(self):
        prompt = build_system_prompt(module_count=300)
        assert "params_schema" in prompt


class TestBlueprintFirst:
    """Blueprint-first instructions MUST appear in execute/yaml prompts."""

    def test_blueprint_in_prompt(self):
        for mode in ("execute", "yaml"):
            prompt = build_system_prompt(module_count=300, mode=mode)
            assert "list_blueprints" in prompt
            assert "use_blueprint" in prompt

        # Toolless mode should NOT have blueprint instructions
        prompt = build_system_prompt(module_count=300, has_tools=False)
        assert "use_blueprint" not in prompt

    def test_no_tool_list_in_prompt(self):
        for mode in ("execute", "yaml"):
            prompt = build_system_prompt(module_count=300, mode=mode)
            assert "## Available tools:" not in prompt


class TestBackwardCompat:
    """Backward-compatible aliases still work."""

    def test_default_prompt_alias(self):
        assert DEFAULT_SYSTEM_PROMPT is LAYER_B_YAML

    def test_execute_prompt_alias(self):
        assert EXECUTE_SYSTEM_PROMPT is LAYER_B_EXECUTE

    def test_language_policy_exported(self):
        assert "Language" in LANGUAGE_POLICY

    def test_failure_policy_exported(self):
        assert "Failure" in FAILURE_POLICY

    def test_valid_modes_include_execute_yaml(self):
        assert "execute" in _VALID_MODES
        assert "yaml" in _VALID_MODES


class TestDetectLanguage:
    """Deterministic language detection from user message."""

    def test_english(self):
        assert detect_language("Hello, how are you?") == "English"

    def test_traditional_chinese(self):
        assert detect_language("你好嗎？這個功能很棒") == "Traditional Chinese (zh-TW)"

    def test_simplified_chinese(self):
        assert detect_language("你好吗这个功能很棒") == "Simplified Chinese (zh-CN)"

    def test_japanese(self):
        assert detect_language("こんにちは") == "Japanese (ja)"

    def test_korean(self):
        assert detect_language("안녕하세요") == "Korean (ko)"

    def test_short_text_defaults_english(self):
        assert detect_language("hi") == "English"

    def test_empty_defaults_english(self):
        assert detect_language("") == "English"


class TestDetectLanguageEndToEnd:
    """Language detection integrates with prompt building."""

    def test_reply_language_override(self):
        prompt = build_system_prompt(module_count=300, reply_language="Japanese (ja)")
        assert "Japanese" in prompt

    def test_policy_still_present_with_override(self):
        lang = detect_language("Search something")
        prompt = build_system_prompt(module_count=300, reply_language=lang)
        assert "POLICY" in prompt
        assert "DISCOVERY" in prompt
        assert "QUALITY GATES" in prompt


class TestReplyLanguageInjection:
    """Reply language override is prepended correctly."""

    def test_forced_language_prepended(self):
        prompt = build_system_prompt(module_count=300, reply_language="French (fr)")
        assert prompt.startswith("⛔ REPLY IN French (fr)")

    def test_no_override_no_prefix(self):
        prompt = build_system_prompt(module_count=300)
        assert not prompt.startswith("⛔ REPLY IN")
