# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Tests for three-layer system prompt architecture."""
from flyto_ai.prompt.system_prompt import (
    build_system_prompt,
    detect_language,
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


class TestDetectLanguage:
    """Deterministic language detection from user message."""

    # --- Basic single-language ---

    def test_english(self):
        assert detect_language("Help me search for Taylor Swift") == "English"

    def test_english_with_code(self):
        assert detect_language("Create a workflow for image.resize") == "English"

    def test_traditional_chinese(self):
        result = detect_language("幫我搜尋泰勒絲的相關資訊")
        assert "Traditional Chinese" in result

    def test_simplified_chinese(self):
        result = detect_language("帮我搜索泰勒丝的信息")
        assert "Chinese" in result

    def test_japanese(self):
        result = detect_language("テイラー・スウィフトを検索してください")
        assert "Japanese" in result

    def test_korean(self):
        result = detect_language("테일러 스위프트를 검색해 주세요")
        assert "Korean" in result

    def test_french(self):
        result = detect_language("Veuillez rechercher les dernières nouvelles sur le concert")
        assert "French" in result

    def test_spanish(self):
        result = detect_language("Por favor busca las últimas noticias sobre el concierto")
        assert "Spanish" in result

    def test_german(self):
        result = detect_language("Bitte suchen Sie nach den neuesten Nachrichten über das Konzert")
        assert "German" in result

    def test_russian(self):
        result = detect_language("Пожалуйста, найдите последние новости о концерте Тейлор Свифт")
        # langdetect may confuse Russian/Bulgarian (both Cyrillic) — either is acceptable
        assert "Russian" in result or "Bulgarian" in result

    # --- Mixed language (CJK + English) ---

    def test_mixed_chinese_english_mostly_chinese(self):
        """Chinese dominates → should detect Chinese."""
        result = detect_language("幫我搜尋 Taylor Swift")
        assert "Chinese" in result

    def test_mixed_chinese_english_mostly_english(self):
        """English dominates → should detect English."""
        result = detect_language("Search for Taylor Swift on Google right now please")
        assert result == "English"

    def test_japanese_with_kanji(self):
        """Japanese with kanji (CJK shared) → hiragana wins."""
        result = detect_language("東京タワーの近くのレストランを探して")
        assert "Japanese" in result

    def test_japanese_kanji_only(self):
        """Pure kanji without kana → detected as Chinese (expected ambiguity)."""
        result = detect_language("東京大学")
        assert "Chinese" in result  # no kana → regex falls to CJK

    def test_korean_with_english(self):
        result = detect_language("Taylor Swift 의 최신 앨범을 검색해 주세요")
        assert "Korean" in result

    # --- Short text edge cases ---

    def test_single_chinese_char(self):
        """Single character still gets detected."""
        result = detect_language("好")
        assert "Chinese" in result

    def test_single_english_word(self):
        """Short Latin text (<15 chars) → fallback to English."""
        assert detect_language("hello") == "English"

    def test_two_chinese_words(self):
        result = detect_language("搜尋泰勒絲")
        assert "Chinese" in result

    def test_single_japanese_word(self):
        result = detect_language("ありがとう")
        assert "Japanese" in result

    def test_single_korean_word(self):
        result = detect_language("감사합니다")
        assert "Korean" in result

    # --- Code / URL heavy inputs ---

    def test_mostly_url(self):
        """URL-heavy input with English words → English."""
        result = detect_language("Open https://www.google.com/search?q=test and extract results")
        assert result == "English"

    def test_code_snippet(self):
        result = detect_language("Run browser.goto then browser.extract to get the data")
        assert result == "English"

    def test_chinese_with_url(self):
        """Chinese instruction with URL → still Chinese."""
        result = detect_language("幫我打開 https://google.com 然後擷取資料")
        assert "Chinese" in result

    # --- Edge cases ---

    def test_empty_string(self):
        assert detect_language("") == "English"

    def test_whitespace_only(self):
        assert detect_language("   ") == "English"

    def test_numbers_only(self):
        """Pure numbers → fallback to English."""
        assert detect_language("12345") == "English"

    def test_emoji_only(self):
        """Emoji → fallback to English."""
        result = detect_language("👍🎉🔥")
        assert result == "English"

    def test_punctuation_only(self):
        result = detect_language("...")
        assert result == "English"

    # --- Determinism ---

    def test_repeated_calls_same_result(self):
        """Same input → same output, no randomness."""
        text = "幫我搜尋泰勒絲的相關資訊"
        results = [detect_language(text) for _ in range(10)]
        assert len(set(results)) == 1


class TestDetectLanguageEndToEnd:
    """Full pipeline: detect_language → build_system_prompt → verify injection."""

    def test_english_input_gets_english_prompt(self):
        lang = detect_language("Help me search for Taylor Swift")
        prompt = build_system_prompt(module_count=300, reply_language=lang)
        assert "REPLY IN English" in prompt

    def test_chinese_input_gets_chinese_prompt(self):
        lang = detect_language("幫我搜尋泰勒絲的相關資訊")
        prompt = build_system_prompt(module_count=300, reply_language=lang)
        assert "REPLY IN Traditional Chinese" in prompt

    def test_japanese_input_gets_japanese_prompt(self):
        lang = detect_language("テイラー・スウィフトを検索してください")
        prompt = build_system_prompt(module_count=300, reply_language=lang)
        assert "REPLY IN Japanese" in prompt

    def test_korean_input_gets_korean_prompt(self):
        lang = detect_language("테일러 스위프트를 검색해 주세요")
        prompt = build_system_prompt(module_count=300, reply_language=lang)
        assert "REPLY IN Korean" in prompt

    def test_override_appears_before_everything(self):
        """Language override is the very first line of the prompt."""
        lang = detect_language("幫我搜尋泰勒絲")
        prompt = build_system_prompt(module_count=300, reply_language=lang)
        first_line = prompt.split("\n")[0]
        assert "⛔ REPLY IN" in first_line

    def test_policy_still_present_with_override(self):
        lang = detect_language("Search something")
        prompt = build_system_prompt(module_count=300, reply_language=lang)
        assert "POLICY" in prompt
        assert "EXECUTION LOOP" in prompt
        assert "QUALITY GATES" in prompt


class TestReplyLanguageInjection:
    """reply_language parameter injects hard override at prompt top."""

    def test_english_override(self):
        prompt = build_system_prompt(module_count=300, reply_language="English")
        assert prompt.startswith("⛔ REPLY IN English")

    def test_chinese_override(self):
        prompt = build_system_prompt(
            module_count=300,
            reply_language="Traditional Chinese (zh-TW)",
        )
        assert "REPLY IN Traditional Chinese" in prompt

    def test_no_override_when_none(self):
        prompt = build_system_prompt(module_count=300, reply_language=None)
        assert not prompt.startswith("⛔ REPLY IN")

    def test_override_before_policy(self):
        prompt = build_system_prompt(module_count=300, reply_language="English")
        reply_pos = prompt.index("REPLY IN English")
        policy_pos = prompt.index("POLICY")
        assert reply_pos < policy_pos
