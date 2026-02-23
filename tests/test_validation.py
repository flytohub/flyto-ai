# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Tests for YAML extraction."""
from flyto_ai.validation import extract_yaml_from_response


class TestExtractYaml:

    def test_extracts_yaml_block(self):
        text = 'Some text\n```yaml\nname: test\nsteps: []\n```\nMore text'
        result = extract_yaml_from_response(text)
        assert result is not None
        assert "name: test" in result

    def test_extracts_yml_block(self):
        text = '```yml\nfoo: bar\n```'
        result = extract_yaml_from_response(text)
        assert result is not None
        assert "foo: bar" in result

    def test_no_yaml_block(self):
        text = "Just some text without any code blocks"
        assert extract_yaml_from_response(text) is None

    def test_non_yaml_code_block(self):
        text = '```python\nprint("hello")\n```'
        assert extract_yaml_from_response(text) is None

    def test_multiple_blocks_returns_first(self):
        text = '```yaml\nfirst: 1\n```\ntext\n```yaml\nsecond: 2\n```'
        result = extract_yaml_from_response(text)
        assert "first: 1" in result

    def test_strips_whitespace(self):
        text = '```yaml\n  name: test  \n```'
        result = extract_yaml_from_response(text)
        assert result == "name: test"
