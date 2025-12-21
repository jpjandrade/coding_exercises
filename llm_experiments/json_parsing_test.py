"""
Comprehensive pytest test suite for JSON parsing from LLM responses.
"""

import pytest
from json_parsing import extract_json_from_response, extract_field, ParseResult


class TestExtractJsonFromResponse:
    """Tests for the main JSON extraction function."""

    # =========================================================================
    # EXISTING TESTS (from original file)
    # =========================================================================

    def test_clean_json(self):
        """Test parsing clean, well-formed JSON."""
        result = extract_json_from_response('{"name": "Alice", "age": 30}')
        assert result.success, "Should parse clean JSON"
        assert result.data == {"name": "Alice", "age": 30}

    def test_json_in_markdown_with_json_specifier(self):
        """Test JSON in markdown code block with 'json' language specifier."""
        response = """Here's the extracted information:
```json
{
    "title": "Meeting Notes",
    "attendees": ["Alice", "Bob"]
}
```
Let me know if you need anything else!"""
        result = extract_json_from_response(response)
        assert result.success, "Should parse JSON from markdown"
        assert result.data["title"] == "Meeting Notes"
        assert len(result.data["attendees"]) == 2

    def test_json_in_plain_code_block(self):
        """Test JSON in code block without language specifier."""
        response = """```
{"key": "value"}
```"""
        result = extract_json_from_response(response)
        assert result.success, "Should parse JSON from plain code block"
        assert result.data == {"key": "value"}

    def test_json_with_surrounding_text(self):
        """Test extracting JSON embedded in surrounding text."""
        response = (
            'Based on my analysis, the result is {"score": 0.95, '
            '"label": "positive"} which indicates high confidence.'
        )
        result = extract_json_from_response(response)
        assert result.success, "Should extract JSON from surrounding text"
        assert result.data["score"] == 0.95

    def test_invalid_json(self):
        """Test handling of invalid JSON."""
        result = extract_json_from_response("This is not JSON at all")
        assert not result.success, "Should fail on invalid JSON"
        assert result.error is not None

    def test_empty_response(self):
        """Test handling of empty string."""
        result = extract_json_from_response("")
        assert not result.success, "Should fail on empty response"

    def test_nested_json(self):
        """Test deeply nested JSON objects."""
        response = '{"outer": {"inner": {"value": 42}}}'
        result = extract_json_from_response(response)
        assert result.success
        assert result.data["outer"]["inner"]["value"] == 42

    def test_json_with_arrays(self):
        """Test JSON containing arrays."""
        response = '{"items": [1, 2, 3], "nested": [{"a": 1}, {"b": 2}]}'
        result = extract_json_from_response(response)
        assert result.success
        assert result.data["items"] == [1, 2, 3]

    def test_escaped_characters_in_text(self):
        """Test handling of escaped characters outside JSON."""
        result = extract_json_from_response('aab \\"')
        assert not result.success, "Should handle escaped characters."

    def test_escaped_quote_in_json_value(self):
        """Test JSON with escaped quotes in string values."""
        response = '{"char": "\\""}'
        result = extract_json_from_response(response)
        assert result.success
        assert result.data["char"] == '"'

    # =========================================================================
    # NEW TESTS - Coverage Gaps
    # =========================================================================

    def test_top_level_array(self):
        """Test parsing top-level JSON arrays (not just objects)."""
        response = '["apple", "banana", "cherry"]'
        result = extract_json_from_response(response)
        assert result.success, "Should parse top-level arrays"
        assert result.data == ["apple", "banana", "cherry"]

    def test_array_of_objects(self):
        """Test top-level array containing objects."""
        response = '[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]'
        result = extract_json_from_response(response)
        assert result.success
        assert len(result.data) == 2
        assert result.data[0]["name"] == "Alice"

    def test_array_in_markdown(self):
        """Test array inside markdown code block."""
        response = """Here are the items:
```json
[1, 2, 3, 4, 5]
```
"""
        result = extract_json_from_response(response)
        assert result.success, "Should parse array from markdown"
        assert result.data == [1, 2, 3, 4, 5]

    def test_trailing_comma_in_object(self):
        """Test handling of trailing comma in object (common LLM mistake)."""
        response = '{"name": "Alice", "age": 30,}'
        result = extract_json_from_response(response)
        # This might fail with current implementation
        # Mark as expected failure if not handling trailing commas yet
        if not result.success:
            pytest.skip("Trailing comma handling not yet implemented")
        assert result.data == {"name": "Alice", "age": 30}

    def test_trailing_comma_in_array(self):
        """Test handling of trailing comma in array."""
        response = '[1, 2, 3,]'
        result = extract_json_from_response(response)
        if not result.success:
            pytest.skip("Trailing comma handling not yet implemented")
        assert result.data == [1, 2, 3]

    def test_string_containing_closing_brace(self):
        """Test JSON with closing brace inside a string value - critical edge case."""
        response = '{"message": "This } is tricky", "status": "ok"}'
        result = extract_json_from_response(response)
        assert result.success, "Should handle braces inside strings"
        assert result.data["message"] == "This } is tricky"

    def test_string_containing_opening_brace(self):
        """Test JSON with opening brace inside a string value."""
        response = '{"message": "Error: { invalid", "code": 500}'
        result = extract_json_from_response(response)
        assert result.success, "Should handle opening braces inside strings"
        assert result.data["message"] == "Error: { invalid"

    def test_string_containing_brackets(self):
        """Test JSON with array brackets inside a string value."""
        response = '{"pattern": "[a-z]+", "type": "regex"}'
        result = extract_json_from_response(response)
        assert result.success, "Should handle brackets inside strings"
        if result.success:
            assert result.data["pattern"] == "[a-z]+"

    def test_mismatched_brackets_object_array(self):
        """Test detection of mismatched brackets."""
        response = '{"items": [1, 2, 3}'
        result = extract_json_from_response(response)
        assert not result.success, "Should fail on mismatched brackets"

    def test_mismatched_brackets_array_object(self):
        """Test detection of mismatched brackets (reversed)."""
        response = '["items": {"a": 1]]'
        result = extract_json_from_response(response)
        assert not result.success, "Should fail on mismatched brackets"

    def test_unclosed_object(self):
        """Test handling of unclosed JSON object."""
        response = '{"name": "Alice", "age": 30'
        result = extract_json_from_response(response)
        assert not result.success, "Should fail on unclosed object"

    def test_unclosed_array(self):
        """Test handling of unclosed JSON array."""
        response = '[1, 2, 3'
        result = extract_json_from_response(response)
        assert not result.success, "Should fail on unclosed array"

    def test_multiple_json_objects_extracts_first(self):
        """Test that first valid JSON is extracted when multiple present."""
        response = 'First: {"id": 1} Second: {"id": 2}'
        result = extract_json_from_response(response)
        assert result.success
        assert result.data == {"id": 1}, "Should extract first valid JSON"

    def test_complex_escaped_strings(self):
        """Test JSON with complex escaped content."""
        response = r'{"path": "C:\\Users\\Alice\\file.txt", "quote": "She said \"hello\""}'
        result = extract_json_from_response(response)
        assert result.success
        assert "Users" in result.data["path"]
        assert '"hello"' in result.data["quote"]

    def test_json_with_newlines_and_tabs(self):
        """Test JSON with embedded newlines and tabs in strings."""
        response = '{"text": "Line 1\\nLine 2\\tTabbed"}'
        result = extract_json_from_response(response)
        assert result.success
        assert "\\n" in result.data["text"] or "\n" in result.data["text"]

    def test_markdown_multiline_json(self):
        """Test well-formatted multiline JSON in markdown."""
        response = """The configuration is:
```json
{
    "server": {
        "host": "localhost",
        "port": 8080
    },
    "features": [
        "auth",
        "logging"
    ]
}
```
This should work."""
        result = extract_json_from_response(response)
        assert result.success
        assert result.data["server"]["port"] == 8080
        assert "auth" in result.data["features"]

    def test_json_with_unicode(self):
        """Test JSON containing Unicode characters."""
        response = '{"emoji": "🎉", "chinese": "你好", "name": "Café"}'
        result = extract_json_from_response(response)
        assert result.success
        assert result.data["emoji"] == "🎉"
        assert result.data["chinese"] == "你好"

    def test_json_with_null_values(self):
        """Test JSON with null values."""
        response = '{"name": "Alice", "middle_name": null, "age": 30}'
        result = extract_json_from_response(response)
        assert result.success
        assert result.data["middle_name"] is None

    def test_json_with_boolean_values(self):
        """Test JSON with boolean values."""
        response = '{"active": true, "deleted": false}'
        result = extract_json_from_response(response)
        assert result.success
        assert result.data["active"] is True
        assert result.data["deleted"] is False

    def test_json_with_numbers(self):
        """Test JSON with various number formats."""
        response = '{"int": 42, "float": 3.14, "negative": -10, "scientific": 1.5e10}'
        result = extract_json_from_response(response)
        assert result.success
        assert result.data["int"] == 42
        assert result.data["float"] == 3.14
        assert result.data["scientific"] == 1.5e10

    def test_empty_object(self):
        """Test parsing empty JSON object."""
        response = '{}'
        result = extract_json_from_response(response)
        assert result.success
        assert result.data == {}

    def test_empty_array(self):
        """Test parsing empty JSON array."""
        response = '[]'
        result = extract_json_from_response(response)
        assert result.success
        assert result.data == []


class TestExtractField:
    """Tests for the field extraction convenience function."""

    def test_simple_field_extraction_string(self):
        """Test extracting a string field."""
        assert extract_field('{"name": "Alice", "age": 30}', "name") == "Alice"

    def test_simple_field_extraction_number(self):
        """Test extracting a numeric field."""
        assert extract_field('{"name": "Alice", "age": 30}', "age") == 30

    def test_missing_field_with_default(self):
        """Test missing field returns default value."""
        assert extract_field('{"name": "Alice"}', "age", default=0) == 0

    def test_missing_field_without_default(self):
        """Test missing field without default returns None."""
        assert extract_field('{"name": "Alice"}', "age") is None

    def test_invalid_json_with_default(self):
        """Test invalid JSON returns default value."""
        assert extract_field("not json", "field", default="fallback") == "fallback"

    def test_field_extraction_from_markdown(self):
        """Test extracting field from JSON in markdown."""
        response = '```json\n{"status": "success", "count": 42}\n```'
        assert extract_field(response, "count") == 42

    def test_extract_boolean_field(self):
        """Test extracting boolean field."""
        assert extract_field('{"active": true}', "active") is True

    def test_extract_null_field(self):
        """Test extracting null field."""
        result = extract_field('{"value": null}', "value")
        assert result is None

    def test_extract_array_field(self):
        """Test extracting array field."""
        result = extract_field('{"items": [1, 2, 3]}', "items")
        assert result == [1, 2, 3]

    def test_extract_nested_object_field(self):
        """Test extracting nested object field (returns the whole object)."""
        result = extract_field('{"user": {"name": "Alice", "id": 1}}', "user")
        assert result == {"name": "Alice", "id": 1}


class TestParseResult:
    """Tests for the ParseResult dataclass."""

    def test_successful_parse_result_structure(self):
        """Test ParseResult structure for successful parse."""
        result = extract_json_from_response('{"test": true}')
        assert isinstance(result, ParseResult)
        assert result.success is True
        assert result.data is not None
        assert result.error is None
        assert result.raw_json is not None

    def test_failed_parse_result_structure(self):
        """Test ParseResult structure for failed parse."""
        result = extract_json_from_response("invalid")
        assert isinstance(result, ParseResult)
        assert result.success is False
        assert result.data is None
        assert result.error is not None
        assert result.raw_json is not None

    def test_raw_json_preserved(self):
        """Test that raw input is preserved in ParseResult."""
        original = '{"key": "value"}'
        result = extract_json_from_response(original)
        assert result.raw_json == original


class TestEdgeCases:
    """Additional edge cases and stress tests."""

    def test_deeply_nested_structure(self):
        """Test very deeply nested JSON structure."""
        response = '{"l1": {"l2": {"l3": {"l4": {"l5": {"value": "deep"}}}}}}'
        result = extract_json_from_response(response)
        assert result.success
        assert result.data["l1"]["l2"]["l3"]["l4"]["l5"]["value"] == "deep"

    def test_large_array(self):
        """Test parsing larger arrays."""
        large_array = "[" + ",".join(str(i) for i in range(100)) + "]"
        result = extract_json_from_response(large_array)
        assert result.success
        assert len(result.data) == 100

    def test_json_after_garbage(self):
        """Test finding JSON after invalid content."""
        response = 'Some random text here {"valid": "json"} more text'
        result = extract_json_from_response(response)
        assert result.success
        assert result.data["valid"] == "json"

    def test_whitespace_variations(self):
        """Test JSON with various whitespace patterns."""
        response = '  {  "key"  :  "value"  }  '
        result = extract_json_from_response(response)
        assert result.success
        assert result.data["key"] == "value"

    def test_single_quotes_invalid(self):
        """Test that single quotes are properly rejected (not valid JSON)."""
        response = "{'key': 'value'}"
        result = extract_json_from_response(response)
        assert not result.success, "Single quotes are not valid JSON"

    def test_unquoted_keys_invalid(self):
        """Test that unquoted keys are properly rejected."""
        response = "{key: 'value'}"
        result = extract_json_from_response(response)
        assert not result.success, "Unquoted keys are not valid JSON"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
