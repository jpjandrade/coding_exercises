"""
EXERCISE 1: Parse LLM API Responses and Extract Structured Data
================================================================

PROBLEM:
Implement a function that extracts structured data from LLM responses. The model
might return JSON in various formats:
- Clean JSON
- JSON wrapped in markdown code blocks (```json ... ```)
- JSON with trailing commas or minor formatting issues
- Multiple JSON objects (return the first valid one)

You should also handle common failure modes gracefully.
"""

import json
from typing import Any, Optional
from dataclasses import dataclass


@dataclass
class ParseResult:
    """Result of parsing an LLM response."""

    success: bool
    data: Any  # Can be dict, list, or other JSON types; None when success=False
    error: Optional[str]
    raw_json: Optional[str]  # The extracted JSON string before parsing


OPENING_PIECES = {"{", "["}
CLOSING_PIECES = {
    "}": "{",
    "]": "[",
}


def _find_closing_brace(input_string, i):
    j = i
    stack = [input_string[i]]
    n = len(input_string)
    while j < n - 1 and stack:
        j += 1

        if j >= n:  # Force end of loop.
            continue

        next_char = input_string[j]
        if stack[-1] == '"':  # We're in a string
            if next_char == "\\":
                j += 1  # Skip escaped character
            if next_char == '"':
                stack.pop()
            continue

        if next_char == '"':
            stack.append('"')

        if next_char in CLOSING_PIECES:
            if CLOSING_PIECES[next_char] != stack[-1]:
                return
            stack.pop()

        if next_char in OPENING_PIECES:
            stack.append(next_char)

    if not stack:
        return j


def _find_first_valid_json(input_string: str) -> str:
    n = len(input_string)

    for i in range(n):
        # We're not dealing with escaped brackets.
        if input_string[i] in ["{", "["]:
            j = _find_closing_brace(input_string, i)

            if j:
                candidate_string = input_string[i : j + 1]
                try:
                    json.loads(candidate_string)
                except json.JSONDecodeError:
                    pass
                else:
                    return candidate_string

    return ""


def extract_json_from_response(response: str) -> ParseResult:
    """
    Extract and parse JSON from an LLM response.

    The response might contain:
    - Pure JSON
    - JSON in markdown code blocks (```json ... ``` or ``` ... ```)
    - JSON with some text before/after
    - Multiple JSON objects (extract the first valid one)

    Args:
        response: Raw string response from an LLM

    Returns:
        ParseResult with success status, parsed data, and any error message

    Examples:
        >>> result = extract_json_from_response('{"name": "Alice", "age": 30}')
        >>> result.success
        True
        >>> result.data
        {'name': 'Alice', 'age': 30}

        >>> result = extract_json_from_response('Here is the data: ```json\n{"x": 1}\n```')
        >>> result.data
        {'x': 1}
    """

    json_string: str = _find_first_valid_json(response)

    if not json_string:
        return ParseResult(
            raw_json=response,
            success=False,
            data=None,
            error="Couldn't find valid json in string.",
        )

    try:
        json_obj = json.loads(json_string)
    except json.JSONDecodeError as e:
        return ParseResult(raw_json=response, success=False, data=None, error=e.msg)

    return ParseResult(raw_json=response, success=True, data=json_obj, error=None)


def extract_field(response: str, field: str, default: Any = None) -> Any:
    """
    Convenience function to extract a specific field from a JSON response.

    Args:
        response: Raw string response from an LLM
        field: The field name to extract
        default: Default value if field not found or parsing fails

    Returns:
        The field value or default

    Example:
        >>> extract_field('{"status": "success", "count": 42}', "count")
        42
        >>> extract_field('{"status": "success"}', "count", default=0)
        0
    """
    response_json = extract_json_from_response(response)

    if not response_json.success or response_json.data is None:
        return default

    return response_json.data.get(field, default)
