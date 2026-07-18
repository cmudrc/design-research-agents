from __future__ import annotations

from design_research_agents._implementations._shared._agent_internal import _code_action_step_runner as runner


def test_calculator_expression_resolution_normalizes_deduplicates_and_filters_candidates() -> None:
    assert runner._resolve_calculator_expressions(
        prompt="Compare (2 + (3 * 4)) with (5 / 2).",
        input_payload={"expressions": [" 1+1 ", 3, "words"], "expression": "1+1"},
    ) == ["1+1", "(2 + (3 * 4))", "(5 / 2)"]
    assert runner._resolve_calculator_expressions(prompt="What is 6 * 7?", input_payload={}) == ["6 * 7"]


def test_parenthesized_expression_parser_handles_nested_unbalanced_and_adjacent_groups() -> None:
    assert runner._extract_parenthesized_expressions("a (1 + (2 * 3)) b (4/2)") == [
        "(1 + (2 * 3))",
        "(4/2)",
    ]
    assert runner._extract_parenthesized_expressions("ignored ) and (unfinished") == []


def test_arithmetic_expression_normalizer_rejects_empty_text_words_and_unsafe_syntax() -> None:
    assert runner._normalize_arithmetic_expression(" ; ") is None
    assert runner._normalize_arithmetic_expression("123") is None
    assert runner._normalize_arithmetic_expression("a + b") is None
    assert runner._normalize_arithmetic_expression("2 + os.system('x')") is None
    assert runner._normalize_arithmetic_expression(" (2 + 3), ") == "(2 + 3)"
