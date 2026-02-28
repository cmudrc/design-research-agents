"""Design evaluation tools for decision and ranking workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from design_research_agents._contracts._tools import (
    ToolMetadata,
    ToolSideEffects,
    ToolSpec,
)
from design_research_agents.tools._sources._inprocess_source import InProcessToolSource


@dataclass(slots=True, frozen=True)
class _Criterion:
    """One weighted decision criterion."""

    name: str
    goal: str
    weight: float


@dataclass(slots=True, frozen=True)
class _Alternative:
    """One decision alternative with criterion scores."""

    alt_id: str
    scores: dict[str, float]


def register_evaluation_tools(source: InProcessToolSource) -> None:
    """Register design-evaluation tools.

    Args:
        source: In-process tool source receiving registrations.
    """
    metadata = ToolMetadata(
        source="core",
        side_effects=ToolSideEffects(),
        timeout_s=10,
        max_output_bytes=131_072,
        risky=False,
    )
    source.register_tool(
        spec=ToolSpec(
            name="eval.decision_matrix",
            description=(
                "Score and rank alternatives against weighted criteria with optional goal-aware normalization."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "alternatives": {"type": "array", "items": {"type": "object"}},
                    "criteria": {"type": "array", "items": {"type": "object"}},
                    "normalize": {"type": "boolean"},
                },
                "required": ["alternatives", "criteria"],
            },
            output_schema={"type": "object"},
            metadata=metadata,
        ),
        handler=_decision_matrix_handler,
    )
    source.register_tool(
        spec=ToolSpec(
            name="eval.pairwise_rank",
            description="Rank alternatives from pairwise outcomes using Copeland scoring (wins - losses + 0.5*ties).",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "alternatives": {
                        "type": "array",
                        "items": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "object"},
                            ]
                        },
                    },
                    "comparisons": {"type": "array", "items": {"type": "object"}},
                    "method": {"type": "string"},
                },
                "required": ["alternatives", "comparisons"],
            },
            output_schema={"type": "object"},
            metadata=metadata,
        ),
        handler=_pairwise_rank_handler,
    )


def _decision_matrix_handler(
    input_dict: Mapping[str, object],
    request_id: str,
    dependencies: Mapping[str, object],
) -> Mapping[str, object]:
    """Run weighted decision matrix scoring.

    Args:
        input_dict: Tool input payload.
        request_id: Request identifier.
        dependencies: Runtime dependencies.

    Returns:
        Ranked alternatives and criterion contributions.
    """
    del request_id, dependencies
    normalize = bool(input_dict.get("normalize", True))
    criteria = _parse_criteria(input_dict.get("criteria"))
    alternatives = _parse_alternatives(input_dict.get("alternatives"))

    ranked: list[dict[str, object]] = []
    for alternative in alternatives:
        contributions: dict[str, float] = {}
        criterion_values: dict[str, float] = {}
        total = 0.0
        for criterion in criteria:
            raw_value = alternative.scores[criterion.name]
            criterion_values[criterion.name] = raw_value
            adjusted = _decision_value(
                alternatives=alternatives,
                criterion=criterion,
                raw_value=raw_value,
                normalize=normalize,
            )
            weighted = criterion.weight * adjusted
            contributions[criterion.name] = weighted
            total += weighted
        ranked.append(
            {
                "alternative": alternative.alt_id,
                "total_score": total,
                "contributions": contributions,
                "scores": criterion_values,
            }
        )

    ranked.sort(
        # Sort by score then id for deterministic ordering when totals tie.
        key=lambda item: (
            cast(float, item["total_score"]),
            cast(str, item["alternative"]),
        ),
        reverse=True,
    )
    for index, item in enumerate(ranked, start=1):
        item["rank"] = index

    return {
        "method": "weighted_decision_matrix",
        "normalize": normalize,
        "criteria": [
            {
                "name": criterion.name,
                "goal": criterion.goal,
                "weight": criterion.weight,
            }
            for criterion in criteria
        ],
        "ranked": ranked,
    }


def _pairwise_rank_handler(
    input_dict: Mapping[str, object],
    request_id: str,
    dependencies: Mapping[str, object],
) -> Mapping[str, object]:
    """Run pairwise ranking with Copeland aggregation.

    Args:
        input_dict: Tool input payload.
        request_id: Request identifier.
        dependencies: Runtime dependencies.

    Returns:
        Pairwise table and Copeland ranking.
    """
    del request_id, dependencies
    method = str(input_dict.get("method", "copeland")).strip().lower()
    if method and method != "copeland":
        raise ValueError("Only method='copeland' is supported in v1.")

    alternative_names = _parse_pairwise_alternatives(input_dict.get("alternatives"))
    comparisons = input_dict.get("comparisons")
    if not isinstance(comparisons, list):
        raise ValueError("comparisons must be a list.")

    scores, matrix = _initialize_pairwise_trackers(alternative_names)

    for index, raw_entry in enumerate(comparisons):
        a_name, b_name, outcome = _parse_comparison_entry(raw_entry, index=index)
        _apply_pairwise_outcome(
            scores=scores,
            matrix=matrix,
            a_name=a_name,
            b_name=b_name,
            outcome=outcome,
            index=index,
        )

    ranking = _build_pairwise_ranking(scores=scores, alternative_names=alternative_names)
    pairwise_table = _build_pairwise_table(matrix)
    return {
        "method": "copeland",
        "pairwise_table": pairwise_table,
        "ranking": ranking,
    }


def _initialize_pairwise_trackers(
    alternative_names: list[str],
) -> tuple[dict[str, dict[str, float]], dict[tuple[str, str], dict[str, int]]]:
    """Create pairwise score and matrix containers.

    Args:
        alternative_names: Alternative ids in run order.

    Returns:
        Initialized score and pairwise matrix mappings.
    """
    scores = {name: {"wins": 0.0, "losses": 0.0, "ties": 0.0} for name in alternative_names}
    matrix: dict[tuple[str, str], dict[str, int]] = {}
    for left in alternative_names:
        for right in alternative_names:
            if left >= right:
                continue
            matrix[(left, right)] = {"left_wins": 0, "right_wins": 0, "ties": 0}
    return scores, matrix


def _apply_pairwise_outcome(
    *,
    scores: dict[str, dict[str, float]],
    matrix: dict[tuple[str, str], dict[str, int]],
    a_name: str,
    b_name: str,
    outcome: str,
    index: int,
) -> None:
    """Apply one pairwise comparison to score and matrix trackers.

    Args:
        scores: Running wins/losses/ties by alternative.
        matrix: Running pairwise matrix.
        a_name: Left alternative.
        b_name: Right alternative.
        outcome: Outcome token (``a``, ``b``, ``tie``).
        index: Comparison index for errors.
    """
    if a_name not in scores or b_name not in scores:
        raise ValueError(f"comparisons[{index}] references unknown alternatives.")
    if a_name == b_name:
        raise ValueError(f"comparisons[{index}] must compare two distinct alternatives.")

    key = (a_name, b_name) if a_name < b_name else (b_name, a_name)
    # Store matrix in canonical lexical order to avoid duplicate mirrored keys.
    oriented = matrix[key]
    if outcome == "tie":
        scores[a_name]["ties"] += 1.0
        scores[b_name]["ties"] += 1.0
        oriented["ties"] += 1
        return

    winner = a_name if outcome == "a" else b_name
    loser = b_name if outcome == "a" else a_name
    if key[0] == winner:
        oriented["left_wins"] += 1
    else:
        oriented["right_wins"] += 1
    scores[winner]["wins"] += 1.0
    scores[loser]["losses"] += 1.0


def _build_pairwise_ranking(
    *,
    scores: dict[str, dict[str, float]],
    alternative_names: list[str],
) -> list[dict[str, object]]:
    """Build ranked Copeland table from score tracker.

    Args:
        scores: Running wins/losses/ties by alternative.
        alternative_names: Alternatives in stable input order.

    Returns:
        Ranked Copeland rows.
    """
    ranking: list[dict[str, object]] = []
    for name in alternative_names:
        wins = float(scores[name]["wins"])
        losses = float(scores[name]["losses"])
        ties = float(scores[name]["ties"])
        ranking.append(
            {
                "alternative": name,
                "wins": int(wins),
                "losses": int(losses),
                "ties": int(ties),
                "copeland_score": wins - losses + (0.5 * ties),
            }
        )
    ranking.sort(
        # Stable tie-breaker keeps outputs deterministic for equal Copeland scores.
        key=lambda item: (
            cast(float, item["copeland_score"]),
            cast(int, item["wins"]),
            cast(str, item["alternative"]),
        ),
        reverse=True,
    )
    for rank_index, row in enumerate(ranking, start=1):
        row["rank"] = rank_index
    return ranking


def _build_pairwise_table(matrix: dict[tuple[str, str], dict[str, int]]) -> list[dict[str, object]]:
    """Serialize sorted pairwise matrix table.

    Args:
        matrix: Pairwise matrix accumulator.

    Returns:
        Sorted pairwise row list.
    """
    return [
        {
            "a": left,
            "b": right,
            "a_wins": counts["left_wins"],
            "b_wins": counts["right_wins"],
            "ties": counts["ties"],
        }
        for (left, right), counts in sorted(matrix.items())
    ]


def _parse_criteria(raw_criteria: object) -> list[_Criterion]:
    """Parse weighted criteria payloads.

    Args:
        raw_criteria: Raw criteria payload.

    Returns:
        Parsed criteria list.
    """
    if not isinstance(raw_criteria, list) or not raw_criteria:
        raise ValueError("criteria must be a non-empty list.")
    parsed: list[_Criterion] = []
    for index, entry in enumerate(raw_criteria):
        if not isinstance(entry, Mapping):
            raise ValueError(f"criteria[{index}] must be an object.")
        name = str(entry.get("name", "")).strip()
        if not name:
            raise ValueError(f"criteria[{index}].name must be non-empty.")
        goal = str(entry.get("goal", "max")).strip().lower()
        if goal not in {"max", "min"}:
            raise ValueError(f"criteria[{index}].goal must be 'max' or 'min'.")
        weight = _coerce_number(entry.get("weight"), label=f"criteria[{index}].weight")
        if weight <= 0:
            raise ValueError(f"criteria[{index}].weight must be > 0.")
        parsed.append(_Criterion(name=name, goal=goal, weight=weight))

    total_weight = sum(item.weight for item in parsed)
    return [_Criterion(name=item.name, goal=item.goal, weight=item.weight / total_weight) for item in parsed]


def _parse_alternatives(raw_alternatives: object) -> list[_Alternative]:
    """Parse alternatives payload for decision matrix.

    Args:
        raw_alternatives: Raw alternatives payload.

    Returns:
        Parsed alternatives list.
    """
    if not isinstance(raw_alternatives, list) or not raw_alternatives:
        raise ValueError("alternatives must be a non-empty list.")
    parsed: list[_Alternative] = []
    for index, entry in enumerate(raw_alternatives):
        if not isinstance(entry, Mapping):
            raise ValueError(f"alternatives[{index}] must be an object.")
        alt_id = str(entry.get("id", entry.get("name", f"alt_{index + 1}"))).strip()
        if not alt_id:
            raise ValueError(f"alternatives[{index}].id must be non-empty.")
        raw_scores = entry.get("scores")
        if isinstance(raw_scores, Mapping):
            score_map = dict(raw_scores)
        else:
            score_map = {key: value for key, value in entry.items() if key not in {"id", "name"}}
        if not score_map:
            raise ValueError(f"alternatives[{index}] must include score fields or scores object.")
        scores: dict[str, float] = {}
        for key, value in score_map.items():
            scores[str(key)] = _coerce_number(
                value,
                label=f"alternatives[{index}].scores[{key}]",
            )
        parsed.append(_Alternative(alt_id=alt_id, scores=scores))
    return parsed


def _decision_value(
    *,
    alternatives: list[_Alternative],
    criterion: _Criterion,
    raw_value: float,
    normalize: bool,
) -> float:
    """Resolve criterion value used for final weighted scoring.

    Args:
        alternatives: Parsed alternatives.
        criterion: Current criterion.
        raw_value: Raw criterion value for one alternative.
        normalize: Whether to apply min-max normalization.

    Returns:
        Goal-aligned criterion score.
    """
    if not normalize:
        return raw_value if criterion.goal == "max" else -raw_value

    values = _criterion_values(alternatives=alternatives, criterion_name=criterion.name)
    min_value = min(values)
    max_value = max(values)
    if max_value == min_value:
        return 1.0
    if criterion.goal == "max":
        return (raw_value - min_value) / (max_value - min_value)
    return (max_value - raw_value) / (max_value - min_value)


def _criterion_values(*, alternatives: list[_Alternative], criterion_name: str) -> list[float]:
    """Collect values for one criterion across alternatives.

    Args:
        alternatives: Parsed alternatives.
        criterion_name: Criterion name.

    Returns:
        Criterion values in alternative order.
    """
    values: list[float] = []
    for alternative in alternatives:
        if criterion_name not in alternative.scores:
            raise ValueError(f"alternative '{alternative.alt_id}' is missing criterion '{criterion_name}'.")
        values.append(alternative.scores[criterion_name])
    return values


def _parse_pairwise_alternatives(raw_alternatives: object) -> list[str]:
    """Parse alternative names for pairwise ranking.

    Args:
        raw_alternatives: Raw alternatives payload.

    Returns:
        Normalized alternative names.
    """
    if not isinstance(raw_alternatives, list) or not raw_alternatives:
        raise ValueError("alternatives must be a non-empty list.")
    names: list[str] = []
    for index, entry in enumerate(raw_alternatives):
        if isinstance(entry, str):
            name = entry.strip()
        elif isinstance(entry, Mapping):
            name = str(entry.get("id", entry.get("name", ""))).strip()
        else:
            raise ValueError(f"alternatives[{index}] must be a string or object.")
        if not name:
            raise ValueError(f"alternatives[{index}] must resolve to a non-empty name.")
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("alternatives must contain unique names.")
    return names


def _parse_comparison_entry(raw_entry: object, *, index: int) -> tuple[str, str, str]:
    """Parse one pairwise comparison entry.

    Args:
        raw_entry: Raw comparison payload.
        index: Entry index.

    Returns:
        Tuple ``(a_name, b_name, outcome)``.
    """
    if not isinstance(raw_entry, Mapping):
        raise ValueError(f"comparisons[{index}] must be an object.")

    if "winner" in raw_entry and "loser" in raw_entry:
        winner = str(raw_entry.get("winner", "")).strip()
        loser = str(raw_entry.get("loser", "")).strip()
        if not winner or not loser:
            raise ValueError(f"comparisons[{index}] winner/loser must be non-empty.")
        return winner, loser, "a"

    a_name = str(raw_entry.get("a", "")).strip()
    b_name = str(raw_entry.get("b", "")).strip()
    outcome = str(raw_entry.get("outcome", "")).strip().lower()
    if not a_name or not b_name:
        raise ValueError(f"comparisons[{index}] requires non-empty a and b.")
    if outcome not in {"a", "b", "tie"}:
        raise ValueError(f"comparisons[{index}].outcome must be 'a', 'b', or 'tie'.")
    return a_name, b_name, outcome


def _coerce_number(raw_value: object, *, label: str) -> float:
    """Coerce one numeric field into float.

    Args:
        raw_value: Raw field value.
        label: Field label for errors.

    Returns:
        Numeric value.
    """
    if isinstance(raw_value, bool):
        raise ValueError(f"{label} must be a number.")
    if isinstance(raw_value, (int, float)):
        return float(raw_value)
    if isinstance(raw_value, str) and raw_value.strip():
        try:
            return float(raw_value.strip())
        except ValueError as exc:
            raise ValueError(f"{label} must be a number.") from exc
    raise ValueError(f"{label} must be a number.")


__all__ = ["register_evaluation_tools"]
