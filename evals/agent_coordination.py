"""Structural checks for multi-agent handoffs and outcomes."""

from __future__ import annotations

from typing import Any, Mapping

from .common import Check, check, expectation, text_contract


def evaluate(case: Mapping[str, Any], candidate: Any) -> list[Check]:
    checks = text_contract(case, candidate)
    expected = expectation(case)
    if not isinstance(candidate, Mapping):
        candidate = {}
    agents = candidate.get("agents", [])
    valid_agents = isinstance(agents, list) and all(
        isinstance(agent, Mapping)
        and isinstance(agent.get("name"), str) and agent["name"]
        and isinstance(agent.get("status"), str)
        for agent in agents
    )
    configured = False

    if "required_agents" in expected:
        configured = True
        required = expected["required_agents"]
        names = [agent["name"] for agent in agents] if valid_agents else []
        passed = (
            isinstance(required, list) and all(isinstance(item, str) for item in required)
            and valid_agents and len(names) == len(set(names))
            and set(names) == set(required)
        )
        checks.append(check(
            "agent_roster",
            passed,
            "Reported agent roster exactly matches the labeled roster."
            if passed else "Agent roster is malformed, duplicated, missing, or unexpected.",
        ))

    if "all_agents_succeeded" in expected:
        configured = True
        required = expected["all_agents_succeeded"]
        passed = (
            required is True and valid_agents and bool(agents)
            and all(agent["status"].casefold() in ("success", "succeeded", "completed") for agent in agents)
        )
        checks.append(check(
            "agent_completion",
            passed,
            "Every reported agent completed successfully."
            if passed else "At least one agent failed, is incomplete, or no agent outcome was provided.",
        ))

    if "expected_handoffs" in expected:
        configured = True
        gold = expected["expected_handoffs"]
        handoffs = candidate.get("handoffs", [])
        def valid_edges(value: Any) -> bool:
            return isinstance(value, list) and all(
                isinstance(edge, Mapping)
                and isinstance(edge.get("from"), str)
                and isinstance(edge.get("to"), str)
                for edge in value
            )
        actual_edges = [(edge["from"], edge["to"]) for edge in handoffs] if valid_edges(handoffs) else []
        gold_edges = [(edge["from"], edge["to"]) for edge in gold] if valid_edges(gold) else []
        passed = valid_edges(gold) and valid_edges(handoffs) and actual_edges == gold_edges
        checks.append(check(
            "agent_handoffs",
            passed,
            "Agent handoffs match the labeled ordered handoff list."
            if passed else "Handoff records differ from the expected agents or order.",
        ))

    if not valid_agents and "required_agents" in expected:
        checks.append(check(
            "agent_record_schema",
            False,
            "agents must be a list of objects with non-empty name and status strings.",
        ))
    if not configured and not any(key in expected for key in (
        "exact", "required_terms", "forbidden_terms", "min_chars", "max_chars"
    )):
        checks.append(check(
            "coordination_ground_truth",
            False,
            "Provide labeled agents, completion requirements, handoffs, or text constraints.",
        ))
    return checks