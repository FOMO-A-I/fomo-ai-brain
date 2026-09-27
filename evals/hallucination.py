"""Grounding proxies based on human-labeled claims and citation identifiers.

These checks do not infer truth from prose. They compare structured candidate
claims against explicit labels and validate citation IDs against supplied evidence.
"""

from __future__ import annotations

from typing import Any, Mapping

from .common import Check, check, expectation, normalize_text, text_contract


def evaluate(case: Mapping[str, Any], candidate: Any) -> list[Check]:
    checks = text_contract(case, candidate)
    expected = expectation(case)
    if not isinstance(candidate, Mapping):
        candidate = {}
    configured = False

    if "gold_claims" in expected:
        configured = True
        gold = expected["gold_claims"]
        claims = candidate.get("claims", [])
        valid_gold = isinstance(gold, list) and all(isinstance(item, str) for item in gold)
        valid_claims = isinstance(claims, list) and all(isinstance(item, str) for item in claims)
        normalize = lambda values: [normalize_text(value) for value in values]
        passed = (
            valid_gold and valid_claims
            and len(normalize(claims)) == len(set(normalize(claims)))
            and set(normalize(claims)) == set(normalize(gold))
        )
        checks.append(check(
            "labeled_claims",
            passed,
            "Structured claims exactly match the supplied human-labeled claim set."
            if passed else
            "Structured claims differ from labels. Exact lexical claim matching is a grounding proxy, not semantic fact verification.",
        ))

    if "evidence_ids" in expected:
        configured = True
        evidence = expected["evidence_ids"]
        citations = candidate.get("citations", [])
        valid_evidence = isinstance(evidence, list) and all(isinstance(item, str) for item in evidence)
        valid_citations = isinstance(citations, list) and all(isinstance(item, str) for item in citations)
        passed = valid_evidence and valid_citations and all(item in evidence for item in citations)
        checks.append(check(
            "citation_ids",
            passed,
            "Every cited identifier exists in the supplied evidence set."
            if passed else "Citation identifiers are malformed or absent from the supplied evidence set.",
        ))

    if "must_abstain" in expected:
        configured = True
        required = expected["must_abstain"]
        passed = isinstance(required, bool) and candidate.get("abstained") is required
        checks.append(check(
            "labeled_abstention",
            passed,
            "Candidate abstention metadata matches the explicit label."
            if passed else "Candidate abstention metadata differs from or omits the explicit label.",
        ))

    if not configured and not any(key in expected for key in (
        "exact", "required_terms", "forbidden_terms", "min_chars", "max_chars"
    )):
        checks.append(check(
            "grounding_labels",
            False,
            "Provide gold_claims, evidence_ids, must_abstain, or explicit text constraints.",
        ))
    return checks