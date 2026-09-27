"""Deterministic, offline evaluation for externally generated FOMO responses.

The framework never calls a model and never executes generated code. See
``python -m evals --help`` for the JSONL runner.
"""

from .runner import EvaluationError, evaluate_case, run_suite

__all__ = ["EvaluationError", "evaluate_case", "run_suite"]