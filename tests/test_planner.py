import unittest

from fomo.reasoning.execution_graph import ExecutionGraph, GraphError
from fomo.reasoning.retry_policy import PermanentFailure, RetryPolicy
from fomo.reasoning.task_decomposition import AGENT_NAMES, Task, decompose_task
from fomo.reasoning.verification import verify_result


class ScriptedBackend:
    def __init__(self, reply):
        self.reply = reply
        self.requests = []

    def complete(self, messages):
        self.requests.append(tuple(messages))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


class ReasoningTests(unittest.TestCase):
    def test_all_36_declared_agent_names_are_valid_task_roles(self):
        self.assertEqual(len(AGENT_NAMES), 36)
        self.assertEqual(
            {
                Task(f"task-{index}", "Perform assigned role", agent=name).agent
                for index, name in enumerate(sorted(AGENT_NAMES), 1)
            },
            AGENT_NAMES,
        )

    def test_decomposition_builds_sequential_tasks_and_bounds_count(self):
        tasks = decompose_task("Build a feature", ["Inspect", "Implement", "Review"])
        self.assertEqual([task.depends_on for task in tasks], [(), ("task-1",), ("task-2",)])
        self.assertEqual([task.status for task in tasks], ["pending"] * 3)
        with self.assertRaises(ValueError):
            decompose_task("request", ["1", "2"], max_tasks=1)

    def test_execution_graph_guards_dependencies_and_transitions(self):
        tasks = [
            Task("first", "First"),
            Task("second", "Second", depends_on=("first",)),
        ]
        graph = ExecutionGraph(tasks)
        self.assertEqual([task.id for task in graph.ready()], ["first"])
        with self.assertRaises(GraphError):
            graph.start("second")
        graph.start("first")
        graph.complete("first", "done")
        graph.start("second")
        graph.complete("second", "also done")
        self.assertTrue(graph.finished)
        with self.assertRaises(GraphError):
            graph.start("first")

    def test_execution_graph_rejects_cycles_and_missing_dependencies(self):
        with self.assertRaises(GraphError):
            ExecutionGraph(
                [
                    Task("first", "First", depends_on=("second",)),
                    Task("second", "Second", depends_on=("first",)),
                ]
            )
        with self.assertRaises(GraphError):
            ExecutionGraph([Task("first", "First", depends_on=("missing",))])

    def test_retry_is_bounded_and_permanent_failures_are_not_retried(self):
        attempts = []
        policy = RetryPolicy(max_attempts=3, initial_delay=0, max_delay=0)

        def flaky():
            attempts.append(1)
            if len(attempts) < 3:
                raise OSError("temporary")
            return "ok"

        self.assertEqual(policy.run(flaky), "ok")
        self.assertEqual(len(attempts), 3)
        attempts.clear()
        with self.assertRaises(PermanentFailure):
            policy.run(lambda: (_ for _ in ()).throw(PermanentFailure("no retry")))
        self.assertEqual(attempts, [])

    def test_output_verification_checks_python_syntax_without_execution(self):
        self.assertTrue(verify_result("value = 42", require_python_syntax=True).passed)
        failed = verify_result("def broken(:", require_python_syntax=True)
        self.assertFalse(failed.passed)
        self.assertIn("syntax error", failed.issues[0])
        self.assertFalse(verify_result("  ").passed)


if __name__ == "__main__":
    unittest.main()