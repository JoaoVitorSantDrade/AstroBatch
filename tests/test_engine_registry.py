from __future__ import annotations

import unittest

from app.engines import (
    EngineDescriptor,
    EngineProfile,
    EngineRegistry,
    EngineUnavailable,
    ExecutionBudget,
)


class EngineRegistryTests(unittest.TestCase):
    def test_resolves_a_registered_engine_for_its_profile(self) -> None:
        registry = EngineRegistry()
        implementation = lambda: "ok"
        registry.register(
            EngineDescriptor(
                "stable", "flow.detector", "Stable", frozenset({EngineProfile.STABLE})
            ),
            implementation,
        )
        self.assertIs(registry.resolve("flow.detector", "stable", EngineProfile.STABLE), implementation)

    def test_rejects_an_engine_outside_its_profile(self) -> None:
        registry = EngineRegistry()
        registry.register(
            EngineDescriptor(
                "fast", "stack.reducer", "Fast", frozenset({EngineProfile.FAST})
            ),
            lambda: None,
        )
        with self.assertRaises(EngineUnavailable):
            registry.resolve("stack.reducer", "fast", EngineProfile.STABLE)

    def test_reports_missing_optional_dependency(self) -> None:
        registry = EngineRegistry()
        registry.register(
            EngineDescriptor(
                "optional", "flow.detector", "Optional", frozenset({EngineProfile.FAST}),
                optional_dependency="astrobatch_test_missing_dependency",
            ),
            lambda: None,
        )
        available, reason = registry.is_available("flow.detector", "optional")
        self.assertFalse(available)
        self.assertIn("astrobatch_test_missing_dependency", reason or "")

    def test_execution_budget_reserves_parallel_kernels_for_single_worker_reductions(self) -> None:
        self.assertFalse(ExecutionBudget.for_pipeline(4).kernel_parallel)
        self.assertTrue(ExecutionBudget.for_pipeline(1).kernel_parallel)


if __name__ == "__main__":
    unittest.main()
