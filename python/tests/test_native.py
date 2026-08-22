from __future__ import annotations

import unittest

try:
    from foamnordic import _native
except ImportError:
    _native = None


@unittest.skipIf(_native is None, "nanobind extension is not installed")
class NativeBindingTests(unittest.TestCase):
    def test_help_and_dir_expose_the_small_native_facade(self) -> None:
        self.assertIn("native Longship allocation plan", _native.plan_longship.__doc__)
        self.assertIn("resource plan", _native.LongshipPlan.__doc__)
        self.assertIn("plan_longship", dir(_native))
        self.assertNotIn("Harbor", dir(_native))

    def test_runtime_plan_uses_native_resource_arithmetic(self) -> None:
        placement = _native.PlacementRequest()
        placement.placement = _native.HostPlacement.ATTACHED
        placement.data_path = _native.DataPathPreference.SHARED_MEMORY
        placement.solver_nodes = 2

        request = _native.LongshipRequest()
        request.name = "two-node"
        request.solver_nodes = 2
        request.solver_tasks = 16
        request.solver_cpus_per_task = 1
        request.host_cpus_per_node = 1
        request.placement = placement

        plan = _native.plan_longship(request)

        self.assertEqual(plan.name, "two-node")
        self.assertEqual(plan.allocation_nodes, 2)
        self.assertEqual(plan.solver_tasks_per_node, 8)
        self.assertEqual(plan.host_tasks, 2)
        self.assertTrue(plan.host_starts_first)
        self.assertTrue(plan.fail_together)

    def test_invalid_request_maps_to_python_exception(self) -> None:
        request = _native.LongshipRequest()
        request.solver_tasks = 0
        with self.assertRaises(ValueError):
            _native.plan_longship(request)


if __name__ == "__main__":
    unittest.main()
