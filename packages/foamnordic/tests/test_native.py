from __future__ import annotations

import pytest

try:
    from foamnordic import _native
except ImportError:
    _native = None


@pytest.mark.skipif(_native is None, reason="nanobind extension is not installed")
def test_help_and_dir_expose_the_small_native_facade() -> None:
    assert "native Longship allocation plan" in _native.plan_longship.__doc__
    assert "resource plan" in _native.LongshipPlan.__doc__
    assert "plan_longship" in dir(_native)
    assert "Harbor" not in dir(_native)
    assert "write_model_manifest" not in dir(_native)


@pytest.mark.skipif(_native is None, reason="nanobind extension is not installed")
def test_runtime_plan_uses_native_resource_arithmetic() -> None:
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

    assert plan.name == "two-node"
    assert plan.allocation_nodes == 2
    assert plan.solver_tasks_per_node == 8
    assert plan.host_tasks == 2
    assert plan.host_starts_first
    assert plan.fail_together


@pytest.mark.skipif(_native is None, reason="nanobind extension is not installed")
def test_invalid_request_maps_to_python_exception() -> None:
    request = _native.LongshipRequest()
    request.solver_tasks = 0
    with pytest.raises(ValueError):
        _native.plan_longship(request)
