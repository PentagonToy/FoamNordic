"""Translation from public declarations to the private native facade."""

from __future__ import annotations

from typing import Any

try:
    from . import _native
except ImportError:
    _native = None


def available() -> bool:
    """Return whether the compiled native planning facade is importable."""

    return _native is not None


def compile_runtime_plan(longship: Any) -> dict[str, object]:
    """Resolve resource arithmetic and lifecycle invariants in native C++."""

    if _native is None:
        raise RuntimeError(
            "FoamNordic's native extension is unavailable. Install a binary "
            "wheel or build the package before compiling a Longship plan."
        )

    scheduler = longship.scheduler
    solver_nodes = 1 if scheduler is None else scheduler.nodes
    solver_tasks = longship.case.ranks if scheduler is None else scheduler.ntasks
    solver_cpus_per_task = (
        1 if scheduler is None else scheduler.cpus_per_task
    )

    placement = _native.PlacementRequest()
    placement.placement = _native.HostPlacement.ATTACHED
    placement.device = _native.InferenceDevice.CPU
    placement.solver_nodes = solver_nodes
    placement.data_path = {
        "auto": _native.DataPathPreference.AUTOMATIC,
        "shm": _native.DataPathPreference.SHARED_MEMORY,
        "uds": _native.DataPathPreference.UNIX_SOCKET,
        "ucx": _native.DataPathPreference.UCX,
        "tcp": _native.DataPathPreference.TCP,
    }[longship.placement.data_path]
    # Availability is verified against the selected build/launch profile later.
    # These flags tell the native planner which explicit declaration is legal.
    placement.shared_memory_available = longship.placement.data_path in {"auto", "shm"}
    placement.unix_socket_available = longship.placement.data_path in {"auto", "uds"}
    placement.ucx_available = longship.placement.data_path == "ucx"

    request = _native.LongshipRequest()
    request.name = longship.name
    request.solver_nodes = solver_nodes
    request.solver_tasks = solver_tasks
    request.solver_cpus_per_task = solver_cpus_per_task
    request.host_cpus_per_node = longship.placement.closure_cpus_per_node
    request.use_closure_host = bool(longship.closures)
    request.placement = placement

    plan = _native.plan_longship(request)
    placement_names = {
        _native.HostPlacement.ATTACHED: "attached",
        _native.HostPlacement.CENTRAL: "central",
    }
    data_path_names = {
        _native.DataPath.SHARED_MEMORY: "shm",
        _native.DataPath.UNIX_SOCKET: "uds",
        _native.DataPath.UCX: "ucx",
        _native.DataPath.TCP: "tcp",
    }
    has_closure = bool(longship.closures)
    return {
        "allocation_nodes": plan.allocation_nodes,
        "allocation_cpus_per_node": plan.allocation_cpus_per_node,
        "solver_tasks": plan.solver_tasks,
        "solver_tasks_per_node": plan.solver_tasks_per_node,
        "solver_cpus_per_task": plan.solver_cpus_per_task,
        "host_tasks": plan.host_tasks,
        "host_cpus_per_task": plan.host_cpus_per_task,
        "placement": {
            "kind": placement_names[plan.placement.placement] if has_closure else "none",
            "data_path": data_path_names[plan.placement.data_path] if has_closure else "none",
            "host_instances": plan.placement.host_instances,
            "same_allocation": plan.placement.same_allocation,
            "same_node": plan.placement.same_node,
            "coupled_lifetime": plan.placement.coupled_lifetime,
            "reason": plan.placement.reason,
        },
        "lifecycle": {
            "host_starts_first": plan.host_starts_first,
            "fail_together": plan.fail_together,
        },
    }
