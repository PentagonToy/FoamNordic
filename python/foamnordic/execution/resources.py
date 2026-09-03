"""Human-readable and Slurm-ready workload resource estimates."""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.spec import Longship


SHM_MIB_PER_RANK_PROGRAM = 32
_MEMORY = re.compile(r"^(\d+(?:\.\d+)?)\s*([KMGT]i?B?|B)?$", re.IGNORECASE)
_UNITS = {"B": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}


def memory_bytes(value: str | None) -> int | None:
    if value is None:
        return None
    match = _MEMORY.fullmatch(value)
    if match is None:
        raise ValueError(f"unsupported Slurm memory value: {value!r}")
    unit = (match.group(2) or "B").upper().removesuffix("B").removesuffix("I") or "B"
    return math.ceil(float(match.group(1)) * _UNITS[unit])


def slurm_memory(value: int) -> str:
    return f"{math.ceil(value / 1024**2)}M"


def readable_bytes(value: int | None) -> str:
    if value is None:
        return "shared"
    for unit, scale in (("TiB", 1024**4), ("GiB", 1024**3), ("MiB", 1024**2)):
        if value >= scale:
            amount = value / scale
            return f"{amount:g} {unit}"
    return f"{value / 1024:g} KiB"


def resource_values(longship: Longship) -> dict[str, object]:
    scheduler = longship.scheduler
    if scheduler is None:
        raise ValueError("scheduled resource estimates require a Slurm declaration")
    programs = len(longship.field_programs)
    model = scheduler.model_resources
    model_active = programs > 0
    solver_cpus = scheduler.ntasks * scheduler.cpus_per_task
    model_cpus_per_node = (
        model.cpus_per_task
        if scheduler.has_model_resources
        else (
            longship.placement.model_cpus_per_node
            or max(1, programs)
        )
    )
    model_cpus = scheduler.nodes * model_cpus_per_node if model_active else 0
    memory_per_cpu = memory_bytes(scheduler.mem_per_cpu)
    model_memory_per_cpu = memory_bytes(model.mem_per_cpu) if model_active else None
    solver_memory = (
        None if memory_per_cpu is None else memory_per_cpu * solver_cpus
    )
    model_memory = (
        None
        if memory_per_cpu is None
        else (
            model_memory_per_cpu * model_cpus
            if model_memory_per_cpu is not None
            else memory_per_cpu * model_cpus
        )
    )
    shm = (
        SHM_MIB_PER_RANK_PROGRAM * 1024**2 * scheduler.ntasks * programs
        if longship.placement.data_path in {"auto", "shm"}
        else 0
    )
    memory_known = solver_memory is not None and (
        not model_active or model_memory is not None
    )
    total_memory = None
    if memory_known:
        total_memory = solver_memory + (model_memory or 0)
        if scheduler.has_model_resources and model.mem_per_cpu is not None:
            total_memory += shm
    return {
        "programs": programs,
        "solver_cpus": solver_cpus,
        "model_cpus": model_cpus,
        "total_cpus": solver_cpus + model_cpus,
        "solver_memory": solver_memory,
        "model_memory": model_memory,
        "total_memory": total_memory,
        "shm": shm,
        "host_instances": scheduler.nodes if model_active else 0,
    }


def display_resources(longship: Longship) -> None:
    import onsaemiro as osm

    scheduler = longship.scheduler
    assert scheduler is not None
    values = resource_values(longship)
    programs = int(values["programs"])
    table = osm.TableMaker(
        title=f"{longship.name} Resource Plan",
        columns=["Resource", "OpenFOAM", "Model", "Allocation"],
        mode="static",
    )
    rows = (
        (
            "Processes",
            f"{scheduler.ntasks} tasks",
            f"{values['host_instances']} host(s)",
            f"{scheduler.nodes} node(s)",
        ),
        (
            "CPU",
            f"{values['solver_cpus']} cores",
            f"{values['model_cpus']} cores",
            f"{values['total_cpus']} cores",
        ),
        (
            "Memory",
            readable_bytes(values["solver_memory"]),
            readable_bytes(values["model_memory"]),
            readable_bytes(values["total_memory"]),
        ),
        (
            "Field programs",
            "-",
            str(programs),
            str(programs),
        ),
        (
            "SHM",
            "-",
            f"~{readable_bytes(values['shm'])}" if values["shm"] else "none",
            f"~{readable_bytes(values['shm'])}" if values["shm"] else "none",
        ),
    )
    for row in rows:
        table.add_row(row)
    table.display()
