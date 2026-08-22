"""Immutable public declarations compiled before a solver starts."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

from . import openfoam
from ._expressions import FieldExpression
from ._native_plan import compile_runtime_plan
from ._paths import PathInput, path_from
from ._plan import CompiledPlan
from ._validation import require_nonempty, require_positive

if TYPE_CHECKING:
    from ._run import Run


def _frozen_mapping(
    value: Mapping[str, FieldExpression], label: str
) -> Mapping[str, FieldExpression]:
    if not value:
        raise ValueError(f"{label} must not be empty")
    normalized: dict[str, FieldExpression] = {}
    for name, expression in value.items():
        key = require_nonempty(name, f"{label} name")
        if not isinstance(expression, FieldExpression):
            raise TypeError(f"{label}[{key!r}] must be a FieldExpression")
        normalized[key] = expression
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class Closure:
    """Bind model artifact tensors to native OpenFOAM expressions and fields."""

    name: str
    artifact: PathInput
    inputs: Mapping[str, FieldExpression]
    outputs: Mapping[str, FieldExpression]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", require_nonempty(self.name, "closure name"))
        artifact = path_from(self.artifact)
        require_nonempty(str(artifact), "artifact")
        object.__setattr__(self, "artifact", artifact)
        object.__setattr__(self, "inputs", _frozen_mapping(self.inputs, "inputs"))
        object.__setattr__(self, "outputs", _frozen_mapping(self.outputs, "outputs"))
        for logical_name, expression in self.outputs.items():
            if expression.operation != "field":
                raise ValueError(f"output {logical_name!r} must bind to a mutable field")

    def to_plan(self) -> dict[str, object]:
        return {
            "name": self.name,
            "artifact": str(self.artifact),
            "inputs": {name: value.to_plan() for name, value in self.inputs.items()},
            "outputs": {name: value.to_plan() for name, value in self.outputs.items()},
        }


@dataclass(frozen=True, slots=True)
class Retention:
    """Byte- and record-bounded observation retention policy."""

    records: int = 64
    maximum_bytes: int = 256 * 1024
    overflow: str = "drop_oldest"

    def __post_init__(self) -> None:
        require_positive(self.records, "retention records")
        require_positive(self.maximum_bytes, "retention maximum_bytes")
        if self.overflow not in {"drop_oldest", "drop_newest"}:
            raise ValueError("retention overflow must be drop_oldest or drop_newest")

    @classmethod
    def latest(cls, records: int, *, maximum_bytes: int) -> "Retention":
        return cls(records=records, maximum_bytes=maximum_bytes)

    def to_plan(self) -> dict[str, object]:
        return {
            "records": self.records,
            "maximum_bytes": self.maximum_bytes,
            "overflow": self.overflow,
        }


@dataclass(frozen=True, slots=True)
class Observe:
    """Read-only summary observations compiled before solver launch."""

    summaries: Mapping[str, tuple[str, ...]]
    every: int = 1
    offset: int = 0
    retention: Retention = dataclass_field(default_factory=Retention)

    def __post_init__(self) -> None:
        require_positive(self.every, "observation every")
        if self.offset < 0:
            raise ValueError("observation offset must not be negative")
        normalized: dict[str, tuple[str, ...]] = {}
        supported = {"min", "max", "mean", "l2"}
        for field_name, reductions in self.summaries.items():
            name = require_nonempty(field_name, "observed field")
            values = tuple(reductions)
            if not values:
                raise ValueError(f"summaries[{name!r}] must not be empty")
            unknown = set(values) - supported
            if unknown:
                raise ValueError(f"unsupported reductions for {name!r}: {sorted(unknown)}")
            normalized[name] = values
        if not normalized:
            raise ValueError("summaries must not be empty")
        object.__setattr__(self, "summaries", MappingProxyType(normalized))

    def to_plan(self) -> dict[str, object]:
        return {
            "summaries": {name: list(values) for name, values in self.summaries.items()},
            "schedule": {"every": self.every, "offset": self.offset},
            "retention": self.retention.to_plan(),
        }


@dataclass(frozen=True, slots=True)
class Attached:
    """Place one native ClosureHost beside each solver node."""

    closure_cpus_per_node: int = 1
    data_path: str = "auto"

    def __post_init__(self) -> None:
        require_positive(self.closure_cpus_per_node, "closure_cpus_per_node")
        if self.data_path not in {"auto", "shm", "uds", "ucx", "tcp"}:
            raise ValueError("data_path must be auto, shm, uds, ucx, or tcp")

    def to_plan(self) -> dict[str, object]:
        return {
            "kind": "attached",
            "closure_cpus_per_node": self.closure_cpus_per_node,
            "data_path": self.data_path,
        }


@dataclass(frozen=True, slots=True)
class Slurm:
    """Slurm resources expressed with native ``#SBATCH`` terminology."""

    account: str
    partition: str
    time: str
    nodes: int
    ntasks: int
    cpus_per_task: int = 1
    mem_per_cpu: str | None = None

    def __post_init__(self) -> None:
        for value, label in (
            (self.account, "account"),
            (self.partition, "partition"),
            (self.time, "time"),
        ):
            require_nonempty(value, label)
            if any(character.isspace() for character in value):
                raise ValueError(f"{label} must not contain whitespace")
        for value, label in (
            (self.nodes, "nodes"),
            (self.ntasks, "ntasks"),
            (self.cpus_per_task, "cpus_per_task"),
        ):
            require_positive(value, label)
        if self.ntasks % self.nodes != 0:
            raise ValueError("ntasks must divide evenly across nodes")
        if self.mem_per_cpu is not None:
            require_nonempty(self.mem_per_cpu, "mem_per_cpu")
            if any(character.isspace() for character in self.mem_per_cpu):
                raise ValueError("mem_per_cpu must not contain whitespace")

    def to_plan(self) -> dict[str, object]:
        return {
            "kind": "slurm",
            "account": self.account,
            "partition": self.partition,
            "time": self.time,
            "nodes": self.nodes,
            "ntasks": self.ntasks,
            "cpus_per_task": self.cpus_per_task,
            "mem_per_cpu": self.mem_per_cpu,
        }


@dataclass(frozen=True, slots=True)
class Longship:
    """Complete declarative coupled workload compiled before solver launch."""

    case: openfoam.Case
    closures: tuple[Closure, ...] = ()
    observations: tuple[Observe, ...] = ()
    placement: Attached = dataclass_field(default_factory=Attached)
    scheduler: Slurm | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            require_nonempty(self.name or self.case.name, "longship name"),
        )
        object.__setattr__(self, "closures", tuple(self.closures))
        object.__setattr__(self, "observations", tuple(self.observations))
        closure_names = [closure.name for closure in self.closures]
        if len(closure_names) != len(set(closure_names)):
            raise ValueError("closure names must be unique")
        output_fields = [
            expression.field_name
            for closure in self.closures
            for expression in closure.outputs.values()
        ]
        if len(output_fields) != len(set(output_fields)):
            raise ValueError("multiple closures must not write the same output field")
        if self.scheduler is not None and self.scheduler.ntasks != self.case.ranks:
            raise ValueError("case ranks must match scheduler ntasks")

    def compile(self) -> CompiledPlan:
        """Validate declarations and return an immutable native-backed plan."""

        runtime = compile_runtime_plan(self)
        value = {
            "schema_version": 1,
            "name": self.name,
            "case": self.case.to_plan(),
            "closures": [closure.to_plan() for closure in self.closures],
            "observations": [item.to_plan() for item in self.observations],
            "placement": self.placement.to_plan(),
            "scheduler": None if self.scheduler is None else self.scheduler.to_plan(),
            "runtime": runtime,
        }
        return CompiledPlan.create(value)

    def launch(
        self,
        *,
        readiness_timeout: float = 120.0,
        termination_grace: float = 30.0,
        start_timeout: float | None = None,
        verbose: bool = True,
    ) -> Run:
        """Prepare an isolated case and return a non-blocking native Run."""

        from ._launch import launch

        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean")
        if start_timeout is not None and start_timeout <= 0:
            raise ValueError("start_timeout must be positive")
        run = launch(
            self,
            readiness_timeout=readiness_timeout,
            termination_grace=termination_grace,
        )
        if verbose:
            if self.scheduler is None:
                print(f"[FoamNordic] Sailing in background: {self.name}")
            else:
                job_id, state = run._wait_for_start(start_timeout)
                identity = job_id or "pending"
                if state == "running":
                    print(
                        f"[FoamNordic] Sailing has launched with Job ID: {identity}"
                    )
                    print(f"[FoamNordic] Sailing in background: {self.name}")
                elif state in {"submitting", "pending", "unknown"}:
                    print(
                        f"[FoamNordic] Sailing remains pending with Job ID: {identity}"
                    )
                else:
                    print(
                        f"[FoamNordic] Sailing reached {state} during launch: "
                        f"Job ID {identity}"
                    )
        elif self.scheduler is not None:
            run._wait_for_start(start_timeout)
        return run
