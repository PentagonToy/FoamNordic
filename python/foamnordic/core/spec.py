"""Immutable public declarations compiled before a solver starts."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
import hashlib
from types import MappingProxyType
from typing import TYPE_CHECKING, Callable, Mapping

from .. import openfoam
from .expressions import FieldExpression
from .native_plan import compile_runtime_plan
from .paths import PathInput, path_from
from .plan import CompiledPlan
from .validation import require_nonempty, require_positive
from ..random import Key, key as random_key

if TYPE_CHECKING:
    from ..combustion.progress_variable import ProgressVariable
    from ..execution.run import Run


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


def _program_key(value: Key | int | None, seed: int | None, label: str) -> Key:
    if value is not None and seed is not None:
        raise ValueError(f"{label} accepts either key or seed, not both")
    if value is not None:
        if isinstance(value, int) and not isinstance(value, bool):
            return random_key(value)
        if not isinstance(value, Key):
            raise TypeError(f"{label} key must be a Random.Key")
        return value
    return random_key(42 if seed is None else seed)


@dataclass(frozen=True, slots=True)
class Operator:
    """Evaluation implementation used by a field program.

    ``model()`` accepts one FNOM manifest.  The manifest selects compiled C++,
    ONNX, Joblib, or Equinox internally, so backend names do not leak into the
    orchestration API. ``function()`` packages a callable into the isolated
    run directory and evaluates it in a resident Python worker.
    """

    kind: str
    source: PathInput | Callable[..., object]

    def __post_init__(self) -> None:
        if self.kind == "model":
            path = path_from(self.source)  # type: ignore[arg-type]
            if path.suffix.lower() != ".fnom":
                raise ValueError("Operator.model requires an .fnom manifest")
            object.__setattr__(self, "source", path)
        elif self.kind == "function":
            if not callable(self.source):
                raise TypeError("Operator.function requires a callable")
        else:
            raise ValueError("operator kind must be model or function")

    @classmethod
    def model(cls, path: PathInput) -> "Operator":
        """Load any supported model backend through one FNOM manifest."""

        return cls("model", path)

    @classmethod
    def function(cls, function: Callable[..., object]) -> "Operator":
        """Declare a managed in-memory field function.

        Input arrays are supplied by logical port name. A function may also
        request ``key``, ``exchange_index``, ``physical_time``, or ``rank``
        as keyword parameters. Random keys are derived deterministically from
        the field-program key, stable program identity, exchange index, and
        optionally the solver rank. Compatibility ``seed`` and NumPy ``rng``
        injection remain available for source compatibility.
        """

        return cls("function", function)

    @property
    def artifact(self):
        return self.source if self.kind == "model" else None

    def to_plan(self) -> dict[str, object]:
        if self.kind == "model":
            return {"kind": "model", "manifest": str(self.source)}
        function = self.source
        identity = "\0".join(
            (
                str(getattr(function, "__module__", "")),
                str(
                    getattr(
                        function,
                        "__qualname__",
                        getattr(function, "__name__", ""),
                    )
                ),
            )
        ).encode("utf-8")
        return {
            "kind": "function",
            "name": getattr(
                function,
                "__qualname__",
                getattr(function, "__name__", "callable"),
            ),
            "module": getattr(function, "__module__", None),
            "identity": f"sha256:{hashlib.sha256(identity).hexdigest()}",
        }


@dataclass(frozen=True, slots=True, init=False)
class Closure:
    """Bind model artifact tensors to native OpenFOAM expressions and fields."""

    name: str
    operator: Operator
    inputs: Mapping[str, FieldExpression]
    outputs: Mapping[str, FieldExpression]
    key: Key

    def __init__(
        self,
        name: str,
        operator: Operator | None = None,
        inputs: Mapping[str, FieldExpression] | None = None,
        outputs: Mapping[str, FieldExpression] | None = None,
        key: Key | int | None = None,
        *,
        artifact: PathInput | None = None,
        seed: int | None = None,
    ) -> None:
        if (artifact is None) == (operator is None):
            raise ValueError("closure requires exactly one of operator or artifact")
        selected = Operator.model(artifact) if operator is None else operator
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "operator", selected)
        object.__setattr__(self, "inputs", {} if inputs is None else inputs)
        object.__setattr__(self, "outputs", {} if outputs is None else outputs)
        object.__setattr__(self, "key", _program_key(key, seed, "closure"))
        self.__post_init__()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", require_nonempty(self.name, "closure name"))
        if not isinstance(self.operator, Operator):
            raise TypeError("closure operator must be an Operator")
        object.__setattr__(self, "inputs", _frozen_mapping(self.inputs, "inputs"))
        object.__setattr__(self, "outputs", _frozen_mapping(self.outputs, "outputs"))
        for logical_name, expression in self.outputs.items():
            if expression.operation != "field":
                raise ValueError(
                    f"output {logical_name!r} must bind to a mutable field"
                )

    @property
    def artifact(self):
        return self.operator.artifact

    def to_plan(self) -> dict[str, object]:
        return {
            "name": self.name,
            "artifact": None if self.artifact is None else str(self.artifact),
            "operator": self.operator.to_plan(),
            "inputs": {name: value.to_plan() for name, value in self.inputs.items()},
            "outputs": {name: value.to_plan() for name, value in self.outputs.items()},
            "key": self.key.to_plan(),
        }

    @property
    def seed(self) -> int:
        """Compatibility view of the low 32 bits of the root entropy."""

        return self.key.entropy[0]


@dataclass(frozen=True, slots=True, init=False)
class Transform:
    """Apply one general field-to-field model at a declared solver boundary."""

    name: str
    operator: Operator
    inputs: Mapping[str, FieldExpression]
    outputs: Mapping[str, FieldExpression]
    at: str = "time_step_start"
    key: Key

    def __init__(
        self,
        name: str,
        operator: Operator,
        inputs: Mapping[str, FieldExpression],
        outputs: Mapping[str, FieldExpression],
        at: str = "time_step_start",
        key: Key | int | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "at", at)
        object.__setattr__(self, "key", _program_key(key, seed, "transform"))
        self.__post_init__()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", require_nonempty(self.name, "transform name"))
        if not self.name.replace("_", "").isalnum() or self.name[0].isdigit():
            raise ValueError("transform name must be an OpenFOAM word")
        if not isinstance(self.operator, Operator):
            raise TypeError("transform operator must be an Operator")
        object.__setattr__(self, "inputs", _frozen_mapping(self.inputs, "inputs"))
        object.__setattr__(self, "outputs", _frozen_mapping(self.outputs, "outputs"))
        for logical_name, expression in self.inputs.items():
            if expression.operation != "field":
                raise NotImplementedError(
                    f"transform input {logical_name!r} currently requires "
                    "a stored field"
                )
        for logical_name, expression in self.outputs.items():
            if expression.operation != "field":
                raise ValueError(
                    f"output {logical_name!r} must bind to a mutable field"
                )
        supported = {
            "time_step_start",
            "outer_corrector",
            "pressure_corrected",
            "time_step_end",
        }
        if self.at not in supported:
            raise ValueError(
                "transform at must be time_step_start, outer_corrector, "
                "pressure_corrected, or time_step_end"
            )

    @property
    def artifact(self):
        return self.operator.artifact

    def to_plan(self) -> dict[str, object]:
        return {
            "name": self.name,
            "operator": self.operator.to_plan(),
            "inputs": {name: value.to_plan() for name, value in self.inputs.items()},
            "outputs": {name: value.to_plan() for name, value in self.outputs.items()},
            "at": self.at,
            "key": self.key.to_plan(),
        }

    @property
    def seed(self) -> int:
        """Compatibility view of the low 32 bits of the root entropy."""

        return self.key.entropy[0]


@dataclass(frozen=True, slots=True)
class Observe:
    """Read-only field summaries sampled at a solver-friendly interval.

    ``interval`` is the only cadence control. Queue length, byte limits, and
    overflow behavior remain private runtime safety details.
    """

    summaries: Mapping[str, tuple[str, ...]]
    interval: int = 1

    def __post_init__(self) -> None:
        require_positive(self.interval, "observation interval")
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
            "schedule": {"interval": self.interval},
        }


@dataclass(frozen=True, slots=True)
class Attached:
    """Place one native ClosureHost beside each solver node."""

    closure_cpus_per_node: int | None = None
    data_path: str = "auto"

    def __post_init__(self) -> None:
        if self.closure_cpus_per_node is not None:
            require_positive(self.closure_cpus_per_node, "closure_cpus_per_node")
        if self.data_path not in {"auto", "shm", "uds", "ucx", "tcp"}:
            raise ValueError("data_path must be auto, shm, uds, ucx, or tcp")

    def to_plan(self) -> dict[str, object]:
        return {
            "kind": "attached",
            "closure_cpus_per_node": (
                "auto"
                if self.closure_cpus_per_node is None
                else self.closure_cpus_per_node
            ),
            "data_path": self.data_path,
        }


@dataclass(frozen=True, slots=True)
class SlurmOpenFOAM:
    """OpenFOAM step resources expressed with ``#SBATCH`` names."""

    nodes: int
    ntasks: int
    cpus_per_task: int = 1
    mem_per_cpu: str | None = None

    def __post_init__(self) -> None:
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
            "nodes": self.nodes,
            "ntasks": self.ntasks,
            "cpus_per_task": self.cpus_per_task,
            "mem_per_cpu": self.mem_per_cpu,
        }


@dataclass(frozen=True, slots=True)
class SlurmModel:
    """Resources for exactly one ClosureHost task per OpenFOAM node."""

    cpus_per_task: int = 1
    mem_per_cpu: str | None = None

    def __post_init__(self) -> None:
        require_positive(self.cpus_per_task, "cpus_per_task")
        if self.mem_per_cpu is not None:
            require_nonempty(self.mem_per_cpu, "mem_per_cpu")
            if any(character.isspace() for character in self.mem_per_cpu):
                raise ValueError("mem_per_cpu must not contain whitespace")

    def to_plan(self) -> dict[str, object]:
        return {
            "cpus_per_task": self.cpus_per_task,
            "mem_per_cpu": self.mem_per_cpu,
        }


@dataclass(frozen=True, slots=True, init=False)
class Slurm:
    """Slurm resources using native ``#SBATCH`` terminology."""

    account: str
    partition: str
    time: str
    _openfoam: SlurmOpenFOAM
    _model: SlurmModel
    _explicit_model: bool

    def __init__(
        self,
        account: str,
        partition: str,
        time: str,
        *,
        openfoam: SlurmOpenFOAM,
        model: SlurmModel | None = None,
    ) -> None:
        for value, label in (
            (account, "account"),
            (partition, "partition"),
            (time, "time"),
        ):
            require_nonempty(value, label)
            if any(character.isspace() for character in value):
                raise ValueError(f"{label} must not contain whitespace")
        if not isinstance(openfoam, SlurmOpenFOAM):
            raise TypeError("openfoam must be created by Slurm.openfoam()")
        if model is not None and not isinstance(model, SlurmModel):
            raise TypeError("model must be created by Slurm.model()")
        object.__setattr__(self, "account", account)
        object.__setattr__(self, "partition", partition)
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "_openfoam", openfoam)
        object.__setattr__(self, "_model", model or SlurmModel())
        object.__setattr__(self, "_explicit_model", model is not None)
        if (
            model is not None
            and model.mem_per_cpu is not None
            and openfoam.mem_per_cpu is None
        ):
            raise ValueError(
                "model mem_per_cpu requires OpenFOAM mem_per_cpu"
            )

    @staticmethod
    def openfoam(
        *,
        nodes: int,
        ntasks: int,
        cpus_per_task: int = 1,
        mem_per_cpu: str | None = None,
    ) -> SlurmOpenFOAM:
        """Declare the OpenFOAM Slurm step with standard directive names."""

        return SlurmOpenFOAM(nodes, ntasks, cpus_per_task, mem_per_cpu)

    @staticmethod
    def model(
        *,
        cpus_per_task: int = 1,
        mem_per_cpu: str | None = None,
    ) -> SlurmModel:
        """Declare resources for exactly one ClosureHost task per node."""

        return SlurmModel(cpus_per_task, mem_per_cpu)

    @property
    def openfoam_resources(self) -> SlurmOpenFOAM:
        return self._openfoam

    @property
    def model_resources(self) -> SlurmModel:
        return self._model

    @property
    def has_model_resources(self) -> bool:
        return self._explicit_model

    @property
    def nodes(self) -> int:
        return self._openfoam.nodes

    @property
    def ntasks(self) -> int:
        return self._openfoam.ntasks

    @property
    def cpus_per_task(self) -> int:
        return self._openfoam.cpus_per_task

    @property
    def mem_per_cpu(self) -> str | None:
        return self._openfoam.mem_per_cpu

    def to_plan(self) -> dict[str, object]:
        plan: dict[str, object] = {
            "kind": "slurm",
            "account": self.account,
            "partition": self.partition,
            "time": self.time,
            "openfoam": self._openfoam.to_plan(),
        }
        if self._explicit_model:
            plan["model"] = self._model.to_plan()
        return plan


@dataclass(frozen=True, slots=True)
class Longship:
    """Complete declarative coupled workload compiled before solver launch."""

    case: openfoam.Case
    closures: tuple[Closure, ...] = ()
    transforms: tuple[Transform, ...] = ()
    observations: tuple[Observe, ...] = ()
    placement: Attached = dataclass_field(default_factory=Attached)
    scheduler: Slurm | None = None
    name: str | None = None
    combustion: ProgressVariable | None = None
    verbose: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            require_nonempty(self.name or self.case.name, "longship name"),
        )
        object.__setattr__(self, "closures", tuple(self.closures))
        object.__setattr__(self, "transforms", tuple(self.transforms))
        object.__setattr__(self, "observations", tuple(self.observations))
        if not isinstance(self.verbose, bool):
            raise TypeError("verbose must be a boolean")
        if self.combustion is not None:
            from ..combustion.progress_variable import ProgressVariable

            if not isinstance(self.combustion, ProgressVariable):
                raise TypeError("combustion must be a Combustion.ProgressVariable")
            if self.closures:
                raise ValueError(
                    "combustion owns its reaction-rate and manifold closures; "
                    "do not also pass closures"
                )
        programs = self.field_programs
        names = [program.name for program in programs]
        if len(names) != len(set(names)):
            raise ValueError("field-program names must be unique")
        writers = [
            (
                expression.field_name,
                "closure" if isinstance(program, Closure) else program.at,
            )
            for program in programs
            for expression in program.outputs.values()
        ]
        if len(writers) != len(set(writers)):
            raise ValueError(
                "multiple field programs must not write the same output field "
                "at the same solver stage"
            )
        if self.scheduler is not None and self.scheduler.ntasks != self.case.ranks:
            raise ValueError("case ranks must match scheduler OpenFOAM ranks")
        if self.verbose and self.scheduler is not None:
            self._display_resources()

    @property
    def closure_programs(self) -> tuple[Closure, ...]:
        if self.combustion is None:
            return self.closures
        return self.combustion.programs(self.case)

    @property
    def field_programs(self) -> tuple[Closure | Transform, ...]:
        return (*self.closure_programs, *self.transforms)

    def compile(self) -> CompiledPlan:
        """Validate declarations and return an immutable native-backed plan."""

        runtime = compile_runtime_plan(self)
        value = {
            "schema_version": 2,
            "name": self.name,
            "case": self.case.to_plan(),
            "closures": [closure.to_plan() for closure in self.closure_programs],
            "transforms": [transform.to_plan() for transform in self.transforms],
            "combustion": (
                None if self.combustion is None else self.combustion.to_plan()
            ),
            "observations": [item.to_plan() for item in self.observations],
            "placement": self.placement.to_plan(),
            "scheduler": None if self.scheduler is None else self.scheduler.to_plan(),
            "runtime": runtime,
        }
        return CompiledPlan.create(value)

    def _display_resources(self) -> None:
        from ..execution.resources import display_resources

        display_resources(self)

    def launch(
        self,
        *,
        readiness_timeout: float = 120.0,
        termination_grace: float = 30.0,
        start_timeout: float | None = None,
        verbose: bool | None = None,
    ) -> Run:
        """Prepare an isolated case and return a non-blocking native Run."""

        from ..execution.launch import launch

        if verbose is not None and not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean or None")
        selected_verbose = self.verbose if verbose is None else verbose
        if start_timeout is not None and start_timeout <= 0:
            raise ValueError("start_timeout must be positive")
        run = launch(
            self,
            readiness_timeout=readiness_timeout,
            termination_grace=termination_grace,
            verbose=selected_verbose,
        )
        if selected_verbose:
            if self.scheduler is None:
                print(f"[FoamNordic] Sailing in background: {self.name}")
            else:
                def report_pending(job_id: str, estimated: str) -> None:
                    suffix = f" (est. start: {estimated})" if estimated else ""
                    print(
                        f"[FoamNordic] Sailing submitted with Job ID: "
                        f"{job_id}{suffix}"
                    )

                job_id, state = run._wait_for_start(
                    start_timeout,
                    pending_callback=report_pending,
                )
                identity = job_id or "pending"
                if state == "running":
                    print(
                        f"[FoamNordic] Sailing has launched with Job ID: {identity}"
                    )
                    started = run._slurm_start_time(identity)
                    if started:
                        print(f"[FoamNordic] Sailing started at: {started}")
                    print(f"[FoamNordic] Sailing in background: {self.name}")
                elif state in {"submitting", "pending", "unknown"}:
                    estimated = (
                        run._slurm_start_time(job_id, estimated=True)
                        if job_id is not None
                        else ""
                    )
                    suffix = f" (est. start: {estimated})" if estimated else ""
                    print(
                        f"[FoamNordic] Sailing remains pending with Job ID: "
                        f"{identity}{suffix}"
                    )
                else:
                    if job_id is not None:
                        started = run._slurm_start_time(job_id)
                        if started:
                            print(f"[FoamNordic] Sailing started at: {started}")
                    print(
                        f"[FoamNordic] Sailing reached {state} during launch: "
                        f"Job ID {identity}"
                    )
        elif self.scheduler is not None:
            run._wait_for_start(start_timeout)
        return run
