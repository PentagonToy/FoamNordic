"""Solver-native function graph tests."""

from __future__ import annotations

import pytest
from foamnordic.core.graph import (
    NativeAlgebraPlan,
    compile_native_program,
    compile_native_transport,
)
from foamnordic.execution.openfoam_rendering import (
    _scheme_requirements,
    render_dictionary,
    render_transform_dictionary,
)

import foamnordic as fno


def keqn_function(k, velocity_grad, nu, delta, C_k=0.094, C_e=1.048):
    k_positive = fno.Math.maximum(k, 1.0e-10)
    sqrt_k = fno.Math.sqrt(k_positive)
    strain = fno.Math.dev(2.0 * fno.Math.symm(velocity_grad))
    nut = C_k * delta * sqrt_k
    production = nut * fno.Math.ddot(velocity_grad, strain)
    dissipation = C_e * sqrt_k / delta
    k_new = fno.Field.transport(
        k,
        diffusivity=nu + nut,
        source=production,
        implicit_sink=dissipation,
        lower_bound=1.0e-10,
    )
    return {"k": k_new, "nut": C_k * delta * fno.Math.sqrt(k_new)}


def test_compile_native_transport_lowers_post_solve_dependency() -> None:
    plan = compile_native_transport(
        keqn_function,
        {
            "k": fno.Field("k"),
            "velocity_grad": fno.Field.grad("U"),
            "nu": fno.Field("nu"),
            "delta": fno.Field.delta(),
        },
        {"k": fno.Field("k"), "nut": fno.Field("nut")},
    )

    assert plan is not None
    assert plan.field.canonical == "k"
    assert plan.lower_bound == pytest.approx(1.0e-10)
    assert plan.outputs["k"].canonical == "k"
    assert not plan.outputs["nut"].contains_transport
    assert "sqrt(k)" in plan.outputs["nut"].canonical


def test_regular_function_is_not_traced() -> None:
    def scale(velocity):
        return {"velocity": velocity * 1.01}

    assert compile_native_transport(
        scale,
        {"velocity": fno.Field("U")},
        {"velocity": fno.Field("U")},
    ) is None


def test_native_function_is_traced_once_when_closure_is_declared() -> None:
    calls = 0

    def transport(k, nu):
        nonlocal calls
        calls += 1
        k_new = fno.Field.transport(
            k,
            diffusivity=nu,
            source=0.0 * k,
            implicit_sink=0.0 * k,
        )
        return {"k": k_new, "nut": 0.0 * k_new}

    closure = fno.Closure(
        name="kEqnFjord",
        operator=fno.Operator.function(transport),
        inputs={"k": fno.Field("k"), "nu": fno.Field("nu")},
        outputs={"k": fno.Field("k"), "nut": fno.Field("nut")},
    )

    assert calls == 1
    assert closure.native_transport is not None
    first = closure.to_plan()
    second = closure.to_plan()
    assert calls == 1
    assert first["native_transport"] == second["native_transport"]


def test_k_equation_transport_rejects_negative_bound() -> None:
    def transport(k, nu):
        k_new = fno.Field.transport(
            k,
            diffusivity=nu,
            source=0.0 * k,
            implicit_sink=0.0 * k,
            lower_bound=-1.0,
        )
        return {"k": k_new, "nut": 0.0 * k_new}

    with pytest.raises(ValueError, match="must not be negative"):
        fno.Closure(
            name="kEqnFjord",
            operator=fno.Operator.function(transport),
            inputs={"k": fno.Field("k"), "nu": fno.Field("nu")},
            outputs={"k": fno.Field("k"), "nut": fno.Field("nut")},
        )


def test_native_transport_renders_without_model_host_address() -> None:
    closure = fno.Closure(
        name="kEqnFjord",
        operator=fno.Operator.function(keqn_function),
        inputs={
            "k": fno.Field("k"),
            "velocity_grad": fno.Field.grad("U"),
            "nu": fno.Field("nu"),
            "delta": fno.Field.delta(),
        },
        outputs={"k": fno.Field("k"), "nut": fno.Field("nut")},
    )
    longship = fno.Longship(
        case=fno.OpenFOAM.Case(case_dir="case", run_dir="runs"),
        closures=(closure,),
        observations=(
            fno.Observe(
                summaries={
                    "U": ("min", "max", "mean", "l2"),
                    "p": ("min", "max", "mean"),
                },
                interval=10,
            ),
        ),
    )
    plan = compile_native_transport(
        keqn_function,
        closure.inputs,
        closure.outputs,
    )
    assert plan is not None

    _, rendered = render_dictionary(
        longship,
        closure,
        "unix:///unused.sock",
        True,
        observation_path="runs/observations.{rank}.jsonl",
        profile_every=7,
        native_program=plan,
    )

    assert "foamNordicTransport" in rendered
    assert "foamNordicClosure" not in rendered
    assert "unix:///unused.sock" not in rendered
    assert 'diffusivity       "add(nu,' in rendered
    assert 'nut                "multiply(' in rendered
    assert "profileEvery       7;" in rendered
    assert "profileProgram     kEqnFjord;" in rendered
    assert 'path        "runs/observations.{rank}.jsonl";' in rendered
    assert "fields      (U p);" in rendered
    assert "every       10;" in rendered
    assert _scheme_requirements(longship) == (
        ("divSchemes", "div(phi,k)", "Gauss linear"),
        ("gradSchemes", "grad(U)", "Gauss linear"),
    )
    runtime = longship.compile().as_dict()["runtime"]
    assert runtime["host_tasks"] == 0
    assert runtime["placement"]["data_path"] == "none"

    quiet = fno.Longship(
        case=longship.case,
        closures=(closure,),
    )
    _, quiet_rendered = render_dictionary(
        quiet,
        closure,
        "unix:///unused.sock",
        True,
        native_program=plan,
    )
    assert "profileEvery       0;" in quiet_rendered
    assert "observation" not in quiet_rendered


def test_nut_algebra_uses_native_openfoam_lane() -> None:
    def smagorinsky(velocity_grad, delta, coefficient=0.01):
        strain = fno.Math.symm(velocity_grad)
        return coefficient * delta * delta * fno.Math.sqrt(
            2.0 * fno.Math.ddot(strain, strain)
        )

    closure = fno.Closure(
        name="nutFjord",
        operator=fno.Operator.function(smagorinsky),
        inputs={
            "velocity_grad": fno.Field.grad("U"),
            "delta": fno.Field.delta(),
        },
        outputs={"nut": fno.Field("nut")},
    )
    longship = fno.Longship(
        case=fno.OpenFOAM.Case(case_dir="case", run_dir="runs"),
        closures=(closure,),
    )

    assert isinstance(closure.native_program, NativeAlgebraPlan)
    assert closure.execution_lane == "native"
    assert longship.compile().as_dict()["runtime"]["host_tasks"] == 0
    _, rendered = render_dictionary(
        longship,
        closure,
        "unix:///unused.sock",
        True,
        native_program=closure.native_program,
    )
    assert "foamNordicAlgebra" in rendered
    assert "foamNordicClosure" not in rendered
    assert 'nut "multiply(' in rendered
    assert "unix:///unused.sock" not in rendered


def test_scalar_bounds_use_native_openfoam_lane() -> None:
    def bounded(value):
        return fno.Math.clip(fno.Math.abs(value), 0.0, 1.0)

    closure = fno.Closure(
        name="nutFjord",
        operator=fno.Operator.function(bounded),
        inputs={"value": fno.Field("nut")},
        outputs={"nut": fno.Field("nut")},
    )

    assert isinstance(closure.native_program, NativeAlgebraPlan)
    assert closure.execution_lane == "native"
    assert closure.native_program.outputs["nut"].canonical == (
        "minimum(maximum(abs(nut),0),1)"
    )


def test_full_smagorinsky_algebra_uses_native_openfoam_lane() -> None:
    def smagorinsky(velocity_grad, delta, C_k=0.0265463553, C_e=1.048):
        strain = fno.Math.symm(velocity_grad)
        trace = fno.Math.einsum("...ii->...", strain)
        deviatoric = fno.Math.dev(strain)
        a = C_e / delta
        b = (2.0 / 3.0) * trace
        c = 2.0 * C_k * delta * fno.Math.ddot(deviatoric, strain)
        discriminant = b**2 + 4.0 * a * c
        sqrt_k = (-b + fno.Math.sqrt(fno.Math.maximum(discriminant, 0.0))) / (2.0 * a)
        return C_k * delta * fno.Math.maximum(sqrt_k, 0.0)

    closure = fno.Closure(
        name="nutFjord",
        operator=fno.Operator.function(smagorinsky),
        inputs={
            "velocity_grad": fno.Field.grad("U"),
            "delta": fno.Field.delta(),
        },
        outputs={"nut": fno.Field("nut")},
    )

    assert isinstance(closure.native_program, NativeAlgebraPlan)
    assert closure.execution_lane == "native"
    assert "divide(1.048,delta)" in closure.native_program.outputs["nut"].canonical


def test_full_wale_algebra_uses_native_openfoam_lane() -> None:
    def wale(velocity_grad, delta, C_w=0.325):
        squared = fno.Math.matmul(velocity_grad, velocity_grad)
        traceless = fno.Math.dev(fno.Math.symm(squared))
        strain = fno.Math.symm(velocity_grad)
        sd2 = fno.Math.maximum(fno.Math.ddot(traceless, traceless), 0.0)
        s2 = fno.Math.maximum(fno.Math.ddot(strain, strain), 0.0)
        numerator = sd2**1.5
        denominator = s2**2.5 + sd2**1.25
        safe = fno.Math.where(denominator > 0.0, denominator, 1.0)
        nut = (C_w * delta) ** 2 * numerator / safe
        return fno.Math.where(denominator > 0.0, nut, 0.0)

    closure = fno.Closure(
        name="nutFjord",
        operator=fno.Operator.function(wale),
        inputs={
            "velocity_grad": fno.Field.grad("U"),
            "delta": fno.Field.delta(),
        },
        outputs={"nut": fno.Field("nut")},
    )

    assert isinstance(closure.native_program, NativeAlgebraPlan)
    assert closure.execution_lane == "native"


def test_nut_algebra_accepts_a_logical_output_name() -> None:
    closure = fno.Closure(
        name="nutFjord",
        operator=fno.Operator.function(lambda delta: 0.01 * delta * delta),
        inputs={"delta": fno.Field.delta()},
        outputs={"eddy_viscosity": fno.Field("nut")},
    )

    assert isinstance(closure.native_program, NativeAlgebraPlan)
    assert closure.execution_lane == "native"


def test_native_algebra_with_update_stays_on_resident_lane() -> None:
    base = fno.Closure(
        name="nutFjord",
        operator=fno.Operator.function(lambda delta: 0.01 * delta * delta),
        inputs={"delta": fno.Field.delta()},
        outputs={"nut": fno.Field("nut")},
    )
    coupled = base.update(
        name="correction",
        operator=fno.Operator.function(lambda nut: nut),
        inputs={"nut": fno.Field("nut")},
        outputs={"corrected_nut": fno.Field("nutCorrected")},
    )

    assert coupled.native_program is None
    assert coupled.execution_lane == "resident"


def test_non_symbolic_function_remains_resident() -> None:
    def model(velocity):
        return float(velocity)

    assert compile_native_program(
        model,
        {"velocity": fno.Field("U")},
        {"velocity": fno.Field("U")},
    ) is None


def test_symbolic_python_condition_remains_resident() -> None:
    def conditional(value):
        if value > 0.0:
            return value
        return -value

    assert compile_native_program(
        conditional,
        {"value": fno.Field("nut")},
        {"value": fno.Field("nut")},
    ) is None


def test_patch_algebra_remains_resident() -> None:
    transform = fno.Transform(
        name="correctInlet",
        operator=fno.Operator.function(lambda velocity: 1.0001 * velocity),
        inputs={"velocity": fno.Field.patch("U", "inlet")},
        outputs={"velocity": fno.Field.patch("U", "inlet")},
        at="time_step_end",
    )

    assert transform.native_program is None
    assert transform.execution_lane == "resident"


def test_symbolic_transform_uses_native_openfoam_lane() -> None:
    def correct_velocity(velocity):
        return 1.0001 * velocity

    transform = fno.Transform(
        name="correctVelocity",
        operator=fno.Operator.function(correct_velocity),
        inputs={"velocity": fno.Field("U")},
        outputs={"velocity": fno.Field("U")},
        at="time_step_end",
    )
    longship = fno.Longship(
        case=fno.OpenFOAM.Case(case_dir="case", run_dir="runs"),
        transforms=(transform,),
    )

    assert isinstance(transform.native_program, NativeAlgebraPlan)
    assert transform.execution_lane == "native"
    assert longship.compile().as_dict()["runtime"]["host_tasks"] == 0
    rendered = render_transform_dictionary(
        transform,
        "unix:///unused.sock",
        True,
        native_program=transform.native_program,
    )
    assert "nativeAlgebra   true;" in rendered
    assert "outputs         (U);" in rendered
    assert 'outputExpressions ("multiply(1.0001,U)");' in rendered
    assert "unix:///unused.sock" not in rendered
