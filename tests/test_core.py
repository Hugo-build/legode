from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from legode import (
    AssemblyError,
    EquationSide,
    EvaluationError,
    Force,
    ODEEnsemble,
    ODEEquation,
    StateLayout,
)


@dataclass(frozen=True)
class Params:
    scale: float = 1.0


class State(Force[Params]):
    required_states = ("x",)

    def __call__(self, t, y, y_dot, layout):
        return self.params.scale * y[layout["x"]]


class Derivative(Force[Params]):
    required_derivatives = ("x",)

    def __call__(self, t, y, y_dot, layout):
        return y_dot[layout["x"]]


class Constant(Force[Params]):
    def __call__(self, t, y, y_dot, layout):
        return self.params.scale


class BadShape(Force[Params]):
    def __call__(self, t, y, y_dot, layout):
        return np.ones(2)


def test_force_requires_dataclass_parameters() -> None:
    with pytest.raises(TypeError, match="dataclass"):
        Constant({"scale": 1.0})  # type: ignore[arg-type]


def test_side_arithmetic_preserves_signed_order() -> None:
    params = Params(2.0)
    side = State(params) + Constant(params) - Constant(Params(0.5))
    assert isinstance(side, EquationSide)
    layout = StateLayout(("x",))
    result = side.evaluate(
        0.0,
        np.array([3.0]),
        np.array([0.0]),
        layout,
        expected_size=1,
        label="test",
    )
    np.testing.assert_allclose(result, [7.5])


def test_empty_side_represents_zero() -> None:
    result = EquationSide().evaluate(
        0.0,
        np.array([1.0]),
        np.array([0.0]),
        StateLayout(("x",)),
        expected_size=1,
        label="zero",
    )
    np.testing.assert_array_equal(result, [0.0])


def test_ensemble_uses_declaration_order_and_residual_is_lhs_minus_rhs() -> None:
    params = Params()

    class VelocityDerivative(Force[Params]):
        required_derivatives = ("velocity",)

        def __call__(self, t, y, y_dot, layout):
            return y_dot[layout["velocity"]]

    velocity = ODEEquation(
        solves_for=("velocity",), lhs=VelocityDerivative(params), rhs=Constant(params)
    )

    class PositionDerivative(Force[Params]):
        required_derivatives = ("position",)

        def __call__(self, t, y, y_dot, layout):
            return y_dot[layout["position"]]

    position = ODEEquation(
        solves_for=("position",), lhs=PositionDerivative(params), rhs=None
    )
    ensemble = ODEEnsemble([velocity, position])
    assert ensemble.layout.names == ("velocity", "position")
    np.testing.assert_allclose(
        ensemble.residual(0.0, [4.0, 5.0], [2.0, 3.0]), [1.0, 3.0]
    )


def test_vector_equation_is_supported() -> None:
    class VectorDerivative(Force[Params]):
        required_derivatives = ("x", "v")

        def __call__(self, t, y, y_dot, layout):
            return y_dot[[layout["x"], layout["v"]]]

    equation = ODEEquation(
        solves_for=("x", "v"), lhs=VectorDerivative(Params()), rhs=None
    )
    ensemble = ODEEnsemble([equation])
    np.testing.assert_array_equal(ensemble.residual(0, [0, 0], [2, 3]), [2, 3])


def test_state_layout_converts_named_values() -> None:
    layout = StateLayout(("x", "v"))
    np.testing.assert_array_equal(layout.vector({"v": 2, "x": 1}), [1, 2])
    assert layout.named([1, 2]) == {"x": 1.0, "v": 2.0}
    with pytest.raises(ValueError, match="missing"):
        layout.vector({"x": 1})


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: ODEEnsemble([]), "at least one"),
        (
            lambda: ODEEnsemble(
                [
                    ODEEquation(("x",), Constant(Params()), None, "one"),
                    ODEEquation(("x",), Constant(Params()), None, "two"),
                ]
            ),
            "both",
        ),
        (
            lambda: ODEEquation(("x", "x"), Constant(Params()), None),
            "same state twice",
        ),
        (lambda: ODEEquation(("x",), None, None), "two empty sides"),
    ],
)
def test_invalid_assembly_is_rejected(factory, message: str) -> None:
    with pytest.raises(AssemblyError, match=message):
        factory()


def test_missing_required_state_is_rejected() -> None:
    ensemble_equation = ODEEquation(("velocity",), State(Params()), None)
    with pytest.raises(AssemblyError, match="unsolved"):
        ODEEnsemble([ensemble_equation])


def test_wrong_shape_and_nonfinite_outputs_are_rejected() -> None:
    wrong = ODEEnsemble([ODEEquation(("x",), BadShape(Params()), None)])
    with pytest.raises(EvaluationError, match="shape"):
        wrong.residual(0.0, [0.0], [0.0])

    infinite = ODEEnsemble([ODEEquation(("x",), Constant(Params(np.inf)), None)])
    with pytest.raises(EvaluationError, match="non-finite"):
        infinite.residual(0.0, [0.0], [0.0])


def test_input_vectors_are_validated() -> None:
    system = ODEEnsemble([ODEEquation(("x",), Derivative(Params()), None)])
    with pytest.raises(EvaluationError, match="shape"):
        system.residual(0.0, [1.0, 2.0], [0.0])
    with pytest.raises(EvaluationError, match="non-finite"):
        system.residual(0.0, [np.nan], [0.0])
