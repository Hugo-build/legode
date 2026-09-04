from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from legode import DerivativeSolveError, EvaluationError, Force, ODEEnsemble, ODEEquation


@dataclass(frozen=True)
class Params:
    rate: float = 2.0


class Derivative(Force[Params]):
    required_derivatives = ("x",)

    def __call__(self, t, y, y_dot, layout):
        return y_dot[layout["x"]]


class Decay(Force[Params]):
    required_states = ("x",)

    def __call__(self, t, y, y_dot, layout):
        return -self.params.rate * y[layout["x"]]


def decay_system(rate: float = 2.0) -> ODEEnsemble:
    params = Params(rate)
    return ODEEnsemble([ODEEquation(("x",), Derivative(params), Decay(params))])


def test_adapter_solves_implicit_derivative_and_reuses_guess() -> None:
    rhs = decay_system().as_scipy_rhs()
    np.testing.assert_allclose(rhs(0.0, [3.0]), [-6.0])
    np.testing.assert_allclose(rhs.last_derivative, [-6.0])
    np.testing.assert_allclose(rhs(0.1, [2.0]), [-4.0])
    rhs.reset()
    assert rhs.last_derivative is None


def test_custom_initial_guess_and_validation() -> None:
    seen: list[float] = []

    def guess(t, y):
        seen.append(t)
        return [-1.0]

    rhs = decay_system().as_scipy_rhs(initial_guess=guess)
    np.testing.assert_allclose(rhs(0.25, [2.0]), [-4.0])
    assert seen == [0.25]

    bad = decay_system().as_scipy_rhs(initial_guess=lambda t, y: [1.0, 2.0])
    with pytest.raises(EvaluationError, match="guess"):
        bad(0.0, [1.0])


def test_solve_ivp_integration_matches_analytic_solution() -> None:
    system = decay_system(rate=1.5)
    result = solve_ivp(
        system.as_scipy_rhs(tol=1e-11),
        (0.0, 2.0),
        [3.0],
        t_eval=np.linspace(0.0, 2.0, 21),
        rtol=1e-9,
        atol=1e-11,
    )
    assert result.success
    np.testing.assert_allclose(result.y[0], 3.0 * np.exp(-1.5 * result.t), rtol=2e-7)


def test_nonlinear_derivative_equation() -> None:
    class CubedDerivative(Force[Params]):
        required_derivatives = ("x",)

        def __call__(self, t, y, y_dot, layout):
            return y_dot[layout["x"]] ** 3

    class State(Force[Params]):
        required_states = ("x",)

        def __call__(self, t, y, y_dot, layout):
            return y[layout["x"]]

    system = ODEEnsemble(
        [ODEEquation(("x",), CubedDerivative(Params()), State(Params()))]
    )
    rhs = system.as_scipy_rhs(initial_guess=lambda t, y: [1.0])
    np.testing.assert_allclose(rhs(0.0, [8.0]), [2.0], rtol=1e-8)


def test_root_failure_contains_diagnostics() -> None:
    class One(Force[Params]):
        def __call__(self, t, y, y_dot, layout):
            return 1.0

    system = ODEEnsemble([ODEEquation(("x",), One(Params()), None)])
    with pytest.raises(DerivativeSolveError) as captured:
        system.as_scipy_rhs()(1.25, [0.0])
    error = captured.value
    assert error.time == 1.25
    assert error.residual_norm == pytest.approx(1.0)
    np.testing.assert_array_equal(error.last_guess, [0.0])
    assert error.solver_message


def test_adapter_validates_tolerance_and_state() -> None:
    system = decay_system()
    with pytest.raises(ValueError, match="tol"):
        system.as_scipy_rhs(tol=0)
    with pytest.raises(ValueError, match="residual_tol"):
        system.as_scipy_rhs(residual_tol=0)
    rhs = system.as_scipy_rhs()
    with pytest.raises(EvaluationError, match="shape"):
        rhs(0.0, [1.0, 2.0])
    with pytest.raises(EvaluationError, match="non-finite"):
        rhs(0.0, [np.inf])
