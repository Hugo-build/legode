"""LEGOde Duffing study: direct integration versus active Kriging.

The dynamic balance is

    mass * acceleration + damping * velocity + stiffness * displacement
        = excitation(time) - nonlinear_restoring(displacement).

The known linear terms stay on the LEGOde LHS. The expensive nonlinear RHS
term is evaluated only at actively selected displacement samples and learned
with a one-dimensional Gaussian process. Both the exact and learned systems
are integrated through LEGOde's ``solve_ivp`` adapter.

The ``# %%`` markers make this file usable as either a normal Python script or
a cell-based notebook in Jupyter-compatible editors.

Run from the project root:

    python notebooks/duffing_ak_legode.py
"""

# %% Imports and data models
from __future__ import annotations

import csv
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from legode import Force, ODEEnsemble, ODEEquation

FloatArray = NDArray[np.float64]
Predictor = Callable[[FloatArray], FloatArray]


@dataclass(frozen=True)
class DuffingParameters:
    """Physical parameters shared by exact LEGOde terms."""

    mass: float = 1.0
    damping: float = 0.12
    stiffness: float = 1.0
    cubic: float = 0.2
    x0: float = 0.15
    v0: float = 0.0

    def __post_init__(self) -> None:
        if self.mass <= 0:
            raise ValueError("mass must be positive")
        if self.cubic < 0:
            raise ValueError("cubic must be non-negative")


@dataclass(frozen=True)
class LearnedRHSParameters:
    """Parameter structure used by the learned RHS force term."""

    predict: Predictor


@dataclass(frozen=True)
class Trajectory:
    t: FloatArray
    x: FloatArray
    v: FloatArray
    x_dot: FloatArray
    v_dot: FloatArray
    rhs_evaluations: int

    @property
    def states(self) -> FloatArray:
        return np.column_stack((self.x, self.v))

    @property
    def derivatives(self) -> FloatArray:
        return np.column_stack((self.x_dot, self.v_dot))


@dataclass(frozen=True)
class BudgetResult:
    budget: int
    residual_rmse: float
    displacement_rmse: float
    displacement_max_error: float
    exact_rhs_evaluations: int


# %% LEGOde terms
class PositionDerivative(Force[DuffingParameters]):
    required_derivatives = ("displacement",)

    def __call__(self, t, y, y_dot, layout):
        return y_dot[layout["displacement"]]


class Velocity(Force[DuffingParameters]):
    required_states = ("velocity",)

    def __call__(self, t, y, y_dot, layout):
        return y[layout["velocity"]]


class Inertia(Force[DuffingParameters]):
    required_derivatives = ("velocity",)

    def __call__(self, t, y, y_dot, layout):
        return self.params.mass * y_dot[layout["velocity"]]


class Damping(Force[DuffingParameters]):
    required_states = ("velocity",)

    def __call__(self, t, y, y_dot, layout):
        return self.params.damping * y[layout["velocity"]]


class LinearRestoring(Force[DuffingParameters]):
    required_states = ("displacement",)

    def __call__(self, t, y, y_dot, layout):
        return self.params.stiffness * y[layout["displacement"]]


class Excitation(Force[DuffingParameters]):
    def __call__(self, t, y, y_dot, layout):
        return excitation(t)


class ExactNonlinearRestoring(Force[DuffingParameters]):
    required_states = ("displacement",)

    def __call__(self, t, y, y_dot, layout):
        x = y[layout["displacement"]]
        return self.params.cubic * x**3


class LearnedNonlinearRestoring(Force[LearnedRHSParameters]):
    required_states = ("displacement",)

    def __call__(self, t, y, y_dot, layout):
        x = np.asarray([y[layout["displacement"]]], dtype=float)
        return float(self.params.predict(x)[0])


def excitation(t: float | FloatArray) -> float | FloatArray:
    """Test excitation shared by exact and learned systems."""
    values = np.asarray(t)
    result = 0.68 * np.sin(0.91 * values + 0.27) + 0.32 * np.sin(1.58 * values + 1.10)
    return float(result) if result.ndim == 0 else result


def exact_nonlinear_rhs(params: DuffingParameters, x: FloatArray) -> FloatArray:
    """The expensive RHS contribution sampled by active Kriging."""
    return params.cubic * np.asarray(x, dtype=float) ** 3


def build_exact_system(params: DuffingParameters) -> ODEEnsemble:
    """Build the exact implicit Duffing system from reusable LEGOde terms."""
    kinematic = ODEEquation(
        solves_for=("displacement",),
        lhs=PositionDerivative(params),
        rhs=Velocity(params),
        name="kinematic balance",
    )
    dynamic = ODEEquation(
        solves_for=("velocity",),
        lhs=Inertia(params) + Damping(params) + LinearRestoring(params),
        rhs=Excitation(params) - ExactNonlinearRestoring(params),
        name="dynamic balance",
    )
    return ODEEnsemble((kinematic, dynamic))


def build_learned_system(
    params: DuffingParameters,
    predict: Predictor,
) -> ODEEnsemble:
    """Replace only the nonlinear RHS term with the Kriging predictor."""
    kinematic = ODEEquation(
        solves_for=("displacement",),
        lhs=PositionDerivative(params),
        rhs=Velocity(params),
        name="kinematic balance",
    )
    dynamic = ODEEquation(
        solves_for=("velocity",),
        lhs=Inertia(params) + Damping(params) + LinearRestoring(params),
        rhs=Excitation(params) - LearnedNonlinearRestoring(LearnedRHSParameters(predict)),
        name="learned dynamic balance",
    )
    return ODEEnsemble((kinematic, dynamic))


# %% Direct and learned integration
def solve_system(
    system: ODEEnsemble,
    params: DuffingParameters,
    t_eval: FloatArray,
    *,
    method: str = "DOP853",
    rtol: float = 1e-9,
    atol: float = 1e-11,
) -> Trajectory:
    """Integrate a LEGOde ensemble and recover derivatives at output times."""
    initial = system.layout.vector(
        {"displacement": params.x0, "velocity": params.v0}
    )
    adapter = system.as_scipy_rhs(tol=1e-11)
    solution = solve_ivp(
        adapter,
        (float(t_eval[0]), float(t_eval[-1])),
        initial,
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    # Use a fresh adapter for deterministic derivative reconstruction in time order.
    derivative_adapter = system.as_scipy_rhs(tol=1e-11)
    derivatives = np.column_stack(
        [
            derivative_adapter(float(time), state)
            for time, state in zip(solution.t, solution.y.T, strict=True)
        ]
    ).T
    return Trajectory(
        t=np.asarray(solution.t),
        x=np.asarray(solution.y[system.layout["displacement"]]),
        v=np.asarray(solution.y[system.layout["velocity"]]),
        x_dot=derivatives[:, system.layout["displacement"]],
        v_dot=derivatives[:, system.layout["velocity"]],
        rhs_evaluations=int(solution.nfev),
    )


def physical_residual_rmse(
    learned_system: ODEEnsemble,
    reference: Trajectory,
) -> float:
    """Evaluate learned LHS-RHS imbalance on the exact trajectory."""
    residuals = np.vstack(
        [
            learned_system.residual(time, state, derivative)
            for time, state, derivative in zip(
                reference.t,
                reference.states,
                reference.derivatives,
                strict=True,
            )
        ]
    )
    dynamic_row = learned_system.layout["velocity"]
    return float(np.sqrt(np.mean(residuals[:, dynamic_row] ** 2)))


# %% Active Kriging on the nonlinear RHS
def make_gp(x_scale: float) -> GaussianProcessRegressor:
    """Create a stable one-dimensional Matérn-5/2 Kriging model."""
    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(
        length_scale=max(0.2 * x_scale, 1e-3),
        length_scale_bounds="fixed",
        nu=2.5,
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-10,
        normalize_y=True,
        optimizer=None,
    )


def initial_sample_indices(x: FloatArray, count: int) -> list[int]:
    """Cover the observed displacement range with deterministic initial samples."""
    if count < 2:
        raise ValueError("At least two initial samples are required")
    targets = np.linspace(float(np.min(x)), float(np.max(x)), count)
    indices: list[int] = []
    for target in targets:
        order = np.argsort(np.abs(x - target))
        index = next(int(candidate) for candidate in order if int(candidate) not in indices)
        indices.append(index)
    return indices


def fit_active_kriging(
    params: DuffingParameters,
    reference: Trajectory,
    budgets: Sequence[int],
) -> tuple[list[BudgetResult], GaussianProcessRegressor, list[int]]:
    """Actively sample the exact RHS and measure the LEGOde residual reduction."""
    if not budgets or sorted(set(budgets)) != list(budgets):
        raise ValueError("budgets must be a non-empty, strictly increasing sequence")
    if budgets[0] < 2 or budgets[-1] > reference.x.size:
        raise ValueError("budgets must fit within the candidate trajectory")

    selected = initial_sample_indices(reference.x, budgets[0])
    selected_set = set(selected)
    results: list[BudgetResult] = []
    final_gp: GaussianProcessRegressor | None = None

    for budget in range(budgets[0], budgets[-1] + 1):
        sample_x = reference.x[selected]
        # These are the only exact nonlinear RHS evaluations charged to the budget.
        sample_rhs = exact_nonlinear_rhs(params, sample_x)
        scale = max(float(np.ptp(reference.x)), 1.0)
        gp = make_gp(scale).fit(sample_x.reshape(-1, 1) / scale, sample_rhs)

        def predict(values: FloatArray, *, _gp=gp, _scale=scale) -> FloatArray:
            array = np.asarray(values, dtype=float)
            return np.asarray(_gp.predict(array.reshape(-1, 1) / _scale)).reshape(array.shape)

        if budget in budgets:
            learned_system = build_learned_system(params, predict)
            learned = solve_system(learned_system, params, reference.t, method="RK45")
            difference = learned.x - reference.x
            results.append(
                BudgetResult(
                    budget=budget,
                    residual_rmse=physical_residual_rmse(learned_system, reference),
                    displacement_rmse=float(np.sqrt(np.mean(difference**2))),
                    displacement_max_error=float(np.max(np.abs(difference))),
                    exact_rhs_evaluations=budget,
                )
            )
            final_gp = gp

        if budget == budgets[-1]:
            break

        _, uncertainty = gp.predict(
            reference.x.reshape(-1, 1) / scale,
            return_std=True,
        )
        uncertainty[np.asarray(sorted(selected_set), dtype=int)] = -np.inf
        next_index = int(np.argmax(uncertainty))
        selected.append(next_index)
        selected_set.add(next_index)

    if final_gp is None:  # pragma: no cover - guarded by budget validation
        raise RuntimeError("No Kriging model was trained")
    return results, final_gp, selected


# %% Reporting
def save_results(
    output_dir: Path,
    reference: Trajectory,
    learned: Trajectory,
    results: Sequence[BudgetResult],
    selected_x: FloatArray,
    params: DuffingParameters,
    predict: Predictor,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "budget_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(BudgetResult.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(result.__dict__ for result in results)

    grid = np.linspace(float(np.min(reference.x)), float(np.max(reference.x)), 500)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes[0, 0].plot(reference.t, reference.x, label="Exact LEGOde", color="#252525")
    axes[0, 0].plot(learned.t, learned.x, "--", label="AK LEGOde", color="#B45B5B")
    axes[0, 0].set(xlabel="Time", ylabel="Displacement", title="Direct and learned solves")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(grid, exact_nonlinear_rhs(params, grid), label="Exact RHS term")
    axes[0, 1].plot(grid, predict(grid), "--", label="Kriging RHS term")
    axes[0, 1].scatter(
        selected_x,
        exact_nonlinear_rhs(params, selected_x),
        s=20,
        color="#B45B5B",
        label="Evaluated samples",
    )
    axes[0, 1].set(xlabel="Displacement", ylabel="Nonlinear restoring", title="RHS learning")
    axes[0, 1].legend(frameon=False)

    budgets = [item.budget for item in results]
    axes[1, 0].semilogy(
        budgets,
        [item.residual_rmse for item in results],
        "o-",
        color="#72856F",
    )
    axes[1, 0].set(
        xlabel="Exact RHS evaluations",
        ylabel="Dynamic residual RMSE",
        title="Physical residual convergence",
    )

    axes[1, 1].semilogy(
        budgets,
        [item.displacement_rmse for item in results],
        "o-",
        color="#817A9F",
    )
    axes[1, 1].set(
        xlabel="Exact RHS evaluations",
        ylabel="Displacement RMSE",
        title="Trajectory convergence",
    )
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.savefig(output_dir / "duffing_legode_ak.png", dpi=180)
    plt.close(fig)

    final = results[-1]
    summary: dict[str, Any] = {
        "state_order": list(("displacement", "velocity")),
        "reference_rhs_evaluations": reference.rhs_evaluations,
        "active_exact_rhs_evaluations": final.exact_rhs_evaluations,
        "final_physical_residual_rmse": final.residual_rmse,
        "final_displacement_rmse": final.displacement_rmse,
        "final_displacement_max_error": final.displacement_max_error,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


# %% End-to-end experiment
def run_study(
    output_dir: Path,
    *,
    duration: float = 20.0,
    points: int = 801,
    max_budget: int = 24,
) -> dict[str, Any]:
    if duration <= 0:
        raise ValueError("duration must be positive")
    if points < 3:
        raise ValueError("points must be at least three")
    if max_budget < 4 or max_budget > points:
        raise ValueError("max_budget must be between four and points")

    params = DuffingParameters()
    t_eval = np.linspace(0.0, duration, points)
    exact_system = build_exact_system(params)
    reference = solve_system(exact_system, params, t_eval)

    proposed = [4, 6, 8, 12, 16, 24, max_budget]
    budgets = sorted({budget for budget in proposed if budget <= max_budget})
    results, final_gp, selected = fit_active_kriging(params, reference, budgets)
    scale = max(float(np.ptp(reference.x)), 1.0)

    def final_predict(values: FloatArray) -> FloatArray:
        array = np.asarray(values, dtype=float)
        return np.asarray(
            final_gp.predict(array.reshape(-1, 1) / scale)
        ).reshape(array.shape)

    learned_system = build_learned_system(params, final_predict)
    learned = solve_system(learned_system, params, t_eval, method="RK45")
    return save_results(
        output_dir,
        reference,
        learned,
        results,
        reference.x[selected],
        params,
        final_predict,
    )


# %% Notebook configuration and execution
try:
    STUDY_DIR = Path(__file__).resolve().parent
except NameError:  # Native .ipynb kernels do not define __file__.
    STUDY_DIR = Path.cwd().resolve()

OUTPUT_DIR = STUDY_DIR / "results" / "duffing"
DURATION = 20.0
POINTS = 801
MAX_BUDGET = 24


if __name__ == "__main__":
    summary = run_study(
        OUTPUT_DIR,
        duration=DURATION,
        points=POINTS,
        max_budget=MAX_BUDGET,
    )
    print(json.dumps(summary, indent=2))
