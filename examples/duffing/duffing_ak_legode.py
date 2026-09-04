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

    python examples/duffing/duffing_ak_legode.py

The ``# %%`` windows can also be run one at a time in an interactive editor.
Each calculation, console summary, and figure is intentionally kept in its own
window so the study can be followed from top to bottom.
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

from legode import Force, ODEEnsemble, ODEEquation, StateLayout

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
# SciPy stores the state in compact numeric arrays.  These two helpers keep the
# numerical indexing out of the physical terms below: callers ask for a named
# state or its named time derivative, and StateLayout supplies the array index.
def state_value(y: FloatArray, layout: StateLayout, name: str) -> float:
    return float(y[layout[name]])


def derivative_value(y_dot: FloatArray, layout: StateLayout, name: str) -> float:
    return float(y_dot[layout[name]])


class PositionDerivative(Force[DuffingParameters]):
    required_derivatives = ("displacement",)

    def __call__(self, t, y, y_dot, layout):
        return derivative_value(y_dot, layout, "displacement")


class Velocity(Force[DuffingParameters]):
    required_states = ("velocity",)

    def __call__(self, t, y, y_dot, layout):
        return state_value(y, layout, "velocity")


class Inertia(Force[DuffingParameters]):
    required_derivatives = ("velocity",)

    def __call__(self, t, y, y_dot, layout):
        acceleration = derivative_value(y_dot, layout, "velocity")
        return self.params.mass * acceleration


class Damping(Force[DuffingParameters]):
    required_states = ("velocity",)

    def __call__(self, t, y, y_dot, layout):
        velocity = state_value(y, layout, "velocity")
        return self.params.damping * velocity


class LinearRestoring(Force[DuffingParameters]):
    required_states = ("displacement",)

    def __call__(self, t, y, y_dot, layout):
        displacement = state_value(y, layout, "displacement")
        return self.params.stiffness * displacement


class Excitation(Force[DuffingParameters]):
    def __call__(self, t, y, y_dot, layout):
        return excitation(t)


class ExactNonlinearRestoring(Force[DuffingParameters]):
    required_states = ("displacement",)

    def __call__(self, t, y, y_dot, layout):
        displacement = state_value(y, layout, "displacement")
        return self.params.cubic * displacement**3


class LearnedNonlinearRestoring(Force[LearnedRHSParameters]):
    required_states = ("displacement",)

    def __call__(self, t, y, y_dot, layout):
        displacement = state_value(y, layout, "displacement")
        sample = np.asarray([displacement], dtype=float)
        return float(self.params.predict(sample)[0])


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
    initial = system.layout.vector({"displacement": params.x0, "velocity": params.v0})
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


# %% Reporting helpers
def print_budget_history(results: Sequence[BudgetResult]) -> None:
    """Print the convergence history in a compact, dependency-free table."""
    print("\nActive-Kriging convergence")
    print("budget    residual RMSE    displacement RMSE    maximum error")
    for result in results:
        print(
            f"{result.budget:>6d}    {result.residual_rmse:>13.4e}    "
            f"{result.displacement_rmse:>17.4e}    "
            f"{result.displacement_max_error:>13.4e}"
        )


def make_summary(
    system: ODEEnsemble,
    reference: Trajectory,
    results: Sequence[BudgetResult],
) -> dict[str, Any]:
    final = results[-1]
    return {
        "state_order": list(system.layout.names),
        "reference_rhs_evaluations": reference.rhs_evaluations,
        "active_exact_rhs_evaluations": final.exact_rhs_evaluations,
        "final_physical_residual_rmse": final.residual_rmse,
        "final_displacement_rmse": final.displacement_rmse,
        "final_displacement_max_error": final.displacement_max_error,
    }


def write_numeric_results(
    output_dir: Path,
    results: Sequence[BudgetResult],
    selected_x: FloatArray,
    summary: dict[str, Any],
) -> None:
    """Write machine-readable results; plotting remains in its own cells."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "budget_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(BudgetResult.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(result.__dict__ for result in results)
    np.savetxt(
        output_dir / "active_samples.csv",
        selected_x,
        delimiter=",",
        header="displacement",
        comments="",
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def plot_solution_comparison(reference: Trajectory, learned: Trajectory):
    """Show the trajectories and make their small difference visible."""
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.0), sharex=True, constrained_layout=True)
    axes[0].plot(reference.t, reference.x, color="#252525", label="Exact LEGOde")
    axes[0].plot(
        learned.t,
        learned.x,
        "--",
        color="#B45B5B",
        label="AK LEGOde",
    )
    axes[0].set(ylabel="Displacement", title="Direct and learned LEGOde solutions")
    axes[0].legend(frameon=False)
    axes[1].plot(
        reference.t,
        learned.x - reference.x,
        color="#B45B5B",
        label="AK minus exact",
    )
    axes[1].set(xlabel="Time", ylabel="Displacement error")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False)
    return fig


def plot_rhs_learning(
    reference: Trajectory,
    selected_x: FloatArray,
    params: DuffingParameters,
    predict: Predictor,
):
    """Show which exact RHS samples define the learned nonlinear term."""
    grid = np.linspace(float(np.min(reference.x)), float(np.max(reference.x)), 500)
    fig, axis = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    axis.plot(grid, exact_nonlinear_rhs(params, grid), color="#252525", label="Exact RHS")
    axis.plot(grid, predict(grid), "--", color="#B45B5B", label="Kriging RHS")
    axis.scatter(
        selected_x,
        exact_nonlinear_rhs(params, selected_x),
        s=24,
        color="#72856F",
        label="Exact samples",
        zorder=3,
    )
    axis.set(
        xlabel="Displacement",
        ylabel="Nonlinear restoring force",
        title="Active learning of the nonlinear RHS",
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    return fig


def plot_budget_convergence(results: Sequence[BudgetResult]):
    """Show physical-equation and trajectory errors against exact sample cost."""
    budgets = [result.budget for result in results]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    axes[0].semilogy(
        budgets,
        [result.residual_rmse for result in results],
        "o-",
        color="#72856F",
    )
    axes[0].set(
        xlabel="Exact RHS evaluations",
        ylabel="Dynamic residual RMSE",
        title="Physical residual convergence",
    )
    axes[1].semilogy(
        budgets,
        [result.displacement_rmse for result in results],
        "o-",
        color="#817A9F",
    )
    axes[1].set(
        xlabel="Exact RHS evaluations",
        ylabel="Displacement RMSE",
        title="Trajectory convergence",
    )
    for axis in axes:
        axis.set_xticks(budgets)
        axis.grid(alpha=0.2)
    return fig


# %% 1. Study configuration
try:
    STUDY_DIR = Path(__file__).resolve().parent
except NameError:  # Native .ipynb kernels do not define __file__.
    STUDY_DIR = Path.cwd().resolve()

OUTPUT_DIR = STUDY_DIR / "results"
DURATION = 20.0
POINTS = 801
MAX_BUDGET = 24

if DURATION <= 0:
    raise ValueError("DURATION must be positive")
if POINTS < 3:
    raise ValueError("POINTS must be at least three")
if MAX_BUDGET < 4 or MAX_BUDGET > POINTS:
    raise ValueError("MAX_BUDGET must be between four and POINTS")

params = DuffingParameters()
t_eval = np.linspace(0.0, DURATION, POINTS)
exact_system = build_exact_system(params)

print("Duffing active-Kriging study")
print(f"  duration: {DURATION:g}")
print(f"  output points: {POINTS}")
print(f"  state vector order: {exact_system.layout.names}")
print("  state vector meaning: y = [displacement, velocity]")


# %% 2. Direct reference solve
reference = solve_system(exact_system, params, t_eval)

print("\nDirect LEGOde reference")
print(f"  RHS evaluations: {reference.rhs_evaluations}")
print(f"  displacement range: [{reference.x.min():.4f}, {reference.x.max():.4f}]")
print(f"  maximum |displacement|: {np.max(np.abs(reference.x)):.4f}")


# %% 3. Active Kriging budget sweep
proposed_budgets = [4, 6, 8, 12, 16, 24, MAX_BUDGET]
budgets = sorted({budget for budget in proposed_budgets if budget <= MAX_BUDGET})
budget_results, final_gp, selected_indices = fit_active_kriging(params, reference, budgets)
selected_x = reference.x[selected_indices]
print_budget_history(budget_results)


# %% 4. Solve with the learned nonlinear RHS
displacement_scale = max(float(np.ptp(reference.x)), 1.0)


def final_predict(values: FloatArray) -> FloatArray:
    values = np.asarray(values, dtype=float)
    scaled_values = values.reshape(-1, 1) / displacement_scale
    return np.asarray(final_gp.predict(scaled_values)).reshape(values.shape)


learned_system = build_learned_system(params, final_predict)
learned = solve_system(learned_system, params, t_eval, method="RK45")
summary = make_summary(learned_system, reference, budget_results)

print("\nFinal learned LEGOde solution")
print(f"  exact nonlinear RHS samples: {summary['active_exact_rhs_evaluations']}")
print(f"  displacement RMSE: {summary['final_displacement_rmse']:.4e}")
print(f"  maximum displacement error: {summary['final_displacement_max_error']:.4e}")
print(f"  physical residual RMSE: {summary['final_physical_residual_rmse']:.4e}")


# %% 5. Plot direct and learned trajectories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
solution_figure = plot_solution_comparison(reference, learned)
solution_path = OUTPUT_DIR / "solution_comparison.png"
solution_figure.savefig(solution_path, dpi=180)
print(f"\nSaved solution comparison: {solution_path}")
plt.show()
plt.close(solution_figure)


# %% 6. Plot exact and learned nonlinear RHS
rhs_figure = plot_rhs_learning(reference, selected_x, params, final_predict)
rhs_path = OUTPUT_DIR / "gp_force_learning.png"
rhs_figure.savefig(rhs_path, dpi=180)
print(f"Saved nonlinear RHS learning plot: {rhs_path}")
plt.show()
plt.close(rhs_figure)


# %% 7. Plot budget convergence
budget_figure = plot_budget_convergence(budget_results)
budget_path = OUTPUT_DIR / "budget_convergence.png"
budget_figure.savefig(budget_path, dpi=180)
print(f"Saved budget convergence plot: {budget_path}")
plt.show()
plt.close(budget_figure)


# %% 8. Export numeric results
write_numeric_results(OUTPUT_DIR, budget_results, selected_x, summary)
print(f"\nNumeric results written to: {OUTPUT_DIR}")
print(json.dumps(summary, indent=2))
