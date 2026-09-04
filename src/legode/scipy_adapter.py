"""Adapter from implicit LEGOde residuals to SciPy's explicit RHS protocol."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import root  # type: ignore[import-untyped]

from .core import ODEEnsemble
from .exceptions import DerivativeSolveError, EvaluationError

FloatArray = NDArray[np.float64]
InitialGuess = Callable[[float, FloatArray], ArrayLike]


class ImplicitRHS:
    """Stateful ``fun(t, y) -> y_dot`` adapter for ``scipy.integrate.solve_ivp``.

    A successful derivative is reused as the next nonlinear solve's initial
    guess. Instances are therefore intended for one integration at a time and
    must not be shared concurrently.
    """

    def __init__(
        self,
        ensemble: ODEEnsemble,
        *,
        method: str = "hybr",
        tol: float | None = None,
        residual_tol: float = 1e-8,
        options: Mapping[str, Any] | None = None,
        initial_guess: InitialGuess | None = None,
    ) -> None:
        if not isinstance(ensemble, ODEEnsemble):
            raise TypeError("ensemble must be an ODEEnsemble")
        if tol is not None and (not np.isfinite(tol) or tol <= 0):
            raise ValueError("tol must be positive and finite")
        if not np.isfinite(residual_tol) or residual_tol <= 0:
            raise ValueError("residual_tol must be positive and finite")
        self.ensemble = ensemble
        self.method = method
        self.tol = tol
        self.residual_tol = float(residual_tol)
        self.options = dict(options or {})
        self.initial_guess = initial_guess
        self._last_derivative: FloatArray | None = None

    @property
    def last_derivative(self) -> FloatArray | None:
        """Return a defensive copy of the last successful derivative."""
        if self._last_derivative is None:
            return None
        return self._last_derivative.copy()

    def reset(self) -> None:
        """Discard continuation state before starting another integration."""
        self._last_derivative = None

    def __call__(self, t: float, y: ArrayLike) -> FloatArray:
        state = np.asarray(y, dtype=float)
        size = len(self.ensemble.layout)
        if state.ndim != 1 or state.shape != (size,):
            raise EvaluationError(f"y must have shape ({size},), got {state.shape}")
        if not np.all(np.isfinite(state)):
            raise EvaluationError("y contains non-finite values")

        guess = self._make_guess(float(t), state)
        result = root(
            lambda candidate: self.ensemble.residual(t, state, candidate),
            guess,
            method=self.method,
            tol=self.tol,
            options=self.options,
        )
        derivative = np.asarray(result.x, dtype=float)
        try:
            residual = self.ensemble.residual(t, state, derivative)
            residual_norm = float(np.linalg.norm(residual))
        except (EvaluationError, ValueError):
            residual_norm = float("inf")

        if (
            derivative.shape != (size,)
            or not np.all(np.isfinite(derivative))
            or not np.isfinite(residual_norm)
            or residual_norm > self.residual_tol
        ):
            raise DerivativeSolveError(
                time=float(t),
                message=str(result.message),
                residual_norm=residual_norm,
                last_guess=guess,
            )
        self._last_derivative = derivative.copy()
        return derivative

    def _make_guess(self, t: float, y: FloatArray) -> FloatArray:
        size = len(self.ensemble.layout)
        if self._last_derivative is not None:
            return self._last_derivative.copy()
        raw_guess: ArrayLike
        if self.initial_guess is None:
            raw_guess = np.zeros(size, dtype=float)
        else:
            raw_guess = self.initial_guess(t, y.copy())
        guess = np.asarray(raw_guess, dtype=float)
        if guess.ndim != 1 or guess.shape != (size,):
            raise EvaluationError(
                f"Initial derivative guess must have shape ({size},), got {guess.shape}"
            )
        if not np.all(np.isfinite(guess)):
            raise EvaluationError("Initial derivative guess contains non-finite values")
        return guess.copy()
