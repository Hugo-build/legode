"""Exceptions raised by LEGOde."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


class LEGOdeError(Exception):
    """Base class for package-specific errors."""


class AssemblyError(LEGOdeError, ValueError):
    """Raised when equations cannot form a valid ODE ensemble."""


class EvaluationError(LEGOdeError, ValueError):
    """Raised when an equation term produces an invalid numerical value."""


class ParameterError(LEGOdeError, ValueError):
    """Raised when configuration cannot be converted to a parameter dataclass."""


class DerivativeSolveError(LEGOdeError, RuntimeError):
    """Raised when an implicit residual cannot be solved for the derivative."""

    def __init__(
        self,
        *,
        time: float,
        message: str,
        residual_norm: float,
        last_guess: NDArray[np.float64],
    ) -> None:
        self.time = float(time)
        self.solver_message = message
        self.residual_norm = float(residual_norm)
        self.last_guess = np.asarray(last_guess, dtype=float).copy()
        super().__init__(
            f"Could not solve for y_dot at t={self.time:g}: {message} "
            f"(residual norm={self.residual_norm:.6g})"
        )

    def as_dict(self) -> dict[str, Any]:
        """Return diagnostic information suitable for logging."""
        return {
            "time": self.time,
            "message": self.solver_message,
            "residual_norm": self.residual_norm,
            "last_guess": self.last_guess.copy(),
        }

