"""Equation composition and automatic ODE ensemble assembly."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, is_dataclass
from types import MappingProxyType
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .exceptions import AssemblyError, EvaluationError

ParamsT = TypeVar("ParamsT")
FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class StateLayout(Mapping[str, int]):
    """Immutable mapping between state names and solver-vector indices."""

    names: tuple[str, ...]
    _indices: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.names:
            raise AssemblyError("A state layout cannot be empty")
        if any(not isinstance(name, str) or not name for name in self.names):
            raise AssemblyError("State names must be non-empty strings")
        if len(set(self.names)) != len(self.names):
            raise AssemblyError("State names must be unique")
        object.__setattr__(
            self,
            "_indices",
            MappingProxyType({name: index for index, name in enumerate(self.names)}),
        )

    def __getitem__(self, name: str) -> int:
        return self._indices[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)

    def __len__(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        """Return the integer index for a named state."""
        try:
            return self._indices[name]
        except KeyError as exc:
            raise KeyError(f"Unknown state {name!r}") from exc

    def vector(self, values: Mapping[str, float]) -> FloatArray:
        """Create a solver vector in layout order from named values."""
        missing = set(self.names).difference(values)
        unknown = set(values).difference(self.names)
        if missing or unknown:
            details = []
            if missing:
                details.append(f"missing {sorted(missing)!r}")
            if unknown:
                details.append(f"unknown {sorted(unknown)!r}")
            raise ValueError("Invalid named state values: " + ", ".join(details))
        return np.asarray([values[name] for name in self.names], dtype=float)

    def named(self, vector: ArrayLike) -> dict[str, float]:
        """Convert a solver vector into a name-to-value dictionary."""
        array = _validate_vector(vector, len(self), "state vector")
        return dict(zip(self.names, array, strict=True))


@runtime_checkable
class Term(Protocol):
    """Structural protocol implemented by numerical equation terms."""

    required_states: tuple[str, ...]
    required_derivatives: tuple[str, ...]

    def __call__(
        self,
        t: float,
        y: FloatArray,
        y_dot: FloatArray,
        layout: StateLayout,
    ) -> ArrayLike: ...


class _ComposableTerm:
    """Arithmetic helpers shared by concrete force terms."""

    def __add__(self, other: Term | EquationSide) -> EquationSide:
        return EquationSide((self,)) + other  # type: ignore[arg-type]

    def __sub__(self, other: Term | EquationSide) -> EquationSide:
        return EquationSide((self,)) - other  # type: ignore[arg-type]

    def __neg__(self) -> EquationSide:
        return EquationSide(((self, -1.0),))  # type: ignore[arg-type]


class Force(_ComposableTerm, ABC, Generic[ParamsT]):
    """Base class for a parameterized callable equation term.

    Subclasses declare their dependencies using ``required_states`` and
    ``required_derivatives`` and implement ``__call__``. The state arrays follow
    the supplied :class:`StateLayout`.
    """

    required_states: tuple[str, ...] = ()
    required_derivatives: tuple[str, ...] = ()

    def __init__(self, params: ParamsT) -> None:
        if not is_dataclass(params) or isinstance(params, type):
            raise TypeError("Force parameters must be a dataclass instance")
        self.params = params
        _validate_dependencies(self.required_states, "required_states")
        _validate_dependencies(self.required_derivatives, "required_derivatives")

    @abstractmethod
    def __call__(
        self,
        t: float,
        y: FloatArray,
        y_dot: FloatArray,
        layout: StateLayout,
    ) -> ArrayLike:
        """Evaluate this term."""


@dataclass(frozen=True, slots=True)
class _SignedTerm:
    term: Term
    coefficient: float = 1.0


class EquationSide:
    """Ordered, signed sum of numerical terms on one side of an equation."""

    __slots__ = ("_terms",)

    def __init__(
        self,
        terms: Iterable[Term | tuple[Term, float]] = (),
    ) -> None:
        normalized: list[_SignedTerm] = []
        for item in terms:
            if isinstance(item, tuple):
                term, coefficient = item
            else:
                term, coefficient = item, 1.0
            if not isinstance(term, Term):
                raise TypeError("Equation sides can contain only callable terms")
            coefficient = float(coefficient)
            if not np.isfinite(coefficient):
                raise ValueError("Term coefficients must be finite")
            normalized.append(_SignedTerm(term, coefficient))
        self._terms = tuple(normalized)

    @property
    def terms(self) -> tuple[tuple[Term, float], ...]:
        return tuple((signed.term, signed.coefficient) for signed in self._terms)

    @property
    def required_states(self) -> tuple[str, ...]:
        return _ordered_union(term.required_states for term, _ in self.terms)

    @property
    def required_derivatives(self) -> tuple[str, ...]:
        return _ordered_union(term.required_derivatives for term, _ in self.terms)

    def __bool__(self) -> bool:
        return bool(self._terms)

    def __add__(self, other: Term | EquationSide) -> EquationSide:
        other_side = as_side(other)
        return EquationSide((*self.terms, *other_side.terms))

    def __radd__(self, other: Term | EquationSide | int) -> EquationSide:
        if other == 0:
            return self
        return as_side(other) + self  # type: ignore[arg-type]

    def __sub__(self, other: Term | EquationSide) -> EquationSide:
        other_side = as_side(other)
        negated = tuple((term, -coefficient) for term, coefficient in other_side.terms)
        return EquationSide((*self.terms, *negated))

    def __neg__(self) -> EquationSide:
        return EquationSide((term, -coefficient) for term, coefficient in self.terms)

    def evaluate(
        self,
        t: float,
        y: FloatArray,
        y_dot: FloatArray,
        layout: StateLayout,
        *,
        expected_size: int,
        label: str,
    ) -> FloatArray:
        total = np.zeros(expected_size, dtype=float)
        for position, signed in enumerate(self._terms):
            raw = signed.term(t, y, y_dot, layout)
            value = _normalize_term_output(raw, expected_size, f"{label} term {position}")
            total += signed.coefficient * value
        if not np.all(np.isfinite(total)):
            raise EvaluationError(f"{label} produced non-finite values")
        return total


def as_side(value: Term | EquationSide | None) -> EquationSide:
    """Normalize a term, side, or ``None`` into an :class:`EquationSide`."""
    if value is None:
        return EquationSide()
    if isinstance(value, EquationSide):
        return value
    if isinstance(value, Term):
        return EquationSide((value,))
    raise TypeError("Expected a callable term, EquationSide, or None")


class ODEEquation:
    """An implicit equation associating residual rows with named derivatives."""

    __slots__ = ("lhs", "name", "rhs", "solves_for")

    def __init__(
        self,
        solves_for: Iterable[str],
        lhs: EquationSide | Term | None,
        rhs: EquationSide | Term | None,
        name: str | None = None,
    ) -> None:
        self.solves_for = tuple(solves_for)
        self.lhs = as_side(lhs)
        self.rhs = as_side(rhs)
        self.name = name
        if not self.solves_for:
            raise AssemblyError("An equation must solve for at least one state")
        if len(set(self.solves_for)) != len(self.solves_for):
            raise AssemblyError("An equation cannot solve for the same state twice")
        if any(not isinstance(value, str) or not value for value in self.solves_for):
            raise AssemblyError("solves_for must contain only non-empty strings")
        if not self.lhs and not self.rhs:
            raise AssemblyError("An equation cannot have two empty sides")

    @property
    def label(self) -> str:
        return self.name or "/".join(self.solves_for)

    @property
    def required_states(self) -> tuple[str, ...]:
        return _ordered_union((self.lhs.required_states, self.rhs.required_states))

    @property
    def required_derivatives(self) -> tuple[str, ...]:
        return _ordered_union(
            (self.lhs.required_derivatives, self.rhs.required_derivatives)
        )

    def residual(
        self,
        t: float,
        y: FloatArray,
        y_dot: FloatArray,
        layout: StateLayout,
    ) -> FloatArray:
        size = len(self.solves_for)
        lhs = self.lhs.evaluate(
            t, y, y_dot, layout, expected_size=size, label=f"{self.label} LHS"
        )
        rhs = self.rhs.evaluate(
            t, y, y_dot, layout, expected_size=size, label=f"{self.label} RHS"
        )
        return lhs - rhs


class ODEEnsemble:
    """Validated square system assembled from named component equations."""

    __slots__ = ("equations", "layout", "_row_slices")

    def __init__(self, equations: Sequence[ODEEquation]) -> None:
        if not equations:
            raise AssemblyError("An ODE ensemble requires at least one equation")
        self.equations = tuple(equations)
        state_names: list[str] = []
        owners: dict[str, str] = {}
        row_slices: list[slice] = []
        row = 0
        for equation in self.equations:
            if not isinstance(equation, ODEEquation):
                raise TypeError("ODEEnsemble accepts only ODEEquation instances")
            for state in equation.solves_for:
                if state in owners:
                    raise AssemblyError(
                        f"State {state!r} is solved by both {owners[state]!r} "
                        f"and {equation.label!r}"
                    )
                owners[state] = equation.label
                state_names.append(state)
            next_row = row + len(equation.solves_for)
            row_slices.append(slice(row, next_row))
            row = next_row

        self.layout = StateLayout(tuple(state_names))
        self._row_slices = tuple(row_slices)
        available = set(self.layout.names)
        for equation in self.equations:
            missing_states = set(equation.required_states).difference(available)
            missing_derivatives = set(equation.required_derivatives).difference(available)
            if missing_states or missing_derivatives:
                missing = sorted(missing_states | missing_derivatives)
                raise AssemblyError(
                    f"Equation {equation.label!r} requires unsolved states {missing!r}"
                )

        if row != len(self.layout):
            raise AssemblyError(
                f"ODE ensemble must be square; got {row} rows for {len(self.layout)} states"
            )

    def residual(self, t: float, y: ArrayLike, y_dot: ArrayLike) -> FloatArray:
        """Evaluate the assembled ``LHS - RHS`` residual vector."""
        state = _validate_vector(y, len(self.layout), "y")
        derivative = _validate_vector(y_dot, len(self.layout), "y_dot")
        result = np.empty(len(self.layout), dtype=float)
        for equation, rows in zip(self.equations, self._row_slices, strict=True):
            result[rows] = equation.residual(t, state, derivative, self.layout)
        if not np.all(np.isfinite(result)):
            raise EvaluationError("The assembled residual contains non-finite values")
        return result

    def as_scipy_rhs(self, **kwargs: Any) -> Any:
        """Create a fresh stateful ``solve_ivp``-compatible derivative adapter."""
        from .scipy_adapter import ImplicitRHS

        return ImplicitRHS(self, **kwargs)


def _validate_dependencies(values: tuple[str, ...], label: str) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{label} must be a tuple of state names")
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{label} must contain only non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} cannot contain duplicates")


def _ordered_union(groups: Iterable[Iterable[str]]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(name for group in groups for name in group))


def _normalize_term_output(value: ArrayLike, size: int, label: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1 or array.shape != (size,):
        raise EvaluationError(
            f"{label} returned shape {array.shape}; expected scalar/shape (1,) "
            f"for one row or shape ({size},)"
        )
    if not np.all(np.isfinite(array)):
        raise EvaluationError(f"{label} produced non-finite values")
    return array


def _validate_vector(value: ArrayLike, size: int, label: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.shape != (size,):
        raise EvaluationError(f"{label} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise EvaluationError(f"{label} contains non-finite values")
    return array
