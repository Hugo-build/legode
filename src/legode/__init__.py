"""LEGOde: composable tools for implicit ordinary differential equations."""

from .core import EquationSide, Force, ODEEnsemble, ODEEquation, StateLayout, Term, as_side
from .exceptions import (
    AssemblyError,
    DerivativeSolveError,
    EvaluationError,
    LEGOdeError,
    ParameterError,
)
from .params import (
    ConfigLoader,
    dump_params,
    load_params,
    params_from_mapping,
    params_to_mapping,
    register_config_loader,
)
from .scipy_adapter import ImplicitRHS, InitialGuess

__all__ = [
    "AssemblyError",
    "ConfigLoader",
    "DerivativeSolveError",
    "EquationSide",
    "EvaluationError",
    "Force",
    "ImplicitRHS",
    "InitialGuess",
    "LEGOdeError",
    "ODEEnsemble",
    "ODEEquation",
    "ParameterError",
    "StateLayout",
    "Term",
    "as_side",
    "dump_params",
    "load_params",
    "params_from_mapping",
    "params_to_mapping",
    "register_config_loader",
]

