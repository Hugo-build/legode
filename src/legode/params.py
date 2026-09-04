"""Typed dataclass parameter conversion and configuration loading."""

from __future__ import annotations

import json
import types
from collections import abc
from collections.abc import Callable, Mapping
from dataclasses import MISSING, Field, fields, is_dataclass
from pathlib import Path
from typing import Annotated, Any, TypeVar, Union, cast, get_args, get_origin, get_type_hints

import numpy as np

from .exceptions import ParameterError

ParamsT = TypeVar("ParamsT")
ConfigLoader = Callable[[Path], Mapping[str, Any]]


def _json_loader(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, Mapping):
        raise ParameterError("The configuration root must be an object/mapping")
    return value


_LOADERS: dict[str, ConfigLoader] = {".json": _json_loader}


def register_config_loader(
    suffix: str,
    loader: ConfigLoader,
    *,
    replace: bool = False,
) -> None:
    """Register a mapping loader for a filename suffix."""
    normalized = _normalize_suffix(suffix)
    if normalized in _LOADERS and not replace:
        raise ValueError(f"A loader is already registered for {normalized!r}")
    if not callable(loader):
        raise TypeError("loader must be callable")
    _LOADERS[normalized] = loader


def params_from_mapping(data: Mapping[str, Any], cls: type[ParamsT]) -> ParamsT:
    """Recursively construct a typed parameter dataclass from a mapping."""
    if not isinstance(data, Mapping):
        raise ParameterError("Parameter data must be a mapping")
    if not isinstance(cls, type) or not is_dataclass(cls):
        raise TypeError("cls must be a dataclass type")
    return _convert_dataclass(data, cls, cls.__name__)


def load_params(
    path: str | Path,
    cls: type[ParamsT],
    *,
    loader: ConfigLoader | None = None,
) -> ParamsT:
    """Load a file and convert its mapping to a parameter dataclass."""
    config_path = Path(path)
    selected = loader
    if selected is None:
        suffix = config_path.suffix.lower()
        selected = _LOADERS.get(suffix)
        if selected is None:
            known = ", ".join(sorted(_LOADERS))
            raise ParameterError(
                f"No configuration loader is registered for {suffix or '<no suffix>'!r}; "
                f"known suffixes: {known}"
            )
    try:
        raw = selected(config_path)
    except ParameterError:
        raise
    except Exception as exc:
        raise ParameterError(f"Could not load parameters from {config_path}: {exc}") from exc
    return params_from_mapping(raw, cls)


def params_to_mapping(params: Any) -> dict[str, Any]:
    """Convert a parameter dataclass to JSON-compatible Python values."""
    if not is_dataclass(params) or isinstance(params, type):
        raise TypeError("params must be a dataclass instance")
    result = _to_json_value(params, type(params).__name__)
    if not isinstance(result, dict):  # pragma: no cover - guaranteed by the check above
        raise TypeError("Expected dataclass serialization to produce a dictionary")
    return result


def dump_params(params: Any, path: str | Path, *, indent: int = 2) -> None:
    """Write a parameter dataclass as JSON."""
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(params_to_mapping(params), stream, indent=indent)
        stream.write("\n")


def _convert_dataclass(data: Mapping[str, Any], cls: type[ParamsT], path: str) -> ParamsT:
    field_map = {item.name: item for item in fields(cast(Any, cls)) if item.init}
    unknown = set(data).difference(field_map)
    if unknown:
        raise ParameterError(f"{path} has unknown fields: {sorted(unknown)!r}")
    hints = get_type_hints(cls, include_extras=True)
    kwargs: dict[str, Any] = {}
    for name, item in field_map.items():
        if name not in data:
            if item.default is MISSING and item.default_factory is MISSING:
                raise ParameterError(f"{path}.{name} is required")
            continue
        hint = hints.get(name, Any)
        kwargs[name] = _convert_value(data[name], hint, f"{path}.{name}", item)
    try:
        return cls(**kwargs)
    except ParameterError:
        raise
    except Exception as exc:
        raise ParameterError(f"Could not construct {path}: {exc}") from exc


def _convert_value(value: Any, hint: Any, path: str, dataclass_field: Field[Any] | None) -> Any:
    origin = get_origin(hint)
    args = get_args(hint)

    if origin is Annotated:
        return _convert_value(value, args[0], path, dataclass_field)
    if hint is Any:
        return value
    if hint is type(None):
        if value is not None:
            raise ParameterError(f"{path} must be null")
        return None
    if origin in (Union, types.UnionType):
        errors: list[str] = []
        for option in args:
            try:
                return _convert_value(value, option, path, dataclass_field)
            except ParameterError as exc:
                errors.append(str(exc))
        raise ParameterError(f"{path} does not match any allowed type: {'; '.join(errors)}")
    if isinstance(hint, type) and is_dataclass(hint):
        if not isinstance(value, Mapping):
            raise ParameterError(f"{path} must be an object/mapping")
        return _convert_dataclass(value, hint, path)
    if hint is np.ndarray or origin is np.ndarray:
        dtype = dataclass_field.metadata.get("dtype") if dataclass_field is not None else None
        try:
            array = np.asarray(value, dtype=dtype)
        except (TypeError, ValueError) as exc:
            raise ParameterError(f"{path} cannot be converted to an ndarray: {exc}") from exc
        if array.dtype == object:
            raise ParameterError(f"{path} produced an object array; use homogeneous JSON values")
        return array
    if origin is list:
        if not isinstance(value, list):
            raise ParameterError(f"{path} must be a list")
        item_hint = args[0] if args else Any
        return [
            _convert_value(item, item_hint, f"{path}[{index}]", None)
            for index, item in enumerate(value)
        ]
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ParameterError(f"{path} must be an array")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(
                _convert_value(item, args[0], f"{path}[{index}]", None)
                for index, item in enumerate(value)
            )
        if args and len(value) != len(args):
            raise ParameterError(f"{path} must contain exactly {len(args)} items")
        return tuple(
            _convert_value(item, item_hint, f"{path}[{index}]", None)
            for index, (item, item_hint) in enumerate(zip(value, args, strict=True))
        ) if args else tuple(value)
    if origin in (dict, Mapping, abc.Mapping):
        if not isinstance(value, Mapping):
            raise ParameterError(f"{path} must be an object/mapping")
        key_hint, value_hint = args if args else (Any, Any)
        return {
            _convert_value(key, key_hint, f"{path}.<key>", None): _convert_value(
                item, value_hint, f"{path}[{key!r}]", None
            )
            for key, item in value.items()
        }
    if hint is bool:
        if not isinstance(value, bool):
            raise ParameterError(f"{path} must be a boolean")
        return value
    if hint is int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ParameterError(f"{path} must be an integer")
        return value
    if hint is float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ParameterError(f"{path} must be a number")
        converted = float(value)
        if not np.isfinite(converted):
            raise ParameterError(f"{path} must be finite")
        return converted
    if hint is str:
        if not isinstance(value, str):
            raise ParameterError(f"{path} must be a string")
        return value
    if isinstance(hint, type) and isinstance(value, hint):
        return value
    raise ParameterError(f"{path} uses unsupported annotation {hint!r}")


def _to_json_value(value: Any, path: str) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _to_json_value(getattr(value, item.name), f"{path}.{item.name}")
            for item in fields(value)
        }
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _to_json_value(item, f"{path}[{key!r}]") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_value(item, f"{path}[]") for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ParameterError(f"{path} contains non-JSON-compatible value {type(value).__name__}")


def _normalize_suffix(suffix: str) -> str:
    if not isinstance(suffix, str) or not suffix.strip():
        raise ValueError("suffix must be a non-empty string")
    normalized = suffix.strip().lower()
    if not normalized.startswith("."):
        normalized = "." + normalized
    if normalized == ".":
        raise ValueError("suffix must contain characters after the dot")
    return normalized
