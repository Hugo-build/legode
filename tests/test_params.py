from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from legode import (
    ParameterError,
    dump_params,
    load_params,
    params_from_mapping,
    params_to_mapping,
    register_config_loader,
)


@dataclass(frozen=True)
class Nested:
    label: str
    enabled: bool = True


@dataclass(frozen=True)
class Parameters:
    mass: float
    samples: np.ndarray = field(metadata={"dtype": "float64"})
    nested: Nested = field(default_factory=lambda: Nested("default"))
    axes: tuple[int, ...] = ()
    aliases: dict[str, int] = field(default_factory=dict)
    note: str | None = None

    def __post_init__(self) -> None:
        if self.mass <= 0:
            raise ValueError("mass must be positive")


def test_recursive_parameter_conversion_and_array_dtype() -> None:
    params = params_from_mapping(
        {
            "mass": 2,
            "samples": [1, 2, 3],
            "nested": {"label": "inner"},
            "axes": [1, 3],
            "aliases": {"left": 4},
            "note": None,
        },
        Parameters,
    )
    assert params.mass == 2.0
    assert params.nested == Nested("inner")
    assert params.axes == (1, 3)
    assert params.aliases == {"left": 4}
    assert params.samples.dtype == np.float64
    np.testing.assert_array_equal(params.samples, [1.0, 2.0, 3.0])


def test_defaults_are_honored() -> None:
    params = params_from_mapping({"mass": 1, "samples": []}, Parameters)
    assert params.nested == Nested("default")
    assert params.note is None


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"samples": []}, "mass is required"),
        ({"mass": 1, "samples": [], "extra": 4}, "unknown fields"),
        ({"mass": "heavy", "samples": []}, "must be a number"),
        ({"mass": 0, "samples": []}, "mass must be positive"),
        ({"mass": 1, "samples": {"bad": "array"}}, "ndarray"),
    ],
)
def test_invalid_parameter_data(data, message: str) -> None:
    with pytest.raises(ParameterError, match=message):
        params_from_mapping(data, Parameters)


def test_json_load_and_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "parameters.json"
    source.write_text(json.dumps({"mass": 3, "samples": [2, 4]}), encoding="utf-8")
    params = load_params(source, Parameters)
    assert params.mass == 3.0
    assert params_to_mapping(params)["samples"] == [2.0, 4.0]

    destination = tmp_path / "saved.json"
    dump_params(params, destination)
    loaded = load_params(destination, Parameters)
    np.testing.assert_array_equal(loaded.samples, params.samples)


def test_unknown_suffix_and_custom_loader(tmp_path: Path) -> None:
    source = tmp_path / "manifest.legodetest"
    source.write_text("ignored", encoding="utf-8")
    with pytest.raises(ParameterError, match="No configuration loader"):
        load_params(source, Parameters)

    register_config_loader(
        ".legodetest", lambda path: {"mass": 5, "samples": [path.stat().st_size]}
    )
    params = load_params(source, Parameters)
    assert params.mass == 5.0
    np.testing.assert_array_equal(params.samples, [7.0])
    with pytest.raises(ValueError, match="already registered"):
        register_config_loader("legodetest", lambda path: {})


def test_explicit_loader_can_ignore_suffix(tmp_path: Path) -> None:
    source = tmp_path / "settings.in"
    source.write_text("value", encoding="utf-8")
    params = load_params(source, Parameters, loader=lambda _: {"mass": 1, "samples": []})
    assert params.mass == 1.0


def test_parameter_api_requires_dataclasses() -> None:
    with pytest.raises(TypeError, match="dataclass type"):
        params_from_mapping({}, dict)
    with pytest.raises(TypeError, match="dataclass instance"):
        params_to_mapping({})

