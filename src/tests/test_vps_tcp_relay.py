"""Tests for the VPS TCP relay helper script (loopback-by-default bind)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "install" / "vps_tcp_relay.py"
)


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("vps_tcp_relay", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


relay = _load()


def test_default_bind_is_loopback() -> None:
    assert relay.parse_args(["27000", "27100"], {}) == (27000, 27100, "127.0.0.1")


def test_env_var_overrides_bind_host() -> None:
    env = {relay.BIND_ENV_VAR: "0.0.0.0"}
    assert relay.parse_args(["1", "2"], env)[2] == "0.0.0.0"


def test_blank_env_var_falls_back_to_loopback() -> None:
    assert relay.parse_args(["1", "2"], {relay.BIND_ENV_VAR: "  "})[2] == "127.0.0.1"


def test_flag_overrides_env_in_both_forms() -> None:
    env = {relay.BIND_ENV_VAR: "0.0.0.0"}
    assert relay.parse_args(["1", "2", "--bind", "10.0.0.5"], env)[2] == "10.0.0.5"
    assert relay.parse_args(["--bind=::1", "1", "2"], env)[2] == "::1"


@pytest.mark.parametrize(
    "argv",
    [["1"], ["1", "2", "3"], ["1", "2", "--bind"], ["1", "2", "--bind="]],
)
def test_invalid_arguments_exit(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        relay.parse_args(argv, {})


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", True),
        ("127.5.5.5", True),
        ("::1", True),
        ("localhost", True),
        ("0.0.0.0", False),
        ("::", False),
        ("203.0.113.10", False),
        ("example.com", False),
    ],
)
def test_is_loopback_host(host: str, expected: bool) -> None:
    assert relay.is_loopback_host(host) is expected
