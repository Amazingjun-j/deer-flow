from types import SimpleNamespace

import pytest

from deerflow.config.app_config import AppConfig
from deerflow.config.sandbox_config import SandboxConfig
from deerflow.config.subagents_config import SubagentsAppConfig
from deerflow.subagents.registry import _resolve_subagents_app_config


def test_resolve_subagents_app_config_accepts_subagents_config() -> None:
    subagents = SubagentsAppConfig()

    assert _resolve_subagents_app_config(subagents) is subagents


def test_resolve_subagents_app_config_extracts_from_app_config() -> None:
    app_config = AppConfig(
        sandbox=SandboxConfig(use="deerflow.sandbox.local:LocalSandboxProvider"),
        subagents=SubagentsAppConfig(timeout_seconds=1234),
    )

    resolved = _resolve_subagents_app_config(app_config)

    assert resolved is app_config.subagents
    assert resolved.timeout_seconds == 1234


def test_resolve_subagents_app_config_rejects_duck_typed_objects() -> None:
    duck_typed = SimpleNamespace(subagents=SubagentsAppConfig())

    with pytest.raises(TypeError, match="AppConfig, SubagentsAppConfig, or None"):
        _resolve_subagents_app_config(duck_typed)
