import importlib

from multi_purpose_mpc_ros.core import runtime_mode


def test_runtime_diagnostics_are_disabled_by_default(monkeypatch):
    monkeypatch.delenv("AICHALLENGE_ENABLE_RUNTIME_DIAGNOSTICS", raising=False)
    assert importlib.reload(runtime_mode).ENABLE_RUNTIME_DIAGNOSTICS is False


def test_runtime_diagnostics_can_be_reenabled(monkeypatch):
    monkeypatch.setenv("AICHALLENGE_ENABLE_RUNTIME_DIAGNOSTICS", "1")
    assert importlib.reload(runtime_mode).ENABLE_RUNTIME_DIAGNOSTICS is True

    monkeypatch.delenv("AICHALLENGE_ENABLE_RUNTIME_DIAGNOSTICS", raising=False)
    importlib.reload(runtime_mode)
