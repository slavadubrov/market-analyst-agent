"""Keep test artifacts away from the operator's run workspaces and effect ledger."""

import pytest


@pytest.fixture(autouse=True)
def isolate_runtime_storage(tmp_path, monkeypatch):
    from market_analyst.runtime import idempotency, workspace

    monkeypatch.setattr(workspace, "_DEFAULT_ROOT", tmp_path / "workspaces")
    monkeypatch.setattr(idempotency, "_DEFAULT_ROOT", tmp_path / "operations")
    monkeypatch.setattr(idempotency, "_store", None)
