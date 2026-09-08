from datetime import UTC, datetime, timedelta

import pytest

from deerflow.config.run_ownership_config import RunOwnershipConfig
from deerflow.runtime import RunManager, RunStatus
from deerflow.runtime.runs.store.memory import MemoryRunStore


def _lease_config() -> RunOwnershipConfig:
    return RunOwnershipConfig(lease_seconds=30, grace_seconds=10, heartbeat_enabled=True)


@pytest.mark.anyio
async def test_peer_idempotency_reuse_does_not_shadow_durable_completion():
    store = MemoryRunStore()
    owner = RunManager(store=store, worker_id="worker-a", run_ownership_config=_lease_config())
    peer = RunManager(store=store, worker_id="worker-b", run_ownership_config=_lease_config())
    original = await owner.create_or_reject("thread-1", idempotency_key="request-1")
    await owner.set_status(original.run_id, RunStatus.running)
    reused = await peer.create_or_reject("thread-1", idempotency_key="request-1")
    assert reused.store_only is True and reused.task is None
    await owner.set_status(original.run_id, RunStatus.success)
    observed = await peer.get(original.run_id)
    assert observed is not None and observed.status == RunStatus.success
    next_run = await peer.create_or_reject("thread-1", idempotency_key="request-2")
    assert next_run.run_id != original.run_id


@pytest.mark.anyio
async def test_peer_idempotency_reuse_does_not_mask_orphan_recovery():
    store = MemoryRunStore()
    owner = RunManager(store=store, worker_id="worker-a", run_ownership_config=_lease_config())
    peer = RunManager(store=store, worker_id="worker-b", run_ownership_config=_lease_config())
    original = await owner.create_or_reject("thread-1", idempotency_key="request-1")
    await owner.set_status(original.run_id, RunStatus.running)
    reused = await peer.create_or_reject("thread-1", idempotency_key="request-1")
    assert reused.store_only is True and reused.owner_worker_id == "worker-a" and reused.task is None
    expired = (datetime.now(UTC) - timedelta(seconds=60)).isoformat()
    assert await store.update_lease(original.run_id, owner_worker_id="worker-a", lease_expires_at=expired) is True
    recovered = await peer.reconcile_orphaned_inflight_runs(error="owner lease expired")
    assert [record.run_id for record in recovered] == [original.run_id]
    stored = await store.get(original.run_id)
    assert stored is not None and stored["status"] == "error"
