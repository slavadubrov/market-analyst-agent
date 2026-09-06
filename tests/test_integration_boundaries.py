"""Explicit opt-in: real disposable Docker services, no model or market API calls."""

import os
import subprocess
import sys
import time
import uuid

import pytest
import redis

from market_analyst.runtime.queue import ack, pull_one, push_run
from market_analyst.tools.code_exec import execute_python_analysis, run_isolated

pytestmark = pytest.mark.integration


@pytest.fixture
def redis_server():
    name = "market-audit-redis-" + uuid.uuid4().hex
    subprocess.run(["docker", "run", "--rm", "-d", "--name", name, "-p", "127.0.0.1::6379", "redis:7-alpine"], check=True, capture_output=True)
    try:
        address = subprocess.check_output(["docker", "port", name, "6379/tcp"], text=True).strip()
        url = "redis://" + address
        client = redis.Redis.from_url(url, decode_responses=True)
        for _ in range(100):
            try:
                if client.ping():
                    break
            except redis.ConnectionError:
                time.sleep(0.05)
        yield url, client
    finally:
        subprocess.run(["docker", "rm", "-f", name], check=False, capture_output=True)


def test_killed_consumer_is_reclaimed_once(redis_server):
    url, client = redis_server
    push_run("fixture", thread_id="killed-worker", client=client)
    code = """import os
from market_analyst.runtime.queue import pull_one
job = pull_one('dying-consumer', block_ms=50)
assert job is not None
os._exit(23)
"""
    result = subprocess.run([sys.executable, "-c", code], env={**os.environ, "REDIS_URL": url}, capture_output=True)
    assert result.returncode == 23
    job = pull_one("replacement", client=client, block_ms=50, reclaim_idle_ms=0)
    assert job is not None and job.thread_id == "killed-worker" and job.deliveries == 2
    ack(job.message_id, client=client)
    assert pull_one("third", client=client, block_ms=50, reclaim_idle_ms=0) is None


def test_dead_letter_is_durable_before_ack(redis_server):
    from market_analyst.runtime.queue import CONSUMER_GROUP, STREAM_KEY, dead_letter

    _, client = redis_server
    push_run("bad", client=client)
    job = pull_one("worker", client=client)
    dead_letter(job.message_id, {"thread_id": job.thread_id}, "fixture failure", client=client)
    assert client.xpending(STREAM_KEY, CONSUMER_GROUP)["pending"] == 0
    assert client.xrange(STREAM_KEY + ":dead")[0][1]["error"] == "fixture failure"


def test_container_isolation_and_resource_limits():
    assert execute_python_analysis.invoke({"code": "print(2 + 2)"}) == "4"
    # Exercise the OS boundary directly, without relying on the AST prefilter.
    assert "PermissionError" in run_isolated("open('/host-write', 'w').write('fixture')") or "Read-only" in run_isolated("open('/host-write', 'w')")
    assert "SECRET_FIXTURE" not in run_isolated("import os; print(os.environ)")
    result = run_isolated("import socket; socket.create_connection(('1.1.1.1', 443), timeout=1)")
    assert result.startswith("Error")
    assert run_isolated("while True: pass", timeout=1).startswith("Error")
    assert "output limit exceeded" in run_isolated("print('x' * 20000)")


def test_process_crash_resumes_postgres_checkpoint(tmp_path):
    name = "market-audit-postgres-" + uuid.uuid4().hex
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-d",
            "--name",
            name,
            "-p",
            "127.0.0.1::5432",
            "-e",
            "POSTGRES_PASSWORD=fixture",
            "-e",
            "POSTGRES_USER=fixture",
            "-e",
            "POSTGRES_DB=fixture",
            "postgres:16-alpine",
        ],
        check=True,
        capture_output=True,
    )
    try:
        for _ in range(100):
            ready = subprocess.run(["docker", "exec", name, "pg_isready", "-U", "fixture"], capture_output=True)
            if ready.returncode == 0:
                break
            time.sleep(0.1)
        address = subprocess.check_output(["docker", "port", name, "5432/tcp"], text=True).strip()
        env = {
            **os.environ,
            "POSTGRES_HOST": "127.0.0.1",
            "POSTGRES_PORT": address.rsplit(":", 1)[1],
            "POSTGRES_USER": "fixture",
            "POSTGRES_PASSWORD": "fixture",
            "POSTGRES_DB": "fixture",
            "HOT_MEMORY_PROVIDER": "postgres",
            "WORKSPACE_ROOT": str(tmp_path / "workspaces"),
            "CHECKPOINT_ENCRYPTION_KEY": "",
        }
        evidence = tmp_path / "effects.txt"
        command = [sys.executable, "tests/checkpoint_worker.py"]
        crashed = subprocess.run([*command, "crash", str(evidence)], env=env, capture_output=True, text=True, timeout=30)
        assert crashed.returncode == 23, crashed.stderr
        for _ in range(2):
            recovered = subprocess.run([*command, "recover", str(evidence)], env=env, capture_output=True, text=True, timeout=30)
            assert recovered.returncode == 0, recovered.stderr
        assert evidence.read_text().splitlines() == ["step-0", "step-1"]
    finally:
        subprocess.run(["docker", "rm", "-f", name], check=False, capture_output=True)
