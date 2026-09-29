"""Persistent RTC-style RPC server for Cosmos Piper14 policy inference."""

from __future__ import annotations

import time
import traceback
from collections.abc import Mapping
from functools import partial
from multiprocessing import AuthenticationError
from multiprocessing.connection import Listener
from typing import Any

from piper_cosmos.deployment.cosmos_piper14_policy import (
    CosmosPiper14PolicyClient,
    CosmosPiper14PolicyConfig,
)


def serve_cosmos_piper14_policy(
    config: CosmosPiper14PolicyConfig | Mapping[str, Any],
    host: str = "127.0.0.1",
    port: int = 8766,
    authkey: str | bytes = "cosmos-piper14",
) -> None:
    key = authkey.encode("utf-8") if isinstance(authkey, str) else authkey
    policy = CosmosPiper14PolicyClient(config)
    listener = Listener((host, int(port)), authkey=key)
    print(f"[cosmos-piper14-policy-server] Listening on {host}:{port}", flush=True)
    try:
        while True:
            try:
                conn = listener.accept()
            except AuthenticationError as exc:
                # A stale or mistyped client authkey must not tear down the
                # persistent model process for every subsequent client.
                print(
                    "[cosmos-piper14-policy-server] Rejected client "
                    f"authentication: {exc}",
                    flush=True,
                )
                continue
            print(
                f"[cosmos-piper14-policy-server] Client connected from {listener.last_accepted}",
                flush=True,
            )
            should_shutdown = _serve_connection(policy, conn)
            conn.close()
            if should_shutdown:
                break
    finally:
        listener.close()
        print("[cosmos-piper14-policy-server] Stopped.", flush=True)


def _serve_connection(policy: CosmosPiper14PolicyClient, conn: Any) -> bool:
    while True:
        try:
            request = conn.recv()
        except (EOFError, ConnectionError):
            return False

        if not isinstance(request, Mapping):
            try:
                conn.send(
                    {
                        "ok": False,
                        "error": f"Expected request mapping, got {type(request)}",
                    }
                )
            except (EOFError, ConnectionError):
                return False
            continue

        op = request.get("op")
        try:
            if op == "update_observation":
                policy.update_observation(request["obs"])
                conn.send({"ok": True})
            elif op == "get_action":
                _send_timed_action(policy, conn, op, policy.get_action)
            elif op == "infer":
                observation = request["obs"]
                _send_timed_action(
                    policy,
                    conn,
                    op,
                    partial(policy.infer, observation),
                )
            elif op == "metadata":
                conn.send({"ok": True, "metadata": policy.metadata()})
            elif op == "timing":
                conn.send({"ok": True, "timing": policy.timing()})
            elif op == "reset":
                policy.reset()
                conn.send({"ok": True})
            elif op == "shutdown":
                conn.send({"ok": True})
                return True
            else:
                conn.send({"ok": False, "error": f"Unknown operation: {op!r}"})
        except (EOFError, ConnectionError):
            return False
        except Exception as exc:  # noqa: BLE001 - RPC boundary serializes failures
            try:
                conn.send(
                    {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
            except (EOFError, ConnectionError):
                return False


def _send_timed_action(
    policy: CosmosPiper14PolicyClient,
    conn: Any,
    op: str,
    inference: Any,
) -> None:
    started = time.perf_counter()
    action = inference()
    dispatch_ms = (time.perf_counter() - started) * 1000.0
    send_started = time.perf_counter()
    conn.send({"ok": True, "action": action.tolist()})
    send_ms = (time.perf_counter() - send_started) * 1000.0
    if policy.config.timing:
        print(
            f"[cosmos-piper14-rpc-timing] op={op} "
            f"dispatch={dispatch_ms:.3f}ms send={send_ms:.3f}ms",
            flush=True,
        )
