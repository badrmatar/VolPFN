"""Pod-local, stop-only watchdog for the return-only benchmark.

This module never reads an API key or creates/terminates resources. It uses the
pod's already configured runpodctl, after a read-only authentication check. Start
it on the pod *before dependency setup*, with the provider's creation timestamp.
The CLI's successful stop request is not proof of EXITED: the operator must still
confirm EXITED through RunPod. A watchdog is independent of the Mac, but cannot
guarantee a dollar cap during a provider/API failure or if this process is killed.

Public-source audit (2026-09-25): https://graphql-spec.runpod.io/ exposes a native
creation-time PodFindAndDeployOnDemandInput.stopAfter DateTime. The current MCP
create surface and runpodctl main do not expose that field. Use native stopAfter
when an authenticated creation path is available, preferably as a second guard.
runpodctl's official README says pods include a pod-scoped CLI credential; this
must be checked on the actual image, not assumed from documentation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time


UTC = timezone.utc
MAX_HOURS = 4.0
MAX_BUDGET_USD = 4.0


def utc_datetime(value: str) -> datetime:
    value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("Creation timestamp must include a timezone")
    return value.astimezone(UTC)


def compute_deadline(created_utc: str, hourly_usd: float, *, max_hours=4.0,
                     budget_usd=4.0, reserve_usd=0.10, buffer_seconds=120.0):
    """Absolute cap includes startup/setup; hourly rate must include running disk.

    The reserve is withheld from compute. Retained stopped storage remains billed
    separately and is not bounded by this process (termination is forbidden).
    """
    values = (hourly_usd, max_hours, budget_usd, reserve_usd, buffer_seconds)
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError("All budget inputs must be finite")
    if hourly_usd <= 0 or not 0 < max_hours <= MAX_HOURS:
        raise ValueError("Positive hourly rate and at most four hours required")
    if not 0 < budget_usd <= MAX_BUDGET_USD or not 0 <= reserve_usd < budget_usd:
        raise ValueError("Budget exceeds authority or reserve consumes budget")
    if buffer_seconds < 0:
        raise ValueError("Stop buffer cannot be negative")
    seconds = min(max_hours * 3600, (budget_usd - reserve_usd) / hourly_usd * 3600)
    seconds = max(0, seconds - buffer_seconds)
    return utc_datetime(created_utc) + timedelta(seconds=seconds)


def validate_self(pod_id: str, env=None):
    env = os.environ if env is None else env
    if not re.fullmatch(r"[A-Za-z0-9_-]{5,80}", pod_id):
        raise ValueError("Invalid pod identifier")
    if env.get("RUNPOD_POD_ID") != pod_id:
        raise ValueError("Refusing to operate outside the explicitly named pod")


class PodCLI:
    """Only reads its named pod and requests that exact pod's stop action."""

    def __init__(self, pod_id, executable="runpodctl", runner=subprocess.run):
        self.pod_id, self.executable, self.runner = pod_id, executable, runner
        self.stop_args = None
        self.get_args = None

    def _run(self, args):
        # Never echo stdout/stderr: CLI responses may include sensitive metadata.
        return self.runner([self.executable, *args], capture_output=True,
                           text=True, timeout=30, check=False)

    def preflight(self):
        candidates = (
            (["pod", "stop"], ["pod", "get", self.pod_id, "--output", "json"]),
            (["stop", "pod"], ["get", "pod", self.pod_id]),
        )
        for stop_prefix, get_args in candidates:
            help_result = self._run([*stop_prefix, "--help"])
            if help_result.returncode:
                continue
            result = self._run(get_args)
            if result.returncode:
                continue
            # Require a successful authenticated read naming this very pod. Do
            # not save the full response, which can contain pod-scoped secrets.
            if not re.search(r"(?<![A-Za-z0-9_-])" + re.escape(self.pod_id) +
                             r"(?![A-Za-z0-9_-])", result.stdout):
                continue
            self.stop_args = [*stop_prefix, self.pod_id]
            self.get_args = get_args
            return {"authenticated_read": True, "command_style": stop_prefix}
        raise RuntimeError("No authenticated pod-local runpodctl stop path; do not launch work")

    def stop(self):
        if self.stop_args is None:
            raise RuntimeError("CLI must pass read-only preflight before stop")
        result = self._run(self.stop_args)
        return result.returncode == 0


def write_status(path: Path, **fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(timestamp_utc=datetime.now(UTC).isoformat(),
                                  **fields), indent=2) + "\n")
    tmp.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--created-utc", required=True)
    parser.add_argument("--hourly-usd", type=float, required=True)
    parser.add_argument("--max-hours", type=float, default=4.0)
    parser.add_argument("--budget-usd", type=float, default=4.0)
    parser.add_argument("--reserve-usd", type=float, default=0.10)
    parser.add_argument("--buffer-seconds", type=float, default=120.0)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--stop-request-file", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    validate_self(args.pod_id)
    deadline = compute_deadline(args.created_utc, args.hourly_usd,
                                max_hours=args.max_hours, budget_usd=args.budget_usd,
                                reserve_usd=args.reserve_usd,
                                buffer_seconds=args.buffer_seconds)
    common = dict(pod_id=args.pod_id, created_utc=args.created_utc,
                  stop_deadline_utc=deadline.isoformat(), hourly_usd=args.hourly_usd,
                  hard_dollar_cap=False, native_provider_timer=False)
    if shutil.which("runpodctl") is None:
        write_status(args.status, status="preflight_failed", reason="runpodctl_missing", **common)
        raise RuntimeError("runpodctl absent; no stop guard armed")
    cli = PodCLI(args.pod_id)
    try:
        checked = cli.preflight()
    except Exception as exc:
        write_status(args.status, status="preflight_failed", reason=type(exc).__name__, **common)
        raise
    if args.preflight_only:
        write_status(args.status, status="preflight_passed_not_armed", **checked, **common)
        return
    force_stop = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: force_stop.__setitem__(0, True))
    write_status(args.status, status="armed", pid=os.getpid(), **checked, **common)
    # Also use monotonic time: moving the wall clock backwards cannot extend work.
    remaining = max(0.0, (deadline - datetime.now(UTC)).total_seconds())
    monotonic_deadline = time.monotonic() + remaining
    while (not force_stop[0] and datetime.now(UTC) < deadline and
           time.monotonic() < monotonic_deadline and
           not (args.stop_request_file and args.stop_request_file.exists())):
        time.sleep(min(5.0, max(0.0, monotonic_deadline - time.monotonic())))
    # Requesting a stop may kill this process before its response arrives. Keep
    # retrying if still alive; do not confuse worker exit with stopped billing.
    attempt = 0
    while True:
        attempt += 1
        write_status(args.status, status="stop_request_in_flight", attempt=attempt, **common)
        try:
            accepted = cli.stop()
            state = "stop_requested_exit_unconfirmed" if accepted else "stop_request_failed"
        except Exception as exc:
            state = "stop_request_error_" + type(exc).__name__
        write_status(args.status, status=state, attempt=attempt, **common)
        time.sleep(30)


if __name__ == "__main__":
    main()
