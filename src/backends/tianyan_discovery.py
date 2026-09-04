"""Read-only machine inventory retrieval for TianYan CQlib."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any, Iterable, Mapping


SENSITIVE_KEYS = {"access_token", "authorization", "token", "login_key", "password"}
MACHINE_NAME_KEYS = ("machineName", "code", "name", "computerName", "alias")


def sanitize_machine(machine: Mapping[str, Any]) -> dict[str, Any]:
    """Keep server machine metadata while excluding possible credential fields."""
    return {
        str(key): value
        for key, value in machine.items()
        if str(key).lower() not in SENSITIVE_KEYS
    }


def machine_code(machine: Mapping[str, Any]) -> str | None:
    for key in ("code", "machineName", "machine_code", "computerCode"):
        value = machine.get(key)
        if value:
            return str(value)
    return None


def display_values(machine: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for key in MACHINE_NAME_KEYS:
        value = machine.get(key)
        if value is not None:
            values.append(str(value))
    return tuple(values)


def candidate_codes(
    machines: Iterable[Mapping[str, Any]], display_name: str
) -> list[str]:
    target = display_name.casefold()
    candidates: list[str] = []
    for machine in machines:
        if any(target in value.casefold() for value in display_values(machine)):
            code = machine_code(machine)
            if code:
                candidates.append(code)
    return sorted(set(candidates))


def build_inventory(
    machines: Iterable[Mapping[str, Any]],
    *,
    cqlib_version: str,
) -> dict[str, Any]:
    sanitized = [sanitize_machine(machine) for machine in machines]
    sanitized.sort(key=lambda item: machine_code(item) or "")
    return {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "TianYanPlatform._send_request(MACHINE_LIST_PATH)",
        "cqlib_version": cqlib_version,
        "read_only": True,
        "machines": sanitized,
        "role_candidates": {
            "primary_noisy_simulator": candidate_codes(sanitized, "密度矩阵带噪声"),
            "ideal_reference_simulator": candidate_codes(sanitized, "全振幅"),
            "hardware": candidate_codes(sanitized, "tianyan-287"),
        },
    }


def query_inventory(login_key: str | None = None) -> dict[str, Any]:
    """Authenticate with CQlib and retrieve the raw machine-list response only."""
    key = login_key or os.environ.get("TIANYAN_LOGIN_KEY")
    if not key:
        raise RuntimeError("TIANYAN_LOGIN_KEY is not set.")
    try:
        from cqlib._version import __version__ as cqlib_version
        from cqlib.quantum_platform import TianYanPlatform
    except ImportError as exc:
        raise RuntimeError(
            "CQlib is not installed. Install requirements-platform.txt before discovery."
        ) from exc

    platform = TianYanPlatform(login_key=key, auto_login=True)
    # CQlib's public listing API formats rows for display and drops headers. This
    # private call is the same read-only endpoint, retained here to preserve codes.
    response = platform._send_request(platform.MACHINE_LIST_PATH)
    machines = response.get("data") if isinstance(response, Mapping) else None
    if not isinstance(machines, list) or not all(isinstance(item, Mapping) for item in machines):
        raise RuntimeError("TianYan machine-list response did not contain a machine array.")
    return build_inventory(machines, cqlib_version=cqlib_version)
