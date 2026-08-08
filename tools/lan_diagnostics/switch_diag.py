"""SSH into a managed switch (e.g. Huawei) and pull interface/error diagnostics.

paramiko is an optional dependency, guarded the same way heavy/optional
imports are guarded elsewhere in this repo's src/ modules.
"""

from __future__ import annotations

import re

try:
    import paramiko
except ImportError:
    paramiko = None

from .models import InterfaceStats


def parse_interface_detail(name: str, raw: str) -> InterfaceStats:
    status_match = re.search(r"current state\s*:?\s*([^\n,]+)", raw, re.IGNORECASE)
    speed_match = re.search(r"speed\s*:?\s*(\d+\s*\w*)", raw, re.IGNORECASE)
    crc_match = re.search(r"CRC\s*:?\s*(\d+)", raw, re.IGNORECASE)
    return InterfaceStats(
        name=name,
        status=status_match.group(1).strip() if status_match else "unknown",
        speed=speed_match.group(1).strip() if speed_match else None,
        crc_errors=int(crc_match.group(1)) if crc_match else None,
    )


class SwitchDiagnostics:
    """Runs read-only `display` commands over SSH against a managed switch."""

    def __init__(
        self,
        host: str,
        username: str,
        password: str | None = None,
        key_filename: str | None = None,
        port: int = 22,
        timeout: float = 10.0,
    ):
        if paramiko is None:
            raise ImportError("paramiko is required for switch diagnostics: pip install paramiko")
        self.host = host
        self.username = username
        self.password = password
        self.key_filename = key_filename
        self.port = port
        self.timeout = timeout

    def _run(self, command: str) -> str:
        client = paramiko.SSHClient()
        # AutoAddPolicy trusts an unknown host key on first connect. Acceptable
        # for a switch on your own LAN reached directly by IP; do not point
        # this at a host over an untrusted network path.
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(
                self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                key_filename=self.key_filename,
                timeout=self.timeout,
            )
            _, stdout, _ = client.exec_command(command, timeout=self.timeout)
            return stdout.read().decode(errors="replace")
        finally:
            client.close()

    def interface_brief(self) -> str:
        return self._run("display interface brief")

    def interface_detail(self, interface: str) -> InterfaceStats:
        raw = self._run(f"display interface {interface}")
        return parse_interface_detail(interface, raw)

    def logbuffer(self) -> str:
        return self._run("display logbuffer")
