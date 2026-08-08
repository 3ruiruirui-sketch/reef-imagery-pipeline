"""Thin wrapper around the `iperf3` CLI for LAN throughput testing."""

from __future__ import annotations

import json
import subprocess

from .models import IperfResult


def run_server(port: int = 5201) -> None:
    """Run `iperf3 -s` in the foreground until interrupted."""
    subprocess.run(["iperf3", "-s", "-p", str(port)])


def parse_iperf_json(raw: str, label: str, bidir: bool = False) -> IperfResult:
    data = json.loads(raw)
    end = data["end"]
    sent_bps = end["sum_sent"]["bits_per_second"]
    received_bps = end["sum_received"]["bits_per_second"]
    retransmits = end["sum_sent"].get("retransmits")
    return IperfResult(
        label=label,
        sent_mbps=round(sent_bps / 1e6, 2),
        received_mbps=round(received_bps / 1e6, 2),
        retransmits=retransmits,
        bidir=bidir,
    )


def run_client(
    server: str,
    port: int = 5201,
    duration: int = 10,
    parallel: int = 1,
    bidir: bool = False,
    label: str | None = None,
) -> IperfResult:
    cmd = ["iperf3", "-c", server, "-p", str(port), "-t", str(duration), "-P", str(parallel), "-J"]
    if bidir:
        cmd.append("--bidir")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 30, check=True)
    return parse_iperf_json(proc.stdout, label=label or server, bidir=bidir)


def compare_results(results: list[IperfResult]) -> str:
    """Render a Markdown table of results, weakest link (lowest received Mbps) first."""
    lines = [
        "| Label | Sent (Mbps) | Received (Mbps) | Retransmits |",
        "|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda r: r.received_mbps):
        retransmits = r.retransmits if r.retransmits is not None else "-"
        lines.append(f"| {r.label} | {r.sent_mbps} | {r.received_mbps} | {retransmits} |")
    return "\n".join(lines)
