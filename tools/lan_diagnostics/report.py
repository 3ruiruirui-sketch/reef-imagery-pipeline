"""Aggregate LAN diagnostic results into a single Markdown report."""

from __future__ import annotations

from datetime import datetime

from .iperf_wrapper import compare_results
from .models import Device, IperfResult, WifiNetwork

CHECKLIST = [
    "Single DHCP server active (on the main router); DHCP disabled on all APs",
    "Same SSID and password on all APs, for seamless roaming",
    "Each AP on a distinct, non-overlapping channel (1, 6, or 11 on 2.4GHz)",
    "APs connected to switch LAN ports (not WAN) via CAT5e or better cable",
    "Static IP assigned to each AP",
]


def build_report(
    devices: list[Device] | None = None,
    dhcp_servers: list[str] | None = None,
    iperf_results: list[IperfResult] | None = None,
    wifi_networks: list[WifiNetwork] | None = None,
    wifi_warnings: list[str] | None = None,
    switch_notes: str | None = None,
) -> str:
    lines = ["# LAN Diagnostic Report", f"_Generated {datetime.now().isoformat(timespec='seconds')}_", ""]

    if devices is not None:
        lines.append("## Discovered devices")
        lines.append(f"{len(devices)} device(s) responded to the ping sweep.")
        lines.append("")
        lines.append("| IP | MAC |")
        lines.append("|---|---|")
        for d in devices:
            lines.append(f"| {d.ip} | {d.mac or '-'} |")
        lines.append("")

    if dhcp_servers is not None:
        lines.append("## DHCP servers detected")
        if len(dhcp_servers) > 1:
            lines.append(
                f"**WARNING:** {len(dhcp_servers)} DHCP servers responded — "
                "disable DHCP on all but one router/AP to avoid IP conflicts."
            )
        elif not dhcp_servers:
            lines.append("No DHCP servers responded.")
        for s in dhcp_servers:
            lines.append(f"- {s}")
        lines.append("")

    if iperf_results:
        lines.append("## iPerf3 throughput results")
        lines.append(compare_results(iperf_results))
        lines.append("")

    if wifi_networks is not None:
        lines.append("## Wi-Fi networks")
        lines.append("| SSID | BSSID | Band | Channel | RSSI |")
        lines.append("|---|---|---|---|---|")
        for n in wifi_networks:
            rssi = n.rssi if n.rssi is not None else "-"
            lines.append(f"| {n.ssid} | {n.bssid} | {n.band} | {n.channel} | {rssi} |")
        lines.append("")

    if wifi_warnings:
        lines.append("## Wi-Fi warnings")
        for w in wifi_warnings:
            lines.append(f"- {w}")
        lines.append("")

    if switch_notes:
        lines.append("## Switch diagnostics")
        lines.append(switch_notes)
        lines.append("")

    lines.append("## Recommended configuration checklist")
    lines.extend(f"- [ ] {item}" for item in CHECKLIST)

    return "\n".join(lines)
