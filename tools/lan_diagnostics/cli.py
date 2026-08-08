"""Command-line entry point for the LAN diagnostic toolkit.

python -m tools.lan_diagnostics.cli discover 192.168.1.0/24 --check-dhcp
python -m tools.lan_diagnostics.cli iperf-server
python -m tools.lan_diagnostics.cli iperf-client 192.168.1.10 --bidir -P 8 -t 30
python -m tools.lan_diagnostics.cli wifi-scan
python -m tools.lan_diagnostics.cli report --subnet 192.168.1.0/24 --wifi --out report.md
"""

from __future__ import annotations

import argparse
import sys

from . import discovery, iperf_wrapper, wifi_scan
from .report import build_report


def _cmd_discover(args: argparse.Namespace) -> None:
    devices = discovery.discover(args.subnet)
    for d in devices:
        print(f"{d.ip}\t{d.mac or '-'}")
    if args.check_dhcp:
        servers = discovery.detect_dhcp_servers(timeout=args.dhcp_timeout)
        print(f"\nDHCP servers responding: {', '.join(servers) if servers else 'none'}")
        if len(servers) > 1:
            print("WARNING: multiple DHCP servers detected — disable DHCP on all but one.")


def _cmd_iperf_server(args: argparse.Namespace) -> None:
    iperf_wrapper.run_server(port=args.port)


def _cmd_iperf_client(args: argparse.Namespace) -> None:
    result = iperf_wrapper.run_client(
        args.server,
        port=args.port,
        duration=args.time,
        parallel=args.parallel,
        bidir=args.bidir,
        label=args.label,
    )
    print(f"{result.label}: sent {result.sent_mbps} Mbps, received {result.received_mbps} Mbps")


def _cmd_wifi_scan(args: argparse.Namespace) -> None:
    networks = wifi_scan.scan()
    for n in networks:
        rssi = n.rssi if n.rssi is not None else ""
        print(f"{n.ssid}\t{n.bssid}\t{n.band}\tch{n.channel}\t{rssi}")
    for w in wifi_scan.detect_channel_conflicts(networks):
        print(f"WARNING: {w}")


def _cmd_report(args: argparse.Namespace) -> None:
    devices = discovery.discover(args.subnet) if args.subnet else None
    dhcp_servers = discovery.detect_dhcp_servers(timeout=args.dhcp_timeout) if args.check_dhcp else None
    networks = wifi_scan.scan() if args.wifi else None
    warnings = wifi_scan.detect_channel_conflicts(networks) if networks else None

    text = build_report(
        devices=devices,
        dhcp_servers=dhcp_servers,
        wifi_networks=networks,
        wifi_warnings=warnings,
    )
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"Report written to {args.out}")
    else:
        print(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lan-diagnostics",
        description="Home LAN diagnostic toolkit: device discovery, iPerf3, " "Wi-Fi channel scan, managed-switch checks.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_discover = sub.add_parser("discover", help="Scan the LAN for devices and check for rogue DHCP servers")
    p_discover.add_argument("subnet", help="CIDR to scan, e.g. 192.168.1.0/24")
    p_discover.add_argument("--check-dhcp", action="store_true", help="Also probe for active DHCP servers (needs root)")
    p_discover.add_argument("--dhcp-timeout", type=float, default=3.0)
    p_discover.set_defaults(func=_cmd_discover)

    p_server = sub.add_parser("iperf-server", help="Run an iperf3 server (foreground)")
    p_server.add_argument("--port", type=int, default=5201)
    p_server.set_defaults(func=_cmd_iperf_server)

    p_client = sub.add_parser("iperf-client", help="Run an iperf3 client test against a server")
    p_client.add_argument("server")
    p_client.add_argument("--port", type=int, default=5201)
    p_client.add_argument("--time", "-t", type=int, default=10)
    p_client.add_argument("--parallel", "-P", type=int, default=1)
    p_client.add_argument("--bidir", action="store_true")
    p_client.add_argument("--label", default=None)
    p_client.set_defaults(func=_cmd_iperf_client)

    p_wifi = sub.add_parser("wifi-scan", help="Scan nearby Wi-Fi networks and flag channel conflicts")
    p_wifi.set_defaults(func=_cmd_wifi_scan)

    p_report = sub.add_parser("report", help="Run available checks and produce a Markdown diagnostic report")
    p_report.add_argument("--subnet", default=None, help="CIDR to scan for devices, e.g. 192.168.1.0/24")
    p_report.add_argument("--check-dhcp", action="store_true")
    p_report.add_argument("--dhcp-timeout", type=float, default=3.0)
    p_report.add_argument("--wifi", action="store_true", help="Include a Wi-Fi channel scan")
    p_report.add_argument("--out", default=None, help="Write the report to this file instead of stdout")
    p_report.set_defaults(func=_cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
