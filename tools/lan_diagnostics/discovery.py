"""LAN device discovery and rogue-DHCP-server detection.

Discovery uses a ping sweep + the OS ARP table rather than raw sockets/scapy,
so it needs no extra dependencies and no elevated privileges. DHCP-server
detection does need root/administrator, since it binds UDP port 68.
"""

from __future__ import annotations

import ipaddress
import platform
import re
import socket
import struct
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from random import randint

from .models import Device

MAC_RE = re.compile(r"([0-9a-fA-F]{1,2}(?::[0-9a-fA-F]{1,2}){5})")
IP_RE = re.compile(r"\(?(\d{1,3}(?:\.\d{1,3}){3})\)?")

_DHCP_MAGIC_COOKIE = bytes([99, 130, 83, 99])


def _ping_once(ip: str, timeout_s: float) -> bool:
    import subprocess

    is_windows = platform.system() == "Windows"
    count_flag = "-n" if is_windows else "-c"
    timeout_flag = "-w" if is_windows else "-W"
    timeout_val = str(int(timeout_s * 1000)) if is_windows else str(max(1, int(timeout_s)))
    cmd = ["ping", count_flag, "1", timeout_flag, timeout_val, ip]
    try:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout_s + 2)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def ping_sweep(subnet: str, max_workers: int = 64, timeout_s: float = 1.0) -> list[str]:
    """Return the IPs in `subnet` (e.g. "192.168.1.0/24") that answer a ping."""
    network = ipaddress.ip_network(subnet, strict=False)
    hosts = [str(h) for h in network.hosts()]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = pool.map(lambda ip: _ping_once(ip, timeout_s), hosts)
        return [ip for ip, ok in zip(hosts, results) if ok]


def parse_arp_output(raw: str) -> dict[str, str]:
    table: dict[str, str] = {}
    for line in raw.splitlines():
        ip_match = IP_RE.search(line)
        mac_match = MAC_RE.search(line)
        if ip_match and mac_match:
            table[ip_match.group(1)] = mac_match.group(1).lower()
    return table


def arp_table() -> dict[str, str]:
    import subprocess

    try:
        raw = subprocess.run(["arp", "-a"], capture_output=True, text=True, timeout=5).stdout
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return {}
    return parse_arp_output(raw)


def discover(subnet: str, max_workers: int = 64, timeout_s: float = 1.0) -> list[Device]:
    """Ping-sweep `subnet` and cross-reference the ARP table for MAC addresses."""
    live_ips = ping_sweep(subnet, max_workers=max_workers, timeout_s=timeout_s)
    macs = arp_table()
    return [Device(ip=ip, mac=macs.get(ip)) for ip in live_ips]


def _build_dhcp_discover(xid: int, mac: bytes) -> bytes:
    packet = struct.pack("!BBBBIHH", 1, 1, 6, 0, xid, 0, 0x8000)
    packet += b"\x00" * 4  # ciaddr
    packet += b"\x00" * 4  # yiaddr
    packet += b"\x00" * 4  # siaddr
    packet += b"\x00" * 4  # giaddr
    packet += mac.ljust(16, b"\x00")  # chaddr
    packet += b"\x00" * 64  # sname
    packet += b"\x00" * 128  # file
    packet += _DHCP_MAGIC_COOKIE
    packet += bytes([53, 1, 1])  # option 53: DHCPDISCOVER
    packet += bytes([255])  # end option
    return packet


def _extract_dhcp_server_id(data: bytes) -> str | None:
    options = data[240:]
    i = 0
    while i < len(options) - 1:
        code = options[i]
        if code == 255:
            break
        if code == 0:
            i += 1
            continue
        length = options[i + 1]
        value = options[i + 2 : i + 2 + length]
        if code == 54 and length == 4:
            return ".".join(str(b) for b in value)
        i += 2 + length
    return None


def detect_dhcp_servers(timeout: float = 3.0) -> list[str]:
    """Broadcast a DHCPDISCOVER and collect the IPs of every server that offers a lease.

    More than one distinct server responding means multiple DHCP servers are
    active on the LAN, which causes the IP conflicts described in the
    diagnostic plan. Requires root/administrator privileges (binds UDP/68).
    """
    mac = uuid.getnode().to_bytes(6, "big")
    xid = randint(0, 0xFFFFFFFF)
    packet = _build_dhcp_discover(xid, mac)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.settimeout(timeout)
    try:
        sock.bind(("0.0.0.0", 68))
    except PermissionError as exc:
        sock.close()
        raise PermissionError("Binding UDP port 68 requires root/administrator privileges (try running with sudo)") from exc

    servers: set[str] = set()
    try:
        sock.sendto(packet, ("255.255.255.255", 67))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            sock.settimeout(max(0.0, remaining))
            try:
                data, addr = sock.recvfrom(1024)
            except socket.timeout:
                break
            if len(data) < 240 or data[4:8] != struct.pack("!I", xid):
                continue
            servers.add(_extract_dhcp_server_id(data) or addr[0])
    finally:
        sock.close()
    return sorted(servers)
