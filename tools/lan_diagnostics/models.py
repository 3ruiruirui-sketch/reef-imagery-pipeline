"""Shared dataclasses for LAN diagnostic results."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Device:
    ip: str
    mac: str | None = None


@dataclass
class IperfResult:
    label: str
    sent_mbps: float
    received_mbps: float
    retransmits: int | None = None
    bidir: bool = False


@dataclass
class WifiNetwork:
    ssid: str
    bssid: str
    channel: int
    rssi: int | None = None
    band: str = "2.4GHz"


@dataclass
class InterfaceStats:
    name: str
    status: str
    speed: str | None = None
    crc_errors: int | None = None
