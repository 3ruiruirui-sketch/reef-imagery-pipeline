"""Wi-Fi network scanning and channel-conflict detection (macOS + Linux)."""

from __future__ import annotations

import platform
import re
import subprocess

from .models import WifiNetwork

MAC_RE = re.compile(r"([0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5})")
NONOVERLAPPING_24GHZ = {1, 6, 11}

_MACOS_AIRPORT = "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport"


def _band_for_channel(channel: int) -> str:
    return "5GHz" if channel > 14 else "2.4GHz"


def parse_airport_output(raw: str) -> list[WifiNetwork]:
    """Parse `airport -s` output. Columns are fixed-width but SSIDs may contain
    spaces, so anchor on the BSSID (a MAC address) and split around it."""
    networks = []
    for line in raw.splitlines()[1:]:
        match = MAC_RE.search(line)
        if not match:
            continue
        ssid = line[: match.start()].strip()
        bssid = match.group(1).lower()
        rest = line[match.end() :].split()
        if len(rest) < 2:
            continue
        try:
            rssi = int(rest[0])
            channel = int(re.match(r"\d+", rest[1]).group())
        except (ValueError, AttributeError):
            continue
        networks.append(WifiNetwork(ssid=ssid, bssid=bssid, channel=channel, rssi=rssi, band=_band_for_channel(channel)))
    return networks


def scan_macos() -> list[WifiNetwork]:
    proc = subprocess.run([_MACOS_AIRPORT, "-s"], capture_output=True, text=True, check=True)
    return parse_airport_output(proc.stdout)


def _split_nmcli_line(line: str) -> list[str]:
    """Split an `nmcli -t -e yes` line on unescaped ':' (nmcli escapes ':' inside
    field values, e.g. inside the BSSID, as '\\:')."""
    return [field.replace("\\:", ":") for field in re.split(r"(?<!\\):", line)]


def parse_nmcli_output(raw: str) -> list[WifiNetwork]:
    networks = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        fields = _split_nmcli_line(line)
        if len(fields) != 4:
            continue
        ssid, bssid, chan, signal = fields
        try:
            channel = int(chan)
            rssi = int(signal)
        except ValueError:
            continue
        networks.append(
            WifiNetwork(
                ssid=ssid,
                bssid=bssid.lower(),
                channel=channel,
                rssi=rssi,
                band=_band_for_channel(channel),
            )
        )
    return networks


def scan_linux() -> list[WifiNetwork]:
    proc = subprocess.run(
        ["nmcli", "-t", "-e", "yes", "-f", "SSID,BSSID,CHAN,SIGNAL", "dev", "wifi", "list"],
        capture_output=True,
        text=True,
        check=True,
    )
    return parse_nmcli_output(proc.stdout)


def scan() -> list[WifiNetwork]:
    system = platform.system()
    if system == "Darwin":
        return scan_macos()
    if system == "Linux":
        return scan_linux()
    raise NotImplementedError(f"Wi-Fi scanning is not implemented for {system}")


def detect_channel_conflicts(networks: list[WifiNetwork], rssi_threshold: int = -75) -> list[str]:
    """Flag APs sharing a channel and 2.4GHz APs on a non-standard channel.

    Only networks at or above `rssi_threshold` are considered, since a weak,
    distant network sharing a channel causes negligible real-world interference.
    """
    warnings: list[str] = []
    strong = [n for n in networks if n.rssi is None or n.rssi >= rssi_threshold]

    by_channel: dict[tuple[str, int], list[WifiNetwork]] = {}
    for n in strong:
        by_channel.setdefault((n.band, n.channel), []).append(n)

    for (band, channel), nets in sorted(by_channel.items()):
        distinct_bssids = {n.bssid for n in nets}
        if len(distinct_bssids) > 1:
            ssids = ", ".join(sorted({n.ssid for n in nets}))
            warnings.append(f"Channel {channel} ({band}) is shared by {len(distinct_bssids)} access points: {ssids}")

    for n in strong:
        if n.band == "2.4GHz" and n.channel not in NONOVERLAPPING_24GHZ:
            warnings.append(
                f"{n.ssid} ({n.bssid}) is on 2.4GHz channel {n.channel}, " "not a non-overlapping channel (use 1, 6, or 11)"
            )

    return warnings
