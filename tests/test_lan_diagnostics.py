"""
Offline unit tests for tools/lan_diagnostics — parsing and pure logic only.

No network, no external processes: all subprocess/CLI output is exercised
against fixed sample strings, so this suite runs unconditionally.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.lan_diagnostics.discovery import parse_arp_output
from tools.lan_diagnostics.iperf_wrapper import compare_results, parse_iperf_json
from tools.lan_diagnostics.models import Device, IperfResult, WifiNetwork
from tools.lan_diagnostics.report import build_report
from tools.lan_diagnostics.switch_diag import parse_interface_detail
from tools.lan_diagnostics.wifi_scan import (
    _split_nmcli_line,
    detect_channel_conflicts,
    parse_airport_output,
    parse_nmcli_output,
)

# ─── discovery ───────────────────────────────────────────────────────────────


def test_parse_arp_output_macos_style():
    raw = (
        "? (192.168.1.1) at ac:1f:6b:11:22:33 on en0 ifscope [ethernet]\n"
        "? (192.168.1.20) at 8:9d:f4:aa:bb:cc on en0 ifscope [ethernet]"
    )
    table = parse_arp_output(raw)
    assert table["192.168.1.1"] == "ac:1f:6b:11:22:33"
    assert table["192.168.1.20"] == "8:9d:f4:aa:bb:cc"


def test_parse_arp_output_ignores_lines_without_mac():
    raw = "? (192.168.1.1) at (incomplete) on en0 ifscope [ethernet]"
    assert parse_arp_output(raw) == {}


# ─── iperf ───────────────────────────────────────────────────────────────────


def _iperf_json(sent_bps: float, received_bps: float, retransmits: int = 0) -> str:
    return json.dumps(
        {
            "end": {
                "sum_sent": {"bits_per_second": sent_bps, "retransmits": retransmits},
                "sum_received": {"bits_per_second": received_bps},
            }
        }
    )


def test_parse_iperf_json():
    raw = _iperf_json(sent_bps=940_000_000, received_bps=935_000_000, retransmits=3)
    result = parse_iperf_json(raw, label="test-run")
    assert result.label == "test-run"
    assert result.sent_mbps == 940.0
    assert result.received_mbps == 935.0
    assert result.retransmits == 3


def test_compare_results_orders_by_received_ascending():
    fast = IperfResult(label="wired", sent_mbps=940, received_mbps=935)
    slow = IperfResult(label="ap-bedroom", sent_mbps=120, received_mbps=80)
    table = compare_results([fast, slow])
    assert table.index("ap-bedroom") < table.index("wired")


# ─── wifi scan ───────────────────────────────────────────────────────────────


def test_parse_airport_output():
    raw = (
        "                            SSID BSSID             RSSI CHANNEL HT CC SECURITY\n"
        "                        HomeNet  ac:1f:6b:11:22:33  -45  6       Y  PT WPA2(PSK/AES/AES)\n"
        "                    HomeNet 5G   ac:1f:6b:11:22:34  -50  40      Y  PT WPA2(PSK/AES/AES)"
    )
    networks = parse_airport_output(raw)
    assert len(networks) == 2
    assert networks[0].ssid == "HomeNet"
    assert networks[0].channel == 6
    assert networks[0].band == "2.4GHz"
    assert networks[0].rssi == -45
    assert networks[1].band == "5GHz"


def test_split_nmcli_line_unescapes_colons_in_bssid():
    line = r"HomeNet:AC\:1F\:6B\:11\:22\:33:6:80"
    fields = _split_nmcli_line(line)
    assert fields == ["HomeNet", "AC:1F:6B:11:22:33", "6", "80"]


def test_parse_nmcli_output():
    raw = r"HomeNet:AC\:1F\:6B\:11\:22\:33:6:80" + "\n" + r"Neighbour:00\:11\:22\:33\:44\:55:6:40"
    networks = parse_nmcli_output(raw)
    assert len(networks) == 2
    assert networks[0].ssid == "HomeNet"
    assert networks[0].bssid == "ac:1f:6b:11:22:33"
    assert networks[0].channel == 6


def test_detect_channel_conflicts_flags_shared_channel():
    networks = [
        WifiNetwork(ssid="AP1", bssid="aa:aa:aa:aa:aa:01", channel=6, rssi=-40, band="2.4GHz"),
        WifiNetwork(ssid="AP2", bssid="aa:aa:aa:aa:aa:02", channel=6, rssi=-45, band="2.4GHz"),
    ]
    warnings = detect_channel_conflicts(networks)
    assert any("Channel 6" in w for w in warnings)


def test_detect_channel_conflicts_flags_non_standard_24ghz_channel():
    networks = [WifiNetwork(ssid="AP1", bssid="aa:aa:aa:aa:aa:01", channel=4, rssi=-40, band="2.4GHz")]
    warnings = detect_channel_conflicts(networks)
    assert any("channel 4" in w for w in warnings)


def test_detect_channel_conflicts_ignores_weak_signals():
    networks = [
        WifiNetwork(ssid="AP1", bssid="aa:aa:aa:aa:aa:01", channel=6, rssi=-40, band="2.4GHz"),
        WifiNetwork(ssid="Distant", bssid="aa:aa:aa:aa:aa:02", channel=6, rssi=-90, band="2.4GHz"),
    ]
    warnings = detect_channel_conflicts(networks, rssi_threshold=-75)
    assert not any("Channel 6" in w for w in warnings)


# ─── switch diagnostics ──────────────────────────────────────────────────────


def test_parse_interface_detail():
    raw = "GigabitEthernet0/0/1 current state : UP\nSpeed : 1000, Duplex: FULL\nCRC: 42"
    stats = parse_interface_detail("GigabitEthernet0/0/1", raw)
    assert stats.status == "UP"
    assert stats.crc_errors == 42


def test_parse_interface_detail_handles_missing_fields():
    stats = parse_interface_detail("eth0", "no useful data here")
    assert stats.status == "unknown"
    assert stats.crc_errors is None


# ─── report ──────────────────────────────────────────────────────────────────


def test_build_report_includes_dhcp_warning_for_multiple_servers():
    text = build_report(dhcp_servers=["192.168.1.1", "192.168.1.5"])
    assert "WARNING" in text
    assert "192.168.1.1" in text
    assert "192.168.1.5" in text


def test_build_report_includes_checklist_and_devices():
    text = build_report(devices=[Device(ip="192.168.1.20", mac="aa:bb:cc:dd:ee:ff")])
    assert "192.168.1.20" in text
    assert "Recommended configuration checklist" in text
