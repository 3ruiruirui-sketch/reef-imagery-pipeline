# LAN Diagnostics

A small standalone toolkit for diagnosing home LAN problems (multiple
Wi-Fi APs, a managed switch, flaky roaming) — unrelated to the reef
imagery pipeline itself, but kept in this repo for convenience.

It implements the checks from a typical "why is my LAN slow" diagnostic
plan: internal throughput testing (iPerf3, not a speed test — that only
measures the WAN link), device/DHCP discovery, Wi-Fi channel-conflict
detection, and managed-switch port diagnostics.

## Requirements

- `iperf3` on the PATH for throughput tests (`brew install iperf3` / `apt install iperf3`).
- `nmcli` on the PATH for Wi-Fi scanning on Linux (macOS uses the bundled `airport` tool).
- `paramiko` (`pip install paramiko`) only if you use `SwitchDiagnostics` to SSH into a managed switch.
- Root/administrator privileges only for `--check-dhcp` (binds UDP port 68).

## Usage

```bash
# 1. Map the network: who's on it, and is more than one DHCP server active?
python -m tools.lan_diagnostics.cli discover 192.168.1.0/24 --check-dhcp

# 2. Test LAN throughput between two points (run the server on a wired machine first)
python -m tools.lan_diagnostics.cli iperf-server
python -m tools.lan_diagnostics.cli iperf-client 192.168.1.10 --bidir -P 8 -t 30

# 3. Check for Wi-Fi channel conflicts
python -m tools.lan_diagnostics.cli wifi-scan

# 4. Generate a single Markdown report combining the above
python -m tools.lan_diagnostics.cli report --subnet 192.168.1.0/24 --check-dhcp --wifi --out lan_report.md
```

Compare iPerf3 results between different access points (client wired to the
switch, then over each AP in turn) to isolate whether a slowdown is the
switch/cabling or a specific AP.

For a managed switch (e.g. Huawei), use `SwitchDiagnostics` directly:

```python
from tools.lan_diagnostics.switch_diag import SwitchDiagnostics

switch = SwitchDiagnostics("192.168.1.2", username="admin", password="...")
print(switch.interface_brief())
print(switch.interface_detail("GigabitEthernet0/0/1").crc_errors)
print(switch.logbuffer())
```

## Ideal topology (avoids the recurring multi-AP issues)

- One DHCP server only (the main router); DHCP disabled on every AP.
- Same SSID/password on all APs for seamless roaming, but a distinct,
  non-overlapping channel per AP (1, 6, or 11 on 2.4GHz).
- APs connected to switch **LAN** ports (not WAN) via CAT5e or better.
- A static IP assigned to each AP.

`report` renders this checklist alongside the live findings so you can see what's already fixed.
