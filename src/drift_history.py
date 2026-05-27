"""
drift_history.py — Historical aggregation of batch drift reports.
=================================================================

Reads exported JSON drift summaries from drift_reports/ and produces
a time-ordered historical artifact (CSV or JSON) for dashboard use.

Resilient: skips malformed files, handles missing fields gracefully,
returns empty results cleanly if no reports exist.

Dependencies: stdlib only (json, os, csv, glob).
"""

import os
import json
import csv
import glob
import logging

log = logging.getLogger(__name__)

# Default reports directory
_REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "drift_reports")

# Fields extracted from each batch report (flat structure for CSV)
HISTORY_FIELDS = [
    "timestamp",
    "pipeline",
    "model_version",
    "schema_version",
    "batch_id",
    "observations",
    "alerts_ok",
    "alerts_warning",
    "alerts_critical",
    "feature_drift_count",
    "score_drift_count",
    "null_spike_count",
    "highest_severity",
    "worst_alert_feature",
    "worst_alert_level",
    "worst_alert_reason",
    "summary",
]


def _flatten_report(report):
    """
    Flatten a single drift report dict into a row dict with HISTORY_FIELDS keys.
    Missing fields default to empty string or 0 as appropriate.
    """
    alerts = report.get("alerts", {})
    worst = report.get("worst_alert") or {}

    return {
        "timestamp": report.get("timestamp", ""),
        "pipeline": report.get("pipeline", ""),
        "model_version": report.get("model_version", ""),
        "schema_version": report.get("schema_version", ""),
        "batch_id": report.get("batch_id", ""),
        "observations": report.get("observations", 0),
        "alerts_ok": alerts.get("ok", 0),
        "alerts_warning": alerts.get("warning", 0),
        "alerts_critical": alerts.get("critical", 0),
        "feature_drift_count": report.get("feature_drift_count", 0),
        "score_drift_count": report.get("score_drift_count", 0),
        "null_spike_count": report.get("null_spike_count", 0),
        "highest_severity": report.get("highest_severity", ""),
        "worst_alert_feature": worst.get("feature", ""),
        "worst_alert_level": worst.get("level", ""),
        "worst_alert_reason": worst.get("reason", ""),
        "summary": report.get("summary", ""),
    }


def load_reports(reports_dir=None):
    """
    Load all drift report JSON files from a directory.

    Args:
        reports_dir: path to directory containing drift_*.json files

    Returns:
        list of dict: parsed report dicts, sorted by timestamp ascending.
                      Malformed files are skipped with a warning.
    """
    directory = reports_dir or _REPORTS_DIR

    if not os.path.isdir(directory):
        log.info(f"No drift_reports directory found at {directory}")
        return []

    pattern = os.path.join(directory, "drift_*.json")
    files = sorted(glob.glob(pattern))

    if not files:
        log.info(f"No drift report files found in {directory}")
        return []

    reports = []
    for filepath in files:
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                log.warning(f"Skipping non-dict report: {filepath}")
                continue
            reports.append(data)
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"Skipping malformed report {filepath}: {e}")
            continue

    # Sort by timestamp (lexicographic on ISO format works correctly)
    reports.sort(key=lambda r: r.get("timestamp", ""))
    return reports


def aggregate(reports_dir=None):
    """
    Load and flatten all drift reports into a list of row dicts.

    Args:
        reports_dir: path to reports directory

    Returns:
        list of dict: one flat dict per batch, ordered by timestamp.
    """
    reports = load_reports(reports_dir)
    return [_flatten_report(r) for r in reports]


def export_history_csv(output_path=None, reports_dir=None):
    """
    Aggregate drift history and write to CSV.

    Args:
        output_path: file path for CSV output (default: drift_reports/history.csv)
        reports_dir: source directory for report JSONs

    Returns:
        str: path to written CSV, or None if no data
    """
    rows = aggregate(reports_dir)
    if not rows:
        log.info("No drift reports to aggregate into CSV.")
        return None

    directory = reports_dir or _REPORTS_DIR
    path = output_path or os.path.join(directory, "history.csv")

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        log.info(f"Drift history CSV exported: {path} ({len(rows)} batches)")
        return path
    except Exception as e:
        log.error(f"Failed to write drift history CSV: {e}")
        return None


def export_history_json(output_path=None, reports_dir=None):
    """
    Aggregate drift history and write to a single JSON array file.

    Args:
        output_path: file path for JSON output (default: drift_reports/history.json)
        reports_dir: source directory for report JSONs

    Returns:
        str: path to written JSON, or None if no data
    """
    rows = aggregate(reports_dir)
    if not rows:
        log.info("No drift reports to aggregate into JSON.")
        return None

    directory = reports_dir or _REPORTS_DIR
    path = output_path or os.path.join(directory, "history.json")

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(rows, f, indent=2)
        log.info(f"Drift history JSON exported: {path} ({len(rows)} batches)")
        return path
    except Exception as e:
        log.error(f"Failed to write drift history JSON: {e}")
        return None
