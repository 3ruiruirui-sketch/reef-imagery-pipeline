"""
drift_report.py — Canonical entry point for HTML drift report generation.
==========================================================================

Re-exports from drift_report_html.py for convenience.

Usage:
    from src.drift_report import export_html, generate_html
    export_html()  # → drift_reports/report.html
"""

from src.drift_report_html import (
    export_html,
    generate_html,
)

__all__ = ["generate_html", "export_html"]
