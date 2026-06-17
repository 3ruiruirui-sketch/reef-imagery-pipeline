"""
report_generator.py — Automated HTML Report Generation for Reef Pipeline.

This module generates an HTML summary after a pipeline execution, detailing:
- Processing date and coordinates
- Data sources used (DGT LiDAR, SDB, etc.)
- Number of reef candidates discovered and validated
"""

import os
import glob
import json
from datetime import datetime

def generate_pipeline_report(output_dir: str, date_str: str, lat: float, lon: float) -> str | None:
    """Generates an HTML report summarizing the pipeline run."""
    try:
        report_path = os.path.join(output_dir, f"reef_report_{date_str}.html")
        
        candidates_file = os.path.join(output_dir, f"reef_candidates_{date_str}_validated.geojson")
        num_candidates = 0
        num_high_conf = 0
        
        if os.path.exists(candidates_file):
            with open(candidates_file, "r") as f:
                data = json.load(f)
                features = data.get("features", [])
                num_candidates = len(features)
                num_high_conf = sum(1 for f in features if f.get("properties", {}).get("validation_class") == "HIGH Confidence")
        
        # Check available sources
        has_lidar = len(glob.glob(os.path.join(output_dir, f"bathy_dgt_lidar_{date_str}.tif"))) > 0
        has_emodnet = len(glob.glob(os.path.join(output_dir, f"bathy_emodnet_{date_str}.tif"))) > 0
        has_s2 = len(glob.glob(os.path.join(output_dir, f"bathy_s2_stumpf_{date_str}.tif"))) > 0
        
        sources_html = ""
        if has_lidar: sources_html += "<li>🟢 DGT LiDAR (50cm)</li>"
        if has_emodnet: sources_html += "<li>🔵 EMODnet (115m)</li>"
        if has_s2: sources_html += "<li>🟡 Sentinel-2 SDB (10m) - Stumpf + Sunglint</li>"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reef Pipeline Report - {date_str}</title>
            <style>
                body {{ font-family: -apple-system, system-ui, sans-serif; margin: 40px; background: #0f172a; color: #f8fafc; }}
                h1 {{ border-bottom: 2px solid #3b82f6; padding-bottom: 10px; }}
                .card {{ background: #1e293b; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                .stat {{ font-size: 24px; font-weight: bold; color: #38bdf8; }}
            </style>
        </head>
        <body>
            <h1>Reef Discovery Pipeline Report</h1>
            <p><strong>Execution Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="card">
                <h2>📍 Location Data</h2>
                <p><strong>Target Date:</strong> {date_str}</p>
                <p><strong>Coordinates:</strong> {lat:.5f}, {lon:.5f}</p>
            </div>
            
            <div class="card">
                <h2>📊 Data Sources Utilized</h2>
                <ul>
                    {sources_html or "<li>N/A</li>"}
                </ul>
            </div>
            
            <div class="card">
                <h2>🪸 Discovery Results</h2>
                <p>Total Candidates Detected: <span class="stat">{num_candidates}</span></p>
                <p>High Confidence Reefs: <span class="stat">{num_high_conf}</span></p>
            </div>
        </body>
        </html>
        """
        
        with open(report_path, "w") as f:
            f.write(html_content)
            
        return report_path
    except Exception as e:
        print(f"Error generating report: {e}")
        return None
