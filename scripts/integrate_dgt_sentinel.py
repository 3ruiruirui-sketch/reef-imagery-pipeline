#!/usr/bin/env python3
"""
Integration example: DGT coastal topography + Sentinel-2 + reef visibility model

Demonstrates:
  1. Extract coastal terrain features (slope, aspect) from DGT MDT-50cm
  2. Download Sentinel-2 L2A imagery
  3. Combine as features for reef visibility/water quality prediction
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coastal_topography import CoastalTopographyAnalyzer
from dgt_sentinel_integrator import DGTSentinelIntegrator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


# Survey sites from algarve_reef_survey.py
SURVEY_SITES = [
    ("tavira_west",          37.1050, -7.6800),
    ("ilha_de_tavira",       37.0950, -7.7200),
    ("fuseta",               37.0600, -7.7600),
    ("olhao_offshore",       37.0350, -7.8200),
    ("faro_east",            37.0100, -7.8800),
    ("praia_de_faro",        36.9800, -7.9400),
    ("ancão_peninsula",      36.9650, -8.0000),
    ("quarteira",            37.0650, -8.1000),
    ("vilamoura",            37.0750, -8.1300),
    ("olhos_de_agua",        37.0900, -8.1600),
    ("pedra_sta_eulalia",    37.069081, -8.210242),
    ("albufeira_reef",       37.0690, -8.2105),
    ("galé",                 37.0560, -8.2296),
    ("salgados",             37.0950, -8.3000),
    ("armacao_de_pera",      37.0700, -8.3600),
]


def compute_bbox_from_sites(sites, margin=0.1):
    """Compute bbox covering all sites with margin."""
    lats = [s[1] for s in sites]
    lons = [s[2] for s in sites]
    return (
        min(lons) - margin,
        min(lats) - margin,
        max(lons) + margin,
        max(lats) + margin
    )


def run_coastal_topography_analysis(output_dir, buffer_m=1000):
    """Extract coastal topography features for all sites."""
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: COASTAL TOPOGRAPHY ANALYSIS")
    logger.info("=" * 70)
    
    bbox = compute_bbox_from_sites(SURVEY_SITES, margin=0.1)
    
    analyzer = CoastalTopographyAnalyzer(
        bbox=bbox,
        output_dir=str(Path(output_dir) / "coastal_topography"),
        cache_tiles=True
    )
    
    result = analyzer.run_analysis(
        sites=SURVEY_SITES,
        buffer_m=buffer_m,
        output_name="algarve_coastal_features"
    )
    
    logger.info("\nCoastal Topography Result:")
    logger.info(json.dumps(result, indent=2))
    
    return result


def run_dgt_sentinel_integration(output_dir, date_start="2024-07-01", date_end="2024-08-31"):
    """Download and integrate MDT-50cm + Sentinel-2."""
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: DGT MDT-50cm + SENTINEL-2 INTEGRATION")
    logger.info("=" * 70)
    
    # Focus on Albufeira area (has both reef and good MDT coverage)
    bbox = (-8.25, 37.04, -8.17, 37.10)
    
    integrator = DGTSentinelIntegrator(
        bbox=bbox,
        output_dir=str(Path(output_dir) / "dgt_sentinel"),
    )
    
    result = integrator.integrate(
        fetch_sentinel=False,  # Set to True if sentinelhub credentials available
        date_start=date_start,
        date_end=date_end,
        cloud_pct=30
    )
    
    logger.info("\nDGT-Sentinel Integration Result:")
    logger.info(json.dumps(result, indent=2))
    
    return result


def create_integration_report(output_dir, coastal_result, integration_result):
    """Create a summary report combining both analyses."""
    report = {
        "title": "Algarve Reef Imagery Pipeline: DGT + Sentinel Integration Report",
        "timestamp": str(pd.Timestamp.now()),
        "survey_sites_count": len(SURVEY_SITES),
        "coastal_topography": coastal_result,
        "dgt_sentinel": integration_result,
        "next_steps": [
            "1. Load coastal_features CSV into visibility model as static features",
            "2. Download Sentinel-2 L2A for date range and reproject to EPSG:3763",
            "3. Stack coastal topography + Sentinel bands for multi-modal reef analysis",
            "4. Validate against in-situ measurements at select dive sites",
        ],
    }
    
    report_path = Path(output_dir) / "integration_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"\nIntegration report saved: {report_path}")
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Integrate DGT coastal topography + Sentinel-2 for reef analysis"
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs/dgt_sentinel_integration",
        help="Output directory (default: ./outputs/dgt_sentinel_integration)"
    )
    parser.add_argument(
        "--buffer-m",
        type=float,
        default=1000,
        help="Buffer radius around dive sites in meters (default: 1000)"
    )
    parser.add_argument(
        "--date-start",
        default="2024-07-01",
        help="Date range start for Sentinel-2 (YYYY-MM-DD, default: 2024-07-01)"
    )
    parser.add_argument(
        "--date-end",
        default="2024-08-31",
        help="Date range end for Sentinel-2 (YYYY-MM-DD, default: 2024-08-31)"
    )
    parser.add_argument(
        "--skip-topography",
        action="store_true",
        help="Skip coastal topography analysis"
    )
    parser.add_argument(
        "--skip-sentinel",
        action="store_true",
        help="Skip DGT-Sentinel integration"
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Survey sites: {len(SURVEY_SITES)}")
    
    results = {}
    
    # Run analyses
    if not args.skip_topography:
        results["coastal_topography"] = run_coastal_topography_analysis(
            output_dir, buffer_m=args.buffer_m
        )
    
    if not args.skip_sentinel:
        results["dgt_sentinel"] = run_dgt_sentinel_integration(
            output_dir, date_start=args.date_start, date_end=args.date_end
        )
    
    # Create integration report
    if results:
        report = create_integration_report(
            output_dir,
            results.get("coastal_topography", {}),
            results.get("dgt_sentinel", {})
        )
    
    logger.info("\n" + "=" * 70)
    logger.info("INTEGRATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\nOutput files in: {output_dir}")
    logger.info("\nKey outputs:")
    logger.info("  - coastal_topography/algarve_coastal_features.csv")
    logger.info("  - coastal_topography/algarve_coastal_features.geojson")
    logger.info("  - dgt_sentinel/MDT_50cm_mosaic_algarve.tif")
    logger.info("  - dgt_sentinel/integrated_mdt_sentinel_algarve.tif")
    logger.info("  - integration_report.json")


if __name__ == "__main__":
    # Ensure pandas is available for report
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas not installed; report generation may be limited")
        pd = None
    
    main()
