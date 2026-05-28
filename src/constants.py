"""
Physical constants and calibration parameters for reef imagery pipeline.

References:
- Dierssen et al. (2003): Remote sensing of shallow coastal waters
- Stumpf et al. (2003): Determination of water depth with high-resolution
  satellite imagery over variable bottom types
- Gordon et al. (1988): Estimation of seawater constituents from radiance
"""

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

# Refractive index of water (Snell's law)
# Source: Standard value for seawater at 589nm (sodium D-line)
N_WATER = 1.333

# =============================================================================
# DEPTH AND OPTICAL LIMITS
# =============================================================================

# Maximum reliable depth for SDB (optical limit of blue/green ratio method)
# Beyond this, signal is dominated by water column, not bottom
SDB_OPTICAL_LIMIT_M = 40.0

# Default target depth for benthic visibility analysis (metres)
DEFAULT_DEPTH_TARGET = 16.0
DEPTH_TARGET = DEFAULT_DEPTH_TARGET  # Alias for backward compatibility

# =============================================================================
# STUMPF SDB ALGORITHM PARAMETERS
# =============================================================================

# Default Stumpf coefficients for Algarve oligotrophic waters (Kd≈0.045)
# Literature values from regional studies
# Stumpf formula: depth = m1 * ln(n*B02) / ln(n*B03) + m0
STUMPF_M0_DEFAULT = -16.0   # Intercept
STUMPF_M1_DEFAULT = 20.0    # Slope (literature value for clear oligotrophic)
STUMPF_N = 1000.0           # Log scaling factor

# Literature m1 for Stumpf n=1000 in clear oligotrophic waters
# Range validated against Dierssen et al. 2003 and regional studies
# Distinct from STUMPF_M1_DEFAULT: used as a physically-grounded fixed slope
# in single-isobath offset calibration, not as a fallback default.
STUMPF_M1_LITERATURE = 20.0

# =============================================================================
# ATTENUATION COEFFICIENTS (Kd490)
# =============================================================================

# Seasonal Kd490 table for Algarve coastal waters (m^-1)
# Based on historical in-situ measurements and satellite climatology
# Format: {month: Kd490_value}
KD490_TABLE = {
    1: 0.055,   # January: winter mixing, higher turbidity
    2: 0.055,   # February: winter mixing
    4: 0.200,   # April: spring bloom onset
    5: 0.200,   # May: peak spring bloom
    9: 0.045,   # September: oligotrophic summer, clearest waters
    10: 0.045,  # October: autumn transition, still clear
}
# Default for months not in table (June-Aug, Nov-Dec): 0.080 m^-1
KD490_DEFAULT = 0.080

# =============================================================================
# GLINT CORRECTION PENALTIES (monthly factors)
# =============================================================================

# Monthly glint penalties for visibility score calculation
# Higher values = less penalty (better conditions)
# Lower values = more penalty (more sun glint risk)
GLINT_PENALTY = {
    1: 0.60,   # January: low sun angle, moderate glint
    2: 0.70,   # February: increasing sun angle
    4: 0.85,   # April: spring
    5: 0.90,   # May: late spring
    9: 0.95,   # September: high sun but stable seas
    10: 0.60,  # October: lower angle but variable weather
}
# Default penalty for months not in table
GLINT_PENALTY_DEFAULT = 0.80

# =============================================================================
# QUALITY THRESHOLDS
# =============================================================================

# SNR threshold for acceptable quality
SNR_THRESHOLD = 3.0

# Cloud cover threshold (percent)
CLOUD_THRESHOLD = 5.0

# High uncertainty threshold for Kd (relative difference from prior)
KD_HIGH_UNCERTAINTY_THRESHOLD = 0.30  # 30%

# =============================================================================
# CALIBRATION AND VALIDATION
# =============================================================================

# Isobath depths used for benthic zone classification (metres)
BENTHIC_ISOBATHS = [10, 20, 30]

# Context isobaths for zone classification only (metres)
CONTEXT_ISOBATHS = [50, 100]

# Pixel buffer around isobath vertices for sampling (pixels)
# ±3 pixels = ±30m at Sentinel-2 10m resolution
BUF_PIX = 3

# =============================================================================
# REFLECTANCE CONSTANTS (for benthic contrast calculation)
# =============================================================================

# Typical bottom reflectance values (BOA, 0-1 scale)
SAND_R = 0.25    # Bright sand
ROCK_R = 0.05    # Dark rock/reef

# Reflectance scaling factor for DN -> BOA conversion
# Sentinel-2 L2A products use 1/10000 scaling
REFLECTANCE_DN_SCALE = 10000.0

# DN threshold to detect unscaled vs already-scaled reflectance
REFLECTANCE_DN_THRESHOLD = 2.0
