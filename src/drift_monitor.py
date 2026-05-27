"""
drift_monitor.py — Lightweight feature & score distribution drift monitor.
===========================================================================

Tracks baseline statistics for active model features and prediction scores.
Detects drift via z-score thresholds, null-rate spikes, and score range violations.

Alerts classified as: OK / WARNING / CRITICAL

Does NOT alter predict_score() contract or block inference.
Integrated as a post-prediction observation layer.
"""

import logging
import numpy as np
from collections import deque

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Baseline statistics (derived from production training data)
# ---------------------------------------------------------------------------

FEATURE_BASELINES = {
    "kd_b02": {"mean": 0.068, "std": 0.017, "min": 0.03, "max": 0.15},
    "water_trans": {"mean": 0.128, "std": 0.056, "min": 0.01, "max": 0.50},
    "signal_strength": {"mean": 39.1, "std": 17.1, "min": 3.0, "max": 200.0},
    "cleanliness": {"mean": 7400.0, "std": 4200.0, "min": 500.0, "max": 16000.0},
}

SCORE_BASELINE = {"mean": 0.60, "std": 0.15, "min": 0.0, "max": 1.0}

# Thresholds (in units of standard deviations or absolute)
Z_WARN = 2.5       # z-score for WARNING
Z_CRIT = 4.0       # z-score for CRITICAL
NULL_RATE_WARN = 0.1   # 10% null rate triggers WARNING
NULL_RATE_CRIT = 0.5   # 50% null rate triggers CRITICAL

# Alert level enum
OK = "OK"
WARNING = "WARNING"
CRITICAL = "CRITICAL"

# Rolling window for score drift (last N predictions)
_SCORE_WINDOW_SIZE = 50
_score_history = deque(maxlen=_SCORE_WINDOW_SIZE)

# Log throttling: suppress repeated alerts
_LOG_EVERY_N = 50  # Re-log every N calls if drift persists
_last_logged_level = OK
_calls_since_log = 0

# Batch counters for summary reporting
_obs_count = 0
_level_counts = {OK: 0, WARNING: 0, CRITICAL: 0}
_feature_drift_count = 0
_score_drift_count = 0
_null_spike_count = 0
_worst_level = OK
_worst_alert = None


def _z_score(value, mean, std):
    """Compute z-score; returns 0.0 if std is zero."""
    if std <= 0:
        return 0.0
    return abs(value - mean) / std


def check_feature_drift(features_dict, schema_features=None):
    """
    Check incoming features against baseline distributions.
    
    Args:
        features_dict: dict of feature_name -> value (from standard_features)
        schema_features: list of active schema feature names to check
        
    Returns:
        dict: {
            "level": "OK" | "WARNING" | "CRITICAL",
            "alerts": list of (feature, level, reason),
            "null_count": int,
        }
    """
    check_features = schema_features or list(FEATURE_BASELINES.keys())
    alerts = []
    null_count = 0
    max_level = OK
    
    for feat in check_features:
        if feat not in FEATURE_BASELINES:
            continue
            
        baseline = FEATURE_BASELINES[feat]
        value = features_dict.get(feat)
        
        # Null / missing check
        if value is None or (isinstance(value, float) and np.isnan(value)):
            null_count += 1
            continue
        
        # Range check
        if value < baseline["min"] or value > baseline["max"]:
            alerts.append((feat, WARNING, f"out of range: {value:.4f} vs [{baseline['min']}, {baseline['max']}]"))
            max_level = WARNING
            continue
        
        # Z-score check
        z = _z_score(value, baseline["mean"], baseline["std"])
        if z >= Z_CRIT:
            alerts.append((feat, CRITICAL, f"z={z:.1f} (value={value:.4f})"))
            max_level = CRITICAL
        elif z >= Z_WARN:
            alerts.append((feat, WARNING, f"z={z:.1f} (value={value:.4f})"))
            if max_level != CRITICAL:
                max_level = WARNING
    
    # Null rate check
    n_checked = len([f for f in check_features if f in FEATURE_BASELINES])
    if n_checked > 0:
        null_rate = null_count / n_checked
        if null_rate >= NULL_RATE_CRIT:
            alerts.append(("_null_rate", CRITICAL, f"null_rate={null_rate:.0%}"))
            max_level = CRITICAL
        elif null_rate >= NULL_RATE_WARN:
            alerts.append(("_null_rate", WARNING, f"null_rate={null_rate:.0%}"))
            if max_level != CRITICAL:
                max_level = WARNING
    
    return {"level": max_level, "alerts": alerts, "null_count": null_count}


def check_score_drift(score):
    """
    Check if prediction score is drifting from expected distribution.
    Uses both single-value range check and rolling window mean drift.
    
    Args:
        score: float, the predicted score
        
    Returns:
        dict: {
            "level": "OK" | "WARNING" | "CRITICAL",
            "alerts": list of (source, level, reason),
            "rolling_mean": float or None,
        }
    """
    alerts = []
    max_level = OK
    
    # Single score range check
    z = _z_score(score, SCORE_BASELINE["mean"], SCORE_BASELINE["std"])
    if z >= Z_CRIT:
        alerts.append(("score_value", CRITICAL, f"z={z:.1f} (score={score:.4f})"))
        max_level = CRITICAL
    elif z >= Z_WARN:
        alerts.append(("score_value", WARNING, f"z={z:.1f} (score={score:.4f})"))
        max_level = WARNING
    
    # Update rolling window
    _score_history.append(score)
    
    # Rolling mean drift (only if enough history)
    rolling_mean = None
    if len(_score_history) >= 10:
        rolling_mean = float(np.mean(_score_history))
        rolling_z = _z_score(rolling_mean, SCORE_BASELINE["mean"], SCORE_BASELINE["std"])
        if rolling_z >= Z_CRIT:
            alerts.append(("rolling_mean", CRITICAL, f"z={rolling_z:.1f} (mean={rolling_mean:.4f} over {len(_score_history)} calls)"))
            max_level = CRITICAL
        elif rolling_z >= Z_WARN:
            alerts.append(("rolling_mean", WARNING, f"z={rolling_z:.1f} (mean={rolling_mean:.4f} over {len(_score_history)} calls)"))
            if max_level != CRITICAL:
                max_level = WARNING
    
    return {"level": max_level, "alerts": alerts, "rolling_mean": rolling_mean}


def observe(features_dict, score, schema_features=None):
    """
    Single entry point: observe one prediction for drift.
    Logs alerts at appropriate levels with throttling to avoid per-call spam.
    Does not alter data or block inference.
    
    Args:
        features_dict: dict of standardized features
        score: float, predicted score
        schema_features: active schema feature list
        
    Returns:
        dict: {"feature_drift": {...}, "score_drift": {...}}
    """
    global _last_logged_level, _calls_since_log
    global _obs_count, _feature_drift_count, _score_drift_count, _null_spike_count
    global _worst_level, _worst_alert
    
    feat_drift = check_feature_drift(features_dict, schema_features)
    score_drift = check_score_drift(score)
    
    # Determine overall severity
    overall = CRITICAL if CRITICAL in (feat_drift["level"], score_drift["level"]) else \
              WARNING if WARNING in (feat_drift["level"], score_drift["level"]) else OK
    
    # Update batch counters
    _obs_count += 1
    _level_counts[overall] = _level_counts.get(overall, 0) + 1
    if feat_drift["level"] != OK:
        _feature_drift_count += 1
    if score_drift["level"] != OK:
        _score_drift_count += 1
    if feat_drift["null_count"] > 0:
        _null_spike_count += 1
    if overall == CRITICAL or (overall == WARNING and _worst_level == OK):
        _worst_level = overall
    all_a = feat_drift["alerts"] + score_drift["alerts"]
    level_rank = {OK: 0, WARNING: 1, CRITICAL: 2}
    _worst_alert = max(all_a, key=lambda a: level_rank.get(a[1], 0)) if len(all_a) > 0 else None
    
    # Throttled logging: log on level change or periodic reminder
    level_changed = (overall != _last_logged_level)
    should_log = (
        overall != OK and (
            level_changed or                   # Severity changed
            _calls_since_log >= _LOG_EVERY_N   # Periodic reminder
        )
    )
    
    if should_log:
        all_alerts = feat_drift["alerts"] + score_drift["alerts"]
        if overall == CRITICAL:
            crit_alerts = [(f, r) for f, l, r in all_alerts if l == CRITICAL]
            log.warning(f"Drift CRITICAL: {crit_alerts}")
        else:
            warn_alerts = [(f, r) for f, l, r in all_alerts if l == WARNING]
            log.warning(f"Drift WARNING: {warn_alerts}")
        _calls_since_log = 1
    elif overall != OK:
        _calls_since_log += 1
    else:
        _calls_since_log = 0
    
    # Log when drift clears
    if overall == OK and _last_logged_level != OK:
        log.debug("Drift cleared — back to normal.")
    
    _last_logged_level = overall
    
    return {"feature_drift": feat_drift, "score_drift": score_drift}


def summary():
    """
    Return a concise batch drift summary.
    
    Returns:
        dict: {
            "total_observations": int,
            "counts": {"OK": int, "WARNING": int, "CRITICAL": int},
            "feature_drift_count": int,
            "score_drift_count": int,
            "null_spike_count": int,
            "worst_level": str,
            "worst_alert": tuple or None,
        }
    """
    return {
        "total_observations": _obs_count,
        "counts": dict(_level_counts),
        "feature_drift_count": _feature_drift_count,
        "score_drift_count": _score_drift_count,
        "null_spike_count": _null_spike_count,
        "worst_level": _worst_level,
        "worst_alert": _worst_alert,
    }


def summary_line():
    """Return a one-line formatted string for batch-end logging."""
    s = summary()
    parts = [
        f"obs={s['total_observations']}",
        f"OK={s['counts'].get(OK, 0)}",
        f"WARN={s['counts'].get(WARNING, 0)}",
        f"CRIT={s['counts'].get(CRITICAL, 0)}",
        f"feat_drift={s['feature_drift_count']}",
        f"score_drift={s['score_drift_count']}",
        f"nulls={s['null_spike_count']}",
    ]
    worst = f" | worst={s['worst_level']}"
    if s['worst_alert']:
        worst += f" ({s['worst_alert'][0]}: {s['worst_alert'][2]})"
    return "Drift summary: " + " ".join(parts) + worst


def log_summary():
    """Log the batch drift summary at INFO level."""
    log.info(summary_line())


def reset():
    """Reset all state: rolling window, throttle, and batch counters."""
    global _last_logged_level, _calls_since_log
    global _obs_count, _feature_drift_count, _score_drift_count, _null_spike_count
    global _worst_level, _worst_alert, _level_counts
    _score_history.clear()
    _last_logged_level = OK
    _calls_since_log = 0
    _obs_count = 0
    _level_counts = {OK: 0, WARNING: 0, CRITICAL: 0}
    _feature_drift_count = 0
    _score_drift_count = 0
    _null_spike_count = 0
    _worst_level = OK
    _worst_alert = None
