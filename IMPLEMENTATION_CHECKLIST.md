# DGT STAC Integration - Implementation Checklist

## Status: ✅ READY FOR TESTING

---

## Phase 1: Setup ✅ COMPLETE

### Files Created
- [x] `src/coastal_topography.py` (21 KB, 600+ lines)
- [x] `src/dgt_sentinel_integrator.py` (15 KB, 500+ lines)
- [x] `scripts/integrate_dgt_sentinel.py` (CLI orchestrator, 200+ lines)
- [x] `docs/DGT_SENTINEL_INTEGRATION.md` (Technical reference)
- [x] `DGT_STAC_GUIDE.md` (User guide, Portuguese focus)
- [x] `TECHNICAL_SUMMARY_DGT_STAC.txt` (Quick reference)
- [x] `requirements.txt` (updated with new dependencies)

### Dependencies Added
- [x] `geopandas>=0.13.0`
- [x] `rasterstats>=0.17.0`
- [x] `rioxarray>=0.13.0`
- [x] `xarray>=2023.12.0`

### Validation
- [x] STAC API connectivity tested (10 tiles found)
- [x] Module imports verified
- [x] Code style follows project conventions
- [x] Logging configured
- [x] Error handling in place
- [x] Caching implemented

---

## Phase 2: Installation (Next)

### Prerequisites
- [ ] Python 3.8+
- [ ] pip or conda
- [ ] ~5 GB disk space for test run

### Steps
```bash
# 1. Install requirements
pip install -r requirements.txt

# 2. Verify installation
python -c "import coastal_topography; print('✓ Import OK')"

# 3. Quick connectivity test
python -c "
from src.coastal_topography import CoastalTopographyAnalyzer
a = CoastalTopographyAnalyzer((-8.25, 37.04, -8.17, 37.10), '/tmp/test')
feats = a.fetch_stac_items(3)
print(f'✓ STAC connectivity: {len(feats)} tiles')
"
```

---

## Phase 3: Pilot Testing

### Test 1: Single Site Feature Extraction
```bash
python -c "
from src.coastal_topography import CoastalTopographyAnalyzer
import logging
logging.basicConfig(level=logging.INFO)

sites = [('test_site', 37.069081, -8.210242)]
bbox = (-8.25, 37.04, -8.17, 37.10)

analyzer = CoastalTopographyAnalyzer(bbox, './test_output')
result = analyzer.run_analysis(sites, buffer_m=1000)

# Check output
import os
assert os.path.exists('./test_output/dem_mosaic_50cm.tif'), 'DEM missing'
assert os.path.exists('./test_output/algarve_coastal_features.csv'), 'CSV missing'
print('✓ All outputs created')
"
```

**Expected Output:**
- DEM mosaic GeoTIFF (~100-200 MB)
- Slope raster (float32)
- Aspect raster (float32)
- CSV with features
- Duration: 5-10 minutes (first run)

**Validation Checklist:**
- [ ] All files created
- [ ] No errors in logs
- [ ] CSV has correct columns (slope_mean, aspect_mean, etc.)
- [ ] GeoTIFFs have EPSG:3763 CRS
- [ ] Feature values are in expected ranges (slope 0-90°, aspect 0-360°)

### Test 2: Multi-Site Analysis
```bash
python scripts/integrate_dgt_sentinel.py \
    --output-dir ./test_algarve \
    --buffer-m 1000 \
    --skip-sentinel  # Skip for now (needs credentials)
```

**Expected Output:**
- Features for all 15 Algarve survey sites
- Integration report JSON
- Duration: 10-15 minutes

**Validation Checklist:**
- [ ] All 15 sites processed
- [ ] CSV has 15 rows
- [ ] No missing values in critical columns
- [ ] Aspect values reasonable (mix of 0-360°)
- [ ] Slope values vary (coastal sites different slopes)

### Test 3: Manual Validation
```bash
# Compare with QGIS hillshade
1. Open QGIS
2. Load: ./test_output/dem_mosaic_50cm.tif
3. Raster → Analysis → Hillshade
4. Compare visually with generated aspect/slope
5. Check slope matches visible terrain
6. Verify aspect matches known exposure (ex. west-facing should be ~270°)
```

**Validation Checklist:**
- [ ] Slope values visually match terrain steepness
- [ ] Aspect cardinal directions correct (N≈0, E≈90, S≈180, W≈270)
- [ ] Nodata handling correct (no artifacts)
- [ ] CRS transformation accurate (no misalignment)

---

## Phase 4: Integration with Pipeline

### Integration Point 1: Visibility Model
```python
# in src/ranking_model.py

# Load coastal features (computed once, reused always)
coastal_features = pd.read_csv("outputs/coastal/algarve_coastal_features.csv")

# Merge with existing reef data
X = pd.merge(reef_data, coastal_features, on="site_name", how="left")

# Use in model
X_features = X[["slope_mean", "slope_p90", "aspect_mean", "sentinel_b02", ...]]
y = X["visibility_score"]

model.fit(X_features, y)
```

**Validation Checklist:**
- [ ] Features load without errors
- [ ] No NaN values (or handled appropriately)
- [ ] Model trains successfully
- [ ] Performance improves vs. baseline
- [ ] Feature importance shows slope/aspect as significant

### Integration Point 2: Drift Monitor
```python
# in src/drift_monitor.py

# Use terrain exposure for plume extent estimation
def estimate_plume(site, wind_speed, wind_direction):
    topo = coastal_features.loc[site]
    exposure = calc_exposure(topo["aspect_mean"], wind_direction)
    plume_extent = base_extent * (1 + exposure * wind_speed/10)
    return plume_extent
```

**Validation Checklist:**
- [ ] Plume estimates reasonable (0-50 km)
- [ ] Exposure factor correlates with known patterns
- [ ] Performance vs. historical data acceptable

### Integration Point 3: Dashboard
```python
# Serve DEM as COG for web visualization
from rio_cogeo.cogeo import cog_translate

cog_translate(
    "outputs/coastal/dem_mosaic_50cm.tif",
    "outputs/coastal/dem_cog.tif",
    dst_kwargs={"COMPRESS": "zstd"}
)

# Add to dashboard map
```

**Validation Checklist:**
- [ ] COG created successfully
- [ ] Tile server can read it
- [ ] Web map displays without lag
- [ ] Hillshade rendering correct

---

## Phase 5: Performance Benchmarking

### Metrics to Measure
- [ ] STAC query latency
- [ ] Download speed (Mbps)
- [ ] Mosaic creation time
- [ ] Slope/aspect computation time
- [ ] Memory usage during operations
- [ ] Disk usage (input + output)

### Targets
- Single site: <5 minutes
- 15 sites: <15 minutes
- Memory peak: <2 GB
- Disk output: <500 MB (with caching)

**Run Test:**
```bash
python -c "
import time, psutil, os
from src.coastal_topography import CoastalTopographyAnalyzer

sites = [(f'site{i}', 37.06+i*0.01, -8.21) for i in range(5)]
analyzer = CoastalTopographyAnalyzer((-8.3, 37.0, -8.1, 37.1), './bench')

t0 = time.time()
result = analyzer.run_analysis(sites)
elapsed = time.time() - t0

print(f'Elapsed: {elapsed:.1f}s')
print(f'Result: {result}')
"
```

---

## Phase 6: Documentation & Deployment

### Documentation
- [ ] README updated with coastal_topography examples
- [ ] Docstrings complete in all modules
- [ ] Example notebooks created (optional)
- [ ] Contributing guide mentions DGT integration

### Code Quality
- [ ] Linting: `pylint src/coastal_topography.py`
- [ ] Type hints: coverage >80%
- [ ] Unit tests: >70% coverage
- [ ] Integration tests with sample data

### Deployment
- [ ] Code pushed to repository
- [ ] PR created with all changes
- [ ] Code review completed
- [ ] CI/CD passing (if configured)
- [ ] Merged to main branch

---

## Common Issues & Troubleshooting

### Issue: "No tiles found"
**Causes:**
- BBox outside Portugal
- STAC API down
- Network timeout

**Solution:**
```python
# Verify bbox
bbox = (-8.25, 37.04, -8.17, 37.10)  # Must be in Portugal
assert -10 < bbox[0] < -6, "Lon out of range"
assert 36 < bbox[1] < 42, "Lat out of range"

# Test connectivity
import requests
r = requests.get(DGT_URL, timeout=10)
r.raise_for_status()
```

### Issue: "Slope/aspect all NaN"
**Causes:**
- DEM nodata not recognized
- Grid resolution issue
- Gradient computation error

**Solution:**
```python
# Check nodata handling
with rasterio.open(dem_path) as src:
    print(f"Nodata value: {src.nodata}")
    dem = src.read(1, masked=True)
    print(f"Valid pixels: {dem.count()}/{dem.size}")
```

### Issue: Memory exhausted
**Causes:**
- Large bbox → many tiles
- Rasterio loading full arrays

**Solution:**
- Reduce bbox size
- Process tiles individually
- Use block-based I/O

### Issue: "sentinelhub authentication fails"
**Solution:**
```bash
# Create config file
mkdir -p ~/.sentinelhub
echo '{
  "sh_client_id": "YOUR_ID",
  "sh_client_secret": "YOUR_SECRET"
}' > ~/.sentinelhub/config.json
```

---

## Success Criteria

### Phase 1 ✅
- [x] All modules created and validated
- [x] Dependencies added to requirements
- [x] Documentation complete
- [x] API connectivity tested

### Phase 2
- [ ] Install dependencies without errors
- [ ] Quick import test passes

### Phase 3
- [ ] Single-site test completes <10 min
- [ ] Features extracted correctly
- [ ] Manual validation passes
- [ ] Values in expected ranges

### Phase 4
- [ ] Integrated with ranking_model
- [ ] Model trains successfully
- [ ] Performance metrics acceptable
- [ ] Dashboard displays context layers

### Phase 5
- [ ] Benchmarks meet targets
- [ ] Memory efficient (<2 GB)
- [ ] Scalable to 15+ sites

### Phase 6
- [ ] Code merged to main
- [ ] CI/CD passing
- [ ] Ready for production

---

## Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| 1: Setup | - | ✅ Complete |
| 2: Install | 15 min | Pending |
| 3: Pilot Test | 1-2 hours | Pending |
| 4: Integration | 2-4 hours | Pending |
| 5: Benchmarking | 1 hour | Pending |
| 6: Deployment | 1-2 hours | Pending |
| **TOTAL** | **~8-12 hours** | - |

---

## Questions / Support

See `docs/DGT_SENTINEL_INTEGRATION.md` for detailed troubleshooting.
Contact maintainer for issues or feature requests.

---

**Last Updated:** June 5, 2024  
**Status:** Ready for Phase 2 (Installation)
