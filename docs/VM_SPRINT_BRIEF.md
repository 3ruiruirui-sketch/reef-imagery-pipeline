# DGT JupyterHub Sprint Brief — credentialed / HPC work

> Scope: only the steps a remote/offline agent **cannot** do, because they need your
> CDSE OAuth, AWS S3 keys, or the 96-core VM. Everything code-level (the 4 fixes below)
> was already done on branch `fix/four-tasks-offline`. Run *this* file on JupyterHub.

## 0. Corrections to the earlier "playbook" (verified against the real code)

Do **not** waste sprint time on these — they were wrong:

- **".env is committed → `git rm --cached .env`"** — false. `.env` is gitignored and not in
  history (`git log --all -- .env` is empty). Nothing to purge. *(Still rotate the tokens you
  pasted into chat — hygiene, not an exposure.)*
- **"Env-var mismatch kills the LiDAR track silently"** — false. `src/pipeline_config.py:152`
  already does `os.environ.get("AWS_ACCESS_KEY_HPC_ID1") or os.environ.get("AWS_ACCESS_KEY_ID1","")`
  with a clear error if neither is set. No rename needed.

Real issues, now fixed in code (verify the branch, then this brief is unblocked):

- RSS units bug in `benchmark_cpu.py` (656 GB was a bytes↔KB confusion, not a leak) — fixed.
- Drift webhook existed but was never called — now env-gated via `DRIFT_WEBHOOK_URL`, failure-safe.
- No ICESat-2 RMSE assertion — added (offline, synthetic).
- `scripts/threshold_sweep.py` added (fixture/`--demo` mode).

---

## Pre-flight (Day 1, ~30 min)

```bash
# On JupyterHub terminal
cd /home/jovyan/reef-imagery-pipeline
git fetch && git checkout fix/four-tasks-offline   # review + merge the code fixes first
rm -f .git/index.lock                              # clear the stale lock the agent hit
pytest tests/ -m "not network" -q                  # baseline; expect green except optional-dep skips
```

1. **Rotate `JUPYTERHUB_API_TOKEN` and `JHUB_TOKEN`** in the JupyterHub UI + your `.env`. They were
   shared in cleartext; treat as burned.
2. Confirm S3 reachability without downloading 500 GB:
   ```bash
   python -c "from src.pipeline_config import PipelineConfig as C; print(C().s3_client_stor001().list_buckets()['Buckets'][:3])"
   ```
   **Pass:** prints buckets. **Fail:** OSError → fix creds before anything else.

---

## Track A — Bathymetry / BVI (credentialed)

### A1. CDSE batch download (the real bottleneck = quota, not compute)
**Why VM:** needs CDSE OAuth; quota ~30k PU/month ⇒ budget it.
Build `notebooks/02_cdse_batch_download.ipynb` that walks the 8 reef sites for the 2024 + 2025
Sep/Oct cloud-free windows via `src/sh_downloader.py::download_patch(..., endpoint="cdse")`.
- Cap to top-3 cloud-free scenes/site/year (24 sites-scenes/year ≈ within quota).
- **Pass/fail:** a manifest CSV lists ≥1 scene per site with cloud% < 10; total PU spent logged < monthly cap.

### A2. Multi-scene Stumpf fusion per site
`src/stumpf_multiscene.py::fuse_scenes` is already wired into `orchestrator_run.main()`.
Run end-to-end per site over the A1 scenes.
- **Pass/fail:** one fused depth GeoTIFF per site; pixel coverage > single-scene baseline; no NaN holes
  in the 0–20 m domain > X%.

### A3. ICESat-2 ground-truth validation (the missing audit piece)
Use the **real** `run_icesat2_validation` against `outputs/icesat2_deep_survey/atl03_all_photons.json`
per reef polygon → `validation_summary.csv` with `rmse_m, bias_m, pearson_r, n_points` per scene.
The offline test added on the branch proves the plumbing; here you run it for real.
- **Pass/fail:** CSV has a row per site; assert **RMSE < 2 m** (or your thesis-defensible bound) and
  flag sites that miss it. This is your Chapter 5 validation table.

### A4. CMEMS Kd490 weekly refresh
`orchestrator_run._activate_cmems_live()` already refreshes on start. Schedule a weekly Jupyter/cron
job that refreshes `KD490_TABLE_LIVE` and writes the payload to `drift_reports/`.
- **Pass/fail:** a dated JSON appears weekly; orchestrator picks up live values (log line confirms).

---

## Track B — LiDAR → UNet → reef mask (this is what the 96 cores are for)

### B1. Tile generation at scale
`scripts/lidar/05_generate_tiles.py` is the I/O bottleneck. Profile on the real
`mosaico_algarve.tif` with 32+ workers; compare wall-time to laptop.
- **Pass/fail:** tiles/sec recorded; chain remains resumable (re-run skips existing tiles).

### B2. UNet training (set `DL_NUM_WORKERS`, batch 32)
Run `notebooks/01_UNet_Reef_Trainer.ipynb`: `batch_size=32`, `num_workers` via
`DL_NUM_WORKERS` (capped at 16 by design), checkpoint every 5 epochs.
- **Pass/fail:** loss curve trends down; `models/unet_reef_best.pth` updated; per-epoch < 300 s
  target from `BENCHMARKS.md` met or the regression explained.

### B3. UNet inference fan-out
`scripts/dgt_inference_unet.py` over `outputs/tiles/` with `multiprocessing.Pool` batches of 32.
- **Pass/fail:** throughput ~5–10k tiles/min on 96 cores; `outputs/masks/tile_*.tif` complete.

### B4. Threshold sweep on REAL validation tiles
The branch added `scripts/threshold_sweep.py` (verified in `--demo`). Point it at real
pred/mask pairs from B3 to pick the operating threshold (vs the static `PipelineConfig.threshold`).
- **Pass/fail:** precision/recall CSV + PNG over 0.3–0.7; chosen threshold beats the default's F1.

### B5. Assemble + paper trail
`06_assemble_mask.py` → `recife_mask.tif`/`.gpkg`. Wire `scripts/watchdog.py` to auto-commit new
masks to a `results-YYYY-MM-DD` branch (keeps `main` clean).
- **Pass/fail:** mask + gpkg produced; auto-commit lands on a dated branch.

---

## Track C — Ops / thesis artifacts

- **C1. Drift webhook live:** export `DRIFT_WEBHOOK_URL=<slack/webhook>` on the VM so the now-wired
  call actually fires. **Pass:** a drift POST shows up after an orchestrator run.
- **C2. Drift dashboard panel:** `notebooks/06_drift_dashboard.ipynb` reading `drift_reports/*.json`
  + `outputs/coastal_topography/algarve_coastal_features.geojson` → 30-day timeline.
- **C3. Reproducibility:** rerun A2 + B3 end-to-end on a fresh Apptainer snapshot; archive the
  command log. This is your "methods are reproducible" claim for the viva.

---

## Suggested order (by ROI, not calendar)

1. Pre-flight + merge code branch + rotate tokens.
2. **A1 CDSE download** (quota-bound — start it early, let it run).
3. **A3 ICESat-2 validation** (cheap, unblocks your validation chapter).
4. **B2/B3 UNet train+infer** (the only thing that genuinely needs the VM).
5. B4 threshold sweep → B5 mask → C drift/repro while you write.

> Everything else (BVI scoring, dashboard tweaks, drift file export) runs fine on your laptop
> against `reef_Output_Master/` fixtures — don't burn VM hours on it.
