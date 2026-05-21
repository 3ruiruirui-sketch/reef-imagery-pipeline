# Hardened Reef Imagery Pipeline v3

A secure, production-grade local Python pipeline for the Albufeira Reef imagery acquisition and analysis. 

This version is hardened against credential leaks, handles CDSE 2FA/TOTP token generation safely, defaults to a free WMS background layer for the DGT orthophoto in QGIS, and automatically falls back to Planetary Computer STAC if CDSE access fails.

---

## 🗺️ Multi-Source Orthophoto Tracks

The v3 pipeline features a unified multi-source orthophoto integration comprising three distinct tracks:

| Track | Details | Resolution | Active in QGIS? |
|---|---|---|---|
| **1. OrtoSat2023 (Pléiades Neo)** | **Primary track**. High-resolution 2023 imagery. Supports True-Color (`CorVerdadeira`) and False-Color NIR (`FalsaCor`). | 30 cm | **Yes** (WMS layers loaded by default; local clip loaded if detected) |
| **2. DGT Ortho 2018/2021** | Traditional aerial campaign. High baseline resolution, great for long-term comparative analysis. | 25 cm | **Yes** (WMS background loaded by default; WCS local clip loaded if `--enable-dgt-download` is forced) |

---

## 🔒 Hardened Security Features

1. **Token-First Workflow**: Pass your bearer token using `CDSE_ACCESS_TOKEN` in your environment or `--cdse-token`. This prevents plaintext passwords from leaking into shell history.
2. **Interactive 2FA/TOTP**: If generating a token via the script, use the optional `--cdse-totp` flag to submit 2FA verification codes securely.
3. **No Stored Plaintext Passwords**: Password fields are completely stripped from `.env.save` writes by default.
4. **Conditional DGT WCS Downloads**: WCS downloads require specialized institutional clearance (frequently returning 400/403 errors). The pipeline skips WCS downloads by default, immediately adding a functional **DGT WMS Background** layer to the QGIS template for zero-friction visualization.

---

## 🚀 Setup & Installation

```bash
pip install -r requirements_v3.txt
```

---

## 📖 Recommended Workflow

### 📡 Activating the 30 cm OrtoSat2023 False-Color Local Analysis
Since DGT's SharePoint download link is gated behind active session cookies:
1. Copy and open this link in your authenticated browser:
   [DGT OrtoSat2023 SharePoint Download](https://dgterritorio.sharepoint.com/sites/EXT-ORTOSAT2023/_layouts/15/download.aspx?SourceUrl=/sites/EXT-ORTOSAT2023/Documentos%20Partilhados/2_OrtoSat2023_FalsaCor/1_Seccoes_OrtoSat2023_FalsaCor/Seccoes_4800/OrtoSat2023_4824_FalsaCor.tif)
2. Save the download as `OrtoSat2023_4824_FalsaCor.tif` inside your configured output directory (e.g. `reef_output_v3`).
3. Re-run:
   ```bash
   python reef_imagery_pipeline_v3.py --step ortho
   ```
   *The pipeline will automatically detect the local file, clip a high-precision subset matching your exact reef buffer window, and load the clipped high-res raster into your QGIS project.*

---

### Option A: Clean Token Workflow (Recommended)
First, generate your Copernicus token:
```bash
export CDSE_ACCESS_TOKEN=$(curl -s -X POST \
  "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "client_id=cdse-public" \
  -d "username=stash.forked.9k@icloud.com" \
  -d "password=YOUR_REAL_PASSWORD" \
  -d "grant_type=password" | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
```

Then, run the full pipeline in one command:
```bash
python reef_imagery_pipeline_v3.py --step all
```

---

### Option B: Interactive Credentials Mode
If you prefer the script to request the token on your behalf, provide credentials directly:
```bash
python reef_imagery_pipeline_v3.py --step all \
  --cdse-user stash.forked.9k@icloud.com \
  --cdse-pass YOUR_REAL_PASSWORD
```

*If 2FA is active on your CDSE account, append the TOTP code:*
```bash
python reef_imagery_pipeline_v3.py --step all \
  --cdse-user stash.forked.9k@icloud.com \
  --cdse-pass YOUR_REAL_PASSWORD \
  --cdse-totp 123456
```

---

### Option C: Planetary Computer Fallback (Zero Setup)
If you do not specify any token or credentials, the pipeline gracefully bypasses CDSE and downloads the correct bands from the Planetary Computer STAC automatically:
```bash
python reef_imagery_pipeline_v3.py --step all
```

---

## 🛠️ CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--step` | *(required)* | `capabilities`, `ortho`, `sentinel`, `ratio`, `qgis`, `gee`, `all` |
| `--date` | `2024-10-15` | Target Sentinel-2 scene date (OP20 default) |
| `--lat` | `37.069071` | Target latitude |
| `--lon` | `-8.210492` | Target longitude |
| `--buffer-m` | `500` | Search/Clip buffer in meters |
| `--coverage` | `Ortos2018-RGB` | DGT CoverageId |
| `--output-dir` | `reef_output_v3` | Destination folder |
| `--cdse-token` | `""` | CDSE Bearer Token (overrides environment variable) |
| `--cdse-user` | `""` | CDSE username/email |
| `--cdse-pass` | `""` | CDSE password |
| `--cdse-totp` | `""` | optional 2FA/TOTP validation code |
| `--enable-dgt-download` | `False` | Force attempts to download DGT WCS orthophotos locally |

---

## 🗺️ Output Artifacts

All results will be saved in `--output-dir` (default: `reef_output_v3`):
* `S2_B02_YYYYMMDD.tif` / `S2_B03_YYYYMMDD.tif`: Downloaded bands (windowed clip).
* `ratio_B02_B03_YYYYMMDD.tif`: Computed contrast ratio raster.
* `ratio_analysis_YYYYMMDD.png`: Diagnostic plot (Blue band, Green band, and ratio).
* `OrtoSat2023_4824_FalsaCor_reef_clip.tif`: Autogenerated high-res 30 cm local clip (if source tile is provided).
* `reef_project_YYYYMMDD.qgs`: A pre-configured QGIS project. 
* `ratio_style.qml`: Pre-defined style layout (RdYlBu 0.8 to 1.2 ramp).
* `gee_reef_export_YYYYMMDD.js`: Earth Engine fallback code.
* `pipeline.log`: Full operation audit log.
