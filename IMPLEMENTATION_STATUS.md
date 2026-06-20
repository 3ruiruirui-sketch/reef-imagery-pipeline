# O que foi implementado

## Itens novos (não estavam no codebase)
| Ficheiro | O que faz |
|---|---|
| `src/pipeline_config.py` | PipelineConfig canónico — paths, S3 endpoints, CRS, paralelismo. Properties derivadas para todos os paths. Credenciais via os.environ (nunca hardcoded). |
| `scripts/lidar/01_stac_query.py` | Consulta STAC DGT, salva `lidar_manifest.json`. Resumível (skip se já existe). Usa PipelineConfig. |
| `scripts/lidar/02_download_s3.py` | Download paralelo S3 com `ThreadPoolExecutor(32)`. Skip se ficheiro já existe. Credenciais via env vars. |
| `scripts/lidar/03_process_laz.py` | PDAL via subprocess, mantém Classes 2+9, `multiprocessing.Pool(64)`. Timeout de 300 s por ficheiro. |
| `scripts/lidar/04_merge_mdt.py` | `rasterio.merge` + reprojecção para EPSG:3763. Output com DEFLATE + tiles 256×256. Resumível. |
| `notebooks/01_UNet_Reef_Trainer.ipynb` | Arquitectura, sanity check, visualização do desequilíbrio de classes e curvas de treino. |

## Correcções ao script de treino
| Item da checklist | Antes | Depois |
|---|---|---|
| `num_workers = min(64, os.cpu_count())` | `NUM_WORKERS = 4` (fixo) | `min(os.cpu_count(), 16)` via env var `DL_NUM_WORKERS` |
| Checkpoints com timestamp cada 5 epochs | só `checkpoint_latest.pth` | + `checkpoint_epoch_005.pth`, `010.pth`, ... |

## Pipeline completa agora rastreada em git
1. `01_stac_query.py` → `lidar_manifest.json` ✅ local + DGT
2. `02_download_s3.py` → `raw/*.laz` / `*.tif` ✅ local + DGT
3. `03_process_laz.py` → `dem/*.tif` (PDAL classes 2+9) ✅ local + DGT
4. `04_merge_mdt.py` → `mosaico_algarve.tif` ✅ local + DGT
5. `05_generate_tiles.py` → `tiles/tile_*.tif` ✅ local
6. `dgt_inference_unet.py` → `masks/tile_*.tif` ✅ local
7. `06_assemble_mask.py` → `recife_mask.tif` + `.gpkg` ✅ local
8. `02_reef_detection_hpc.ipynb` → validação folium ✅ local
