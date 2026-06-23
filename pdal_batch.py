#!/usr/bin/env python3
"""
Processamento PDAL em batch de todos os LAZ DGT -> DEMs GeoTIFF.
Versao 2: trata ficheiros sem pontos de classe 2+9 com fallback.
Corre em background no DGT JupyterHub.
"""
import json
import multiprocessing
import os
import subprocess
import tempfile
import time
from pathlib import Path

LAZ_DIR = Path("/home/jovyan/reef_pipeline/data/lidar/laz")
DEM_DIR = Path("/home/jovyan/reef_output/dem")
LOG = Path("/home/jovyan/reef-imagery-pipeline/pdal_batch.log")

DEM_DIR.mkdir(parents=True, exist_ok=True)


def _run_pipeline(pipeline: dict, tmp_prefix: str, timeout: int = 600):
    """Correr um pipeline PDAL. Retorna (returncode, stderr)."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix=tmp_prefix, delete=False
    ) as tmp:
        json.dump(pipeline, tmp)
        tmp_path = tmp.name
    try:
        r = subprocess.run(
            ["pdal", "pipeline", tmp_path],
            capture_output=True,
            timeout=timeout,
        )
        return r.returncode, r.stderr.decode()[-300:]
    except subprocess.TimeoutExpired:
        return -1, "timeout"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def process_one(args):
    laz_path, dem_path = args

    # Saltar se DEM ja existe e e valido (>0 bytes)
    if dem_path.exists() and dem_path.stat().st_size > 0:
        return ("skip", laz_path.name, 0.0)

    # Remover DEM vazio anterior se existir
    if dem_path.exists():
        dem_path.unlink()

    t0 = time.time()

    # Tentativa 1: filtrar apenas classes 2 (terreno) e 9 (agua)
    pipeline_filtered = {
        "pipeline": [
            {"type": "readers.las", "filename": str(laz_path)},
            {
                "type": "filters.range",
                "limits": "Classification[2:2],Classification[9:9]",
            },
            {
                "type": "writers.gdal",
                "filename": str(dem_path),
                "gdaldriver": "GTiff",
                "output_type": "idw",
                "resolution": 0.5,
                "radius": 1.0,
                "data_type": "float32",
                "gdalopts": "COMPRESS=DEFLATE,TILED=YES,BLOCKXSIZE=256,BLOCKYSIZE=256",
            },
        ]
    }

    rc, stderr = _run_pipeline(pipeline_filtered, "pdal_filtered_")
    elapsed = time.time() - t0

    if rc == 0 and dem_path.exists() and dem_path.stat().st_size > 0:
        return ("ok", laz_path.name, elapsed)

    # Tentativa 2: sem filtro de classe (usa todos os pontos)
    # Util para ficheiros de zona costeira sem terreno nu classificado
    t1 = time.time()
    if dem_path.exists():
        dem_path.unlink()

    pipeline_all = {
        "pipeline": [
            {"type": "readers.las", "filename": str(laz_path)},
            {
                "type": "writers.gdal",
                "filename": str(dem_path),
                "gdaldriver": "GTiff",
                "output_type": "idw",
                "resolution": 0.5,
                "radius": 1.0,
                "data_type": "float32",
                "gdalopts": "COMPRESS=DEFLATE,TILED=YES,BLOCKXSIZE=256,BLOCKYSIZE=256",
            },
        ]
    }

    rc2, stderr2 = _run_pipeline(pipeline_all, "pdal_all_")
    elapsed2 = time.time() - t1

    if rc2 == 0 and dem_path.exists() and dem_path.stat().st_size > 0:
        return ("fallback", laz_path.name, elapsed + elapsed2)

    # Ambas as tentativas falharam
    if dem_path.exists():
        dem_path.unlink()
    err_msg = stderr2[-120:] if stderr2 else stderr[-120:]
    return ("err", laz_path.name + ": " + err_msg, elapsed + elapsed2)


if __name__ == "__main__":
    laz_files = sorted(LAZ_DIR.glob("*.laz"))
    args = [(f, DEM_DIR / (f.stem + ".tif")) for f in laz_files]

    # Contar pendentes
    pending = [(laz, dem) for laz, dem in args
               if not dem.exists() or dem.stat().st_size == 0]
    already_done = len(args) - len(pending)

    t0 = time.time()
    ok = skip = fallback = err = 0

    with open(str(LOG), "w") as log:
        log.write(
            f"PDAL batch v2: {len(laz_files)} LAZ files total "
            f"({already_done} ja completos, {len(pending)} pendentes), 32 workers\n"
        )
        log.flush()

        with multiprocessing.Pool(processes=32) as pool:
            for i, (status, msg, elapsed) in enumerate(
                pool.imap_unordered(process_one, pending), 1
            ):
                if status == "ok":
                    ok += 1
                elif status == "skip":
                    skip += 1
                elif status == "fallback":
                    fallback += 1
                else:
                    err += 1
                line = (
                    f"[{i:>3}/{len(pending)}] {status}: {msg}  ({elapsed:.0f}s)"
                )
                log.write(line + "\n")
                log.flush()

        total = time.time() - t0
        summary = (
            f"\nDone in {total / 60:.1f} min — "
            f"ok={ok} fallback={fallback} skip={skip} err={err} "
            f"total_dems={ok+fallback+already_done}/{len(laz_files)}\n"
        )
        log.write(summary)

    print(summary.strip())
