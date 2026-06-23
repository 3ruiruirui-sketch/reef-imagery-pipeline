#!/usr/bin/env python3
"""
06_assemble_mask.py — Reconstituição do mosaico de máscaras de recife.

Lê as máscaras de probabilidade por tile (produzidas por dgt_inference_unet.py),
reconstitui o mosaico de dimensão original com weighted blend nas zonas de overlap,
e exporta o resultado final em dois formatos:

    {output_dir}/recife_mask.tif   — raster binário EPSG:3763 (uint8, 0/1)
    {output_dir}/recife_mask.gpkg  — polígonos vectorizados (requer gdal/fiona)

Estratégia de blend nas zonas de overlap (tile_overlap=32 px):
    Cada pixel é a média das probabilidades de todos os tiles que o cobrem.
    Acumulador de soma + acumulador de contagem → divide no final.
    Threshold 0.5 sobre a probabilidade média → máscara binária.

Input:
    {output_dir}/tiles_index.json   (gerado por 05_generate_tiles.py)
    {output_dir}/masks/tile_*.tif   (gerado por dgt_inference_unet.py)
Output:
    {output_dir}/recife_mask.tif
    {output_dir}/recife_mask.gpkg   (opcional — requer fiona + shapely)

Corre no DGT JupyterHub (96 cores AMD EPYC-Milan, 503 GB RAM, CPU-only).
Um mosaico Algarve típico ≈ 10 000 × 20 000 px float32 ≈ 800 MB — ok com 503 GB RAM.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('reef_pipeline.log', mode='a')],
)
logger = logging.getLogger(__name__)

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    logger.error("rasterio não instalado")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    tqdm = lambda x, **kw: x  # type: ignore[assignment]
    HAS_TQDM = False


DEFAULT_OUTPUT_DIR = Path('/home/jovyan/reef_output')
REEF_THRESHOLD     = 0.5


def assemble_mask(
    output_dir: Path  = DEFAULT_OUTPUT_DIR,
    threshold: float  = REEF_THRESHOLD,
    vectorize: bool   = True,
) -> dict:
    """
    Reconstitui o mosaico de máscaras a partir dos tiles de probabilidade.

    Args:
        output_dir: directório com tiles_index.json e masks/
        threshold:  limiar de probabilidade para máscara binária
        vectorize:  se True, tenta vectorizar para GeoPackage

    Returns:
        dict com reef_fraction, output_tif, output_gpkg (se gerado)
    """
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio é obrigatório")

    index_path = output_dir / 'tiles_index.json'
    if not index_path.exists():
        raise FileNotFoundError(f"Índice não encontrado: {index_path}")

    with open(str(index_path)) as f:
        idx = json.load(f)

    raster_h  = idx['raster_height']
    raster_w  = idx['raster_width']
    entries   = idx['tiles']
    masks_dir  = output_dir / 'masks'
    mosaic_path = Path(idx['mosaic'])

    logger.info("Reconstituindo mosaico: %d×%d px  com %d tiles",
                raster_w, raster_h, len(entries))

    # Acumuladores — cabem em RAM (34k×36k × 4 B ≈ 5 GB para prob_sum)
    # float32 é suficiente: max 4 tiles em overlap → valor máximo 4.0, dentro da precisão float32
    prob_sum   = np.zeros((raster_h, raster_w), dtype=np.float32)
    count_map  = np.zeros((raster_h, raster_w), dtype=np.uint8)

    missing = 0
    pbar    = tqdm(entries, desc="Montagem", unit="tile")
    for entry in pbar:
        mask_path = masks_dir / entry['filename']
        if not mask_path.exists():
            missing += 1
            continue
        try:
            with rasterio.open(str(mask_path)) as src:
                prob = src.read(1).astype(np.float32)
        except Exception as exc:
            logger.warning("Falha ao ler %s: %s", entry['filename'], exc)
            missing += 1
            continue

        row_off = entry['row_off']
        col_off = entry['col_off']
        win_h   = entry['win_h']
        win_w   = entry['win_w']

        # Clip para área válida do tile (pode ser menor que tile_size nas bordas)
        prob_clip = prob[:win_h, :win_w]

        row_end = min(row_off + win_h, raster_h)
        col_end = min(col_off + win_w, raster_w)
        actual_h = row_end - row_off
        actual_w = col_end - col_off

        prob_sum[row_off:row_end, col_off:col_end] += prob_clip[:actual_h, :actual_w]
        count_map[row_off:row_end, col_off:col_end] += 1

    if missing > 0:
        logger.warning("%d tiles sem máscara correspondente (descartados)", missing)

    # Normalizar: probabilidade média nos overlaps
    valid = count_map > 0
    mean_prob = np.full((raster_h, raster_w), fill_value=-1.0, dtype=np.float32)
    mean_prob[valid] = prob_sum[valid] / count_map[valid]

    # Máscara binária só onde há cobertura — pixeis sem tiles ficam como 0 (não-recife)
    mask_binary = np.where(valid, (mean_prob >= threshold), 0).astype(np.uint8)
    reef_frac   = float(mask_binary.mean())
    reef_px     = int(mask_binary.sum())
    logger.info("Fracção de recife: %.4f  (%d pixels de %d)",
                reef_frac, reef_px, raster_h * raster_w)

    # Obter profile geoespacial do mosaico original
    with rasterio.open(str(mosaic_path)) as src:
        mosaic_profile = src.profile.copy()

    out_tif = output_dir / 'recife_mask.tif'
    out_profile = mosaic_profile.copy()
    out_profile.update(dtype='uint8', count=1, nodata=255, compress='deflate',
                       predictor=2, tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(str(out_tif), 'w', **out_profile) as dst:
        dst.write(mask_binary, 1)
        dst.update_tags(
            reef_fraction=str(round(reef_frac, 6)),
            reef_pixels=str(reef_px),
            threshold=str(threshold),
            source='ReefUNet v3 — DGT MDT-50cm',
        )
    logger.info("Máscara raster exportada: %s", out_tif)

    # Probabilidade contínua (para análise pós-processamento)
    out_prob_tif = output_dir / 'recife_prob.tif'
    prob_profile = mosaic_profile.copy()
    prob_profile.update(dtype='float32', count=1, nodata=-1.0, compress='deflate')
    with rasterio.open(str(out_prob_tif), 'w', **prob_profile) as dst:
        dst.write(mean_prob, 1)
    logger.info("Probabilidades exportadas: %s", out_prob_tif)

    result = {
        'output_tif':    str(out_tif),
        'output_prob':   str(out_prob_tif),
        'reef_fraction': round(reef_frac, 6),
        'reef_pixels':   reef_px,
        'raster_h':      raster_h,
        'raster_w':      raster_w,
        'threshold':     threshold,
        'tiles_missing': missing,
    }

    # Vectorização opcional — requer fiona + shapely (podem não estar no HPC)
    out_gpkg = output_dir / 'recife_mask.gpkg'
    if vectorize:
        try:
            from rasterio.features import shapes
            import fiona
            from shapely.geometry import shape, mapping

            polys = []
            for geom, val in shapes(mask_binary, transform=mosaic_profile['transform']):
                if val == 1:
                    polys.append(shape(geom))

            if polys:
                schema = {'geometry': 'Polygon', 'properties': {'id': 'int'}}
                with fiona.open(str(out_gpkg), 'w', driver='GPKG', crs=str(mosaic_profile['crs']),
                                schema=schema) as dst:
                    for i, poly in enumerate(polys):
                        dst.write({'geometry': mapping(poly), 'properties': {'id': i}})
                logger.info("Vectorizado: %d polígonos → %s", len(polys), out_gpkg)
                result['output_gpkg'] = str(out_gpkg)
            else:
                logger.warning("Nenhum pixel de recife detectado — GeoPackage vazio")
        except ImportError as exc:
            logger.info("Vectorização ignorada (fiona/shapely não disponíveis): %s", exc)
        except Exception as exc:
            logger.warning("Falha na vectorização: %s", exc)

    logger.info("Pipeline concluída — validar com: 02_reef_detection_hpc.ipynb")
    return result


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Reconstitui mosaico de máscaras de recife')
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--threshold',  type=float, default=REEF_THRESHOLD)
    parser.add_argument('--no-vectorize', action='store_true',
                        help='Salta vectorização para GeoPackage')
    args = parser.parse_args()

    result = assemble_mask(
        output_dir = Path(args.output_dir),
        threshold  = args.threshold,
        vectorize  = not args.no_vectorize,
    )
    logger.info("Resultado: %s", result)
    sys.exit(0)
