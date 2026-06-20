"""
pipeline_config.py — Configuração canónica do Reef Imagery Pipeline.

Ponto único de configuração para todos os scripts da pipeline.
NUNCA hardcodar paths, endpoints S3 ou credenciais fora desta classe.

Uso:
    from src.pipeline_config import PipelineConfig
    cfg = PipelineConfig()
    s3  = cfg.s3_client_stor001()
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('reef_pipeline.log', mode='a'),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """
    Configuração central do Reef Imagery Pipeline — DGT HPC JupyterHub.

    Infraestrutura:
        CPU:  2x AMD EPYC 7643 = 96 cores (sem restrições SLURM)
        RAM:  503 GB (~485 GB livres)
        GPU:  NENHUMA — PyTorch corre estritamente em CPU
        ENV:  Contentor Apptainer (credenciais S3 injetadas como env vars)
        WD:   /home/jovyan/

    S3 MinIO DGT:
        stor-001 → bucket lidar     (nuvens de pontos LAZ)
        stor-002 → bucket mdt-50cm  (MDT GeoTIFF 50 cm + ortofotos)

    Regra crítica: ListObjects BLOQUEADO → usar exclusivamente STAC API.
    """

    # Diretórios principais
    output_dir:  Path = field(default_factory=lambda: Path('/home/jovyan/reef_output'))
    aoi_geojson: Path = field(default_factory=lambda: Path('/home/jovyan/reef_output/aoi_algarve.geojson'))

    # STAC API DGT
    stac_url: str = 'https://dgt-be.a.incd.pt:8081/stac'

    # S3 buckets
    lidar_bucket: str = 'lidar'
    mdt_bucket:   str = 'mdt-50cm'

    # S3 endpoints (internos ao Apptainer)
    s3_endpoint_stor001: str = 'http://stor-001.internal.dgt.pt'
    s3_endpoint_stor002: str = 'http://stor-002.internal.dgt.pt'

    # AOI Algarve (Loulé + Albufeira) — EPSG:4326 para STAC
    aoi_bbox: tuple = (-8.25, 37.00, -8.00, 37.15)

    # Paralelismo — explorar os 96 cores
    n_workers:    int = field(default_factory=lambda: min(64, os.cpu_count() or 1))
    n_io_threads: int = 32  # I/O-bound: downloads S3 (ThreadPoolExecutor)

    # Inferência ReefUNet
    device:       str   = 'cpu'   # NUNCA 'cuda' — sem GPU
    batch_size:   int   = 32      # tiles 256×256 em CPU ≈ 256 MB/batch
    tile_size:    int   = 256
    tile_overlap: int   = 32      # evita artefactos nas bordas
    threshold:    float = 0.5     # limiar sigmoid → máscara binária

    # CRS
    crs_work:  str = 'EPSG:3763'  # PT-TM06/ETRS89 — sistema oficial PT
    crs_query: str = 'EPSG:4326'  # WGS84 para queries STAC

    seed: int = 42

    def __post_init__(self) -> None:
        for sub in ('raw', 'dem', 'tiles', 'masks', 'checkpoints'):
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)

    # ── Paths derivados ───────────────────────────────────────────────────────

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / 'lidar_manifest.json'

    @property
    def mosaic_path(self) -> Path:
        return self.output_dir / 'mosaico_algarve.tif'

    @property
    def tiles_dir(self) -> Path:
        return self.output_dir / 'tiles'

    @property
    def masks_dir(self) -> Path:
        return self.output_dir / 'masks'

    @property
    def tiles_index_path(self) -> Path:
        return self.output_dir / 'tiles_index.json'

    @property
    def best_model_path(self) -> Path:
        """Pesos do modelo: repo local primeiro, fallback para output_dir."""
        local = Path('/home/jovyan/reef-imagery-pipeline/models/unet_reef_best.pth')
        return local if local.exists() else self.output_dir / 'unet_reef_best.pth'

    @property
    def reef_mask_path(self) -> Path:
        return self.output_dir / 'recife_mask.tif'

    @property
    def reef_prob_path(self) -> Path:
        return self.output_dir / 'recife_prob.tif'

    @property
    def reef_gpkg_path(self) -> Path:
        return self.output_dir / 'recife_mask.gpkg'

    @property
    def training_log_path(self) -> Path:
        return Path('/home/jovyan/reef-imagery-pipeline/training_log.json')

    @property
    def inference_log_path(self) -> Path:
        return self.output_dir / 'inference_log.json'

    @property
    def stride(self) -> int:
        return self.tile_size - self.tile_overlap

    # ── Credenciais S3 — injetadas pelo Apptainer, NUNCA hardcoded ───────────

    @property
    def s3_access_key(self) -> str:
        v = os.environ.get('AWS_ACCESS_KEY_HPC_ID1', '')
        if not v:
            raise EnvironmentError(
                "AWS_ACCESS_KEY_HPC_ID1 não definida.\n"
                "Esta variável é injetada automaticamente pelo Apptainer DGT.\n"
                "Fora do Apptainer: export AWS_ACCESS_KEY_HPC_ID1=<key>"
            )
        return v

    @property
    def s3_secret_key(self) -> str:
        v = os.environ.get('AWS_SECRET_KEY_HPC_ID1', '')
        if not v:
            raise EnvironmentError(
                "AWS_SECRET_KEY_HPC_ID1 não definida.\n"
                "Esta variável é injetada automaticamente pelo Apptainer DGT."
            )
        return v

    def s3_client_stor001(self):
        """boto3 client para stor-001 (lidar LAZ). Credenciais via env vars."""
        import boto3
        return boto3.client(
            's3',
            endpoint_url=self.s3_endpoint_stor001,
            aws_access_key_id=self.s3_access_key,
            aws_secret_access_key=self.s3_secret_key,
        )

    def s3_client_stor002(self):
        """boto3 client para stor-002 (mdt-50cm GeoTIFF). Credenciais via env vars."""
        import boto3
        return boto3.client(
            's3',
            endpoint_url=self.s3_endpoint_stor002,
            aws_access_key_id=self.s3_access_key,
            aws_secret_access_key=self.s3_secret_key,
        )
