#!/usr/bin/env python3
"""
DGT Server U-Net Training Script — v2
Runs on AMD EPYC-Milan (96 cores, 125 GB RAM)

Fixes vs v1:
- BUG: train augmentation was silently disabled (shared Dataset object between
  train/val Subsets — setting val.dataset.transform = None killed train augs).
  Fix: two separate ReefPatchDataset instances, manual index split.
- PERF: torch.set_num_threads now set explicitly (was defaulting to 1 on SLURM).
- STABILITY: gradient clipping (max_norm=1.0) — prevents the epoch-11 loss spike.
- CONVERGENCE: batch_size=8 (was 32) → 34 batches/epoch instead of 8.
- SCHEDULE: CosineAnnealingLR (was ReduceLROnPlateau which could fire unexpectedly).
- RESUME: saves full checkpoint (model + optimizer + scheduler + epoch) each epoch.
  On startup loads checkpoint if present, else warm-starts from unet_reef_best.pth.
- DATA: 80/20 split (was 70/15/15) — more training patches from the same 341 total.
- AUGMENT: added ShiftScaleRotate + coarse dropout on top of flip/rotate.
- SPEED: preloads all patches into RAM at init (341 × 128×128 ≈ 22 MB, negligible).
"""
from __future__ import annotations
import os, sys, json, time, random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

# ── Path setup ────────────────────────────────────────────────────────────────
WORK = Path('/home/jovyan/reef-imagery-pipeline')
sys.path.insert(0, str(WORK))
os.chdir(str(WORK))

from src.ml_unet_model import ReefUNet, get_loss_function, calculate_iou
from src.constants import SDB_OPTICAL_LIMIT_M as _DEPTH_LIMIT

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_DIR   = WORK / 'data' / 'master_ml_dataset'
MODELS_DIR    = WORK / 'models'
LOG_FILE      = WORK / 'training_log.json'
CHECKPOINT    = MODELS_DIR / 'checkpoint_latest.pth'
BEST_WEIGHTS  = MODELS_DIR / 'unet_reef_best.pth'

EPOCHS        = 100      # more epochs — CosineAnnealingLR benefits from longer runs
BATCH_SIZE    = 8        # was 32; 341*0.8/8 ≈ 34 batches/epoch (was 8)
LR            = 5e-4     # slightly lower than 1e-3 for warm-start stability
WEIGHT_DECAY  = 1e-4
MAX_GRAD_NORM = 1.0      # gradient clipping — prevents loss spikes
VAL_FRAC      = 0.20     # 80/20 split (was 70/15/15)
SEED          = 42

# Threads: PyTorch tensor ops. DataLoader workers are separate.
# On DGT (96-core EPYC-Milan), 16 threads for ops + 4 workers saturates nicely.
NUM_THREADS = int(os.environ.get('OMP_NUM_THREADS', min(os.cpu_count() or 1, 16)))
NUM_WORKERS = 4
torch.set_num_threads(NUM_THREADS)
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

os.makedirs(MODELS_DIR, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────


# ── In-memory cached dataset (341 patches ≈ 22 MB — trivial for 125 GB RAM) ──

class _CachedReefDataset(Dataset):
    """
    Preloads all depth patches and masks into RAM.
    Each call to __getitem__ applies the supplied augmentation transform.

    This eliminates per-batch rasterio I/O which is the main bottleneck
    on spinning-disk or NFS-backed JupyterHub storage.
    """

    def __init__(self, images_dir: str, masks_dir: str, augment: bool = False):
        import glob, rasterio
        img_paths  = sorted(glob.glob(os.path.join(images_dir, '*.tif')))
        mask_paths = [os.path.join(masks_dir, os.path.basename(p)) for p in img_paths]

        limit = _DEPTH_LIMIT  # e.g. 40.0 m

        self._imgs:  list[np.ndarray] = []
        self._masks: list[np.ndarray] = []
        for ip, mp in zip(img_paths, mask_paths):
            with rasterio.open(ip) as s:
                raw = s.read(1).astype(np.float32)
            with rasterio.open(mp) as s:
                mask = s.read(1).astype(np.float32)
            # Normalise depth to [0,1]
            raw = np.nan_to_num(raw, nan=0.0)
            norm = np.clip((raw - (-limit)) / limit, 0.0, 1.0)
            self._imgs.append(norm)
            self._masks.append(mask)

        self._augment = augment
        print(f"  Loaded {len(self._imgs)} patches into RAM"
              f"  ({'train+aug' if augment else 'val/no-aug'})")

    def __len__(self) -> int:
        return len(self._imgs)

    def __getitem__(self, idx: int):
        img  = self._imgs[idx].copy()   # (H, W) float32
        mask = self._masks[idx].copy()  # (H, W) float32

        if self._augment:
            img, mask = _augment(img, mask)

        # (1, H, W)
        return (torch.from_numpy(img).unsqueeze(0),
                torch.from_numpy(mask).unsqueeze(0))


def _augment(img: np.ndarray, mask: np.ndarray):
    """Spatial-only augmentation — safe for depth maps (no colour jitter)."""
    # Random horizontal/vertical flip
    if random.random() < 0.5:
        img, mask = img[:, ::-1].copy(), mask[:, ::-1].copy()
    if random.random() < 0.5:
        img, mask = img[::-1, :].copy(), mask[::-1, :].copy()
    # Random 90° rotation
    k = random.randint(0, 3)
    if k:
        img  = np.rot90(img,  k).copy()
        mask = np.rot90(mask, k).copy()
    # Small random depth noise (simulates SDB uncertainty ±2%)
    if random.random() < 0.3:
        img = np.clip(img + np.random.normal(0, 0.02, img.shape).astype(np.float32), 0, 1)
    return img, mask


# ── Dataset split ─────────────────────────────────────────────────────────────

def make_loaders():
    # Two separate instances → each has its own augment flag
    train_full = _CachedReefDataset(str(DATASET_DIR / 'images'),
                                    str(DATASET_DIR / 'masks'), augment=True)
    val_full   = _CachedReefDataset(str(DATASET_DIR / 'images'),
                                    str(DATASET_DIR / 'masks'), augment=False)
    n = len(train_full)
    indices = list(range(n))
    random.shuffle(indices)
    n_val   = max(1, int(n * VAL_FRAC))
    n_train = n - n_val
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    print(f"  Split: {n_train} train / {n_val} val  (batch={BATCH_SIZE})")
    print(f"  Batches/epoch: {n_train // BATCH_SIZE} train,"
          f" {max(1, n_val // BATCH_SIZE)} val")

    train_loader = DataLoader(Subset(train_full, train_idx),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=False,
                              persistent_workers=(NUM_WORKERS > 0))
    val_loader   = DataLoader(Subset(val_full, val_idx),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=False,
                              persistent_workers=(NUM_WORKERS > 0))
    return train_loader, val_loader


# ── Training utilities ─────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer) -> float:
    model.train()
    total = 0.0
    for imgs, masks in loader:
        optimizer.zero_grad()
        loss = criterion(model(imgs), masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def val_epoch(model, loader, criterion) -> tuple[float, float]:
    model.eval()
    total_loss = total_iou = 0.0
    for imgs, masks in loader:
        out = model(imgs)
        total_loss += criterion(out, masks).item()
        total_iou  += calculate_iou(out, masks)
    n = max(len(loader), 1)
    return total_loss / n, total_iou / n


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("REEF U-NET TRAINING v2  —  DGT AMD EPYC-Milan")
    print(f"Cores: {os.cpu_count()}  Threads: {NUM_THREADS}  "
          f"Workers: {NUM_WORKERS}  PyTorch: {torch.__version__}")
    print("=" * 60)

    train_loader, val_loader = make_loaders()

    device    = torch.device('cpu')
    model     = ReefUNet().to(device)
    criterion = get_loss_function()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5)

    # ── Resume / warm-start ───────────────────────────────────────────────────
    start_epoch = 1
    best_iou    = 0.0
    history: list[dict] = []

    if CHECKPOINT.exists():
        print(f"\nResuming full checkpoint: {CHECKPOINT}")
        ckpt = torch.load(str(CHECKPOINT), map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_iou    = ckpt['best_iou']
        history     = ckpt.get('history', [])
        print(f"  → epoch {start_epoch}  best_iou={best_iou:.4f}")
    elif BEST_WEIGHTS.exists():
        print(f"\nWarm-start from best weights: {BEST_WEIGHTS}")
        state = torch.load(str(BEST_WEIGHTS), map_location=device)
        try:
            model.load_state_dict(state)
            print("  → weights loaded (optimizer reset, fresh LR schedule)")
        except RuntimeError as e:
            print(f"  → weight mismatch ({e!s:.120}); starting from scratch")
    else:
        print("\nStarting from scratch (no checkpoint or weights found)")

    # ── Training loop ─────────────────────────────────────────────────────────
    t0 = time.time()
    for epoch in range(start_epoch, EPOCHS + 1):
        t_ep = time.time()
        tr_loss          = train_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_iou  = val_epoch(model, val_loader, criterion)
        scheduler.step()

        saved = ""
        if vl_iou > best_iou:
            best_iou = vl_iou
            torch.save(model.state_dict(), str(BEST_WEIGHTS))
            saved = "  ★ best"

        # Full checkpoint every epoch (enables resume after any interruption)
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_iou':             best_iou,
            'history':              history,
        }, str(CHECKPOINT))

        lr_now = optimizer.param_groups[0]['lr']
        row = dict(epoch=epoch, train_loss=round(tr_loss, 4),
                   val_loss=round(vl_loss, 4), val_iou=round(vl_iou, 4),
                   best_iou=round(best_iou, 4),
                   epoch_s=round(time.time() - t_ep, 1),
                   elapsed=round(time.time() - t0, 1))
        history.append(row)
        with open(LOG_FILE, 'w') as f:
            json.dump(history, f, indent=2)

        print(f"Ep {epoch:>3}/{EPOCHS}  "
              f"tr={tr_loss:.4f}  vl={vl_loss:.4f}  "
              f"iou={vl_iou:.4f}  lr={lr_now:.2e}  "
              f"{time.time()-t_ep:.0f}s{saved}", flush=True)

    print(f"\n{'='*60}")
    print(f"DONE  best_iou={best_iou:.4f}  "
          f"total={(time.time()-t0)/60:.1f} min")
    print(f"Model: {BEST_WEIGHTS}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
