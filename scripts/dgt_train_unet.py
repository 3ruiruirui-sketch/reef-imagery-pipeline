#!/usr/bin/env python3
"""
DGT Server U-Net Training Script
Runs on AMD EPYC-Milan (64 cores, 125 GB RAM)
"""
import os, sys, shutil, json, time
from glob import glob
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, str(Path('/home/jovyan/reef_ml')))
os.chdir('/home/jovyan/reef_ml')

from src.ml_unet_dataset import ReefPatchDataset, get_training_augmentation, get_validation_augmentation
from src.ml_unet_model import ReefUNet, get_loss_function, calculate_iou

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_DIR = '/home/jovyan/reef_ml/data/master_ml_dataset'
MODELS_DIR  = '/home/jovyan/reef_ml/models'
EPOCHS      = 50      # More epochs on powerful server
BATCH_SIZE  = 32      # Large batch (125 GB RAM)
NUM_WORKERS = 16      # Use 16 of 64 cores
LR          = 1e-3
SEED        = 42
LOG_FILE    = '/home/jovyan/reef_ml/training_log.json'
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(MODELS_DIR, exist_ok=True)
torch.manual_seed(SEED)

print("=" * 60)
print("REEF U-NET TRAINING  —  DGT AMD EPYC-Milan Server")
print(f"Cores : {os.cpu_count()}    PyTorch : {torch.__version__}")
print(f"Device: CPU (no GPU detected)")
print("=" * 60)

# ── DATASET ───────────────────────────────────────────────────────────────────
full_dataset = ReefPatchDataset(
    images_dir=f'{DATASET_DIR}/images',
    masks_dir=f'{DATASET_DIR}/masks',
    transform=get_training_augmentation()
)
n_total = len(full_dataset)
n_train = int(0.70 * n_total)
n_val   = int(0.15 * n_total)
n_test  = n_total - n_train - n_val

print(f"\nDataset: {n_total} patches → {n_train} train / {n_val} val / {n_test} test")

train_ds, val_ds, test_ds = random_split(
    full_dataset, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(SEED)
)
val_ds.dataset.transform  = get_validation_augmentation()
test_ds.dataset.transform = get_validation_augmentation()

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# ── MODEL ─────────────────────────────────────────────────────────────────────
device    = torch.device('cpu')
model     = ReefUNet(encoder_name="resnet18").to(device)
criterion = get_loss_function()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# ── TRAINING LOOP ─────────────────────────────────────────────────────────────
best_iou  = 0.0
history   = []
t0 = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        out  = model(images)
        loss = criterion(out, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    val_iou  = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            out       = model(images)
            val_loss += criterion(out, masks).item()
            val_iou  += calculate_iou(out, masks)

    n_val_batches = max(len(val_loader), 1)
    avg_train = train_loss / max(len(train_loader), 1)
    avg_val   = val_loss   / n_val_batches
    avg_iou   = val_iou    / n_val_batches

    scheduler.step(avg_val)

    saved = ""
    if avg_iou > best_iou:
        best_iou = avg_iou
        torch.save(model.state_dict(), f'{MODELS_DIR}/unet_reef_best.pth')
        saved = "  ← BEST"

    row = dict(epoch=epoch, train_loss=round(avg_train, 4),
               val_loss=round(avg_val, 4), val_iou=round(avg_iou, 4),
               best_iou=round(best_iou, 4), elapsed=round(time.time()-t0, 1))
    history.append(row)
    with open(LOG_FILE, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"Epoch {epoch:>2}/{EPOCHS}  train={avg_train:.4f}  val={avg_val:.4f}  IoU={avg_iou:.4f}{saved}")

# ── FINAL TEST ────────────────────────────────────────────────────────────────
print("\nEvaluating on held-out test set...")
model.load_state_dict(torch.load(f'{MODELS_DIR}/unet_reef_best.pth', map_location=device))
model.eval()
test_iou = 0.0
with torch.no_grad():
    for images, masks in test_loader:
        images, masks = images.to(device), masks.to(device)
        out       = model(images)
        test_iou += calculate_iou(out, masks)
test_iou /= max(len(test_loader), 1)

print(f"\n{'='*60}")
print(f"TRAINING COMPLETE")
print(f"Best Validation IoU : {best_iou:.4f}")
print(f"Test IoU            : {test_iou:.4f}")
print(f"Total time          : {(time.time()-t0)/60:.1f} min")
print(f"Model saved to      : {MODELS_DIR}/unet_reef_best.pth")
print(f"Log saved to        : {LOG_FILE}")
print(f"{'='*60}")
