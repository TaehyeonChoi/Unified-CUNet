"""
KITTI Training Script for Unified-CUNet and Baseline Models
============================================================

Mirrors train.py exactly in structure:
  - Same build_model() factory          → same 6 model types
  - Same Trainer class                  → AdamW, cosine annealing, warmup,
                                          mixed precision, early stopping,
                                          gradient clipping, FPS protocol
  - Same loss functions                 → UnifiedCUNetLoss / HSPCLoss / SimpleLoss
  - Same checkpoint format              → loadable by evaluate.py / evaluate_kitti.py

Only difference from train.py: the dataset layer.
  ModelNet40Dataset  →  KITTIDataset (self-contained .bin reader + patch sampler)

KITTI dataset structure
-----------------------
    <kitti_dir>/
        training/
            velodyne/  *.bin   ← used for train + val
            calib/     *.txt
            image_2/   *.png
        testing/
            velodyne/  *.bin   ← used by evaluate_kitti.py only
            calib/     *.txt
            image_2/   *.png

Train / val split
-----------------
Frames from training/velodyne/ are randomly split:
  train : (1 - --val_fraction) fraction  (default 85 %)
  val   : --val_fraction fraction         (default 15 %)

Patch sampling (matches evaluate_kitti.py exactly)
--------------------------------------------------
1. Sample a random seed point in the LiDAR frame.
2. Take the k = num_points nearest neighbours.
3. Centre and scale to unit sphere.

KITTI-specific augmentation
----------------------------
Only Z-axis rotation is applied (vehicle heading), never X/Y flip — those
would produce physically impossible upside-down LiDAR scans.
  • Random yaw rotation  (full 360°)
  • Scale jitter         (×[0.9, 1.1])
  • Point-wise jitter    (σ = 0.01, clip at ±0.05)

Usage
------
# Train Unified-CUNet from scratch on KITTI
python train_kitti.py \\
    --model_type unified_cunet \\
    --kitti_dir ./data/kitti \\
    --compression_ratio 0.5 --upsample_ratio 4

# Train all six models sequentially (convenience wrapper)
python train_kitti.py \\
    --model_type all \\
    --kitti_dir ./data/kitti

# Resume a run
python train_kitti.py \\
    --model_type fps_punet \\
    --kitti_dir ./data/kitti \\
    --resume ./checkpoints_kitti/fps_punet_latest.pt

# Quick smoke-test (3 frames)
python train_kitti.py \\
    --model_type unified_cunet \\
    --kitti_dir ./data/kitti \\
    --max_frames 3 --epochs 2 --batch_size 2

Checkpoint naming (matches evaluate_kitti.py search order):
    <save_dir>/<model_type>_best.pt
    <save_dir>/<model_type>_latest.pt
"""

import os
import sys
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

# ── path setup (mirrors train.py) ────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unified_cunet import create_unified_cunet, create_unified_cunet_mlp
from models.baselines import (
    fps_subsample,
    PUNet, PUGACNet, ADCoSNet, DimensionFixedModel,
)
from models.losses import UnifiedCUNetLoss, SimpleLoss
from models.HSPC_Net import HSPCNet, HSPCLoss


# ─────────────────────────────────────────────────────────────────────────────
# KITTI dataset
# ─────────────────────────────────────────────────────────────────────────────

def _discover_train_bin_files(kitti_dir: str) -> list:
    """
    Return sorted list of .bin paths from training/velodyne/ only.

    Expected layout:
        <kitti_dir>/training/velodyne/*.bin   ← train + val frames (labelled)
        <kitti_dir>/testing/velodyne/*.bin    ← used by evaluate_kitti.py only

    Falls back to a recursive search if the standard layout is not found.
    """
    # Primary: KITTI Object Detection layout
    train_bins = sorted(glob.glob(
        os.path.join(kitti_dir, 'training', 'velodyne', '*.bin')))
    if train_bins:
        return train_bins, 'object'

    # Fallback A — KITTI Odometry (all sequences, flattened)
    odo = sorted(glob.glob(
        os.path.join(kitti_dir, 'sequences', '*', 'velodyne', '*.bin')))
    if odo:
        return odo, 'odometry'

    # Fallback B — KITTI Raw
    raw = sorted(glob.glob(
        os.path.join(kitti_dir, '*', '*', 'velodyne_points', 'data', '*.bin')))
    if raw:
        return raw, 'raw'

    # Fallback C — recursive
    fb = sorted(glob.glob(
        os.path.join(kitti_dir, '**', '*.bin'), recursive=True))
    return fb, 'fallback'


def _read_bin(path: str) -> np.ndarray:
    """Read KITTI .bin → [N, 3] XYZ float32 (intensity dropped)."""
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return pts[:, :3]


def _sample_patch(scene_pts: np.ndarray, patch_size: int,
                  rng: random.Random) -> np.ndarray:
    """
    Sample one patch of `patch_size` points from a LiDAR frame.
    Returns None if the frame has fewer points than needed.
    """
    N = len(scene_pts)
    if N < patch_size:
        return None
    idx  = rng.randrange(N)
    seed = scene_pts[idx]
    d    = ((scene_pts - seed) ** 2).sum(1)
    nn   = np.argpartition(d, patch_size)[:patch_size]
    patch = scene_pts[nn].copy()
    # Normalise to unit sphere
    patch -= patch.mean(0)
    scale  = np.linalg.norm(patch, axis=1).max() + 1e-8
    patch /= scale
    return patch.astype(np.float32)


def _augment_patch(patch: np.ndarray, rng: random.Random) -> np.ndarray:
    """
    KITTI-safe data augmentation.

    - Random yaw (Z-axis) rotation only — preserves physical LiDAR geometry.
      Never flip X or Y: upside-down point clouds are not meaningful for LiDAR.
    - Scale jitter:  ×Uniform[0.9, 1.1]
    - Point jitter:  N(0, 0.01), clipped at ±0.05
    """
    # Yaw rotation around Z-axis
    theta = rng.uniform(0, 2 * np.pi)
    c, s  = np.cos(theta), np.sin(theta)
    Rz    = np.array([[c, -s, 0],
                      [s,  c, 0],
                      [0,  0, 1]], dtype=np.float32)
    patch = patch @ Rz.T

    # Scale jitter
    scale = rng.uniform(0.9, 1.1)
    patch = patch * scale

    # Point jitter
    noise = np.clip(np.random.randn(*patch.shape).astype(np.float32) * 0.01,
                    -0.05, 0.05)
    patch = patch + noise

    return patch


class KITTIDataset(Dataset):
    """
    KITTI LiDAR point cloud dataset for training Unified-CUNet.

    Each __getitem__ returns (patch_tensor, 0) where:
      patch_tensor : torch.FloatTensor [3, num_points]  (BCN format, matches train.py)
      0            : dummy label (Trainer._prepare_batch ignores labels)

    Patches are pre-sampled at construction time from all frames in `bin_files`,
    then shuffled each epoch via DataLoader.  This is more memory-efficient than
    sampling on-the-fly for large KITTI sequences.

    Parameters
    ----------
    bin_files    : list of str — paths to .bin LiDAR frames
    num_points   : int         — points per patch
    patches_per_frame : int    — patches to sample per frame
    augment      : bool        — apply KITTI-safe augmentation
    seed         : int         — random seed for reproducibility
    """

    def __init__(self, bin_files: list, num_points: int = 2048,
                 patches_per_frame: int = 20, augment: bool = True,
                 seed: int = 42):
        super().__init__()
        self.num_points   = num_points
        self.augment      = augment
        self._rng_aug     = random.Random(seed + 1)   # separate rng for augmentation

        rng = random.Random(seed)
        self.patches = []   # list of np.ndarray [num_points, 3]

        print(f"  Loading {len(bin_files)} frame(s), "
              f"{patches_per_frame} patch(es) each ...")

        skipped = 0
        for path in bin_files:
            try:
                pts = _read_bin(path)
            except Exception as e:
                skipped += 1
                continue
            for _ in range(patches_per_frame):
                patch = _sample_patch(pts, num_points, rng)
                if patch is not None:
                    self.patches.append(patch)

        if skipped:
            print(f"  Warning: {skipped} frame(s) skipped (read errors).")
        print(f"  Total patches: {len(self.patches)}")

        if not self.patches:
            raise RuntimeError(
                "KITTIDataset: no patches were sampled.  "
                "Check --kitti_dir and --num_points.")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx].copy()   # [N, 3]

        if self.augment:
            patch = _augment_patch(patch, self._rng_aug)

        # [N, 3] → [3, N]  (BCN format — matches train.py / Trainer._prepare_batch)
        tensor = torch.from_numpy(patch.T)   # float32
        return tensor, 0                     # (data, dummy_label)


def create_kitti_dataloaders(kitti_dir: str,
                              num_points: int = 2048,
                              patches_per_frame: int = 20,
                              batch_size: int = 4,
                              num_workers: int = 4,
                              val_fraction: float = 0.15,
                              max_frames: int = None,
                              seed: int = 42):
    """
    Build train and val DataLoaders from KITTI training/velodyne/ frames.

    Split strategy
    --------------
    All frames in training/velodyne/ are randomly split:
      train : ceil((1 - val_fraction) * n_frames)
      val   : floor(val_fraction * n_frames)

    testing/velodyne/ is intentionally excluded — those frames are reserved
    for evaluate_kitti.py.

    Returns
    -------
    train_loader, val_loader
    """
    all_files, layout = _discover_train_bin_files(kitti_dir)
    if not all_files:
        raise FileNotFoundError(
            f"No KITTI .bin files found under '{kitti_dir}/training/velodyne/'.\n"
            f"Expected layout:\n"
            f"  {kitti_dir}/training/velodyne/*.bin\n"
            f"  {kitti_dir}/testing/velodyne/*.bin")

    print(f"\n  KITTI layout         : {layout}")
    print(f"  training/ frames     : {len(all_files)}")

    if max_frames:
        all_files = all_files[:max_frames]
        print(f"  Capped to            : {len(all_files)} frames (--max_frames)")

    # Random train / val split (deterministic via seed)
    rng_split = random.Random(seed)
    shuffled  = all_files[:]
    rng_split.shuffle(shuffled)

    n_val       = max(1, int(len(shuffled) * val_fraction))
    val_files   = shuffled[:n_val]
    train_files = shuffled[n_val:]

    print(f"  Train frames         : {len(train_files)}")
    print(f"  Val frames           : {len(val_files)}"
          f"  ({val_fraction:.0%} of total)")

    if not train_files:
        raise RuntimeError(
            "No training frames after split. Reduce --val_fraction or "
            "add more frames to training/velodyne/.")
    if not val_files:
        raise RuntimeError(
            "No validation frames after split. Increase --val_fraction or "
            "add more frames to training/velodyne/.")

    print("\n  Building train dataset...")
    train_ds = KITTIDataset(
        train_files, num_points=num_points,
        patches_per_frame=patches_per_frame,
        augment=True, seed=seed)

    print("\n  Building val dataset...")
    val_ds = KITTIDataset(
        val_files, num_points=num_points,
        patches_per_frame=patches_per_frame,
        augment=False, seed=seed + 99)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Model factory  (identical to train.py — only comments differ)
# ─────────────────────────────────────────────────────────────────────────────

FPS_MODEL_TYPES = {'fps_punet', 'fps_pugacnet', 'fps_adcosnet', 'fps_hspcnet'}

ALL_MODEL_TYPES = [
    'fps_adcosnet', 'fps_pugacnet', 'fps_punet',
    'fps_hspcnet', 'unified_cunet_mlp', 'unified_cunet',
]


def build_model(args):
    """
    Instantiate the correct model — verbatim copy of train.py's build_model().

    Returns
    -------
    model          : nn.Module
    fps_preprocess : bool
    fps_ratio      : float
    """
    mtype = args.model_type.lower()
    alpha = args.compression_ratio
    beta  = int(args.upsample_ratio)
    N     = args.num_points
    N_c   = max(32, int(N * alpha))

    if mtype == 'unified_cunet':
        print(f"\nCreating Unified-CUNet  (alpha={alpha}, beta=x{beta})")
        model = create_unified_cunet(
            compression_ratio=alpha, upsample_ratio=float(beta),
            hidden_dim=64, num_transformer_blocks=4)
        return model, False, alpha

    if mtype == 'unified_cunet_mlp':
        print(f"\nCreating Unified-CUNet-MLP  (alpha={alpha}, beta=x{beta})  [ablation]")
        model = create_unified_cunet_mlp(
            compression_ratio=alpha, upsample_ratio=float(beta),
            hidden_dim=64, num_transformer_blocks=4)
        return model, False, alpha

    if mtype == 'fps_punet':
        print(f"\nCreating FPS+PU-Net  (alpha={alpha}, beta=x{beta}, N_c={N_c})")
        model = DimensionFixedModel(
            PUNet(num_points=N_c, up_ratio=beta),
            required_input_format='BNC', name='PU-Net')
        return model, True, alpha

    if mtype == 'fps_pugacnet':
        print(f"\nCreating FPS+PU-GACNet  (alpha={alpha}, beta=x{beta}, N_c={N_c})")
        model = DimensionFixedModel(
            PUGACNet(in_channels=3, out_channels=3, up_factor=beta),
            required_input_format='BCN', name='PU-GACNet')
        return model, True, alpha

    if mtype == 'fps_adcosnet':
        print(f"\nCreating FPS+ADCoSNet  (alpha={alpha}, beta=x{beta}, N_c={N_c})")
        model = DimensionFixedModel(
            ADCoSNet(in_channels=3, cs_ratio=1.0, num_points=N_c),
            required_input_format='BCN', name='ADCoSNet')
        return model, True, alpha

    if mtype == 'fps_hspcnet':
        print(f"\nCreating FPS+HSPC-Net  (alpha={alpha}, beta=x{beta}, N_c={N_c})")
        model = HSPCNet(up_ratio=beta, d_model=128, num_heads=4)
        return model, True, alpha

    raise ValueError(
        f"Unknown --model_type '{args.model_type}'. "
        "Choose from: unified_cunet, unified_cunet_mlp, "
        "fps_punet, fps_pugacnet, fps_adcosnet, fps_hspcnet  (or 'all')")


# ─────────────────────────────────────────────────────────────────────────────
# Trainer  (verbatim copy of train.py's Trainer — no changes)
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Generic trainer shared by all model types.

    When fps_preprocess=True the loop applies FPS (fps_compression_ratio)
    to every batch before the forward pass.  The loss target is always
    the original full-resolution point cloud.

    This class is a verbatim copy of train.py's Trainer; the only
    difference is that it receives a KITTI DataLoader instead of ModelNet40.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        loss_fn,
        device,
        save_dir,
        model_name='model',
        lr=0.0002,
        weight_decay=0.01,
        num_epochs=50,
        warmup_epochs=5,
        early_stopping_patience=15,
        mixed_precision=True,
        fps_preprocess=False,
        fps_compression_ratio=0.5,
    ):
        self.model                = model.to(device)
        self.train_loader         = train_loader
        self.val_loader           = val_loader
        self.loss_fn              = loss_fn
        self.device               = device
        self.save_dir             = save_dir
        self.model_name           = model_name
        self.num_epochs           = num_epochs
        self.warmup_epochs        = warmup_epochs
        self.early_stopping_patience = early_stopping_patience
        self.mixed_precision      = mixed_precision
        self.fps_preprocess       = fps_preprocess
        self.fps_compression_ratio = fps_compression_ratio

        os.makedirs(save_dir, exist_ok=True)

        self.optimizer = optim.AdamW(
            model.parameters(), lr=lr,
            weight_decay=weight_decay, betas=(0.9, 0.999))
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=1, eta_min=1e-6)
        self.scaler = (
            torch.amp.GradScaler('cuda')
            if mixed_precision and torch.cuda.is_available()
            else None)

        self.best_loss        = float('inf')
        self.patience_counter = 0
        self.train_losses     = []
        self.val_losses       = []
        self.lr_history       = []
        self.initial_lr       = lr
        self.max_lr           = 0.001

    # ── helpers ───────────────────────────────────────────────────────────────

    def _prepare_batch(self, batch_data):
        """
        Returns (model_inputs, targets).

        targets      : original full-resolution cloud  [B, 3, N]
        model_inputs : FPS-downsampled cloud [B, 3, N_c]  if fps_preprocess
                       otherwise same as targets
        """
        inputs, _ = batch_data
        inputs    = inputs.to(self.device)   # [B, 3, N]
        targets   = inputs.clone()

        if self.fps_preprocess:
            N_c = max(32, int(inputs.size(2) * self.fps_compression_ratio))
            model_inputs = fps_subsample(inputs, N_c)   # [B, 3, N_c]
        else:
            model_inputs = inputs

        return model_inputs, targets

    def _forward(self, model_inputs):
        """Handle both Unified-CUNet (requires mode kwarg) and baselines."""
        try:
            return self.model(model_inputs, mode='full')
        except TypeError:
            return self.model(model_inputs)

    # ── one epoch ─────────────────────────────────────────────────────────────

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss    = 0.0
        valid_batches = 0
        total_batches = len(self.train_loader)
        bar_len       = 30

        # Learning-rate warm-up
        if epoch < self.warmup_epochs:
            lr = (self.initial_lr
                  + (self.max_lr - self.initial_lr) * (epoch / self.warmup_epochs))
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
        self.lr_history.append(self.optimizer.param_groups[0]['lr'])

        for batch_idx, batch_data in enumerate(self.train_loader):
            try:
                model_inputs, targets = self._prepare_batch(batch_data)
                self.optimizer.zero_grad(set_to_none=True)

                if self.scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = self._forward(model_inputs)
                        loss    = self.loss_fn(outputs, targets)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self._forward(model_inputs)
                    loss    = self.loss_fn(outputs, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                if not (torch.isnan(loss) or torch.isinf(loss)):
                    epoch_loss    += loss.item()
                    valid_batches += 1

                if (batch_idx + 1) % 5 == 0:
                    prog   = (batch_idx + 1) / total_batches
                    filled = int(bar_len * prog)
                    bar    = '=' * filled + '-' * (bar_len - filled)
                    lstr   = f"Loss: {loss.item():.4f}"
                    if hasattr(self.loss_fn, 'loss_components'):
                        cd = self.loss_fn.loss_components.get('chamfer')
                        if cd is not None:
                            lstr += f" (CD: {cd:.4f})"
                    print(f"\r  [{bar}] {prog*100:.1f}% | {lstr}", end='')

            except Exception as e:
                print(f"\n  Warning: Batch {batch_idx} error: {e}")
                continue

        avg = epoch_loss / max(valid_batches, 1)
        self.train_losses.append(avg)
        return avg

    # ── validation ────────────────────────────────────────────────────────────

    def validate(self):
        self.model.eval()
        val_loss      = 0.0
        valid_batches = 0

        with torch.no_grad():
            for batch_data in self.val_loader:
                try:
                    model_inputs, targets = self._prepare_batch(batch_data)
                    outputs = self._forward(model_inputs)
                    loss    = self.loss_fn(outputs, targets)
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        val_loss      += loss.item()
                        valid_batches += 1
                except Exception:
                    continue

        avg = val_loss / max(valid_batches, 1)
        self.val_losses.append(avg)
        return avg

    # ── full loop ─────────────────────────────────────────────────────────────

    def train(self):
        print(f"\n{'='*60}")
        print(f"Training  : {self.model_name}")
        print(f"Device    : {self.device}")
        print(f"Epochs    : {self.num_epochs}")
        if self.fps_preprocess:
            print(f"FPS preprocess : True  (alpha={self.fps_compression_ratio})")
        print(f"Mixed precision: {self.mixed_precision}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            print(f"Epoch {epoch+1}/{self.num_epochs}")

            train_loss = self.train_epoch(epoch)
            val_loss   = self.validate()

            if epoch >= self.warmup_epochs:
                self.scheduler.step()

            elapsed = time.time() - epoch_start
            print(
                f"\n  Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f} | "
                f"Time: {elapsed:.1f}s"
            )

            if val_loss < self.best_loss:
                self.best_loss        = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_loss, is_best=True)
                print("  ✓ New best model saved!")
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break

            print()

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed in {total_time/60:.1f} minutes")
        print(f"Best validation loss: {self.best_loss:.6f}")
        print(f"{'='*60}")

        return {
            'train_losses': self.train_losses,
            'val_losses':   self.val_losses,
            'lr_history':   self.lr_history,
            'best_loss':    self.best_loss,
        }

    # ── checkpoints ───────────────────────────────────────────────────────────

    def save_checkpoint(self, epoch, loss, is_best=False):
        ckpt = {
            'epoch':                epoch + 1,
            'model_state_dict':     self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss':                 loss,
            'train_losses':         self.train_losses,
            'val_losses':           self.val_losses,
        }
        torch.save(ckpt,
                   os.path.join(self.save_dir, f'{self.model_name}_latest.pt'))
        if is_best:
            torch.save(ckpt,
                       os.path.join(self.save_dir, f'{self.model_name}_best.pt'))

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.train_losses = ckpt.get('train_losses', [])
        self.val_losses   = ckpt.get('val_losses', [])
        self.best_loss    = ckpt.get('loss', float('inf'))
        return ckpt['epoch']


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — single model
# ─────────────────────────────────────────────────────────────────────────────

def train_model(args):
    device = torch.device(
        'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"\nLoading KITTI dataset from {args.kitti_dir} ...")
    train_loader, val_loader = create_kitti_dataloaders(
        kitti_dir=args.kitti_dir,
        num_points=args.num_points,
        patches_per_frame=args.patches_per_frame,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
        max_frames=args.max_frames,
        seed=args.seed,
    )
    print(f"\nTrain batches: {len(train_loader)}"
          f"  |  Val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model, fps_preprocess, fps_ratio = build_model(args)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters    : {total:,}")
    print(f"  Trainable parameters: {trainable:,}")

    # ── Loss — identical selection logic to train.py ──────────────────────────
    if args.simple_loss:
        loss_fn = SimpleLoss()
    elif args.model_type == 'fps_hspcnet':
        loss_fn = HSPCLoss(lambda_cd=1.0, lambda_nc=0.2, lambda_rep=0.1)
    else:
        loss_fn = UnifiedCUNetLoss(
            lambda_cd=1.0,
            lambda_smoothness=0.1,
            lambda_distribution=0.5,
            lambda_edge=0.3,
            lambda_normal=0.2,
        )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        device=device,
        save_dir=args.save_dir,
        model_name=args.model_name,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        early_stopping_patience=args.patience,
        mixed_precision=not args.no_mixed_precision,
        fps_preprocess=fps_preprocess,
        fps_compression_ratio=fps_ratio,
    )

    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        epoch = trainer.load_checkpoint(args.resume)
        print(f"  Resumed from epoch {epoch}")

    history = trainer.train()

    history_path = os.path.join(
        args.save_dir, f'{args.model_name}_history.pt')
    torch.save(history, history_path)
    print(f"\nTraining history saved to {history_path}")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — train all six models sequentially  (--model_type all)
# ─────────────────────────────────────────────────────────────────────────────

def train_all(args):
    """Train every model type in the registry one after another."""
    print(f"\n{'#'*70}")
    print(f"#  Training ALL models on KITTI  "
          f"(alpha={args.compression_ratio}, beta=×{int(args.upsample_ratio)})")
    print(f"{'#'*70}")

    results = {}
    for mtype in ALL_MODEL_TYPES:
        print(f"\n{'═'*70}")
        print(f"  Model: {mtype}")
        print(f"{'═'*70}")

        args.model_type = mtype
        args.model_name = mtype     # each model gets its own checkpoint stem

        try:
            history = train_model(args)
            results[mtype] = history['best_loss']
            print(f"  ✓ {mtype} best val loss: {history['best_loss']:.6f}")
        except Exception as e:
            print(f"  ✗ {mtype} failed: {e}")
            results[mtype] = None

    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for mtype, loss in results.items():
        status = f"{loss:.6f}" if loss is not None else "FAILED"
        print(f"  {mtype:<28}: {status}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train point cloud models from scratch on KITTI',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Train Unified-CUNet
  python train_kitti.py --model_type unified_cunet --kitti_dir ./data/kitti

  # Train all six models sequentially
  python train_kitti.py --model_type all --kitti_dir ./data/kitti

  # Custom compression/upsampling ratios
  python train_kitti.py --model_type fps_punet --kitti_dir ./data/kitti \\
      --compression_ratio 0.75 --upsample_ratio 6

  # Resume a run
  python train_kitti.py --model_type fps_punet --kitti_dir ./data/kitti \\
      --resume ./checkpoints_kitti/fps_punet_latest.pt

  # Quick smoke-test (3 frames, 2 epochs)
  python train_kitti.py --model_type unified_cunet --kitti_dir ./data/kitti \\
      --max_frames 3 --epochs 2 --batch_size 2
""")

    # ── Model ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        '--model_type', type=str, default='unified_cunet',
        choices=['unified_cunet', 'unified_cunet_mlp',
                 'fps_punet', 'fps_pugacnet', 'fps_adcosnet', 'fps_hspcnet',
                 'all'],
        help=(
            "Which model to train (default: unified_cunet):\n"
            "  unified_cunet     — proposed method\n"
            "  unified_cunet_mlp — MLP ablation variant\n"
            "  fps_punet         — FPS + PU-Net   sequential baseline\n"
            "  fps_pugacnet      — FPS + PU-GACNet sequential baseline\n"
            "  fps_adcosnet      — FPS + ADCoSNet  sequential baseline\n"
            "  fps_hspcnet       — FPS + HSPC-Net  sequential baseline\n"
            "  all               — train all six models sequentially"
        ))

    # ── KITTI data ─────────────────────────────────────────────────────────────
    parser.add_argument('--kitti_dir', type=str, required=True,
                        help='Root directory of KITTI dataset.\n'
                             'Expected layout:\n'
                             '  <kitti_dir>/training/velodyne/*.bin   ← train + val\n'
                             '  <kitti_dir>/testing/velodyne/*.bin    ← eval only')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Cap on training/ frames loaded (default: all)')
    parser.add_argument('--patches_per_frame', type=int, default=20,
                        help='Patches to sample per LiDAR frame (default: 20)')
    parser.add_argument('--val_fraction', type=float, default=0.15,
                        help='Fraction of training/ frames used for validation (default: 0.15)')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='Points per patch — must match evaluate_kitti.py (default: 2048)')

    # ── Ratios ─────────────────────────────────────────────────────────────────
    parser.add_argument('--compression_ratio', type=float, default=0.75,
                        help='α — FPS compression ratio (default: 0.5)')
    parser.add_argument('--upsample_ratio', type=float, default=4.0,
                        help='β — upsampling factor (default: 4)')

    # ── Naming / output ────────────────────────────────────────────────────────
    parser.add_argument('--model_name', type=str, default=None,
                        help='Checkpoint filename stem (defaults to --model_type)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_kitti',
                        help='Checkpoint output directory (default: ./checkpoints_kitti)')

    # ── Hyperparameters — same defaults as train.py ───────────────────────────
    parser.add_argument('--batch_size',    type=int,   default=4)
    parser.add_argument('--epochs',        type=int,   default=50)
    parser.add_argument('--lr',            type=float, default=0.0002)
    parser.add_argument('--weight_decay',  type=float, default=0.01)
    parser.add_argument('--warmup_epochs', type=int,   default=5)
    parser.add_argument('--patience',      type=int,   default=15)
    parser.add_argument('--num_workers',   type=int,   default=4)
    parser.add_argument('--seed',          type=int,   default=42)

    # ── Flags ─────────────────────────────────────────────────────────────────
    parser.add_argument('--no_cuda',             action='store_true')
    parser.add_argument('--no_mixed_precision',  action='store_true')
    parser.add_argument('--simple_loss',         action='store_true',
                        help='Use Chamfer-only loss (faster, for debugging)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint path to resume from (single model only)')

    args = parser.parse_args()

    # Default model_name = model_type (unless 'all', handled in train_all)
    if args.model_name is None and args.model_type != 'all':
        args.model_name = args.model_type

    if args.model_type == 'all':
        train_all(args)
    else:
        train_model(args)


if __name__ == '__main__':
    main()
