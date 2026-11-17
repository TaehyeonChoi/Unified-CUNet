"""
Training Script for Unified-CUNet and Baseline Models

Implements the training procedure described in the paper:
    - AdamW optimizer with learning rate 0.0002
    - Weight decay 0.01
    - Cosine annealing scheduler with warm restarts
    - Batch size 4
    - 50 epochs with early stopping
    - Mixed precision training

Model types (--model_type):
    unified_cunet       Proposed method (default)
    unified_cunet_mlp   MLP-variant ablation (Reviewer 2, Comment 1)
    fps_punet           FPS + PU-Net  sequential baseline
    fps_pugacnet        FPS + PU-GACNet sequential baseline
    fps_adcosnet        FPS + ADCoSNet  sequential baseline

For FPS+X baselines, FPS preprocessing is applied automatically
using --compression_ratio as alpha.  The loss target is always
the original full-resolution point cloud.
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unified_cunet import create_unified_cunet, create_unified_cunet_mlp
from models.baselines import (
    fps_subsample,
    PUNet, PUGACNet, ADCoSNet, DimensionFixedModel,
)
from models.losses import UnifiedCUNetLoss, SimpleLoss
from datasets.modelnet40 import ModelNet40Dataset, create_dataloaders
from models.HSPC_Net import HSPCNet, HSPCLoss


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

FPS_MODEL_TYPES = {'fps_punet', 'fps_pugacnet', 'fps_adcosnet', 'fps_hspcnet'}


def build_model(args):
    """
    Instantiate the correct model based on --model_type.

    Returns
    -------
    model          : nn.Module
    fps_preprocess : bool   — True for FPS+X baselines
    fps_ratio      : float  — alpha value for FPS
    """
    mtype = args.model_type.lower()
    alpha = args.compression_ratio
    beta  = int(args.upsample_ratio)
    N     = args.num_points
    N_c   = max(32, int(N * alpha))

    # Proposed method
    if mtype == 'unified_cunet':
        print(f"\nCreating Unified-CUNet  (alpha={alpha}, beta=x{beta})")
        model = create_unified_cunet(
            compression_ratio=alpha,
            upsample_ratio=float(beta),
            hidden_dim=64,
            num_transformer_blocks=4,
        )
        return model, False, alpha

    # MLP ablation variant
    if mtype == 'unified_cunet_mlp':
        print(f"\nCreating Unified-CUNet-MLP  (alpha={alpha}, beta=x{beta})  [ablation]")
        model = create_unified_cunet_mlp(
            compression_ratio=alpha,
            upsample_ratio=float(beta),
            hidden_dim=64,
            num_transformer_blocks=4,
        )
        return model, False, alpha

    # FPS + PU-Net
    if mtype == 'fps_punet':
        print(f"\nCreating FPS+PU-Net  (alpha={alpha}, beta=x{beta}, N_c={N_c})")
        model = DimensionFixedModel(
            PUNet(num_points=N_c, up_ratio=beta),
            required_input_format='BNC',
            name='PU-Net',
        )
        return model, True, alpha

    # FPS + PU-GACNet
    if mtype == 'fps_pugacnet':
        print(f"\nCreating FPS+PU-GACNet  (alpha={alpha}, beta=x{beta}, N_c={N_c})")
        model = DimensionFixedModel(
            PUGACNet(in_channels=3, out_channels=3, up_factor=beta),
            required_input_format='BCN',
            name='PU-GACNet',
        )
        return model, True, alpha

    # FPS + ADCoSNet
    if mtype == 'fps_adcosnet':
        print(f"\nCreating FPS+ADCoSNet  (alpha={alpha}, beta=x{beta}, N_c={N_c})")
        model = DimensionFixedModel(
            ADCoSNet(in_channels=3, cs_ratio=1.0, num_points=N_c),
            required_input_format='BCN',
            name='ADCoSNet',
        )
        return model, True, alpha

    # FPS + HSPC-Net
    if mtype == 'fps_hspcnet':
        print(f"\nCreating FPS+HSPC-Net  (alpha={alpha}, beta=x{beta}, N_c={N_c})")
        model = HSPCNet(up_ratio=beta, d_model=128, num_heads=4)
        return model, True, alpha

    raise ValueError(
        f"Unknown --model_type '{args.model_type}'. "
        "Choose from: unified_cunet, unified_cunet_mlp, "
        "fps_punet, fps_pugacnet, fps_adcosnet, fps_hspcnet"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Generic trainer shared by all model types.

    When fps_preprocess=True the loop applies FPS (fps_compression_ratio)
    to every batch before the forward pass.  The loss target is always
    the original full-resolution point cloud.
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
        self.model       = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.loss_fn     = loss_fn
        self.device      = device
        self.save_dir    = save_dir
        self.model_name  = model_name
        self.num_epochs  = num_epochs
        self.warmup_epochs = warmup_epochs
        self.early_stopping_patience = early_stopping_patience
        self.mixed_precision = mixed_precision
        self.fps_preprocess = fps_preprocess
        self.fps_compression_ratio = fps_compression_ratio

        os.makedirs(save_dir, exist_ok=True)

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=1, eta_min=1e-6
        )
        self.scaler = (
            torch.amp.GradScaler('cuda')
            if mixed_precision and torch.cuda.is_available()
            else None
        )

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
        inputs  = inputs.to(self.device)      # [B, 3, N]  BCN format
        targets = inputs.clone()

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

                # Progress bar
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
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f} | Time: {elapsed:.1f}s"
            )

            if val_loss < self.best_loss:
                self.best_loss        = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_loss, is_best=True)
                print("  \u2713 New best model saved!")
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
            'train_losses' : self.train_losses,
            'val_losses'   : self.val_losses,
            'lr_history'   : self.lr_history,
            'best_loss'    : self.best_loss,
        }

    # ── checkpoints ───────────────────────────────────────────────────────────

    def save_checkpoint(self, epoch, loss, is_best=False):
        ckpt = {
            'epoch'               : epoch + 1,
            'model_state_dict'    : self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss'                : loss,
            'train_losses'        : self.train_losses,
            'val_losses'          : self.val_losses,
        }
        torch.save(ckpt, os.path.join(self.save_dir, f'{self.model_name}_latest.pt'))
        if is_best:
            torch.save(ckpt, os.path.join(self.save_dir, f'{self.model_name}_best.pt'))

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
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def train_model(args):
    device = torch.device(
        'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    )
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Dataset
    print(f"\nLoading dataset from {args.data_dir}...")
    train_loader, val_loader, _ = create_dataloaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_points=args.num_points,
        num_workers=args.num_workers,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    model, fps_preprocess, fps_ratio = build_model(args)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters    : {total:,}")
    print(f"  Trainable parameters: {trainable:,}")

    # Loss
    loss_fn = (
        SimpleLoss() if args.simple_loss
        else HSPCLoss(lambda_cd=1.0, lambda_nc=0.2, lambda_rep=0.1)
        if args.model_type == 'fps_hspcnet'
        else UnifiedCUNetLoss(
            lambda_cd=1.0,
            lambda_smoothness=0.1,
            lambda_distribution=0.5,
            lambda_edge=0.3,
            lambda_normal=0.2,
        )
    )

    # Trainer
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
        print(f"Resuming from epoch {epoch}")

    history = trainer.train()

    history_path = os.path.join(args.save_dir, f'{args.model_name}_history.pt')
    torch.save(history, history_path)
    print(f"\nTraining history saved to {history_path}")

    return history


def main():
    parser = argparse.ArgumentParser(
        description='Train point cloud models',
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Model selection
    parser.add_argument(
        '--model_type', type=str, default='unified_cunet',
        choices=['unified_cunet', 'unified_cunet_mlp',
                 'fps_punet', 'fps_pugacnet', 'fps_adcosnet', 'fps_hspcnet'],
        help=(
            "Which model to train:\n"
            "  unified_cunet     — proposed method (default)\n"
            "  unified_cunet_mlp — MLP ablation variant\n"
            "  fps_punet         — FPS → PU-Net   sequential baseline\n"
            "  fps_pugacnet      — FPS → PU-GACNet sequential baseline\n"
            "  fps_adcosnet      — FPS → ADCoSNet  sequential baseline\n"
            "  fps_hspcnet       — FPS → HSPC-Net  sequential baseline\n"
            "For fps_* types, FPS is applied automatically with alpha=--compression_ratio."
        )
    )

    # Dataset
    parser.add_argument('--data_dir',   type=str, default='./ModelNet40')
    parser.add_argument('--num_points', type=int, default=2048)

    # Ratios
    parser.add_argument('--compression_ratio', type=float, default=0.5,
                        help='alpha — FPS compression ratio (also used as fps_alpha for fps_* models)')
    parser.add_argument('--upsample_ratio',    type=float, default=4.0,
                        help='beta — upsampling factor')

    # Naming
    parser.add_argument('--model_name', type=str, default=None,
                        help='Checkpoint stem (defaults to --model_type)')
    parser.add_argument('--save_dir',   type=str, default='./checkpoints')

    # Hyperparameters
    parser.add_argument('--batch_size',    type=int,   default=4)
    parser.add_argument('--epochs',        type=int,   default=50)
    parser.add_argument('--lr',            type=float, default=0.0002)
    parser.add_argument('--weight_decay',  type=float, default=0.01)
    parser.add_argument('--warmup_epochs', type=int,   default=5)
    parser.add_argument('--patience',      type=int,   default=15)
    parser.add_argument('--num_workers',   type=int,   default=4)
    parser.add_argument('--seed',          type=int,   default=42)

    # Flags
    parser.add_argument('--no_cuda',            action='store_true')
    parser.add_argument('--no_mixed_precision', action='store_true')
    parser.add_argument('--simple_loss',        action='store_true',
                        help='Use Chamfer-only loss (faster, for debugging)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint path to resume from')

    args = parser.parse_args()

    if args.model_name is None:
        args.model_name = args.model_type

    train_model(args)


if __name__ == '__main__':
    main()
