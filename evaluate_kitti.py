"""
KITTI Real-World Evaluation Script for Unified-CUNet
=====================================================

Evaluates all six models on KITTI outdoor LiDAR scenes using the same
6-metric protocol as evaluate.py (CD, HD, P2S, Uniformity, NC, CC).

Adapted from run_kitti_experiments.py (KITTIConfig layout, directory
conventions, batch-capped evaluation) but with:
  • Our model registry (FPS+ADCoSNet, FPS+PU-GACNet, FPS+PU-Net,
    FPS+HSPC-Net, Unified-CUNet-MLP, Unified-CUNet)
  • Our FPS preprocessing protocol — all baselines receive FPS-compressed
    input at the same α as Unified-CUNet (no random permutation)
  • Our 6-metric suite (CD, HD, P2S, Uniformity, NC, CC)
  • No dependency on external model/metric/dataset modules from the
    original run_kitti_experiments.py

KITTI dataset structure
-----------------------
    <kitti_dir>/
        training/
            velodyne/  *.bin   ← use with --split training (trained models)
            calib/     *.txt
            image_2/   *.png
        testing/
            velodyne/  *.bin   ← default evaluation split (no GT labels needed)
            calib/     *.txt
            image_2/   *.png

Use --split training to evaluate on training/ frames (e.g. to cross-check
against train_kitti.py results).  Default is testing/.

KITTI .bin format: each point is [x, y, z, intensity] as float32.
Intensity is dropped; only XYZ is used.

Patch sampling
--------------
Each LiDAR frame typically contains 100k–150k points.  We sample fixed-size
patches (--num_points, default 2048) around random seed points using k-NN,
normalise each patch to the unit sphere, and feed it to the models — exactly
as in evaluate_scannet.py.  This preserves consistency with ModelNet40 training.

Usage
------
# Evaluate on testing/ frames (default), single config
python evaluate_kitti.py \\
    --kitti_dir ./data/kitti \\
    --checkpoint_dir ./checkpoints_kitti \\
    --compression_ratio 0.5 --upsample_ratio 4

# All four (alpha, beta) configs
python evaluate_kitti.py \\
    --kitti_dir ./data/kitti \\
    --checkpoint_dir ./checkpoints_kitti \\
    --full_comparison

# Quick smoke-test (5 frames, 4 patches each)
python evaluate_kitti.py \\
    --kitti_dir ./data/kitti \\
    --checkpoint_dir ./checkpoints_kitti \\
    --max_frames 5 --patches_per_frame 4

# Evaluate on training/ frames instead
python evaluate_kitti.py \\
    --kitti_dir ./data/kitti \\
    --checkpoint_dir ./checkpoints_kitti \\
    --split training

Outputs (written to --save_dir, default ./results_kitti/):
    kitti_results_a<alpha>_b<beta>.txt   — human-readable table
    kitti_table_a<alpha>_b<beta>.tex     — LaTeX snippet (paste into paper)
"""

import os
import sys
import glob
import argparse
import random
import struct
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

# ── path setup (mirrors evaluate.py) ─────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unified_cunet import create_unified_cunet, create_unified_cunet_mlp
from models.baselines import (
    fps_subsample,
    PUNet, PUGACNet, ADCoSNet, DimensionFixedModel,
)
from models.HSPC_Net import HSPCNet


# ─────────────────────────────────────────────────────────────────────────────
# KITTI configuration (mirrors KITTIConfig from run_kitti_experiments.py,
# restricted to settings relevant for our evaluation)
# ─────────────────────────────────────────────────────────────────────────────

class KITTIConfig:
    """
    KITTI-specific evaluation configuration.

    Mirrors the KITTIConfig class from run_kitti_experiments.py but
    restricted to the settings needed for our model evaluation.
    """

    def __init__(self):
        # Data
        self.dataset          = 'kitti'
        self.num_points       = 2048          # patch size (matches ModelNet40 training)
        self.patches_per_frame = 10           # random patches sampled per LiDAR frame
        self.max_frames       = None          # cap on frames evaluated (None = all)

        # Compression / upsampling
        self.compression_ratio = 0.5         # α
        self.up_ratio          = 4           # β

        # Evaluation
        self.eval_batch_size  = 1            # kept at 1 to match evaluate.py
        self.max_eval_batches = 50           # max patches per model (from original)
        self.gpu_id           = 0
        self.use_gpu          = True

        # Derived (recalculated on update)
        self._update_derived()

    def _update_derived(self):
        self.input_points  = max(32, int(self.num_points * self.compression_ratio))
        self.output_points = int(self.num_points * self.up_ratio)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._update_derived()

    def summary(self):
        return (f"  num_points={self.num_points}  α={self.compression_ratio}"
                f"  β=×{self.up_ratio}"
                f"  input_points={self.input_points}"
                f"  output_points={self.output_points}")


# ─────────────────────────────────────────────────────────────────────────────
# Inline FPS (avoids fragile cross-module dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _fps(pts: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Farthest Point Sampling. pts: [B, 3, N] → [B, 3, n_samples]"""
    B, _, N = pts.shape
    device   = pts.device
    selected = torch.zeros(B, n_samples, dtype=torch.long, device=device)
    dist     = torch.full((B, N), float('inf'), device=device)
    current  = torch.zeros(B, dtype=torch.long, device=device)

    for i in range(n_samples):
        selected[:, i] = current
        cur_pts = pts[torch.arange(B), :, current].unsqueeze(2)   # [B, 3, 1]
        d       = ((pts - cur_pts) ** 2).sum(dim=1)               # [B, N]
        dist    = torch.min(dist, d)
        current = dist.argmax(dim=1)

    idx = selected.unsqueeze(1).expand(-1, 3, -1)
    return pts.gather(2, idx)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics (self-contained; identical to evaluate_scannet.py)
# ─────────────────────────────────────────────────────────────────────────────

def _pairwise_sq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """[N,3],[M,3] → [N,M] squared distances."""
    a2 = (a ** 2).sum(1, keepdims=True)
    b2 = (b ** 2).sum(1, keepdims=True).T
    return np.maximum(a2 + b2 - 2 * (a @ b.T), 0.0)


def chamfer_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    d = _pairwise_sq(pred, gt)
    return float(d.min(1)[0].mean() + d.min(0)[0].mean())


def hausdorff_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    d = _pairwise_sq(pred, gt)
    return float(max(np.sqrt(np.maximum(d.min(1)[0].max(), 0)),
                     np.sqrt(np.maximum(d.min(0)[0].max(), 0))))


def point_to_surface(pred: np.ndarray, gt: np.ndarray, k: int = 5) -> float:
    d      = _pairwise_sq(pred, gt)
    nn_idx = d.argmin(1)
    dists  = []
    for i, j in enumerate(nn_idx):
        knn   = np.argsort(_pairwise_sq(gt[j:j+1], gt)[0])[:k]
        nbrs  = gt[knn]
        ctr   = nbrs.mean(0)
        _, _, vh = np.linalg.svd(nbrs - ctr)
        normal = vh[-1] / (np.linalg.norm(vh[-1]) + 1e-8)
        dists.append(abs((pred[i] - gt[j]) @ normal))
    return float(np.mean(dists)) if dists else 0.0


def _estimate_normals(pts: np.ndarray, k: int = 20) -> np.ndarray:
    d      = _pairwise_sq(pts, pts)
    np.fill_diagonal(d, np.inf)
    nn_idx = np.argsort(d, axis=1)[:, :k]
    normals = []
    for i in range(len(pts)):
        nbrs = pts[nn_idx[i]] - pts[i]
        _, _, vh = np.linalg.svd(nbrs)
        normals.append(vh[-1])
    return np.array(normals)


def normal_consistency(pred: np.ndarray, gt: np.ndarray, k: int = 20) -> float:
    pn  = _estimate_normals(pred, k)
    gn  = _estimate_normals(gt,   k)
    nn  = _pairwise_sq(pred, gt).argmin(1)
    return float(np.abs((pn * gn[nn]).sum(1)).mean())


def curvature_consistency(pred: np.ndarray, gt: np.ndarray, k: int = 20) -> float:
    def _curvature(pts, k):
        d = _pairwise_sq(pts, pts)
        np.fill_diagonal(d, np.inf)
        nn_idx = np.argsort(d, axis=1)[:, :k]
        curv = []
        for i in range(len(pts)):
            nbrs = pts[nn_idx[i]] - pts[i]
            _, s, _ = np.linalg.svd(nbrs)
            curv.append(s[-1] / (s.sum() + 1e-8))
        return np.array(curv)

    pc  = _curvature(pred, k)
    gc  = _curvature(gt,   k)
    nn  = _pairwise_sq(pred, gt).argmin(1)
    return float(1.0 - np.abs(pc - gc[nn]).mean())


def uniformity(pts: np.ndarray, k: int = 10) -> float:
    d = _pairwise_sq(pts, pts)
    np.fill_diagonal(d, np.inf)
    knn = np.sort(d, axis=1)[:, :k]
    return float(np.sqrt(np.maximum(knn, 0)).mean(1).var())


def compute_all_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    return {
        'chamfer_distance':      chamfer_distance(pred, gt),
        'hausdorff_distance':    hausdorff_distance(pred, gt),
        'point_to_surface':      point_to_surface(pred, gt),
        'uniformity':            uniformity(pred),
        'normal_consistency':    normal_consistency(pred, gt),
        'curvature_consistency': curvature_consistency(pred, gt),
    }


# ─────────────────────────────────────────────────────────────────────────────
# KITTI data loading
# ─────────────────────────────────────────────────────────────────────────────

def _read_bin(path: str) -> np.ndarray:
    """
    Read a KITTI Velodyne .bin file.
    Each point is stored as (x, y, z, intensity) float32.
    Returns [N, 3] XYZ array.
    """
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return pts[:, :3]          # drop intensity


def _discover_bin_files(kitti_dir: str, split: str = 'testing') -> list:
    """
    Return sorted list of .bin paths for the requested split.

    split='testing'  → <kitti_dir>/testing/velodyne/*.bin   (default)
    split='training' → <kitti_dir>/training/velodyne/*.bin

    Falls back to a recursive search if the standard layout is not found.
    """
    primary = sorted(glob.glob(
        os.path.join(kitti_dir, split, 'velodyne', '*.bin')))
    if primary:
        return primary

    # Fallback: any .bin under kitti_dir (odometry / raw layouts)
    fallback = sorted(glob.glob(
        os.path.join(kitti_dir, '**', '*.bin'), recursive=True))
    return fallback


def _sample_patches(scene_pts: np.ndarray,
                    patch_size: int,
                    n_patches: int,
                    rng: random.Random) -> list:
    """
    Sample n_patches fixed-size patches from a KITTI frame.

    Strategy (mirrors evaluate_scannet.py):
      1. Choose a random seed point.
      2. Take patch_size nearest neighbours.
      3. Centre and scale to unit sphere.

    Returns list of np.ndarray [patch_size, 3].
    """
    N = len(scene_pts)
    if N < patch_size:
        return []

    patches = []
    seed_indices = rng.choices(range(N), k=n_patches)

    for idx in seed_indices:
        seed  = scene_pts[idx]
        dists = ((scene_pts - seed) ** 2).sum(1)
        nn    = np.argpartition(dists, patch_size)[:patch_size]
        patch = scene_pts[nn].copy()
        # Normalise to unit sphere
        patch -= patch.mean(0)
        scale  = np.linalg.norm(patch, axis=1).max() + 1e-8
        patch /= scale
        patches.append(patch.astype(np.float32))

    return patches


# ─────────────────────────────────────────────────────────────────────────────
# Model registry (identical to evaluate_scannet.py / evaluate.py)
# ─────────────────────────────────────────────────────────────────────────────

def _build_unified_cunet(alpha, beta, N):
    model = create_unified_cunet(
        compression_ratio=alpha, upsample_ratio=float(beta),
        hidden_dim=64, num_transformer_blocks=4)
    return model, False, alpha

def _build_unified_cunet_mlp(alpha, beta, N):
    model = create_unified_cunet_mlp(
        compression_ratio=alpha, upsample_ratio=float(beta),
        hidden_dim=64, num_transformer_blocks=4)
    return model, False, alpha

def _build_fps_punet(alpha, beta, N):
    N_c = max(32, int(N * alpha))
    model = DimensionFixedModel(
        PUNet(num_points=N_c, up_ratio=beta),
        required_input_format='BNC', name='PU-Net')
    return model, True, alpha

def _build_fps_pugacnet(alpha, beta, N):
    model = DimensionFixedModel(
        PUGACNet(in_channels=3, out_channels=3, up_factor=beta),
        required_input_format='BCN', name='PU-GACNet')
    return model, True, alpha

def _build_fps_adcosnet(alpha, beta, N):
    N_c = max(32, int(N * alpha))
    model = DimensionFixedModel(
        ADCoSNet(in_channels=3, cs_ratio=1.0, num_points=N_c),
        required_input_format='BCN', name='ADCoSNet')
    return model, True, alpha

def _build_fps_hspcnet(alpha, beta, N):
    model = HSPCNet(up_ratio=beta, d_model=128, num_heads=4)
    return model, True, alpha

MODEL_REGISTRY = [
    dict(key='fps_adcosnet',      display='fps_adcosnet',         build=_build_fps_adcosnet),
    dict(key='fps_pugacnet',      display='FPS+PU-GACNet',        build=_build_fps_pugacnet),
    dict(key='fps_punet',         display='FPS+PU-Net',           build=_build_fps_punet),
    dict(key='fps_hspcnet',       display='FPS+HSPC-Net',         build=_build_fps_hspcnet),
    dict(key='unified_cunet_mlp', display='Unified-CUNet-MLP',    build=_build_unified_cunet_mlp),
    dict(key='unified_cunet',     display='Unified-CUNet (Ours)', build=_build_unified_cunet),
]

CKPT_KEYS = [m['key'] for m in MODEL_REGISTRY]

METRIC_KEYS   = ['chamfer_distance', 'hausdorff_distance', 'point_to_surface',
                 'uniformity', 'normal_consistency', 'curvature_consistency']
METRIC_LABELS = ['CD', 'HD', 'P2S', 'Uniformity', 'NC', 'CC']
METRIC_HIGHER = [False, False, False, False, True, True]


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_checkpoint(model, path, device):
    if not path or not os.path.exists(path):
        print(f"    [skip] checkpoint not found: {path}")
        return model, False
    ckpt  = torch.load(path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    epoch = ckpt.get('epoch', '?')
    print(f"    loaded  {path}  (epoch {epoch})")
    return model, True


def _resolve_checkpoint(args, key):
    """
    Checkpoint resolution — mirrors run_kitti_experiments.py's multi-path
    search pattern.  Priority:
      1. Explicit --ckpt_<key> flag
      2. <checkpoint_dir>/<key>_best.pt  (our naming convention)
      3. <checkpoint_dir>/<key>/best_model.pth  (run_kitti_experiments.py convention)
    """
    flag = getattr(args, f'ckpt_{key}', None)
    if flag:
        return flag

    candidates = [
        os.path.join(args.checkpoint_dir, f'{key}_best.pt'),
        os.path.join(args.checkpoint_dir, key, 'best_model.pth'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    return candidates[0]     # return primary path (will trigger "not found" warning)


def _model_forward(model, x):
    try:
        return model(x, mode='full')
    except TypeError:
        return model(x)


# ─────────────────────────────────────────────────────────────────────────────
# Metric accumulator
# ─────────────────────────────────────────────────────────────────────────────

class Accumulator:
    def __init__(self):
        self._sums  = {}
        self._count = 0

    def update(self, d: dict):
        for k, v in d.items():
            self._sums[k] = self._sums.get(k, 0.0) + float(v)
        self._count += 1

    def average(self) -> dict:
        if self._count == 0:
            return {}
        return {k: v / self._count for k, v in self._sums.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model_on_patches(model, patches: list, device,
                               fps_preprocess: bool, fps_ratio: float,
                               max_eval_batches: int) -> dict:
    """
    Run model over a capped list of patches, return averaged metric dict.

    Mirrors run_kitti_experiments.py's max_eval_batches cap to keep wall
    time predictable across the large KITTI dataset.
    """
    model.eval()
    acc = Accumulator()

    # Apply the same cap used in run_kitti_experiments.py
    patches_to_run = patches[:max_eval_batches]

    with torch.no_grad():
        for idx, patch in enumerate(patches_to_run):
            try:
                # patch: [N, 3] → tensor [1, 3, N]
                x = torch.from_numpy(patch.T[None]).to(device)      # [1, 3, N]
                target_np = patch.copy()                             # ground truth

                if fps_preprocess:
                    N_c  = max(32, int(x.size(2) * fps_ratio))
                    minp = _fps(x, N_c)
                else:
                    minp = x

                out  = _model_forward(model, minp)                  # [1, 3, N_out]
                pred = out.squeeze(0).cpu().numpy()                 # [3, N_out]
                if pred.shape[0] == 3:
                    pred = pred.T                                    # [N_out, 3]

                acc.update(compute_all_metrics(pred, target_np))

                if (idx + 1) % 10 == 0:
                    print(f"      {idx+1}/{len(patches_to_run)} patches...",
                          flush=True)

            except Exception as e:
                print(f"    Warning: patch {idx}: {e}")
                continue

    return acc.average()


# ─────────────────────────────────────────────────────────────────────────────
# Display / save
# ─────────────────────────────────────────────────────────────────────────────

def _best_per_metric(results: dict) -> dict:
    best = {}
    for ki, k in enumerate(METRIC_KEYS):
        vals = [m[k] for m in results.values() if k in m]
        if not vals:
            best[k] = None
            continue
        best[k] = max(vals) if METRIC_HIGHER[ki] else min(vals)
    return best


def print_table(results: dict, title: str = ''):
    best  = _best_per_metric(results)
    width = max(len(n) for n in results) + 2
    hdr   = f"{'Method':<{width}}" + "".join(f"{l:>12}" for l in METRIC_LABELS)
    sep   = '─' * len(hdr)

    print(f"\n{'='*70}")
    if title:
        print(f"  {title}")
    print(f"{'='*70}\n")
    print(hdr)
    print(sep)

    for name, m in results.items():
        row = f"{name:<{width}}"
        for ki, k in enumerate(METRIC_KEYS):
            v   = m.get(k, float('nan'))
            fmt = f"{v:.6f}" if k == 'uniformity' else f"{v:.4f}"
            is_best = (best[k] is not None and abs(v - best[k]) < 1e-7)
            row += f"{'*'+fmt if is_best else fmt:>12}"
        print(row)
    print('\n* = best in column')


def save_txt(results: dict, path: str, title: str = ''):
    best = _best_per_metric(results)
    with open(path, 'w') as f:
        f.write("KITTI Evaluation Results\n")
        f.write(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Config    : {title}\n")
        f.write('=' * 70 + '\n\n')
        f.write(f"{'Method':<28}" +
                "".join(f"{l:>10}" for l in METRIC_LABELS) + '\n')
        f.write('-' * 70 + '\n')
        for name, m in results.items():
            row = f"{name:<28}"
            for ki, k in enumerate(METRIC_KEYS):
                v   = m.get(k, float('nan'))
                fmt = f"{v:.6f}" if k == 'uniformity' else f"{v:.4f}"
                is_best = (best[k] is not None and abs(v - best[k]) < 1e-7)
                row += f"{'*'+fmt if is_best else fmt:>10}"
            f.write(row + '\n')
        f.write('\n* = best in column\n')
    print(f"Saved -> {path}")


def save_latex(results: dict, path: str, config_str: str = ''):
    best = _best_per_metric(results)
    with open(path, 'w') as f:
        f.write("% Auto-generated KITTI table — paste into paper\n")
        f.write("\\begin{table}[!ht]\n"
                "\\begin{adjustwidth}{-2.25in}{0in}\n"
                "\\centering\n")
        f.write(
            f"\\caption{{\\textbf{{Quantitative comparison on KITTI"
            f" outdoor LiDAR scenes ({config_str}).}}"
            " Bold values indicate best performance."
            " All baselines receive FPS-compressed input at the same"
            " $\\alpha$ as Unified-CUNet."
            " Patches of 2,048 points are sampled from each LiDAR frame"
            " and normalised to the unit sphere.}}\n")
        f.write("\\begin{tabular}{|l|l|l|l|l|l|l|}\\hline\n")
        f.write("\\textbf{Method} & \\textbf{CD} & \\textbf{HD} &"
                " \\textbf{P2S} & \\textbf{Uniformity} &"
                " \\textbf{NC} & \\textbf{CC} \\\\ \\thickhline\n")
        for name, m in results.items():
            cells = []
            for k in METRIC_KEYS:
                v   = m.get(k, 0.0)
                fmt = f"{v:.6f}" if k == 'uniformity' else f"{v:.4f}"
                is_best = (best[k] is not None and abs(v - best[k]) < 1e-7)
                cells.append(f"\\textbf{{{fmt}}}" if is_best else fmt)
            f.write(f"{name} & " + " & ".join(cells) + " \\\\ \\hline\n")
        f.write("\\end{tabular}\n"
                "\\label{table:kitti}\n"
                "\\end{adjustwidth}\n"
                "\\end{table}\n")
    print(f"Saved -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation routine
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(args):
    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device(
        'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"\nDevice: {device}")

    rng   = random.Random(args.seed)
    alpha = args.compression_ratio
    beta  = int(args.upsample_ratio)
    N     = args.num_points

    # ── Build KITTIConfig (for logging consistency with run_kitti_experiments) ──
    cfg = KITTIConfig()
    cfg.update(compression_ratio=alpha, up_ratio=beta, num_points=N,
               patches_per_frame=args.patches_per_frame,
               max_frames=args.max_frames,
               max_eval_batches=args.max_eval_batches)
    print(f"\nKITTI Configuration:{cfg.summary()}")

    # ── Discover .bin frames ──────────────────────────────────────────────────
    split = getattr(args, 'split', 'testing')
    print(f"\nScanning {args.kitti_dir}/{split}/velodyne/ for .bin files...")
    bin_files = _discover_bin_files(args.kitti_dir, split=split)
    if not bin_files:
        raise FileNotFoundError(
            f"No .bin files found under '{args.kitti_dir}/{split}/velodyne/'.\n"
            f"Expected layout:\n"
            f"  {args.kitti_dir}/testing/velodyne/*.bin\n"
            f"  {args.kitti_dir}/training/velodyne/*.bin\n"
            f"Use --split training to read from training/ instead.")

    if args.max_frames:
        bin_files = bin_files[:args.max_frames]

    print(f"Found {len(bin_files)} frame(s)  |  "
          f"{args.patches_per_frame} patch(es) each  |  "
          f"patch_size={N}")

    # ── Load frames and sample patches ───────────────────────────────────────
    print("\nLoading frames and sampling patches...")
    all_patches = []
    for bin_path in bin_files:
        try:
            pts = _read_bin(bin_path)
        except Exception as e:
            print(f"  [skip] {os.path.basename(bin_path)}: {e}")
            continue
        patches = _sample_patches(pts, N, args.patches_per_frame, rng)
        all_patches.extend(patches)
        if len(all_patches) % 100 == 0 or bin_files.index(bin_path) < 5:
            print(f"  {os.path.basename(bin_path)}: {len(pts):>7} pts  "
                  f"→  {len(patches)} patches  "
                  f"(total so far: {len(all_patches)})")

    if not all_patches:
        raise RuntimeError(
            "No patches generated.  Check frame files and --num_points.")

    # Apply max_eval_batches cap (matches run_kitti_experiments.py behaviour)
    if args.max_eval_batches and len(all_patches) > args.max_eval_batches:
        rng.shuffle(all_patches)
        all_patches = all_patches[:args.max_eval_batches]
        print(f"Capped to {args.max_eval_batches} patches "
              "(--max_eval_batches, matches run_kitti_experiments.py)")

    print(f"\nTotal patches for evaluation: {len(all_patches)}")

    # ── Model selection ───────────────────────────────────────────────────────
    if args.models:
        keys_to_run = [k.strip() for k in args.models.split(',')]
    else:
        keys_to_run = [m['key'] for m in MODEL_REGISTRY]

    os.makedirs(args.save_dir, exist_ok=True)
    results = {}

    # ── Evaluation loop ───────────────────────────────────────────────────────
    for entry in MODEL_REGISTRY:
        key     = entry['key']
        display = entry['display']

        if key not in keys_to_run:
            continue

        print(f"\n{'─'*60}")
        print(f"  {display}  (alpha={alpha}, beta=×{beta})")
        print(f"{'─'*60}")

        model, fps_preprocess, fps_ratio = entry['build'](alpha, beta, N)

        ckpt_path = _resolve_checkpoint(args, key)
        model, loaded = _load_checkpoint(model, ckpt_path, device)
        if not loaded:
            print("    [WARNING] Random weights — results are not meaningful.")

        model = model.to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f"    Parameters: {params:,}"
              + ("  |  FPS preprocess ON" if fps_preprocess else ""))

        m = evaluate_model_on_patches(
            model, all_patches, device,
            fps_preprocess=fps_preprocess,
            fps_ratio=fps_ratio,
            max_eval_batches=len(all_patches))   # already capped above

        results[display] = m

        nc = m.get('normal_consistency', 0)
        cc = m.get('curvature_consistency', 0)
        cd = m.get('chamfer_distance', 0)
        print(f"    CD={cd:.4f}  NC={nc:.4f}  CC={cc:.4f}")

    # ── Output ────────────────────────────────────────────────────────────────
    config_str = f"alpha={alpha}, beta=×{beta}"
    title      = f"KITTI Real-World  ({config_str})"
    print_table(results, title=title)

    stem = f"a{alpha}_b{beta}"
    save_txt  (results, os.path.join(args.save_dir, f"kitti_results_{stem}.txt"), title)
    save_latex(results, os.path.join(args.save_dir, f"kitti_table_{stem}.tex"),   config_str)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Full comparison (all four (alpha, beta) configs)
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS = [(0.50, 4), (0.75, 4), (0.50, 6), (0.75, 6)]


def run_full_comparison(args):
    all_results = {}
    for alpha, beta in CONFIGS:
        print(f"\n{'#'*70}")
        print(f"#  KITTI  alpha={alpha}  beta=×{beta}")
        print(f"{'#'*70}")
        args.compression_ratio = alpha
        args.upsample_ratio    = float(beta)
        all_results[(alpha, beta)] = run_evaluation(args)

    print(f"\n\n{'='*120}")
    print(" KITTI FULL COMPARISON SUMMARY")
    print(f"{'='*120}")
    for (alpha, beta), results in all_results.items():
        print(f"\n  alpha={alpha}  beta=×{beta}")
        print_table(results)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='KITTI real-world evaluation for Unified-CUNet',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Evaluate on testing/ frames (default)
  python evaluate_kitti.py --kitti_dir ./data/kitti --checkpoint_dir ./checkpoints_kitti

  # All four (alpha, beta) configs
  python evaluate_kitti.py --kitti_dir ./data/kitti --checkpoint_dir ./checkpoints_kitti \\
      --full_comparison

  # Quick smoke-test (5 frames, 4 patches each)
  python evaluate_kitti.py --kitti_dir ./data/kitti --checkpoint_dir ./checkpoints_kitti \\
      --max_frames 5 --patches_per_frame 4

  # Evaluate on training/ frames instead
  python evaluate_kitti.py --kitti_dir ./data/kitti --checkpoint_dir ./checkpoints_kitti \\
      --split training

  # Evaluate only specific models
  python evaluate_kitti.py --kitti_dir ./data/kitti --checkpoint_dir ./checkpoints_kitti \\
      --models unified_cunet,fps_punet
""")

    # KITTI
    parser.add_argument('--kitti_dir', type=str, required=True,
                        help='Root directory of KITTI dataset.\n'
                             'Expected layout:\n'
                             '  <kitti_dir>/testing/velodyne/*.bin    ← default eval split\n'
                             '  <kitti_dir>/training/velodyne/*.bin   ← optional (--split training)')
    parser.add_argument('--split', type=str, default='testing',
                        choices=['testing', 'training'],
                        help='Which KITTI split to evaluate on (default: testing)')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Max LiDAR frames to load (default: all)')
    parser.add_argument('--patches_per_frame', type=int, default=10,
                        help='Random patches to sample per frame (default: 10)')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='Points per patch (default: 2048, matches ModelNet40 training)')
    parser.add_argument('--max_eval_batches', type=int, default=50,
                        help='Hard cap on patches evaluated per model (default: 50)')

    # Ratios
    parser.add_argument('--compression_ratio', type=float, default=0.75,
                        help='α — compression ratio (default: 0.5)')
    parser.add_argument('--upsample_ratio', type=float, default=4.0,
                        help='β — upsampling ratio (default: 4)')

    # Checkpoints
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_kitti',
                        help='Directory for *_best.pt checkpoints (default: ./checkpoints_kitti)\n'
                             'Also checks <dir>/<key>/best_model.pth')
    for k in CKPT_KEYS:
        parser.add_argument(f'--ckpt_{k}', type=str, default=None,
                            help=f'Explicit checkpoint path for {k}')

    # Model selection
    parser.add_argument('--models', type=str, default=None,
                        help=('Comma-separated model keys (default: all).\n'
                              'Keys: unified_cunet, unified_cunet_mlp,\n'
                              '      fps_punet, fps_pugacnet,\n'
                              '      fps_adcosnet, fps_hspcnet'))

    # Output
    parser.add_argument('--save_dir', type=str, default='./results_kitti',
                        help='Output directory (default: ./results_kitti)')

    # Mode
    parser.add_argument('--full_comparison', action='store_true',
                        help='Run all four (alpha, beta) configs')

    # Other
    parser.add_argument('--seed',    type=int, default=42)
    parser.add_argument('--no_cuda', action='store_true')

    args = parser.parse_args()

    if args.full_comparison:
        run_full_comparison(args)
    else:
        run_evaluation(args)


if __name__ == '__main__':
    main()
