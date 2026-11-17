"""
Evaluation Script for Unified-CUNet — Integrated Comparison

Evaluates ALL models for a given (alpha, beta) configuration in one run
and prints a single side-by-side comparison table, exactly as in the paper.

Typical usage
-------------
# Single (alpha, beta) config — evaluate all models
python evaluate.py \
    --compression_ratio 0.5 --upsample_ratio 4 \
    --checkpoint_dir ./checkpoints

# All four (alpha, beta) configs — generates Tables 1-4
python evaluate.py \
    --checkpoint_dir ./checkpoints \
    --full_comparison

Checkpoint naming convention (matches train.py defaults):
    <checkpoint_dir>/unified_cunet_best.pt
    <checkpoint_dir>/unified_cunet_mlp_best.pt
    <checkpoint_dir>/fps_punet_best.pt
    <checkpoint_dir>/fps_pugacnet_best.pt
    <checkpoint_dir>/fps_adcosnet_best.pt

You can override any individual path with the explicit flags
--ckpt_unified_cunet, --ckpt_unified_cunet_mlp, etc.

Outputs (written to --save_dir):
    results_a<alpha>_b<beta>.txt    — human-readable table
    table_a<alpha>_b<beta>.tex      — LaTeX table snippet (paste into paper)
"""

import os
import sys
import argparse
import torch
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unified_cunet import create_unified_cunet, create_unified_cunet_mlp
from models.baselines import (
    fps_subsample,
    PUNet, PUGACNet, ADCoSNet, DimensionFixedModel,
)
from metrics.evaluation import compute_all_metrics, MetricsAccumulator
from datasets.modelnet40 import create_dataloaders
from models.HSPC_Net import HSPCNet


# ─────────────────────────────────────────────────────────────────────────────
# Model registry
#
# Each entry defines:
#   key          internal name used for checkpoint lookup and result dict
#   display      name printed in tables (matches paper)
#   build(a,b,N) factory that returns (model, fps_preprocess, fps_ratio)
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
    N_c = max(32, int(N * alpha))
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

# Ordered list — controls the row order in the output table
MODEL_REGISTRY = [
    dict(key='fps_adcosnet',        display='FPS+ADCoSNet',          build=_build_fps_adcosnet),
    dict(key='fps_pugacnet',        display='FPS+PU-GACNet',         build=_build_fps_pugacnet),
    dict(key='fps_punet',           display='FPS+PU-Net',            build=_build_fps_punet),
    dict(key='fps_hspcnet',         display='FPS+HSPC-Net',          build=_build_fps_hspcnet),
    dict(key='unified_cunet_mlp',   display='Unified-CUNet-MLP',     build=_build_unified_cunet_mlp),
    dict(key='unified_cunet',       display='Unified-CUNet (Ours)',   build=_build_unified_cunet),
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_checkpoint(model, path, device):
    """Load weights from path; silently skip if path is None or missing."""
    if not path:
        return model, False           # (model, loaded)
    if not os.path.exists(path):
        print(f"    [skip] checkpoint not found: {path}")
        return model, False
    ckpt  = torch.load(path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    epoch = ckpt.get('epoch', '?')
    loss  = ckpt.get('loss',  '?')
    print(f"    loaded  {path}  (epoch {epoch}, val_loss {loss})")
    return model, True


def _model_forward(model, x):
    try:
        return model(x, mode='full')
    except TypeError:
        return model(x)


def _resolve_checkpoint(args, key):
    """
    Return checkpoint path for a given model key.
    Explicit --ckpt_<key> flag takes priority, then default naming convention.
    """
    flag_name = f'ckpt_{key}'
    explicit  = getattr(args, flag_name, None)
    if explicit:
        return explicit
    return os.path.join(args.checkpoint_dir, f'{key}_best.pt')


# ─────────────────────────────────────────────────────────────────────────────
# Metrics constants
# ─────────────────────────────────────────────────────────────────────────────

METRIC_KEYS   = ['chamfer_distance', 'hausdorff_distance', 'point_to_surface',
                 'uniformity', 'normal_consistency', 'curvature_consistency']
METRIC_LABELS = ['CD', 'HD', 'P2S', 'Uniformity', 'NC', 'CC']
# True = higher is better, False = lower is better
METRIC_HIGHER = [False, False, False, False, True, True]


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, test_loader, device,
                   fps_preprocess=False, fps_ratio=0.5,
                   max_samples=None):
    model.eval()
    acc = MetricsAccumulator()

    with torch.no_grad():
        for idx, (inputs, _) in enumerate(test_loader):
            if max_samples is not None and idx >= max_samples:
                break
            try:
                inputs = inputs.to(device)          # [1, 3, N]
                target = inputs.clone()

                if fps_preprocess:
                    N_c  = max(32, int(inputs.size(2) * fps_ratio))
                    minp = fps_subsample(inputs, N_c)
                else:
                    minp = inputs

                out = _model_forward(model, minp)   # [1, 3, N_out]

                # numpy [N, 3]
                pred = out.cpu().numpy()
                tgt  = target.cpu().numpy()
                if pred.shape[1] == 3: pred = pred.transpose(0, 2, 1)
                if tgt.shape[1]  == 3: tgt  = tgt.transpose(0, 2, 1)

                acc.update(compute_all_metrics(pred[0], tgt[0]))

                if (idx + 1) % 50 == 0:
                    n = max_samples or len(test_loader)
                    print(f"      {idx+1}/{n} samples...", flush=True)

            except Exception as e:
                print(f"    Warning: sample {idx}: {e}")
                continue

    return acc.get_average()


# ─────────────────────────────────────────────────────────────────────────────
# Display & save
# ─────────────────────────────────────────────────────────────────────────────

def _best_per_metric(results):
    """Return best value per metric key across all entries in results dict."""
    best = {}
    for ki, k in enumerate(METRIC_KEYS):
        vals = [m.get(k, float('nan')) for m in results.values()
                if not np.isnan(m.get(k, float('nan')))]
        if not vals:
            best[k] = None
            continue
        best[k] = max(vals) if METRIC_HIGHER[ki] else min(vals)
    return best


def print_table(results, title=''):
    best = _best_per_metric(results)
    W    = 24
    col  = 12

    print(f"\n{'='*(W + col*len(METRIC_LABELS))}")
    if title:
        print(f" {title}")
    print(f"{'='*(W + col*len(METRIC_LABELS))}")
    print(f"{'Method':<{W}}" + "".join(f"{l:>{col}}" for l in METRIC_LABELS))
    print("-" * (W + col * len(METRIC_LABELS)))

    for name, m in results.items():
        row = f"{name:<{W}}"
        for ki, k in enumerate(METRIC_KEYS):
            v   = m.get(k, float('nan'))
            fmt = f"{v:.6f}" if k == 'uniformity' else f"{v:.4f}"
            # Mark best with asterisk
            is_best = (best[k] is not None and not np.isnan(v)
                       and abs(v - best[k]) < 1e-8)
            row += f"{'*'+fmt if is_best else fmt:>{col}}"
        print(row)

    print("=" * (W + col * len(METRIC_LABELS)))
    print("  * = best in column")


def save_txt(results, path, config_str=''):
    best = _best_per_metric(results)
    with open(path, 'w') as f:
        f.write(f"Evaluation Results\n")
        f.write(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if config_str:
            f.write(f"Config    : {config_str}\n")
        f.write("=" * 70 + "\n\n")
        # Header
        f.write(f"{'Method':<24}")
        for l in METRIC_LABELS:
            f.write(f"{l:>12}")
        f.write("\n" + "-" * 70 + "\n")
        for name, m in results.items():
            f.write(f"{name:<24}")
            for ki, k in enumerate(METRIC_KEYS):
                v   = m.get(k, float('nan'))
                fmt = f"{v:.6f}" if k == 'uniformity' else f"{v:.4f}"
                is_best = (best[k] is not None and not np.isnan(v)
                           and abs(v - best[k]) < 1e-8)
                f.write(f"{'*'+fmt if is_best else fmt:>12}")
            f.write("\n")
        f.write("\n* = best in column\n")
    print(f"Saved -> {path}")


def save_latex(results, path, config_str=''):
    best = _best_per_metric(results)
    with open(path, 'w') as f:
        f.write("% Auto-generated — paste into paper\n")
        f.write("\\begin{table}[!ht]\n"
                "\\begin{adjustwidth}{-2.25in}{0in}\n"
                "\\centering\n")
        caption = config_str or "Quantitative comparison"
        f.write(f"\\caption{{\\textbf{{{caption}.}}"
                " Bold values indicate best performance.}}\n")
        f.write("\\begin{tabular}{|l|l|l|l|l|l|l|}\\hline\n")
        f.write("\\textbf{Method} & \\textbf{CD} & \\textbf{HD} & "
                "\\textbf{P2S} & \\textbf{Uniformity} & "
                "\\textbf{NC} & \\textbf{CC} \\\\ \\thickhline\n")
        for name, m in results.items():
            cells = []
            for k in METRIC_KEYS:
                v   = m.get(k, 0.0)
                fmt = f"{v:.6f}" if k == 'uniformity' else f"{v:.4f}"
                is_best = (best[k] is not None
                           and abs(v - best[k]) < 1e-8)
                cells.append(f"\\textbf{{{fmt}}}" if is_best else fmt)
            f.write(f"{name} & " + " & ".join(cells) + " \\\\ \\hline\n")
        f.write("\\end{tabular}\n"
                "\\end{adjustwidth}\n"
                "\\end{table}\n")
    print(f"Saved -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation function for one (alpha, beta) config
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(args):
    device = torch.device(
        'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    alpha = args.compression_ratio
    beta  = int(args.upsample_ratio)
    N     = args.num_points

    # Dataset (shared across all models)
    print(f"\nLoading test data from {args.data_dir}...")
    _, _, test_loader = create_dataloaders(
        root_dir=args.data_dir, batch_size=1,
        num_points=N, num_workers=args.num_workers)
    print(f"Test samples: {len(test_loader)}"
          + (f"  (capped at {args.max_samples})" if args.max_samples else ""))

    # Determine which models to run
    if args.models:
        keys_to_run = [k.strip() for k in args.models.split(',')]
    else:
        keys_to_run = [m['key'] for m in MODEL_REGISTRY]

    os.makedirs(args.save_dir, exist_ok=True)
    results = {}   # display_name -> metric dict

    for entry in MODEL_REGISTRY:
        key     = entry['key']
        display = entry['display']

        if key not in keys_to_run:
            continue

        print(f"\n{'─'*60}")
        print(f"  {display}  (alpha={alpha}, beta=x{beta})")
        print(f"{'─'*60}")

        model, fps_preprocess, fps_ratio = entry['build'](alpha, beta, N)

        ckpt_path = _resolve_checkpoint(args, key)
        model, loaded = _load_checkpoint(model, ckpt_path, device)
        if not loaded:
            print(f"    [WARNING] Running with random weights — results are meaningless.")

        model = model.to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f"    Parameters: {params:,}"
              + ("  |  FPS preprocess ON" if fps_preprocess else ""))

        m = evaluate_model(
            model, test_loader, device,
            fps_preprocess=fps_preprocess,
            fps_ratio=fps_ratio,
            max_samples=args.max_samples)

        results[display] = m

        # Per-model quick summary
        nc  = m.get('normal_consistency', 0)
        cc  = m.get('curvature_consistency', 0)
        cd  = m.get('chamfer_distance', 0)
        print(f"    CD={cd:.4f}  NC={nc:.4f}  CC={cc:.4f}")

    # ── Combined table ──────────────────────────────────────────────────────
    config_str = f"(alpha={alpha}, beta=x{beta})"
    title      = f"Quantitative Comparison  {config_str}"
    print_table(results, title=title)

    stem = f"a{alpha}_b{beta}"
    save_txt  (results, os.path.join(args.save_dir, f"results_{stem}.txt"),  title)
    save_latex(results, os.path.join(args.save_dir, f"table_{stem}.tex"),    title)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Full comparison  (Tables 1-4 in one shot)
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS = [(0.50, 4), (0.75, 4), (0.50, 6), (0.75, 6)]


def run_full_comparison(args):
    all_results = {}

    for alpha, beta in CONFIGS:
        print(f"\n{'#'*70}")
        print(f"#  Configuration: alpha={alpha}, beta=x{beta}")
        print(f"{'#'*70}")
        args.compression_ratio = alpha
        args.upsample_ratio    = float(beta)
        all_results[(alpha, beta)] = run_evaluation(args)

    # Master summary
    print(f"\n\n{'='*120}")
    print(" FULL COMPARISON SUMMARY")
    print(f"{'='*120}")
    for (alpha, beta), results in all_results.items():
        print(f"\n  alpha={alpha}  beta=x{beta}")
        print_table(results)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate all point cloud models in one run',
        formatter_class=argparse.RawTextHelpFormatter)

    # Ratios
    parser.add_argument('--compression_ratio', type=float, default=0.5,
                        help='alpha (default: 0.5)')
    parser.add_argument('--upsample_ratio',    type=float, default=4.0,
                        help='beta  (default: 4)')

    # Checkpoints
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory containing *_best.pt files (default: ./checkpoints)')
    # Per-model overrides
    parser.add_argument('--ckpt_unified_cunet',     type=str, default=None)
    parser.add_argument('--ckpt_unified_cunet_mlp', type=str, default=None)
    parser.add_argument('--ckpt_fps_punet',         type=str, default=None)
    parser.add_argument('--ckpt_fps_pugacnet',      type=str, default=None)
    parser.add_argument('--ckpt_fps_adcosnet',      type=str, default=None)
    parser.add_argument('--ckpt_fps_hspcnet',       type=str, default=None)

    # Model selection
    parser.add_argument('--models', type=str, default=None,
                        help=('Comma-separated list of model keys to evaluate.\n'
                              'Default: all models.\n'
                              'Keys: unified_cunet, unified_cunet_mlp,\n'
                              '      fps_punet, fps_pugacnet, fps_adcosnet,\n'
                              '      fps_hspcnet'))

    # Dataset
    parser.add_argument('--data_dir',   type=str, default='./ModelNet40')
    parser.add_argument('--num_points', type=int, default=2048)

    # Output
    parser.add_argument('--save_dir',    type=str, default='./results')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Cap test samples per model (default: full test set)')

    # Mode
    parser.add_argument('--full_comparison', action='store_true',
                        help='Run all four (alpha, beta) configs (Tables 1-4)')

    # Other
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--no_cuda',     action='store_true')

    args = parser.parse_args()

    if args.full_comparison:
        run_full_comparison(args)
    else:
        run_evaluation(args)


if __name__ == '__main__':
    main()
