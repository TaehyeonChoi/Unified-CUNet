"""
Main Experiment Script for Unified-CUNet (FIXED - v3)

KEY FIX: Baseline upsampling models (PU-Net, PU-GACNet) now receive
FPS-compressed input during BOTH training AND evaluation.

Pipeline per model type:
    ┌─────────────────┬──────────────────────────────────────────────────────────┐
    │ Model           │ Pipeline                                                 │
    ├─────────────────┼──────────────────────────────────────────────────────────┤
    │ Unified-CUNet   │ Input(2048) → [internal FPS+upsample] → Compare to Orig │
    │ PU-Net          │ Input(2048) → FPS(2048×α) → [upsample] → Compare to Orig│
    │ PU-GACNet       │ Input(2048) → FPS(2048×α) → [upsample] → Compare to Orig│
    │ ADCoSNet        │ Input(2048) → [internal compress]      → Compare to Orig │
    └─────────────────┴──────────────────────────────────────────────────────────┘

Reproduces all experiments from the paper:
    1. Train Unified-CUNet and baseline models
    2. Evaluate on ModelNet40 test set
    3. Generate comparison tables (Tables 1-4)

Usage:
    python main.py --mode train     # Train all models
    python main.py --mode evaluate  # Evaluate and generate tables
    python main.py --mode all       # Train and evaluate
"""

import os
import sys
import argparse
import torch
import numpy as np
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from models.unified_cunet import create_unified_cunet
from models.baselines import create_baseline_models, ADCoSNet, PUNet, PUGACNet
from models.losses import UnifiedCUNetLoss, SimpleLoss
from metrics.evaluation import compute_all_metrics, print_metrics
from datasets.modelnet40 import ModelNet40Dataset, create_dataloaders
from train import Trainer
from evaluate import Evaluator, load_model


# ============================================================
# Models that need FPS preprocessing (upsampling-only baselines)
# ADCoSNet handles compression internally → no FPS needed
# PU-Net, PU-GACNet are upsampling-only → need FPS-compressed input
# ============================================================
MODELS_REQUIRING_FPS = {'PU-Net', 'PU-GACNet'}


def print_header(title):
    """Print formatted header."""
    width = 70
    print("\n" + "=" * width)
    print(f" {title}".center(width))
    print("=" * width + "\n")


def train_unified_cunet(args, train_loader, val_loader, config):
    """
    Train Unified-CUNet model.

    Unified-CUNet handles compression internally (FPS inside forward()),
    so no external FPS preprocessing is needed.
    """
    alpha, beta = config
    model_name = f'unified_cunet_a{alpha}_b{int(beta)}'

    print_header(f"Training Unified-CUNet (α={alpha}, β={beta})")

    # Check if already trained
    checkpoint_path = os.path.join(args.save_dir, f'{model_name}_best.pt')
    if os.path.exists(checkpoint_path) and not args.force_train:
        print(f"Checkpoint exists at {checkpoint_path}")
        print("Skipping training (use --force_train to override)")
        return checkpoint_path

    # Create model
    model = create_unified_cunet(
        compression_ratio=alpha,
        upsample_ratio=beta,
        hidden_dim=64,
        num_transformer_blocks=4
    )

    # Create loss function (full multi-component loss for proposed method)
    loss_fn = UnifiedCUNetLoss(
        lambda_cd=1.0,
        lambda_smoothness=0.1,
        lambda_distribution=0.5,
        lambda_edge=0.3,
        lambda_normal=0.2
    )

    # Create trainer — NO FPS preprocessing (handled internally)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        device=args.device,
        save_dir=args.save_dir,
        model_name=model_name,
        lr=0.0002,
        weight_decay=0.01,
        num_epochs=args.epochs,
        warmup_epochs=5,
        early_stopping_patience=15,
        mixed_precision=not args.no_mixed_precision,
        apply_fps=False,
        compression_ratio=alpha
    )

    # Train
    trainer.train()

    return checkpoint_path


def train_baselines(args, train_loader, val_loader, config):
    """
    Train baseline models with correct preprocessing per model type.

    CRITICAL (v3):
        - PU-Net, PU-GACNet: apply_fps=True → receive FPS-compressed input
        - ADCoSNet: apply_fps=False → receives full input (compresses internally)

    All baselines use SimpleLoss (Chamfer Distance only) as they are
    single-task models without Unified-CUNet's multi-component loss.
    """
    alpha, beta = config

    baselines = create_baseline_models(
        compression_ratio=alpha,
        upsample_ratio=beta,
        num_points=args.num_points
    )

    for name, model in baselines.items():
        model_name = f'{name.lower()}_a{alpha}_b{int(beta)}'

        # Determine if this model needs FPS preprocessing
        needs_fps = name in MODELS_REQUIRING_FPS

        print_header(f"Training {name} (α={alpha}, β={beta})")
        if needs_fps:
            print(f"  >>> FPS preprocessing ENABLED (upsampling-only model)")
            print(f"  >>> Input: 2048 pts → FPS → {int(2048 * alpha)} pts → Model")
        else:
            print(f"  >>> FPS preprocessing DISABLED (model handles compression)")
            print(f"  >>> Input: 2048 pts → Model (internal compression)")

        # Check if already trained
        checkpoint_path = os.path.join(args.save_dir, f'{model_name}_best.pt')
        if os.path.exists(checkpoint_path) and not args.force_train:
            print(f"Checkpoint exists at {checkpoint_path}")
            print("Skipping training (use --force_train to override)")
            continue

        # Use simple Chamfer loss for baselines
        loss_fn = SimpleLoss()

        # Create trainer with correct FPS setting
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            device=args.device,
            save_dir=args.save_dir,
            model_name=model_name,
            lr=0.0002,
            weight_decay=0.01,
            num_epochs=args.epochs,
            warmup_epochs=5,
            early_stopping_patience=15,
            mixed_precision=not args.no_mixed_precision,
            apply_fps=needs_fps,
            compression_ratio=alpha
        )

        trainer.train()


def evaluate_configuration(args, test_loader, config):
    """
    Evaluate all models for a specific (α, β) configuration.

    The Evaluator automatically applies FPS preprocessing to models
    listed in MODELS_REQUIRING_FPS (PU-Net, PU-GACNet).
    """
    alpha, beta = config

    print_header(f"Evaluating Configuration (α={alpha}, β={beta})")

    # Create evaluator
    evaluator = Evaluator(args.device, args.results_dir)

    # Create and load Unified-CUNet
    unified_cunet = create_unified_cunet(
        compression_ratio=alpha,
        upsample_ratio=beta,
        hidden_dim=64,
        num_transformer_blocks=4
    )

    checkpoint_path = os.path.join(args.save_dir, f'unified_cunet_a{alpha}_b{int(beta)}_best.pt')
    unified_cunet = load_model(unified_cunet, checkpoint_path, args.device)

    models = {'Unified-CUNet': unified_cunet}

    # Create and load baselines
    if not args.unified_only:
        baselines = create_baseline_models(
            compression_ratio=alpha,
            upsample_ratio=beta,
            num_points=args.num_points
        )

        for name, model in baselines.items():
            checkpoint_path = os.path.join(
                args.save_dir,
                f'{name.lower()}_a{alpha}_b{int(beta)}_best.pt'
            )
            model = load_model(model, checkpoint_path, args.device)
            models[name] = model

    # Evaluate all models (FPS applied automatically per model type)
    evaluator.evaluate_all_models(
        models, test_loader, args.max_eval_samples,
        compression_ratio=alpha
    )

    # Print and save results
    config_str = f"with (α, β) = ({alpha}, ×{int(beta)})"
    evaluator.print_comparison_table(f"Quantitative Comparison {config_str}")
    evaluator.save_results(f'results_a{alpha}_b{int(beta)}.txt')
    evaluator.generate_latex_table(
        f'table_a{alpha}_b{int(beta)}.tex',
        config_str
    )

    return evaluator.results


def run_all_experiments(args):
    """Run all experiments from the paper."""

    # Configurations from Tables 1-4
    configs = [
        (0.50, 4),
        (0.75, 4),
        (0.50, 6),
        (0.75, 6)
    ]

    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # Set device
    args.device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {args.device}")

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Load data
    print_header("Loading Dataset")
    print(f"Dataset path: {args.data_dir}")

    train_loader, val_loader, test_loader = create_dataloaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_points=args.num_points,
        num_workers=args.num_workers
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test samples: {len(test_loader)}")

    # Print pipeline summary
    print_header("Pipeline Configuration (v3 - FIXED)")
    print("  Model            │ FPS Preprocess │ Rationale")
    print("  ─────────────────┼────────────────┼──────────────────────────────────")
    print("  Unified-CUNet    │ NO             │ Internal FPS in forward()")
    print("  PU-Net           │ YES            │ Upsampling-only model")
    print("  PU-GACNet        │ YES            │ Upsampling-only model")
    print("  ADCoSNet         │ NO             │ Internal learned compression")
    print()

    # Training phase
    if args.mode in ['train', 'all']:
        print_header("TRAINING PHASE")

        for config in configs:
            # Train Unified-CUNet
            train_unified_cunet(args, train_loader, val_loader, config)

            # Train baselines (with correct FPS per model type)
            if not args.unified_only:
                train_baselines(args, train_loader, val_loader, config)

    # Evaluation phase
    if args.mode in ['evaluate', 'all']:
        print_header("EVALUATION PHASE")

        all_results = {}
        for config in configs:
            results = evaluate_configuration(args, test_loader, config)
            all_results[config] = results

        # Generate summary
        generate_summary(args, all_results)


def generate_summary(args, all_results):
    """Generate summary of all experiments."""
    print_header("EXPERIMENT SUMMARY")

    summary_path = os.path.join(args.results_dir, 'experiment_summary.txt')

    with open(summary_path, 'w') as f:
        f.write("Unified-CUNet Experiment Summary (v3 - Fixed Pipeline)\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        f.write("Pipeline Configuration:\n")
        f.write("  Unified-CUNet : full input → internal compress+upsample → compare to original\n")
        f.write("  PU-Net        : full input → FPS compress → upsample → compare to original\n")
        f.write("  PU-GACNet     : full input → FPS compress → upsample → compare to original\n")
        f.write("  ADCoSNet      : full input → internal compress+recon → compare to original\n")
        f.write("\n" + "=" * 80 + "\n\n")

        for config, results in all_results.items():
            alpha, beta = config
            f.write(f"\nConfiguration: (α={alpha}, β={beta})\n")
            f.write("-" * 40 + "\n")

            for model_name, result in results.items():
                metrics = result['metrics']
                f.write(f"\n{model_name}:\n")
                for metric_name, value in metrics.items():
                    f.write(f"  {metric_name}: {value:.6f}\n")

            f.write("\n")

        # Best results summary
        f.write("\n" + "=" * 80 + "\n")
        f.write("BEST RESULTS BY METRIC\n")
        f.write("=" * 80 + "\n")

        metric_names = ['chamfer_distance', 'hausdorff_distance', 'point_to_surface',
                       'uniformity', 'normal_consistency', 'curvature_consistency']

        for metric in metric_names:
            best_value = None
            best_model = None
            best_config = None

            for config, results in all_results.items():
                for model_name, result in results.items():
                    value = result['metrics'].get(metric, None)
                    if value is not None:
                        # Higher is better for NC and CC
                        is_better = (metric in ['normal_consistency', 'curvature_consistency']
                                   and (best_value is None or value > best_value))
                        is_better = is_better or (
                            metric not in ['normal_consistency', 'curvature_consistency']
                            and (best_value is None or value < best_value)
                        )

                        if is_better:
                            best_value = value
                            best_model = model_name
                            best_config = config

            f.write(f"\n{metric}:\n")
            f.write(f"  Best: {best_model} (α={best_config[0]}, β={best_config[1]})\n")
            f.write(f"  Value: {best_value:.6f}\n")

    print(f"Summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Unified-CUNet Experiments (v3 - Fixed Pipeline)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Fix (v3):
    PU-Net and PU-GACNet now receive FPS-compressed input during
    both training and evaluation, matching the paper's protocol.
    ADCoSNet and Unified-CUNet are unchanged (handle compression internally).

Examples:
    # Train all models
    python main.py --mode train --data_dir ./ModelNet40

    # Evaluate pretrained models
    python main.py --mode evaluate --data_dir ./ModelNet40

    # Full pipeline (train + evaluate)
    python main.py --mode all --data_dir ./ModelNet40

    # Train only Unified-CUNet (skip baselines)
    python main.py --mode train --unified_only

    # Quick test with synthetic data
    python main.py --mode all --epochs 5 --max_eval_samples 20
        """
    )

    # Mode argument
    parser.add_argument('--mode', type=str, default='all',
                        choices=['train', 'evaluate', 'all'],
                        help='Execution mode: train, evaluate, or all')

    # Dataset arguments
    parser.add_argument('--data_dir', type=str, default='./ModelNet40',
                        help='Path to ModelNet40 dataset')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='Number of points per sample')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--force_train', action='store_true',
                        help='Force training even if checkpoint exists')

    # Evaluation arguments
    parser.add_argument('--max_eval_samples', type=int, default=100,
                        help='Maximum samples for evaluation')

    # Model arguments
    parser.add_argument('--unified_only', action='store_true',
                        help='Only train/evaluate Unified-CUNet')

    # Directory arguments
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory to save results')

    # Hardware arguments
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--no_mixed_precision', action='store_true',
                        help='Disable mixed precision training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Print configuration
    print_header("UNIFIED-CUNET EXPERIMENTS (v3 - FIXED PIPELINE)")
    print(f"Mode: {args.mode}")
    print(f"Data directory: {args.data_dir}")
    print(f"Number of points: {args.num_points}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"CUDA: {'Disabled' if args.no_cuda else 'Enabled'}")
    print(f"Mixed precision: {'Disabled' if args.no_mixed_precision else 'Enabled'}")

    # Run experiments
    run_all_experiments(args)

    print_header("EXPERIMENTS COMPLETED")


if __name__ == '__main__':
    main()
