# Unified-CUNet

Official implementation of **"Unified-CUNet: A transformer-based framework for surface-aware point cloud compression and upsampling"**, published in *PLOS ONE* (2026).

Unified-CUNet integrates point cloud compression and upsampling into a single transformer-based architecture. Conventional pipelines treat the two tasks independently, so information needed for high-quality upsampling is discarded during compression and errors introduced by compression are amplified afterwards. Unified-CUNet shares a feature extractor and four transformer blocks across both stages, which preserves feature continuity through the pipeline and removes the separate Farthest Point Sampling (FPS) preprocessing step that sequential baselines require.

The method achieves competitive distance-based accuracy (Chamfer Distance, Hausdorff Distance, Point-to-Surface) while consistently obtaining the best Normal Consistency and Curvature Consistency across all tested configurations on both ModelNet40 and the KITTI outdoor LiDAR dataset, using the smallest parameter count (0.24M) among the compared methods.

## Repository contents

```
models/
  unified_cunet.py      Proposed architecture and the MLP ablation variant
  baselines.py          Baseline wrappers, FPS subsampling, model factory
  PU_Net.py             PU-Net baseline
  PU_GACNet.py          PU-GACNet baseline
  ADCoSNet.py           ADCoSNet baseline
  HSPC_Net.py           HSPC-Net baseline
  losses.py             Composite loss (Chamfer + smoothness, distribution, edge, normal)
datasets/
  modelnet40.py         ModelNet40 dataset and dataloaders
metrics/
  evaluation.py         CD, HD, P2S, Uniformity, NC, CC
train.py                Train a single model on ModelNet40
evaluate.py             Evaluate all models on ModelNet40, emits comparison tables
train_kitti.py          Train on KITTI
evaluate_kitti.py       Evaluate on KITTI
main.py                 Legacy all-in-one driver (earlier baseline set)
```

## Requirements

- Python 3.9 or later
- PyTorch 2.0 or later with CUDA
- NumPy, SciPy, scikit-learn, tqdm, matplotlib

```bash
pip install -r requirements.txt
```

All results in the paper were produced on a single NVIDIA RTX 4070 (12 GB) with CUDA 12.1.

## Datasets

### ModelNet40

Download ModelNet40 and point `--data_dir` at the extracted directory. Each model is sampled to 2,048 surface points; 9,843 models are used for training and 2,468 for testing. Random rotation, scaling, and jittering are applied during training.

### KITTI

Download the KITTI 3D object detection Velodyne point clouds and point `--kitti_dir` at the root, which should contain `training/velodyne/` and `testing/velodyne/`. Each `.bin` frame stores points as `[x, y, z, intensity]` in float32; intensity is discarded. Fixed-size patches of `--num_points` points are sampled around random seed points using k-NN and normalised to the unit sphere, so that the input distribution matches ModelNet40 training.

## Pretrained weights

Trained checkpoints for every model and configuration reported in the paper are included in this repository, organised by configuration. Directory names encode the compression ratio α and the upsampling ratio β, so `CR_50_UP_4` corresponds to (α, β) = (0.50, ×4).

```
checkpoints/              ModelNet40
  CR_50_UP_4/
  CR_50_UP_6/
  CR_75_UP_4/
  CR_75_UP_6/
checkpoints_kitti/        KITTI
  CR_50_UP_4/
  CR_75_UP_4/
```

Each directory holds `unified_cunet_best.pt` and `unified_cunet_mlp_best.pt`. The KITTI experiments in the paper are reported for β = ×4 only, so `checkpoints_kitti/` contains two configurations rather than four.

Weights are released for the proposed model and its ablation variant only. The four sequential baselines are reimplementations of methods published by other authors, and we provide the training scripts rather than redistributing their weights. Every baseline was trained under exactly the same protocol as the proposed model, described under [Training](#training) and fixed by the default arguments in `train.py`, including the random seed. Running `train.py` with the corresponding `--model_type` therefore reproduces the baseline checkpoints used for the reported comparisons.

## Models

| `--model_type` | Method in the paper | Parameters |
|---|---|---|
| `fps_adcosnet` | FPS+ADCoSNet | 1.08M |
| `fps_punet` | FPS+PU-Net | 0.52M |
| `fps_pugacnet` | FPS+PU-GACNet | 0.14M |
| `fps_hspcnet` | FPS+HSPC-Net | 0.98M |
| `unified_cunet_mlp` | Unified-CUNet-MLP (ablation) | 0.30M |
| `unified_cunet` | Unified-CUNet (proposed) | 0.24M |

Pretrained weights are released for `unified_cunet` and `unified_cunet_mlp`; the four baselines can be trained with the scripts provided here.

For the `fps_*` sequential baselines, FPS preprocessing is applied automatically using `--compression_ratio` as the compression ratio α. The loss target is always the original full-resolution point cloud, so all methods are trained and evaluated under the same protocol.

Parameter counts are for the (α, β) = (0.50, ×4) configuration.

## Training

Train the proposed model on ModelNet40:

```bash
python train.py \
    --model_type unified_cunet \
    --data_dir ./ModelNet40 \
    --compression_ratio 0.5 \
    --upsample_ratio 4 \
    --save_dir ./checkpoints/CR_50_UP_4
```

Train a sequential baseline:

```bash
python train.py \
    --model_type fps_punet \
    --data_dir ./ModelNet40 \
    --compression_ratio 0.5 \
    --upsample_ratio 4 \
    --save_dir ./checkpoints/CR_50_UP_4
```

Checkpoints are written to `<save_dir>/<model_type>_best.pt`, with the training history alongside as `<model_type>_history.pt`. Point `--save_dir` at the directory matching the configuration being trained, so that the checkpoints line up with the layout described above.

Default hyperparameters match the paper: AdamW with learning rate 0.0002 and weight decay 0.01, cosine annealing with warm restarts, 5 warm-up epochs, batch size 4, 50 epochs with early stopping (patience 15), and mixed precision enabled. Pass `--simple_loss` to train with Chamfer distance alone, which is faster and useful for debugging, and `--resume <path>` to continue from a checkpoint.

### KITTI

```bash
python train_kitti.py \
    --model_type unified_cunet \
    --kitti_dir ./data/kitti \
    --compression_ratio 0.5 \
    --upsample_ratio 4 \
    --checkpoint_dir ./checkpoints_kitti/CR_50_UP_4
```

Pass `--model_type all` to train all six models in sequence.

## Evaluation

Evaluate every model for one configuration and print a side-by-side table:

```bash
python evaluate.py \
    --compression_ratio 0.5 \
    --upsample_ratio 4 \
    --checkpoint_dir ./checkpoints/CR_50_UP_4 \
    --data_dir ./ModelNet40 \
    --models unified_cunet,unified_cunet_mlp
```

`--models` restricts the run to the methods whose checkpoints are present. Drop the flag once the baselines have been trained, and `evaluate.py` will then evaluate all six methods and reproduce the full comparison tables from the paper.

Reproduce all four configurations reported in the paper, (α, β) ∈ {0.50, 0.75} × {×4, ×6}, by running the four commands in turn:

```bash
python evaluate.py --compression_ratio 0.50 --upsample_ratio 4 \
    --checkpoint_dir ./checkpoints/CR_50_UP_4 --data_dir ./ModelNet40
python evaluate.py --compression_ratio 0.75 --upsample_ratio 4 \
    --checkpoint_dir ./checkpoints/CR_75_UP_4 --data_dir ./ModelNet40
python evaluate.py --compression_ratio 0.50 --upsample_ratio 6 \
    --checkpoint_dir ./checkpoints/CR_50_UP_6 --data_dir ./ModelNet40
python evaluate.py --compression_ratio 0.75 --upsample_ratio 6 \
    --checkpoint_dir ./checkpoints/CR_75_UP_6 --data_dir ./ModelNet40
```

Add `--models unified_cunet,unified_cunet_mlp` to each command to evaluate only the released checkpoints.

Because the checkpoints are organised into one directory per configuration, run the configurations separately as shown above rather than with the `--full_comparison` flag, which sweeps all four configurations against a single `--checkpoint_dir`.

Results are written to `--save_dir` (default `./results`) as `results_a<alpha>_b<beta>.txt` and as a LaTeX table `table_a<alpha>_b<beta>.tex`. Use `--models` to restrict the run to a comma-separated subset of model keys, `--max_samples` to cap the number of test samples, and `--ckpt_<key>` to override an individual checkpoint path.

### KITTI

```bash
python evaluate_kitti.py --compression_ratio 0.50 --upsample_ratio 4 \
    --kitti_dir ./data/kitti --checkpoint_dir ./checkpoints_kitti/CR_50_UP_4
python evaluate_kitti.py --compression_ratio 0.75 --upsample_ratio 4 \
    --kitti_dir ./data/kitti --checkpoint_dir ./checkpoints_kitti/CR_75_UP_4
```

Outputs are written to `./results_kitti` by default. `--split training` evaluates on the training frames instead of the test frames, and `--max_frames` with `--patches_per_frame` gives a quick smoke test.

## Metrics

Six metrics are reported. Chamfer Distance, Hausdorff Distance, Point-to-Surface Distance, and Uniformity are lower-is-better; Normal Consistency and Curvature Consistency are higher-is-better. The last two measure how well local surface properties survive the compression and upsampling round trip, which is the primary design objective of the method.

## Citation

```bibtex
@article{choi2026unifiedcunet,
  author  = {Choi, Taehyeon and Park, Jinwan},
  title   = {Unified-CUNet: A transformer-based framework for surface-aware point cloud compression and upsampling},
  journal = {PLOS ONE},
  year    = {2026},
  doi     = {10.1371/journal.pone.0363249}
}
```

## License

Released under the MIT License. See [LICENSE](LICENSE).

## Contact

Taehyeon Choi(taehyeon [at] korea.ac.kr), Jinwan Park
Department of Image Science and Arts, Graduate School of Advanced Imaging Science, Multimedia and Film, Chung-Ang University, Seoul 06974, South Korea

Corresponding author: Jinwan Park
