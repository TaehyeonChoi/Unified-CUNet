"""
Baseline Models for Comparison

Implements the three baseline methods used in the paper's experimental comparison:
    1. ADCoSNet - Deep Convolutional Compressed Sensing Network
    2. PU-Net - Point Cloud Upsampling Network
    3. PU-GACNet - Graph Attention Convolution Network for Upsampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# ADCoSNet: Deep Convolutional Compressed Sensing Network
# Reference: Shinde & Durbha, Remote Sensing, 2023
# ============================================================

class MEMDModule(nn.Module):
    """Multivariate Empirical Mode Decomposition Module."""
    def __init__(self, num_channels=3, num_imfs=6):
        super(MEMDModule, self).__init__()
        self.num_imfs = num_imfs
        
        self.envelope_estimators = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(num_channels, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(64, num_channels, kernel_size=3, padding=1)
            ) for _ in range(num_imfs)
        ])

    def forward(self, x):
        imfs = []
        residue = x.clone()
        
        for estimator in self.envelope_estimators:
            envelopes = estimator(residue)
            mean_envelope = torch.mean(envelopes, dim=1, keepdim=True)
            imf = residue - mean_envelope
            imfs.append(imf)
            residue = residue - imf
        
        return torch.stack(imfs, dim=1), residue


class ConvCSNet(nn.Module):
    """Convolutional Compressive Sensing Network."""
    def __init__(self, in_channels=3, hidden_channels=64, cs_ratio=0.5, num_points=1024):
        super(ConvCSNet, self).__init__()
        self.cs_ratio = cs_ratio
        self.num_points = num_points
        self.hidden_channels = hidden_channels
        self.compressed_size = int(num_points * cs_ratio)
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU()
        )
        
        self.residue_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.compress_layer = nn.Linear(num_points, self.compressed_size)
        
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Conv1d(hidden_channels // 2, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, residue):
        batch_size = x.size(0)
        features = self.encoder(x)
        residue_features = self.residue_proj(residue)
        combined = features + residue_features
        
        # Compress each channel
        compressed_list = []
        for i in range(self.hidden_channels):
            channel_features = combined[:, i, :]
            compressed_channel = self.compress_layer(channel_features)
            compressed_list.append(compressed_channel.unsqueeze(1))
        
        compressed = torch.cat(compressed_list, dim=1)
        reconstructed = self.decoder(compressed)
        
        return reconstructed


class ADCoSNet(nn.Module):
    """
    ADCoSNet: Adaptive Deep Convolutional Compressed Sensing Network.
    
    Uses multivariate empirical mode decomposition with convolutional
    compressive sensing for point cloud compression.
    """
    def __init__(self, in_channels=3, cs_ratio=0.5, num_points=1024):
        super(ADCoSNet, self).__init__()
        self.memd = MEMDModule(in_channels)
        self.cs_net = ConvCSNet(in_channels, cs_ratio=cs_ratio, num_points=num_points)
        self.cs_ratio = cs_ratio
        self.num_points = num_points

    def forward(self, x):
        # Adapt to actual input size
        actual_points = x.size(2)
        if actual_points != self.num_points:
            self.cs_net.num_points = actual_points
            self.cs_net.compressed_size = int(actual_points * self.cs_ratio)
            self.cs_net.compress_layer = nn.Linear(
                actual_points, self.cs_net.compressed_size
            ).to(x.device)
        
        imfs, residue = self.memd(x)
        reconstructed = self.cs_net(x, residue)
        return reconstructed


# ============================================================
# PU-Net: Point Cloud Upsampling Network
# Reference: Yu et al., CVPR 2018
# ============================================================

class PointFeatureEmbedding(nn.Module):
    """Hierarchical feature learning with multi-level aggregation."""
    def __init__(self, num_points):
        super(PointFeatureEmbedding, self).__init__()
        
        self.level1_mlp = nn.Sequential(
            nn.Conv1d(3, 32, 1), nn.ReLU(),
            nn.Conv1d(32, 32, 1), nn.ReLU(),
            nn.Conv1d(32, 64, 1)
        )
        
        self.level2_mlp = nn.Sequential(
            nn.Conv1d(64, 64, 1), nn.ReLU(),
            nn.Conv1d(64, 64, 1), nn.ReLU(),
            nn.Conv1d(64, 128, 1)
        )
        
        self.level3_mlp = nn.Sequential(
            nn.Conv1d(128, 128, 1), nn.ReLU(),
            nn.Conv1d(128, 128, 1), nn.ReLU(),
            nn.Conv1d(128, 256, 1)
        )
        
        self.level4_mlp = nn.Sequential(
            nn.Conv1d(256, 256, 1), nn.ReLU(),
            nn.Conv1d(256, 256, 1), nn.ReLU(),
            nn.Conv1d(256, 512, 1)
        )
        
        self.feature_reduction = nn.Conv1d(64 + 128 + 256 + 512, 64, 1)

    def forward(self, x):
        if x.size(1) != 3 and x.size(2) == 3:
            x = x.transpose(1, 2).contiguous()
        
        f1 = self.level1_mlp(x)
        f2 = self.level2_mlp(f1)
        f3 = self.level3_mlp(f2)
        f4 = self.level4_mlp(f3)
        
        features = torch.cat([f1, f2, f3, f4], dim=1)
        features = self.feature_reduction(features)
        
        return features.transpose(1, 2).contiguous()


class FeatureExpansionPUNet(nn.Module):
    """Multi-branch feature expansion for PU-Net."""
    def __init__(self, input_channels=64, output_channels=128, up_ratio=4):
        super(FeatureExpansionPUNet, self).__init__()
        self.up_ratio = up_ratio
        
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_channels, output_channels, 1),
                nn.ReLU(),
                nn.Conv1d(output_channels, output_channels, 1)
            ) for _ in range(up_ratio)
        ])

    def forward(self, x):
        x = x.transpose(1, 2)
        branch_features = [branch(x) for branch in self.branches]
        out = torch.cat(branch_features, dim=2)
        return out.transpose(1, 2).contiguous()


class PUNet(nn.Module):
    """
    PU-Net: Point Cloud Upsampling Network.
    
    Hierarchical feature learning with multi-branch expansion
    for high-quality point cloud upsampling.
    """
    def __init__(self, num_points=1024, up_ratio=4):
        super(PUNet, self).__init__()
        self.up_ratio = up_ratio
        
        self.feature_embedding = PointFeatureEmbedding(num_points)
        self.feature_expansion = FeatureExpansionPUNet(64, 128, up_ratio)
        self.coord_reconstruction = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        # Handle input format
        if x.size(1) == 3 and x.size(2) != 3:
            x = x.transpose(1, 2).contiguous()
        
        features = self.feature_embedding(x)
        expanded = self.feature_expansion(features)
        points = self.coord_reconstruction(expanded)
        
        return points.transpose(1, 2).contiguous()


# ============================================================
# PU-GACNet: Graph Attention Convolution Network
# Reference: Han et al., Image and Vision Computing, 2022
# ============================================================

class GraphAttentionConvolution(nn.Module):
    """Graph attention convolution layer."""
    def __init__(self, in_channels, out_channels, k=9):
        super(GraphAttentionConvolution, self).__init__()
        self.k = k
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.transform = nn.Conv1d(in_channels, out_channels, 1)
        
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, 1, 1),
            nn.Softmax(dim=-1)
        )

    def get_graph_feature(self, x):
        batch_size, num_dims, num_points = x.size()
        k = min(self.k, num_points)
        
        x_trans = x.transpose(2, 1).contiguous()
        
        inner = -2 * torch.matmul(x_trans, x)
        xx = torch.sum(x_trans ** 2, dim=2, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        
        idx = pairwise_distance.topk(k=k, dim=-1)[1]
        
        feature_list = []
        for bi in range(batch_size):
            feature = x[bi].unsqueeze(0)
            neighbor_features = torch.index_select(
                feature, 2, idx[bi].view(-1)
            ).view(1, num_dims, num_points, k)
            feature_list.append(neighbor_features)
        
        neighbor_features = torch.cat(feature_list, dim=0)
        central_feature = x.unsqueeze(-1).repeat(1, 1, 1, k)
        edge_feature = torch.cat([central_feature, neighbor_features - central_feature], dim=1)
        
        return edge_feature

    def forward(self, x):
        batch_size, num_dims, num_points = x.size()
        k = min(self.k, num_points)
        
        edge_feature = self.get_graph_feature(x)
        attention_weights = self.attention(edge_feature)
        
        transformed = self.transform(x)
        transformed = transformed.unsqueeze(-1).repeat(1, 1, 1, k)
        
        out = (transformed * attention_weights).sum(dim=-1)
        
        return out


class EdgeAwareNodeShuffle(nn.Module):
    """Edge-aware node shuffle for upsampling."""
    def __init__(self, in_channels, up_factor=4):
        super(EdgeAwareNodeShuffle, self).__init__()
        self.up_factor = up_factor
        
        self.edge_conv = nn.Sequential(
            nn.Conv1d(in_channels * 2, in_channels, 1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU()
        )
        
        self.expand_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels * up_factor, 1),
            nn.BatchNorm1d(in_channels * up_factor),
            nn.ReLU()
        )

    def forward(self, x, edge_info):
        batch_size, channels, num_points = x.size()
        
        combined = torch.cat([x, edge_info], dim=1)
        edge_features = self.edge_conv(combined)
        
        expanded = self.expand_conv(edge_features)
        output = expanded.view(batch_size, channels, self.up_factor, num_points)
        output = output.permute(0, 1, 3, 2).contiguous()
        output = output.view(batch_size, channels, -1)
        
        return output


class PUGACNet(nn.Module):
    """
    PU-GACNet: Point Cloud Upsampling with Graph Attention.
    
    Uses graph attention convolution to capture local geometric
    structures for high-quality upsampling.
    """
    def __init__(self, in_channels=3, out_channels=3, up_factor=4):
        super(PUGACNet, self).__init__()
        self.up_factor = up_factor
        
        self.gac1 = nn.Sequential(
            GraphAttentionConvolution(in_channels, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        self.gac2 = nn.Sequential(
            GraphAttentionConvolution(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        self.ens = EdgeAwareNodeShuffle(128, up_factor)
        
        self.reconstruction = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, out_channels, 1)
        )

    def forward(self, x):
        # Ensure BCN format
        if x.size(2) == 3 and x.size(1) != 3:
            x = x.transpose(1, 2).contiguous()
        
        f1 = self.gac1(x)
        f2 = self.gac2(f1)
        
        edge_info = f2.detach()
        expanded = self.ens(f2, edge_info)
        
        output = self.reconstruction(expanded)
        
        return output


# ============================================================
# Dimension Handler Wrapper (FIXED to accept mode parameter)
# ============================================================

class DimensionFixedModel(nn.Module):
    """
    Wrapper that handles input/output dimension formats.
    
    Ensures consistent tensor format handling across different models.
    Now accepts 'mode' parameter for compatibility with Unified-CUNet training.
    """
    def __init__(self, model, required_input_format='BCN', name='Model'):
        super(DimensionFixedModel, self).__init__()
        self.model = model
        self.required_format = required_input_format
        self.name = name

    def forward(self, x, mode=None, **kwargs):
        """
        Forward pass with optional mode parameter (ignored for baselines).
        
        Args:
            x: Input tensor
            mode: Ignored for baseline models (for compatibility with Unified-CUNet)
            **kwargs: Additional arguments (ignored)
        """
        # Detect input format
        original_format = 'BCN' if x.size(1) == 3 else 'BNC'
        
        # Convert to required format
        if original_format != self.required_format:
            x = x.transpose(1, 2).contiguous()
        
        # Forward through model
        output = self.model(x)
        
        # Convert output back if needed
        if isinstance(output, torch.Tensor):
            if original_format == 'BNC' and output.size(1) == 3:
                output = output.transpose(1, 2).contiguous()
        
        return output


# ============================================================
# FPS Preprocessing Utilities
# Used for sequential baseline evaluation (FPS + Upsampler)
# ============================================================

def fps_subsample(points, num_output):
    """
    Farthest Point Sampling — self-contained, no external dependencies.

    Args:
        points: [B, 3, N] (BCN format)
        num_output: Number of points to keep (N_c = alpha * N)

    Returns:
        Subsampled point cloud [B, 3, num_output]
    """
    device = points.device
    # Work in BNC for distance computation
    xyz = points.transpose(1, 2).contiguous()  # [B, N, 3]
    B, N, _ = xyz.shape
    num_output = min(num_output, N)

    centroids = torch.zeros(B, num_output, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_idx = torch.arange(B, device=device)

    for i in range(num_output):
        centroids[:, i] = farthest
        centroid = xyz[batch_idx, farthest].unsqueeze(1)        # [B, 1, 3]
        dist = ((xyz - centroid) ** 2).sum(dim=-1)              # [B, N]
        distance = torch.min(distance, dist)
        farthest = distance.argmax(dim=-1)

    # Gather selected points
    # centroids: [B, num_output] → index into BCN points
    idx = centroids.unsqueeze(1).expand(-1, 3, -1)             # [B, 3, num_output]
    return points.gather(2, idx)                                # [B, 3, num_output]


class SequentialFPSBaseline(nn.Module):
    """
    Wrapper for sequential pipeline baselines: FPS → Upsampler.

    This implements the FPS+X baselines required for fair comparison
    with Unified-CUNet in the reviewer-requested ablation (Reviewer 1,
    Major Comment 1 and Reviewer 3, Major Concern 3).

    Pipeline:
        Input (N_in points)
            → FPS subsample to N_c = alpha * N_in
            → Upsampling model
            → Output (beta * N_c points)

    The output is then compared against the original N_in-point cloud
    using the same metrics as all other methods.

    Args:
        upsampler: A baseline upsampling model (PUNet, PUGACNet, etc.)
        compression_ratio: alpha — fraction of points to retain after FPS
        name: Display name for logging/reporting
    """
    def __init__(self, upsampler, compression_ratio=0.5, name='FPS+Model'):
        super(SequentialFPSBaseline, self).__init__()
        self.upsampler = upsampler
        self.compression_ratio = compression_ratio
        self.name = name

    def forward(self, x, mode=None):
        """
        Args:
            x:    Input point cloud [B, 3, N]
            mode: Ignored (for API compatibility with Unified-CUNet)

        Returns:
            Upsampled point cloud [B, 3, beta * alpha * N]
        """
        N = x.size(2)
        num_compressed = max(32, int(N * self.compression_ratio))

        # Step 1 — FPS subsampling (geometry-only, same as Unified-CUNet)
        x_fps = fps_subsample(x, num_compressed)               # [B, 3, N_c]

        # Step 2 — Upsampling model
        return self.upsampler(x_fps)


def create_baseline_models(compression_ratio=0.5, upsample_ratio=4, num_points=2048):
    """
    Create all baseline models for comparison.
    
    Returns:
        Dictionary of model name -> model instance
    """
    models = {
        'ADCoSNet': DimensionFixedModel(
            ADCoSNet(
                in_channels=3,
                cs_ratio=compression_ratio,
                num_points=num_points
            ),
            required_input_format='BCN',
            name='ADCoSNet'
        ),
        
        'PU-Net': DimensionFixedModel(
            PUNet(
                num_points=num_points,
                up_ratio=int(upsample_ratio)
            ),
            required_input_format='BNC',
            name='PU-Net'
        ),
        
        'PU-GACNet': DimensionFixedModel(
            PUGACNet(
                in_channels=3,
                out_channels=3,
                up_factor=int(upsample_ratio)
            ),
            required_input_format='BCN',
            name='PU-GACNet'
        )
    }
    
    return models


def create_fps_baseline_models(compression_ratio=0.5, upsample_ratio=4, num_points=2048):
    """
    Create FPS+X sequential pipeline baselines for fair comparison.

    Each model first applies FPS to reduce point count by compression_ratio,
    then applies its upsampling network.  The input to the upsampler is
    N_c = compression_ratio * num_points points; the upsampler is
    instantiated with up_ratio = beta so the final output has
    beta * N_c = beta * alpha * N points — identical to Unified-CUNet output.

    Returns:
        Dictionary of model name -> SequentialFPSBaseline instance
    """
    N_c = max(32, int(num_points * compression_ratio))

    fps_models = {
        'FPS+ADCoSNet': SequentialFPSBaseline(
            upsampler=DimensionFixedModel(
                ADCoSNet(in_channels=3, cs_ratio=1.0, num_points=N_c),
                required_input_format='BCN',
                name='ADCoSNet'
            ),
            compression_ratio=compression_ratio,
            name='FPS+ADCoSNet'
        ),

        'FPS+PU-Net': SequentialFPSBaseline(
            upsampler=DimensionFixedModel(
                PUNet(num_points=N_c, up_ratio=int(upsample_ratio)),
                required_input_format='BNC',
                name='PU-Net'
            ),
            compression_ratio=compression_ratio,
            name='FPS+PU-Net'
        ),

        'FPS+PU-GACNet': SequentialFPSBaseline(
            upsampler=DimensionFixedModel(
                PUGACNet(in_channels=3, out_channels=3, up_factor=int(upsample_ratio)),
                required_input_format='BCN',
                name='PU-GACNet'
            ),
            compression_ratio=compression_ratio,
            name='FPS+PU-GACNet'
        ),
    }

    return fps_models


if __name__ == "__main__":
    # Test all baseline models
    print("Testing baseline models...")
    
    batch_size = 2
    num_points = 1024
    x = torch.randn(batch_size, 3, num_points)
    
    models = create_baseline_models(
        compression_ratio=0.5,
        upsample_ratio=4,
        num_points=num_points
    )
    
    for name, model in models.items():
        print(f"\n{name}:")
        output = model(x, mode='full')
        print(f"  Input: {x.shape} -> Output: {output.shape}")

    print("\nTesting FPS+baseline models...")
    fps_models = create_fps_baseline_models(
        compression_ratio=0.5,
        upsample_ratio=4,
        num_points=num_points
    )
    for name, model in fps_models.items():
        print(f"\n{name}:")
        output = model(x, mode='full')
        print(f"  Input: {x.shape} -> FPS({int(num_points*0.5)}) -> Output: {output.shape}")

    print("\n\u2713 All baseline model tests passed!")
