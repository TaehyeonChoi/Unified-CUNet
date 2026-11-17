# Fixed PU-Net.py - The issue is with the input format to Conv1d

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PointFeatureEmbedding(nn.Module):
    """Hierarchical feature learning with multi-level feature aggregation"""
    def __init__(self, num_points):
        super().__init__()
        self.num_points = num_points

        # Hierarchical feature learning layers
        self.level1_mlp = nn.Sequential(
            nn.Conv1d(3, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1)
        )

        self.level2_mlp = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1)
        )

        self.level3_mlp = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1)
        )

        self.level4_mlp = nn.Sequential(
            nn.Conv1d(256, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 512, 1)
        )

        # Feature reduction for concatenated features
        self.feature_reduction = nn.Conv1d(64 + 128 + 256 + 512, 64, 1)

    def forward(self, x):
        """
        Args:
            x: Input points, shape [B, N, 3] OR [B, 3, N]
        Returns:
            Embedded features, shape [B, N, 64]
        """
        # Handle both input formats
        if x.size(1) != 3 and x.size(2) == 3:
            # Input is [B, N, 3], need to transpose
            x = x.transpose(1, 2).contiguous()
        # Now x is [B, 3, N]

        # Multi-level feature extraction
        f1 = self.level1_mlp(x)
        f2 = self.level2_mlp(f1)
        f3 = self.level3_mlp(f2)
        f4 = self.level4_mlp(f3)

        # Feature concatenation and reduction
        features = torch.cat([f1, f2, f3, f4], dim=1)
        features = self.feature_reduction(features)

        # Return in [B, N, C] format
        return features.transpose(1, 2).contiguous()

class FeatureExpansion(nn.Module):
    """Multi-branch feature expansion as described in paper"""
    def __init__(self, input_channels=64, output_channels=128, up_ratio=4):
        super().__init__()
        self.up_ratio = up_ratio

        # Create r separate branches with reduced correlation
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_channels, output_channels, 1),
                nn.ReLU(),
                nn.Conv1d(output_channels, output_channels, 1)
            ) for _ in range(up_ratio)
        ])

    def forward(self, x):
        """
        Args:
            x: Input features [B, N, C]
        Returns:
            Expanded features [B, N*up_ratio, C]
        """
        # Convert to [B, C, N] for Conv1d
        x = x.transpose(1, 2)

        # Process through each branch
        branch_features = []
        for branch in self.branches:
            branch_features.append(branch(x))

        # Combine branch outputs
        out = torch.cat(branch_features, dim=2)  # [B, C, N*up_ratio]
        return out.transpose(1, 2).contiguous()  # [B, N*up_ratio, C]

class CoordinateReconstruction(nn.Module):
    """Final MLP to reconstruct 3D coordinates"""
    def __init__(self, input_channels=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.mlp(x)

class PUNet(nn.Module):
    """Complete PU-Net architecture - Fixed version"""
    def __init__(self, num_points=1024, up_ratio=4):
        super().__init__()
        self.num_points = num_points
        self.up_ratio = up_ratio

        # Three main components
        self.feature_embedding = PointFeatureEmbedding(num_points)
        self.feature_expansion = FeatureExpansion(
            input_channels=64,
            output_channels=128,
            up_ratio=up_ratio
        )
        self.coordinate_reconstruction = CoordinateReconstruction(128)

    def forward(self, x):
        """
        Args:
            x: Input point cloud [B, N, 3] or [B, 3, N]
        Returns:
            Upsampled point cloud [B, N*up_ratio, 3]
        """
        # The DimensionFixedModel wrapper ensures input is [B, N, 3]
        # But let's handle both formats just in case
        
        features = self.feature_embedding(x)  # [B, N, 64]
        expanded_features = self.feature_expansion(features)  # [B, N*up_ratio, 128]
        points = self.coordinate_reconstruction(expanded_features)  # [B, N*up_ratio, 3]
        
        return points

def earth_movers_distance(pred, gt, eps=1e-12):
    """Improved Earth Mover's Distance loss with better numerical stability"""
    # Ensure valid input dimensions
    if pred.size(0) != gt.size(0):
        raise ValueError(f"Batch sizes don't match: {pred.size(0)} vs {gt.size(0)}")

    # Calculate pairwise distances
    dist = torch.cdist(pred, gt)

    # Add small epsilon to avoid numerical instability
    dist = dist + eps

    # Calculate minimum distances
    min_dist = torch.min(dist, dim=2)[0]

    # Apply log-scale normalization to prevent extremely large values
    loss = torch.log1p(min_dist)

    return torch.mean(loss)

def compute_repulsion_loss(points, k=16, h=0.03, eps=1e-12):
    """Improved repulsion loss with better numerical stability"""
    # Calculate pairwise distances
    dist = torch.cdist(points, points)

    # Add small epsilon to avoid numerical instability
    dist = dist + eps

    # Get k nearest neighbors (excluding self)
    k = min(k, dist.shape[-1] - 1)
    val, _ = torch.topk(dist, k=k+1, dim=2, largest=False)
    val = val[:, :, 1:]  # Exclude self-distance

    # Calculate repulsion term with numerical stability
    loss = torch.mean(torch.exp(-torch.clamp(val, min=0.0, max=10.0)**2 / h**2))

    return loss
