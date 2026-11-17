# Fixed ADCoSNet.py - The issue is with matrix multiplication dimensions

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MEMDModule(nn.Module):
    """Multivariate Empirical Mode Decomposition Module"""
    def __init__(self, num_channels=3, num_imfs=6):
        super(MEMDModule, self).__init__()
        self.num_imfs = num_imfs

        # Learnable convolutional layers for envelope estimation
        self.envelope_estimators = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(num_channels, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(64, num_channels, kernel_size=3, padding=1)
            ) for _ in range(num_imfs)
        ])

    def sifting_process(self, x, estimator):
        # Estimate upper and lower envelopes
        envelopes = estimator(x)
        mean_envelope = torch.mean(envelopes, dim=1, keepdim=True)
        return x - mean_envelope

    def forward(self, x):
        # x: [B, C=3, N]
        imfs = []
        residue = x.clone()

        # Iterative sifting process
        for estimator in self.envelope_estimators:
            imf = self.sifting_process(residue, estimator)
            imfs.append(imf)
            residue = residue - imf

        return torch.stack(imfs, dim=1), residue  # [B, num_imfs, C, N], [B, C, N]

class ConvCSNet(nn.Module):
    """Convolutional Compressive Sensing Network - Fixed version"""
    def __init__(self, in_channels=3, hidden_channels=64, cs_ratio=0.5, num_points=1024):
        super(ConvCSNet, self).__init__()
        self.cs_ratio = cs_ratio
        self.num_points = num_points
        self.hidden_channels = hidden_channels
        
        # Calculate actual compressed size
        self.compressed_size = int(num_points / cs_ratio)  # Note: cs_ratio is actually expansion ratio

        # Feature extraction
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU()
        )

        # Residue projection to match feature channels
        self.residue_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)

        # Fixed: Use linear layer instead of matrix multiplication
        # This compresses from num_points to compressed_size
        self.compress_layer = nn.Linear(num_points, self.compressed_size)

        # Reconstruction decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels//2),
            nn.ReLU(),
            nn.Conv1d(hidden_channels//2, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, residue):
        # x: [B, C, N], residue: [B, C, N]
        batch_size, channels, num_points = x.size()

        # Extract features
        features = self.encoder(x)  # [B, hidden_channels, N]

        # Project residue to match feature dimensions
        residue_features = self.residue_proj(residue)  # [B, hidden_channels, N]

        # Combine features
        combined_features = features + residue_features  # [B, hidden_channels, N]

        # Compression using linear layer
        # Transpose to [B, hidden_channels, N] -> [B, N, hidden_channels] for linear layer
        combined_features_t = combined_features.transpose(1, 2)  # [B, N, hidden_channels]
        
        # Apply compression across spatial dimension
        compressed_list = []
        for i in range(self.hidden_channels):
            channel_features = combined_features[:, i, :]  # [B, N]
            compressed_channel = self.compress_layer(channel_features)  # [B, compressed_size]
            compressed_list.append(compressed_channel.unsqueeze(1))
        
        compressed = torch.cat(compressed_list, dim=1)  # [B, hidden_channels, compressed_size]

        # Reconstruction
        reconstructed = self.decoder(compressed)  # [B, C, compressed_size]

        return reconstructed

class ADCoSNet(nn.Module):
    """Complete ADCoSNet Architecture - Fixed version"""
    def __init__(self, in_channels=3, cs_ratio=0.5, num_points=1024):
        super(ADCoSNet, self).__init__()
        self.memd = MEMDModule(in_channels)
        self.cs_net = ConvCSNet(in_channels, cs_ratio=cs_ratio, num_points=num_points)
        self.cs_ratio = cs_ratio
        self.num_points = num_points

    def forward(self, x):
        # x: [B, 3, N]
        # Ensure we have the expected number of points
        if x.size(2) != self.num_points:
            # Adapt to actual input size
            actual_points = x.size(2)
            self.cs_net.num_points = actual_points
            self.cs_net.compressed_size = int(actual_points / self.cs_ratio)
            # Reinitialize the compression layer with correct dimensions
            self.cs_net.compress_layer = nn.Linear(actual_points, self.cs_net.compressed_size).to(x.device)
        
        imfs, residue = self.memd(x)
        reconstructed = self.cs_net(x, residue)
        return reconstructed

def chamfer_distance(pred, target):
    """Chamfer Distance Loss"""
    # Handle different input formats
    if pred.size(2) == 3:  # [B, N, 3]
        pred = pred.transpose(1, 2)  # [B, 3, N]
    if target.size(2) == 3:  # [B, M, 3]
        target = target.transpose(1, 2)  # [B, 3, M]
    
    # Now transpose back for distance computation
    pred = pred.transpose(1, 2)  # [B, N, 3]
    target = target.transpose(1, 2)  # [B, M, 3]
    
    dist = torch.cdist(pred, target)
    loss1 = torch.mean(dist.min(dim=2)[0])
    loss2 = torch.mean(dist.min(dim=1)[0])
    return loss1 + loss2
